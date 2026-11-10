"""t0-alpha's backend (`T0Model`): `tfc-t0==0.5.0`'s `T0Forecaster`, loaded from the card's two files, and the research's
fine-tune (`zoo_tsfm/code/t0_ft.py`, `zoo_followup/code/t0_roll.py`) through the library's trainer (`_torch.finetune`).

- **Inputs** as the research's adapter (`zoo_tsfm/code/zoo.py` `T0`): the context `(B, V, T)` is the target then each past
  series as rows of one multivariate series, the known-ahead series `(B, F, T + S)` its `future_covariates`, float32; the
  forecast is variate 0's, at the research's 19 levels (the median, index 9, is a native level of t0-alpha's five).
- **The fine-tune loss** of a window is t0's native pinball loss over its target day's steps and its five native levels, from
  one differentiable forward pass over the rollout buffer (as `forward_q` there), divided by the mean absolute step-to-step
  change of its context plus 1e-3, so markets and regimes weigh alike. Windows of one S share a forward pass; a step that
  forecasts no period of the target day (or has no actual) is masked, and a missing value in a context is masked by t0 as in
  `predict` (the scale is then the mean of the changes it has).
- **State**: the base weights are kept on the CPU at the first fine-tune; `reset()` loads them back, `save_state` writes the
  fine-tuned state as safetensors and `load_state` reads one back.

Needs the `[t0]` extra; importing this module imports torch and t0.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from ..._extras import MissingExtraError
from . import LEVELS_19, Backend, FineTuneData, FineTuneOutcome, staged

try:
    import safetensors.torch
    import torch
    from t0 import T0Forecaster
    from t0.data import VariateType, round_up, time_series_from_array
    from t0.model.rollout import RolloutManager
except ImportError:  # pragma: no cover - the core-only CI job has no t0
    raise MissingExtraError("T0Model", "t0") from None

FILES = ("config.json", "model.safetensors")


class T0Backend(Backend):
    """`T0Forecaster` on `device`, from `resolved` (the card's `config.json` and `model.safetensors`, each by name)."""

    levels = LEVELS_19

    def __init__(self, resolved: Any, device: Any):
        # huggingface_hub's mixin reads the config, then the safetensors file (and prints that it loads from a directory).
        with staged(resolved, FILES) as root, contextlib.redirect_stdout(io.StringIO()):
            model = T0Forecaster.from_pretrained(str(root))
        expected = set(model.state_dict())
        with safetensors.safe_open(str(resolved.path("model.safetensors")), framework="pt") as stored:
            found = set(stored.keys())
        if expected - found:
            raise ValueError(
                f"{resolved.ref}: the weights lack {sorted(expected - found)[:5]} of T0Forecaster's tensors"
            )
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self._base: Optional[Dict[str, Any]] = None
        self._fine_tuned = False

    # --- forecasts ---------------------------------------------------------------------------------------------------

    @staticmethod
    def _arrays(target: np.ndarray, past: Mapping[str, np.ndarray], known: Mapping[str, np.ndarray]):
        """(context (B, V, T), future covariates (B, F, T + S) or None), float32: the target, then the past series."""
        context = np.stack([target, *past.values()], axis=1).astype(np.float32)
        future = np.stack(list(known.values()), axis=1).astype(np.float32) if known else None
        return context, future

    def predict(self, call: Any, options: Mapping[str, Any]) -> np.ndarray:
        context, future = self._arrays(call.target, call.past, call.known)
        with torch.no_grad():
            out = self.model.predict(
                torch.as_tensor(context, device=self.device),
                horizon=int(call.steps_ahead),
                quantile_levels=list(self.levels),
                future_covariates=None if future is None else torch.as_tensor(future, device=self.device),
            )
        quantiles = out.quantiles.float().cpu().numpy()
        return quantiles[:, 0] if quantiles.ndim == 4 else quantiles  # (B, V, S, Q): the target is variate 0

    # --- the fine-tune -----------------------------------------------------------------------------------------------

    def window_losses(
        self, context: np.ndarray, future: Optional[np.ndarray], targets: np.ndarray, steps: int
    ) -> "torch.Tensor":
        """One loss per window of one S (`steps`): the native pinball loss of its finite targets (B, S) over the target
        variate's forecast, divided by its context's mean absolute change plus 1e-3 (t0_ft.py's `loss_fn` per window)."""
        model, device = self.model, self.device
        native = torch.tensor(model.config.quantile_levels, dtype=torch.float32, device=device)
        series, _ = time_series_from_array(
            torch.as_tensor(context, device=device),
            steps,
            device,
            None if future is None else torch.as_tensor(future, device=device),
            None,
            None,
        )
        width = context.shape[-1]
        rollout = RolloutManager(model)
        buffer = rollout.prepare_rollout_buffer(series, steps, width)
        patch = model.patch_size
        ahead = round_up(steps, patch)
        predicted = rollout.predict_step(buffer.time_slice(0, round_up(width, patch) + ahead), ahead)
        predicted = predicted[buffer.variate_type[:, -1] == VariateType.TARGET]
        variates = context.shape[1]
        predicted = predicted.reshape(context.shape[0], variates, *predicted.shape[1:])[:, 0][:, :steps, :]
        finite = np.isfinite(targets)
        y = torch.as_tensor(np.where(finite, targets, 0.0), dtype=torch.float32, device=device)
        error = y[..., None] - predicted
        pinball = torch.maximum(native * error, (native - 1) * error)
        if finite.all():
            per_window = pinball.mean(dim=(1, 2))
        else:
            mask = torch.as_tensor(finite, dtype=torch.float32, device=device)[..., None]
            per_window = (pinball * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) * len(native))
        # A context with a gap (masked by t0) scales by the changes it has; a whole one by t0_ft.py's np.mean, to the bit.
        scale = np.array(
            [(np.nanmean if np.isnan(row).any() else np.mean)(np.abs(np.diff(row))) + 1e-3 for row in context[:, 0]],
            dtype=np.float32,
        )
        return per_window / torch.as_tensor(scale, dtype=torch.float32, device=device)

    def _losses(self, data: FineTuneData, indices: np.ndarray) -> "torch.Tensor":
        """The loss of each window of `indices`, in their order: one forward pass per S."""
        parts: list = [None] * len(indices)
        for steps, positions in data.by_steps(indices).items():
            chosen = [int(indices[position]) for position in positions]
            items = [data.items[i] for i in chosen]
            context = np.stack([np.stack([item.target, *(item.past[n] for n in data.past)]) for item in items]).astype(
                np.float32
            )
            future = (
                np.stack([np.stack([item.known[n] for n in data.known_ahead]) for item in items]).astype(np.float32)
                if data.known_ahead
                else None
            )
            targets = np.stack([data.targets[i] for i in chosen]).astype(np.float32)
            losses = self.window_losses(context, future, targets, steps)
            for k, position in enumerate(positions):
                parts[position] = losses[k]
        return torch.stack(parts)

    def _keep_base(self) -> None:
        if self._base is None:
            self._base = {key: value.detach().to("cpu", copy=True) for key, value in self.model.state_dict().items()}

    def fine_tune(self, data: FineTuneData, settings: Mapping[str, Any], seed: int) -> FineTuneOutcome:
        from .._torch.finetune import FineTuneWindows, fine_tune

        self.reset()
        self._keep_base()
        self._fine_tuned = True
        windows = FineTuneWindows(
            dates=data.dates, cut=data.cut, loss=lambda model, indices: self._losses(data, indices)
        )
        options = {key: settings[key] for key in ("weight_decay", "min_delta", "eval_batch_size") if key in settings}
        result = fine_tune(
            self.model,
            windows,
            lr=settings["lr"],
            max_steps=settings["max_steps"],
            batch_size=settings["batch_size"],
            holdout_days=settings["holdout_days"],
            eval_every=settings["eval_every"],
            patience=settings["patience"],
            grad_clip=settings["grad_clip"],
            seed=int(seed),
            **options,
        )
        return FineTuneOutcome(result.best_step, result.data_end, result.stats)

    def save_state(self, path: Path) -> None:
        state = {key: value.detach().to("cpu").contiguous() for key, value in self.model.state_dict().items()}
        safetensors.torch.save_file(state, str(path))

    def load_state(self, path: Path) -> None:
        self._keep_base()
        self.model.load_state_dict(safetensors.torch.load_file(str(path), device="cpu"))
        self.model.eval()
        self._fine_tuned = True

    def reset(self) -> None:
        if self._fine_tuned and self._base is not None:
            self.model.load_state_dict(self._base)
        self.model.eval()
        self._fine_tuned = False
