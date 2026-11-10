"""TimesFM-3's backend (`TimesFMModel`): `timesfm[torch]==3.0.2`'s `TimesFM3Forecaster`, loaded from the card's two files (its
hub mixin reads the directory's `config.json`, so the architecture is the checkpoint's own), called as the research's adapter
called it (`zoo_tsfm/code/zoo_extra.py` `TimesFM3`).

- **Inputs**: each item's context `(T,)` float32; each past series a row of its `past_only_covariates` `(P, T)`, each
  known-ahead series a row of its `past_future_covariates` `(F, T + S)`, in the model's order. The package fills a gap by
  linear interpolation (a gap at an end takes the nearest value: nothing after a series' newest known step), drops leading NaN
  steps and reads at most 15,360 steps.
- **Levels**: its nine deciles, sorted by the package per step; the forecast is the median, index 4.
- **Calls**: one `predict_batch` per call of the model's `batch_size` items, in one batch of that size.

Needs the `[timesfm]` extra; importing this module imports torch and timesfm3.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

import numpy as np

from ..._extras import MissingExtraError
from . import Backend, staged

try:
    import torch  # noqa: F401 - the forecaster's own torch, imported here so the extra is checked before it loads
    from timesfm3.torch.timesfm3_forecaster import TimesFM3Forecaster
except ImportError:  # pragma: no cover - the core-only CI job has no timesfm
    raise MissingExtraError("TimesFMModel", "timesfm") from None

FILES = ("config.json", "model.safetensors")
DECILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class TimesFMBackend(Backend):
    """`TimesFM3Forecaster` on `device`, from `resolved` (`config.json` and `model.safetensors`, each by name)."""

    levels = DECILES

    def __init__(self, resolved: Any, device: Any):
        with staged(resolved, FILES) as root:
            self.forecaster = TimesFM3Forecaster.from_pretrained(str(root), device=str(device), local_files_only=True)
        levels = tuple(round(float(level), 6) for level in self.forecaster.config.quantiles)
        if levels != DECILES:
            raise ValueError(f"TimesFM-3's quantile levels are {levels}, not the deciles {DECILES}")

    def predict(self, call: Any, options: Mapping[str, Any]) -> np.ndarray:
        steps = int(call.steps_ahead)
        size = call.target.shape[0]
        contexts = [call.target[b].astype(np.float32) for b in range(size)]
        past = (
            [np.stack([v[b] for v in call.past.values()]).astype(np.float32) for b in range(size)]
            if call.past
            else None
        )
        known = (
            [np.stack([v[b] for v in call.known.values()]).astype(np.float32) for b in range(size)]
            if call.known
            else None
        )
        # The whole call in one batch of its size: the package batches by `per_core_batch_size`.
        self.forecaster.config = dataclasses.replace(self.forecaster.config, per_core_batch_size=size)
        outputs = self.forecaster.predict_batch(
            contexts=contexts,
            horizon=steps,
            past_only_covariates=past,
            past_future_covariates=known,
            return_quantiles=True,
        )
        return np.stack([np.asarray(output.quantiles, dtype=np.float32)[:steps] for output in outputs])
