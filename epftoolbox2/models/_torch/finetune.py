"""Fine-tuning a pretrained network on the windows of one anchor, with early stopping on its newest dates.

The recipe of the research's t0 fine-tune (`epftoolbox2-research/zoo_tsfm/code/t0_ft.py`): AdamW (weight decay 0 by default), a
batch of distinct training windows drawn per step from numpy's generator seeded by `seed`, the gradient norm clipped, the
loss on the holdout (the windows whose target date is in the last `holdout_days` calendar days of the data) evaluated before
the first step and every `eval_every` steps, a step kept as best when it beats the best by more than `min_delta`, a stop after
`patience` evaluations without one, and the best state restored. A window is the unit of the loss: `FineTuneWindows.loss(model,
indices)` returns one loss per window, whatever the model (quantile loss divided by the context's scale for t0).

The library asserts the cut: a window whose target date is after `windows.cut` (the newest target the anchor's information
allows) raises `AssertionError` before any step, so a fine-tune can never train on a target it couldn't know.
"""

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..._extras import MissingExtraError

try:
    import torch
except ImportError:  # pragma: no cover - the core-only CI job has no torch
    raise MissingExtraError("the torch fine-tune", "torch") from None


def _day(value: Any) -> np.datetime64:
    return np.datetime64(value, "D") if not isinstance(value, dt.datetime) else np.datetime64(value.date(), "D")


@dataclass(frozen=True)
class FineTuneWindows:
    """The windows of one fine-tune: `dates[i]` is the target date of window i (the newest target its loss reads), `cut` the
    newest target date the fine-tune may read, and `loss(model, indices)` a tensor of one loss per window of `indices` (an
    int array), differentiable in the model's parameters."""

    dates: Sequence[Any]
    cut: Any
    loss: Callable[["torch.nn.Module", np.ndarray], "torch.Tensor"]


@dataclass
class FineTuneResult:
    """The best state (`state_dict`, detached copies), the step it was reached at (0: the starting weights were best), the
    newest target date the fine-tune read (`data_end`, a `datetime.date`), and `stats` (the holdout loss at step 0 and at each
    evaluation, the steps run, the counts of training and holdout windows)."""

    state_dict: Dict[str, "torch.Tensor"]
    best_step: int
    data_end: dt.date
    stats: Dict[str, Any] = field(default_factory=dict)


def split_holdout(dates: Sequence[Any], holdout_days: int) -> Tuple[np.ndarray, np.ndarray, np.datetime64]:
    """(training windows, holdout windows, data end): the holdout is every window whose date is in the last `holdout_days`
    calendar days up to the newest date (`data end`), the training windows the rest."""
    days = np.array([_day(d) for d in dates], dtype="datetime64[D]")
    if len(days) == 0:
        raise ValueError("fine_tune needs at least one window")
    end = days.max()
    hold = days > end - np.timedelta64(int(holdout_days), "D")
    return np.flatnonzero(~hold), np.flatnonzero(hold), end


def fine_tune(
    model: "torch.nn.Module",
    windows: FineTuneWindows,
    *,
    lr: float = 1e-5,
    max_steps: int = 400,
    batch_size: int = 16,
    holdout_days: int = 61,
    eval_every: int = 25,
    patience: int = 4,
    grad_clip: float = 1.0,
    seed: int = 0,
    weight_decay: float = 0.0,
    min_delta: float = 1e-4,
    eval_batch_size: int = 32,
) -> FineTuneResult:
    """Fine-tune `model` in place on `windows` and return its best state (also loaded into `model`, left in eval mode).

    Every random draw comes from `seed`: the batches (numpy's generator) and torch's generator (for dropout), forked so the
    caller's stays as it was. `grad_clip=0` clips nothing."""
    days = np.array([_day(d) for d in windows.dates], dtype="datetime64[D]")
    cut = _day(windows.cut)
    late = np.flatnonzero(days > cut)
    if len(late):
        raise AssertionError(
            f"fine_tune: {len(late)} window(s) have a target after the cut {cut} (the first: window {late[0]}, target "
            f"{days[late[0]]}); a fine-tune may only read targets known at the anchor's issue time"
        )
    if int(max_steps) < 0 or int(batch_size) < 1 or int(eval_every) < 1 or int(patience) < 1 or int(holdout_days) < 1:
        raise ValueError("fine_tune needs max_steps >= 0 and batch_size, eval_every, patience, holdout_days >= 1")
    train, hold, end = split_holdout(days, int(holdout_days))
    if len(train) == 0:
        raise ValueError(f"fine_tune: every window is in the {holdout_days}-day holdout; give it older windows")
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("fine_tune: the model has no parameter that requires grad")
    device = params[0].device
    devices = [device.index or 0] if device.type == "cuda" else []
    rng = np.random.default_rng(int(seed))
    batch = min(int(batch_size), len(train))

    def holdout_loss() -> float:
        model.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(hold), int(eval_batch_size)):
                total += float(windows.loss(model, hold[start : start + int(eval_batch_size)]).sum())
        return total / len(hold)

    history: List[Tuple[int, float]] = []
    with torch.random.fork_rng(devices=devices), torch.enable_grad():
        torch.manual_seed(int(seed))
        opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))
        base = holdout_loss()
        history.append((0, base))
        best, best_step, bad, step = base, 0, 0, 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for step in range(1, int(max_steps) + 1):
            model.train()
            chosen = train[rng.choice(len(train), batch, replace=False)]
            loss = windows.loss(model, chosen).mean()
            opt.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
            opt.step()
            if step % int(eval_every):
                continue
            h = holdout_loss()
            history.append((step, h))
            if h < best - float(min_delta):
                best, best_step, bad = h, step, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= int(patience):
                    break
    model.load_state_dict(best_state)
    model.eval()
    stats = {
        "holdout": history,
        "base_loss": base,
        "best_loss": best if math.isfinite(best) else math.nan,
        "steps": step,
        "train_windows": int(len(train)),
        "holdout_windows": int(len(hold)),
    }
    return FineTuneResult(state_dict=best_state, best_step=best_step, data_end=end.astype(dt.date), stats=stats)
