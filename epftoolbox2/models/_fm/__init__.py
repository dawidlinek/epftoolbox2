"""The backends of the foundation-model families (step 6i; 6j adds PatchTST-FM, Timestral and TimesFM-3 here).

A family's heavy calls sit behind a small `Backend` (decision 23), so every library-side line runs in CI with a deterministic
numpy fake, and with the real package in the extras jobs. A backend is built by its model class (`FoundationModel._load_backend`)
once per process, from the weights the pre-flight resolved (`ResolvedWeights`: each file opened by name, never a directory as a
whole) inside `_weights.offline()`, and is called with history batches (`models._history.Batch`):

- `predict(call) -> (batch_size, S, len(levels))`: the quantiles of each step ahead of each item of a call (target `(B, T)`,
  past `{name: (B, T)}`, known-ahead `{name: (B, T + S)}`, S = `call.steps_ahead`), float, at `levels`. The model's forecast is
  the 0.5 level of them (decision 24); a family without one names its point level in `point_level`. Each item's output must not
  depend on the other items of its call (verify's co-batch check).
- `fine_tune(data, settings, seed) -> FineTuneOutcome` (families with `FoundationModel._can_fine_tune`): fine-tune from the base
  weights on `data` (`FineTuneData`, whose newest target is at or before its cut, asserted by the model before the call and by
  the trainer), leaving the backend on the best state; `save_state(path)` writes that state, `load_state(path)` puts a stored one
  back, and `reset()` the base weights, which every fine-tune starts from (decision 13: never from the previous anchor's).

Which inputs a family takes is the model class's (`_takes_known`, `_takes_past`, `_accepts_missing`), so a univariate family
refuses covariates when it is built, without its package. Nothing here imports an extra; `t0` and `chronos` do, and only the
model's `_load_backend` imports them.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple

import numpy as np

# The quantile levels the research asked every family for (`zoo_tsfm/code/zoo.py` LEVELS: 0.05 to 0.95 by 0.05, the median at
# index 9). The median of a family whose native levels hold 0.5 is that native level, whichever others are asked for.
LEVELS_19: Tuple[float, ...] = tuple(round(0.05 * k, 2) for k in range(1, 20))


@dataclass(frozen=True, eq=False)
class FineTuneData:
    """The windows of one fine-tune: window i is the forecast issued on `items[i].issue_day` at the unit's horizon, read
    through the history accessor as a forecast issued then (its context cut by the information set at its own issue time),
    with `targets[i]` (S_i,) the target on its steps ahead, NaN on a step that forecasts no period of the target day and where
    the data has none, and `dates[i]` its target day (local). `cut` is the newest local target date the fine-tune may read: a
    window after it raises before any step."""

    items: Tuple[Any, ...]
    targets: Tuple[np.ndarray, ...]
    dates: np.ndarray  # datetime64[D]
    cut: Any  # datetime.date
    past: Tuple[str, ...] = ()
    known_ahead: Tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.items)

    def by_steps(self, indices: Sequence[int]) -> Dict[int, list]:
        """The positions in `indices` (in their order) of each S, smallest S first: one forward pass per S."""
        out: Dict[int, list] = {}
        for position, index in enumerate(indices):
            out.setdefault(self.items[int(index)].steps_ahead, []).append(position)
        return dict(sorted(out.items()))


@dataclass
class FineTuneOutcome:
    """What a backend's fine-tune gives back: the best step (0: the base weights were best), the newest target date it read,
    and the trainer's statistics (holdout losses, steps run, window counts)."""

    best_step: int
    data_end: Any
    stats: Dict[str, Any] = field(default_factory=dict)


class Backend:
    """A foundation-model family's heavy calls (see the module's docstring). `levels` are the quantile levels `predict`
    returns, sorted; `point_level` the one the forecast is (0.5)."""

    levels: Tuple[float, ...] = (0.5,)
    point_level: float = 0.5

    def predict(self, call: Any) -> np.ndarray:  # pragma: no cover - every family defines it
        raise NotImplementedError

    def fine_tune(self, data: FineTuneData, settings: Mapping[str, Any], seed: int) -> FineTuneOutcome:
        raise NotImplementedError(f"{type(self).__name__} doesn't fine-tune")

    def save_state(self, path: Path) -> None:
        raise NotImplementedError(f"{type(self).__name__} doesn't fine-tune")

    def load_state(self, path: Path) -> None:
        raise NotImplementedError(f"{type(self).__name__} doesn't fine-tune")

    def reset(self) -> None:
        """Back to the base weights (a family that doesn't fine-tune never leaves them)."""

    @property
    def point_index(self) -> int:
        """Where `point_level` is among `levels`."""
        levels = [float(level) for level in self.levels]
        if self.point_level not in levels:
            raise ValueError(f"{type(self).__name__}: its levels {levels} hold no {self.point_level}")
        return levels.index(self.point_level)


@contextlib.contextmanager
def staged(resolved: Any, names: Sequence[str]) -> Iterator[Path]:
    """A temporary directory holding a link to each of `names` of `resolved` (`ResolvedWeights.path(name)`) and nothing else,
    for a package whose loader takes a directory (`from_pretrained`): it sees exactly the files the card names, never a file
    that happens to lie next to them (an adapter config, another checkpoint). Copies where a link can't be made. Removed on
    exit."""
    root = Path(tempfile.mkdtemp(prefix="epftoolbox2-weights-"))
    try:
        for name in names:
            source = Path(resolved.path(name)).resolve()
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(source, target)
            except OSError:
                shutil.copyfile(source, target)
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


__all__ = ["LEVELS_19", "Backend", "FineTuneData", "FineTuneOutcome", "staged"]
