"""`FallbackEnsemble`: a ladder of members, each forecast taken from the first member that has it.

Like `AverageEnsemble` it fits nothing and reads no data: `ensembles._compute` hands it which member has a finite forecast of
which planned key, and it says which member serves each key. Its records are that member's forecasts, with `rung` and `member`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import Choice, Ensemble

PER_DAY = "day"
PER_KEY = "key"
PER = (PER_DAY, PER_KEY)

# What a record of a fallback adds to an ensemble's: the member's place on the ladder (0-based) and its name.
RUNG = "rung"
MEMBER = "member"


class FallbackEnsemble(Ensemble):
    """The forecasts of the first member that has them, in member order: rung 0 first, then rung 1, and so on.

    `per="day"` takes each (issue day, horizon, target day) whole from one member: the first whose forecasts of every planned key of
    that day exist with a finite prediction, so a delivered day is never stitched from two models. A key the clock doesn't have on
    the issue day or the target day isn't planned, so a spring day needs 23 hours (92 quarter-hours). `per="key"` takes each key
    from the first member that has it. A skip record is no forecast, whatever its status (`missing_test_predictors`, or
    `check_failed` from checks before export), so the ladder moves on past it. A day (or key) no member has is missing, counted in
    `forecast_counts()` as `member_missing`.

    The records carry the member's prediction and actual, plus `rung` (0-based) and `member` (its name); their `status` stays
    "ok", so every table reads them as forecasts. `report.rungs(name)` scores each rung on what it served. It chooses nothing on
    validation, so it has forecasts in every window on file, unless a member has test forecasts only (a `top` ensemble), and then
    it has test forecasts only, as a plain average over one has.

    Args:
        members: Names of the models or ensembles of the pipeline, best first: the order is the ladder. A `top` ensemble may be
            one of them; a `FallbackEnsemble` may not be a member of a `top` ensemble.
        per: `"day"` (a whole target day from one member) or `"key"` (each key from the first member that has it).
        name: Names the ensemble in every table and its results file.
    """

    def __init__(self, members: Sequence[str], *, per: str = PER_DAY, name: str):
        super().__init__(members, name)
        self.per = per
        self._check()

    def _check(self) -> None:
        super()._check()
        if self.per not in PER:
            raise ValueError(f"per must be one of {PER}, got {self.per!r}")

    def choose(self, predictions: np.ndarray, actual: np.ndarray) -> Choice:
        raise TypeError(f"{self.name}: a FallbackEnsemble chooses nothing on validation; its order is its members'")

    def combine(self, predictions: np.ndarray) -> np.ndarray:
        """The first member's prediction: with every member finite, rung 0 serves every key."""
        return predictions[:, 0].copy()

    def serve(self, held: np.ndarray, groups: np.ndarray, planned: np.ndarray) -> np.ndarray:
        """The rung that serves each key, -1 where none does.

        `held[i, j]` says whether member j has a finite forecast of key i, which is planned (False for a key the clock doesn't have).
        The keys are in key order, so the keys of one (issue day, target day, horizon) are next to each other: `groups[i]` is that
        day's position, from 0 up without a gap, and `planned[g]` how many keys day g plans, whether any member has them or not.
        """
        if self.per == PER_KEY:
            return np.where(held.any(axis=1), np.argmax(held, axis=1), -1)
        if not len(groups):
            return np.empty(0, dtype=np.int64)
        starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
        whole = np.add.reduceat(held.astype(np.int64), starts, axis=0) == np.asarray(planned)[:, None]
        day_rung = np.where(whole.any(axis=1), np.argmax(whole, axis=1), -1)
        rung = day_rung[groups]
        # A key of a served day that its member lacks isn't planned: no other member fills it in.
        lacking = (rung >= 0) & ~held[np.arange(len(rung)), np.maximum(rung, 0)]
        return np.where(lacking, -1, rung)


__all__ = ["FallbackEnsemble", "MEMBER", "PER_DAY", "PER_KEY", "RUNG"]
