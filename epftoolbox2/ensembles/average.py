from __future__ import annotations

import numbers
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .base import Choice, Ensemble

VALIDATION = "validation"

# MAEs are compared at this many significant digits: the noise of averaging identical members is a tie, and rounding each
# number once, unlike a tolerance between two of them, keeps the comparison transitive.
_COMPARED_DIGITS = 12


def _compared(mae: float) -> float:
    return float(f"{mae:.{_COMPARED_DIGITS - 1}e}")


def _mae(prediction: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - actual)))


class AverageEnsemble(Ensemble):
    """The mean of the members' predictions, on the forecast keys all the members it averages hold.

    `top=None` averages every member and has forecasts in every window on file. Otherwise it averages the members chosen on
    validation and has forecasts in the test window alone: its validation score is in sample, and it lives in
    `report.choices()`.

    - `top=k` keeps the k members with the lowest validation MAE, and `top=1` means "pick the best".
    - `top="validation"` chooses k as well: the k whose average of the best k members has the lowest validation MAE.

    What counts for choosing, per test period served (with `validation="previous_year"`, one per 12 months): forecasts that
    period's validation serves (`Segment.serves`: issued in it, for a day before its first test issue day), on the keys every
    member holds with a finite prediction and a finite actual. MAEs are compared at 12 significant digits. A tie goes to the
    earlier member when ranking and to the smaller k when choosing k. The choice is frozen for the test period it serves,
    while the members keep refitting.

    The average is `np.mean` over the chosen members in member order, whatever the ranking, so `top=k` equals an
    `AverageEnsemble` of what it chose, bit for bit. A key that a chosen member has no finite prediction for is missing for the
    ensemble and counted in `forecast_counts()` as `member_missing`. It is never averaged over the rest.

    Args:
        members: Names of the models or ensembles to average. An ensemble may not be a member of a `top` ensemble, which
            would rank it on forecasts already chosen on the same days. A plain average of ensembles is fine.
        top: None, a number of members from 1 to `len(members)`, or `"validation"`.
        by: Where `top` chooses: `"validation"`, the only value. The test period never chooses anything.
        name: Names the ensemble in every table and its results file.
    """

    def __init__(
        self,
        members: Sequence[str],
        top: Union[int, str, None] = None,
        by: str = VALIDATION,
        name: str = "avg",
    ):
        super().__init__(members, name)
        self.top = int(top) if isinstance(top, numbers.Integral) and not isinstance(top, bool) else top
        self.by = by
        self._check()

    def _check(self) -> None:
        super()._check()
        if self.by != VALIDATION:
            raise ValueError(f"by must be {VALIDATION!r}: the test period never chooses anything. Got {self.by!r}")
        if self.top is None or self.top == VALIDATION:
            return
        if isinstance(self.top, bool) or not isinstance(self.top, int):
            raise ValueError(f"top must be None, a number of members or {VALIDATION!r}, got {self.top!r}")
        if not 1 <= self.top <= len(self.members):
            raise ValueError(f"top must be from 1 to {len(self.members)}, the number of members, got {self.top}")

    @property
    def needs_validation(self) -> bool:
        return self.top is not None

    def choose(self, predictions: np.ndarray, actual: np.ndarray) -> Choice:
        count = len(self.members)
        member_mae = [_mae(predictions[:, j], actual) for j in range(count)]
        ranked = sorted(range(count), key=lambda j: (_compared(member_mae[j]), j))
        k_mae: Optional[Dict[int, float]] = None
        if self.top == VALIDATION:
            k_mae = {size: self._mae_of(predictions, actual, ranked[:size]) for size in range(1, count + 1)}
            k = min(k_mae, key=lambda size: (_compared(k_mae[size]), size))
        else:
            k = count if self.top is None else self.top
        kept = sorted(ranked[:k])
        return Choice(
            chosen=[self.members[j] for j in kept],
            mae=self._mae_of(predictions, actual, kept),
            member_mae=dict(zip(self.members, member_mae)),
            k_mae=k_mae,
        )

    def combine(self, predictions: np.ndarray) -> np.ndarray:
        return np.mean(predictions, axis=1)

    def _mae_of(self, predictions: np.ndarray, actual: np.ndarray, columns: List[int]) -> float:
        """Member order, as `combine` gets them on test: the average of what was chosen doesn't depend on how it ranked."""
        return _mae(self.combine(predictions[:, sorted(columns)]), actual)
