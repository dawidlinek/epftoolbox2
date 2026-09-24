from typing import Optional

import numpy as np
import pandas as pd

from ._utils import ONE_DAY, MAX_LISTED, add_messages, format_step, infer_step, to_step
from .base import Validator
from .result import ValidationResult


class ContinuityValidator(Validator):
    """Checks that the index is a sorted grid with one step, no duplicates and no missing timestamps.

    Args:
        freq: Expected step ("15min", "1h", "1D"). None infers it as the most common step in the data.

    Steps are measured in elapsed time, so a tz-aware local index is continuous across DST switches (the local clock
    skips or repeats an hour, the elapsed step stays 1h). Steps of a day or more on a tz-aware index are measured on
    the local calendar, where a day lasts 23 or 25 hours on switch days. A naive index is compared as it is, so a
    naive local wall-clock index shows the DST switches as a gap and a duplicate.

    Errors: duplicate timestamps, an unsorted index, gaps (steps longer than `freq`) and steps that are not a whole
    multiple of `freq` (mixed resolutions or an off-grid index). Only whole columns of missing rows are seen here;
    use `CoverageValidator` for values missing in single columns.
    """

    def __init__(self, freq: Optional[str] = None):
        self.freq = freq
        self._step = to_step(freq) if freq is not None else None

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        result = self._result()

        if df.empty:
            result.warnings.append("DataFrame is empty")
            return result

        if not isinstance(df.index, pd.DatetimeIndex):
            result.add_error("Index is not a DatetimeIndex")
            return result

        index = df.index
        duplicated = index.duplicated()
        if duplicated.any():
            repeated = index[duplicated].unique()
            result.add_error(
                f"Index has {int(duplicated.sum())} duplicate timestamps (first: {repeated[0]}, "
                f"{len(repeated)} distinct timestamps repeated)"
            )
            result.info["duplicate_count"] = int(duplicated.sum())
            result.info["duplicates"] = list(repeated[:MAX_LISTED])

        unique = index[~duplicated]
        if not unique.is_monotonic_increasing:
            backward = np.flatnonzero(np.diff(unique.asi8) < 0)
            first = backward[0]
            result.add_error(
                f"Index is not sorted: {len(backward)} timestamps are earlier than the row before "
                f"(first: {unique[first]} followed by {unique[first + 1]})"
            )
            result.info["unsorted_count"] = len(backward)
            unique = unique.sort_values()

        step = self._step if self._step is not None else infer_step(unique)
        if step is None:
            result.warnings.append("Fewer than two distinct timestamps: the step can't be checked")
            return result
        freq = self.freq if self.freq is not None else format_step(step)
        result.info["expected_freq"] = freq
        result.info["freq_inferred"] = self.freq is None

        # Day-or-longer steps on a local index are compared on the wall clock (a local day may last 23 or 25 hours).
        times = unique.tz_localize(None) if unique.tz is not None and step % ONE_DAY == pd.Timedelta(0) else unique
        deltas = times[1:] - times[:-1]

        irregular = np.flatnonzero((deltas % step) != pd.Timedelta(0))
        if len(irregular):
            examples = [{"start": unique[i], "end": unique[i + 1], "step": deltas[i]} for i in irregular[:MAX_LISTED]]
            first = examples[0]
            result.add_error(
                f"{len(irregular)} steps are not a multiple of {freq} (first: {first['start']} to {first['end']}, "
                f"{format_step(first['step'])}); the index mixes resolutions or is off the {freq} grid"
            )
            result.info["irregular_steps"] = len(irregular)
            result.info["irregular"] = examples
        else:
            result.info["irregular_steps"] = 0

        gap_positions = np.flatnonzero(deltas > step)
        gaps, messages = [], []
        for i in gap_positions:
            missing = int(np.ceil(deltas[i] / step)) - 1
            gaps.append({"start": unique[i], "end": unique[i + 1], "duration": deltas[i], "missing": missing})
            messages.append(
                f"Gap detected (expected {freq} frequency): no rows between {unique[i]} and {unique[i + 1]} "
                f"({missing} missing timestamps)"
            )
        if gaps:
            result.is_valid = False
            add_messages(result.errors, messages, "gaps")
            result.info["gaps"] = gaps
            result.info["gap_count"] = len(gaps)
            result.info["missing_periods"] = sum(g["missing"] for g in gaps)

        return result
