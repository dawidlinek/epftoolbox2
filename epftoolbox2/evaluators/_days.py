"""Which days of a frame of forecasts hold all their periods, and which hours all their quarter-hours.

A schedule (`EconomicEvaluator`) and a daily aggregate (`EvaluationReport.aggregate`) both need a whole day, and both
judge it here: by the period ids of the day, not by their number. An hourly aggregate of quarter-hours
(`aggregate("1h")`), the within-hour shape (`ShapeMAEEvaluator`) and the split of the squared error
(`EvaluationReport.error_split`) need whole hours, judged here too (`whole_hours`).
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .base import FORECAST_KEYS

# The local hours a clock change can fall in: the only periods a whole day may lack.
_SWITCH_HOURS = (1, 2, 3)

# Quarter-hours in an hour, and the hour codes a day key leaves room for (a day numbered through has 25 hours).
QUARTERS = 4
_HOURS_PER_KEY = 25


def day_codes(frame: pd.DataFrame, what: str = "This metric") -> np.ndarray:
    """A dense integer code per row for its (run_date, target_date, horizon): one forecast day of one model.

    Raises ValueError when the frame lacks one of `FORECAST_KEYS` or has a row without one.
    """
    missing = [k for k in FORECAST_KEYS if k not in frame.columns]
    if missing:
        raise ValueError(
            f"{what} needs the forecast keys {list(FORECAST_KEYS)}, but the frame lacks {missing}. Result records need "
            "run_date, target_date, hour and horizon."
        )
    code = np.zeros(len(frame), dtype=np.int64)
    for column in ("run_date", "target_date", "horizon"):
        labels, uniques = pd.factorize(frame[column].to_numpy(), sort=True)
        if (labels < 0).any():
            raise ValueError(f"{what}: a frame has rows with a missing forecast key ({column}).")
        code = code * len(uniques) + labels
    if frame["hour"].isna().any():
        raise ValueError(f"{what}: a frame has rows with a missing forecast key (hour).")
    return pd.factorize(code, sort=True)[0].astype(np.int64)


def hour_codes(day: np.ndarray, period: np.ndarray) -> np.ndarray:
    """A code per row for its hour of its forecast day: `day * 25 + period // 4` (see `whole_hours`)."""
    return np.asarray(day, dtype=np.int64) * _HOURS_PER_KEY + np.asarray(period, dtype=np.int64) // QUARTERS


def whole_hours(day: np.ndarray, period: np.ndarray, finite: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The rows of the hours that hold all four quarter-hours with finite values: `(rows, hour_codes)`.

    `day` is a code per row for its forecast day (`day_codes`), `period` the quarter-hour of the day (0-95, or 0-99 for a
    day numbered through) and `finite` whether the row's prediction and actual are both finite. An hour is `period // 4`
    of its day, and it is whole when its four quarter-hours are there and finite: the repeated autumn hour, which has no
    actual, and an hour with a quarter-hour missing are not. `rows` are positions in the frame, ordered by (day, period),
    four per whole hour, so `values[rows].reshape(-1, 4)` holds one hour per row; `hour_codes` is `day * 25 + hour` of
    each, in the same order. Raises ValueError when a quarter-hour of a day appears twice.
    """
    period = np.asarray(period, dtype=np.int64)
    hour_key = hour_codes(day, period)
    order = np.lexsort((period, hour_key))
    if ((hour_key[order][1:] == hour_key[order][:-1]) & (period[order][1:] == period[order][:-1])).any():
        raise ValueError("Whole hours need each forecast key once; the frame has duplicated keys.")
    kept = order[np.asarray(finite, dtype=bool)[order]]
    codes, counts = np.unique(hour_key[kept], return_counts=True)
    whole = codes[counts == QUARTERS]
    rows = kept[np.isin(hour_key[kept], whole)]
    return rows, whole


def hour_of_code(codes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """`(day code, hour of the day)` of each code `whole_hours` returns."""
    return codes // _HOURS_PER_KEY, codes % _HOURS_PER_KEY


_PER_HOUR = {"1h": 1, "15min": QUARTERS}


def check_period_ids(hour: Sequence[int], freq: str, what: str) -> None:
    """Raise when the period ids are beyond what results at `freq` hold: a day of 25 hours numbered through has ids up to
    24 (99 at 15 minutes), so a larger id is a quarter-hour read as an hour, as plain 15-minute records without a `freq`
    key are (read as "1h", the default). Scoring them would put the peak hours at 02:00-04:59 without a word."""
    per_hour = _PER_HOUR.get(freq)
    if per_hour is None or not len(hour):
        return
    largest = int(np.max(hour))
    if largest >= 25 * per_hour:
        raise ValueError(
            f"{what}: the records have period ids up to {largest}, but {freq} results have at most {25 * per_hour - 1}. "
            'They are probably quarter-hours read as hours: give each record "freq": "15min" (a run writes it).'
        )


def periods_per_hour(hour: Sequence[int]) -> int:
    """4 when the period ids say quarter-hours (an id above 24), else 1: the frame's largest id says what a period is."""
    return 4 if len(hour) and np.max(hour) > 24 else 1


def whole_day_ids(per_hour: int) -> Dict[int, List[Tuple[np.ndarray, bool]]]:
    """`{periods in the day: [(the period ids such a whole day may hold, whether it must be a Sunday)]}`.

    A real run numbers periods by the clock, so a day of 23 hours lacks the hour it skips, also on the day after the spring
    switch and on the autumn switch day, when that hour has no target. Other results may number a short day through, and
    that is also what a day looks like that only lacks its last hour, so it counts only on a Sunday, when clocks change.
    """
    clock = np.arange(24 * per_hour)
    return {
        24 * per_hour: [(clock, False)],
        25 * per_hour: [(np.arange(25 * per_hour), False)],
        23 * per_hour: [
            (np.arange(23 * per_hour), True),
            *((np.delete(clock, np.arange(hour * per_hour, (hour + 1) * per_hour)), False) for hour in _SWITCH_HOURS),
        ],
    }


def whole_days(day: np.ndarray, hour: np.ndarray, sunday: np.ndarray, per_hour: int) -> List[Tuple[int, np.ndarray]]:
    """The days that hold all their periods, by length: `[(periods in the day, day codes in ascending order)]`.

    `day` is a dense integer code per row (0 to n - 1) and `hour` the period id, both sorted by (day, hour), one row per
    period; `sunday` says for each day code whether its target date is a Sunday. A day is whole when its ids are those of
    a day of 23, 24 or 25 hours (92, 96 or 100 quarter-hours); see `whole_day_ids`. The rows of the days of one length
    reshape to `(len(days), periods)`.
    """
    length = np.bincount(day) if len(day) else np.empty(0, dtype=np.int64)
    found: List[Tuple[int, np.ndarray]] = []
    for periods, allowed in whole_day_ids(per_hour).items():
        days = np.flatnonzero(length == periods)
        if not len(days):
            continue
        ids = hour[np.isin(day, days)].reshape(-1, periods)
        on_sunday = sunday[days]
        whole = np.any(
            [(ids == template).all(axis=1) & (on_sunday | (not sunday_only)) for template, sunday_only in allowed],
            axis=0,
        )
        if whole.any():
            found.append((periods, days[whole]))
    return found
