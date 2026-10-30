"""Which days of a frame of forecasts hold all their periods.

A schedule (`EconomicEvaluator`) and a daily aggregate (`EvaluationReport.aggregate`) both need a whole day, and both
judge it here: by the period ids of the day, not by their number.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

# The local hours a clock change can fall in: the only periods a whole day may lack.
_SWITCH_HOURS = (1, 2, 3)


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
            [(ids == template).all(axis=1) & (on_sunday | ~sunday_only) for template, sunday_only in allowed], axis=0
        )
        if whole.any():
            found.append((periods, days[whole]))
    return found
