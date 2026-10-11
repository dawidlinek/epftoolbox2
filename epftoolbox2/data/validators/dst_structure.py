from typing import Optional

import numpy as np
import pandas as pd

from ._utils import ONE_DAY, add_messages, as_utc, format_step, infer_step, to_step
from .base import Validator
from .result import ValidationResult


class DSTStructureValidator(Validator):
    """Checks that every local calendar day has one row per period: 23, 24 or 25 hours × periods per hour.

    Args:
        tz: Time zone that defines the local days, e.g. "Europe/Warsaw". None uses the index's own time zone (UTC
            for a naive index, which the source contract makes UTC). Set it to check a UTC frame straight from the
            sources.
        freq: Period length ("15min", "1h"). None infers it as the most common step.
        allow_partial_edges: Report a first or last day with too few rows as a warning, not an error. Pipeline dates
            are UTC midnights, so the local days at both ends of a frame are usually partial.

    The expected row count of each day comes from the time zone (the spring switch day has 23 hours in Europe, the
    autumn one 25), so a naive wall-clock index, a filled-in DST hour, a duplicated row or a missing hour all show
    up as a day with the wrong count. `stats` lists every local day with its hours, rows and expected rows.
    """

    def __init__(self, tz: Optional[str] = None, freq: Optional[str] = None, allow_partial_edges: bool = True):
        if tz is not None:
            try:
                pd.Timestamp("2000-01-01", tz=tz)
            except Exception as exc:
                raise ValueError(f"Unknown time zone {tz!r}") from exc
        self.tz = tz
        self.freq = freq
        self.allow_partial_edges = allow_partial_edges
        self._step = to_step(freq) if freq is not None else None

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        result = self._result()

        if df.empty:
            result.warnings.append("DataFrame is empty")
            return result

        if not isinstance(df.index, pd.DatetimeIndex):
            result.add_error("Index is not a DatetimeIndex")
            return result

        tz = self.tz if self.tz is not None else df.index.tz
        local = as_utc(df.index).tz_convert(tz if tz is not None else "UTC")
        result.info["tz"] = str(local.tz)

        step = self._step if self._step is not None else infer_step(local)
        if step is None:
            result.warnings.append("Fewer than two distinct timestamps: rows per day can't be checked")
            return result
        result.info["freq"] = self.freq if self.freq is not None else format_step(step)
        if step >= ONE_DAY:
            result.warnings.append(f"Step is {format_step(step)}: rows per local day are checked for sub-daily data")
            return result

        days = local.tz_localize(None).normalize()
        all_days = pd.date_range(days.min(), days.max(), freq="D")
        counts = pd.Series(1, index=days).groupby(level=0).size().reindex(all_days, fill_value=0)

        # Length of each local day from its midnight to the next (23 or 25 hours on DST switch days).
        bounds = all_days.append(all_days[-1:] + ONE_DAY)
        midnights = bounds.tz_localize(
            local.tz, ambiguous=np.ones(len(bounds), dtype=bool), nonexistent="shift_forward"
        )
        lengths = midnights[1:] - midnights[:-1]
        stats = pd.DataFrame(
            {
                "date": all_days.date,
                "hours": np.asarray(lengths / pd.Timedelta(hours=1)),
                "rows": counts.to_numpy(),
                "expected": np.ceil(np.asarray(lengths / step)).astype(int),
            }
        )
        result.stats = stats
        result.info["rows_per_day"] = {int(k): int(v) for k, v in stats["rows"].value_counts().sort_index().items()}
        result.info["dst_days"] = list(stats.loc[stats["hours"] != 24, "date"])

        errors, warnings, bad_days, partial_days = [], [], [], []
        last = len(stats) - 1
        for pos, row in stats[stats["rows"] != stats["expected"]].iterrows():
            day = {"date": row["date"], "rows": int(row["rows"]), "expected": int(row["expected"])}
            message = (
                f"Local day {row['date']} ({row['hours']:g} h in {local.tz}) has {day['rows']} rows, "
                f"expected {day['expected']}"
            )
            if self.allow_partial_edges and pos in (0, last) and day["rows"] < day["expected"]:
                partial_days.append(day)
                warnings.append(f"{message} (partial {'first' if pos == 0 else 'last'} day)")
            else:
                bad_days.append(day)
                errors.append(message)

        if errors:
            result.is_valid = False
            add_messages(result.errors, errors, "days")
        add_messages(result.warnings, warnings, "days")
        result.info["bad_days"] = bad_days
        if partial_days:
            result.info["partial_days"] = partial_days
        return result
