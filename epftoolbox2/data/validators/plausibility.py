from collections.abc import Mapping
from fnmatch import fnmatchcase
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ._utils import column_values, matching_columns, true_runs
from .base import Validator
from .result import ValidationResult

# SDAC harmonised clearing price limits in EUR/MWh (HMMCP methodology, ACER): max +4000 since 2022, min -600 from
# trading day 2026-05-28 (it was -500 before). Both move automatically when prices approach them, so override
# `ranges` if they change or for other markets.
SDAC_PRICE_LIMITS = [-600.0, 4000.0]
DEFAULT_RANGES: Dict[str, List[Optional[float]]] = {
    "price*": SDAC_PRICE_LIMITS,
    "load*": [1.0, None],  # MW; zeros reported during outages are not a real load
}
DEFAULT_CONSTANT_COLUMNS = ["price*", "load_actual*"]


class PlausibilityValidator(Validator):
    """Checks that values lie in plausible ranges and don't stay frozen for too long.

    Args:
        ranges: {column or glob pattern: [min, max]}, bounds inclusive, None for no bound. A column uses the first
            pattern it matches. None uses `DEFAULT_RANGES`: `price*` within the SDAC harmonised limits
            (-600 to 4000 EUR/MWh) and `load*` at least 1 MW. `{}` turns the range check off.
        max_constant_run: Longest run of identical consecutive values allowed (in rows) before a warning. None turns
            the check off.
        constant_columns: Columns or glob patterns checked for constant runs. None uses `price*` and
            `load_actual*`; calendar dummies and daily values upsampled to hours are constant by design.

    Values out of range are errors. Long constant runs are warnings: they are typical of a filled outage or a
    frozen feed, but a price can genuinely repeat for hours. Only numeric columns are checked.
    """

    def __init__(
        self,
        ranges: Optional[Dict[str, List[Optional[float]]]] = None,
        max_constant_run: Optional[int] = 24,
        constant_columns: Optional[List[str]] = None,
    ):
        self.ranges = None if ranges is None else self._check_ranges(ranges)
        if max_constant_run is not None:
            if not isinstance(max_constant_run, (int, np.integer)) or max_constant_run < 1:
                raise ValueError(f"max_constant_run must be a positive integer or None, got {max_constant_run!r}")
            max_constant_run = int(max_constant_run)
        self.max_constant_run = max_constant_run
        self.constant_columns = [constant_columns] if isinstance(constant_columns, str) else constant_columns

    @staticmethod
    def _check_ranges(ranges) -> Dict[str, List[Optional[float]]]:
        if not isinstance(ranges, Mapping):
            raise ValueError(f"ranges must be a dict {{column or pattern: [min, max]}}, got {ranges!r}")
        checked = {}
        for pattern, bounds in ranges.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"ranges['{pattern}'] must be [min, max], got {bounds!r}")
            low, high = (None if b is None else float(b) for b in bounds)
            if low is not None and high is not None and low > high:
                raise ValueError(f"ranges['{pattern}']: min {low} is above max {high}")
            checked[str(pattern)] = [low, high]
        return checked

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        result = self._result()

        if df.empty:
            result.warnings.append("DataFrame is empty")
            return result

        numeric = [c for c in dict.fromkeys(df.columns) if pd.api.types.is_numeric_dtype(column_values(df, c))]
        ranges = DEFAULT_RANGES if self.ranges is None else self.ranges
        rules = {}
        for pattern, bounds in ranges.items():
            matched = [c for c in numeric if c not in rules and fnmatchcase(str(c), pattern)]
            if not matched and self.ranges is not None:
                result.warnings.append(f"Range pattern '{pattern}' matches no numeric column")
            for column in matched:
                rules[column] = (pattern, bounds)

        stats: Dict[str, dict] = {}
        out_of_range = {}
        for column, (pattern, (low, high)) in rules.items():
            values = column_values(df, column).to_numpy(dtype=float)
            outside = np.zeros(len(values), dtype=bool)
            if low is not None:
                outside |= values < low
            if high is not None:
                outside |= values > high
            count = int(outside.sum())
            has_values = not np.isnan(values).all()
            vmin = float(np.nanmin(values)) if has_values else np.nan
            vmax = float(np.nanmax(values)) if has_values else np.nan
            stats[column] = {"column": column, "rule": pattern, "low": low, "high": high, "min": vmin, "max": vmax}
            stats[column]["out_of_range"] = count
            if count:
                out_of_range[column] = count
                first = df.index[np.flatnonzero(outside)[0]]
                result.add_error(
                    f"Column '{column}' has {count} values outside [{_bound(low)}, {_bound(high)}] (rule '{pattern}'); "
                    f"first at {first}, min {vmin:g}, max {vmax:g}"
                )
        if out_of_range:
            result.info["out_of_range"] = out_of_range

        if self.max_constant_run is not None:
            patterns = DEFAULT_CONSTANT_COLUMNS if self.constant_columns is None else self.constant_columns
            longest = {}
            for column in matching_columns(numeric, patterns):
                series = column_values(df, column)
                if not series.index.is_monotonic_increasing:
                    series = series.sort_index(kind="stable")
                values = series.to_numpy(dtype=float)
                starts, lengths = true_runs(values[1:] == values[:-1])  # NaN never equals NaN, so gaps break runs
                run, start = (int(lengths.max()) + 1, int(starts[np.argmax(lengths)])) if len(lengths) else (1, 0)
                longest[column] = run
                row = stats.setdefault(column, {"column": column})
                row["longest_constant_run"] = run
                row["constant_value"] = values[start] if len(values) else np.nan
                row["constant_start"] = series.index[start] if len(values) else None
                too_long = int((lengths + 1 > self.max_constant_run).sum())
                if too_long:
                    more = f"; {too_long} runs are longer than {self.max_constant_run} rows" if too_long > 1 else ""
                    result.warnings.append(
                        f"Column '{column}' repeats {values[start]:g} for {run} consecutive rows from "
                        f"{series.index[start]} to {series.index[start + run - 1]} (more than "
                        f"{self.max_constant_run}){more}"
                    )
            result.info["longest_constant_run"] = longest

        result.info["checked_columns"] = list(stats)
        result.stats = pd.DataFrame(list(stats.values()))
        return result


def _bound(value: Optional[float]) -> str:
    return "none" if value is None else f"{value:g}"
