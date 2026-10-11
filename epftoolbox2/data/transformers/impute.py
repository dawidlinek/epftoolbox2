import numpy as np
import pandas as pd

from epftoolbox2.logging import get_logger

from .base import Transformer

logger = get_logger(__name__)

_METHODS = ("ffill", "previous_day", "previous_week")
_SEASONAL_STEP_DAYS = {"previous_day": 1, "previous_week": 7}


class ImputeTransformer(Transformer):
    """Fill missing values with leak-safe methods only.

    Two methods are offered, both causal: a filled value never depends on a row dated later than it.

    * ``"ffill"``: repeats the last valid value forward.
    * ``"previous_day"`` / ``"previous_week"``: uses the value at the same local wall-clock time on the
      previous local calendar day or week. Local days are 23, 24 or 25 hours on a DST switch (see
      ``TimezoneTransformer``); the wall-clock hour is matched, not a fixed 24 h offset. If that day is
      itself missing in the *original* (unimputed) data, the search keeps going back one more day or week
      at a time until it finds a real value.

    Interpolation, backward fill and any other method that could use a later value are refused.

    A value is only ever filled between two real observations of its own column: nothing is invented
    before the column's first observed value, and nothing after its last one, since the rows after it may
    simply not have happened yet.

    Args:
        columns: Column name(s) to impute. Required.
        method: ``"ffill"`` (default), ``"previous_day"`` or ``"previous_week"``.
        limit: How long a gap may be before it is left unfilled, measured from the last real value seen
            before the gap: a positive integer number of rows, a pandas duration string (e.g. ``"6h"``,
            ``"2D"``), or ``None`` for no limit. Default ``None`` (fill every gap the method can reach,
            however long).
        flag: Add a ``{column}_imputed`` column (0/1) marking which values were filled. If that column
            already exists (e.g. this is a second `ImputeTransformer` over the same data), the new flags
            are added to it rather than replacing it. Default ``True``.
        suffix: If non-empty, write the filled series to a new column ``{column}{suffix}`` and leave the
            original column untouched; the flag column is then ``{column}{suffix}_imputed``. Default
            ``""`` (overwrite the column in place).

    Example:
        >>> ImputeTransformer(columns=["price"], method="previous_day", limit="2D").transform(df)
    """

    def __init__(
        self,
        columns: list[str] | str | None = None,
        method: str = "ffill",
        limit: int | str | None = None,
        flag: bool = True,
        suffix: str = "",
    ):
        if not columns:
            raise ValueError("ImputeTransformer: at least one column is required")
        if method not in _METHODS:
            raise ValueError(f"Invalid method: {method!r}. Must be one of {list(_METHODS)}; other methods look ahead")
        _validate_limit(limit)

        self.columns = [columns] if isinstance(columns, str) else list(columns)
        self.method = method
        self.limit = limit
        self.flag = flag
        self.suffix = suffix

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_frame(df)
        result = df.copy()
        for col in self.columns:
            filled, mask = self._impute_column(df[col])
            out_name = f"{col}{self.suffix}"
            result[out_name] = filled
            logger.info("%s: %d value(s) imputed, %d still missing", col, int(mask.sum()), int(filled.isna().sum()))
            if self.flag:
                flag_name = f"{out_name}_imputed"
                new_flag = mask
                if flag_name in df.columns:
                    new_flag = df[flag_name].astype(bool) | mask
                result[flag_name] = new_flag.astype(int)
        return result

    def _check_frame(self, df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        missing = set(self.columns) - set(df.columns)
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {sorted(missing)}")
        if df.index.has_duplicates:
            duplicated = df.index[df.index.duplicated()].unique()
            raise ValueError(
                f"ImputeTransformer: the index has {len(duplicated)} duplicate timestamp(s), e.g. {duplicated[0]}"
            )
        if not df.index.is_monotonic_increasing:
            raise ValueError("ImputeTransformer: the index must be sorted")

    def _impute_column(self, s: pd.Series) -> tuple[pd.Series, pd.Series]:
        fillable, has_data = _fillable_mask(s, self.limit)
        if not has_data:
            return s.copy(), pd.Series(False, index=s.index)

        if self.method == "ffill":
            candidate = s.ffill()
        else:
            candidate = _seasonal_fill(s, _SEASONAL_STEP_DAYS[self.method])

        mask = s.isna() & fillable & candidate.notna()
        filled = s.where(~mask, candidate)
        return filled, mask


def _validate_limit(limit) -> None:
    if limit is None:
        return
    if isinstance(limit, str):
        try:
            duration = pd.Timedelta(limit)
        except (ValueError, TypeError):
            raise _limit_error(limit) from None
        if duration <= pd.Timedelta(0):
            raise _limit_error(limit)
        return
    if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or limit <= 0:
        raise _limit_error(limit)


def _limit_error(limit) -> ValueError:
    return ValueError(
        f"Invalid limit: {limit!r}. Use a positive integer number of rows, a positive pandas duration string "
        "(e.g. '6h', '2D'), or None"
    )


def _fillable_mask(s: pd.Series, limit) -> tuple[pd.Series, bool]:
    """Rows eligible to be filled: not before the column's first value, not after its last, and within
    `limit` of the last real value seen before the row's own gap started."""
    valid = s.notna()
    if not valid.any():
        return pd.Series(False, index=s.index), False

    positions = np.arange(len(s))
    last_valid_pos = positions[valid.to_numpy()][-1]
    anchor_pos = pd.Series(np.where(valid.to_numpy(), positions, np.nan), index=s.index).ffill()
    has_anchor = anchor_pos.notna()
    within_series = pd.Series(positions <= last_valid_pos, index=s.index)

    if limit is None:
        within_limit = pd.Series(True, index=s.index)
    elif isinstance(limit, str):
        anchor_ts = pd.Series(s.index, index=s.index).where(valid).ffill()
        elapsed = pd.Series(s.index, index=s.index) - anchor_ts
        within_limit = elapsed <= pd.Timedelta(limit)
    else:
        run_len = positions - anchor_pos.to_numpy()
        within_limit = pd.Series(run_len <= limit, index=s.index)

    return has_anchor & within_limit & within_series, True


def _seasonal_fill(s: pd.Series, step_days: int) -> pd.Series:
    """Value at the same local wall-clock time `step_days` (or a multiple of it) earlier, walking further
    back through the *original* series until a real value is found."""
    idx = s.index
    local_day = idx.normalize()
    time_of_day = np.asarray(idx.hour) * 3600 + np.asarray(idx.minute) * 60 + np.asarray(idx.second)

    valid = s.notna().to_numpy()
    result = pd.Series(np.nan, index=idx)
    missing = np.flatnonzero(~valid)
    if not valid.any() or len(missing) == 0:
        return result

    # Only real (non-imputed) observations are ever used as a seasonal source; ties (a wall-clock hour that
    # occurs twice on a DST fall-back day) keep the earlier occurrence.
    source = pd.Series(s.to_numpy()[valid], index=pd.MultiIndex.from_arrays([local_day[valid], time_of_day[valid]]))
    source = source[~source.index.duplicated(keep="first")]

    day_arr, tod_arr = local_day[missing], time_of_day[missing]
    unresolved = np.ones(len(missing), dtype=bool)
    found = np.full(len(missing), np.nan)
    max_steps = len(local_day.unique())
    for n in range(1, max_steps + 1):
        if not unresolved.any():
            break
        candidate_days = _shift_local_days(day_arr[unresolved], -step_days * n)
        keys = pd.MultiIndex.from_arrays([candidate_days, tod_arr[unresolved]])
        hits = source.reindex(keys).to_numpy()
        got = ~pd.isna(hits)
        if got.any():
            positions = np.flatnonzero(unresolved)
            found[positions[got]] = hits[got]
            unresolved[positions[got]] = False

    result.iloc[missing] = found
    return result


def _shift_local_days(index: pd.DatetimeIndex, days: int) -> pd.DatetimeIndex:
    """`index` (local calendar-day timestamps) shifted by `days` local calendar days, so the wall-clock
    time is kept across a DST switch (a "day" is 23, 24 or 25 hours long, not a fixed `Timedelta`)."""
    if index.tz is None:
        return index + pd.Timedelta(days=days)
    wall = index.tz_localize(None) + pd.Timedelta(days=days)
    return wall.tz_localize(index.tz, ambiguous=True, nonexistent="shift_forward")
