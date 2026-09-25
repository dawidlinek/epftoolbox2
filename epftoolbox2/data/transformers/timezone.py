import re
import warnings
from datetime import timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .._market_timezone import resolve as _resolve_timezone
from .._market_timezone import require as _require_timezone
from .base import Transformer

DEFAULT_DAILY_COLUMNS = ["load_forecast_daily_min", "load_forecast_daily_max"]
_UTC_OFFSET = re.compile(r"^(?:UTC)?([+-])(\d{1,2})(?::?(\d{2}))?$", re.IGNORECASE)


class TimezoneTransformer(Transformer):
    """Convert the index to a local time zone and put daily values on local midnight.

    Sources return UTC. After conversion, local days have 23, 24 or 25 hours; nothing is added or removed
    (there is no 02:00 on the spring switch day and two 02:00 rows on the autumn one).

    ``daily_columns`` hold one value per day, e.g. ENTSO-E's week-ahead load forecast min/max (A31). ENTSO-E
    always stamps such a value at the *area's own* local midnight, whatever that area's UTC offset is. In CET
    that instant is 23:00 UTC the day before in winter and 22:00 UTC in summer; a later ``ResampleTransformer``
    would give it to the wrong hours unless it is snapped first. The rule applied: **a value at timestamp t
    belongs to the calendar date of t in ``daily_anchor``, and is moved to 00:00 local time (``target_tz``) on
    that date.** The default anchor is ``target_tz`` itself (None), which is right whenever the daily column
    comes from the market's own area, in every zone — including EET/EEST and zones with a fixed UTC+3 offset,
    which a fixed ``"UTC+02:00"`` anchor gets wrong every summer (or every day, for a zone that never observes
    UTC+2). Pass an explicit ``daily_anchor`` only when the column's real anchor differs from ``target_tz``.

    Raises ``ValueError`` if a daily column holds more than one value for a day (it was probably resampled
    before this transformer; run ``TimezoneTransformer`` first). If the local midnight a value belongs to is
    not in the index, that row is added (all other columns NaN). This is routine at the edges (in winter the
    first A31 value is stamped an hour before the first local day starts); inside the index it means the data
    has a hole there, and a ``UserWarning`` is emitted.

    Args:
        target_tz: Target IANA time zone, e.g. ``"Europe/Warsaw"``. None (default) inherits
            ``DataPipeline.timezone``; transforming without either raises.
        daily_columns: Columns holding one value per day. Defaults to
            ``["load_forecast_daily_min", "load_forecast_daily_max"]``; columns not in the frame are skipped.
        daily_anchor: Time zone whose calendar date a daily value belongs to: a UTC offset such as
            ``"UTC+02:00"`` or an IANA name. None (default) uses the resolved ``target_tz`` itself, matching
            where ENTSO-E stamps A31 data.

    Example:
        >>> transformer = TimezoneTransformer(
        ...     target_tz="Europe/Warsaw",
        ...     daily_columns=["load_forecast_daily_min", "load_forecast_daily_max"],
        ... )
        >>> df = transformer.transform(df)
    """

    # Set by DataPipeline._bind_component_timezones() when `target_tz` is left unset; never written by users.
    _pipeline_timezone: Optional[str] = None

    def __init__(
        self,
        target_tz: Optional[str] = None,
        daily_columns: list[str] | None = None,
        daily_anchor: Optional[str] = None,
    ):
        self.target_tz = target_tz
        self.daily_columns = list(daily_columns) if daily_columns is not None else list(DEFAULT_DAILY_COLUMNS)
        self.daily_anchor = daily_anchor
        self._validate_timezone()
        if daily_anchor is not None:
            _parse_anchor(daily_anchor)  # validate eagerly; re-parsed per transform() since None tracks target_tz

    def _validate_timezone(self) -> None:
        if self.target_tz is None:
            return
        try:
            ZoneInfo(self.target_tz)
        except (KeyError, ValueError):
            raise ValueError(f"Invalid timezone: '{self.target_tz}'")

    @property
    def market_timezone(self) -> Optional[str]:
        """Resolved zone: `target_tz` if set, else the one a `DataPipeline` inherited it from (`None` if neither)."""
        return _resolve_timezone(self.target_tz, self)

    def _require_market_timezone(self) -> str:
        return _require_timezone(self.market_timezone, "TimezoneTransformer.target_tz")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        tz = self._require_market_timezone()
        result = df.copy()

        if result.index.tz is None:
            result.index = result.index.tz_localize("UTC").tz_convert(tz)
        else:
            result.index = result.index.tz_convert(tz)

        for col in self.daily_columns:
            if col in result.columns:
                result = self._snap_to_local_midnight(result, col, tz)

        return result

    def _snap_to_local_midnight(self, result: pd.DataFrame, col: str, tz: str) -> pd.DataFrame:
        values = result[col].dropna()
        if values.empty:
            return result

        anchor = _parse_anchor(self.daily_anchor) if self.daily_anchor is not None else ZoneInfo(tz)
        days = values.index.tz_convert(anchor).tz_localize(None).normalize()
        per_day = days.value_counts()
        crowded = per_day[per_day > 1].sort_index()
        if len(crowded):
            raise ValueError(
                f"TimezoneTransformer: daily column '{col}' has more than one value per day on {len(crowded)} "
                f"day(s), e.g. {crowded.index[0].date()} has {crowded.iloc[0]}. daily_columns must hold one value "
                "per day; if the column was resampled already, run TimezoneTransformer before ResampleTransformer."
            )

        midnights = days.tz_localize(tz, ambiguous=np.ones(len(days), dtype=bool), nonexistent="shift_forward").as_unit(
            result.index.unit
        )
        move = values.index != midnights
        if not move.any():
            return result

        absent = midnights[move].difference(result.index)
        if len(absent):
            # In winter the A31 value for the first local day is stamped an hour before the other sources start,
            # so adding that row is routine. A missing midnight among the other columns' rows is a hole in the data.
            others = result.drop(columns=self.daily_columns, errors="ignore")
            rows = result.index[others.notna().to_numpy().any(axis=1)]
            inside = absent[(absent > rows[0]) & (absent < rows[-1])] if len(rows) else absent[:0]
            if len(inside):
                warnings.warn(
                    f"TimezoneTransformer: the index has no row at local midnight for {len(inside)} value(s) of "
                    f"daily column '{col}' (first: {inside[0]}); those rows were added.",
                    UserWarning,
                    stacklevel=3,
                )
            result = result.reindex(result.index.union(absent))

        result.loc[values.index[move], col] = np.nan
        result.loc[midnights[move], col] = values.to_numpy()[move]
        return result


def _parse_anchor(anchor: str):
    if isinstance(anchor, str):
        text = anchor.strip()
        match = _UTC_OFFSET.match(text)
        if text.upper() == "UTC":
            return timezone.utc
        if match:
            hours, minutes = int(match.group(2)), int(match.group(3) or 0)
            if hours <= 14 and minutes < 60:
                offset = timedelta(hours=hours, minutes=minutes)
                return timezone(-offset if match.group(1) == "-" else offset)
        elif text:
            try:
                return ZoneInfo(text)
            except (KeyError, ValueError):
                pass
    raise ValueError(
        f"Invalid daily_anchor: {anchor!r}. Use a UTC offset such as 'UTC+02:00' or an IANA time zone name"
    )
