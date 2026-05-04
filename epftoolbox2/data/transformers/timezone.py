import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from .base import Transformer


class TimezoneTransformer(Transformer):
    """Transformer that converts DataFrame index timezone.

    Optionally snaps sparse daily columns to local midnight after conversion.
    ENTSOE week-ahead (A31) data uses a fixed UTC+2 anchor for period starts,
    so in winter (UTC+1 zones like CET) daily values land at 23:00 local time
    instead of midnight. This transformer rounds each non-null value to the
    *nearest* local midnight (either current or next day) so that a subsequent
    ffill correctly covers each calendar day with its own daily value.

    Example:
        >>> transformer = TimezoneTransformer(
        ...     target_tz="Europe/Warsaw",
        ...     daily_columns=["load_forecast_daily_min", "load_forecast_daily_max"],
        ... )
        >>> df = transformer.transform(df)
    """

    def __init__(self, target_tz: str, daily_columns: list[str] | None = None):
        """
        Args:
            target_tz: Target timezone name (e.g., "Europe/Warsaw", "America/New_York")
            daily_columns: Columns holding one value per day that should be
                re-snapped to local midnight after tz conversion.
                Defaults to ["load_forecast_daily_min", "load_forecast_daily_max"].
        """
        self.target_tz = target_tz
        self.daily_columns = daily_columns if daily_columns is not None else [
            "load_forecast_daily_min", "load_forecast_daily_max"
        ]
        self._validate_timezone()

    def _validate_timezone(self) -> None:
        try:
            ZoneInfo(self.target_tz)
        except KeyError:
            raise ValueError(f"Invalid timezone: '{self.target_tz}'")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        result = df.copy()

        if result.index.tz is None:
            result.index = result.index.tz_localize("UTC").tz_convert(self.target_tz)
        else:
            result.index = result.index.tz_convert(self.target_tz)

        # Snap daily columns to local midnight after tz conversion
        for col in self.daily_columns:
            if col not in result.columns:
                continue

            non_null = result[col].dropna()
            if non_null.empty:
                continue

            # Add 1h before normalizing so that 23:xx snaps to the *next*
            # midnight while 00:xx/01:xx/02:xx still snap to the current one.
            # This handles ENTSOE's UTC+2-anchored period starts, which land at
            # 23:00 local time in UTC+1 (winter) zones instead of at midnight.
            midnight_index = (non_null.index + pd.Timedelta('1h')).normalize()
            needs_shift = non_null.index != midnight_index
            if not needs_shift.any():
                continue

            to_move = non_null[needs_shift]
            target_idx = midnight_index[needs_shift]
            valid = target_idx.isin(result.index)

            result.loc[to_move.index[valid], col] = np.nan
            new_series = pd.Series(to_move.values[valid], index=target_idx[valid])
            result.loc[new_series.index, col] = new_series.values

        return result
