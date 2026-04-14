import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from .base import Transformer


class TimezoneTransformer(Transformer):
    """Transformer that converts DataFrame index timezone.

    Optionally snaps sparse daily columns to local midnight after conversion.
    When daily values (e.g. load_forecast_daily_min) are stamped at UTC 00:00,
    tz conversion shifts them to 01:00/02:00 local time. A subsequent ffill
    resample would then cover 00:00 of that local day with the *previous*
    day's value. Passing such columns via ``daily_columns`` moves each
    non-null value back to 00:00 of the same local calendar day.

    Example:
        >>> transformer = TimezoneTransformer(
        ...     target_tz="Europe/Warsaw",
        ...     daily_columns=["load_forecast_daily_min", "load_forecast_daily_max"],
        ... )
        >>> df = transformer.transform(df)
    """

    def __init__(self, target_tz: str, daily_columns: list[str] = ["load_forecast_daily_min", "load_forecast_daily_max"]):
        """
        Args:
            target_tz: Target timezone name (e.g., "Europe/Warsaw", "America/New_York")
            daily_columns: Columns holding one value per day that should be
                re-snapped to local midnight after tz conversion.
        """
        self.target_tz = target_tz
        self.daily_columns = daily_columns
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

            midnight_index = non_null.index.normalize()
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
