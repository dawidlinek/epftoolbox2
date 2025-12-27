import pandas as pd
from zoneinfo import ZoneInfo
from .base import Transformer


class TimezoneTransformer(Transformer):
    """Transformer that converts DataFrame index timezone.
    Example:
        >>> transformer = TimezoneTransformer(target_tz="Europe/Warsaw")
        >>> df = transformer.transform(df)
    """

    def __init__(self, target_tz: str):
        """
        Args:
            target_tz: Target timezone name (e.g., "Europe/Warsaw", "America/New_York")
        """
        self.target_tz = target_tz
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

        return result
