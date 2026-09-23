from typing import Optional

import pandas as pd

from .base import Transformer

# ENTSO-E day-ahead wind and solar forecasts (A69), as EntsoeSource names them.
_RENEWABLE_FORECASTS = (
    "generation_solar_forecast",
    "generation_wind_onshore_forecast",
    "generation_wind_offshore_forecast",
)


class ResidualLoadTransformer(Transformer):
    """Residual load forecast: load forecast minus wind and solar generation forecasts.

    Built from forecast columns only; a column whose name doesn't contain "forecast" is rejected, because
    realised load or generation would make the result look ahead. A missing value in any input gives a
    missing result (a missing forecast is not treated as zero).

    The result is only as available as its inputs. ENTSO-E's day-ahead wind and solar forecast for D+1 (A69)
    may be published as late as 18:00 on D, after a forecast issued in the morning of D, so using
    `residual_load_forecast_d+1` there assumes it was available in time.

    Args:
        load: Load forecast column. Default "load_forecast".
        renewables: Wind and solar forecast columns to subtract. If None, uses whichever of
            generation_solar_forecast, generation_wind_onshore_forecast and generation_wind_offshore_forecast
            are present (at least one must be).
        output: Name of the new column. Default "residual_load_forecast".

    Example:
        >>> df = ResidualLoadTransformer().transform(df)  # adds residual_load_forecast
    """

    def __init__(
        self,
        load: str = "load_forecast",
        renewables: Optional[list[str]] = None,
        output: str = "residual_load_forecast",
    ):
        self.load = load
        self.renewables = (
            None if renewables is None else ([renewables] if isinstance(renewables, str) else list(renewables))
        )
        self.output = output
        not_forecasts = [c for c in [load, *(self.renewables or [])] if "forecast" not in c.lower()]
        if not_forecasts:
            raise ValueError(
                f"ResidualLoadTransformer uses forecast columns only; {not_forecasts} look like realised data"
            )
        if self.renewables is not None and not self.renewables:
            raise ValueError("renewables must name at least one column (or be None)")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.renewables is None:
            renewables = [c for c in _RENEWABLE_FORECASTS if c in df.columns]
            if not renewables:
                raise ValueError(
                    f"No wind or solar forecast column found; expected any of {list(_RENEWABLE_FORECASTS)}"
                )
        else:
            renewables = self.renewables
        missing = [c for c in [self.load, *renewables] if c not in df.columns]
        if missing:
            raise ValueError(f"ResidualLoadTransformer: columns not found: {missing}")
        if self.output in df.columns:
            raise ValueError(f"ResidualLoadTransformer: column '{self.output}' already exists")

        # Plain arithmetic, not sum(): a missing input must give a missing result.
        residual = df[self.load].astype(float)
        for column in renewables:
            residual = residual - df[column].astype(float)
        result = df.copy()
        result[self.output] = residual
        return result
