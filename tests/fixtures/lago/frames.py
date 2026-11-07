"""The LEAR frame of a Lago benchmark market, built by the documented chain (Predictor Presets), from the fixture's data or from
the cached Zenodo file.

`tests/models/test_lear_lago.py` reads the fixture (`FixtureSource`), the reproduction outside CI
(`epftoolbox2-research/library_check_step4/lear/`) the whole file through `EpftoolboxBenchmarkSource`: both through
`lear_pipeline`, so they build the same columns the same way.
"""

from pathlib import Path

import pandas as pd

from epftoolbox2.data.sources import CalendarSource, EpftoolboxBenchmarkSource
from epftoolbox2.data.transformers import (
    DailyProfileTransformer,
    DSTNormalizeTransformer,
    LagTransformer,
    TimezoneTransformer,
)
from epftoolbox2.pipelines import DataPipeline
from epftoolbox2.presets import WEEKDAYS, profile_columns

HERE = Path(__file__).resolve().parent

# The two exogenous series of each market in the order of its file, which `LEARModel.lago` needs (its docstring lists them).
EXOGENOUS = {
    "NP": ["load_forecast", "wind_forecast"],
    "PJM": ["load_forecast", "zonal_load_forecast"],
    "FR": ["generation_forecast", "load_forecast"],
    "BE": ["generation_forecast_fr", "load_forecast_fr"],
    "DE": ["load_forecast_amprion", "wind_solar_forecast"],
}
# The country `CalendarSource` is given for the weekday dummies. With `holidays=False` it makes those alone, which depend on the
# local date only, not on the country; Nord Pool and PJM have no entry of their own, so they borrow one.
CALENDAR_COUNTRY = {"NP": "NO", "PJM": "DE", "FR": "FR", "BE": "BE", "DE": "DE"}
# The issue days of the fixture: the day before the first target day of Lago's test period.
FIXTURE_ISSUE_DAYS = {"DE": "2016-01-03", "BE": "2015-01-03"}


class FixtureSource(EpftoolboxBenchmarkSource):
    """`EpftoolboxBenchmarkSource` over `data_{market}.parquet`: the file's three columns as pandas 1.5 parsed them for the
    original, on the file's naive wall clock, mapped to UTC as the source maps the file."""

    def __init__(self, dataset: str, path: Path = None):
        super().__init__(dataset)
        self._parquet = Path(path) if path is not None else HERE / f"data_{self.dataset}.parquet"

    def _load(self) -> pd.DataFrame:
        raw = pd.read_parquet(self._parquet)
        frame = raw.astype(float)
        index = pd.DatetimeIndex(raw.index)
        frame.index = index.tz_localize(self.index_timezone).tz_convert("UTC").rename("datetime")
        return frame


def lear_pipeline(source: EpftoolboxBenchmarkSource, horizon: int = 1) -> DataPipeline:
    """The chain of the Predictor Presets page on a benchmark source: weekday dummies, 24 slots a day, the profiles of the price
    and of both exogenous series, the price lagged 1, 2 and 6 days, the exogenous series 6 days and -1 … -horizon days, and the
    dummies -1 … -horizon days."""
    exogenous = EXOGENOUS[source.dataset]
    leads = list(range(-1, -horizon - 1, -1))
    zone = source.index_timezone
    return (
        DataPipeline(timezone=zone)
        .add_source(source)
        .add_source(CalendarSource(CALENDAR_COUNTRY[source.dataset], weekday="onehot", holidays=False, timezone=zone))
        .add_transformer(TimezoneTransformer())
        .add_transformer(DSTNormalizeTransformer(freq="1h"))
        .add_transformer(DailyProfileTransformer(columns=["price", *exogenous]))
        .add_transformer(LagTransformer(columns=profile_columns("price"), lags=[1, 2, 6], freq="day"))
        .add_transformer(
            LagTransformer(columns=[c for x in exogenous for c in profile_columns(x)], lags=[*leads, 6], freq="day")
        )
        .add_transformer(LagTransformer(columns=[f"is_{d}" for d in WEEKDAYS], lags=leads, freq="day"))
    )


def lear_frame(source: EpftoolboxBenchmarkSource, first: str, last: str, horizon: int = 1) -> pd.DataFrame:
    """`lear_pipeline(source, horizon)` run over the local dates `first` to `last`, without the DST flag column."""
    frame = lear_pipeline(source, horizon).run(first, last, cache=False)
    return frame.drop(columns=["is_dst_adjusted"], errors="ignore")
