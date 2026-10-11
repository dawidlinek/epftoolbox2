import datetime
import multiprocessing
import os
import time
from unittest.mock import Mock

import numpy as np
import pytest
import pandas as pd

from epftoolbox2.data import cache_manager as cache_manager_module
from epftoolbox2.data.sources.base import DataSource
from epftoolbox2.data.sources.open_meteo import OpenMeteoSource
from epftoolbox2.data.transformers.base import Transformer
from epftoolbox2.data.transformers import TimezoneTransformer
from epftoolbox2.data.validators.base import Validator
from epftoolbox2.data.validators.result import ValidationResult
from epftoolbox2.pipelines import DataPipeline
from epftoolbox2.pipelines.data_pipeline import DataValidationError


class MockDataSource(DataSource):
    """Mock data source for testing"""

    def __init__(self, data: pd.DataFrame = None, prefix: str = ""):
        self.data = data
        self.prefix = prefix
        self.fetch_called = False
        self.fetch_args = None

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        self.fetch_called = True
        self.fetch_args = (start, end)
        if self.data is None:
            return pd.DataFrame()
        return self.data.copy()

    def _validate_config(self) -> bool:
        return True

    def get_cache_config(self) -> dict:
        return {"source_type": "mock", "prefix": self.prefix}


class MockTransformer(Transformer):
    """Mock transformer for testing"""

    def __init__(self, suffix: str = "_transformed"):
        self.suffix = suffix
        self.transform_called = False

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.transform_called = True
        result = df.copy()
        result.columns = [f"{col}{self.suffix}" for col in result.columns]
        return result


@pytest.fixture
def sample_dataframe_1():
    """First sample DataFrame"""
    dates = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    return pd.DataFrame({"price": [10.0, 20.0, 30.0, 40.0, 50.0]}, index=dates)


@pytest.fixture
def sample_dataframe_2():
    """Second sample DataFrame with different columns"""
    dates = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    return pd.DataFrame({"load": [100, 200, 300, 400, 500]}, index=dates)


@pytest.fixture
def sample_dataframe_partial():
    """DataFrame with partial overlap"""
    dates = pd.date_range("2024-01-01 02:00", periods=3, freq="h", tz="UTC")
    return pd.DataFrame({"weather": [1.0, 2.0, 3.0]}, index=dates)


class TestDataPipelineInit:
    """Test DataPipeline initialization"""

    def test_init_single_source(self, sample_dataframe_1):
        """Test initialization with a single source"""
        source = MockDataSource(data=sample_dataframe_1)
        pipeline = DataPipeline(sources=[source])

        assert len(pipeline.sources) == 1
        assert len(pipeline.transformers) == 0

    def test_init_multiple_sources(self, sample_dataframe_1, sample_dataframe_2):
        """Test initialization with multiple sources"""
        source1 = MockDataSource(data=sample_dataframe_1)
        source2 = MockDataSource(data=sample_dataframe_2)
        pipeline = DataPipeline(sources=[source1, source2])

        assert len(pipeline.sources) == 2

    def test_init_with_transformers(self, sample_dataframe_1):
        """Test initialization with transformers"""
        source = MockDataSource(data=sample_dataframe_1)
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")
        pipeline = DataPipeline(sources=[source], transformers=[transformer])

        assert len(pipeline.transformers) == 1

    def test_init_empty_allows_builder(self):
        """Test that empty constructor is allowed for builder pattern"""
        pipeline = DataPipeline()
        assert len(pipeline.sources) == 0
        assert len(pipeline.transformers) == 0


class TestDataPipelineBuilder:
    """Test DataPipeline builder pattern"""

    def test_add_source(self, sample_dataframe_1):
        """Test adding a source via builder"""
        source = MockDataSource(data=sample_dataframe_1)
        pipeline = DataPipeline().add_source(source)

        assert len(pipeline.sources) == 1
        assert pipeline.sources[0] is source

    def test_add_source_returns_self(self, sample_dataframe_1):
        """Test that add_source returns self for chaining"""
        source = MockDataSource(data=sample_dataframe_1)
        pipeline = DataPipeline()
        result = pipeline.add_source(source)

        assert result is pipeline

    def test_add_multiple_sources(self, sample_dataframe_1, sample_dataframe_2):
        """Test adding multiple sources via chaining"""
        source1 = MockDataSource(data=sample_dataframe_1)
        source2 = MockDataSource(data=sample_dataframe_2)
        pipeline = DataPipeline().add_source(source1).add_source(source2)

        assert len(pipeline.sources) == 2

    def test_add_transformer(self, sample_dataframe_1):
        """Test adding a transformer via builder"""
        source = MockDataSource(data=sample_dataframe_1)
        transformer = MockTransformer()
        pipeline = DataPipeline().add_source(source).add_transformer(transformer)

        assert len(pipeline.transformers) == 1

    def test_add_transformer_returns_self(self):
        """Test that add_transformer returns self for chaining"""
        transformer = MockTransformer()
        pipeline = DataPipeline()
        result = pipeline.add_transformer(transformer)

        assert result is pipeline

    def test_full_builder_chain(self, sample_dataframe_1, sample_dataframe_2):
        """Test full builder pattern with multiple sources and transformers"""
        source1 = MockDataSource(data=sample_dataframe_1)
        source2 = MockDataSource(data=sample_dataframe_2)
        transformer1 = MockTransformer(suffix="_first")
        transformer2 = TimezoneTransformer(target_tz="Europe/Warsaw")

        pipeline = (
            DataPipeline()
            .add_source(source1)
            .add_source(source2)
            .add_transformer(transformer1)
            .add_transformer(transformer2)
        )

        assert len(pipeline.sources) == 2
        assert len(pipeline.transformers) == 2

    def test_add_source_invalid_type(self):
        """Test that add_source rejects invalid types"""
        pipeline = DataPipeline()
        with pytest.raises(TypeError, match="must be a DataSource"):
            pipeline.add_source("not a source")

    def test_add_transformer_invalid_type(self):
        """Test that add_transformer rejects invalid types"""
        pipeline = DataPipeline()
        with pytest.raises(TypeError, match="must be a Transformer"):
            pipeline.add_transformer("not a transformer")


class TestDataPipelineRun:
    """Test DataPipeline run method"""

    def test_run_single_source(self, sample_dataframe_1):
        """Test running pipeline with a single source"""
        source = MockDataSource(data=sample_dataframe_1)
        pipeline = DataPipeline(sources=[source])

        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        result = pipeline.run(start, end)

        assert source.fetch_called
        assert len(result) == 5
        assert "price" in result.columns

    def test_run_multiple_sources_merge(self, sample_dataframe_1, sample_dataframe_2):
        """Test that multiple sources are merged correctly"""
        source1 = MockDataSource(data=sample_dataframe_1)
        source2 = MockDataSource(data=sample_dataframe_2)
        pipeline = DataPipeline(sources=[source1, source2])

        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        result = pipeline.run(start, end)

        assert len(result) == 5
        assert "price" in result.columns
        assert "load" in result.columns

    def test_run_partial_overlap_outer_join(self, sample_dataframe_1, sample_dataframe_partial):
        """Test that partial overlap uses outer join"""
        source1 = MockDataSource(data=sample_dataframe_1)
        source2 = MockDataSource(data=sample_dataframe_partial)
        pipeline = DataPipeline(sources=[source1, source2])

        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        result = pipeline.run(start, end)

        assert len(result) == 5  # Outer join preserves all rows
        assert "price" in result.columns
        assert "weather" in result.columns
        # First two rows should have NaN for weather
        assert pd.isna(result["weather"].iloc[0])
        assert pd.isna(result["weather"].iloc[1])
        assert result["weather"].iloc[2] == 1.0

    def test_run_with_transformer(self, sample_dataframe_1):
        """Test that transformers are applied"""
        source = MockDataSource(data=sample_dataframe_1)
        transformer = MockTransformer(suffix="_test")
        pipeline = DataPipeline(sources=[source], transformers=[transformer])

        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        result = pipeline.run(start, end)

        assert transformer.transform_called
        assert "price_test" in result.columns

    def test_run_multiple_transformers_chain(self, sample_dataframe_1):
        """Test that multiple transformers are applied in order"""
        source = MockDataSource(data=sample_dataframe_1)
        transformer1 = MockTransformer(suffix="_first")
        transformer2 = MockTransformer(suffix="_second")
        pipeline = DataPipeline(sources=[source], transformers=[transformer1, transformer2])

        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        result = pipeline.run(start, end)

        # Column should have both suffixes applied in order
        assert "price_first_second" in result.columns

    def test_run_timezone_transformer(self, sample_dataframe_1):
        """Test with real TimezoneTransformer"""
        source = MockDataSource(data=sample_dataframe_1)
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")
        pipeline = DataPipeline(sources=[source], transformers=[transformer])

        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        result = pipeline.run(start, end)

        assert str(result.index.tz) == "Europe/Warsaw"

    def test_run_empty_source_returns_empty(self):
        """Test that empty source returns empty DataFrame"""
        source = MockDataSource(data=None)
        pipeline = DataPipeline(sources=[source])

        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        result = pipeline.run(start, end)

        assert result.empty

    def test_run_validates_timestamps(self, sample_dataframe_1):
        """Test that end must be after start"""
        source = MockDataSource(data=sample_dataframe_1)
        pipeline = DataPipeline(sources=[source])

        with pytest.raises(ValueError, match="must be after"):
            pipeline.run(
                start=pd.Timestamp("2024-01-02", tz="UTC"),
                end=pd.Timestamp("2024-01-01", tz="UTC"),
            )

    def test_run_localizes_naive_timestamps(self, sample_dataframe_1):
        """Test that naive timestamps are localized to UTC"""
        source = MockDataSource(data=sample_dataframe_1)
        pipeline = DataPipeline(sources=[source])

        start = pd.Timestamp("2024-01-01")  # Naive
        end = pd.Timestamp("2024-01-02")  # Naive
        pipeline.run(start, end)

        assert source.fetch_called
        # Check that the source received UTC timestamps
        assert source.fetch_args[0].tzinfo is not None
        assert source.fetch_args[1].tzinfo is not None

    def test_run_no_sources_raises_error(self):
        """Test that run with no sources raises error"""
        pipeline = DataPipeline()

        with pytest.raises(ValueError, match="At least one data source"):
            pipeline.run(
                start=pd.Timestamp("2024-01-01", tz="UTC"),
                end=pd.Timestamp("2024-01-02", tz="UTC"),
            )

    def test_run_with_builder_pattern(self, sample_dataframe_1):
        """Test running pipeline built with builder pattern"""
        source = MockDataSource(data=sample_dataframe_1)
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")

        result = (
            DataPipeline()
            .add_source(source)
            .add_transformer(transformer)
            .run(
                start=pd.Timestamp("2024-01-01", tz="UTC"),
                end=pd.Timestamp("2024-01-02", tz="UTC"),
            )
        )

        assert len(result) == 5
        assert str(result.index.tz) == "Europe/Warsaw"


class TestDataPipelineCache:
    """Test pipeline caching functionality"""

    def test_run_with_cache_disabled_default(self, sample_dataframe_1):
        """Test that cache is disabled by default"""
        source = MockDataSource(data=sample_dataframe_1)
        pipeline = DataPipeline(sources=[source])

        result = pipeline.run(start=pd.Timestamp("2024-01-01", tz="UTC"), end=pd.Timestamp("2024-01-02", tz="UTC"))

        assert len(result) == 5
        assert source.fetch_called

    def test_run_with_cache_enabled(self, tmp_path, monkeypatch):
        """cache=True stores the source output and serves the second run without calling the source."""
        monkeypatch.chdir(tmp_path)
        dates = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        source = MockDataSource(data=pd.DataFrame({"price": range(24)}, index=dates))
        pipeline = DataPipeline(sources=[source])
        start, end = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC")

        first = pipeline.run(start=start, end=end, cache=True)
        assert source.fetch_called
        assert list((tmp_path / ".cache" / "sources").glob("*/data_*.parquet"))

        source.fetch_called = False
        second = pipeline.run(start=start, end=end, cache=True)

        assert not source.fetch_called
        pd.testing.assert_frame_equal(first, second, check_freq=False)
        assert len(second) == 24

    def test_range_the_source_did_not_return_is_fetched_again(self, sample_dataframe_1, tmp_path, monkeypatch):
        """The source returned 00:00-04:00 for a whole-day request: only that part counts as cached."""
        monkeypatch.chdir(tmp_path)
        source = MockDataSource(data=sample_dataframe_1)
        pipeline = DataPipeline(sources=[source])
        start, end = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC")

        pipeline.run(start=start, end=end, cache=True)
        again = pipeline.run(start=start, end=end, cache=True)

        assert source.fetch_args == (pd.Timestamp("2024-01-01 05:00", tz="UTC"), end)
        assert len(again) == 5


# ---------------------------------------------------------------------------
# Source cache (cache=True) — regression tests for the review findings
# ---------------------------------------------------------------------------

_FULL = pd.date_range("2023-12-01", "2025-01-01", freq="1h", tz="UTC")
_RNG = np.random.default_rng(7)
_SERIES = pd.DataFrame(
    {"price": _RNG.normal(90, 10, len(_FULL)), "load": _RNG.normal(16000, 500, len(_FULL))}, index=_FULL
)


class RangeSource(DataSource):
    """Deterministic source over a fixed series.

    Like the real sources it may return rows outside [start, end] (ENTSO-E load starts one day early, Open-Meteo
    runs to end + horizon) and nothing after `now` (data that is not published yet).
    """

    def __init__(self, name="range", before="0h", after="0h", now=None, columns=("price", "load")):
        self.name = name
        self.before = pd.Timedelta(before)
        self.after = pd.Timedelta(after)
        self.now = now
        self.columns = list(columns)
        self.calls = []

    def fetch(self, start, end):
        self.calls.append((start, end))
        hi = end + self.after
        if self.now is not None:
            hi = min(hi, self.now)
        return _SERIES.loc[start - self.before : hi, self.columns].copy()

    def _validate_config(self):
        return True

    def get_cache_config(self):
        return {"source_type": self.name, "columns": self.columns}


class EntsoeLikeSource(DataSource):
    """Mimics EntsoeSource(type=["load", "price"]): load from start - 1 day, price from start, both to end
    inclusive, and a daily minimum stamped at local midnight."""

    def fetch(self, start, end):
        daily = _SERIES["load"].tz_convert("Europe/Warsaw").resample("D").min().tz_convert("UTC")
        load = pd.DataFrame({"load_actual": _SERIES["load"], "load_forecast_daily_min": daily})
        load = load.truncate(before=start - pd.Timedelta(days=1), after=end)
        price = _SERIES[["price"]].truncate(before=start, after=end)
        return pd.concat([load, price], axis=1)

    def _validate_config(self):
        return True

    def get_cache_config(self):
        return {"source_type": "entsoe_like"}


class EntsoeLikeClippingSource(DataSource):
    """Like `EntsoeLikeSource`, but also mimics `EntsoeSource._clip`: a daily value whose (one-local-day) period
    covers `start` — i.e. it started before `start` — is carried forward and re-stamped at `start`, unless
    `start` already is that value's own stamp. Real ENTSO-E daily values sit at the source's own local midnight,
    so this only bites when `start` falls strictly inside a local day, e.g. a per-source cache resuming at a
    boundary that isn't the area's own midnight (DST/timezone review finding 2)."""

    def __init__(self, tz="Europe/Warsaw"):
        self.tz = tz

    def _load(self):
        return _SERIES["load"]

    def fetch(self, start, end):
        load = self._load()
        daily = load.tz_convert(self.tz).resample("D").min().tz_convert("UTC")  # one value per local day
        df = pd.DataFrame({"load_actual": load, "load_forecast_daily_min": daily})
        df = df.truncate(before=start - pd.Timedelta(days=1), after=end)

        values = df["load_forecast_daily_min"].dropna()
        before = values[values.index < start]
        carried = None
        if not before.empty:
            period_start = before.index[-1]
            period_end = (period_start.tz_convert(self.tz) + pd.Timedelta(days=1)).tz_convert("UTC")
            if period_end > start:
                carried = before.iloc[-1]

        df = df[(df.index >= start) & (df.index < end)].copy()
        if carried is not None and start not in values.index:
            if start not in df.index:
                df = df.reindex(df.index.union([start]))
            df.loc[start, "load_forecast_daily_min"] = carried
        return df

    def _daily_zone(self):
        """Like `EntsoeSource._daily_zone`: the zone a daily value is stamped at when no pipeline zone is bound."""
        return self.tz

    def _validate_config(self):
        return True

    def get_cache_config(self):
        return {"source_type": "entsoe_like_clipping", "tz": self.tz}


class EntsoeLikeResolutionSwitchSource(EntsoeLikeClippingSource):
    """Like `EntsoeLikeClippingSource`, but the load rows are hourly up to 2024-06-12 23:00 UTC and 15-minute from
    2024-06-13 00:00 UTC, as PL's ENTSO-E load is. The cache cannot tell whether the 45 minutes after the last
    hourly row are missing, so it leaves them as a hole and fetches them on their own next time."""

    def _load(self):
        hourly = pd.date_range("2024-06-10", "2024-06-12 23:00", freq="1h", tz="UTC")
        quarter_hourly = pd.date_range("2024-06-13", "2024-06-16 23:45", freq="15min", tz="UTC")
        index = hourly.append(quarter_hourly)
        return pd.Series(np.arange(len(index), dtype=float) % 29 + 100.0, index=index)

    def get_cache_config(self):
        return {"source_type": "entsoe_like_resolution_switch", "tz": self.tz}


class EntsoeLikeRevisedSource(EntsoeLikeResolutionSwitchSource):
    """Like `EntsoeLikeResolutionSwitchSource`, with every load value moved by `offset` (a later revision). The cache
    config does not mention it, so revisions share one cache."""

    def __init__(self, tz="Europe/Warsaw", offset=0.0):
        super().__init__(tz)
        self.offset = offset

    def _load(self):
        return super()._load() + self.offset


class EntsoeLikeEmptyRefetchSource(EntsoeLikeResolutionSwitchSource):
    """Like `EntsoeLikeResolutionSwitchSource`, but a fetch that ends at 2024-06-13 00:00 UTC or earlier, which is
    only the refetch of the hole's local day, gets an empty frame, as ENTSO-E answers "No matching data"."""

    def __init__(self, tz="Europe/Warsaw"):
        super().__init__(tz)
        self.calls = []

    def fetch(self, start, end):
        self.calls.append((start, end))
        if end <= pd.Timestamp("2024-06-13", tz="UTC"):
            return pd.DataFrame()
        return super().fetch(start, end)

    def get_cache_config(self):
        return {"source_type": "entsoe_like_empty_refetch", "tz": self.tz}


def _utc(s):
    return pd.Timestamp(s, tz="UTC")


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    """Run in an empty directory so `.cache/sources` starts cold; freeze the cache's clock far in the future."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cache_manager_module, "_now", lambda: _utc("2100-01-01"))
    return tmp_path


class TestSourceCache:
    def test_cache_true_equals_cache_false(self, in_tmp):
        """pipelines M2 / empirical F5c: both paths return exactly the requested range (end inclusive)."""
        start, end = "2024-01-01", "2024-02-01"

        def pipe():
            return DataPipeline().add_source(RangeSource(before="1D", after="2D"))

        fresh = pipe().run(start, end, cache=False)
        miss = pipe().run(start, end, cache=True)
        hit = pipe().run(start, end, cache=True)

        assert fresh.index.min() == _utc(start)
        assert fresh.index.max() == _utc(end) + pd.Timedelta("23h")
        pd.testing.assert_frame_equal(fresh, miss, check_freq=False)
        pd.testing.assert_frame_equal(fresh, hit, check_freq=False)

    def test_uncached_source_is_clipped_too(self, in_tmp):
        """Sources without a cache config (e.g. CsvSource) are clipped to the resolved range too."""
        src = RangeSource(before="1D", after="2D")
        src.get_cache_config = lambda: None
        df = DataPipeline().add_source(src).run("2024-01-01", "2024-01-03", cache=True)
        assert df.index.min() == _utc("2024-01-01")
        assert df.index.max() == _utc("2024-01-03 23:00")
        assert not (in_tmp / ".cache" / "sources").exists() or not list((in_tmp / ".cache" / "sources").iterdir())

    def test_cached_split_run_equals_fresh_run(self, in_tmp):
        """entsoe H3 / empirical F5b: no half-empty rows at the seam between two cached ranges."""
        DataPipeline().add_source(EntsoeLikeSource()).run("2024-01-01", "2024-01-03", cache=True)
        cached = DataPipeline().add_source(EntsoeLikeSource()).run("2024-01-01", "2024-01-05", cache=True)
        fresh = DataPipeline().add_source(EntsoeLikeSource()).run("2024-01-01", "2024-01-05", cache=False)

        pd.testing.assert_frame_equal(fresh, cached, check_freq=False)
        assert cached.loc[_utc("2024-01-02 23:00")].notna().all()
        assert cached.loc[_utc("2024-01-03 00:00"), ["load_actual", "price"]].notna().all()

    def test_cached_split_lands_exactly_on_the_inclusive_end_boundary(self, in_tmp):
        """`end="2024-01-02"` now includes that whole day; a second fetch starting the next calendar day must
        neither duplicate nor skip a row at that seam."""
        first = DataPipeline().add_source(EntsoeLikeSource()).run("2024-01-01", "2024-01-02", cache=True)
        cached = DataPipeline().add_source(EntsoeLikeSource()).run("2024-01-03", "2024-01-04", cache=True)
        combined = pd.concat([first, cached]).sort_index()
        fresh = DataPipeline().add_source(EntsoeLikeSource()).run("2024-01-01", "2024-01-04", cache=False)

        assert first.index.max() == _utc("2024-01-02 23:00")
        assert cached.index.min() == _utc("2024-01-03 00:00")
        pd.testing.assert_frame_equal(fresh, combined, check_freq=False)

    def test_split_run_in_three_pieces_equals_fresh_run(self, in_tmp):
        for s, e in [("2024-01-10", "2024-01-12"), ("2024-01-01", "2024-01-03")]:
            DataPipeline().add_source(EntsoeLikeSource()).run(s, e, cache=True)
        cached = DataPipeline().add_source(EntsoeLikeSource()).run("2024-01-01", "2024-01-20", cache=True)
        fresh = DataPipeline().add_source(EntsoeLikeSource()).run("2024-01-01", "2024-01-20", cache=False)
        pd.testing.assert_frame_equal(fresh, cached, check_freq=False)

    @pytest.mark.parametrize("tz", ["Europe/Warsaw", "Europe/Helsinki"])
    def test_second_cached_run_does_not_double_stamp_a_daily_value(self, in_tmp, monkeypatch, tz):
        """DST/timezone review finding 2: the per-source cache's unsettled boundary is floored at the market's
        local midnight, not at UTC midnight, which (almost) never lines up with a source's own local midnight.
        Resuming a fetch there would make EntsoeSource._clip carry the day's already-cached daily value forward
        again, re-stamped at that non-local-midnight boundary — a second value for the same local day, which
        TimezoneTransformer rejects.
        A cached, split run must equal a fresh one, including daily columns, in PL and in an EET zone."""
        start, end = _utc("2024-10-25"), _utc("2024-10-29")

        def pipe():
            return DataPipeline(
                sources=[EntsoeLikeClippingSource(tz)], transformers=[TimezoneTransformer()], timezone=tz
            )

        monkeypatch.setattr(cache_manager_module, "_now", lambda: _utc("2024-10-30 12:00"))
        pipe().run(start, end, cache=True)  # first piece: cached only up to the unsettled boundary

        monkeypatch.setattr(cache_manager_module, "_now", lambda: _utc("2024-11-15 12:00"))
        cached = pipe().run(start, end, cache=True)  # resumes from that boundary

        fresh = pipe().run(start, end, cache=False)
        pd.testing.assert_frame_equal(fresh, cached, check_freq=False)

    @pytest.mark.parametrize("tz", ["Europe/Warsaw", "Europe/Helsinki"])
    def test_second_cached_run_does_not_double_stamp_a_daily_value_without_a_pipeline_timezone(
        self, in_tmp, monkeypatch, tz
    ):
        """The unsettled boundary of the cache is floored in the source's own zone when the pipeline has none,
        not in UTC, so the resumed fetch does not start inside a local day either."""
        start, end = _utc("2024-10-25"), _utc("2024-10-29")

        def pipe():
            return DataPipeline(
                sources=[EntsoeLikeClippingSource(tz)], transformers=[TimezoneTransformer(target_tz=tz)]
            )

        monkeypatch.setattr(cache_manager_module, "_now", lambda: _utc("2024-10-30 12:00"))
        pipe().run(start, end, cache=True)

        monkeypatch.setattr(cache_manager_module, "_now", lambda: _utc("2024-11-15 12:00"))
        cached = pipe().run(start, end, cache=True)

        fresh = pipe().run(start, end, cache=False)
        pd.testing.assert_frame_equal(fresh, cached, check_freq=False)

    @pytest.mark.parametrize("tz", ["Europe/Warsaw", "Europe/Helsinki"])
    def test_a_hole_inside_a_local_day_does_not_double_stamp_a_daily_value(self, in_tmp, tz):
        """The cache fetches a hole on its own, and the hole after the last hourly row of a series that turns into
        15-minute rows starts inside a local day whose daily value sits in the neighbouring piece. A fetch from
        there would carry that value forward (EntsoeSource._clip) and stamp the day a second time, which
        TimezoneTransformer rejects, so every run after the first would fail. Holes are fetched from the start of
        their local day instead, and every cached run must equal a fresh one."""
        start, end = _utc("2024-06-10"), _utc("2024-06-16")

        def pipe():
            return DataPipeline(
                sources=[EntsoeLikeResolutionSwitchSource(tz)], transformers=[TimezoneTransformer()], timezone=tz
            )

        fresh = pipe().run(start, end, cache=False)
        for _ in range(3):  # the first run fetches everything, the next ones fetch the hole
            pd.testing.assert_frame_equal(fresh, pipe().run(start, end, cache=True), check_freq=False)

    @pytest.mark.parametrize("tz", ["Europe/Warsaw", "Europe/Helsinki"])
    def test_a_hole_does_not_double_stamp_a_daily_value_without_a_pipeline_timezone(self, in_tmp, tz):
        """The same without `DataPipeline(timezone=...)`: the hole starts from the start of its day in the source's
        own zone (`_daily_zone`), and every cached run equals a fresh one."""
        start, end = _utc("2024-06-10"), _utc("2024-06-16")

        def pipe():
            return DataPipeline(
                sources=[EntsoeLikeResolutionSwitchSource(tz)], transformers=[TimezoneTransformer(target_tz=tz)]
            )

        fresh = pipe().run(start, end, cache=False)
        for _ in range(3):
            pd.testing.assert_frame_equal(fresh, pipe().run(start, end, cache=True), check_freq=False)

    def test_the_zone_holes_start_in_is_the_pipelines_else_the_sources_own(self):
        from epftoolbox2.data.sources import EntsoeSource
        from epftoolbox2.pipelines.data_pipeline import _cache_day_zone

        fi = EntsoeSource("FI", "key", ["load"])
        assert _cache_day_zone(fi, None) == "Europe/Helsinki"
        assert _cache_day_zone(fi, "Europe/Warsaw") == "Europe/Warsaw"
        assert _cache_day_zone(RangeSource(), None) is None
        DataPipeline([fi], timezone="Europe/Warsaw")  # binding the market zone makes it the source's
        assert _cache_day_zone(fi, None) == "Europe/Warsaw"

    def test_a_source_without_a_zone_keeps_holes_as_they_are_without_a_pipeline_timezone(self, in_tmp):
        source = RangeSource()
        assert not hasattr(source, "_daily_zone")
        start, end = _utc("2024-01-01"), _utc("2024-01-03")
        pipe = DataPipeline([source])
        pipe.run(start, end, cache=True)
        source.calls.clear()
        pipe.run(start, _utc("2024-01-05"), cache=True)
        assert source.calls == [(end, _utc("2024-01-05"))]

    def test_an_empty_refetch_of_a_hole_keeps_the_rows_already_cached(self, in_tmp):
        """The hole's local day is fetched again from its midnight (22:00Z) and the source answers with nothing.
        The rows of that day that were cached (22:00Z and 23:00Z, with the day's daily value) must stay in the
        result, not vanish because they were dropped before the refetch."""
        start, end = _utc("2024-06-10"), _utc("2024-06-16")

        def pipe():
            source = EntsoeLikeEmptyRefetchSource()
            return DataPipeline(sources=[source], timezone="Europe/Warsaw"), source

        first_pipe, _ = pipe()
        first = first_pipe.run(start, end, cache=True)
        assert _utc("2024-06-12 22:00") in first.index and _utc("2024-06-12 23:00") in first.index
        assert (
            first.loc[_utc("2024-06-12 22:00"), "load_forecast_daily_min"]
            == first["load_forecast_daily_min"].dropna().loc[_utc("2024-06-12 22:00")]
        )

        for _ in range(2):
            second_pipe, source = pipe()
            second = second_pipe.run(start, end, cache=True)
            assert source.calls == [(_utc("2024-06-12 22:00"), _utc("2024-06-13"))]  # the hole was asked for
            pd.testing.assert_frame_equal(first, second, check_freq=False)

    def test_a_refetch_that_returns_rows_still_replaces_the_cached_rows_of_its_range(self, in_tmp):
        """Only an empty answer keeps the cached rows: when the refetch returns rows for the hole's local day, they
        replace what was cached there, and the rest of the cache is untouched."""
        start, end = _utc("2024-06-10"), _utc("2024-06-16")

        first = DataPipeline(sources=[EntsoeLikeRevisedSource()], timezone="Europe/Warsaw").run(start, end, cache=True)
        second = DataPipeline(sources=[EntsoeLikeRevisedSource(offset=1000.0)], timezone="Europe/Warsaw").run(
            start, end, cache=True
        )
        day = second.index >= _utc("2024-06-12 22:00")
        assert second.index.equals(first.index)
        assert (second.loc[day & (second.index < _utc("2024-06-13")), "load_actual"] > 1000).all()
        pd.testing.assert_frame_equal(first.loc[~day], second.loc[~day], check_freq=False)
        # The 15-minute rows after the hole were cached whole, so they are the first source's values.
        assert (second.loc[second.index >= _utc("2024-06-13"), "load_actual"] < 1000).all()

    def test_unpublished_tail_is_fetched_again_next_day(self, in_tmp, monkeypatch):
        """pipelines H3 / entsoe H4 / empirical F5d: the daily forecast-only recipe with cache=True."""
        src = RangeSource(name="live", columns=["load"])
        pipe = DataPipeline().add_source(src)

        for day, (start, end) in enumerate([("2024-05-01", "2024-06-18"), ("2024-05-02", "2024-06-19")]):
            now = _utc("2024-06-10 09:00") + pd.Timedelta(days=day)
            src.now = now
            monkeypatch.setattr(cache_manager_module, "_now", lambda now=now: now)
            cached = pipe.run(start, end, cache=True)

        fresh = pipe.run("2024-05-02", "2024-06-19", cache=False)
        pd.testing.assert_frame_equal(fresh, cached, check_freq=False)
        assert cached["load"].last_valid_index() == _utc("2024-06-11 09:00")

    def test_open_meteo_provisional_rows_are_not_cached_as_covered(self, in_tmp, monkeypatch):
        """integration check f: an Open-Meteo row whose determining issue time hasn't passed yet is
        provisional (it substitutes "now" for the issue time and uses whatever run has been published so
        far, see OpenMeteoSource._build_features). Such a row is always within a day or two of "now", so
        the per-source cache's 2-day unsettled boundary must already keep it from being recorded as
        covered. Fetch a wide, mostly-old range twice: the part far in the past must be served from cache (the second
        fetch requests far less than the full range), but the tail around "now" must still be requested
        again instead of being served a stale, wrongly-covered provisional value."""
        from tests.data.sources.test_open_meteo import make_archive

        archive = make_archive()
        before_issue = pd.Timestamp("2024-01-11 08:00", tz="Europe/Warsaw").tz_convert("UTC")
        after_issue = pd.Timestamp("2024-01-11 11:00", tz="Europe/Warsaw").tz_convert("UTC")

        def fetch(now):
            source = OpenMeteoSource(
                latitude=52.23, longitude=21.01, columns=["temperature_2m"], timezone="Europe/Warsaw", horizon=1
            )
            source.session = Mock()
            source.session.get.side_effect = archive
            source.console.quiet = True
            source._now = lambda: now
            monkeypatch.setattr(cache_manager_module, "_now", lambda: now)
            DataPipeline().add_source(source).run("2023-06-01", "2024-01-12", cache=True)
            return source

        fetch(before_issue)  # cold cache: everything is fetched once
        n_calls_before = len(archive.calls)
        settled_source = fetch(after_issue)  # same wide range, now after the issue time

        second_fetch_calls = archive.calls[n_calls_before:]
        assert second_fetch_calls  # the tail was requested again, not served from a wrongly-covered cache
        requested_starts = [pd.Timestamp(c["start_date"], tz="UTC") for c in second_fetch_calls]
        # Most of 2023-06-01..2024-01-12 was served from cache: only a small tail near "now" was re-requested.
        assert min(requested_starts) > pd.Timestamp("2023-12-01", tz="UTC")
        assert settled_source.session.get.called

    def test_deleted_cache_file_is_refetched(self, in_tmp):
        """entsoe M4: a deleted data file must not silently drop the source's columns."""

        def pipe():
            return DataPipeline(
                [RangeSource("price_src", columns=["price"]), RangeSource("load_src", columns=["load"])]
            )

        first = pipe().run("2024-01-01", "2024-01-03", cache=True)
        files = sorted((in_tmp / ".cache" / "sources").glob("*/data_*.parquet"))
        assert len(files) == 2
        files[0].unlink()

        again = pipe().run("2024-01-01", "2024-01-03", cache=True)
        pd.testing.assert_frame_equal(first, again, check_freq=False)

    def test_concurrent_runs_fetch_once_and_agree(self, in_tmp):
        """pipelines H1 / entsoe H5: SLURM array tasks starting together on a cold shared cache."""
        results, fetches = _run_concurrently(in_tmp, cache=True)
        assert len(set(results)) == 1, results
        assert results[0][0] == len(pd.date_range("2024-01-01", "2024-07-02", freq="h", inclusive="left"))
        assert fetches == 1


class SlowLoggingSource(DataSource):
    """Slow source that appends one line per fetch to a log file (counts API calls across processes)."""

    def __init__(self, log_path):
        self.log_path = log_path

    def fetch(self, start, end):
        with open(self.log_path, "a") as f:
            f.write(f"{start} {end}\n")
        time.sleep(1.0)
        return _SERIES.loc[start:end].copy()

    def _validate_config(self):
        return True

    def get_cache_config(self):
        return {"source_type": "slow_logging"}


def _pipeline_task(workdir, cache, queue):
    os.chdir(workdir)
    try:
        df = DataPipeline([SlowLoggingSource(os.path.join(workdir, "fetch.log"))]).run(
            "2024-01-01", "2024-07-01", cache=cache
        )
        queue.put((len(df), round(float(df.sum().sum()), 6)))
    except Exception as e:  # noqa: BLE001 - report any failure to the parent
        queue.put((type(e).__name__, str(e)[:200]))


def _run_concurrently(workdir, cache, n=3):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_pipeline_task, args=(str(workdir), cache, queue)) for _ in range(n)]
    for p in procs:
        p.start()
    results = [queue.get(timeout=180) for _ in procs]
    for p in procs:
        p.join(timeout=60)
    fetches = len((workdir / "fetch.log").read_text().splitlines())
    return results, fetches


# ---------------------------------------------------------------------------
# Single-file cache (cache="file.csv" / "file.parquet" / directory)
# ---------------------------------------------------------------------------


class TestFileCache:
    def _pipe(self, **kw):
        return DataPipeline().add_source(RangeSource(**kw))

    def test_refetches_when_dates_change(self, in_tmp):
        """pipelines M3 / empirical F5e: a different range must not return the frame of another range."""
        self._pipe().run("2024-01-01", "2024-03-01", cache="cache.csv")
        other = self._pipe().run("2024-02-01", "2024-05-01", cache="cache.csv")
        fresh = self._pipe().run("2024-02-01", "2024-05-01")
        pd.testing.assert_frame_equal(fresh, other, check_freq=False)

    def test_refetches_when_sources_change(self, in_tmp):
        self._pipe(columns=["price", "load"]).run("2024-01-01", "2024-02-01", cache="cache.csv")
        other = self._pipe(columns=["price"]).run("2024-01-01", "2024-02-01", cache="cache.csv")
        assert list(other.columns) == ["price"]

    def test_hit_does_not_call_the_source(self, in_tmp):
        self._pipe().run("2024-01-01", "2024-02-01", cache="cache.csv")
        src = RangeSource()
        DataPipeline().add_source(src).run("2024-01-01", "2024-02-01", cache="cache.csv")
        assert src.calls == []

    @pytest.mark.parametrize("name", ["cache.csv", "cache"])
    def test_file_without_a_fingerprint_is_rebuilt(self, in_tmp, name):
        """A cache file without a fingerprint is never trusted."""
        pd.DataFrame({"price": [1.0]}, index=pd.DatetimeIndex(["2020-01-01"], tz="UTC")).to_csv(in_tmp / name)
        got = self._pipe().run("2024-01-01", "2024-02-01", cache=name)
        pd.testing.assert_frame_equal(got, self._pipe().run("2024-01-01", "2024-02-01"), check_freq=False)
        assert (in_tmp / name).is_file()

    @pytest.mark.parametrize("name", ["cache.csv", "cache.parquet"])
    def test_reload_is_exact(self, in_tmp, name):
        """Floats, dtypes and the UTC index survive the file (read_csv's fast float parser would lose a digit)."""
        from epftoolbox2.data.sources import CalendarSource

        def pipe():
            return (
                DataPipeline()
                .add_source(RangeSource())
                .add_source(CalendarSource("PL", holidays="name", weekday="onehot", daylight=True))
            )

        fresh = pipe().run("2024-01-01", "2024-03-01")
        pipe().run("2024-01-01", "2024-03-01", cache=name)
        reload = pipe().run("2024-01-01", "2024-03-01", cache=name)
        pd.testing.assert_frame_equal(fresh, reload, check_freq=False, check_exact=True)

    def test_refetches_when_csv_input_changes(self, in_tmp):
        from epftoolbox2.data.sources import CsvSource

        path = in_tmp / "input.csv"
        path.write_text("datetime,price\n2024-01-01 00:00,1.0\n2024-01-01 01:00,2.0\n")
        DataPipeline().add_source(CsvSource(str(path))).run("2024-01-01", "2024-01-02", cache="cache.csv")
        path.write_text("datetime,price\n2024-01-01 00:00,1.0\n2024-01-01 01:00,5.0\n2024-01-01 02:00,6.0\n")
        again = DataPipeline().add_source(CsvSource(str(path))).run("2024-01-01", "2024-01-02", cache="cache.csv")
        assert again["price"].tolist() == [1.0, 5.0, 6.0]

    def test_expands_environment_variables(self, in_tmp, monkeypatch):
        monkeypatch.setenv("EPF_TEST_JOB", "job42")
        self._pipe().run("2024-01-01", "2024-01-02", cache="scratch/$EPF_TEST_JOB/data.csv")
        assert (in_tmp / "scratch" / "job42" / "data.csv").exists()

    def test_unset_environment_variable_fails_before_fetching(self, in_tmp, monkeypatch):
        monkeypatch.delenv("EPF_TEST_UNSET", raising=False)
        src = RangeSource()
        with pytest.raises(ValueError, match="EPF_TEST_UNSET"):
            DataPipeline().add_source(src).run("2024-01-01", "2024-01-02", cache="scratch/${EPF_TEST_UNSET}/data.csv")
        assert src.calls == []

    @pytest.mark.parametrize("target", ["existing_dir", "new_dir/", "no_suffix"])
    def test_directory_holds_one_file_per_configuration(self, in_tmp, target):
        if target == "existing_dir":
            (in_tmp / target).mkdir()
        a = self._pipe().run("2024-01-01", "2024-01-05", cache=target)
        b = self._pipe().run("2024-02-01", "2024-02-05", cache=target)
        folder = in_tmp / target.rstrip("/")
        assert folder.is_dir()
        assert len(list(folder.glob("data_*.parquet"))) == 2
        pd.testing.assert_frame_equal(a, self._pipe().run("2024-01-01", "2024-01-05", cache=target), check_freq=False)
        pd.testing.assert_frame_equal(b, self._pipe().run("2024-02-01", "2024-02-05", cache=target), check_freq=False)

    def test_fingerprint_file_does_not_contain_api_key(self, in_tmp):
        src = RangeSource()
        src.api_key = "SECRET-KEY-123"
        DataPipeline().add_source(src).run("2024-01-01", "2024-01-02", cache="cache.csv")
        for f in in_tmp.iterdir():
            if f.is_file():
                assert "SECRET-KEY-123" not in f.read_text(errors="ignore"), f.name

    def test_concurrent_runs_fetch_once_and_agree(self, in_tmp):
        """pipelines H1: parallel tasks sharing one cache file must not read it half-written."""
        results, fetches = _run_concurrently(in_tmp, cache="shared.csv")
        assert len(set(results)) == 1, results
        assert fetches == 1

    def test_refetches_once_the_unsettled_tail_becomes_settled(self, in_tmp, monkeypatch):
        """pipelines-integration: a fixed `end` beyond "now" must not cache the not-yet-settled tail forever:
        the per-source cache never records that tail as covered, and the single-file cache must not either.
        Once the settled boundary has moved past it, the file must be refetched, not served stale."""
        day1, day2 = _utc("2024-06-10 09:00"), _utc("2024-06-13 09:00")

        monkeypatch.setattr(cache_manager_module, "_now", lambda: day1)
        src1 = RangeSource()
        DataPipeline().add_source(src1).run("2024-06-01", "2024-06-15", cache="cache.parquet")
        assert src1.calls  # cold cache: fetched

        monkeypatch.setattr(cache_manager_module, "_now", lambda: day2)
        src2 = RangeSource()
        DataPipeline().add_source(src2).run("2024-06-01", "2024-06-15", cache="cache.parquet")
        assert src2.calls  # day1's settled boundary does not cover "now" on day2: refetched

    def test_does_not_refetch_when_the_whole_range_was_already_settled(self, in_tmp, monkeypatch):
        """The staleness check must not defeat caching for a plain historical range."""
        day1, day2 = _utc("2024-06-10 09:00"), _utc("2024-06-13 09:00")

        monkeypatch.setattr(cache_manager_module, "_now", lambda: day1)
        src1 = RangeSource()
        DataPipeline().add_source(src1).run("2024-01-01", "2024-02-01", cache="cache.parquet")
        assert src1.calls

        monkeypatch.setattr(cache_manager_module, "_now", lambda: day2)
        src2 = RangeSource()
        DataPipeline().add_source(src2).run("2024-01-01", "2024-02-01", cache="cache.parquet")
        assert not src2.calls  # "end" was already settled when fetched on day1 — the file is still trusted


# ---------------------------------------------------------------------------
# Date semantics (timezone=...)
# ---------------------------------------------------------------------------


class TestDateSemantics:
    """pipelines H4: date strings as local calendar days with `timezone`, UTC midnights without it."""

    def test_without_timezone_dates_are_utc_midnights_end_inclusive(self):
        """A date-only `end` covers that whole UTC day, same as with `timezone` set (just UTC instead of local)."""
        df = DataPipeline([RangeSource()]).run("2024-01-01", "2024-01-03")
        assert df.index.min() == _utc("2024-01-01")
        assert df.index.max() == _utc("2024-01-03 23:00")
        assert len(df) == 72

    def test_timezone_dates_are_whole_local_days(self):
        df = DataPipeline([RangeSource()], timezone="Europe/Warsaw").run("2024-01-01", "2024-01-03")
        local = df.index.tz_convert("Europe/Warsaw")
        assert local.min() == pd.Timestamp("2024-01-01 00:00", tz="Europe/Warsaw")
        assert local.max() == pd.Timestamp("2024-01-03 23:00", tz="Europe/Warsaw")
        assert len(df) == 72

    @pytest.mark.parametrize("timezone", [None, "Europe/Warsaw"])
    def test_a_plain_date_names_the_same_whole_days_as_its_string(self, timezone):
        """YAML turns an unquoted `end: 2024-01-31` into a `datetime.date`. It must end through the end of 01-31,
        as the quoted string does, and not at its midnight (one day short)."""
        pipe = DataPipeline([RangeSource()], timezone=timezone)
        quoted = pipe.resolve_range("2024-01-01", "2024-01-31")
        assert pipe.resolve_range(datetime.date(2024, 1, 1), datetime.date(2024, 1, 31)) == quoted
        assert quoted[1] - quoted[0] == pd.Timedelta(days=31)
        df = pipe.run(datetime.date(2024, 1, 1), datetime.date(2024, 1, 31))
        assert df.index.max() == quoted[1] - pd.Timedelta(hours=1)
        assert len(df) == 31 * 24

    def test_a_plain_date_end_is_the_whole_local_day_on_a_dst_day(self):
        pipe = DataPipeline([RangeSource()], timezone="Europe/Warsaw")
        assert len(pipe.run(datetime.date(2024, 3, 31), datetime.date(2024, 3, 31))) == 23

    def test_a_datetime_or_timestamp_is_still_an_exact_instant(self):
        pipe = DataPipeline([RangeSource()], timezone="Europe/Warsaw")
        exact = pipe.resolve_range(datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 31))
        assert exact[1] == pd.Timestamp("2024-01-30 23:00", tz="UTC")
        assert pipe.resolve_range(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31")) == exact

    @pytest.mark.parametrize("day, hours", [("2024-03-31", 23), ("2024-10-27", 25), ("2024-06-01", 24)])
    def test_timezone_end_day_is_complete_on_dst_days(self, day, hours):
        df = DataPipeline([RangeSource()], timezone="Europe/Warsaw").run(day, day)
        assert len(df) == hours

    def test_timezone_keywords_use_the_market_date(self, monkeypatch):
        from epftoolbox2.pipelines import data_pipeline as dp_module

        # 23:30 UTC on 2024-06-09 is already 2024-06-10 in Warsaw
        monkeypatch.setattr(dp_module, "_now", lambda tz: pd.Timestamp("2024-06-09 23:30", tz="UTC").tz_convert(tz))
        pipe = DataPipeline([RangeSource()], timezone="Europe/Warsaw")
        df = pipe.run("today_d-1", "today")
        local = df.index.tz_convert("Europe/Warsaw")
        assert local.min() == pd.Timestamp("2024-06-09 00:00", tz="Europe/Warsaw")
        assert local.max() == pd.Timestamp("2024-06-10 23:00", tz="Europe/Warsaw")

    def test_timezone_naive_timestamps_are_local_instants(self):
        pipe = DataPipeline([RangeSource()], timezone="Europe/Warsaw")
        df = pipe.run(pd.Timestamp("2024-01-01 06:00"), "2024-01-01 09:00")
        assert df.index.min() == _utc("2024-01-01 05:00")
        assert df.index.max() == _utc("2024-01-01 07:00")

    def test_aware_timestamps_are_used_as_given(self):
        pipe = DataPipeline([RangeSource()], timezone="Europe/Warsaw")
        df = pipe.run(_utc("2024-01-01 06:00"), _utc("2024-01-01 09:00"))
        assert (df.index.min(), df.index.max()) == (_utc("2024-01-01 06:00"), _utc("2024-01-01 08:00"))

    def test_invalid_timezone_raises(self):
        with pytest.raises(ValueError, match="timezone"):
            DataPipeline(timezone="Europe/Warsw")

    def test_timezone_mismatched_with_timezone_transformer_raises(self):
        """pipelines review, finding 3: DataPipeline.timezone decides what a local calendar day is, so a
        TimezoneTransformer converting to a different zone silently splits that day across two local dates."""
        with pytest.raises(ValueError, match="America/New_York.*Europe/Warsaw|Europe/Warsaw.*America/New_York"):
            DataPipeline(
                sources=[RangeSource()],
                timezone="America/New_York",
                transformers=[TimezoneTransformer(target_tz="Europe/Warsaw")],
            )

    def test_timezone_matching_timezone_transformer_is_allowed(self):
        pipeline = DataPipeline(
            sources=[RangeSource()],
            timezone="Europe/Warsaw",
            transformers=[TimezoneTransformer(target_tz="Europe/Warsaw")],
        )
        assert pipeline.timezone == "Europe/Warsaw"

    def test_timezone_mismatch_added_via_builder_is_caught_before_running(self):
        """The mismatch isn't only checked at construction: add_transformer can introduce it afterwards."""
        pipeline = DataPipeline(sources=[RangeSource()], timezone="America/New_York").add_transformer(
            TimezoneTransformer(target_tz="Europe/Warsaw")
        )
        with pytest.raises(ValueError, match="America/New_York.*Europe/Warsaw|Europe/Warsaw.*America/New_York"):
            pipeline.run("2024-01-01", "2024-01-02")

    def test_timezone_mismatched_with_open_meteo_source_raises(self):
        """integration check b: a weather source computing its own local calendar day/issue time in a
        different zone than DataPipeline.timezone would silently disagree with it."""
        from epftoolbox2.data.sources import OpenMeteoSource

        with pytest.raises(ValueError, match="America/New_York.*Europe/Warsaw|Europe/Warsaw.*America/New_York"):
            DataPipeline(
                sources=[OpenMeteoSource(52.23, 21.01, timezone="Europe/Warsaw")],
                timezone="America/New_York",
            )

    def test_timezone_mismatched_with_open_meteo_single_runs_api_raises(self):
        from epftoolbox2.data.sources import OpenMeteoSource

        with pytest.raises(ValueError, match="America/New_York.*Europe/Warsaw|Europe/Warsaw.*America/New_York"):
            DataPipeline(
                sources=[OpenMeteoSource(52.23, 21.01, timezone="Europe/Warsaw", api="single_runs")],
                timezone="America/New_York",
            )

    def test_timezone_mismatched_with_open_meteo_source_locations_raises(self):
        """A source built with `locations` still has its own `timezone`, checked at the parent level."""
        from epftoolbox2.data.sources import OpenMeteoSource

        with pytest.raises(ValueError, match="America/New_York.*Europe/Warsaw|Europe/Warsaw.*America/New_York"):
            DataPipeline(
                sources=[
                    OpenMeteoSource(
                        locations={"warsaw": {"latitude": 52.23, "longitude": 21.01}},
                        columns=["temperature_2m"],
                        timezone="Europe/Warsaw",
                    )
                ],
                timezone="America/New_York",
            )

    def test_timezone_mismatched_with_calendar_source_raises(self):
        from epftoolbox2.data.sources import CalendarSource

        with pytest.raises(ValueError, match="America/New_York.*Europe/Warsaw|Europe/Warsaw.*America/New_York"):
            DataPipeline(sources=[CalendarSource(country="PL", timezone="Europe/Warsaw")], timezone="America/New_York")

    def test_calendar_source_without_explicit_timezone_inherits_the_pipeline_s(self):
        """CalendarSource(timezone=None) doesn't fall back to the country's own zone once a DataPipeline gives
        it one: DataPipeline.timezone is the single place the market zone needs to be written."""
        from epftoolbox2.data.sources import CalendarSource

        source = CalendarSource(country="PL")
        assert source.timezone is None
        pipeline = DataPipeline(sources=[source], timezone="America/New_York")
        assert pipeline.timezone == "America/New_York"
        assert source.market_timezone == "America/New_York"

    def test_timezone_matching_open_meteo_source_is_allowed(self):
        from epftoolbox2.data.sources import OpenMeteoSource

        pipeline = DataPipeline(
            sources=[OpenMeteoSource(52.23, 21.01, timezone="Europe/Warsaw")],
            timezone="Europe/Warsaw",
        )
        assert pipeline.timezone == "Europe/Warsaw"

    def test_csv_source_timezone_is_not_checked_against_pipeline_timezone(self, tmp_path):
        """CsvSource.timezone is the input file's own zone, not the market's, so it is excluded."""
        from epftoolbox2.data.sources import CsvSource

        csv_path = tmp_path / "p.csv"
        csv_path.write_text("ts;price\n2024-01-01 00:00;1.0\n")
        pipeline = DataPipeline(
            sources=[CsvSource(str(csv_path), datetime_column="ts", timezone="America/New_York")],
            timezone="Europe/Warsaw",
        )
        assert pipeline.timezone == "Europe/Warsaw"

    def test_no_timezone_with_timezone_transformer_is_allowed(self):
        """Without DataPipeline.timezone there is no local-day boundary to disagree about."""
        pipeline = DataPipeline(sources=[RangeSource()], transformers=[TimezoneTransformer(target_tz="Europe/Warsaw")])
        assert pipeline.timezone is None

    def test_forecast_only_end_equal_to_last_horizon_day_gives_every_forecast(self, monkeypatch):
        """With timezone, `end = run day + H` covers the last target day, so all 24 * H forecasts are produced."""
        from collections import Counter

        from epftoolbox2.data.sources import CalendarSource
        from epftoolbox2.data.transformers import LagTransformer
        from epftoolbox2.models import OLSModel

        monkeypatch.setenv("MAX_PROCESSES", "1")
        monkeypatch.setenv("THREADS_PER_PROCESS", "1")
        run_day, horizon = "2024-06-10", 3
        pipe = (
            DataPipeline(timezone="Europe/Warsaw")
            .add_source(RangeSource(columns=["price"]))
            .add_source(CalendarSource("PL", holidays="binary", weekday="number"))
            .add_transformer(TimezoneTransformer("Europe/Warsaw"))
            .add_transformer(LagTransformer(["price"], [1, 7], "day"))
            .add_transformer(LagTransformer(["weekday"], list(range(-horizon, 1)), "day"))
        )
        end = (pd.Timestamp(run_day) + pd.Timedelta(days=horizon)).date().isoformat()
        df = pipe.run("2024-03-01", end)

        model = OLSModel(["price_d-1", "price_d-7", "weekday_d+{horizon}"], training_window=60)
        ref = model.run(df, test_start=run_day, test_end=run_day, horizon=horizon, forecast_only=True)
        per_horizon = Counter(r["horizon"] for r in ref._results)
        assert len(ref._results) == ref.count == 24 * horizon
        assert dict(per_horizon) == {h: 24 for h in range(1, horizon + 1)}


# ---------------------------------------------------------------------------
# Column collisions
# ---------------------------------------------------------------------------


class TestColumnCollisions:
    """pipelines M1: two sources producing the same column name must not be kept silently."""

    def test_duplicate_column_across_sources_raises_naming_sources(self):
        from epftoolbox2.data.sources import CalendarSource

        pipe = DataPipeline([CalendarSource("PL"), CalendarSource("DE")])
        with pytest.raises(ValueError, match=r"is_holiday.*CalendarSource #1.*CalendarSource #2"):
            pipe.run("2024-01-01", "2024-01-02")

    def test_prefix_resolves_the_collision(self):
        from epftoolbox2.data.sources import CalendarSource

        df = DataPipeline([CalendarSource("PL"), CalendarSource("DE", prefix="de_")]).run("2024-01-01", "2024-01-02")
        assert {"is_holiday", "de_is_holiday"} <= set(df.columns)

    def test_collision_is_detected_on_the_cached_path_too(self, in_tmp):
        pipe = DataPipeline([RangeSource("a", columns=["price"]), RangeSource("b", columns=["price", "load"])])
        with pytest.raises(ValueError, match=r"price.*RangeSource #1.*RangeSource #2"):
            pipe.run("2024-01-01", "2024-01-02", cache=True)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


class RecordingValidator(Validator):
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.seen_columns = None

    def validate(self, df):
        self.seen_columns = list(df.columns)
        result = ValidationResult()
        if self.fail:
            result.is_valid = False
            result.errors.append("something is wrong")
        return result


class TestValidation:
    def _pipe(self, **kw):
        return DataPipeline(sources=[RangeSource(columns=["price"])], transformers=[MockTransformer("_t")], **kw)

    def test_default_warns_and_keeps_results(self):
        validator = RecordingValidator(fail=True)
        pipe = self._pipe(validators=[validator])
        assert pipe.on_invalid == "warn"
        df = pipe.run("2024-01-01", "2024-01-02")
        assert not df.empty
        ((name, result),) = pipe.validation_results.items()
        assert name == "validators.RecordingValidator"
        assert result.is_valid is False

    def test_raise_stops_the_pipeline(self):
        pipe = self._pipe(validators=[RecordingValidator(fail=True)], on_invalid="raise")
        with pytest.raises(DataValidationError, match="RecordingValidator.*something is wrong"):
            pipe.run("2024-01-01", "2024-01-02")

    def test_passing_validators_do_not_raise(self):
        pipe = self._pipe(validators=[RecordingValidator()], on_invalid="raise")
        assert not pipe.run("2024-01-01", "2024-01-02").empty

    def test_source_validators_see_raw_source_output(self):
        source_check, output_check = RecordingValidator(), RecordingValidator()
        pipe = self._pipe(source_validators=[source_check], validators=[output_check])
        pipe.run("2024-01-01", "2024-01-02")
        assert source_check.seen_columns == ["price"]
        assert output_check.seen_columns == ["price_t"]
        assert list(pipe.validation_results) == [
            "source_validators.RecordingValidator",
            "validators.RecordingValidator",
        ]

    def test_failing_source_validator_raises_before_transformers(self):
        transformer = MockTransformer()
        pipe = DataPipeline(
            sources=[RangeSource()],
            transformers=[transformer],
            source_validators=[RecordingValidator(fail=True)],
            on_invalid="raise",
        )
        with pytest.raises(DataValidationError):
            pipe.run("2024-01-01", "2024-01-02")
        assert not transformer.transform_called

    def test_add_source_validator_builder(self):
        pipe = DataPipeline().add_source_validator(RecordingValidator())
        assert len(pipe.source_validators) == 1
        with pytest.raises(TypeError, match="Validator"):
            pipe.add_source_validator("not a validator")

    def test_repeated_validator_classes_get_distinct_names(self):
        pipe = self._pipe(validators=[RecordingValidator(), RecordingValidator()])
        pipe.run("2024-01-01", "2024-01-02")
        assert list(pipe.validation_results) == ["validators.RecordingValidator", "validators.RecordingValidator#2"]

    def test_results_are_reset_on_each_run(self):
        pipe = self._pipe(validators=[RecordingValidator()])
        pipe.run("2024-01-01", "2024-01-02")
        pipe.validators = []
        pipe.run("2024-01-01", "2024-01-02")
        assert pipe.validation_results == {}

    def test_invalid_on_invalid_value_raises(self):
        with pytest.raises(ValueError, match="on_invalid"):
            DataPipeline(on_invalid="ignore")


class TestDataPipelineSerialization:
    """Test pipeline serialization (save/load)"""

    def test_to_dict_empty_pipeline(self):
        """Test serializing empty pipeline to dict"""
        pipeline = DataPipeline()
        config = pipeline.to_dict()

        assert config == {
            "timezone": None,
            "on_invalid": "warn",
            "sources": [],
            "source_validators": [],
            "transformers": [],
            "validators": [],
        }

    def test_to_dict_with_source(self):
        """Test serializing pipeline with source to dict"""
        from epftoolbox2.data.sources import CalendarSource

        pipeline = DataPipeline().add_source(CalendarSource(country="PL"))
        config = pipeline.to_dict()

        assert len(config["sources"]) == 1
        assert config["sources"][0]["class"] == "CalendarSource"
        assert config["sources"][0]["params"]["country"] == "PL"

    def test_to_dict_with_transformer(self):
        """Test serializing pipeline with transformer to dict"""
        pipeline = DataPipeline().add_transformer(TimezoneTransformer(target_tz="Europe/Warsaw"))
        config = pipeline.to_dict()

        assert len(config["transformers"]) == 1
        assert config["transformers"][0]["class"] == "TimezoneTransformer"
        assert config["transformers"][0]["params"]["target_tz"] == "Europe/Warsaw"

    def test_to_dict_with_validator(self):
        """Test serializing pipeline with validator to dict"""
        from epftoolbox2.data.validators import CoverageValidator

        pipeline = DataPipeline().add_validator(CoverageValidator(columns=["test"]))
        config = pipeline.to_dict()

        assert len(config["validators"]) == 1
        assert config["validators"][0]["class"] == "CoverageValidator"
        assert config["validators"][0]["params"]["columns"] == ["test"]

    def test_save_and_load(self, tmp_path):
        """Test saving and loading pipeline from YAML"""
        from epftoolbox2.data.sources import CalendarSource
        from epftoolbox2.data.validators import CoverageValidator

        pipeline = (
            DataPipeline()
            .add_source(CalendarSource(country="PL", holidays="binary", weekday="number"))
            .add_transformer(TimezoneTransformer(target_tz="Europe/Warsaw"))
            .add_validator(CoverageValidator(freq="1h"))
        )

        yaml_path = tmp_path / "pipeline.yaml"
        pipeline.save(yaml_path)

        assert yaml_path.exists()

        loaded = DataPipeline.load(yaml_path)

        assert len(loaded.sources) == 1
        assert len(loaded.transformers) == 1
        assert len(loaded.validators) == 1
        assert type(loaded.sources[0]).__name__ == "CalendarSource"
        assert type(loaded.transformers[0]).__name__ == "TimezoneTransformer"
        assert type(loaded.validators[0]).__name__ == "CoverageValidator"

    def test_loaded_pipeline_runs(self, tmp_path):
        """Test that loaded pipeline can run successfully"""
        from epftoolbox2.data.sources import CalendarSource

        pipeline = DataPipeline().add_source(CalendarSource(country="PL", holidays="binary", weekday=False))

        yaml_path = tmp_path / "pipeline.yaml"
        pipeline.save(yaml_path)

        loaded = DataPipeline.load(yaml_path)
        result = loaded.run(start="2024-01-01", end="2024-01-02")

        assert not result.empty
        assert "is_holiday" in result.columns

    def test_save_creates_valid_yaml(self, tmp_path):
        """Test that saved file is valid YAML"""
        import yaml

        pipeline = DataPipeline().add_transformer(TimezoneTransformer(target_tz="UTC"))

        yaml_path = tmp_path / "pipeline.yaml"
        pipeline.save(yaml_path)

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        assert "sources" in config
        assert "transformers" in config
        assert "validators" in config

    def test_load_unknown_component_raises_error(self, tmp_path):
        """Test that loading unknown component raises error"""
        import yaml

        yaml_path = tmp_path / "invalid.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump({"sources": [{"class": "UnknownSource", "params": {}}], "transformers": [], "validators": []}, f)

        with pytest.raises(ValueError, match="Unknown"):
            DataPipeline.load(yaml_path)

    def test_round_trip_preserves_params(self, tmp_path):
        """Test that save/load preserves all parameters"""
        from epftoolbox2.data.sources import CalendarSource
        from epftoolbox2.data.transformers import ResampleTransformer

        pipeline = (
            DataPipeline()
            .add_source(CalendarSource(country="DE", holidays="onehot", weekday="name", month="number"))
            .add_transformer(ResampleTransformer(freq="1h", agg="sum"))
        )

        yaml_path = tmp_path / "pipeline.yaml"
        pipeline.save(yaml_path)
        loaded = DataPipeline.load(yaml_path)

        assert loaded.sources[0].country == "DE"
        assert loaded.sources[0].holidays == "onehot"
        assert loaded.sources[0].weekday == "name"
        assert loaded.sources[0].month == "number"
        assert loaded.transformers[0].freq == "1h"
        assert loaded.transformers[0].agg == "sum"
