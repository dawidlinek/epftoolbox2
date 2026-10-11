"""DataPipeline.timezone as the single place the market zone needs to be written.

TimezoneTransformer.target_tz, CalendarSource.timezone and OpenMeteoSource.timezone may all be left unset and
inherit DataPipeline.timezone instead — including, for a source built with `locations`, through each of its
private per-location instances (see DataPipeline._bind_component_timezones). Explicit-and-equal and
explicit-and-different-raises are covered in tests/pipelines/test_pipeline.py (TestDateSemantics); this file
covers inheritance itself, the "missing everywhere" error, YAML round trips and the information-set check.
"""

from unittest.mock import Mock

import pandas as pd
import pytest
import yaml

from epftoolbox2.data.sources import (
    CalendarSource,
    OpenMeteoSource,
)
from epftoolbox2.data.sources.base import DataSource
from epftoolbox2.data.transformers import TimezoneTransformer
from epftoolbox2.information_set import source_rules
from epftoolbox2.pipelines import DataPipeline
from tests.data.sources.test_open_meteo import make_archive

TZ = "Europe/Warsaw"


class _RangeSource(DataSource):
    """Minimal deterministic source: one column of hourly integers over [start, end)."""

    def __init__(self, columns=("price",)):
        self.columns = list(columns)

    def fetch(self, start, end):
        index = pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")
        return pd.DataFrame({c: range(len(index)) for c in self.columns}, index=index)

    def _validate_config(self):
        return True

    def get_cache_config(self):
        return None


def _open_meteo(**kwargs):
    kwargs.setdefault("latitude", 52.23)
    kwargs.setdefault("longitude", 21.01)
    kwargs.setdefault("columns", ["temperature_2m"])
    kwargs.setdefault("horizon", 1)
    source = OpenMeteoSource(**kwargs)
    source.session = Mock()
    source.session.get.side_effect = make_archive()
    source.console.quiet = True
    return source


_RANGE = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-03", tz="UTC"))


# ---------------------------------------------------------------------------
# Inheritance: the component's own zone is unset and it takes DataPipeline.timezone
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_calendar_source(self):
        source = CalendarSource(country="PL")
        DataPipeline(sources=[source], timezone="America/New_York")
        assert source.market_timezone == "America/New_York"
        assert source.timezone is None  # the public attribute is never mutated

    def test_open_meteo_source(self):
        source = _open_meteo()
        DataPipeline(sources=[source], timezone=TZ)
        assert source.market_timezone == TZ
        assert source.timezone is None

    def test_open_meteo_source_single_runs_api(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, columns=["temperature_2m"], api="single_runs")
        DataPipeline(sources=[source], timezone=TZ)
        assert source.market_timezone == TZ
        assert source.timezone is None

    def test_open_meteo_source_with_locations(self):
        source = OpenMeteoSource(
            locations={"warsaw": {"latitude": 52.23, "longitude": 21.01, "weight": 1.0}},
            agg="mean",
            columns=["temperature_2m"],
        )
        DataPipeline(sources=[source], timezone=TZ)
        assert source.market_timezone == TZ
        assert source.timezone is None
        assert source._location_sources[0].market_timezone == TZ
        assert source._location_sources[0].timezone is None

    def test_entsoe_source(self):
        # EntsoeSource has no zone parameter of its own (nothing to leave unset); it only uses the bound zone
        # to anchor a neighbouring area's week-ahead daily value (A31) onto the market's own local day — see
        # EntsoeSource._daily_zone and tests/data/sources/test_entsoe.py::TestWeekAheadDailyMarketZone.
        from epftoolbox2.data.sources import EntsoeSource

        source = EntsoeSource("FI", "test-key", type=["load"])
        assert source.market_timezone is None
        DataPipeline(sources=[source], timezone=TZ)
        assert source.market_timezone == TZ

    def test_timezone_transformer(self):
        transformer = TimezoneTransformer()
        DataPipeline(sources=[_RangeSource()], transformers=[transformer], timezone=TZ)
        assert transformer.market_timezone == TZ
        assert transformer.target_tz is None

    def test_open_meteo_source_gives_identical_data_as_explicit(self):
        explicit = _open_meteo(timezone=TZ)
        inherited = _open_meteo()
        a = DataPipeline(sources=[explicit]).run(*_RANGE)
        b = DataPipeline(sources=[inherited], timezone=TZ).run(*_RANGE)
        pd.testing.assert_frame_equal(a, b)

    def test_timezone_transformer_gives_identical_data_as_explicit(self):
        # Both pipelines set DataPipeline.timezone=TZ (it also decides what the date strings mean below); only
        # the transformer's own target_tz differs, explicit vs inherited.
        a = DataPipeline(sources=[_RangeSource()], transformers=[TimezoneTransformer(target_tz=TZ)], timezone=TZ).run(
            "2024-01-01", "2024-01-02"
        )
        b = DataPipeline(sources=[_RangeSource()], transformers=[TimezoneTransformer()], timezone=TZ).run(
            "2024-01-01", "2024-01-02"
        )
        pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Missing everywhere: neither the component nor the pipeline has a zone
# ---------------------------------------------------------------------------


class TestMissingEverywhereRaises:
    def test_open_meteo_source(self):
        pipe = DataPipeline(sources=[_open_meteo()])
        with pytest.raises(ValueError, match="timezone is required"):
            pipe.run(*_RANGE)

    def test_open_meteo_source_single_runs_api(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, columns=["temperature_2m"], api="single_runs")
        pipe = DataPipeline(sources=[source])
        with pytest.raises(ValueError, match="timezone is required"):
            pipe.run(*_RANGE)

    def test_open_meteo_source_with_locations(self):
        source = OpenMeteoSource(
            locations={"warsaw": {"latitude": 52.23, "longitude": 21.01, "weight": 1.0}},
            agg="mean",
            columns=["temperature_2m"],
        )
        pipe = DataPipeline(sources=[source])
        with pytest.raises(ValueError, match="timezone is required"):
            pipe.run(*_RANGE)

    def test_timezone_transformer(self):
        pipe = DataPipeline(sources=[_RangeSource()], transformers=[TimezoneTransformer()])
        with pytest.raises(ValueError, match="has no time zone"):
            pipe.run("2024-01-01", "2024-01-02")

    def test_standalone_open_meteo_source_without_a_pipeline(self):
        # Never attached to a DataPipeline at all: still a clear error, not a silent wrong default.
        source = _open_meteo()
        with pytest.raises(ValueError, match="timezone is required"):
            source.fetch(*_RANGE)


# ---------------------------------------------------------------------------
# save() writes the zone once
# ---------------------------------------------------------------------------


class TestSaveWritesTheZoneOnce:
    def test_save(self, tmp_path):
        pipe = DataPipeline(
            sources=[CalendarSource(country="PL"), _open_meteo()],
            transformers=[TimezoneTransformer()],
            timezone=TZ,
        )
        path = tmp_path / "dp.yaml"
        pipe.save(path)
        config = yaml.safe_load(path.read_text())

        assert config["timezone"] == TZ
        assert config["sources"][0]["params"]["timezone"] is None
        assert config["sources"][1]["params"]["timezone"] is None
        assert config["transformers"][0]["params"]["target_tz"] is None


# ---------------------------------------------------------------------------
# Load and rerun gives identical data
# ---------------------------------------------------------------------------


class TestLoadAndRerun:
    def test_calendar_and_open_meteo(self, tmp_path):
        original = DataPipeline(
            sources=[CalendarSource(country="PL", holidays=False, weekday=False), _open_meteo()],
            transformers=[TimezoneTransformer()],
            timezone=TZ,
        )
        path = tmp_path / "dp.yaml"
        original.save(path)
        loaded = DataPipeline.load(path)

        # The loaded OpenMeteoSource is a fresh instance: mock its HTTP the same way as the original's.
        weather = loaded.sources[1]
        weather.session = Mock()
        weather.session.get.side_effect = make_archive()
        weather.console.quiet = True

        a = original.run(*_RANGE)
        b = loaded.run(*_RANGE)
        pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# The information-set check sees the inherited zone
# ---------------------------------------------------------------------------


class TestInformationSetSeesTheInheritedZone:
    def test_source_rules_resolves_even_with_an_inherited_timezone(self):
        # weather_cutoff is anchored to run:00 UTC, not to the source's own market zone, so source_rules must
        # not need `market_timezone` (and therefore a bound DataPipeline) to resolve a rule at all.
        source = _open_meteo(run="09")
        dp = DataPipeline(sources=[source], timezone=TZ)

        rules = source_rules(dp.sources)

        assert rules["temperature_2m"].kind == "weather_cutoff"
        assert rules["temperature_2m"].time == "09:00"


# ---------------------------------------------------------------------------
# One component instance shared by two pipelines
# ---------------------------------------------------------------------------


class TestSharedComponent:
    def test_second_pipeline_with_another_zone_raises(self):
        source = CalendarSource(country="PL")
        DataPipeline(sources=[source], timezone="Europe/Warsaw")
        with pytest.raises(ValueError, match="another DataPipeline"):
            DataPipeline(sources=[source], timezone="Europe/London")
        assert source.market_timezone == "Europe/Warsaw"

    def test_second_pipeline_with_the_same_zone_is_fine(self):
        transformer = TimezoneTransformer()
        DataPipeline(transformers=[transformer], timezone="Europe/Warsaw")
        DataPipeline(transformers=[transformer], timezone="Europe/Warsaw")

    def test_rebinding_by_the_same_pipeline_is_fine(self):
        source = CalendarSource(country="PL")
        pipeline = DataPipeline(sources=[source], timezone="Europe/Warsaw")
        pipeline.timezone = "Europe/Berlin"
        pipeline._bind_component_timezones()
        assert source.market_timezone == "Europe/Berlin"
