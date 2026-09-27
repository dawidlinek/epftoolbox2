"""OpenMeteoSource's `locations` parameter: several locations fetched by another instance of the same class
(sharing `api`/`run`) and combined into one weighted series, per-location columns, or both.

Behaviour tests here were carried over from the removed OpenMeteoWeightedSource (its multi-location logic moved
into OpenMeteoBase, shared across both `api` modes — see `epftoolbox2/data/sources/_open_meteo_common.py`).
"""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.sources.open_meteo import OpenMeteoSource
from epftoolbox2.pipelines import DataPipeline

TZ = "Europe/Warsaw"

LOCATIONS = {
    "warsaw": {"latitude": 52.23, "longitude": 21.01, "weight": 3.0},
    "krakow": {"latitude": 50.06, "longitude": 19.94, "weight": 1.0},
}
LOCATIONS_NO_WEIGHT = {
    "warsaw": {"latitude": 52.23, "longitude": 21.01},
    "krakow": {"latitude": 50.06, "longitude": 19.94},
}


def _frame(index, **columns):
    return pd.DataFrame(columns, index=index)


def make_source(locations=None, fetch_returns=None, **kwargs):
    """Build a source with `locations` and stub each underlying instance's `fetch` with `fetch_returns[i]`."""
    kwargs.setdefault("columns", ["temperature_2m"])
    kwargs.setdefault("horizon", 1)
    kwargs.setdefault("timezone", TZ)
    source = OpenMeteoSource(locations=locations or LOCATIONS, **kwargs)
    if fetch_returns is not None:
        for underlying, frame in zip(source._location_sources, fetch_returns):
            underlying.fetch = lambda *a, _f=frame, **k: _f
    return source


# ---------------------------------------------------------------------------
# Mutual exclusivity with latitude/longitude
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("api", ["previous_runs", "single_runs"])
class TestMutualExclusivity:
    def test_locations_and_point_together_raise(self, api):
        with pytest.raises(ValueError, match="mutually exclusive"):
            OpenMeteoSource(
                latitude=52.23, longitude=21.01, locations=LOCATIONS, timezone=TZ, columns=["temperature_2m"], api=api
            )

    def test_neither_given_raises(self, api):
        with pytest.raises(ValueError, match="either latitude and longitude"):
            OpenMeteoSource(timezone=TZ, columns=["temperature_2m"], api=api)

    def test_only_latitude_given_raises(self, api):
        with pytest.raises(ValueError, match="both be set"):
            OpenMeteoSource(latitude=52.23, timezone=TZ, columns=["temperature_2m"], api=api)

    def test_only_longitude_given_raises(self, api):
        with pytest.raises(ValueError, match="both be set"):
            OpenMeteoSource(longitude=21.01, timezone=TZ, columns=["temperature_2m"], api=api)


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self):
        source = make_source()
        assert source.keep_locations is False
        assert source.prefix == ""
        assert len(source._location_sources) == 2
        assert all(isinstance(s, OpenMeteoSource) for s in source._location_sources)

    def test_settings_reach_the_underlying_sources(self):
        source = make_source(columns=["temperature_2m"], horizon=3, model="icon_seamless", api="single_runs", run="06")
        for underlying, name in zip(source._location_sources, ("warsaw", "krakow")):
            assert underlying.latitude == LOCATIONS[name]["latitude"]
            assert underlying.longitude == LOCATIONS[name]["longitude"]
            assert underlying.horizon == 3
            assert underlying.model == "icon_seamless"
            assert underlying.api == "single_runs"
            assert underlying.run == "06"
            assert underlying.prefix == ""  # locations carry their own name-based prefixing instead

    def test_rejects_non_bool_keep_locations(self):
        with pytest.raises(ValueError, match="keep_locations must be True or False"):
            make_source(keep_locations="yes")

    def test_weights_are_normalised(self):
        source = make_source()
        np.testing.assert_allclose(source._location_weights, [0.75, 0.25])


# ---------------------------------------------------------------------------
# fetch: weighted average
# ---------------------------------------------------------------------------


class TestFetchWeightedAverage:
    def test_weighted_average_matches_manual_calculation(self):
        index = pd.date_range("2024-01-10", periods=2, freq="1h", tz="UTC")
        source = make_source(
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0, 20.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0, 0.0]}),
            ]
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        # weights normalise to [0.75, 0.25]: 0.75*10 + 0.25*30 = 15; 0.75*20 + 0.25*0 = 15
        np.testing.assert_allclose(df["temperature_2m_d+1"].tolist(), [15.0, 15.0])

    def test_nan_at_any_location_makes_the_weighted_value_nan(self):
        # The NaN row is not last, so the trailing-row trim does not remove it.
        index = pd.date_range("2024-01-10", periods=3, freq="1h", tz="UTC")
        source = make_source(
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0, np.nan, 10.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0, 5.0, 30.0]}),
            ]
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert df["temperature_2m_d+1"].iloc[0] == 15.0
        assert np.isnan(df["temperature_2m_d+1"].iloc[1])
        assert df["temperature_2m_d+1"].iloc[2] == 15.0

    def test_prefix_applied_to_weighted_columns(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0]}),
            ],
            prefix="pl",
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert list(df.columns) == ["pl_temperature_2m_d+1"]

    def test_indices_are_unioned_and_missing_rows_are_nan_weighted(self):
        idx1 = pd.date_range("2024-01-10 00:00", periods=2, freq="1h", tz="UTC")
        idx2 = pd.date_range("2024-01-10 01:00", periods=2, freq="1h", tz="UTC")  # one hour later, one hour short
        source = make_source(
            fetch_returns=[
                _frame(idx1, **{"temperature_2m_d+1": [10.0, 20.0]}),
                _frame(idx2, **{"temperature_2m_d+1": [30.0, 40.0]}),
            ]
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        # 00:00 only in location 1 -> NaN; 01:00 and 02:00 both present -> weighted; last row trimmed by the tail
        # trim only if fully NaN, so 00:00 (partially NaN combined -> NaN) is dropped only if it's the very end.
        assert pd.Timestamp("2024-01-10 00:00", tz="UTC") in df.index
        assert np.isnan(df.loc["2024-01-10 00:00", "temperature_2m_d+1"])
        np.testing.assert_allclose(df.loc["2024-01-10 01:00", "temperature_2m_d+1"], 0.75 * 20 + 0.25 * 30)

    def test_all_empty_frames_return_empty_dataframe(self):
        source = make_source(fetch_returns=[pd.DataFrame(), pd.DataFrame()])
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert df.empty

    def test_trailing_all_nan_rows_are_dropped(self):
        index = pd.date_range("2024-01-10", periods=3, freq="1h", tz="UTC")
        source = make_source(
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0, 20.0, np.nan]}),
                _frame(index, **{"temperature_2m_d+1": [30.0, 40.0, np.nan]}),
            ]
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert len(df) == 2


class TestWindDirection:
    def test_circular_mean_wraps_around_zero(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            locations={
                "a": {"latitude": 52.23, "longitude": 21.01, "weight": 1.0},
                "b": {"latitude": 50.06, "longitude": 19.94, "weight": 1.0},
            },
            fetch_returns=[
                _frame(index, **{"wind_direction_10m_d+1": [350.0]}),
                _frame(index, **{"wind_direction_10m_d+1": [10.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        # A plain arithmetic mean would give 180 deg (due south); the circular mean must be near 0/360.
        value = df["wind_direction_10m_d+1"].iloc[0]
        assert value < 5 or value > 355


class TestCategoricalColumns:
    def test_weather_code_is_not_averaged(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            fetch_returns=[
                _frame(index, **{"weather_code_d+1": [1.0], "temperature_2m_d+1": [10.0]}),
                _frame(index, **{"weather_code_d+1": [2.0], "temperature_2m_d+1": [30.0]}),
            ]
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert "weather_code_d+1" not in df.columns
        assert "temperature_2m_d+1" in df.columns

    def test_keep_locations_exposes_weather_code_per_location_only(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            fetch_returns=[
                _frame(index, **{"weather_code_d+1": [1.0], "temperature_2m_d+1": [10.0]}),
                _frame(index, **{"weather_code_d+1": [2.0], "temperature_2m_d+1": [30.0]}),
            ],
            keep_locations=True,
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert "weather_code_d+1" not in df.columns
        assert df["warsaw_weather_code_d+1"].iloc[0] == 1.0
        assert df["krakow_weather_code_d+1"].iloc[0] == 2.0


class TestKeepLocations:
    def test_per_location_columns_present_with_prefix(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0]}),
            ],
            keep_locations=True,
            prefix="pl",
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert "pl_temperature_2m_d+1" in df.columns
        assert "pl_warsaw_temperature_2m_d+1" in df.columns
        assert "pl_krakow_temperature_2m_d+1" in df.columns
        assert df["pl_warsaw_temperature_2m_d+1"].iloc[0] == 10.0
        assert df["pl_krakow_temperature_2m_d+1"].iloc[0] == 30.0

    def test_default_does_not_include_per_location_columns(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0]}),
            ]
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert list(df.columns) == ["temperature_2m_d+1"]


# ---------------------------------------------------------------------------
# fetch: without weights, per-location columns only
# ---------------------------------------------------------------------------


class TestPerLocationOnly:
    def test_only_per_location_columns_are_returned(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            locations=LOCATIONS_NO_WEIGHT,
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert list(df.columns) == ["warsaw_temperature_2m_d+1", "krakow_temperature_2m_d+1"]
        assert df["warsaw_temperature_2m_d+1"].iloc[0] == 10.0
        assert df["krakow_temperature_2m_d+1"].iloc[0] == 30.0

    def test_prefix_applies_to_per_location_columns(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            locations=LOCATIONS_NO_WEIGHT,
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0]}),
            ],
            prefix="pl",
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert list(df.columns) == ["pl_warsaw_temperature_2m_d+1", "pl_krakow_temperature_2m_d+1"]

    def test_keep_locations_has_no_extra_effect_without_weights(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            locations=LOCATIONS_NO_WEIGHT,
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0]}),
            ],
            keep_locations=True,
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert list(df.columns) == ["warsaw_temperature_2m_d+1", "krakow_temperature_2m_d+1"]


class TestSharedNow:
    def test_every_location_fetches_against_the_same_now(self):
        # Fetching locations one after another can let wall-clock time advance between them; if each location
        # resolved "now" independently, live rows near the cutoff could end up on different sides of it for
        # different locations, mixing vintages within what should be one consistent row.
        source = make_source()
        seen_now = []
        for underlying in source._location_sources:

            def fake_fetch(start, end, _underlying=underlying):
                seen_now.append(_underlying._now())
                return pd.DataFrame()

            underlying.fetch = fake_fetch

        ticks = iter(
            [
                pd.Timestamp("2024-01-11 08:00", tz="UTC"),
                pd.Timestamp("2024-01-11 09:00", tz="UTC"),
                pd.Timestamp("2024-01-11 10:00", tz="UTC"),
            ]
        )
        source._now = lambda: next(ticks)
        source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert len(seen_now) == len(source._location_sources)
        assert len(set(seen_now)) == 1, f"locations used different 'now' values: {seen_now}"


# ---------------------------------------------------------------------------
# Cache config and serialization
# ---------------------------------------------------------------------------


class TestCacheConfigAndSerialization:
    def test_cache_config_contains_locations_and_no_latitude_longitude(self):
        source = OpenMeteoSource(locations=LOCATIONS, columns=["temperature_2m"], timezone=TZ)
        config = source.get_cache_config()
        assert config["locations"] == [
            {"name": "warsaw", "latitude": 52.23, "longitude": 21.01, "weight": 3.0},
            {"name": "krakow", "latitude": 50.06, "longitude": 19.94, "weight": 1.0},
        ]
        assert "latitude" not in config
        assert "longitude" not in config
        assert "keep_locations" not in config  # only included when True

    def test_changing_a_weight_changes_the_cache_config(self):
        base = OpenMeteoSource(locations=LOCATIONS, columns=["temperature_2m"], timezone=TZ).get_cache_config()
        changed = dict(LOCATIONS, warsaw={**LOCATIONS["warsaw"], "weight": 5.0})
        other = OpenMeteoSource(locations=changed, columns=["temperature_2m"], timezone=TZ).get_cache_config()
        assert base != other

    def test_changing_a_shared_setting_changes_the_cache_config(self):
        base = OpenMeteoSource(locations=LOCATIONS, columns=["temperature_2m"], horizon=2, timezone=TZ)
        other = OpenMeteoSource(locations=LOCATIONS, columns=["temperature_2m"], horizon=3, timezone=TZ)
        assert base.get_cache_config() != other.get_cache_config()

    def test_keep_locations_changes_the_cache_config(self):
        base = OpenMeteoSource(locations=LOCATIONS, columns=["temperature_2m"], timezone=TZ).get_cache_config()
        other = OpenMeteoSource(
            locations=LOCATIONS, columns=["temperature_2m"], timezone=TZ, keep_locations=True
        ).get_cache_config()
        assert base != other

    def test_locations_and_single_point_configs_are_never_equal(self):
        point = OpenMeteoSource(latitude=52.23, longitude=21.01, columns=["temperature_2m"], timezone=TZ)
        located = OpenMeteoSource(
            locations={"warsaw": {"latitude": 52.23, "longitude": 21.01}}, columns=["temperature_2m"], timezone=TZ
        )
        assert point.get_cache_config() != located.get_cache_config()

    def test_yaml_round_trip(self, tmp_path):
        source = OpenMeteoSource(
            locations=LOCATIONS,
            columns=["temperature_2m"],
            horizon=3,
            model="icon_seamless",
            timezone=TZ,
            prefix="pl",
            keep_locations=True,
        )
        path = tmp_path / "dp.yaml"
        DataPipeline().add_source(source).save(path)
        loaded = DataPipeline.load(path).sources[0]
        assert isinstance(loaded, OpenMeteoSource)
        assert loaded.get_cache_config() == source.get_cache_config()
        assert loaded.keep_locations is True

    def test_yaml_round_trip_with_compact_string_locations(self, tmp_path):
        source = OpenMeteoSource(
            locations="warsaw:52.23:21.01:3.0, krakow:50.06:19.94:1.0",
            columns=["temperature_2m"],
            timezone=TZ,
        )
        path = tmp_path / "dp.yaml"
        DataPipeline().add_source(source).save(path)
        loaded = DataPipeline.load(path).sources[0]
        assert isinstance(loaded, OpenMeteoSource)
        assert loaded.get_cache_config() == source.get_cache_config()

    def test_yaml_round_trip_reproduces_identical_data(self, tmp_path):
        """A `locations` dict round-tripped through YAML must fetch identically (mocked HTTP)."""
        from unittest.mock import Mock

        from tests.data.sources.test_open_meteo import make_archive

        def mock_http(src):
            for underlying in src._location_sources:
                underlying.session = Mock()
                underlying.session.get.side_effect = make_archive(step_h=1)
                underlying.console.quiet = True

        source = OpenMeteoSource(
            locations=LOCATIONS, columns=["temperature_2m"], horizon=1, model="icon_seamless", timezone=TZ
        )
        mock_http(source)

        path = tmp_path / "dp.yaml"
        DataPipeline().add_source(source).save(path)
        loaded = DataPipeline.load(path).sources[0]
        mock_http(loaded)

        start, end = pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-12", tz="UTC")
        original = source.fetch(start, end)
        reloaded = loaded.fetch(start, end)
        pd.testing.assert_frame_equal(original, reloaded)


# ---------------------------------------------------------------------------
# api="single_runs": the same locations logic, inherited from OpenMeteoBase
# ---------------------------------------------------------------------------


class TestSingleRunsApiLocations:
    def test_weighted_average(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            api="single_runs",
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert df["temperature_2m_d+1"].iloc[0] == 0.75 * 10 + 0.25 * 30

    def test_per_location_columns_without_weights(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            api="single_runs",
            locations=LOCATIONS_NO_WEIGHT,
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert list(df.columns) == ["warsaw_temperature_2m_d+1", "krakow_temperature_2m_d+1"]

    def test_cache_config_contains_locations(self):
        source = OpenMeteoSource(locations=LOCATIONS, columns=["temperature_2m"], timezone=TZ, api="single_runs")
        config = source.get_cache_config()
        assert "locations" in config
        assert "latitude" not in config
        assert config["api"] == "single_runs"
