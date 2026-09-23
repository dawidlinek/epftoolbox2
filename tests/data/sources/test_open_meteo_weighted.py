import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.sources.open_meteo import OpenMeteoSource
from epftoolbox2.data.sources.open_meteo_single_runs import OpenMeteoSingleRunsSource
from epftoolbox2.data.sources.open_meteo_weighted import OpenMeteoWeightedSource, parse_locations
from epftoolbox2.pipelines import DataPipeline

LOCATIONS = [
    {"name": "warsaw", "latitude": 52.23, "longitude": 21.01, "weight": 3.0},
    {"name": "krakow", "latitude": 50.06, "longitude": 19.94, "weight": 1.0},
]


def _frame(index, **columns):
    return pd.DataFrame(columns, index=index)


def make_source(locations=None, fetch_returns=None, **kwargs):
    """Build a weighted source and stub each underlying source's `fetch` with `fetch_returns[i]`."""
    kwargs.setdefault("source_params", {"columns": ["temperature_2m"], "horizon": 1})
    source = OpenMeteoWeightedSource(locations=locations or LOCATIONS, **kwargs)
    if fetch_returns is not None:
        for underlying, frame in zip(source._sources, fetch_returns):
            underlying.fetch = lambda *a, _f=frame, **k: _f
    return source


# ---------------------------------------------------------------------------
# parse_locations
# ---------------------------------------------------------------------------


class TestParseLocations:
    def test_parses_list_of_dicts(self):
        parsed = parse_locations(LOCATIONS)
        assert parsed == LOCATIONS

    def test_parses_compact_string(self):
        parsed = parse_locations("warsaw:52.23:21.01:3.0, krakow:50.06:19.94:1.0")
        assert parsed == LOCATIONS

    def test_string_with_wrong_field_count_raises(self):
        with pytest.raises(ValueError, match="name:latitude:longitude:weight"):
            parse_locations("warsaw:52.23:21.01")

    def test_string_with_non_numeric_field_raises(self):
        with pytest.raises(ValueError, match="name:latitude:longitude:weight"):
            parse_locations("warsaw:abc:21.01:3.0")

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="non-empty list"):
            parse_locations([])
        with pytest.raises(ValueError, match="non-empty list"):
            parse_locations("")

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="exactly name, latitude, longitude and weight"):
            parse_locations([{"name": "warsaw", "latitude": 52.23, "longitude": 21.01}])

    def test_extra_key_raises(self):
        bad = [{"name": "warsaw", "latitude": 52.23, "longitude": 21.01, "weight": 1.0, "extra": 1}]
        with pytest.raises(ValueError, match="exactly name, latitude, longitude and weight"):
            parse_locations(bad)

    def test_duplicate_name_raises(self):
        dup = [LOCATIONS[0], dict(LOCATIONS[0])]
        with pytest.raises(ValueError, match="duplicate location name"):
            parse_locations(dup)

    @pytest.mark.parametrize("name", ["has space", "has-dash", "", "accént"])
    def test_invalid_name_characters_raise(self, name):
        bad = [{"name": name, "latitude": 52.23, "longitude": 21.01, "weight": 1.0}]
        with pytest.raises(ValueError, match="letters, digits and underscores"):
            parse_locations(bad)

    @pytest.mark.parametrize("weight", [0, -1.0, float("nan"), "1.0", True])
    def test_invalid_weight_raises(self, weight):
        bad = [{"name": "warsaw", "latitude": 52.23, "longitude": 21.01, "weight": weight}]
        with pytest.raises(ValueError, match="weight"):
            parse_locations(bad)


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self):
        source = OpenMeteoWeightedSource(locations=LOCATIONS)
        assert source.source == "OpenMeteoSource"
        assert source.keep_locations is False
        assert source.prefix == ""
        assert len(source._sources) == 2
        assert all(isinstance(s, OpenMeteoSource) for s in source._sources)

    def test_can_use_single_runs_source(self):
        source = OpenMeteoWeightedSource(locations=LOCATIONS, source="OpenMeteoSingleRunsSource")
        assert all(isinstance(s, OpenMeteoSingleRunsSource) for s in source._sources)

    def test_rejects_unknown_source(self):
        with pytest.raises(ValueError, match="source must be one of"):
            OpenMeteoWeightedSource(locations=LOCATIONS, source="NotASource")

    @pytest.mark.parametrize("reserved", ["latitude", "longitude", "prefix"])
    def test_rejects_reserved_source_params(self, reserved):
        with pytest.raises(ValueError, match="source_params must not set"):
            OpenMeteoWeightedSource(locations=LOCATIONS, source_params={reserved: 1})

    def test_rejects_non_bool_keep_locations(self):
        with pytest.raises(ValueError, match="keep_locations must be True or False"):
            OpenMeteoWeightedSource(locations=LOCATIONS, keep_locations="yes")

    def test_source_params_reach_the_underlying_sources(self):
        source = OpenMeteoWeightedSource(
            locations=LOCATIONS, source_params={"columns": ["temperature_2m"], "horizon": 3, "model": "icon_seamless"}
        )
        for underlying, loc in zip(source._sources, LOCATIONS):
            assert underlying.latitude == loc["latitude"]
            assert underlying.longitude == loc["longitude"]
            assert underlying.horizon == 3
            assert underlying.model == "icon_seamless"

    def test_weights_are_normalised(self):
        source = OpenMeteoWeightedSource(locations=LOCATIONS)
        np.testing.assert_allclose(source._weights, [0.75, 0.25])


# ---------------------------------------------------------------------------
# fetch: weighted averages
# ---------------------------------------------------------------------------


class TestFetchWeightedAverage:
    def test_weighted_average_matches_manual_calculation(self):
        index = pd.date_range("2024-01-10", periods=2, freq="1h", tz="UTC")
        source = make_source(
            fetch_returns=[
                _frame(index, temperature_2m_d1=[10.0, 20.0]).rename(
                    columns={"temperature_2m_d1": "temperature_2m_d+1"}
                ),
                _frame(index, x=[30.0, 0.0]).rename(columns={"x": "temperature_2m_d+1"}),
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
            locations=[
                {"name": "a", "latitude": 52.23, "longitude": 21.01, "weight": 1.0},
                {"name": "b", "latitude": 50.06, "longitude": 19.94, "weight": 1.0},
            ],
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
# Cache config and serialization
# ---------------------------------------------------------------------------


class TestCacheConfigAndSerialization:
    def test_cache_config_contains_locations_and_source_config(self):
        source = OpenMeteoWeightedSource(locations=LOCATIONS, source_params={"columns": ["temperature_2m"]})
        config = source.get_cache_config()
        assert config["source_type"] == "open_meteo_weighted"
        assert config["schema_version"] >= 1
        assert config["locations"] == LOCATIONS
        assert "latitude" not in config["source_config"]
        assert "longitude" not in config["source_config"]
        assert "prefix" not in config["source_config"]

    def test_changing_a_weight_changes_the_cache_config(self):
        base = OpenMeteoWeightedSource(locations=LOCATIONS).get_cache_config()
        changed_locations = [dict(LOCATIONS[0], weight=5.0), LOCATIONS[1]]
        other = OpenMeteoWeightedSource(locations=changed_locations).get_cache_config()
        assert base != other

    def test_changing_source_params_changes_the_cache_config(self):
        base = OpenMeteoWeightedSource(locations=LOCATIONS, source_params={"horizon": 2}).get_cache_config()
        other = OpenMeteoWeightedSource(locations=LOCATIONS, source_params={"horizon": 3}).get_cache_config()
        assert base != other

    def test_keep_locations_changes_the_cache_config(self):
        base = OpenMeteoWeightedSource(locations=LOCATIONS).get_cache_config()
        other = OpenMeteoWeightedSource(locations=LOCATIONS, keep_locations=True).get_cache_config()
        assert base != other

    def test_yaml_round_trip(self, tmp_path):
        source = OpenMeteoWeightedSource(
            locations=LOCATIONS,
            source="OpenMeteoSource",
            source_params={"columns": ["temperature_2m"], "horizon": 3, "model": "icon_seamless"},
            prefix="pl",
            keep_locations=True,
        )
        path = tmp_path / "dp.yaml"
        DataPipeline().add_source(source).save(path)
        loaded = DataPipeline.load(path).sources[0]
        assert isinstance(loaded, OpenMeteoWeightedSource)
        assert loaded.get_cache_config() == source.get_cache_config()
        assert loaded.keep_locations is True

    def test_yaml_round_trip_with_compact_string_locations(self, tmp_path):
        source = OpenMeteoWeightedSource(
            locations="warsaw:52.23:21.01:3.0, krakow:50.06:19.94:1.0",
            source_params={"columns": ["temperature_2m"]},
        )
        path = tmp_path / "dp.yaml"
        DataPipeline().add_source(source).save(path)
        loaded = DataPipeline.load(path).sources[0]
        assert isinstance(loaded, OpenMeteoWeightedSource)
        assert loaded.get_cache_config() == source.get_cache_config()
