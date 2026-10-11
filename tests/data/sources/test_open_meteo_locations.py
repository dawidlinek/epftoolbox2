"""OpenMeteoSource's `locations` parameter: several locations fetched by another instance of the same class
(sharing `api`/`run`) and combined via `agg` into one or more aggregated series, per-location columns, or both.

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
# agg / keep_locations validation
# ---------------------------------------------------------------------------


class TestAggValidation:
    def test_agg_with_single_point_raises(self):
        with pytest.raises(ValueError, match="agg is only valid with locations"):
            OpenMeteoSource(latitude=52.23, longitude=21.01, agg="mean", columns=["temperature_2m"], timezone=TZ)

    def test_keep_locations_with_single_point_raises(self):
        with pytest.raises(ValueError, match="keep_locations is only valid with locations"):
            OpenMeteoSource(
                latitude=52.23, longitude=21.01, keep_locations=True, columns=["temperature_2m"], timezone=TZ
            )

    def test_keep_locations_without_agg_raises(self):
        with pytest.raises(ValueError, match="keep_locations=True requires agg"):
            make_source(locations=LOCATIONS_NO_WEIGHT, keep_locations=True)

    def test_weights_without_agg_raise(self):
        with pytest.raises(ValueError, match='"mean"'):
            make_source()  # LOCATIONS is weighted; no agg given at all

    def test_weights_with_non_mean_agg_raise(self):
        with pytest.raises(ValueError, match='"mean"'):
            make_source(agg="max")

    def test_weights_with_list_without_mean_raise(self):
        with pytest.raises(ValueError, match='"mean"'):
            make_source(agg=["max", "min"])

    def test_weights_with_mean_in_a_list_is_accepted(self):
        source = make_source(agg=["mean", "max"])
        assert source._agg == ("mean", "max")

    @pytest.mark.parametrize("agg", ["mean", "median", "min", "max", "sum"])
    def test_each_agg_value_is_accepted(self, agg):
        source = make_source(locations=LOCATIONS_NO_WEIGHT, agg=agg)
        assert source._agg == (agg,)

    def test_invalid_agg_string_raises(self):
        with pytest.raises(ValueError, match="agg must be"):
            make_source(locations=LOCATIONS_NO_WEIGHT, agg="average")

    def test_empty_agg_list_raises(self):
        with pytest.raises(ValueError, match="agg must be"):
            make_source(locations=LOCATIONS_NO_WEIGHT, agg=[])

    def test_duplicate_agg_in_list_raises(self):
        with pytest.raises(ValueError, match="not repeat"):
            make_source(locations=LOCATIONS_NO_WEIGHT, agg=["max", "max"])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self):
        source = make_source(locations=LOCATIONS_NO_WEIGHT)
        assert source.agg is None
        assert source.keep_locations is False
        assert source.prefix == ""
        assert len(source._location_sources) == 2
        assert all(isinstance(s, OpenMeteoSource) for s in source._location_sources)

    def test_settings_reach_the_underlying_sources(self):
        source = make_source(
            columns=["temperature_2m"], horizon=3, model="icon_seamless", api="single_runs", run="06", agg="mean"
        )
        for underlying, name in zip(source._location_sources, ("warsaw", "krakow")):
            assert underlying.latitude == LOCATIONS[name]["latitude"]
            assert underlying.longitude == LOCATIONS[name]["longitude"]
            assert underlying.horizon == 3
            assert underlying.model == "icon_seamless"
            assert underlying.api == "single_runs"
            assert underlying.run == "06"
            assert underlying.prefix == ""  # locations carry their own name-based prefixing instead
            assert underlying.agg is None  # not forwarded: each underlying source is a single point

    def test_rejects_non_bool_keep_locations(self):
        with pytest.raises(ValueError, match="keep_locations must be True or False"):
            make_source(agg="mean", keep_locations="yes")

    def test_weights_are_normalised(self):
        source = make_source(agg="mean")
        np.testing.assert_allclose(source._location_weights, [0.75, 0.25])


# ---------------------------------------------------------------------------
# fetch: agg="mean" (weighted when locations carry weights, else plain)
# ---------------------------------------------------------------------------


class TestFetchMean:
    def test_weighted_mean_matches_manual_calculation(self):
        index = pd.date_range("2024-01-10", periods=2, freq="1h", tz="UTC")
        source = make_source(
            agg="mean",
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0, 20.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0, 0.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        # weights normalise to [0.75, 0.25]: 0.75*10 + 0.25*30 = 15; 0.75*20 + 0.25*0 = 15
        np.testing.assert_allclose(df["temperature_2m_mean_d+1"].tolist(), [15.0, 15.0])

    def test_unweighted_mean_without_location_weights(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            locations=LOCATIONS_NO_WEIGHT,
            agg="mean",
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert df["temperature_2m_mean_d+1"].iloc[0] == 20.0  # plain average of 10 and 30

    def test_nan_at_any_location_makes_the_mean_nan(self):
        # The NaN row is not last, so the trailing-row trim does not remove it.
        index = pd.date_range("2024-01-10", periods=3, freq="1h", tz="UTC")
        source = make_source(
            agg="mean",
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0, np.nan, 10.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0, 5.0, 30.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert df["temperature_2m_mean_d+1"].iloc[0] == 15.0
        assert np.isnan(df["temperature_2m_mean_d+1"].iloc[1])
        assert df["temperature_2m_mean_d+1"].iloc[2] == 15.0

    def test_prefix_applied_to_aggregated_columns(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            agg="mean",
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0]}),
            ],
            prefix="pl",
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert list(df.columns) == ["pl_temperature_2m_mean_d+1"]

    def test_indices_are_unioned_and_missing_rows_are_nan(self):
        idx1 = pd.date_range("2024-01-10 00:00", periods=2, freq="1h", tz="UTC")
        idx2 = pd.date_range("2024-01-10 01:00", periods=2, freq="1h", tz="UTC")  # one hour later, one hour short
        source = make_source(
            agg="mean",
            fetch_returns=[
                _frame(idx1, **{"temperature_2m_d+1": [10.0, 20.0]}),
                _frame(idx2, **{"temperature_2m_d+1": [30.0, 40.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        # 00:00 only in location 1 -> NaN; 01:00 and 02:00 both present -> weighted; last row trimmed by the tail
        # trim only if fully NaN, so 00:00 (partially NaN combined -> NaN) is dropped only if it's the very end.
        assert pd.Timestamp("2024-01-10 00:00", tz="UTC") in df.index
        assert np.isnan(df.loc["2024-01-10 00:00", "temperature_2m_mean_d+1"])
        np.testing.assert_allclose(df.loc["2024-01-10 01:00", "temperature_2m_mean_d+1"], 0.75 * 20 + 0.25 * 30)

    def test_all_empty_frames_return_empty_dataframe(self):
        source = make_source(agg="mean", fetch_returns=[pd.DataFrame(), pd.DataFrame()])
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert df.empty

    def test_trailing_all_nan_rows_are_dropped(self):
        index = pd.date_range("2024-01-10", periods=3, freq="1h", tz="UTC")
        source = make_source(
            agg="mean",
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0, 20.0, np.nan]}),
                _frame(index, **{"temperature_2m_d+1": [30.0, 40.0, np.nan]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert len(df) == 2


# ---------------------------------------------------------------------------
# fetch: the other aggregations (always unweighted)
# ---------------------------------------------------------------------------


class TestOtherAggregations:
    @pytest.mark.parametrize("agg, expected", [("median", 20.0), ("min", 10.0), ("max", 30.0), ("sum", 40.0)])
    def test_unweighted_aggregation_matches_manual_calculation(self, agg, expected):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            locations=LOCATIONS_NO_WEIGHT,
            agg=agg,
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert df[f"temperature_2m_{agg}_d+1"].iloc[0] == expected

    def test_nan_at_any_location_makes_it_nan(self):
        # The NaN row is not last, so the trailing-row trim (drops rows with no value at all) does not remove it.
        index = pd.date_range("2024-01-10", periods=2, freq="1h", tz="UTC")
        source = make_source(
            locations=LOCATIONS_NO_WEIGHT,
            agg="max",
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0, 10.0]}),
                _frame(index, **{"temperature_2m_d+1": [np.nan, 30.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert np.isnan(df["temperature_2m_max_d+1"].iloc[0])
        assert df["temperature_2m_max_d+1"].iloc[1] == 30.0

    def test_weights_have_no_effect_on_non_mean_aggregations(self):
        # "mean" is weighted; every other aggregation is always unweighted, even when weights are given
        # (weights are otherwise rejected unless "mean" is one of the requested aggregations).
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            agg=["mean", "max"],
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert df["temperature_2m_max_d+1"].iloc[0] == 30.0  # not weight-adjusted
        assert df["temperature_2m_mean_d+1"].iloc[0] == 0.75 * 10 + 0.25 * 30  # weighted

    def test_weighted_mean_and_median_together(self):
        # Same combination as above, with "median" standing in for every unweighted aggregation: it must stay
        # the plain (unweighted) median of the three locations' values even though "mean" in the same call is
        # weighted.
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            locations={
                "warsaw": {"latitude": 52.23, "longitude": 21.01, "weight": 3.0},
                "krakow": {"latitude": 50.06, "longitude": 19.94, "weight": 1.0},
                "gdansk": {"latitude": 54.35, "longitude": 18.65, "weight": 1.0},
            },
            agg=["mean", "median"],
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0]}),
                _frame(index, **{"temperature_2m_d+1": [20.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        # weights normalise to [0.6, 0.2, 0.2]: 0.6*10 + 0.2*20 + 0.2*30 = 16
        assert df["temperature_2m_mean_d+1"].iloc[0] == pytest.approx(16.0)
        assert df["temperature_2m_median_d+1"].iloc[0] == 20.0  # plain median of [10, 20, 30], not weight-adjusted


class TestAggListNaming:
    def test_list_suffixes_every_entry_even_with_one(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            locations=LOCATIONS_NO_WEIGHT,
            agg=["max"],
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert list(df.columns) == ["temperature_2m_max_d+1"]

    def test_bare_string_is_suffixed_the_same_way_as_a_one_element_list(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            locations=LOCATIONS_NO_WEIGHT,
            agg="max",
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert list(df.columns) == ["temperature_2m_max_d+1"]

    def test_multiple_aggregations_each_get_their_own_column(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            locations=LOCATIONS_NO_WEIGHT,
            agg=["min", "max", "median", "sum"],
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert list(df.columns) == [
            "temperature_2m_min_d+1",
            "temperature_2m_max_d+1",
            "temperature_2m_median_d+1",
            "temperature_2m_sum_d+1",
        ]
        assert df["temperature_2m_min_d+1"].iloc[0] == 10.0
        assert df["temperature_2m_max_d+1"].iloc[0] == 30.0
        assert df["temperature_2m_median_d+1"].iloc[0] == 20.0
        assert df["temperature_2m_sum_d+1"].iloc[0] == 40.0


class TestWindDirection:
    def test_circular_mean_wraps_around_zero(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            locations={
                "a": {"latitude": 52.23, "longitude": 21.01, "weight": 1.0},
                "b": {"latitude": 50.06, "longitude": 19.94, "weight": 1.0},
            },
            agg="mean",
            fetch_returns=[
                _frame(index, **{"wind_direction_10m_d+1": [350.0]}),
                _frame(index, **{"wind_direction_10m_d+1": [10.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        # A plain arithmetic mean would give 180 deg (due south); the circular mean must be near 0/360.
        value = df["wind_direction_10m_mean_d+1"].iloc[0]
        assert value < 5 or value > 355

    def test_direction_is_skipped_for_max(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            locations=LOCATIONS_NO_WEIGHT,
            agg="max",
            columns=["temperature_2m", "wind_direction_10m"],
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0], "wind_direction_10m_d+1": [350.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0], "wind_direction_10m_d+1": [10.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert "wind_direction_10m_max_d+1" not in df.columns
        assert df["temperature_2m_max_d+1"].iloc[0] == 30.0

    def test_direction_kept_for_mean_but_skipped_for_max_in_a_list(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            locations=LOCATIONS_NO_WEIGHT,
            agg=["mean", "max"],
            columns=["wind_direction_10m"],
            fetch_returns=[
                _frame(index, **{"wind_direction_10m_d+1": [350.0]}),
                _frame(index, **{"wind_direction_10m_d+1": [10.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert list(df.columns) == ["wind_direction_10m_mean_d+1"]


class TestWeatherCodeColumn:
    def test_weather_code_is_aggregated_like_any_other_numeric_column(self):
        # WMO codes are ordinal (0 = clear, higher = worse), so the usual aggregations are meaningful.
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            locations=LOCATIONS_NO_WEIGHT,
            agg="max",
            columns=["temperature_2m", "weather_code"],
            fetch_returns=[
                _frame(index, **{"weather_code_d+1": [1.0], "temperature_2m_d+1": [10.0]}),
                _frame(index, **{"weather_code_d+1": [2.0], "temperature_2m_d+1": [30.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert df["weather_code_max_d+1"].iloc[0] == 2.0
        assert df["temperature_2m_max_d+1"].iloc[0] == 30.0

    def test_keep_locations_also_keeps_the_per_location_weather_code(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            agg="mean",
            columns=["temperature_2m", "weather_code"],
            keep_locations=True,
            fetch_returns=[
                _frame(index, **{"weather_code_d+1": [1.0], "temperature_2m_d+1": [10.0]}),
                _frame(index, **{"weather_code_d+1": [2.0], "temperature_2m_d+1": [30.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert df["weather_code_mean_d+1"].iloc[0] == 1.25  # weighted mean: 0.75*1 + 0.25*2
        assert df["warsaw_weather_code_d+1"].iloc[0] == 1.0
        assert df["krakow_weather_code_d+1"].iloc[0] == 2.0


class TestKeepLocations:
    def test_per_location_columns_present_with_prefix(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            agg="mean",
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0]}),
            ],
            keep_locations=True,
            prefix="pl",
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert "pl_temperature_2m_mean_d+1" in df.columns
        assert "pl_warsaw_temperature_2m_d+1" in df.columns
        assert "pl_krakow_temperature_2m_d+1" in df.columns
        assert df["pl_warsaw_temperature_2m_d+1"].iloc[0] == 10.0
        assert df["pl_krakow_temperature_2m_d+1"].iloc[0] == 30.0

    def test_default_does_not_include_per_location_columns(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            agg="mean",
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert list(df.columns) == ["temperature_2m_mean_d+1"]


# ---------------------------------------------------------------------------
# fetch: agg=None, per-location columns only
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


class TestSharedNow:
    def test_every_location_fetches_against_the_same_now(self):
        # Fetching locations one after another can let wall-clock time advance between them; if each location
        # resolved "now" independently, live rows near the cutoff could end up on different sides of it for
        # different locations, mixing vintages within what should be one consistent row.
        source = make_source(agg="mean")
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
        source = OpenMeteoSource(locations=LOCATIONS, agg="mean", columns=["temperature_2m"], timezone=TZ)
        config = source.get_cache_config()
        assert config["locations"] == [
            {"name": "warsaw", "latitude": 52.23, "longitude": 21.01, "weight": 3.0},
            {"name": "krakow", "latitude": 50.06, "longitude": 19.94, "weight": 1.0},
        ]
        assert "latitude" not in config
        assert "longitude" not in config
        assert "keep_locations" not in config  # only included when True
        assert config["agg"] == ["mean"]

    def test_changing_a_weight_changes_the_cache_config(self):
        base = OpenMeteoSource(
            locations=LOCATIONS, agg="mean", columns=["temperature_2m"], timezone=TZ
        ).get_cache_config()
        changed = dict(LOCATIONS, warsaw={**LOCATIONS["warsaw"], "weight": 5.0})
        other = OpenMeteoSource(
            locations=changed, agg="mean", columns=["temperature_2m"], timezone=TZ
        ).get_cache_config()
        assert base != other

    def test_changing_a_shared_setting_changes_the_cache_config(self):
        base = OpenMeteoSource(locations=LOCATIONS, agg="mean", columns=["temperature_2m"], horizon=2, timezone=TZ)
        other = OpenMeteoSource(locations=LOCATIONS, agg="mean", columns=["temperature_2m"], horizon=3, timezone=TZ)
        assert base.get_cache_config() != other.get_cache_config()

    def test_keep_locations_changes_the_cache_config(self):
        base = OpenMeteoSource(
            locations=LOCATIONS, agg="mean", columns=["temperature_2m"], timezone=TZ
        ).get_cache_config()
        other = OpenMeteoSource(
            locations=LOCATIONS, agg="mean", columns=["temperature_2m"], timezone=TZ, keep_locations=True
        ).get_cache_config()
        assert base != other

    def test_agg_as_string_and_as_one_element_list_share_a_cache_config(self):
        # They also produce identically-named output columns (see TestAggListNaming), so they can share a
        # cache key.
        bare = OpenMeteoSource(
            locations=LOCATIONS_NO_WEIGHT, agg="max", columns=["temperature_2m"], timezone=TZ
        ).get_cache_config()
        listed = OpenMeteoSource(
            locations=LOCATIONS_NO_WEIGHT, agg=["max"], columns=["temperature_2m"], timezone=TZ
        ).get_cache_config()
        assert bare == listed

    def test_locations_and_single_point_configs_are_never_equal(self):
        point = OpenMeteoSource(latitude=52.23, longitude=21.01, columns=["temperature_2m"], timezone=TZ)
        located = OpenMeteoSource(
            locations={"warsaw": {"latitude": 52.23, "longitude": 21.01}}, columns=["temperature_2m"], timezone=TZ
        )
        assert point.get_cache_config() != located.get_cache_config()

    def test_yaml_round_trip(self, tmp_path):
        source = OpenMeteoSource(
            locations=LOCATIONS,
            agg="mean",
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
        assert loaded.agg == "mean"

    def test_yaml_round_trip_with_agg_list(self, tmp_path):
        source = OpenMeteoSource(
            locations=LOCATIONS_NO_WEIGHT, agg=["max", "min"], columns=["temperature_2m"], timezone=TZ
        )
        path = tmp_path / "dp.yaml"
        DataPipeline().add_source(source).save(path)
        loaded = DataPipeline.load(path).sources[0]
        assert loaded.agg == ["max", "min"]
        assert loaded.get_cache_config() == source.get_cache_config()

    def test_yaml_round_trip_with_compact_string_locations(self, tmp_path):
        source = OpenMeteoSource(
            locations="warsaw:52.23:21.01:3.0, krakow:50.06:19.94:1.0",
            agg="mean",
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
            locations=LOCATIONS,
            agg="mean",
            columns=["temperature_2m"],
            horizon=1,
            model="icon_seamless",
            timezone=TZ,
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
    def test_weighted_mean(self):
        index = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        source = make_source(
            api="single_runs",
            agg="mean",
            fetch_returns=[
                _frame(index, **{"temperature_2m_d+1": [10.0]}),
                _frame(index, **{"temperature_2m_d+1": [30.0]}),
            ],
        )
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert df["temperature_2m_mean_d+1"].iloc[0] == 0.75 * 10 + 0.25 * 30

    def test_per_location_columns_without_agg(self):
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
        source = OpenMeteoSource(
            locations=LOCATIONS, agg="mean", columns=["temperature_2m"], timezone=TZ, api="single_runs"
        )
        config = source.get_cache_config()
        assert "locations" in config
        assert "latitude" not in config
        assert config["api"] == "single_runs"
