import warnings

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.transformers import ResampleTransformer, TimezoneTransformer


class TestTimezoneTransformerInit:
    """Test TimezoneTransformer initialization"""

    def test_init_valid_timezone(self):
        """Test initialization with valid timezone"""
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")
        assert transformer.target_tz == "Europe/Warsaw"

    def test_init_utc(self):
        """Test initialization with UTC"""
        transformer = TimezoneTransformer(target_tz="UTC")
        assert transformer.target_tz == "UTC"

    def test_init_invalid_timezone(self):
        """Test initialization with invalid timezone raises error"""
        with pytest.raises(ValueError, match="Invalid timezone"):
            TimezoneTransformer(target_tz="Invalid/Timezone")


class TestTimezoneTransformerTransform:
    """Test TimezoneTransformer transform method"""

    @pytest.fixture
    def sample_utc_dataframe(self):
        """Create a sample DataFrame with UTC index"""
        dates = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        return pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=dates)

    @pytest.fixture
    def sample_naive_dataframe(self):
        """Create a sample DataFrame with timezone-naive index"""
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        return pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=dates)

    def test_transform_utc_to_warsaw(self, sample_utc_dataframe):
        """Test converting UTC to Europe/Warsaw"""
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")
        result = transformer.transform(sample_utc_dataframe)

        assert str(result.index.tz) == "Europe/Warsaw"
        # Warsaw is UTC+1 in winter
        assert result.index[0].hour == 1  # 00:00 UTC -> 01:00 Warsaw

    def test_transform_utc_to_new_york(self, sample_utc_dataframe):
        """Test converting UTC to America/New_York"""
        transformer = TimezoneTransformer(target_tz="America/New_York")
        result = transformer.transform(sample_utc_dataframe)

        assert str(result.index.tz) == "America/New_York"
        # New York is UTC-5 in winter
        assert result.index[0].hour == 19  # 00:00 UTC on Jan 1 -> 19:00 Dec 31 NY

    def test_transform_naive_to_warsaw(self, sample_naive_dataframe):
        """Test converting naive timestamps (assumed UTC) to Europe/Warsaw"""
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")
        result = transformer.transform(sample_naive_dataframe)

        assert str(result.index.tz) == "Europe/Warsaw"
        assert result.index[0].hour == 1  # 00:00 UTC -> 01:00 Warsaw

    def test_transform_preserves_data(self, sample_utc_dataframe):
        """Test that transform preserves the data values"""
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")
        result = transformer.transform(sample_utc_dataframe)

        assert list(result["value"]) == [1, 2, 3, 4, 5]

    def test_transform_does_not_modify_original(self, sample_utc_dataframe):
        """Test that transform returns a copy, not modifying original"""
        original_tz = str(sample_utc_dataframe.index.tz)
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")
        transformer.transform(sample_utc_dataframe)

        assert str(sample_utc_dataframe.index.tz) == original_tz

    def test_transform_invalid_index_type(self):
        """Test that non-DatetimeIndex raises error"""
        df = pd.DataFrame({"value": [1, 2, 3]}, index=[0, 1, 2])
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")

        with pytest.raises(ValueError, match="DatetimeIndex"):
            transformer.transform(df)

    def test_transform_multiple_columns(self):
        """Test transform with multiple columns"""
        dates = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "price": [10.0, 20.0, 30.0],
                "load": [100, 200, 300],
            },
            index=dates,
        )

        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")
        result = transformer.transform(df)

        assert list(result.columns) == ["price", "load"]
        assert list(result["price"]) == [10.0, 20.0, 30.0]
        assert list(result["load"]) == [100, 200, 300]


class TestTimezoneTransformerDailySnap:
    """Test daily_columns midnight-snapping for winter and summer cases."""

    def _make_df(self, hourly_utc_start: str, periods: int, daily_utc_ts: str, daily_value: float):
        """Build a DataFrame with hourly rows and one sparse daily column."""
        idx = pd.date_range(hourly_utc_start, periods=periods, freq="h", tz="UTC")
        df = pd.DataFrame({"load": range(periods)}, index=idx)
        df["daily_min"] = float("nan")
        if daily_utc_ts in df.index:
            df.loc[daily_utc_ts, "daily_min"] = daily_value
        else:
            # Insert the daily row even if it's outside the hourly grid
            extra = pd.DataFrame(
                {"load": float("nan"), "daily_min": daily_value},
                index=pd.DatetimeIndex([pd.Timestamp(daily_utc_ts, tz="UTC")]),
            )
            df = pd.concat([df, extra]).sort_index()
        return df

    def test_winter_snap_23xx_to_next_midnight(self):
        """An explicit UTC+2 anchor (not this data's own zone) puts daily value at 23:00+01:00 CET winter.
        Should snap to 00:00+01:00 of the *next* local day (Wednesday), not Tuesday."""
        # 2025-11-18 22:00 UTC = 2025-11-18 23:00+01:00 CET (1h before Wed midnight)
        df = self._make_df("2025-11-17 00:00", periods=72, daily_utc_ts="2025-11-18 22:00", daily_value=99.0)
        transformer = TimezoneTransformer(
            target_tz="Europe/Madrid", daily_columns=["daily_min"], daily_anchor="UTC+02:00"
        )
        result = transformer.transform(df)

        # Value must land at Wednesday midnight (2025-11-19 00:00+01:00)
        wednesday_midnight = pd.Timestamp("2025-11-19 00:00", tz="Europe/Madrid")
        assert result.loc[wednesday_midnight, "daily_min"] == 99.0

        # Tuesday midnight must NOT have the value
        tuesday_midnight = pd.Timestamp("2025-11-18 00:00", tz="Europe/Madrid")
        assert pd.isna(result.loc[tuesday_midnight, "daily_min"])

    def test_summer_snap_00xx_already_midnight(self):
        """CEST summer: the default anchor (target_tz) puts daily value at 00:00+02:00.
        Already at midnight — no shift should occur."""
        # 2025-07-15 22:00 UTC = 2025-07-16 00:00+02:00 CEST (midnight)
        df = self._make_df("2025-07-15 00:00", periods=72, daily_utc_ts="2025-07-15 22:00", daily_value=55.0)
        transformer = TimezoneTransformer(target_tz="Europe/Madrid", daily_columns=["daily_min"])
        result = transformer.transform(df)

        wednesday_midnight = pd.Timestamp("2025-07-16 00:00", tz="Europe/Madrid")
        assert result.loc[wednesday_midnight, "daily_min"] == 55.0

    def test_winter_snap_01xx_to_current_midnight(self):
        """UTC+2 period-start (00:00 UTC) converted to UTC+1: lands at 01:00+01:00.
        Should snap to 00:00+01:00 of the *same* local day."""
        # 2025-11-19 00:00 UTC = 2025-11-19 01:00+01:00 CET
        df = self._make_df("2025-11-18 00:00", periods=72, daily_utc_ts="2025-11-19 00:00", daily_value=77.0)
        transformer = TimezoneTransformer(target_tz="Europe/Madrid", daily_columns=["daily_min"])
        result = transformer.transform(df)

        target_midnight = pd.Timestamp("2025-11-19 00:00", tz="Europe/Madrid")
        assert result.loc[target_midnight, "daily_min"] == 77.0

    def test_already_at_midnight_no_shift(self):
        """Value already at local midnight: needs_shift must be False, no movement."""
        # 2025-11-18 23:00 UTC = 2025-11-19 00:00+01:00 CET (exactly midnight)
        df = self._make_df("2025-11-18 00:00", periods=72, daily_utc_ts="2025-11-18 23:00", daily_value=42.0)
        transformer = TimezoneTransformer(target_tz="Europe/Madrid", daily_columns=["daily_min"])
        result = transformer.transform(df)

        midnight = pd.Timestamp("2025-11-19 00:00", tz="Europe/Madrid")
        assert result.loc[midnight, "daily_min"] == 42.0

    def test_non_daily_columns_unaffected(self):
        """Columns not in daily_columns must not be moved."""
        idx = pd.date_range("2025-11-18 00:00", periods=48, freq="h", tz="UTC")
        df = pd.DataFrame({"load": range(48), "price": range(48)}, index=idx)
        transformer = TimezoneTransformer(target_tz="Europe/Madrid", daily_columns=["daily_min"])
        result = transformer.transform(df)
        # price column should be untouched (no snapping)
        assert list(result["price"]) == list(range(48))


class TestDailyColumnPrefixMatching:
    """`daily_columns` matches a column by its exact name or by a `f"_{name}"` suffix, so a prefixed neighbour's
    column (e.g. `EntsoeSource(country_code="FI", prefix="fi")`'s `fi_load_forecast_daily_min`) is recognised
    under the default `daily_columns` without listing every prefix by name."""

    def test_prefixed_column_is_snapped_like_the_bare_one(self):
        idx = pd.date_range("2025-01-13 00:00", periods=48, freq="h", tz="UTC")
        df = pd.DataFrame({"load": range(48)}, index=idx)
        df["fi_load_forecast_daily_min"] = float("nan")
        df.loc[pd.Timestamp("2025-01-13 23:00", tz="UTC"), "fi_load_forecast_daily_min"] = 10.0
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")  # default daily_columns
        result = transformer.transform(df)
        value = result["fi_load_forecast_daily_min"].dropna()
        assert list(value.index) == [pd.Timestamp("2025-01-14 00:00", tz="Europe/Warsaw")]
        assert value.iloc[0] == 10.0

    def test_unrelated_column_with_the_same_ending_is_not_matched(self):
        # only an exact name or one ending in "_<daily column>" counts, not merely containing the substring
        idx = pd.date_range("2025-01-13 00:00", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"xload_forecast_daily_min": range(24)}, index=idx)
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")
        result = transformer.transform(df)
        assert list(result["xload_forecast_daily_min"]) == list(range(24))

    def test_custom_daily_columns_also_match_by_suffix(self):
        idx = pd.date_range("2025-01-13 00:00", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"a_custom_daily": [1.0] + [float("nan")] * 23}, index=idx)
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw", daily_columns=["custom_daily"])
        result = transformer.transform(df)
        value = result["a_custom_daily"].dropna()
        assert list(value.index) == [pd.Timestamp("2025-01-13 00:00", tz="Europe/Warsaw")]


def _a31_frame(first_utc: str, days: int, stamps: list[str], values: list[float], col="load_forecast_daily_min"):
    """Hourly UTC frame plus a sparse daily column holding ENTSO-E A31-style values at the given UTC stamps."""
    idx = pd.date_range(first_utc, periods=24 * days, freq="1h", tz="UTC")
    df = pd.DataFrame({"load_actual": 1.0}, index=idx)
    df[col] = np.nan
    for stamp, value in zip(stamps, values):
        df.loc[pd.Timestamp(stamp, tz="UTC"), col] = value
    return df.sort_index()


class TestDailyAnchorRule:
    """review F7: a daily value belongs to its calendar date in `daily_anchor` (default: the resolved `target_tz`
    itself — ENTSO-E always stamps A31 at the *area's own* local midnight, not a fixed UTC+2 offset; DST/timezone
    review finding 1). Passing an explicit `daily_anchor` (e.g. a fixed UTC offset shared by several zones) is
    still supported and exercised below, decoupled from whatever the default happens to be."""

    def test_defaults(self):
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")
        assert transformer.daily_columns == ["load_forecast_daily_min", "load_forecast_daily_max"]
        assert transformer.daily_anchor is None

    @pytest.mark.parametrize(
        "tz", ["Europe/Warsaw", "Europe/Helsinki", "Europe/Lisbon", "Europe/London", "America/New_York", "Asia/Tokyo"]
    )
    @pytest.mark.parametrize("stamp, day", [("2024-01-09 22:00", "2024-01-10"), ("2024-07-09 22:00", "2024-07-10")])
    def test_explicit_utc_plus_2_anchor_lands_on_its_local_day(self, tz, stamp, day):
        """A value genuinely anchored to a fixed UTC+2 offset (e.g. a non-ENTSO-E source) belongs to the same
        calendar date in every target zone, via an explicit `daily_anchor="UTC+02:00"`."""
        first = (pd.Timestamp(stamp) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        df = _a31_frame(first, 5, [stamp], [111.0])
        out = TimezoneTransformer(target_tz=tz, daily_anchor="UTC+02:00").transform(df)
        landed = out["load_forecast_daily_min"].dropna()
        assert landed.index.tolist() == [pd.Timestamp(day, tz=tz)]
        assert landed.iloc[0] == 111.0

    @pytest.mark.parametrize(
        "stamps, days",
        [
            (["2024-03-29 22:00", "2024-03-30 22:00", "2024-03-31 22:00"], ["2024-03-30", "2024-03-31", "2024-04-01"]),
            (["2024-10-25 22:00", "2024-10-26 22:00", "2024-10-27 22:00"], ["2024-10-26", "2024-10-27", "2024-10-28"]),
        ],
    )
    def test_dst_switch_days_explicit_anchor(self, stamps, days):
        """Same explicit-UTC+2-anchor mechanism, exercised across both DST switch directions."""
        df = _a31_frame(stamps[0][:10], 5, stamps, [1.0, 2.0, 3.0])
        out = TimezoneTransformer(target_tz="Europe/Warsaw", daily_anchor="UTC+02:00").transform(df)
        landed = out["load_forecast_daily_min"].dropna()
        assert landed.index.tolist() == [pd.Timestamp(d, tz="Europe/Warsaw") for d in days]
        assert landed.tolist() == [1.0, 2.0, 3.0]

    def test_values_stamped_at_local_midnight_east_of_utc_plus_2(self):
        """Data stamped at local midnight in EEST (UTC+3) still works with an explicit daily_anchor=target_tz
        (now also the default — see TestDefaultAnchorIsTargetZone)."""
        df = _a31_frame("2024-07-08", 4, ["2024-07-09 21:00"], [5.0])  # 00:00 EEST on 2024-07-10
        out = TimezoneTransformer(target_tz="Europe/Helsinki", daily_anchor="Europe/Helsinki").transform(df)
        landed = out["load_forecast_daily_min"].dropna()
        assert landed.index.tolist() == [pd.Timestamp("2024-07-10", tz="Europe/Helsinki")]


class TestDefaultAnchorIsTargetZone:
    """DST/timezone review finding 1: ENTSO-E always stamps A31 daily values at the *area's own* local midnight.
    The default `daily_anchor` (None) must therefore resolve to `target_tz`, not a fixed UTC+2 offset — which
    silently misplaces every value in EET/EEST zones every summer, and every value, year-round, in a zone with
    a fixed UTC+3 offset (real FI data: 336/336 hourly rows wrong under the old default, 2024-06-03..16)."""

    @pytest.mark.parametrize(
        "tz, local_day",
        [
            ("Europe/Helsinki", "2024-01-10"),  # EET winter, UTC+2 — coincides with the old fixed anchor
            ("Europe/Helsinki", "2024-07-10"),  # EEST summer, UTC+3 — the old fixed anchor got this wrong
            ("Europe/Istanbul", "2024-01-10"),  # fixed UTC+3, no DST — always wrong under the old anchor
            ("Europe/Istanbul", "2024-07-10"),
        ],
    )
    def test_value_at_the_areas_own_midnight_lands_on_its_own_day(self, tz, local_day):
        local_midnight = pd.Timestamp(local_day, tz=tz)
        stamp = local_midnight.tz_convert("UTC").strftime("%Y-%m-%d %H:%M")
        first = (local_midnight - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        df = _a31_frame(first, 5, [stamp], [111.0])
        out = TimezoneTransformer(target_tz=tz).transform(df)  # default daily_anchor
        landed = out["load_forecast_daily_min"].dropna()
        assert landed.index.tolist() == [local_midnight]
        assert landed.iloc[0] == 111.0

    @pytest.mark.parametrize(
        "tz",
        ["Europe/Warsaw", "Europe/Helsinki"],  # CET/CEST (correct under the old default too) and EET/EEST
    )
    @pytest.mark.parametrize(
        "local_days",
        [
            ["2024-03-30", "2024-03-31", "2024-04-01"],  # spring switch
            ["2024-10-26", "2024-10-27", "2024-10-28"],  # autumn switch
        ],
    )
    def test_consecutive_days_across_a_dst_switch(self, tz, local_days):
        midnights = [pd.Timestamp(d, tz=tz) for d in local_days]
        stamps = [t.tz_convert("UTC").strftime("%Y-%m-%d %H:%M") for t in midnights]
        values = [float(i) for i in range(1, len(midnights) + 1)]
        first = (midnights[0] - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        df = _a31_frame(first, 8, stamps, values)
        out = TimezoneTransformer(target_tz=tz).transform(df)  # default daily_anchor
        landed = out["load_forecast_daily_min"].dropna()
        assert landed.index.tolist() == midnights
        assert landed.tolist() == values

    @pytest.mark.parametrize("anchor", ["UTC+02:00", "+02:00", "UTC+2", "utc+0200", "UTC", "UTC-05:00", "Europe/Paris"])
    def test_accepted_anchor_formats(self, anchor):
        assert TimezoneTransformer(target_tz="Europe/Warsaw", daily_anchor=anchor).daily_anchor == anchor

    @pytest.mark.parametrize("anchor", ["Mars/Olympus", "UTC+25:00", "+2h", ""])
    def test_invalid_anchor_raises(self, anchor):
        with pytest.raises(ValueError, match="daily_anchor"):
            TimezoneTransformer(target_tz="Europe/Warsaw", daily_anchor=anchor)


class TestDailyColumnChecks:
    def test_dense_daily_column_raises(self):
        """review F7: running ResampleTransformer first made every hour snap to midnight (98 -> 5 values)."""
        stamps = ["2023-12-31 22:00"] + [f"2024-01-0{d} 22:00" for d in (1, 2, 3)]
        df = _a31_frame("2023-12-31", 5, stamps, [100.0, 200.0, 300.0, 400.0])
        dense = df.copy()
        dense["load_forecast_daily_min"] = dense["load_forecast_daily_min"].ffill()
        with pytest.raises(ValueError, match="more than one value"):
            TimezoneTransformer(target_tz="Europe/Warsaw").transform(dense)

    def test_resample_before_timezone_raises(self):
        stamps = [f"2024-01-0{d} 22:00" for d in (1, 2, 3)]
        df = _a31_frame("2024-01-01", 4, stamps, [1.0, 2.0, 3.0])
        dense = ResampleTransformer(freq="1h").transform(df)
        with pytest.raises(ValueError, match="TimezoneTransformer before ResampleTransformer"):
            TimezoneTransformer(target_tz="Europe/Warsaw").transform(dense)

    def test_sparse_column_keeps_every_value(self):
        stamps = [f"2024-01-0{d} 22:00" for d in (1, 2, 3, 4)]
        df = _a31_frame("2024-01-01", 6, stamps, [1.0, 2.0, 3.0, 4.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = TimezoneTransformer(target_tz="Europe/Warsaw", daily_anchor="UTC+02:00").transform(df)
        assert out["load_forecast_daily_min"].notna().sum() == 4
        assert len(out) == len(df)

    def test_missing_midnight_at_the_edge_is_added(self):
        """review F7: a value whose local midnight is not in the index used to stay at 23:00 of the day before."""
        idx = pd.date_range("2024-01-08", "2024-01-09 21:00", freq="1h", tz="UTC")
        df = pd.DataFrame({"load_actual": 1.0, "load_forecast_daily_min": np.nan}, index=idx)
        extra = pd.DataFrame(
            {"load_actual": np.nan, "load_forecast_daily_min": 111.0},
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-09 22:00", tz="UTC")]),
        )
        df = pd.concat([df, extra])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = TimezoneTransformer(target_tz="Europe/Warsaw", daily_anchor="UTC+02:00").transform(df)

        midnight = pd.Timestamp("2024-01-10 00:00", tz="Europe/Warsaw")
        assert out["load_forecast_daily_min"].dropna().index.tolist() == [midnight]
        assert pd.isna(out.loc[midnight, "load_actual"])
        assert out.index.is_unique and out.index.is_monotonic_increasing
        assert len(out) == len(df) + 1

    def test_first_winter_value_gets_its_own_midnight_row(self):
        """In winter the A31 value for the first local day is stamped 23:00 local, before the first hourly row."""
        df = _a31_frame("2024-01-01", 2, ["2023-12-31 22:00", "2024-01-01 22:00"], [1.0, 2.0])
        out = TimezoneTransformer(target_tz="Europe/Warsaw", daily_anchor="UTC+02:00").transform(df)
        landed = out["load_forecast_daily_min"].dropna()
        assert landed.index.tolist() == [pd.Timestamp(d, tz="Europe/Warsaw") for d in ["2024-01-01", "2024-01-02"]]
        assert len(out) == len(df) + 1  # the row the value came from stays, now empty

    def test_missing_midnight_inside_the_index_warns(self):
        df = _a31_frame("2024-01-08", 3, ["2024-01-08 22:00"], [7.0])
        df = df.drop(pd.Timestamp("2024-01-08 23:00", tz="UTC"))  # the local midnight of 2024-01-09

        with pytest.warns(UserWarning, match="local midnight"):
            out = TimezoneTransformer(target_tz="Europe/Warsaw", daily_anchor="UTC+02:00").transform(df)

        midnight = pd.Timestamp("2024-01-09 00:00", tz="Europe/Warsaw")
        assert out["load_forecast_daily_min"].dropna().index.tolist() == [midnight]
        assert len(out) == len(df) + 1

    def test_absent_daily_column_is_ignored(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        out = TimezoneTransformer(target_tz="Europe/Warsaw").transform(pd.DataFrame({"x": 1.0}, index=idx))
        assert list(out.columns) == ["x"]


def test_daily_anchor_round_trips_through_yaml(tmp_path):
    from epftoolbox2.pipelines import DataPipeline

    original = DataPipeline().add_transformer(
        TimezoneTransformer(target_tz="Europe/Lisbon", daily_columns=["dmin"], daily_anchor="UTC+01:00")
    )
    original.save(tmp_path / "dp.yaml")
    loaded = DataPipeline.load(tmp_path / "dp.yaml").transformers[0]
    assert (loaded.target_tz, loaded.daily_columns, loaded.daily_anchor) == ("Europe/Lisbon", ["dmin"], "UTC+01:00")
