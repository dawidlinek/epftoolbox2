import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.transformers import LagTransformer, ResampleTransformer, TimezoneTransformer, Transformer
from tests.synthetic import make_market_data

TZ = "Europe/Warsaw"


def local_dates(index: pd.DatetimeIndex) -> np.ndarray:
    return index.tz_convert(TZ).normalize().tz_localize(None).to_numpy()


class TestResampleTransformerInit:
    def test_defaults(self):
        transformer = ResampleTransformer()
        assert transformer.freq == "1h"
        assert transformer.agg == "mean"
        assert transformer.fill == "ffill"
        assert transformer.method is None
        assert transformer.columns is None

    def test_is_transformer(self):
        assert isinstance(ResampleTransformer(), Transformer)

    def test_per_column_agg_and_fill(self):
        transformer = ResampleTransformer(agg={"default": "first", "energy": "sum"}, fill={"price": None})
        assert transformer.agg == {"default": "first", "energy": "sum"}
        assert transformer.fill == {"price": None}

    def test_fill_none_string_is_normalised(self):
        assert ResampleTransformer(fill="none").fill is None

    @pytest.mark.parametrize("agg", ["linear", "cubic", {"price": "interpolate"}])
    def test_invalid_agg(self, agg):
        with pytest.raises(ValueError, match="agg"):
            ResampleTransformer(agg=agg)

    @pytest.mark.parametrize("fill", ["linear", "bfill", {"price": "interpolate"}])
    def test_invalid_fill(self, fill):
        with pytest.raises(ValueError, match="fill"):
            ResampleTransformer(fill=fill)

    def test_invalid_freq(self):
        with pytest.raises(ValueError, match="freq"):
            ResampleTransformer(freq="every tuesday")

    def test_method_and_agg_together_raise(self):
        with pytest.raises(ValueError, match="method"):
            ResampleTransformer(method="sum", agg="first")


class TestLegacyParameters:
    """`method` and `columns` still work, translated to per-column agg/fill with a FutureWarning."""

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Invalid method"):
            ResampleTransformer(method="invalid")

    @pytest.mark.parametrize(
        "method, agg", [("linear", "mean"), ("ffill", "first"), ("bfill", "last"), ("sum", "sum"), ("last", "last")]
    )
    def test_method_selects_aggregation(self, method, agg):
        idx = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")
        df = pd.DataFrame({"x": np.arange(8.0)}, index=idx)
        with pytest.warns(FutureWarning, match="method"):
            legacy = ResampleTransformer(freq="1h", method=method)
        expected = ResampleTransformer(freq="1h", agg=agg).transform(df)
        pd.testing.assert_frame_equal(legacy.transform(df), expected)

    def test_legacy_attributes_survive(self):
        with pytest.warns(FutureWarning):
            transformer = ResampleTransformer(freq="15min", method="ffill", columns="price")
        assert transformer.method == "ffill"
        assert transformer.columns == ["price"]

    def test_columns_limit_the_method_to_those_columns(self):
        """review F6: method='sum', columns=['energy_mwh'] used to sum the price too (100 -> 400)."""
        idx = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")
        df = pd.DataFrame({"energy_mwh": 10.0, "price": 100.0}, index=idx)
        with pytest.warns(FutureWarning):
            transformer = ResampleTransformer(freq="1h", method="sum", columns=["energy_mwh"])
        out = transformer.transform(df)
        assert out["energy_mwh"].tolist() == [40.0, 40.0]
        assert out["price"].tolist() == [100.0, 100.0]

    def test_unlisted_columns_are_not_filled(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0], "load": [100.0, 200.0, 300.0]}, index=idx)
        with pytest.warns(FutureWarning):
            out = ResampleTransformer(freq="12h", method="ffill", columns=["price"]).transform(df)
        assert out["price"].tolist() == [10.0, 10.0, 20.0, 20.0, 30.0, 30.0]
        assert out["load"].iloc[[0, 2, 4]].tolist() == [100.0, 200.0, 300.0]
        assert out["load"].iloc[[1, 3, 5]].isna().all()

    def test_two_step_example_pattern_equals_one_step(self):
        """Examples 01/06 resampled daily columns with method='ffill' first; that must not change other columns."""
        raw = _raw_source_frame()
        local = TimezoneTransformer(TZ).transform(raw)
        with pytest.warns(FutureWarning):
            first = ResampleTransformer(
                freq="1h", columns=["load_forecast_daily_min", "load_forecast_daily_max"], method="ffill"
            )
        two_step = ResampleTransformer(freq="1h").transform(first.transform(local))
        one_step = ResampleTransformer(freq="1h").transform(local)
        pd.testing.assert_frame_equal(two_step, one_step)

    def test_missing_legacy_column_raises(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
        with pytest.warns(FutureWarning):
            transformer = ResampleTransformer(columns=["non_existent"])
        with pytest.raises(ValueError, match="Columns not found"):
            transformer.transform(pd.DataFrame({"value": 1.0}, index=idx))


def _raw_source_frame(start="2024-01-01", days=12, gap=None) -> pd.DataFrame:
    """UTC frame shaped like DataPipeline source output: 15-min load, hourly price, A31 daily values at 22:00 UTC."""
    first = pd.Timestamp(start, tz="UTC")
    quarter = pd.date_range(first, periods=days * 96, freq="15min")
    hourly = pd.date_range(first, periods=days * 24, freq="1h")
    rng = np.random.default_rng(3)
    load = pd.DataFrame({"load_actual": 1000 + rng.normal(0, 50, len(quarter))}, index=quarter)
    price = pd.DataFrame({"price": 50 + rng.normal(0, 10, len(hourly))}, index=hourly)
    if gap is not None:
        price.loc[gap[0] : gap[1], "price"] = np.nan
    daily_idx = pd.date_range(first - pd.Timedelta(hours=2), periods=days + 1, freq="D")
    daily = pd.DataFrame(
        {
            "load_forecast_daily_min": 800.0 + 10 * np.arange(days + 1),
            "load_forecast_daily_max": 1500.0 + 10 * np.arange(days + 1),
        },
        index=daily_idx,
    )
    return pd.concat([load, price, daily], axis=1)


class TestNoLookAhead:
    """review F1 / empirical F3: filling must never use later values or invent values past the last one."""

    def test_three_day_price_gap_stays_nan(self):
        df = make_market_data(days=30)[["load", "wind", "is_weekend", "is_weekday", "price"]].tz_convert("UTC")
        gap = (df.index >= "2023-01-10") & (df.index < "2023-01-13")
        df.loc[gap, "price"] = np.nan

        out = TimezoneTransformer(TZ).transform(df)
        out = ResampleTransformer(freq="1h").transform(out)
        out = LagTransformer(columns=["price"], lags=[1], freq="day").transform(out)

        in_gap = (out.index >= pd.Timestamp("2023-01-10", tz="UTC")) & (
            out.index < pd.Timestamp("2023-01-13", tz="UTC")
        )
        assert in_gap.sum() == 72
        assert out.loc[in_gap, "price"].isna().all()
        lagged = out.index.isin(out.index[in_gap] + pd.Timedelta(days=1))
        assert out.loc[lagged, "price_d-1"].isna().all()
        assert out.loc[~in_gap, "price"].notna().all()

    @pytest.mark.parametrize("freq", ["1h", "15min"])
    @pytest.mark.parametrize("issue_day", ["2024-01-04", "2024-01-05", "2024-01-06", "2024-01-08"])
    def test_features_on_day_d_ignore_values_from_d_plus_1(self, freq, issue_day):
        """Perturb every raw value dated >= D+1 (local); nothing on rows dated <= D may change."""
        raw = _raw_source_frame(gap=("2024-01-04 06:00", "2024-01-07 12:00"))
        cutoff = pd.Timestamp(issue_day) + pd.Timedelta(days=1)
        perturbed = raw.copy()
        later = local_dates(raw.index) >= np.datetime64(cutoff)
        perturbed.loc[later] = perturbed.loc[later] + 1000.0

        def pipeline(frame):
            out = TimezoneTransformer(TZ).transform(frame)
            out = ResampleTransformer(freq=freq).transform(out)
            return LagTransformer(columns=["price", "load_actual"], lags=[1, 7], freq="day").transform(out)

        base, moved = pipeline(raw), pipeline(perturbed)
        rows = local_dates(base.index) < np.datetime64(cutoff)
        pd.testing.assert_frame_equal(base.loc[rows], moved.loc[rows])

    def test_trailing_values_are_not_extrapolated(self):
        """empirical F3: rows after the last price must stay NaN, not repeat the last price."""
        idx = pd.date_range("2024-01-01", periods=72, freq="1h", tz="UTC")
        df = pd.DataFrame({"price": np.arange(72.0), "load_forecast": 1.0}, index=idx)
        df.loc[idx[48:], "price"] = np.nan
        out = ResampleTransformer(freq="1h").transform(df)
        assert out["price"].iloc[48:].isna().all()
        assert out["price"].iloc[:48].tolist() == list(np.arange(48.0))

    def test_leading_and_interior_nans_stay(self):
        idx = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
        df = pd.DataFrame({"price": [np.nan, np.nan, 3.0, np.nan, 5.0, 6.0]}, index=idx)
        out = ResampleTransformer(freq="1h").transform(df)
        np.testing.assert_array_equal(out["price"].to_numpy(), df["price"].to_numpy())

    def test_values_are_not_rounded(self):
        """review F11: round(3) turned 0.00049 into 0."""
        idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
        df = pd.DataFrame({"capacity_factor": [0.12345, 0.00049, 0.5]}, index=idx)
        out = ResampleTransformer(freq="1h").transform(df)
        assert out["capacity_factor"].tolist() == [0.12345, 0.00049, 0.5]


class TestUpsampling:
    """review F2: hourly period values step-fill to 15 min; no ramps, and the last period is complete."""

    def test_hourly_to_quarter_hour_step_fill(self):
        df = make_market_data(days=3)[["price", "is_weekend", "load"]]
        out = ResampleTransformer(freq="15min").transform(df)

        hourly = out.index.floor("h")
        for col in df.columns:
            expected = df[col].reindex(hourly).to_numpy()
            np.testing.assert_array_equal(out[col].to_numpy(), expected)
        assert set(out["is_weekend"].unique()) <= {0.0, 1.0}

        last_day = out.index.normalize() == out.index[-1].normalize()
        assert last_day.sum() == 96
        assert out.index[-1] == df.index[-1] + pd.Timedelta(minutes=45)

    def test_monday_dummy_stays_binary_across_midnight(self):
        idx = pd.date_range("2024-01-07 22:00", periods=4, freq="1h", tz="Europe/Berlin")
        df = pd.DataFrame({"is_monday": (idx.dayofweek == 0).astype(float)}, index=idx)
        out = ResampleTransformer(freq="15min").transform(df)
        sunday_late = out.loc["2024-01-07 23:00":"2024-01-07 23:45", "is_monday"]
        assert sunday_late.tolist() == [0.0, 0.0, 0.0, 0.0]

    def test_upsampling_on_spring_dst_day(self):
        idx = pd.date_range("2024-03-30 23:00", "2024-03-31 21:00", freq="1h", tz="UTC").tz_convert(TZ)
        df = pd.DataFrame({"price": np.arange(len(idx), dtype=float)}, index=idx)
        out = ResampleTransformer(freq="15min").transform(df)
        assert len(out) == 23 * 4
        assert out["price"].notna().all()
        assert (out["price"].to_numpy() == np.repeat(df["price"].to_numpy(), 4)).all()

    def test_missing_hour_is_not_filled_by_neighbours(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
        df = pd.DataFrame({"price": [1.0, np.nan, 3.0, 4.0]}, index=idx)
        out = ResampleTransformer(freq="15min").transform(df)
        assert out["price"].iloc[4:8].isna().all()
        assert out["price"].iloc[:4].tolist() == [1.0] * 4
        assert out["price"].iloc[8:].tolist() == [3.0] * 4 + [4.0] * 4

    def test_fill_none_leaves_new_slots_empty(self):
        idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
        df = pd.DataFrame({"price": [1.0, 2.0], "load": [10.0, 20.0]}, index=idx)
        out = ResampleTransformer(freq="15min", fill={"price": None}).transform(df)
        assert out["load"].tolist() == [10.0] * 4 + [20.0] * 4
        assert out["price"].iloc[[0, 4]].tolist() == [1.0, 2.0]
        assert out["price"].drop(out.index[[0, 4]]).isna().all()

    def test_sum_columns_are_split_so_totals_are_kept(self):
        idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
        df = pd.DataFrame({"energy_mwh": [40.0, 80.0]}, index=idx)
        out = ResampleTransformer(freq="15min", agg="sum").transform(df)
        assert out["energy_mwh"].tolist() == [10.0] * 4 + [20.0] * 4
        back = ResampleTransformer(freq="1h", agg="sum").transform(out)
        assert back["energy_mwh"].tolist() == [40.0, 80.0]

    def test_resolution_switch_inside_one_column(self):
        """DE prices are hourly until the 15-min SDAC switch; both parts must end up on the 15-min grid."""
        hourly = pd.date_range("2025-09-30 20:00", periods=4, freq="1h", tz="UTC")
        quarter = pd.date_range("2025-10-01 00:00", periods=8, freq="15min", tz="UTC")
        price = pd.concat(
            [pd.Series([1.0, 2.0, 3.0, 4.0], index=hourly), pd.Series(np.arange(10.0, 18.0), index=quarter)]
        )
        out = ResampleTransformer(freq="15min").transform(price.to_frame("price"))
        assert out["price"].iloc[:16].tolist() == [1.0] * 4 + [2.0] * 4 + [3.0] * 4 + [4.0] * 4
        assert out["price"].iloc[16:].tolist() == list(np.arange(10.0, 18.0))

        hourly_out = ResampleTransformer(freq="1h").transform(price.to_frame("price"))
        assert hourly_out["price"].tolist() == [1.0, 2.0, 3.0, 4.0, 11.5, 15.5]


class TestDownsampling:
    def test_mixed_energy_and_price_quarter_hours_to_hours(self):
        idx = pd.date_range("2024-01-01", periods=16, freq="15min", tz="UTC")
        df = pd.DataFrame({"energy_mwh": 10.0, "price": 100.0 + np.arange(16.0)}, index=idx)
        df = df.drop(idx[4:8])  # hour 01:00 missing entirely
        df.loc[idx[9:11]] = np.nan  # hour 02:00 half missing

        out = ResampleTransformer(freq="1h", agg={"price": "mean", "energy_mwh": "sum"}).transform(df)

        assert len(out) == 4
        assert out["energy_mwh"].iloc[[0, 3]].tolist() == [40.0, 40.0]
        assert out["price"].iloc[[0, 3]].tolist() == [101.5, 113.5]
        assert out.iloc[1].isna().all()
        assert out.iloc[2].isna().all()

    def test_mean_of_quarter_hours_not_the_first(self):
        """empirical F2: the old two-step pattern kept only the :00 quarter-hour of every 15-min series."""
        idx = pd.date_range("2024-10-28 05:00", periods=8, freq="15min", tz="UTC")
        load = pd.Series([17829.0, 18472.0, 19047.0, 19743.0, 1.0, 2.0, 3.0, 4.0], index=idx)
        out = ResampleTransformer(freq="1h").transform(load.to_frame("load_actual"))
        assert out["load_actual"].iloc[0] == pytest.approx(18772.75)

    def test_hourly_column_in_quarter_hour_frame_keeps_its_value(self):
        quarter = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")
        hourly = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
        df = pd.concat(
            [pd.DataFrame({"load": np.arange(8.0)}, index=quarter), pd.DataFrame({"price": [5.0, 7.0]}, index=hourly)],
            axis=1,
        )
        for agg in ["mean", "sum", "first", "last", "min", "max", "median"]:
            out = ResampleTransformer(freq="1h", agg=agg).transform(df)
            assert out["price"].tolist() == [5.0, 7.0], agg

    def test_local_days_on_dst_switches(self):
        q = pd.date_range("2024-03-30 23:00", "2024-03-31 22:00", freq="15min", tz="UTC", inclusive="left")
        out = ResampleTransformer(freq="1h").transform(pd.DataFrame({"x": 1.0}, index=q.tz_convert(TZ)))
        assert out.groupby(out.index.date).size().tolist() == [23]
        assert out["x"].notna().all()
        q = pd.date_range("2024-10-26 22:00", "2024-10-27 23:00", freq="15min", tz="UTC", inclusive="left")
        out = ResampleTransformer(freq="1h").transform(pd.DataFrame({"x": 1.0}, index=q.tz_convert(TZ)))
        assert out.groupby(out.index.date).size().tolist() == [25]

    def test_daily_bins_follow_local_days_and_incomplete_days_are_nan(self):
        idx = pd.date_range("2024-03-29 00:00", "2024-04-02 00:00", freq="1h", tz="UTC", inclusive="left")
        df = pd.DataFrame({"x": 1.0}, index=idx.tz_convert(TZ))  # starts at 01:00 local, ends at 01:00 local
        out = ResampleTransformer(freq="1D").transform(df)
        assert [d.date().isoformat() for d in out.index] == [
            "2024-03-29",
            "2024-03-30",
            "2024-03-31",
            "2024-04-01",
            "2024-04-02",
        ]
        assert out["x"].isna().tolist() == [True, False, False, False, True]

    def test_string_column_does_not_crash_mean(self):
        """review F11: a string column (e.g. weekday_name) made the default aggregation raise."""
        idx = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")
        df = pd.DataFrame({"price": np.arange(8.0), "weekday_name": "Monday"}, index=idx)
        out = ResampleTransformer(freq="1h").transform(df)
        assert out["weekday_name"].tolist() == ["Monday", "Monday"]
        assert out["price"].tolist() == [1.5, 5.5]

    def test_numeric_aggregation_requested_for_string_column_raises(self):
        idx = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")
        df = pd.DataFrame({"weekday_name": "Monday"}, index=idx)
        with pytest.raises(ValueError, match="weekday_name"):
            ResampleTransformer(freq="1h", agg={"weekday_name": "mean"}).transform(df)


class TestDailyColumns:
    """review F8: daily values snapped to local midnight cover exactly their own local day."""

    def test_daily_value_covers_its_local_day_only(self):
        h_idx = pd.date_range("2024-01-01", "2024-01-05", freq="1h", tz="UTC", inclusive="left")
        df = pd.DataFrame({"load_actual": 1.0}, index=h_idx)
        df["load_forecast_daily_min"] = np.nan
        for i, d in enumerate(pd.date_range("2023-12-31 22:00", periods=5, freq="1D", tz="UTC")):
            df.loc[d, "load_forecast_daily_min"] = 100.0 * (i + 1)
        df = df.sort_index()

        out = ResampleTransformer(freq="1h").transform(TimezoneTransformer(TZ, daily_anchor="UTC+02:00").transform(df))
        day2 = out.loc["2024-01-02", "load_forecast_daily_min"]
        assert len(day2) == 24
        assert (day2 == 200.0).all()

    @pytest.mark.parametrize("freq", ["1h", "15min"])
    @pytest.mark.parametrize(
        "first_utc, last_utc, days, hours",
        [
            ("2024-03-29 23:00", "2024-04-01 21:00", ["2024-03-30", "2024-03-31", "2024-04-01"], [24, 23, 24]),
            ("2024-10-25 22:00", "2024-10-28 22:00", ["2024-10-26", "2024-10-27", "2024-10-28"], [24, 25, 24]),
        ],
    )
    def test_daily_value_fills_23_and_25_hour_days(self, freq, first_utc, last_utc, days, hours):
        index = pd.date_range(first_utc, last_utc, freq="1h", tz="UTC").tz_convert(TZ)
        df = pd.DataFrame({"load_actual": 1.0}, index=index)
        midnights = pd.DatetimeIndex([pd.Timestamp(d, tz=TZ) for d in days])
        df["daily"] = pd.Series([1.0, 2.0, 3.0], index=midnights)
        out = ResampleTransformer(freq=freq).transform(df)
        per_day = out["daily"].groupby(out.index.date).agg(["min", "max", "count"])
        assert per_day["min"].tolist() == [1.0, 2.0, 3.0]
        assert per_day["max"].tolist() == [1.0, 2.0, 3.0]
        per_hour = 4 if freq == "15min" else 1
        assert per_day["count"].tolist() == [h * per_hour for h in hours]

    def test_missing_day_is_not_filled_from_the_day_before(self):
        hours = pd.date_range("2024-01-01", periods=6 * 24, freq="1h", tz=TZ)
        df = pd.DataFrame({"load_actual": 1.0}, index=hours)
        df["daily"] = np.nan
        for day, value in [("2024-01-01", 1.0), ("2024-01-02", 2.0), ("2024-01-04", 4.0), ("2024-01-05", 5.0)]:
            df.loc[pd.Timestamp(day, tz=TZ), "daily"] = value
        out = ResampleTransformer(freq="1h").transform(df)
        assert out.loc["2024-01-03", "daily"].isna().all()
        assert out.loc["2024-01-06", "daily"].isna().all()
        assert (out.loc["2024-01-02", "daily"] == 2.0).all()
        assert (out.loc["2024-01-04", "daily"] == 4.0).all()


class TestFrameHandling:
    def test_invalid_index_type(self):
        with pytest.raises(ValueError, match="DatetimeIndex"):
            ResampleTransformer().transform(pd.DataFrame({"value": [1, 2, 3]}, index=[0, 1, 2]))

    def test_duplicate_timestamps_raise(self):
        idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 01:00"], tz="UTC")
        with pytest.raises(ValueError, match="duplicate"):
            ResampleTransformer().transform(pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=idx))

    def test_unknown_agg_column_raises(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
        with pytest.raises(ValueError, match="Columns not found"):
            ResampleTransformer(agg={"nope": "sum"}).transform(pd.DataFrame({"value": 1.0}, index=idx))

    def test_empty_dataframe(self):
        df = pd.DataFrame({"value": []}, index=pd.DatetimeIndex([], dtype="datetime64[ns]"))
        assert len(ResampleTransformer().transform(df)) == 0

    def test_single_row(self):
        df = pd.DataFrame({"value": [1.0]}, index=pd.date_range("2024-01-01", periods=1, freq="h"))
        out = ResampleTransformer(freq="30min").transform(df)
        assert out["value"].iloc[0] == 1.0

    def test_preserves_timezone_columns_and_input(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="h", tz=TZ)
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0], "load": [100.0, 200.0, 300.0]}, index=idx)
        original = df.copy()
        out = ResampleTransformer(freq="30min").transform(df)
        assert str(out.index.tz) == TZ
        assert list(out.columns) == ["price", "load"]
        assert out.index.freq == pd.Timedelta("30min")
        pd.testing.assert_frame_equal(df, original)

    def test_naive_index(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        out = ResampleTransformer(freq="2h").transform(pd.DataFrame({"value": np.arange(5.0)}, index=idx))
        assert out["value"].iloc[:2].tolist() == [0.5, 2.5]
        assert np.isnan(out["value"].iloc[2])  # 04:00-06:00 holds one of two hours

    def test_unsorted_input_is_sorted(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
        df = pd.DataFrame({"value": np.arange(4.0)}, index=idx)
        out = ResampleTransformer(freq="1h").transform(df.iloc[::-1])
        assert out["value"].tolist() == [0.0, 1.0, 2.0, 3.0]
