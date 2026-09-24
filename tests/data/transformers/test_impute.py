import logging

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.transformers import (
    ImputeTransformer,
    LagTransformer,
    ResampleTransformer,
    TimezoneTransformer,
    Transformer,
)
from epftoolbox2.pipelines import DataPipeline
from tests.synthetic import make_market_data

TZ = "Europe/Warsaw"
NAN = np.nan


def hourly(values, start="2024-01-01", tz="UTC"):
    idx = pd.date_range(start, periods=len(values), freq="1h", tz=tz)
    return pd.DataFrame({"price": np.asarray(values, dtype=float)}, index=idx)


def assert_same(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float))


class TestInit:
    def test_defaults(self):
        t = ImputeTransformer(columns="price")
        assert (t.columns, t.method, t.limit, t.flag, t.suffix) == (["price"], "ffill", "1D", True, "")
        assert isinstance(t, Transformer)

    @pytest.mark.parametrize("columns", [None, [], ""])
    def test_columns_required(self, columns):
        with pytest.raises(ValueError, match="column"):
            ImputeTransformer(columns=columns)

    @pytest.mark.parametrize("method", ["linear", "bfill", "interpolate", "nearest"])
    def test_only_leak_safe_methods(self, method):
        with pytest.raises(ValueError, match="method"):
            ImputeTransformer(columns="price", method=method)

    @pytest.mark.parametrize("limit", [0, -3, True, "abc", "-1h", "0h", 1.5])
    def test_invalid_limit(self, limit):
        with pytest.raises(ValueError, match="limit"):
            ImputeTransformer(columns="price", limit=limit)

    @pytest.mark.parametrize("limit", [None, 3, "6h", "2D", np.int64(4)])
    def test_valid_limit(self, limit):
        assert ImputeTransformer(columns="price", limit=limit).limit == limit


class TestForwardFill:
    def test_limit_in_slots(self):
        out = ImputeTransformer(columns="price", limit=2).transform(hourly([1, NAN, NAN, NAN, 5]))
        assert_same(out["price"], [1, 1, 1, NAN, 5])
        assert out["price_imputed"].tolist() == [0, 1, 1, 0, 0]

    def test_limit_as_duration(self):
        out = ImputeTransformer(columns="price", limit="2h").transform(hourly([1, NAN, NAN, NAN, 5]))
        assert_same(out["price"], [1, 1, 1, NAN, 5])

    def test_no_limit(self):
        out = ImputeTransformer(columns="price", limit=None).transform(hourly([1, NAN, NAN, NAN, 5]))
        assert_same(out["price"], [1, 1, 1, 1, 5])

    def test_nothing_before_the_first_or_after_the_last_value(self):
        """Filling the tail would invent actuals for rows that are still in the future."""
        out = ImputeTransformer(columns="price", limit=None).transform(hourly([NAN, 1, NAN, 3, NAN, NAN]))
        assert_same(out["price"], [NAN, 1, 1, 3, NAN, NAN])
        assert out["price_imputed"].tolist() == [0, 0, 1, 0, 0, 0]

    def test_other_columns_untouched(self):
        df = hourly([1, NAN, 3])
        df["load"] = [NAN, 2.0, NAN]
        out = ImputeTransformer(columns="price").transform(df)
        assert_same(out["load"], [NAN, 2, NAN])
        assert "load_imputed" not in out.columns

    def test_input_not_modified(self):
        df = hourly([1, NAN, 3])
        ImputeTransformer(columns="price").transform(df)
        assert np.isnan(df["price"].iloc[1])
        assert list(df.columns) == ["price"]


class TestSeasonalNaive:
    def test_previous_day_same_local_hour(self):
        df = make_market_data(start="2024-01-01", days=5, tz=TZ)[["price"]]
        gap = df.index.normalize() == pd.Timestamp("2024-01-03", tz=TZ)
        truth = df["price"].copy()
        df.loc[gap, "price"] = NAN

        out = ImputeTransformer(columns="price", method="previous_day").transform(df)

        day_before = truth[df.index.normalize() == pd.Timestamp("2024-01-02", tz=TZ)].to_numpy()
        assert_same(out.loc[gap, "price"], day_before)
        assert out.loc[gap, "price_imputed"].eq(1).all()
        assert out.loc[~gap, "price_imputed"].eq(0).all()

    def test_previous_week(self):
        df = make_market_data(start="2024-01-01", days=15, tz=TZ)[["price"]]
        gap = df.index.normalize() == pd.Timestamp("2024-01-10", tz=TZ)
        truth = df["price"].copy()
        df.loc[gap, "price"] = NAN
        out = ImputeTransformer(columns="price", method="previous_week").transform(df)
        week_before = truth[df.index.normalize() == pd.Timestamp("2024-01-03", tz=TZ)].to_numpy()
        assert_same(out.loc[gap, "price"], week_before)

    def test_multi_day_gap_uses_the_last_observed_day(self):
        values = np.arange(24 * 6, dtype=float)
        df = hourly(values, tz=TZ)
        days = df.index.normalize()
        gap = (days >= pd.Timestamp("2024-01-03", tz=TZ)) & (days <= pd.Timestamp("2024-01-05", tz=TZ))
        df.loc[gap, "price"] = NAN
        out = ImputeTransformer(columns="price", method="previous_day", limit=None).transform(df)
        jan2 = values[24:48]
        assert_same(out.loc[gap, "price"], np.tile(jan2, 3))

    def test_limit_applies_to_seasonal_fill(self):
        df = hourly(np.arange(24 * 4, dtype=float), tz=TZ)
        df.iloc[30:40, 0] = NAN
        out = ImputeTransformer(columns="price", method="previous_day", limit="3h").transform(df)
        assert_same(out["price"].iloc[30:33], np.arange(6.0, 9.0))
        assert out["price"].iloc[33:40].isna().all()

    @pytest.mark.parametrize(
        "switch_day, day_after",
        [("2024-03-31", "2024-04-01"), ("2024-10-27", "2024-10-28")],
    )
    def test_dst_days_use_local_calendar_days(self, switch_day, day_after):
        """Same local hour yesterday, not 24 h earlier: DST days have 23 or 25 hours."""
        idx = pd.date_range(
            "2024-03-28" if switch_day < "2024-06" else "2024-10-24", periods=24 * 6, freq="1h", tz="UTC"
        )
        local = idx.tz_convert(TZ)
        # value = local wall-clock hour + 100 * local day of month, so the right source is easy to check
        df = pd.DataFrame({"price": (local.hour + 100 * local.day).astype(float)}, index=local)
        for day in (switch_day, day_after):
            df.loc[df.index.normalize() == pd.Timestamp(day, tz=TZ), "price"] = NAN

        out = ImputeTransformer(columns="price", method="previous_day", limit=None).transform(df)

        for day in (switch_day, day_after):
            rows = out.index.normalize() == pd.Timestamp(day, tz=TZ)
            filled = out.loc[rows, "price"]
            assert filled.notna().all(), day
            source_day = (pd.Timestamp(switch_day) - pd.Timedelta(days=1)).day
            # every filled value is the same local hour from the last observed day before the gap
            assert (filled.to_numpy() % 100 == out.index[rows].hour).all(), day
            assert (filled.to_numpy() // 100 == source_day).all(), day

    def test_never_uses_later_values(self):
        """Perturb every value dated >= D+1; imputed values on rows dated <= D must not change."""
        df = make_market_data(start="2024-03-20", days=20, tz=TZ)[["price"]]
        rng = np.random.default_rng(0)
        df.loc[rng.random(len(df)) < 0.3, "price"] = NAN
        df.loc["2024-03-30":"2024-04-01", "price"] = NAN
        for method in ["ffill", "previous_day", "previous_week"]:
            transformer = ImputeTransformer(columns="price", method=method, limit=None)
            base = transformer.transform(df)
            for day in ["2024-03-28", "2024-03-31", "2024-04-02"]:
                cutoff = pd.Timestamp(day, tz=TZ) + pd.Timedelta(days=1)
                moved = df.copy()
                moved.loc[moved.index >= cutoff, "price"] += 1000.0
                out = transformer.transform(moved)
                rows = base.index < cutoff
                pd.testing.assert_frame_equal(base.loc[rows], out.loc[rows])


class TestFlagsAndSuffix:
    def test_suffix_keeps_the_original_column(self):
        out = ImputeTransformer(columns="price", suffix="_filled").transform(hourly([1, NAN, 3]))
        assert_same(out["price"], [1, NAN, 3])
        assert_same(out["price_filled"], [1, 1, 3])
        assert out["price_filled_imputed"].tolist() == [0, 1, 0]
        assert "price_imputed" not in out.columns

    def test_flag_can_be_turned_off(self):
        out = ImputeTransformer(columns="price", flag=False).transform(hourly([1, NAN, 3]))
        assert list(out.columns) == ["price"]

    def test_second_imputer_adds_to_existing_flags(self):
        df = hourly([1, NAN, NAN, NAN, NAN, 6])
        once = ImputeTransformer(columns="price", limit=1).transform(df)
        twice = ImputeTransformer(columns="price", limit=None).transform(once)
        assert once["price_imputed"].tolist() == [0, 1, 0, 0, 0, 0]
        assert twice["price_imputed"].tolist() == [0, 1, 1, 1, 1, 0]

    def test_logs_counts_per_column(self):
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        logger = logging.getLogger("epftoolbox2.data.transformers.impute")
        logger.addHandler(handler)
        try:
            df = hourly([1, NAN, NAN, 4, NAN])
            df["load"] = [NAN, 1.0, NAN, 3.0, 4.0]
            ImputeTransformer(columns=["price", "load"], limit=1).transform(df)
        finally:
            logger.removeHandler(handler)
        messages = [r.getMessage() for r in records]
        assert any("price" in m and "1 value(s) imputed" in m and "2 still missing" in m for m in messages)
        assert any("load" in m and "1 value(s) imputed" in m and "1 still missing" in m for m in messages)


class TestFrameChecks:
    def test_missing_column(self):
        with pytest.raises(ValueError, match="Columns not found"):
            ImputeTransformer(columns="nope").transform(hourly([1, 2]))

    def test_index_type(self):
        with pytest.raises(ValueError, match="DatetimeIndex"):
            ImputeTransformer(columns="price").transform(pd.DataFrame({"price": [1.0]}, index=[0]))

    def test_unsorted_index(self):
        with pytest.raises(ValueError, match="sorted"):
            ImputeTransformer(columns="price").transform(hourly([1, NAN, 3]).iloc[::-1])

    def test_duplicate_index(self):
        df = pd.concat([hourly([1, NAN]), hourly([3])])
        with pytest.raises(ValueError, match="duplicate"):
            ImputeTransformer(columns="price").transform(df)

    def test_all_missing_column(self):
        out = ImputeTransformer(columns="price").transform(hourly([NAN, NAN]))
        assert out["price"].isna().all()
        assert out["price_imputed"].tolist() == [0, 0]


def test_pipeline_order_with_lagged_copy():
    """Timezone -> Resample -> Impute (into a copy) -> Lag: features are complete, the target keeps its gap."""
    raw = make_market_data(start="2024-01-01", days=20, tz=TZ)[["price", "load"]].tz_convert("UTC")
    raw.loc["2024-01-10 05:00":"2024-01-10 08:00", "price"] = NAN

    df = TimezoneTransformer(TZ).transform(raw)
    df = ResampleTransformer(freq="1h").transform(df)
    df = ImputeTransformer(columns="price", method="previous_day", suffix="_filled").transform(df)
    df = LagTransformer(columns=["price_filled", "price_filled_imputed"], lags=[1], freq="day").transform(df)

    assert df.loc["2024-01-10 06:00":"2024-01-10 09:00", "price"].isna().all()
    assert df["price_filled"].loc["2024-01-10"].notna().all()
    assert df["price_filled_d-1"].loc["2024-01-11"].notna().all()
    assert df["price_filled_imputed_d-1"].loc["2024-01-11"].sum() == 4


def test_yaml_round_trip(tmp_path):
    original = DataPipeline().add_transformer(
        ImputeTransformer(columns=["price"], method="previous_week", limit="2D", flag=False, suffix="_f")
    )
    original.save(tmp_path / "dp.yaml")
    loaded = DataPipeline.load(tmp_path / "dp.yaml").transformers[0]
    assert isinstance(loaded, ImputeTransformer)
    assert (loaded.columns, loaded.method, loaded.limit, loaded.flag, loaded.suffix) == (
        ["price"],
        "previous_week",
        "2D",
        False,
        "_f",
    )
