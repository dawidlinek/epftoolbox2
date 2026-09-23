"""Daily-feature transformers: profiles, daily stats, residual load, spike flags."""

import warnings

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.transformers import (
    DailyProfileTransformer,
    DailyStatsTransformer,
    DSTNormalizeTransformer,
    ResidualLoadTransformer,
    SpikeFlagTransformer,
)
from epftoolbox2.pipelines import DataPipeline

TZ = "Europe/Warsaw"


def local_frame(start: str, end: str, freq: str = "1h", seed: int = 0) -> pd.DataFrame:
    index = pd.date_range(start, end, freq=freq, tz=TZ, inclusive="left")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"price": rng.normal(50, 10, len(index)), "load": rng.normal(1000, 50, len(index))}, index=index
    )


def local(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts, tz=TZ)


def assert_rows_up_to_day_unchanged(transformer, df: pd.DataFrame, issue_day: str, columns=None):
    """Changing every value dated after the issue day must not change any feature on rows up to that day."""
    day = pd.Timestamp(issue_day).date()
    perturbed = df.copy()
    later = df.index.date > day
    perturbed.loc[later, perturbed.select_dtypes("number").columns] *= 3.0
    perturbed.loc[later, perturbed.select_dtypes("number").columns] += 1000.0
    a, b = transformer.transform(df), transformer.transform(perturbed)
    new = [c for c in a.columns if c not in df.columns] if columns is None else columns
    upto = a.index.date <= day
    pd.testing.assert_frame_equal(a.loc[upto, new], b.loc[upto, new])


# --------------------------------------------------------------------------- DailyProfileTransformer


def test_profile_copies_the_days_values_onto_every_row():
    df = local_frame("2024-01-01", "2024-01-04")
    out = DailyProfileTransformer(columns="price").transform(df)

    names = [f"price_h{h:02d}" for h in range(24)]
    assert list(out.columns) == ["price", "load", *names]
    for day in ["2024-01-01", "2024-01-02", "2024-01-03"]:
        rows = out.loc[day]
        expected = df.loc[day, "price"].to_numpy()
        assert (rows[names].to_numpy() == expected).all()


def test_profile_of_quarter_hours_uses_slot_names():
    df = local_frame("2024-01-01", "2024-01-03", freq="15min")
    out = DailyProfileTransformer(columns="price", freq="15min").transform(df)
    assert "price_s00" in out.columns and "price_s95" in out.columns
    assert out.loc[local("2024-01-02 17:30"), "price_s05"] == df.loc[local("2024-01-02 01:15"), "price"]


def test_profile_on_dst_days_warns_and_leaves_skipped_slot_empty():
    df = local_frame("2024-03-30", "2024-04-02")
    with pytest.warns(UserWarning, match="DSTNormalizeTransformer"):
        out = DailyProfileTransformer(columns="price").transform(df)
    assert out.loc["2024-03-31", "price_h02"].isna().all()
    assert out.loc[local("2024-03-31 12:00"), "price_h03"] == df.loc[local("2024-03-31 03:00"), "price"]

    autumn = local_frame("2024-10-26", "2024-10-29")
    with pytest.warns(UserWarning):
        out = DailyProfileTransformer(columns="price").transform(autumn)
    first = autumn.loc[pd.Timestamp("2024-10-27 02:00+02:00"), "price"]
    assert (out.loc["2024-10-27", "price_h02"] == first).all()


def test_profile_after_dst_normalisation_is_complete_without_warning():
    df = DSTNormalizeTransformer().transform(local_frame("2024-03-30", "2024-04-02"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = DailyProfileTransformer(columns="price").transform(df)
    assert out.loc["2024-03-31", [f"price_h{h:02d}" for h in range(24)]].notna().all().all()


def test_profile_is_leak_free():
    df = local_frame("2024-10-20", "2024-11-03")
    for day in ["2024-10-26", "2024-10-27", "2024-10-28"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            assert_rows_up_to_day_unchanged(DailyProfileTransformer(columns=["price", "load"]), df, day)


def test_profile_rejects_off_grid_data_and_name_clashes():
    with pytest.raises(ValueError, match="grid"):
        DailyProfileTransformer(columns="price").transform(local_frame("2024-01-01", "2024-01-02", freq="15min"))
    df = DailyProfileTransformer(columns="price").transform(local_frame("2024-01-01", "2024-01-03"))
    with pytest.raises(ValueError, match="price_h00"):
        DailyProfileTransformer(columns="price").transform(df)


# --------------------------------------------------------------------------- DailyStatsTransformer


def test_daily_stats_per_local_day():
    df = local_frame("2024-01-01", "2024-01-04")
    out = DailyStatsTransformer(columns="price").transform(df)

    day = df.loc["2024-01-02", "price"]
    rows = out.loc["2024-01-02"]
    assert (rows["price_day_min"] == day.min()).all()
    assert (rows["price_day_max"] == day.max()).all()
    assert np.allclose(rows["price_day_mean"], day.mean())
    assert (rows["price_day_last"] == day.iloc[-1]).all()


def test_daily_stats_only_on_complete_days():
    # UTC bounds: the first local day starts at 01:00, so it is incomplete; so is a day with a missing value.
    index = pd.date_range("2024-01-01", "2024-01-05", freq="1h", tz="UTC", inclusive="left").tz_convert(TZ)
    df = pd.DataFrame({"price": np.arange(len(index), dtype=float)}, index=index)
    df.loc[local("2024-01-03 05:00"), "price"] = np.nan

    out = DailyStatsTransformer(columns="price", stats=["max"]).transform(df)

    assert out.loc["2024-01-01", "price_day_max"].isna().all()
    assert out.loc["2024-01-03", "price_day_max"].isna().all()
    assert out.loc["2024-01-02", "price_day_max"].notna().all()


def test_daily_stats_count_dst_days_as_complete():
    for start, end, day in [("2024-03-30", "2024-04-02", "2024-03-31"), ("2024-10-26", "2024-10-29", "2024-10-27")]:
        df = local_frame(start, end)
        out = DailyStatsTransformer(columns="price", stats=["mean", "last"]).transform(df)
        assert np.allclose(out.loc[day, "price_day_mean"], df.loc[day, "price"].mean())
        assert (out.loc[day, "price_day_last"] == df.loc[local(f"{day} 23:00"), "price"]).all()


def test_daily_stats_are_leak_free():
    df = local_frame("2024-10-20", "2024-11-03", freq="15min")
    for day in ["2024-10-26", "2024-10-27"]:
        assert_rows_up_to_day_unchanged(DailyStatsTransformer(columns=["price", "load"], freq="15min"), df, day)


def test_daily_stats_reject_unknown_stat():
    with pytest.raises(ValueError, match="median"):
        DailyStatsTransformer(columns="price", stats=["median"])


# --------------------------------------------------------------------------- ResidualLoadTransformer


def forecast_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=48, freq="1h", tz=TZ)
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "load_forecast": rng.normal(20000, 1000, 48),
            "generation_solar_forecast": rng.uniform(0, 3000, 48),
            "generation_wind_onshore_forecast": rng.uniform(0, 5000, 48),
            "load_actual": rng.normal(20000, 1000, 48),
        },
        index=index,
    )


def test_residual_load_subtracts_wind_and_solar_forecasts():
    df = forecast_frame()
    out = ResidualLoadTransformer().transform(df)
    expected = df["load_forecast"] - df["generation_solar_forecast"] - df["generation_wind_onshore_forecast"]
    pd.testing.assert_series_equal(out["residual_load_forecast"], expected, check_names=False)


def test_residual_load_missing_input_gives_missing_output():
    df = forecast_frame()
    df.iloc[3, df.columns.get_loc("generation_solar_forecast")] = np.nan
    out = ResidualLoadTransformer().transform(df)
    assert np.isnan(out["residual_load_forecast"].iloc[3])


def test_residual_load_uses_forecast_columns_only():
    with pytest.raises(ValueError, match="forecast"):
        ResidualLoadTransformer(load="load_actual")
    with pytest.raises(ValueError, match="forecast"):
        ResidualLoadTransformer(renewables=["generation_solar"])


def test_residual_load_needs_its_columns():
    with pytest.raises(ValueError, match="generation_wind_offshore_forecast"):
        ResidualLoadTransformer(renewables=["generation_wind_offshore_forecast"]).transform(forecast_frame())
    with pytest.raises(ValueError, match="wind or solar"):
        ResidualLoadTransformer().transform(
            forecast_frame().drop(columns=["generation_solar_forecast", "generation_wind_onshore_forecast"])
        )


# --------------------------------------------------------------------------- SpikeFlagTransformer


def flat_prices(days: int = 60, freq: str = "1h") -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=days * (24 if freq == "1h" else 96), freq=freq, tz=TZ)
    rng = np.random.default_rng(3)
    profile = 50 + 20 * np.sin(2 * np.pi * index.hour / 24)
    return pd.DataFrame({"price": profile + rng.normal(0, 1, len(index))}, index=index)


def test_spike_flags_compare_with_the_same_slot_on_previous_days():
    df = flat_prices()
    df.loc[local("2024-02-10 19:00"), "price"] += 50.0
    df.loc[local("2024-02-11 04:00"), "price"] -= 50.0

    out = SpikeFlagTransformer(columns="price", window=28, k=5.0).transform(df)

    assert out.loc[local("2024-02-10 19:00"), "price_spike"] == 1
    assert out.loc[local("2024-02-11 04:00"), "price_spike"] == 1
    assert out.loc["2024-02-01":"2024-02-28", "price_spike"].sum() == 2
    # Not enough history in the first days of the data.
    assert out.loc["2024-01-01", "price_spike"].isna().all()


def test_spike_flag_sides():
    df = flat_prices()
    df.loc[local("2024-02-10 19:00"), "price"] += 50.0
    df.loc[local("2024-02-11 04:00"), "price"] -= 50.0
    upper = SpikeFlagTransformer(columns="price", k=5.0, side="upper").transform(df)
    lower = SpikeFlagTransformer(columns="price", k=5.0, side="lower").transform(df)
    assert upper.loc[local("2024-02-11 04:00"), "price_spike"] == 0
    assert lower.loc[local("2024-02-11 04:00"), "price_spike"] == 1
    assert lower.loc[local("2024-02-10 19:00"), "price_spike"] == 0


def test_spike_flags_use_only_earlier_days():
    df = flat_prices(freq="15min")
    for day in ["2024-02-05", "2024-02-06"]:
        assert_rows_up_to_day_unchanged(SpikeFlagTransformer(columns="price"), df, day)
    # A value on D changes only its own flag, not the flags of other rows on D.
    perturbed = df.copy()
    perturbed.loc[local("2024-02-05 10:00"), "price"] += 500.0
    a = SpikeFlagTransformer(columns="price").transform(df)
    b = SpikeFlagTransformer(columns="price").transform(perturbed)
    changed = a.index[
        (a["price_spike"] != b["price_spike"])
        & a.index.to_series().dt.date.eq(pd.Timestamp("2024-02-05").date()).to_numpy()
    ]
    assert list(changed) == [local("2024-02-05 10:00")]


def test_spike_flags_handle_dst_days():
    index = pd.date_range("2024-09-01", "2024-11-15", freq="1h", tz=TZ, inclusive="left")
    df = pd.DataFrame({"price": 50 + np.random.default_rng(0).normal(0, 1, len(index))}, index=index)
    out = SpikeFlagTransformer(columns="price").transform(df)
    assert out.loc["2024-10-27", "price_spike"].notna().all()
    assert len(out) == len(df)


def test_spike_flag_rejects_bad_parameters():
    with pytest.raises(ValueError, match="side"):
        SpikeFlagTransformer(columns="price", side="up")
    with pytest.raises(ValueError, match="window"):
        SpikeFlagTransformer(columns="price", window=0)
    with pytest.raises(ValueError, match="min_periods"):
        SpikeFlagTransformer(columns="price", window=10, min_periods=11)


# --------------------------------------------------------------------------- UTC warnings and YAML


@pytest.mark.parametrize(
    "transformer",
    [
        DailyProfileTransformer(columns="price"),
        DailyStatsTransformer(columns="price"),
        SpikeFlagTransformer(columns="price"),
    ],
)
def test_local_day_features_warn_on_utc_index(transformer):
    index = pd.date_range("2024-01-01", periods=24 * 40, freq="1h", tz="UTC")
    df = pd.DataFrame({"price": np.arange(len(index), dtype=float)}, index=index)
    with pytest.warns(UserWarning, match="UTC"):
        transformer.transform(df)


@pytest.mark.parametrize(
    "transformer",
    [
        DailyProfileTransformer(columns=["price", "load"], freq="15min"),
        DailyStatsTransformer(columns="price", stats=["min", "last"], freq="15min"),
        ResidualLoadTransformer(renewables=["generation_solar_forecast"], output="rl"),
        SpikeFlagTransformer(columns="price", window=14, k=4.0, min_periods=10, side="upper"),
    ],
)
def test_yaml_round_trip(tmp_path, transformer):
    path = tmp_path / "dp.yaml"
    DataPipeline().add_transformer(transformer).save(path)
    loaded = DataPipeline.load(path).transformers[0]
    assert type(loaded) is type(transformer)
    assert vars(loaded) == vars(transformer)
