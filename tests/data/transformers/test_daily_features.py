"""Daily-feature transformers: profiles and daily stats."""

import warnings

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.transformers import DailyProfileTransformer, DailyStatsTransformer, DSTNormalizeTransformer

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
