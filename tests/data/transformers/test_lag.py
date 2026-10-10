"""LagTransformer: calendar-day lags on local time, naming, and information-set checks."""

import warnings

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.sources import CalendarSource
from epftoolbox2.data.transformers import LagTransformer, TimezoneTransformer
from tests.synthetic import calendar_shift

TZ = "Europe/Warsaw"


def local(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts, tz=TZ)


def local_frame(start: str, end: str, freq: str = "1h", seed: int = 0) -> pd.DataFrame:
    index = pd.date_range(start, end, freq=freq, tz=TZ, inclusive="left")
    return pd.DataFrame({"price": np.random.default_rng(seed).normal(50, 10, len(index))}, index=index)


# --------------------------------------------------------------------------- calendar-day lags across DST


def test_day_lag_keeps_wall_clock_hour_across_dst():
    # Day lags shift by local calendar days, not by elapsed time.
    index = pd.date_range("2024-03-29 22:00", "2024-04-01 00:00", freq="1h", tz="UTC").tz_convert(TZ)
    df = pd.DataFrame({"price": np.arange(len(index), dtype=float)}, index=index)

    out = LagTransformer(columns=["price"], lags=[1], freq="day").transform(df)

    on_switch_day = local("2024-03-31 10:00")
    day_before = local("2024-03-30 10:00")
    assert out.loc[on_switch_day, "price_d-1"] == df.loc[day_before, "price"]


def test_day_lag_on_the_repeated_autumn_hour():
    df = local_frame("2024-10-25", "2024-10-30")
    out = LagTransformer(columns="price", lags=[1, -1], freq="day").transform(df)
    first = pd.Timestamp("2024-10-27 02:00+02:00")
    second = pd.Timestamp("2024-10-27 02:00+01:00")

    # Both occurrences on the switch day look back to the single 02:00 of the day before.
    assert out.loc[first, "price_d-1"] == df.loc[local("2024-10-26 02:00"), "price"]
    assert out.loc[second, "price_d-1"] == df.loc[local("2024-10-26 02:00"), "price"]
    # The day after looks back to the first occurrence.
    assert out.loc[local("2024-10-28 02:00"), "price_d-1"] == df.loc[first, "price"]
    assert out.loc[local("2024-10-26 02:00"), "price_d+1"] == df.loc[first, "price"]
    # 03:00 on the switch day is 03:00 the day before, not 26 hours earlier.
    assert out.loc[local("2024-10-27 03:00"), "price_d-1"] == df.loc[local("2024-10-26 03:00"), "price"]


def test_day_lag_of_the_skipped_spring_hour_is_nan():
    df = local_frame("2024-03-29", "2024-04-03")
    out = LagTransformer(columns="price", lags=[1], freq="day").transform(df)
    assert np.isnan(out.loc[local("2024-04-01 02:00"), "price_d-1"])


@pytest.mark.parametrize("freq", ["1h", "15min"])
def test_day_lags_match_reference_calendar_shift_over_a_year(freq):
    df = local_frame("2024-01-01", "2025-01-01", freq=freq)
    out = LagTransformer(columns="price", lags=[1, 2, 7, -1, -7], freq="day").transform(df)
    for lag, name in [(1, "price_d-1"), (2, "price_d-2"), (7, "price_d-7"), (-1, "price_d+1"), (-7, "price_d+7")]:
        np.testing.assert_array_equal(out[name].to_numpy(), calendar_shift(df["price"], -lag).to_numpy(), name)


def test_hour_lags_stay_elapsed_time():
    df = local_frame("2024-03-30", "2024-04-02")
    out = LagTransformer(columns="price", lags=[24], freq="hour").transform(df)
    row = local("2024-03-31 10:00")
    assert out.loc[row, "price_h-24"] == df.loc[row - pd.Timedelta(hours=24), "price"]
    assert row - pd.Timedelta(hours=24) == local("2024-03-30 09:00")


def test_multi_day_frequency_counts_calendar_days():
    df = local_frame("2024-03-25", "2024-04-05")
    out = LagTransformer(columns="price", lags=[1], freq="2D").transform(df)
    row = local("2024-04-01 10:00")
    assert out.loc[row, "price_d-2"] == df.loc[local("2024-03-30 10:00"), "price"]


def test_calendar_dummy_leads_point_at_the_right_day():
    # F9: with elapsed-time leads, is_monday_d+1 was 0 on Sunday 2024-10-27 00:00 and 1 on Saturday 2024-03-30 23:00.
    source = CalendarSource(country="PL", holidays="binary", weekday="onehot")
    raw = source.fetch(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))
    df = TimezoneTransformer(TZ).transform(raw)
    cols = ["is_monday", "is_sunday", "is_holiday"]
    out = LagTransformer(columns=cols, lags=list(range(-7, 0)), freq="day").transform(df)

    assert out.loc[local("2024-10-27 00:00"), "is_monday_d+1"] == 1
    assert out.loc[local("2024-10-27 00:00"), "is_sunday_d+1"] == 0
    assert out.loc[local("2024-03-30 23:00"), "is_monday_d+1"] == 0

    by_date = df.groupby(df.index.date)[cols].first()
    dates = pd.Series(out.index.date, index=out.index)
    wrong = 0
    for k in range(1, 8):
        target = dates + pd.Timedelta(days=k)
        for col in cols:
            truth = target.map(by_date[col])
            lead = out[f"{col}_d+{k}"]
            wrong += int((lead.notna() & truth.notna() & (lead != truth)).sum())
    assert wrong == 0


def test_positive_day_lags_use_only_past_data():
    # Changing every value dated after D must not change any d-k feature on rows of D.
    df = local_frame("2024-10-20", "2024-11-05")
    issue_day = pd.Timestamp("2024-10-27").date()
    perturbed = df.copy()
    perturbed.loc[df.index.date > issue_day, "price"] += 1000.0

    lagger = LagTransformer(columns="price", lags=[1, 2, 7], freq="day")
    a, b = lagger.transform(df), lagger.transform(perturbed)
    on_d = a.index.date <= issue_day
    pd.testing.assert_frame_equal(
        a.loc[on_d, ["price_d-1", "price_d-2", "price_d-7"]], b.loc[on_d, ["price_d-1", "price_d-2", "price_d-7"]]
    )


# --------------------------------------------------------------------------- warnings


def test_day_lag_on_utc_index_warns():
    df = pd.DataFrame({"price": np.arange(72.0)}, index=pd.date_range("2024-01-01", periods=72, freq="1h", tz="UTC"))
    with pytest.warns(UserWarning, match="UTC"):
        LagTransformer(columns="price", lags=[1], freq="day").transform(df)


@pytest.mark.parametrize(
    "index, freq",
    [
        (pd.date_range("2024-01-01", periods=72, freq="1h", tz=TZ), "day"),
        (pd.date_range("2024-01-01", periods=72, freq="1h"), "day"),
        (pd.date_range("2024-01-01", periods=72, freq="1h", tz="UTC"), "hour"),
    ],
)
def test_no_warning_for_local_or_naive_day_lags_or_utc_hour_lags(index, freq):
    df = pd.DataFrame({"price": np.arange(72.0)}, index=index)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        LagTransformer(columns="price", lags=[1], freq=freq).transform(df)


# --------------------------------------------------------------------------- F11: parameters and naming


@pytest.fixture
def small_frame():
    index = pd.date_range("2024-01-01", periods=96, freq="1h", tz=TZ)
    return pd.DataFrame({"price": np.arange(96.0), "load": np.arange(96.0) + 1000}, index=index)


def test_elapsed_day_multiples_are_not_named_as_day_lags(small_frame):
    out = LagTransformer(columns="price", lags=[1], freq="day").transform(small_frame)
    out = LagTransformer(columns="price", lags=[24], freq="60min").transform(out)
    assert "price_h-24" in out.columns
    assert list(out.columns).count("price_d-1") == 1
    assert LagTransformer(columns="p", lags=[96], freq="15min")._format_lag_name("p", 96) == "p_h-24"
    assert LagTransformer(columns="p", lags=[1], freq="24h")._format_lag_name("p", 1) == "p_h-24"
    assert LagTransformer(columns="p", lags=[4], freq="15min")._format_lag_name("p", 4) == "p_h-1"
    assert LagTransformer(columns="p", lags=[-1], freq="1D")._format_lag_name("p", -1) == "p_d+1"


def test_name_collision_with_existing_column_raises(small_frame):
    once = LagTransformer(columns="price", lags=[1], freq="day").transform(small_frame)
    with pytest.raises(ValueError, match="price_d-1"):
        LagTransformer(columns="price", lags=[1], freq="day").transform(once)


def test_repeated_lags_create_one_column(small_frame):
    out = LagTransformer(columns="price", lags=[1, 1], freq="day").transform(small_frame)
    assert list(out.columns) == ["price", "load", "price_d-1"]


@pytest.mark.parametrize("lags", [np.int64(1), np.array([1, 2]), [np.int32(1), 2], range(1, 3)])
def test_numpy_and_range_lags_become_python_ints(lags):
    transformer = LagTransformer(columns="price", lags=lags, freq="day")
    assert all(type(lag) is int for lag in transformer.lags)


@pytest.mark.parametrize("lags", [1.5, [1, 2.0], "1", [True]])
def test_non_integer_lags_raise(lags):
    with pytest.raises(TypeError, match="integer"):
        LagTransformer(columns="price", lags=lags)


def test_columns_none_lags_all_columns(small_frame):
    out = LagTransformer(columns=None, lags=[1], freq="day").transform(small_frame)
    assert list(out.columns) == ["price", "load", "price_d-1", "load_d-1"]


def test_empty_columns_list_raises():
    with pytest.raises(ValueError, match="At least one column"):
        LagTransformer(columns=[], lags=[1])


def test_lag_zero_adds_no_duplicate_column(small_frame):
    out = LagTransformer(columns="price", lags=range(-2, 1), freq="day").transform(small_frame)
    assert "price_d-0" not in out.columns
    assert list(out.columns) == ["price", "load", "price_d+2", "price_d+1"]


def test_only_lag_zero_raises():
    with pytest.raises(ValueError, match="lag 0"):
        LagTransformer(columns="price", lags=[0])


def test_all_columns_only_with_backward_lags():
    # Forward lags of every column would include realised series and the target (price_d+1 would collide
    # with the model's own target columns).
    with pytest.raises(ValueError, match="columns=None"):
        LagTransformer(columns=None, lags=[1, -1], freq="day")


@pytest.mark.parametrize("tz", ["UTC", "Etc/UTC", "GMT", "Etc/GMT"])
def test_utc_like_zones_warn(tz):
    df = pd.DataFrame({"price": np.arange(72.0)}, index=pd.date_range("2024-01-01", periods=72, freq="1h", tz=tz))
    with pytest.warns(UserWarning, match="UTC"):
        LagTransformer(columns="price", lags=[1], freq="day").transform(df)
