import datetime

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.validators import LagConsistencyValidator
from tests.synthetic import calendar_shift, make_market_data

TZ = "Europe/Warsaw"
SWITCH_DAYS = {datetime.date(2024, 3, 31), datetime.date(2024, 10, 27)}


def year_index(freq="1h"):
    return pd.date_range("2024-01-01", "2025-01-01", freq=freq, tz="UTC", inclusive="left").tz_convert(TZ)


def elapsed_shift(series, days):
    """What LagTransformer does today: the value `days` × 24 h earlier (later for negative days)."""
    return pd.Series(series.to_numpy(), index=series.index + pd.Timedelta(days=days)).reindex(series.index)


def lagged_frame(freq, shift):
    index = year_index(freq)
    price = pd.Series(np.arange(len(index), dtype=float), index=index)
    return pd.DataFrame(
        {
            "price": price,
            "price_d-1": shift(price, 1),
            "price_d-7": shift(price, 7),
            "price_d+1": shift(price, -1),
        }
    )


def calendar(series, days):
    return calendar_shift(series, -days)


def differs(a, b):
    return ~((a.isna() & b.isna()) | (a == b))


@pytest.mark.parametrize("freq", ["1h", "15min"])
def test_calendar_day_lags_are_consistent(freq):
    result = LagConsistencyValidator().validate(lagged_frame(freq, calendar))

    assert result.is_valid, result.errors
    assert result.info["mismatch_counts"] == {"price_d-1": 0, "price_d-7": 0, "price_d+1": 0}
    assert result.info["checked"]["price_d-7"] > 0


def test_synthetic_market_data_is_consistent():
    df = make_market_data(days=150)  # load_d+1, wind_d+2, ..., price_d-1 across the spring switch

    result = LagConsistencyValidator().validate(df)

    assert result.is_valid, result.errors
    assert "load_d+2" in result.info["checked"]


@pytest.mark.parametrize("freq", ["1h", "15min"])
def test_elapsed_time_lags_are_caught_around_dst(freq):
    df = lagged_frame(freq, elapsed_shift)
    reference = lagged_frame(freq, calendar)

    result = LagConsistencyValidator().validate(df)

    assert not result.is_valid
    for column in ["price_d-1", "price_d-7", "price_d+1"]:
        wrong = differs(df[column], reference[column])
        assert result.info["mismatch_counts"][column] == int(wrong.sum()) > 0
    wrong_d1 = df.index[differs(df["price_d-1"], reference["price_d-1"])]
    assert set(wrong_d1.date) <= {d + datetime.timedelta(days=k) for d in SWITCH_DAYS for k in (0, 1)}
    first = result.info["mismatches"]["price_d-1"][0]
    assert first["timestamp"].date() in SWITCH_DAYS
    assert "'price_d-1' is not 'price' 1 local calendar day earlier" in result.errors[0]


def test_known_issue_row_is_flagged():
    # The row pinned by test_lag_transformer_day_lag_keeps_wall_clock_hour_across_dst.
    df = lagged_frame("1h", elapsed_shift)

    result = LagConsistencyValidator(columns=["price_d-1"]).validate(df)

    row = result.stats.set_index("column").loc["price_d-1"]
    flagged = {m["timestamp"] for m in result.info["mismatches"]["price_d-1"]}
    assert row["mismatches"] > 0
    assert pd.Timestamp("2024-03-31 03:00", tz=TZ) in flagged
    assert pd.Timestamp("2024-03-31 10:00", tz=TZ) in flagged


def test_hourly_counts_per_year():
    result = LagConsistencyValidator().validate(lagged_frame("1h", elapsed_shift))

    # One lag day: the switch day and the early hours of the next. Seven lag days: a week after each switch.
    assert result.info["mismatch_counts"]["price_d-1"] == 49
    assert result.info["mismatch_counts"]["price_d-7"] == 337


def test_utc_index_needs_the_local_time_zone():
    df = lagged_frame("1h", calendar)
    df.index = df.index.tz_convert("UTC")

    assert LagConsistencyValidator(tz=TZ).validate(df).is_valid
    assert not LagConsistencyValidator().validate(df).is_valid  # UTC days are not the local days the lags used


def test_edges_that_cannot_be_verified_are_skipped():
    full = lagged_frame("1h", calendar)
    sliced = full.iloc[24 * 30 : 24 * 60]  # lags computed on the longer frame keep values at the edges

    result = LagConsistencyValidator().validate(sliced)

    assert result.is_valid
    row = result.stats.set_index("column").loc["price_d-7"]
    assert row["unverifiable"] == 24 * 7


def test_lag_with_value_where_base_is_missing_is_flagged():
    df = lagged_frame("1h", calendar).iloc[: 24 * 20].copy()
    df.loc[df.index[100], "price"] = np.nan

    result = LagConsistencyValidator(columns=["price_d-1"]).validate(df)

    assert result.info["mismatch_counts"]["price_d-1"] == 1


def test_tolerance():
    df = lagged_frame("1h", calendar).iloc[: 24 * 20].copy()
    df["price_d-1"] += 1e-12

    assert LagConsistencyValidator().validate(df).is_valid
    assert not LagConsistencyValidator(tolerance=0.0).validate(df).is_valid


def test_non_numeric_columns():
    index = year_index()[: 24 * 10]
    names = pd.Series(index.day_name(), index=index)
    df = pd.DataFrame({"day": names, "day_d-1": calendar(names, 1)})

    assert LagConsistencyValidator().validate(df).is_valid
    df["day_d-1"] = names
    assert not LagConsistencyValidator().validate(df).is_valid


def test_forecast_columns_without_a_base_are_not_checked():
    index = year_index()[:48]
    df = pd.DataFrame({"temperature_d+1": 1.0, "price": 1.0}, index=index)

    result = LagConsistencyValidator().validate(df)

    assert result.is_valid
    assert len(result.warnings) == 1


def test_explicit_columns_must_be_day_lags_of_present_columns():
    index = year_index()[:48]
    df = pd.DataFrame({"price": 1.0, "price_h-24": 1.0, "load_d-1": 1.0}, index=index)

    result = LagConsistencyValidator(columns=["price_h-24", "load_d-1", "nope_d-1"]).validate(df)

    assert not result.is_valid
    assert len(result.errors) == 3


def test_nullable_integer_columns():
    index = year_index()[: 24 * 10]
    load = pd.Series(pd.array(range(len(index)), dtype="Int64"), index=index)
    lagged = pd.Series(pd.array(calendar(load.astype(float), 1).astype("Int64")), index=index)
    df = pd.DataFrame({"load": load, "load_d-1": lagged})

    assert LagConsistencyValidator().validate(df).is_valid


def test_invalid_arguments_raise():
    with pytest.raises(ValueError, match="time zone"):
        LagConsistencyValidator(tz="Nowhere/Land")
    with pytest.raises(ValueError, match="tolerance"):
        LagConsistencyValidator(tolerance=-1.0)
