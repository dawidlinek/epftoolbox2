"""DSTNormalizeTransformer: 24 (or 96) wall-clock slots per local day."""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.transformers import DSTNormalizeTransformer, LagTransformer, TimezoneTransformer
from epftoolbox2.models import OLSModel
from tests.synthetic import make_market_data

TZ = "Europe/Warsaw"
FLAG = "is_dst_adjusted"


def local_frame(start: str, end: str, freq: str = "1h", seed: int = 0) -> pd.DataFrame:
    index = pd.date_range(start, end, freq=freq, tz=TZ, inclusive="left")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"price": rng.normal(50, 10, len(index)), "load": rng.normal(1000, 50, len(index))}, index=index
    )


def rows_per_day(df: pd.DataFrame) -> pd.Series:
    return df.groupby(df.index.date).size()


@pytest.mark.parametrize("freq, slots", [("1h", 24), ("15min", 96)])
def test_every_local_day_gets_the_same_number_of_slots(freq, slots):
    df = local_frame("2024-01-01", "2025-01-01", freq=freq)
    assert set(rows_per_day(df)) == {slots - slots // 24, slots, slots + slots // 24}

    out = DSTNormalizeTransformer(freq=freq).transform(df)

    assert out.index.tz is None
    assert out.index.is_unique and out.index.is_monotonic_increasing
    assert set(rows_per_day(out)) == {slots}
    assert out.index[0] == pd.Timestamp("2024-01-01 00:00")
    assert list(out.columns) == ["price", "load", FLAG]
    assert out[FLAG].sum() == 2 * (slots // 24)


def test_spring_hour_is_the_average_of_its_neighbours():
    df = local_frame("2024-03-30", "2024-04-02")
    out = DSTNormalizeTransformer().transform(df)

    before, after = df.loc[pd.Timestamp("2024-03-31 01:00", tz=TZ)], df.loc[pd.Timestamp("2024-03-31 03:00", tz=TZ)]
    filled = out.loc[pd.Timestamp("2024-03-31 02:00")]
    assert filled["price"] == pytest.approx((before["price"] + after["price"]) / 2)
    assert filled["load"] == pytest.approx((before["load"] + after["load"]) / 2)
    assert out.index[out[FLAG] == 1].tolist() == [pd.Timestamp("2024-03-31 02:00")]
    # Every other slot is the original value.
    real = out[out[FLAG] == 0]
    np.testing.assert_array_equal(real["price"].to_numpy(), df["price"].to_numpy())


def test_repeated_autumn_hour_is_averaged():
    df = local_frame("2024-10-26", "2024-10-29")
    out = DSTNormalizeTransformer().transform(df)

    first = df.loc[pd.Timestamp("2024-10-27 02:00+02:00"), "price"]
    second = df.loc[pd.Timestamp("2024-10-27 02:00+01:00"), "price"]
    assert out.loc[pd.Timestamp("2024-10-27 02:00"), "price"] == pytest.approx((first + second) / 2)
    assert out.index[out[FLAG] == 1].tolist() == [pd.Timestamp("2024-10-27 02:00")]
    assert len(out) == len(df) - 1


def test_quarter_hours_on_switch_days():
    df = local_frame("2024-03-30", "2024-04-01", freq="15min")
    out = DSTNormalizeTransformer(freq="15min").transform(df)
    before = df.loc[pd.Timestamp("2024-03-31 01:45", tz=TZ), "price"]
    after = df.loc[pd.Timestamp("2024-03-31 03:00", tz=TZ), "price"]
    filled = out.loc["2024-03-31 02:00":"2024-03-31 02:45", "price"]
    assert len(filled) == 4
    np.testing.assert_allclose(filled.to_numpy(), (before + after) / 2)

    df = local_frame("2024-10-26", "2024-10-28", freq="15min")
    out = DSTNormalizeTransformer(freq="15min").transform(df)
    for minute in (0, 15, 30, 45):
        wall = pd.Timestamp(f"2024-10-27 02:{minute:02d}")
        pair = df.loc[[pd.Timestamp(f"{wall}{offset}").tz_convert(TZ) for offset in ("+02:00", "+01:00")], "price"]
        assert out.loc[wall, "price"] == pytest.approx(pair.mean())


def test_missing_values_are_not_averaged_away():
    df = local_frame("2024-10-26", "2024-10-29")
    df.loc[pd.Timestamp("2024-10-27 02:00+01:00"), "price"] = np.nan
    spring = local_frame("2024-03-30", "2024-04-02")
    spring.loc[pd.Timestamp("2024-03-31 03:00", tz=TZ), "price"] = np.nan

    assert np.isnan(DSTNormalizeTransformer().transform(df).loc[pd.Timestamp("2024-10-27 02:00"), "price"])
    assert np.isnan(DSTNormalizeTransformer().transform(spring).loc[pd.Timestamp("2024-03-31 02:00"), "price"])


def test_gaps_and_partial_edge_days_are_left_alone():
    # UTC-midnight bounds give partial first/last local days; a missing row is a real gap, not a DST slot.
    index = pd.date_range("2024-03-29", "2024-04-02", freq="1h", tz="UTC", inclusive="left").tz_convert(TZ)
    df = pd.DataFrame({"price": np.arange(len(index), dtype=float)}, index=index).drop(index[40])

    out = DSTNormalizeTransformer().transform(df)

    assert out.index[0] == pd.Timestamp("2024-03-29 01:00")
    assert out.index[-1] == index[-1].tz_localize(None)
    assert index[40].tz_localize(None) not in out.index
    assert len(out) == len(df) + 1  # only the skipped spring hour is added


def test_non_numeric_columns_take_first_occurrence_and_previous_slot():
    df = local_frame("2024-03-30", "2024-04-02")
    df["weekday_name"] = df.index.day_name()
    autumn = local_frame("2024-10-26", "2024-10-29")
    autumn["weekday_name"] = autumn.index.day_name()

    assert DSTNormalizeTransformer().transform(df).loc[pd.Timestamp("2024-03-31 02:00"), "weekday_name"] == "Sunday"
    assert DSTNormalizeTransformer().transform(autumn).loc[pd.Timestamp("2024-10-27 02:00"), "weekday_name"] == "Sunday"


def test_tz_parameter_converts_a_utc_index_first():
    index = pd.date_range("2024-10-25", "2024-10-30", freq="1h", tz="UTC", inclusive="left")
    df = pd.DataFrame({"price": np.arange(len(index), dtype=float)}, index=index)

    direct = DSTNormalizeTransformer(tz=TZ).transform(df)
    two_step = DSTNormalizeTransformer().transform(TimezoneTransformer(TZ, daily_columns=[]).transform(df))

    pd.testing.assert_frame_equal(direct, two_step)


@pytest.mark.parametrize(
    "index, match",
    [
        (pd.date_range("2024-01-01", periods=48, freq="1h"), "time-zone-aware"),
        (pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC"), "UTC"),
        (pd.date_range("2024-01-01", periods=48, freq="15min", tz=TZ), "15min|grid|resample"),
    ],
)
def test_rejects_indexes_it_cannot_normalise(index, match):
    df = pd.DataFrame({"price": np.arange(len(index), dtype=float)}, index=index)
    with pytest.raises(ValueError, match=match):
        DSTNormalizeTransformer(freq="1h").transform(df)


def test_rejects_existing_flag_column_and_bad_freq():
    df = local_frame("2024-01-01", "2024-01-03")
    df[FLAG] = 0
    with pytest.raises(ValueError, match=FLAG):
        DSTNormalizeTransformer().transform(df)
    with pytest.raises(ValueError, match="freq"):
        DSTNormalizeTransformer(freq="7h")


def test_changing_data_after_d_does_not_change_rows_of_d():
    df = local_frame("2024-10-20", "2024-11-03")
    for issue_day in ["2024-10-26", "2024-10-27"]:
        day = pd.Timestamp(issue_day).date()
        perturbed = df.copy()
        perturbed.loc[df.index.date > day] += 1000.0
        a = DSTNormalizeTransformer().transform(df)
        b = DSTNormalizeTransformer().transform(perturbed)
        upto = a.index.date <= day
        pd.testing.assert_frame_equal(a[upto], b[upto])
    spring = local_frame("2024-03-25", "2024-04-05")
    day = pd.Timestamp("2024-03-31").date()
    perturbed = spring.copy()
    perturbed.loc[spring.index.date > day] += 1000.0
    a, b = DSTNormalizeTransformer().transform(spring), DSTNormalizeTransformer().transform(perturbed)
    pd.testing.assert_frame_equal(a[a.index.date <= day], b[b.index.date <= day])


# --------------------------------------------------------------------------- with the model


def normalised_market_data(df: pd.DataFrame) -> pd.DataFrame:
    base = DSTNormalizeTransformer().transform(df[["load", "wind", "is_weekend", "is_weekday", "price"]])
    base = LagTransformer(columns=["load", "wind", "is_weekend", "is_weekday"], lags=[-1, -2], freq="day").transform(
        base
    )
    return LagTransformer(columns="price", lags=[1], freq="day").transform(base)


PREDICTORS = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "is_weekday_d+{horizon}", "price_d-1"]


@pytest.fixture
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


@pytest.mark.parametrize("run_day", ["2023-10-29", "2023-10-30", "2023-10-31"])
def test_model_on_normalised_data_has_no_lookahead_after_fall_back(single_process, run_day):
    # The same scenario as the fall-back known issue in tests/test_known_issues.py, on normalised data.
    raw = make_market_data(start="2023-08-01", days=120, noise=5.0, seed=1)
    scrambled = raw.copy()
    future = pd.Series(raw.index.date >= pd.Timestamp(run_day).date(), index=raw.index)
    scrambled.loc[future, "price"] = 1e4 + np.arange(future.sum())

    def predictions(data):
        ref = OLSModel(PREDICTORS, training_window=40).run(
            normalised_market_data(data), test_start=run_day, test_end=run_day, horizon=2
        )
        return {(r["hour"], r["horizon"]): r["prediction"] for r in ref._results}

    original, perturbed = predictions(raw), predictions(scrambled)
    assert len(original) == 48
    assert [key for key in original if abs(original[key] - perturbed[key]) > 1e-9] == []


def test_model_on_normalised_data_scores_every_hour_of_the_spring_day(single_process):
    data = normalised_market_data(make_market_data(noise=5.0, seed=1))
    ref = OLSModel(PREDICTORS, training_window=40).run(data, test_start="2023-03-25", test_end="2023-03-26", horizon=2)

    assert len(ref._results) == ref.count == 2 * 24 * 2
    for r in ref._results:
        assert r["actual"] == pytest.approx(data.loc[pd.Timestamp(f"{r['target_date']} {r['hour']:02d}:00"), "price"])


def test_yaml_round_trip(tmp_path):
    from epftoolbox2.pipelines import DataPipeline

    path = tmp_path / "dp.yaml"
    DataPipeline().add_transformer(DSTNormalizeTransformer(freq="15min", tz=TZ)).save(path)
    loaded = DataPipeline.load(path).transformers[0]

    assert isinstance(loaded, DSTNormalizeTransformer)
    assert (loaded.freq, loaded.tz) == ("15min", TZ)
