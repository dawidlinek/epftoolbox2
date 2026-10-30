"""The data contract BaseModel.run checks before any fit, and how skipped or failed tasks are reported.

Each case here would run silently with wrong or missing forecasts (review findings C1, F3, M5 and the
worker error handling) if it were not checked: it must fail loudly or be counted.
"""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.transformers import DSTNormalizeTransformer, LagTransformer, ResampleTransformer
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.models import OLSModel, _contract
from epftoolbox2.models.base import BaseModel
from epftoolbox2.pipelines import ModelPipeline
from tests.synthetic import PREDICTORS, make_market_data


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


def run_ols(df, freq="1h", predictors=PREDICTORS, run_day="2023-04-12", horizon=2, **kwargs):
    model = OLSModel(predictors, training_window=45)
    return model.run(df, test_start=run_day, test_end=run_day, horizon=horizon, freq=freq, **kwargs)


# ---------------------------------------------------------------------------
# Frequency (C1)
# ---------------------------------------------------------------------------


def test_quarter_hour_data_with_hourly_freq_raises():
    df = make_market_data(days=120, freq="15min")
    with pytest.raises(ValueError, match="freq='1h'"):
        run_ols(df, freq="1h")


def test_hourly_data_with_quarter_hour_freq_raises():
    df = make_market_data(days=120, freq="1h")
    with pytest.raises(ValueError, match="freq='15min'"):
        run_ols(df, freq="15min")


def test_mixed_resolution_raises_even_when_the_most_common_step_matches():
    hourly = make_market_data(days=120, freq="1h")
    quarter = make_market_data(days=120, freq="15min")
    # One day of 15-minute rows inside otherwise hourly data: several rows per (day, hour).
    day = quarter.index.date == pd.Timestamp("2023-03-01").date()
    df = pd.concat([hourly[hourly.index.date != pd.Timestamp("2023-03-01").date()], quarter[day]]).sort_index()
    with pytest.raises(ValueError, match="more than one row"):
        run_ols(df, freq="1h")


class RowCounter(BaseModel):
    """Forecasts nothing useful; its only 'coefficient' is the number of training rows it was given."""

    def _fit_predict(self, train_x, train_y, test_x):
        return 0.0, [float(train_x.shape[0])]


def training_rows(df, freq, run_day="2023-04-20", training_window=90):
    ref = RowCounter(PREDICTORS, training_window=training_window).run(
        df, test_start=run_day, test_end=run_day, horizon=1, freq=freq
    )
    assert ref.produced == ref.count
    return {r["hour"]: int(r["coefficients"][0]) for r in ref._results}


def hourly_then_quarter_hour(cutover="2023-04-01"):
    """15-minute data that has only the :00 rows before `cutover`, like ENTSO-E DE_LU/PL prices (hourly before
    2025-10-01, 15-minute after). The hourly part spans the 2023-03-26 spring switch."""
    quarter = make_market_data(days=150, freq="15min")
    return quarter[(quarter.index >= pd.Timestamp(cutover, tz=quarter.index.tz)) | (quarter.index.minute == 0)]


def test_hourly_then_quarter_hour_data_raises_with_quarter_hour_freq():
    """Rows are most commonly 15 min apart and no quarter-hour slot has two rows, so this passed before: the
    :15/:30/:45 periods then trained on the 18 post-cutover days of a 90-day window, the :00 ones on all 90."""
    df = hourly_then_quarter_hour()
    with pytest.raises(ValueError, match="coarser") as info:
        run_ols(df, freq="15min", run_day="2023-04-20")
    message = str(info.value)
    assert "from 2023-01-01 00:00:00+01:00 to 2023-03-31 23:00:00+02:00" in message
    assert "1 h apart" in message
    assert "ResampleTransformer(freq='15min')" in message


def hourly_stretches(df, *ranges):
    """Keep only the :00 rows inside each [start, end) local range."""
    keep = np.ones(len(df), dtype=bool)
    for start, end in ranges:
        inside = (df.index >= pd.Timestamp(start, tz=df.index.tz)) & (df.index < pd.Timestamp(end, tz=df.index.tz))
        keep &= ~inside | (df.index.minute == 0)
    return df[keep]


def test_hourly_stretches_inside_quarter_hour_data_raise():
    quarter = make_market_data(days=120, freq="15min")
    df = hourly_stretches(quarter, ("2023-03-01", "2023-03-03"), ("2023-04-10", "2023-04-12"))
    with pytest.raises(ValueError, match="coarser") as info:
        run_ols(df, freq="15min")
    message = str(info.value)
    # 00:00 on 2023-03-01 still has 23:45 the day before as its neighbour.
    assert "47 rows from 2023-03-01 01:00:00+01:00 to 2023-03-02 23:00:00+01:00 are 1 h apart" in message
    assert "and so are 47 rows in 1 later stretch(es)" in message


@pytest.mark.parametrize("end", ["2023-03-01 13:00", "2023-03-02"])  # 3 h and 14 h
def test_hourly_stretch_shorter_than_a_day_counts_as_a_gap(end):
    """Like a missing day, it costs each quarter-hour period at most one training day."""
    df = hourly_stretches(make_market_data(days=120, freq="15min"), ("2023-03-01 10:00", end))
    _contract.check_index(df, "15min")


def test_hourly_stretch_is_refused_from_one_day_of_rows_without_a_neighbour():
    """The rows at either end of a stretch still have a 15-minute neighbour, so a whole local day leaves 23."""
    quarter = make_market_data(days=120, freq="15min")
    _contract.check_index(hourly_stretches(quarter, ("2023-03-01", "2023-03-02")), "15min")
    with pytest.raises(ValueError, match="24 rows from 2023-03-01 01:00:00\\+01:00 to 2023-03-02 00:00:00\\+01:00"):
        _contract.check_index(hourly_stretches(quarter, ("2023-03-01", "2023-03-02 01:00")), "15min")


@pytest.mark.parametrize("freq", ["1h", "15min"])
def test_scattered_missing_rows_pass(freq):
    """Rows lost at random (e.g. by dropna()) leave many rows with both neighbours missing, but unevenly spaced."""
    df = make_market_data(days=2 * 365, freq=freq)
    df = df[np.random.default_rng(0).random(len(df)) > 0.2]
    _contract.check_index(df, freq)


def test_daily_rows_before_hourly_ones_raise():
    hourly = make_market_data(days=120)
    early = hourly.index < pd.Timestamp("2023-02-01", tz=hourly.index.tz)
    df = hourly[~early | (hourly.index.hour == 0)]
    with pytest.raises(ValueError, match="from 2023-01-01 00:00:00\\+01:00 to 2023-01-31 00:00:00\\+01:00"):
        run_ols(df)


def test_rows_off_the_hourly_grid_raise():
    """A 10:30 row in an hour without a 10:00 row is still one row per (day, hour), but period 10 would then
    mix 10:00 and 10:30 values."""
    hourly = make_market_data(days=120)
    quarter = make_market_data(days=120, freq="15min")
    moved = hourly.index[(hourly.index.hour == 10) & (hourly.index.month == 2)]
    df = pd.concat([hourly.drop(index=moved), quarter.loc[moved + pd.Timedelta(minutes=30)]]).sort_index()
    with pytest.raises(ValueError, match="whole number") as info:
        run_ols(df)
    assert "2023-02-01 09:00:00+01:00 and 2023-02-01 10:30:00+01:00" in str(info.value)


def test_resampling_the_mixed_frame_gives_every_period_the_same_training_rows():
    df = ResampleTransformer(freq="15min").transform(hourly_then_quarter_hour())
    rows = training_rows(df, "15min")
    # 90-day window. The 02:00-02:45 periods (8-11) have no row on the 2023-03-26 spring switch and no
    # d+1 target on the day before it.
    assert rows == {p: 88 if 8 <= p <= 11 else 90 for p in range(96)}


@pytest.mark.parametrize("freq", ["1h", "15min"])
def test_gaps_and_dst_switches_pass_the_resolution_check(freq):
    """Missing days, missing periods and a lone row between two gaps are gaps, not a coarser resolution. The
    23/25-hour days (92/100 quarter-hours) of a local index are regular in elapsed time."""
    df = make_market_data(start="2023-03-01", days=250, freq=freq)
    per_day = pd.Series(df.index.date).value_counts()
    slots = 24 if freq == "1h" else 96
    assert {slots - slots // 24, slots + slots // 24} <= set(per_day)
    tz = df.index.tz
    step = pd.Timedelta(freq)
    lone = pd.Timestamp("2023-06-01 10:00", tz=tz)
    gaps = [
        df.index[df.index.date == pd.Timestamp("2023-05-10").date()],  # a whole day
        pd.DatetimeIndex([pd.Timestamp("2023-05-20 07:00", tz=tz)]),  # one period
        pd.DatetimeIndex([lone - step, lone + step]),  # a lone row between two gaps
    ]
    df = df.drop(index=gaps[0].append(gaps[1:]))
    rows = training_rows(df, freq, run_day="2023-07-01", training_window=60)
    # A missing row costs two training rows: its own and the d+1 target of the day before. So 60 days less two
    # for the missing day, and two more in the periods of 07:00 and of the lone row's dropped neighbours.
    short = {7, 9, 11} if freq == "1h" else {28, 39, 41}
    assert rows == {p: 56 if p in short else 58 for p in range(slots)}


@pytest.mark.parametrize("freq", ["1h", "15min"])
def test_dst_normalized_naive_index_passes(freq):
    df = DSTNormalizeTransformer(freq=freq).transform(make_market_data(start="2023-03-01", days=250, freq=freq))
    assert df.index.tz is None
    assert training_rows(df, freq, run_day="2023-07-01", training_window=60)


def test_fall_back_dst_repeated_hour_is_allowed():
    df = make_market_data(start="2023-09-01", days=80)
    assert (pd.Series(df.index.date).value_counts() == 25).any()
    ref = run_ols(df, run_day="2023-10-30")
    assert ref.produced == ref.count


def test_duplicate_timestamps_raise():
    df = make_market_data(days=120)
    df = pd.concat([df, df.iloc[[100]]]).sort_index()
    with pytest.raises(ValueError, match="duplicate"):
        run_ols(df)


def test_utc_index_with_day_lag_column_raises():
    """integration check d: a UTC index alone only warns (its own target shift is at least internally
    consistent, see tests/models/test_target_shift.py); a UTC index with an existing '_d-<k>' predictor
    must raise instead, since that column's values were computed against a different set of calendar-day
    boundaries than the ones now derived from the UTC index (e.g. epf data -> epf model losing the zone)."""
    df = make_market_data(days=120)  # has a price_d-1 column, per tests/synthetic.py
    df.index = df.index.tz_convert("UTC")
    with pytest.raises(ValueError, match="UTC"):
        run_ols(df, predictors=PREDICTORS + ["price_d-1"])


def test_utc_index_without_day_lag_column_only_warns():
    df = make_market_data(days=120).drop(columns=["price_d-1"])
    df.index = df.index.tz_convert("UTC")
    with pytest.warns(UserWarning, match="UTC"):
        run_ols(df)


def test_unsorted_rows_give_the_same_forecasts_as_sorted():
    df = make_market_data(days=120, noise=5.0, seed=4)
    shuffled = df.sample(frac=1.0, random_state=0)
    a = {(r["hour"], r["horizon"]): r["prediction"] for r in run_ols(df)._results}
    b = {(r["hour"], r["horizon"]): r["prediction"] for r in run_ols(shuffled)._results}
    assert a.keys() == b.keys()
    assert all(a[k] == pytest.approx(b[k], abs=1e-9) for k in a)


# ---------------------------------------------------------------------------
# Names: target collisions (F3), reserved ids (M5), missing / non-numeric predictors
# ---------------------------------------------------------------------------


def test_existing_target_columns_raise():
    df = make_market_data(days=120, horizon=3)
    lagged = LagTransformer(columns=["price"], lags=[-1, -2], freq="day").transform(df)
    with pytest.raises(ValueError, match=r"price_d\+1"):
        run_ols(lagged, horizon=3)


@pytest.mark.parametrize("column", ["hour", "day"])
def test_reserved_column_names_raise(column):
    df = make_market_data(days=120)
    df[column] = 1.0
    with pytest.raises(ValueError, match=f"'{column}'"):
        run_ols(df)


def test_predictor_naming_a_model_target_raises():
    df = make_market_data(days=120)
    with pytest.raises(ValueError, match="target"):
        run_ols(df, predictors=PREDICTORS + ["price_d+{horizon}"])


def test_missing_predictor_raises_before_any_result_is_written(tmp_path):
    df = make_market_data(days=120)
    out = tmp_path / "ols.jsonl"
    with pytest.raises(ValueError, match="load_d\\+3"):
        run_ols(df, predictors=["load_d+{horizon}", "wind_d+1"], horizon=3, save_to=str(out))
    assert not out.exists()


def test_pipeline_checks_every_model_before_running_any(tmp_path):
    df = make_market_data(days=120)
    pipeline = (
        ModelPipeline()
        .add_model(OLSModel(PREDICTORS, training_window=45, name="good"))
        .add_model(OLSModel(["not_a_column"], training_window=45, name="bad"))
    )
    with pytest.raises(ValueError, match="not_a_column"):
        pipeline.run(df, test_start="2023-04-12", test_end="2023-04-12", horizon=1, save_dir=str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_unused_string_columns_are_ignored():
    df = make_market_data(days=120)
    df["holiday_name"] = np.where(df.index.dayofweek == 6, "Sunday", None)
    ref = run_ols(df)
    assert ref.produced == ref.count == 48


def test_non_numeric_predictor_raises():
    df = make_market_data(days=120)
    df["holiday_name"] = "none"
    with pytest.raises(ValueError, match="holiday_name"):
        run_ols(df, predictors=PREDICTORS + ["holiday_name"])


def test_duplicate_model_names_raise():
    df = make_market_data(days=120)
    pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, name="OLS")).add_model(OLSModel(PREDICTORS, name="OLS"))
    with pytest.raises(ValueError, match="unique"):
        pipeline.run(df, test_start="2023-04-12", test_end="2023-04-12", horizon=1)


# ---------------------------------------------------------------------------
# Skipped and failed tasks
# ---------------------------------------------------------------------------


def test_missing_test_predictors_are_counted_and_reported():
    df = make_market_data(days=120)
    issue_rows = (df.index.date == pd.Timestamp("2023-04-12").date()) & np.isin(df.index.hour, [3, 4])
    df.loc[issue_rows, "load_d+1"] = np.nan

    report = (
        ModelPipeline()
        .add_model(OLSModel(PREDICTORS, training_window=45, name="OLS"))
        .add_evaluator(MAEEvaluator())
        .run(df, test_start="2023-04-12", test_end="2023-04-12", horizon=2)
    )
    ref = report.refs["OLS"]
    assert (ref.count, ref.produced) == (48, 46)
    assert ref.skipped == {"missing_test_predictors": 2}

    counts = report.forecast_counts().set_index("model")
    assert counts.loc["OLS", "planned"] == 48
    assert counts.loc["OLS", "produced"] == 46
    assert counts.loc["OLS", "missing_test_predictors"] == 2
    assert report.summary().set_index("model").loc["OLS", "forecasts"] == 46


def test_missing_test_row_is_skipped_not_mislabelled():
    """A gap on the issue day must not make the worker forecast from the previous day's row."""
    df = make_market_data(days=120, noise=5.0, seed=2)
    gap = pd.Timestamp("2023-04-12 07:00", tz=df.index.tz)
    df = df.drop(index=gap)

    ref = run_ols(df)
    produced = {(r["hour"], r["horizon"]) for r in ref._results}
    assert (7, 1) not in produced and (7, 2) not in produced
    assert ref.skipped == {"missing_test_row": 2}
    for r in ref._results:
        expected = df.loc[pd.Timestamp(f"{r['target_date']} {r['hour']:02d}:00", tz=df.index.tz), "price"]
        assert r["actual"] == pytest.approx(expected)


class FailingModel(BaseModel):
    def _fit_predict(self, train_x, train_y, test_x):
        raise RuntimeError("boom")


class SometimesInvalidModel(BaseModel):
    """Raises ValueError (as scikit-learn does for degenerate input) on every other call."""

    calls = 0

    def _fit_predict(self, train_x, train_y, test_x):
        type(self).calls += 1
        if type(self).calls % 2 == 0:
            raise ValueError("degenerate input")
        return 0.0, []


def test_unexpected_worker_error_fails_the_run():
    df = make_market_data(days=120)
    with pytest.raises(RuntimeError, match="boom"):
        FailingModel(PREDICTORS, training_window=45).run(df, test_start="2023-04-12", test_end="2023-04-12", horizon=1)


def test_unexpected_worker_error_fails_the_run_in_threaded_mode(monkeypatch):
    monkeypatch.setenv("THREADS_PER_PROCESS", "4")
    df = make_market_data(days=120)
    with pytest.raises(RuntimeError, match="boom"):
        FailingModel(PREDICTORS, training_window=45).run(df, test_start="2023-04-12", test_end="2023-04-12", horizon=1)


def test_fit_value_errors_are_counted():
    df = make_market_data(days=120)
    SometimesInvalidModel.calls = 0
    ref = SometimesInvalidModel(PREDICTORS, training_window=45).run(
        df, test_start="2023-04-12", test_end="2023-04-12", horizon=1
    )
    assert ref.count == 24
    assert ref.produced == 12
    assert ref.skipped == {"fit_error": 12}


def test_run_that_produces_nothing_raises():
    df = make_market_data(days=120)
    df.loc[df.index.date == pd.Timestamp("2023-04-12").date(), "load_d+1"] = np.nan
    with pytest.raises(RuntimeError, match="no forecasts"):
        run_ols(df, horizon=1)
