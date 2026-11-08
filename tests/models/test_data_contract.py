"""The data contract BaseModel.run checks before any fit, and how skipped or failed tasks are reported.

Each case here would run silently with wrong or missing forecasts (review findings C1, F3, M5 and the
worker error handling) if it were not checked: it must fail loudly or be counted.
"""

import datetime
import json

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.transformers import DSTNormalizeTransformer, LagTransformer, ResampleTransformer
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.models import LassoCVModel, OLSModel, _contract
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


def hourly_then_quarter_hour(cutover="2023-04-01", days=150):
    """15-minute data that has only the :00 rows before `cutover`, like ENTSO-E DE_LU/PL prices (hourly before
    2025-10-01, 15-minute after). The hourly part spans the 2023-03-26 spring switch."""
    quarter = make_market_data(days=days, freq="15min")
    return quarter[(quarter.index >= pd.Timestamp(cutover, tz=quarter.index.tz)) | (quarter.index.minute == 0)]


def test_hourly_then_quarter_hour_data_raises_with_quarter_hour_freq():
    """Rows are most commonly 15 min apart and no quarter-hour slot has two rows, so this passed before: the
    :15/:30/:45 periods then trained on the 18 post-cutover days of a 90-day window, the :00 ones on all 90."""
    df = hourly_then_quarter_hour()
    with pytest.raises(ValueError, match="coarser") as info:
        run_ols(df, freq="15min", run_day="2023-04-20")
    message = str(info.value)
    # The run reads back to 2023-03-03 (2023-04-20 less 45 + 2 + 1 days); older rows can't reach a fit.
    assert "from 2023-03-03 00:00:00+01:00 to 2023-03-31 23:00:00+02:00" in message
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
        run_ols(df, run_day="2023-02-15")


def test_rows_off_the_hourly_grid_raise():
    """A 10:30 row in an hour without a 10:00 row is still one row per (day, hour), but period 10 would then
    mix 10:00 and 10:30 values."""
    hourly = make_market_data(days=120)
    quarter = make_market_data(days=120, freq="15min")
    moved = hourly.index[(hourly.index.hour == 10) & (hourly.index.month == 2)]
    df = pd.concat([hourly.drop(index=moved), quarter.loc[moved + pd.Timedelta(minutes=30)]]).sort_index()
    with pytest.raises(ValueError, match="whole number") as info:
        run_ols(df, run_day="2023-03-10")
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


# ---------------------------------------------------------------------------
# Only the days a run can read are checked for spacing
# ---------------------------------------------------------------------------

SWITCH = "2025-10-01"
LAST_MONTH = {"test_start": "2026-03-28", "test_end": "2026-03-29", "horizon": 2, "freq": "15min"}  # spring switch


@pytest.fixture(scope="module")
def long_history():
    """3 years of hourly rows, then 6 months of 15-minute ones, like DE_LU/PL prices around 2025-10-01."""
    quarter = make_market_data(start="2022-10-01", days=1278, freq="15min", noise=5.0, seed=3)
    return quarter[(quarter.index >= pd.Timestamp(SWITCH, tz=quarter.index.tz)) | (quarter.index.minute == 0)]


def records(ref):
    """Every field of every forecast. JSON writes the shortest float repr that round-trips, so equal text is equal bits."""
    return json.dumps(sorted(ref._results, key=lambda r: (r["day_in_test"], r["hour"], r["horizon"])))


def local_days(df, first, last):
    """Rows on local days first..last, both included."""
    wall = df.index.tz_localize(None)
    return df[(wall >= pd.Timestamp(first)) & (wall < pd.Timestamp(last) + pd.Timedelta(days=1))]


@pytest.mark.parametrize(
    "model_cls, model_kwargs, run_kwargs",
    [
        (OLSModel, {}, LAST_MONTH),
        (LassoCVModel, {"cv": 3}, {**LAST_MONTH, "test_end": LAST_MONTH["test_start"], "horizon": 1}),  # slower
    ],
    ids=["OLS", "LassoCV"],
)
def test_unread_coarse_history_leaves_the_forecasts_bitwise_identical(
    long_history, model_cls, model_kwargs, run_kwargs
):
    """The proof that the scope is safe: the same run on the frame without the hourly part gives the same bits."""
    only_quarter_hours = long_history[long_history.index >= pd.Timestamp(SWITCH, tz=long_history.index.tz)]
    full = model_cls(PREDICTORS, training_window=90, **model_kwargs).run(long_history, **run_kwargs)
    sliced = model_cls(PREDICTORS, training_window=90, **model_kwargs).run(only_quarter_hours, **run_kwargs)
    assert full.produced > 0.9 * full.count
    assert records(full) == records(sliced)
    assert (full.count, full.produced, full.skipped) == (sliced.count, sliced.produced, sliced.skipped)


def test_a_run_reads_exactly_the_days_inside_the_margin(long_history):
    """`read_days` is one day wider on each side than what is read: with exactly those rows nothing changes, and
    without the first or the last of them something does, so the bound is neither too narrow nor needlessly wide."""

    def run(frame):
        return records(OLSModel(PREDICTORS, training_window=90).run(frame, **LAST_MONTH))

    first, last = _contract.read_days(LAST_MONTH["test_start"], LAST_MONTH["test_end"], 90, LAST_MONTH["horizon"])
    assert (first, last) == (pd.Timestamp("2025-12-25"), pd.Timestamp("2026-04-01"))
    day = pd.Timedelta(days=1)
    baseline = run(long_history)
    assert run(local_days(long_history, first + day, last - day)) == baseline
    assert run(local_days(long_history, first + 2 * day, last - day)) != baseline
    assert run(local_days(long_history, first + day, last - 2 * day)) != baseline


def test_training_window_reaching_the_coarse_part_raises(long_history):
    kwargs = {**LAST_MONTH, "test_start": "2026-03-15", "test_end": "2026-03-16"}
    with pytest.raises(ValueError, match="coarser") as info:
        OLSModel(PREDICTORS, training_window=365).run(long_history, **kwargs)
    # 2026-03-15 less 365 + 2 + 1 days; the older hourly rows are not read.
    assert "from 2025-03-12 00:00:00+01:00 to 2025-09-30 23:00:00+02:00" in str(info.value)
    assert "2025-03-12 to 2026-03-19" in str(info.value)


@pytest.mark.parametrize(
    "test_start, test_end, training_window, message",
    [
        ("2025-09-29", "2025-11-30", 7, "coarser"),  # mostly 15-minute rows
        ("2025-09-20", "2025-10-10", 90, "most commonly 1 h apart"),  # mostly hourly rows
        ("2025-06-01", "2025-06-02", 7, "most commonly 1 h apart"),  # only hourly rows
    ],
)
def test_test_window_overlapping_the_coarse_part_raises(long_history, test_start, test_end, training_window, message):
    with pytest.raises(ValueError, match=message):
        OLSModel(PREDICTORS, training_window=training_window).run(
            long_history, test_start=test_start, test_end=test_end, horizon=1, freq="15min"
        )


def test_coarse_history_outside_every_training_window_passes():
    df = hourly_then_quarter_hour(days=300)
    rows = training_rows(df, "15min", run_day="2023-09-01", training_window=45)
    assert rows == {p: 45 for p in range(96)}


@pytest.mark.parametrize(
    "run_day, raises",
    [
        ("2023-05-18", False),  # the checked days start on the switch, 2023-04-01
        # They start a day earlier: 24 hourly rows, in the margin but not read. It takes 24 only because the cut leaves
        # 00:00 with no row before it, so it counts as having no neighbour. A whole hourly day with 15-minute rows on
        # both sides leaves 23 and passes. So this case moves with the one-day threshold and with how the first row of
        # the checked days is counted; it says nothing about which days are read, so don't widen read_days to fix it.
        ("2023-05-17", True),
    ],
)
def test_the_checked_days_start_one_day_before_the_first_day_read(run_day, raises):
    df = hourly_then_quarter_hour(days=300)
    if raises:
        with pytest.raises(ValueError, match="from 2023-03-31 00:00:00\\+02:00 to 2023-03-31 23:00:00\\+02:00"):
            training_rows(df, "15min", run_day=run_day, training_window=45)
    else:
        assert training_rows(df, "15min", run_day=run_day, training_window=45)


@pytest.mark.parametrize(
    "days_after_run_day, raises",
    [(3, False), (1, True)],  # after everything read; on the targets of the run day
)
def test_coarse_rows_after_the_test_window_count_only_if_read(days_after_run_day, raises):
    quarter = make_market_data(days=120, freq="15min")
    run_day = pd.Timestamp("2023-04-12")
    coarse_from = pd.Timestamp(run_day + pd.Timedelta(days=days_after_run_day), tz=quarter.index.tz)
    df = quarter[(quarter.index < coarse_from) | (quarter.index.minute == 0)]
    if raises:
        with pytest.raises(ValueError, match="coarser"):
            training_rows(df, "15min", run_day="2023-04-12", training_window=45)
    else:
        assert training_rows(df, "15min", run_day="2023-04-12", training_window=45)


def test_no_test_window_checks_the_whole_frame():
    df = hourly_then_quarter_hour(days=300)
    whole_frame = "from 2023-01-01 00:00:00\\+01:00 to 2023-03-31 23:00:00\\+02:00"
    with pytest.raises(ValueError, match=whole_frame):
        _contract.check_index(df, "15min")
    with pytest.raises(ValueError, match=whole_frame):
        OLSModel(PREDICTORS, training_window=45)._check_inputs(df, "price", 1, "15min")


@pytest.mark.parametrize("training_window", ["expanding", [45, "expanding"]])
def test_an_expanding_training_window_checks_the_whole_frame(training_window):
    """An expanding window trains on every row from the frame's first date, so the hourly history is read."""
    df = hourly_then_quarter_hour(days=300)
    model = OLSModel(PREDICTORS, training_window=training_window)
    with pytest.raises(ValueError, match="from 2023-01-01 00:00:00\\+01:00"):
        model._check_inputs(df, "price", 1, "15min", None, "2023-09-01", "2023-09-01")


@pytest.mark.parametrize("training_window", [45.0, None, "45"])
def test_a_training_window_that_is_not_an_int_is_refused(training_window):
    """It used to check the whole frame and run; such a window is now refused where it is set (`resolve_training_window`)."""
    with pytest.raises(ValueError, match="training_window takes whole numbers of days"):
        OLSModel(PREDICTORS, training_window=training_window)


def test_a_numpy_int_training_window_is_an_int():
    df = hourly_then_quarter_hour(days=300)
    OLSModel(PREDICTORS, training_window=np.int64(45))._check_inputs(
        df, "price", 1, "15min", None, "2023-09-01", "2023-09-01"
    )


def test_duplicates_are_refused_outside_the_days_read():
    df = make_market_data(days=120)
    df = pd.concat([df, df.iloc[[10]]]).sort_index()  # on 2023-01-01, long before the days read
    with pytest.raises(ValueError, match="duplicate"):
        run_ols(df)


def test_several_rows_in_a_slot_are_refused_outside_the_days_read():
    """Finer rows are the look-ahead case, so this check stays on the whole frame, even for rows the run won't read."""
    df = hourly_then_quarter_hour(days=300)
    with pytest.raises(ValueError, match="more than one row"):
        OLSModel(PREDICTORS, training_window=7).run(
            df, test_start="2023-02-15", test_end="2023-02-15", horizon=1, freq="1h"
        )


@pytest.fixture
def checked_days(monkeypatch):
    """The `days` of every check_index call in the test (the checks run in this process, not in the workers)."""
    seen = []
    check_index = _contract.check_index

    def spy(data, freq, days=None):
        seen.append(days)
        return check_index(data, freq, days)

    monkeypatch.setattr(_contract, "check_index", spy)
    return seen


@pytest.mark.parametrize("as_date", [False, True])
def test_pipeline_precheck_reads_the_same_days_as_run(tmp_path, as_date, checked_days):
    df = hourly_then_quarter_hour(days=300)
    day = (lambda s: datetime.date.fromisoformat(s)) if as_date else (lambda s: s)

    def run(run_day, save_dir=None):
        pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, training_window=45, name="OLS"))
        return pipeline.run(
            df, test_start=day(run_day), test_end=day(run_day), horizon=1, freq="15min", save_dir=save_dir
        )

    assert run("2023-09-01").refs["OLS"].produced == 96
    # The pre-check, which the run then uses (`run(checked=)`): 2023-09-01 less 45 + 1 + 1 days, plus 1 + 1 days.
    assert checked_days == [(pd.Timestamp("2023-07-16"), pd.Timestamp("2023-09-03"))]
    # A run on its own checks the same days.
    OLSModel(PREDICTORS, training_window=45, name="OLS").run(
        df, day("2023-09-01"), day("2023-09-01"), horizon=1, freq="15min"
    )
    assert checked_days == [(pd.Timestamp("2023-07-16"), pd.Timestamp("2023-09-03"))] * 2
    with pytest.raises(ValueError, match="coarser"):
        run("2023-04-20", save_dir=str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def history_up_to_today(hourly_until_days_ago):
    """Hourly rows until that many days ago, then 15-minute ones through the day after tomorrow."""
    today = pd.Timestamp(datetime.date.today())
    quarter = make_market_data(start=str((today - pd.Timedelta(days=300)).date()), days=303, freq="15min", horizon=1)
    switch = pd.Timestamp(today - pd.Timedelta(days=hourly_until_days_ago), tz=quarter.index.tz)
    return quarter[(quarter.index >= switch) | (quarter.index.minute == 0)]


@pytest.mark.parametrize("use_pipeline", [False, True])
def test_forecast_only_checks_the_days_read_from_today(use_pipeline, checked_days):
    def run(df):
        model = OLSModel(PREDICTORS, training_window=45, name="OLS")
        if use_pipeline:
            return ModelPipeline().add_model(model).run(df, horizon=1, freq="15min", forecast_only=True).refs["OLS"]
        return model.run(df, horizon=1, freq="15min", forecast_only=True)

    ref = run(history_up_to_today(hourly_until_days_ago=100))
    today = datetime.date.today().isoformat()
    assert ref.test_start == today and ref.produced > 0
    # A pipeline's run uses its pre-check (`run(checked=)`): one check either way.
    assert checked_days == [_contract.read_days(today, today, 45, 1)]
    with pytest.raises(ValueError, match="coarser"):
        run(history_up_to_today(hourly_until_days_ago=20))


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
