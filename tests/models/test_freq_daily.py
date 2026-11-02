"""`freq="1D"`: one model per day on a frame with one row per local date, each stamped at the time of day it stands for.

A daily frame cut from an hourly one at noon is held against period 12 of the hourly run, which shares no daily code, key by key and
bit for bit; the forecasts against a fit with numpy alone; the availability of each kind of rule on a daily row against instants
written down with pandas; and the probes against models that leak on purpose, against a frame stamped at midnight, which hides a
late input, and against a control that can't reach a model.
"""

import datetime
import json
import warnings
from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest

import epftoolbox2.information_set as information_module
from epftoolbox2.information_set import Availability, InformationSet, InformationSetError
from epftoolbox2.models import OLSModel, Use
from epftoolbox2.models._calendar import IssueCalendar, clock_periods, periods_per_day
from epftoolbox2.models._contract import check_index, missing_test_rows, period_index
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.models.base import BaseModel
from epftoolbox2.pipelines import ModelPipeline
from tests.synthetic import PREDICTORS, calendar_shift, make_market_data, with_daily_columns

SWITCH_DAYS = {"spring": "2023-03-26", "autumn": "2023-10-29"}
WINDOW = 40
NOON = "12:00"
RULES = {
    "load": Availability.published("10:00", 7),  # its `_d+h` columns stand for a load forecast
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "price": Availability.published("13:00", 1),
    "load_actual": Availability.realised("1h"),
    "actual_daily_*": Availability.realised("1h"),
}
# Known by noon on D whatever the date: a forecast, a price of D-1, an hour lag of two, a daily statistic of D-1.
DAILY_INPUTS = PREDICTORS + ["price_d-1", "load_actual_h-2", "actual_daily_max_d-1"]


class InProcessPool:
    """Stands in for the worker pool, whose start is most of the time these runs would take."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        _worker_init(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, task, *args):
        future = Future()
        try:
            future.set_result(task(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)


def day_of(day, offset=0):
    return (pd.Timestamp(day) + pd.Timedelta(days=offset)).date().isoformat()


def hourly_market(switch=None, horizon=3, seed=3, days=100, zone="Europe/Warsaw"):
    """Hourly rows from 70 days before a switch (or from 1 February), with the columns of `with_daily_columns`."""
    start = day_of(SWITCH_DAYS[switch], -70) if switch else "2023-02-01"
    data = make_market_data(start=start, days=days, tz=zone, horizon=horizon, noise=5.0, seed=seed)
    return with_daily_columns(data)


@pytest.fixture(scope="module")
def year():
    """Hourly rows over both switches of 2023, and the daily frame of their noons."""
    hourly = hourly_market(days=300)
    return hourly, hourly.at_time(NOON)


def info(issue=NOON, zone="Europe/Warsaw", **more):
    return InformationSet(issue, zone, rules={**RULES, **more})


def by_key(records):
    return {(r["run_date"], r["hour"], r["horizon"]): r for r in records}


def as_text(record, *drop):
    """A record as JSON without some keys: equal text is equal to the bit, and a NaN equals itself."""
    return json.dumps({k: v for k, v in record.items() if k not in drop}, sort_keys=True)


# ---------------------------------------------------------------------------
# The frame contract
# ---------------------------------------------------------------------------


def test_a_daily_frame_is_one_period_on_the_clock_of_every_date():
    assert periods_per_day("1D") == 1
    for day in ("2023-03-26", "2023-10-29", "2023-06-01"):
        assert clock_periods(day, "1D", "Europe/Warsaw") == (0,)
    calendar = IssueCalendar("1D", "Europe/Warsaw", [1, 2, 3])
    assert calendar.count("2023-03-20", "2023-11-05") == 231 * 3
    assert calendar.tasks(datetime.date(2023, 3, 25), [0, 1]) == [(0, h, d) for d in (0, 1) for h in (1, 2, 3)]


@pytest.mark.parametrize("stamp", ["12:00", "00:00", "23:30", "06:15"])
@pytest.mark.parametrize("zone", ["Europe/Warsaw", "UTC", None])
def test_one_row_a_date_at_one_time_that_exists_on_every_date_passes(year, stamp, zone):
    _, daily = year
    index = daily.index.tz_localize(None).normalize() + pd.Timedelta(stamp + ":00")
    frame = daily.set_axis(index if zone is None else index.tz_localize(zone))
    check_index(frame, "1D")
    gappy = frame.drop(frame.index[[5, 6, 40, 200]])  # gaps are allowed, as at any freq
    check_index(gappy, "1D")
    assert (period_index(gappy.index, "1D") == 0).all()
    dates = [day.date() for day in pd.to_datetime(["2023-02-06", "2023-02-07", "2023-02-08"])]
    assert missing_test_rows(gappy.index, "1D", dates) == [
        (0, pd.Timestamp("2023-02-06")),
        (0, pd.Timestamp("2023-02-07")),
    ]


def test_hourly_data_with_freq_1d_raises_and_says_to_take_one_time_of_day(year):
    hourly, _ = year
    with pytest.raises(ValueError, match=r"most commonly 1 h apart.*at_time\('12:00'\).*freq='1h'"):
        check_index(hourly, "1D")
    with pytest.raises(ValueError, match="at_time"):
        OLSModel(PREDICTORS, training_window=WINDOW).run(hourly, "2023-04-01", "2023-04-02", horizon=1, freq="1D")


def test_two_rows_on_a_date_raise(year):
    _, daily = year
    extra = daily.iloc[[10]].set_axis(daily.index[[10]] + pd.Timedelta(hours=6))
    with pytest.raises(ValueError, match=r"1 local date\(s\) have more than one row \(e\.g\. 2023-02-11\)"):
        check_index(pd.concat([daily, extra]).sort_index(), "1D")


def test_rows_at_several_times_of_day_raise(year):
    _, daily = year
    index = daily.index.to_series()
    index.iloc[30] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match=r"one wall-clock time of day.*2 times \(12:00, 13:00\)"):
        check_index(daily.set_axis(pd.DatetimeIndex(index)), "1D")
    with pytest.raises(ValueError, match="one wall-clock time"):
        OLSModel(PREDICTORS, training_window=WINDOW).run(
            daily.set_axis(pd.DatetimeIndex(index)), "2023-04-01", "2023-04-02", horizon=1, freq="1D"
        )


@pytest.mark.parametrize(("stamp", "date"), [("02:30", "2023-03-26"), ("02:30", "2023-10-29")])
def test_a_time_the_clocks_skip_or_repeat_raises(stamp, date):
    """Warsaw skips 02:30 on 26 March and repeats it on 29 October: that date's row would be another instant, or two. The date
    has no row here, and still the stamp is refused."""
    wall = pd.date_range(day_of(date, -20), day_of(date, 20), freq="D") + pd.Timedelta(stamp + ":00")
    wall = wall[wall.normalize() != pd.Timestamp(date)]
    frame = pd.DataFrame({"price": np.arange(len(wall), dtype=float)}, index=wall.tz_localize("Europe/Warsaw"))
    with pytest.raises(
        ValueError, match=rf"at 02:30, which the clock of Europe/Warsaw skips or repeats on 1 date\(s\).*{date}"
    ):
        check_index(frame, "1D")
    check_index(frame.tz_localize(None), "1D")  # a naive index has every time once


# ---------------------------------------------------------------------------
# A daily run is period 12 of an hourly run
# ---------------------------------------------------------------------------


def weekday_lists(ctx):
    """Another list on weekdays, so that a list's mask and its first test day matter."""
    return PREDICTORS + (["price_d-1"] if ctx.issue_day.weekday() < 5 else [])


def weekend_matching(ctx):
    """Forecasts of a weekend train on the days whose forecast was of a weekend too."""
    return Use(PREDICTORS, train_on="matching") if ctx.target_day.weekday() >= 5 else PREDICTORS


LISTS = {
    "fixed": dict(predictors=PREDICTORS + ["price_d-1", "load_actual_h-2"]),
    "function": dict(predictors=weekday_lists),
    "matching": dict(predictors=weekend_matching),
    "published": dict(predictors=PREDICTORS + ["price_d-1"], newest_target="published"),
}


@pytest.mark.parametrize("switch", list(SWITCH_DAYS))
@pytest.mark.parametrize("lists", list(LISTS))
def test_a_daily_run_is_period_12_of_an_hourly_run_bit_for_bit(year, switch, lists):
    """Issue days S-2 to S+3 of each switch, horizons 1 to 3: every record of the noon frame is the hourly run's record of
    period 12, prediction and actual and everything else to the bit, but for `hour` and `freq`. Period 12 is one row a date at
    noon too, chosen by the same dates, so nothing about a switch is left to tell them apart."""
    hourly, daily = year
    S = SWITCH_DAYS[switch]
    first, last = day_of(S, -2), day_of(S, 3)
    settings = dict(LISTS[lists])
    extra = {"information_set": info()} if settings.get("newest_target") == "published" else {}

    def run(frame, freq):
        model = OLSModel(training_window=WINDOW, **settings)
        return model.run(frame, first, last, horizon=3, freq=freq, **extra)._results

    hourly_records = {(r["run_date"], r["horizon"]): r for r in run(hourly, "1h") if r["hour"] == 12}
    daily_records = {(r["run_date"], r["horizon"]): r for r in run(daily, "1D")}
    assert len(daily_records) == 6 * 3
    assert daily_records.keys() == hourly_records.keys()
    for key, record in daily_records.items():
        assert (record["hour"], record["freq"]) == (0, "1D")
        assert as_text(record, "hour", "freq") == as_text(hourly_records[key], "hour", "freq"), key
    assert all(np.isfinite(r["actual"]) for r in daily_records.values())


# ---------------------------------------------------------------------------
# By hand
# ---------------------------------------------------------------------------


def hand_daily_forecast(daily, target, names, day, horizon, newest=-1, window=WINDOW):
    """The forecast of a daily frame with numpy alone: the rows dated D-W-h ... D + min(k - h, -1), one a date, each with the
    target of h dates later at its time of day, the 0/1 columns as they are and the others standardised on the rows that train."""
    dates = daily.index.tz_localize(None).normalize()
    D = pd.Timestamp(day)
    last = D + pd.Timedelta(days=min(newest - horizon, -1))
    rows = np.flatnonzero((dates >= last - pd.Timedelta(days=window - 1)) & (dates <= last))
    values = pd.Series(daily[target].to_numpy(), index=dates)
    y = values.reindex(dates[rows] + pd.Timedelta(days=horizon)).to_numpy()
    x = daily[names].to_numpy(dtype=float)
    usable = np.isfinite(y) & np.isfinite(x[rows]).all(axis=1)
    y, train = y[usable], x[rows][usable]
    scaled = np.array([not set(np.unique(daily[name].dropna())) <= {0.0, 1.0} for name in names])
    means = np.where(scaled, train.mean(axis=0), 0.0)
    deviations = np.where(scaled, train.std(axis=0, ddof=1), 1.0)
    coefficients, *_ = np.linalg.lstsq((train - means) / deviations, (y - y.mean()) / y.std(ddof=1), rcond=None)
    test = x[np.flatnonzero(dates == D)[0]]
    return float(((test - means) / deviations) @ coefficients * y.std(ddof=1) + y.mean()), dates[rows][usable]


@pytest.mark.parametrize("switch", list(SWITCH_DAYS))
def test_an_ols_forecast_of_a_daily_frame_is_a_hand_fit_on_w_dates(year, switch):
    """Over both switches every forecast trains on W rows, one a date, the newest dated D-1-h: a 23- or 25-hour date is one row
    like any other."""
    _, daily = year
    S = SWITCH_DAYS[switch]
    found = by_key(
        OLSModel(DAILY_INPUTS, training_window=WINDOW)
        .run(daily, day_of(S, -1), day_of(S, 2), horizon=3, freq="1D")
        ._results
    )
    assert len(found) == 4 * 3
    price = pd.Series(daily["price"].to_numpy(), index=daily.index.tz_localize(None).normalize())
    for (run_date, _, h), record in found.items():
        names = [name.replace("{horizon}", str(h)) for name in DAILY_INPUTS]
        prediction, dates = hand_daily_forecast(daily, "price", names, run_date, h)
        assert len(dates) == WINDOW and dates.is_unique
        assert dates.max() == pd.Timestamp(day_of(run_date, -1 - h))
        assert record["prediction"] == pytest.approx(prediction, abs=1e-9), (run_date, h)
        assert record["actual"] == price[pd.Timestamp(day_of(run_date, h))]


# ---------------------------------------------------------------------------
# What a daily row's inputs hold, and when they are known
# ---------------------------------------------------------------------------

ISSUE_DAYS = [
    "2023-03-24",
    "2023-03-26",
    "2023-03-27",
    "2023-10-29",
    "2023-10-30",
]  # a Friday, both switch days, the days after


def at(day, offset, time, zone="Europe/Warsaw"):
    """The instant of a wall-clock time `offset` dates after `day` in `zone`, the later one where the clocks repeat it."""
    wall = pd.Timestamp(day_of(day, offset) + " " + time)
    return wall.tz_localize(zone, ambiguous=False)


def friday_before_the_week_of(day):
    monday = pd.Timestamp(day) - pd.Timedelta(days=pd.Timestamp(day).weekday())
    return day_of(monday, -3)


# Each kind of rule on a row stamped at noon, and the instant its value is known, written down from the rule. `realised` named
# alone is the value of the whole date, known an hour after the date ends (decision 12); with an hour shift it is an hour's.
TABLE = {
    "is_weekend_d+3": (Availability.static(), lambda D: None),
    "load_d+3": (Availability.published("10:00", 7), lambda D: at(D, -4, "10:00", "Europe/Brussels")),
    "fc_d+3": (
        Availability.weekly("friday", "10:00", "Europe/Warsaw"),
        lambda D: at(friday_before_the_week_of(day_of(D, 3)), 0, "10:00"),
    ),
    "price_d-1": (Availability.published("13:00", 1), lambda D: at(D, -2, "13:00", "Europe/Brussels")),
    "actual_daily_max": (Availability.realised("1h"), lambda D: at(D, 1, "00:00") + pd.Timedelta(hours=1)),
    "actual_daily_max_d-1": (Availability.realised("1h"), lambda D: at(D, 0, "00:00") + pd.Timedelta(hours=1)),
    "load_actual_h-1": (Availability.realised("1h"), lambda D: at(D, 0, "13:00")),
    "load_actual_h-2": (Availability.realised("1h"), lambda D: at(D, 0, "12:00")),
    "w_d+3": (Availability.weather_cutoff("00", "8h"), lambda D: at(D, 0, "08:00", "UTC")),
    "snap_d+2": (Availability.as_of("06:00", "Europe/Warsaw"), lambda D: at(D, 0, "06:00")),
    "wr_d+1": (Availability.weather_run("6h"), lambda D: at(D, 0, "18:00")),
    "load_actual_h05_d-1": (Availability.realised("1h"), lambda D: at(D, -1, "07:00")),
    "load_actual_day_max_d-1": (Availability.realised("1h"), lambda D: at(D, 0, "00:00") + pd.Timedelta(hours=1)),
}


def test_each_rule_on_a_daily_row_is_known_when_written_down_by_hand():
    """Over both switches: a date's value is known after its 23 or 25 hours, an hour lag is elapsed time from the stamp, a
    profile and a day aggregate are read at their slot, as on hourly rows, and the rest by the day their shifts reach."""
    rules = {"load": TABLE["load_d+3"][0], "fc": TABLE["fc_d+3"][0], "price": TABLE["price_d-1"][0]}
    rules |= {"is_weekend": Availability.static(), "w": TABLE["w_d+3"][0], "snap": TABLE["snap_d+2"][0]}
    rules |= {"wr": TABLE["wr_d+1"][0], "load_actual": Availability.realised("1h")}
    rules |= {"actual_daily_*": Availability.realised("1h")}
    information = InformationSet(NOON, "Europe/Warsaw", rules=rules)
    index = pd.date_range("2023-03-01 12:00", "2023-11-30 12:00", freq="D", tz="Europe/Warsaw")
    days = pd.DatetimeIndex(ISSUE_DAYS)
    clock = information._issue_clock(days, index, "1D")
    origin = pd.Timestamp(ISSUE_DAYS[0], tz="Europe/Warsaw")
    for name, (_, by_hand) in TABLE.items():
        found = clock.available(information._resolve_for(name, 60), np.array([0]), 60)[0]
        for day, minutes in zip(ISSUE_DAYS, found):
            expected = by_hand(day)
            expected = -np.inf if expected is None else (expected - origin) / pd.Timedelta(minutes=1)
            assert minutes == expected, (name, day, clock.label(minutes, ISSUE_DAYS.index(day)))
    late = information.violations({(0, 1): list(TABLE)}, "price", "1D", index, issue_days=ISSUE_DAYS)[0]
    assert sorted({v.predictor for v in late}) == ["actual_daily_max", "load_actual_h-1", "wr_d+1"]
    assert information._resolve_for("load_actual_s30", 60).rule is None  # a half-hour slot needs a rule of its own


def test_a_daily_statistic_is_late_on_its_date_whatever_the_issue_time():
    """At 14:00 an hourly reading of the noon row (known 13:00) would let the date's maximum through; the date isn't over."""
    index = pd.date_range("2023-03-01 12:00", "2023-11-30 12:00", freq="D", tz="Europe/Warsaw")
    for issue in ("12:00", "14:00", "23:30"):
        late = info(issue).violations(
            {(0, 1): ["actual_daily_max", "load_actual_h-1"]}, "price", "1D", index, "2023-03-20", "2023-04-02"
        )[0]
        assert {v.predictor for v in late} == (
            {"actual_daily_max", "load_actual_h-1"} if issue == "12:00" else {"actual_daily_max"}
        )


def test_newest_targets_of_a_daily_frame_are_one_row_of_k(year):
    """`realised("1h")` at noon gives k = -1, at 14:00 too (the date ends at midnight); a price published 13:00 the day before
    gives 0 at noon and 1 at 14:00."""
    _, daily = year
    days = (
        pd.date_range("2023-03-20", "2023-04-02").date.tolist()
        + pd.date_range("2023-10-25", "2023-11-02").date.tolist()
    )
    for issue, price_k in (("12:00", 0), ("14:00", 1)):
        k = info(issue).newest_targets("actual_daily_max", "1D", daily.index, days, 3)
        assert k.shape == (1, len(days)) and (k == -1).all()
        assert (info(issue).newest_targets("price", "1D", daily.index, days, 3) == price_k).all()


def test_a_published_daily_statistic_trains_up_to_d_2_as_the_spec_s_table_says(year):
    """The 3c table's row `actual_daily_max`, 1D at noon, issue 11:45 Brussels: k = -1, so the newest row at h = 1 is D-2, by
    hand, and every forecast equals the "d-1" one to the bit."""
    _, daily = year
    names = PREDICTORS + ["actual_daily_max_d-1"]
    brussels = info("11:45", "Europe/Brussels")
    first, last = "2023-10-27", "2023-10-31"

    def run(newest_target):
        model = OLSModel(names, training_window=WINDOW, newest_target=newest_target)
        records = model.run(daily, first, last, target="actual_daily_max", horizon=3, freq="1D", cut_rules=brussels)
        return by_key(records._results)

    published, d_1 = run("published"), run("d-1")
    assert len(published) == 5 * 3
    for key, record in published.items():
        run_date, _, h = key
        assert record["newest_target_day"] == day_of(run_date, -1)
        prediction, dates = hand_daily_forecast(
            daily, "actual_daily_max", [n.replace("{horizon}", str(h)) for n in names], run_date, h
        )
        assert dates.max() == pd.Timestamp(day_of(run_date, -1 - h))
        assert record["prediction"] == pytest.approx(prediction, abs=1e-9)
        assert as_text(record, "newest_target_day") == as_text(d_1[key])


def with_the_date_if_published(ctx):
    """Adds the date's own maximum where `ctx` says it is out: never, on a daily frame, at any issue time on that date."""
    out = DAILY_INPUTS + (["actual_daily_max"] if ctx.published("actual_daily_max") else [])
    return out + (["price_d-1"] if not ctx.published("actual_daily_max_d-1") else [])


@pytest.mark.parametrize("issue", ["12:00", "14:00"])
def test_a_predictor_function_on_a_daily_frame_sees_one_period_and_what_is_out(year, issue):
    """One period, 0, at each horizon; the date's own maximum is never out on it, at 14:00 either, and the day before's always."""
    _, daily = year
    model = OLSModel(with_the_date_if_published, training_window=WINDOW)
    with pytest.warns(UserWarning, match=r"published\('actual_daily_max'\) was never True.*known D\+1 01:00"):
        plan = model._plan_predictors(
            daily, "price", 2, "1D", info(issue), "2023-03-20", "2023-04-02", availability="raise"
        )
    assert sorted(plan.assignment) == [(0, 1), (0, 2)]
    assert plan.lists == [tuple(name.replace("{horizon}", str(h)) for name in DAILY_INPUTS) for h in (1, 2)]


DATE_RULES = {
    "price_daily_max": Availability.published("13:00", 1),  # Brussels: a date's prices out at 13:00 the day before
    "fuel_ttf": Availability.daily_close("18:00"),  # Brussels: a trading day's close at 18:00 on it
}


def market_of_dates(zone, stamp):
    """Daily rows of a market in `zone` stamped at `stamp`, with the maximum price of each local date and of the next, and a
    value of each local date that closes on it."""
    hourly = make_market_data(start="2023-01-01", days=200, tz=zone, horizon=2, noise=5.0, seed=4)
    local_date = pd.Series(hourly.index.tz_localize(None).normalize(), index=hourly.index)
    hourly["price_daily_max"] = hourly["price"].groupby(local_date).transform("max")
    hourly["price_daily_max_d+1"] = calendar_shift(hourly["price_daily_max"], 1)
    hourly["price_daily_max_d-1"] = calendar_shift(hourly["price_daily_max"], -1)
    hourly["fuel_ttf"] = hourly["load"].groupby(local_date).transform("mean")
    return hourly.at_time(stamp)


@pytest.mark.parametrize("stamp", ["00:00", "12:00"])
@pytest.mark.parametrize("name", ["price_daily_max_d+1", "fuel_ttf"])
def test_a_helsinki_date_is_known_when_the_later_of_its_two_brussels_days_is(stamp, name):
    """A Helsinki date runs from 23:00 to 23:00 Brussels, so its prices are out when the later Brussels day's are, at 13:00 the
    day before it, and its close at 18:00 on it. At a midnight stamp the row's own instant is on the earlier Brussels day: read
    there, both passed as known a day early. By hand, then through a run that refuses late inputs."""
    daily = market_of_dates("Europe/Helsinki", stamp)
    information = InformationSet("11:45", "Europe/Brussels", rules={**RULES, **DATE_RULES})
    days = ["2023-02-14", "2023-06-14"]
    clock = information._issue_clock(pd.DatetimeIndex(days), daily.index, "1D")
    origin = pd.Timestamp(days[0], tz="Europe/Helsinki")  # the clock counts from the data zone's midnight
    by_hand = {
        "price_daily_max_d+1": lambda D: at(D, 0, "13:00", "Europe/Brussels"),
        "fuel_ttf": lambda D: at(D, 0, "18:00", "Europe/Brussels"),
        "price_daily_max_d-1": lambda D: at(D, -2, "13:00", "Europe/Brussels"),
    }
    for column, expected in by_hand.items():
        found = clock.available(information._resolve_for(column, 60), np.array([0]), 60)[0]
        assert found.tolist() == [(expected(day) - origin) / pd.Timedelta(minutes=1) for day in days], column
    model = OLSModel(PREDICTORS + [name], training_window=WINDOW, name="m")
    pipeline = ModelPipeline(information_set=information, availability="raise").add_model(model)
    with pytest.raises(InformationSetError, match=name.replace("+", r"\+")):
        pipeline.run(daily, test=("2023-06-01", "2023-06-03"), horizon=2, freq="1D")


@pytest.mark.parametrize(
    ("data_zone", "rule_zone", "stamp"),
    [
        ("Asia/Kolkata", "Asia/Kathmandu", "23:45"),
        ("Asia/Kolkata", "Asia/Kathmandu", "23:59"),
        ("Australia/Adelaide", "Australia/Sydney", "23:30"),
    ],
)
def test_a_date_s_value_is_never_known_before_its_stamp_reads_it(data_zone, rule_zone, stamp):
    """Read at its date's last slot, 23:00, a value stamped later could fall on the rule zone's earlier day: Kathmandu's midnight
    is 23:45 in Kolkata. So a date's value is read at the later of the two, and is never known before the row's own stamp says."""
    rules = {
        "p": Availability.published("13:00", 1, rule_zone),
        "w": Availability.weekly("friday", "10:00", rule_zone),
        "c": Availability.daily_close("18:00", rule_zone),
    }
    information = InformationSet("11:45", rule_zone, rules=rules)
    wall = pd.date_range("2023-01-01", "2023-12-31", freq="D") + pd.Timedelta(stamp + ":00")
    index = wall.tz_localize(data_zone)
    later = 0
    for run_day in ["2023-04-02", "2023-06-15", "2023-10-01"]:
        ctx, day, minute, first = information_module._rows_around_issue(index, run_day, information)
        for name in ["p", "p_d+1", "p_d-1", "w_d+3", "w_d+9", "c", "c_d-1"]:
            col = information._resolve_for(name, 60)
            as_dates = information_module._available_array(col, col.rule, day, minute, 60, ctx, first, daily=True)
            at_stamp = information_module._available_array(col, col.rule, day, minute, 60, ctx, first, daily=False)
            assert (as_dates >= at_stamp).all(), (name, run_day)
            later += int((as_dates > at_stamp).sum())
    assert later == 0  # a stamp after 23:00 is the date's latest instant: read there, nothing moves


def test_where_the_date_is_one_brussels_day_the_stamp_changes_nothing():
    """Warsaw's dates are Brussels' dates, so a date's value is known at the same instant at any stamp."""
    information = InformationSet("11:45", "Europe/Brussels", rules={**RULES, **DATE_RULES})
    days = pd.DatetimeIndex(["2023-02-14", "2023-03-26", "2023-10-29"])
    names = ["price_daily_max_d+1", "price_daily_max_d-1", "fuel_ttf", "price_d-1", "load_d+2"]
    found = {}
    for stamp in ("00:00", "12:00", "23:00"):
        index = market_of_dates("Europe/Warsaw", stamp).index
        clock = information._issue_clock(days, index, "1D")
        found[stamp] = [clock.available(information._resolve_for(n, 60), np.array([0]), 60)[0].tolist() for n in names]
    assert found["00:00"] == found["12:00"] == found["23:00"]


@pytest.mark.parametrize("zone", ["America/Havana", "Europe/Warsaw"])
def test_a_date_whose_midnight_the_clock_skips_still_has_its_daily_row_judged(zone):
    """Havana skips 00:00 to 00:59 on 12 March 2023: hour 0 isn't on that date's clock, but a daily row is, so its own maximum
    is late there as on any date, not a slot with no verdict."""
    index = pd.date_range("2023-02-01 12:00", "2023-04-30 12:00", freq="D", tz=zone)
    late = info(NOON, zone).violations({(0, 1): ["actual_daily_max"]}, "price", "1D", index, issue_days=["2023-03-12"])[
        0
    ]
    assert [v.predictor for v in late] == ["actual_daily_max"]


# ---------------------------------------------------------------------------
# The probes on daily frames
# ---------------------------------------------------------------------------


class TrainingRowsOnly(BaseModel):
    """Reads its training rows and never the row it forecasts from, so only a change to a training row can move it."""

    def _fit_predict(self, train_x, train_y, test_x):
        return float(train_y.mean() + train_x[:, 0].max()), []


def verified(daily, predictors, target="price", issue=NOON, model=OLSModel, days=5, **kwargs):
    pipeline = ModelPipeline(information_set=info(issue)).add_model(model(predictors, training_window=WINDOW, name="m"))
    first, last = kwargs.pop("period", ("2023-10-27", "2023-10-31"))
    return pipeline.verify(daily, first, last, target=target, freq="1D", days=days, **kwargs)


@pytest.mark.parametrize("switch", list(SWITCH_DAYS))
@pytest.mark.parametrize("target", ["price", "actual_daily_max"])
def test_verify_is_ok_on_a_daily_frame_with_every_kind_of_input(year, switch, target):
    """`x_d+h`, `x_d-1`, `x_h-2` and a daily statistic of D-1, on the days of both switches, for a price and for a daily
    statistic as the target."""
    _, daily = year
    S = SWITCH_DAYS[switch]
    report = verified(daily, DAILY_INPUTS, target, horizon=2, period=(day_of(S, -2), day_of(S, 2)))
    report.assert_ok()
    assert len(report.checks) == 5 and all(check.control.moved for check in report.checks)


def test_verify_is_ok_with_issue_days_and_horizons_on_a_daily_frame(year):
    _, daily = year
    report = verified(
        daily, DAILY_INPUTS, horizons=[2, 3], issue_days=["friday"], days=3, period=("2023-03-01", "2023-04-30")
    )
    report.assert_ok()
    assert {pd.Timestamp(check.run_day).weekday() for check in report.checks} == {4}


@pytest.mark.parametrize("late", ["actual_daily_max", "load_actual_h-1"])
def test_a_daily_statistic_of_d_or_the_hour_before_the_stamp_is_a_leak(year, late):
    _, daily = year
    report = verified(daily, DAILY_INPUTS + [late], horizon=2)
    assert report.status("m") == "leak"
    assert all(check.probe.changed for check in report.checks)


def test_a_frame_stamped_at_midnight_hides_the_hour_before_noon(year):
    """Decision 11: the S02 file holds at midnight what is true at noon. Its `load_actual_h-1` is the load of 11:00 on D,
    known at 13:00, but at a midnight stamp the name says 23:00 on D-1, known by 01:00, and the check and the probe believe it.
    At noon both see the leak."""
    hourly, daily = year
    names = DAILY_INPUTS + ["load_actual_h-1"]
    midnight = daily.set_axis(daily.index.normalize())
    eleven = hourly["load_actual"].at_time("11:00")
    assert np.array_equal(
        midnight["load_actual_h-1"].to_numpy(), eleven.reindex(daily.index - pd.Timedelta(hours=1)).to_numpy()
    )
    with pytest.warns(UserWarning, match="stamped at 00:00.*restamp"):  # what can't be seen, only warned about
        verified(daily.set_axis(daily.index.normalize()), names, horizon=2).assert_ok()
    assert verified(daily, names, horizon=2).status("m") == "leak"
    expanded = {(0, 1): names}
    assert not info().violations(expanded, "price", "1D", midnight.index, "2023-10-27", "2023-10-31")[0]
    assert [
        v.predictor for v in info().violations(expanded, "price", "1D", daily.index, "2023-10-27", "2023-10-31")[0]
    ] == ["load_actual_h-1"]


def midnight_warnings(daily, names, information_set=None, period=("2023-10-27", "2023-10-28")):
    """The warnings about a midnight stamp a run of `names` on `daily` gives, and its records."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = OLSModel(names, training_window=WINDOW, name="m")
        records = model.run(daily, *period, horizon=1, freq="1D", cut_rules=information_set)._results
    return [str(w.message) for w in caught if "restamp" in str(w.message)], records


def test_a_midnight_stamp_with_hour_lags_warns_and_moves_nothing(year, monkeypatch):
    """Decision 11: neither the check nor a probe can see a midnight stamp on rows that hold noon's values, so a run says what
    to look at when the rows are stamped at 00:00 and a predictor counts hours from the stamp. Nothing is refused, nothing moves."""
    hourly, daily = year
    names = PREDICTORS + ["load_actual_h-1"]
    midnight = daily.set_axis(daily.index.normalize())
    found, records = midnight_warnings(midnight, names)
    assert (
        len(found) == 1
        and "m: the daily rows are stamped at 00:00, midnight" in found[0]
        and "load_actual_h-1" in found[0]
    )
    with monkeypatch.context() as patched:
        patched.setattr("epftoolbox2.models._contract.warn_about_a_midnight_stamp", lambda *args, **kwargs: None)
        silent, quiet = midnight_warnings(midnight, names)
    assert silent == [] and [as_text(r) for r in records] == [as_text(r) for r in quiet]
    assert midnight_warnings(daily, names)[0] == []  # noon
    assert midnight_warnings(midnight, PREDICTORS + ["price_d-1"])[0] == []  # no hour counted from the stamp
    assert midnight_warnings(hourly.at_time("00:00"), names)[0]  # warned too: only the user knows what the rows hold


@pytest.mark.parametrize(("stamp", "warns"), [("01:00", True), ("23:00", True), ("02:00", False), ("12:00", False)])
def test_a_stamp_within_a_rule_zone_s_offset_of_midnight_warns(stamp, warns):
    """Helsinki's 01:00 is Brussels' midnight, so with a Brussels rule on an hour-shifted predictor it warns, and 02:00 doesn't."""
    daily = market_of_dates("Europe/Helsinki", stamp)
    daily = daily.assign(**{"price_h-1": daily["price"]})
    information = InformationSet("11:45", "Europe/Brussels", rules={**RULES, **DATE_RULES})
    found, _ = midnight_warnings(daily, PREDICTORS + ["price_h-1"], information, ("2023-06-01", "2023-06-02"))
    assert bool(found) == warns
    if warns:
        assert f"stamped at {stamp}, within 01:00 of midnight in the zone of a rule" in found[0]


def test_the_positive_control_can_fail_on_a_daily_frame(year):
    """A model that reads only its training rows can't be reached through the row it forecasts from: inconclusive."""
    _, daily = year
    report = verified(daily, DAILY_INPUTS, model=TrainingRowsOnly, horizon=2, days=2)
    assert report.status("m") == "inconclusive"
    assert all(check.probe.ok and check.control.moved == [] and check.control.unmoved for check in report.checks)


def from_hours(source):
    """What `rebuild` makes of hourly source rows: the daily columns from the measured load, and the noon rows."""
    return with_daily_columns(source).at_time(NOON)


def test_verify_with_rebuild_reads_the_hourly_source_of_a_daily_model_by_its_hours(year):
    """With `rebuild` the probes perturb the source, whose rows are hours: the load of 10:00 is known at 12:00 there, so
    `load_actual_h-2` built from it is no leak, while the 11:00 one behind `load_actual_h-1` is perturbed and moves the forecast."""
    hourly, _ = year
    raw = ["price", "load", "wind", "is_weekend", "is_weekday", "load_actual", "price_d-1"]
    source = hourly[raw + [f"{c}_d+{h}" for h in (1, 2, 3) for c in ("load", "wind", "is_weekend", "is_weekday")]]
    report = verified(source, DAILY_INPUTS, horizon=2, days=2, rebuild=from_hours)
    report.assert_ok()
    assert all(check.control.moved for check in report.checks)
    report = verified(source, DAILY_INPUTS + ["load_actual_h-1"], horizon=2, days=2, rebuild=from_hours)
    assert report.status("m") == "leak"


@pytest.mark.parametrize(("stamp", "issue"), [("00:00", "11:45"), ("12:00", "14:00")])
def test_with_rebuild_a_whole_date_s_value_the_source_repeats_on_every_hour_is_a_leak(year, stamp, issue):
    """The source holds D's maximum on every hour of D, and an hour of it is known by the issue time read as an hour. On the
    daily row it is D's maximum, known only after D: a leak with `rebuild` as without."""
    hourly = year[0].drop(columns=["price_daily_max"])  # no rule, so the probe of the source would refuse it
    names = PREDICTORS + ["price_d-1", "load_actual_h-2", "actual_daily_max"]
    plain = verified(hourly.at_time(stamp), names, issue=issue, horizon=2, days=2)
    rebuilt = verified(hourly, names, issue=issue, horizon=2, days=2, rebuild=lambda source: source.at_time(stamp))
    assert plain.status("m") == rebuilt.status("m") == "leak"
    assert all(check.probe.perturbed.get("actual_daily_max") for check in rebuilt.checks)


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


def test_explain_shows_a_daily_row_at_its_stamp(year, capsys):
    _, daily = year
    names = ["price_d-1", "load_actual_h-1", "actual_daily_max", "actual_daily_max_d-1"]
    pipeline = ModelPipeline(information_set=info()).add_model(OLSModel(names, training_window=WINDOW, name="m"))
    out = pipeline.explain(daily, model="m", test=("2023-10-27", "2023-10-31"), freq="1D")
    assert "period 0 (12:00)" in capsys.readouterr().out
    table = out["predictors"].set_index("predictor")
    assert table.loc["price_d-1", "target-relative"] == "T-2 12:00"
    assert table.loc["load_actual_h-1", "target-relative"] == "T-1 11:00"
    assert table["known"].to_dict() == {
        "price_d-1": "D-2 13:00",
        "load_actual_h-1": "D 13:00",
        "actual_daily_max": "D+1 01:00",
        "actual_daily_max_d-1": "D 01:00",
    }
    assert table["status"].to_dict() == {
        "price_d-1": "ok",
        "load_actual_h-1": "late",
        "actual_daily_max": "late",
        "actual_daily_max_d-1": "ok",
    }
    with pytest.raises(ValueError, match="0-0 with freq='1D'"):
        pipeline.explain(daily, model="m", hour=12, freq="1D", echo=False)
