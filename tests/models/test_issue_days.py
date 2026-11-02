"""`issue_days=`: a run that issues forecasts on the weekdays it lists, each what a run of every day gives it.

Training is untouched and masks are read on a window's first calendar day, so a Friday run's records are held key by key against
the Fridays of a run of every day, which plans other days and shares none of the weekday filtering. Counts are written down from
the calendar and the clock by hand. A column that turns from 0/1 to continuous on the first day of the test makes the day a mask
is read on visible in the forecasts, and the probes are held against a column published weekly, which is late on a Monday and
not on a Friday, and against a control that can't reach a Friday's list.
"""

import datetime
import json
import warnings
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from epftoolbox2 import cli
from epftoolbox2.ensembles import AverageEnsemble
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.information_set import Availability, InformationSet, InformationSetError
from epftoolbox2.models import OLSModel, Use
from epftoolbox2.models._calendar import IssueCalendar, resolve_issue_days
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline, Workflow
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path
from epftoolbox2.results.windows import RunWindows
from tests.synthetic import PREDICTORS, calendar_shift, make_market_data

SWITCH_DAYS = {
    "spring": "2023-03-26",
    "autumn": "2023-10-29",
}  # Sundays, so the Friday before aims at them at horizon 2
WINDOW = 40
FRIDAY = 4
WEEKDAYS = ["Monday", "TUESDAY", "wednesday", "Thursday", "friday", "saturday", "sunday"]
RULES = {
    "load": Availability.published("10:00", 10),
    "wind": Availability.published("10:00", 10),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
}
CALENDAR = ["is_weekend_d+{horizon}", "is_weekday_d+{horizon}"]
TIMESTAMPS = ("created_at", "updated_at")


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


def weekday_of(day):
    return datetime.date.fromisoformat(day).weekday()


def market(switch, freq="1h", horizon=3, seed=3):
    """The 70 days before a switch and 25 after, with drivers out to `horizon` days."""
    return make_market_data(
        start=day_of(SWITCH_DAYS[switch], -70), days=95, freq=freq, horizon=horizon, noise=5.0, seed=seed
    )


def by_key(records):
    """Records by forecast key, each as its JSON text, so that a NaN `actual` equals itself."""
    return {(r["run_date"], r["hour"], r["horizon"]): json.dumps(r, sort_keys=True) for r in records}


def fridays(keyed):
    return {key: record for key, record in keyed.items() if json.loads(record)["run_weekday"] == FRIDAY}


def without_list(keyed):
    """A predictor function's records name their list; a fixed list's have no such key."""
    return {key: {k: v for k, v in json.loads(r).items() if k != "predictor_list"} for key, r in keyed.items()}


def weekday_lists(ctx):
    """Another list on weekdays, so that a list's mask and its first test day matter."""
    return PREDICTORS + (["price_d-1"] if ctx.issue_day.weekday() < 5 else [])


def weekend_matching(ctx):
    """Forecasts of a weekend train on the days whose forecast was of a weekend too: Friday's training days are other weekdays."""
    return Use(PREDICTORS, train_on="matching") if ctx.target_day.weekday() >= 5 else PREDICTORS


LISTS = {"fixed": PREDICTORS + ["price_d-1"], "function": weekday_lists, "matching": weekend_matching}


# ---------------------------------------------------------------------------
# What issue_days= takes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (None, None),
        (["friday"], (4,)),
        (["FRIDAY"], (4,)),
        (("Thursday", "monday"), (0, 3)),
        (iter(["sunday", "saturday"]), (5, 6)),
        (WEEKDAYS, None),
    ],
)
def test_issue_days_are_weekday_names_in_any_case(given, expected):
    assert resolve_issue_days(given) == expected


@pytest.mark.parametrize(
    ("given", "message"),
    [
        ([], "empty"),
        (["fri"], "weekday names"),
        ([4], "weekday names"),
        ([None], "weekday names"),
        (["friday", "Friday"], "more than once"),
        ("friday", "list of weekday names"),
        (4, "list of weekday names"),
    ],
)
def test_what_issue_days_refuses(given, message):
    with pytest.raises(ValueError, match=message):
        resolve_issue_days(given)


def test_a_window_without_a_listed_weekday_raises_at_every_entry_point(tmp_path):
    """2023-10-23 to 2023-10-26 is Monday to Thursday."""
    data = market("autumn")
    pipeline = ModelPipeline(information_set=InformationSet(rules=RULES)).add_model(
        OLSModel(PREDICTORS, training_window=WINDOW, name="m")
    )
    week = ("2023-10-23", "2023-10-26")
    pipeline.run(data, test=week, horizon=1, save_dir=tmp_path)
    for call in (
        lambda: pipeline.run(data, test=week, horizon=1, issue_days=["friday"]),
        lambda: pipeline.run(data, validation=week, test=("2023-10-27", "2023-11-03"), issue_days=["friday"]),
        lambda: pipeline.report(tmp_path, test=week, horizon=1, issue_days=["friday"]),
        lambda: pipeline.verify(data, *week, issue_days=["friday"]),
        lambda: pipeline.models[0].run(data, *week, horizon=1, issue_days=["friday"]),
    ):
        with pytest.raises(ValueError, match=r"has no friday"):
            call()


# ---------------------------------------------------------------------------
# Every day, however it is said, is a run without issue_days, byte for byte
# ---------------------------------------------------------------------------


def snapshot(directory):
    """Every file of a results directory: records by their sorted lines, manifests without their timestamps."""
    out = {}
    for path in sorted(Path(directory).iterdir()):
        if path.name.endswith(".manifest.json"):
            manifest = json.loads(path.read_text())
            out[path.name] = {key: value for key, value in manifest.items() if key not in TIMESTAMPS}
        else:
            out[path.name] = sorted(line for line in path.read_text().splitlines() if line)
    return out


def small_pipeline(info=None):
    pipeline = ModelPipeline(information_set=info or InformationSet(rules=RULES))
    pipeline.add_model(OLSModel(LISTS["fixed"], training_window=WINDOW, name="fixed"))
    pipeline.add_model(OLSModel(weekday_lists, training_window=WINDOW, name="function"))
    pipeline.add_ensemble(AverageEnsemble(["fixed", "function"], name="average"))
    return pipeline.add_evaluator(MAEEvaluator())


def test_all_seven_weekdays_are_a_run_without_issue_days(tmp_path):
    data = market("spring")
    windows = {"validation": ("2023-03-15", "2023-03-21"), "test": ("2023-03-22", "2023-03-28")}
    plain = small_pipeline().run(data, horizon=2, save_dir=tmp_path / "plain", **windows)
    seven = small_pipeline().run(data, horizon=2, save_dir=tmp_path / "seven", issue_days=WEEKDAYS, **windows)

    assert snapshot(tmp_path / "plain") == snapshot(tmp_path / "seven")
    assert "issue_days" not in json.loads(manifest_path(tmp_path / "seven" / "fixed.jsonl").read_text())
    pd.testing.assert_frame_equal(plain.summary(), seven.summary())
    pd.testing.assert_frame_equal(plain.forecast_counts(), seven.forecast_counts())
    assert all(ref.issue_days is None for ref in seven.refs.values())

    def workflow(**days):
        return Workflow(data_start="2023-01-01", data_end="2023-04-01", model_test=windows["test"], **days)

    assert workflow(model_issue_days=WEEKDAYS).to_dict() == workflow().to_dict()
    assert "issue_days" not in workflow().to_dict()["model_pipeline"]


# ---------------------------------------------------------------------------
# A Friday run is the Fridays of a run of every day
# ---------------------------------------------------------------------------

# Every level of each axis, and every pair of levels of two, as the tests rules ask: switch x freq x kind of list x windows. A
# matching list, horizons=[3, 9] and newest_target="published" are a cell each. Every test period starts on a Wednesday, and
# the Friday two days before each switch aims at it at horizon 2.
CELLS = [
    ("spring", "1h", "fixed", "plain", {}),
    ("spring", "15min", "function", "validation", {}),
    ("autumn", "1h", "function", "plain", {}),
    ("autumn", "15min", "fixed", "validation", {}),
    ("autumn", "1h", "fixed", "validation", {}),
    ("spring", "15min", "function", "plain", {}),
    ("autumn", "1h", "matching", "validation", {}),
    ("spring", "1h", "function", "plain", {"horizons": [3, 9]}),
    ("autumn", "1h", "fixed", "plain", {"newest_target": "published"}),
]


@pytest.mark.parametrize(("switch", "freq", "lists", "windows", "extra"), CELLS)
def test_a_friday_run_is_the_fridays_of_a_run_of_every_day(switch, freq, lists, windows, extra):
    S = SWITCH_DAYS[switch]
    horizons = extra.get("horizons")
    published = extra.get("newest_target") == "published"
    data = market(switch, freq=freq, horizon=9 if horizons else 3)
    info = None
    if published:  # The price is out at 13:00 the day before delivery and the issue is at 11:45: D's price trains.
        info = InformationSet("11:45", "Europe/Brussels", rules={**RULES, "price": Availability.published("13:00", 1)})
    test = (day_of(S, -11 if horizons else -4), day_of(S, 8))
    validation = (day_of(S, -18), day_of(S, -6)) if windows == "validation" else None
    run_windows = RunWindows.resolve(validation=validation, test=test)
    assert {weekday_of(start) for _, (start, _) in run_windows.periods} == {2}  # Wednesdays

    def run(**days):
        model = OLSModel(LISTS[lists], training_window=WINDOW, newest_target=extra.get("newest_target", "d-1"))
        kwargs = {"horizons": horizons} if horizons else {"horizon": 3}
        if windows == "validation":
            kwargs["windows"] = run_windows
        else:
            kwargs.update(test_start=test[0], test_end=test[1])
        return model.run(data, freq=freq, information_set=info, **kwargs, **days)

    every = run()
    friday = run(issue_days=["Friday"])
    expected = fridays(by_key(every._results))
    assert by_key(friday._results) == expected
    assert friday.issue_days == (FRIDAY,) and every.issue_days is None
    issued = {key[0] for key in expected}
    assert {weekday_of(day) for day in issued} == {FRIDAY}
    assert len(issued) == (4 if windows == "validation" else 3 if horizons else 2)
    aimed = {(key[0], key[2]) for key in expected if day_of(key[0], key[2]) == S}
    assert aimed == ({(day_of(S, -9), 9)} if horizons else {(day_of(S, -2), 2)})
    if windows == "validation":
        assert set(friday.by_window) == {"validation", "test"}
        days = {r["run_date"]: r["day_in_test"] for r in friday._results}
        assert days[day_of(S, -2)] == 2 and days[day_of(S, -16)] == 2  # calendar days from the window's Wednesday
    if published:
        assert {json.loads(r)["newest_target_day"] for key, r in expected.items() if key[2] == 2} == issued


# ---------------------------------------------------------------------------
# Masks: read on the window's first calendar day when the first issue day uses the list
# ---------------------------------------------------------------------------

SPRING_WEDNESDAY = (
    "2023-03-22"  # the first day of the test below; its first Friday is 2023-03-24, its second 2023-03-31
)
# The drivers alone: with the two weekday dummies, which a model leaves raw and which sum to 1, OLS spans a constant and gives a
# raw and a centred `regime` the same forecasts, so no mask could be seen.
DRIVERS = ["load_d+{horizon}", "wind_d+{horizon}"]
MASKED = DRIVERS + ["regime"]


def with_regime(data, first_continuous=SPRING_WEDNESDAY):
    """`regime`: the weekend flag, 0 or 1, before `first_continuous` and a continuous value from then on.

    A mask is read from training rows dated up to D - 1 - h, so at h=1 the column is binary on the Wednesday and continuous on
    the Friday after it: it is left raw or standardised by which day the mask is read on, and the forecasts move with that.
    """
    dates = pd.DatetimeIndex(data.index.tz_localize(None)).normalize()
    later = np.asarray(dates >= pd.Timestamp(first_continuous))
    regime = data["is_weekend"].to_numpy(dtype=float).copy()
    regime[later] += np.linspace(0.1, 0.9, int(later.sum()))
    return data.assign(regime=regime)


def on_fridays(ctx):
    """The fixed list on Fridays, something else on other days: in a Friday run it is the list of every issue day."""
    return MASKED if ctx.issue_day.weekday() == FRIDAY else DRIVERS


def from_the_second_friday(ctx):
    """A list that starts on 2023-03-31, the second issue day of a Friday run and the tenth calendar day of the test."""
    return MASKED if ctx.issue_day >= datetime.date(2023, 3, 31) else DRIVERS


TESTS = {
    "plain": {"test_start": SPRING_WEDNESDAY, "test_end": "2023-04-07"},
    "validation": {
        "windows": RunWindows.resolve(validation=(SPRING_WEDNESDAY, "2023-03-28"), test=("2023-03-29", "2023-04-07"))
    },
}


@pytest.mark.parametrize("windows", ["plain", "validation"])
def test_a_mask_is_read_on_the_first_calendar_day_of_its_window(windows):
    data = with_regime(market("spring"))

    def run(lists, **kwargs):
        return by_key(
            OLSModel(lists, training_window=WINDOW).run(data, horizon=3, **{**TESTS[windows], **kwargs})._results
        )

    every_day = run(MASKED)
    on_friday = run(MASKED, issue_days=["friday"])
    assert on_friday == fridays(every_day)

    # A Friday-only function is the fixed list of a Friday run, its mask included.
    assert without_list(run(on_fridays, issue_days=["friday"])) == without_list(on_friday)

    # A list from the second Friday on is not used on the first issue day: its mask is read on that Friday, as in a run of every
    # day.
    assert run(from_the_second_friday, issue_days=["friday"]) == fridays(run(from_the_second_friday))

    # What makes these tests see a mask: read on the first Friday, the same forecasts move at horizon 1.
    if windows == "plain":
        first_friday = run(MASKED, issue_days=["friday"], test_start="2023-03-24")
        moved = {
            key[2]
            for key, record in first_friday.items()
            if abs(json.loads(record)["prediction"] - json.loads(on_friday[key])["prediction"]) > 1e-6
        }
        assert moved == {1}


def until_april(ctx):
    """The list on the Fridays before 7 April."""
    return MASKED if ctx.issue_day.weekday() == FRIDAY and ctx.issue_day < datetime.date(2023, 4, 7) else DRIVERS


def while_aux(ctx):
    """The list while `aux` is out: whether it is, on 7 April, says nothing about 24 March."""
    return MASKED if ctx.published("aux") else DRIVERS


AUX_RULES = {**RULES, "regime": Availability.static(), "aux": Availability.static()}


def test_a_later_end_of_the_test_moves_no_earlier_forecast():
    """A mask read on the first issue day that uses the list, when the list missed a later Friday, made the forecasts of 24
    March depend on whether the test went on to 7 April (24 of 24 moved, by up to 0.41 with LassoCV)."""
    data = with_regime(market("spring"))

    def run(end):
        model = OLSModel(until_april, training_window=WINDOW)
        return by_key(model.run(data, SPRING_WEDNESDAY, end, horizon=3, issue_days=["friday"])._results)

    shorter, longer = run("2023-03-31"), run("2023-04-07")
    assert {key[0] for key in shorter} == {"2023-03-24", "2023-03-31"}
    assert {key: longer[key] for key in shorter} == shorter


def test_a_later_friday_s_data_moves_no_earlier_forecast():
    """`aux` missing on 7 April changes the list of that Friday alone, and no forecast of an earlier one."""
    data = with_regime(market("spring")).assign(aux=1.0)
    gone = data.copy()
    gone.loc[np.asarray(pd.DatetimeIndex(data.index.tz_localize(None)).normalize() == "2023-04-07"), "aux"] = np.nan
    info = InformationSet(rules=AUX_RULES)

    def run(frame):
        model = OLSModel(while_aux, training_window=WINDOW)
        test = {"test_start": SPRING_WEDNESDAY, "test_end": "2023-04-07", "horizon": 3, "issue_days": ["friday"]}
        return by_key(model.run(frame, information_set=info, **test)._results)

    every_friday, not_the_last = run(data), run(gone)
    earlier = {key: record for key, record in every_friday.items() if key[0] < "2023-04-07"}
    assert {key: not_the_last[key] for key in earlier} == earlier
    assert {json.loads(r)["predictor_list"] for key, r in not_the_last.items() if key[0] == "2023-04-07"} != {
        json.loads(r)["predictor_list"] for key, r in earlier.items()
    }


def test_data_after_a_friday_moves_none_of_its_forecasts_through_a_column_with_no_value():
    """A list used on the first Friday reads its mask on Wednesday 22 March, where `late` has no value yet (its rows go up to 20
    March at h=1). That column is read again from the first Friday on, the Friday itself included: read from the second, the
    forecasts of 24 March would take it from rows dated after their issue day."""
    data = market("spring")
    dates = np.asarray(pd.DatetimeIndex(data.index.tz_localize(None)).normalize())
    late = np.full(len(data), np.nan)
    late[dates == np.datetime64("2023-03-21")] = 0.0
    late[dates >= np.datetime64("2023-03-22")] = 1.0
    late[dates >= np.datetime64("2023-03-23")] = data["is_weekend"].to_numpy()[dates >= np.datetime64("2023-03-23")]
    after = dates >= np.datetime64("2023-03-25")
    continuous = late.copy()
    continuous[after] += np.linspace(0.1, 0.9, int(after.sum()))

    def until_7_april(ctx):
        return DRIVERS + ["late"] if ctx.issue_day < datetime.date(2023, 4, 7) else DRIVERS

    def run(values):
        model = OLSModel(until_7_april, training_window=WINDOW)
        test = {"horizon": 1, "issue_days": ["friday"]}
        return by_key(model.run(data.assign(late=values), SPRING_WEDNESDAY, "2023-04-07", **test)._results)

    binary, later_continuous = run(late), run(continuous)
    first_friday = {key: record for key, record in binary.items() if key[0] == "2023-03-24"}
    assert len(first_friday) == 24
    assert {key: later_continuous[key] for key in first_friday} == first_friday


def on_wednesdays_and_fridays(ctx):
    return MASKED if ctx.issue_day.weekday() in (2, FRIDAY) else DRIVERS


def from_thursday_30_march(ctx):
    return MASKED if ctx.issue_day >= datetime.date(2023, 3, 30) else DRIVERS


# Where a Friday run and a run of every day read a list's mask on different days: (function, test start, the day the mask is read in
# the run of every day, in the Friday run, and the first day `regime` is continuous, so that at h=1 it is binary in the rows of one
# and not of the other). Each mask day is the forecast's issue day or earlier, so neither reads past D-1-h.
DIFFERING = {
    "fridays alone": (on_fridays, SPRING_WEDNESDAY, "2023-03-24", "2023-03-22", "2023-03-22"),
    "wednesdays and fridays from a monday": (
        on_wednesdays_and_fridays,
        "2023-03-20",
        "2023-03-22",
        "2023-03-20",
        "2023-03-20",
    ),
    "from a thursday": (from_thursday_30_march, SPRING_WEDNESDAY, "2023-03-30", "2023-03-31", "2023-03-29"),
}


@pytest.mark.parametrize("case", list(DIFFERING))
def test_where_a_friday_record_and_a_run_of_every_day_s_differ(case):
    """A list used on the Friday run's first issue day reads its mask on the window's first day, and in a run of every day on the
    first day that uses it: a Friday-only list on the first Friday, a Wednesday-and-Friday list on the first Wednesday. A list
    first used on a later Friday reads it there, and in a run of every day on the day before if that one uses it. Where a column is
    binary on one of those days and not on the other, the forecasts of that list differ; with a column binary throughout they
    are equal."""
    fn, start, _, _, turns = DIFFERING[case]
    test = {"test_start": start, "test_end": "2023-03-31", "horizon": 1}
    base = market("spring")

    def both(data):
        every = fridays(by_key(OLSModel(fn, training_window=WINDOW).run(data, **test)._results))
        friday = by_key(OLSModel(fn, training_window=WINDOW).run(data, issue_days=["friday"], **test)._results)
        assert set(every) == set(friday)
        return {key for key in friday if json.loads(friday[key])["prediction"] != json.loads(every[key])["prediction"]}

    assert both(base.assign(regime=base["is_weekend"])) == set()
    differ = both(with_regime(base, turns))
    # The keys of that list alone: from 31 March for a list from a Thursday, every Friday for the others.
    assert {key[0] for key in differ} == ({"2023-03-31"} if case == "from a thursday" else {"2023-03-24", "2023-03-31"})


# ---------------------------------------------------------------------------
# What is planned: by hand
# ---------------------------------------------------------------------------

# Fridays from 2023-03-01 to 2023-04-15: 3, 10, 17, 24 and 31 March, 7 and 14 April. Sundays: 5, 12, 19 and 26 March, 2 and 9 April.
MARCH = ("2023-03-01", "2023-04-15")
FRIDAYS, SUNDAYS = 7, 6


@pytest.mark.parametrize(
    ("freq", "weekdays", "expected"),
    [
        # Friday 24 March aims at the spring switch at horizon 2, which skips 02:00, or four quarter-hours.
        ("1h", ["friday"], FRIDAYS * 3 * 24 - 1),
        ("15min", ["friday"], FRIDAYS * 3 * 96 - 4),
        # Sunday 26 March is the switch itself, at all three horizons.
        ("1h", ["friday", "sunday"], (FRIDAYS + SUNDAYS) * 3 * 24 - 1 - 3),
        (
            "1h",
            None,
            46 * 3 * 24 - 3 - 3,
        ),  # every day: the switch is the target of 25 March (h=1) and 24 and 23 March too
    ],
)
def test_planned_keys_by_hand(freq, weekdays, expected):
    calendar = IssueCalendar(freq, "Europe/Warsaw", (1, 2, 3), resolve_issue_days(weekdays))
    assert calendar.count(*MARCH) == expected
    if weekdays == ["friday"]:
        assert [day.isoformat() for day in calendar.issue_days(*MARCH)] == [
            "2023-03-03", "2023-03-10", "2023-03-17", "2023-03-24", "2023-03-31", "2023-04-07", "2023-04-14",
        ]  # fmt: skip
    # The autumn switch repeats an hour and skips none: Friday 27 October aims at it and loses nothing. Six Fridays, 6 October
    # to 10 November.
    autumn = IssueCalendar(freq, "Europe/Warsaw", (1, 2, 3), (FRIDAY,))
    assert autumn.count("2023-10-01", "2023-11-15") == 6 * 3 * (24 if freq == "1h" else 96)


@pytest.fixture(scope="module")
def spring_data():
    return make_market_data(start="2023-01-01", days=110, horizon=3, noise=5.0, seed=5)


def test_a_run_and_its_report_plan_the_listed_weekdays_and_miss_nothing(tmp_path, spring_data):
    planned = (FRIDAYS + SUNDAYS) * 3 * 24 - 4
    pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="m"))
    report = pipeline.run(spring_data, test=MARCH, horizon=3, issue_days=["sunday", "Friday"], save_dir=tmp_path)
    ref = report.refs["m"]
    assert (ref.count, ref.produced, ref.skipped) == (planned, planned, {})
    lines = (tmp_path / "m.jsonl").read_text().splitlines()
    assert len(lines) == planned and {json.loads(line)["run_weekday"] for line in lines} == {4, 6}
    assert json.loads(manifest_path(tmp_path / "m.jsonl").read_text())["issue_days"] == ["friday", "sunday"]

    # A report reads the weekdays from the manifest, or plans the ones it is told: Fridays alone are the records of Fridays.
    counts = (
        ModelPipeline()
        .add_model(OLSModel(PREDICTORS, name="m"))
        .report(tmp_path, test=MARCH, horizon=3)
        .forecast_counts()
    )
    assert counts[["planned", "produced", "missing"]].values.tolist() == [[planned, planned, 0]]
    only = ModelPipeline().add_model(OLSModel(PREDICTORS, name="m"))
    only = only.report(tmp_path, test=MARCH, horizon=3, issue_days=["friday"])
    assert only.forecast_counts()[["planned", "produced", "missing"]].values.tolist() == [[FRIDAYS * 72 - 1] * 2 + [0]]
    assert set(pd.to_datetime(only.predictions()["run_date"]).dt.weekday) == {FRIDAY}


def test_a_report_of_the_fridays_of_a_run_of_every_day_reads_only_them(tmp_path, spring_data):
    every = small_pipeline().run(spring_data, test=MARCH, horizon=3, save_dir=tmp_path / "every")
    small_pipeline().run(spring_data, test=MARCH, horizon=3, issue_days=["friday"], save_dir=tmp_path / "friday")
    told = small_pipeline().report(tmp_path / "every", test=MARCH, horizon=3, issue_days=["friday"])
    on_file = small_pipeline().report(tmp_path / "friday", test=MARCH, horizon=3)
    pd.testing.assert_frame_equal(told.summary(), on_file.summary())
    pd.testing.assert_frame_equal(told.forecast_counts(), on_file.forecast_counts())
    assert told.forecast_counts()["planned"].tolist() == [FRIDAYS * 72 - 1] * 3
    assert every.forecast_counts()["planned"].tolist() == [46 * 72 - 6] * 3


def test_models_run_on_other_weekdays_are_not_reported_or_averaged_together(tmp_path, spring_data):
    one = ModelPipeline().add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="a"))
    one.run(spring_data, test=MARCH, horizon=1, issue_days=["friday"], save_dir=tmp_path)
    other = ModelPipeline().add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="b"))
    other.run(spring_data, test=MARCH, horizon=1, save_dir=tmp_path)
    both = ModelPipeline()
    both.add_model(OLSModel(PREDICTORS, name="a")).add_model(OLSModel(PREDICTORS, name="b"))
    with pytest.raises(ValueError, match=r"different weekdays.*a: \['friday'\]; b: every day"):
        both.report(tmp_path, test=MARCH, horizon=1)
    # Told the Fridays, both report them; an average of the two is still of members run on other days.
    assert (
        both.report(tmp_path, test=MARCH, horizon=1, issue_days=["friday"]).forecast_counts()["planned"].tolist()
        == [FRIDAYS * 24] * 2
    )
    both.add_ensemble(AverageEnsemble(["a", "b"], name="e"))
    with pytest.raises(ManifestMismatchError, match=r"issue_days \{'a': \['friday'\], 'b': every day\}"):
        both.report(tmp_path, test=MARCH, horizon=1, issue_days=["friday"])


@pytest.mark.parametrize("other", [{}, {"issue_days": ["monday"]}, {"issue_days": ["friday", "monday"]}])
def test_a_resume_with_other_issue_days_is_refused(tmp_path, spring_data, other):
    small_pipeline().run(spring_data, test=MARCH, horizon=1, issue_days=["friday"], save_dir=tmp_path)
    with pytest.raises(ManifestMismatchError, match="issue_days"):
        small_pipeline().run(spring_data, test=MARCH, horizon=1, save_dir=tmp_path, **other)


def test_a_friday_run_resumes_to_what_a_fresh_one_gives(tmp_path, spring_data):
    windows = {"validation": ("2023-03-01", "2023-03-14"), "test": ("2023-03-15", "2023-04-15")}
    small_pipeline().run(spring_data, horizon=2, issue_days=["friday"], save_dir=tmp_path / "a", **windows)
    for name, kept in (("fixed", 100), ("function", 151)):
        path = tmp_path / "a" / f"{name}.jsonl"
        path.write_text("".join(path.read_text().splitlines(keepends=True)[:kept]))
    small_pipeline().run(spring_data, horizon=2, issue_days=["friday"], save_dir=tmp_path / "a", **windows)
    small_pipeline().run(spring_data, horizon=2, issue_days=["friday"], save_dir=tmp_path / "fresh", **windows)
    assert snapshot(tmp_path / "a") == snapshot(tmp_path / "fresh")


# ---------------------------------------------------------------------------
# validation="previous_year" on Fridays
# ---------------------------------------------------------------------------


def test_previous_year_on_fridays(tmp_path):
    """The test period starts on Sunday 2023-03-19, so the year before ends on Saturday the 18th: its last Friday, the 17th,
    aims at the test at horizon 3. Fridays of that year: 25 March 2022 to 17 March 2023, 52 of them."""
    data = make_market_data(start="2022-01-20", days=450, horizon=3, noise=5.0, seed=8)
    test = ("2023-03-19", "2023-04-07")  # Fridays 24 and 31 March, 7 April

    def pipeline():
        p = ModelPipeline()
        p.add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="a"))
        p.add_model(OLSModel(PREDICTORS + ["price_d-1"], training_window=WINDOW, name="b"))
        return p.add_ensemble(AverageEnsemble(["a", "b"], top=1, name="best")).add_evaluator(MAEEvaluator())

    report = pipeline().run(
        data, validation="previous_year", test=test, horizons=[3], issue_days=["friday"], save_dir=tmp_path
    )
    assert report.windows.mode == "previous_year"
    assert report.windows.validation == ("2022-03-19", "2023-03-18")
    counts = report.validation.forecast_counts().set_index("model")
    assert (
        counts.loc[["a", "b"], ["planned", "produced", "target_in_test"]].values.tolist()
        == [[52 * 24, 51 * 24, 24]] * 2
    )
    assert report.forecast_counts().set_index("model")["planned"].to_dict() == {"a": 72, "b": 72, "best": 72}
    choices = report.choices()
    assert choices["ensemble"].tolist() == ["best"] and choices["validation_forecasts"].tolist() == [51 * 24]

    # The windows are each what a Friday run over it alone gives, as for a run of every day.
    alone = OLSModel(PREDICTORS, training_window=WINDOW, name="a").run(data, *test, horizons=[3], issue_days=["friday"])
    stored = [json.loads(line) for line in (tmp_path / "a.jsonl").read_text().splitlines()]
    in_test = by_key({k: v for k, v in r.items() if k != "window"} for r in stored if r.get("window") == "test")
    assert in_test == by_key(alone._results)
    assert {r["day_in_test"] for r in stored if r["run_date"] == "2022-03-25"} == {6}


def test_a_choice_on_few_fridays_warns_however_many_days_they_are_for(tmp_path):
    """Four Fridays of March at horizons 1 to 7 are for 28 target days, 4 to 31 March, so the target days don't warn: the issue
    days do, as the choice rests on four issues."""
    data = make_market_data(start="2023-01-01", days=110, horizon=7, noise=5.0, seed=5)
    pipeline = ModelPipeline()
    pipeline.add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="a"))
    pipeline.add_model(OLSModel(PREDICTORS + ["price_d-1"], training_window=WINDOW, name="b"))
    pipeline.add_ensemble(AverageEnsemble(["a", "b"], top=1, name="best"))
    windows = {"validation": ("2023-03-01", "2023-03-28"), "test": ("2023-04-04", "2023-04-10")}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = pipeline.run(data, horizon=7, issue_days=["friday"], save_dir=tmp_path, **windows)
    messages = [str(w.message) for w in caught]
    assert not [m for m in messages if "target days" in m]
    assert [m for m in messages if "best:" in m and "issued on 4 days (" in m]
    assert report.choices()["validation_forecasts"].tolist() == [4 * 7 * 24 - 1]  # 02:00 of 26 March, the switch


# ---------------------------------------------------------------------------
# The information set: what is late on a Monday isn't on a Friday
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def weekly():
    """`tso`, a load forecast published on Fridays at 10:00 for the next Monday-to-Sunday week: `tso_d+9` is out by a Friday's
    issue (it is next Sunday's) and not by a Monday's (next Wednesday's, out on Friday)."""
    data = make_market_data(start="2023-01-01", days=130, horizon=9, noise=5.0, seed=6)
    tso = data["load"] + np.random.default_rng(1).normal(0, 20, len(data))
    return data.assign(**{"tso_d+9": calendar_shift(tso.rename("tso"), 9)})


WEEKLY = {**RULES, "tso": Availability.weekly("friday", "10:00", "Europe/Warsaw")}
WITH_TSO = PREDICTORS + ["tso_d+9"]


def a31_on_fridays(ctx):
    return WITH_TSO if ctx.issue_day.weekday() == FRIDAY else PREDICTORS


@pytest.mark.parametrize("lists", [WITH_TSO, a31_on_fridays])
def test_a_weekly_input_is_known_on_fridays_and_late_on_mondays(weekly, lists):
    test = ("2023-04-03", "2023-04-16")
    pipeline = ModelPipeline(information_set=InformationSet(rules=WEEKLY), availability="raise")
    pipeline.add_model(OLSModel(lists, training_window=WINDOW, name="m"))
    pipeline.run(weekly, test=test, horizons=[9], issue_days=["friday"])
    if lists is WITH_TSO:
        with pytest.raises(InformationSetError, match="tso_d\\+9"):
            pipeline.run(weekly, test=test, horizons=[9])
        with pytest.raises(InformationSetError, match="tso_d\\+9"):
            pipeline.run(weekly, test=test, horizons=[9], issue_days=["friday", "monday"])

    # At "warn" each window lists what its own issue days read late: nothing on Fridays, in either window.
    listing = ModelPipeline(information_set=InformationSet(rules=WEEKLY), availability="warn")
    listing.add_model(OLSModel(lists, training_window=WINDOW, name="m"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = listing.run(
            weekly, validation=("2023-03-13", "2023-03-24"), test=test, horizons=[9], issue_days=["friday"]
        )
    assert report.availability().empty and report.validation.availability().empty
    if lists is WITH_TSO:
        with pytest.warns(UserWarning, match="tso_d\\+9 is late"):
            every_day = listing.run(weekly, validation=("2023-03-13", "2023-03-24"), test=test, horizons=[9])
        assert set(every_day.validation.availability()["predictor"]) == {"tso_d+9"}


VERIFY = ("2023-04-03", "2023-04-30")


def verified(model, data, rules=WEEKLY, **kwargs):
    pipeline = ModelPipeline(information_set=InformationSet(rules=rules)).add_model(model)
    return pipeline.verify(data, *VERIFY, horizons=[9], **kwargs)


def test_verify_samples_the_listed_weekdays_alone(weekly):
    model = OLSModel(WITH_TSO + ["price_d-1"], training_window=WINDOW, name="m")
    on_fridays_ = verified(model, weekly, days=3, issue_days=["friday"])
    assert [weekday_of(day) for day in on_fridays_.issue_days] == [FRIDAY] * 3
    on_fridays_.assert_ok()
    on_mondays = verified(model, weekly, days=3, issue_days=["monday"])
    assert [weekday_of(day) for day in on_mondays.issue_days] == [0] * 3
    assert on_mondays.status("m") == "leak"
    # Fewer Fridays than asked for: all of them.
    assert verified(model, weekly, days=9, issue_days=["friday"]).issue_days == [
        "2023-04-07", "2023-04-14", "2023-04-21", "2023-04-28",
    ]  # fmt: skip


def test_verify_plans_a_function_on_the_listed_weekdays_alone(weekly):
    """A function that reads `tso_d+9` on Mondays, where it is late, uses nothing late in a Friday run: verify plans the
    whole period on Fridays, so it is no leak there, and a leak of a run of every day whichever days are sampled."""
    model = OLSModel(
        lambda ctx: (WITH_TSO if ctx.issue_day.weekday() == 0 else PREDICTORS) + ["price_d-1"],
        training_window=WINDOW,
        name="m",
    )
    verified(model, weekly, days=2, issue_days=["friday"]).assert_ok()
    every_day = verified(model, weekly, days=2, issue_days=["friday", "saturday"])
    assert every_day.status("m") == "ok"  # no Monday in the plan either
    assert verified(model, weekly, days=2, issue_days=["friday", "monday"]).status("m") == "leak"


def test_the_control_can_fail_on_fridays(weekly):
    """A control that can't fail is no control: on Fridays this list is the calendar alone, which no recent value moves."""
    model = OLSModel(
        lambda ctx: CALENDAR if ctx.issue_day.weekday() == FRIDAY else PREDICTORS + ["price_d-1"],
        training_window=WINDOW,
        name="m",
    )
    report = verified(model, weekly, days=2, issue_days=["friday"])
    assert report.status("m") == "inconclusive"
    assert any("moved no forecast" in line for line in report.problems())
    verified(model, weekly, days=2, issue_days=["monday"]).assert_ok()


# ---------------------------------------------------------------------------
# YAML and the command line
# ---------------------------------------------------------------------------


def test_a_workflow_keeps_its_issue_days_through_yaml(tmp_path):
    workflow = Workflow(
        data_start="2023-01-01", data_end="2023-04-01", model_test=MARCH, model_issue_days=["Monday", "FRIDAY"]
    )
    workflow.save(tmp_path / "experiment.yaml")
    written = yaml.safe_load((tmp_path / "experiment.yaml").read_text())["model_pipeline"]
    assert written["issue_days"] == ["monday", "friday"]
    loaded = Workflow.load(tmp_path / "experiment.yaml")
    assert loaded.model_issue_days == ["monday", "friday"]
    assert loaded.to_dict() == workflow.to_dict()
    with pytest.raises(ValueError, match="weekday names"):
        Workflow(data_start="2023-01-01", data_end="2023-04-01", model_issue_days=["fri"])


def test_a_workflow_runs_and_reports_its_issue_days(tmp_path):
    index = pd.date_range("2023-01-01", "2023-04-01", freq="1h", tz="UTC", inclusive="left")
    prices = 100 + 20 * pd.Series(range(len(index))).mod(24).to_numpy() / 24.0
    pd.DataFrame({"datetime": index.strftime("%Y-%m-%d %H:%M:%S"), "price": prices}).to_csv(
        tmp_path / "prices.csv", index=False
    )
    config = {
        "data_pipeline": {
            "start": "2023-01-01",
            "end": "2023-03-31",
            "sources": [{"class": "CsvSource", "params": {"file_path": str(tmp_path / "prices.csv")}}],
            "transformers": [
                {"class": "TimezoneTransformer", "params": {"target_tz": "Europe/Warsaw"}},
                {"class": "LagTransformer", "params": {"columns": ["price"], "lags": [1, 7], "freq": "day"}},
            ],
        },
        "model_pipeline": {
            "test": ["2023-03-01", "2023-03-14"],
            "horizon": 2,
            "issue_days": ["friday"],
            "save_dir": "results",
            "models": [
                {"class": "OLSModel", "params": {"predictors": ["price_d-1", "price_d-7"], "training_window": 30}}
            ],
        },
    }
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(config))
    Workflow.load(tmp_path / "experiment.yaml").run()
    lines = (tmp_path / "results" / "model.jsonl").read_text().splitlines()
    assert sorted({json.loads(line)["run_date"] for line in lines}) == ["2023-03-03", "2023-03-10"]
    counts = Workflow.load(tmp_path / "experiment.yaml").report().forecast_counts()
    assert counts[["planned", "produced"]].values.tolist() == [[2 * 2 * 24, 2 * 2 * 24]]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("friday", ["friday"]), ("Friday,MONDAY", ["monday", "friday"]), (",".join(WEEKDAYS), None)],
)
@pytest.mark.parametrize("command", ["model", "verify"])
def test_the_command_line_reads_weekday_names(command, text, expected):
    window = ["--test-start", "2023-03-01", "--test-end", "2023-03-31"] if command == "verify" else []
    args = cli.build_parser().parse_args([command, "mp.yaml", "--data", "d.csv", *window, "--issue-days", text])
    assert args.issue_days == expected


@pytest.mark.parametrize("text", ["fri", "friday,friday", "", "4"])
def test_the_command_line_refuses_what_isn_t_a_weekday(text):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["model", "mp.yaml", "--data", "d.csv", "--issue-days", text])


def test_epf_model_with_issue_days_gives_the_in_process_forecasts(tmp_path, spring_data):
    from epftoolbox2._frame_csv import save_frame_csv

    predictors = PREDICTORS + ["price_d-1"]
    ModelPipeline().add_model(OLSModel(predictors, training_window=WINDOW, name="OLS")).save(tmp_path / "mp.yaml")
    data = spring_data[["price", "load", "wind", "is_weekend", "is_weekday", "price_d-1"]].join(
        spring_data[[f"{name}_d+{h}" for h in (1, 2) for name in ("load", "wind", "is_weekend", "is_weekday")]]
    )
    save_frame_csv(data, tmp_path / "data.csv")
    argv = ["model", str(tmp_path / "mp.yaml"), "--data", str(tmp_path / "data.csv"), "--test-start", "2023-03-20"]
    argv += [
        "--test-end",
        "2023-04-02",
        "--horizon",
        "2",
        "--issue-days",
        "friday",
        "--save-dir",
        str(tmp_path / "out"),
    ]
    cli.cmd_model(cli.build_parser().parse_args(argv))

    stored = by_key(json.loads(line) for line in (tmp_path / "out" / "ols.jsonl").read_text().splitlines())
    in_process = by_key(
        OLSModel(predictors, training_window=WINDOW, name="OLS")
        .run(data, test_start="2023-03-20", test_end="2023-04-02", horizon=2, issue_days=["friday"])
        ._results
    )
    assert {key: json.loads(r)["prediction"] for key, r in stored.items()} == {
        key: json.loads(r)["prediction"] for key, r in in_process.items()
    }
    assert {key[0] for key in stored} == {"2023-03-24", "2023-03-31"}
