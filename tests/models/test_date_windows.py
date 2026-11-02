"""Rows chosen by date and forecasts planned by the clock (`epftoolbox2.models._calendar`).

The first tests were xfails of `tests/test_known_issues.py`. The rest hold the semantics against what doesn't share code with them:
a copy of the row cut the dates replaced, compared key by key with a table computed from the rule; forecasts read from the frame
with numpy; and counts by hand. A forecast that moves on a day the table doesn't name fails here, and so does one that should move
and doesn't.
"""

import datetime
import json
import warnings
from collections import Counter
from concurrent.futures import Future

import dateutil.tz
import numpy as np
import pandas as pd
import pytest

from epftoolbox2.ensembles import AverageEnsemble
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import OLSModel, Use, _contract, _worker
from epftoolbox2.models import base as base_module
from epftoolbox2.models._calendar import IssueCalendar, periods_per_day, training_span
from epftoolbox2.models._worker import SKIP_MISSING_TEST_ROW, _fit_one_numpy, _worker_init, issue_row, training_rows
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path, read_zone, rows_unverified
from epftoolbox2.results.report import EvaluationReport
from tests.synthetic import PREDICTORS, calendar_shift, make_market_data

# Both zones change their clocks on these dates of 2023, Warsaw at 02:00 and London at 01:00.
SWITCH_DAYS = {"spring": "2023-03-26", "autumn": "2023-10-29"}
SWITCH_HOUR = {"Europe/Warsaw": 2, "Europe/London": 1}
FREQS = {"1h": 1, "15min": 4}
WINDOW = 40
RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
}


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


def market(switch, zone="Europe/Warsaw", freq="1h", horizon=3, seed=3):
    """A frame with `WINDOW` days and more on both sides of the switch."""
    return make_market_data(
        start=day_of(SWITCH_DAYS[switch], -70), days=100, tz=zone, freq=freq, horizon=horizon, noise=5.0, seed=seed
    )


def switch_periods(zone, freq):
    per_hour = FREQS[freq]
    return range(SWITCH_HOUR[zone] * per_hour, (SWITCH_HOUR[zone] + 1) * per_hour)


def by_key(records):
    return {(r["run_date"], r["hour"], r["horizon"]): r for r in records}


def forecasts(data, day, horizon=3, freq="1h", last=None, predictors=PREDICTORS, window=WINDOW):
    return by_key(
        OLSModel(predictors, training_window=window)
        .run(data, test_start=day, test_end=last or day, horizon=horizon, freq=freq)
        ._results
    )


# ---------------------------------------------------------------------------
# What tests/test_known_issues.py pinned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("run_day", ["2023-10-30", "2023-10-31"])
def test_no_lookahead_in_days_after_fall_back_dst(run_day):
    df = make_market_data(start="2023-08-01", days=120, noise=5.0, seed=1)
    predictors = PREDICTORS + ["price_d-1"]
    scrambled = df.copy()
    future = pd.Series(df.index.date >= pd.Timestamp(run_day).date(), index=df.index)
    scrambled.loc[future, "price"] = 1e4 + np.arange(future.sum())
    scrambled["price_d-1"] = calendar_shift(scrambled["price"], -1)

    def predictions(data):
        ref = OLSModel(predictors, training_window=40).run(data, test_start=run_day, test_end=run_day, horizon=2)
        return {(r["hour"], r["horizon"]): r["prediction"] for r in ref._results}

    original, perturbed = predictions(df), predictions(scrambled)
    changed = [key for key in original if abs(original[key] - perturbed[key]) > 1e-9]
    assert changed == []


def test_no_lookahead_in_a_matching_set_holding_the_fall_back_day():
    """Monday and Tuesday targets have a list of their own, so a forecast issued on the Monday after the switch trains on the
    day of the switch, which has two rows at hour 2. The newest of them must not reach the target dated on the issue day."""
    df = make_market_data(start="2023-08-20", days=90, noise=5.0, seed=3, horizon=1)
    run_day = "2023-10-30"

    def mondays_and_tuesdays(ctx):
        return Use(PREDICTORS if ctx.target_day.weekday() in (0, 1) else PREDICTORS[:3], train_on="matching")

    def forecast(data):
        ref = OLSModel(mondays_and_tuesdays, training_window=45).run(
            data, test_start=run_day, test_end=run_day, horizon=1
        )
        return {r["hour"]: r["prediction"] for r in ref._results}[2]

    raised = df.copy()
    raised.loc[f"{run_day} 02:00", "price"] += 100.0

    assert abs(forecast(df) - forecast(raised)) <= 1e-9


def test_spring_dst_day_forecasts_carry_their_own_target():
    df = make_market_data(noise=5.0, seed=1)
    ref = OLSModel(PREDICTORS, training_window=40).run(df, test_start="2023-03-26", test_end="2023-03-26", horizon=2)

    produced = {(r["hour"], r["horizon"]) for r in ref._results}
    mislabelled = [
        (r["hour"], r["horizon"])
        for r in ref._results
        if r["actual"]
        != pytest.approx(df.loc[pd.Timestamp(f"{r['target_date']} {r['hour']:02d}:00", tz=df.index.tz), "price"])
    ]
    assert mislabelled == []
    assert len(produced) == ref.count


@pytest.mark.parametrize("run_day", ["2023-10-30", "2023-10-31"])
@pytest.mark.parametrize("target_as_input", [[], ["price"]], ids=["plain", "target as an input"])
def test_verify_passes_in_the_days_after_fall_back_dst(run_day, target_as_input):
    df = make_market_data(start="2023-08-01", days=120, noise=5.0, seed=1)
    pipeline = ModelPipeline(information_set=InformationSet(rules=RULES))
    pipeline.add_model(OLSModel(PREDICTORS + ["price_d-1"] + target_as_input, training_window=40))

    pipeline.verify(df, run_day, run_day, horizon=2, days=1).assert_ok()


# ---------------------------------------------------------------------------
# Differential: the row cut this replaced, against a table computed from the rule
# ---------------------------------------------------------------------------


def old_cut(days, day, window, hz):
    """The row cut, copied from before dates: the rows of [D - W - h, D], all but the last 1 + h."""
    return np.flatnonzero((days >= day - window - hz) & (days <= day))[: -(1 + hz)]


def by_old_cut(days, day, window, hz):
    """The row cut as the slice `training_rows` gives."""
    rows = old_cut(days, day, window, hz)
    return slice(int(rows[0]), int(rows[-1]) + 1) if len(rows) else slice(0, 0)


def table(switch, in_the_switch_hour, S, D, h):
    """What the rule of dates says of a key: (does it exist, the dates the row cut holds besides the rows of the rule, the dates it
    lacks of them, does that change what trains)."""
    if not in_the_switch_hour:
        return True, [], [], False
    if switch == "autumn":
        if S + 1 <= D <= S + h:
            return True, [D - h], [], True  # the first row after the cut was short by one, its target on D
        if D == S:
            return True, [S - h], [], False  # its target is the repeated hour, NaN, so the row drops
        return True, [], [], False
    if D == S or D + h == S:
        return False, [], [], False
    if S + 1 <= D <= S + h:
        return True, [], [D - 1 - h], D >= S + 2  # at S + 1 that row's target is the skipped hour, NaN
    return True, [], [], False


def rows_of(data, freq, horizon):
    model = OLSModel([], training_window=WINDOW)
    model._set_freq(freq)
    model._load_periods(data, ["price"], horizon, "price")
    return model


def price_frame(switch, zone, freq):
    start = pd.Timestamp(day_of(SWITCH_DAYS[switch], -60), tz=zone)
    index = pd.date_range(start, start + pd.Timedelta(days=90), freq=freq, inclusive="left")
    return pd.DataFrame({"price": np.random.default_rng(0).normal(50, 5, len(index))}, index=index)


@pytest.mark.parametrize("zone", list(SWITCH_HOUR))
@pytest.mark.parametrize("freq", list(FREQS))
@pytest.mark.parametrize("switch", list(SWITCH_DAYS))
@pytest.mark.parametrize("H", [1, 3, 7])
def test_rows_equal_the_row_cut_except_on_the_keys_the_table_names(zone, freq, switch, H):
    data = price_frame(switch, zone, freq)
    model = rows_of(data, freq, H)
    switch_date = datetime.date.fromisoformat(SWITCH_DAYS[switch])
    S = switch_date.toordinal() - model._first_ordinal
    calendar = IssueCalendar(freq, data.index.tz, range(1, H + 1))
    issue_days = range(S - H - 3, S + H + 4)
    first = datetime.date.fromordinal(issue_days[0] + model._first_ordinal)
    tasks = set(calendar.tasks(first, [D - issue_days[0] for D in issue_days]))
    in_hour = set(switch_periods(zone, freq))
    # The periods outside the hour have rows on the same dates, so one of them stands for all, keys included.
    plain = [period for period in range(periods_per_day(freq)) if period not in in_hour]
    read = [model._hour_days[period][model._hour_days[period] <= issue_days[-1]] for period in plain]
    assert all(np.array_equal(days, read[0]) for days in read)
    moved = 0
    for period in [*in_hour, plain[0]]:
        days = model._hour_days[period]
        for h in range(1, H + 1):
            finite = np.isfinite(model._hour_data[period][f"price_d+{h}"].to_numpy())
            for D in issue_days:
                exists, extra, lacking, moves = table(switch, period in in_hour, S, D, h)
                where = (period, h, D - S)
                assert ((period, h, D - issue_days[0]) in tasks) == exists, where
                if not exists:
                    continue
                new = np.arange(len(days))[training_rows(days, D, WINDOW, h)]
                old = old_cut(days, D, WINDOW, h)
                assert Counter(days[old]) - Counter(days[new]) == Counter(extra), where
                assert Counter(days[new]) - Counter(days[old]) == Counter(lacking), where
                # The dates of the rule, and only those, are in the new rows.
                assert set(days[new]) <= set(range(D - WINDOW - h, D - h)), where
                changed = old[finite[old]].tolist() != new[finite[new]].tolist()
                assert changed == moves, where
                moved += moves
                row = issue_row(days, D)
                assert row == np.flatnonzero(days == D)[-1], where
    # The size of the move: per switch, with these periods in the hour of the switch.
    assert moved == (H * (H + 1) // 2 if switch == "autumn" else H * (H - 1) // 2) * FREQS[freq]


@pytest.mark.parametrize("freq", list(FREQS))
@pytest.mark.parametrize("switch", list(SWITCH_DAYS))
def test_forecasts_equal_the_row_cut_bit_for_bit_except_on_those_keys(monkeypatch, freq, switch):
    H = 3
    data = market(switch, freq=freq, horizon=H)
    S = SWITCH_DAYS[switch]
    first, last = day_of(S, 1), day_of(S, 3)
    after = forecasts(data, first, horizon=H, freq=freq, last=last)

    monkeypatch.setattr(_worker, "training_rows", by_old_cut)
    before = forecasts(data, first, horizon=H, freq=freq, last=last)

    in_hour = set(switch_periods("Europe/Warsaw", freq))
    origin = pd.Timestamp(S)
    assert before.keys() == after.keys()
    moved = 0
    for key, record in after.items():
        run_date, period, h = key
        D = (pd.Timestamp(run_date) - origin).days
        _, _, _, moves = table(switch, period in in_hour, 0, D, h)
        if moves:
            assert before[key]["prediction"] != record["prediction"], key
            assert before[key]["actual"] == record["actual"] or np.isnan(record["actual"]), key
            moved += 1
        else:
            assert json.dumps(before[key]) == json.dumps(record), key
    assert moved == (H * (H + 1) // 2 if switch == "autumn" else H * (H - 1) // 2) * FREQS[freq]


# ---------------------------------------------------------------------------
# Hand-computed
# ---------------------------------------------------------------------------


def hand_forecast(data, freq, period, horizon, day, names, window=WINDOW):
    """The forecast read from the frame with numpy alone: rows by date, targets by the wall clock (a repeated hour has none), the
    binary columns as they are and the others standardised on the rows that train, and a least-squares fit."""
    wall = data.index.tz_localize(None) if data.index.tz is not None else data.index
    dates = wall.normalize()
    per_hour = FREQS[freq]
    periods = wall.hour * per_hour + wall.minute // (60 // per_hour)
    D = pd.Timestamp(day)
    newest = D - pd.Timedelta(days=1 + horizon)
    oldest = newest - pd.Timedelta(days=window - 1)
    here = np.asarray(periods == period)
    rows = np.flatnonzero(here & np.asarray((dates >= oldest) & (dates <= newest)))
    prices = pd.Series(data["price"].to_numpy(), index=wall)
    once = prices[~prices.index.duplicated(keep=False)]
    target = once.reindex(wall[rows] + pd.Timedelta(days=horizon)).to_numpy()
    x = data[names].to_numpy(dtype=float)
    test_row = np.flatnonzero(here & np.asarray(dates == D))[-1]
    usable = np.isfinite(target) & np.isfinite(x[rows]).all(axis=1)
    used, y, train = rows[usable], target[usable], x[rows][usable]
    scaled = np.array(["is_week" not in name for name in names])
    means = np.where(scaled, train.mean(axis=0), 0.0)
    deviations = np.where(scaled, train.std(axis=0, ddof=1), 1.0)
    coefficients, *_ = np.linalg.lstsq((train - means) / deviations, (y - y.mean()) / y.std(ddof=1), rcond=None)
    prediction = float(((x[test_row] - means) / deviations) @ coefficients * y.std(ddof=1) + y.mean())
    return prediction, used, dates, float(once.reindex([wall[test_row] + pd.Timedelta(days=horizon)]).iloc[0])


@pytest.mark.parametrize("zone", list(SWITCH_HOUR))
@pytest.mark.parametrize("freq", list(FREQS))
@pytest.mark.parametrize("switch", list(SWITCH_DAYS))
def test_a_switch_hour_forecast_equals_a_hand_fit_on_rows_chosen_by_date(zone, freq, switch):
    H = 3 if freq == "1h" else 2
    data = market(switch, zone=zone, freq=freq, horizon=H)
    S = SWITCH_DAYS[switch]
    found = forecasts(data, S, horizon=H, freq=freq, last=day_of(S, 3))
    origin = pd.Timestamp(S)
    checked = doubled = 0
    for (run_date, period, h), record in found.items():
        if period not in switch_periods(zone, freq):
            continue
        names = [name.replace("{horizon}", str(h)) for name in PREDICTORS]
        prediction, used, dates, actual = hand_forecast(data, freq, period, h, run_date, names)
        assert record["prediction"] == pytest.approx(prediction, abs=1e-9), (run_date, period, h)
        assert record["actual"] == pytest.approx(actual, nan_ok=True), (run_date, period, h)
        assert dates[used].max() <= pd.Timestamp(run_date) - pd.Timedelta(days=1 + h)
        checked += 1
        doubled += int((dates[used] == origin).sum() == 2)
    # S to S+3 at each horizon and period of the hour; in spring S itself has none.
    assert checked == (4 if switch == "autumn" else 3) * H * FREQS[freq]
    # In autumn, both rows of the doubled hour train, from S + h + 1 on.
    assert doubled == (sum(3 - h for h in range(1, H + 1)) * FREQS[freq] if switch == "autumn" else 0)


@pytest.mark.parametrize(
    ("offset", "binary"),
    [(-1, True), (-2, False)],
    ids=["a number the day after the span: binary", "a number on its last date: scaled"],
)
def test_the_binary_columns_are_read_from_the_rows_of_the_span(offset, binary):
    """A column of 0s and 1s with a number on one date: binary for this forecast if that date is after the span (D-1 at h=1), so
    left as it is, and scaled if it is the span's last date (D-1-h). A span one day off either way gets one of them wrong."""
    data = make_market_data(start="2023-04-01", days=80, noise=5.0, seed=8, horizon=1)
    D = "2023-06-10"
    rng = np.random.default_rng(1)
    data["flag"] = rng.integers(0, 2, len(data)).astype(float)
    data.loc[data.index.strftime("%Y-%m-%d") == day_of(D, offset), "flag"] = 7.0
    names = ["load_d+1", "flag"]
    found = forecasts(data, D, horizon=1, predictors=names)[(D, 5, 1)]

    model = rows_of(data, "1h", 1)
    days = model._hour_days[5]
    rows = training_rows(days, pd.Timestamp(D).date().toordinal() - model._first_ordinal, WINDOW, 1)
    x = data[names].to_numpy(dtype=float)[data.index.hour == 5][rows]
    y = model._hour_data[5]["price_d+1"].to_numpy()[rows]
    test = data.loc[f"{D} 05:00", names].to_numpy(dtype=float)
    assert (set(x[:, 1]) == {0.0, 1.0}) == binary
    means = np.array([x[:, 0].mean(), 0.0 if binary else x[:, 1].mean()])
    deviations = np.array([x[:, 0].std(ddof=1), 1.0 if binary else x[:, 1].std(ddof=1)])
    coefficients, *_ = np.linalg.lstsq((x - means) / deviations, (y - y.mean()) / y.std(ddof=1), rcond=None)
    by_hand = ((test - means) / deviations) @ coefficients * y.std(ddof=1) + y.mean()
    assert found["prediction"] == pytest.approx(by_hand, abs=1e-9)


def test_both_rows_of_a_doubled_hour_train_with_one_target():
    data = market("autumn", horizon=1)
    S = SWITCH_DAYS["autumn"]
    names = [name.replace("{horizon}", "1") for name in PREDICTORS]
    _, used, dates, _ = hand_forecast(data, "1h", 2, 1, day_of(S, 2), names)
    assert (dates[used] == pd.Timestamp(S)).sum() == 2

    model = rows_of(data, "1h", 1)
    rows = np.flatnonzero(model._hour_data[2].index.tz_localize(None).normalize() == pd.Timestamp(S))
    targets = model._hour_data[2]["price_d+1"].to_numpy()[rows]
    assert len(rows) == 2 and targets[0] == targets[1] == data.loc[f"{day_of(S, 1)} 02:00", "price"]


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("zone", "days"),
    [("Europe/Warsaw", range(4)), ("Europe/London", [1])],
    ids=["Warsaw, S to S+3", "London, S+1"],
)
@pytest.mark.parametrize("switch", list(SWITCH_DAYS))
def test_verify_is_ok_on_the_days_of_a_switch_with_its_positive_control(zone, days, switch):
    data = market(switch, zone=zone)
    S = SWITCH_DAYS[switch]
    pipeline = ModelPipeline(information_set=InformationSet(rules=RULES))
    pipeline.add_model(OLSModel(PREDICTORS + ["price_d-1"], training_window=WINDOW, name="m"))

    report = pipeline.verify(data, day_of(S, days[0]), day_of(S, days[-1]), horizon=3, days=len(days))

    report.assert_ok()
    assert report.issue_days == [day_of(S, offset) for offset in days]
    assert all(check.control.moved for check in report.checks)


@pytest.mark.parametrize("switch", list(SWITCH_DAYS))
def test_verify_at_15_minutes_on_the_days_after_a_switch(switch):
    data = market(switch, freq="15min", horizon=2)
    S = SWITCH_DAYS[switch]
    pipeline = ModelPipeline(information_set=InformationSet(rules=RULES))
    pipeline.add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="m"))
    pipeline.verify(data, day_of(S, 2), day_of(S, 2), horizon=2, freq="15min", days=1).assert_ok()


@pytest.mark.parametrize(
    ("switch", "offset", "h"),
    [
        ("autumn", 2, 1),  # the row on S, doubled, has its target on S + 1
        ("autumn", 2, 2),
        ("autumn", 3, 3),
        ("spring", 3, 1),
        ("spring", 2, 2),  # the row the row cut lost: its target is on D - 1
        ("spring", 4, 2),
    ],
)
def test_moving_the_target_of_the_day_before_moves_the_switch_hour_forecast_and_the_issue_day_s_doesnt(
    switch, offset, h
):
    """So the newest row isn't dropped by accident, and nothing newer is read."""
    data = market(switch)
    D = day_of(SWITCH_DAYS[switch], offset)
    key = (D, 2, h)
    original = forecasts(data, D)

    def moved(day):
        changed = data.copy()
        changed.loc[f"{day} 02:00", "price"] += 100.0
        return forecasts(changed, D)

    newest = moved(day_of(D, -1))
    assert abs(newest[key]["prediction"] - original[key]["prediction"]) > 1e-6
    others = {k for k in original if k[1] != 2}
    assert all(newest[k]["prediction"] == original[k]["prediction"] for k in others)

    on_the_issue_day = moved(D)
    assert on_the_issue_day[key]["prediction"] == original[key]["prediction"]


def one_day_late(anchor, window, hz, newest=-1):
    return training_span(anchor + 1, window, hz, newest)


def one_day_early(anchor, window, hz, newest=-1):
    return training_span(anchor - 1, window, hz, newest)


@pytest.mark.parametrize("switch", list(SWITCH_DAYS))
def test_a_cut_one_day_late_is_a_leak_on_the_days_of_a_switch(monkeypatch, switch):
    data = market(switch)
    S = SWITCH_DAYS[switch]
    pipeline = ModelPipeline(information_set=InformationSet(rules=RULES))
    pipeline.add_model(OLSModel(PREDICTORS + ["price_d-1"], training_window=WINDOW, name="m"))
    pipeline.verify(data, day_of(S, 2), day_of(S, 2), horizon=3, days=1).assert_ok()

    monkeypatch.setattr(_worker, "training_span", one_day_late)
    assert (
        pipeline.verify(data, day_of(S, 2), day_of(S, 2), horizon=3, days=1, positive_control=False).status("m")
        == "leak"
    )


@pytest.mark.parametrize("wrong", [one_day_late, one_day_early])
def test_a_span_off_by_a_day_fails_the_hand_fit_and_the_control(monkeypatch, wrong):
    data = market("autumn")
    D = day_of(SWITCH_DAYS["autumn"], 2)
    names = [name.replace("{horizon}", "1") for name in PREDICTORS]
    prediction, *_ = hand_forecast(data, "1h", 2, 1, D, names)
    assert forecasts(data, D)[(D, 2, 1)]["prediction"] == pytest.approx(prediction, abs=1e-9)

    monkeypatch.setattr(_worker, "training_span", wrong)
    assert forecasts(data, D)[(D, 2, 1)]["prediction"] != pytest.approx(prediction, abs=1e-9)


# ---------------------------------------------------------------------------
# Planned and produced
# ---------------------------------------------------------------------------


def two_springs():
    """Two spring switches, 2023-03-26 and 2024-03-31, and the autumn one between them."""
    return make_market_data(start="2023-01-01", days=500, noise=5.0, seed=4, horizon=2)


@pytest.fixture(scope="module")
def two_years():
    return two_springs()


VALIDATION = ("2023-03-24", "2023-03-28")
TEST = ("2024-03-29", "2024-04-02")


def small_pipeline(ensemble):
    pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="a"))
    pipeline.add_model(OLSModel(PREDICTORS[:3], training_window=WINDOW, name="b"))
    if ensemble:
        pipeline.add_ensemble(AverageEnsemble(members=["a", "b"], name="avg"))
    return pipeline.add_evaluator(MAEEvaluator())


@pytest.mark.parametrize("ensemble", [False, True], ids=["models", "with an ensemble"])
def test_count_equals_report_s_planned_per_window(tmp_path, two_years, ensemble):
    pipeline = small_pipeline(ensemble)
    run = pipeline.run(two_years, validation=VALIDATION, test=TEST, horizon=2, save_dir=tmp_path)
    merged = pipeline.report(tmp_path, validation=VALIDATION, test=TEST, horizon=2)

    # S at 2 horizons, S-1 at h=1, S-2 at h=2: 4 of 5 * 2 * 24 aren't on the clock, in each window.
    expected = 5 * 2 * 24 - 4
    for report in (run, run.validation, merged, merged.validation):
        names = ["a", "b", "avg"] if ensemble else ["a", "b"]
        assert list(report.refs) == names
        counts = report.forecast_counts().set_index("model")
        for name in names:
            assert report.refs[name].count == expected
            assert report.refs[name].produced == expected
            assert counts.loc[name, "planned"] == expected
            assert counts.loc[name, "missing"] == 0
            assert report.refs[name].skipped == {}
    whole = IssueCalendar("1h", two_years.index.tz, [1, 2])
    assert whole.count(*TEST) == whole.count(*VALIDATION) == expected


@pytest.mark.parametrize("ensemble", [False, True], ids=["models", "with an ensemble"])
def test_count_equals_report_s_planned_in_a_plain_run(tmp_path, two_years, ensemble):
    pipeline = small_pipeline(ensemble)
    run = pipeline.run(two_years, test=VALIDATION, horizon=2, save_dir=tmp_path)
    merged = pipeline.report(tmp_path, test=VALIDATION, horizon=2)
    for report in (run, merged):
        counts = report.forecast_counts().set_index("model")
        for name in report.refs:
            assert (counts.loc[name, "planned"], counts.loc[name, "produced"]) == (5 * 2 * 24 - 4, 5 * 2 * 24 - 4)
            assert "member_missing" not in counts.columns

    # A report told of a smaller horizon than its members were run with: the ensemble plans what they planned.
    if ensemble:
        assert merged.refs["avg"].horizon == 2


def test_a_run_ending_before_a_spring_switch_plans_nothing_aimed_at_the_hour_it_skips():
    data = market("spring", horizon=1)
    D = day_of(SWITCH_DAYS["spring"], -1)
    ref = OLSModel(PREDICTORS, training_window=WINDOW).run(data, D, D, horizon=1)
    assert ref.count == ref.produced == 23 and ref.skipped == {}
    assert 2 not in {r["hour"] for r in ref._results}


def test_a_real_gap_is_still_planned_and_counted_where_the_clock_s_gap_isn_t():
    data = market("spring", horizon=1)
    S = SWITCH_DAYS["spring"]
    gappy = data[~((data.index.date == pd.Timestamp(day_of(S, 1)).date()) & (data.index.hour == 5))]
    ref = OLSModel(PREDICTORS, training_window=WINDOW).run(gappy, S, day_of(S, 1), horizon=1)
    # S has 23 hours, S+1 has 24 of which one lacks its row.
    assert ref.count == 23 + 24
    assert ref.skipped == {"missing_test_row": 1}
    assert ref.produced == 46


def test_the_clock_s_gap_is_no_missing_row_and_a_worker_asked_for_it_says_so():
    data = market("spring", horizon=1)
    S = SWITCH_DAYS["spring"]
    assert _contract.missing_test_rows(data.index, "1h", [pd.Timestamp(S)]) == []
    lost = data[~((data.index.strftime("%Y-%m-%d") == S) & (data.index.hour == 5))]
    assert _contract.missing_test_rows(lost.index, "1h", [pd.Timestamp(S)]) == [(5, pd.Timestamp(S))]

    model = rows_of(data, "1h", 1)
    arrays = {
        2: {
            "x": model._hour_data[2][["price"]].to_numpy(dtype=float),
            "y": model._hour_data[2][["price_d+1"]].to_numpy(dtype=float),
            "days": model._hour_days[2],
            "col_index": {"price": 0},
        }
    }
    offset = pd.Timestamp(S).date().toordinal() - model._first_ordinal
    config = {
        "offset": offset,
        "test_start": S,
        "training_window": WINDOW,
        "scalable_masks": {(2, 1): np.array([True])},
        "scaler": "standard",
        "threads_per_process": 1,
    }
    _worker_init(arrays, {(2, 1): ["price"]}, config, OLSModel, {})
    assert _fit_one_numpy(2, 1, 0) == SKIP_MISSING_TEST_ROW
    assert _fit_one_numpy(2, 1, 1)["run_date"] == day_of(S, 1)


def test_an_hour_the_clock_skips_half_of_is_planned_with_its_row():
    """Lord Howe skips 02:00 to 02:29 on 2024-10-06, so an hourly frame has its period 2 at 02:30 that day: the forecast issued
    from it is planned and made, and so is the one aimed at it, whose 02:00 target doesn't exist."""
    zone = "Australia/Lord_Howe"
    index = pd.date_range(
        pd.Timestamp("2024-09-01", tz=zone), pd.Timestamp("2024-10-20", tz=zone), freq="1h", inclusive="left"
    )
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"x": rng.normal(size=len(index)), "price": rng.normal(50, 10, len(index))}, index=index)
    assert [t.strftime("%H:%M") for t in data.loc["2024-10-06"].index if t.hour == 2] == ["02:30"]
    ref = OLSModel(["x"], training_window=10).run(data, "2024-10-05", "2024-10-07", horizon=1)
    records = by_key(ref._results)
    assert ref.count == ref.produced == 3 * 24 and ref.skipped == {}
    assert ("2024-10-06", 2, 1) in records and np.isnan(records[("2024-10-05", 2, 1)]["actual"])


def unplanned_only(ctx):
    """A list with a late column, and a late read, only where the clock has no 02:00: the spring switch day."""
    if ctx.issue_day == datetime.date(2023, 3, 26) and ctx.period == 2:
        ctx.value("late")
        return PREDICTORS + ["late"]
    return PREDICTORS


@pytest.mark.parametrize("level", ["warn", "raise"])
def test_a_list_only_on_a_slot_the_clock_skips_is_neither_listed_nor_refused(level):
    data = make_market_data(noise=5.0, seed=42)
    data["late"] = data["load"]
    info = InformationSet("11:00", "Europe/Warsaw", rules={**RULES, "late": Availability.published("13:00", -1)})
    pipeline = ModelPipeline(information_set=info, availability=level)
    pipeline.add_model(OLSModel(unplanned_only, training_window=WINDOW, name="m"))
    report = pipeline.run(data, test=("2023-03-25", "2023-03-27"), horizon=1)
    assert report.availability().empty
    assert sorted(entry["forecasts"] for entry in report.refs["m"].predictor_lists) == [0, report.refs["m"].count]
    assert pipeline.verify(data, "2023-03-26", "2023-03-26", horizon=1, days=1).status("m") == "ok"


def test_the_lists_of_a_predictor_function_count_the_planned_forecasts(two_years):
    def all_the_same(ctx):
        return PREDICTORS

    ref = OLSModel(all_the_same, training_window=WINDOW).run(two_years, *VALIDATION, horizon=2)
    assert sum(entry["forecasts"] for entry in ref.predictor_lists) == ref.count == ref.produced == 5 * 2 * 24 - 4


def test_a_matching_forecast_the_clock_doesnt_have_isnt_counted_before_any_fit(two_years):
    def matching(ctx):
        return Use(PREDICTORS, train_on="matching")

    model = OLSModel(matching, training_window=WINDOW)
    plan = model._plan_predictors(two_years, "price", 2, "1h", None, *VALIDATION, warn_unpublished=False)
    model._load_periods(two_years, ["price", *sorted({n for names in plan.lists for n in names})], 2, "price")
    counts = model._matching_day_counts(plan, "price")
    S = plan.position_of("2023-03-26")
    assert counts[(2, 1)][S] == counts[(2, 2)][S] == -1  # no row to issue from
    assert counts[(2, 1)][S - 1] == counts[(2, 2)][S - 2] == -1  # no slot to score on
    assert counts[(2, 1)][S + 1] > 0 and counts[(3, 1)][S] > 0

    ref = OLSModel(matching, training_window=WINDOW).run(two_years, *VALIDATION, horizon=2)
    for record in ref._results:
        count = counts[(record["hour"], record["horizon"])][plan.position_of(record["run_date"])]
        assert count == record["training_days"]


# ---------------------------------------------------------------------------
# aggregate() on switch days, on real runs
# ---------------------------------------------------------------------------


def aggregated(data, first, last, horizon, freq, save_dir=None):
    pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="m"))
    pipeline.add_evaluator(MAEEvaluator())
    report = pipeline.run(data, test=(first, last), horizon=horizon, freq=freq, save_dir=save_dir)
    return report, report.aggregate("1D", "max")


def daily_keys(report):
    return {(str(r["run_date"]), str(r["target_date"]), r["horizon"]) for r in report.refs["m"]._results}


@pytest.mark.parametrize("freq", list(FREQS))
@pytest.mark.parametrize("horizon", [1, 2], ids=["H1", "H2"])
def test_aggregate_drops_the_day_after_spring_at_every_horizon_set(freq, horizon):
    data = market("spring", freq=freq, horizon=2)
    S = SWITCH_DAYS["spring"]
    hourly, daily = aggregated(data, day_of(S, -1), day_of(S, 2), horizon, freq)

    keys = daily_keys(daily)
    # The spring day is whole over its 23 hours, or 92 quarter-hours.
    assert (day_of(S, -1), S, 1) in keys
    # The day after it, from S: no forecast of 02:00 was issued, so that day lacks one id at each horizon, also at H=1.
    for h in range(1, horizon + 1):
        assert (S, day_of(S, h), h) not in keys
    # Every other day is whole.
    every = {
        (day_of(S, offset), day_of(S, offset + h), h)
        for offset in range(-1, 3)
        for h in range(1, horizon + 1)
        if not (horizon == 2 and offset == -2)
    }
    dropped = {key for key in every if key[0] == S}
    assert keys == every - dropped
    assert daily.refs["m"].skipped == {"incomplete_day": len(dropped)}
    counts = daily.forecast_counts().set_index("model").loc["m"]
    assert (counts["planned"], counts["produced"], counts["incomplete_day"]) == (len(every), len(keys), len(dropped))

    # The whole spring day is made of the periods the clock has.
    spring = [r for r in hourly.refs["m"]._results if r["target_date"] == S and r["horizon"] == 1]
    assert len(spring) == 23 * FREQS[freq]


@pytest.mark.parametrize("freq", list(FREQS))
@pytest.mark.parametrize("horizon", [1, 2], ids=["H1", "H2"])
def test_aggregate_drops_the_autumn_day_for_its_repeated_hour_without_an_actual(freq, horizon):
    data = market("autumn", freq=freq, horizon=2)
    S = SWITCH_DAYS["autumn"]
    _, daily = aggregated(data, day_of(S, -2), day_of(S, 1), horizon, freq)
    keys = daily_keys(daily)
    every = {(day_of(S, offset), day_of(S, offset + h), h) for offset in range(-2, 2) for h in range(1, horizon + 1)}
    dropped = {key for key in every if key[1] == S}
    assert dropped and keys == every - dropped
    assert daily.refs["m"].skipped == {"incomplete_day": len(dropped)}


def test_an_ensemble_aggregates_by_the_clock_too():
    data = market("spring", horizon=1)
    S = SWITCH_DAYS["spring"]
    pipeline = small_pipeline(True)
    daily = pipeline.run(data, test=(day_of(S, -1), day_of(S, 1)), horizon=1).aggregate("1D", "max")
    for name in ("a", "avg"):
        keys = {(str(r["run_date"]), str(r["target_date"])) for r in daily.refs[name]._results}
        assert (S, day_of(S, 1)) not in keys and (day_of(S, -1), S) in keys
        assert daily.refs[name].skipped == {"incomplete_day": 1}


def test_aggregate_without_a_zone_keeps_the_rule_of_the_records():
    """Plain lists say nothing of a clock: a day lacking a period no model has is taken at its word."""
    data = market("spring", horizon=1)
    S = SWITCH_DAYS["spring"]
    report, daily = aggregated(data, day_of(S, -1), day_of(S, 1), 1, "1h")
    records = list(report.refs["m"]._results)
    assert (S, day_of(S, 1), 1) not in daily_keys(daily)

    plain = EvaluationReport({"m": records}, [MAEEvaluator()]).aggregate("1D", "max")
    assert (S, day_of(S, 1), 1) in {
        (str(r["run_date"]), str(r["target_date"]), r["horizon"]) for r in plain.refs["m"]._results
    }


def test_aggregate_from_files_equals_aggregate_of_the_run(tmp_path):
    data = market("spring", horizon=2)
    S = SWITCH_DAYS["spring"]
    report, daily = aggregated(data, day_of(S, -1), day_of(S, 2), 2, "1h", save_dir=tmp_path)
    pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="m"))
    merged = pipeline.report(tmp_path, day_of(S, -1), day_of(S, 2), horizon=2)
    assert merged.refs["m"].zone == "Europe/Warsaw"
    assert merged.refs["m"].count == report.refs["m"].count
    from_files = merged.aggregate("1D", "max")
    assert daily_keys(from_files) == daily_keys(daily)
    assert from_files.refs["m"].skipped == daily.refs["m"].skipped


@pytest.mark.parametrize(
    ("zone", "freq", "switch"),
    [
        ("America/Havana", "1h", "2024-03-10"),  # skips its midnight
        ("America/Nuuk", "1h", "2023-03-25"),  # skips 22:00
        ("Australia/Lord_Howe", "15min", "2024-10-06"),  # skips half an hour
    ],
)
def test_aggregate_keeps_the_short_day_of_any_clock_and_drops_the_day_after_from_the_switch_day(zone, freq, switch):
    first = day_of(switch, -50)
    data = make_market_data(start=first, days=60, tz=zone, freq=freq, horizon=1, noise=5.0, seed=2)
    _, daily = aggregated(data, day_of(switch, -1), day_of(switch, 1), 1, freq)
    keys = daily_keys(daily)
    assert (day_of(switch, -1), switch, 1) in keys
    assert (switch, day_of(switch, 1), 1) not in keys and (day_of(switch, 1), day_of(switch, 2), 1) in keys
    assert daily.refs["m"].skipped == {"incomplete_day": 1}


def test_a_report_from_the_files_of_a_dateutil_zone_knows_the_clock(tmp_path):
    data = market("spring", horizon=1)
    data.index = data.index.tz_convert(dateutil.tz.gettz("Europe/Warsaw"))
    S = SWITCH_DAYS["spring"]
    report, daily = aggregated(data, day_of(S, -1), day_of(S, 1), 1, "1h", save_dir=tmp_path)
    assert json.loads(manifest_path(tmp_path / "m.jsonl").read_text())["data_timezone"] == "Europe/Warsaw"
    pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="m"))
    merged = pipeline.report(tmp_path, day_of(S, -1), day_of(S, 1), horizon=1)
    assert merged.refs["m"].count == report.refs["m"].count == 3 * 24 - 2
    assert daily_keys(merged.aggregate("1D", "max")) == daily_keys(daily)


def test_a_zone_without_a_name_is_said_at_run_time(tmp_path):
    data = make_market_data(start="2023-04-01", days=60, tz="UTC", horizon=1, noise=5.0, seed=1).drop(
        columns="price_d-1"
    )
    data.index = data.index.tz_convert(dateutil.tz.tzoffset(None, 3600))
    with pytest.warns(UserWarning, match="has no name that reads back"):
        OLSModel(PREDICTORS, training_window=20).run(
            data, "2023-05-20", "2023-05-20", horizon=1, save_to=tmp_path / "m.jsonl"
        )


# ---------------------------------------------------------------------------
# Gaps, old manifests, frames without a switch
# ---------------------------------------------------------------------------


def test_a_gappy_frame_trains_on_rows_up_to_the_target_of_the_day_before():
    """A row lost in the last days before the issue cost the row cut the newest rows, as it counted back from the issue day."""
    data = make_market_data(start="2023-04-01", days=80, noise=5.0, seed=8, horizon=2)
    D = "2023-06-10"
    gappy = data[~((data.index.strftime("%Y-%m-%d") == day_of(D, -2)) & (data.index.hour == 5))]

    found = forecasts(gappy, D, horizon=2)
    for h in (1, 2):
        names = [name.replace("{horizon}", str(h)) for name in PREDICTORS]
        prediction, used, dates, _ = hand_forecast(gappy, "1h", 5, h, D, names)
        assert found[(D, 5, h)]["prediction"] == pytest.approx(prediction, abs=1e-9)
        # At h=2 the newest row is the rule's, D-3, whose target is on D-1. At h=1 that row is the one lost, and D-3's target is too.
        assert dates[used].max() == pd.Timestamp(D) - pd.Timedelta(days=3 if h == 2 else 4)

    model = rows_of(gappy, "1h", 2)
    days = model._hour_days[5]
    day = pd.Timestamp(D).date().toordinal() - model._first_ordinal
    assert days[training_rows(days, day, WINDOW, 2)].max() == day - 3
    assert days[old_cut(days, day, WINDOW, 2)].max() == day - 4


def test_a_real_gap_on_the_issue_day_is_no_row_and_no_row_in_the_span_is_too_few():
    data = make_market_data(start="2023-04-01", days=80, noise=5.0, seed=8, horizon=1)
    D = "2023-06-10"
    no_issue_row = data[~((data.index.strftime("%Y-%m-%d") == D) & (data.index.hour == 5))]
    ref = OLSModel(PREDICTORS, training_window=WINDOW).run(no_issue_row, D, D, horizon=1)
    assert ref.skipped == {"missing_test_row": 1}

    # The whole span of hour 5 is gone: two dates, D-3 and D-2, with a window of 2.
    span = [day_of(D, -3), day_of(D, -2)]
    no_span = data[~(np.isin(data.index.strftime("%Y-%m-%d"), span) & (data.index.hour == 5))]
    ref = OLSModel(PREDICTORS, training_window=2).run(no_span, D, D, horizon=1)
    assert ref.skipped == {"too_few_rows": 1}
    # A row outside it doesn't help: the row dated D-1 has its target on D.
    assert (5, 1) not in {(r["hour"], r["horizon"]) for r in ref._results}


def test_a_window_of_few_rows_still_trains():
    """`too_few_rows` is no row in the span: the row cut asked for more rows than it would cut."""
    data = make_market_data(start="2023-04-01", days=80, noise=5.0, seed=8, horizon=1)
    ref = OLSModel(PREDICTORS[:1], training_window=2).run(data, "2023-06-10", "2023-06-10", horizon=1)
    assert ref.produced == ref.count == 24


def write_old_manifest(path):
    stored = json.loads(path.read_text())
    assert stored.pop("training_cut") == {"rows": "dates"}
    path.write_text(json.dumps(stored))


def old_results(tmp_path, data, **run):
    pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="m"))
    pipeline.run(data, save_dir=tmp_path, horizon=1, **run)
    write_old_manifest(manifest_path(tmp_path / "m.jsonl"))
    return pipeline


def test_a_manifest_without_the_field_is_refused_on_resume_and_the_override_goes_on(tmp_path):
    data = market("spring", horizon=1)
    period = (SWITCH_DAYS["spring"], day_of(SWITCH_DAYS["spring"], 1))
    pipeline = old_results(tmp_path, data, test=period)

    with pytest.raises(ManifestMismatchError, match="training_cut.*before training rows were chosen by date"):
        pipeline.run(data, test=period, horizon=1, save_dir=tmp_path)

    with pytest.warns(UserWarning, match="may hold records written before training rows were chosen by date"):
        report = pipeline.run(data, test=period, horizon=1, save_dir=tmp_path, allow_manifest_mismatch=True)
    assert report.refs["m"].count == 47
    unverified = {"rows": "dates", "stored": "unverified"}
    assert json.loads(manifest_path(tmp_path / "m.jsonl").read_text())["training_cut"] == unverified
    pipeline.run(data, test=period, horizon=1, save_dir=tmp_path)  # resumes without the override, and keeps saying so
    assert json.loads(manifest_path(tmp_path / "m.jsonl").read_text())["training_cut"] == unverified


def test_a_manifest_without_the_field_is_refused_in_a_run_with_windows(tmp_path):
    data = market("spring", horizon=1)
    S = SWITCH_DAYS["spring"]
    windows = {"validation": (day_of(S, -3), day_of(S, -1)), "test": (S, day_of(S, 1))}
    pipeline = old_results(tmp_path, data, **windows)
    with pytest.raises(ManifestMismatchError, match="training_cut"):
        pipeline.run(data, horizon=1, save_dir=tmp_path, **windows)


def test_a_report_of_old_results_says_so_and_plans_by_the_clock(tmp_path):
    data = market("spring", horizon=1)
    S = SWITCH_DAYS["spring"]
    pipeline = old_results(tmp_path, data, test=(S, S))
    with pytest.warns(UserWarning, match=r"\['m'\] may hold records written before training rows were chosen by date"):
        report = pipeline.report(tmp_path, test=(S, S), horizon=1)
    assert report.refs["m"].zone == "Europe/Warsaw"
    assert report.refs["m"].count == 23


def test_a_report_of_today_s_results_says_nothing_of_old_records(tmp_path):
    data = market("spring", horizon=1)
    S = SWITCH_DAYS["spring"]
    pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="m"))
    pipeline.run(data, test=(S, S), horizon=1, save_dir=tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.report(tmp_path, test=(S, S), horizon=1)
    assert not [w for w in caught if "training rows" in str(w.message) or "clock doesn't have" in str(w.message)]


@pytest.mark.parametrize(
    ("stored", "zone"),
    [
        ("Europe/Warsaw", "Europe/Warsaw"),
        ("UTC+02:00", "UTC+02:00"),
        ("None", None),
        ("pytz.FixedOffset(60)", None),  # `str` of some time zones doesn't read back
    ],
)
def test_the_zone_of_a_manifest(tmp_path, stored, zone):
    path = tmp_path / "m.manifest.json"
    for cut in ({"rows": "dates"}, {"rows": "dates", "stored": "unverified"}, None):
        path.write_text(json.dumps({"data_timezone": stored, **({} if cut is None else {"training_cut": cut})}))
        assert read_zone(path) == zone
        assert rows_unverified(path) == (cut != {"rows": "dates"})
    assert read_zone(tmp_path / "no.manifest.json") is None
    assert rows_unverified(tmp_path / "no.manifest.json")


def last_row_up_to(days, day):
    """The row the row cut forecast from: the last one up to the issue day, whatever its date."""
    row = int(np.searchsorted(days, day, side="right")) - 1
    return row if row >= 0 else None


def stage_with_the_row_cut(monkeypatch, pipeline, data, save_dir, **run):
    """Results as the row cut wrote them: every slot planned, rows counted, and the test row the last one up to the issue day."""
    with monkeypatch.context() as patch:
        patch.setattr(_worker, "training_rows", by_old_cut)
        patch.setattr(_worker, "issue_row", last_row_up_to)
        patch.setattr(base_module, "IssueCalendar", lambda freq, zone, horizons: IssueCalendar(freq, None, horizons))
        pipeline.run(data, save_dir=save_dir, **run)


@pytest.mark.parametrize("how", ["override", "no manifest"])
def test_results_staged_by_the_row_cut_and_accepted_are_warned_about_every_time_and_lose_the_slot_the_clock_skips(
    tmp_path, monkeypatch, how
):
    data = market("spring", horizon=2)
    S = SWITCH_DAYS["spring"]
    run = dict(test=(day_of(S, -1), day_of(S, 1)), horizon=2)
    pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="m"))
    pipeline.add_evaluator(MAEEvaluator())
    stage_with_the_row_cut(monkeypatch, pipeline, data, tmp_path, **run)
    stale = (S, 2, 2)  # built from the row of S-1, with the price of S+1 as its actual
    assert stale in by_key(json.loads(line) for line in (tmp_path / "m.jsonl").read_text().splitlines())
    manifest = manifest_path(tmp_path / "m.jsonl")
    if how == "override":
        write_old_manifest(manifest)
        with pytest.raises(ManifestMismatchError, match="training_cut"):
            pipeline.run(data, save_dir=tmp_path, **run)
        with pytest.warns(UserWarning, match="its manifest was written before rows were chosen by date"):
            pipeline.run(data, save_dir=tmp_path, allow_manifest_mismatch=True, **run)
    else:
        manifest.unlink()
        with pytest.warns(UserWarning, match="they have no manifest"):
            pipeline.run(data, save_dir=tmp_path, **run)
    assert json.loads(manifest.read_text())["training_cut"] == {"rows": "dates", "stored": "unverified"}
    pipeline.run(data, save_dir=tmp_path, **run)  # later resumes go on, and keep saying so

    fresh = ModelPipeline().add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="m")).run(data, **run)
    for _ in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = pipeline.report(tmp_path, test=run["test"], horizon=2)
        said = [str(w.message) for w in caught]
        assert any("['m'] may hold records written before training rows were chosen by date" in m for m in said), said
        assert any("clock doesn't have" in m and "{'m': 1}" in m for m in said), said
        keys = set(by_key(report.refs["m"].iter_records()))
        assert stale not in keys and keys == set(by_key(fresh.refs["m"].iter_records()))
        counts = report.forecast_counts().set_index("model").loc["m"]
        assert counts["planned"] == fresh.refs["m"].count and counts["produced"] == len(keys)
        assert report.summary().set_index("model").loc["m", "forecasts"] == len(keys)
    report.refs["m"].zone = None  # without a zone nothing tells the slot apart, and a table read before isn't reused
    assert report.summary().set_index("model").loc["m", "forecasts"] == len(keys) + 1


def test_a_ref_with_a_zone_leaves_out_a_record_of_a_slot_the_clock_skips():
    data = market("spring", horizon=1)
    S = SWITCH_DAYS["spring"]
    ref = OLSModel(PREDICTORS, training_window=WINDOW).run(data, S, S, horizon=1)
    stale = {**ref._results[0], "hour": 2, "run_date": S, "target_date": day_of(S, 1)}
    ref._results.append(stale)
    assert ref.off_the_clock() == 1 and len(list(ref.iter_records())) == 23
    assert EvaluationReport({"m": ref}, [MAEEvaluator()]).summary().set_index("model").loc["m", "forecasts"] == 23
    ref.zone = None
    assert ref.off_the_clock() == 0 and len(list(ref.iter_records())) == 24


def test_a_manifest_of_today_resumes_and_the_second_run_has_nothing_to_do(tmp_path, capsys):
    data = market("spring", horizon=1)
    S = SWITCH_DAYS["spring"]
    pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="m"))
    first = pipeline.run(data, test=(S, day_of(S, 1)), horizon=1, save_dir=tmp_path)
    capsys.readouterr()
    again = pipeline.run(data, test=(S, day_of(S, 1)), horizon=1, save_dir=tmp_path)
    assert "All 47 tasks completed" in capsys.readouterr().out
    assert first.refs["m"].produced == again.refs["m"].produced == first.refs["m"].count


def test_a_plain_result_grown_into_windows_compares_its_actuals_by_the_issue_day_row(tmp_path):
    data = market("spring", horizon=1)
    S = SWITCH_DAYS["spring"]
    period = (S, day_of(S, 1))
    pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="m"))
    pipeline.run(data, test=period, horizon=1, save_dir=tmp_path)
    validation = (day_of(S, -3), day_of(S, -1))
    pipeline.run(data, validation=validation, test=period, horizon=1, save_dir=tmp_path)


def test_a_stale_record_of_a_slot_the_clock_doesnt_have_is_refused_when_a_plain_result_is_grown(tmp_path):
    data = market("spring", horizon=2)
    S = SWITCH_DAYS["spring"]
    period = (S, day_of(S, 1))
    pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="m"))
    pipeline.run(data, test=period, horizon=2, save_dir=tmp_path)
    path = tmp_path / "m.jsonl"
    first = json.loads(path.read_text().splitlines()[0])
    # What the row cut wrote for (S, period 2, h=2): the row of S-1, whose target at h=2 is the price of S+1 at 02:00.
    old_actual = float(data.loc[f"{day_of(S, 1)} 02:00", "price"])
    stale = {**first, "run_date": S, "target_date": day_of(S, 2), "hour": 2, "horizon": 2, "actual": old_actual}
    with open(path, "a") as f:
        f.write(json.dumps(stale) + "\n")

    with pytest.raises(ManifestMismatchError, match="another `actual`"):
        pipeline.run(data, validation=(day_of(S, -4), day_of(S, -2)), test=period, horizon=2, save_dir=tmp_path)


@pytest.mark.parametrize(
    ("start", "tz"),
    [("2023-04-01", "Europe/Warsaw"), ("2023-01-01", "UTC"), ("2023-01-01", None)],
    ids=["no switch in the frame", "a UTC index over a switch date", "a naive index over a switch date"],
)
@pytest.mark.parametrize("freq", list(FREQS))
def test_a_frame_without_switches_is_bitwise_the_row_cut_and_today_s_count(monkeypatch, start, tz, freq):
    data = make_market_data(start=start, days=100, tz=tz or "UTC", freq=freq, horizon=2, noise=5.0, seed=6)
    data = data.drop(columns="price_d-1")  # an index in UTC refuses a day lag made in another zone
    if tz is None:
        data.index = data.index.tz_localize(None)
    first, last = ("2023-06-01", "2023-06-03") if start == "2023-04-01" else ("2023-03-25", "2023-03-27")

    def run():
        ref = OLSModel(PREDICTORS, training_window=WINDOW).run(data, first, last, horizon=2, freq=freq)
        return ref, sorted(json.dumps(r) for r in ref._results)

    ref, today = run()
    assert ref.count == ref.produced == 3 * 2 * 24 * FREQS[freq]

    monkeypatch.setattr(_worker, "training_rows", by_old_cut)
    _, cut = run()
    assert today == cut
