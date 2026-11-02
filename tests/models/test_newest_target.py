"""`newest_target`, per model (plan decision D1): "d-1", or every target its rule says is known by the issue time.

The table of newest targets and the forecasts are held against what doesn't share code with them: k written down from each rule
by hand, rows chosen and fitted with numpy from the frame. The probes are held against models that leak on purpose, and the
positive control against a model it can't reach.
"""

import datetime
import json
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import epftoolbox2.information_set as information_set_module
import epftoolbox2.verification as verification_module
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.information_set import (
    Availability,
    InformationSet,
    InformationSetError,
    leakage_probe,
    perturb_recent,
    perturb_unknown,
    positive_control,
)
from epftoolbox2.models import LassoCVModel, OLSModel, Use
from epftoolbox2.models._calendar import NewestTargets
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.models.base import BaseModel
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path, rows_unverified
from tests.synthetic import PREDICTORS, calendar_shift, make_market_data

SWITCH_DAYS = {"spring": "2023-03-26", "autumn": "2023-10-29"}
# Europe/Warsaw skips 02:00 to 02:59 on the spring day: period 2, or 8 to 11 at 15 minutes.
SKIPPED = {"1h": [2], "15min": [8, 9, 10, 11]}
STEP = {"1h": 60, "15min": 15}
WINDOW = 40
INPUT_RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
}
# The rows of the table in the spec, by the rule of the target, `price` here whatever it stands for, and the issue time.
CASES = {
    "price published 13:00 D-1": (Availability.published("13:00", 1), "11:45"),
    "load realised 1h": (Availability.realised("1h"), "11:45"),
    "price issued 14:00": (Availability.published("13:00", 1), "14:00"),
}


def newest_by_hand(case, freq, period):
    """The newest day, relative to D, whose target the rule makes known by the issue time on an ordinary day, written down from
    the rule. `published 13:00 D-1` at 11:45: D's prices, not D+1's. `realised 1h` at 11:45: a period is known 1 h after it ends,
    so up to the one ending 10:45, 10:00 hourly (period 9) and 10:45 at 15 minutes (period 42; known exactly at 11:45, which
    counts). At 14:00 the prices of D+1, published 13:00 on D, are out too."""
    if case == "load realised 1h":
        return 0 if period <= {"1h": 9, "15min": 42}[freq] else -1
    return 1 if case == "price issued 14:00" else 0


def k_by_hand(newest, day, skipped_days, horizon):
    """k of an issue day: every target from D-31 to D+k known, at most horizon - 1. A slot the clock skips has no value to be
    late, so it doesn't stop the walk."""
    for offset in range(-31, horizon):
        if offset > newest and day + datetime.timedelta(days=offset) not in skipped_days:
            return min(offset - 1, horizon - 1)
    return horizon - 1


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


def info_for(case, **more):
    rule, issue = CASES[case]
    return InformationSet(issue, "Europe/Brussels", rules={**INPUT_RULES, "price": rule, **more})


def market(switch, freq="1h", horizon=2, seed=3, zone="Europe/Warsaw"):
    return make_market_data(
        start=day_of(SWITCH_DAYS[switch], -70), days=100, tz=zone, freq=freq, horizon=horizon, noise=5.0, seed=seed
    )


def by_key(records):
    return {(r["run_date"], r["hour"], r["horizon"]): r for r in records}


def run(data, first, last, info, newest_target="published", horizon=2, freq="1h", predictors=PREDICTORS, **kwargs):
    model = OLSModel(predictors, training_window=WINDOW, newest_target=newest_target, **kwargs)
    return by_key(
        model.run(data, test_start=first, test_end=last, horizon=horizon, freq=freq, information_set=info)._results
    )


_NEWEST_TARGETS = InformationSet.newest_targets


def one_day_more(self, *args, **kwargs):
    """`newest_targets` with k + 1: the mutation the probes must catch."""
    return _NEWEST_TARGETS(self, *args, **kwargs) + 1


class TrainingRowsOnly(BaseModel):
    """Reads its training rows and never the row it forecasts from, so only a change to a training row can move it: what the
    positive control must leave alone."""

    def _fit_predict(self, train_x, train_y, test_x):
        return float(train_y.mean() + train_x[:, 0].max()), []


# ---------------------------------------------------------------------------
# The keyword
# ---------------------------------------------------------------------------


def test_newest_target_is_d_1_or_published_and_anything_else_raises():
    assert OLSModel(PREDICTORS).newest_target == "d-1"
    assert LassoCVModel(PREDICTORS, newest_target="published").newest_target == "published"
    for wrong in ["D-1", "published ", None, -1, "d-0"]:
        with pytest.raises(ValueError, match="newest_target must be one of"):
            OLSModel(PREDICTORS, newest_target=wrong)
    model = OLSModel(PREDICTORS)
    with pytest.raises(ValueError, match="newest_target must be one of"):
        model.newest_target = "latest"
    assert "newest_target" not in model._model_kwargs  # the cut travels in the worker config


@pytest.mark.parametrize("level", ["none", "warn", "raise"])
def test_published_without_an_information_set_or_a_rule_for_the_target_raises_at_every_level(level):
    data = market("autumn")
    pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, newest_target="published", name="m"))
    with pytest.raises(InformationSetError, match="needs an information set"):
        pipeline.run(data, test=("2023-10-10", "2023-10-11"), horizon=2)
    with pytest.raises(InformationSetError, match="needs an information set"):
        OLSModel(PREDICTORS, newest_target="published").run(data, "2023-10-10", "2023-10-11", horizon=2)

    no_rule = InformationSet(rules=INPUT_RULES, use_defaults=False, unknown="ignore")
    pipeline = ModelPipeline(information_set=no_rule, availability=level)
    pipeline.add_model(OLSModel(PREDICTORS, newest_target="published", name="m"))
    with pytest.raises(InformationSetError, match="'price' has none"):
        pipeline.run(data, test=("2023-10-10", "2023-10-11"), horizon=2)
    # Without a rule for its target, "d-1" runs at the levels that let a column without a rule through.
    if level != "raise":
        pipeline = ModelPipeline(information_set=no_rule, availability=level).add_model(OLSModel(PREDICTORS, name="m"))
        assert pipeline.run(data, test=("2023-10-10", "2023-10-11"), horizon=2).refs["m"].produced == 96


def test_a_rule_that_doesn_t_know_the_target_of_31_days_before_raises():
    data = market("autumn")
    info = InformationSet("11:45", rules={**INPUT_RULES, "price": Availability.published("00:00", -32)})
    with pytest.raises(InformationSetError, match="looks back 31 days"):
        run(data, "2023-10-10", "2023-10-10", info)
    info = InformationSet("11:45", rules={**INPUT_RULES, "price": Availability.published("00:00", -31)})
    k = info.newest_targets("price", "1h", data.index, ["2023-10-10"], 2)
    assert (k == -31).all()


# ---------------------------------------------------------------------------
# The table, by hand
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("freq", ["1h", "15min"])
@pytest.mark.parametrize("horizon", [1, 2, 3])
@pytest.mark.parametrize("case", list(CASES))
def test_the_newest_targets_are_the_table_of_the_spec_across_both_switches(case, horizon, freq):
    """Every issue day from 35 days before the spring switch to 35 after the autumn one, so that the walk back from each
    crosses a switch. Ordinary days have the spec's k. Only the day before the spring switch differs, in the hour it skips:
    the switch day has no value there to be late, so k is one day more, up to the cap."""
    index = pd.date_range("2023-02-01", "2023-12-31", freq=freq, tz="Europe/Warsaw")
    days = [d.date() for d in pd.date_range(day_of(SWITCH_DAYS["spring"], -35), day_of(SWITCH_DAYS["autumn"], 35))]
    found = info_for(case).newest_targets("price", freq, index, days, horizon)

    spring = pd.Timestamp(SWITCH_DAYS["spring"]).date()
    ordinary = np.array([[min(newest_by_hand(case, freq, period), horizon - 1)] for period in range(found.shape[0])])
    expected = np.repeat(ordinary, len(days), axis=1)
    for period in SKIPPED[freq]:
        expected[period] = [k_by_hand(newest_by_hand(case, freq, period), day, {spring}, horizon) for day in days]
    assert np.array_equal(found, expected)
    # Where that differs from an ordinary day: the day whose newest target would be the spring switch day.
    differs = {(int(p), days[int(d)]) for p, d in np.argwhere(expected != ordinary)}
    assert differs == (
        {(p, spring - datetime.timedelta(days=int(ordinary[p, 0]) + 1)) for p in SKIPPED[freq]}
        if ordinary[SKIPPED[freq][0], 0] < horizon - 1
        else set()
    )


def test_on_a_fixed_offset_a_brussels_rule_reads_the_summer_s_last_hour_as_tomorrow():
    """On a UTC+1 index, 23:00 in summer is 00:00 of the next day in Brussels, which the price of D-1's auction doesn't cover: by
    a Brussels rule that target is known only at 13:00 on D, after the issue time. The benchmark files' index only labels their
    wall clock, and their source's rule, on that clock, gives k = 0 all year."""
    from epftoolbox2.data.sources import EpftoolboxBenchmarkSource

    index = pd.date_range("2016-01-01", "2016-12-31 23:00", freq="1h", tz="Etc/GMT-1")
    days = ["2016-01-15", "2016-07-15", "2016-12-15"]
    k = info_for("price published 13:00 D-1").newest_targets("price", "1h", index, days, 2)
    assert (k[:23] == 0).all()
    assert k[23].tolist() == [0, -1, 0]
    on_its_clock = InformationSet("11:45").with_source_rules([EpftoolboxBenchmarkSource("DE")])
    assert (on_its_clock.newest_targets("price", "1h", index, days, 2) == 0).all()


# ---------------------------------------------------------------------------
# Forecasts, by hand
# ---------------------------------------------------------------------------


def hand_forecast(data, freq, period, horizon, day, names, newest, window=WINDOW):
    """The forecast read from the frame with numpy alone: rows by date up to D + min(k - h, -1), W of them, targets by the wall
    clock (a repeated hour has none), the binary columns as they are and the others standardised on the rows that train."""
    wall = data.index.tz_localize(None)
    dates = wall.normalize()
    periods = wall.hour * (60 // STEP[freq]) + wall.minute // STEP[freq]
    D = pd.Timestamp(day)
    last = D + pd.Timedelta(days=min(newest - horizon, -1))
    first = last - pd.Timedelta(days=window - 1)
    here = np.asarray(periods == period)
    rows = np.flatnonzero(here & np.asarray((dates >= first) & (dates <= last)))
    prices = pd.Series(data["price"].to_numpy(), index=wall)
    once = prices[~prices.index.duplicated(keep=False)]
    target = once.reindex(wall[rows] + pd.Timedelta(days=horizon)).to_numpy()
    x = data[names].to_numpy(dtype=float)
    test_row = np.flatnonzero(here & np.asarray(dates == D))[-1]
    usable = np.isfinite(target) & np.isfinite(x[rows]).all(axis=1)
    y, train = target[usable], x[rows][usable]
    scaled = np.array(["is_week" not in name for name in names])
    means = np.where(scaled, train.mean(axis=0), 0.0)
    deviations = np.where(scaled, train.std(axis=0, ddof=1), 1.0)
    coefficients, *_ = np.linalg.lstsq((train - means) / deviations, (y - y.mean()) / y.std(ddof=1), rcond=None)
    return float(((x[test_row] - means) / deviations) @ coefficients * y.std(ddof=1) + y.mean()), dates[rows][usable]


@pytest.mark.parametrize("freq", ["1h", "15min"])
@pytest.mark.parametrize("switch", list(SWITCH_DAYS))
@pytest.mark.parametrize("case", list(CASES))
def test_a_published_forecast_equals_a_hand_fit_on_the_rows_its_rule_allows(case, switch, freq):
    """Issue days S-2 to S+3 of each switch, so both switch days and the day before the spring one, where the skipped hour's k
    is one day more. At 15 minutes the hour of the switch and the periods around 42 are fitted by hand, and every record's
    newest target day is checked."""
    H = 2
    data = market(switch, freq=freq, horizon=H)
    S = SWITCH_DAYS[switch]
    first, last = day_of(S, -2), day_of(S, 3)
    found = run(data, first, last, info_for(case), horizon=H, freq=freq)
    skipped_days = {pd.Timestamp(S).date()} if switch == "spring" else set()
    fitted = [*SKIPPED[freq], 0, 9, 10, 23] if freq == "1h" else [*SKIPPED[freq], 41, 42, 43, 44, 95]
    checked = 0
    for (run_date, period, h), record in found.items():
        day = pd.Timestamp(run_date).date()
        k = k_by_hand(newest_by_hand(case, freq, period), day, skipped_days if period in SKIPPED[freq] else set(), H)
        assert record["newest_target_day"] == day_of(run_date, min(k, h - 1)), (run_date, period, h)
        if freq == "1h" or period in fitted:
            names = [name.replace("{horizon}", str(h)) for name in PREDICTORS]
            prediction, dates = hand_forecast(data, freq, period, h, run_date, names, k)
            assert record["prediction"] == pytest.approx(prediction, abs=1e-9), (run_date, period, h)
            assert dates.max() < pd.Timestamp(run_date)  # no row dated on the issue day or later
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("h", [2, 3])
def test_on_the_day_before_a_spring_switch_the_skipped_hour_trains_on_w_1_rows(h):
    """Warsaw skips 02:00 on 2023-03-26. Issued on 2023-03-25, the switch hour's newest target is the switch day (k = 1): that
    slot has no value to be late, so the walk goes past it. The window is still W dates, ending on the row whose target is the
    skipped slot, which drops: W-1 rows, as "d-1" has for that hour on the day after the switch. `newest_target_day` is that
    date, a target that never trained. An hour the clock has keeps k = 0 and W rows."""
    W, day = 30, "2023-03-25"
    data = market("spring", horizon=3)
    model = OLSModel(PREDICTORS, training_window=W, newest_target="published")
    found = by_key(model.run(data, day, day, horizon=3, information_set=info_for("price published 13:00 D-1"))._results)
    names = [name.replace("{horizon}", str(h)) for name in PREDICTORS]

    prediction, dates = hand_forecast(data, "1h", 2, h, day, names, 1, window=W)
    assert len(dates) == W - 1 and dates.max() == pd.Timestamp(day_of(day, -h))
    assert found[(day, 2, h)]["prediction"] == pytest.approx(prediction, abs=1e-9)
    assert found[(day, 2, h)]["newest_target_day"] == "2023-03-26"
    one_day_short, dates = hand_forecast(data, "1h", 2, h, day, names, 0, window=W)
    assert len(dates) == W and found[(day, 2, h)]["prediction"] != pytest.approx(one_day_short, abs=1e-9)

    prediction, dates = hand_forecast(data, "1h", 3, h, day, names, 0, window=W)
    assert len(dates) == W
    assert found[(day, 3, h)]["prediction"] == pytest.approx(prediction, abs=1e-9)
    assert found[(day, 3, h)]["newest_target_day"] == day_of(day, min(0, h - 1))

    # "d-1" on the day after the switch: the same W-1 rows for that hour.
    d_1 = by_key(OLSModel(PREDICTORS, training_window=W).run(data, "2023-03-27", "2023-03-27", horizon=3)._results)
    prediction, dates = hand_forecast(data, "1h", 2, h, "2023-03-27", names, -1, window=W)
    assert len(dates) == W - 1
    assert d_1[("2023-03-27", 2, h)]["prediction"] == pytest.approx(prediction, abs=1e-9)


@pytest.mark.parametrize("case", list(CASES))
def test_the_newest_rows_are_those_of_the_table(case):
    """Newest row at h=1 per the spec: D-1 where k >= 0 and D-2 where k = -1, and never a row dated D (the cap at 14:00)."""
    data = market("autumn")
    day = "2023-10-10"
    found = run(data, day, day, info_for(case))
    for period in range(24):
        names = [name.replace("{horizon}", "1") for name in PREDICTORS]
        k = newest_by_hand(case, "1h", period)
        prediction, dates = hand_forecast(data, "1h", period, 1, day, names, k)
        assert dates.max() == pd.Timestamp(day_of(day, -1 if k >= 0 else -2))
        assert found[(day, period, 1)]["prediction"] == pytest.approx(prediction, abs=1e-9)
        assert found[(day, period, 1)]["newest_target_day"] == day_of(day, min(k, 0))


@pytest.mark.parametrize(
    ("newest_target", "binary"), [("published", False), ("d-1", True)], ids=["published: scaled", "d-1: binary"]
)
def test_the_binary_columns_are_read_from_the_rows_the_rule_allows(newest_target, binary):
    """A column of 0s and 1s with a number on D-1: in the span of a published forecast at h=1, so scaled, and after the span of
    "d-1", so left as it is. A mask read from the span of "d-1" would leave it unscaled in the published forecast."""
    data = make_market_data(start="2023-04-01", days=80, noise=5.0, seed=8, horizon=1)
    D = "2023-06-10"
    data["flag"] = np.random.default_rng(1).integers(0, 2, len(data)).astype(float)
    data.loc[data.index.strftime("%Y-%m-%d") == day_of(D, -1), "flag"] = 7.0
    names = ["load_d+1", "flag"]
    info = info_for("price published 13:00 D-1", flag=Availability.static())
    found = run(data, D, D, info, newest_target=newest_target, horizon=1, predictors=names)[(D, 5, 1)]

    wall = data.index.tz_localize(None)
    dates = wall.normalize()
    last = pd.Timestamp(D) - pd.Timedelta(days=1 if newest_target == "published" else 2)
    rows = np.flatnonzero((wall.hour == 5) & (dates >= last - pd.Timedelta(days=WINDOW - 1)) & (dates <= last))
    x = data[names].to_numpy(dtype=float)[rows]
    y = calendar_shift(data["price"], 1).to_numpy()[rows]
    assert (set(x[:, 1]) == {0.0, 1.0}) == binary
    means = np.array([x[:, 0].mean(), 0.0 if binary else x[:, 1].mean()])
    deviations = np.array([x[:, 0].std(ddof=1), 1.0 if binary else x[:, 1].std(ddof=1)])
    coefficients, *_ = np.linalg.lstsq((x - means) / deviations, (y - y.mean()) / y.std(ddof=1), rcond=None)
    test = data.loc[f"{D} 05:00", names].to_numpy(dtype=float)
    assert found["prediction"] == pytest.approx(((test - means) / deviations) @ coefficients * y.std(ddof=1) + y.mean())


# ---------------------------------------------------------------------------
# Two-sided perturbation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("day", ["2023-10-10", day_of(SWITCH_DAYS["autumn"], 1), day_of(SWITCH_DAYS["autumn"])])
def test_moving_a_known_target_moves_the_published_forecast_and_an_unknown_one_moves_nothing(day):
    """`realised 1h` at 11:45: the load of D is known up to period 9. Moving it at period 5 moves that period's published
    forecasts (h=1 trains on the row of D-1, h=2 on the row of D-2, both with a target on D) and no "d-1" one; moving it at
    period 12, or any cell of D+1, moves nothing."""
    data = market("autumn")
    info = info_for("load realised 1h")
    published, d_1 = run(data, day, day, info), run(data, day, day, info, newest_target="d-1")

    def moved(cells):
        changed = data.copy()
        for when in cells:
            changed.loc[when, "price"] += 100.0
        new_published, new_d_1 = run(changed, day, day, info), run(changed, day, day, info, newest_target="d-1")
        return (
            {key for key in published if new_published[key]["prediction"] != published[key]["prediction"]},
            {key for key in d_1 if new_d_1[key]["prediction"] != d_1[key]["prediction"]},
        )

    assert moved([f"{day} 05:00"]) == ({(day, 5, 1), (day, 5, 2)}, set())
    assert moved([f"{day} 12:00"]) == (set(), set())
    assert moved([f"{day_of(day, 1)} {hour:02d}:00" for hour in range(24)]) == (set(), set())
    unknown, counts, _ = perturb_unknown(data, day, info, columns=["price"])
    assert counts["price"] > 24
    after = run(unknown, day, day, info)
    assert {key: r["prediction"] for key, r in after.items()} == {key: r["prediction"] for key, r in published.items()}


@pytest.mark.parametrize("case", ["load realised 1h", "price published 13:00 D-1"])
def test_one_day_more_than_the_rule_allows_is_a_leak_for_verify(monkeypatch, case):
    data = market("autumn")
    pipeline = ModelPipeline(information_set=info_for(case))
    pipeline.add_model(OLSModel(PREDICTORS, training_window=WINDOW, newest_target="published", name="m"))
    report = pipeline.verify(data, "2023-10-10", "2023-10-10", horizon=2, days=1)
    report.assert_ok()

    monkeypatch.setattr(InformationSet, "newest_targets", one_day_more)
    report = pipeline.verify(data, "2023-10-10", "2023-10-10", horizon=2, days=1, positive_control=False)
    assert report.status("m") == "leak"
    changed = report.checks[0].probe.changed
    # At h=1 the cap keeps k + 1 = 1 of the prices at D-1; at h=2 it trains on the price of D+1. The load of D past period 9
    # leaks at h=1 already.
    expected = {(p, 2) for p in range(24)} if case.startswith("price") else {(p, 1) for p in range(10, 24)}
    assert expected <= set(changed)


# ---------------------------------------------------------------------------
# Where the rule gives -1, "published" is "d-1"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("freq", ["1h", "15min"])
@pytest.mark.parametrize("switch", list(SWITCH_DAYS))
def test_published_equals_d_1_bit_for_bit_where_the_rule_gives_minus_one(switch, freq):
    """Prices published 13:00 on their own day: at 11:45 D's are out on no period, D-1's on every one."""
    data = market(switch, freq=freq)
    S = SWITCH_DAYS[switch]
    info = InformationSet("11:45", rules={**INPUT_RULES, "price": Availability.published("13:00", 0)})
    days = [d.date() for d in pd.date_range(day_of(S, -2), day_of(S, 3))]
    k = info.newest_targets("price", freq, data.index, days, 2).copy()
    # Every planned key; the spring day's skipped hour, where k is 0 having no value of its own, has no row to issue from.
    off_the_clock = [(p, days.index(pd.Timestamp(S).date())) for p in SKIPPED[freq]] if switch == "spring" else []
    for period, position in off_the_clock:
        assert k[period, position] == 0
        k[period, position] = -1
    assert (k == -1).all()

    published = run(data, day_of(S, -2), day_of(S, 3), info, freq=freq)
    d_1 = run(data, day_of(S, -2), day_of(S, 3), info, newest_target="d-1", freq=freq)
    assert published.keys() == d_1.keys()
    for key, record in published.items():
        assert record.pop("newest_target_day") == day_of(key[0], -1)
        assert json.dumps(record) == json.dumps(d_1[key])


# ---------------------------------------------------------------------------
# Resume, manifests, YAML
# ---------------------------------------------------------------------------


def lines(path):
    return sorted(line for line in Path(path).read_text().splitlines() if line)


def pipeline_of(info, level=None, newest_target="published", **model):
    pipeline = ModelPipeline(information_set=info, availability=level)
    return pipeline.add_model(
        OLSModel(PREDICTORS, training_window=WINDOW, newest_target=newest_target, name="m", **model)
    ).add_evaluator(MAEEvaluator())


def crash(path, kept):
    """What a run that stopped after `kept` forecasts leaves: the first of them in the file."""
    path = Path(path)
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:kept]))


def test_a_level_change_resumes_and_moves_nothing(tmp_path):
    data = market("autumn")
    info = info_for("load realised 1h")
    test = ("2023-10-27", "2023-10-31")
    pipeline_of(info, "warn").run(data, test=test, horizon=2, save_dir=tmp_path / "a")
    crash(tmp_path / "a" / "m.jsonl", 100)
    first = lines(tmp_path / "a" / "m.jsonl")
    pipeline_of(info, "none").run(data, test=test, horizon=2, save_dir=tmp_path / "a")
    crash(tmp_path / "a" / "m.jsonl", 170)
    pipeline_of(info, "raise").run(data, test=test, horizon=2, save_dir=tmp_path / "a")
    pipeline_of(info, "raise").run(data, test=test, horizon=2, save_dir=tmp_path / "b")
    resumed = lines(tmp_path / "a" / "m.jsonl")
    assert set(first) <= set(resumed)
    assert resumed == lines(tmp_path / "b" / "m.jsonl")
    assert len(resumed) == 5 * 2 * 24
    assert not rows_unverified(manifest_path(tmp_path / "a" / "m.jsonl"))
    cut = json.loads(manifest_path(tmp_path / "a" / "m.jsonl").read_text())["training_cut"]
    assert cut == {
        "rows": "dates",
        "newest_target": "published",
        "target_rule": {"kind": "realised", "delay": "1h"},
        "issue_time": "11:45",
        "issue_timezone": "Europe/Brussels",
    }


@pytest.mark.parametrize(
    "other",
    [
        InformationSet("11:00", "Europe/Brussels", rules={**INPUT_RULES, "price": Availability.realised("1h")}),
        InformationSet("11:45", "Europe/Warsaw", rules={**INPUT_RULES, "price": Availability.realised("1h")}),
        InformationSet("11:45", "Europe/Brussels", rules={**INPUT_RULES, "price": Availability.realised("2h")}),
    ],
    ids=["issue time", "issue zone", "target rule"],
)
def test_changing_the_issue_time_or_the_target_rule_refuses_a_resume(tmp_path, other):
    data = market("autumn")
    pipeline_of(info_for("load realised 1h")).run(data, test=("2023-10-27", "2023-10-28"), horizon=2, save_dir=tmp_path)
    with pytest.raises(ManifestMismatchError, match="training_cut"):
        pipeline_of(other).run(data, test=("2023-10-27", "2023-10-28"), horizon=2, save_dir=tmp_path)
    # A rule of another column, a note on the target's rule and "d-1" models don't care.
    noted = Availability.realised("1h", note="A65 actual load")
    pipeline_of(info_for("load realised 1h", price=noted, early=Availability.static())).run(
        data, test=("2023-10-27", "2023-10-28"), horizon=2, save_dir=tmp_path
    )
    pipeline_of(info_for("load realised 1h"), newest_target="d-1").run(
        data, test=("2023-10-27", "2023-10-28"), horizon=2, save_dir=tmp_path / "d1"
    )
    pipeline_of(other, newest_target="d-1").run(
        data, test=("2023-10-27", "2023-10-28"), horizon=2, save_dir=tmp_path / "d1"
    )
    # Nor may one cut resume into the other's results.
    with pytest.raises(ManifestMismatchError, match="training_cut"):
        pipeline_of(info_for("load realised 1h"), newest_target="d-1").run(
            data, test=("2023-10-27", "2023-10-28"), horizon=2, save_dir=tmp_path
        )


def without_provenance(path):
    manifest = json.loads(Path(path).read_text())
    for key in ("created_at", "updated_at", "git_sha", "package_version"):
        manifest.pop(key)
    return manifest


@pytest.mark.parametrize("info", [None, info_for("price published 13:00 D-1")], ids=["no information set", "with one"])
def test_an_explicit_d_1_is_byte_identical_to_leaving_it_out(tmp_path, info):
    data = market("spring")
    reports = {}
    for name, keyword in [("omitted", {}), ("explicit", {"newest_target": "d-1"})]:
        pipeline = ModelPipeline(information_set=info)
        pipeline.add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="m", **keyword))
        pipeline.add_evaluator(MAEEvaluator())
        reports[name] = pipeline.run(data, test=("2023-03-24", "2023-03-28"), horizon=2, save_dir=tmp_path / name)
        pipeline.save(tmp_path / f"{name}.yaml")
    assert lines(tmp_path / "omitted" / "m.jsonl") == lines(tmp_path / "explicit" / "m.jsonl")
    assert "newest_target" not in (tmp_path / "omitted" / "m.jsonl").read_text()
    assert without_provenance(manifest_path(tmp_path / "omitted" / "m.jsonl")) == without_provenance(
        manifest_path(tmp_path / "explicit" / "m.jsonl")
    )
    assert json.loads(manifest_path(tmp_path / "omitted" / "m.jsonl").read_text())["training_cut"] == {"rows": "dates"}
    pd.testing.assert_frame_equal(reports["omitted"].summary(), reports["explicit"].summary())
    pd.testing.assert_frame_equal(reports["omitted"].forecast_counts(), reports["explicit"].forecast_counts())
    assert (tmp_path / "omitted.yaml").read_bytes() == (tmp_path / "explicit.yaml").read_bytes()
    assert b"newest_target" not in (tmp_path / "omitted.yaml").read_bytes()


@pytest.mark.parametrize(
    ("case", "switch", "validation", "test", "H"),
    [
        ("load realised 1h", "autumn", ("2023-10-20", "2023-10-23"), ("2023-10-28", "2023-10-31"), 2),
        ("price published 13:00 D-1", "spring", ("2023-03-18", "2023-03-21"), ("2023-03-24", "2023-03-27"), 3),
    ],
    ids=["k the same every day", "k one more on the day before the spring switch"],
)
def test_with_validation_each_window_is_what_a_plain_run_over_it_gives(tmp_path, case, switch, validation, test, H):
    """The worker reads k by the task's place among the run's issue days, which skip the days between the windows. Only a k that
    differs between days can tell a place from a calendar day: in spring the switch hour's k is 1 on 2023-03-25 and 0 on every
    other day, so a table read over every calendar day from the first issue day would give 2023-03-27 the k of 2023-03-25, and at
    h=2 train on the prices of 2023-03-28, published at 13:00 on 2023-03-27, after the issue."""
    data = market(switch, horizon=H)
    info = info_for(case)
    issue_days = [d.date() for p in (validation, test) for d in pd.date_range(*p)]
    k = info.newest_targets("price", "1h", data.index, issue_days, H)
    if switch == "spring":
        assert k[2].tolist() == [0, 0, 0, 0, 0, 1, 0, 0] and (k[3] == 0).all()
    both = pipeline_of(info).run(data, validation=validation, test=test, horizon=H, save_dir=tmp_path / "both")
    stored = [json.loads(line) for line in lines(tmp_path / "both" / "m.jsonl")]
    for window, period in [("validation", validation), ("test", test)]:
        plain = pipeline_of(info).run(data, test=period, horizon=H, save_dir=tmp_path / window)
        assert plain.refs["m"].produced == plain.refs["m"].count
        own = sorted(json.dumps({k: v for k, v in r.items() if k != "window"}) for r in stored if r["window"] == window)
        assert own == sorted(json.dumps(json.loads(line)) for line in lines(tmp_path / window / "m.jsonl"))
    assert both.refs["m"].produced == both.refs["m"].count
    if switch == "spring":
        newest = {(r["run_date"], r["hour"], r["horizon"]): r["newest_target_day"] for r in stored}
        assert newest[("2023-03-27", 2, 2)] == "2023-03-27"
        assert newest[("2023-03-25", 2, 2)] == "2023-03-26"


def test_published_is_saved_to_yaml_and_loads_back(tmp_path):
    pipeline = pipeline_of(info_for("price published 13:00 D-1"))
    pipeline.add_model(LassoCVModel(PREDICTORS, newest_target="published", name="lasso"))
    pipeline.save(tmp_path / "p.yaml")
    assert (tmp_path / "p.yaml").read_text().count("newest_target: published") == 2
    loaded = ModelPipeline.load(tmp_path / "p.yaml")
    assert [model.newest_target for model in loaded.models] == ["published", "published"]
    loaded.save(tmp_path / "again.yaml")
    assert (tmp_path / "again.yaml").read_bytes() == (tmp_path / "p.yaml").read_bytes()


def test_explain_shows_the_newest_target_of_the_forecast_shown():
    data = market("autumn")
    pipeline = pipeline_of(info_for("load realised 1h"))
    pipeline.add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="d1"))
    for hour, target, row in [(5, "D", "D-1"), (12, "D-1", "D-2")]:
        shown = pipeline.explain(data, "m", test=("2023-10-10", "2023-10-12"), hour=hour, echo=False)
        assert shown["newest_target"].to_dict("records") == [
            {"newest target": target, "newest row": row, "issue days": 3}
        ]
    shown = pipeline.explain(data, "m", test=("2023-10-10", "2023-10-12"), horizon=2, hour=5, echo=False)
    assert shown["newest_target"].to_dict("records") == [{"newest target": "D", "newest row": "D-2", "issue days": 3}]
    assert "newest_target" not in pipeline.explain(data, "d1", test=("2023-10-10", "2023-10-12"), echo=False)


# ---------------------------------------------------------------------------
# train_on="matching"
# ---------------------------------------------------------------------------

TWO = ["load_d+{horizon}", "wind_d+{horizon}"]
MATCHING_RULES = {"signal": Availability.published("10:00", 0), "early": Availability.published("10:15", 0)}


def by_signal(ctx):
    """Its list depends on a value of the row's day before, so a training day's list is what the function said as of that
    day's issue time."""
    return Use([*TWO, "early"] if ctx.value("signal_d-1") > 0 else TWO, train_on="matching")


@pytest.fixture(scope="module")
def signal_frame():
    data = make_market_data(start="2023-08-01", days=110, noise=5.0, seed=5, horizon=2)
    rng = np.random.default_rng(2)
    data["signal"] = rng.normal(0, 1, len(data))
    data["signal_d-1"] = calendar_shift(data["signal"], -1)
    data["early"] = rng.normal(0, 1, len(data))
    return data


def matching_by_hand(data, period, horizon, day, newest):
    """The matching forecast with numpy and scikit-learn: the rows of the span its rule allows whose day had the same list, the
    target by the wall clock, every column standardised on the rows that train."""
    wall = data.index.tz_localize(None)
    dates = wall.normalize()
    D = pd.Timestamp(day)
    last = D + pd.Timedelta(days=min(newest - horizon, -1))
    here = np.asarray(wall.hour == period)
    rows = np.flatnonzero(here & np.asarray((dates >= last - pd.Timedelta(days=WINDOW - 1)) & (dates <= last)))
    test_row = np.flatnonzero(here & np.asarray(dates == D))[-1]
    signal = data["signal_d-1"].to_numpy()
    rows = rows[(signal[rows] > 0) == (signal[test_row] > 0)]
    names = [n.replace("{horizon}", str(horizon)) for n in ([*TWO, "early"] if signal[test_row] > 0 else TWO)]
    prices = pd.Series(data["price"].to_numpy(), index=wall)
    once = prices[~prices.index.duplicated(keep=False)]
    y = once.reindex(wall[rows] + pd.Timedelta(days=horizon)).to_numpy()
    x = data[names].to_numpy(dtype=float)
    usable = np.isfinite(y) & np.isfinite(x[rows]).all(axis=1)
    train, y = x[rows][usable], y[usable]
    means, deviations = train.mean(axis=0), train.std(axis=0, ddof=1)
    fit = LinearRegression(fit_intercept=False).fit((train - means) / deviations, (y - y.mean()) / y.std(ddof=1))
    prediction = fit.predict(((x[test_row] - means) / deviations)[None, :])[0] * y.std(ddof=1) + y.mean()
    return float(prediction), len(set(dates[rows][usable]))


def test_a_published_matching_forecast_is_a_hand_fit_on_its_set(signal_frame):
    info = info_for("load realised 1h", **MATCHING_RULES)
    model = OLSModel(by_signal, training_window=WINDOW, newest_target="published")
    found = by_key(model.run(signal_frame, "2023-10-27", "2023-10-31", horizon=2, information_set=info)._results)
    assert len(found) == 5 * 2 * 24
    for (day, period, h), record in found.items():
        if period not in (2, 5, 9, 10, 17):
            continue
        prediction, days = matching_by_hand(
            signal_frame, period, h, day, newest_by_hand("load realised 1h", "1h", period)
        )
        assert record["prediction"] == pytest.approx(prediction, abs=1e-9), (day, period, h)
        assert record["training_days"] == days
    # What explain() and the warning about thin sets count before any fit is what the worker trained on.
    pipeline = ModelPipeline(information_set=info).add_model(
        OLSModel(by_signal, training_window=WINDOW, newest_target="published", name="m")
    )
    lists = pipeline.explain(signal_frame, "m", test=("2023-10-27", "2023-10-31"), horizon=2, echo=False)["lists"]
    for row in lists.to_dict("records"):
        days = [
            r["training_days"]
            for r in found.values()
            if r["horizon"] == row["horizon"] and r["predictor_list"] == row["hash"]
        ]
        assert (row["smallest days"], row["median days"], row["largest days"]) == (
            min(days),
            np.median(days),
            max(days),
        )


def test_a_published_matching_run_resumes_bit_for_bit_and_its_set_is_stable_under_the_probe(signal_frame, tmp_path):
    info = info_for("load realised 1h", **MATCHING_RULES)

    def pipeline():
        return ModelPipeline(information_set=info).add_model(
            OLSModel(by_signal, training_window=WINDOW, newest_target="published", name="m")
        )

    test = ("2023-10-27", "2023-11-01")
    pipeline().run(signal_frame, test=test, horizon=2, save_dir=tmp_path / "a")
    crash(tmp_path / "a" / "m.jsonl", 111)
    pipeline().run(signal_frame, test=test, horizon=2, save_dir=tmp_path / "a")
    pipeline().run(signal_frame, test=test, horizon=2, save_dir=tmp_path / "b")
    assert lines(tmp_path / "a" / "m.jsonl") == lines(tmp_path / "b" / "m.jsonl")
    assert len(lines(tmp_path / "a" / "m.jsonl")) == 6 * 2 * 24

    day = "2023-10-30"
    model = OLSModel(by_signal, training_window=WINDOW, newest_target="published")
    original = by_key(model.run(signal_frame, day, day, horizon=2, information_set=info)._results)
    perturbed, counts, _ = perturb_unknown(signal_frame, day, info)
    assert counts["price"] and counts["signal"] and counts["early"]
    after = by_key(model.run(perturbed, day, day, horizon=2, information_set=info)._results)
    assert {k: (r["matching_days"], r["prediction"]) for k, r in after.items()} == {
        k: (r["matching_days"], r["prediction"]) for k, r in original.items()
    }
    probe = leakage_probe(model, signal_frame, day, info, horizon=2)
    probe.assert_ok()
    pipeline().verify(signal_frame, "2023-10-27", "2023-11-01", horizon=2, days=2).assert_ok()


def test_a_published_matching_forecast_trained_one_day_late_is_a_leak(signal_frame, monkeypatch):
    info = info_for("load realised 1h", **MATCHING_RULES)
    pipeline = ModelPipeline(information_set=info).add_model(
        OLSModel(by_signal, training_window=WINDOW, newest_target="published", name="m")
    )
    monkeypatch.setattr(InformationSet, "newest_targets", one_day_more)
    report = pipeline.verify(signal_frame, "2023-10-30", "2023-10-30", horizon=2, days=1, positive_control=False)
    assert report.status("m") == "leak"


def test_a_target_older_than_d_1_reaches_back_further_and_reads_those_days(signal_frame):
    """Prices out four days after delivery: k = -4, so the span starts 3 days before the one of "d-1", and the training days a
    matching list asks about and the days the frame is checked on reach back as far."""
    info = InformationSet(
        "11:45", rules={**INPUT_RULES, **MATCHING_RULES, "price": Availability.published("12:00", -3)}
    )
    day = "2023-10-30"
    model = OLSModel(by_signal, training_window=WINDOW, newest_target="published")
    plan = model._plan_predictors(signal_frame, "price", 2, "1h", info, day, day)
    assert plan.reach == 3
    assert plan.first_grid_day == pd.Timestamp(day).date() - datetime.timedelta(days=WINDOW + 2 + 3)
    found = by_key(model.run(signal_frame, day, day, horizon=2, information_set=info)._results)
    for (run_date, period, h), record in found.items():
        assert record["newest_target_day"] == day_of(run_date, -4)
        if period in (0, 13):
            prediction, days = matching_by_hand(signal_frame, period, h, run_date, -4)
            assert record["prediction"] == pytest.approx(prediction, abs=1e-9)
            assert record["training_days"] == days
    assert NewestTargets([day], np.full((24, 1), -4)).reach == 3
    assert NewestTargets([day], np.full((24, 1), 0)).reach == 0


# ---------------------------------------------------------------------------
# verify and the probes see through "published"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("switch", list(SWITCH_DAYS))
@pytest.mark.parametrize(
    ("case", "target_as_input"),
    [("load realised 1h", []), ("price published 13:00 D-1", ["price"])],
    ids=["plain", "target as an input"],
)
def test_verify_is_ok_for_a_published_model_on_the_days_of_a_switch(switch, case, target_as_input):
    """The probe perturbs what the target's rule calls unknown, not the target from D on, which a published model trains on
    legally; a copy of the target trained on has the target's rule. The positive control moves every forecast."""
    data = market(switch)
    S = SWITCH_DAYS[switch]
    pipeline = ModelPipeline(information_set=info_for(case))
    pipeline.add_model(
        OLSModel(PREDICTORS + target_as_input, training_window=WINDOW, newest_target="published", name="m")
    )
    report = pipeline.verify(data, day_of(S, 0), day_of(S, 3), horizon=2, days=4)
    report.assert_ok()
    assert all(not check.control.unmoved for check in report.checks)


def test_the_planted_leak_is_caught_with_the_target_as_an_input(monkeypatch):
    """The copy of the target the probe trains on decides its own newest targets: with a static rule it would train on every
    price up to D+1, and the probe, which perturbs what that rule calls unknown, would see nothing."""
    data = market("autumn")
    pipeline = ModelPipeline(information_set=info_for("price published 13:00 D-1"))
    pipeline.add_model(OLSModel(PREDICTORS + ["price"], training_window=WINDOW, newest_target="published", name="m"))
    pipeline.verify(data, "2023-10-10", "2023-10-10", horizon=2, days=1).assert_ok()
    monkeypatch.setattr(InformationSet, "newest_targets", one_day_more)
    report = pipeline.verify(data, "2023-10-10", "2023-10-10", horizon=2, days=1, positive_control=False)
    assert report.status("m") == "leak"
    assert {(p, 2) for p in range(24)} <= set(report.checks[0].probe.changed)


def test_the_copy_of_the_target_has_the_target_s_newest_targets(monkeypatch):
    """A model that reads its target is probed and controlled on a copy of it, trained on under another name. Under "published"
    the copy has the target's rule, so it trains on the rows the model does: with a static rule it would train on every target
    up to D+1 and the control would test another model. The main probe still catches a planted leak either way (above), so this
    is held here."""
    data = market("autumn")
    info = info_for("price published 13:00 D-1")
    seen = []
    control = verification_module.positive_control

    def spy(model, frame, run_day, rules, target, **kwargs):
        seen.append((rules, target, frame.index, run_day))
        return control(model, frame, run_day, rules, target, **kwargs)

    monkeypatch.setattr(verification_module, "positive_control", spy)
    pipeline = ModelPipeline(information_set=info)
    pipeline.add_model(OLSModel(PREDICTORS + ["price"], training_window=WINDOW, newest_target="published", name="m"))
    pipeline.verify(data, "2023-10-10", "2023-10-10", horizon=2, days=1).assert_ok()
    rules, label, index, day = seen[0]
    assert label == "price__as_target"
    copy = rules.newest_targets(label, "1h", index, [day], 2)
    assert np.array_equal(copy, info.newest_targets("price", "1h", index, [day], 2))
    assert (copy == 0).all()


def test_the_probe_refuses_the_target_boundary_for_a_published_model_and_gives_its_run_the_rules_alone():
    data = market("autumn")
    info = info_for("price published 13:00 D-1")
    model = OLSModel(PREDICTORS, training_window=WINDOW, newest_target="published")
    with pytest.raises(ValueError, match="check_target_boundary"):
        leakage_probe(model, data, "2023-10-10", info, horizon=2, check_target_boundary=True)
    with pytest.raises(ValueError, match="check_target_boundary"):
        leakage_probe(
            lambda: OLSModel(PREDICTORS, newest_target="published"),
            data,
            "2023-10-10",
            info,
            check_target_boundary=True,
        )
    leakage_probe(model, data, "2023-10-10", info, horizon=2).assert_ok()
    # The run gets the rules for its cut and nothing else: a late input isn't refused, it is seen to move.
    data["load_d+8"] = calendar_shift(data["load"], 8)
    late = OLSModel(PREDICTORS + ["load_d+8"], training_window=WINDOW, newest_target="published")
    probe = leakage_probe(late, data, "2023-10-10", info, horizon=2)
    assert not probe.ok and len(probe.changed) == 48
    with pytest.raises(ValueError, match="information_set or cut_rules"):
        late.run(data, "2023-10-10", "2023-10-10", information_set=info, cut_rules=info.copy())


@pytest.mark.parametrize("target_as_input", [[], ["price"]], ids=["plain", "target as an input"])
def test_the_positive_control_can_fail_for_a_published_model(target_as_input):
    """A model that reads only its training rows can't be reached through the row it forecasts from, so the control must move
    nothing. Under "published" the row of D-1 trains at h=1: a control that changed it would move this model, and every model,
    and never fail."""
    data = market("autumn")
    pipeline = ModelPipeline(information_set=info_for("price published 13:00 D-1"))
    pipeline.add_model(
        TrainingRowsOnly(PREDICTORS + target_as_input, training_window=WINDOW, newest_target="published", name="m")
    )
    report = pipeline.verify(data, "2023-10-10", "2023-10-11", horizon=2, days=2)
    assert report.status("m") == "inconclusive"
    assert all(check.control.moved == [] and check.control.unmoved for check in report.checks)
    assert all(check.probe.ok for check in report.checks)


def test_the_control_of_a_published_model_changes_no_row_before_the_issue_day():
    data = market("autumn")
    info = info_for("price published 13:00 D-1")
    day = "2023-10-10"
    dates = data.index.tz_localize(None).normalize()
    for first_row_day, oldest in [(-1, day_of(day, -1)), (0, day)]:
        perturbed, counts, _ = perturb_recent(data, day, info, first_row_day=first_row_day)
        same = (perturbed == data) | (perturbed.isna() & data.isna())
        changed = ~same.all(axis=1).to_numpy()
        assert counts and dates[changed].min() == pd.Timestamp(oldest)


def rebuild(source):
    """The model's inputs built from source data: a day lag of the price and the day-ahead load and wind."""
    out = source.copy()
    out["price_d-1"] = calendar_shift(out["price"], -1)
    for h in (1, 2):
        for column in ("load", "wind", "is_weekend", "is_weekday"):
            out[f"{column}_d+{h}"] = calendar_shift(out[column], h)
    return out


def test_with_rebuild_the_control_reaches_yesterday_s_price_and_no_training_row():
    """Under "published" the built row of D-1 trains, so every built row before D is put back, while the raw prices of D-1
    still change: `price_d-1` on D, the newest price lag, is what moves a model that has nothing fresher."""
    source = market("autumn")[["price", "load", "wind", "is_weekend", "is_weekday"]]
    info = info_for("price published 13:00 D-1")
    day = "2023-10-10"
    lags_only = OLSModel(
        ["price_d-1", "is_weekend_d+{horizon}", "is_weekday_d+{horizon}"],
        training_window=WINDOW,
        newest_target="published",
    )
    moved = positive_control(lags_only, source, day, info, horizon=2, rebuild=rebuild)
    assert len(moved.moved) == 48
    blind = TrainingRowsOnly(PREDICTORS, training_window=WINDOW, newest_target="published")
    assert positive_control(blind, source, day, info, horizon=2, rebuild=rebuild).moved == []

    pipeline = ModelPipeline(information_set=info).add_model(
        OLSModel(["price_d-1", *PREDICTORS], training_window=WINDOW, newest_target="published", name="m")
    )
    pipeline.verify(source, day, day, horizon=2, days=1, rebuild=rebuild).assert_ok()


def test_with_rebuild_the_planted_leak_is_caught(monkeypatch):
    source = market("autumn")[["price", "load", "wind", "is_weekend", "is_weekday"]]
    pipeline = ModelPipeline(information_set=info_for("price published 13:00 D-1")).add_model(
        OLSModel(["price_d-1", *PREDICTORS], training_window=WINDOW, newest_target="published", name="m")
    )
    monkeypatch.setattr(InformationSet, "newest_targets", one_day_more)
    report = pipeline.verify(
        source, "2023-10-10", "2023-10-10", horizon=2, days=1, positive_control=False, rebuild=rebuild
    )
    assert report.status("m") == "leak"


def test_a_published_model_lists_no_training_target_and_d_1_does():
    """The check of the newest training target is the boundary of "d-1": for "published" it is known by construction, so a rule
    that makes D-1 late refuses "d-1" and not "published"."""
    data = market("autumn")
    info = InformationSet("11:45", rules={**INPUT_RULES, "price": Availability.published("14:30", -1)})
    with pytest.raises(InformationSetError, match="training target"):
        run(data, "2023-10-10", "2023-10-10", info, newest_target="d-1")
    found = run(data, "2023-10-10", "2023-10-10", info)
    assert {r["newest_target_day"] for r in found.values()} == {"2023-10-08"}

    for function in (None, lambda ctx: PREDICTORS):
        predictors = PREDICTORS if function is None else function
        published = ModelPipeline(information_set=info).add_model(
            OLSModel(predictors, training_window=WINDOW, newest_target="published", name="m")
        )
        d_1 = ModelPipeline(information_set=info).add_model(OLSModel(predictors, training_window=WINDOW, name="m"))
        assert published.run(data, test=("2023-10-10", "2023-10-10"), horizon=2).availability().empty
        listed = d_1.run(data, test=("2023-10-10", "2023-10-10"), horizon=2).availability()
        assert set(listed["role"]) == {"training target"}


def test_a_table_is_computed_once_for_what_it_is_computed_from(monkeypatch):
    data = market("autumn")
    info = info_for("load realised 1h")
    calls = []
    compute = InformationSet._newest_targets

    def counted(self, *args):
        calls.append(args[-1])
        return compute(self, *args)

    monkeypatch.setattr(InformationSet, "_newest_targets", counted)
    monkeypatch.setattr(information_set_module, "_NEWEST_TABLES", {})
    pipeline = pipeline_of(info)
    pipeline.run(data, test=("2023-10-10", "2023-10-11"), horizon=2)
    assert len(calls) == 1
    pipeline_of(info_for("load realised 1h")).run(data, test=("2023-10-10", "2023-10-11"), horizon=2)
    assert len(calls) == 1  # an equal information set
    pipeline_of(info).run(data, test=("2023-10-10", "2023-10-11"), horizon=1)
    pipeline_of(info_for("price issued 14:00")).run(data, test=("2023-10-10", "2023-10-11"), horizon=2)
    assert len(calls) == 3
    table = info.newest_targets("price", "1h", data.index, ["2023-10-10"], 2)
    with pytest.raises(ValueError):
        table[0, 0] = 5
