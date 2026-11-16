"""Checks before export (step 8f, `epftoolbox2.results.checks`): `ForecastChecks` on one member-day, the periods the clock gives
a day (switch days at 1h, 15 min and 1D), `check_records` over held records, and `_Store.overrule` in both formats.

What a day fails is compared with an oracle that knows nothing of the library: a local date's periods come from
`pd.date_range` in the zone, and each check is written out with pandas on the day's predictions.
"""

import datetime
import json
import math

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.models._worker import forecast_record, skip_record
from epftoolbox2.results.checks import (
    CHECK_FAILED,
    CHECK_NAMES,
    ForecastChecks,
    check_failed_record,
    check_records,
    day_products,
    planned_periods,
)
from epftoolbox2.results.store import open_store, read_records

ZONE = "Europe/Berlin"
SPRING, AUTUMN, PLAIN = "2026-03-29", "2026-10-25", "2026-10-21"


@pytest.fixture(autouse=True)
def offline(no_network):
    yield


def plus(day, days):
    return (datetime.date.fromisoformat(day) + datetime.timedelta(days=days)).isoformat()


def wall_periods(day, freq, zone=ZONE):
    """The oracle: a local date's period ids from its instants in the zone (a repeated hour's id once), and their count."""
    if freq == "1D":
        return [0], 1
    start, end = pd.Timestamp(day).tz_localize(zone), pd.Timestamp(plus(day, 1)).tz_localize(zone)
    stamps = pd.date_range(start, end, freq="h" if freq == "1h" else "15min", inclusive="left")
    per_hour = 1 if freq == "1h" else 4
    ids = stamps.hour * per_hour + stamps.minute // 15 * (per_hour > 1)
    return sorted(set(ids.tolist())), len(stamps)


# ---------------------------------------------------------------------------------------------------------------------
# The clock's periods
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "day, freq, periods, products",
    [
        (SPRING, "15min", 92, 92),
        (PLAIN, "15min", 96, 96),
        (AUTUMN, "15min", 96, 100),
        (SPRING, "1h", 23, 23),
        (PLAIN, "1h", 24, 24),
        (AUTUMN, "1h", 24, 25),
        (SPRING, "1D", 1, 1),
        (AUTUMN, "1D", 1, 1),
    ],
)
def test_the_periods_of_a_switch_day_are_the_clock_s(day, freq, periods, products):
    planned = planned_periods(freq, ZONE)(plus(day, -1), 1)  # issued the day before, on an ordinary day
    ids, instants = wall_periods(day, freq)
    assert list(planned) == ids and len(planned) == periods
    assert day_products(day, freq, ZONE) == instants == products


def test_a_forecast_issued_on_a_spring_day_lacks_its_skipped_period_on_every_target_day():
    for freq, gone in (("1h", {2}), ("15min", {8, 9, 10, 11})):
        for horizon in (1, 3):
            planned = planned_periods(freq, ZONE)(SPRING, horizon)
            ids, _ = wall_periods(plus(SPRING, horizon), freq)
            assert set(planned) == set(ids) - gone
    assert planned_periods("1h", None)(SPRING, 0) == tuple(range(24))  # no zone: every period
    assert day_products(AUTUMN, "15min", None) == 96


# ---------------------------------------------------------------------------------------------------------------------
# One member-day
# ---------------------------------------------------------------------------------------------------------------------


def oracle(values, planned, checks):
    """The checks a day fails, written out on its own: `values` period -> prediction (None: no value)."""
    order = sorted(planned)
    held = pd.Series({p: (np.nan if values[p] is None else values[p]) for p in order if p in values}, dtype=float)
    finite = held[np.isfinite(held)]
    failed = []
    if checks["periods"] and len(held) < len(order):
        failed.append("periods")
    if checks["finite"] and len(finite) < len(held):
        failed.append("finite")
    low, high = checks["range"] or (None, None)
    if checks["range"] and (
        ((finite < low).any() if low is not None else False) or ((finite > high).any() if high is not None else False)
    ):
        failed.append("range")
    if checks["not_constant"] and len(finite) >= 2 and finite.nunique() == 1:
        failed.append("not_constant")
    if checks["max_step"] is not None:
        pairs = [(a, b) for a, b in zip(order, order[1:]) if a in finite.index and b in finite.index]
        if any(abs(finite[b] - finite[a]) > checks["max_step"] for a, b in pairs):
            failed.append("max_step")
    return failed


SETTINGS = [
    {"periods": True, "finite": True, "range": None, "not_constant": True, "max_step": None},
    {"periods": False, "finite": False, "range": (0.0, 200.0), "not_constant": False, "max_step": None},
    {"periods": False, "finite": True, "range": (None, 150.0), "not_constant": False, "max_step": 30.0},
    {"periods": True, "finite": False, "range": (-50.0, None), "not_constant": True, "max_step": 5.0},
]


@pytest.mark.parametrize("settings", SETTINGS)
def test_each_check_fails_exactly_the_days_the_oracle_fails(settings):
    rng = np.random.default_rng(3)
    checks = ForecastChecks(**settings)
    seen = set()
    for trial in range(400):
        freq = ("1h", "15min")[trial % 2]
        day = (SPRING, AUTUMN, PLAIN)[trial % 3]
        planned = planned_periods(freq, ZONE)(plus(day, -1), 1)
        base = 100 + 10 * np.sin(np.arange(96) / 6) + rng.normal(0, 1, 96)
        values = {p: float(base[p]) for p in planned}
        kind = trial % 7
        if kind == 1:
            values.pop(int(rng.choice(planned)))
        elif kind == 2:
            values[int(rng.choice(planned))] = float(rng.choice([np.nan, np.inf, -np.inf])) if trial % 3 else None
        elif kind == 3:
            values[int(rng.choice(planned))] = float(rng.choice([-80.0, 250.0, 160.0]))
        elif kind == 4:
            values = {p: 42.0 for p in planned}
        elif kind == 5:
            at = int(rng.integers(1, len(planned)))
            values[planned[at]] += float(rng.choice([6.0, 40.0]))
        elif kind == 6:
            values[999] = 1e9  # no period of the day: read by no check
        found = checks.failures(values, planned)
        expected = oracle(values, planned, settings)
        assert [name for name, _ in found] == expected, (trial, values)
        seen.update(expected)
        assert all(what for _, what in found)
    enabled = {name for name in CHECK_NAMES if settings[name] not in (False, None)}
    assert seen == enabled  # every check on fails some day, and passes others


def test_a_detail_names_what_failed():
    checks = ForecastChecks(range=(0, 200), max_step=20)
    planned = tuple(range(24))
    values = {p: 100.0 + p for p in planned if p not in (5, 6)}
    values[7], values[8], values[9] = math.nan, 260.0, 100.0
    found = dict(checks.failures(values, planned))
    assert found["periods"] == "22 of 24, missing 5, 6"
    assert found["finite"] == "1 not finite (7 at nan)"
    assert found["range"] == "1 outside [0, 200] (8 at 260)"
    assert found["max_step"].startswith("160 from period 8 to 9, above 20")
    assert dict(ForecastChecks().failures({p: 3.0 for p in planned}, planned)) == {"not_constant": "all 24 at 3"}
    many = dict(ForecastChecks().failures({0: 1.0}, planned))["periods"]
    assert many == "1 of 24, missing 1, 2, 3, 4, 5, 6 and 17 more"
    # A one-period day ("1D") is never one value all day, and has no neighbours.
    assert ForecastChecks(max_step=1).failures({0: 5.0}, (0,)) == []
    # The bounds are in the range, and a step of exactly max_step is allowed.
    assert ForecastChecks(range=(0, 200), max_step=200).failures({0: 0.0, 1: 200.0}, (0, 1)) == []
    # Neighbours in clock order: on a spring day 01:00 and 03:00, whose step alone is too large here.
    spring = planned_periods("1h", ZONE)(plus(SPRING, -1), 1)
    ramp = {p: float(p) for p in spring}
    ramp.update({p: ramp[p] + 50.0 for p in spring if p >= 3})
    assert dict(ForecastChecks(max_step=10).failures(ramp, spring)) == {"max_step": "52 from period 1 to 3, above 10"}


def test_settings_are_checked_and_read_back():
    default = ForecastChecks()
    assert default.to_dict() == {
        "periods": True,
        "finite": True,
        "range": None,
        "not_constant": True,
        "max_step": None,
    }
    set_ = ForecastChecks(periods=False, range=(-500, 4000), max_step=300)
    assert set_.range == (-500.0, 4000.0) and set_.max_step == 300.0
    assert ForecastChecks.from_dict(json.loads(json.dumps(set_.to_dict()))) == set_
    assert set_ != default and repr(set_).startswith("ForecastChecks(periods=False")
    for bad in (
        {"range": (5, 5)},
        {"range": (None, None)},
        {"range": "0-100"},
        {"range": (0, 1, 2)},
        {"range": (0, math.inf)},
        {"max_step": 0},
        {"max_step": -1},
        {"max_step": True},
    ):
        with pytest.raises(ValueError):
            ForecastChecks(**bad)
    with pytest.raises(TypeError):
        ForecastChecks(periods=1)
    with pytest.raises(TypeError):
        ForecastChecks(True)  # keyword only


# ---------------------------------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------------------------------


def day_records(run_date, horizon, freq, predictions, window=None, anchor=None, skipped=()):
    """A member-day's records as a run writes them: forecasts, and skip records of the periods in `skipped`."""
    out = []
    for period, value in predictions.items():
        record = forecast_record(
            datetime.date.fromisoformat(run_date),
            period,
            horizon,
            3,
            value,
            50.0 + period,
            [0.5],
            freq,
            "abc123",
            window=window,
            refit_anchor=anchor,
        )
        out.append(record)
    for period in skipped:
        out.append(
            skip_record(
                datetime.date.fromisoformat(run_date),
                period,
                horizon,
                3,
                50.0,
                freq,
                window,
                "missing_test_predictors: load_d+1",
                "abc123",
                anchor or run_date,
            )
        )
    return out


def test_check_records_turns_the_failing_member_days_and_keeps_the_rest():
    planned = planned_periods("1h", ZONE)
    curve = {p: 100.0 + p for p in planned(plus(SPRING, -1), 1)}
    records = {
        "a": day_records(plus(SPRING, -1), 1, "1h", curve)  # a 23-hour day, whole: passes
        + day_records(PLAIN, 1, "1h", {**{p: 100.0 + p for p in range(24)}, 4: 900.0}, window="test")  # range
        + day_records(PLAIN, 2, "1h", {p: 100.0 + p for p in range(24) if p != 7}, skipped=(7,)),  # periods
        "b": day_records(AUTUMN, 1, "1h", {p: 70.0 for p in range(24)}, anchor="2026-10-19"),  # not_constant
    }
    checks = ForecastChecks(range=(0, 500))
    outcome = check_records(records, checks, planned, freq="1h", zone=ZONE)
    assert outcome.checked == 4
    a, b = outcome.records["a"], outcome.records["b"]
    assert a[:23] == records["a"][:23]
    failed_range = a[23:47]
    assert all(r["status"] == CHECK_FAILED and r["prediction"] is None for r in failed_range)
    assert {r["detail"] for r in failed_range} == {"range: 1 outside [0, 500] (4 at 900)"}
    # A skip record's layout, its key fields, actual, window and list hash kept.
    original = records["a"][23]
    assert list(failed_range[0]) == [
        "run_date",
        "target_date",
        "run_weekday",
        "hour",
        "horizon",
        "day_in_test",
        "prediction",
        "actual",
        "coefficients",
        "freq",
        "window",
        "status",
        "list_hash",
        "refit_anchor",
        "revision",
        "detail",
    ]
    assert failed_range[0]["actual"] == original["actual"] and failed_range[0]["window"] == "test"
    assert failed_range[0]["coefficients"] == [] and failed_range[0]["revision"] == original["revision"]
    periods_day = a[47:]
    assert [r["status"] for r in periods_day] == [CHECK_FAILED] * 23 + ["missing_test_predictors"]
    assert periods_day[-1] == records["a"][-1]  # the key's own skip record stays
    assert periods_day[0]["detail"] == "periods: 23 of 24, missing 7"
    assert all(r["status"] == CHECK_FAILED and r["refit_anchor"] == "2026-10-19" for r in b)
    assert [(f["member"], f["run_date"], f["horizon"], sorted(f["failed"])) for f in outcome.findings] == [
        ("a", PLAIN, 1, ["range"]),
        ("a", PLAIN, 2, ["periods"]),
        ("b", AUTUMN, 1, ["not_constant"]),
    ]
    autumn = outcome.findings[2]
    assert autumn["target_date"] == plus(AUTUMN, 1) and autumn["periods"] == 24 and autumn["forecasts"] == 24
    assert outcome.to_dict(checks) == {"checks": checks.to_dict(), "member_days": 4, "failed": outcome.findings}
    # The autumn day itself as a target: 24 periods for 25 hours, 100 quarter-hours at 15 minutes.
    quarter = check_records(
        {"q": day_records(plus(AUTUMN, -1), 1, "15min", {p: 1.0 for p in range(96)})},
        ForecastChecks(),
        planned_periods("15min", ZONE),
        freq="15min",
        zone=ZONE,
    )
    assert quarter.findings[0]["periods"] == 96 and quarter.findings[0]["products"] == 100


def test_check_records_reads_no_actual():
    planned = planned_periods("1h", ZONE)
    records = day_records(PLAIN, 1, "1h", {**{p: 100.0 for p in range(24)}, 3: 101.0}) + day_records(
        PLAIN, 2, "1h", {p: 5.0 for p in range(24)}
    )
    checks = ForecastChecks(max_step=0.5)
    first = check_records({"m": records}, checks, planned)
    rng = np.random.default_rng(0)
    moved = [{**r, "actual": float(rng.normal(0, 1e6))} for r in records]
    second = check_records({"m": moved}, checks, planned)
    assert first.findings == second.findings
    assert [r["status"] for r in first.records["m"]] == [r["status"] for r in second.records["m"]]


def test_a_member_day_without_forecasts_is_not_checked_and_a_repeated_key_raises():
    planned = planned_periods("1h", ZONE)
    only_skips = day_records(PLAIN, 1, "1h", {}, skipped=range(24))
    outcome = check_records({"m": only_skips}, ForecastChecks(), planned)
    assert outcome.checked == 0 and outcome.records["m"] == only_skips and not outcome.findings
    twice = day_records(PLAIN, 1, "1h", {p: float(p) for p in range(24)}) + day_records(PLAIN, 1, "1h", {3: 1.0})
    with pytest.raises(ValueError, match="two forecasts of period 3"):
        check_records({"m": twice}, ForecastChecks(), planned)


def test_a_failed_record_of_a_live_or_ensemble_record_keeps_its_key_order():
    record = {
        "run_date": PLAIN,
        "target_date": plus(PLAIN, 1),
        "run_weekday": 2,
        "hour": 0,
        "horizon": 1,
        "day_in_test": 0,
        "prediction": 1.0,
        "actual": None,
        "freq": "1h",
        "status": "ok",
        "revision": 3,
    }
    failed = check_failed_record(record, "finite: x")
    assert failed["revision"] == 3 and "list_hash" not in failed and failed["detail"] == "finite: x"
    assert list(failed)[-3:] == ["status", "revision", "detail"]


# ---------------------------------------------------------------------------------------------------------------------
# The store: a skip record over a forecast on file
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("save_format", ["jsonl", "parquet"])
def test_overrule_writes_a_skip_record_one_revision_above_the_forecast(tmp_path, save_format):
    path = tmp_path / f"m.{save_format}"
    records = day_records(PLAIN, 1, "1h", {p: float(p) for p in range(3)})
    store = open_store(path, test_start=PLAIN, lock=True, save_coefficients=True)
    for record in records:
        store.save(dict(record))
    store.close()
    store = open_store(path, test_start=PLAIN, lock=True, save_coefficients=True)
    assert store.is_done(1, 1, 0)
    store.overrule(check_failed_record(records[1], "range: x"))
    assert not store.is_done(1, 1, 0)
    store.close()
    chosen = {r["hour"]: r for r in read_records(path, statuses=None)}
    assert chosen[1]["status"] == CHECK_FAILED and chosen[1]["revision"] == 2
    assert chosen[0]["status"] == "ok" and chosen[0]["revision"] == 1
    assert [r["hour"] for r in read_records(path)] == [0, 2]  # the forecasts the readers take
    # A later forecast of the key is written at that revision, and wins over the skip record there.
    store = open_store(path, test_start=PLAIN, lock=True, save_coefficients=True)
    store.save(dict(records[1]))
    store.close()
    chosen = {r["hour"]: r for r in read_records(path, statuses=None)}
    assert chosen[1]["status"] == "ok" and chosen[1]["revision"] == 2


def test_overrule_refuses_a_forecast_not_yet_written(tmp_path, monkeypatch):
    record = day_records(PLAIN, 1, "1h", {0: 1.0})[0]
    store = open_store(tmp_path / "m.jsonl", test_start=PLAIN, lock=True, save_coefficients=True)
    try:
        monkeypatch.setattr(store, "_enqueue", lambda record, layout=None: None)  # saved, never written
        store.save(dict(record))
        with pytest.raises(ValueError, match="isn't written yet"):
            store.overrule(check_failed_record(record, "x"))
    finally:
        store.close()


# ---------------------------------------------------------------------------------------------------------------------
# The gate of a run, on its own
# ---------------------------------------------------------------------------------------------------------------------


class FakeStore:
    """What the gate reads of a store: whether it has results, and its forecasts. Like a store whose writer thread still holds
    the skips decided before any fit (`unwritten`), it refuses to be read before `flush()` has written them."""

    def __init__(self, records, unwritten=0):
        self.records, self.unwritten = records, unwritten

    def has_results(self):
        return bool(self.records)

    def flush(self):
        self.unwritten = 0

    def iter_lines(self, cols=None):
        if self.unwritten:
            raise AssertionError("read while the writer thread still appends: flush the store first")
        return iter(self.records)


class FakeState:
    """What the gate reads of a run's state, and the skips it is told of."""

    def __init__(self, tasks, pending, store):
        self.tasks, self.pending, self.store = tasks, pending, store
        self.skipped = []

    def skip(self, task, outcome, example, sink, overrule=False):
        self.skipped.append((task, outcome.partition(": ")[0], overrule))


class FakeSink:
    def __init__(self):
        self.saved = []

    def save(self, record):
        self.saved.append(record)


def test_the_gate_checks_a_day_whose_new_records_are_all_skips_with_its_forecasts_on_file():
    """A member-day whose keys left to do are all skips decided before any fit (no task) and whose other keys are on file is
    checked when the run starts, from the forecasts on file alone: here they lack the skipped periods, so `periods` fails and
    they are overruled. A day with nothing left to do isn't checked again."""
    from epftoolbox2.models._checks import CheckGate

    first = datetime.date.fromisoformat(PLAIN)
    on_file = [{"hour": p, "horizon": 1, "run_date": PLAIN, "prediction": 100.0 + p} for p in range(20)]
    on_file += [{"hour": p, "horizon": 1, "run_date": plus(PLAIN, 1), "prediction": 100.0 + p} for p in range(24)]
    state = FakeState(tasks=[], pending=[(p, 1, 0) for p in range(20, 24)], store=FakeStore(on_file, unwritten=4))
    gate = CheckGate(ForecastChecks(), "1h", ZONE, first)
    sink = FakeSink()
    gate.finish(state, sink)
    assert sorted(state.skipped) == [((p, 1, 0), CHECK_FAILED, True) for p in range(20)]
    assert sink.saved == []


def test_the_gate_holds_a_day_until_every_task_of_it_is_back():
    from epftoolbox2.models._checks import CheckGate

    first = datetime.date.fromisoformat(PLAIN)
    tasks = [(p, 1, 0) for p in range(24)]
    state = FakeState(tasks=tasks, pending=tasks, store=None)
    gate = CheckGate(ForecastChecks(), "1h", ZONE, first)
    sink = FakeSink()
    day = day_records(PLAIN, 1, "1h", {p: 100.0 + p for p in range(24)})
    gate.collect(state, day[:10], [], sink)
    assert sink.saved == []  # 14 periods still out
    gate.collect(state, day[10:23], [((23, 1, 0), "missing_test_predictors")], sink)
    assert sink.saved == [] and [s[1] for s in state.skipped] == ["missing_test_predictors"] + [CHECK_FAILED] * 23
    assert not any(s[2] for s in state.skipped)
