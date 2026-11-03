"""Vintage classes and validity periods of availability rules (step 3h).

A class (`Availability(vintage=...)`) says what the archive of a column holds and decides no number. A validity period says on
which days a rule holds, read on the day the value is delivered (for `as_of` and `weather_cutoff` the day it is read on), and a
value no rule of its column holds for is never known. Everything is held against what shares no code with it: the day a value is
delivered or read worked out with pandas from the row's own instant, k written down from the rules by hand, and runs with and
without the metadata compared bit for bit.
"""

import dataclasses
import datetime
import json
from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest

import epftoolbox2.information_set as information_set_module
from epftoolbox2.data.sources import EntsoeSource, OpenMeteoSource
from epftoolbox2.data.sources.benchmark import EpftoolboxBenchmarkSource
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.information_set import (
    ASSUMPTIONS,
    NEVER,
    Availability,
    InformationSet,
    InformationSetError,
    RuleChain,
    default_rules,
    is_revised,
    source_rules,
    vintage_of,
)
from epftoolbox2.models import OLSModel, Use
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline
from tests.synthetic import PREDICTORS, make_market_data

PL = "Europe/Warsaw"
INPUT_RULES = {
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


def day(text, offset=0):
    return (pd.Timestamp(text) + pd.Timedelta(days=offset)).date().isoformat()


# ---------------------------------------------------------------------------------------------------------------------------
# The rule: a class, a period, written only when set
# ---------------------------------------------------------------------------------------------------------------------------


def test_every_constructor_takes_a_class_and_writes_it_only_when_set():
    made = {
        "static": (Availability.static, ()),
        "realised": (Availability.realised, ("2h",)),
        "published": (Availability.published, ("13:00", 1)),
        "daily_close": (Availability.daily_close, ("18:00",)),
        "weekly": (Availability.weekly, ("friday", "10:00")),
        "weather_run": (Availability.weather_run, ("6h",)),
        "as_of": (Availability.as_of, ("09:00",)),
        "weather_cutoff": (Availability.weather_cutoff, ("00", "8h")),
    }
    for name, (make, args) in made.items():
        plain, classed = make(*args), make(*args, vintage="snapshot")
        assert plain.vintage is None and classed.vintage == "snapshot", name
        assert "vintage" not in plain.to_dict() and "valid_from" not in plain.to_dict(), name
        assert classed.to_dict() == {**plain.to_dict(), "vintage": "snapshot"}, name
        assert Availability.from_dict(classed.to_dict()) == classed, name
        assert classed.describe() == plain.describe(), name  # a class decides nothing, and says nothing in a listing
    assert (
        Availability("published", time="13:00", days_before=1, vintage="latest_revision").vintage == "latest_revision"
    )


def test_the_dict_of_a_rule_is_what_it_was_unless_a_class_or_a_period_is_set():
    """Old files load and are written back as they were: the new keys are there only when set."""
    assert Availability.published("13:00", 1).to_dict() == {
        "kind": "published",
        "time": "13:00",
        "days_before": 1,
        "timezone": "Europe/Brussels",
    }
    assert Availability.realised("1h", note="n").to_dict() == {"kind": "realised", "delay": "1h", "note": "n"}
    rule = Availability.published("14:30", -4, PL, note="n", vintage="last_write_stamp").during(
        "2026-09-14", "2026-10-07"
    )
    assert list(rule.to_dict().items()) == [
        ("kind", "published"),
        ("time", "14:30"),
        ("days_before", -4),
        ("timezone", PL),
        ("vintage", "last_write_stamp"),
        ("valid_from", "2026-09-14"),
        ("valid_until", "2026-10-07"),
        ("note", "n"),
    ]
    assert Availability.from_dict(rule.to_dict()) == rule


def test_a_period_is_a_span_of_calendar_dates():
    rule = Availability.realised("1h")
    assert rule.during(valid_from=datetime.date(2026, 10, 8)).valid_from == "2026-10-08"
    assert rule.during(valid_until=pd.Timestamp("2026-10-07")).valid_until == "2026-10-07"
    assert rule.during("2026-10-08", "2026-10-08").bounded and not rule.bounded
    assert rule.during(valid_from="2026-10-08").during().valid_from is None  # `during()` sets both sides
    assert (
        rule.during(valid_from="2026-10-08").describe()
        == "known 1h after the end of the period (valid from 2026-10-08)"
    )
    assert rule.during(valid_until="2026-10-07").describe().endswith("(valid to 2026-10-07)")
    assert rule.during("2026-01-01", "2026-10-07").describe().endswith("(valid 2026-01-01 to 2026-10-07)")
    with pytest.raises(ValueError, match="after valid_until"):
        rule.during("2026-10-08", "2026-10-07")
    with pytest.raises(ValueError, match="not an instant"):
        rule.during(valid_from="2026-10-08 12:00")
    with pytest.raises(ValueError, match="not an instant"):
        rule.during(valid_from=pd.Timestamp("2026-10-08", tz=PL))
    with pytest.raises(ValueError, match="valid_from must be a date"):
        rule.during(valid_from="the eighth")
    with pytest.raises(ValueError, match="Unknown vintage class"):
        Availability.realised("1h", vintage="revised")


def test_a_list_of_rules_holds_on_disjoint_periods_in_the_order_of_their_days():
    early = Availability.published("14:30", -4, PL).during(valid_until="2026-10-07")
    late = Availability.published("14:30", -1, PL).during(valid_from="2026-10-08")
    info = InformationSet(rules={"sk": [late, early], "price": Availability.published("13:00", 1)})
    chain = info.rules["sk"]
    assert (
        isinstance(chain, RuleChain)
        and chain.rules == (early, late)
        and len(chain) == 2
        and list(chain) == [early, late]
    )
    assert chain.kind == "published"
    assert RuleChain((early, Availability.realised("1h").during(valid_from="2026-10-09"))).kind == "chain"
    assert info.to_dict()["rules"]["sk"] == [early.to_dict(), late.to_dict()]
    assert InformationSet.from_dict(info.to_dict()).rules == info.rules
    assert chain.describe() == f"{early.describe()}; {late.describe()}"
    # A gap is allowed; a chain of one rule is one.
    assert RuleChain((early, late.during(valid_from="2026-10-20"))).rules[1].valid_from == "2026-10-20"
    assert len(InformationSet(rules={"x": [Availability.static()]}).rules["x"]) == 1
    overlapping = [
        [early, late.during(valid_from="2026-10-07")],  # one shared day
        [early, Availability.published("14:30", -1, PL)],  # open on both sides
        [early.during(), late],  # open before, and before the other one too
        [late, late.during(valid_from="2026-11-01")],  # open after
    ]
    for rules in overlapping:
        with pytest.raises(ValueError, match="disjoint periods"):
            InformationSet(rules={"sk": rules})
    with pytest.raises(ValueError, match="at least one rule"):
        InformationSet(rules={"sk": []})


def test_the_built_in_rules_carry_their_classes_and_a_rule_of_yours_has_none():
    rules = default_rules()
    revised = ["load_actual", "load_forecast", "load_forecast_daily_min", "load_forecast_daily_max"]
    revised += ["generation_*_forecast", "generation_*"]
    assert {name: rules[name].vintage for name in revised} == dict.fromkeys(revised, "latest_revision")
    assert rules["price"].vintage == "snapshot"
    assert {rule.vintage for rule in rules.values() if rule.kind == "static"} == {None}
    assert ASSUMPTIONS["entsoe_a69_by_10"][1].vintage == "latest_revision"
    weather = OpenMeteoSource(latitude=52.2, longitude=21.0, prefix="warsaw", columns=["temperature_2m"], timezone=PL)
    found = source_rules([EntsoeSource("PL", "key", ["load"]), weather, EpftoolboxBenchmarkSource("PJM")])
    assert found["load_forecast_daily_max"].vintage == "latest_revision"
    assert found["warsaw_temperature_2m"].vintage == "snapshot"
    assert found["price"].vintage == "snapshot"
    assert {rule.vintage for name, rule in found.items() if name != "price" and rule.kind == "published"} == {None}
    info = InformationSet(rules={"price": Availability.published("13:00", 1)})
    assert vintage_of(info.resolve("price_d-1").rule) == "unknown"  # yours replaces the built-in one, class and all
    assert vintage_of(InformationSet().resolve("price_d-1").rule) == "snapshot"
    assert vintage_of(InformationSet().resolve("generation_wind_forecast_d+1").rule) == "latest_revision"
    assert vintage_of(None) == "unknown"
    assert not is_revised(None) and not is_revised(rules["price"]) and is_revised(rules["load_actual"])
    chain = RuleChain((rules["price"].during(valid_until="2026-01-01"), rules["load_actual"].during("2026-01-02")))
    assert vintage_of(chain) == "snapshot, latest_revision" and is_revised(chain)


# ---------------------------------------------------------------------------------------------------------------------------
# A chain is each rule on its days
# ---------------------------------------------------------------------------------------------------------------------------

SWITCHES = {"spring": "2023-03-26", "autumn": "2023-10-29"}
STEPS = {"1h": 60, "15min": 15, "1D": 60}
NAMES = ["x", "x_d-1", "x_d+1", "x_h-3", "x_d-2_h-13"]


def by_hand(name, instant, read):
    """The local date of the value `name` holds on a row at `instant`: delivered, or with `read` the day it is read on, where a
    leading `_d+k` is the source's own lead and moves nothing. Day shifts keep the wall clock, hour shifts are elapsed time."""
    wall = instant.tz_localize(None)
    if name == "x":
        return wall.date()
    if name == "x_d-1":
        return (wall - pd.Timedelta(days=1)).date()
    if name == "x_d+1":
        return wall.date() if read else (wall + pd.Timedelta(days=1)).date()
    if name == "x_h-3":
        return (instant - pd.Timedelta(hours=3)).tz_convert(PL).date()
    # x_d-2_h-13: 13 hours earlier, then two calendar days back on the wall clock.
    earlier = (instant - pd.Timedelta(hours=13)).tz_convert(PL).tz_localize(None)
    return (earlier - pd.Timedelta(days=2)).date()


@pytest.mark.parametrize(
    "freq, switch, row_day",
    [("1h", "spring", 0), ("1h", "autumn", -3), ("15min", "spring", -3), ("15min", "autumn", 0), ("1D", "autumn", 0)],
)
def test_a_chain_is_each_rule_on_its_days_with_a_gap_and_never_outside(freq, switch, row_day):
    """Three rules of three kinds: A to the switch day, nothing the day after, B for three days, C (`as_of`, read on the day it is
    read on) from then on. On every cell the chain is the rule whose period holds the value's day, worked out here from the row's
    instant, and never known where none does; a slot the clock skips has no value either way. The boundary days and the switch
    are in the grid, and so are names whose delivery day differs from the row's, and from the day an `as_of` value is read on."""
    s = pd.Timestamp(SWITCHES[switch])
    a = Availability.published("13:00", 1, PL)
    b = Availability.realised("2h")
    c = Availability.as_of("09:00", PL)
    a_until, b_from, b_until, c_from = s, s + pd.Timedelta(days=2), s + pd.Timedelta(days=4), s + pd.Timedelta(days=5)
    chain = [
        a.during(valid_until=a_until.date()),
        b.during(b_from.date(), b_until.date()),
        c.during(valid_from=c_from.date()),
    ]
    data = make_market_data(start=day(s, -20), days=40, tz=PL, freq="1h" if freq == "1D" else freq)
    if freq == "1D":
        data = data.at_time("12:00")
    step = STEPS[freq]
    days = pd.date_range(s - pd.Timedelta(days=4), s + pd.Timedelta(days=9))
    infos = {key: InformationSet("11:45", PL, rules={"x": rule}) for key, rule in zip("ABC", (a, b, c))}
    infos["chain"] = InformationSet("11:45", PL, rules={"x": chain})
    clock = infos["chain"]._issue_clock(days, data.index, freq)
    periods = np.arange(1 if freq == "1D" else 1440 // step)
    minute = (12 * 60) if freq == "1D" else None
    instants = {
        (p, i): (issue_day + pd.Timedelta(days=row_day, minutes=minute if minute is not None else period * step))
        # A repeated hour is its second occurrence, the row a model forecasts from; a skipped one has no row.
        .tz_localize(PL, ambiguous=False, nonexistent="NaT")
        for p, period in enumerate(periods)
        for i, issue_day in enumerate(days)
    }
    for name in NAMES:
        got = {
            key: clock.available(info._resolve_for(name, step), periods, step, row_day) for key, info in infos.items()
        }
        assert np.isfinite(got["A"][got["A"] > -np.inf]).all()
        claimed = {"A": 0, "B": 0, "C": 0, None: 0}
        for p, period in enumerate(periods):
            for i, issue_day in enumerate(days):
                instant = instants[(p, i)]
                cell = got["chain"][p, i]
                if instant is pd.NaT:
                    assert cell == -np.inf, (name, period, issue_day)
                    continue
                delivered = pd.Timestamp(by_hand(name, instant, read=False))
                read = pd.Timestamp(by_hand(name, instant, read=True)) - pd.Timedelta(days=row_day)
                claims = [
                    key
                    for key, inside in (
                        ("A", delivered <= a_until),
                        ("B", b_from <= delivered <= b_until),
                        ("C", read >= c_from),
                    )
                    if inside
                ]
                # Off the issue day's own row an `as_of` value is read on D, so C and a rule of its delivery day can both
                # claim it: the later time counts.
                expected = max((got[key][p, i] for key in claims), default=np.inf)
                for key in claims or [None]:
                    claimed[key] += 1
                assert cell == expected, (freq, name, int(period), issue_day.date(), claims, cell, expected)
        assert all(claimed.values()), (name, claimed)  # every rule and the gap, in every grid


def test_a_value_no_rule_holds_for_is_late_in_the_check_ctx_explain_and_the_probe():
    """Two rules with a gap of a day: what the gap's day delivers is never known. The check refuses it, the listing shows it as
    never, `ctx.published` says no, `explain` calls it late and the probe perturbs it; the days around it are as each rule says."""
    data = make_market_data(start="2023-03-01", days=40, tz=PL)
    chain = [
        Availability.published("10:00", 2, PL).during(valid_until="2023-03-20"),
        Availability.published("10:00", 2, PL).during(valid_from="2023-03-22"),
    ]
    info = InformationSet(
        "11:45", PL, rules={**INPUT_RULES, "load": chain, "price": Availability.published("13:00", 1)}
    )
    expanded = {(period, 1): ["load_d+1"] for period in range(24)}
    found, _ = info.violations(expanded, "price", "1h", data.index, "2023-03-18", "2023-03-24")
    assert {(v.period, v.available_label) for v in found} == {(period, NEVER) for period in range(24)}
    with pytest.raises(InformationSetError, match="known never"):
        info.check(expanded, "price", "1h", data.index, "2023-03-18", "2023-03-24")
    # load_d+1 of issue day 20 March is the load of the 21st, in the gap; of the 19th and the 21st it isn't.
    info.check(expanded, "price", "1h", data.index, "2023-03-18", "2023-03-19")
    info.check(expanded, "price", "1h", data.index, "2023-03-21", "2023-03-24")

    fixed = ModelPipeline(information_set=info).add_model(OLSModel(PREDICTORS, training_window=10, name="m"))
    table = fixed.explain(data, "m", test=("2023-03-20", "2023-03-20"), horizon=1, hour=10, echo=False)["predictors"]
    assert table.set_index("predictor").loc["load_d+1", ["known", "status"]].tolist() == [NEVER, "late"]
    listed = fixed.run(data, test=("2023-03-19", "2023-03-21"), horizon=1).availability()
    assert set(listed["known"]) == {NEVER} and set(listed["predictor"]) == {"load_d+1"}

    seen = {}

    def reads(ctx):
        seen[ctx.issue_day.isoformat()] = ctx.published("load_d+1")
        return PREDICTORS

    pipeline = ModelPipeline(information_set=info).add_model(OLSModel(reads, training_window=10, name="m"))
    pipeline.run(data, test=("2023-03-19", "2023-03-21"), horizon=1)
    assert seen == {"2023-03-19": True, "2023-03-20": False, "2023-03-21": True}
    perturbed, counts, _ = information_set_module.perturb_unknown(data, "2023-03-20", info, columns=["load_d+1"])
    rows = data.index.tz_localize(None).normalize() == pd.Timestamp("2023-03-20")
    assert (perturbed.loc[rows, "load_d+1"] != data.loc[rows, "load_d+1"]).all()  # the gap's value, on D's own row


# ---------------------------------------------------------------------------------------------------------------------------
# `published` on the PL settlement chain, by hand
# ---------------------------------------------------------------------------------------------------------------------------

# PSE's settlement series: a business date was first out at 14:30 four days later up to 2026-10-07, and one day later from
# 2026-10-08 on (markets/REPORT.md section 5.4).
PL_CHAIN = [
    Availability.published("14:30", -4, PL, vintage="last_write_stamp").during(valid_until="2026-10-07"),
    Availability.published("14:30", -1, PL, vintage="last_write_stamp").during(valid_from="2026-10-08"),
]
ISSUE_DAYS = [day("2026-10-05", d) for d in range(10)]  # Monday 5 October to Wednesday 14 October


def k_by_hand(issue_day):
    """At 11:45 on D every target from D-31 to D+k must be known. Through D = 11 October the value of D-4 is out only at 14:30 on
    D (business date D-4 up to 7 October), so D-5 is the newest: on 9 and 10 October the value of 8 October (out on the 9th) is
    known, but those of 6 and 7 October aren't yet, and 7 October's comes only on the 11th, after its issue time. On 12 October 7
    October is out, and 8 to 10 October each the day after: D-2. From then on the one-day lag holds."""
    return -5 if issue_day <= "2026-10-11" else -2


def pl_frame(freq="1h"):
    data = make_market_data(start="2026-07-15", days=100, tz=PL, freq="1h" if freq == "1D" else freq, horizon=3, seed=5)
    return data.at_time("12:00") if freq == "1D" else data


def pl_info(rule=None):
    return InformationSet("11:45", PL, rules={**INPUT_RULES, "price": PL_CHAIN if rule is None else rule})


@pytest.mark.parametrize("freq", ["1h", "15min", "1D"])
def test_the_newest_target_on_the_pl_settlement_chain_is_the_hand_table(freq):
    data = pl_frame(freq)
    info = pl_info()
    days = pd.DatetimeIndex(ISSUE_DAYS)
    k = info.newest_targets("price", freq, data.index, days, horizon=1)
    assert k.shape[0] == {"1h": 24, "15min": 96, "1D": 1}[freq]
    assert {d: set(column.tolist()) for d, column in zip(ISSUE_DAYS, k.T)} == {d: {k_by_hand(d)} for d in ISSUE_DAYS}
    # The value of 8 October is known on the 10th, before those of the 6th and 7th: the walk stops at the first unknown one.
    clock = info._issue_clock(days, data.index, freq)
    col = info._resolve_for("price", 60 if freq != "15min" else 15)
    on_the_10th, on_the_11th = ISSUE_DAYS.index("2026-10-10"), ISSUE_DAYS.index("2026-10-11")
    known = {
        row_day: bool((clock.available(col, np.array([0]), 60, row_day) <= clock.issue)[0, on_the_10th])
        for row_day in (-2, -3, -4, -5)
    }
    assert known == {-2: True, -3: False, -4: False, -5: True}
    seventh = clock.available(col, np.array([0]), 60, -4)[0, on_the_11th]  # 7 October, read on the 11th
    assert clock.label(seventh, on_the_11th) == "D 14:30"


RUN_CASES = {
    "plain": {},
    "15min": {"freq": "15min"},
    "daily": {"freq": "1D"},
    "horizons": {"horizons": [1, 3]},
    "issue_days": {"issue_days": ["monday", "friday", "saturday", "sunday"]},
    "refit": {"refit": "7D"},
}


@pytest.mark.parametrize("case", RUN_CASES)
def test_a_published_model_trains_on_the_hand_table_through_the_chain(case):
    """Each record's newest target date is D + min(k, h - 1), k from the hand table; with `refit` the anchor's k (Mondays, 5 and 12
    October). Through 11 October the forecasts are bit for bit those of the four-day lag alone, though the one-day lag holds for
    targets from 8 October; from 12 October those of the one-day lag alone, which on 8 to 11 October would have trained on
    targets the chain says weren't out yet."""
    extra = dict(RUN_CASES[case])
    freq = extra.pop("freq", "1h")
    refit = extra.pop("refit", None)
    data = pl_frame(freq)
    horizon = {} if "horizons" in extra else {"horizon": 2}

    def run(rule=None):
        model = OLSModel(PREDICTORS, training_window=30, newest_target="published", refit=refit)
        ref = model.run(
            data,
            test_start=ISSUE_DAYS[0],
            test_end=ISSUE_DAYS[-1],
            freq=freq,
            information_set=pl_info(rule),
            **horizon,
            **extra,
        )
        return {(r["run_date"], r["hour"], r["horizon"]): r for r in ref._results}

    records = run()
    assert records
    for (run_date, _, h), record in records.items():
        anchor = run_date if refit is None else ("2026-10-05" if run_date < "2026-10-12" else "2026-10-12")
        assert record["newest_target_day"] == day(anchor, min(k_by_hand(anchor), h - 1)), (run_date, h)
    if "issue_days" in extra:
        assert {pd.Timestamp(d).day_name() for d, _, _ in records} == {"Monday", "Friday", "Saturday", "Sunday"}
    if refit is not None:
        return
    four_days, one_day = run(PL_CHAIN[0].during()), run(PL_CHAIN[1].during())
    assert records == {key: (four_days if key[0] <= "2026-10-11" else one_day)[key] for key in records}
    middle = [key for key in records if "2026-10-08" <= key[0] <= "2026-10-11"]
    assert middle and all(one_day[key]["newest_target_day"] > records[key]["newest_target_day"] for key in middle)


# ---------------------------------------------------------------------------------------------------------------------------
# Bitwise: the metadata moves nothing
# ---------------------------------------------------------------------------------------------------------------------------


def classed(rule, vintage):
    return dataclasses.replace(rule, vintage=vintage)


def reads_the_load(ctx):
    names = list(PREDICTORS)
    if ctx.published("load_d+1") and ctx.value("load_d+1") > 1000:
        names.append("price_d-1")
    return Use(names, train_on="matching")


@pytest.mark.parametrize("predictors", ["fixed", "function"])
def test_forecasts_are_bitwise_with_and_without_classes_and_periods_that_cover_everything(predictors):
    """The same rules with a class each, with a period from long before the data to long after it, and split in two at a day:
    every record and table is the same, for a published newest target too."""
    data = make_market_data(start="2023-02-01", days=70, tz=PL, horizon=2, noise=5.0, seed=8)
    plain = {**INPUT_RULES, "price": Availability.published("13:00", 1)}
    split = {
        name: [rule.during(valid_until="2023-03-26"), rule.during(valid_from="2023-03-27")]
        for name, rule in plain.items()
    }
    covering = {
        name: classed(rule, "latest_revision").during("1990-01-01", "2090-12-31") for name, rule in plain.items()
    }
    lists = PREDICTORS + ["price_d-1"] if predictors == "fixed" else reads_the_load

    def report(rules):
        pipeline = ModelPipeline(information_set=InformationSet("11:45", PL, rules=rules))
        pipeline.add_model(OLSModel(lists, training_window=20, name="d1"))
        pipeline.add_model(OLSModel(lists, training_window=20, newest_target="published", name="published"))
        pipeline.add_evaluator(MAEEvaluator())
        out = pipeline.run(data, test=("2023-03-20", "2023-04-02"), horizon=2)
        records = {name: sorted(map(repr, ref._results)) for name, ref in out.refs.items()}
        return records, out.summary(), out.by_horizon(), out.availability()

    base = report(plain)
    for rules in (split, covering):
        other = report(rules)
        assert other[0] == base[0]
        pd.testing.assert_frame_equal(other[1], base[1])
        pd.testing.assert_frame_equal(other[2], base[2])
        assert len(other[3]) == len(base[3])


# ---------------------------------------------------------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------------------------------------------------------

OLD_FILE = """models:
- class: OLSModel
  params:
    predictors:
    - load_d+{horizon}
    - price_d-1
    training_window: 30
    name: ols
    scaler: standard
evaluators: []
exporters: []
information_set:
  issue_time: '11:45'
  issue_timezone: Europe/Warsaw
  rules:
    load:
      kind: published
      time: '10:00'
      days_before: 1
      timezone: Europe/Warsaw
    flows:
      kind: realised
      delay: 2h
      note: measured
    fuel_*:
      kind: weekly
      weekday: friday
      time: '10:00'
      timezone: Europe/Brussels
    is_*:
      kind: static
    weather:
      kind: weather_cutoff
      time: 00:00
      delay: 8h
      timezone: UTC
    reanalysis:
      kind: as_of
      time: 09:00
      timezone: Europe/Brussels
    run:
      kind: weather_run
      delay: 6h
  assumptions:
  - entsoe_a69_by_10
  unknown: warn
  use_defaults: false
availability: raise
"""


def test_an_old_file_is_byte_identical_after_a_round_trip(tmp_path):
    source = tmp_path / "old.yaml"
    source.write_text(OLD_FILE)
    again = tmp_path / "again.yaml"
    ModelPipeline.load(source).save(again)
    assert again.read_text() == OLD_FILE


def test_chains_classes_and_periods_survive_a_round_trip(tmp_path):
    rules = {
        "sk": PL_CHAIN,
        "price": Availability.published("13:00", 1, vintage="snapshot"),
        "load_*": Availability.realised("1h", vintage="latest_revision").during(valid_from="2015-01-01"),
        "pdgsz": [
            Availability.published("16:00", 1, PL, vintage="row_stamp").during("2024-06-01", "2024-12-31"),
            Availability.published("15:00", 1, PL, vintage="row_stamp").during(valid_from="2025-01-02"),
        ],
    }
    pipeline = ModelPipeline(information_set=InformationSet("11:45", PL, rules=rules))
    pipeline.add_model(OLSModel(["sk_d-5"], training_window=30, name="ols"))
    path = tmp_path / "p.yaml"
    pipeline.save(path)
    text = path.read_text()
    assert "valid_until: '2026-10-07'" in text and "vintage: last_write_stamp" in text
    loaded = ModelPipeline.load(path)
    assert loaded.information_set.rules == InformationSet("11:45", PL, rules=rules).rules
    assert isinstance(loaded.information_set.rules["sk"], RuleChain)
    again = tmp_path / "again.yaml"
    loaded.save(again)
    assert again.read_text() == text
    # A period written by hand as a YAML date loads as the same rule.
    by_hand = text.replace("'2026-10-07'", "2026-10-07")
    path.write_text(by_hand)
    assert ModelPipeline.load(path).information_set.rules["sk"] == loaded.information_set.rules["sk"]


# ---------------------------------------------------------------------------------------------------------------------------
# Older values under rules over time: the training rows, the slots of a day, and the newest target's window
# ---------------------------------------------------------------------------------------------------------------------------


def pl_rows():
    """Hourly Warsaw rows from July to October 2026: a settlement series `sk` and what reads it, a target `fs`, `x` and `s`."""
    index = pd.date_range("2026-07-01", "2026-10-25 23:00", freq="1h", tz=PL)
    rng = np.random.default_rng(0)
    columns = ["sk", "sk_d-2", "fs", "x_d-1", "price"]
    data = pd.DataFrame({name: rng.normal(size=len(index)) for name in columns}, index=index)
    data["sk_d-2"] = 0.5 * data["sk_d-2"] + data["price"]  # a predictor the fit leans on, so the probe sees it
    data["x_d-1"] = 0.5 * data["x_d-1"] + data["price"]
    return data.assign(s=1.0)


SHORTENING = {
    "fs": [
        Availability.published("18:00", -2, PL).during(valid_until="2026-10-07"),
        Availability.published("18:00", 0, PL).during(valid_from="2026-10-08"),
    ]
}


def run_at_raise(predictors, rules, day, target="price", window=60, **model):
    info = InformationSet(
        "11:45", PL, rules={"price": Availability.published("13:00", 1, PL), "s": Availability.static(), **rules}
    )
    OLSModel(predictors, training_window=window, **model).run(
        pl_rows(), test_start=day, test_end=day, horizon=1, target=target, information_set=info
    )
    return info


def sk_d2(ctx):
    return ["sk_d-2"]


@pytest.mark.parametrize("predictors", [["sk_d-2"], sk_d2], ids=["fixed", "function"])
def test_a_training_row_a_shortening_lag_leaves_unknown_is_refused_as_the_probe_finds(predictors):
    """`sk_d-2` of issue day 10 October is the value of the 8th, out on the 9th; the row dated the 8th, which a forecast at h=1
    trains on, holds the 6th's, out only at 14:30 on the 10th. The issue day's row is known and the training row isn't: refused,
    and the probe moves every forecast. From 12 October every row is known: it runs, and the probe moves nothing."""
    for day_ in ("2026-10-10", "2026-10-11"):
        with pytest.raises(InformationSetError, match="a training row holds a value known only then"):
            run_at_raise(predictors, {"sk": PL_CHAIN}, day_)
    info = run_at_raise(predictors, {"sk": PL_CHAIN}, "2026-10-12")
    model = lambda: OLSModel(["sk_d-2"], training_window=60)  # noqa: E731
    late = information_set_module.leakage_probe(model, pl_rows(), "2026-10-10", info, horizon=1, columns=["sk_d-2"])
    assert len(late.changed) == 24
    clean = information_set_module.leakage_probe(model, pl_rows(), "2026-10-12", info, horizon=1, columns=["sk_d-2"])
    assert clean.ok


def test_a_training_target_a_shortening_lag_leaves_unknown_is_refused():
    """`fs` was out two days after delivery up to 7 October and on the day from the 8th. On issue day 9 October the target of D-1
    (the 8th) is out, and that of D-2 (the 7th), which trains at h=1, comes only on the 9th at 18:00."""
    with pytest.raises(InformationSetError, match="training target fs"):
        run_at_raise(["s"], SHORTENING, "2026-10-09", target="fs")
    run_at_raise(["s"], SHORTENING, "2026-10-12", target="fs")


def test_a_window_reaching_before_valid_from_is_refused_and_one_after_it_runs():
    rules = {"x": Availability.realised("1h").during(valid_from="2026-09-01")}
    with pytest.raises(InformationSetError, match="x_d-1 for period 0 is known never"):
        run_at_raise(["x_d-1"], rules, "2026-10-10", window=60)
    info = run_at_raise(["x_d-1"], rules, "2026-10-10", window=30)
    model = lambda window: lambda: OLSModel(["x_d-1"], training_window=window)  # noqa: E731
    found = information_set_module.leakage_probe(model(60), pl_rows(), "2026-10-10", info, horizon=1, columns=["x_d-1"])
    assert len(found.changed) == 24
    assert information_set_module.leakage_probe(
        model(30), pl_rows(), "2026-10-10", info, horizon=1, columns=["x_d-1"]
    ).ok


def test_the_listing_says_a_value_is_late_on_a_training_row_and_never_known_is_null_in_the_manifest(tmp_path):
    rules = {
        "price": Availability.published("13:00", 1, PL),
        "x": Availability.realised("1h").during(valid_from="2026-09-01"),
    }
    pipeline = ModelPipeline(information_set=InformationSet("11:45", PL, rules={**rules, "sk": PL_CHAIN}))
    pipeline.add_model(OLSModel(["sk_d-2", "x_d-1"], training_window=60, name="m"))
    listed = pipeline.run(
        pl_rows(), test=("2026-10-10", "2026-10-10"), horizon=1, save_dir=str(tmp_path)
    ).availability()
    assert set(listed["predictor"]) == {"sk_d-2", "x_d-1"}
    assert set(listed["note"]) == {information_set_module._TRAINING_ROWS_HINT}
    assert set(listed.loc[listed["predictor"] == "x_d-1", "known"]) == {NEVER}
    text = (tmp_path / "m.manifest.json").read_text()
    assert "Infinity" not in text
    stored = [row for row in json.loads(text)["availability"]["listing"] if row["predictor"] == "x_d-1"]
    assert stored and all(row["minutes_late"] is None and row["known"] == NEVER for row in stored)


def test_the_newest_target_s_window_must_be_known_back_to_its_oldest_target():
    """A published model's window holds targets back to D + k - W + 1. Under rules over time a known D-31 says nothing of
    those, so a window reaching before `valid_from` raises at every level, and a shorter one runs. At 11:45 the hour 10:00 of D
    is known only at 12:00, so period 10's newest target is D-1 and a window of 70 reaches D-70, while `valid_from` is D-56."""
    rules = {"fs": Availability.realised("1h").during(valid_from="2026-08-15")}
    for level in ("raise", "none"):
        info = InformationSet("11:45", PL, rules={**rules, "s": Availability.static()})
        with pytest.raises(InformationSetError, match="training target 'fs' of 70 days before"):
            OLSModel(["s"], training_window=70, newest_target="published").run(
                pl_rows(),
                test_start="2026-10-10",
                test_end="2026-10-10",
                horizon=1,
                target="fs",
                information_set=info,
                availability=level,
            )
    run_at_raise(["s"], rules, "2026-10-10", target="fs", window=40, newest_target="published")


def test_a_rule_that_doesn_t_change_over_time_reads_no_training_row(monkeypatch):
    """Only a column whose rule changes over time is read on its training rows: nothing else is slower or moves."""

    def never(*args, **kwargs):
        raise AssertionError("a training row was read")

    monkeypatch.setattr(information_set_module._RowsKnown, "latest", never)
    rules = {"sk": Availability.published("14:30", -1, PL), "x": Availability.realised("1h"), **SHORTENING}
    run_at_raise(["sk_d-2", "x_d-1"], {k: v for k, v in rules.items() if k != "fs"}, "2026-10-12")
    run_at_raise(sk_d2, {"sk": rules["sk"]}, "2026-10-12")


def verdict(names, index, day, freq="1h", rules=None):
    info = InformationSet("11:45", PL, rules=rules or {"sk": PL_CHAIN, "price": Availability.published("13:00", 1, PL)})
    periods = [0] if freq == "1D" else range(24)
    found, _ = info.violations({(p, 1): names for p in periods}, "price", freq, index, day, day)
    return {name: ("late" if any(v.predictor == name for v in found) else "known") for name in names}


def test_a_day_aggregate_is_known_when_every_slot_of_its_day_is():
    """`sk_h-12_day_max_d-3` on 11 October is the maximum of sk from 12:00 on the 7th to 11:00 on the 8th: the 7th's values come
    on the 11th at 14:30, the 8th's on the 9th. The last slot is known, the first isn't. On a UTC index a UTC date holds the last
    two hours of the next Warsaw date: `sk_day_max_d-3` of 10 October (UTC 7 October) has 22 slots out only on the 11th."""
    warsaw = pd.date_range("2026-08-01", "2026-10-25 23:00", freq="1h", tz=PL)
    assert verdict(["sk_h-12_day_max_d-3", "sk_h-12_d-3"], warsaw, "2026-10-11") == dict.fromkeys(
        ["sk_h-12_day_max_d-3", "sk_h-12_d-3"], "late"
    )
    assert verdict(["sk_h-12_day_max_d-3"], warsaw, "2026-10-12") == {"sk_h-12_day_max_d-3": "known"}
    utc = pd.date_range("2026-08-01", "2026-10-25 23:00", freq="1h", tz="UTC")
    assert verdict(["sk_day_max_d-3"], utc, "2026-10-10") == {"sk_day_max_d-3": "late"}
    single = {"sk": Availability.published("14:30", -1, PL), "price": Availability.published("13:00", 1, PL)}
    assert verdict(["sk_day_max_d-3"], utc, "2026-10-10", rules=single) == {"sk_day_max_d-3": "known"}  # as before
    # The probe perturbs what the check calls late: every aggregate cell of 10 October's rows.
    sk = pd.Series(np.arange(len(utc), dtype=float), index=utc)
    data = pd.DataFrame({"sk": sk, "sk_day_max": sk.groupby(utc.date).transform("max")}, index=utc)
    data["sk_day_max_d-3"] = data["sk_day_max"].shift(72)
    info = InformationSet("11:45", PL, rules={"sk": PL_CHAIN})
    out, _, _ = information_set_module.perturb_unknown(data, "2026-10-10", info, columns=["sk_day_max_d-3"])
    rows = utc.normalize() == pd.Timestamp("2026-10-10", tz="UTC")
    assert int((out["sk_day_max_d-3"] != data["sk_day_max_d-3"])[rows].sum()) == 24


def test_a_daily_row_of_another_zone_is_known_when_every_slot_of_its_date_is():
    """A Helsinki date runs from 23:00 Warsaw time on the day before: `sk_d-2` of 10 October holds the Helsinki 8 October, whose
    first hour is Warsaw's 7 October, out on the 11th. On the 11th, the Helsinki 9 October is all out."""
    helsinki = pd.date_range("2026-08-01 12:00", "2026-10-25 12:00", freq="1D", tz="Europe/Helsinki")
    assert verdict(["sk_d-2"], helsinki, "2026-10-10", "1D") == {"sk_d-2": "late"}
    assert verdict(["sk_d-2"], helsinki, "2026-10-11", "1D") == {"sk_d-2": "known"}


def test_a_refit_model_s_rows_are_those_of_its_refit_day_s_fit():
    """With `refit="7D"` the forecasts of 10 and 11 October use the fit of Monday 5 October, whose rows end on the 3rd: `sk_d-2`
    there holds the 1st's value, out on the 5th at 14:30, known by the 10th. Rows of 8 and 9 October, late on the 10th, are no
    fit's: it runs, and the probe moves nothing. With `"1MS"` a fit of 1 October reads rows back to 31 August, where `x` (valid
    from 1 September) is never known: refused, though a fit per forecast with the same window runs, and the probe moves."""
    run_at_raise(["sk_d-2"], {"sk": PL_CHAIN}, "2026-10-10", refit="7D")
    run_at_raise(sk_d2, {"sk": PL_CHAIN}, "2026-10-11", refit="7D")
    with pytest.raises(InformationSetError):
        run_at_raise(["sk_d-2"], {"sk": PL_CHAIN}, "2026-10-10")
    info = run_at_raise(["sk_d-2"], {"sk": PL_CHAIN}, "2026-10-10", refit="7D")
    model = lambda **kw: lambda: OLSModel(["sk_d-2"], training_window=60, **kw)  # noqa: E731
    assert information_set_module.leakage_probe(model(refit="7D"), pl_rows(), "2026-10-10", info, columns=["sk_d-2"]).ok
    rules = {"x": Availability.realised("1h").during(valid_from="2026-09-01")}
    run_at_raise(["x_d-1"], rules, "2026-10-10", window=30)
    with pytest.raises(InformationSetError, match="x_d-1 for period 0 is known never"):
        run_at_raise(["x_d-1"], rules, "2026-10-10", window=30, refit="1MS")
    info = run_at_raise(["x_d-1"], rules, "2026-10-10", window=30)
    late = information_set_module.leakage_probe(
        lambda: OLSModel(["x_d-1"], training_window=30, refit="1MS"), pl_rows(), "2026-10-10", info, columns=["x_d-1"]
    )
    assert not late.ok
