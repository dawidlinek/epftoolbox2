"""Predictor functions of `ctx`: one plan before any fit, one list per forecast, and nothing about it changes a number."""

import inspect
import json
import re
import sys
import zlib
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from epftoolbox2 import presets
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.information_set import Availability, InformationSet, InformationSetError, leakage_probe
from epftoolbox2.models import LassoCVModel, OLSModel, PredictorContext, PredictorPlan, Use
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.models.context import build_plan, list_hash
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import ManifestMismatchError
from epftoolbox2.verification import sample_issue_days
from tests.synthetic import PREDICTORS, calendar_shift, make_market_data

DOCS_PAGE = (
    Path(__file__).resolve().parents[2] / "docs/src/content/docs/model-pipeline/predictor-functions.mdx"
)  # its `lear_*` examples run below
RULES = {
    # The synthetic drivers stand for forecasts published well before their delivery.
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "price": Availability.published("13:00", 1),
    "early": Availability.published("10:15", 0),  # row D holds the value for D+1, published 10:15 on D
}
INFO = InformationSet("11:45", "Europe/Warsaw", rules=RULES)
TEST = ("2023-04-05", "2023-04-14")
MISSING_DAYS = ("2023-04-07", "2023-04-10")  # `early` has no value on these issue days


class InProcessPool:
    """Stands in for the worker pool, whose start is most of the time these runs would take. Only the resume test,
    which is about the number of processes, uses the real one."""

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


@pytest.fixture(scope="module")
def df():
    data = make_market_data(noise=5.0, seed=7)
    data["early"] = data["load_d+1"] + np.random.default_rng(1).normal(0, 20, len(data))
    data["early_d+1"] = calendar_shift(data["early"], 1)
    data.loc[data.index.tz_localize(None).normalize().isin(pd.to_datetime(MISSING_DAYS)), "early"] = np.nan
    return data


def key(record):
    return record["run_date"], record["hour"], record["horizon"]


def predictions(ref):
    return {key(r): r["prediction"] for r in ref._results}


def lines(records):
    """Records as text, which compares them bit for bit (and NaN equal to NaN), whatever the order they finished in."""
    return sorted(json.dumps(r, sort_keys=True) for r in records)


def without_list(records):
    return [{k: v for k, v in r.items() if k != "predictor_list"} for r in records]


def stored_lines(path):
    return sorted(line for line in Path(path).read_text().splitlines() if line)


# ---------------------------------------------------------------------------
# A function returning the static list is the static list
# ---------------------------------------------------------------------------


def static_list(ctx):
    return PREDICTORS


SWITCHES = {
    "spring": dict(start="2023-01-01", days=120, test=("2023-03-25", "2023-03-27")),
    "autumn": dict(start="2023-08-01", days=120, test=("2023-10-28", "2023-10-30")),
}
LASSO_DAYS = {"spring": "2023-03-26", "autumn": "2023-10-29"}


def switch_data(spec, freq):
    return make_market_data(start=spec["start"], days=spec["days"], freq=freq, horizon=3, noise=5.0, seed=3)


@pytest.mark.parametrize("switch", list(SWITCHES))
@pytest.mark.parametrize("freq", ["1h", "15min"])
@pytest.mark.parametrize("horizon", [1, 3])
def test_ols_on_a_function_returning_the_static_list_is_bitwise_identical(switch, freq, horizon):
    spec = SWITCHES[switch]
    frame = switch_data(spec, freq)
    common = dict(
        test_start=spec["test"][0], test_end=spec["test"][1], horizon=horizon, freq=freq, information_set=INFO
    )
    static = OLSModel(PREDICTORS, training_window=45).run(frame, **common)
    function = OLSModel(static_list, training_window=45).run(frame, **common)

    assert lines(static._results) == lines(without_list(function._results))
    assert static.predictor_lists is None
    assert len(function.predictor_lists) == horizon  # {horizon} gives another list per horizon
    assert {r["predictor_list"] for r in function._results} == {d["hash"] for d in function.predictor_lists}
    assert function.count == static.count and function.produced == static.produced
    assert function.skipped == static.skipped


@pytest.mark.parametrize(
    ("switch", "freq", "horizon"),
    [("spring", "1h", 1), ("autumn", "1h", 3), ("autumn", "15min", 1), ("spring", "15min", 3)],
)
def test_lasso_on_a_function_returning_the_static_list_is_bitwise_identical(switch, freq, horizon):
    frame = switch_data(SWITCHES[switch], freq)
    day = LASSO_DAYS[switch]
    common = dict(test_start=day, test_end=day, horizon=horizon, freq=freq, information_set=INFO)
    static = LassoCVModel(PREDICTORS, training_window=14, cv=2, max_iter=200).run(frame, **common)
    function = LassoCVModel(static_list, training_window=14, cv=2, max_iter=200).run(frame, **common)
    assert lines(static._results) == lines(without_list(function._results))


def turns_non_binary(frame, column, day):
    out = frame.copy()
    binary = (np.random.default_rng(5).random(len(frame)) > 0.5).astype(float)
    rows = out.index.tz_localize(None).normalize() >= pd.Timestamp(day)
    binary[rows] = binary[rows] * 0.5 + 0.25
    out[column] = binary
    return out


NOT_ALWAYS_BINARY = PREDICTORS + ["flag"]


def not_always_binary_list(ctx):
    return NOT_ALWAYS_BINARY


def test_the_identity_holds_for_a_column_that_stops_being_binary_inside_the_test_period(df):
    """A fixed list reads its mask from the first test day's window, and so must the function: "over all test days" would
    scale this column for all of them, and let a later value change an earlier forecast."""
    frame = turns_non_binary(df, "flag", "2023-04-08")
    rules = InformationSet("11:45", "Europe/Warsaw", rules={**RULES, "flag": Availability.static()})
    common = dict(test_start=TEST[0], test_end=TEST[1], horizon=1, information_set=rules)
    static = OLSModel(NOT_ALWAYS_BINARY, training_window=14).run(frame, **common)
    function = OLSModel(not_always_binary_list, training_window=14).run(frame, **common)
    assert lines(static._results) == lines(without_list(function._results))


def test_the_identity_holds_for_a_column_that_starts_inside_the_test_period(df):
    frame = df.copy()
    frame["late_start"] = frame["load"] * 3.0
    frame.loc[frame.index.tz_localize(None) < pd.Timestamp("2023-04-08"), "late_start"] = np.nan
    rules = InformationSet("11:45", "Europe/Warsaw", rules={**RULES, "late_start": Availability.static()})
    common = dict(test_start=TEST[0], test_end=TEST[1], horizon=1, information_set=rules)

    def with_it(ctx):
        return PREDICTORS + ["late_start"]

    static = LassoCVModel(PREDICTORS + ["late_start"], training_window=14, cv=3, max_iter=500).run(frame, **common)
    function = LassoCVModel(with_it, training_window=14, cv=3, max_iter=500).run(frame, **common)
    assert lines(static._results) == lines(without_list(function._results))
    assert function.skipped == static.skipped


# ---------------------------------------------------------------------------
# Fixed lists and (horizon) / (horizon, hour) callables are untouched
# ---------------------------------------------------------------------------


def test_horizon_and_hour_callables_are_not_given_a_context(df):
    seen = []

    def by_horizon(horizon):
        seen.append(horizon)
        return PREDICTORS

    def by_horizon_and_hour(horizon, hour):
        seen.append((horizon, hour))
        return PREDICTORS

    static = OLSModel(PREDICTORS, training_window=45).run(df, *TEST, horizon=1)
    for function in (by_horizon, by_horizon_and_hour):
        model = OLSModel(function, training_window=45)
        ref = model.run(df, *TEST, horizon=1)
        assert model._ctx_fn is None
        assert ref.predictor_lists is None
        assert lines(ref._results) == lines(static._results)
        assert all("predictor_list" not in record for record in ref._results)
    assert 1 in seen and (1, 23) in seen


def test_a_fixed_list_gets_no_list_hash_and_its_manifest_no_function(df, tmp_path):
    path = tmp_path / "fixed.jsonl"
    OLSModel(PREDICTORS, training_window=45).run(df, "2023-04-05", "2023-04-05", horizon=1, save_to=str(path))
    assert "predictor_list" not in path.read_text()
    manifest = json.loads((tmp_path / "fixed.manifest.json").read_text())
    assert "predictor_function" not in manifest and "predictor_lists" not in manifest


def test_a_plan_is_refused_for_a_model_with_fixed_lists(df):
    plan = OLSModel(static_list)._plan_predictors(df, "price", 1, "1h", INFO, *TEST)
    with pytest.raises(ValueError, match="another predictor function"):
        OLSModel(PREDICTORS).run(df, *TEST, horizon=1, plan=plan)


# ---------------------------------------------------------------------------
# The context
# ---------------------------------------------------------------------------


def test_the_context_says_which_forecast_it_is_and_never_holds_the_frame(df):
    seen = []

    def records_itself(ctx):
        seen.append(ctx)
        return PREDICTORS

    OLSModel(records_itself, training_window=45).run(df, "2023-04-06", "2023-04-06", horizon=2)
    assert len(seen) == 48
    ctx = next(c for c in seen if c.horizon == 2 and c.period == 13)
    assert isinstance(ctx, PredictorContext)
    assert ctx.hour == ctx.period == 13
    assert str(ctx.issue_day) == "2023-04-06" and str(ctx.target_day) == "2023-04-08"
    assert ctx.target_weekday == 5  # a Saturday
    assert ctx.zone == "Europe/Warsaw" and ctx.freq == "1h"
    assert "2023-04-06" in repr(ctx)
    assert not any(isinstance(getattr(ctx, name), pd.DataFrame) for name in dir(ctx) if not name.startswith("_"))


def test_published_takes_one_name_or_several_and_expands_the_horizon(df):
    answers = {}

    def asks(ctx):
        if ctx.period == 12:
            answers[(str(ctx.issue_day), ctx.horizon)] = (
                ctx.published("load_d+{horizon}"),
                ctx.published(["load_d+{horizon}", "wind_d+{horizon}"]),
                ctx.published(c for c in ["early"]),
            )
        return PREDICTORS

    OLSModel(asks, training_window=45).run(df, "2023-04-06", "2023-04-07", horizon=1, information_set=INFO)
    assert answers[("2023-04-06", 1)] == (True, True, True)
    assert answers[("2023-04-07", 1)] == (True, True, False)  # `early` is missing on that issue day


def test_a_misspelt_column_raises_whatever_the_other_columns_answer(df):
    def misspelt_after_a_false(ctx):
        return PREDICTORS + (["early"] if ctx.published(["early_d+1", "eraly"]) else [])

    with pytest.raises(KeyError, match="eraly"):
        OLSModel(misspelt_after_a_false, training_window=45).run(df, *TEST, horizon=1, information_set=INFO)

    def misspelt_value(ctx):
        return PREDICTORS if ctx.value("eraly") > 0 else []

    with pytest.raises(KeyError, match="eraly"):
        OLSModel(misspelt_value, training_window=45).run(df, *TEST, horizon=1)


def test_published_is_false_for_a_column_known_later_with_a_warning_naming_the_rule(df):
    def later(ctx):
        assert not ctx.published(["early_d+1"])
        return PREDICTORS

    with pytest.warns(UserWarning, match=r"never True.*early_d\+1.*known D\+1 10:15, after the issue time"):
        OLSModel(later, training_window=45).run(df, *TEST, horizon=1, information_set=INFO)


def test_a_published_argument_that_is_never_true_over_the_run_warns_once(df, recwarn):
    frame = df.assign(empty=np.nan)
    rules = InformationSet("11:45", "Europe/Warsaw", rules={**RULES, "empty": Availability.static()})

    def asks_twice(ctx):
        ctx.published("empty")
        ctx.published("early")  # True on most days, False on two
        return PREDICTORS

    OLSModel(asks_twice, training_window=45).run(frame, *TEST, horizon=1, information_set=rules)
    messages = [str(w.message) for w in recwarn.list if "never True" in str(w.message)]
    assert len(messages) == 1
    assert "'empty' has no value on any forecast's row" in messages[0]


def test_without_an_information_set_only_a_missing_value_is_unpublished(df):
    seen = {}

    def asks(ctx):
        seen[(str(ctx.issue_day), ctx.period)] = (
            ctx.published("early_d+1"),
            ctx.published("early"),
            ctx.value("early_d+1"),
        )
        return PREDICTORS

    OLSModel(asks, training_window=45).run(df, "2023-04-06", "2023-04-07", horizon=1)
    # `early_d+1` is late by the rule, which nobody asks here; `early` has its hole on 2023-04-07.
    late_published, published, late_value = seen[("2023-04-07", 12)]
    assert late_published is True and published is False
    assert late_value == df.loc["2023-04-07 12:00", "early_d+1"]


def test_value_returns_the_cell_of_the_forecasts_row_and_nan_for_a_missing_one(df):
    values = {}

    def reads(ctx):
        values[(str(ctx.issue_day), ctx.period)] = ctx.value("early")
        return PREDICTORS

    OLSModel(reads, training_window=45).run(df, "2023-04-06", "2023-04-07", horizon=1, information_set=INFO)
    assert values[("2023-04-06", 9)] == df.loc["2023-04-06 09:00", "early"]
    assert isinstance(values[("2023-04-06", 9)], float)
    assert np.isnan(values[("2023-04-07", 9)])


def test_value_known_after_the_issue_time_is_refused(df):
    def sneaky(ctx):
        return PREDICTORS + (["price_d-1"] if ctx.value("early_d+1") > 0 else [])

    with pytest.raises(InformationSetError, match=r"early_d\+1.*known D\+1 10:15, after the issue time") as refused:
        OLSModel(sneaky, training_window=45).run(df, *TEST, horizon=1, information_set=INFO)
    assert any("sneaky" in note and "2023-04-05" in note for note in refused.value.__notes__)


def test_only_nan_counts_as_missing_and_a_non_numeric_column_reads_as_itself(df):
    frame = df.assign(infinite=np.inf, stamp=pd.Timestamp("2023-01-01"), label="x")
    read = {}

    def reads(ctx):
        read[ctx.period] = (ctx.published("infinite"), ctx.value("infinite"), ctx.value("stamp"), ctx.value("label"))
        return PREDICTORS

    OLSModel(reads, training_window=45).run(frame, "2023-04-06", "2023-04-06", horizon=1)
    assert read[5] == (True, np.inf, pd.Timestamp("2023-01-01"), "x")


def test_a_column_without_a_rule_is_not_handed_out_whatever_unknown_says(df):
    frame = df.assign(mystery=1.0)

    def asks(ctx):
        return PREDICTORS if ctx.published("mystery") else PREDICTORS

    rules = InformationSet("11:45", "Europe/Warsaw", rules=RULES, unknown="ignore")
    with pytest.raises(InformationSetError, match="no availability rule for 'mystery'"):
        OLSModel(asks, training_window=45).run(frame, *TEST, horizon=1, information_set=rules)


def test_a_repeated_autumn_hour_reads_the_row_the_model_forecasts_from():
    frame = make_market_data(start="2023-08-01", days=120, noise=5.0, seed=3)
    day = "2023-10-29"
    twice = frame[frame.index.tz_localize(None) == pd.Timestamp(f"{day} 02:00")]
    assert len(twice) == 2 and twice["wind"].iloc[0] != twice["wind"].iloc[1]
    read = {}

    def reads(ctx):
        read[ctx.period] = ctx.value("wind")
        return PREDICTORS

    OLSModel(reads, training_window=45).run(frame, day, day, horizon=1)
    assert read[2] == twice["wind"].iloc[1]


def test_the_hour_skipped_in_spring_has_no_cell():
    frame = make_market_data(noise=5.0, seed=3)
    day = "2023-03-26"
    read = {}

    def reads(ctx):
        read[ctx.period] = (ctx.value("wind"), ctx.published("wind"))
        return PREDICTORS

    OLSModel(reads, training_window=45).run(frame, day, day, horizon=1)
    assert np.isnan(read[2][0]) and read[2][1] is False
    assert read[3][1] is True


@pytest.mark.parametrize(
    "day",
    ["2023-04-12", "2023-03-26", "2023-10-29"],
    ids=["plain", "spring", "autumn"],
)
@pytest.mark.parametrize("zone", ["Europe/Warsaw", "America/New_York"])
def test_published_agrees_with_the_check_on_every_period_including_profiles_and_aggregates(day, zone):
    """`published` and the information-set check read the same availability path: a profile column at its slot, an
    aggregate at its day's last slot, each through the column it was built from."""
    frame = make_market_data(start="2023-01-01", days=320, noise=5.0, seed=3, tz="Europe/Warsaw").rename(
        columns={"load": "load_actual"}
    )
    names = {
        "load_actual": "plain",
        "load_actual_h05": "early slot",
        "load_actual_h23": "late slot",
        "load_actual_h23_d-1": "yesterday's late slot",
        "load_actual_day_max": "today's aggregate",
        "load_actual_day_max_d-1": "yesterday's aggregate",
        "load_forecast_h12_d+1": "tomorrow's slot of a forecast",
        "price_h05": "a price's slot",
    }
    frame = frame.assign(**{name: 1.0 for name in names})
    info = InformationSet("11:45", zone)
    report = {}

    def asks(ctx):
        for name in names:
            report[(name, ctx.period)] = ctx.published(name)
        return ["wind"]

    plan = build_plan(asks, frame, [pd.Timestamp(day).date()], 1, "1h", info, warn_unpublished=False)
    assert plan.calls == 24
    for name in names:
        found, _ = info.violations({(period, 1): [name] for period in range(24)}, "price", "1h", frame.index, day, day)
        late = {v.period for v in found if v.role == "predictor"}
        has_row = {
            period: bool(((frame.index.tz_localize(None).normalize() == day) & (frame.index.hour == period)).any())
            for period in range(24)
        }
        assert {p for p in range(24) if not report[(name, p)]} == late | {p for p, on in has_row.items() if not on}, (
            name
        )


# ---------------------------------------------------------------------------
# One plan, built once, not on the model
# ---------------------------------------------------------------------------

CALLS = {"count": 0}


def counts_calls(ctx):
    CALLS["count"] += 1
    return PREDICTORS


def test_the_pipeline_plans_each_forecast_once_and_keeps_nothing_on_the_model(df):
    CALLS["count"] = 0
    model = OLSModel(counts_calls, training_window=45, name="counted")
    report = (
        ModelPipeline(information_set=INFO).add_model(model).add_evaluator(MAEEvaluator()).run(df, *TEST, horizon=2)
    )
    assert CALLS["count"] == 10 * 24 * 2
    counts = report.forecast_counts().set_index("model")
    assert counts.loc["counted", "planned"] == counts.loc["counted", "produced"] == 480
    assert report.summary()["MAE"].notna().all()
    assert not [name for name, value in vars(model).items() if isinstance(value, PredictorPlan)]


def test_a_report_merged_from_the_saved_results_of_a_function(df, tmp_path):
    pipeline = (
        ModelPipeline(information_set=INFO)
        .add_model(OLSModel(by_horizon_period_and_publication, training_window=45, name="dyn"))
        .add_model(OLSModel(PREDICTORS, training_window=45, name="fixed"))
        .add_evaluator(MAEEvaluator())
    )
    ran = pipeline.run(df, *TEST, horizon=1, save_dir=str(tmp_path))
    merged = pipeline.report(tmp_path, *TEST, horizon=1)
    assert merged.summary().equals(ran.summary())
    assert merged.predictions().equals(ran.predictions())
    assert list(merged.forecast_counts()["produced"]) == [240, 240]


def test_forecast_only_plans_the_forecast_days(df):
    pipeline = ModelPipeline(information_set=INFO).add_model(
        OLSModel(by_horizon_period_and_publication, training_window=45, name="dyn")
    )
    report = pipeline.run(df, "2023-04-05", "2023-04-05", horizon=1, forecast_only=True)
    assert report.refs["dyn"].produced == 24
    assert len(report.refs["dyn"].predictor_lists) == 2


def test_a_direct_run_plans_once_too(df):
    CALLS["count"] = 0
    OLSModel(counts_calls, training_window=45).run(df, *TEST, horizon=1, information_set=INFO)
    assert CALLS["count"] == 240


def test_a_plan_is_refused_for_another_run(df):
    model = OLSModel(static_list, training_window=45)
    plan = model._plan_predictors(df, "price", 1, "1h", INFO, *TEST)
    with pytest.raises(ValueError, match="another test period"):
        model.run(df, "2023-04-05", "2023-04-06", horizon=1, information_set=INFO, plan=plan)
    with pytest.raises(ValueError, match="another horizon"):
        model.run(df, *TEST, horizon=2, information_set=INFO, plan=plan)
    with pytest.raises(ValueError, match="another data"):
        model.run(df.copy(), *TEST, horizon=1, information_set=INFO, plan=plan)
    with pytest.raises(ValueError, match="another information set"):
        model.run(df, *TEST, horizon=1, information_set=INFO.copy(), plan=plan)
    with pytest.raises(ValueError, match="another predictor function"):
        OLSModel(counts_calls).run(df, *TEST, horizon=1, information_set=INFO, plan=plan)


def test_availability_is_computed_once_per_distinct_name_however_many_lists(df, monkeypatch):
    from epftoolbox2 import information_set as module

    def many_lists(ctx):
        return PREDICTORS + (["price_d-1"] if ctx.issue_day.day % 2 else []) + [f"wind_d+{1}"] * 0

    calls = []
    real = module._IssueClock.available

    def counted(self, col, periods, step, row_day=0):
        calls.append(col.name)
        return real(self, col, periods, step, row_day)

    monkeypatch.setattr(module._IssueClock, "available", counted)
    model = OLSModel(many_lists, training_window=45)
    plan = model._plan_predictors(df, "price", 1, "1h", INFO, *TEST)
    assert len(plan.lists) == 2
    model._check_inputs(df, "price", 1, "1h", INFO, *TEST, plan=plan)
    names = {name.replace("{horizon}", "1") for name in PREDICTORS} | {"price_d-1"}
    assert sorted(calls) == sorted([*names, "price"])  # `price` is the newest training target


def test_lists_are_compared_after_expansion_and_with_their_training_rule():
    assert list_hash(["a", "b"]) != list_hash(["b", "a"])
    assert list_hash(["a"], "all") != list_hash(["a"], "other")


def as_use(ctx):
    return Use(PREDICTORS)


def test_use_with_train_on_all_is_the_plain_list(df):
    plain = OLSModel(static_list, training_window=45).run(df, *TEST, horizon=1, information_set=INFO)
    used = OLSModel(as_use, training_window=45).run(df, *TEST, horizon=1, information_set=INFO)
    assert lines(plain._results) == lines(used._results)
    assert used.predictor_lists == plain.predictor_lists
    assert used.predictor_lists[0]["train_on"] == "all"


def test_use_refuses_another_rule_and_a_string():
    with pytest.raises(ValueError, match="train_on must be"):
        Use(PREDICTORS, train_on="weekends")
    with pytest.raises(TypeError, match="list of column names"):
        Use("price_d-1")


@pytest.mark.parametrize(
    ("returned", "error", "message"),
    [
        ("price_d-1", TypeError, "returned str"),
        (None, TypeError, "returned NoneType"),
        ([], ValueError, "empty list"),
        (["price_d-1", 3], TypeError, "other than names"),
        (["load_d+{horizon}", "load_d+1"], ValueError, "repeated names"),
    ],
)
def test_a_function_returning_something_unusable_raises(df, returned, error, message):
    def bad(ctx):
        return returned

    with pytest.raises(error, match=message):
        OLSModel(bad, training_window=45).run(df, *TEST, horizon=1)


def test_a_name_the_data_does_not_have_or_the_target_itself_is_refused_like_in_a_fixed_list(df):
    def unknown_column(ctx):
        return PREDICTORS + ["nope"]

    with pytest.raises(ValueError, match="Predictor column.*not in the data.*nope"):
        OLSModel(unknown_column, training_window=45).run(df, *TEST, horizon=1)

    def target_itself(ctx):
        return PREDICTORS + ["price_d+{horizon}"]

    with pytest.raises(ValueError, match=r"price_d\+1.*look-ahead"):
        OLSModel(target_itself, training_window=45).run(df, *TEST, horizon=1)


def test_a_function_needs_the_test_period_to_be_planned(df):
    with pytest.raises(ValueError, match="needs the test period"):
        OLSModel(static_list)._check_inputs(df, "price", 1, "1h")


# ---------------------------------------------------------------------------
# A dynamic function gives what a static model with each forecast's list gives
# ---------------------------------------------------------------------------


def by_horizon_period_and_publication(ctx):
    x = ["load_d+{horizon}", "is_weekend_d+{horizon}", "is_weekday_d+{horizon}"]
    if ctx.horizon == 1:
        x.append("wind_d+{horizon}")
    if 8 <= ctx.period <= 18:
        x.append("price_d-1")
    if ctx.published(["early"]):
        x.append("early")
    return x


@pytest.mark.parametrize(
    ("make", "test"),
    [
        (lambda p: OLSModel(p, training_window=45), TEST),
        (lambda p: LassoCVModel(p, training_window=14, cv=3, max_iter=500), ("2023-04-06", "2023-04-07")),
    ],
    ids=["OLS", "LassoCV"],
)
def test_each_forecast_equals_a_static_model_with_its_list(df, make, test):
    dynamic = make(by_horizon_period_and_publication).run(df, *test, horizon=2, information_set=INFO)
    assert dynamic.produced == dynamic.count
    records = {key(r): r for r in dynamic._results}
    assert len(dynamic.predictor_lists) == 8  # 2 horizons x daylight or not x `early` there or not
    checked = 0
    for entry in dynamic.predictor_lists:
        span = entry["first_issue_day"], entry["last_issue_day"]
        static = predictions(make(entry["predictors"]).run(df, *span, horizon=2))
        for forecast, record in records.items():
            if record["predictor_list"] == entry["hash"]:
                assert record["prediction"] == pytest.approx(static[forecast], abs=1e-9)
                checked += 1
    assert checked == len(records)


def test_forecasts_are_grouped_by_the_list_the_function_returns(df):
    ref = OLSModel(by_horizon_period_and_publication, training_window=45).run(
        df, *TEST, horizon=1, information_set=INFO
    )
    described = {(d["n_predictors"], d["forecasts"], d["issue_days"]) for d in ref.predictor_lists}
    # 11 daylight hours (08-18), 2 issue days without the early column
    assert described == {(5, 13 * 8, 8), (6, 11 * 8, 8), (4, 13 * 2, 2), (5, 11 * 2, 2)}
    assert ref.produced == 240
    assert {r["predictor_list"] for r in ref._results} == {d["hash"] for d in ref.predictor_lists}


def test_each_list_trains_on_the_rows_where_its_columns_exist(df):
    """Rows without `early` stay out of the fit of the list that holds it and in the fit of the list that doesn't."""
    run_day, period = "2023-04-14", 12
    ref = OLSModel(by_horizon_period_and_publication, training_window=20).run(
        df, run_day, run_day, horizon=1, information_set=INFO
    )
    got = predictions(ref)[(run_day, period, 1)]

    rows = df[(df.index.hour == period)].loc[:run_day].tail(22)
    names = ["load_d+1", "is_weekend_d+1", "is_weekday_d+1", "wind_d+1", "price_d-1", "early"]
    x, y = rows[names].to_numpy(), rows["price"].shift(-1).to_numpy()
    train = np.arange(len(rows) - 2)
    train = train[np.isfinite(x[train]).all(axis=1) & np.isfinite(y[train])]
    assert len(train) == 18  # two of the 20 rows have no `early`
    scaled = [0, 3, 4, 5]
    mean, std = x[train][:, scaled].mean(axis=0), x[train][:, scaled].std(axis=0, ddof=1)
    design = x.copy()
    design[:, scaled] = (design[:, scaled] - mean) / std
    y_mean, y_std = y[train].mean(), y[train].std(ddof=1)
    beta = np.linalg.lstsq(design[train], (y[train] - y_mean) / y_std, rcond=None)[0]
    assert got == pytest.approx(design[-1] @ beta * y_std + y_mean, abs=1e-9)


@pytest.mark.parametrize("values", ["continuous", "binary"])
def test_a_column_starting_inside_the_test_period_is_scaled_from_when_it_has_values(df, values):
    """Its list starts with it, with no training row at all. A mask read from that empty window alone would leave the
    column unscaled for good. The later forecasts must equal a static model that starts when the column has a history."""
    frame = df.copy()
    rng = np.random.default_rng(2)
    frame["late_start"] = (
        frame["load"] * 3.0 if values == "continuous" else (rng.random(len(frame)) > 0.5).astype(float)
    )
    frame.loc[frame.index.tz_localize(None) < pd.Timestamp("2023-04-01"), "late_start"] = np.nan
    rules = InformationSet("11:45", "Europe/Warsaw", rules={**RULES, "late_start": Availability.static()})

    def when_published(ctx):
        return PREDICTORS + (["late_start"] if ctx.published("late_start") else [])

    make = lambda p: LassoCVModel(p, training_window=14, cv=3, max_iter=500)  # noqa: E731
    dynamic = make(when_published).run(frame, "2023-03-31", "2023-04-06", horizon=1, information_set=rules)
    assert dynamic.skipped.get("no_finite_training_rows", 0) == 48  # its first two days have nothing to train on
    static = predictions(make(PREDICTORS + ["late_start"]).run(frame, "2023-04-05", "2023-04-06", horizon=1))
    produced = predictions(dynamic)
    for forecast, prediction in static.items():
        assert produced[forecast] == pytest.approx(prediction, abs=1e-9)


# ---------------------------------------------------------------------------
# The information set
# ---------------------------------------------------------------------------


def test_a_late_column_in_one_returned_list_is_refused_naming_the_model_list_columns_and_periods(df):
    def late_at_hour_3(ctx):
        return PREDICTORS + (["early_d+1"] if ctx.period == 3 else [])

    model = OLSModel(late_at_hour_3, training_window=45, name="dynamic")
    with pytest.raises(InformationSetError) as refused:
        model.run(df, *TEST, horizon=1, information_set=INFO)
    message = str(refused.value)
    assert re.search(r"^dynamic, predictor list 2 of 2 \(\w{12}\)", message)
    assert "early_d+1" in message and "Periods 3, horizons 1" in message
    assert "known D+1 10:15, issued D 11:45" in message


def test_the_pipeline_refuses_a_late_list_before_any_fit(df, monkeypatch):
    def late_at_hour_3(ctx):
        return PREDICTORS + (["early_d+1"] if ctx.period == 3 else [])

    def no_fit(self, *args):
        raise AssertionError("a model ran")

    monkeypatch.setattr(OLSModel, "_fit_predict", no_fit)
    pipeline = (
        ModelPipeline(information_set=INFO, availability="raise")
        .add_model(OLSModel(PREDICTORS, training_window=45, name="fine"))
        .add_model(OLSModel(late_at_hour_3, training_window=45, name="late"))
    )
    with pytest.raises(InformationSetError, match="late, predictor list 2 of 2"):
        pipeline.run(df, *TEST, horizon=1)


def test_a_column_is_judged_on_the_cells_its_list_reads(df):
    """The check is per cell, not per name: by its weekly rule `load_forecast_daily_max_d+9` (published on Fridays for
    the next Monday to Sunday) is known on Fridays only, so a list using it on Fridays passes, and a fixed list or a list
    using it on another day doesn't."""
    name = "load_forecast_daily_max_d+9"
    frame = df.assign(**{name: 1.0})

    def on_fridays(ctx):
        return PREDICTORS + ([name] if ctx.issue_day.weekday() == 4 else [])

    def on_thursdays(ctx):
        return PREDICTORS + ([name] if ctx.issue_day.weekday() == 3 else [])

    fine = OLSModel(on_fridays, training_window=45)
    plan = fine._plan_predictors(frame, "price", 1, "1h", INFO, *TEST)
    fine._check_inputs(frame, "price", 1, "1h", INFO, *TEST, plan=plan)
    with pytest.raises(InformationSetError, match="predictor list 2 of 2"):
        OLSModel(on_thursdays, training_window=45)._check_inputs(frame, "price", 1, "1h", INFO, *TEST)
    with pytest.raises(InformationSetError):
        OLSModel(PREDICTORS + [name], training_window=45)._check_inputs(frame, "price", 1, "1h", INFO, *TEST)


def test_no_rule_for_a_listed_column_follows_the_unknown_policy(df):
    frame = df.assign(mystery=1.0)

    def with_mystery(ctx):
        return PREDICTORS + ["mystery"]

    with pytest.raises(InformationSetError, match="no availability rule.*mystery"):
        OLSModel(with_mystery, training_window=45).run(frame, *TEST, horizon=1, information_set=INFO)
    tolerant = InformationSet("11:45", "Europe/Warsaw", rules=RULES, unknown="warn")
    with pytest.warns(UserWarning, match="no availability rule.*mystery"):
        OLSModel(with_mystery, training_window=45).run(frame, *TEST, horizon=1, information_set=tolerant)


def test_a_target_not_known_at_the_issue_time_is_refused_like_for_a_fixed_list(df):
    frame = df.assign(slow=df["price"], **{"slow_d-1": df["price_d-1"]})
    rules = InformationSet(
        "11:45", "Europe/Warsaw", rules={**RULES, "slow": Availability.published("13:00", -1)}
    )  # known only the day after delivery, at 13:00: yesterday's value isn't known at 11:45

    def plain(ctx):
        return PREDICTORS

    with pytest.raises(InformationSetError, match="training target"):
        OLSModel(PREDICTORS, training_window=45).run(frame, *TEST, target="slow", horizon=1, information_set=rules)
    with pytest.raises(InformationSetError, match="training target"):
        OLSModel(plain, training_window=45).run(frame, *TEST, target="slow", horizon=1, information_set=rules)


def test_leakage_probe_passes_for_a_function_of_published_cells(df):
    columns = ["price", "price_d-1", "early", *[p.replace("{horizon}", "1") for p in PREDICTORS]]
    result = leakage_probe(
        lambda: OLSModel(by_horizon_period_and_publication, training_window=45),
        df[columns],
        "2023-04-07",
        INFO,
        horizon=1,
    )
    result.assert_ok()
    assert result.forecasts == 24


def test_verify_reports_a_dynamic_model_ok(df):
    report = (
        ModelPipeline(information_set=INFO)
        .add_model(OLSModel(by_horizon_period_and_publication, training_window=45, name="dyn"))
        .verify(df, *TEST, horizon=2, days=2)
    )
    report.assert_ok()
    assert report.table().iloc[0]["status"] == "ok"


def test_verify_finds_a_late_name_in_a_list(df):
    def late_at_noon(ctx):
        return PREDICTORS + (["early_d+1"] if ctx.period == 12 else [])

    report = (
        ModelPipeline(information_set=INFO)
        .add_model(OLSModel(late_at_noon, training_window=45, name="late"))
        .verify(df, *TEST, horizon=1, days=1, positive_control=False)
    )
    assert report.status("late") == "leak"
    assert report.checks[0].probe.changed == [(12, 1)]


def test_verify_refuses_a_late_value():
    frame = make_market_data(noise=5.0, seed=7).assign(**{"secret_d+1": 1.0})
    rules = InformationSet("11:45", "Europe/Warsaw", rules={**RULES, "secret": Availability.published("13:00", 0)})

    def peeks(ctx):
        return PREDICTORS if ctx.value("secret_d+1") > 0 else PREDICTORS + ["price_d-1"]

    pipeline = ModelPipeline(information_set=rules, availability="raise").add_model(
        OLSModel(peeks, training_window=45, name="peeks")
    )
    with pytest.raises(InformationSetError, match=r"secret_d\+1"):
        pipeline.verify(frame, *TEST, horizon=1, days=1)


def leaking_rebuild(frame):
    """Builds `signal_d-1` from the next day: by its name and rule it is known."""
    out = frame.copy()
    out["signal_d-1"] = calendar_shift(out["signal"], 1)
    return out


def test_verify_finds_a_list_chosen_by_a_value_a_transformer_leaked():
    """The forecast reads nothing late. Only the choice of its list does, through a legally named column that holds the
    next day's value, so only `rebuild` can show it."""
    source = make_market_data(noise=5.0, seed=7, days=150)
    source["signal"] = np.random.default_rng(8).normal(size=len(source))
    rules = InformationSet("11:45", "Europe/Warsaw", rules={**RULES, "signal": Availability.realised("1h")})

    def chooses(ctx):
        return PREDICTORS if ctx.value("signal_d-1") < 100 else PREDICTORS + ["price_d-1"]

    pipeline = ModelPipeline(information_set=rules).add_model(OLSModel(chooses, training_window=45, name="chooses"))
    report = pipeline.verify(source, *TEST, horizon=1, days=1, positive_control=False, rebuild=leaking_rebuild)
    assert report.status("chooses") == "leak"


def positive_signal():
    source = make_market_data(noise=5.0, seed=7, days=150)
    source["signal"] = 20.0 + np.random.default_rng(1).normal(0, 2, len(source))
    rules = InformationSet("11:45", "Europe/Warsaw", rules={**RULES, "signal": Availability.realised("1h")})
    return source, rules


def properly_lagged(frame):
    return frame.assign(**{"signal_d-1": calendar_shift(frame["signal"], -1)})


def stays_on_the_branch_the_noise_lands_on(ctx):
    return PREDICTORS if ctx.value("signal_d-1") > 0 else PREDICTORS + ["price_d-1"]


def test_verify_finds_a_list_chosen_by_a_leaked_value_even_if_the_noise_leaves_the_list_alone():
    """Large positive noise lands where `> 0` already was, so no forecast moves. The cell the function read did."""
    source, rules = positive_signal()
    pipeline = ModelPipeline(information_set=rules).add_model(
        OLSModel(stays_on_the_branch_the_noise_lands_on, training_window=45, name="gate")
    )
    report = pipeline.verify(source, *TEST, horizon=1, days=1, positive_control=False, rebuild=leaking_rebuild)
    check = report.checks[0]
    assert report.status("gate") == "leak"
    assert check.probe.max_abs_change == 0.0  # the forecasts alone say nothing here
    assert check.probe.moved_reads == {"signal_d-1": 24}
    assert "read through ctx and moved" in report.problems()[0]
    with pytest.raises(AssertionError, match="Read through ctx and moved"):
        check.probe.assert_ok()


def test_verify_passes_the_same_function_on_a_properly_built_column():
    source, rules = positive_signal()
    pipeline = ModelPipeline(information_set=rules).add_model(
        OLSModel(stays_on_the_branch_the_noise_lands_on, training_window=45, name="gate")
    )
    report = pipeline.verify(source, *TEST, horizon=1, days=2, rebuild=properly_lagged)
    assert report.status("gate") == "ok", report.problems()


def test_asking_whether_a_late_cell_is_published_is_not_reading_it(df):
    """The answer is False whatever the cell holds, and the probe perturbs that cell."""

    def asks_about_a_late_column(ctx):
        return PREDICTORS + (["price_d-1"] if ctx.published("early_d+1") else [])

    with pytest.warns(UserWarning, match="never True"):
        model = OLSModel(asks_about_a_late_column, training_window=45)
        model.run(df, "2023-04-05", "2023-04-05", horizon=1, information_set=INFO)
    result = leakage_probe(lambda: OLSModel(asks_about_a_late_column, training_window=45), df, "2023-04-07", INFO)
    result.assert_ok()
    assert result.moved_reads == {}


def on_sundays_only(ctx):
    return PREDICTORS + (["signal_d-1"] if ctx.issue_day.weekday() == 6 else ["price_d-1"])


def test_verify_does_not_pass_a_column_only_unsampled_days_read():
    """A fixed list reads the same columns on every sampled day. A function may read a leaking one on Sundays only."""
    source = make_market_data(noise=5.0, seed=7, days=150)
    source["signal"] = np.random.default_rng(8).normal(size=len(source))
    rules = InformationSet("11:45", "Europe/Warsaw", rules={**RULES, "signal": Availability.realised("1h")})
    week = ("2023-04-03", "2023-04-09")  # 2023-04-09 is the Sunday
    pipeline = ModelPipeline(information_set=rules).add_model(OLSModel(on_sundays_only, training_window=45, name="m"))

    some_days = pipeline.verify(source, week[0], "2023-04-08", horizon=1, days=1, rebuild=leaking_rebuild)
    assert some_days.checks[0].not_sampled == []  # that period has no Sunday, so nothing is left unread
    assert some_days.status("m") == "ok"

    weekdays = next(
        seed
        for seed in range(50)
        if pd.Timestamp(sample_issue_days(source.index, *week, days=1, seed=seed)[0]).weekday() != 6
    )
    sampled = pipeline.verify(source, *week, horizon=1, days=1, seed=weekdays, rebuild=leaking_rebuild)
    assert sampled.status("m") == "inconclusive"
    assert "['signal_d-1'] only on issue days outside the sample" in sampled.problems()[0]

    every_day = pipeline.verify(source, *week, horizon=1, days=7, rebuild=leaking_rebuild)
    assert every_day.status("m") == "leak"


def test_verify_counts_a_column_a_function_only_reads_as_an_input():
    """A column that only chooses the list is an input like the others: a rule is needed for it to be probed, and a
    target read that way mustn't trip the training boundary."""
    frame = make_market_data(noise=5.0, seed=7)
    info = InformationSet("11:45", "Europe/Warsaw", rules=RULES)

    def reads_todays_price(ctx):
        return PREDICTORS if ctx.value("price") < 1e5 else PREDICTORS + ["price_d-1"]

    report = (
        ModelPipeline(information_set=info)
        .add_model(OLSModel(reads_todays_price, training_window=45, name="reader"))
        .verify(frame, *TEST, horizon=1, days=1)
    )
    assert report.status("reader") == "ok"


# ---------------------------------------------------------------------------
# Binary-column masks
# ---------------------------------------------------------------------------


def test_a_later_value_does_not_change_an_earlier_forecast_through_the_scaler_mask(df):
    """The mask of a list comes from its first test day, so a column turning non-binary later can't reach back."""
    frame = turns_non_binary(df, "flag", "2023-04-10")
    rules = InformationSet("11:45", "Europe/Warsaw", rules={**RULES, "flag": Availability.static()})
    make = lambda: OLSModel(not_always_binary_list, training_window=14)  # noqa: E731
    short = make().run(frame, "2023-04-05", "2023-04-08", horizon=1, information_set=rules)
    long = make().run(frame, "2023-04-05", "2023-04-14", horizon=1, information_set=rules)
    assert predictions(short).items() <= predictions(long).items()


# ---------------------------------------------------------------------------
# Resuming
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kept",
    [lambda line: zlib.crc32(line.encode()) % 3 == 0, lambda line: '"run_date": "2023-04-05"' not in line],
    ids=["a third of the lines", "all but the first test day"],
)
def test_resume_with_another_process_count_is_bitwise_a_fresh_run(df, tmp_path, monkeypatch, kept):
    monkeypatch.undo()  # the real pool, as the number of processes is the point
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    frame = df.iloc[:2800]
    fresh_path, path = tmp_path / "fresh.jsonl", tmp_path / "resumed.jsonl"
    make = lambda: OLSModel(by_horizon_period_and_publication, training_window=45)  # noqa: E731

    monkeypatch.setenv("MAX_PROCESSES", "1")
    make().run(frame, *TEST, horizon=2, information_set=INFO, save_to=str(fresh_path))
    make().run(frame, *TEST, horizon=2, information_set=INFO, save_to=str(path))
    assert stored_lines(path) == stored_lines(fresh_path)

    lines_before = [line for line in path.read_text().splitlines() if kept(line)]
    assert 0 < len(lines_before) < 480
    path.write_text("\n".join(lines_before) + "\n")  # an interrupted run

    monkeypatch.setenv("MAX_PROCESSES", "3")
    ref = make().run(frame, *TEST, horizon=2, information_set=INFO, save_to=str(path))
    assert ref.produced == 480
    assert stored_lines(path) == stored_lines(fresh_path)
    assert (
        json.loads((tmp_path / "resumed.manifest.json").read_text())["predictor_lists"]
        == json.loads((tmp_path / "fresh.manifest.json").read_text())["predictor_lists"]
    )


def test_the_manifest_records_the_function_and_every_distinct_list(df, tmp_path):
    OLSModel(by_horizon_period_and_publication, training_window=45).run(
        df, *TEST, horizon=1, information_set=INFO, save_to=str(tmp_path / "m.jsonl")
    )
    manifest = json.loads((tmp_path / "m.manifest.json").read_text())
    assert manifest["predictor_function"]["path"].endswith(":by_horizon_period_and_publication")
    assert re.fullmatch(r"\w{12}", manifest["predictor_function"]["fingerprint"])
    assert len(manifest["predictor_lists"]) == 4
    stored = {json.loads(line)["predictor_list"] for line in (tmp_path / "m.jsonl").read_text().splitlines()}
    assert stored == {entry["hash"] for entry in manifest["predictor_lists"]}
    for entry in manifest["predictor_lists"]:
        assert entry["hash"] == list_hash(entry["predictors"], entry["train_on"])
        assert entry["binary_masks"]
    assert "price_d-1" in manifest["predictors"] and "early" in manifest["predictors"]


def other_function(ctx):
    return PREDICTORS


def test_a_changed_function_that_changes_a_stored_list_raises(df, tmp_path):
    path = str(tmp_path / "r.jsonl")
    OLSModel(by_horizon_period_and_publication, training_window=45).run(
        df, *TEST, horizon=1, information_set=INFO, save_to=path
    )
    with pytest.raises(ManifestMismatchError, match="another predictor list"):
        OLSModel(other_function, training_window=45).run(df, *TEST, horizon=1, information_set=INFO, save_to=path)


EVERY_NTH_DAY = 2


def reads_a_global(ctx):
    return PREDICTORS + (["price_d-1"] if ctx.issue_day.day % EVERY_NTH_DAY == 0 else [])


def test_resume_after_a_change_the_source_hash_cannot_see_raises(df, tmp_path, monkeypatch):
    path = str(tmp_path / "g.jsonl")
    OLSModel(reads_a_global, training_window=45).run(df, *TEST, horizon=1, information_set=INFO, save_to=path)
    # Same source and the same names over the window: only which day gets which list changes.
    monkeypatch.setitem(reads_a_global.__globals__, "EVERY_NTH_DAY", 3)
    with pytest.raises(ManifestMismatchError, match="another predictor list.*2023-04-"):
        OLSModel(reads_a_global, training_window=45).run(df, *TEST, horizon=1, information_set=INFO, save_to=path)


def test_a_published_answer_changed_by_a_new_rule_raises(df, tmp_path):
    """The information set is part of what a list depends on, though a fixed list's run doesn't compare it."""

    def when_early(ctx):
        return PREDICTORS + (["price_d-1"] if ctx.published("early") else [])

    path = str(tmp_path / "rule.jsonl")
    module_level = when_early  # local functions run; only saving to YAML needs an import path
    OLSModel(module_level, training_window=45).run(df, *TEST, horizon=1, information_set=INFO, save_to=path)
    stricter = InformationSet("09:00", "Europe/Warsaw", rules=RULES)
    with pytest.raises(ManifestMismatchError, match="another predictor list"):
        OLSModel(module_level, training_window=45).run(df, *TEST, horizon=1, information_set=stricter, save_to=path)


def test_a_change_that_leaves_every_stored_list_alone_resumes_and_is_recorded(df, tmp_path):
    def first_version(ctx):
        return PREDICTORS

    def second_version(ctx):
        # Same answer, other source.
        return list(PREDICTORS)

    path = tmp_path / "s.jsonl"
    OLSModel(first_version, training_window=45).run(df, *TEST, horizon=1, information_set=INFO, save_to=str(path))
    fresh = stored_lines(path)
    kept = path.read_text().splitlines()[:50]
    path.write_text("\n".join(kept) + "\n")
    OLSModel(second_version, training_window=45).run(df, *TEST, horizon=1, information_set=INFO, save_to=str(path))
    assert stored_lines(path) == fresh
    manifest = json.loads((tmp_path / "s.manifest.json").read_text())
    assert manifest["predictor_function"]["path"].endswith("second_version")


def test_allow_manifest_mismatch_resumes_over_a_changed_list(df, tmp_path):
    path = str(tmp_path / "a.jsonl")
    OLSModel(reads_a_global, training_window=45).run(
        df, "2023-04-05", "2023-04-06", horizon=1, information_set=INFO, save_to=path
    )
    OLSModel(other_function, training_window=45).run(
        df,
        "2023-04-05",
        "2023-04-06",
        horizon=1,
        information_set=INFO,
        save_to=path,
        allow_manifest_mismatch=True,
    )
    assert len(Path(path).read_text().splitlines()) == 48


FIRST_USE = "2023-04-12"


def uses_flag_from(ctx):
    return NOT_ALWAYS_BINARY if str(ctx.issue_day) >= FIRST_USE else PREDICTORS


def test_a_stored_forecast_whose_binary_columns_a_fresh_run_would_change_raises(df, tmp_path, monkeypatch):
    """The function changes on pending days only, so every stored list stays. But the list's first test day moves to a
    window where `flag` is still binary, which would change how the stored forecasts of that list were scaled."""
    frame = turns_non_binary(df, "flag", "2023-04-08")
    rules = InformationSet("11:45", "Europe/Warsaw", rules={**RULES, "flag": Availability.static()})
    path = tmp_path / "flag.jsonl"
    make = lambda: OLSModel(uses_flag_from, training_window=30)  # noqa: E731
    make().run(frame, "2023-04-08", "2023-04-13", horizon=1, information_set=rules, save_to=str(path))
    records = [line for line in path.read_text().splitlines() if json.loads(line)["run_date"] >= FIRST_USE]
    path.write_text("\n".join(records) + "\n")

    # On 2023-04-09 the newest training row is still before the column's first non-binary value.
    monkeypatch.setitem(uses_flag_from.__globals__, "FIRST_USE", "2023-04-09")
    with pytest.raises(ManifestMismatchError, match="other binary columns"):
        make().run(frame, "2023-04-08", "2023-04-13", horizon=1, information_set=rules, save_to=str(path))


def by_part_of_the_day(ctx):
    if ctx.period < 12:
        return NOT_ALWAYS_BINARY if str(ctx.issue_day) < "2023-04-08" else PREDICTORS
    return NOT_ALWAYS_BINARY if str(ctx.issue_day) >= "2023-04-08" else PREDICTORS


def test_a_narrower_run_keeps_the_masks_a_later_resume_compares(df, tmp_path):
    """The middle run doesn't use the morning's list at all. If it forgot the masks the morning's stored forecasts used,
    the third run, whose first test day changes them, would resume over them."""
    frame = df.assign(flag=(np.random.default_rng(0).random(len(df)) > 0.5).astype(float))
    turned = frame.index.tz_localize(None) >= pd.Timestamp("2023-04-04")
    frame.loc[turned, "flag"] = frame.loc[turned, "load"]
    rules = InformationSet("11:45", "Europe/Warsaw", rules={**RULES, "flag": Availability.static()})
    path = tmp_path / "chain.jsonl"

    def run(start):
        OLSModel(by_part_of_the_day, training_window=20).run(
            frame, start, "2023-04-10", horizon=1, information_set=rules, save_to=str(path)
        )

    run("2023-04-06")
    run("2023-04-08")
    with pytest.raises(ManifestMismatchError, match="other binary columns"):
        run("2023-04-05")


def test_results_of_fixed_lists_and_of_a_function_do_not_mix(df, tmp_path):
    path = str(tmp_path / "mix.jsonl")
    OLSModel(PREDICTORS, training_window=45).run(df, *TEST, horizon=1, save_to=path)
    with pytest.raises(ManifestMismatchError, match="predictor"):
        OLSModel(static_list, training_window=45).run(df, *TEST, horizon=1, save_to=path)

    other = str(tmp_path / "mix2.jsonl")
    OLSModel(static_list, training_window=45).run(df, *TEST, horizon=1, save_to=other)
    with pytest.raises(ManifestMismatchError, match="predictor_layout_hash"):
        OLSModel(PREDICTORS, training_window=45).run(df, *TEST, horizon=1, save_to=other)


def test_records_without_a_list_hash_are_refused_when_there_is_no_manifest_to_tell(df, tmp_path):
    path = tmp_path / "old.jsonl"
    OLSModel(PREDICTORS, training_window=45).run(df, *TEST, horizon=1, save_to=str(path))
    (tmp_path / "old.manifest.json").unlink()
    with pytest.raises(ManifestMismatchError, match="another predictor list"):
        OLSModel(static_list, training_window=45).run(df, *TEST, horizon=1, save_to=str(path))


def test_stored_records_outside_the_test_period_are_not_compared_and_their_lists_stay_in_the_manifest(df, tmp_path):
    path = tmp_path / "w.jsonl"
    OLSModel(reads_a_global, training_window=45).run(
        df, "2023-04-05", "2023-04-08", horizon=1, information_set=INFO, save_to=str(path)
    )
    before = json.loads((tmp_path / "w.manifest.json").read_text())["predictor_lists"]
    # 2023-04-08 only uses one of the two lists, and has the same last row hashed, so the data still compares equal.
    ref = OLSModel(reads_a_global, training_window=45).run(
        df, "2023-04-08", "2023-04-08", horizon=1, information_set=INFO, save_to=str(path)
    )
    assert len(ref.predictor_lists) == 1
    after = json.loads((tmp_path / "w.manifest.json").read_text())["predictor_lists"]
    assert {e["hash"] for e in after} == {e["hash"] for e in before}


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------

STUDY = """
from tests.synthetic import PREDICTORS


def by_period(ctx):
    return PREDICTORS + (["price_d-1"] if 8 <= ctx.period <= 18 else [])


class Holder:
    def method(self, ctx):
        return PREDICTORS

    def nested(self):
        def inner(ctx):
            return PREDICTORS

        return inner


def wrapped(ctx):
    return PREDICTORS


_original = wrapped


def wrapped(ctx):  # noqa: F811
    return _original(ctx)
"""


@pytest.fixture
def study(tmp_path, monkeypatch):
    (tmp_path / "my_study_predictors.py").write_text(STUDY)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield __import__("my_study_predictors")
    sys.modules.pop("my_study_predictors", None)


def test_a_module_level_function_round_trips_by_import_path_with_identical_forecasts(df, tmp_path, study):
    path = tmp_path / "mp.yaml"
    pipeline = ModelPipeline(information_set=INFO).add_model(OLSModel(study.by_period, training_window=45, name="dyn"))
    pipeline.save(path)

    saved = yaml.safe_load(path.read_text())["models"][0]["params"]
    assert saved["predictors"] == {"function": "my_study_predictors:by_period"}

    loaded = ModelPipeline.load(path)
    assert loaded.models[0]._ctx_fn is study.by_period
    assert loaded.to_dict() == pipeline.to_dict()
    first = pipeline.run(df, *TEST, horizon=1).refs["dyn"]
    second = loaded.run(df, *TEST, horizon=1).refs["dyn"]
    assert lines(first._results) == lines(second._results)


@pytest.mark.parametrize(
    ("make_function", "reason"),
    [
        (lambda study: (lambda ctx: PREDICTORS), "it is a lambda"),
        (lambda study: study.Holder().nested(), "it is a nested function"),
        (lambda study: study.Holder().method, "it is a bound method"),
        (lambda study: study._original, "gives another object"),
    ],
    ids=["lambda", "nested", "bound method", "shadowed"],
)
def test_a_function_that_cannot_be_imported_back_raises_on_save(tmp_path, study, make_function, reason):
    path = tmp_path / "mp.yaml"
    path.write_text("kept")
    pipeline = ModelPipeline().add_model(OLSModel(make_function(study), name="x"))
    with pytest.raises(TypeError, match=rf"Cannot save OLSModel\('x'\).*{reason}"):
        pipeline.save(path)
    assert path.read_text() == "kept"  # the failure leaves the file alone


def test_a_function_of_the_running_script_raises_on_save(tmp_path, monkeypatch):
    def script_function(ctx):
        return PREDICTORS

    monkeypatch.setattr(script_function, "__module__", "__main__")
    monkeypatch.setattr(script_function, "__qualname__", "script_function")
    with pytest.raises(TypeError, match="defined in the script being run"):
        ModelPipeline().add_model(OLSModel(script_function, name="x")).save(tmp_path / "mp.yaml")


def test_horizon_callables_still_cannot_be_saved(tmp_path):
    with pytest.raises(TypeError, match="callable predictors can't be saved to YAML yet"):
        ModelPipeline().add_model(OLSModel(lambda horizon: PREDICTORS, name="x")).save(tmp_path / "mp.yaml")


@pytest.mark.parametrize(
    ("predictors", "error", "message"),
    [
        ("{function: no.such.module:fn}", ValueError, "Cannot import predictor function"),
        ("{function: not_a_path}", ValueError, "package.module:qualname"),
        ("{other: thing}", ValueError, "predictors must be a list"),
        ("{function: 'json:dumps'}", TypeError, "one parameter named `ctx`"),
    ],
)
def test_a_saved_function_that_cannot_be_loaded_raises(tmp_path, predictors, error, message):
    path = tmp_path / "mp.yaml"
    path.write_text(f"models:\n- class: OLSModel\n  params:\n    predictors: {predictors}\n    name: x\n")
    with pytest.raises(error, match=message):
        ModelPipeline.load(path)


# ---------------------------------------------------------------------------
# The examples on the docs page
# ---------------------------------------------------------------------------


def docs_examples():
    """What the docs page's python blocks define, executed the way a reader would."""
    source = DOCS_PAGE.read_text()
    namespace = {}
    for block in re.findall(r"```python\n(.*?)```", source, flags=re.DOTALL):
        if "def lear_" in block or "lear_x =" in block or "no_tso =" in block:
            exec(block, namespace)
    return namespace


@pytest.fixture(scope="module")
def lear_frame():
    """The hourly columns of the LEAR list and of the docs' examples. Only names and missing values matter here."""
    examples = docs_examples()
    frame = make_market_data(days=120, noise=5.0, seed=7)[["price"]].copy()
    rng = np.random.default_rng(0)
    hours = [f"h{h:02d}" for h in range(24)]
    names = set(examples["lear_x"]) | set(examples["no_tso"])
    names = {n.replace("{horizon}", str(h)) for n in names for h in (1, 2)}
    names |= {f"solar_fc_{hour}_d+{h}" for hour in hours for h in (1, 2)}
    names |= {f"exaa_price_{hour}_d+{h}" for hour in hours for h in (1, 2)}
    names |= {"load_actual", "load_forecast"}
    columns = {name: rng.normal(size=len(frame)) for name in sorted(names)}
    frame = pd.concat([frame, pd.DataFrame(columns, index=frame.index)], axis=1)
    for name in frame.columns:
        if name.startswith("is_"):
            frame[name] = (frame[name] > 0).astype(float)
    # EXAA is out on most days, not on 2023-04-07.
    exaa = [c for c in frame.columns if c.startswith("exaa_price")]
    frame.loc[frame.index.tz_localize(None).normalize() == pd.Timestamp("2023-04-07"), exaa] = np.nan
    return frame


def test_the_docs_lear_dynamic_gives_the_lists_the_page_says(lear_frame):
    examples = docs_examples()
    info = examples["info"]
    model = OLSModel(examples["lear_dynamic"], training_window=56, name="LEAR_dyn")
    plan = model._plan_predictors(lear_frame, "price", 2, "1h", info, "2023-04-05", "2023-04-08")
    model._check_inputs(lear_frame, "price", 2, "1h", info, "2023-04-05", "2023-04-08", plan=plan)
    assert plan.calls == 4 * 2 * 24
    assert len(plan.lists) == 6  # daylight or not x EXAA or not for horizon 1, and EXAA is never out for horizon 2
    described = {tuple(entry["horizons"]) for entry in plan.describe()}
    assert described == {(1,), (2,)}
    noon = plan.lists[plan.list_id_of(12, 1, "2023-04-05")]
    assert "solar_fc_h12_d+1" in noon and "exaa_price_h12_d+1" in noon and "load_forecast_h00_d+1" in noon
    missing = plan.lists[plan.list_id_of(12, 1, "2023-04-07")]
    assert "solar_fc_h12_d+1" in missing and "exaa_price_h12_d+1" not in missing
    second = plan.lists[plan.list_id_of(0, 2, "2023-04-05")]
    assert "load_forecast_h00_d+2" not in second and "solar_fc_h12_d+2" not in second


def test_the_docs_lear_asof_switches_when_the_actual_value_is_late(lear_frame):
    examples = docs_examples()
    info = examples["info"]
    model = OLSModel(examples["lear_asof"], training_window=56, name="asof")
    plan = model._plan_predictors(lear_frame, "price", 1, "1h", info, "2023-04-05", "2023-04-08")
    model._check_inputs(lear_frame, "price", 1, "1h", info, "2023-04-05", "2023-04-08", plan=plan)
    assert len(plan.lists) == 2
    assert "load_actual" in plan.lists[plan.list_id_of(9, 1, "2023-04-05")]
    assert "load_forecast" in plan.lists[plan.list_id_of(10, 1, "2023-04-05")]


def test_load_actual_at_ten_is_late_at_a_quarter_to_twelve(lear_frame):
    """Why the docs switch at 10 and not at the issue time's hour: 10:00-11:00 is known 12:00 by the built-in rule."""
    examples = docs_examples()
    info = examples["info"]

    def until_the_issue_hour(ctx):
        return examples["lear_x"] + ["load_actual" if ctx.period < 11 else "load_forecast"]

    model = OLSModel(until_the_issue_hour, training_window=56, name="naive")
    with pytest.raises(InformationSetError, match=r"load_actual.*Periods 10, horizons 1"):
        model.run(lear_frame, "2023-04-05", "2023-04-08", horizon=1, information_set=info)


def test_the_signature_of_use_and_the_public_names():
    assert list(inspect.signature(Use).parameters) == ["predictors", "train_on"]
    assert inspect.signature(Use).parameters["train_on"].default == "all"
    assert presets.lear_predictors(["load_forecast"])  # what the docs' examples build on
