"""Issue-time checks at three levels: "none" checks nothing, "warn" lists and goes on, "raise" refuses before any fit.

The level decides what happens around an input known after the issue time. It never changes a number, so a run can resume
at another level, and what `warn` listed follows the results: into the manifest, a report built from it, and the terminal.
"""

import json
import warnings
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from epftoolbox2 import cli
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.exporters import TerminalExporter
from epftoolbox2.information_set import (
    AVAILABILITY_LEVELS,
    Availability,
    InformationSet,
    InformationSetError,
    leakage_probe,
)
from epftoolbox2.models import OLSModel
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.models.context import list_hash
from epftoolbox2.pipelines import ModelPipeline, Workflow
from epftoolbox2.results.availability import COLUMNS
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path
from tests.synthetic import PREDICTORS, calendar_shift, make_market_data

RULES = {
    # The synthetic drivers stand for forecasts published well before their delivery.
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "price": Availability.published("13:00", 1),
    "early": Availability.published("10:15", 0),  # row D holds the value for D+1, published 10:15 on D
    "observed": Availability.realised("1h"),
}
INFO = InformationSet("11:45", "Europe/Warsaw", rules=RULES)
TEST = ("2023-04-05", "2023-04-07")
EXPANDED = [p.replace("{horizon}", "1") for p in PREDICTORS]
REALISED = "known 1h after the end of the period"
EARLY_RULE = "published 10:15 Europe/Brussels on the delivery day"


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


@pytest.fixture(scope="module")
def df():
    data = make_market_data(noise=5.0, seed=7)
    data["early"] = data["load_d+1"] + np.random.default_rng(1).normal(0, 20, len(data))
    data["early_d+1"] = calendar_shift(data["early"], 1)
    data["early_d+2"] = calendar_shift(data["early"], 2)
    data["observed"] = data["load"] + np.random.default_rng(2).normal(0, 20, len(data))
    data["mystery"] = np.random.default_rng(3).normal(0, 1, len(data))
    return data


def ols(predictors=None, name="m"):
    return OLSModel(PREDICTORS if predictors is None else predictors, training_window=45, name=name)


def pipeline(*models, level=None, info=INFO):
    out = ModelPipeline(information_set=info, availability=level)
    for model in models:
        out.add_model(model)
    return out


def rows(report, *columns):
    table = report.availability()
    return table[list(columns) if columns else COLUMNS].values.tolist()


def lines(path):
    return sorted(line for line in Path(path).read_text().splitlines() if line)


def records(report):
    return {
        name: sorted(json.dumps(r, sort_keys=True) for r in report._iter_ref(ref)) for name, ref in report.refs.items()
    }


def availability_warnings(captured):
    return [str(w.message) for w in captured if str(w.message).startswith("Availability:")]


def no_warnings_of_ours(captured):
    return [str(w.message) for w in captured if "availability" in str(w.message).lower()]


# ---------------------------------------------------------------------------
# Which level a pipeline runs at
# ---------------------------------------------------------------------------


def test_no_issue_time_means_no_checks():
    assert ModelPipeline().availability == "none"
    assert ModelPipeline(availability="none").availability == "none"
    assert ModelPipeline(issue_time=None).availability == "none"


def test_an_issue_time_or_an_information_set_defaults_to_warn():
    assert ModelPipeline(issue_time="11:45", issue_timezone="Europe/Warsaw").availability == "warn"
    assert ModelPipeline(information_set=InformationSet()).availability == "warn"


@pytest.mark.parametrize("level", AVAILABILITY_LEVELS)
def test_each_level_can_be_asked_for_with_either_way_of_giving_the_issue_time(level):
    assert ModelPipeline(issue_time="11:45", availability=level).availability == level
    assert ModelPipeline(information_set=InformationSet(), availability=level).availability == level


@pytest.mark.parametrize("level", ["warn", "raise"])
def test_warn_and_raise_need_an_issue_time(level):
    with pytest.raises(ValueError, match="issue_time or information_set"):
        ModelPipeline(availability=level)


@pytest.mark.parametrize("level", ["ignore", "WARN", "", 1])
def test_an_unknown_level_raises(level):
    with pytest.raises(ValueError, match="availability must be one of"):
        ModelPipeline(issue_time="11:45", availability=level)


def test_a_level_set_without_an_issue_time_is_refused_also_when_the_issue_time_went_away(df):
    a_pipeline = pipeline(ols(), level="raise")
    a_pipeline.information_set = None
    with pytest.raises(ValueError, match="issue_time or information_set"):
        a_pipeline.run(df, *TEST, horizon=1)


# ---------------------------------------------------------------------------
# What warn lists
# ---------------------------------------------------------------------------


def test_a_fixed_list_is_listed_with_its_periods_and_horizons(df):
    report = pipeline(ols(PREDICTORS + ["observed", "price_d-1", "mystery"])).run(df, *TEST, horizon=2)
    assert rows(report) == [
        ["m", "mystery", "predictor", "no rule", "", "0-23", "1-2", "", "", "", ""],
        ["m", "observed", "predictor", "late", "", "10-23", "1-2", REALISED, "D+1 01:00", "D 11:45", ""],
    ]
    assert report.forecast_counts().loc[0, "missing"] == 0


def test_a_horizon_callable_is_listed_per_name(df):
    def by_horizon(horizon):
        return [*(p.replace("{horizon}", str(horizon)) for p in PREDICTORS), f"early_d+{horizon}"]

    report = pipeline(ols(by_horizon)).run(df, *TEST, horizon=2)
    assert rows(report, "predictor", "status", "periods", "horizons", "known", "issued") == [
        ["early_d+1", "late", "0-23", "1", "D+1 10:15", "D 11:45"],
        ["early_d+2", "late", "0-23", "2", "D+2 10:15", "D 11:45"],
    ]


def test_a_horizon_and_hour_callable_is_listed_with_exactly_its_uses(df):
    """`observed` is late from period 10. Horizon 1 uses it before noon and horizon 2 after, so neither row may claim
    periods 10-23 at horizons 1-2."""

    def by_horizon_and_hour(horizon, hour):
        names = [p.replace("{horizon}", str(horizon)) for p in PREDICTORS]
        return names + (["observed"] if (hour < 12) == (horizon == 1) else [])

    report = pipeline(ols(by_horizon_and_hour)).run(df, *TEST, horizon=2)
    assert rows(report, "predictor", "periods", "horizons", "known") == [
        ["observed", "10-11", "1", "D 13:00"],
        ["observed", "12-23", "2", "D+1 01:00"],
    ]


def late_at_noon(ctx):
    return PREDICTORS + (["early_d+1"] if ctx.period == 12 else [])


def observed_by_period(ctx):
    return PREDICTORS + (["observed"] if ctx.period < 12 else ["observed", "price_d-1"])


def test_a_predictor_function_is_listed_per_list(df):
    report = pipeline(ols(observed_by_period)).run(df, *TEST, horizon=1)
    morning = list_hash([*EXPANDED, "observed"])
    afternoon = list_hash([*EXPANDED, "observed", "price_d-1"])
    assert sorted(rows(report, "predictor", "status", "list", "periods", "horizons", "known")) == sorted(
        [
            ["observed", "late", morning, "10-11", "1", "D 13:00"],
            ["observed", "late", afternoon, "12-23", "1", "D+1 01:00"],
        ]
    )


def test_a_list_is_listed_only_for_the_forecasts_that_use_it(df):
    report = pipeline(ols(late_at_noon)).run(df, *TEST, horizon=1)
    assert rows(report, "predictor", "list", "periods", "known") == [
        ["early_d+1", list_hash([*EXPANDED, "early_d+1"]), "12", "D+1 10:15"]
    ]


def test_every_models_late_predictors_are_reported_not_just_the_first_models(df):
    """The prototype's `check()` table, without a `check()` step: a run reports every model, and doesn't stop at one."""
    report = pipeline(
        ols(PREDICTORS, name="static"),
        ols(late_at_noon, name="dyn"),
        ols(PREDICTORS + ["early_d+1"], name="late"),
        ols(PREDICTORS + ["mystery"], name="unruled"),
    ).run(df, *TEST, horizon=1)
    assert rows(report, "model", "predictor", "status", "periods") == [
        ["dyn", "early_d+1", "late", "12"],
        ["late", "early_d+1", "late", "0-23"],
        ["unruled", "mystery", "no rule", "0-23"],
    ]
    assert report.forecast_counts()["missing"].tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize("unknown", ["raise", "warn", "ignore"])
def test_a_column_without_a_rule_is_listed_whatever_unknown_says_and_is_no_error(df, unknown, recwarn):
    info = InformationSet("11:45", "Europe/Warsaw", rules=RULES, unknown=unknown)
    report = pipeline(ols(PREDICTORS + ["mystery"]), info=info).run(df, *TEST, horizon=1)
    assert rows(report, "predictor", "status", "periods", "horizons") == [["mystery", "no rule", "0-23", "1"]]
    assert not [w for w in recwarn if "no availability rule" in str(w.message)]  # `unknown`'s own message is for raise


@pytest.mark.parametrize("unknown", ["raise", "warn", "ignore"])
def test_in_raise_mode_unknown_behaves_as_it_always_did(df, unknown):
    info = InformationSet("11:45", "Europe/Warsaw", rules=RULES, unknown=unknown)
    a_pipeline = pipeline(ols(PREDICTORS + ["mystery"]), level="raise", info=info)
    if unknown == "raise":
        with pytest.raises(InformationSetError, match="no availability rule"):
            a_pipeline.run(df, *TEST, horizon=1)
    elif unknown == "warn":
        with pytest.warns(UserWarning, match="no availability rule"):
            report = a_pipeline.run(df, *TEST, horizon=1)
        assert report.availability().empty
    else:
        assert a_pipeline.run(df, *TEST, horizon=1).availability().empty


def test_a_late_training_target_is_listed_under_its_role(df):
    """The boundary `check` also holds: the newest training target is D-1's, here known only on D+1."""
    info = InformationSet("11:45", "Europe/Warsaw", rules={**RULES, "slow": Availability.published("12:00", -2)})
    frame = df.assign(slow=df["price"])
    report = pipeline(ols(), info=info).run(frame, *TEST, target="slow", horizon=1)
    assert rows(report, "predictor", "role", "status", "periods", "horizons", "known") == [
        ["slow", "training target", "late", "0-23", "1", "D+1 12:00"]
    ]
    with pytest.raises(InformationSetError, match="training target slow"):
        pipeline(ols(), info=info, level="raise").run(frame, *TEST, target="slow", horizon=1)


def test_a_late_training_target_of_a_predictor_function_is_listed_too(df):
    info = InformationSet("11:45", "Europe/Warsaw", rules={**RULES, "slow": Availability.published("12:00", -2)})
    frame = df.assign(slow=df["price"], mystery_target=df["price"])

    def constant(ctx):
        return PREDICTORS

    report = pipeline(ols(constant), info=info).run(frame, *TEST, target="slow", horizon=2)
    assert rows(report, "predictor", "role", "status", "periods", "horizons") == [
        ["slow", "training target", "late", "0-23", "1-2"]
    ]
    report = pipeline(ols(constant), info=info).run(frame, *TEST, target="mystery_target", horizon=2)
    assert rows(report, "predictor", "role", "status", "periods", "horizons") == [
        ["mystery_target", "training target", "no rule", "0-23", "1-2"]
    ]


def test_a_hint_that_would_put_it_in_time_is_kept(df):
    frame = df.assign(generation_solar_forecast=1.0)
    frame["generation_solar_forecast_d+1"] = 1.0
    frame["generation_solar_forecast_d+1"] += np.random.default_rng(0).normal(0, 1, len(frame))
    info = InformationSet("11:45", "Europe/Warsaw", rules=RULES)
    report = pipeline(ols(PREDICTORS + ["generation_solar_forecast_d+1"]), info=info).run(frame, *TEST, horizon=1)
    assert rows(report, "predictor", "status", "note")[0][:2] == ["generation_solar_forecast_d+1", "late"]
    assert "entsoe_a69_by_10" in rows(report, "note")[0][0]


# ---------------------------------------------------------------------------
# Where the listing shows
# ---------------------------------------------------------------------------


def test_nothing_to_list_gives_an_empty_table_with_the_same_columns_and_no_warning(df, recwarn):
    for level in ("warn", "raise", "none"):
        report = pipeline(ols(), level=level).run(df, *TEST, horizon=1)
        assert report.availability().empty
        assert list(report.availability().columns) == COLUMNS
    assert availability_warnings(recwarn) == []
    assert ModelPipeline().add_model(ols()).run(df, *TEST, horizon=1).availability().empty


def test_a_report_from_plain_lists_has_an_empty_table(df):
    from epftoolbox2.results.report import EvaluationReport

    assert list(EvaluationReport({"a": []}, []).availability().columns) == COLUMNS


def test_one_warning_covers_every_model_and_says_what_to_do(df, recwarn):
    pipeline(ols(PREDICTORS + ["observed"], name="a"), ols(PREDICTORS + ["mystery"], name="b")).run(
        df, *TEST, horizon=1
    )
    messages = availability_warnings(recwarn)
    assert len(messages) == 1
    assert "2 input(s) of 2 model(s)" in messages[0]
    assert "a: observed is late (periods 10-23, horizons 1; known D+1 01:00, issued D 11:45)" in messages[0]
    assert "b: mystery has no rule (periods 0-23, horizons 1)" in messages[0]
    assert 'availability="raise"' in messages[0]


def test_the_warning_comes_before_any_fit(df, monkeypatch):
    fit = OLSModel._fit_predict
    warned_by_then = []

    def noting_fit(self, *args):
        warned_by_then.append(len(availability_warnings(captured)))
        return fit(self, *args)

    monkeypatch.setattr(OLSModel, "_fit_predict", noting_fit)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        pipeline(ols(PREDICTORS + ["observed"])).run(df, *TEST, horizon=1)
    assert warned_by_then and min(warned_by_then) == 1


def test_a_model_run_directly_warns_for_itself(df, recwarn):
    ref = ols(PREDICTORS + ["observed"]).run(df, *TEST, horizon=1, information_set=INFO, availability="warn")
    assert len(availability_warnings(recwarn)) == 1
    assert ref.availability_listing[0]["predictor"] == "observed"


def test_a_model_run_directly_refuses_by_default(df):
    with pytest.raises(InformationSetError, match="observed"):
        ols(PREDICTORS + ["observed"]).run(df, *TEST, horizon=1, information_set=INFO)
    with pytest.raises(ValueError, match="availability must be one of"):
        ols().run(df, *TEST, horizon=1, availability="maybe")


def test_the_terminal_shows_it_first(df, capsys):
    a_pipeline = (
        pipeline(ols(PREDICTORS + ["observed"]))
        .add_evaluator(MAEEvaluator())
        .add_exporter(TerminalExporter(show=["summary"]))
    )
    a_pipeline.run(df, *TEST, horizon=1)
    out = capsys.readouterr().out
    assert out.index("Availability") < out.index("Summary")
    assert "observed" in out and "10-23" in out and "D+1" in out


def test_the_terminal_says_nothing_with_nothing_to_list(df, capsys):
    pipeline(ols()).add_evaluator(MAEEvaluator()).add_exporter(TerminalExporter()).run(df, *TEST, horizon=1)
    assert "Availability" not in capsys.readouterr().out


def test_the_manifest_holds_the_level_and_the_listing(df, tmp_path):
    report = pipeline(ols(PREDICTORS + ["observed"])).run(df, *TEST, horizon=1, save_dir=str(tmp_path))
    stored = json.loads(manifest_path(tmp_path / "m.jsonl").read_text())["availability"]
    assert stored["level"] == "warn"
    assert [(r["predictor"], r["status"], r["periods"]) for r in stored["listing"]] == [("observed", "late", "10-23")]
    assert report.availability()["predictor"].tolist() == ["observed"]


def test_a_merge_of_separate_runs_shows_what_they_listed_without_duplicates(df, tmp_path, recwarn, capsys):
    """Like a SLURM array: each model on its own, then one report, from a pipeline that knows no issue time at all."""
    pipeline(ols(PREDICTORS + ["observed"], name="a")).run(df, *TEST, horizon=1, save_dir=str(tmp_path))
    pipeline(ols(late_at_noon, name="b")).run(df, *TEST, horizon=1, save_dir=str(tmp_path))
    recwarn.clear()

    merging = (
        ModelPipeline()
        .add_model(ols(PREDICTORS + ["observed"], name="a"))
        .add_model(ols(late_at_noon, name="b"))
        .add_evaluator(MAEEvaluator())
        .add_exporter(TerminalExporter(show=["summary"]))
    )
    report = merging.report(tmp_path, test_start=TEST[0], test_end=TEST[1], horizon=1)
    assert rows(report, "model", "predictor", "periods") == [["a", "observed", "10-23"], ["b", "early_d+1", "12"]]
    assert not report.availability().duplicated().any()
    assert len(availability_warnings(recwarn)) == 1
    out = capsys.readouterr().out
    assert out.index("Availability") < out.index("Summary")


def test_results_that_predate_the_listing_merge_with_an_empty_table(df, tmp_path, recwarn):
    ModelPipeline().add_model(ols()).run(df, *TEST, horizon=1, save_dir=str(tmp_path))
    report = ModelPipeline().add_model(ols()).report(tmp_path, test_start=TEST[0], test_end=TEST[1], horizon=1)
    assert report.availability().empty
    assert availability_warnings(recwarn) == []


@pytest.mark.parametrize("predictors", [PREDICTORS + ["observed", "mystery"], late_at_noon], ids=["fixed", "function"])
def test_a_resumed_run_lists_what_a_fresh_one_does_without_duplicates(df, tmp_path, predictors, recwarn):
    fresh = pipeline(ols(predictors)).run(df, *TEST, horizon=2, save_dir=str(tmp_path / "fresh"))
    expected = fresh.availability()
    assert len(expected)

    # Done completely, then interrupted half-way.
    again = pipeline(ols(predictors)).run(df, *TEST, horizon=2, save_dir=str(tmp_path / "fresh"))
    pd.testing.assert_frame_equal(again.availability(), expected)

    partial = tmp_path / "partial"
    pipeline(ols(predictors)).run(df, *TEST, horizon=2, save_dir=str(partial))
    stored = (partial / "m.jsonl").read_text().splitlines()
    (partial / "m.jsonl").write_text("\n".join(stored[: len(stored) // 2]) + "\n")
    recwarn.clear()
    resumed = pipeline(ols(predictors)).run(df, *TEST, horizon=2, save_dir=str(partial))
    pd.testing.assert_frame_equal(resumed.availability(), expected)
    assert not resumed.availability().duplicated().any()
    assert len(availability_warnings(recwarn)) == 1
    assert lines(partial / "m.jsonl") == lines(tmp_path / "fresh" / "m.jsonl")

    stored_again = json.loads(manifest_path(partial / "m.jsonl").read_text())["availability"]["listing"]
    assert (
        stored_again == json.loads(manifest_path(tmp_path / "fresh" / "m.jsonl").read_text())["availability"]["listing"]
    )


def morning_then_afternoon(ctx):
    late = ctx.period < 12 if ctx.issue_day.day == 5 else ctx.period >= 12
    return PREDICTORS + (["early_d+1"] if late else [])


def test_a_narrower_run_keeps_what_was_listed_for_the_days_it_leaves_out(df, tmp_path):
    """Records outside a narrower run stay in the file, so a report over all of them must still show their rows."""
    model = lambda: ols(morning_then_afternoon)  # noqa: E731
    first = pipeline(model()).run(df, *TEST, horizon=1, save_dir=str(tmp_path))
    assert rows(first, "predictor", "periods") == [["early_d+1", "0-23"]]

    narrower = pipeline(model()).run(df, TEST[1], TEST[1], horizon=1, save_dir=str(tmp_path))
    assert rows(narrower, "predictor", "periods") == [["early_d+1", "0-23"]]
    merged = ModelPipeline().add_model(model()).report(tmp_path, test_start=TEST[0], test_end=TEST[1], horizon=1)
    assert rows(merged, "predictor", "periods") == [["early_d+1", "0-23"]]


def test_rules_that_changed_replace_what_was_listed(df, tmp_path):
    """Old rows would be a claim the current rules no longer make."""
    pipeline(ols(PREDICTORS + ["observed"])).run(df, *TEST, horizon=1, save_dir=str(tmp_path))
    relaxed = InformationSet("11:45", "Europe/Warsaw", rules={**RULES, "observed": Availability.static()})
    report = pipeline(ols(PREDICTORS + ["observed"]), info=relaxed).run(df, *TEST, horizon=1, save_dir=str(tmp_path))
    assert report.availability().empty
    assert json.loads(manifest_path(tmp_path / "m.jsonl").read_text())["availability"]["listing"] == []


def test_joined_and_aggregated_reports_keep_the_listing(df):
    a = pipeline(ols(PREDICTORS + ["observed"], name="a")).add_evaluator(MAEEvaluator()).run(df, *TEST, horizon=1)
    b = pipeline(ols(PREDICTORS + ["mystery"], name="b")).add_evaluator(MAEEvaluator()).run(df, *TEST, horizon=1)
    assert a.join(b).availability()["model"].tolist() == ["a", "b"]
    assert a.aggregate("1D", "max").availability()["predictor"].tolist() == ["observed"]


def test_forecast_only_lists_too(df):
    report = pipeline(ols(PREDICTORS + ["observed"])).run(df, TEST[0], TEST[0], horizon=1, forecast_only=True)
    assert rows(report, "predictor") == [["observed"]]


# ---------------------------------------------------------------------------
# ctx at each level
# ---------------------------------------------------------------------------

MISSING_ROW = ("2023-04-06", 3)  # `sparse` has no value at this issue day and period


@pytest.fixture(scope="module")
def sparse(df):
    frame = df.assign(sparse=df["early"])
    stamp = pd.Timestamp(f"{MISSING_ROW[0]} {MISSING_ROW[1]:02d}:00", tz="Europe/Warsaw")
    frame.loc[stamp, "sparse"] = np.nan
    return frame


SPARSE_RULES = {**RULES, "sparse": Availability.published("10:15", 0)}
SPARSE_INFO = InformationSet("11:45", "Europe/Warsaw", rules=SPARSE_RULES)


def answers_of_published(frame, level, column, info=SPARSE_INFO):
    seen = {}

    def asks(ctx):
        seen[(ctx.issue_day.isoformat(), ctx.period)] = ctx.published(column)
        return PREDICTORS

    pipeline(ols(asks), level=level, info=info).run(frame, *TEST, horizon=1)
    return seen


@pytest.mark.parametrize("column", ["sparse", "early_d+1", "observed"])
def test_published_answers_alike_at_every_level(sparse, column, recwarn):
    answers = {level: answers_of_published(sparse, level, column) for level in AVAILABILITY_LEVELS}
    assert answers["none"] == answers["warn"] == answers["raise"]
    if column == "sparse":
        assert not answers["warn"][MISSING_ROW] and sum(answers["warn"].values()) == 71
    if column == "early_d+1":  # known only on D+1, so never
        assert not any(answers["warn"].values())
    if column == "observed":  # 1 h after the period: out for 00:00-10:00, not from 10:00-11:00 on
        assert all(answers["warn"][(TEST[0], period)] for period in range(10))
        assert not any(answers["warn"][(TEST[0], period)] for period in range(10, 24))


def by_publication(ctx):
    """A function that passes `raise`: it only reads what is known and only uses what it may."""
    names = list(PREDICTORS)
    if ctx.published("sparse"):
        names.append("sparse")
    if ctx.period < 10 and ctx.value("observed") > 1000:
        names.append("observed")
    return names


@pytest.mark.parametrize("level", ["warn", "none"])
def test_a_model_that_passes_raise_gives_bitwise_identical_forecasts_at_the_other_levels(sparse, level, tmp_path):
    def run(chosen, where):
        return pipeline(ols(by_publication), level=chosen, info=SPARSE_INFO).run(
            sparse, *TEST, horizon=2, save_dir=str(tmp_path / where)
        )

    checked, other = run("raise", "raise"), run(level, level)
    assert checked.availability().empty and other.availability().empty
    assert lines(tmp_path / "raise" / "m.jsonl") == lines(tmp_path / level / "m.jsonl")
    assert records(checked) == records(other)


def reads_late(ctx):
    return PREDICTORS if ctx.value("observed") > 0 else PREDICTORS + ["price_d-1"]


def test_a_late_value_is_refused_by_raise_and_listed_by_warn(df):
    with pytest.raises(InformationSetError, match=r"ctx\.value\('observed'\)"):
        pipeline(ols(reads_late), level="raise").run(df, *TEST, horizon=1)

    report = pipeline(ols(reads_late), level="warn").run(df, *TEST, horizon=1)
    assert rows(report, "predictor", "role", "status", "list", "periods", "known", "issued") == [
        ["observed", "ctx.value", "late", "", "10-23", "D+1 01:00", "D 11:45"]
    ]


def test_a_late_value_is_returned_at_none_and_nothing_is_listed(df, recwarn):
    seen = []

    def reads(ctx):
        value = ctx.value("observed")
        seen.append(value)
        return PREDICTORS

    report = pipeline(ols(reads), level="none").run(df, *TEST, horizon=1)
    assert report.availability().empty
    assert availability_warnings(recwarn) == []
    assert np.isfinite(seen).all()


def test_warn_returns_the_value_the_frame_holds(df):
    seen = {}

    def reads(ctx):
        seen[(ctx.issue_day.isoformat(), ctx.period)] = ctx.value("observed")
        return PREDICTORS

    pipeline(ols(reads), level="warn").run(df, TEST[0], TEST[0], horizon=1)
    stamp = pd.Timestamp(f"{TEST[0]} 15:00", tz="Europe/Warsaw")
    assert seen[(TEST[0], 15)] == df.loc[stamp, "observed"]


def test_a_column_without_a_rule_read_through_ctx(sparse):
    """At raise an error whatever `unknown` says, as it always was. At warn published is just "not NaN", and it is listed."""
    frame = sparse.drop(columns="sparse").assign(mine=sparse["sparse"])

    def asks(ctx):
        return PREDICTORS + (["price_d-1"] if ctx.published("mine") else [])

    def reads(ctx):
        return PREDICTORS + (["price_d-1"] if ctx.value("mine") > 0 else [])

    for function in (asks, reads):
        with pytest.raises(InformationSetError, match="no availability rule"):
            pipeline(ols(function), level="raise").run(frame, *TEST, horizon=1)

    assert rows(
        pipeline(ols(asks), level="warn").run(frame, *TEST, horizon=1), "predictor", "role", "status", "periods"
    ) == [["mine", "ctx.published", "no rule", "0-23"]]
    assert rows(
        pipeline(ols(reads), level="warn").run(frame, *TEST, horizon=1), "predictor", "role", "status", "periods"
    ) == [["mine", "ctx.value", "no rule", "0-23"]]
    assert pipeline(ols(asks), level="none").run(frame, *TEST, horizon=1).availability().empty

    info = InformationSet("11:45", "Europe/Warsaw", rules=RULES)
    answers = answers_of_published(frame, "warn", "mine", info)
    assert not answers[MISSING_ROW] and sum(answers.values()) == 71  # True iff the value is not NaN
    assert answers == answers_of_published(frame, "none", "mine", info)


def test_a_column_without_a_rule_in_a_list_of_a_function_is_listed_per_list(df):
    def with_mystery(ctx):
        return PREDICTORS + (["mystery"] if ctx.period < 6 else [])

    report = pipeline(ols(with_mystery)).run(df, *TEST, horizon=2)
    assert rows(report, "predictor", "status", "list", "periods", "horizons") == [
        ["mystery", "no rule", list_hash([*EXPANDED, "mystery"]), "0-5", "1"],
        ["mystery", "no rule", list_hash([p.replace("{horizon}", "2") for p in PREDICTORS] + ["mystery"]), "0-5", "2"],
    ]


def test_without_an_information_set_ctx_only_looks_at_nan(sparse):
    seen = {}

    def asks(ctx):
        seen[(ctx.issue_day.isoformat(), ctx.period)] = (ctx.published("sparse"), ctx.value("observed"))
        return PREDICTORS

    report = ModelPipeline().add_model(ols(asks)).run(sparse, *TEST, horizon=1)
    assert report.availability().empty
    assert sum(answer for answer, _ in seen.values()) == 71
    assert not seen[MISSING_ROW][0]


def test_a_plan_made_at_another_level_is_refused(df):
    model = ols(by_publication)
    frame = df.assign(sparse=df["early"])
    info = SPARSE_INFO
    plan = model._plan_predictors(frame, "price", 1, "1h", info, *TEST, availability="warn")
    with pytest.raises(ValueError, match="availability level"):
        model.run(frame, *TEST, horizon=1, information_set=info, plan=plan, availability="raise")
    assert model.run(frame, *TEST, horizon=1, information_set=info, plan=plan, availability="warn").produced == 72


# ---------------------------------------------------------------------------
# The checks don't change numbers
# ---------------------------------------------------------------------------


def test_warn_and_none_give_bitwise_identical_records_and_results_files(df, tmp_path):
    predictors = PREDICTORS + ["observed", "early_d+1", "mystery"]
    warned = pipeline(ols(predictors), level="warn").run(df, *TEST, horizon=2, save_dir=str(tmp_path / "warn"))
    silent = pipeline(ols(predictors), level="none").run(df, *TEST, horizon=2, save_dir=str(tmp_path / "none"))
    plain = ModelPipeline().add_model(ols(predictors)).run(df, *TEST, horizon=2, save_dir=str(tmp_path / "plain"))
    assert lines(tmp_path / "warn" / "m.jsonl") == lines(tmp_path / "none" / "m.jsonl")
    assert lines(tmp_path / "warn" / "m.jsonl") == lines(tmp_path / "plain" / "m.jsonl")
    assert records(warned) == records(silent) == records(plain)


def test_warn_and_none_give_the_same_in_memory_records(df):
    predictors = PREDICTORS + ["observed", "mystery"]
    warned = pipeline(ols(predictors), level="warn").run(df, *TEST, horizon=1)
    plain = ModelPipeline().add_model(ols(predictors)).run(df, *TEST, horizon=1)
    assert records(warned) == records(plain)


@pytest.mark.parametrize("predictors", [PREDICTORS + ["observed"], late_at_noon], ids=["fixed", "function"])
@pytest.mark.parametrize("first, then", [("warn", "none"), ("none", "warn")])
def test_changing_only_the_level_resumes(df, tmp_path, predictors, first, then):
    pipeline(ols(predictors), level=first).run(df, *TEST, horizon=1, save_dir=str(tmp_path / "split"))
    path = tmp_path / "split" / "m.jsonl"
    stored = path.read_text().splitlines()
    path.write_text("\n".join(stored[: len(stored) // 3]) + "\n")
    report = pipeline(ols(predictors), level=then).run(df, *TEST, horizon=1, save_dir=str(tmp_path / "split"))
    assert report.forecast_counts().loc[0, "missing"] == 0

    pipeline(ols(predictors), level=then).run(df, *TEST, horizon=1, save_dir=str(tmp_path / "whole"))
    assert lines(path) == lines(tmp_path / "whole" / "m.jsonl")


def test_a_clean_model_resumes_through_every_level(df, tmp_path):
    for level in ("none", "warn", "raise", "none", "raise", "warn"):
        pipeline(ols(by_period), level=level).run(df, *TEST, horizon=1, save_dir=str(tmp_path))
    pipeline(ols(by_period)).run(df, *TEST, horizon=1, save_dir=str(tmp_path))


def by_period(ctx):
    return PREDICTORS + (["price_d-1"] if ctx.period < 12 else [])


def test_a_real_change_still_stops_a_resume_at_any_level(df, tmp_path):
    pipeline(ols(), level="warn").run(df, *TEST, horizon=1, save_dir=str(tmp_path))
    with pytest.raises(ManifestMismatchError, match="predictor_layout_hash"):
        pipeline(ols(PREDICTORS + ["price_d-1"]), level="warn").run(df, *TEST, horizon=1, save_dir=str(tmp_path))


def test_level_none_adds_nothing_to_the_manifest(df, tmp_path):
    expected = {
        "model_class",
        "predictors",
        "predictor_layout_hash",
        "target",
        "horizon",
        "freq",
        "training_window",
        "model_params",
        "data_columns_hash",
        "data_timezone",
        "information_set",
        "test_start",
        "test_end",
        "package_version",
        "git_sha",
        "created_at",
        "updated_at",
        "schema_version",
    }
    ModelPipeline().add_model(ols()).run(df, *TEST, horizon=1, save_dir=str(tmp_path / "plain"))
    plain = json.loads(manifest_path(tmp_path / "plain" / "m.jsonl").read_text())
    assert set(plain) == expected and plain["information_set"] is None

    pipeline(ols(), level="none").run(df, *TEST, horizon=1, save_dir=str(tmp_path / "issue"))
    issued = json.loads(manifest_path(tmp_path / "issue" / "m.jsonl").read_text())
    assert set(issued) == expected and issued["information_set"] == INFO.to_dict()


def test_a_warn_run_then_none_drops_the_listing_and_warn_again_finds_it(df, tmp_path):
    predictors = PREDICTORS + ["observed"]
    pipeline(ols(predictors), level="warn").run(df, *TEST, horizon=1, save_dir=str(tmp_path))
    report = pipeline(ols(predictors), level="none").run(df, *TEST, horizon=1, save_dir=str(tmp_path))
    assert report.availability().empty
    assert "availability" not in json.loads(manifest_path(tmp_path / "m.jsonl").read_text())
    report = pipeline(ols(predictors), level="warn").run(df, *TEST, horizon=1, save_dir=str(tmp_path))
    assert rows(report, "predictor") == [["observed"]]


# ---------------------------------------------------------------------------
# verify and the probe do their job at every level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", AVAILABILITY_LEVELS)
def test_a_late_predictor_fails_verify_at_every_level(df, level):
    verified = pipeline(ols(PREDICTORS + ["observed"]), level=level).verify(
        df, *TEST, horizon=1, days=1, positive_control=False
    )
    assert verified.status("m") == "leak"
    assert not verified.ok


@pytest.mark.parametrize("level", ["none", "warn"])
def test_a_late_list_of_a_function_fails_verify(df, level):
    verified = pipeline(ols(late_at_noon), level=level).verify(df, *TEST, horizon=1, days=1, positive_control=False)
    assert verified.status("m") == "leak"


@pytest.mark.parametrize("level", ["none", "warn"])
def test_a_late_value_is_a_leak_where_the_run_would_only_list_it(df, level):
    verified = pipeline(ols(reads_late), level=level).verify(df, *TEST, horizon=1, days=1, positive_control=False)
    assert verified.status("m") == "leak"
    assert "read through ctx though known later" in " ".join(verified.problems())
    check = verified.checks[0]
    assert check.probe.late_reads == {"observed": 14}
    assert sorted(check.probe.changed) == [(period, 1) for period in range(10, 24)]


def test_a_late_value_stops_verify_at_raise(df):
    with pytest.raises(InformationSetError, match=r"ctx\.value\('observed'\)"):
        pipeline(ols(reads_late), level="raise").verify(df, *TEST, horizon=1, days=1, positive_control=False)


def test_a_late_value_the_noise_cannot_reach_is_still_a_leak(df):
    """The probe perturbs numbers only, so it can't move a text column. The read itself is the leak."""
    frame = df.assign(regime=np.where(df["load"] > 1000, "high", "low"))
    info = InformationSet("11:45", "Europe/Warsaw", rules={**RULES, "regime": Availability.realised("1h")})

    def by_regime(ctx):
        return PREDICTORS + (["price_d-1"] if ctx.value("regime") == "high" else [])

    verified = pipeline(ols(by_regime), info=info).verify(frame, *TEST, horizon=1, days=1, positive_control=False)
    assert verified.status("m") == "leak"
    assert verified.checks[0].probe.late_reads == {"regime": 14}
    assert verified.checks[0].probe.moved_reads == {}


@pytest.mark.parametrize("level", AVAILABILITY_LEVELS)
def test_the_probe_passes_a_clean_function_at_every_level(sparse, level):
    probe = leakage_probe(
        lambda: ols(by_publication),
        sparse.drop(columns="mystery"),
        TEST[0],
        SPARSE_INFO,
        horizon=1,
        availability=level,
        check_target_boundary=True,
    )
    probe.assert_ok()
    assert probe.forecasts == 24
    assert probe.late_reads == {}


@pytest.mark.parametrize("level", AVAILABILITY_LEVELS)
def test_verify_passes_a_clean_function_at_every_level(df, level):
    frame = df.assign(sparse=df["early"])
    verified = pipeline(ols(by_publication), level=level, info=SPARSE_INFO).verify(frame, *TEST, horizon=1, days=1)
    assert verified.status("m") == "ok", verified.problems()


def test_the_probe_refuses_a_late_value_by_default(df):
    frame = df.drop(columns="mystery")
    with pytest.raises(InformationSetError, match=r"ctx\.value\('observed'\)"):
        leakage_probe(lambda: ols(reads_late), frame, TEST[0], INFO, horizon=1)
    with pytest.raises(AssertionError, match="Read through ctx though known later"):
        leakage_probe(lambda: ols(reads_late), frame, TEST[0], INFO, horizon=1, availability="warn").assert_ok()


# ---------------------------------------------------------------------------
# YAML and the CLI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", AVAILABILITY_LEVELS)
def test_a_level_asked_for_is_saved_and_loaded(tmp_path, level):
    path = tmp_path / "mp.yaml"
    ModelPipeline(issue_time="11:45", availability=level).add_model(ols()).save(path)
    assert yaml.safe_load(path.read_text())["availability"] == level
    assert ModelPipeline.load(path).availability == level


def test_a_level_never_asked_for_stays_the_default(tmp_path):
    path = tmp_path / "mp.yaml"
    ModelPipeline(issue_time="11:45").add_model(ols()).save(path)
    assert "availability" not in yaml.safe_load(path.read_text())
    assert ModelPipeline.load(path).availability == "warn"

    ModelPipeline().add_model(ols()).save(path)
    assert "availability" not in yaml.safe_load(path.read_text())
    assert ModelPipeline.load(path).availability == "none"


def test_an_old_file_with_an_issue_time_now_warns_instead_of_raising(df, tmp_path):
    path = tmp_path / "old.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "class": "OLSModel",
                        "params": {"predictors": PREDICTORS + ["observed"], "training_window": 45, "name": "m"},
                    }
                ],
                "evaluators": [],
                "exporters": [],
                "information_set": {
                    "issue_time": "11:45",
                    "issue_timezone": "Europe/Warsaw",
                    "unknown": "raise",
                    "rules": {name: rule.to_dict() for name, rule in RULES.items()},
                },
            }
        )
    )
    loaded = ModelPipeline.load(path)
    assert loaded.availability == "warn"
    with pytest.warns(UserWarning, match="Availability"):
        report = loaded.run(df, *TEST, horizon=1)
    assert rows(report, "predictor", "status") == [["observed", "late"]]


def test_a_file_with_warn_and_no_issue_time_is_refused_on_load(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"models": [], "evaluators": [], "exporters": [], "availability": "raise"}))
    with pytest.raises(ValueError, match="issue_time or information_set"):
        ModelPipeline.load(path)


def workflow_config(tmp_path, predictors, **model_pipeline):
    index = pd.date_range("2023-01-01", "2023-04-01", freq="1h", tz="UTC", inclusive="left")
    rng = np.random.default_rng(0)
    pd.DataFrame({"datetime": index.strftime("%Y-%m-%d %H:%M:%S"), "price": rng.normal(100, 10, len(index))}).to_csv(
        tmp_path / "prices.csv", index=False
    )
    return {
        "data_pipeline": {
            "start": "2023-01-01",
            "end": "2023-03-31",
            "sources": [{"class": "CsvSource", "params": {"file_path": str(tmp_path / "prices.csv")}}],
            "transformers": [
                {"class": "TimezoneTransformer", "params": {"target_tz": "Europe/Warsaw"}},
                {"class": "LagTransformer", "params": {"columns": ["price"], "lags": [1, -2], "freq": "day"}},
            ],
        },
        "model_pipeline": {
            "test_start": "2023-03-01",
            "test_end": "2023-03-02",
            "horizon": 1,
            "models": [
                {"class": "OLSModel", "params": {"predictors": predictors, "training_window": 30, "name": "OLS"}},
                {"class": "OLSModel", "params": {"predictors": ["price_d-1"], "training_window": 30, "name": "Clean"}},
            ],
            "evaluators": [{"class": "MAEEvaluator", "params": {}}],
            "information_set": {"issue_time": "11:45", "issue_timezone": "Europe/Warsaw"},
            **model_pipeline,
        },
    }


def test_a_workflow_defaults_to_warn_and_takes_the_level_inline(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(workflow_config(tmp_path, ["price_d-1", "price_d+2"])))
    with pytest.warns(UserWarning, match="Availability"):
        report = Workflow.load(path).run()
    assert rows(report, "model", "predictor", "status") == [["OLS", "price_d+2", "late"]]

    path.write_text(yaml.safe_dump(workflow_config(tmp_path, ["price_d-1", "price_d+2"], availability="raise")))
    with pytest.raises(InformationSetError, match=r"price_d\+2"):
        Workflow.load(path).run()

    path.write_text(yaml.safe_dump(workflow_config(tmp_path, ["price_d-1", "price_d+2"], availability="none")))
    assert Workflow.load(path).run().availability().empty


def test_a_workflow_saves_the_inline_level(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(workflow_config(tmp_path, ["price_d-1"], availability="raise")))
    saved = tmp_path / "again.yaml"
    Workflow.load(path).save(saved)
    assert yaml.safe_load(saved.read_text())["model_pipeline"]["availability"] == "raise"
    assert Workflow.load(saved)._build_model_pipeline().availability == "raise"


def test_a_workflow_with_a_level_and_no_issue_time_is_refused(tmp_path):
    config = workflow_config(tmp_path, ["price_d-1"], availability="raise")
    del config["model_pipeline"]["information_set"]
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="issue_time or information_set"):
        Workflow.load(path).run()


def test_an_array_of_tasks_merges_into_one_listing(tmp_path, capsys):
    """SLURM array: one model per task. `Workflow.report()` knows the listing only from the manifests."""
    config = workflow_config(
        tmp_path,
        ["price_d-1", "price_d+2"],
        save_dir="results",
        exporters=[{"class": "TerminalExporter", "params": {"show": ["summary"]}}],
    )
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    for index in (0, 1):
        config["environment"] = {"model_index": index}
        path.write_text(yaml.safe_dump(config))
        Workflow.load(path).run()
    del config["environment"]
    path.write_text(yaml.safe_dump(config))
    capsys.readouterr()

    with pytest.warns(UserWarning, match="Availability") as captured:
        report = Workflow.load(path).report()
    assert len(availability_warnings(captured)) == 1
    assert rows(report, "model", "predictor", "periods", "horizons") == [["OLS", "price_d+2", "0-23", "1"]]
    out = capsys.readouterr().out
    assert out.index("Availability") < out.index("Summary")


def model_argv(tmp_path, local_df, predictors, *extra):
    from epftoolbox2._frame_csv import save_frame_csv

    data_path = tmp_path / "data.csv"
    save_frame_csv(local_df, data_path)
    yaml_path = tmp_path / "mp.yaml"
    ModelPipeline().add_model(ols(predictors)).save(yaml_path)
    return [
        "model",
        str(yaml_path),
        "--data",
        str(data_path),
        "--test-start",
        TEST[0],
        "--test-end",
        TEST[0],
        "--horizon",
        "1",
        *extra,
    ]


def test_epf_model_takes_the_level(df, tmp_path, monkeypatch):
    frame = df[["price", "price_d-1"]].rename(columns={"price_d-1": "price_lag"})
    frame["load_actual"] = df["load"]
    frame["load_actual_d-1"] = df["load"].shift(24)
    predictors = ["load_actual_d-1", "load_actual"]
    clock = ["--issue-time", "11:45", "--issue-timezone", "Europe/Warsaw"]

    def run(*extra):
        monkeypatch.setattr("sys.argv", ["epf", *model_argv(tmp_path, frame.iloc[24:], predictors, *clock, *extra)])
        cli.main()

    with pytest.warns(UserWarning, match="Availability"):
        run()
    with pytest.raises(InformationSetError, match="load_actual"):
        run("--availability", "raise")
    run("--availability", "none")


def test_epf_model_refuses_a_level_without_an_issue_time(df, tmp_path, monkeypatch, capsys):
    frame = df[["price", "load"]].rename(columns={"load": "load_actual"})
    monkeypatch.setattr("sys.argv", ["epf", *model_argv(tmp_path, frame, ["load_actual"], "--availability", "raise")])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 1
    assert "issue_time or information_set" in capsys.readouterr().out


def test_epf_model_offers_the_levels_there_are():
    parser = cli.build_parser()
    actions = {
        action.dest: action for sub in parser._subparsers._group_actions for action in sub.choices["model"]._actions
    }
    assert tuple(actions["availability"].choices) == AVAILABILITY_LEVELS
