"""Vintage classes in results (step 3h): what a run records of each input's class, what a report shows, and when a comparison
says that its models read one series with different classes. Nothing here may move a number."""

import dataclasses
import json
import warnings
from concurrent.futures import Future

import pandas as pd
import pytest

import epftoolbox2.information_set as information_set_module
from epftoolbox2.ensembles import AverageEnsemble
from epftoolbox2.evaluators import DMEvaluator, MAEEvaluator
from epftoolbox2.exporters import TerminalExporter
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import OLSModel
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import manifest_digest
from epftoolbox2.results.ref import ModelResultRef
from epftoolbox2.results.report import EvaluationReport
from epftoolbox2.results.vintages import COLUMNS, mixed_classes, vintage_rows
from tests.forecast_records import forecast_records
from tests.synthetic import make_market_data

PL = "Europe/Warsaw"
USER_RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
}
TEST = ("2023-03-20", "2023-03-26")
PROVENANCE = ("package_version", "git_sha", "created_at", "updated_at")


class InProcessPool:
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
def data():
    frame = make_market_data(start="2023-02-01", days=60, tz=PL, horizon=2, noise=5.0, seed=9)
    return frame.assign(**{"load_forecast": frame["load"], "load_forecast_d+1": frame["load_d+1"]})


def manifest_of(path):
    manifest = json.loads(path.read_text())
    for key in PROVENANCE:
        manifest.pop(key)
    return manifest


def reads_the_price(ctx):
    """Reads the built-in `price` through `ctx`, a snapshot, and adds the revised day-ahead load forecast on high days."""
    names = ["wind_d+1", "is_weekend_d+1"]
    if ctx.value("price_d-1") > 100:
        names.append("load_forecast_d+1")
    return names


def test_a_run_records_each_input_s_class_on_its_ref_and_in_its_manifest_and_a_report_reads_it_back(data, tmp_path):
    pipeline = ModelPipeline(information_set=InformationSet("11:45", PL, rules=USER_RULES))
    pipeline.add_model(OLSModel(["load_forecast_d+1", "wind_d+1", "price_d-1"], training_window=20, name="fixed"))
    pipeline.add_model(OLSModel(reads_the_price, training_window=20, name="function"))
    pipeline.add_model(OLSModel(["load_d+1", "wind_d+1"], training_window=20, name="yours"))
    pipeline.add_evaluator(MAEEvaluator())
    report = pipeline.run(data, test=TEST, horizon=1, save_dir=str(tmp_path))
    row = {"load_forecast_d+1": "latest_revision", "price_d-1": "snapshot", "wind_d+1": "unknown"}
    series = {
        "load_forecast_d+1": "load_forecast",
        "price_d-1": "price",
        "wind_d+1": "wind",
        "is_weekend_d+1": "is_weekend",
    }
    assert report.refs["fixed"].vintages == [
        {"predictor": name, "series": series[name], "class": row[name], "ctx": False} for name in sorted(row)
    ]
    assert report.refs["function"].vintages == [
        {"predictor": "is_weekend_d+1", "series": "is_weekend", "class": "unknown", "ctx": False},
        {"predictor": "load_forecast_d+1", "series": "load_forecast", "class": "latest_revision", "ctx": False},
        {"predictor": "price_d-1", "series": "price", "class": "snapshot", "ctx": True},
        {"predictor": "wind_d+1", "series": "wind", "class": "unknown", "ctx": False},
    ]
    assert report.refs["yours"].vintages is None  # no input with a class: nothing recorded
    manifests = {name: manifest_of(tmp_path / f"{name}.manifest.json") for name in ("fixed", "function", "yours")}
    assert manifests["fixed"]["vintages"] == report.refs["fixed"].vintages
    assert manifests["function"]["vintages"] == report.refs["function"].vintages
    assert "vintages" not in manifests["yours"]
    table = report.vintages()
    assert list(table.columns) == COLUMNS
    assert table.to_dict("records") == [
        {"model": model, **row} for model in ("fixed", "function") for row in report.refs[model].vintages
    ]
    reported = pipeline.report(str(tmp_path), test=TEST, horizon=1)
    pd.testing.assert_frame_equal(reported.vintages(), table)
    assert reported.aggregate().vintages().equals(table)
    # A run with no information set knows no rule, and so no class.
    alone = OLSModel(["price_d-1"], training_window=20).run(data, test_start=TEST[0], test_end=TEST[1], horizon=1)
    assert alone.vintages is None
    assert list(EvaluationReport({}, []).vintages().columns) == COLUMNS


def test_the_rows_of_a_model_are_its_inputs_and_what_it_read_through_ctx():
    info = InformationSet("11:45", PL, rules={**USER_RULES, "flows": Availability.realised("2h", vintage="row_stamp")})
    rows = vintage_rows(info, "1h", ["price_d-1", "load_d+1", "flows_h05_d-1"], ["flows_h05_d-1", "load_actual_h-2"])
    assert rows == [
        {"predictor": "flows_h05_d-1", "series": "flows", "class": "row_stamp", "ctx": True},
        {"predictor": "load_actual_h-2", "series": "load_actual", "class": "latest_revision", "ctx": True},
        {"predictor": "load_d+1", "series": "load", "class": "unknown", "ctx": False},
        {"predictor": "price_d-1", "series": "price", "class": "snapshot", "ctx": False},
    ]
    assert vintage_rows(info, "1h", ["load_d+1", "nothing_d-1"]) is None  # unknown, and a column with no rule at all
    assert vintage_rows(None, "1h", ["price_d-1"]) is None


def test_a_class_on_every_built_in_rule_moves_no_forecast(monkeypatch, data, tmp_path):
    """The built-in rules without their classes, as before this step, against with them: records byte for byte, manifests
    alike but for `vintages`."""
    rules = {"load": USER_RULES["load"], "wind": USER_RULES["wind"], "is_weekend": Availability.static()}
    lists = ["load_forecast_d+1", "wind_d+1", "is_weekend_d+1", "price_d-1"]

    def run(where):
        info = InformationSet("11:45", PL, rules=rules)
        OLSModel(lists, training_window=20).run(
            data, test_start=TEST[0], test_end=TEST[1], horizon=1, information_set=info, save_to=where
        )

    run(str(tmp_path / "with" / "m.jsonl"))
    without = {
        name: dataclasses.replace(rule, vintage=None) for name, rule in information_set_module._built_in_rules().items()
    }
    monkeypatch.setattr(information_set_module, "_built_in_rules", lambda: without)
    run(str(tmp_path / "without" / "m.jsonl"))
    lines = {where: sorted((tmp_path / where / "m.jsonl").read_text().splitlines()) for where in ("with", "without")}
    assert lines["with"] == lines["without"]
    manifests = {where: manifest_of(tmp_path / where / "m.manifest.json") for where in ("with", "without")}
    assert "vintages" not in manifests["without"]
    assert manifests["with"].pop("vintages") == [
        {"predictor": "is_weekend_d+1", "series": "is_weekend", "class": "unknown", "ctx": False},
        {"predictor": "load_forecast_d+1", "series": "load_forecast", "class": "latest_revision", "ctx": False},
        {"predictor": "price_d-1", "series": "price", "class": "snapshot", "ctx": False},
        {"predictor": "wind_d+1", "series": "wind", "class": "unknown", "ctx": False},
    ]
    assert manifests["with"] == manifests["without"]


def reads_the_load_forecast_early(ctx):
    if ctx.issue_day.isoformat() < "2023-03-23":
        ctx.value("load_forecast_d+1")
    return ["wind_d+1", "is_weekend_d+1"]


def test_classes_are_recorded_never_compared_and_kept_for_what_an_earlier_run_read(data, tmp_path):
    """A resume with other classes, of an input and of the target a published model reads its cut from, goes on and writes the
    same records; the manifest has this run's classes. A narrower resume keeps the class of what only the earlier run read."""
    path = tmp_path / "m.jsonl"

    def run(vintage, test=TEST, predictors=("load_forecast_d+1", "wind_d+1")):
        rules = {
            **USER_RULES,
            "wind": Availability.published("10:00", 7, vintage=vintage),
            "price": Availability.published("13:00", 1, vintage=vintage),
        }
        model = OLSModel(list(predictors), training_window=20, newest_target="published")
        model.run(
            data,
            test_start=test[0],
            test_end=test[1],
            horizon=1,
            information_set=InformationSet("11:45", PL, rules=rules),
            save_to=str(path),
        )
        return sorted(path.read_text().splitlines()), manifest_of(path.with_name("m.manifest.json"))

    first, before = run(None)
    again, after = run("snapshot")
    assert again == first
    assert before["training_cut"] == after["training_cut"]
    classes = {row["predictor"]: row["class"] for row in after["vintages"]}
    assert classes == {"load_forecast_d+1": "latest_revision", "wind_d+1": "snapshot"}
    assert {row["predictor"]: row["class"] for row in before["vintages"]}["wind_d+1"] == "unknown"

    path = tmp_path / "f" / "m.jsonl"
    model = OLSModel(reads_the_load_forecast_early, training_window=20)
    info = InformationSet("11:45", PL, rules=USER_RULES)
    for test in (TEST, ("2023-03-24", "2023-03-26")):
        model.run(data, test_start=test[0], test_end=test[1], horizon=1, information_set=info, save_to=str(path))
    kept = manifest_of(path.with_name("m.manifest.json"))["vintages"]
    assert [(row["predictor"], row["ctx"]) for row in kept] == [
        ("is_weekend_d+1", False),
        ("load_forecast_d+1", True),
        ("wind_d+1", False),
    ]
    # Other rules leave nothing of the earlier run's: what it said may no longer be true.
    other = InformationSet("11:45", PL, rules={**USER_RULES, "wind": Availability.published("09:00", 7)})
    model.run(data, test_start="2023-03-24", test_end="2023-03-26", horizon=1, information_set=other, save_to=str(path))
    assert "vintages" not in manifest_of(path.with_name("m.manifest.json"))  # what it read now has no class


def test_an_ensemble_reads_its_models_classes_and_its_manifest_does_not_move_with_them(data, tmp_path):
    pipeline = ModelPipeline(information_set=InformationSet("11:45", PL, rules=USER_RULES))
    pipeline.add_model(OLSModel(["load_forecast_d+1", "wind_d+1"], training_window=20, name="a"))
    pipeline.add_model(OLSModel(["price_d-1", "wind_d+1"], training_window=20, name="b"))
    pipeline.add_ensemble(AverageEnsemble(members=["a", "b"], name="avg"))
    report = pipeline.run(data, test=TEST, horizon=1, save_dir=str(tmp_path))
    rows = report.vintages()
    assert sorted(rows[rows["model"] == "avg"]["predictor"]) == ["load_forecast_d+1", "price_d-1", "wind_d+1"]
    member = json.loads((tmp_path / "a.manifest.json").read_text())
    assert "vintages" in member
    assert manifest_digest(member) == manifest_digest({k: v for k, v in member.items() if k != "vintages"})
    ensemble = json.loads((tmp_path / "avg.manifest.json").read_text())
    assert "vintages" not in ensemble


# ---------------------------------------------------------------------------------------------------------------------------
# Compared models reading one series with different classes
# ---------------------------------------------------------------------------------------------------------------------------

DAYS = pd.date_range("2024-01-01", periods=60)


def ref(name, seed, classes):
    """A model's ref over plain records, reading `load` (and `price`) with the given classes."""
    out = ModelResultRef(name, None, 0, "", "", 1, _results=forecast_records(seed, DAYS, noise=3.0 + seed))
    out.vintages = [
        {"predictor": f"{series}_d-1", "series": series, "class": vintage, "ctx": False}
        for series, vintage in classes.items()
    ] or None
    return out


def report_of(classes):
    refs = {name: ref(name, seed, classes.get(name, {})) for seed, name in enumerate(["a", "b", "c"])}
    return EvaluationReport(refs, [MAEEvaluator(), DMEvaluator(benchmark="a")])


def vintage_warnings(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table = call()
    return table, [str(w.message) for w in caught if "vintage classes" in str(w.message)]


TABLES = {
    "tests": lambda r: r.tests(),
    "compare": lambda r: r.compare("a", "b"),
    "leaderboard": lambda r: r.leaderboard(),
    "summary": lambda r: r.summary(),
    "by_horizon": lambda r: r.by_horizon(),
}


@pytest.mark.parametrize("table", TABLES)
def test_compared_models_with_different_classes_of_one_series_warn_and_the_table_says_which(table):
    mixed = report_of(
        {"a": {"load": "snapshot", "price": "snapshot"}, "b": {"load": "latest_revision", "price": "snapshot"}}
    )
    found, said = vintage_warnings(lambda: TABLES[table](mixed))
    assert len(said) == 1 and "'load'" in said[0] and "'price'" not in said[0]
    assert found.attrs["vintages"] == {"load": {"a": "snapshot", "b": "latest_revision"}}


@pytest.mark.parametrize("table", TABLES)
def test_equal_classes_and_undeclared_ones_neither_warn_nor_change_a_table(table):
    plain = report_of({})
    expected, _ = vintage_warnings(lambda: TABLES[table](plain))
    for classes in (
        {"a": {"load": "latest_revision"}, "b": {"load": "latest_revision"}, "c": {"load": "latest_revision"}},
        {"a": {"load": "snapshot"}, "b": {}, "c": {"price": "latest_revision"}},  # read by one model only
        {"a": {"load": "snapshot"}, "b": {"load": "unknown"}},  # an undeclared class says nothing to compare
    ):
        found, said = vintage_warnings(lambda: TABLES[table](report_of(classes)))
        assert said == [] and "vintages" not in found.attrs, classes
        pd.testing.assert_frame_equal(found, expected)


def test_a_test_against_the_benchmark_compares_each_model_with_the_benchmark_only():
    """b and c read `load` with different classes, and the benchmark a doesn't read it: no pair of `tests()` or of the DM
    column compares b with c, but `leaderboard()` ranks them all on one MAE."""
    classes = {"b": {"load": "snapshot"}, "c": {"load": "latest_revision"}}
    for table in ("tests", "summary"):
        found, said = vintage_warnings(lambda: TABLES[table](report_of(classes)))
        assert said == [] and "vintages" not in found.attrs, table
    found, said = vintage_warnings(lambda: report_of(classes).leaderboard())
    assert said and found.attrs["vintages"] == {"load": {"b": "snapshot", "c": "latest_revision"}}
    assert mixed_classes({"b": ref("b", 1, classes["b"]).vintages}, [("b", "c")]) == {}


def test_the_terminal_exporter_prints_the_revised_inputs_next_to_availability(capsys):
    revised = report_of({"a": {"load": "latest_revision", "price": "snapshot"}, "b": {"flows": "last_write_stamp"}})
    TerminalExporter(show=[]).export(revised)
    out = capsys.readouterr().out
    assert "Vintages: inputs read from an archive of revised values" in out
    assert "load_d-1" in out and "flows_d-1" in out and "price_d-1" not in out
    TerminalExporter(show=[]).export(report_of({"a": {"price": "snapshot"}}))
    TerminalExporter(show=[]).export(report_of({}))
    assert "Vintages" not in capsys.readouterr().out
