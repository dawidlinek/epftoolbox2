"""Engine groups (step 5h): `ModelPipeline(engine="on")` runs the engine-capable models of a refit rule in one pool, on one read of
the data, and every record, manifest and table is what `engine="off"` gives, to the bit.

The grid of `test_engine_on_equals_off` covers every level of every axis and every pair of levels across its cells, not every
cell (tests.md): OLS, LassoCV, Ridge and LEAR; refit None, "7D" and "auto"; one window and a list; a fixed list and a predictor
function; `binary_mask` None and "fit"; 1h, 15 min and 1D; the spring and the autumn switch; a plain run, validation + test and
Fridays; `newest_target` "d-1" and "published"; JSONL and Parquet. The larger grid, the 200-model group and the 1,000-model
measurement are in `epftoolbox2-research/library_check_step5/engine_groups/`.
"""

import datetime
import gc
import json
import random
import types
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from threadpoolctl import threadpool_limits

from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import LassoCVModel, LEARModel, NaiveModel, OLSModel, RidgeModel, Use
from epftoolbox2.models import _engine, _engine_worker
from epftoolbox2.models import _worker as worker
from epftoolbox2.models.base import BaseModel, _RunState
from epftoolbox2.pipelines import ModelPipeline, Workflow
from epftoolbox2.results.store import ResultStore, iter_file_records
from epftoolbox2.scalers.standard import StandardScaler
from tests.synthetic import PREDICTORS, calendar_shift, make_market_data

PROVENANCE = ("package_version", "git_sha", "created_at", "updated_at", "record_schema", "kernels")
PUBLISHED = {"price": Availability.published("13:00", 1)}


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.delenv("ENGINE_MAX_MODELS", raising=False)
    monkeypatch.delenv("ENGINE_MAX_COLUMNS", raising=False)


class InProcessPool:
    """Stands in for an engine pool where a test runs many groups and compares engine runs with each other only."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        _engine_worker._engine_load(*initargs)

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

    def shutdown(self, wait=True, cancel_futures=False):
        pass


@pytest.fixture
def in_process(monkeypatch):
    monkeypatch.setattr(_engine, "ProcessPoolExecutor", InProcessPool)
    with threadpool_limits(limits=1):  # what a pool's worker runs on
        yield


# --- predictor functions (module level: a plan holds them) ------------------------------------------------------------


def by_weekday(ctx):
    names = [f"load_d+{ctx.horizon}", f"wind_d+{ctx.horizon}", f"is_weekend_d+{ctx.horizon}"]
    return names if ctx.target_day.weekday() < 4 else names + ["price_d-1"]


def with_aux(ctx):
    names = [f"load_d+{ctx.horizon}", f"wind_d+{ctx.horizon}", "aux"]
    return names if ctx.issue_day.weekday() != 2 else names + [f"is_weekend_d+{ctx.horizon}"]


def matching_list(ctx):
    names = [f"load_d+{ctx.horizon}", f"wind_d+{ctx.horizon}"]
    if ctx.target_day.weekday() >= 5:
        return Use(names, train_on="matching")
    return names + [f"is_weekend_d+{ctx.horizon}", f"is_weekday_d+{ctx.horizon}"]


class OwnFit(OLSModel):
    """Overrides the fit without `_fit_stats`: not engine-capable, runs on its own."""

    def _fit_predict(self, train_x, train_y, test_x):
        return super()._fit_predict(train_x, train_y, test_x)

    def _fit_model(self, train_x, train_y):
        return super()._fit_model(train_x, train_y)

    def _predict_fitted(self, fitted, test_x):
        return super()._predict_fitted(fitted, test_x)


class Crashing(OLSModel):
    """OLS (engine-capable) whose fits a test makes raise, through the worker functions it patches."""


# --- frames -----------------------------------------------------------------------------------------------------------


def holes(data, day, hours=(3, 4), drop_hour=7, column="load_d+1"):
    data = data.copy()
    issue = (data.index.date == pd.Timestamp(day).date()) & np.isin(data.index.hour, list(hours))
    data.loc[issue, column] = np.nan
    gone = data.index[(data.index.date == pd.Timestamp(day).date()) & (data.index.hour == drop_hour)]
    return data.drop(index=gone)


def with_aux_column(data, change, seed):
    """`aux`: the weekend flag (0/1) before `change`, continuous after, so masks read on different rows differ."""
    after = data.index >= pd.Timestamp(change, tz=data.index.tz)
    aux = data["is_weekend"].to_numpy(dtype=float).copy()
    aux[after] = np.random.default_rng(seed).normal(0.5, 0.3, after.sum())
    data["aux"] = aux
    return data


@pytest.fixture(scope="module")
def spring():
    data = make_market_data(start="2022-12-01", days=150, noise=5.0, seed=41, horizon=3)
    data = with_aux_column(data, "2023-03-25", 42)
    return holes(data, "2023-03-27")


@pytest.fixture(scope="module")
def autumn():
    data = make_market_data(start="2023-06-01", days=160, noise=5.0, seed=43, horizon=3)
    data = with_aux_column(data, "2023-10-28", 44)
    return holes(data, "2023-10-30", column="wind_d+2")


@pytest.fixture(scope="module")
def quarter():
    data = make_market_data(start="2023-09-01", days=62, noise=5.0, seed=45, horizon=1, freq="15min")
    data = with_aux_column(data, "2023-10-27", 46)
    issue = (data.index.date == pd.Timestamp("2023-10-30").date()) & (data.index.hour == 5)
    data.loc[issue & (data.index.minute == 15), "load_d+1"] = np.nan
    return data.drop(index=data.index[issue & (data.index.minute == 45)])


@pytest.fixture(scope="module")
def daily():
    data = make_market_data(start="2023-01-01", days=200, noise=5.0, seed=47, horizon=7).at_time("12:00")
    data = with_aux_column(data.copy(), "2023-05-20", 48)
    data.iloc[150, data.columns.get_loc("load_d+3")] = np.nan
    return data


# --- comparing results ------------------------------------------------------------------------------------------------


def lines(path):
    return sorted(line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


def stored(path):
    """A results file's or directory's records, every one as written, as sorted JSON (Parquet parts aren't reproducible)."""
    return sorted(json.dumps(record, sort_keys=True) for record in iter_file_records(Path(path)))


def manifest(path):
    content = json.loads(Path(path).read_text())
    return {key: value for key, value in content.items() if key not in PROVENANCE}


def same_results(one, other):
    """Two save_dirs hold the same results: every model's records (as sorted lines, or a Parquet directory's records) and
    every manifest without provenance. Returns how many files were compared."""
    names = sorted(p.name for p in Path(one).iterdir() if not p.name.endswith(".lock"))
    assert names == sorted(p.name for p in Path(other).iterdir() if not p.name.endswith(".lock"))
    for name in names:
        a, b = Path(one) / name, Path(other) / name
        if name.endswith(".jsonl"):
            assert lines(a) == lines(b), name
        elif name.endswith(".parquet"):
            assert stored(a) == stored(b), name
        elif name.endswith(".manifest.json"):
            assert manifest(a) == manifest(b), name
    return len(names)


def tables(report):
    out = {name: table for name, table in report.compute_all().items()}
    out["forecast_counts"] = report.forecast_counts()
    out["by_day"] = report.by_day()
    return out


def same_tables(one, other):
    a, b = tables(one), tables(other)
    assert list(a) == list(b)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name], check_exact=True)


def run_both(build, data, tmp_path, **arguments):
    """The pipeline `build(engine)` run with engine off and on into two save_dirs; the two reports."""
    reports = {}
    for engine in ("off", "on"):
        reports[engine] = build(engine).run(data, save_dir=str(tmp_path / engine), **arguments)
    return reports["off"], reports["on"]


def by_name(models):
    return {model.name: model for model in models}


# --- engine-capable models and groups ---------------------------------------------------------------------------------


def test_which_models_the_engine_takes():
    for model in (OLSModel(PREDICTORS), LassoCVModel(PREDICTORS), RidgeModel(PREDICTORS), LEARModel(PREDICTORS)):
        assert model._engine_capable()
        assert not model._engine_capable(forecast_only=True)
    assert Crashing(PREDICTORS)._engine_capable()  # a subclass that keeps its parent's four methods
    assert not NaiveModel(["price_d-1"])._engine_capable()
    assert not OwnFit(PREDICTORS)._engine_capable()  # its own fit, its parent's _fit_stats

    class OwnRun(OLSModel):
        def _run_tasks(self, tasks, state, sink):
            return super()._run_tasks(tasks, state, sink)

    assert not OwnRun(PREDICTORS)._engine_capable()  # the group does what it overrides

    class NoStats(BaseModel):
        def _fit_predict(self, x, y, t):
            return 0.0, []

        def _fit_model(self, x, y):
            return None

        def _predict_fitted(self, fitted, t):
            return 0.0, []

    assert not NoStats(PREDICTORS)._engine_capable()


def test_a_matching_plan_is_not_taken(spring):
    model = OLSModel(matching_list, 28, name="matching")
    plan = model._plan_predictors(spring, "price", 1, "1h", None, "2023-03-24", "2023-03-24")
    assert plan.uses_matching and not model._engine_capable(plan)
    other = OLSModel(by_weekday, 28, name="function")
    assert other._engine_capable(other._plan_predictors(spring, "price", 1, "1h", None, "2023-03-24", "2023-03-24"))


def test_groups_are_refit_rules_and_chunks_are_cut_in_list_order(monkeypatch):
    models = [
        OLSModel(["a", "b"], 28, name="m0"),
        RidgeModel(["a"], 28, refit="7D", name="m1"),
        OLSModel(["c"], [14, 120], refit="auto", name="m2"),  # 1D and 7D: block 7D
        LEARModel(["a", "d"], 28, name="m3"),
        OLSModel(["b"], [14, 28], refit="auto", name="m4"),  # 1D only
    ]
    groups = _engine.engine_groups(models)
    assert [[m.name for m in group] for group in groups] == [["m0", "m3"], ["m1", "m2"], ["m4"]]

    class OwnFrames(OLSModel):
        def _preprocess(self, data, horizon, target):
            return super()._preprocess(data, horizon, target)

    mixed = [models[0], OwnFrames(["a"], 28, name="own"), models[3], OwnFrames(["b"], 28, name="own2")]
    groups = _engine.engine_groups(mixed)
    assert [[m.name for m in group] for group in groups] == [["m0", "m3"], ["own"], ["own2"]]  # each instance its own
    with pytest.raises(ValueError, match="reads its own period frames"):
        _engine.run_group(mixed[1::2], {}, {"own": ["a"], "own2": ["b"]})

    columns = {"m0": ["a", "b"], "m1": ["a"], "m2": ["c"], "m3": ["a", "d"], "m4": ["b"]}
    assert [[m.name for m in c] for c in _engine.engine_chunks(models, columns, 2, 100)] == [
        ["m0", "m1"],
        ["m2", "m3"],
        ["m4"],
    ]
    assert [[m.name for m in c] for c in _engine.engine_chunks(models, columns, 100, 3)] == [
        ["m0", "m1", "m2"],
        ["m3", "m4"],
    ]
    assert [[m.name for m in c] for c in _engine.engine_chunks(models, columns, 100, 1)] == [
        ["m0"],
        ["m1"],
        ["m2"],
        ["m3"],
        ["m4"],
    ]

    assert _engine.engine_limits() == (20_000, 4_096)
    monkeypatch.setenv("ENGINE_MAX_MODELS", "7")
    monkeypatch.setenv("ENGINE_MAX_COLUMNS", "50")
    assert _engine.engine_limits() == (7, 50)
    for bad in ("0", "-3", "many", "1.5"):
        monkeypatch.setenv("ENGINE_MAX_COLUMNS", bad)
        with pytest.raises(ValueError, match="ENGINE_MAX_COLUMNS must be a whole number from 1"):
            _engine.engine_limits()


def test_engine_is_checked_and_saved_only_when_on(tmp_path):
    with pytest.raises(ValueError, match="engine must be one of"):
        ModelPipeline(engine="yes")
    pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, 28, name="ols"))
    with pytest.raises(ValueError, match="engine must be one of"):
        pipeline.engine = "check"  # step 5j's
    assert "engine" not in pipeline.to_dict()
    pipeline.engine = "on"
    assert pipeline.to_dict()["engine"] == "on"
    pipeline.save(tmp_path / "pipeline.yaml")
    assert ModelPipeline.load(tmp_path / "pipeline.yaml").engine == "on"
    off = ModelPipeline().add_model(OLSModel(PREDICTORS, 28, name="ols"))
    off.save(tmp_path / "off.yaml")
    assert "engine" not in yaml.safe_load((tmp_path / "off.yaml").read_text())
    assert ModelPipeline.load(tmp_path / "off.yaml").engine == "off"


@pytest.mark.parametrize("word, engine", [("on", "on"), ("off", "off"), ('"on"', "on"), ("'off'", "off")])
def test_engine_written_by_hand_in_yaml(tmp_path, word, engine):
    """`engine: on` unquoted is YAML 1.1's True and `off` its False: both read as the word, in a pipeline file and in a
    workflow's `model_pipeline`, quoted or not; a workflow saves the word back."""
    ModelPipeline().add_model(OLSModel(PREDICTORS, 28, name="ols")).save(tmp_path / "pipeline.yaml")
    text = (tmp_path / "pipeline.yaml").read_text()
    (tmp_path / "hand.yaml").write_text(f"engine: {word}\n{text}")
    assert ModelPipeline.load(tmp_path / "hand.yaml").engine == engine
    (tmp_path / "experiment.yaml").write_text(
        "data_pipeline:\n  start: '2023-01-01'\n  end: '2023-02-01'\n"
        f"model_pipeline:\n  path: pipeline.yaml\n  test: ['2023-01-20', '2023-01-21']\n  engine: {word}\n"
    )
    workflow = Workflow.load(tmp_path / "experiment.yaml")
    assert workflow._build_model_pipeline().engine == engine
    workflow.save(tmp_path / "again.yaml")
    assert yaml.safe_load((tmp_path / "again.yaml").read_text())["model_pipeline"]["engine"] == engine
    assert Workflow.load(tmp_path / "again.yaml")._build_model_pipeline().engine == engine
    with pytest.raises(ValueError, match="engine must be one of"):
        ModelPipeline(engine=True)  # the Python API takes the words alone


def test_a_workflow_overrides_the_pipeline_engine(tmp_path):
    pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, 28, name="ols"))
    pipeline.save(tmp_path / "pipeline.yaml")
    workflow = {
        "data_pipeline": {"start": "2023-01-01", "end": "2023-02-01"},
        "model_pipeline": {"path": "pipeline.yaml", "test": ["2023-01-20", "2023-01-21"], "engine": "on"},
    }
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(workflow))
    loaded = Workflow.load(tmp_path / "experiment.yaml")
    assert loaded._build_model_pipeline().engine == "on"
    assert loaded.to_dict()["model_pipeline"]["engine"] == "on"
    workflow["model_pipeline"]["engine"] = "sometimes"
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(workflow))
    with pytest.raises(ValueError, match="engine must be one of"):
        Workflow.load(tmp_path / "experiment.yaml")._build_model_pipeline()
    workflow["model_pipeline"].pop("engine")
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(workflow))
    assert Workflow.load(tmp_path / "experiment.yaml")._build_model_pipeline().engine == "off"


def squared_load(self, frame, horizon, target):
    """A `_preprocess` set on one instance: `load_d+1` squared."""
    frame = frame.copy()
    frame["load_d+1"] = frame["load_d+1"] ** 2
    return BaseModel._preprocess(self, frame, horizon, target)


def test_a_method_set_on_an_instance_is_the_instances(spring, tmp_path):
    """A frame reader set on an instance (`model._preprocess = ...`) makes it a group of its own, which reads its frames
    with it; a method the group does in a model's place set on an instance (`_run_tasks`, `_build_hour_arrays`) keeps the
    model out of the engine. Either way engine on equals engine off (before, the patched `_preprocess` was grouped with the
    plain model and read through BaseModel's)."""

    def models():
        plain, patched = OLSModel(PREDICTORS, 28, name="plain"), OLSModel(PREDICTORS, 28, name="patched")
        patched._preprocess = types.MethodType(squared_load, patched)
        arrays = OLSModel(PREDICTORS, 28, name="arrays")
        arrays._build_hour_arrays = types.MethodType(BaseModel._build_hour_arrays, arrays)
        return [plain, patched, arrays]

    plain, patched, arrays = models()
    assert plain._engine_capable() and patched._engine_capable() and not arrays._engine_capable()
    assert patched._frame_readers()[1] is patched
    assert [[m.name for m in g] for g in _engine.engine_groups([plain, patched])] == [["plain"], ["patched"]]
    off, on = run_both(build(models), spring, tmp_path, test=("2023-03-24", "2023-03-25"), horizon=1)
    assert same_results(tmp_path / "off", tmp_path / "on") == 6
    same_tables(off, on)
    assert lines(tmp_path / "on" / "plain.jsonl") != lines(tmp_path / "on" / "patched.jsonl")


def test_run_alone_classifies_on_the_plan_the_run_uses(spring, monkeypatch):
    """`run_alone` of a predictor function without a plan is the model's own run: its plan (which decides a matching list)
    isn't made until the run; with the plan, as the probes pass it, a matching one is the model's own run too."""
    grouped = []
    real_group = _engine.run_group
    monkeypatch.setattr(_engine, "run_group", lambda *a, **k: grouped.append(1) or real_group(*a, **k))
    days = ("2023-03-24", "2023-03-24")
    for model in (OLSModel(matching_list, 28, name="m"), OLSModel(by_weekday, 28, name="f")):
        mine = _engine.run_alone(model, spring, *days, horizon=1)
        theirs = type(model)(model._ctx_fn, 28, name=model.name).run(spring, *days, horizon=1)
        assert sorted(map(json.dumps, mine._results)) == sorted(map(json.dumps, theirs._results))
    assert grouped == []
    planned = OLSModel(matching_list, 28, name="m")
    plan = planned._plan_predictors(spring, "price", 1, "1h", None, *days)
    _engine.run_alone(planned, spring, *days, horizon=1, plan=plan)
    assert grouped == []
    fixed = OLSModel(PREDICTORS, 28, name="ols")
    _engine.run_alone(fixed, spring, *days, horizon=1)
    assert grouped == [1]


# --- engine on equals engine off --------------------------------------------------------------------------------------


class Lagged(OLSModel):
    """Builds a column of its frames from a setting of the instance: two instances can't share frames."""

    def __init__(self, *args, lag=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lag = lag

    def _preprocess(self, data, horizon, target):
        out = super()._preprocess(data, horizon, target)
        out["load_d+1"] = out["load_d+1"].shift(24 * self.lag)
        return out


def test_instances_that_read_their_own_frames_each_read_theirs(spring, tmp_path):
    """Two instances of a class with its own `_preprocess`, different only in what it reads (a lag), with a plain OLS
    between them: each is a group of its own, so engine on equals engine off for both (with one group they shared the first
    instance's frames, and the second forecast from the first's lag)."""

    def models():
        return [
            Lagged(PREDICTORS, 28, lag=1, name="lag1"),
            OLSModel(PREDICTORS, 28, name="ols"),
            Lagged(PREDICTORS, 28, lag=5, name="lag5"),
        ]

    assert all(model._engine_capable() for model in models())
    assert [[m.name for m in g] for g in _engine.engine_groups(models())] == [["lag1"], ["ols"], ["lag5"]]
    off, on = run_both(build(models), spring, tmp_path, test=("2023-03-24", "2023-03-25"), horizon=1)
    assert same_results(tmp_path / "off", tmp_path / "on") == 6
    same_tables(off, on)
    assert lines(tmp_path / "on" / "lag1.jsonl") != lines(tmp_path / "on" / "lag5.jsonl")


def spring_models():
    return [
        OLSModel(PREDICTORS, 28, name="ols"),
        LassoCVModel([*PREDICTORS, "aux"], 28, cv=3, refit="7D", binary_mask="fit", name="lasso_7d_fit"),
        RidgeModel([*PREDICTORS, "price_d-1"], [14, 28], refit="7D", name="ridge_7d_list"),
        LEARModel([*PREDICTORS, "price_d-1", "aux"], [28, 90], refit="auto", name="lear_auto"),
        OLSModel(by_weekday, 28, binary_mask="fit", name="ols_function_fit"),
        RidgeModel(with_aux, 28, refit="7D", name="ridge_function_7d"),
        NaiveModel(["price_d-1"], name="naive"),
        OLSModel(matching_list, 28, name="matching"),
        OwnFit(PREDICTORS, 28, name="own_fit"),
    ]


def autumn_models():
    names = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "aux"]
    return [
        OLSModel([*names, "price_d-1"], 28, newest_target="published", name="ols_published"),
        RidgeModel(names, [14, 28], binary_mask="fit", name="ridge_list_fit"),
        LassoCVModel(by_weekday, 28, cv=3, refit="7D", name="lasso_function_7d"),
        LEARModel(
            [*names, "price_d-1"], [28, 120], refit="auto", newest_target="published", name="lear_auto_published"
        ),
        OLSModel(names, [14, 120], refit="auto", name="ols_auto"),
        NaiveModel(["price_d-1"], name="naive"),
    ]


def quarter_models():
    return [
        OLSModel([*PREDICTORS, "aux"], [7, 14], binary_mask="fit", name="ols_list_fit"),
        OLSModel(PREDICTORS, 14, refit="7D", name="ols_7d"),
        RidgeModel(PREDICTORS, [7, 14], refit="auto", name="ridge_auto"),
        LEARModel(with_aux, 14, name="lear_function"),
    ]


def daily_models():
    names = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "aux"]
    return [
        OLSModel([*names, "price_d-1"], 56, newest_target="published", binary_mask="fit", name="ols_published_fit"),
        RidgeModel(by_weekday, [42, 56], refit="7D", name="ridge_function_7d"),
        RidgeModel(names, [42, 90], refit="auto", name="ridge_auto"),
        LassoCVModel(names, [42, 56], cv=3, binary_mask="fit", name="lasso_list_fit"),
        LassoCVModel(by_weekday, 56, cv=3, refit="auto", newest_target="published", name="lasso_function_auto"),
        LEARModel([*names, "price_d-1"], [42, 56], name="lear_list"),
        LEARModel([*names, "price_d-1"], 56, refit="7D", name="lear_7d"),
        NaiveModel(["price_d-1"], name="naive"),
    ]


def autumn_validation_models():
    names = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "aux"]
    return [
        RidgeModel(names, [14, 28], binary_mask="fit", name="ridge_list_fit"),
        OLSModel(names, [14, 120], refit="auto", name="ols_auto"),
        LEARModel(by_weekday, [28, 90], refit="auto", name="lear_function_auto"),
        LassoCVModel(with_aux, 28, cv=3, refit="7D", binary_mask="fit", name="lasso_function_7d"),
        NaiveModel(["price_d-1"], name="naive"),
    ]


def build(models, rules=None):
    def pipeline(engine):
        info = None if rules is None else InformationSet("11:45", "Europe/Brussels", rules=rules)
        out = ModelPipeline(information_set=info, availability="none" if info else None, engine=engine)
        for model in models():
            out.add_model(model)
        return out

    return pipeline


# Each cell: the frame, the pipeline, the run's arguments. Pairs of levels across the cells: 1h with the spring switch (plain,
# JSONL) and the autumn one (Fridays, published, Parquet; validation, JSONL); 15 min over the autumn switch with validation,
# Parquet; 1D on Fridays with published, JSONL. Each class with refit None, "7D" and "auto", a window list, a function and
# `binary_mask="fit"` in some cell; the models the engine doesn't take (Naive, a matching list, a fit without `_fit_stats`)
# in the same pipelines.
CELLS = {
    "1h-spring-plain-jsonl": ("spring", spring_models, None, dict(test=("2023-03-24", "2023-03-27"), horizon=2)),
    "1h-autumn-fridays-published-parquet": (
        "autumn",
        autumn_models,
        PUBLISHED,
        dict(test=("2023-10-27", "2023-11-03"), horizons=[1, 3], issue_days=["friday"], save_format="parquet"),
    ),
    "15min-autumn-validation-parquet": (
        "quarter",
        quarter_models,
        None,
        dict(
            validation=("2023-10-26", "2023-10-26"),
            test=("2023-10-28", "2023-10-29"),
            horizon=1,
            freq="15min",
            save_format="parquet",
        ),
    ),
    "1D-fridays-published-jsonl": (
        "daily",
        daily_models,
        PUBLISHED,
        dict(test=("2023-05-05", "2023-06-16"), horizons=[1, 3, 7], freq="1D", issue_days=["friday"]),
    ),
    "1h-autumn-validation-jsonl": (
        "autumn",
        autumn_validation_models,
        None,
        dict(validation=("2023-10-23", "2023-10-24"), test=("2023-10-28", "2023-10-30"), horizon=2),
    ),
}


@pytest.mark.parametrize("cell", list(CELLS))
def test_engine_on_equals_off(cell, request, tmp_path):
    frame, models, rules, arguments = CELLS[cell]
    data = request.getfixturevalue(frame)
    off, on = run_both(build(models, rules), data, tmp_path, **arguments)
    assert same_results(tmp_path / "off", tmp_path / "on") >= 2 * len(models())
    same_tables(off, on)
    if "validation" in arguments:
        same_tables(off.validation, on.validation)
    assert off.forecast_counts()["produced"].gt(0).all()


def test_engine_on_equals_off_in_memory(spring):
    """Without a save_dir: the refs' records, and the tables."""
    off = build(spring_models)("off").run(spring, "2023-03-24", "2023-03-26", horizon=2)
    on = build(spring_models)("on").run(spring, "2023-03-24", "2023-03-26", horizon=2)
    for name in by_name(spring_models()):
        mine = sorted(json.dumps(r, sort_keys=True) for r in on.refs[name]._results)
        assert mine == sorted(json.dumps(r, sort_keys=True) for r in off.refs[name]._results)
    same_tables(off, on)


def test_threads_per_process(spring, tmp_path, monkeypatch):
    """Two task threads per process in the engine's jobs: the records of one thread."""
    monkeypatch.setenv("MAX_PROCESSES", "2")
    pick = lambda: spring_models()[:4]  # noqa: E731
    build(pick)("on").run(spring, "2023-03-24", "2023-03-26", horizon=2, save_dir=str(tmp_path / "one"))
    monkeypatch.setenv("THREADS_PER_PROCESS", "2")
    build(pick)("on").run(spring, "2023-03-24", "2023-03-26", horizon=2, save_dir=str(tmp_path / "two"))
    same_results(tmp_path / "one", tmp_path / "two")


# --- a model alone, in a group, shuffled, chunked; processes ----------------------------------------------------------

POOL = 60  # columns the many-model frame has


@pytest.fixture(scope="module")
def wide():
    """A daily frame with 60 weather-like columns, a few cells missing, for groups of many models."""
    base = make_market_data(start="2023-01-01", days=120, noise=5.0, seed=51, horizon=1).at_time("12:00").copy()
    rng = np.random.default_rng(52)
    columns = {f"w{i:02d}": rng.normal(0, 1, len(base)) + 0.01 * base["load"].to_numpy() for i in range(POOL)}
    wide = pd.concat([base, pd.DataFrame(columns, index=base.index)], axis=1)
    for i in range(0, POOL, 7):
        wide.iloc[rng.integers(0, len(wide), 3), wide.columns.get_loc(f"w{i:02d}")] = np.nan
    return wide


def many_models(n=200, seed=53):
    rng = random.Random(seed)
    models = []
    for i in range(n):
        names = [f"w{j:02d}" for j in sorted(rng.sample(range(POOL), rng.randint(2, 8)))]
        kind = i % 4
        if kind == 0:
            models.append(OLSModel(["load_d+1", *names], rng.choice([21, 28, 35]), name=f"m{i:03d}"))
        elif kind == 1:
            models.append(OLSModel(["load_d+1", *names], [14, 28], binary_mask="fit", name=f"m{i:03d}"))
        elif kind == 2:
            models.append(RidgeModel(["load_d+1", *names], 28, name=f"m{i:03d}"))
        else:
            models.append(OLSModel(["is_weekend_d+1", *names], 21, name=f"m{i:03d}"))
    return models


def run_models(models, data, save_dir, engine="on"):
    pipeline = ModelPipeline(engine=engine)
    for model in models:
        pipeline.add_model(model)
    return pipeline.run(data, "2023-04-10", "2023-04-14", horizon=1, freq="1D", save_dir=str(save_dir))


def files_of(save_dir, name):
    return lines(Path(save_dir) / f"{name}.jsonl"), manifest(Path(save_dir) / f"{name}.manifest.json")


def test_a_model_is_the_same_alone_in_a_group_shuffled_and_chunked(wide, tmp_path, monkeypatch, in_process):
    models = many_models()
    names = [model.name for model in models]
    run_models(models, wide, tmp_path / "group")
    shuffled = many_models()
    random.Random(54).shuffle(shuffled)
    run_models(shuffled, wide, tmp_path / "shuffled")
    monkeypatch.setenv("ENGINE_MAX_MODELS", "7")
    run_models(many_models(), wide, tmp_path / "seven")
    monkeypatch.delenv("ENGINE_MAX_MODELS")
    monkeypatch.setenv("ENGINE_MAX_COLUMNS", "50")
    run_models(many_models(), wide, tmp_path / "columns")
    monkeypatch.delenv("ENGINE_MAX_COLUMNS")
    for other in ("shuffled", "seven", "columns"):
        assert same_results(tmp_path / "group", tmp_path / other) == 2 * len(models)
    for name in random.Random(55).sample(names, 8):
        run_models([m for m in many_models() if m.name == name], wide, tmp_path / "alone" / name)
        assert files_of(tmp_path / "alone" / name, name) == files_of(tmp_path / "group", name)


def test_one_and_three_processes(wide, tmp_path, monkeypatch):
    """Real pools: 1 and 3 processes, and engine off, give the same files."""
    models = lambda: many_models(24)  # noqa: E731
    run_models(models(), wide, tmp_path / "one")
    monkeypatch.setenv("MAX_PROCESSES", "3")
    run_models(models(), wide, tmp_path / "three")
    run_models(models(), wide, tmp_path / "off", engine="off")
    assert same_results(tmp_path / "one", tmp_path / "three") == 48
    assert same_results(tmp_path / "one", tmp_path / "off") == 48


def test_the_union_holds_each_models_rows_and_columns(wide, in_process, monkeypatch):
    """A chunk's arrays have each model's own rows (days, instants, targets) whatever the union, and its columns by name: what a
    span's rows are (the period, horizon and the dates of its rows) is the same in any group."""
    seen = {}
    real = _engine_worker._engine_load

    def spy(hour_arrays, models):
        seen.setdefault("arrays", []).append(hour_arrays)
        return real(hour_arrays, models)

    monkeypatch.setattr(InProcessPool, "__init__", lambda self, *a, initargs=(), **k: spy(*initargs))
    models = many_models(12)
    pipeline = ModelPipeline(engine="on")
    for model in models:
        pipeline.add_model(model)
    pipeline.run(wide, "2023-04-10", "2023-04-11", horizon=1, freq="1D")
    alone = ModelPipeline(engine="on").add_model(many_models(12)[5])
    alone.run(wide, "2023-04-10", "2023-04-11", horizon=1, freq="1D")
    group, own = seen["arrays"]
    for period in own:
        for key in ("days", "timestamps", "y"):
            np.testing.assert_array_equal(group[period][key], own[period][key])
            assert group[period][key].dtype == own[period][key].dtype
        for name, column in own[period]["col_index"].items():
            mine = group[period]["x"][:, group[period]["col_index"][name]]
            assert mine.tobytes() == own[period]["x"][:, column].tobytes()
        assert len(group[period]["col_index"]) > len(own[period]["col_index"])


def test_masks_read_on_the_union_frames_are_the_models_own(spring, monkeypatch):
    """A model's masks read from a group's period frames (the union of columns, of several dtypes) equal those of its own
    frames, its training sample is the rows-then-columns take of before, and a run reads each first-day mask once (the plain
    run's manifest asks for them again)."""
    names = [*PREDICTORS, "aux"]
    frame = spring.assign(flag=(spring["is_weekend"] > 0), count=spring["is_weekend"].astype(int))
    alone = OLSModel(names, 28)
    columns = list(dict.fromkeys(name for hz in (1, 2) for name in alone._expand_predictors(hz)))
    alone._set_freq("1h")
    alone._load_periods(frame, ["price", *columns], 2, "price")
    grouped = OLSModel(names, 28)
    grouped._set_freq("1h")
    periods = grouped._load_periods(frame, ["price", "flag", "count", "wind_d+3", *columns, "load_d+3"], 2, "price")
    grouped._use_periods(periods, frame, ["price", *columns], 2, "price")
    first = (datetime.date(2023, 3, 24) - frame.index[0].date()).days
    for model in (alone, grouped):
        model._offset, model._last_day, model._newest = first, first + 3, None
    for hour in (0, 2, 23):
        for hz in (1, 2):
            expanded = alone._expand_predictors(hz, hour=hour)
            mine = grouped._training_sample(hour, first, hz, expanded)
            rows = worker.training_rows(grouped._hour_days[hour], first, 28, hz)
            assert mine.tobytes() == grouped._hour_data[hour].iloc[rows][expanded].to_numpy(dtype=np.float64).tobytes()
            assert mine.tobytes() == alone._training_sample(hour, first, hz, expanded).tobytes()
    calls = []
    real = StandardScaler.get_scalable_mask
    monkeypatch.setattr(StandardScaler, "get_scalable_mask", staticmethod(lambda x: calls.append(1) or real(x)))
    masks = grouped._precompute_scalable_masks([1, 2])
    reads = len(calls)
    again = grouped._precompute_scalable_masks([1, 2], first, first)
    assert len(calls) == reads  # the first day's masks again, without a read
    own = alone._precompute_scalable_masks([1, 2])
    assert all(np.array_equal(masks[key], own[key]) and np.array_equal(again[key], own[key]) for key in own)
    assert all(not again[key].all() for key in again)  # the 0/1 columns are left unscaled
    grouped._release()
    assert grouped._first_day_masks == {}


def test_the_parent_holds_one_job_and_one_chunk_at_a_time(wide, tmp_path, monkeypatch, in_process):
    """Memory: a job's records leave the parent with its future once collected (not when the chunk's pool ends), and a
    chunk's runs (their tasks, their stores' done keys) are finished and closed before the next chunk starts, so nothing of
    them is alive but the refs."""
    monkeypatch.setenv("ENGINE_MAX_MODELS", "2")
    held, seen = [], {}
    real_collect, real_prepare = BaseModel._collect, BaseModel._prepare_run

    def collect(self, state, results, skipped, sink):
        held.append(sum(1 for o in gc.get_objects() if isinstance(o, Future) and o.done() and not o.cancelled()))
        return real_collect(self, state, results, skipped, sink)

    def prepare(self, *args, **kwargs):
        if self.name == "m002":  # the second chunk's first model
            gc.collect()
            seen["states"] = sum(1 for o in gc.get_objects() if isinstance(o, _RunState))
            for name in ("m000", "m001"):  # the first chunk's stores are closed: their locks are free
                ResultStore(tmp_path / "group" / f"{name}.jsonl", lock=True).close()
        return real_prepare(self, *args, **kwargs)

    monkeypatch.setattr(BaseModel, "_collect", collect)
    monkeypatch.setattr(BaseModel, "_prepare_run", prepare)
    pipeline = ModelPipeline(engine="on")
    for model in many_models(4):
        pipeline.add_model(model)
    report = pipeline.run(wide, "2023-03-20", "2023-04-14", horizon=1, freq="1D", save_dir=str(tmp_path / "group"))
    # A collect per model and job (26 days, one period, two models a chunk). The stand-in pool has every job done at once, so
    # the futures alive at a job's collect are those not collected yet (about 26 down to 1 in each chunk), none collected
    # before: it was about 26 throughout, every collected job's records alive until the chunk's pool ended.
    assert len(held) == 4 * 26
    assert held[0] >= 25 and held[52] >= 25
    assert max(held[48:52]) <= 3 and max(held[100:]) <= 3
    assert seen["states"] == 0
    assert report.forecast_counts()["produced"].between(24, 26).all()  # a few skipped for missing cells


# --- union isolation --------------------------------------------------------------------------------------------------


def test_a_column_one_model_reads_moves_no_other_model(wide, tmp_path, in_process):
    models = many_models(40)
    reads = {}
    for model in models:
        for name in model.predictors:
            reads.setdefault(name, []).append(model.name)
    alone = next(name for name, readers in sorted(reads.items()) if len(readers) == 1 and name.startswith("w"))
    reader = reads[alone][0]
    run_models(many_models(40), wide, tmp_path / "base")
    perturbed = wide.copy()
    perturbed[alone] = perturbed[alone] * 3.0 + 7.0
    run_models(many_models(40), perturbed, tmp_path / "perturbed")
    for model in models:
        a = lines(tmp_path / "base" / f"{model.name}.jsonl")
        b = lines(tmp_path / "perturbed" / f"{model.name}.jsonl")
        assert (a == b) == (model.name != reader), model.name


# --- non-capable models in the same pipeline --------------------------------------------------------------------------


def test_models_the_engine_does_not_take_run_on_their_own(spring, tmp_path, monkeypatch):
    ran_alone = []
    real_run = BaseModel.run

    def spy(self, *args, **kwargs):
        ran_alone.append(self.name)
        return real_run(self, *args, **kwargs)

    monkeypatch.setattr(BaseModel, "run", spy)
    grouped = []
    real_group = _engine.run_group

    def group_spy(models, arguments, columns=None):
        grouped.append([model.name for model in models])
        return real_group(models, arguments, columns)

    monkeypatch.setattr("epftoolbox2.pipelines.model_pipeline.run_group", group_spy)
    pipeline = build(spring_models)("on")
    pipeline.run(spring, "2023-03-24", "2023-03-25", horizon=1, save_dir=str(tmp_path / "on"))
    assert ran_alone == ["naive", "matching", "own_fit"]  # in list order, after the groups
    # LEAR's [28, 90] under "auto" refits its 90-day window on "7D": the block of the 7D models.
    assert grouped == [["ols", "ols_function_fit"], ["lasso_7d_fit", "ridge_7d_list", "lear_auto", "ridge_function_7d"]]
    ran_alone.clear()
    grouped.clear()
    build(spring_models)("on").run(spring, "2023-03-24", "2023-03-25", horizon=1, forecast_only=True)
    assert grouped == [] and len(ran_alone) == len(spring_models())


# --- crash and resume, models_per_task --------------------------------------------------------------------------------

CRASH_DAY = datetime.date(2023, 3, 28)  # the test's fifth day


def crash_models():
    return [
        OLSModel(PREDICTORS, 28, name="ols"),
        Crashing(PREDICTORS, [14, 28], name="crashing"),
        RidgeModel(PREDICTORS, [14, 28], refit="7D", name="ridge"),
        Crashing(PREDICTORS, 28, refit="7D", name="crashing_refit"),
        LEARModel([*PREDICTORS, "price_d-1"], 28, name="lear"),
    ]


@pytest.mark.parametrize("which", ["member", "unit"])
def test_a_crash_mid_group_then_resume_equals_fresh(spring, tmp_path, monkeypatch, which):
    period = ("2023-03-24", "2023-03-30")

    def run(save_dir):
        p = ModelPipeline(engine="on")
        for model in crash_models():
            p.add_model(model)
        return p.run(spring, *period, horizon=2, save_dir=str(save_dir))

    run(tmp_path / "fresh")
    first = spring.index[0].date()
    real_member, real_unit = worker._fit_member, worker._fit_unit

    def member(*args, **kwargs):
        if isinstance(args[11], Crashing) and args[2] == 3 and args[4] == (CRASH_DAY - first).days:
            raise RuntimeError("planted crash in a member fit")
        return real_member(*args, **kwargs)

    def unit(unit_, tasks, inputs=None):
        if isinstance(inputs.model, Crashing) and unit_.period == 3 and unit_.anchor == datetime.date(2023, 3, 27):
            raise RuntimeError("planted crash in a unit fit")
        return real_unit(unit_, tasks, inputs)

    monkeypatch.setattr(
        worker, "_fit_member" if which == "member" else "_fit_unit", member if which == "member" else unit
    )
    name = "crashing" if which == "member" else "crashing_refit"
    with pytest.raises(RuntimeError, match=f"^{name}: forecasting failed for .*period 3: .*planted crash"):
        run(tmp_path / "crashed")
    monkeypatch.setattr(worker, "_fit_member", real_member)
    monkeypatch.setattr(worker, "_fit_unit", real_unit)
    crashed = {p.name: len(lines(p)) for p in (tmp_path / "crashed").glob("*.jsonl")}
    fresh = {p.name: len(lines(p)) for p in (tmp_path / "fresh").glob("*.jsonl")}
    assert crashed[f"{name}.jsonl"] < fresh[f"{name}.jsonl"]
    run(tmp_path / "crashed")
    same_results(tmp_path / "fresh", tmp_path / "crashed")


JOB_LOG = None  # where `counted_job` appends a line per job a worker runs (set before the pool forks)


def counted_job(*args, **kwargs):
    with open(JOB_LOG, "a") as f:
        f.write("job\n")
    return _engine_worker.run_job(*args, **kwargs)


def test_a_failed_write_stops_the_group_early_and_a_resume_equals_fresh(spring, tmp_path, monkeypatch):
    """A store that fails while the parent saves (here the third save of one model) stops the group at once: no job not yet
    started runs (before, leaving the pool's `with` waited for all 120), the error is the store's, and a resume equals the
    fresh run."""
    global JOB_LOG

    def run(save_dir):
        p = ModelPipeline(engine="on")
        for model in (OLSModel(PREDICTORS, 28, name=f"ols{i}") for i in range(3)):
            p.add_model(model)
        return p.run(spring, "2023-03-24", "2023-03-28", horizon=1, save_dir=str(save_dir))

    run(tmp_path / "fresh")
    JOB_LOG = tmp_path / "jobs.txt"
    monkeypatch.setattr(_engine, "run_job", counted_job)
    real_save, saves = ResultStore.save, []

    def failing(self, result):
        if self.path.name == "ols1.jsonl":
            saves.append(1)
            if len(saves) == 3:
                raise OSError("disk full (planted)")
        return real_save(self, result)

    monkeypatch.setattr(ResultStore, "save", failing)
    with pytest.raises(OSError, match="disk full"):
        run(tmp_path / "crashed")
    ran = len(JOB_LOG.read_text().split())
    assert ran <= 8, ran  # the three collected, the queued and the one running, of 5 days x 24 periods
    monkeypatch.setattr(ResultStore, "save", real_save)
    run(tmp_path / "crashed")
    same_results(tmp_path / "fresh", tmp_path / "crashed")


def write_workflow(directory, frame, k=None, index=None):
    csv = frame[["price", *[c for c in frame.columns if c.endswith("_d+1")], "price_d-1"]].tz_convert("UTC")
    csv = csv.reset_index(names="datetime")
    csv["datetime"] = csv["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    csv.to_csv(directory / "frame.csv", index=False)
    models = [
        {
            "class": "OLSModel",
            "params": {
                "predictors": ["load_d+1", "wind_d+1", "is_weekend_d+1"],
                "training_window": w,
                "name": f"ols{w}",
            },
        }
        for w in (14, 21, 28)
    ] + [
        {
            "class": "RidgeModel",
            "params": {
                "predictors": ["load_d+1", "wind_d+1"],
                "training_window": [14, 28],
                "refit": "7D",
                "name": "ridge",
            },
        },
        {
            "class": "LEARModel",
            "params": {
                "predictors": ["load_d+1", "wind_d+1", "is_weekend_d+1", "price_d-1"],
                "training_window": 28,
                "name": "lear",
            },
        },
        {"class": "NaiveModel", "params": {"predictors": ["price_d-1"], "name": "naive"}},
        {
            "class": "LassoCVModel",
            "params": {"predictors": ["load_d+1", "wind_d+1"], "training_window": 28, "cv": 3, "name": "lasso"},
        },
        {
            "class": "OLSModel",
            "params": {
                "predictors": ["load_d+1", "is_weekend_d+1"],
                "training_window": [14, 28],
                "refit": "auto",
                "name": "ols_auto",
            },
        },
    ]
    config = {
        **({"environment": {"model_index": index, "models_per_task": k}} if index is not None else {}),
        "data_pipeline": {
            "start": "2022-12-01",
            "end": "2023-04-30",
            "sources": [{"class": "CsvSource", "params": {"file_path": str(directory / "frame.csv")}}],
            "transformers": [{"class": "TimezoneTransformer", "params": {"target_tz": "Europe/Warsaw"}}],
        },
        "model_pipeline": {
            "test": ["2023-03-24", "2023-03-27"],
            "target": "price",
            "horizon": 1,
            "save_dir": "results",
            "engine": "on",
            "models": models,
            "evaluators": [{"class": "MAEEvaluator", "params": {}}],
        },
    }
    (directory / "experiment.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return directory / "experiment.yaml"


def test_models_per_task_merge_to_one_run(spring, tmp_path):
    one = tmp_path / "one"
    one.mkdir()
    report = Workflow.load(write_workflow(one, spring)).run()
    for k in (1, 7, 8):
        directory = tmp_path / f"k{k}"
        directory.mkdir()
        for index in range(-(-8 // k)):
            Workflow.load(write_workflow(directory, spring, k, index)).run()
        same_results(one / "results", directory / "results")
        merged = Workflow.load(write_workflow(directory, spring)).report()
        pd.testing.assert_frame_equal(merged.summary(), report.summary(), check_exact=True)


# --- verify through the engine ----------------------------------------------------------------------------------------

RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "load_actual": Availability.realised("1h"),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
}


class TrainsOnIssueDayTarget(OLSModel):
    """Its newest training row has the price of the issue day as its target, one day past the model's boundary; its frames
    are its own (`_frame_readers`), so the engine reads them for it alone."""

    def _preprocess(self, data, horizon, target):
        out = super()._preprocess(data, horizon, target)
        out[f"{target}_d+1"] = calendar_shift(out[target], 2)
        return out


def test_verify_runs_each_probe_through_the_engine(spring, monkeypatch):
    groups = []
    real_group = _engine.run_group

    def group_spy(models, arguments, columns=None):
        groups.append([model.name for model in models])
        return real_group(models, arguments, columns)

    monkeypatch.setattr(_engine, "run_group", group_spy)
    pipeline = ModelPipeline(information_set=InformationSet(rules=RULES), engine="on")
    pipeline.add_model(OLSModel([*PREDICTORS, "price_d-1"], 28, name="ols"))
    pipeline.add_model(RidgeModel([*PREDICTORS, "price_d-1"], 28, refit="7D", name="ridge"))
    pipeline.add_model(NaiveModel(["price_d-1"], name="naive"))
    report = pipeline.verify(spring, "2023-03-20", "2023-03-30", days=1)
    report.assert_ok()
    assert set(report.table()["status"]) == {"ok"}
    assert {name for group in groups for name in group} == {"ols", "ridge"} and all(len(g) == 1 for g in groups)
    assert len(groups) == 2 * 4  # each model: probe and control, on the original and the perturbed frame

    off = ModelPipeline(information_set=InformationSet(rules=RULES))
    for model in pipeline.models:
        off.add_model(model)
    assert off.verify(spring, "2023-03-20", "2023-03-30", days=1).table().equals(report.table())


def test_verify_through_the_engine_finds_a_leak(spring):
    frame = spring.assign(load_actual=spring["load"])
    pipeline = ModelPipeline(information_set=InformationSet(rules=RULES), engine="on")
    pipeline.add_model(OLSModel([*PREDICTORS, "load_actual"], 28, name="late"))
    pipeline.add_model(TrainsOnIssueDayTarget(PREDICTORS, 28, name="sneaky"))
    assert pipeline.models[1]._engine_capable()
    report = pipeline.verify(frame, "2023-04-12", "2023-04-12", days=1, positive_control=False)
    table = report.table().set_index("model")
    assert table.loc["late", "status"] == "leak" and table.loc["late", "changed"] == 14
    assert table.loc["sneaky", "status"] == "leak" and table.loc["sneaky", "changed"] == 24
