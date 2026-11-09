"""`LightGBMModel`, `XGBoostModel` and `CatBoostModel` (step 6o): gradient-boosted trees on the refit calendar, pooled over the
periods of a day by default, each package an extra of its own.

Tests that need a package skip without it (`pytest.importorskip`): the `[lightgbm]`, `[xgboost]` and `[catboost]` extras run
them in the `gbm-extra` CI job; the core-only job runs the rest (settings, cards, the pre-flight, saving and loading without the
package). Runs go through an in-process pool unless a test says otherwise: a fit on one OpenMP thread leaves no OpenMP team in
this process, so the forked pools of later tests are safe. XGBoost forms a team even with `n_jobs=1` unless OpenMP is held at
one thread (the models do so; a fit by hand here must too), and a forked child that then runs it on several threads hangs in
GNU OpenMP: `threads=2` runs in a process of its own.
"""

import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from epftoolbox2._extras import MissingExtraError
from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import CatBoostModel, LightGBMModel, OLSModel, ResidualModel, XGBoostModel
from epftoolbox2.models import gbm
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.models.base import _usable_cpu_count
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path
from epftoolbox2.scalers.target import invariant_params
from tests.models import adapter_suite as suite
from tests.synthetic import calendar_shift, make_market_data

INPUTS = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "price_d-1", "period"]
EACH_INPUTS = INPUTS[:-1]  # `period` is a feature of a pooled fit only: a per-period unit's rows are of one period
TEST = ("2023-03-13", "2023-03-26")  # two "7D" blocks, Monday to Sunday; the spring switch is on 26 March (Warsaw)
RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
}
INFO = InformationSet(rules=RULES)
ROOT = Path(__file__).parent.parent.parent
GOLDEN_FILE = ROOT / "tests" / "golden" / "gbm_metrics.json"

# Each family: its class, its package, and few, small trees for the tests.
FAMILIES = {
    "lightgbm": (LightGBMModel, {"n_estimators": 20, "min_child_samples": 5}),
    "xgboost": (XGBoostModel, {"n_estimators": 20}),
    "catboost": (CatBoostModel, {"iterations": 20, "depth": 4}),
}


class InProcessPool:
    """Stands in for the worker pool, as in tests/models/test_residual.py: the fits run in this process."""

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

    def shutdown(self, wait=True, cancel_futures=False):
        pass


@pytest.fixture(autouse=True)
def single_process(monkeypatch, tmp_path):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)
    monkeypatch.chdir(tmp_path)  # a package writing files would write them here, where a test can see them


def forked(monkeypatch):
    """Real worker processes (fork on Linux) from here on in the test."""
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", ProcessPoolExecutor)


@pytest.fixture(scope="module")
def market():
    """1 January to 20 April 2023 in Warsaw, over the spring switch."""
    return make_market_data(start="2023-01-01", days=110, horizon=2, noise=5.0, seed=3)


@pytest.fixture(params=sorted(FAMILIES))
def family(request):
    """A family whose package is installed: (package, factory of models named `name` with the family's small settings)."""
    package = request.param
    pytest.importorskip(package)
    cls, small = FAMILIES[package]

    def make(name="tree", inputs=None, **params):
        inputs = inputs or (EACH_INPUTS if params.get("periods") == "each" else INPUTS)
        return cls(inputs, **{"training_window": 28, **small, **params}, name=name)

    return package, make


def keyed(records):
    return {(r["run_date"], r["hour"], r["horizon"]): r for r in records}


def run(model, data, test=TEST, horizon=1, **kwargs):
    return ModelPipeline().add_model(model).run(data, *test, horizon=horizon, **kwargs)


def records(model, data, test=TEST, horizon=1, **kwargs):
    """The records of a pipeline run of `model`, in memory."""
    return run(model, data, test, horizon, **kwargs).refs[model.name]._results


# ---------------------------------------------------------------------------------------------------------------------------
# Without the packages: settings, cards, the pre-flight, saving and loading
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [LightGBMModel, XGBoostModel, CatBoostModel])
def test_settings_cards_and_what_is_refused(cls):
    model = cls(INPUTS)
    package = model.card.extra
    assert model._unit_kind == "pooled" and cls(EACH_INPUTS, periods="each")._unit_kind == "period"
    assert model.card.runtime == (
        {"lightgbm": "lightgbm>=4.5", "xgboost": "xgboost>=2.1", "catboost": "catboost>=1.2.9"}[package],
    )
    assert model.card.family == package and model.card.corpus == "the run's data" and model.card.commercial_use
    assert model.card.weights is None and not model._engine_capable()
    assert (model.objective, model.periods, model.threads, model.random_state) == ("l1", "pooled", 1, 0)
    assert model._model_kwargs["threads"] == 1 and model._runtime_fields()["threads_per_fit"] == 1
    for key, bad in (
        ("periods", "hourly"),
        ("objective", "huber"),
        ("threads", 0),
        ("threads", 1.5),
        ("threads", True),
        ("random_state", "0"),
    ):
        with pytest.raises(ValueError, match=key):
            cls(INPUTS, **{key: bad})
    with pytest.raises(ValueError, match="periods can't change"):
        model.periods = "each"
    assert cls(INPUTS, threads=np.int64(3)).threads == 3 and type(cls(INPUTS, threads=np.int64(3)).threads) is int
    with pytest.raises(ValueError, match="needs refit"):
        cls(INPUTS, refit=None).run(make_market_data(days=60), "2023-02-20", "2023-02-21", horizon=1)


def test_xgboost_s_device_is_recorded_and_a_gpu_takes_one_process(monkeypatch):
    assert XGBoostModel(INPUTS)._runtime_fields() == {"device": "cpu", "threads_per_fit": 1}
    gpu = XGBoostModel(INPUTS, device="cuda:1")
    assert gpu._runtime_fields() == {"device": "cuda", "threads_per_fit": 1} and gpu._model_kwargs["device"] == "cuda:1"
    for bad in ("gpu", "cuda:", "CPU", None):
        with pytest.raises(ValueError, match="device"):
            XGBoostModel(INPUTS, device=bad)
    monkeypatch.delenv("MAX_PROCESSES")
    assert gpu._executor_config() == (1, 1)
    monkeypatch.setenv("MAX_PROCESSES", "3")
    assert gpu._executor_config() == (1, 3)


@pytest.mark.parametrize("cls", [LightGBMModel, XGBoostModel, CatBoostModel])
def test_a_run_takes_as_many_processes_as_the_cpus_hold_fits_of_its_threads(cls, monkeypatch):
    """Without MAX_PROCESSES, a fit of `threads` threads leaves the CPUs to fewer processes; a ResidualModel's learner stage runs
    with its learner's count."""
    monkeypatch.delenv("MAX_PROCESSES")
    cpus = _usable_cpu_count()
    assert cls(INPUTS)._executor_config() == (1, cpus)
    assert cls(INPUTS, threads=4)._executor_config() == (1, max(1, cpus // 4))
    monkeypatch.setenv("THREADS_PER_PROCESS", "2")
    assert cls(INPUTS, threads=4)._executor_config() == (2, max(1, cpus // 8))
    learner = cls(INPUTS + ["base_fc_d+{horizon}"], threads=4)
    residual = ResidualModel(OLSModel(["load_d+{horizon}"], training_window=21), learner, training_window=28)
    assert residual._executor_config() == learner._executor_config()
    monkeypatch.setenv("MAX_PROCESSES", "5")
    assert cls(INPUTS, threads=4)._executor_config() == (2, 5)


@pytest.mark.parametrize("cls", [LightGBMModel, XGBoostModel, CatBoostModel])
def test_without_its_package_a_model_saves_and_loads_and_a_fit_names_the_extra(cls, market, tmp_path, monkeypatch):
    """The class imports, saves and loads with its package missing (it is imported by the first fit), and a fit called
    directly raises `MissingExtraError` naming the extra; a run's pre-flight refuses before any model's run starts."""
    package = cls(INPUTS).card.extra
    monkeypatch.setitem(sys.modules, package, None)  # `import package` raises ModuleNotFoundError
    model = cls(INPUTS, threads=2, random_state=7, name="tree")
    path = tmp_path / "pipeline.yaml"
    ModelPipeline().add_model(model).save(path)
    (loaded,) = ModelPipeline.load(path).models
    assert type(loaded) is cls and loaded._model_kwargs == model._model_kwargs
    with pytest.raises(MissingExtraError, match=rf"pip install 'epftoolbox2\[{package}\]'"):
        model._fit_model(np.zeros((3, 5)), np.zeros(3))
    suite.check_preflight_without_extra(lambda name, **p: cls(INPUTS, name=name, **p), monkeypatch, market, TEST)


def test_an_import_error_inside_the_package_is_its_own(monkeypatch):
    """Only the package's own absence is a missing extra: an error importing something the package needs goes through."""

    def broken(name, *args, **kwargs):
        raise ModuleNotFoundError("No module named 'narwhals'", name="narwhals")

    monkeypatch.setattr(gbm.importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError, match="narwhals") as error:
        LightGBMModel(INPUTS)._estimator()
    assert not isinstance(error.value, MissingExtraError)


def test_importing_the_package_and_the_tree_classes_imports_no_tree_package():
    code = (
        "import sys\n"
        "import epftoolbox2, epftoolbox2.models\n"
        "from epftoolbox2.models import LightGBMModel, XGBoostModel, CatBoostModel\n"
        "from epftoolbox2.pipelines import ModelPipeline\n"
        "ModelPipeline().add_model(LightGBMModel(['price_d-1'])).add_model(XGBoostModel(['price_d-1'], name='x'))\n"
        "loaded = [m for m in ('lightgbm', 'xgboost', 'catboost') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------------------------------------------------------
# With the packages
# ---------------------------------------------------------------------------------------------------------------------------


def _by_hand(package, objective, **params):
    """The family's estimator as the research's script makes it (`ztree.py`'s `make_model`), from the package itself."""
    if package == "lightgbm":
        import lightgbm

        return lightgbm.LGBMRegressor(
            objective=objective, n_jobs=1, random_state=0, verbose=-1, deterministic=True, force_row_wise=True, **params
        )
    if package == "xgboost":
        import xgboost

        objective = {"l1": "reg:absoluteerror", "l2": "reg:squarederror"}[objective]
        return xgboost.XGBRegressor(
            objective=objective, tree_method="hist", device="cpu", n_jobs=1, random_state=0, **params
        )
    import catboost

    objective = {"l1": "MAE", "l2": "RMSE"}[objective]
    return catboost.CatBoostRegressor(
        loss_function=objective, random_seed=0, verbose=0, allow_writing_files=False, thread_count=1, **params
    )


HAND_PARAMS = {
    "lightgbm": dict(
        n_estimators=20,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=1.0,
    ),
    "xgboost": dict(
        n_estimators=20,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.5,
        min_child_weight=5.0,
        reg_lambda=1.0,
    ),
    "catboost": dict(iterations=20, learning_rate=0.05, depth=4, l2_leaf_reg=5.0),
}


@pytest.mark.parametrize("objective", ["l1", "l2"])
def test_a_pooled_fit_is_the_package_s_on_the_stacked_rows_by_hand(family, objective, market, tmp_path):
    """One pooled unit by hand: the (date, period) rows of its span with finite inputs and target, by date then period, each
    period's target through its own Invariant, the package's estimator at the research's settings fitted on them; a forecast is
    its row's prediction mapped back. To the bit, and no file written (CatBoost writes none)."""
    package, make = family
    model = make(objective=objective)
    made = model.run(market, "2023-03-14", "2023-03-14", horizon=1)._results
    local = pd.DatetimeIndex(market.index.tz_localize(None))
    names = [name.replace("{horizon}", "1") for name in INPUTS]
    target = calendar_shift(market["price"], 1).to_numpy()
    table = market.assign(period=local.hour, date=local.normalize(), target=target)
    span = (table["date"] >= pd.Timestamp("2023-02-12")) & (table["date"] <= pd.Timestamp("2023-03-11"))
    train = table[span].dropna(subset=[*names, "target"]).sort_values(["date", "period"], kind="stable")
    centers = {p: invariant_params(group["target"].to_numpy()) for p, group in train.groupby("period")}
    y = np.array([np.arcsinh((v - centers[p][0]) / centers[p][1]) for v, p in zip(train["target"], train["period"])])
    issue = table[table["date"] == pd.Timestamp("2023-03-14")].groupby("period").tail(1)
    estimator = _by_hand(package, objective, **HAND_PARAMS[package])
    with threadpool_limits(
        limits=1, user_api="openmp"
    ):  # XGBoost leaves an OpenMP team behind otherwise (module docstring)
        fitted = estimator.fit(train[names].to_numpy(), y)
        z = np.asarray(fitted.predict(issue[names].to_numpy()), dtype=np.float64)
    expected = {int(p): centers[p][0] + centers[p][1] * np.sinh(v) for p, v in zip(issue["period"], z)}
    assert {r["hour"]: r["prediction"] for r in made} == expected
    assert list(tmp_path.iterdir()) == []


def test_the_estimator_has_the_research_s_fixed_settings(family):
    """What every fit runs with besides the parameters: LightGBM deterministic and row-wise, XGBoost's histogram method on the
    model's device, CatBoost silent and writing no files; each on `threads` threads and the model's seed."""
    package, make = family
    params = make(threads=3, random_state=11)._estimator().get_params()
    fixed = {
        "lightgbm": {"deterministic": True, "force_row_wise": True, "n_jobs": 3, "random_state": 11, "verbose": -1},
        "xgboost": {"tree_method": "hist", "device": "cpu", "n_jobs": 3, "random_state": 11},
        "catboost": {"allow_writing_files": False, "thread_count": 3, "random_seed": 11, "verbose": 0},
    }[package]
    assert {key: params.get(key) for key in fixed} == fixed


def test_forecasts_are_the_same_in_any_process_count_forked_or_here(family, market, monkeypatch):
    """Pooled and per-period: the forecasts of this process (an in-process pool), of one forked worker and of three are the
    same bits."""
    _, make = family
    for params in ({}, {"periods": "each"}):
        here = keyed(records(make(**params), market))
        with monkeypatch.context() as patch:
            forked(patch)
            for processes in ("1", "3"):
                patch.setenv("MAX_PROCESSES", processes)
                assert keyed(records(make(**params), market)) == here, (params, processes)
        assert len(here) == 14 * 24 - 2  # less the spring target day's 02:00 and the switch day's own 02:00


def test_threads_are_given_to_the_package_recorded_and_compared_on_resume(family, market, tmp_path):
    """A fit on two threads runs, its manifest records the thread count and the package's version, and a resume with another
    thread count, or over results of another version, refuses before any fit. The two-thread run is a process of its own,
    whose workers fork from a parent that ran no OpenMP code on several threads (one that had hangs its child in GNU OpenMP)."""
    package, make = family
    cls, small = FAMILIES[package]
    save = tmp_path / "run"
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
        "from tests.synthetic import make_market_data\n"
        "from epftoolbox2.models import LightGBMModel, XGBoostModel, CatBoostModel\n"
        "from epftoolbox2.pipelines import ModelPipeline\n"
        "market = make_market_data(start='2023-01-01', days=110, horizon=2, noise=5.0, seed=3)\n"
        f"model = {cls.__name__}({INPUTS!r}, training_window=28, threads=2, name='tree', **{small!r})\n"
        f"ModelPipeline().add_model(model).run(market, *{TEST!r}, horizon=1, save_dir={str(save)!r})\n"
    )
    env = {**os.environ, "MAX_PROCESSES": "2", "THREADS_PER_PROCESS": "1"}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    path = manifest_path(save / "tree.jsonl")
    manifest = json.loads(path.read_text())
    assert manifest["runtime"]["threads_per_fit"] == 2 and manifest["model_params"]["threads"] == 2
    assert manifest["runtime"]["packages"] == {package: importlib.metadata.version(package)}
    with pytest.raises(ManifestMismatchError, match="runtime.threads_per_fit"):
        run(make(threads=1), market, save_dir=save)
    manifest["runtime"]["packages"][package] = "0.1.0"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ManifestMismatchError, match=f"runtime.packages.{package}"):
        run(make(threads=2), market, save_dir=save)


def test_xgboost_s_device_is_compared_on_resume(market, tmp_path):
    pytest.importorskip("xgboost")
    make = lambda: XGBoostModel(INPUTS, training_window=28, n_estimators=10, name="tree")  # noqa: E731
    run(make(), market, save_dir=tmp_path)
    path = manifest_path(tmp_path / "tree.jsonl")
    manifest = json.loads(path.read_text())
    assert manifest["runtime"]["device"] == "cpu"
    manifest["runtime"]["device"] = "cuda"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ManifestMismatchError, match="runtime.device"):
        run(make(), market, save_dir=tmp_path)


@pytest.mark.filterwarnings("ignore:overflow encountered in sinh")  # the control's noise, mapped back by sinh
def test_the_adapter_suite(family, market, tmp_path, monkeypatch):
    _, make = family

    def named(name, **params):
        return make(name, **params)

    suite.check_yaml_round_trip(named, market, tmp_path, TEST, horizon=1)
    suite.check_loop_and_shards(named, market, tmp_path / "loop", TEST, horizon=1, variants=({}, {"periods": "each"}))
    suite.check_resume(named, market, tmp_path / "resume", TEST, horizon=1)
    suite.check_engine(named, market, TEST, horizon=1)
    suite.check_provenance(named, market, tmp_path / "provenance", TEST)
    # More trees than the other checks, so that the control's noise moves every forecast out of its leaf.
    more = {
        "lightgbm": {"n_estimators": 60},
        "xgboost": {"n_estimators": 60},
        "catboost": {"iterations": 200, "depth": 6},
    }
    suite.check_verify(lambda name: named(name, **more[family[0]]), market, INFO, ["2023-03-20", "2023-03-26"], items=6)


def test_the_adapter_suite_s_calendar_counts(family):
    """23 forecasts for the spring target day, 24 for the autumn one with the repeated hour's actual NaN."""
    _, make = family
    both = pd.concat(
        [
            make_market_data(start="2023-01-15", days=75, horizon=1, noise=5.0, seed=3),
            make_market_data(start="2023-08-25", days=70, horizon=1, noise=5.0, seed=3),
        ]
    )
    suite.check_dst_counts(lambda name, **p: make(name, **p), both, "2023-03-26", "2023-10-29")


def test_as_a_residual_learner(family, market, tmp_path):
    """A tree learns an OLS base's residuals: the residual model's forecasts are the base's plus the tree's, its manifest and
    provenance show the learner's card and package, and it resumes to a fresh run's bits."""
    package, make = family

    def residual(name, **params):
        learner = make(inputs=INPUTS + ["base_fc_d+{horizon}"])
        base = OLSModel(
            ["load_d+{horizon}", "wind_d+{horizon}", "price_d-1"], training_window=21, binary_mask="fit", name="ols"
        )
        return ResidualModel(base, learner, training_window=28, name=name, **params)

    run(residual("hybrid"), market, horizon=2, save_dir=tmp_path / "run")
    lines = (tmp_path / "run" / "hybrid.jsonl").read_text().splitlines()
    made = [json.loads(line) for line in lines if json.loads(line)["status"] == "ok"]
    assert made and all(r["prediction"] - r["base_prediction"] != 0 for r in made)
    manifest = json.loads(manifest_path(tmp_path / "run" / "hybrid.jsonl").read_text())
    assert manifest["runtime"]["packages"] == {package: importlib.metadata.version(package)}
    suite.check_provenance(residual, market, tmp_path / "provenance", TEST, horizon=2)
    suite.check_resume(residual, market, tmp_path / "resume", TEST, horizon=2)


# ---------------------------------------------------------------------------------------------------------------------------
# The golden numbers
# ---------------------------------------------------------------------------------------------------------------------------

# Each family's numbers move between its package's releases, and the 3.15 and newest-dependencies CI cells resolve newer ones
# than the lock's: a model's numbers are compared on the release the file was written with (`version` in its entry), and a
# model whose package isn't installed (catboost on 3.14t) is skipped.
GOLDEN_RTOL = 1e-9


def golden_models():
    """name -> (package, model): the default pooled fit of each family and LightGBM per period with squared error."""
    return {
        "LightGBM": ("lightgbm", LightGBMModel(INPUTS, training_window=28, n_estimators=40, min_child_samples=10)),
        "LightGBM_each_l2": (
            "lightgbm",
            LightGBMModel(EACH_INPUTS, training_window=28, periods="each", objective="l2", n_estimators=40),
        ),
        "XGBoost": ("xgboost", XGBoostModel(INPUTS, training_window=28, n_estimators=40)),
        "CatBoost": ("catboost", CatBoostModel(INPUTS, training_window=28, iterations=40, depth=6)),
    }


def compute_gbm_metrics(names) -> dict:
    """The golden test's data (`make_market_data(noise=5.0, seed=42)`, over the spring switch) and two weeks to 27 March at
    horizon 2: MAE, RMSE, MAE per horizon and the forecast count of each model in `names`."""
    df = make_market_data(noise=5.0, seed=42)
    pipeline = ModelPipeline().add_evaluator(MAEEvaluator()).add_evaluator(RMSEEvaluator())
    for name, (_, model) in golden_models().items():
        if name in names:
            model.name = name
            pipeline.add_model(model)
    report = pipeline.run(df, test_start="2023-03-14", test_end="2023-03-27", target="price", horizon=2)
    tables = report.compute_all()
    metrics = {name: {"forecasts": len(details)} for name, details in report.iter_details()}
    for row in tables["summary"].to_dict("records"):
        metrics[row["model"]].update({k: float(row[k]) for k in ("MAE", "RMSE")})
    for row in tables["horizon"].to_dict("records"):
        metrics[row["model"]][f"MAE_h{row['horizon']}"] = float(row["MAE"])
    return metrics


def _version(package):
    return importlib.metadata.version(package) if importlib.util.find_spec(package) is not None else None


def test_gbm_metrics_match_golden():
    models = golden_models()
    if os.environ.get("UPDATE_GOLDEN") == "1":
        missing = sorted({package for package, _ in models.values() if _version(package) is None})
        assert not missing, f"the golden file holds every family: install {missing}"
        actual = compute_gbm_metrics(set(models))
        written = {
            name: {**values, "package": models[name][0], "version": _version(models[name][0])}
            for name, values in actual.items()
        }
        GOLDEN_FILE.write_text(json.dumps(written, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"golden values written to {GOLDEN_FILE}")
    expected = json.loads(GOLDEN_FILE.read_text())
    assert set(expected) == set(models)
    skipped = {}
    for name, values in expected.items():
        installed = _version(values["package"])
        if installed is None:
            skipped[name] = f"{values['package']} isn't installed"
        elif installed != values["version"]:
            skipped[name] = f"{values['package']} {installed} here, {values['version']} in the file"
    compared = set(expected) - set(skipped)
    if compared:
        actual = compute_gbm_metrics(compared)
        for name in compared:
            values = {k: v for k, v in expected[name].items() if k not in ("package", "version")}
            assert actual[name]["forecasts"] == values.pop("forecasts"), f"{name}: number of forecasts changed"
            for key, value in values.items():
                assert actual[name][key] == pytest.approx(value, rel=GOLDEN_RTOL), f"{name} {key}"
    if skipped:
        pytest.skip(f"not compared (each family's numbers move between its releases): {skipped}")


def test_the_gbm_golden_is_of_the_lock_s_versions():
    """Each family's release in the golden file is the one `uv.lock` pins for Python 3.12 and later (the `gbm-extra` job's),
    so the locked cells compare it. Moving the lock's package means writing the file again."""
    lock = (ROOT / "uv.lock").read_text()
    for name, values in json.loads(GOLDEN_FILE.read_text()).items():
        found = re.findall(rf'\[\[package\]\]\nname = "{values["package"]}"\nversion = "([^"]+)"', lock)
        assert found, f"no {values['package']} in uv.lock"
        newest = max(found, key=lambda text: tuple(int(part) for part in text.split(".")))
        assert values["version"] == newest, (name, values["version"], found)
