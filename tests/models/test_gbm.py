"""`LightGBMModel`, `XGBoostModel` and `CatBoostModel` (step 6o): gradient-boosted trees on the refit calendar, pooled over the
periods of a day by default, each package an extra of its own.

Tests that need a package skip without it (`pytest.importorskip`): the `[lightgbm]`, `[xgboost]` and `[catboost]` extras run
them in the `gbm-extra` CI job; the core-only job runs the rest (settings, cards, the pre-flight, saving and loading without the
package). Runs go through an in-process pool unless a test says otherwise: a fit on one OpenMP thread leaves no OpenMP team in
this process, so the forked pools of later tests are safe. XGBoost forms a team even with `n_jobs=1` unless OpenMP is held at
one thread (the models do so; a fit by hand here must too), and a forked child that then runs it on several threads hangs in
GNU OpenMP, which is why `threads > 1` is refused where pools fork: the run on two threads that a spawned pool allows goes in a
process of its own.
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
from epftoolbox2.models import _cards, _worker, gbm
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
    monkeypatch.setattr(gbm, "_gpu_name", lambda device, visible: "a GPU")
    gpu = XGBoostModel(INPUTS, device="cuda:1")
    assert gpu._runtime_fields() == {"device": "cuda", "threads_per_fit": 1, "device_name": "a GPU"}
    assert gpu._model_kwargs["device"] == "cuda:1"
    for bad in ("gpu", "cuda:", "CPU", None):
        with pytest.raises(ValueError, match="device"):
            XGBoostModel(INPUTS, device=bad)
    monkeypatch.delenv("MAX_PROCESSES")
    assert gpu._executor_config() == (1, 1)
    monkeypatch.setenv("MAX_PROCESSES", "3")
    assert gpu._executor_config() == (1, 3)


@pytest.mark.parametrize("cls", [LightGBMModel, XGBoostModel, CatBoostModel])
def test_a_run_takes_as_many_processes_as_the_cpus_hold_fits_of_its_threads(cls, monkeypatch):
    """Without MAX_PROCESSES, a fit of `threads` threads leaves the CPUs to fewer processes where workers are spawned (the only
    place it may have more than one); a ResidualModel's learner stage runs with its learner's count."""
    monkeypatch.delenv("MAX_PROCESSES")
    monkeypatch.setattr(_worker, "_POOLS_SPAWN", True)
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
def test_more_than_one_thread_is_refused_where_the_pool_forks_before_any_fit(cls, market, monkeypatch):
    """Where the library forks its workers (Linux, macOS; `MAX_PROCESSES=1` too) a fit on several OpenMP threads can hang, so
    `threads > 1` is refused before any fit: in the pre-flight of a pipeline, before a model's own pool starts, and before a
    ResidualModel's base runs. One thread is never refused, and a spawned pool takes more."""
    monkeypatch.setattr(_worker, "_POOLS_SPAWN", False)
    fits, bases = [], []
    monkeypatch.setattr(gbm._BoostedTrees, "_fit_model", lambda self, x, y: fits.append(1))
    monkeypatch.setattr(ResidualModel, "_run_base", staticmethod(lambda *a, **k: bases.append(1) or ({}, {})))
    model = cls(INPUTS, training_window=28, threads=2, name="tree")
    message = r"threads=2 can't run here.*MAX_PROCESSES"
    with pytest.raises(ValueError, match=message):
        model._executor_config()
    with pytest.raises(ValueError, match=message):
        run(model, market)
    with pytest.raises(ValueError, match=message):
        model.run(market, *TEST, horizon=1)
    learner = cls(INPUTS + ["base_fc_d+{horizon}"], threads=2)
    hybrid = ResidualModel(OLSModel(["load_d+{horizon}"], training_window=21), learner, training_window=28)
    with pytest.raises(ValueError, match=message):
        run(hybrid, market)
    with pytest.raises(ValueError, match=message):
        hybrid.run(market, *TEST, horizon=1)
    assert fits == [] and bases == []
    cls(INPUTS)._preflight({})  # one thread: never refused
    cls(INPUTS)._executor_config()
    monkeypatch.setattr(_worker, "_POOLS_SPAWN", True)
    model._executor_config()


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
        model._fit_model(np.arange(15.0).reshape(3, 5), np.arange(3.0))
    suite.check_preflight_without_extra(lambda name, **p: cls(INPUTS, name=name, **p), monkeypatch, market, TEST)


@pytest.mark.parametrize("cls", [LightGBMModel, XGBoostModel, CatBoostModel])
def test_every_setting_reaches_the_workers(cls):
    """A worker rebuilds the model from `_model_kwargs`: every parameter of the class but the ones BaseModel's run reads in the
    parent (the list, the window, the refit rule, the name and the keyword-only ones) must be in it, or a worker would fit with
    its default."""
    import inspect

    parent_side = {"predictors", "training_window", "refit", "name", "scaler", "newest_target", "binary_mask"}
    parent_side |= {"normalise_by", "bounds", "zero_when"}  # a per-row target: the worker reads them from its config
    parent_side |= {"target_transform"}  # a pooled unit's: the worker reads it from its config (`config["units"]`)
    assert set(cls(INPUTS)._model_kwargs) == set(inspect.signature(cls).parameters) - parent_side


@pytest.mark.parametrize("cls", [LightGBMModel, XGBoostModel, CatBoostModel])
def test_target_transform_is_a_pooled_fit_s_and_saved_only_when_set(cls, tmp_path):
    """Without the package: a pooled fit takes "per_period", "pooled" and "none", `periods="each"` the default alone (when
    built and when set later), another value raises; a default model's YAML has no `target_transform`, a set one's has it after
    `periods` and loads back to it."""
    for value in ("pooled", "none"):
        with pytest.raises(ValueError, match="with periods='each' each period's target goes through scaler="):
            cls(EACH_INPUTS, periods="each", target_transform=value)
        each = cls(EACH_INPUTS, periods="each")
        with pytest.raises(ValueError, match="with periods='each'"):
            each.target_transform = value
    with pytest.raises(ValueError, match="target_transform must be one of"):
        cls(INPUTS, target_transform="asinh")
    texts = {}
    for value in ("per_period", "none"):
        path = tmp_path / f"{value}.yaml"
        ModelPipeline().add_model(cls(INPUTS, target_transform=value, name="tree")).save(path)
        texts[value] = path.read_text()
        (loaded,) = ModelPipeline.load(path).models
        assert type(loaded) is cls and loaded.target_transform == value
    assert "target_transform" not in texts["per_period"]
    assert texts["none"] == texts["per_period"].replace(
        "    periods: pooled\n", "    periods: pooled\n    target_transform: none\n"
    )


@pytest.mark.parametrize("cls", [LightGBMModel, XGBoostModel, CatBoostModel])
def test_every_fit_and_prediction_runs_inside_the_thread_scope(cls, monkeypatch):
    """`_fit_model`, `_predict_block` and `_predict_fitted` call the estimator inside the scope of the model's `threads`, and the
    estimator is built before it (the scope limits only the OpenMP runtimes already loaded)."""
    events = []

    class Scope:
        def __init__(self, threads):
            self.threads = threads

        def __enter__(self):
            events.append(("enter", self.threads))

        def __exit__(self, *exc_info):
            events.append(("exit", self.threads))

    class Estimator:
        def fit(self, x, y):
            events.append(("fit", len(x)))
            return self

        def predict(self, rows):
            events.append(("predict", len(rows)))
            return np.zeros(len(rows), dtype=np.float32)

    model = cls(INPUTS, threads=3)
    monkeypatch.setattr(gbm, "_thread_scope", Scope)
    monkeypatch.setattr(model, "_estimator", lambda: events.append(("build", None)) or Estimator())
    fitted = model._fit_model(np.arange(20.0).reshape(4, 5), np.arange(4.0))
    predictions, coefficients = model._predict_block(fitted, np.zeros((2, 5)))
    assert predictions.dtype == np.float64 and coefficients == [[], []]
    assert model._predict_fitted(fitted, np.zeros(5)) == (0.0, [])
    assert events == [
        ("build", None),
        ("enter", 3),
        ("fit", 4),
        ("exit", 3),
        ("enter", 3),
        ("predict", 2),
        ("exit", 3),
        ("enter", 3),
        ("predict", 1),
        ("exit", 3),
    ]


class FakeDistribution:
    """What `importlib.metadata.distribution` gives for an installed distribution, as far as the cards read it."""

    def __init__(self, version):
        self.version = version

    def read_text(self, name):
        return None


def installed_as(monkeypatch, fakes):
    """`importlib.metadata.distribution` with the distributions of `fakes` (name -> version, None for not installed) in place of
    the real ones; every other name as it is."""
    real = importlib.metadata.distribution

    def distribution(name):
        if name in fakes:
            if fakes[name] is None:
                raise importlib.metadata.PackageNotFoundError(name)
            return FakeDistribution(fakes[name])
        return real(name)

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)


def test_xgboost_cpu_stands_in_for_xgboost_and_is_recorded_as_such(monkeypatch):
    """`xgboost-cpu` (XGBoost's CPU-only build, `RUNTIME_ALIASES`) installs the `xgboost` module under another name: the pin
    `xgboost>=2.1` takes its version and the pre-flight passes, and the manifest records it under `xgboost` with the build named,
    so a resume over the full build's results refuses (the builds' forecasts differ)."""
    installed_as(monkeypatch, {"xgboost": None, "xgboost-cpu": "3.4.1"})
    monkeypatch.setattr(_cards, "has_extra", lambda name: True)
    assert _cards.runtime_providers("xgboost") == ["xgboost-cpu"]
    assert _cards.installed_distribution("xgboost") == ("3.4.1", None)
    assert _cards.runtime_versions(gbm.XGBOOST_CARD) == {"xgboost": "3.4.1 (xgboost-cpu)"}
    assert not [problem for problem in XGBoostModel(INPUTS)._preflight({}) if problem.kind == "runtime"]
    installed_as(monkeypatch, {"xgboost": "3.4.1", "xgboost-cpu": None})
    assert _cards.runtime_versions(gbm.XGBOOST_CARD) == {"xgboost": "3.4.1"}


def test_both_xgboost_builds_installed_refuse_before_any_fit(monkeypatch):
    """With the full build and `xgboost-cpu` installed together, one overwrote the other's module and which one runs is unknown:
    the pre-flight refuses."""
    installed_as(monkeypatch, {"xgboost": "3.4.1", "xgboost-cpu": "3.4.1"})
    monkeypatch.setattr(_cards, "has_extra", lambda name: True)
    (problem,) = [problem for problem in XGBoostModel(INPUTS)._preflight({}) if problem.kind == "runtime"]
    assert (
        "xgboost and xgboost-cpu are both installed" in problem.message and "uninstall all but one" in problem.message
    )
    with pytest.raises(MissingExtraError, match="xgboost and xgboost-cpu are both installed"):
        _cards.preflight([XGBoostModel(INPUTS)])
    assert _cards.runtime_versions(gbm.XGBOOST_CARD) == {"xgboost": "3.4.1 (xgboost + xgboost-cpu)"}


def test_a_look_alike_module_of_another_distribution_stands_in_for_nothing(monkeypatch):
    """Only the alias table names stand-ins: a distribution that provides a module of a pin's name (a look-alike `lightgbm` or
    `torch`) is never looked up, so a runtime that isn't installed stays not installed."""
    installed_as(monkeypatch, {"lightgbm": None, "torch": None, "lightgbm-lookalike": "4.7.0", "not-torch": "2.9.0"})
    monkeypatch.setattr(
        importlib.metadata,
        "packages_distributions",
        lambda: {"lightgbm": ["lightgbm-lookalike"], "torch": ["not-torch"]},
    )
    monkeypatch.setattr(_cards, "has_extra", lambda name: True)
    assert _cards.installed_distribution("lightgbm") is None and _cards.installed_distribution("torch") is None
    assert _cards.runtime_versions(gbm.LIGHTGBM_CARD) == {"lightgbm": None}
    (problem,) = LightGBMModel(INPUTS)._preflight({})
    assert problem.kind == "runtime" and "which is not installed" in problem.message


def test_a_resume_across_xgboost_builds_refuses(market, tmp_path, monkeypatch):
    """Results made with the full XGBoost build refuse a resume with `xgboost-cpu` (and the other way round), naming the
    runtime entry, before any fit."""
    pytest.importorskip("xgboost")
    make = lambda: XGBoostModel(INPUTS, training_window=28, n_estimators=10, name="tree")  # noqa: E731
    run(make(), market, save_dir=tmp_path)
    version = importlib.metadata.version("xgboost")
    installed_as(monkeypatch, {"xgboost": None, "xgboost-cpu": version})
    monkeypatch.setattr(gbm._BoostedTrees, "_fit_model", lambda *a: pytest.fail("a fit ran"))
    with pytest.raises(ManifestMismatchError) as refused:
        run(make(), market, save_dir=tmp_path)
    assert f"runtime.packages.xgboost: '{version}' (existing results) vs '{version} (xgboost-cpu)' (this run)" in str(
        refused.value
    )


@pytest.mark.parametrize("free_threaded, version", [(True, (3, 14)), (False, (3, 15)), (False, (3, 12))])
def test_catboost_s_pre_flight_says_where_it_has_no_wheels(free_threaded, version, monkeypatch):
    """Without catboost on Python 3.15 or a free-threaded build the pre-flight says that it has no wheels there (pip would try
    to build it from source); elsewhere it names the extra to install."""
    monkeypatch.setattr(gbm, "_free_threaded_build", lambda: free_threaded)
    monkeypatch.setattr(gbm.sys, "version_info", version)
    monkeypatch.setattr(_cards, "has_extra", lambda name: name != "catboost")
    (problem,) = CatBoostModel(INPUTS)._preflight({})
    if free_threaded or version >= (3, 15):
        assert "no wheels for Python 3.15 or free-threaded builds" in problem.message
        assert "Python 3.11-3.14 build with the GIL" in problem.message
    else:
        assert problem.message.endswith("pip install 'epftoolbox2[catboost]'")
    assert problem.extra == "catboost"


def test_the_gpu_s_name_comes_from_nvidia_smi_through_the_visible_devices(monkeypatch, request):
    """`runtime.device_name` of an XGBoost GPU run: nvidia-smi's name of the device, its index mapped through
    CUDA_VISIBLE_DEVICES; "cuda" where it can't be read. A CPU run records none."""
    asked = []

    def fake_run(command, **kwargs):
        asked.append(command[-1])
        return subprocess.CompletedProcess(command, 0, stdout="NVIDIA H100 80GB HBM3\n", stderr="")

    real = gbm._gpu_name
    real.cache_clear()
    request.addfinalizer(real.cache_clear)  # no fake name stays in the cache
    monkeypatch.setattr(gbm.subprocess, "run", fake_run)
    assert gbm._gpu_name("cuda", None) == "NVIDIA H100 80GB HBM3" and asked == ["0"]
    assert gbm._gpu_name("cuda:1", "3,GPU-abc") == "NVIDIA H100 80GB HBM3" and asked[-1] == "GPU-abc"
    assert gbm._gpu_name("cuda:2", "3,5") == "cuda"  # not among the visible devices
    monkeypatch.setattr(gbm.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi")))
    assert gbm._gpu_name("cuda:3", None) == "cuda"
    monkeypatch.setattr(gbm, "_gpu_name", lambda device, visible: f"name of {device}")
    assert XGBoostModel(INPUTS, device="cuda:1")._runtime_fields()["device_name"] == "name of cuda:1"
    assert "device_name" not in XGBoostModel(INPUTS)._runtime_fields()


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


def test_a_pooled_fit_with_target_transform_none_is_the_package_s_on_the_raw_stacked_rows_by_hand(family, market):
    """`target_transform="none"`: the same stacked rows, the package's estimator fitted on their raw targets (no centre, no
    scale, no asinh), a forecast its row's prediction as it is. To the bit; every forecast moves against the default's."""
    package, make = family
    made = {r["hour"]: r["prediction"] for r in records(make(target_transform="none"), market, ("2023-03-14",) * 2)}
    local = pd.DatetimeIndex(market.index.tz_localize(None))
    names = [name.replace("{horizon}", "1") for name in INPUTS]
    target = calendar_shift(market["price"], 1).to_numpy()
    table = market.assign(period=local.hour, date=local.normalize(), target=target)
    span = (table["date"] >= pd.Timestamp("2023-02-12")) & (table["date"] <= pd.Timestamp("2023-03-11"))
    train = table[span].dropna(subset=[*names, "target"]).sort_values(["date", "period"], kind="stable")
    issue = table[table["date"] == pd.Timestamp("2023-03-14")].groupby("period").tail(1)
    estimator = _by_hand(package, "l1", **HAND_PARAMS[package])
    with threadpool_limits(limits=1, user_api="openmp"):  # XGBoost leaves an OpenMP team behind otherwise
        fitted = estimator.fit(train[names].to_numpy(), train["target"].to_numpy(dtype=np.float64))
        z = np.asarray(fitted.predict(issue[names].to_numpy()), dtype=np.float64)
    expected = {int(p): float(v) for p, v in zip(issue["period"], z)}
    assert made == expected and len(made) == 24
    default = records(make(), market, ("2023-03-14",) * 2)
    assert all(r["prediction"] != made[r["hour"]] for r in default)


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


def test_one_row_at_a_time_gives_the_block_call_s_forecasts(family, market):
    """`block_predict=False` (and so `_predict_fitted`) predicts each forecast's row on its own: the same bits as the block call,
    a tree's prediction of a row reading that row alone."""
    _, make = family
    blocks = keyed(records(make(periods="each"), market))
    single = make(periods="each")
    single.block_predict = False
    assert keyed(records(single, market)) == blocks


def test_two_threads_where_workers_are_spawned_give_one_thread_s_bits(family):
    """Where workers are spawned (`_POOLS_SPAWN`, Windows) `threads=2` runs: the same forecasts as one thread, bit for bit. In a
    process of its own (an in-process pool there), since a fit on two OpenMP threads in this one would put the forked pools of
    later tests at risk."""
    package, _ = family
    cls, small = FAMILIES[package]
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
        "from concurrent.futures import Future\n"
        "import epftoolbox2.models.base as base\n"
        "from epftoolbox2.models import _worker, LightGBMModel, XGBoostModel, CatBoostModel\n"
        "from epftoolbox2.pipelines import ModelPipeline\n"
        "from tests.synthetic import make_market_data\n"
        "class Pool:\n"
        "    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):\n"
        "        _worker._worker_init(*initargs)\n"
        "    def __enter__(self): return self\n"
        "    def __exit__(self, *a): return False\n"
        "    def submit(self, task, *args):\n"
        "        f = Future(); f.set_result(task(*args)); return f\n"
        "    def shutdown(self, wait=True, cancel_futures=False): pass\n"
        "base.ProcessPoolExecutor = Pool\n"
        "_worker._POOLS_SPAWN = True\n"
        "market = make_market_data(start='2023-01-01', days=110, horizon=2, noise=5.0, seed=3)\n"
        "out = []\n"
        "for threads in (1, 2):\n"
        f"    model = {cls.__name__}({INPUTS!r}, training_window=28, threads=threads, name='tree', **{small!r})\n"
        f"    ref = ModelPipeline().add_model(model).run(market, *{TEST!r}, horizon=1).refs['tree']\n"
        "    out.append(sorted((r['run_date'], r['hour'], r['prediction']) for r in ref._results))\n"
        "assert out[0] and out[0] == out[1], 'two threads moved a forecast'\n"
    )
    env = {**os.environ, "MAX_PROCESSES": "1", "THREADS_PER_PROCESS": "1"}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, result.stderr[-3000:]


def test_a_pooled_fit_takes_a_per_row_target(family, market):
    """`normalise_by`, `bounds` and `zero_when` on a pooled tree: each row's target divided by its own row's scale, a forecast
    the run on y / c times its issue row's c, clipped in multiples of c, and 0 with no fit where the issue row's flag is 1; to
    the bit (step 7c's per-row target, as `HistGBModel` takes it)."""
    from tests.models.test_row_targets import SCALE, divided, issue_values, times_scale, with_columns

    _, make = family
    data = with_columns(market)
    on_ratio = keyed(records(make(), divided(data)))
    expected = times_scale(data, on_ratio)
    bounds = (1.5, 2.2)
    both = keyed(records(make(normalise_by=SCALE, bounds=bounds, zero_when="flag"), data))
    flags, scales = issue_values(data, on_ratio, "flag"), issue_values(data, on_ratio, SCALE)
    assert both.keys() == on_ratio.keys() and 0 < sum(flags.values()) < len(flags)
    clipped = 0
    for key, record in both.items():
        if flags[key] == 1.0:
            assert (record["prediction"], record["coefficients"], record["detail"]) == (0.0, [], "zero_when"), key
        else:
            c = scales[key]
            assert record["prediction"] == min(max(expected[key], bounds[0] * c), bounds[1] * c), key
            clipped += record["prediction"] != expected[key]
    assert clipped > 0


def test_threads_and_the_version_are_recorded_and_compared_on_resume(family, market, tmp_path, monkeypatch):
    """The manifest records the thread count and the package's version (`runtime`; `threads` is in `model_params` too), and a
    resume with another thread count, or over results of another version, refuses before any fit."""
    package, make = family
    save = tmp_path / "run"
    run(make(), market, save_dir=save)
    path = manifest_path(save / "tree.jsonl")
    manifest = json.loads(path.read_text())
    assert manifest["runtime"]["threads_per_fit"] == 1 and manifest["model_params"]["threads"] == 1
    assert manifest["runtime"]["packages"] == {package: importlib.metadata.version(package)}
    monkeypatch.setattr(_worker, "_POOLS_SPAWN", True)  # two threads pass the fork check: the manifest refuses them
    monkeypatch.setattr(gbm._BoostedTrees, "_fit_model", lambda *a: pytest.fail("a fit ran"))
    with pytest.raises(ManifestMismatchError) as refused:
        run(make(threads=2), market, save_dir=save)
    assert "runtime.threads_per_fit: 1 (existing results) vs 2 (this run)" in str(refused.value)
    assert "'threads': 1" in str(refused.value) and "'threads': 2" in str(refused.value)  # model_params, named too
    manifest["runtime"]["packages"][package] = "0.1.0"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ManifestMismatchError, match=f"runtime.packages.{package}"):
        run(make(), market, save_dir=save)


def test_package_errors_stop_the_run_and_data_errors_stay_value_errors(family, market, monkeypatch):
    """A package's own error (XGBoostError is a ValueError) is a RuntimeError, in the fit and the predictions, so a run stops
    instead of skipping a window's forecasts as `fit_error`; any other ValueError (of the data) goes through as it is. Real
    settings the packages refuse stop a pipeline run."""
    package, make = family
    model = make()
    error = model._error_types()[0]

    class Failing:
        def __init__(self, raised):
            self.raised = raised

        def fit(self, x, y):
            raise self.raised

        def predict(self, rows):
            raise self.raised

    for raised in (error("device out of memory"), ValueError("the data")):
        monkeypatch.setattr(model, "_estimator", lambda raised=raised: Failing(raised))
        expected = RuntimeError if isinstance(raised, error) else ValueError
        with pytest.raises(expected, match="device out of memory" if expected is RuntimeError else "the data"):
            model._fit_model(np.arange(15.0).reshape(3, 5), np.arange(3.0))
        with pytest.raises(expected):
            model._predict_block(Failing(raised), np.zeros((2, 5)))
    bad = {"lightgbm": {"num_leaves": 1}, "xgboost": {"subsample": 1.5}, "catboost": {"depth": 99}}[package]
    with pytest.raises(RuntimeError, match=f"{type(model).__name__}: {package} failed"):
        run(make(**bad), market)


@pytest.mark.parametrize("objective", ["l1", "l2"])
def test_catboost_predicts_a_constant_where_it_refuses_degenerate_data(objective):
    """CatBoost refuses a target with no spread, inputs with none and a single row (`CatBoostError`), where LightGBM and
    XGBoost fit: the model predicts the target's value, or the loss's optimum (median for l1, mean for l2), without calling
    CatBoost; other data reaches CatBoost (and a genuine CatBoost error still stops the run, tested above)."""
    model = CatBoostModel(INPUTS, objective=objective)
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    optimum = float(np.median(y)) if objective == "l1" else float(np.mean(y))
    cases = [
        (rng.normal(size=(50, 4)), np.full(50, 7.5), 7.5),  # a target with no spread
        (np.ones((50, 4)), y, optimum),  # inputs with none
        (rng.normal(size=(1, 4)), np.array([3.25]), 3.25),  # one row
    ]
    for x, target, expected in cases:
        assert model._degenerate(x, target) == expected
        fitted = model._fit_model(x, target)
        assert model._predict_block(fitted, np.zeros((3, 4)))[0].tolist() == [expected] * 3
        assert model._predict_fitted(fitted, np.zeros(4)) == (expected, [])
    assert model._degenerate(rng.normal(size=(50, 4)), y) is None
    assert LightGBMModel(INPUTS)._degenerate(np.ones((50, 4)), y) is None  # the others fit such data themselves


def test_catboost_each_period_forecasts_zero_where_a_period_s_targets_are_all_zero(market):
    """`periods="each"` on a frame whose 03:00 price is 0 every day (a generation target's night): that period's fits see a
    target with no spread, which CatBoost refuses, and forecast 0; every other period's forecasts are the run's on the frame
    as it was, to the bit."""
    pytest.importorskip("catboost")
    _, small = FAMILIES["catboost"]
    zeroed = market.copy()
    zeroed.loc[zeroed.index.hour == 3, "price"] = 0.0

    def make():
        return CatBoostModel(EACH_INPUTS, training_window=28, periods="each", name="tree", **small)

    before = keyed(records(make(), market))
    after = keyed(records(make(), zeroed))
    assert after.keys() == before.keys()
    night = [key for key in after if key[1] == 3]
    assert night and all(after[key]["prediction"] == 0.0 for key in night)
    assert all(after[key] == before[key] for key in after if key[1] != 3)


def test_xgboost_on_a_gpu_must_have_fitted_on_one(monkeypatch):
    """Asked for a GPU, a fit that XGBoost moved to the CPU raises (a RuntimeError, which stops the run), so a manifest never
    records "cuda" for CPU forecasts; checked from the booster's own configuration after every fit."""
    pytest.importorskip("xgboost")

    class Fitted:
        def __init__(self, device):
            self.device = device

        def get_booster(self):
            return self

        def save_config(self):
            return json.dumps({"learner": {"generic_param": {"device": self.device}}})

    gpu = XGBoostModel(INPUTS, device="cuda:1")
    with pytest.raises(RuntimeError, match=r"device='cuda:1' was asked for, but XGBoost fitted on 'cpu'"):
        gpu._check_fitted(Fitted("cpu"))
    gpu._check_fitted(Fitted("cuda:1"))
    XGBoostModel(INPUTS)._check_fitted(Fitted("cpu"))


def test_xgboost_asked_for_a_gpu_with_none_visible_stops_the_run():
    """With no GPU visible (CUDA_VISIBLE_DEVICES empty, in a process of its own, before CUDA starts), XGBoost moves the fit to the
    CPU with a warning, and the run stops instead of recording "cuda" for CPU forecasts."""
    pytest.importorskip("xgboost")
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
        "import numpy as np\n"
        "from epftoolbox2.models import XGBoostModel\n"
        "model = XGBoostModel(['a', 'b'], n_estimators=3, device='cuda')\n"
        "rng = np.random.default_rng(0)\n"
        "try:\n"
        "    model._fit_model(rng.normal(size=(50, 2)), rng.normal(size=50))\n"
        "except RuntimeError as error:\n"
        "    print('refused:', error)\n"
        "else:\n"
        "    raise SystemExit('a CPU fit passed as a GPU one')\n"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0 and "refused:" in result.stdout, result.stdout + result.stderr[-2000:]


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

# Each family's numbers move between its package's releases, and the newest-dependencies CI job resolves newer ones than the
# lock's: a model's numbers are compared on the release the file was written with (`version` in its entry), and a model whose
# package isn't installed (catboost on 3.14t; every family in the core-only cells) is skipped, one test per model.
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


def test_write_the_gbm_golden_file():
    """`UPDATE_GOLDEN=1` writes the file, with every family installed; skipped otherwise."""
    if os.environ.get("UPDATE_GOLDEN") != "1":
        pytest.skip("UPDATE_GOLDEN=1 writes the golden file")
    models = golden_models()
    missing = sorted({package for package, _ in models.values() if _version(package) is None})
    assert not missing, f"the golden file holds every family: install {missing}"
    actual = compute_gbm_metrics(set(models))
    written = {
        name: {**values, "package": models[name][0], "version": _version(models[name][0])}
        for name, values in actual.items()
    }
    GOLDEN_FILE.write_text(json.dumps(written, indent=2, sort_keys=True) + "\n")


def test_the_golden_file_holds_every_golden_model():
    assert set(json.loads(GOLDEN_FILE.read_text())) == set(golden_models())


@pytest.mark.parametrize("name", sorted(golden_models()))
def test_gbm_metrics_match_golden(name):
    """One model's golden numbers, compared on the release of its package the file was written with: skipped, saying which,
    where the package isn't installed or is another release."""
    if os.environ.get("UPDATE_GOLDEN") == "1":
        pytest.skip("the golden file is being written")
    values = dict(json.loads(GOLDEN_FILE.read_text())[name])
    package, version = values.pop("package"), values.pop("version")
    installed = _version(package)
    if installed is None:
        pytest.skip(f"{name}: {package} isn't installed")
    if installed != version:
        pytest.skip(f"{name}: {package} {installed} here, {version} in the file (its numbers move between releases)")
    actual = compute_gbm_metrics({name})[name]
    assert actual["forecasts"] == values.pop("forecasts"), f"{name}: number of forecasts changed"
    for key, value in values.items():
        assert actual[key] == pytest.approx(value, rel=GOLDEN_RTOL), f"{name} {key}"


def test_the_gbm_golden_is_of_the_lock_s_versions():
    """Each family's release in the golden file is the one `uv.lock` pins for Python 3.12 and later (the `gbm-extra` job's),
    so the locked cells compare it. Moving the lock's package means writing the file again."""
    lock = (ROOT / "uv.lock").read_text()
    for name, values in json.loads(GOLDEN_FILE.read_text()).items():
        found = re.findall(rf'\[\[package\]\]\nname = "{values["package"]}"\nversion = "([^"]+)"', lock)
        assert found, f"no {values['package']} in uv.lock"
        newest = max(found, key=lambda text: _cards.parse_version(text).key)
        assert values["version"] == newest, (name, values["version"], found)
