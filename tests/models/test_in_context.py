"""`TabPFNModel` and `TabICLModel` (step 6e): in-context tabular regressors on the refit calendar, one block call per unit.

The core job runs every library-side line with a numpy backend in place of the package (`tests/models/in_context_fakes.py`:
`_load_backend` replaced, a fake checkpoint fetched into a cache of the test's own): settings, cards, the lean import, the
pre-flight, YAML, the adapter suite (verify with its control, rerun and co-batch checks, loops and shards, resume, engine, the
DST counts, provenance), the context by hand, the backend loaded once per process under the offline scope, errors that stop a
run, the runtime fields compared on resume, and the planted leak and shift that verify must catch. The `[tabpfn]` and
`[tabicl]` extras run the tests marked `package`, with the real package's regressor around a fake `fit`/`predict` (its
settings, the version guard, one checkpoint read per process), and `-m weights` runs verify with the real weights on the CPU
where `EPFTOOLBOX2_WEIGHTS_DIR` holds them (integration, never in the fast suite).
"""

import importlib.util
import json
from datetime import date
import os
import subprocess
import sys
import types
import warnings
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epftoolbox2._extras import MissingExtraError
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import TabICLModel, TabPFNModel, _in_context, _weights
from epftoolbox2.models._calendar import refit_anchor, training_span
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path
from epftoolbox2.scalers.registry import make_scaler
from tests.models import adapter_suite as suite
from tests.models.in_context_fakes import (
    FAMILIES,
    KernelBackend,
    install_fake_weights,
    kernel_mean,
    patch_package,
    use_fake_backend,
)
from tests.synthetic import make_market_data

INPUTS = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "price_d-1"]
TEST = ("2023-03-13", "2023-03-26")  # two "7D" blocks, Monday to Sunday; the spring switch is on 26 March (Warsaw)
RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
}
INFO = InformationSet(rules=RULES)
ROOT = Path(__file__).parent.parent.parent
CLASSES = [TabPFNModel, TabICLModel]
HEAVY = ("torch", "tabpfn", "tabicl", "autogluon", "huggingface_hub", "transformers", "safetensors")


class InProcessPool:
    """Stands in for the worker pool, as in tests/models/test_gbm.py: the fits run in this process, so the spies see them."""

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
    monkeypatch.setattr(_in_context, "_BACKENDS", {})
    monkeypatch.chdir(tmp_path)  # a package writing files would write them here


@pytest.fixture(scope="module")
def market():
    """1 January to 20 April 2023 in Warsaw, over the spring switch."""
    return make_market_data(start="2023-01-01", days=110, horizon=2, noise=5.0, seed=3)


@pytest.fixture(params=CLASSES, ids=lambda cls: cls.__name__)
def faked(request, monkeypatch, tmp_path):
    """A family with a fake checkpoint in the cache and the numpy backend: (class, factory of models named `name`, the list
    of backends loaded)."""
    cls = request.param
    install_fake_weights(monkeypatch, tmp_path / "weights", cls)
    loaded = use_fake_backend(monkeypatch, cls)

    def make(name="icl", inputs=None, **params):
        return cls(inputs or INPUTS, **{"training_window": 28, "device": "cpu", **params}, name=name)

    return cls, make, loaded


def keyed(records):
    return {(r["run_date"], r["hour"], r["horizon"]): r for r in records}


def run(model, data, test=TEST, horizon=1, **kwargs):
    return ModelPipeline().add_model(model).run(data, *test, horizon=horizon, **kwargs)


def records(model, data, test=TEST, horizon=1, **kwargs):
    return run(model, data, test, horizon, **kwargs).refs[model.name]._results


# ---------------------------------------------------------------------------------------------------------------------------
# Without the packages: settings, cards, the lean import, the pre-flight, saving and loading
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSES)
def test_settings_cards_and_what_is_refused(cls):
    model = cls(INPUTS)
    estimators = {TabPFNModel: 4, TabICLModel: 8}[cls]
    assert (model.training_window, model.refit, model.n_estimators, model.random_state) == (728, "7D", estimators, 0)
    assert (model.device, model.weights, model.block_predict) == ("auto", None, True)
    assert model.scaler == {"name": "asinh", "params": {"apply_to": "target", "standardise": False}}
    assert model._unit_kind == "period" and not model._engine_capable() and model._predicts_together
    assert model._model_kwargs == {"n_estimators": estimators, "random_state": 0, "device": "auto"}
    assert cls(INPUTS, weights="/w/file.ckpt")._model_kwargs["weights"] == "/w/file.ckpt"
    card = model.card
    assert card.extra == {TabPFNModel: "tabpfn", TabICLModel: "tabicl"}[cls] and card.commercial_use
    assert card.stated_verdict == "clean (synthetic)" and card.corpus == "synthetic priors" and card.revenue_cap is None
    assert card.runtime[0].startswith(card.extra) and card.runtime[1:] == ("torch", "scikit-learn", "scipy", "numpy")
    assert model._PRECISION["cpu"] == "float32" and "float16" in model._PRECISION["cuda"]
    if cls is TabPFNModel:
        assert card.weights.repo == "Prior-Labs/TabPFN-v2-reg" and list(card.weights.files) == [
            "tabpfn-v2-regressor.ckpt"
        ]
        assert card.runtime[0] == "tabpfn>=9.1,<10" and "PriorLabs-TabPFN" in card.attribution
    else:
        assert card.weights.repo == "jingang/TabICL"
        assert list(card.weights.files) == ["tabicl-regressor-v2-20260212.ckpt"] and card.licence == "BSD-3-Clause"
    for key, bad in (
        ("n_estimators", 0),
        ("n_estimators", 1.5),
        ("random_state", "0"),
        ("random_state", True),
        ("device", "gpu"),
        ("device", "cuda:"),
        ("block_predict", 1),
        ("weights", 3),
    ):
        with pytest.raises(ValueError, match=key):
            cls(INPUTS, **{key: bad})
    assert type(cls(INPUTS, n_estimators=np.int64(3)).n_estimators) is int
    assert cls(INPUTS, weights=Path("/w/x.ckpt")).weights == "/w/x.ckpt"


@pytest.mark.parametrize("cls", CLASSES)
def test_every_setting_reaches_the_workers(cls):
    """A worker rebuilds the model from `_model_kwargs`: every parameter but the ones BaseModel's run reads in the parent
    (and `weights`, there only when set) must be in it, or a worker would fit with its default."""
    import inspect

    parent_side = {"predictors", "training_window", "refit", "name", "scaler", "newest_target", "binary_mask"}
    parent_side |= {"normalise_by", "bounds", "zero_when", "block_predict", "weights"}
    assert set(cls(INPUTS)._model_kwargs) == set(inspect.signature(cls).parameters) - parent_side
    again = cls(
        predictors=[], **cls(INPUTS, n_estimators=3, random_state=5, device="cpu", weights="/w/f")._model_kwargs
    )
    assert (again.n_estimators, again.random_state, again.device, again.weights) == (3, 5, "cpu", "/w/f")


def import_stubs(directory: Path, log: Path) -> Path:
    """A directory of stand-in packages, one per heavy module, each of which writes its name to `log` and raises ImportError
    when imported: put first on PYTHONPATH, an import of any of them shows in the log whether or not the real package is
    installed and whether or not the code catches the ImportError (a check of `sys.modules` alone passes vacuously where the
    package is missing, as in the core job)."""
    for name in HEAVY:
        package = directory / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            f"with open({str(log)!r}, 'a') as handle:\n    handle.write({name!r} + '\\n')\n"
            f"raise ImportError('{name} imported by a lean path')\n"
        )
    return directory


def test_importing_building_and_saving_imports_no_model_package(tmp_path):
    """Dawid's lean-package condition: importing the package, building a calendar data pipeline and fetching from it, and
    building, saving and loading every new model class import none of torch, the families' packages, autogluon,
    huggingface_hub, transformers or safetensors, and download nothing. Each is a stub that logs its import and raises."""
    log = tmp_path / "imported.log"
    stubs = import_stubs(tmp_path / "stubs", log)
    code = (
        "import epftoolbox2\n"
        "from epftoolbox2.data.sources import CalendarSource\n"
        "from epftoolbox2.pipelines import DataPipeline, ModelPipeline\n"
        "from epftoolbox2.models import TabPFNModel, TabICLModel\n"
        "pipe = DataPipeline().add_source(CalendarSource(country='PL', timezone='Europe/Warsaw'))\n"
        "frame = pipe.run('2024-01-01', '2024-01-03', cache=False)\n"
        "assert len(frame) > 0\n"
        "models = ModelPipeline()\n"
        "for cls in (TabPFNModel, TabICLModel):\n"
        "    models.add_model(cls(['is_holiday_d+{horizon}'], name=cls.__name__))\n"
        f"models.save({str(tmp_path / 'models.yaml')!r})\n"
        f"loaded = ModelPipeline.load({str(tmp_path / 'models.yaml')!r})\n"
        "assert [type(m).__name__ for m in loaded.models] == ['TabPFNModel', 'TabICLModel']\n"
        "assert [m._model_kwargs['device'] for m in loaded.models] == ['auto', 'auto']\n"
    )
    path = os.pathsep.join([str(stubs), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env = {**os.environ, "EPFTOOLBOX2_WEIGHTS_DIR": str(tmp_path / "no-weights"), "PYTHONPATH": path}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path, env=env)
    assert result.returncode == 0, result.stderr
    assert not log.exists(), f"imported: {log.read_text().split()}"
    assert not (tmp_path / "no-weights").exists(), "something wrote into the weights cache"


def test_the_import_stubs_catch_an_import():
    """The stubs see an import that code catches: the check above can fail."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        log = Path(directory) / "imported.log"
        stubs = import_stubs(Path(directory) / "stubs", log)
        code = "try:\n    import torch\nexcept ImportError:\n    pass\n"
        env = {**os.environ, "PYTHONPATH": str(stubs)}
        subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=directory)
        assert log.read_text().split() == ["torch"]


@pytest.mark.parametrize("cls", CLASSES)
def test_without_its_package_a_model_saves_and_loads_and_a_fit_names_the_extra(cls, market, tmp_path, monkeypatch):
    """The class imports, saves and loads with its package missing, a fit called directly raises `MissingExtraError` naming
    the extra (with the weights in place), and a run's pre-flight refuses before any model's run starts."""
    extra = cls._card.extra
    monkeypatch.setitem(sys.modules, extra, None)  # `import tabpfn` raises ModuleNotFoundError
    model = cls(INPUTS, n_estimators=2, random_state=7, device="cpu", name="icl")
    path = tmp_path / "pipeline.yaml"
    ModelPipeline().add_model(model).save(path)
    (loaded,) = ModelPipeline.load(path).models
    assert type(loaded) is cls and loaded._model_kwargs == model._model_kwargs
    assert "weights" not in path.read_text()
    install_fake_weights(monkeypatch, tmp_path / "weights", cls, real_extra=True)
    suite.check_preflight_without_extra(lambda name, **p: cls(INPUTS, name=name, **p), monkeypatch, market, TEST)
    with pytest.raises(MissingExtraError, match=rf"pip install 'epftoolbox2\[{extra}\]'"):
        model._fit_model(np.arange(15.0).reshape(3, 5), np.arange(3.0))


@pytest.mark.parametrize("cls", CLASSES)
def test_without_weights_the_pre_flight_names_the_fetch_command(cls, market, tmp_path, monkeypatch):
    from epftoolbox2.models._weights import MissingWeightsError

    install_fake_weights(monkeypatch, tmp_path / "fetched", cls)  # the extra and its pins pass
    monkeypatch.setenv(_weights.WEIGHTS_DIR_ENV, str(tmp_path / "empty"))
    with pytest.raises(MissingWeightsError, match=f"epf weights fetch {cls.__name__}"):
        run(cls(INPUTS, training_window=28, device="cpu"), market)


@pytest.mark.parametrize("cls", CLASSES)
def test_a_default_model_s_yaml_has_no_weights_and_a_set_one_loads_back(cls, tmp_path):
    texts = {}
    for weights in (None, "/w/checkpoint.ckpt"):
        path = tmp_path / f"{weights is None}.yaml"
        ModelPipeline().add_model(cls(INPUTS, weights=weights, name="icl")).save(path)
        texts[weights] = path.read_text()
        (loaded,) = ModelPipeline.load(path).models
        assert loaded.weights == weights and loaded.refit == "7D" and loaded.block_predict is True
    assert "weights" not in texts[None] and "weights: /w/checkpoint.ckpt" in texts["/w/checkpoint.ckpt"]


# ---------------------------------------------------------------------------------------------------------------------------
# The adapter, with the numpy backend
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:overflow encountered in sinh")  # the control's noise, mapped back by sinh
def test_the_adapter_suite(faked, market, tmp_path, monkeypatch):
    cls, make, _ = faked
    suite.check_yaml_round_trip(make, market, tmp_path, TEST, horizon=1)
    suite.check_loop_and_shards(
        make, market, tmp_path / "loop", TEST, horizon=1, variants=({}, {"n_estimators": 2}, {"block_predict": False})
    )
    suite.check_resume(make, market, tmp_path / "resume", TEST, horizon=1)
    suite.check_engine(make, market, TEST, horizon=1)
    suite.check_provenance(make, market, tmp_path / "provenance", TEST)
    # A block's first, middle and last day, the day before the switch and the switch day.
    suite.check_verify(
        make, market, INFO, ["2023-03-13", "2023-03-16", "2023-03-19", "2023-03-25", "2023-03-26"], items=6
    )


def test_the_adapter_suite_s_calendar_counts(faked):
    """23 forecasts for the spring target day, 24 for the autumn one with the repeated hour's actual NaN."""
    _, make, _ = faked
    both = pd.concat(
        [
            make_market_data(start="2023-01-15", days=75, horizon=1, noise=5.0, seed=3),
            make_market_data(start="2023-08-25", days=70, horizon=1, noise=5.0, seed=3),
        ]
    )
    suite.check_dst_counts(lambda name, **p: make(name, **p), both, "2023-03-26", "2023-10-29")


def test_a_unit_s_context_and_call_by_hand(faked, market):
    """One unit by hand: the backend is shown the rows of its period dated in `training_span(anchor, W, h)` with every input
    finite, raw, and their targets through the Invariant (asinh of median and MAD, `standardise=False`), and is called once on
    the issue rows of the block's 7 days, raw; a forecast is its row's median mapped back by the Invariant's inverse."""
    cls, make, loaded = faked
    model = make()
    out = keyed(records(model, market, test=("2023-03-15", "2023-03-15"), horizon=1))
    (backend,) = loaded
    period, h, day = 9, 1, pd.Timestamp("2023-03-15").date()
    anchor = refit_anchor(day, "7D")
    first, last = (date.fromordinal(int(d)) for d in training_span(anchor.toordinal(), 28, h))
    local = market.index.tz_localize(None)
    columns = [name.format(horizon=h) for name in INPUTS]
    rows = market[
        (local.hour == period) & (local.normalize() >= pd.Timestamp(first)) & (local.normalize() <= pd.Timestamp(last))
    ]
    target = rows.index.map(lambda t: t + pd.DateOffset(days=1))
    y = market["price"].reindex(target).to_numpy()
    x = rows[columns].to_numpy()
    keep = np.isfinite(x).all(axis=1) & np.isfinite(y)
    scaler = make_scaler(model.scaler)
    x_scaled, y_scaled, _ = scaler.fit_transform(x[keep].copy(), y[keep].copy(), x[keep][:1].copy(), scalable_mask=None)
    assert np.array_equal(x_scaled, x[keep]), "the inputs reach the context raw"
    fits = [fit for fit in backend.fits if len(fit[1]) == keep.sum() and np.array_equal(fit[0], x[keep])]
    assert len(fits) == 1 and np.array_equal(fits[0][1], y_scaled)
    median = np.median(y[keep])
    assert np.allclose(
        fits[0][1], np.arcsinh((y[keep] - median) / (np.median(np.abs(y[keep] - median)) / 0.6744897501960817))
    )
    issue = market[(local.hour == period) & (local.normalize() >= pd.Timestamp(anchor))]
    issue = issue[issue.index.tz_localize(None).normalize() <= pd.Timestamp(anchor) + pd.Timedelta(days=6)][columns]
    (call,) = [c for c in backend.calls if np.array_equal(c, issue.to_numpy())]
    assert call.shape == (7, len(INPUTS))
    expected = scaler.inverse(kernel_mean(x[keep], y_scaled, issue.to_numpy()[2], 0.5))
    assert out[("2023-03-15", period, h)]["prediction"] == pytest.approx(expected, rel=1e-12)
    assert out[("2023-03-15", period, h)]["coefficients"] == []


def test_one_row_at_a_time_gives_the_block_call_s_forecasts(faked, market):
    """The numpy backend has no cross-row dependence, so `block_predict=False` (one row a call) gives the block's bits; one
    call per unit with it on, one per forecast without."""
    _, make, loaded = faked
    blocked = keyed(records(make(), market))
    (backend,) = loaded  # the same backend serves both runs: `block_predict` changes the calls, not the model
    block_calls = list(backend.calls)
    single = keyed(records(make(block_predict=False), market))
    # 14 issue days of 24 periods, less the 02:00 the clock skips on 26 March, as an issue day and as a target day
    assert sorted(blocked) == sorted(single) and len(blocked) == 14 * 24 - 2
    assert all(blocked[k]["prediction"] == single[k]["prediction"] for k in blocked)
    assert len(block_calls) == 2 * 24 and {len(call) for call in block_calls} == {7}  # two blocks of 24 periods
    assert len(backend.calls) - len(block_calls) == len(single) and {
        len(c) for c in backend.calls[len(block_calls) :]
    } == {1}


def test_the_backend_is_loaded_once_per_process_under_the_offline_scope(faked, market, monkeypatch):
    """Every unit of a run (and of a second run in the process) uses one backend, loaded with the resolved checkpoint's path
    while HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE are 1, which are put back afterwards; another device or seed loads another."""
    cls, make, loaded = faked
    seen = []
    real = cls._load_backend

    def spy(self, resolved, device):
        seen.append((os.environ.get("HF_HUB_OFFLINE"), os.environ.get("TRANSFORMERS_OFFLINE"), device))
        return real(self, resolved, device)

    monkeypatch.setattr(cls, "_load_backend", spy)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    records(make(), market)
    records(make(), market)
    assert seen == [("1", "1", "cpu")] and len(loaded) == 1
    assert "HF_HUB_OFFLINE" not in os.environ
    (name,) = cls._card.weights.files
    assert Path(loaded[0].path).name == name and Path(loaded[0].path).parent.name == cls._card.weights.revision
    records(make(random_state=1), market)
    assert len(loaded) == 2


def test_a_run_reaches_for_no_network(faked, market, no_network):
    _, make, _ = faked
    assert records(make(), market)


def test_a_given_weights_file_reaches_the_workers(faked, market, tmp_path):
    """`weights="/path"` travels in `_model_kwargs`, so each worker's model loads that file, not the cache's."""
    cls, make, loaded = faked
    (name,) = cls._card.weights.files
    given = tmp_path / "elsewhere" / name
    given.parent.mkdir()
    given.write_bytes((_weights.cache_dir(cls._card.weights) / name).read_bytes())
    records(make(weights=str(given)), market, test=("2023-03-13", "2023-03-13"))
    assert [Path(b.path) for b in loaded] == [given]


def test_the_package_s_errors_stop_the_run_and_equal_targets_need_no_call(faked, market, monkeypatch):
    """A ValueError (or anything else) from the package's fit or predict stops the run naming the model, never a silent
    `fit_error`; a context whose targets are all equal predicts that value without calling the package."""
    cls, make, loaded = faked

    def broken(self, x, y):
        raise ValueError("bad context")

    monkeypatch.setattr(KernelBackend, "fit", broken)
    with pytest.raises(RuntimeError, match="failed to fit: ValueError: bad context"):
        records(make(), market, test=("2023-03-13", "2023-03-13"))
    model = make()
    fitted = model._fit_model(np.ones((5, 3)), np.full(5, 0.25))
    predictions, coefficients = model._predict_block(fitted, np.zeros((7, 3)))
    assert predictions.tolist() == [0.25] * 7 and coefficients == [[]] * 7


def test_the_runtime_fields_and_weights_are_recorded_and_compared_on_resume(faked, market, tmp_path):
    cls, make, _ = faked
    run(make(), market, save_dir=tmp_path)
    path = manifest_path(tmp_path / f"{make().name}.jsonl")
    manifest = json.loads(path.read_text())
    assert manifest["runtime"]["device"] == "cpu" and manifest["runtime"]["threads_per_fit"] == 1
    assert manifest["runtime"]["precision"] == "float32"
    assert manifest["model_params"]["block_predict"] is True and manifest["model_params"]["device"] == "cpu"
    assert manifest["card"]["family"] == cls._card.family and manifest["weights"]["reference"] is True
    for field, other in (("device", "cuda"), ("precision", "bfloat16 autocast")):
        changed = json.loads(json.dumps(manifest))
        changed["runtime"][field] = other
        path.write_text(json.dumps(changed))
        with pytest.raises(ManifestMismatchError, match=f"runtime.{field}"):
            run(make(), market, save_dir=tmp_path)


@pytest.mark.parametrize("cls", CLASSES)
def test_processes_one_fit_each_and_six_per_gpu(cls, monkeypatch):
    from epftoolbox2.models.base import _usable_cpu_count

    cpus = max(1, _usable_cpu_count())
    model = cls(INPUTS)
    monkeypatch.setattr(cls, "_run_devices", lambda self: ("cpu",))
    monkeypatch.setenv(
        "THREADS_PER_PROCESS", "4"
    )  # a fit's torch settings are its process's: one task thread, whatever
    monkeypatch.delenv("MAX_PROCESSES")
    assert model._executor_config() == (1, cpus)
    monkeypatch.setattr(cls, "_run_devices", lambda self: ("cuda:0",))
    assert model._executor_config() == (1, min(6, cpus))
    monkeypatch.setattr(cls, "_run_devices", lambda self: ("cuda:0", "cuda:1", "cuda:2", "cuda:3"))
    assert model._executor_config() == (1, min(24, cpus))
    monkeypatch.setenv("MAX_PROCESSES", "3")
    assert model._executor_config() == (1, 3)


class FakeDevice:
    """A `torch.device` stand-in: its type and index, printed as torch prints one."""

    def __init__(self, text):
        self.type, _, index = text.partition(":")
        self.index = int(index) if index else None
        self.text = text

    def __str__(self):
        return self.text


@pytest.mark.parametrize("cls", CLASSES)
def test_auto_is_every_usable_gpu_of_gpu_0_s_model_and_cuda_n_that_one(cls, monkeypatch):
    """`_run_devices`: "auto" resolving to a GPU (GPU 0's probe) is every GPU whose probe passes, in index order, of GPU 0's
    model; "cuda:N" is that GPU; the CPU and a missing torch are ("cpu",); the answer is kept for the model's `device` and
    asked again when it changes. The runtime names every GPU used (`device_name`, warn-only)."""
    from epftoolbox2.models._torch import runtime

    asked = []
    monkeypatch.setattr(_in_context, "has_torch", lambda: True)
    monkeypatch.setattr(_in_context, "cuda_device_count", lambda: 3)
    monkeypatch.setattr(runtime, "cuda_probe", lambda index: (index != 1, "NVIDIA H100 80GB HBM3", "ok"))
    monkeypatch.setattr(runtime, "cuda_capability", lambda index: "9.0")

    def resolved(self):
        asked.append(self.device)
        return FakeDevice({"auto": "cuda:0", "cuda": "cuda:0", "cpu": "cpu"}.get(self.device, self.device))

    monkeypatch.setattr(cls, "_resolved_device", resolved)
    model = cls(INPUTS)
    assert model._run_devices() == ("cuda:0", "cuda:2") and model._run_devices() == ("cuda:0", "cuda:2")
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "torch", types.ModuleType("torch"))  # the CUDA branch reads nothing of it
        fields = model._runtime_fields()
    assert fields["device_name"] == "NVIDIA H100 80GB HBM3 x 2 (cuda:0, cuda:2)" and fields["cuda_capability"] == "9.0"
    model.device = "cuda:1"
    assert model._run_devices() == ("cuda:1",)
    model.device = "cpu"
    assert model._run_devices() == ("cpu",) and asked == ["auto", "cuda:1", "cpu"]
    monkeypatch.setattr(cls, "_resolved_device", _in_context.InContextModel._resolved_device)
    monkeypatch.setattr(_in_context, "has_torch", lambda: False)
    assert cls(INPUTS)._run_devices() == ("cpu",)


@pytest.mark.parametrize("cls", CLASSES)
def test_auto_leaves_out_gpus_of_another_model_with_a_warning(cls, monkeypatch):
    """An A100 at index 0 and an H100 at index 1: under "auto" the run keeps GPU 0's model alone (float16 forecasts differ
    between the two, and which unit runs where depends on the scheduling) and says which GPU it left out; the manifest's
    runtime names the A100 only, so a resume on an H100 refuses on its capability."""
    from epftoolbox2.models._torch import runtime

    gpus = {
        0: ("NVIDIA A100-SXM4-80GB", "8.0"),
        1: ("NVIDIA H100 80GB HBM3", "9.0"),
        2: ("NVIDIA A100-SXM4-80GB", "8.0"),
    }
    monkeypatch.setattr(_in_context, "has_torch", lambda: True)
    monkeypatch.setattr(_in_context, "cuda_device_count", lambda: 3)
    monkeypatch.setattr(runtime, "cuda_probe", lambda index: (True, gpus[index][0], "ok"))
    monkeypatch.setattr(runtime, "cuda_capability", lambda index: gpus[index][1])
    monkeypatch.setattr(cls, "_resolved_device", lambda self: FakeDevice("cuda:0"))
    model = cls(INPUTS, name="icl")
    with pytest.warns(UserWarning, match=r"leaves out cuda:1 \(NVIDIA H100 80GB HBM3, capability 9.0\)"):
        assert model._run_devices() == ("cuda:0", "cuda:2")
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "torch", types.ModuleType("torch"))
        fields = model._runtime_fields()
    assert (fields["cuda_capability"], fields["device_name"]) == ("8.0", "NVIDIA A100-SXM4-80GB x 2 (cuda:0, cuda:2)")


def test_worker_index_is_the_last_part_of_a_process_s_identity(monkeypatch):
    """A pool's workers are told apart by the last part of their identity: `(3,)` is worker 2 of a top-level pool, and
    `(12, 1)`, `(12, 2)` are workers 0 and 1 of a pool inside the parent's twelfth child (the first part is every one's)."""
    import multiprocessing

    for identity, index in (((), 0), ((1,), 0), ((3,), 2), ((12, 1), 0), ((12, 2), 1), ((12, 4, 2), 1)):
        monkeypatch.setattr(
            multiprocessing, "current_process", lambda identity=identity: types.SimpleNamespace(_identity=identity)
        )
        assert _in_context.worker_index() == index, identity


def test_tabicl_s_precision_check_stops_a_fit_that_autocasts_where_the_manifest_says_it_doesn_t():
    """On a GPU TabICL autocasts from 1,024 rows or 60 columns: a regressor that says it autocasts a 1,023 x 59 context stops
    the run, as one that doesn't for 1,024 x 59 does; on a CPU any autocast stops it."""
    from epftoolbox2.models import tabicl

    def backend(on_cpu):
        made = object.__new__(tabicl._Backend)
        made.on_cpu, made.device = on_cpu, "cpu" if on_cpu else "cuda:0"
        return made

    def regressor(autocast):
        return types.SimpleNamespace(_resolve_amp_fa3=lambda: (autocast, False))

    with pytest.raises(RuntimeError, match="autocasts=True on cuda:0 for a context of 1023 rows and 59 columns"):
        backend(False)._check_precision(regressor(True), (1023, 59))
    with pytest.raises(RuntimeError, match="autocasts=False on cuda:0 for a context of 1024 rows"):
        backend(False)._check_precision(regressor(False), (1024, 59))
    with pytest.raises(RuntimeError, match="on cpu"):
        backend(True)._check_precision(regressor(True), (728, 175))
    backend(False)._check_precision(regressor(True), (728, 175))
    backend(False)._check_precision(regressor(False), (1023, 59))
    backend(True)._check_precision(regressor(False), (728, 175))


def test_tabpfn_s_precision_check_stops_a_fit_that_autocasts_on_a_cpu():
    """A regressor reporting autocast on a CPU (tabpfn's "auto" on an AVX512-BF16 or AMX CPU), or a forced dtype other than
    float32, stops the run; on a GPU, one that doesn't autocast does."""
    from epftoolbox2.models import tabpfn

    torch = types.SimpleNamespace(float32="torch.float32")

    def backend(on_cpu):
        made = object.__new__(tabpfn._Backend)
        made.on_cpu, made.device, made.torch = on_cpu, "cpu" if on_cpu else "cuda:0", torch
        return made

    def regressor(autocast, forced):
        return types.SimpleNamespace(use_autocast_=autocast, forced_inference_dtype_=forced)

    with pytest.raises(RuntimeError, match="autocast=True, dtype=None on cpu, not the float32 the manifest records"):
        backend(True)._check_precision(regressor(True, None))
    with pytest.raises(RuntimeError, match="on cpu"):
        backend(True)._check_precision(regressor(False, "torch.bfloat16"))
    with pytest.raises(RuntimeError, match="not the float16 autocast"):
        backend(False)._check_precision(regressor(False, None))
    backend(True)._check_precision(regressor(False, "torch.float32"))
    backend(False)._check_precision(regressor(True, None))


def test_a_gpu_fit_gets_the_pinned_memory_settings():
    """On a GPU nothing a fit computes with is sized by the card's free memory: tabpfn's memory saving off, TabICL's batches
    at 50,000 in each of its three managers and its outputs kept on the GPU, with autocast left to the packages' "auto"."""
    from epftoolbox2.models import tabicl, tabpfn

    made = []

    class Regressor:
        def __init__(self, *args, **kwargs):
            made.append(kwargs)
            self.use_autocast_, self.forced_inference_dtype_ = True, None

        create_default_for_version = classmethod(lambda cls, version, **kwargs: cls(**kwargs))

        def fit(self, x, y):
            pass

        def _resolve_amp_fa3(self):
            return True, False

    x, y = np.zeros((735, 175)), np.zeros(735)
    backend = object.__new__(tabpfn._Backend)
    backend.__dict__.update(on_cpu=False, device="cuda:0", torch=None, regressor=Regressor, version="v2", specs="specs")
    backend.__dict__.update(precision="auto", n_estimators=4, random_state=0)
    backend.fit(x, y)
    assert made.pop()["memory_saving_mode"] is False and tabpfn.TabPFNModel(INPUTS)._memory_settings("cuda") == {
        "memory_saving": False
    }
    backend = object.__new__(tabicl._Backend)
    backend.__dict__.update(on_cpu=False, device="cuda:0", use_amp="auto", regressor=Regressor, path="weights")
    backend.__dict__.update(n_estimators=8, random_state=0, loaded=(None, None, None))
    backend.fit(x, y)
    kwargs = made.pop()
    assert kwargs["offload_mode"] == "gpu" and kwargs["use_amp"] == "auto"
    assert kwargs["inference_config"] == dict.fromkeys(("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG"), {"min_batch_size": 50_000})
    assert tabicl.TabICLModel(INPUTS)._memory_settings("cuda") == {"min_batch_size": 50_000, "offload": "gpu"}


class _CudaCalls(types.ModuleType):
    """A stand-in for `torch.cuda` that records every attribute the code reads of it (any call would start CUDA)."""

    def __init__(self, seen):
        super().__init__("torch.cuda")
        self._seen = seen

    def __getattr__(self, name):
        self._seen.append(name)
        raise AssertionError(f"torch.cuda.{name} read in the parent")


@pytest.mark.parametrize("cls", CLASSES)
def test_the_parent_never_touches_torch_cuda_while_resolving_a_gpu_run(cls, monkeypatch):
    """Resolving a GPU run's devices, its runtime fields, its processes and its workers' inputs reads nothing of `torch.cuda`
    in the parent (the probes run in subprocesses): a parent that started CUDA would break its forked workers. The stand-in
    records any read, so the check can fail with CPU torch too."""
    from epftoolbox2.models._torch import runtime

    seen = []
    fake = types.ModuleType("torch")
    fake.cuda = _CudaCalls(seen)
    fake.backends = types.SimpleNamespace(cpu=types.SimpleNamespace(get_cpu_capability=lambda: "AVX2"))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setitem(sys.modules, "torch.cuda", fake.cuda)
    monkeypatch.setattr(_in_context, "has_torch", lambda: True)
    monkeypatch.setattr(_in_context, "cuda_device_count", lambda: 2)
    monkeypatch.setattr(runtime, "cuda_probe", lambda index: (True, "NVIDIA H100 80GB HBM3", "ok"))
    monkeypatch.setattr(runtime, "cuda_capability", lambda index: "9.0")
    monkeypatch.setattr(runtime, "resolve_device", lambda device: FakeDevice("cuda:0"))
    monkeypatch.delenv("MAX_PROCESSES")
    model = cls(INPUTS, device="auto")
    assert model._run_devices() == ("cuda:0", "cuda:1")
    assert model._runtime_fields()["device"] == "cuda" and model._executor_config()[0] == 1
    assert seen == []


def test_the_parent_resolves_the_device_once_and_every_worker_fits_on_its_answer(faked, market, monkeypatch):
    """The parent's answer reaches every fit, whatever a later resolution would say: `_resolved_device` answers CUDA first
    and the CPU afterwards, and every backend is loaded on the parent's CUDA device; each worker checks the device once."""
    cls, make, loaded = faked
    answers = iter([FakeDevice("cuda:0")] + [FakeDevice("cpu")] * 100)
    devices, checked = [], []
    monkeypatch.setattr(_in_context, "has_torch", lambda: True)
    monkeypatch.setattr(_in_context, "cuda_device_count", lambda: 0)
    monkeypatch.setattr(cls, "_resolved_device", lambda self: next(answers))
    monkeypatch.setattr(cls, "_runtime_fields", lambda self: {"device": self._run_devices()[0].split(":")[0]})
    monkeypatch.setattr(cls, "_check_worker_device", lambda self, device: checked.append(device))
    monkeypatch.setattr(_in_context, "_CHECKED_DEVICES", set())
    real = cls._load_backend

    def spy(self, resolved, device):
        devices.append(device)
        return real(self, resolved, device)

    monkeypatch.setattr(cls, "_load_backend", spy)
    out = records(make(device="auto"), market, test=("2023-03-13", "2023-03-19"))
    assert out and devices == ["cuda:0"] and checked == ["cuda:0"]
    assert [b.device for b in loaded] == ["cuda:0"]


def test_worker_k_fits_on_gpu_k_modulo_the_number_of_gpus(monkeypatch):
    from epftoolbox2.models import _worker
    from epftoolbox2.models._calendar import FitUnit

    model = TabPFNModel(INPUTS)
    monkeypatch.setattr(TabPFNModel, "_check_worker_device", lambda self, device: None)
    monkeypatch.setattr(_worker, "_W_CONFIG", {"in_context_devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]})
    picked = []
    for k in range(6):
        monkeypatch.setattr(_in_context, "worker_index", lambda k=k: k)
        with _worker.fit_scope(FitUnit(0, 1, 0, None, 0), 28):
            picked.append(model._device_text())
    assert picked == ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:0", "cuda:1"]
    monkeypatch.setattr(TabPFNModel, "_run_devices", lambda self: ("cpu",))
    assert model._device_text() == "cpu"  # a fit outside a run's task resolves its own


def test_a_worker_that_can_t_use_its_parent_s_gpu_stops_the_run():
    pytest.importorskip("torch")
    import torch

    if torch.cuda.is_available() and torch.cuda.device_count() > 7:
        pytest.skip("this machine has a GPU 7")
    with pytest.raises(
        RuntimeError, match="the run resolved cuda:7 in its parent process, and this worker can't use it"
    ):
        TabPFNModel(INPUTS, name="icl")._check_worker_device("cuda:7")
    TabPFNModel(INPUTS)._check_worker_device("cpu")


def test_a_backend_that_fails_to_load_stops_the_run(faked, market, monkeypatch):
    """A load's ValueError (a checkpoint the package can't read) stops the run, never a `fit_error` skip of every unit."""
    cls, make, _ = faked

    def broken(self, resolved, device):
        raise ValueError("not a checkpoint")

    monkeypatch.setattr(cls, "_load_backend", broken)
    with pytest.raises(RuntimeError, match="failed to load: ValueError: not a checkpoint"):
        records(make(), market, test=("2023-03-13", "2023-03-13"))


@pytest.mark.parametrize("torch_found", [True, False])
def test_workers_pin_torch_to_one_thread_where_torch_is_installed(faked, market, monkeypatch, torch_found):
    """With torch installed a run's config pins its workers' torch to one thread (`_worker._pool_init`); without it (a numpy
    fake in the core job) nothing would import torch."""
    if torch_found:
        pytest.importorskip("torch")
    _, make, _ = faked
    configs = []
    real = _in_context.BaseModel._worker_inputs

    def spy(self, state, threads):
        expanded, config = real(self, state, threads)
        configs.append(config)
        return expanded, config

    monkeypatch.setattr(_in_context.BaseModel, "_worker_inputs", spy)
    monkeypatch.setattr(_in_context, "has_torch", lambda: torch_found)
    records(make(), market, test=("2023-03-13", "2023-03-13"))
    assert configs and all(config.get("torch_threads") == (1 if torch_found else None) for config in configs)


# ---------------------------------------------------------------------------------------------------------------------------
# Planted leaks: what verify must catch
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("planted", [{"leak": 1e-3}, {"shift": True}], ids=["sees-its-call", "next-day-s-row"])
@pytest.mark.filterwarnings("ignore:overflow encountered in sinh")
def test_verify_catches_a_backend_that_sees_its_call_or_reads_the_next_day(faked, market, monkeypatch, planted):
    """A backend that adds 1e-3 x the mean of its call's first column (it sees the other rows of its block), or that
    forecasts each row from the next row of its call (the next day's inputs, which the issue time doesn't know): verify
    reports a leak on a block's first day, through the co-batch check: the next day's inputs here are all known at the issue
    time (published a week ahead), so the plain probe sees nothing; what moves is the forecast under the other rows' noise."""
    cls, make, _ = faked
    use_fake_backend(monkeypatch, cls, **planted)
    report = (
        ModelPipeline(information_set=INFO)
        .add_model(make("icl"))
        .verify(market, "2023-03-13", "2023-03-13", horizon=1, days=1, rerun=True, co_batch=True, device="cpu")
    )
    row = report.table().iloc[0]
    assert report.status("icl") == "leak", report.problems()
    assert row["co_batch"] > 0 and row["changed"] == 0


# ---------------------------------------------------------------------------------------------------------------------------
# With the real packages (the [tabpfn] and [tabicl] extras), their model calls faked
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.fixture(params=sorted(FAMILIES))
def packaged(request, monkeypatch, tmp_path):
    """A family whose package is installed, the real regressor around a fake `fit`/`predict`, a fake checkpoint in the cache:
    (class, factory, what the regressor saw)."""
    family = request.param
    pytest.importorskip(family)
    cls = FAMILIES[family]
    install_fake_weights(monkeypatch, tmp_path / "weights", cls, real_extra=True)
    for (
        name
    ) in _weights.OFFLINE_VARIABLES:  # what a fit sees is the adapter's offline scope, not the caller's environment
        monkeypatch.delenv(name, raising=False)
    seen = patch_package(monkeypatch, family)

    def make(name="icl", inputs=None, **params):
        return cls(inputs or INPUTS, **{"training_window": 28, "device": "cpu", **params}, name=name)

    return cls, make, seen


def test_the_real_regressor_gets_the_research_s_settings(packaged, market):
    cls, make, seen = packaged
    out = records(make(), market, test=("2023-03-13", "2023-03-14"))
    assert out and set(seen["predicts"]) == {"median"}
    made = seen["made"][0]
    (name,) = cls._card.weights.files
    assert made["device"] == "cpu" and made["random_state"] == 0
    assert len(seen["loads"]) == 1, "the network is built once per process"
    load = seen["loads"][0]
    assert Path(load["path"]).name == name and load["offline"] == "1"
    if cls is TabPFNModel:
        import torch
        from tabpfn.base import ModelSpecs

        assert made["n_estimators"] == 4 and made["n_preprocessing_jobs"] == 1
        assert made["inference_precision"] is torch.float32, "float32 pinned on a CPU (auto autocasts to bf16 on some)"
        assert made["memory_saving_mode"] is False, "not sized by free memory"
        assert all(
            isinstance(m["model_path"], ModelSpecs) and m["model_path"] is made["model_path"] for m in seen["made"]
        )
        assert load["download_if_not_exists"] is False and load["force_inference_dtype"] is torch.float32
    else:
        assert (made["n_estimators"], made["batch_size"], made["allow_auto_download"]) == (8, 8, False)
        assert made["use_amp"] is False and Path(made["model_path"]).name == name
        managers = made["inference_config"]
        assert {key: value["min_batch_size"] for key, value in managers.items()} == dict.fromkeys(
            ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG"), 50_000
        )
    assert len(seen["fits"]) == 24 and set(seen["fit_offline"]) == {"1"}, "every fit runs offline"


def test_the_adapter_suite_with_the_real_package(packaged, market, tmp_path):
    _, make, _ = packaged
    suite.check_yaml_round_trip(make, market, tmp_path, TEST, horizon=1)
    suite.check_resume(make, market, tmp_path / "resume", TEST, horizon=1)
    suite.check_provenance(make, market, tmp_path / "provenance", TEST)
    suite.check_verify(make, market, INFO, ["2023-03-13", "2023-03-26"], items=6)


def test_the_runtime_records_the_package_and_torch(packaged, market, tmp_path):
    cls, make, _ = packaged
    import torch

    run(make(), market, save_dir=tmp_path, test=("2023-03-13", "2023-03-13"))
    manifest = json.loads(manifest_path(tmp_path / f"{make().name}.jsonl").read_text())
    import importlib.metadata as metadata

    packages = manifest["runtime"]["packages"]
    assert packages["torch"] == torch.__version__ and packages[cls._card.extra] == metadata.version(cls._card.extra)
    for name in ("scikit-learn", "scipy", "numpy"):  # they preprocess every context
        assert packages[name] == metadata.version(name)
    assert manifest["runtime"]["cpu_capability"] == torch.backends.cpu.get_cpu_capability()
    assert manifest["runtime"]["precision"] == "float32" and manifest["runtime"]["device_name"]
    assert not torch.cuda.is_initialized(), "a run started CUDA in the parent before its pool forked"


def test_tabpfn_never_reaches_a_download_path(monkeypatch, tmp_path, market, no_network):
    """The backend calls tabpfn's own loader by its real signature with downloads off; on a weights file that isn't a
    checkpoint the run stops at the load with the loader's own error, and no download path of tabpfn's (the Hugging Face
    one, or the direct URL that HF_HUB_OFFLINE doesn't stop) is reached. Without tabpfn the run stops at the pre-flight,
    naming the extra."""
    install_fake_weights(monkeypatch, tmp_path / "weights", TabPFNModel, real_extra=True)
    monkeypatch.setattr(_in_context, "_BACKENDS", {})
    model = TabPFNModel(INPUTS, training_window=28, device="cpu")
    if importlib.util.find_spec("tabpfn") is None:
        with pytest.raises(MissingExtraError, match=r"\[tabpfn\]"):
            records(model, market, test=("2023-03-13", "2023-03-13"))
        return
    import inspect
    import urllib.request

    import tabpfn.model_loading as loading

    attempts, calls = [], []

    def refuse(name):
        def call(*args, **kwargs):
            attempts.append(name)
            raise AssertionError(f"tabpfn tried to download ({name})")

        return call

    for name in ("download_model", "_try_huggingface_downloads", "_try_direct_downloads"):
        monkeypatch.setattr(loading, name, refuse(name))
    monkeypatch.setattr(urllib.request, "urlopen", refuse("urlopen"))
    real = loading.load_model_criterion_config

    def bound(*args, **kwargs):
        calls.append(inspect.signature(real).bind(*args, **kwargs).arguments)  # a TypeError for a call it doesn't take
        return real(*args, **kwargs)

    monkeypatch.setattr(loading, "load_model_criterion_config", bound)
    with pytest.raises(RuntimeError, match="failed to load") as stopped:
        records(model, market, test=("2023-03-13", "2023-03-13"))
    (arguments,) = calls
    assert arguments["download_if_not_exists"] is False and arguments["version"] == "v2"
    with pytest.raises(Exception) as direct:  # the loader's own error on that file, called as the backend calls it
        real(**arguments)
    cause = stopped.value.__cause__
    assert type(cause) is type(direct.value) and str(cause) == str(direct.value)
    assert not isinstance(cause, AssertionError) and attempts == []


def test_tabpfn_refuses_weights_its_package_reads_as_another_version(monkeypatch, tmp_path, market):
    """Only the v2 regressor: a checkpoint named as the newer, non-commercial weights is refused before any fit uses it."""
    pytest.importorskip("tabpfn")
    patch_package(monkeypatch, "tabpfn")
    other = tmp_path / "tabpfn-v2.5-regressor-v2.5_default.ckpt"
    other.write_bytes(b"not v2")
    model = TabPFNModel(INPUTS, training_window=28, device="cpu", weights=str(other))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # not the reference weights, once the card pins their hash
        with pytest.raises(RuntimeError, match="not v2"):
            records(model, market, test=("2023-03-13", "2023-03-13"))


# ---------------------------------------------------------------------------------------------------------------------------
# With the real weights (-m weights): verify on the CPU over five issue days
# ---------------------------------------------------------------------------------------------------------------------------


def _real_weights(cls):
    from epftoolbox2.models._cards import resolve_model_weights
    from epftoolbox2.models._weights import MissingWeightsError

    if importlib.util.find_spec(cls._card.extra) is None:
        pytest.skip(f"needs the [{cls._card.extra}] extra")
    try:
        resolve_model_weights(cls(INPUTS))
    except MissingWeightsError as error:
        pytest.skip(f"no weights: {error}")


@pytest.mark.integration
@pytest.mark.weights
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.filterwarnings("ignore:overflow encountered in sinh")
def test_verify_with_the_real_weights_on_the_cpu(cls, market):
    _real_weights(cls)
    days = ["2023-03-13", "2023-03-16", "2023-03-19", "2023-03-25", "2023-03-26"]
    report = suite.check_verify(
        lambda name: cls(INPUTS, training_window=56, device="cpu", name=name), market, INFO, days, items=6
    )
    assert report.ok
