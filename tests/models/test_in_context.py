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
    assert card.runtime[0].startswith(card.extra) and "torch" in card.runtime
    if cls is TabPFNModel:
        assert card.weights.repo == "Prior-Labs/TabPFN-v2-reg" and list(card.weights.files) == [
            "tabpfn-v2-regressor.ckpt"
        ]
        assert card.runtime[0] == "tabpfn>=6,<10" and "PriorLabs-TabPFN" in card.attribution
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


def test_importing_building_and_saving_imports_no_model_package(tmp_path):
    """Dawid's lean-package condition: importing the package, building a calendar data pipeline and fetching from it, and
    building, saving and loading every new model class imports none of torch, the families' packages, autogluon,
    huggingface_hub, transformers or safetensors, and downloads nothing."""
    code = (
        "import sys\n"
        "import pandas as pd\n"
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
        f"heavy = {HEAVY!r}\n"
        "found = sorted(m for m in sys.modules if m.split('.')[0] in heavy)\n"
        "assert not found, found\n"
    )
    env = {**os.environ, "EPFTOOLBOX2_WEIGHTS_DIR": str(tmp_path / "no-weights")}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path, env=env)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "no-weights").exists(), "something wrote into the weights cache"


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
    assert manifest["model_params"]["block_predict"] is True and manifest["model_params"]["device"] == "cpu"
    assert manifest["card"]["family"] == cls._card.family and manifest["weights"]["reference"] is True
    manifest["runtime"]["device"] = "cuda"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ManifestMismatchError, match="runtime.device"):
        run(make(), market, save_dir=tmp_path)


@pytest.mark.parametrize("cls", CLASSES)
def test_processes_one_fit_each_and_six_on_a_gpu(cls, monkeypatch):
    from epftoolbox2.models.base import _usable_cpu_count

    class Device:
        def __init__(self, kind):
            self.type = kind
            self.index = 0

    model = cls(INPUTS)
    monkeypatch.setattr(cls, "_resolved_device", lambda self: Device("cpu"))
    monkeypatch.setenv(
        "THREADS_PER_PROCESS", "4"
    )  # a fit's torch settings are its process's: one task thread, whatever
    monkeypatch.delenv("MAX_PROCESSES")
    assert model._executor_config() == (1, max(1, _usable_cpu_count()))
    monkeypatch.setattr(cls, "_resolved_device", lambda self: Device("cuda"))
    assert model._executor_config() == (1, min(6, max(1, _usable_cpu_count())))
    monkeypatch.setenv("MAX_PROCESSES", "3")
    assert model._executor_config() == (1, 3)


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
    assert Path(made["model_path"]).name == name and made["device"] == "cpu" and made["random_state"] == 0
    if cls is TabPFNModel:
        assert made["n_estimators"] == 4 and made["n_preprocessing_jobs"] == 1
    else:
        assert (made["n_estimators"], made["batch_size"], made["allow_auto_download"]) == (8, 8, False)
        assert len(seen["loads"]) == 1, "the checkpoint is read once per process"
    assert len(seen["fits"]) == 24


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
    packages = manifest["runtime"]["packages"]
    assert packages["torch"] == torch.__version__ and packages[cls._card.extra]
    assert manifest["runtime"]["cpu_capability"] == torch.backends.cpu.get_cpu_capability()
    assert manifest["runtime"]["device_name"]


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
