"""`TorchModel`, `MLPHourlyModel` and `MLPDailyModel` (step 6l): a module of the user's on the library's trainer and rows.

The prototype's nine tests (`epftoolbox2-research/torchmodel/`, ported): the module's signature, a local or `__main__` class
refused by `save()`, the YAML round trip and the warning for a changed module, the day-level refusal, padded vmap members
equal to themselves alone, the +1000 look-ahead perturbation with its control, the spring switch, a bitwise resume, vmap
refusing `torch.nn.Dropout` and a loss by import path. Then the adapter suite (`tests/models/adapter_suite.py`), the presets
against `TorchModel(module=MLP)` to the bit, named blocks, predictor functions of `ctx`, the seeds, the seed mean, a diverged
seed, the runtime in the manifest, the TF32 and thread scope of a fit, and a pool of processes against one.

Needs the [torch] extra (the torch-extra CI job); skipped without it. The fits are tiny (a few epochs of 4-8 units on 28 days)
and run in this process unless a test says otherwise (`InProcessPool`, as in `tests/models/test_day_units.py`).
"""

import datetime
import json
import math
import subprocess
import sys
import textwrap
import warnings
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from epftoolbox2 import nets  # noqa: E402
from epftoolbox2.data.transformers import DailyProfileTransformer  # noqa: E402
from epftoolbox2.information_set import Availability, InformationSet, InformationSetError  # noqa: E402
from epftoolbox2.models import BaseModel, MLPDailyModel, MLPHourlyModel, TorchModel  # noqa: E402
from epftoolbox2.models import torch_model as tm  # noqa: E402
from epftoolbox2.models._calendar import FitUnit, unit_seed  # noqa: E402
from epftoolbox2.models._torch import runtime, trainer  # noqa: E402
from epftoolbox2.models._worker import _worker_init  # noqa: E402
from epftoolbox2.pipelines import ModelPipeline  # noqa: E402
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path  # noqa: E402
from tests.models import adapter_suite as suite  # noqa: E402
from tests.synthetic import calendar_shift, make_market_data  # noqa: E402

HOURS = (0, 8, 12, 18)
WINDOW = 28
FAST = dict(training_window=WINDOW, seeds=2, max_epochs=4, patience=2, lr=1e-2, batch_size=16)
WEEK = ("2023-03-20", "2023-03-26")  # one "7D" block, Monday to Sunday; the spring switch is its Sunday (Europe/Warsaw)
SUITE_TEST = ("2023-03-15", "2023-03-28")
RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
}
VERIFY_INFO = InformationSet(rules=RULES)


def slot(hour, freq="1h"):
    return f"h{hour:02d}" if freq == "1h" else f"s{hour * 4:02d}"


def day_columns(freq="1h"):
    """Day-level predictors: D-1's prices and the target day's load at `HOURS`, the target day's weekend flag."""
    price = [f"price_{slot(hour, freq)}_d-1" for hour in HOURS]
    load = [f"load_{slot(hour, freq)}_d+{{horizon}}" for hour in HOURS]
    return [*price, *load, "is_weekend_d+{horizon}"]


DAY = day_columns()
PERIOD = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "price_d-1"]
BLOCKS = {"price": DAY[:4], "load": DAY[4:8], "calendar": DAY[8:]}


def frame(start="2023-01-01", days=109, tz="Europe/Warsaw", freq="1h", horizon=2, seed=3):
    """`make_market_data` with daily profiles of price (lagged a day) and load (led by each horizon), by calendar days."""
    data = make_market_data(start=start, days=days, tz=tz, freq=freq, horizon=horizon, noise=5.0, seed=seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "DailyProfileTransformer: some local days", UserWarning)
        profiles = DailyProfileTransformer(columns=["price", "load"], freq=freq).transform(data[["price", "load"]])
    out = data.copy()
    for hour in HOURS:
        out[f"price_{slot(hour, freq)}_d-1"] = calendar_shift(profiles[f"price_{slot(hour, freq)}"], -1)
        for h in range(1, horizon + 1):
            out[f"load_{slot(hour, freq)}_d+{h}"] = calendar_shift(profiles[f"load_{slot(hour, freq)}"], h)
    return out


@pytest.fixture(scope="module")
def market():
    """1 January to 20 April 2023 in Warsaw, over the spring switch."""
    return frame()


class InProcessPool:
    """Stands in for the worker pool: the fits run in this process."""

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
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)


@pytest.fixture
def torch_state():
    """Put back torch's thread count and precision flags after a test that changes them."""
    threads = torch.get_num_threads()
    state = runtime.precision_state()
    yield
    runtime._restore_precision(state)
    torch.set_num_threads(threads)


# --- modules of a user's own ---------------------------------------------------------------------------------------------


class FlatNet(torch.nn.Module):
    """A flat module: `nets` layers, so a vmap stack gives each member its own numbers."""

    def __init__(self, n_inputs, n_outputs, width=8):
        super().__init__()
        self.net = torch.nn.Sequential(
            nets.Linear(n_inputs, width), nets.Activation("relu"), nets.Linear(width, n_outputs)
        )

    def forward(self, x):
        return self.net(x)


class BlockNet(torch.nn.Module):
    """Named blocks: one small layer per block, their outputs concatenated. `seen` keeps each forward's block widths and
    `binary` which blocks hold 0/1 values only (a scaler leaves binary columns as they are)."""

    seen: list = []
    binary: list = []

    def __init__(self, input_sizes, n_outputs, width=4):
        super().__init__()
        self.blocks = torch.nn.ModuleDict({name: nets.Linear(size, width) for name, size in input_sizes.items()})
        self.head = nets.Linear(width * len(input_sizes), n_outputs)

    def forward(self, blocks):
        BlockNet.seen.append({name: tuple(t.shape[-1:]) for name, t in blocks.items()})
        BlockNet.binary.append({name: bool(((t == 0) | (t == 1)).all()) for name, t in blocks.items()})
        return self.head(torch.relu(torch.cat([self.blocks[name](t) for name, t in blocks.items()], dim=-1)))


class TorchDropNet(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, 4), torch.nn.Dropout(0.1), torch.nn.Linear(4, n_outputs)
        )

    def forward(self, x):
        return self.net(x)


def squared(prediction, target):  # a user's loss, elementwise: the library masks and averages
    return (prediction - target) ** 2


def daily(name="Adapter", predictors=DAY, **params):
    return MLPDailyModel(predictors, **{"hidden": [8], "name": name, **FAST, **params})


def hourly(name="Adapter", predictors=PERIOD, **params):
    """One seed: 24 fits a unit, where a day model has one."""
    return MLPHourlyModel(predictors, **{"hidden": [4], "name": name, **FAST, "seeds": 1, **params})


def flat(name="Adapter", output="day", **params):
    names = DAY if output == "day" else PERIOD
    return TorchModel(module=FlatNet, predictors=names, output=output, name=name, **{**FAST, **params})


def blocks(name="Adapter", inputs=BLOCKS, **params):
    return TorchModel(module=BlockNet, inputs=inputs, name=name, **{**FAST, **params})


def run(model, data, test=WEEK, horizon=1, **kwargs):
    return model.run(data, *test, horizon=horizon, **kwargs)


def keyed(records):
    return {(r["run_date"], r["hour"], r["horizon"]): r["prediction"] for r in records}


def records_by_key(records):
    """Every record, keyed: a pool's jobs come back in no fixed order."""
    return {(r["run_date"], r["hour"], r["horizon"]): r for r in records}


def lines(path):
    return sorted(line for line in Path(path).read_text().splitlines() if line.strip())


# --- 1. the module's signature (prototype) --------------------------------------------------------------------------------


def test_the_module_contract_is_checked_when_the_model_is_built():
    """`__init__(n_inputs, n_outputs, **module_params)` for a flat list, `(input_sizes, n_outputs, ...)` for named blocks; a
    parameter the module doesn't take, or one it needs that module_params lacks, raises before any run."""
    with pytest.raises(TypeError, match="input_sizes"):
        TorchModel(module=FlatNet, inputs=BLOCKS)
    with pytest.raises(TypeError, match="n_inputs"):
        TorchModel(module=BlockNet, predictors=DAY)
    with pytest.raises(TypeError, match="n_inputs"):
        TorchModel(module=nets.Dropout, predictors=DAY)
    with pytest.raises(TypeError, match=r"takes no \['depth'\]"):
        TorchModel(module=FlatNet, predictors=DAY, module_params={"depth": 2})

    class Needs(torch.nn.Module):
        def __init__(self, n_inputs, n_outputs, width):
            super().__init__()

    with pytest.raises(TypeError, match=r"needs \['width'\]"):
        TorchModel(module=Needs, predictors=DAY)
    TorchModel(module=Needs, predictors=DAY, module_params={"width": 3})
    with pytest.raises(ValueError, match="not both"):
        TorchModel(module=BlockNet, predictors=DAY[:3], inputs=BLOCKS)
    with pytest.raises(ValueError, match="give predictors"):
        TorchModel(module=FlatNet)
    with pytest.raises(ValueError, match="output must be"):
        TorchModel(module=FlatNet, predictors=DAY, output="hourly")
    with pytest.raises(ValueError, match="loss must be"):
        TorchModel(module=FlatNet, predictors=DAY, loss=squared)
    with pytest.raises(ValueError, match="seeds"):
        TorchModel(module=FlatNet, predictors=DAY, seeds=0)
    with pytest.raises(TypeError, match="plain value"):
        TorchModel(module=FlatNet, predictors=DAY, module_params={"width": FlatNet})
    with pytest.raises(ValueError, match="scaler='standard' only"):
        TorchModel(module=FlatNet, predictors=PERIOD, output="period", scaler="asinh")
    with pytest.raises(ValueError, match="needs refit"):
        TorchModel(module=FlatNet, predictors=DAY, refit=None, **FAST)  # refused when built
    with pytest.raises(TypeError, match="module_params must be a dict"):
        TorchModel(module=FlatNet, predictors=DAY, module_params=[("width", 3)])
    period = TorchModel(module=FlatNet, predictors=PERIOD, output="period", refit=None, **FAST)
    with pytest.raises(ValueError, match="needs refit"):
        period.output = "day"
    assert (period.output, period._unit_kind) == ("period", "period")  # refused before anything was set
    day = TorchModel(module=FlatNet, predictors=DAY, **FAST)
    with pytest.raises(ValueError, match="needs refit"):
        day.refit = None
    with pytest.raises(ValueError, match="output must be"):
        day.output = "hour"
    assert (day.output, day._unit_kind, day.refit) == ("day", "day", "7D")


# --- 2. save() refuses a class without an import path (prototype) --------------------------------------------------------


def test_save_refuses_a_module_from_main_or_a_function_which_still_runs(market, tmp_path):
    def make():
        class Local(FlatNet):
            pass

        return Local

    local = TorchModel(module=make(), predictors=DAY, name="local", **FAST)
    with pytest.raises(TypeError, match="defined inside a function"):
        ModelPipeline().add_model(local).save(tmp_path / "local.yaml")
    main = type("MainNet", (FlatNet,), {"__module__": "__main__"})
    with pytest.raises(TypeError, match="script being run"):
        ModelPipeline().add_model(TorchModel(module=main, predictors=DAY, **FAST)).save(tmp_path / "main.yaml")
    assert not (tmp_path / "local.yaml").exists() and not (tmp_path / "main.yaml").exists()
    # It runs: a worker finds the class again by the key the model's kwargs hold, a key no import path can be.
    key = local._model_kwargs["module"]
    assert key.startswith("<unsaved>") and "<locals>" in key
    assert keyed(run(local, market)._results)


def scaled(scale):
    """A local class whose forecasts move with `scale`: two of them share a name, and so a key."""

    class Scaled(FlatNet):
        SCALE = scale

        def forward(self, x):
            return super().forward(x) * type(self).SCALE

    return Scaled


@pytest.mark.parametrize("pool", ["in process", "forked"])
def test_two_local_classes_of_one_name_each_forecast_with_their_own(market, monkeypatch, pool):
    """Classes without an import path share a key when they share a name (so a notebook cell run again resumes); each run hands
    its workers its own class, in this process or in a pool of forked processes."""
    if pool == "forked":
        monkeypatch.undo()
        monkeypatch.setenv("MAX_PROCESSES", "2")
    one, two = (TorchModel(module=scaled(s), predictors=DAY, name="s", **FAST) for s in (1.0, 2.0))
    assert one._model_kwargs["module"] == two._model_kwargs["module"]
    first, second, again = (keyed(run(m, market, test=("2023-03-20", "2023-03-21"))._results) for m in (one, two, one))
    assert first and first == again and first != second
    if pool == "forked":
        main = TorchModel(module=type("MainNet", (FlatNet,), {"__module__": "__main__"}), predictors=DAY, **FAST)
        assert keyed(run(main, market, test=("2023-03-20", "2023-03-21"))._results)


def test_a_module_reloaded_since_builds_its_new_class_and_the_old_model_its_old(tmp_path, monkeypatch):
    """A notebook edits its module's file and reloads it: a model built after the reload rebuilds the new class in a worker (its
    import path is looked up, never a registered class of that name), one built before still its own old class."""
    import importlib

    monkeypatch.syspath_prepend(str(tmp_path))
    source = textwrap.dedent(
        """
        import torch

        class Net(torch.nn.Module):
            V = {v}

            def __init__(self, n_inputs, n_outputs):
                super().__init__()
                self.version = type(self).V
        """
    )
    path = tmp_path / "user_nets_reload.py"
    path.write_text(source.format(v=1))
    module = importlib.import_module("user_nets_reload")
    old = TorchModel(module=module.Net, predictors=["a_d-1"], name="old")
    path.write_text(source.format(v=2) + "\n# edited\n")
    importlib.reload(module)
    new = TorchModel(module=module.Net, predictors=["a_d-1"], name="new")
    assert new._model_kwargs["module"] == "user_nets_reload:Net"
    assert (
        old._model_kwargs["module"] == "<unsaved>user_nets_reload:Net"
    )  # its class is no longer what the path imports
    assert old.module_hash != new.module_hash
    built = {m.name: TorchModel(predictors=[], **m._model_kwargs)._module_class()(3, 1).version for m in (new, old)}
    assert built == {"new": 2, "old": 1}
    # Only `<unsaved>` keys read the registry: a class registered under an import path never stands in for what it imports.
    monkeypatch.setitem(tm._UNIMPORTABLE, "epftoolbox2.nets:MLP", FlatNet)
    assert tm._resolve_module("epftoolbox2.nets:MLP", "test") is nets.MLP


def test_a_notebook_cell_run_again_resumes(market, tmp_path):
    """The cell that defines a `__main__` class runs again (a new class of the same name and source), then the run resumes
    into its save_dir: the same key, the same source hash, so no manifest mismatch."""
    namespace = {"__name__": "__main__", "torch": torch, "nets": nets}
    cell = textwrap.dedent(
        """
        class CellNet(torch.nn.Module):
            def __init__(self, n_inputs, n_outputs):
                super().__init__()
                self.l = nets.Linear(n_inputs, n_outputs)

            def forward(self, x):
                return self.l(x)
        """
    )
    save_dir = tmp_path / "nb"
    exec(cell, namespace)
    first = TorchModel(module=namespace["CellNet"], predictors=DAY, name="t", **FAST)
    ModelPipeline().add_model(first).run(market, "2023-03-20", "2023-03-21", horizon=1, save_dir=save_dir)
    exec(cell, namespace)
    again = TorchModel(module=namespace["CellNet"], predictors=DAY, name="t", **FAST)
    assert again.module is not first.module and again._model_kwargs == first._model_kwargs
    results = ModelPipeline.results_path(save_dir, again, "jsonl")
    before = lines(results)
    results.write_text("".join(f"{line}\n" for line in before[::2]))  # cut short
    ModelPipeline().add_model(again).run(market, "2023-03-20", "2023-03-21", horizon=1, save_dir=save_dir)
    assert lines(results) == before


# --- 3. YAML round trip and a changed module (prototype) ------------------------------------------------------------------


def test_the_yaml_round_trip_and_a_changed_module_warns(tmp_path):
    pipeline = (
        ModelPipeline()
        .add_model(TorchModel(module=nets.MLP, predictors=DAY, module_params={"hidden": (8,)}, name="t", **FAST))
        .add_model(daily(name="m"))
        .add_model(hourly(name="h", refit=None))
        .add_model(blocks(name="b", module_params={"width": 3}))
    )
    path = tmp_path / "p.yaml"
    pipeline.save(path)
    text = path.read_text()
    assert "module: epftoolbox2.nets:MLP" in text and f"module: {__name__}:BlockNet" in text and "&id" not in text
    saved = {entry["params"]["name"]: entry for entry in __import__("yaml").safe_load(text)["models"]}
    assert saved["t"]["params"]["module_hash"] == tm.module_source_hash(nets.MLP)
    assert saved["t"]["params"]["module_params"] == {"hidden": [8]}
    assert {"module", "module_hash", "output", "inputs"}.isdisjoint(saved["m"]["params"])
    assert saved["m"]["class"] == "MLPDailyModel" and saved["m"]["params"]["hidden"] == [8]
    assert saved["b"]["params"]["inputs"] == BLOCKS
    loaded = ModelPipeline.load(path)
    assert loaded.models[0].module is nets.MLP and loaded.models[3].module is BlockNet
    for original, again in zip(pipeline.models, loaded.models):
        assert type(again) is type(original) and again._model_kwargs == original._model_kwargs
    loaded.save(tmp_path / "again.yaml")
    assert (tmp_path / "again.yaml").read_text() == text
    with pytest.warns(UserWarning, match="changed since this configuration was saved"):
        TorchModel(module="epftoolbox2.nets:MLP", predictors=DAY, module_hash="0" * 16)
    path.write_text(text.replace(saved["t"]["params"]["module_hash"], "f" * 16, 1))
    with pytest.warns(UserWarning, match=r"epftoolbox2.nets:MLP changed"):
        ModelPipeline.load(path)


def test_the_module_hash_is_of_its_whole_file_and_its_package_version(tmp_path, monkeypatch):
    """The prototype hashed the class's own source and missed a change to `nets.Dropout`: the hash is the file's, so a change
    to another class of the file moves it, and so does the package's version."""
    monkeypatch.syspath_prepend(str(tmp_path))
    source = textwrap.dedent(
        """
        import torch

        class Helper(torch.nn.Module):
            SCALE = {scale}

        class Net(torch.nn.Module):
            def __init__(self, n_inputs, n_outputs):
                super().__init__()
        """
    )
    hashes = []
    for i, scale in enumerate((1, 2)):
        (tmp_path / f"user_nets_{i}.py").write_text(source.format(scale=scale))
        hashes.append(tm.module_source_hash(f"user_nets_{i}:Net"))  # by path, before any import
        module = __import__(f"user_nets_{i}")
        assert tm.module_source_hash(module.Net) == hashes[-1]  # the same once imported
    assert hashes[0] != hashes[1]
    before = tm.module_source_hash(nets.MLP)
    monkeypatch.setattr(tm, "_distribution_version", lambda top: "epftoolbox2 99.0")
    assert tm.module_source_hash(nets.MLP) != before


# --- 4. the day-level refusal (prototype) --------------------------------------------------------------------------------


@pytest.mark.parametrize("make", [daily, blocks], ids=["predictors", "blocks"])
def test_day_output_refuses_a_column_that_changes_within_the_day(market, make):
    """A per-period column raises, naming its daily profile as the fix, in a flat list and inside a named block alike (blocks
    are flattened for every check)."""
    if make is daily:
        model = daily(predictors=["price_d-1", *DAY[4:]])
    else:
        model = blocks(inputs={**BLOCKS, "price": ["price_d-1"]})
    with pytest.raises(ValueError, match=r"'price_d-1'.*price_h00_d-1 … price_h23_d-1"):
        run(model, market)


# --- 5. padded vmap members equal themselves alone (prototype) ------------------------------------------------------------


def test_in_a_padded_vmap_stack_each_seed_equals_itself_trained_alone(market, monkeypatch):
    """`stack="vmap", pad_to=4` with three seeds: each seed's forecasts are those of that seed trained alone in a stack of
    its own (same inputs, key and nominal rows), to the bit."""
    calls = []
    real = trainer.train_members

    def spy(factory, x, y, y_mask, members, **kw):
        result = real(factory, x, y, y_mask, members, **kw)
        calls.append((factory, x, y, y_mask, members, kw, result))
        return result

    monkeypatch.setattr(trainer, "train_members", spy)
    model = daily(stack="vmap", pad_to=4, seeds=3, hidden=[6], dropout=0.2)
    assert keyed(run(model, market, test=("2023-03-20", "2023-03-21"))._results)
    assert calls
    for factory, x, y, y_mask, members, kw, result in calls[:1]:
        assert kw["stack"] == "vmap" and kw["pad_to"] == 4 and len(members) == 3
        together = result.predict(x)
        for i, member in enumerate(members):
            alone = real(factory, x, y, y_mask, [member], **kw).predict(x)
            assert np.array_equal(alone[0], together[i])


# --- 6. the +1000 look-ahead perturbation and its control (prototype) -----------------------------------------------------


@pytest.mark.parametrize("output", ["day", "period"])
def test_targets_from_the_anchor_on_move_nothing_and_the_anchor_s_eve_moves_it(market, output):
    """Every target (price) from the block's anchor on +1000: the block's forecasts are bitwise the same; the control from
    the anchor's previous day on (its target is the newest a fit trains on) moves them. The inputs are left as they are."""

    def model():
        # Every training date trains (no early-stopping share), and with a squared loss a target's size reaches the weights
        # (an L1 gradient is the sign alone, which +1000 on a target above the forecast leaves as it is).
        return (daily if output == "day" else hourly)(early_stopping=0.0, loss="mse")

    base = keyed(run(model(), market)._results)
    assert base

    def moved(first_day):
        data = market.copy()
        later = data.index.tz_localize(None).normalize() >= pd.Timestamp(first_day)
        data.loc[later, "price"] += 1000.0
        return keyed(run(model(), data)._results)

    assert moved(WEEK[0]) == base
    control = moved("2023-03-19")
    assert control.keys() == base.keys() and all(control[k] != base[k] for k in base)


# --- 7. the spring switch (prototype) -------------------------------------------------------------------------------------


@pytest.mark.parametrize("make", [daily, hourly], ids=["day", "period"])
def test_the_spring_switch_target_day_has_23_forecasts_and_its_missing_slot_trains_on_nothing(
    market, make, monkeypatch
):
    """No forecast of 02:00 on 26 March (the hour the clock skips), 23 for that target day; a day unit's target slot of that
    hour is masked in training (never another period's value)."""
    masks = []
    real = TorchModel._fit_days

    def spy(self, x, y, y_mask):
        masks.append((y.copy(), y_mask.copy()))
        return real(self, x, y, y_mask)

    monkeypatch.setattr(TorchModel, "_fit_days", spy)
    records = run(make(), market, test=("2023-03-25", "2023-03-25"))._results
    assert sorted(r["hour"] for r in records) == [h for h in range(24) if h != 2]
    later = keyed(run(make(), market, test=("2023-03-30", "2023-03-30"))._results)
    if make is daily:
        y, mask = masks[1]  # the unit of 27 March, whose span holds the switch day's targets (issued 25 March)
        assert (~mask[:, 2]).sum() == 1 and np.isnan(y[~mask[:, 2], 2]).all()
    assert len(later) == 24


# --- 8. a bitwise resume (prototype), and the adapter suite ---------------------------------------------------------------


@pytest.fixture
def registered(monkeypatch):
    from epftoolbox2.pipelines.model_pipeline import COMPONENT_REGISTRY

    assert {"TorchModel", "MLPHourlyModel", "MLPDailyModel"} <= set(COMPONENT_REGISTRY["models"])


MAKERS = {"daily": daily, "hourly": hourly, "blocks": blocks}
# A period model fits 24 units where a day model fits one: a shorter run, still over a block boundary (anchors 13 and 20 March).
SUITE_RUNS = {"daily": (SUITE_TEST, 2), "hourly": (("2023-03-19", "2023-03-21"), 1), "blocks": (SUITE_TEST, 2)}


@pytest.mark.parametrize("kind", MAKERS)
def test_the_adapter_suite(market, tmp_path, monkeypatch, registered, kind):
    """YAML round trip, a loop and its shards, a resume after deleting 40 % of the records (bitwise), `engine="on"` (which
    takes none of them), the pre-flight without torch, and the card in `report.provenance()`."""
    make = MAKERS[kind]
    test, horizon = SUITE_RUNS[kind]
    suite.check_yaml_round_trip(make, market, tmp_path, test, horizon=horizon)
    variants = ({}, {"training_window": 21}, {"random_state": 1})
    suite.check_loop_and_shards(make, market, tmp_path / "loop", test, horizon=horizon, variants=variants)
    suite.check_resume(make, market, tmp_path / "resume", test, horizon=horizon)
    suite.check_engine(make, market, test, horizon=horizon)
    suite.check_preflight_without_extra(make, monkeypatch, market, test)
    suite.check_provenance(make, market, tmp_path / "provenance", test)


@pytest.mark.filterwarnings("ignore:overflow encountered in sinh")  # the control's 1e6 noise, mapped back by sinh
@pytest.mark.parametrize("kind", ["daily", "hourly", "blocks"])
def test_verify_is_ok_with_its_control_rerun_and_co_batch(market, kind):
    """`verify` on a block's first and last (spring) day: nothing moves under the probe, everything under the control, two
    runs agree, and a day model's six other block days replaced by noise move nothing (a block rule of 7 days)."""
    days = [WEEK[0], WEEK[1]]
    suite.check_verify(MAKERS[kind], market, VERIFY_INFO, days, items=None if kind == "hourly" else 6)


@pytest.mark.parametrize("kind", ["daily", "hourly"])
def test_the_calendar_counts_over_both_switches_and_at_15_minutes(kind):
    both = pd.concat([frame(start="2023-02-14", days=46, horizon=1), frame(start="2023-09-17", days=46, horizon=1)])
    suite.check_dst_counts(MAKERS[kind], both, "2023-03-26", "2023-10-29")
    quarter = frame(start="2023-01-01", days=60, freq="15min", horizon=1)
    names = day_columns("15min") if kind == "daily" else PERIOD

    def make(name, **params):
        return MAKERS[kind](name=name, predictors=names, **params)

    suite.check_quarter_hours(make, quarter, ("2023-02-20", "2023-02-20"))


# --- 9. vmap refuses torch's dropout; a loss by import path (prototype) ---------------------------------------------------


def test_vmap_refuses_torch_dropout_before_any_fit_and_a_loss_by_import_path_is_used(market, monkeypatch):
    fits = []
    real = trainer.train_members
    monkeypatch.setattr(trainer, "train_members", lambda *a, **k: fits.append(1) or real(*a, **k))
    model = TorchModel(module=TorchDropNet, predictors=DAY, stack="vmap", pad_to=2, **FAST)
    with pytest.raises(ValueError, match="nets.Dropout"):
        run(model, market)
    assert fits == []  # refused in the parent, not as each fit's fit_error
    assert keyed(run(TorchModel(module=TorchDropNet, predictors=DAY, stack="loop", **FAST), market)._results)
    by_path = keyed(run(flat(loss=f"{__name__}:squared"), market)._results)
    assert by_path == keyed(run(flat(loss="mse"), market)._results)
    assert by_path != keyed(run(flat(loss="l1"), market)._results)
    with pytest.raises(ValueError, match="Cannot import the loss"):
        run(flat(loss=f"{__name__}:no_such_loss"), market)


# --- the presets, blocks, predictor functions, seeds --------------------------------------------------------------------


@pytest.mark.parametrize("refit", ["7D", None])
def test_the_presets_are_torch_model_with_nets_mlp_to_the_bit(market, refit):
    """`MLPDailyModel(...)` gives `TorchModel(module=MLP, output="day", ...)`'s forecasts and `MLPHourlyModel(...)`
    `TorchModel(module=MLP, output="period", ...)`'s, record for record."""
    if refit is not None:
        preset = run(daily(dropout=0.1), market, horizon=2)._results
        generic = TorchModel(
            module=nets.MLP, predictors=DAY, module_params={"hidden": [8], "dropout": 0.1}, output="day", **FAST
        )
        assert records_by_key(preset) == records_by_key(run(generic, market, horizon=2)._results)
    test = ("2023-03-20", "2023-03-21")
    preset = run(hourly(refit=refit), market, test=test, horizon=2)._results
    generic = TorchModel(
        module=nets.MLP,
        predictors=PERIOD,
        module_params={"hidden": [4]},
        output="period",
        refit=refit,
        **{**FAST, "seeds": 1},
    )
    assert preset and records_by_key(preset) == records_by_key(run(generic, market, test=test, horizon=2)._results)


def test_named_blocks_reach_the_module_by_name_and_their_split_is_in_the_manifest(market, tmp_path):
    """The module gets one tensor per block, in the order given; the flattened list is what every check reads (an input the
    information set calls late is refused); two splits of the same columns are different models to a resume."""
    BlockNet.seen.clear()
    BlockNet.binary.clear()
    model = blocks()
    assert model.predictors == DAY
    assert keyed(run(model, market)._results)
    assert {"price": (4,), "load": (4,), "calendar": (1,)} in BlockNet.seen
    # Each block holds its own columns: the weekend flag (binary, left unscaled) is the calendar block's alone.
    assert {"price": False, "load": False, "calendar": True} in BlockNet.binary
    late = blocks(inputs={**BLOCKS, "calendar": ["is_weekend_d+{horizon}", "price_h08_d+{horizon}"]})
    data = market.assign(**{"price_h08_d+1": market["price_h08_d-1"]})
    with pytest.raises(InformationSetError, match="price_h08_d"):
        ModelPipeline(information_set=VERIFY_INFO, availability="raise").add_model(late).run(data, *WEEK, horizon=1)
    save_dir = tmp_path / "split"
    ModelPipeline().add_model(blocks(name="b")).run(market, *WEEK, horizon=1, save_dir=save_dir)
    params = json.loads(manifest_path(ModelPipeline.results_path(save_dir, blocks(name="b"), "jsonl")).read_text())
    assert params["model_params"]["inputs"] == BLOCKS
    other = blocks(name="b", inputs={"price": DAY[:2], "rest": DAY[2:]}, module_params={"width": 4})
    with pytest.raises(ManifestMismatchError, match="inputs"):
        ModelPipeline().add_model(other).run(market, *WEEK, horizon=1, save_dir=save_dir)


def by_issue_day(ctx):
    """One input fewer from Thursday on, another list at horizon 2: a list may change with both."""
    names = DAY[:-1] if ctx.issue_day.weekday() >= 3 else DAY
    return names if ctx.horizon == 1 else names[1:]


def by_period(ctx):
    return DAY if ctx.period < 12 else DAY[1:]


def test_a_predictor_function_may_vary_by_horizon_and_issue_day_not_by_the_period_of_a_day_model(market, monkeypatch):
    widths = []
    real = TorchModel._fit_days

    def spy(self, x, y, y_mask):
        widths.append(x.shape[1])
        return real(self, x, y, y_mask)

    monkeypatch.setattr(TorchModel, "_fit_days", spy)
    records = run(
        daily(predictors=by_issue_day, refit="1D"), market, test=("2023-03-21", "2023-03-24"), horizon=2
    )._results
    assert sorted(set(widths)) == [7, 8, 9]
    assert len({r["predictor_list"] for r in records}) == 4
    with pytest.raises(ValueError, match="not with the period"):
        run(daily(predictors=by_period), market)
    period_records = run(hourly(predictors=lambda ctx: PERIOD if ctx.period < 12 else PERIOD[1:]), market)._results
    assert len({r["predictor_list"] for r in period_records}) == 2  # a period model takes one per period


def test_each_seed_is_the_unit_s_hash_and_never_the_model_s_name(market, monkeypatch):
    """Member s of a fit is seeded by `unit_seed(random_state, unit, s)`: the day unit (None, h, anchor, list, member), for
    `refit=None` the period unit of the issue day. Another name gives the same forecasts; another `random_state` other ones."""
    keys = []
    real = trainer.train_members

    def spy(factory, x, y, y_mask, members, **kw):
        keys.append([m.seed_key for m in members])
        return real(factory, x, y, y_mask, members, **kw)

    monkeypatch.setattr(trainer, "train_members", spy)
    first = keyed(run(daily(seeds=3), market, test=("2023-03-21", "2023-03-21"))._results)
    unit = FitUnit(None, 1, datetime.date(2023, 3, 20), None, 0)
    assert keys == [[unit_seed(0, unit, s) for s in range(3)]]
    assert keyed(run(daily(name="other", seeds=3), market, test=("2023-03-21", "2023-03-21"))._results) == first
    assert keyed(run(daily(seeds=3, random_state=1), market, test=("2023-03-21", "2023-03-21"))._results) != first
    keys.clear()
    run(hourly(refit=None, seeds=1), market, test=("2023-03-21", "2023-03-22"))
    days = (datetime.date(2023, 3, 21), datetime.date(2023, 3, 22))
    assert sorted(k for (k,) in keys) == sorted(
        unit_seed(0, FitUnit(p, 1, d, None, 0), 0) for d in days for p in range(24)
    )


class TwoMembers(BaseModel):
    """A numpy day model of two members that predicts z (the mean of its training targets per output) and z + 1, or one of
    them (`only`): the forecast of both must be the mean of each one's forecast in target units, not of their z."""

    _unit_kind = "day"

    def __init__(self, predictors=DAY, training_window=WINDOW, only=None, **kw):
        kw.setdefault("refit", "7D")
        super().__init__(predictors, training_window, **kw)
        self.only = only

    @property
    def _model_kwargs(self):
        return {"only": self.only}

    def _fit_predict(self, train_x, train_y, test_x):  # pragma: no cover - a day unit fits through _fit_days
        raise NotImplementedError

    def _fit_days(self, x, y, y_mask):
        return np.nanmean(np.where(y_mask, y, np.nan), axis=0)

    def _predict_days(self, fitted, rows):
        z = np.broadcast_to(fitted, (len(rows), len(fitted)))
        both = np.stack([z, z + 1.0])
        return both if self.only is None else both[self.only]


def test_a_forecast_of_several_members_is_their_mean_in_target_units(market):
    both = keyed(TwoMembers().run(market, *WEEK, horizon=1)._results)
    first = keyed(TwoMembers(only=0).run(market, *WEEK, horizon=1)._results)
    second = keyed(TwoMembers(only=1).run(market, *WEEK, horizon=1)._results)
    assert both.keys() == first.keys() == second.keys()
    assert all(both[k] == math.fsum([first[k], second[k]]) / 2 for k in both)


def test_a_seed_that_diverged_makes_its_fit_s_forecasts_fit_errors_never_a_mean_of_the_others(
    market, monkeypatch, tmp_path
):
    real = trainer.train_members
    calls = []

    def first_fails(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            failed = result.failed.copy()
            failed[1] = True
            return trainer.TrainResult(result.predict, result.best_epoch, failed, result.stats)
        return result

    monkeypatch.setattr(trainer, "train_members", first_fails)
    report = ModelPipeline().add_model(daily()).run(market, "2023-03-25", "2023-03-28", horizon=1, save_dir=tmp_path)
    records = [json.loads(line) for line in lines(ModelPipeline.results_path(tmp_path, daily(), "jsonl"))]
    skipped = [r for r in records if r["status"] != "ok"]
    assert skipped and {r["run_date"] for r in skipped} == {"2023-03-25", "2023-03-26"}  # the first block's
    assert {r["status"] for r in skipped} == {"fit_error"} and all(
        "seed [1] of 2 diverged" in r["detail"] for r in skipped
    )
    assert report.forecast_counts().iloc[0]["produced"] == 48


# --- the manifest, the scope of a fit, processes --------------------------------------------------------------------------


def test_the_runtime_is_in_the_manifest_and_a_resume_on_other_settings_refuses(market, tmp_path, monkeypatch):
    save_dir = tmp_path / "run"
    ModelPipeline().add_model(daily()).run(market, *WEEK, horizon=1, save_dir=save_dir)
    stored = json.loads(manifest_path(ModelPipeline.results_path(save_dir, daily(), "jsonl")).read_text())
    runtime_entry = stored["runtime"]
    assert runtime_entry["packages"]["torch"] == torch.__version__
    expected = {"torch": torch.__version__, "device": "cpu", "dtype": "float32", "tf32": False, "threads_per_fit": 1}
    assert {k: runtime_entry[k] for k in expected} == expected
    assert (runtime_entry["stack"], runtime_entry["pad_to"]) == ("loop", None) and runtime_entry["cpu_capability"]
    assert stored["model_params"]["module_hash"] == tm.module_source_hash(nets.MLP) and stored["unit_kind"] == "day"
    assert stored["card"]["family"] == "torch" and stored["card"]["verdicts"]["test"] == "clean"
    with pytest.raises(ManifestMismatchError, match="stack"):
        ModelPipeline().add_model(daily(stack="vmap", pad_to=2)).run(market, *WEEK, horizon=1, save_dir=save_dir)
    with pytest.raises(ManifestMismatchError, match="tf32"):
        ModelPipeline().add_model(daily(tf32=True)).run(market, *WEEK, horizon=1, save_dir=save_dir)
    monkeypatch.setattr(tm, "_distribution_version", lambda top: "epftoolbox2 99.0")  # nets.py of another release
    with pytest.raises(ManifestMismatchError, match="module_hash"):
        ModelPipeline().add_model(daily()).run(market, *WEEK, horizon=1, save_dir=save_dir)


class ScopeProbe(FlatNet):
    """Records torch's thread count and matmul precision in each forward."""

    seen: list = []

    def forward(self, x):
        ScopeProbe.seen.append((torch.get_num_threads(), torch.get_float32_matmul_precision()))
        return super().forward(x)


def test_a_fit_runs_on_one_torch_thread_with_tf32_off_and_puts_both_back(market, torch_state):
    """The caller's settings (4 threads, TF32 on) don't reach a fit, and they are back after the run."""
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    ScopeProbe.seen.clear()
    run(TorchModel(module=ScopeProbe, predictors=DAY, **FAST), market, test=("2023-03-20", "2023-03-20"))
    assert ScopeProbe.seen and set(ScopeProbe.seen) == {(1, "highest")}
    assert (torch.get_num_threads(), torch.get_float32_matmul_precision()) == (4, "high")


@pytest.mark.parametrize("make", [daily, hourly], ids=["day", "period"])
def test_a_pool_of_forked_processes_gives_this_process_s_forecasts(market, make, monkeypatch, tmp_path):
    """Two worker processes, each pinned to one torch thread by the pool initialiser, against the fits in this process: the
    same records, line for line (the loop at one thread is the same in any process)."""

    def into(save_dir):
        ModelPipeline().add_model(make()).run(market, *SUITE_TEST, horizon=2, save_dir=save_dir)

    into(tmp_path / "here")
    monkeypatch.undo()
    monkeypatch.setenv("MAX_PROCESSES", "2")
    monkeypatch.setenv("THREADS_PER_PROCESS", "3")  # not used by a torch model: one fit at a time per process
    into(tmp_path / "pool")
    assert lines(ModelPipeline.results_path(tmp_path / "here", make(), "jsonl"))
    assert lines(ModelPipeline.results_path(tmp_path / "pool", make(), "jsonl")) == lines(
        ModelPipeline.results_path(tmp_path / "here", make(), "jsonl")
    )


def test_without_torch_the_models_build_save_and_load_and_the_run_names_the_extra(tmp_path):
    """With torch missing, the three classes build, save and load from YAML (a module path stays a path, hashed from its
    file without an import), and `run` raises `MissingExtraError` for [torch] before any fit."""
    code = textwrap.dedent(
        f"""
        import sys
        sys.modules["torch"] = None  # as if the extra were missing
        import pandas as pd
        from epftoolbox2._extras import MissingExtraError
        from epftoolbox2.models import MLPDailyModel, MLPHourlyModel, TorchModel
        from epftoolbox2.pipelines import ModelPipeline
        models = [MLPDailyModel(["a_d-1"]), MLPHourlyModel(["a_d-1"]), TorchModel("epftoolbox2.nets:MLP", ["a_d-1"], name="t")]
        assert models[2].module == "epftoolbox2.nets:MLP" and len(models[2].module_hash) == 16
        pipeline = ModelPipeline()
        for model in models:
            pipeline.add_model(model)
        pipeline.save({str(tmp_path / "p.yaml")!r})
        loaded = ModelPipeline.load({str(tmp_path / "p.yaml")!r})
        assert [m._model_kwargs for m in loaded.models] == [m._model_kwargs for m in models]
        index = pd.date_range("2024-01-01", periods=24 * 30, freq="h", tz="Europe/Berlin")
        frame = pd.DataFrame({{"price": 1.0, "a_d-1": 1.0}}, index=index)
        try:
            loaded.run(frame, "2024-01-20", "2024-01-21", horizon=1)
        except MissingExtraError as exc:
            assert exc.extra == "torch", exc
        else:
            raise AssertionError("ran without torch")
        assert "torch" not in [m for m in sys.modules if sys.modules[m] is not None and m.startswith("torch")]
        print("ok")
        """
    )
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root)
    assert out.stdout.strip().splitlines()[-1:] == ["ok"], out.stderr[-3000:]


def test_the_steps_of_an_epoch_come_from_a_full_window_and_each_window_of_a_list_fits_on_its_own(market, monkeypatch):
    """A fit whose span starts before the data trains as many steps as a full window (nominal rows: the window's days); an
    expanding window's are its rows. With a list of windows each is its own fit, its first member the model with that window
    alone."""
    calls = []
    real = trainer.train_members

    def spy(factory, x, y, y_mask, members, **kw):
        calls.append((len(x), kw["nominal_rows"]))
        return real(factory, x, y, y_mask, members, **kw)

    monkeypatch.setattr(trainer, "train_members", spy)
    gappy = market.copy()  # three dates of the span without an input, so the fit has 25 rows of a 28-day window
    gappy.loc[
        gappy.index.tz_localize(None).normalize().isin(pd.to_datetime(["2023-03-01", "2023-03-05", "2023-03-09"])),
        DAY[0],
    ] = np.nan
    run(daily(), gappy)
    run(daily(training_window="expanding"), market)
    run(daily(), market)
    (short, nominal), (rows, expanding), (full, full_nominal) = calls
    assert short == WINDOW - 3 and nominal == WINDOW
    assert expanding == rows > WINDOW
    assert full == WINDOW and full_nominal == WINDOW
    both = run(daily(training_window=[21, 28]), market)._results
    alone = keyed(run(daily(training_window=21), market)._results)
    assert all(len(r["member_predictions"]) == 2 for r in both)
    assert {(r["run_date"], r["hour"], r["horizon"]): r["member_predictions"][0] for r in both} == alone


def test_a_period_network_trains_on_its_target_s_invariant(market, monkeypatch):
    """The scaler hands a period unit its standardised targets; the network trains on their Invariant, which is the raw
    targets' asinh((y - median) / MAD) (median and MAD move with an affine map), as a day unit's outputs do."""
    seen = []
    real = trainer.train_members

    def spy(factory, x, y, y_mask, members, **kw):
        seen.append(np.asarray(y)[:, 0].copy())
        return real(factory, x, y, y_mask, members, **kw)

    monkeypatch.setattr(trainer, "train_members", spy)
    records = run(hourly(predictors=PERIOD[:3]), market, test=("2023-03-21", "2023-03-21"))._results
    assert records and len(seen) == 24
    local = market.index.tz_localize(None)
    raw = market["price"].to_numpy()
    # Period 0's training targets: the prices at 00:00 of the 28 days up to the anchor's eve (anchor Monday 20 March, newest
    # target 19 March), each the target of the row of the day before.
    days = pd.date_range("2023-02-20", "2023-03-19")
    targets = np.array([raw[(local == day).argmax()] for day in days])
    median = np.median(targets)
    mad = np.median(np.abs(targets - median)) / 0.6744897501960817
    np.testing.assert_allclose(np.sort(seen[0]), np.sort(np.arcsinh((targets - median) / mad)), rtol=0, atol=1e-12)


def test_a_period_forecast_is_its_seeds_mean_in_target_units():
    """Each seed's output mapped back by the target's Invariant, then their mean: not the Invariant of their mean."""
    from epftoolbox2.scalers.target import MultiTargetAsinh

    class Members:
        def predict(self, rows):
            return np.array([[[-1.0]], [[2.0]]])

    target = MultiTargetAsinh().fit(np.arange(10.0)[:, None])
    fitted = tm._Fitted(Members(), "cpu", target)
    prediction, coefficients = hourly()._predict_fitted(fitted, np.zeros((1, 3)))
    expected = math.fsum(float(target.inverse(np.array(z), 0)) for z in (-1.0, 2.0)) / 2
    assert prediction == expected and coefficients == []
    assert prediction != float(target.inverse(np.array(0.5), 0))


def test_one_fit_per_process_every_cpu_by_default_and_one_process_on_a_gpu(monkeypatch):
    from epftoolbox2.models.base import _usable_cpu_count

    monkeypatch.delenv("MAX_PROCESSES")
    monkeypatch.setenv("THREADS_PER_PROCESS", "4")  # not a torch model's
    assert daily()._executor_config() == (1, _usable_cpu_count())
    monkeypatch.setattr(runtime, "resolve_device", lambda device: torch.device("cuda", 0))
    assert daily()._executor_config() == (1, 1)
    monkeypatch.setenv("MAX_PROCESSES", "3")
    assert daily()._executor_config() == (1, 3)


def test_a_network_is_refused_by_pooled_model_when_built_and_by_a_residual_model_as_its_learner(market):
    """`PooledModel` fits outside the library's fit units, which a network's seeds come from, and a `ResidualModel` runs its
    learner in its own pool without the network's executor config and torch threads: both refuse a network when built (a
    network as the base is a model of its own run). The run-time guard stays for any other caller without a fit unit."""
    from epftoolbox2.models import PooledModel, ResidualModel, RidgeModel

    for make in (hourly, daily):
        with pytest.raises(ValueError, match="can't be pooled over zones"):
            PooledModel(make(), name="pooled")
    learner = hourly(predictors=["price_d-1", "base_fc_d+{horizon}"])
    with pytest.raises(ValueError, match="isn't supported as a learner"):
        ResidualModel(RidgeModel(["price_d-1"], training_window=40, name="b"), learner, training_window=56, name="res")
    with pytest.raises(RuntimeError, match="not PooledModel"):
        hourly()._fit_model(np.zeros((3, 2)), np.zeros(3))


def test_a_class_is_saved_by_its_import_path_only_where_its_component_lists_it():
    """`TorchModel._CLASS_PARAMS` names `module`, which `ModelPipeline._serialize_param` writes by import path; `to_yaml_value`
    refuses a class anywhere else (a DataPipeline parameter, a model attribute that isn't listed)."""
    from epftoolbox2.pipelines._yaml import to_yaml_value

    with pytest.raises(TypeError, match="callable"):
        to_yaml_value(nets.MLP, "Some.param")
    pipeline = ModelPipeline().add_model(flat(name="listed"))
    assert pipeline.to_dict()["models"][0]["params"]["module"] == f"{__name__}:FlatNet"
    stray = flat(name="stray")
    stray.helper = FlatNet  # a public class attribute no `_CLASS_PARAMS` lists
    with pytest.raises(TypeError, match="stray.*helper"):
        ModelPipeline().add_model(stray).to_dict()


# --- review 6l: seeds from a list's content, early stopping by dates, the device's provenance -----------------------------


def from_wednesday(ctx):
    """Every input before Wednesday 22 March, one fewer from then on: the plan numbers the two lists by first appearance."""
    return DAY if ctx.issue_day < datetime.date(2023, 3, 22) else DAY[:-1]


def by_month(ctx):
    return DAY if ctx.issue_day.month == 2 else DAY[:-1]


def test_a_predictor_function_s_forecasts_are_the_same_whatever_days_the_run_holds(market):
    """A unit is seeded by its list's content, not by the id the plan gave it: a later test start, a validation window before
    the test, or Fridays only give the common forecasts to the bit (118 of 118 moved when the id seeded them)."""
    whole = keyed(run(daily(predictors=from_wednesday), market, test=("2023-03-20", "2023-03-26"))._results)
    later = keyed(run(daily(predictors=from_wednesday), market, test=("2023-03-22", "2023-03-26"))._results)
    common = set(whole) & set(later)
    assert len(common) >= 100 and all(whole[k] == later[k] for k in common)

    from epftoolbox2.results.windows import RunWindows

    validation, test = ("2023-02-20", "2023-02-26"), ("2023-03-20", "2023-03-26")
    both = daily(predictors=by_month).run(market, horizon=1, windows=RunWindows.resolve(validation, test))._results
    in_both = {(r["run_date"], r["hour"], r["horizon"]): r["prediction"] for r in both if r.get("window") == "test"}
    alone = keyed(run(daily(predictors=by_month), market, test=test)._results)
    assert alone and in_both == alone

    fridays = keyed(
        run(daily(predictors=from_wednesday), market, test=("2023-03-20", "2023-03-26"), issue_days=["friday"])._results
    )
    assert fridays and all(fridays[k] == whole[k] for k in fridays)


def test_a_refit_none_fit_of_a_window_list_seeds_each_window_as_its_own_member(market, monkeypatch):
    keys = []
    real = trainer.train_members

    def spy(factory, x, y, y_mask, members, **kw):
        keys.append(members[0].seed_key)
        return real(factory, x, y, y_mask, members, **kw)

    monkeypatch.setattr(trainer, "train_members", spy)
    run(hourly(refit=None, training_window=[21, 28]), market, test=("2023-03-21", "2023-03-21"))
    day = datetime.date(2023, 3, 21)
    expected = {unit_seed(0, FitUnit(p, 1, day, None, m), 0) for p in range(24) for m in (0, 1)}
    assert set(keys) == expected and len(keys) == 48


@pytest.fixture(scope="module")
def autumn():
    """1 September to 9 November 2023 in Warsaw, over the autumn switch (29 October: 02:00 twice)."""
    return frame(start="2023-09-01", days=70, seed=4)


def test_early_stopping_holds_out_dates_and_a_repeated_hour_s_rows_together(autumn, monkeypatch):
    """A period unit whose span holds the autumn switch has two rows of period 2 on 29 October: the early-stopping share is of
    dates, and both rows fall on one side; on other periods (a row a date) the split is the rows' as before."""
    splits = []
    real_split = trainer.member_split

    def spy(seed_key, train_rows, early_stopping, groups=None):
        fit, val = real_split(seed_key, train_rows, early_stopping, groups)
        splits.append((np.asarray(groups), fit, val, real_split(seed_key, train_rows, early_stopping)))
        return fit, val

    monkeypatch.setattr(trainer, "member_split", spy)
    run(hourly(predictors=PERIOD[:3], seeds=4), autumn, test=("2023-11-06", "2023-11-06"))  # its span holds 29 October
    doubled = [s for s in splits if len(np.unique(s[0])) < len(s[0])]
    assert len(doubled) == 4  # period 2's unit, one split per seed
    for groups, fit, val, _ in doubled:
        date = np.flatnonzero(np.bincount(groups - groups.min()) == 2)[0] + groups.min()
        twin = np.flatnonzero(groups == date)
        assert set(twin) <= set(fit) or set(twin) <= set(val)
        assert len(set(groups[val])) == round(0.2 * len(np.unique(groups)))
    single = [s for s in splits if len(np.unique(s[0])) == len(s[0])]
    assert single and all(np.array_equal(f, rf) and np.array_equal(v, rv) for _, f, v, (rf, rv) in single)


def test_the_device_s_model_is_recorded_and_a_resume_on_another_warns(market, tmp_path, monkeypatch):
    save_dir = tmp_path / "device"
    ModelPipeline().add_model(daily()).run(market, *WEEK, horizon=1, save_dir=save_dir)
    stored = json.loads(manifest_path(ModelPipeline.results_path(save_dir, daily(), "jsonl")).read_text())
    assert stored["runtime"]["device_name"] == runtime.cpu_name()
    monkeypatch.setattr(runtime, "cpu_name", lambda: "Another CPU 9000")
    with pytest.warns(UserWarning, match="runtime.device_name"):
        ModelPipeline().add_model(daily()).run(market, *WEEK, horizon=1, save_dir=save_dir)
