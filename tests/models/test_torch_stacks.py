"""Torch stacks (step 6m, `models/_torch/groups.py`): the networks of an `add_model` loop trained together on one device.

Speed only: each model's records are those of its own run, to the bit (CPU, padded vmap stacks), alone, in a 48-model loop, in
another order and in `models_per_task`-style shards; for day and period outputs, `refit=None`, a list of training windows and
named blocks; after a resume and a crash; with models that run alone next to them, a session class, a member that diverged, a
stack that raised and the linear engine. Which models stack and under what key; `TORCH_STACKS=off`.

Needs the [torch] extra (the torch-extra CI job); skipped without it. The fits are tiny (a few epochs of small networks on 28
days) and run in this process unless a test says otherwise (`InProcessPool`); `-m gpu` holds the H100 check.
"""

import concurrent.futures
import datetime
import json
import math
import random
import warnings
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from epftoolbox2 import nets  # noqa: E402
from epftoolbox2.models import LEARModel, MLPDailyModel, MLPHourlyModel, OLSModel, RidgeModel, TorchModel  # noqa: E402
from epftoolbox2.models import _engine, _engine_worker  # noqa: E402
from epftoolbox2.models._torch import groups, runtime  # noqa: E402
from epftoolbox2.models._worker import _worker_init  # noqa: E402
from epftoolbox2.pipelines import ModelPipeline  # noqa: E402
from epftoolbox2.results.manifest import manifest_path  # noqa: E402
from tests.models.test_torch_model import BLOCKS, DAY, PERIOD, FlatNet, frame  # noqa: E402

FAST = dict(training_window=28, seeds=2, max_epochs=4, patience=2, lr=1e-2, batch_size=16, stack="vmap", pad_to=4)
TEST = WEEK = ("2023-03-20", "2023-03-26")  # one "7D" block, Monday to Sunday (the spring switch is its Sunday)
LOSS_BOOM = "tests.models.test_torch_stacks:boom"


@pytest.fixture(scope="module")
def market():
    """1 January to 20 April 2023 in Warsaw, over the spring switch."""
    return frame()


class InProcessPool:
    """Stands in for a pool, a model's own (`_worker_init`), an engine group's (`_engine_load`) or a stack group's
    (`groups._stack_load`), without their native-thread limits: the fits run here."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        loaders = {groups._stack_init: groups._stack_load, _engine_worker._engine_init: _engine_worker._engine_load}
        loaders.get(initializer, _worker_init)(*initargs)

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
def single_process(monkeypatch, request):
    """One process, the pools in this one; and no stack may fall back to its calls alone unless the test asks for it
    (`fallbacks_expected`): a stack that raised would still give each model's own numbers, so the equality tests alone
    wouldn't see it."""
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.delenv(groups.TORCH_STACKS_VAR, raising=False)
    monkeypatch.delenv(groups.STACK_MEMBERS_VAR, raising=False)
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)
    monkeypatch.setattr(_engine, "ProcessPoolExecutor", InProcessPool)
    monkeypatch.setattr(groups, "ProcessPoolExecutor", InProcessPool)
    monkeypatch.setattr(groups, "STATS", {"fallbacks": 0, "alone": 0})
    yield
    if "fallbacks_expected" not in request.fixturenames:
        assert groups.STATS["fallbacks"] == 0, "a stack raised and its calls were trained alone"


@pytest.fixture
def fallbacks_expected():
    """This test makes a stack raise on purpose."""


@pytest.fixture
def vouched(monkeypatch):
    """The test's own modules stack on the CPU too: they are made of `nets` layers and operations exact in every element
    (products, `torch.where`, `torch.cat`), which a CPU stack computes slot by slot alike; the library vouches for `nets.MLP`
    alone (`groups._cpu_exact`)."""
    monkeypatch.setattr(groups, "_cpu_exact", lambda model: True)


def names_of(chunk):
    """The names of a stack's members' models, in slot order (a worker's model is rebuilt without its name: its place's)."""
    return [groups._S_MODELS[request.payload["place"]][0] for request, _, _ in chunk]


@pytest.fixture
def stacks(monkeypatch):
    """Every stack trained: the names of its real members' models (in slot order)."""
    seen = []
    real = groups._train_stack

    def spy(chunk, *args):
        trained = real(chunk, *args)
        seen.append(names_of(chunk))  # only a stack that trained
        return trained

    monkeypatch.setattr(groups, "_train_stack", spy)
    return seen


def boom(prediction, target):  # a loss by import path that raises once a fit trains
    raise RuntimeError("boom: the loss refused")


class BlowUp(torch.nn.Module):
    """A module whose output is infinite for a third of the members (by their stream key, which its `nets.Dropout` holds) when
    `dropout=0.5`, a buffer (so modules of other values stack together): those members never have a finite early-stopping
    loss (they diverged), and the others train on."""

    def __init__(self, n_inputs, n_outputs, dropout=0.0):
        super().__init__()
        self.net = torch.nn.Sequential(nets.Linear(n_inputs, 6), nets.Activation("relu"), nets.Linear(6, n_outputs))
        self.drop = nets.Dropout(0.0)
        self.register_buffer("on", torch.tensor(dropout == 0.5))

    def forward(self, x):
        broken = self.on & (self.drop.seed % 3 == 0)
        return self.drop(self.net(x)) * torch.where(broken, math.inf, 1.0)


class Scaled(torch.nn.Module):
    """A module that keeps its `dropout` as a plain attribute and scales its output by it: two models that differ in it share a
    key (`dropout` isn't in it) but not a stack (its attributes are its structure), so each keeps its own."""

    def __init__(self, n_inputs, n_outputs, dropout=0.0):
        super().__init__()
        self.net = torch.nn.Sequential(nets.Linear(n_inputs, 6), nets.Activation("relu"), nets.Linear(6, n_outputs))
        self.scale = 1.0 + dropout

    def forward(self, x):
        return self.net(x) * self.scale


class SigNet(torch.nn.Module):
    """`nets.Linear` layers with `torch.sigmoid` between them, not `nets.Activation`: over a stacked tensor a CPU's vectorised
    sigmoid gives the last slot other last bits, so on a CPU this module runs alone."""

    def __init__(self, n_inputs, n_outputs, width=5):
        super().__init__()
        self.a = nets.Linear(n_inputs, width)
        self.b = nets.Linear(width, n_outputs)

    def forward(self, x):
        return self.b(torch.sigmoid(self.a(x)))


class RandProj(torch.nn.Module):
    """A random projection drawn from torch's generator when it is built, kept as a plain attribute (not a buffer): vmap would
    run every slot with the first slot's, so its calls train alone (`groups._structure_of`)."""

    def __init__(self, n_inputs, n_outputs, width=4):
        super().__init__()
        self.proj = torch.randn(n_inputs, width)
        self.a = nets.Linear(width, 4)
        self.act = nets.Activation("relu")
        self.b = nets.Linear(4, n_outputs)

    def forward(self, x):
        return self.b(self.act(self.a(x @ self.proj)))


class Perm(torch.nn.Module):
    """A permutation of the inputs drawn when it is built, kept as a list: alone as `RandProj`."""

    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.perm = torch.randperm(n_inputs).tolist()
        self.net = torch.nn.Sequential(nets.Linear(n_inputs, 6), nets.Activation("relu"), nets.Linear(6, n_outputs))

    def forward(self, x):
        return self.net(x[..., self.perm])


class Coin(torch.nn.Module):
    """A coin tossed when it is built, kept as a plain attribute: the same under the probe's two seeds (no flip under 0 and
    1) but not under every member's own build seed, so only the exact check of a stack's members (`trainer._StackFit`'s
    `structure`) keeps them apart."""

    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.flip = bool(torch.rand(()) < 0.3)
        self.net = torch.nn.Sequential(nets.Linear(n_inputs, 6), nets.Activation("relu"), nets.Linear(6, n_outputs))

    def forward(self, x):
        out = self.net(x)
        return -out if self.flip else out


class Blocks(torch.nn.Module):
    """Named blocks, one layer each, then a head: of `nets` layers and exact operations, so it stacks."""

    def __init__(self, input_sizes, n_outputs, width=4):
        super().__init__()
        self.blocks = torch.nn.ModuleDict({name: nets.Linear(size, width) for name, size in input_sizes.items()})
        self.act = nets.Activation("relu")
        self.head = nets.Linear(width * len(input_sizes), n_outputs)

    def forward(self, blocks):
        return self.head(self.act(torch.cat([self.blocks[name](t) for name, t in blocks.items()], dim=-1)))


def daily(name, predictors=DAY, **params):
    return MLPDailyModel(predictors, **{"hidden": [8], "dropout": 0.1, "name": name, **FAST, **params})


def hourly(name, **params):
    return MLPHourlyModel(PERIOD, **{"hidden": [4], "name": name, **FAST, "seeds": 1, **params})


def loop_of(n, make=daily, seed=0, **params):
    """`n` models of one search (plan section 3.12): learning rates and dropout rates drawn, two widths, so a group holds two
    structures of one key's shapes, and a dropout of 0 (no `Dropout` layer) a third. Two learning rates, so that models share
    stacks (a stack's members are ordered by their rate)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        hidden = [8] if i % 3 else [8, 4]
        dropout = 0.0 if i % 5 == 0 else round(rng.uniform(0.05, 0.4), 2)
        lr = rng.choice([3e-3, 1e-2])
        out.append(make(f"m{i:02d}", hidden=hidden, dropout=dropout, lr=lr, **params))
    return out


def run_pipeline(models, data, test=TEST, horizon=2, **kwargs):
    pipeline = ModelPipeline()
    for model in models:
        pipeline.add_model(model)
    return pipeline.run(data, *test, horizon=horizon, **kwargs)


def records(report):
    """Each model's records by key, every field of each; a key twice is an error (a forecast made twice)."""
    out = {}
    for name, ref in report.refs.items():
        if ref._results is None:
            continue
        out[name] = {(r["run_date"], r["hour"], r["horizon"]): r for r in ref._results}
        assert len(out[name]) == len(ref._results), f"{name}: a key forecast twice"
    return out


def alone(models, data, monkeypatch, **kwargs):
    """Each model's records from its own run (`TORCH_STACKS=off`)."""
    with monkeypatch.context() as patch:
        patch.setenv(groups.TORCH_STACKS_VAR, "off")
        return records(run_pipeline(models, data, **kwargs))


def same(left, right):
    """Two runs' records, field for field and bit for bit (a NaN equals a NaN)."""
    assert left.keys() == right.keys()
    for name in left:
        assert left[name].keys() == right[name].keys(), name
        for key, record in left[name].items():
            other = right[name][key]
            assert record.keys() == other.keys(), (name, key)
            for field, value in record.items():
                if isinstance(value, float) and np.isnan(value):
                    assert isinstance(other[field], float) and np.isnan(other[field]), (name, key, field)
                else:
                    assert value == other[field], (name, key, field, value, other[field])


def lines(path):
    return sorted(line for line in Path(path).read_text().splitlines() if line.strip())


# --- which models stack ------------------------------------------------------------------------------------------------


class OwnFit(MLPDailyModel):
    def _fit_rows(self, x, y, y_mask):
        return super()._fit_rows(x, y, y_mask)


def test_which_models_stack_and_under_what_key(monkeypatch):
    key = groups.stack_key
    base = daily("a")
    assert key(base) is not None
    assert key(daily("b", dropout=0.3, lr=1e-3, seeds=4, random_state=5)) == key(base)  # each member's own
    assert key(MLPDailyModel(DAY[:4], **{"hidden": [8], "dropout": 0.1, "name": "c", **FAST})) == key(base)  # its rows
    flat = TorchModel(
        module=nets.MLP, module_params={"hidden": [8], "dropout": 0.2}, output="day", predictors=DAY, name="t", **FAST
    )
    assert key(flat) == key(base)  # the preset is TorchModel with nets.MLP
    for other in (
        daily("d", hidden=[16]),
        daily("e", activation="tanh"),
        daily("f", pad_to=8),
        daily("g", training_window=21),
        daily("h", refit="1D"),
        daily("i", max_epochs=5),
        daily("j", batch_size=8),
        daily("k", weight_decay=1e-4),
        daily("l", tf32=True),
        hourly("m"),
    ):
        assert key(other) is not None and key(other) != key(base), other.name
    assert key(daily("n", stack="auto")) is None and key(daily("o", stack="loop")) is None  # a loop on the CPU
    assert key(base, forecast_only=True) is None
    assert key(OwnFit(DAY, hidden=[8], dropout=0.1, name="p", **FAST)) is None
    patched = daily("q")
    patched._members_out = lambda fitted, rows: None
    assert key(patched) is None
    assert key(LEARModel(["price_d-1"], 28, name="lear")) is None

    # On a CPU only nets.MLP with an exact loss stacks: another module, or a loss of its own, runs alone.
    own_module = TorchModel(module=FlatNet, predictors=DAY, name="own", **FAST)
    own_loss = daily("r", loss="tests.models.test_torch_model:squared")
    assert key(daily("s", loss="mse")) is not None
    assert key(own_module) is None and key(own_loss) is None
    monkeypatch.setattr(groups, "_cpu_exact", lambda model: True)  # what a GPU decides (every element computed alike)
    assert key(own_module) is not None and key(own_loss) is not None
    monkeypatch.undo()

    models = [
        daily("a"),
        hourly("b"),
        daily("c", dropout=0.0),
        LEARModel(["price_d-1"], 28),
        hourly("d"),
        daily("e", hidden=[3]),
    ]
    assert [[m.name for m in group] for group in groups.stack_groups(models)] == [["a", "c"], ["b", "d"]]
    monkeypatch.setenv(groups.TORCH_STACKS_VAR, "check")
    assert [[m.name for m in group] for group in groups.stack_groups(models)] == [["a", "c"], ["b", "d"]]
    monkeypatch.setenv(groups.TORCH_STACKS_VAR, "off")
    assert groups.stack_groups(models) == []
    monkeypatch.setenv(groups.TORCH_STACKS_VAR, "sometimes")
    with pytest.raises(ValueError, match="TORCH_STACKS"):
        groups.stack_groups(models)


@pytest.mark.parametrize(("variable", "value"), [("TORCH_STACKS", "sometimes"), ("TORCH_STACK_MEMBERS", "0")])
def test_a_wrong_setting_refuses_before_any_model_writes(market, tmp_path, monkeypatch, variable, value):
    """`ModelPipeline.run` checks both before the engine groups or any model write a file."""
    monkeypatch.setenv(variable, value)
    pipeline = ModelPipeline(engine="on")
    pipeline.add_model(OLSModel(["load_h08_d+{horizon}", "price_d-1"], 28, name="ols"))
    for model in loop_of(2):
        pipeline.add_model(model)
    with pytest.raises(ValueError, match=variable):
        pipeline.run(market, *TEST, horizon=1, save_dir=tmp_path / "run")
    assert not (tmp_path / "run").exists() or not any((tmp_path / "run").iterdir())


def test_kept_holds_every_method_a_stack_replaces_or_a_unit_calls():
    """A model whose class overrides one of these runs alone (`stack_key`): the engine's replaced methods, every unit kind's
    fit methods and the refit methods are among them."""
    from epftoolbox2.models import base
    from epftoolbox2.models._units import UNIT_METHODS

    needed = (
        set(base._ENGINE_REPLACES)
        | set(base._REFIT_METHODS)
        | {name for names in UNIT_METHODS.values() for name in names}
    )
    assert needed <= set(groups._KEPT), sorted(needed - set(groups._KEPT))


def test_describe_hashes_every_element_of_an_array():
    """A plain numpy attribute is compared element by element (a repr shortens a large array)."""
    a = np.zeros(5000)
    b = a.copy()
    b[2500] = 1.0
    assert repr(a) == repr(b) and groups._describe(a) != groups._describe(b)
    assert groups._describe(a) == groups._describe(a.copy())


def test_structure_keeps_layers_apart_but_not_dropout_rates():
    def built(**params):
        with torch.random.fork_rng(devices=[]):
            return groups._structure(nets.MLP(7, 24, **params))

    assert built(hidden=(8,), dropout=0.1) == built(hidden=(8,), dropout=0.4)
    assert built(hidden=(8,), dropout=0.1) != built(hidden=(8,), dropout=0.0)  # no Dropout layer
    assert built(hidden=(8,)) != built(hidden=(8,), activation="tanh")
    assert built(hidden=(8,)) != built(hidden=(9,))
    # Built under two seeds of torch's generator: a structure drawn from it is none to stack on (its calls train alone).
    groups._S_STRUCTURES.clear()
    for place, module in enumerate([lambda: RandProj(7, 24), lambda: Perm(7, 24)]):
        assert groups._structure_of(place, module, 7, 24) is None
    for place, module in enumerate([lambda: nets.MLP(7, 24, hidden=(8,), dropout=0.1), lambda: Scaled(7, 24, 0.3)], 2):
        assert groups._structure_of(place, module, 7, 24) is not None
    groups._S_STRUCTURES.clear()


# --- alone, in a loop, shuffled, in shards -------------------------------------------------------------------------------


def test_a_model_alone_equals_it_in_a_loop_shuffled_and_in_shards(market, monkeypatch, stacks):
    """16 models of the search (S02 checked the 48 of plan section 3.12 at the research's shapes: research folder)."""
    models = loop_of(16)
    own = alone(models, market, monkeypatch, test=WEEK)
    assert not stacks
    together = records(run_pipeline(models, market, test=WEEK))
    same(own, together)
    assert any(len(set(stack)) > 1 for stack in stacks), "no stack held members of two models"
    assert sum(len(stack) for stack in stacks) == 16 * 2 * 2  # models, seeds, horizons

    stacks.clear()
    shuffled = models[:]
    random.Random(1).shuffle(shuffled)
    same(own, records(run_pipeline(shuffled, market, test=WEEK)))
    shards = {}
    for first in range(0, 16, 5):  # models_per_task=5: blocks of 5, 5, 5 and 1 models in list order
        shards.update(records(run_pipeline(models[first : first + 5], market, test=WEEK)))
    same(own, shards)
    produced = [len(r) for r in own.values()]
    assert min(produced) > 0 and len(set(produced)) == 1


@pytest.mark.parametrize(
    "case",
    [
        "hourly",
        "hourly_refit_none",
        "window_list",
        "named_blocks",
        "jobs_of_one_member",
        "inputs_of_every_horizon",
        "attributes_of_the_module",
        "row_targets",
        "sigmoid_on_a_cpu",
        "attributes_drawn_at_build",
    ],
)
def test_stacked_records_equal_each_model_s_own(market, monkeypatch, stacks, case, request):
    if case in ("named_blocks", "attributes_of_the_module", "attributes_drawn_at_build"):
        request.getfixturevalue("vouched")
    if case == "hourly":
        models = loop_of(3, make=hourly)
    elif case == "hourly_refit_none":
        models = [
            hourly(f"n{i}", refit=None, training_window=[21, 28], dropout=d, random_state=i)
            for i, d in enumerate([0.1, 0.2])
        ]
    elif case == "window_list":
        models = loop_of(4, training_window=[21, 28])
    elif case == "named_blocks":
        models = [
            TorchModel(module=Blocks, inputs=BLOCKS, name=f"b{i}", **{**FAST, "lr": lr})
            for i, lr in enumerate([1e-2, 3e-3, 1e-2])
        ]
    elif case == "jobs_of_one_member":  # a job per fit unit, a window list's units of a block together
        monkeypatch.setenv(groups.STACK_MEMBERS_VAR, "1")
        models = loop_of(3, training_window=[21, 28])
    elif case == "row_targets":  # each model's own per-row target (step 7c) in stacks shared with models of none
        row_targets = [
            {},
            {"normalise_by": "load_d+{horizon}", "bounds": (0.0, 0.08)},
            {"zero_when": "is_weekend_d+{horizon}", "bounds": (None, 60.0)},
        ]
        models = [hourly(f"r{i}", lr=1e-2, random_state=i, **target) for i, target in enumerate(row_targets)]
        assert len({groups.stack_key(model) for model in models}) == 1
    elif case == "inputs_of_every_horizon":  # the same inputs (and rows) for both horizons, other targets, in one stack
        models = [daily(f"m{i}", predictors=DAY[:4], seeds=1, dropout=d) for i, d in enumerate([0.1, 0.2, 0.3])]
    elif case == "sigmoid_on_a_cpu":  # a module's own torch.sigmoid: alone on a CPU, its own numbers
        models = [TorchModel(module=SigNet, predictors=DAY, name=f"g{i}", random_state=i, **FAST) for i in range(3)]
        assert {groups.stack_key(model) for model in models} == {None}
    elif case == "attributes_drawn_at_build":  # vouched for, keyed together, but each call trained alone in the job
        models = [
            TorchModel(module=module, predictors=DAY, name=f"{module.__name__}{i}", random_state=i, **FAST)
            for module in (RandProj, Perm, Coin)
            for i in range(2)
        ]
        assert groups._structure_of(-1, lambda: Coin(7, 24), 7, 24) is not None  # the probe's seeds agree on its coin
    else:
        models = [
            TorchModel(
                module=Scaled, predictors=DAY, module_params={"dropout": d}, random_state=i, name=f"s{i}", **FAST
            )
            for i, d in enumerate([0.0, 0.3, 0.0, 0.3])
        ]
        assert len({groups.stack_key(model) for model in models}) == 1
    test = TEST
    horizon = 2
    if case == "hourly_refit_none":
        test, horizon = ("2023-03-20", "2023-03-20"), 1
    elif case in ("hourly", "row_targets"):
        test, horizon = WEEK, 1
    own = alone(models, market, monkeypatch, test=test, horizon=horizon)
    together = records(run_pipeline(models, market, test=test, horizon=horizon))
    same(own, together)
    if case == "sigmoid_on_a_cpu":
        assert not stacks and groups.STATS["alone"] == 0
    elif case == "attributes_drawn_at_build":
        # RandProj's and Perm's calls alone by the probe (models, horizons); Coin's by the exact check where a stack's members'
        # coins differ (counted as alone, not as fallbacks), the others in a stack of equal coins (`same` above).
        stacked = sum(len(stack) for stack in stacks) // 2  # calls of two seeds
        assert all(name.startswith("Coin") for stack in stacks for name in stack)
        assert groups.STATS["alone"] == 4 * 2 + 2 * 2 - stacked and stacked < 2 * 2
        assert {groups._S_MODELS[place][0] for place in groups._S_UNSTACKABLE} == {"Coin0", "Coin1"}
    else:
        assert stacks and (case == "jobs_of_one_member" or any(len(set(stack)) > 1 for stack in stacks))
    if case == "row_targets":  # the targets took effect: forecasts of 0 on weekends, and r1's are its own scaled ones
        assert any(r["prediction"] == 0.0 and r.get("detail") == "zero_when" for r in own["r2"].values())
        assert any(own["r1"][key]["prediction"] != own["r0"][key]["prediction"] for key in own["r0"])


def test_a_pool_of_forked_processes_gives_each_model_s_own_records(market, monkeypatch):
    models = loop_of(6)
    own = alone(models, market, monkeypatch)
    monkeypatch.setenv("MAX_PROCESSES", "2")
    monkeypatch.setattr(groups, "ProcessPoolExecutor", concurrent.futures.ProcessPoolExecutor)
    monkeypatch.setenv(groups.STACK_MEMBERS_VAR, "8")  # several jobs, so both processes work
    same(own, records(run_pipeline(models, market)))


# --- resume, crash ---------------------------------------------------------------------------------------------------------


def test_a_resume_and_a_crash_then_resume_equal_a_fresh_run(market, tmp_path, monkeypatch):
    models = loop_of(5)
    fresh = tmp_path / "fresh"
    run_pipeline(models, market, save_dir=fresh)

    resumed = tmp_path / "resumed"
    resumed.mkdir()
    for i, model in enumerate(models):  # cut short: none of one model's records, a quarter of another's, ...
        done = Path(ModelPipeline.results_path(fresh, model, "jsonl")).read_text().splitlines(keepends=True)
        Path(ModelPipeline.results_path(resumed, model, "jsonl")).write_text("".join(done[: len(done) * i // 4]))
        manifest = manifest_path(ModelPipeline.results_path(fresh, model, "jsonl"))
        (resumed / manifest.name).write_text(manifest.read_text())
    run_pipeline(models, market, save_dir=resumed)
    for model in models:
        assert lines(ModelPipeline.results_path(resumed, model, "jsonl")) == lines(
            ModelPipeline.results_path(fresh, model, "jsonl")
        )

    crashed = tmp_path / "crashed"
    real = groups.run_stack_job
    jobs = []

    def second_fails(job, *args):
        jobs.append(1)
        if len(jobs) == 2:
            raise RuntimeError("the node went away")
        return real(job, *args)

    monkeypatch.setenv(groups.STACK_MEMBERS_VAR, "1")  # a job per block
    monkeypatch.setattr(groups, "run_stack_job", second_fails)
    with pytest.raises(RuntimeError, match="the node went away"):
        run_pipeline(models, market, save_dir=crashed)
    monkeypatch.setattr(groups, "run_stack_job", real)
    run_pipeline(models, market, save_dir=crashed)
    for model in models:
        assert lines(ModelPipeline.results_path(crashed, model, "jsonl")) == lines(
            ModelPipeline.results_path(fresh, model, "jsonl")
        )


# --- next to other models, a session class, failures -----------------------------------------------------------------------


def scaled(scale):
    """A local class (a session token for its key) whose forecasts move with its class attribute: two of one name are two
    keys, and a stack running one's code for the other's members would move them."""

    class Scaled(FlatNet):
        SCALE = scale

        def forward(self, x):
            return super().forward(x) * type(self).SCALE

    return Scaled


def test_models_of_other_keys_run_alone_and_a_session_class_stacks_with_its_own(market, monkeypatch, vouched):
    one, two = scaled(1.0), scaled(3.0)

    def session(cls, name, lr):
        return TorchModel(module=cls, predictors=DAY, name=name, **{**FAST, "lr": lr})

    models = [
        session(one, "one_a", 1e-2),
        daily("loop", stack="loop"),
        session(two, "two_a", 1e-2),
        daily("wide", hidden=[16]),
        session(one, "one_b", 3e-3),
        LEARModel(["price_d-1", "load_h08_d+{horizon}"], 28, name="lear"),
        session(two, "two_b", 3e-3),
    ]
    assert groups.stack_key(models[0]) != groups.stack_key(models[2])
    ran = []
    real = groups.run_stack_group

    def spy(group, arguments, columns):
        ran.append([model.name for model in group])
        return real(group, arguments, columns)

    monkeypatch.setattr("epftoolbox2.pipelines.model_pipeline.run_stack_group", spy)
    own = alone(models, market, monkeypatch)
    together = records(run_pipeline(models, market))
    assert ran == [["one_a", "one_b"], ["two_a", "two_b"]]
    same(own, together)


def test_a_member_that_diverged_is_its_fit_s_error_and_moves_no_other_member(market, monkeypatch, stacks, vouched):
    models = [
        TorchModel(
            module=BlowUp, predictors=DAY, module_params={"dropout": dropout}, name=f"m{i:02d}", **{**FAST, "lr": lr}
        )
        for i, (dropout, lr) in enumerate([(0.0, 1e-2), (0.5, 1e-2), (0.1, 3e-3), (0.0, 2e-2)])
    ]
    assert len({groups.stack_key(model) for model in models}) == 1
    own = alone(models, market, monkeypatch)
    assert 0 < len(own["m01"]) < len(own["m00"]) == len(own["m02"]) == len(own["m03"])  # fit errors: not in the records
    together = records(run_pipeline(models, market))
    same(own, together)
    assert any("m01" in stack and len(set(stack)) > 1 for stack in stacks)


def test_a_stack_that_raises_trains_its_calls_alone_warns_once_and_counts(market, monkeypatch, fallbacks_expected):
    models = loop_of(4)
    own = alone(models, market, monkeypatch)
    real = groups._train_stack
    raised = []

    def first_raises(chunk, *args):
        if not raised:
            raised.append(names_of(chunk))
            raise RuntimeError("out of memory")
        return real(chunk, *args)

    monkeypatch.setattr(groups, "_train_stack", first_raises)
    monkeypatch.setattr(groups, "_FALLBACK_WARNED", False)
    with pytest.warns(RuntimeWarning, match="a stack raised RuntimeError\\('out of memory'\\)") as caught:
        same(own, records(run_pipeline(models, market)))
    assert raised and groups.STATS["fallbacks"] == len(raised[0]) // 2  # the stack's calls, of 2 seeds each
    assert sum("a stack raised" in str(w.message) for w in caught) == 1  # once per process


def test_a_fit_that_raises_stops_the_group_naming_its_model(market, vouched, fallbacks_expected):
    """A loss that raises (not a ValueError, so no fit_error): its own run raises, and so does a group of it, naming it."""
    with pytest.raises(RuntimeError, match="^bad: forecasting failed") as alone_error:
        daily("bad", loss=LOSS_BOOM).run(market, *TEST, horizon=1)
    assert "boom" in str(alone_error.value)
    stacked = [daily("bad", loss=LOSS_BOOM), daily("bad_too", loss=LOSS_BOOM, lr=3e-3)]
    with (
        pytest.raises(RuntimeError, match=r"^bad: forecasting failed for the refit block") as error,
        warnings.catch_warnings(),
    ):
        warnings.filterwarnings("ignore", "torch stacks: a stack raised", RuntimeWarning)  # its stack fell back first
        run_pipeline(stacked, market, horizon=1)
    assert "boom" in str(error.value)


def test_the_engine_groups_are_unchanged_and_torch_stacks_run_after_them(market, monkeypatch):
    linear = [
        OLSModel(["load_h08_d+{horizon}", "price_d-1"], 28, name="ols"),
        RidgeModel(["load_h08_d+{horizon}", "price_d-1"], 28, name="ridge"),
    ]
    torch_models = [daily("m00"), daily("m01", lr=3e-3), daily("m02", dropout=0.3)]
    order = []
    real_engine, real_stack = _engine.run_group, groups.run_stack_group

    def engine_spy(group, arguments, columns=None, check=False):
        order.append(("engine", [model.name for model in group]))
        return real_engine(group, arguments, columns, check=check)

    def stack_spy(group, arguments, columns):
        order.append(("stack", [model.name for model in group]))
        return real_stack(group, arguments, columns)

    monkeypatch.setattr("epftoolbox2.pipelines.model_pipeline.run_group", engine_spy)
    monkeypatch.setattr("epftoolbox2.pipelines.model_pipeline.run_stack_group", stack_spy)
    both = ModelPipeline(engine="on")
    for model in [*linear, *torch_models]:
        both.add_model(model)
    report = records(both.run(market, *TEST, horizon=2))
    assert order == [("engine", ["ols", "ridge"]), ("stack", ["m00", "m01", "m02"])]
    only_linear = ModelPipeline(engine="on")
    for model in linear:
        only_linear.add_model(model)
    same({name: report[name] for name in ("ols", "ridge")}, records(only_linear.run(market, *TEST, horizon=2)))
    same({model.name: report[model.name] for model in torch_models}, alone(torch_models, market, monkeypatch))


def test_zones_stack_per_zone(market, monkeypatch):
    other = frame(seed=7)
    models = loop_of(3)
    with monkeypatch.context() as patch:
        patch.setenv(groups.TORCH_STACKS_VAR, "off")
        own = run_pipeline(models, {"PL": market, "CZ": other})
    together = run_pipeline(models, {"PL": market, "CZ": other})
    for zone in ("PL", "CZ"):
        same(records(own.zones[zone]), records(together.zones[zone]))


# --- GPU (WCSS H100; `-m gpu`, never in the fast suite) ---------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize("tf32", [False, True])
def test_on_gpu_a_model_alone_equals_it_in_a_loop_and_shuffled(market, monkeypatch, tf32):
    if not runtime.cuda_probe(0)[0]:
        pytest.skip("needs a usable CUDA device")
    from torch.func import stack_module_state

    from epftoolbox2.models._torch import trainer

    # A stack's state drawn at once on the GPU (`_stacked_counters`) is its slots' built one by one, to the bit.
    device = torch.device("cuda:0")
    factories = [lambda d=d: nets.MLP(247, 24, hidden=(256, 128), dropout=d) for d in (0.1, 0.3)] * 40
    slots = [trainer._Member(k, trainer.stream_key(k), 1e-3, np.arange(4), np.arange(2)) for k in range(80)]
    first = trainer._build(factories[0], slots[0], device, torch.float32)
    fast = trainer._stacked_counters(first, factories, slots, device, torch.float32)
    slow = stack_module_state([trainer._build(f, m, device, torch.float32) for f, m in zip(factories, slots)])
    assert all(torch.equal(fast[i][name], slow[i][name]) for i in (0, 1) for name in slow[i])

    models = loop_of(12, device="cuda", tf32=tf32, pad_to=16)
    own = alone(models, market, monkeypatch)
    same(own, records(run_pipeline(models, market)))
    shuffled = models[::-1]
    same(own, records(run_pipeline(shuffled, market)))


def test_calls_share_a_stack_s_rows_only_where_inputs_targets_and_mask_are_equal():
    """Four calls with the same inputs in one stack: two alike (their rows shared), one of other targets and one of another
    mask, each trained as alone, to the bit (a call that read another's targets or mask would not be)."""
    from epftoolbox2.models._torch import trainer

    groups._S_STRUCTURES.clear()  # place 0 is this test's model
    groups._S_UNSTACKABLE.clear()
    rng = np.random.default_rng(5)
    x = rng.normal(size=(40, 5))
    y = rng.normal(size=(40, 2))
    mask = np.ones((40, 2), dtype=bool)
    other_mask = mask.copy()
    other_mask[::3, 1] = False
    model = daily("m")
    settings = dict(
        loss="l1", lr=1e-2, max_epochs=3, patience=2, batch_size=8, early_stopping=0.25, weight_decay=0.0, stack="vmap",
        pad_to=4, dtype="float32", nominal_rows=40, blocks=None,
    )  # fmt: skip

    def factory():
        return nets.MLP(5, 2, hidden=(4,))

    calls = [(y, mask), (y + 1.0, mask), (y, other_mask), (y, mask)]
    requests = []
    for i, (target, kept) in enumerate(calls):
        members = [trainer.MemberSpec(seed_key=1000 + i, train_rows=np.arange(40))]
        payload = dict(
            place=0, model=model, factory=factory, x=x, y=target, y_mask=kept, members=members, settings=settings
        )
        requests.append(groups._Request(groups._FIT, (i,), payload))
    groups._train(groups._Session([]), requests)
    groups._S_STRUCTURES.clear()
    groups._S_UNSTACKABLE.clear()
    rows = rng.normal(size=(3, 5))
    for request, (target, kept) in zip(requests, calls):
        result, device = request.result
        ((fit, slot),) = result.slots
        assert len({id(r.result[0].slots[0][0]) for r in requests}) == 1  # one stack
        alone = trainer.train_members(factory, x, target, kept, request.payload["members"], device=device, **settings)
        assert np.array_equal(alone.stats["best_val"], result.stats["best_val"])
        block = np.broadcast_to(rows, (fit.slots, *rows.shape)).copy()
        stacked = fit.run_slots(torch.as_tensor(block, dtype=torch.float32))[slot]
        assert np.array_equal(alone.predict(rows)[0], stacked)


@pytest.mark.parametrize("hidden", [(8,), (8, 4)])
def test_a_stack_of_the_library_s_layers_is_drawn_at_once_as_its_slots_built_one_by_one(hidden):
    """`_stacked_counters` (an MLP's stack: one build per factory, every slot's weights drawn from its key at once) against
    `stack_module_state` of each slot's own module (`_build`), every tensor to the bit: weights, biases, dropout rates of
    three factories, seeds, steps; a module with a layer of torch's own is built slot by slot."""
    from torch.func import stack_module_state

    from epftoolbox2.models._torch import trainer

    factories = [lambda d=d: nets.MLP(7, 24, hidden=hidden, dropout=d) for d in (0.1, 0.3, 0.1)]
    chosen = [factories[i % 3] for i in range(70)]  # more slots than one draw (`_DRAW_SLOTS`)
    slots = [trainer._Member(k, trainer.stream_key(k), 1e-3, np.arange(4), np.arange(2)) for k in range(70)]
    device = torch.device("cpu")
    first = trainer._build(chosen[0], slots[0], device, torch.float32)
    assert trainer._counter_only(first)
    fast_params, fast_buffers = trainer._stacked_counters(first, chosen, slots, device, torch.float32)
    modules = [trainer._build(f, m, device, torch.float32) for f, m in zip(chosen, slots)]
    slow_params, slow_buffers = stack_module_state(modules)
    for fast, slow in ((fast_params, slow_params), (fast_buffers, slow_buffers)):
        assert list(fast) == list(slow)
        for name in slow:
            assert fast[name].dtype == slow[name].dtype and fast[name].shape == slow[name].shape, name
            assert fast[name].requires_grad == slow[name].requires_grad and fast[name].is_leaf, name
            assert torch.equal(fast[name], slow[name]), name
    assert not trainer._counter_only(FlatNet(7, 24)) and not trainer._counter_only(Scaled(7, 24))


def test_jobs_are_cut_at_whole_units_in_block_order():
    day = datetime.date
    pieces = [(day(2023, 3, d), (1,), place, [(None, 1, d)], 4, 1) for d in (6, 13, 20, 27) for place in (0, 1)]
    cuts = groups._jobs(pieces, 6)
    assert [(start.day, last.day) for start, last, _ in cuts] == [(6, 6), (13, 13), (20, 20), (27, 27)]
    cuts = groups._jobs(pieces, 16)
    assert [(start.day, last.day) for start, last, _ in cuts] == [(6, 13), (20, 27)]
    assert cuts[0][2] == {0: [(None, 1, 6), (None, 1, 13)], 1: [(None, 1, 6), (None, 1, 13)]}
    big = [(day(2023, 3, 6), (1,), place, [(None, 1, 6)], 1, groups.MAX_JOB_ITEMS // 2 + 1) for place in range(3)]
    assert [list(job) for _, _, job in groups._jobs(big, 10**6)] == [[0], [1], [2]]
    assert groups._job_members(1000, 16, 4) == 64 and groups._job_members(40, 16, 4) == 16
    many = [(day(2023, 3, 6), (k,), 0, [(None, 1, 6)], 1, 1) for k in range(3000)]  # seeds=1: a member an item
    assert [len(job[0]) for _, _, job in groups._jobs(many, 2048, max(groups.MAX_JOB_ITEMS, 2048))] == [2048, 952]


def test_check_mode_trains_a_sample_alone_and_records_it(market, tmp_path, monkeypatch):
    """`TORCH_STACKS=check` (every call sampled here): each manifest's `stack_check` counts its calls, none differing; a call
    whose numbers alone differ from the stack's is counted, its largest difference recorded, and the model warns; a call
    that raised when trained alone is counted apart (`failed`: nothing compared), and the model warns so."""
    monkeypatch.setenv(groups.TORCH_STACKS_VAR, "check")
    monkeypatch.setattr(groups, "CHECK_EVERY", 1)
    models = [daily("c0"), daily("c1", lr=3e-3), daily("c2", dropout=0.3)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        run_pipeline(models, market, test=WEEK, save_dir=tmp_path / "ok")

    def stored(where, model):
        return json.loads(manifest_path(ModelPipeline.results_path(tmp_path / where, model, "jsonl")).read_text())

    for model in models:
        assert stored("ok", model)["stack_check"] == {"calls": 2, "differing": 0, "failed": 0, "max_abs": 0.0}

    real = groups._check_alone

    def off_by_one(session, result, *args):
        real(session, result, *args)
        alone = result.check
        result.check = trainer_result_shifted(alone, 1.0)

    monkeypatch.setattr(groups, "_check_alone", off_by_one)
    with pytest.warns(RuntimeWarning, match="torch stack check: 2 of 2 calls checked gave other bits"):
        run_pipeline(models, market, test=WEEK, save_dir=tmp_path / "off")
    for model in models:
        assert stored("off", model)["stack_check"] == {"calls": 2, "differing": 2, "failed": 0, "max_abs": 1.0}

    def refused(*args, **kwargs):
        raise RuntimeError("out of memory, alone")

    def alone_raises(session, result, call, model, trainer, *rest):
        real(session, result, call, model, SimpleNamespace(train_members=refused), *rest)

    monkeypatch.setattr(groups, "_check_alone", alone_raises)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_pipeline(models, market, test=WEEK, save_dir=tmp_path / "failed")
    said = [str(w.message) for w in caught if "torch stack check" in str(w.message)]
    assert len(said) == 3 and all("2 of 2 calls checked raised when trained or predicted alone" in s for s in said)
    assert not any("gave other bits" in s for s in said)
    for model in models:
        assert stored("failed", model)["stack_check"] == {"calls": 2, "differing": 0, "failed": 2, "max_abs": 0.0}
        assert lines(ModelPipeline.results_path(tmp_path / "failed", model, "jsonl")) == lines(
            ModelPipeline.results_path(tmp_path / "ok", model, "jsonl")
        )  # the stacks' records, all of them


def trainer_result_shifted(result, by):
    """A `TrainResult` whose forecasts are `result`'s plus `by`."""
    from epftoolbox2.models._torch import trainer

    return trainer.TrainResult(lambda rows: result.predict(rows) + by, result.best_epoch, result.failed, result.stats)


def test_an_interrupt_while_a_round_is_released_ends_the_job(market, monkeypatch):
    """A Ctrl-C after a round was served and before its items were let go (the reviewer's injected interrupt): the waiting
    items are let go with `_Aborted`, every thread joined, and the interrupt raised, never a hang."""
    import faulthandler

    real = groups._Session._release
    hit = []

    def release(self, batch):
        if batch and not hit:
            hit.append(1)
            raise KeyboardInterrupt
        return real(self, batch)

    monkeypatch.setattr(groups._Session, "_release", release)
    faulthandler.dump_traceback_later(120, exit=True)  # a hang ends the test process with a traceback
    try:
        with pytest.raises(KeyboardInterrupt):
            run_pipeline(loop_of(3), market, test=WEEK)
    finally:
        faulthandler.cancel_dump_traceback_later()
    assert hit


def test_a_stacked_model_s_files_are_its_own_run_s(market, tmp_path, monkeypatch):
    """Its records line for line and its manifest but for the times it was written: nothing names the group."""
    models = loop_of(3)
    with monkeypatch.context() as patch:
        patch.setenv(groups.TORCH_STACKS_VAR, "off")
        run_pipeline(models, market, save_dir=tmp_path / "alone")
    run_pipeline(models, market, save_dir=tmp_path / "stacked")

    def manifest(where, model):
        found = json.loads(manifest_path(ModelPipeline.results_path(tmp_path / where, model, "jsonl")).read_text())
        return {key: value for key, value in found.items() if key not in ("created_at", "updated_at")}

    for model in models:
        stacked = manifest("stacked", model)
        assert stacked["runtime"]["stack"] == "vmap" and stacked["runtime"]["pad_to"] == 4
        assert stacked == manifest("alone", model)
        assert lines(ModelPipeline.results_path(tmp_path / "stacked", model, "jsonl")) == lines(
            ModelPipeline.results_path(tmp_path / "alone", model, "jsonl")
        )
