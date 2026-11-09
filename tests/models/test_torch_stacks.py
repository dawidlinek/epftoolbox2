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
from concurrent.futures import Future
from pathlib import Path

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
TEST = ("2023-03-13", "2023-03-26")  # two "7D" blocks
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
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.delenv(groups.TORCH_STACKS_VAR, raising=False)
    monkeypatch.delenv(groups.STACK_MEMBERS_VAR, raising=False)
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)
    monkeypatch.setattr(_engine, "ProcessPoolExecutor", InProcessPool)
    monkeypatch.setattr(groups, "ProcessPoolExecutor", InProcessPool)


def names_of(chunk):
    """The names of a stack's members' models, in slot order (a worker's model is rebuilt without its name: its place's)."""
    return [groups._S_MODELS[request.payload["place"]][0] for request, _, _ in chunk]


@pytest.fixture
def stacks(monkeypatch):
    """Every stack trained: the names of its real members' models (in slot order)."""
    seen = []
    real = groups._train_stack

    def spy(chunk, *args):
        seen.append(names_of(chunk))
        return real(chunk, *args)

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

    models = [
        daily("a"),
        hourly("b"),
        daily("c", dropout=0.0),
        LEARModel(["price_d-1"], 28),
        hourly("d"),
        daily("e", hidden=[3]),
    ]
    assert [[m.name for m in group] for group in groups.stack_groups(models)] == [["a", "c"], ["b", "d"]]
    monkeypatch.setenv(groups.TORCH_STACKS_VAR, "off")
    assert groups.stack_groups(models) == []
    monkeypatch.setenv(groups.TORCH_STACKS_VAR, "sometimes")
    with pytest.raises(ValueError, match="TORCH_STACKS"):
        groups.stack_groups(models)


def test_structure_keeps_layers_apart_but_not_dropout_rates():
    def built(**params):
        with torch.random.fork_rng(devices=[]):
            return groups._structure(nets.MLP(7, 24, **params))

    assert built(hidden=(8,), dropout=0.1) == built(hidden=(8,), dropout=0.4)
    assert built(hidden=(8,), dropout=0.1) != built(hidden=(8,), dropout=0.0)  # no Dropout layer
    assert built(hidden=(8,)) != built(hidden=(8,), activation="tanh")
    assert built(hidden=(8,)) != built(hidden=(9,))


# --- alone, in a loop, shuffled, in shards -------------------------------------------------------------------------------


def test_a_model_alone_equals_it_in_a_48_model_loop_shuffled_and_in_shards(market, monkeypatch, stacks):
    models = loop_of(48)
    own = alone(models, market, monkeypatch)
    assert not stacks
    together = records(run_pipeline(models, market))
    same(own, together)
    assert any(len(set(stack)) > 1 for stack in stacks), "no stack held members of two models"
    assert sum(len(stack) for stack in stacks) == 48 * 2 * 2 * 2  # models, seeds, horizons, blocks

    stacks.clear()
    shuffled = models[:]
    random.Random(1).shuffle(shuffled)
    same(own, records(run_pipeline(shuffled, market)))
    shards = {}
    for first in range(0, 48, 20):  # models_per_task=20: blocks of 20, 20 and 8 models in list order
        shards.update(records(run_pipeline(models[first : first + 20], market)))
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
    ],
)
def test_stacked_records_equal_each_model_s_own(market, monkeypatch, stacks, case):
    if case == "hourly":
        models = loop_of(4, make=hourly)
    elif case == "hourly_refit_none":
        models = loop_of(3, make=hourly, refit=None, training_window=[21, 28])
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
    elif case == "inputs_of_every_horizon":  # the same inputs (and rows) for both horizons, other targets, in one stack
        models = [daily(f"m{i}", predictors=DAY[:4], seeds=1, dropout=d) for i, d in enumerate([0.1, 0.2, 0.3])]
    else:
        models = [
            TorchModel(
                module=Scaled, predictors=DAY, module_params={"dropout": d}, random_state=i, name=f"s{i}", **FAST
            )
            for i, d in enumerate([0.0, 0.3, 0.0, 0.3])
        ]
        assert len({groups.stack_key(model) for model in models}) == 1
    test = ("2023-03-20", "2023-03-21") if case == "hourly_refit_none" else TEST
    horizon = 1 if case == "hourly_refit_none" else 2
    own = alone(models, market, monkeypatch, test=test, horizon=horizon)
    together = records(run_pipeline(models, market, test=test, horizon=horizon))
    same(own, together)
    assert stacks and (case == "jobs_of_one_member" or any(len(set(stack)) > 1 for stack in stacks))


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


def test_models_of_other_keys_run_alone_and_a_session_class_stacks_with_its_own(market, monkeypatch):
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


def test_a_member_that_diverged_is_its_fit_s_error_and_moves_no_other_member(market, monkeypatch, stacks):
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


def test_a_stack_that_raises_trains_its_calls_alone(market, monkeypatch):
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
    same(own, records(run_pipeline(models, market)))
    assert raised


def test_a_fit_that_raises_stops_the_group_naming_its_model(market):
    """A loss that raises (not a ValueError, so no fit_error): its own run raises, and so does a group of it, naming it."""
    with pytest.raises(RuntimeError, match="^bad: forecasting failed") as alone_error:
        daily("bad", loss=LOSS_BOOM).run(market, *TEST, horizon=1)
    assert "boom" in str(alone_error.value)
    stacked = [daily("bad", loss=LOSS_BOOM), daily("bad_too", loss=LOSS_BOOM, lr=3e-3)]
    with pytest.raises(RuntimeError, match=r"^bad: forecasting failed for the refit block") as error:
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
