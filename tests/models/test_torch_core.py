"""The torch core (step 6g): nets, the trainer's determinism, early stopping, the TF32 and thread scope, thread pinning in the
pool initialiser, and the fine-tune helper. Needs the [torch] extra (the torch-extra CI job); skipped without it.

The determinism claims are the research's (`torchmodel/REPORT.md` section 3, `nn/REPORT.md` section 5): a member is bitwise the
same alone and in a padded stack of any size or order, at any thread count, and with other learning rates in the stack; the
loop at one thread is bitwise the same in another process. `epftoolbox2-research/library_check_step6/torch_core/` holds the
check against the research trainer (stk.py) and the measures; the GPU tests here (`-m gpu`) need a CUDA device.
"""

import hashlib
import math
import multiprocessing
import subprocess
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from epftoolbox2 import nets  # noqa: E402
from epftoolbox2.models import OLSModel  # noqa: E402
from epftoolbox2.models._torch import finetune, runtime, trainer  # noqa: E402
from epftoolbox2.models._torch.optim import StackedAdam  # noqa: E402
from epftoolbox2.models._torch.trainer import MemberSpec, member_split, train_members  # noqa: E402
from epftoolbox2.models._worker import _pool_init  # noqa: E402

# Subprocesses import the library from this checkout, whatever the working directory of the test run.
ROOT = Path(__file__).resolve().parents[2]
K, P, ROWS = 20, 6, 240
FAST = dict(max_epochs=6, patience=3, lr=1e-2, batch_size=32, nominal_rows=200)


def data(seed=0, n=ROWS, k=K, outputs=P):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, k))
    y = np.tanh(x[:, :outputs] @ rng.normal(size=(outputs, outputs))) + 0.1 * rng.normal(size=(n, outputs))
    return x, y, rng.normal(size=(7, k))


def mlp(dropout=0.1, hidden=(24, 12)):
    return lambda: nets.MLP(K, P, hidden=hidden, dropout=dropout)


def spans(n_members, first=0):
    """Each member its own span of rows (windows of different starts and lengths, all within the nominal 200)."""
    return [np.arange(first + 3 * i, first + 3 * i + 150 + 7 * (i % 5)) for i in range(n_members)]


def fit(keys, rows, stack="vmap", factory=None, lrs=None, x=None, y=None, mask=None, **kw):
    if x is None:
        x, y, _ = data()
    members = [MemberSpec(k, r, None if lrs is None else lrs[i]) for i, (k, r) in enumerate(zip(keys, rows))]
    settings = dict(FAST, **kw)
    return train_members(factory or mlp(), x, y, mask, members, stack=stack, **settings)


@pytest.fixture
def torch_state():
    """Put back torch's thread count and precision flags after a test that changes them."""
    threads = torch.get_num_threads()
    state = runtime.precision_state()
    yield
    runtime._restore_precision(state)
    torch.set_num_threads(threads)


# --- streams and layers -----------------------------------------------------------------------------------------------


def _stk_pcg_hash(x):
    x = (x * 747796405 + 2891336453) & 0xFFFFFFFF
    w = (((x >> ((x >> 28) + 4)) ^ x) * 277803737) & 0xFFFFFFFF
    return (w >> 22) ^ w


def _stk_net_uniform(base, tag, shape_tail):
    """stk.py's (dnn.py's) net_uniform, copied: the stream the library must keep."""
    c = _stk_pcg_hash((base + ((int(tag) * 2654435761) & 0xFFFFFFFF)) & 0xFFFFFFFF)
    n = int(np.prod(shape_tail))
    j = torch.arange(n, dtype=torch.int64)
    u = _stk_pcg_hash((c[:, None] ^ _stk_pcg_hash(j)[None, :]) & 0xFFFFFFFF)
    return (u.float() + 0.5) / 4294967296.0


def test_counter_streams_are_the_research_trainers():
    keys = [0, 1, 2**31 + 5, 2**32 - 1]
    base = torch.tensor(keys, dtype=torch.int64)
    for tag in (0, 7, nets.INIT_TAG, 300 * 1_000_000 * 8 + 3):
        ref = _stk_net_uniform(base, tag, (3, 5))
        mine = torch.stack([nets.counter_uniform(k, tag, (15,), dtype=torch.float32) for k in keys])
        assert torch.equal(ref, mine)
    assert trainer.stream_key(12 * 1000003 + 4) == int(_stk_pcg_hash(torch.tensor((12 * 1000003 + 4) & 0xFFFFFFFF)))


def test_dropout_mask_is_the_stream_of_its_step_and_number_and_identity_in_eval():
    layer = nets.Dropout(0.3)
    x = torch.ones(4, 50)
    layer.seed.fill_(123)
    layer.step.fill_(1_000_001)
    layer.number = 1
    out = layer(x)
    u = _stk_net_uniform(torch.tensor([123]), 1_000_001 * 8 + 1, (4, 50)).view(4, 50)
    assert torch.equal(out, x * (u >= torch.tensor(0.3)).float() / (1 - torch.tensor(0.3)))
    assert 0.5 < float((out == 0).float().mean()) / 0.3 < 1.5
    layer.step.fill_(1_000_002)
    assert not torch.equal(layer(x), out)
    layer.eval()
    assert torch.equal(layer(x), x)
    with pytest.raises(ValueError):
        nets.Dropout(1.0)


def test_mlp_layers_and_their_counter_init():
    net = nets.MLP(5, 3, hidden=[8, 4], dropout=0.2, activation="tanh")
    kinds = [type(m).__name__ for m in net.net]
    assert kinds == ["Linear", "Tanh", "Dropout", "Linear", "Tanh", "Dropout", "Linear"]
    assert [type(m).__name__ for m in nets.MLP(5, 3, hidden=[8], dropout=0.0).net] == ["Linear", "ReLU", "Linear"]
    nets.seed_counters(net, 77)
    first, last = net.net[0], net.net[6]
    assert torch.equal(first.weight, (2 * nets.counter_uniform(77, nets.INIT_TAG, (5, 8)) - 1) * (1 / math.sqrt(5)))
    assert torch.equal(last.bias, (2 * nets.counter_uniform(77, nets.INIT_TAG + 5, (1, 3)) - 1) * (1 / math.sqrt(4)))
    assert [m.number for m in net.modules() if isinstance(m, nets.Dropout)] == [0, 1]
    assert net(torch.zeros(2, 5)).shape == (2, 3)
    for bad in (dict(activation="swish"), dict(dropout=1.2), dict(hidden=[0])):
        with pytest.raises(ValueError):
            nets.MLP(5, 3, **bad)


def test_stacked_linear_computes_each_slot_as_the_member_alone_in_any_slot():
    """7 x 12 inputs and 6 outputs: a shape where the CPU's batched product (baddbmm) gives odd slots other bits."""
    from torch.func import functional_call, stack_module_state, vmap

    layers = [nets.Linear(12, 6) for _ in range(8)]
    for i, layer in enumerate(layers):
        nets.seed_counters(layer, i)
    base = nets.Linear(12, 6)
    base.stacked = True
    run = vmap(lambda p, rows: functional_call(base, p, (rows,)))
    x = torch.randn(8, 7, 12)
    params, _ = stack_module_state(layers)
    out = run(params, x)
    for i, layer in enumerate(layers):
        assert torch.equal(out[i], layer(x[i].clone()))
    same = {
        name: p.detach()[:1].expand(8, *p.shape[1:]).contiguous().requires_grad_(True) for name, p in params.items()
    }
    rows = x[:1].expand(8, 7, 12).contiguous().requires_grad_(True)
    result = run(same, rows)
    (result * torch.linspace(-1, 1, 42).view(7, 6)).sum().backward()
    for t in (result, same["weight"].grad, same["bias"].grad, rows.grad):
        assert all(torch.equal(t[0], t[s]) for s in range(8))


# --- determinism of the trainer --------------------------------------------------------------------------------------


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_a_member_alone_equals_it_in_padded_stacks_at_each_thread_count(threads, torch_state):
    keys = [101 + i for i in range(16)]
    rows = spans(16)
    x, y, issue = data()
    with runtime.torch_scope("cpu", threads=threads):
        alone = fit(keys[:1], rows[:1], pad_to=16).predict(issue)
        four = fit(keys[:4], rows[:4], pad_to=16).predict(issue)
        sixteen = fit(keys, rows, pad_to=16).predict(issue)
        alone4 = fit(keys[:1], rows[:1], pad_to=4).predict(issue)
        in4 = fit(keys[:4], rows[:4], pad_to=4).predict(issue)
    assert np.array_equal(alone[0], four[0]) and np.array_equal(alone[0], sixteen[0])
    assert np.array_equal(four, sixteen[:4])
    assert np.array_equal(alone4[0], in4[0])
    with runtime.torch_scope("cpu", threads=1):
        reference = fit(keys[:4], rows[:4], pad_to=16).predict(issue)
    assert np.array_equal(four, reference), "a padded stack moved with the thread count"


def test_learning_rate_variants_in_one_stack_equal_each_alone():
    lrs = [3e-3, 1e-2, 3e-2, 5e-2] * 2
    keys = [7, 7, 7, 7, 8, 8, 8, 8]
    rows = spans(8)
    _, _, issue = data()
    together = fit(keys, rows, lrs=lrs, pad_to=8).predict(issue)
    for i in (0, 3, 6):
        alone = fit([keys[i]], [rows[i]], lrs=[lrs[i]], pad_to=8).predict(issue)
        assert np.array_equal(alone[0], together[i])
    assert not np.array_equal(together[0], together[1])


@pytest.mark.parametrize("stack", ["vmap", "loop"])
def test_permuting_members_leaves_each_members_result_unchanged(stack):
    keys = [31, 32, 33, 34, 35]
    rows = spans(5)
    _, _, issue = data()
    order = [3, 0, 4, 2, 1]
    a = fit(keys, rows, stack=stack, pad_to=8)
    b = fit([keys[i] for i in order], [rows[i] for i in order], stack=stack, pad_to=8)
    pa, pb = a.predict(issue), b.predict(issue)
    for position, i in enumerate(order):
        assert np.array_equal(pa[i], pb[position])
        assert a.best_epoch[i] == b.best_epoch[position]


def test_per_member_blocks_predict_each_members_own_rows():
    keys, rows = [1, 2, 3], spans(3)
    x, y, issue = data()
    result = fit(keys, rows, pad_to=4)
    blocks = np.stack([issue, issue * 2, issue[::-1]])
    each = result.predict(blocks)
    for i in range(3):
        assert np.array_equal(each[i], result.predict(blocks[i])[i])


_LOOP_SCRIPT = textwrap.dedent(
    """
    import hashlib, sys
    import numpy as np
    import torch
    from epftoolbox2 import nets
    from epftoolbox2.models._torch.trainer import MemberSpec, train_members
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(240, 20))
    y = np.tanh(x[:, :6] @ rng.normal(size=(6, 6))) + 0.1 * rng.normal(size=(240, 6))
    issue = rng.normal(size=(7, 20))
    members = [MemberSpec(5, np.arange(0, 150)), MemberSpec(6, np.arange(40, 200))]
    result = train_members(lambda: nets.MLP(20, 6, hidden=(24, 12), dropout=0.1), x, y, None, members, stack="loop",
                           max_epochs=6, patience=3, lr=1e-2, batch_size=32, nominal_rows=200)
    sys.stdout.write(hashlib.sha256(result.predict(issue).tobytes()).hexdigest())
    """
)


def test_the_loop_at_one_thread_is_bitwise_the_same_in_other_processes(torch_state):
    hashes = [
        subprocess.run(
            [sys.executable, "-c", _LOOP_SCRIPT], capture_output=True, text=True, check=True, cwd=ROOT
        ).stdout.strip()
        for _ in range(2)
    ]
    x, y, issue = data()
    with runtime.torch_scope("cpu", threads=1):  # the forecasts too: the loop's products move with the thread count
        here = train_members(
            mlp(),
            x,
            y,
            None,
            [MemberSpec(5, np.arange(0, 150)), MemberSpec(6, np.arange(40, 200))],
            stack="loop",
            **FAST,
        )
        digest = hashlib.sha256(here.predict(issue).tobytes()).hexdigest()
    assert hashes[0] == hashes[1] == digest


def test_a_member_never_reads_a_row_outside_its_span():
    x, y, issue = data()
    keys, rows = [11, 12], [np.arange(20, 170), np.arange(60, 200)]
    used = np.unique(np.concatenate(rows))
    for stack in ("vmap", "loop"):
        ref = fit(keys, rows, stack=stack, pad_to=4, x=x, y=y).predict(issue)
        x2, y2 = x.copy(), y.copy()
        outside = np.setdiff1d(np.arange(ROWS), used)
        x2[outside], y2[outside] = np.nan, 1e9
        assert np.array_equal(ref, fit(keys, rows, stack=stack, pad_to=4, x=x2, y=y2).predict(issue))
        x3, y3 = x.copy(), y.copy()
        y3[rows[0][-1]] += 100.0  # a row of the first member's span (and the second's) moves both
        assert not np.array_equal(ref[0], fit(keys, rows, stack=stack, pad_to=4, x=x3, y=y3).predict(issue)[0])


def test_the_early_stopping_split_holds_training_dates_only():
    rows = np.arange(30, 180)
    fit_rows, val_rows = member_split(99, rows, 0.2)
    assert len(val_rows) == round(0.2 * len(rows)) and len(fit_rows) + len(val_rows) == len(rows)
    assert set(fit_rows) | set(val_rows) == set(rows) and not set(fit_rows) & set(val_rows)
    assert np.all(np.diff(fit_rows) > 0) and np.all(np.diff(val_rows) > 0)
    again = member_split(99, rows[::-1], 0.2)  # the set of rows decides, not their order
    assert np.array_equal(again[0], fit_rows) and np.array_equal(again[1], val_rows)
    assert not np.array_equal(member_split(100, rows, 0.2)[1], val_rows)
    assert len(member_split(99, rows, 0.0)[1]) == 0


def test_masked_target_slots_do_not_count():
    x, y, issue = data()
    mask = np.ones_like(y, dtype=bool)
    mask[::5, 2] = False
    keys, rows = [3], [np.arange(0, 180)]
    ref = fit(keys, rows, mask=mask, x=x, y=y).predict(issue)
    y2 = y.copy()
    y2[~mask] = np.nan
    assert np.array_equal(ref, fit(keys, rows, x=x, y=y2).predict(issue))  # NaN is a missing slot too
    y2[~mask] = 1e6
    assert np.array_equal(ref, fit(keys, rows, mask=mask, x=x, y=y2).predict(issue))


class Planted:
    """L1, until `after` training steps: then gradient ascent, so the weights run away and the early-stopping loss (computed
    without grad, so always plain L1) gets worse."""

    def __init__(self, after):
        self.after, self.calls = after, 0

    def __call__(self, prediction, target):
        err = (prediction - target).abs()
        if torch.is_grad_enabled():
            self.calls += 1
            if self.calls > self.after:
                return -50.0 * err
        return err


@pytest.mark.parametrize("stack", ["vmap", "loop"])
def test_early_stopping_restores_the_best_weights_after_a_planted_divergence(stack):
    _, _, issue = data()
    keys, rows = [21], [np.arange(0, 200)]
    steps = trainer._plan([MemberSpec(21, rows[0])], ROWS, 0.2, 1e-2, 200, 32, np.ones((ROWS, P), bool))[1].steps
    diverge_after = 5
    planted = fit(keys, rows, stack=stack, loss=Planted(diverge_after * steps), max_epochs=40, patience=4, lr=3e-3)
    best = int(planted.best_epoch[0])
    assert 1 <= best <= diverge_after < planted.stats["epochs"][0]
    plain = fit(keys, rows, stack=stack, max_epochs=best, patience=4, lr=3e-3)
    assert int(plain.best_epoch[0]) == best
    assert np.array_equal(planted.predict(issue), plain.predict(issue))


def test_without_early_stopping_every_member_trains_max_epochs_and_keeps_its_last_weights():
    result = fit([1, 2], spans(2), early_stopping=0.0, max_epochs=4, pad_to=2)
    assert result.best_epoch.tolist() == [4, 4] and result.stats["epochs"].tolist() == [4, 4]
    assert result.stats["val_rows"] == 0


class TorchDropoutNet(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, 8), torch.nn.Dropout(0.2), torch.nn.Linear(8, n_outputs)
        )

    def forward(self, x):
        return self.net(x)


class BatchNormNet(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, 8), torch.nn.BatchNorm1d(8), torch.nn.Linear(8, n_outputs)
        )

    def forward(self, x):
        return self.net(x)


class RandomNet(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.lin = torch.nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        return self.lin(x) + 0.01 * torch.randn_like(x[:, :1])


@pytest.mark.parametrize("module", [TorchDropoutNet, BatchNormNet])
def test_torch_dropout_and_batch_norm_are_refused_under_vmap_and_run_in_the_loop(module):
    with pytest.raises(ValueError, match="nets.Dropout"):
        fit([1], spans(1), factory=lambda: module(K, P), stack="vmap", pad_to=2)
    _, _, issue = data()
    alone = fit([1], spans(1), factory=lambda: module(K, P), stack="loop").predict(issue)
    pair = fit([2, 1], [spans(2)[1], spans(1)[0]], factory=lambda: module(K, P), stack="loop").predict(issue)
    assert np.array_equal(alone[0], pair[1]), "the loop seeds torch's generator per member"


def test_a_random_operation_in_forward_is_refused_under_vmap():
    with pytest.raises(RuntimeError, match="random"):
        fit([1], spans(1), factory=lambda: RandomNet(K, P), stack="vmap", pad_to=2)


def l1_times_two(prediction, target):  # a loss by import path: elementwise, the library masks and reduces
    return 2 * (prediction - target).abs()


def test_losses_by_name_import_path_and_callable():
    _, _, issue = data()
    by_path = fit([4], spans(1), loss=f"{__name__}:l1_times_two").predict(issue)
    by_callable = fit([4], spans(1), loss=l1_times_two).predict(issue)
    assert np.array_equal(by_path, by_callable)
    assert np.isfinite(fit([4], spans(1), loss="mse").predict(issue)).all()
    with pytest.raises(ValueError, match="elementwise"):
        fit([4], spans(1), loss=lambda p, t: (p - t).abs().mean())
    for bad in ("huber", "nomodule:"):
        with pytest.raises(ValueError):
            fit([4], spans(1), loss=bad)


def test_named_blocks_reach_the_module_as_views():
    class Blocks(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nets.Linear(8, P), nets.Linear(K - 8, P)

        def forward(self, blocks):
            return self.a(blocks["a"]) + self.b(blocks["b"])

    _, _, issue = data()
    blocks = {"a": slice(0, 8), "b": slice(8, K)}
    alone = fit([5], spans(1), factory=Blocks, blocks=blocks, pad_to=4).predict(issue)
    pair = fit([6, 5], spans(2)[::-1], factory=Blocks, blocks=blocks, pad_to=4).predict(issue)
    assert np.array_equal(alone[0], pair[1])


def test_shapes_are_fixed_by_the_configuration_and_bad_calls_raise():
    result = fit([1], [np.arange(100)], nominal_rows=200)
    assert result.stats["steps_per_epoch"] == 5 and result.stats["val_rows"] == 40  # round(0.8 * 200) / 32; 0.2 * 200
    with pytest.raises(ValueError, match="nominal_rows"):
        fit([1], [np.arange(230)], nominal_rows=200)
    with pytest.raises(ValueError, match="outside"):
        fit([1], [np.arange(ROWS - 5, ROWS + 5)])
    with pytest.raises(ValueError, match="seed_key"):
        fit([-1], [np.arange(50)])
    x, y, _ = data()
    x[10, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit([1], [np.arange(50)], x=x, y=y)


def test_stacked_adam_freezes_an_inactive_member_exactly():
    p = torch.zeros(3, 2, requires_grad=True)
    opt = StackedAdam([p], [0.1, 0.2, 0.1])
    for active in ([True, True, True], [True, False, True]):
        before = p.detach().clone()
        (p * torch.tensor([[1.0], [1.0], [2.0]])).sum().backward()
        opt.step(torch.tensor(active))
        opt.zero_grad()
    assert torch.equal(p[1], before[1]) and not torch.equal(p[0], before[0])
    assert torch.allclose(p[0], -0.2 * torch.ones(2)) and torch.allclose(p[2], -0.2 * torch.ones(2))


# --- the scope, the runtime fields and the pool initialiser ----------------------------------------------------------


def _legacy():
    return (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )


@pytest.mark.parametrize("raise_inside", [False, True])
def test_torch_scope_puts_back_tf32_flags_precision_and_threads(raise_inside, torch_state):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("medium")
    torch.set_num_threads(3)
    before, state = _legacy(), runtime.precision_state()

    class Boom(Exception):
        pass

    try:
        with runtime.torch_scope("cpu", tf32=False, threads=1) as device:
            assert device == torch.device("cpu")
            assert _legacy() == (False, False, "highest") and torch.get_num_threads() == 1
            with runtime.torch_scope("cpu", tf32=True, threads=2):
                assert _legacy() == (True, True, "high") and torch.get_num_threads() == 2
            assert _legacy() == (False, False, "highest") and torch.get_num_threads() == 1
            if raise_inside:
                raise Boom
    except Boom:
        assert raise_inside
    assert _legacy() == before and runtime.precision_state() == state and torch.get_num_threads() == 3


def test_torch_scope_puts_back_a_state_set_through_the_new_interface(torch_state):
    if not hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        pytest.skip("this torch has no fp32_precision attributes")
    torch.backends.cuda.matmul.fp32_precision = (
        "tf32"  # the legacy flags can't express this mix: they refuse to be read
    )
    state = runtime.precision_state()
    with runtime.torch_scope("cpu", tf32=False, threads=None):
        assert torch.backends.cuda.matmul.fp32_precision == "ieee"
    assert runtime.precision_state() == state


def test_runtime_fields_are_resolved_settings():
    fields = runtime.runtime_fields(device="cpu")
    assert fields == {
        "device": "cpu",
        "dtype": "float32",
        "tf32": False,
        "threads_per_fit": 1,
        "stack": "loop",
        "pad_to": None,
    }
    assert runtime.runtime_fields(device="cpu", stack="vmap")["pad_to"] == runtime.DEFAULT_PAD
    assert runtime.runtime_fields(
        device="cpu", stack="vmap", pad_to=256, dtype="float64", tf32=True, threads_per_fit=2
    ) == {
        "device": "cpu",
        "dtype": "float64",
        "tf32": True,
        "threads_per_fit": 2,
        "stack": "vmap",
        "pad_to": 256,
    }
    for bad in (dict(dtype="float16"), dict(stack="stacked"), dict(stack="vmap", pad_to=0)):
        with pytest.raises(ValueError):
            runtime.runtime_fields(device="cpu", **bad)


def test_device_resolution_without_a_usable_gpu():
    if runtime.cuda_probe(0)[0]:
        pytest.skip("a usable GPU is present")
    assert runtime.resolve_device("auto") == torch.device("cpu")
    assert runtime.resolve_device(None) == torch.device("cpu")
    assert runtime.resolve_device("cpu") == torch.device("cpu")
    assert runtime.device_name("auto") == "cpu"
    with pytest.raises(RuntimeError, match="device='cuda'"):
        runtime.resolve_device("cuda")


def _report_threads():
    return torch.get_num_threads(), torch.get_num_interop_threads()


def test_the_pool_initialiser_pins_torch_threads_in_a_forked_worker(torch_state):
    torch.set_num_threads(3)
    torch.ones(64, 64) @ torch.ones(64, 64)  # the parent has used its thread pool
    for wanted in (1, 2):
        with ProcessPoolExecutor(
            1,
            mp_context=multiprocessing.get_context("fork"),
            initializer=_pool_init,
            initargs=({}, {}, {"torch_threads": wanted}, OLSModel, {}),
        ) as pool:
            assert pool.submit(_report_threads).result()[0] == wanted


def test_the_pool_initialiser_pins_torch_in_a_fresh_process_and_imports_none_without_the_setting():
    code = textwrap.dedent(
        """
        import sys
        from epftoolbox2.models import OLSModel
        from epftoolbox2.models._worker import _pool_init
        _pool_init({}, {}, {}, OLSModel, {})
        assert "torch" not in sys.modules, "a core model's pool imported torch"
        _pool_init({}, {}, {"torch_threads": 2}, OLSModel, {})
        import torch
        print(torch.get_num_threads(), torch.get_num_interop_threads())
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT
    ).stdout.split()
    assert out == ["2", "1"]


def test_importing_the_library_and_the_torch_package_imports_no_torch():
    code = textwrap.dedent(
        """
        import sys
        import epftoolbox2, epftoolbox2.models, epftoolbox2.pipelines
        import epftoolbox2.models._torch, epftoolbox2.models._torch.runtime
        assert "torch" not in sys.modules, sorted(m for m in sys.modules if m.startswith("torch"))
        sys.modules["torch"] = None  # as if the extra were missing
        from epftoolbox2._extras import MissingExtraError
        for name in ("epftoolbox2.nets", "epftoolbox2.models._torch.trainer"):
            try:
                __import__(name)
            except MissingExtraError as exc:
                assert exc.extra == "torch", exc
            else:
                raise AssertionError(name + " imported without torch")
        print("ok")
        """
    )
    assert (
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT
        ).stdout.strip()
        == "ok"
    )


# --- fine_tune --------------------------------------------------------------------------------------------------------


def _windows(n_days=120, cut=None, bias=3.0):
    """Windows of a module that should learn a bias: one window per day, its loss the squared error of its target."""
    days = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-01") + n_days)
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.normal(size=(n_days, 1)), dtype=torch.float32)
    y = 2 * x + bias

    def loss(model, idx):
        idx = torch.as_tensor(np.asarray(idx))
        return ((model(x[idx]) - y[idx]) ** 2).mean(dim=1)

    return finetune.FineTuneWindows(dates=days, cut=days[-1] if cut is None else cut, loss=loss), days


def test_fine_tune_refuses_a_window_after_the_cut_before_any_step():
    torch.manual_seed(0)
    model = torch.nn.Linear(1, 1)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    windows, days = _windows(cut=np.datetime64("2024-03-01"))
    with pytest.raises(AssertionError, match="after the cut"):
        finetune.fine_tune(model, windows)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_fine_tune_holds_out_the_newest_days_and_returns_the_best_step_and_data_end():
    torch.manual_seed(0)
    model = torch.nn.Linear(1, 1)
    windows, days = _windows()
    train_idx, hold_idx, end = finetune.split_holdout(days, 30)
    assert end == days[-1] and len(hold_idx) == 30 and set(hold_idx) == set(range(90, 120))
    rng_state = torch.random.get_rng_state()
    result = finetune.fine_tune(model, windows, lr=0.05, max_steps=200, batch_size=8, holdout_days=30, eval_every=10)
    assert torch.equal(torch.random.get_rng_state(), rng_state), "the caller's generator moved"
    assert result.data_end == days[-1].astype(object) and result.stats["holdout_windows"] == 30
    assert result.stats["train_windows"] == 90
    assert result.best_step > 0 and result.best_step % 10 == 0
    assert result.stats["best_loss"] < result.stats["base_loss"]
    assert dict(result.stats["holdout"])[result.best_step] == result.stats["best_loss"]
    assert all(torch.equal(result.state_dict[k], v) for k, v in model.state_dict().items())
    torch.manual_seed(0)
    again = finetune.fine_tune(
        torch.nn.Linear(1, 1), windows, lr=0.05, max_steps=200, batch_size=8, holdout_days=30, eval_every=10
    )
    assert again.best_step == result.best_step
    assert all(torch.equal(again.state_dict[k], result.state_dict[k]) for k in result.state_dict)


def test_fine_tune_keeps_the_starting_weights_when_nothing_improves():
    torch.manual_seed(0)
    model = torch.nn.Linear(1, 1)
    start = {k: v.clone() for k, v in model.state_dict().items()}
    windows, _ = _windows()
    result = finetune.fine_tune(model, windows, lr=0.0, max_steps=60, eval_every=10, patience=2, holdout_days=30)
    assert result.best_step == 0 and result.stats["steps"] == 20
    assert all(torch.equal(start[k], v) for k, v in model.state_dict().items())


# --- GPU (WCSS H100; `-m gpu`, never in the fast suite) ---------------------------------------------------------------


def _gpu():
    if not runtime.cuda_probe(0)[0]:
        pytest.skip("needs a usable CUDA device")
    return torch.device("cuda:0")


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize("tf32", [False, True])
def test_on_gpu_a_member_alone_equals_it_in_16_and_in_256_padded(tf32, torch_state):
    _gpu()
    rng = np.random.default_rng(1)
    n, k = 900, 247
    x = rng.normal(size=(n, k))
    y = np.tanh(x[:, :24] + 0.5 * x[:, 24:48]) + 0.1 * rng.normal(size=(n, 24))
    issue = rng.normal(size=(7, k))
    keys = [int(v) for v in rng.integers(0, 2**62, 256)]
    rows = [np.arange(s, s + 728) for s in rng.integers(0, n - 728, 256)]
    kw = dict(max_epochs=20, patience=5, lr=1e-3, batch_size=64, nominal_rows=728, pad_to=256, device="cuda")

    def run(order):
        members = [MemberSpec(keys[i], rows[i]) for i in order]
        factory = lambda: nets.MLP(k, 24, hidden=(256, 128), dropout=0.1)  # noqa: E731
        with runtime.torch_scope("cuda", tf32=tf32, threads=1):
            return train_members(factory, x, y, None, members, stack="vmap", **kw).predict(issue)

    alone, sixteen, full = run([0]), run(range(16)), run(range(256))
    assert np.array_equal(alone[0], sixteen[0]) and np.array_equal(alone[0], full[0])
    assert np.array_equal(sixteen, full[:16])
    assert np.array_equal(run(range(15, -1, -1))[::-1], sixteen), "a member moved with its slot"
