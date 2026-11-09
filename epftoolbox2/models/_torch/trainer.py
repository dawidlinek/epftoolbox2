"""The trainer of the neural-network families: one call trains the members of one fit.

A port of the research's stacked trainer (`epftoolbox2-research/code_from_wcss/nn/code/stk.py`: `train`, `_init`, the
`net_uniform`/`pcg_hash` streams, padding, the fixed validation row count and steps per epoch, the best-weights restore) and of
the TorchModel prototype's `_torch_core.py` (loop and vmap, the vmap eligibility check, a loss by name or import path).
`epftoolbox2-research/library_check_step6/torch_core/` holds the check that this trainer, given stk.py's inputs and member keys,
gives stk.py's forecasts.

**Inputs.** `x` (n, k) and `y` (n, P) are already scaled (the unit standardises its inputs and transforms its targets); `y_mask`
(n, P) says which target slots exist (a DST gap, a missing value), and only those count in a loss. Each row is one local date
of the unit. A member is a `MemberSpec(seed_key, train_rows, lr)`: it trains on its `train_rows` alone, the rows of its span, and
never reads another row of `x` or `y`.

**What makes a member's numbers its own.** Every random choice comes from the member's `seed_key`: its early-stopping dates, a
random share of its own training rows (`member_split`, numpy's generator seeded by the key: stk.py's `split_indices`); its
32-bit stream key `pcg32(seed_key)`, from which come the initial weights of every `nets.Linear`, the order of its rows in each
epoch and the masks of every `nets.Dropout` (counter-based hashes, stk.py's `net_uniform`); torch's generator, seeded by the key
around the member's build (and, in the loop, its training) for any other layer. The shapes are fixed by the configuration, not by
the other members: `steps_per_epoch` from the nominal training rows (`nominal_rows`, the rows of a full window: a member with
fewer rows wraps around its own order, so every step is a full batch), the validation rows padded to the nominal count, and a
stack padded to `pad_to` members with dummies (copies of its first member with other keys, results discarded). Then a member
is bitwise the same alone, in a stack of 4 or 16 or in any order (`nn/REPORT.md` section 5: unpadded stacks drifted by 2.39
EUR/MWh with TF32, 7.29 without).

**Two ways to train them** (`stack`), which differ in arithmetic and so in numbers, recorded in the manifest:
- "loop": one member at a time with its own module. Any module works, `torch.nn.Dropout` and batch norm too.
- "vmap": the members of a stack together (`torch.func.stack_module_state`, `vmap(functional_call)`), one `StackedAdam` with a
  learning rate per member and exact freezing of stopped members. A stack has `pad_to` slots (`runtime.DEFAULT_PAD` unless set).
  `torch.nn.Dropout`, batch norm and any random operation in `forward` are refused: their numbers would depend on the stack.
  `nets.Linear` runs as one `baddbmm` over the stack on a GPU and as one product per slot on fresh copies on the CPU, stk.py's
  arithmetic either way and the same in any slot (a CPU's batched product isn't: `nets._per_slot_linear`); on the CPU a stack
  runs on one intra-op thread, so it is also the same at any thread count.
"auto" is the loop on a CPU, a stack on a GPU (decision 18).

**Early stopping**: each epoch ends with the member's loss on its early-stopping rows (in eval mode); a member whose loss didn't
improve for `patience` epochs stops (its weights then never move again) and its best weights are restored. With
`early_stopping=0` every member trains `max_epochs` and keeps its last weights. Losses: "l1", "mse", a "package.module:function"
or a callable, elementwise `(prediction, target) -> loss` in the scaled space; the library masks the slots without a target and
takes each member's mean over its batch (stk.py's `mean` over rows and outputs).

The caller owns the device, TF32 and thread scope (`runtime.torch_scope`) and the threads of the process.
"""

import contextlib
import copy
import hashlib
import importlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..._extras import MissingExtraError

try:
    import torch
    from torch.func import functional_call, stack_module_state, vmap
except ImportError:  # pragma: no cover - the core-only CI job has no torch
    raise MissingExtraError("the torch trainer", "torch") from None

from ... import nets
from .optim import Adam, StackedAdam
from .runtime import DTYPES, resolve_device, resolve_pad, resolve_stack

_M32 = 0xFFFFFFFF
_SPREAD = 2654435761
# A row order is drawn per epoch from tag epoch * _EPOCH_TAG (stk.py), a training step's stream is epoch * _EPOCH_TAG + step + 1.
_EPOCH_TAG = 1_000_000
# Padding in a hash key sorts after every real row (hashes are below 2**32).
_LAST = 1 << 62
# Keys of the dummy members that pad a stack; their results are discarded.
_DUMMY_KEY = 0x5EED_0000_0000


def _pcg32(x: int) -> int:
    """`nets.pcg_hash` of one Python int below 2**32."""
    x = (x * 747796405 + 2891336453) & _M32
    w = (((x >> ((x >> 28) + 4)) ^ x) * 277803737) & _M32
    return (w >> 22) ^ w


def stream_key(seed_key: int) -> int:
    """A member's 32-bit stream key: its initial weights, row orders and dropout masks (stk.py's `base`)."""
    return _pcg32(int(seed_key) & _M32)


def _torch_seed(seed_key: int, purpose: str) -> int:
    """A seed of torch's generator for one purpose of one member ("build", "train")."""
    digest = hashlib.blake2b(f"{int(seed_key)}:{purpose}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF_FFFF_FFFF


@dataclass(frozen=True)
class MemberSpec:
    """One member of a fit: `seed_key` (a non-negative int below 2**64, such as `_calendar.unit_seed`), the rows of `x` it trains
    on (`train_rows`, its span: the early-stopping rows are a share of them) and its learning rate (None: the call's `lr`)."""

    seed_key: int
    train_rows: Sequence[int]
    lr: Optional[float] = None


@dataclass(frozen=True)
class TrainResult:
    """`predict(x)`: (members, rows, outputs) float64 forecasts in the scaled target space, from each member's restored weights;
    `x` is (rows, k) for every member or (members, rows, k), one block per member. `best_epoch` per member (1-based; the last
    epoch without early stopping) and `stats` (epochs trained and best early-stopping loss per member, the stack, its size,
    the number of stacks, steps per epoch, early-stopping rows, nominal rows, seconds)."""

    predict: Callable[[np.ndarray], np.ndarray]
    best_epoch: np.ndarray
    stats: Dict[str, Any] = field(default_factory=dict)


def member_split(seed_key: int, train_rows: Sequence[int], early_stopping: float) -> Tuple[np.ndarray, np.ndarray]:
    """(fit rows, early-stopping rows) of a member, both sorted and taken from its `train_rows` only: a random
    round(early_stopping * n) of its n rows, from numpy's generator seeded by `seed_key` (stk.py's `split_indices`)."""
    rows = np.sort(np.asarray(train_rows, dtype=np.int64))
    if rows.ndim != 1 or (len(rows) > 1 and (np.diff(rows) == 0).any()):
        raise ValueError("train_rows must be distinct row numbers")
    n_val = int(round(early_stopping * len(rows))) if early_stopping > 0 else 0
    perm = np.random.default_rng(int(seed_key)).permutation(len(rows))
    return rows[np.sort(perm[n_val:])], rows[np.sort(perm[:n_val])]


def _l1(prediction, target):
    return (prediction - target).abs()


def _mse(prediction, target):
    return (prediction - target) ** 2


LOSSES: Dict[str, Callable] = {"l1": _l1, "mse": _mse}


def resolve_loss(loss: Union[str, Callable]) -> Callable:
    """The elementwise loss: "l1", "mse", "package.module:function" or a callable `(prediction, target) -> loss`."""
    if callable(loss):
        return loss
    if not isinstance(loss, str):
        raise TypeError(f"loss must be 'l1', 'mse', 'package.module:function' or a callable, got {loss!r}")
    if loss in LOSSES:
        return LOSSES[loss]
    module, _, qualname = loss.partition(":")
    if not module or not qualname:
        raise ValueError(f"loss={loss!r}: use 'l1', 'mse' or an elementwise function as 'package.module:function'")
    try:
        found: Any = importlib.import_module(module)
        for part in qualname.split("."):
            found = getattr(found, part)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot import the loss {loss!r}: {exc}") from exc
    if not callable(found):
        raise TypeError(f"The loss {loss!r} is not callable")
    return found


_STACK_BREAKERS = (
    torch.nn.modules.dropout._DropoutNd,
    torch.nn.modules.batchnorm._BatchNorm,
)


def check_vmappable(module: "torch.nn.Module") -> None:
    """Refuse a module whose stacked training would depend on the other members: torch's dropout (one mask drawn for the whole
    stack, so a member's mask depends on its slot), batch norm and instance norm with running statistics (updated in place).
    Random operations inside `forward` are refused by vmap itself (`randomness="error"`)."""
    for name, sub in module.named_modules():
        tracking = isinstance(sub, torch.nn.modules.instancenorm._InstanceNorm) and sub.track_running_stats
        if isinstance(sub, _STACK_BREAKERS) or tracking:
            where = f"{type(module).__name__}.{name}" if name else type(module).__name__
            raise ValueError(
                f"{where} is {type(sub).__name__}: with stack='vmap' a member's result would depend on the other members. "
                "Use epftoolbox2.nets.Dropout (counter-based), leave out batch norm, or train with stack='loop'."
            )


@dataclass
class _Member:
    key: int
    stream: int
    lr: float
    fit: np.ndarray
    val: np.ndarray


@dataclass
class _Shapes:
    nominal: int
    val_rows: int
    steps: int
    batch: int


def _inputs(blocks: Optional[Dict[str, slice]]):
    if blocks is None:
        return lambda x: x
    return lambda x: {name: x[..., sl] for name, sl in blocks.items()}


def _member_loss(loss_fn, out, target, mask):
    """Each member's mean loss over its rows and outputs, counting only slots with a target: (members,)."""
    err = loss_fn(out, target)
    if err.shape != out.shape:
        raise ValueError(f"The loss must be elementwise: it returned shape {tuple(err.shape)} for {tuple(out.shape)}")
    dims = (out.dim() - 2, out.dim() - 1)
    return torch.where(mask, err, torch.zeros_like(err)).sum(dims) / mask.sum(dims).to(err.dtype)


def _row_orders(
    streams: "torch.Tensor", epoch: int, fit_idx: "torch.Tensor", fit_mask: "torch.Tensor"
) -> "torch.Tensor":
    """Each member's order of its fit rows for one epoch (stk.py: the hash of its key and the epoch, padding last)."""
    c = nets.pcg_hash((streams + ((epoch * _EPOCH_TAG * _SPREAD) & _M32)) & _M32)
    j = nets.pcg_hash(torch.arange(fit_idx.shape[1], device=fit_idx.device, dtype=torch.int64))
    keys = nets.pcg_hash((c[:, None] ^ j[None, :]) & _M32).masked_fill(~fit_mask, _LAST)
    return torch.argsort(keys, dim=1, stable=True)


def _padded(rows: Sequence[np.ndarray], width: int) -> Tuple[np.ndarray, np.ndarray]:
    """(index, valid) arrays of shape (len(rows), width): each member's rows, padded with its first row."""
    idx = np.zeros((len(rows), width), dtype=np.int64)
    ok = np.zeros((len(rows), width), dtype=bool)
    for i, r in enumerate(rows):
        idx[i, : len(r)] = r
        idx[i, len(r) :] = r[0] if len(r) else 0
        ok[i, : len(r)] = True
    return idx, ok


def _build(module_factory: Callable[[], "torch.nn.Module"], member: _Member, device, dtype) -> "torch.nn.Module":
    """A member's module: built on the CPU with torch's generator seeded by the member (so a layer of torch's own starts the
    same on every device), moved to the device, then its counter-based layers seeded by its stream key there (the same values
    on every device: integer hashes and correctly rounded float steps)."""
    with torch.random.fork_rng(devices=_cuda_devices(device)):
        torch.manual_seed(_torch_seed(member.key, "build"))
        module = module_factory()
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"module_factory must return a torch.nn.Module, got {type(module).__name__}")
    module = module.to(device=device, dtype=dtype)
    nets.seed_counters(module, member.stream)
    return module


def _cuda_devices(device) -> List[int]:
    return [device.index or 0] if device.type == "cuda" else []


@contextlib.contextmanager
def _stack_threads(device):
    """A stack on the CPU runs on one intra-op thread: its products are one per slot (`nets._per_slot_linear`, so a member is the
    same in any slot), and a product's bits move with its thread count, so one thread makes a stack the same at any count.
    On a GPU nothing changes."""
    if torch.device(device).type != "cpu" or torch.get_num_threads() == 1:
        yield
        return
    saved = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(saved)


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_members(
    module_factory: Callable[[], "torch.nn.Module"],
    x: np.ndarray,
    y: np.ndarray,
    y_mask: Optional[np.ndarray],
    members: Sequence[MemberSpec],
    *,
    loss: Union[str, Callable] = "l1",
    lr: float = 3e-4,
    max_epochs: int = 300,
    patience: int = 20,
    batch_size: int = 64,
    early_stopping: float = 0.2,
    weight_decay: float = 0.0,
    stack: str = "loop",
    pad_to: Optional[int] = None,
    device: Any = "cpu",
    dtype: str = "float32",
    nominal_rows: Optional[int] = None,
    blocks: Optional[Dict[str, slice]] = None,
) -> TrainResult:
    """Train each member on its rows of (x, y) and return their restored networks (`TrainResult.predict`).

    `module_factory()` returns a fresh module (`(batch, k)` in, `(batch, P)` out, or a dict of `blocks` views of the columns
    in); `nominal_rows` is the training rows of a full window (default: the most any member has), which fixes the steps per
    epoch and the early-stopping rows, so pass it for a member to be the same whatever it trains with. See the module
    docstring for the rest."""
    if dtype not in DTYPES:
        raise ValueError(f"dtype={dtype!r}: use one of {DTYPES}")
    if not members:
        raise ValueError("train_members needs at least one member")
    if not 0.0 <= float(early_stopping) < 1.0:
        raise ValueError(f"early_stopping={early_stopping!r}: a share of the training rows in [0, 1)")
    if int(max_epochs) < 1 or int(batch_size) < 1 or int(patience) < 1:
        raise ValueError("max_epochs, batch_size and patience must be at least 1")
    resolved = resolve_device(device)
    kind = resolve_stack(stack, resolved)
    pad = resolve_pad(kind, pad_to)
    torch_dtype = getattr(torch, dtype)
    loss_fn = resolve_loss(loss)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError(
            f"x must be (rows, inputs) and y (rows, outputs) with the same rows, got {x.shape} and {y.shape}"
        )
    mask = np.isfinite(y) if y_mask is None else (np.asarray(y_mask, dtype=bool) & np.isfinite(y))
    if mask.shape != y.shape:
        raise ValueError(f"y_mask has shape {mask.shape}, y {y.shape}")

    plan = _plan(members, len(x), float(early_stopping), float(lr), nominal_rows, int(batch_size), mask)
    shapes = plan[1]
    used = np.unique(np.concatenate([np.concatenate([m.fit, m.val]) for m in plan[0]]))
    if not np.isfinite(x[used]).all():
        raise ValueError("x has a value that is not finite on a training row of a member")

    X = torch.as_tensor(x, dtype=torch_dtype, device=resolved)
    Y = torch.as_tensor(np.where(mask, y, 0.0), dtype=torch_dtype, device=resolved)
    M = torch.as_tensor(mask, device=resolved)
    data = (X, Y, M)
    settings = dict(
        loss_fn=loss_fn,
        max_epochs=int(max_epochs),
        patience=int(patience),
        early=float(early_stopping) > 0,
        weight_decay=float(weight_decay),
        inputs=_inputs(blocks),
        device=resolved,
        dtype=torch_dtype,
    )
    started = time.perf_counter()
    with torch.enable_grad():
        if kind == "loop":
            fits = [_LoopFit(module_factory, member, data, shapes, settings) for member in plan[0]]
        else:
            fits = []
            with _stack_threads(resolved):
                for start in range(0, len(plan[0]), pad):
                    fits.append(_StackFit(module_factory, plan[0][start : start + pad], pad, data, shapes, settings))
    _sync(resolved)
    seconds = time.perf_counter() - started
    n_members = len(plan[0])
    best_epoch = np.concatenate([f.best_epoch for f in fits])[:n_members]
    stats = {
        "epochs": np.concatenate([f.epochs for f in fits])[:n_members],
        "best_val": np.concatenate([f.best_val for f in fits])[:n_members],
        "stack": kind,
        "pad_to": pad,
        "stacks": len(fits) if kind == "vmap" else 0,
        "steps_per_epoch": shapes.steps,
        "val_rows": shapes.val_rows,
        "nominal_rows": shapes.nominal,
        "train_seconds": seconds,
    }

    def predict(rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim == 2:
            rows = np.broadcast_to(rows, (n_members, *rows.shape))
        if rows.ndim != 3 or rows.shape[0] != n_members or rows.shape[2] != x.shape[1]:
            raise ValueError(
                f"predict takes (rows, {x.shape[1]}) or ({n_members}, rows, {x.shape[1]}), got {rows.shape}"
            )
        out, done = [], 0
        for f in fits:
            take = f.real
            block = np.array(rows[done : done + take], dtype=np.float64)
            out.append(f.predict(torch.as_tensor(block, dtype=torch_dtype, device=resolved)))
            done += take
        return np.concatenate(out)

    return TrainResult(predict=predict, best_epoch=best_epoch, stats=stats)


def _plan(members, n_rows, early, lr, nominal_rows, batch, mask) -> Tuple[List[_Member], _Shapes]:
    out = []
    for i, spec in enumerate(members):
        key = int(spec.seed_key)
        if key < 0 or key >= 1 << 64:
            raise ValueError(f"member {i}: seed_key must be a non-negative int below 2**64, got {spec.seed_key!r}")
        fit, val = member_split(key, spec.train_rows, early)
        rows = np.concatenate([fit, val])
        if len(rows) and (rows.min() < 0 or rows.max() >= n_rows):
            raise ValueError(f"member {i}: train_rows outside the {n_rows} rows of x")
        if len(fit) == 0:
            raise ValueError(f"member {i}: no training rows left (it has {len(rows)})")
        if early and not mask[val].any():
            raise ValueError(
                f"member {i}: early_stopping={early} leaves it {len(val)} early-stopping rows with no target; give it more "
                "training rows or set early_stopping=0"
            )
        out.append(_Member(key, stream_key(key), lr if spec.lr is None else float(spec.lr), fit, val))
    largest = max(len(m.fit) + len(m.val) for m in out)
    nominal = largest if nominal_rows is None else int(nominal_rows)
    if nominal < largest:
        raise ValueError(f"nominal_rows={nominal_rows} is below the {largest} training rows of a member")
    val_rows = int(round(early * nominal)) if early > 0 else 0
    fit_rows = int(round((1 - early) * nominal)) if early > 0 else nominal
    return out, _Shapes(nominal, val_rows, max(1, math.ceil(fit_rows / batch)), batch)


class _LoopFit:
    """One member alone with its own module: the loop."""

    real = 1

    def __init__(self, module_factory, member: _Member, data, shapes: _Shapes, s: Dict[str, Any]):
        X, Y, M = data
        device = s["device"]
        self.inputs = s["inputs"]
        self.seed = _torch_seed(member.key, "train")
        self.devices = _cuda_devices(device)
        with torch.random.fork_rng(devices=self.devices):
            torch.manual_seed(self.seed)
            module = _build(module_factory, member, device, s["dtype"])
            self.module = module
            params = [p for p in module.parameters() if p.requires_grad]
            opt = Adam(params, member.lr, weight_decay=s["weight_decay"])
            fit = torch.as_tensor(member.fit, device=device)
            fit_mask = torch.ones((1, len(member.fit)), dtype=torch.bool, device=device)
            stream = torch.tensor([member.stream], dtype=torch.int64, device=device)
            arange = torch.arange(shapes.batch, device=device)
            n_fit = len(member.fit)
            dropouts = [sub for sub in module.modules() if isinstance(sub, nets.Dropout)]
            if s["early"]:
                val_idx, val_ok = _padded([member.val], shapes.val_rows)
                vi = torch.as_tensor(val_idx[0], device=device)
                xv, yv = X[vi], Y[vi]
                mv = M[vi] & torch.as_tensor(val_ok[0], device=device)[:, None]
            # The best state is the module's state_dict (its parameters and saved buffers, a batch norm's running statistics
            # too); nets.Dropout's buffers aren't saved, so a restore leaves them.
            best = {k: v.detach().clone() for k, v in module.state_dict().items()}
            best_loss, bad, best_epoch, epoch = math.inf, 0, s["max_epochs"], 0
            for epoch in range(1, s["max_epochs"] + 1):
                module.train()
                order = fit[_row_orders(stream, epoch, fit[None, :], fit_mask)[0]]
                for step in range(shapes.steps):
                    idx = order[(step * shapes.batch + arange) % n_fit]
                    for sub in dropouts:
                        sub.step.fill_(epoch * _EPOCH_TAG + step + 1)
                    out = module(self.inputs(X[idx]))
                    per = _member_loss(s["loss_fn"], out[None], Y[idx][None], M[idx][None])[0]
                    loss = torch.where(torch.isfinite(per), per, torch.zeros_like(per))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                if not s["early"]:
                    continue
                module.eval()
                with torch.no_grad():
                    vl = float(_member_loss(s["loss_fn"], module(self.inputs(xv))[None], yv[None], mv[None])[0])
                if not math.isfinite(vl):
                    vl = math.inf
                if vl < best_loss:
                    best_loss, bad, best_epoch = vl, 0, epoch
                    with torch.no_grad():
                        for k, v in module.state_dict().items():
                            best[k].copy_(v)
                else:
                    bad += 1
                    if bad >= s["patience"]:
                        break
            if s["early"]:
                module.load_state_dict(best)
        module.eval()
        self.best_epoch = np.array([best_epoch])
        self.epochs = np.array([epoch])
        self.best_val = np.array([best_loss if s["early"] else math.nan])

    def predict(self, rows: "torch.Tensor") -> np.ndarray:
        with torch.random.fork_rng(devices=self.devices), torch.no_grad():
            torch.manual_seed(self.seed)
            return self.module(self.inputs(rows[0])).to(torch.float64).cpu().numpy()[None]


class _StackFit:
    """The members of one stack together, padded to `pad` slots: vmap over their stacked modules."""

    def __init__(self, module_factory, chunk: List[_Member], pad: int, data, shapes: _Shapes, s: Dict[str, Any]):
        X, Y, M = data
        device = s["device"]
        self.real = len(chunk)
        first = chunk[0]
        slots = chunk + [
            _Member(_DUMMY_KEY + j, stream_key(_DUMMY_KEY + j), first.lr, first.fit, first.val)
            for j in range(pad - len(chunk))
        ]
        modules = [_build(module_factory, member, device, s["dtype"]) for member in slots]
        check_vmappable(modules[0])
        base = copy.deepcopy(modules[0])
        for sub in base.modules():
            if isinstance(sub, nets.Linear):
                sub.stacked = True
        params, buffers = stack_module_state(modules)
        del modules
        self.base, self.params, self.buffers = base, params, buffers
        step_buffers = [
            f"{name}.step" if name else "step" for name, sub in base.named_modules() if isinstance(sub, nets.Dropout)
        ]
        inputs = s["inputs"]

        def one(p, b, rows):
            return functional_call(base, (p, b), (inputs(rows),))

        self.forward = vmap(one, in_dims=(0, 0, 0), randomness="error")
        S = len(slots)
        trained = [name for name, p in params.items() if p.requires_grad]
        plist = [params[name] for name in trained]
        opt = StackedAdam(plist, [m.lr for m in slots], weight_decay=s["weight_decay"])
        fit_idx, fit_ok = _padded([m.fit for m in slots], max(len(m.fit) for m in slots))
        fit_idx = torch.as_tensor(fit_idx, device=device)
        fit_ok = torch.as_tensor(fit_ok, device=device)
        n_fit = fit_ok.sum(1)
        streams = torch.as_tensor([m.stream for m in slots], dtype=torch.int64, device=device)
        arange = torch.arange(shapes.batch, device=device)
        if s["early"]:
            val_idx, val_ok = _padded([m.val for m in slots], shapes.val_rows)
            vi = torch.as_tensor(val_idx, device=device)
            xv, yv = X[vi], Y[vi]
            mv = M[vi] & torch.as_tensor(val_ok, device=device)[..., None]
        best = {name: params[name].detach().clone() for name in trained}
        best_loss = torch.full((S,), math.inf, dtype=s["dtype"], device=device)
        bad = torch.zeros(S, dtype=torch.long, device=device)
        active = torch.ones(S, dtype=torch.bool, device=device)
        epochs = torch.zeros(S, dtype=torch.long, device=device)
        best_epoch = torch.full((S,), s["max_epochs"], dtype=torch.long, device=device)
        for epoch in range(1, s["max_epochs"] + 1):
            base.train()
            epochs += active.long()
            order = _row_orders(streams, epoch, fit_idx, fit_ok)
            for step in range(shapes.steps):
                pos = torch.gather(order, 1, (step * shapes.batch + arange)[None, :] % n_fit[:, None])
                idx = torch.gather(fit_idx, 1, pos)
                for name in step_buffers:
                    buffers[name].fill_(epoch * _EPOCH_TAG + step + 1)
                out = self.forward(params, buffers, X[idx])
                per = _member_loss(s["loss_fn"], out, Y[idx], M[idx])
                loss = torch.where(active & torch.isfinite(per), per, torch.zeros_like(per)).sum()
                opt.zero_grad()
                loss.backward()
                opt.step(active)
            if not s["early"]:
                continue
            base.eval()
            with torch.no_grad():
                vl = _member_loss(s["loss_fn"], self.forward(params, buffers, xv), yv, mv)
                vl = torch.where(torch.isfinite(vl), vl, torch.full_like(vl, math.inf))
                improved = (vl < best_loss) & active
                best_loss = torch.where(improved, vl, best_loss)
                best_epoch = torch.where(improved, torch.full_like(best_epoch, epoch), best_epoch)
                for name in trained:
                    p = params[name]
                    best[name].copy_(torch.where(improved.view((S,) + (1,) * (p.dim() - 1)), p, best[name]))
                bad = torch.where(improved, torch.zeros_like(bad), bad + active.long())
                active = active & (bad < s["patience"])
            if not bool(active[: self.real].any()):
                break
        with torch.no_grad():
            if s["early"]:
                for name in trained:
                    params[name].copy_(best[name])
            for p in params.values():
                p.requires_grad_(False)
        base.eval()
        self.slots = S
        self.best_epoch = best_epoch.cpu().numpy()[: self.real]
        self.epochs = epochs.cpu().numpy()[: self.real]
        self.best_val = (best_loss.cpu().numpy() if s["early"] else np.full(S, math.nan))[: self.real]

    def predict(self, rows: "torch.Tensor") -> np.ndarray:
        """rows (real members, m, k), padded to the stack's slots with copies of the first member's."""
        if rows.shape[0] < self.slots:
            rows = torch.cat([rows, rows[:1].expand(self.slots - rows.shape[0], *rows.shape[1:])])
        with torch.no_grad(), _stack_threads(rows.device):
            out = self.forward(self.params, self.buffers, rows.contiguous())
        return out[: self.real].to(torch.float64).cpu().numpy()
