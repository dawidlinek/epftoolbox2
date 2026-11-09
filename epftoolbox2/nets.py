"""Networks for the neural-network families, and the counter-based layers that keep a stacked fit the same for each member.

Needs the `[torch]` extra (`pip install 'epftoolbox2[torch]'`); importing this module without torch raises `MissingExtraError`.
Nothing else of the library imports it, so `import epftoolbox2` never loads torch.

The module contract of the neural-network families: `__init__(n_inputs, n_outputs, **module_params)` and `forward(x)` with `x`
of shape (batch, n_inputs), returning (batch, n_outputs) in the library's scaled target space. A module never sees dates, rows
or targets.

The library's trainer (`epftoolbox2.models._torch`) trains several members of one fit (seeds, configurations) one after another
("loop") or together as one stack (`torch.func.vmap`, "vmap"), padded to a fixed size. Two layers here make a member's numbers
depend on that member alone, wherever it trains:

- `Dropout`: its mask is a hash of the member's key, the training step, the layer and the element, not a draw from torch's
  generator. `torch.nn.Dropout` under vmap draws one mask for the whole stack, so a member's mask would depend on its slot;
  the trainer refuses it there (it is allowed in the loop).
- `Linear`: `x @ W + b` with W of shape (in, out). Its initial weights come from the member's counter-based stream (uniform in
  plus or minus 1/sqrt(in), `torch.nn.Linear`'s default distribution), and stacked members run as one product with the bias per
  slot (`torch.baddbmm` on a GPU, each slot's own `torch.addmm` on fresh copies on the CPU), the arithmetic of the research's
  stacked trainer and the same in any slot.

`seed_counters(module, key)` is what the trainer calls on each member's module: it numbers the `Linear` and `Dropout` layers in
`module.modules()` order and gives them the member's key. Outside the trainer a `Dropout` repeats its mask until its `step`
buffer changes (`layer.step.fill_(k)`).
"""

import math
import threading
from typing import Dict, Sequence, Tuple

from ._extras import MissingExtraError

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - the core-only CI job has no torch
    raise MissingExtraError("epftoolbox2.nets", "torch") from None

__all__ = ["MLP", "Activation", "Dropout", "Linear", "seed_counters", "counter_uniform", "pcg_hash", "ACTIVATIONS"]

_M32 = 0xFFFFFFFF
# Knuth's multiplicative constant: a stream's tag is spread by it before it is added to the member's key (stk.py's net_uniform).
_SPREAD = 2654435761
# The stream of Linear layer k's weight is tag INIT_TAG + 2k, of its bias INIT_TAG + 2k + 1 (stk.py's `_init`).
INIT_TAG = 50_000_000
# A Dropout's stream at a training step is tag step * DROPOUT_LAYERS + its number (stk.py: `rng * 8 + k`).
DROPOUT_LAYERS = 8

# The activations of `Activation` and `MLP`, each as torch.nn's module of that name computes it (default parameters).
ACTIVATIONS = {
    "relu": torch.relu,
    "gelu": nn.functional.gelu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "softplus": nn.functional.softplus,
    "elu": nn.functional.elu,
    "selu": nn.functional.selu,
    "silu": nn.functional.silu,
    "leaky_relu": nn.functional.leaky_relu,
}


def pcg_hash(x: "torch.Tensor") -> "torch.Tensor":
    """The PCG output hash of int64 tensors holding 32-bit values (every product stays below 2**63)."""
    x = (x * 747796405 + 2891336453) & _M32
    w = (((x >> ((x >> 28) + 4)) ^ x) * 277803737) & _M32
    return (w >> 22) ^ w


def _spread(tag) -> "torch.Tensor":
    """(tag * _SPREAD) mod 2**32 for an int64 tensor `tag` >= 0, in 16-bit halves so that no product reaches 2**63."""
    tag = tag & _M32
    low, high = _SPREAD & 0xFFFF, _SPREAD >> 16
    return (tag * low + (((tag * high) & 0xFFFF) << 16)) & _M32


_INDEX_HASHES: Dict[Tuple[int, str], "torch.Tensor"] = {}
_INDEX_LOCK = threading.Lock()


def _index_hash(n: int, device: "torch.device") -> "torch.Tensor":
    """pcg_hash(arange(n)) on `device`, kept: every stream of n values hashes the same element numbers."""
    key = (n, str(device))
    found = _INDEX_HASHES.get(key)
    if found is None:
        found = pcg_hash(torch.arange(n, device=device, dtype=torch.int64))
        with _INDEX_LOCK:
            if len(_INDEX_HASHES) > 64:
                _INDEX_HASHES.clear()
            _INDEX_HASHES[key] = found
    return found


def _stream(key: "torch.Tensor", spread_tag: "torch.Tensor", n: int, device, dtype) -> "torch.Tensor":
    """n uniform(0, 1) values from the member key `key` (int64, below 2**32) and a spread tag (below 2**32)."""
    c = pcg_hash((key + spread_tag) & _M32)
    u = pcg_hash((c ^ _index_hash(n, device)) & _M32)
    return (u.to(dtype) + 0.5) / 4294967296.0


def counter_uniform(key: int, tag: int, shape: Sequence[int], device=None, dtype=None) -> "torch.Tensor":
    """Uniform(0, 1) values of `shape` from the stream (key, tag): the same on every device and in every stack, since it depends
    on the member's 32-bit `key` and the integer `tag` alone (stk.py's `net_uniform` for one member)."""
    device = torch.device("cpu") if device is None else torch.device(device)
    dtype = torch.get_default_dtype() if dtype is None else dtype
    n = int(math.prod(shape))
    base = torch.tensor(int(key) & _M32, dtype=torch.int64, device=device)
    spread = torch.tensor((int(tag) * _SPREAD) & _M32, dtype=torch.int64, device=device)
    return _stream(base, spread, n, device, dtype).view(*shape)


class Linear(nn.Module):
    """`x @ weight + bias` with `weight` of shape (in_features, out_features) and `bias` (1, out_features).

    The trainer draws the initial weights from the member's counter-based stream (`seed_counters`), uniform in plus or minus
    1/sqrt(in_features) as `torch.nn.Linear`'s default; stacked members (`stacked = True`, set by the trainer on the module it
    runs under vmap) are one call over the stack (`_define_stacked_linear`). Inputs with more than two dimensions multiply on the
    last one.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features, self.out_features = int(in_features), int(out_features)
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight = nn.Parameter(torch.empty(self.in_features, self.out_features).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(1, self.out_features).uniform_(-bound, bound))
        self.stacked = False

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        if self.stacked:
            return _stacked_linear(x, self.weight, self.bias)
        if x.dim() == 2:
            return torch.addmm(self.bias, x, self.weight)
        return torch.matmul(x, self.weight) + self.bias

    def reset_from_key(self, key: int, number: int) -> None:
        """Draw the weights from the stream of member `key`, as layer `number` (stk.py's `_init`: weight tag INIT_TAG + 2
        number, bias INIT_TAG + 2 number + 1)."""
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            for param, tag in ((self.weight, INIT_TAG + 2 * number), (self.bias, INIT_TAG + 2 * number + 1)):
                u = counter_uniform(key, tag, tuple(param.shape), device=param.device, dtype=param.dtype)
                param.copy_((2 * u - 1) * bound)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class Activation(nn.Module):
    """An elementwise activation by name (`ACTIVATIONS`), as torch.nn's module of that name computes it. Stacked members
    (`stacked = True`, set by the trainer) on the CPU take it slot by slot on fresh copies: over the whole stacked tensor a
    vectorised kernel gives the elements of its tail, which belong to the last slot, the scalar path, and for sigmoid, silu,
    elu or softplus other last bits than the member alone gets (`_define_stacked_activation`). On a GPU it runs over the
    stack as it is."""

    def __init__(self, name: str = "relu"):
        super().__init__()
        if name not in ACTIVATIONS:
            raise ValueError(f"Activation({name!r}): use one of {sorted(ACTIVATIONS)}")
        self.name = name
        self.stacked = False

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        if self.stacked and x.device.type == "cpu":
            return _stacked_activation(x, self.name)
        return ACTIVATIONS[self.name](x)  # a GPU's elementwise kernels compute every element alike

    def extra_repr(self) -> str:
        return self.name


class Dropout(nn.Module):
    """Dropout whose mask is a hash of (member key, training step, layer, element), not a draw from a generator.

    The trainer writes the member's key into the `seed` buffer, the layer's number into `number` and the step into `step`, so a
    member gets the same masks trained alone, in a loop or in a stack of any size. Identity in eval mode. The rate is a buffer,
    so members of one stack may differ in it.
    """

    def __init__(self, p: float = 0.1):
        super().__init__()
        p = float(p)
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout(p) needs 0 <= p < 1, got {p}")
        self.p = p
        self.number = 0
        self.register_buffer("rate", torch.tensor(p, dtype=torch.float32), persistent=False)
        self.register_buffer("seed", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("step", torch.zeros((), dtype=torch.int64), persistent=False)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        if not self.training:
            return x
        spread = _spread(self.step * DROPOUT_LAYERS + self.number)
        u = _stream(self.seed, spread, x.numel(), x.device, x.dtype).view(x.shape)
        return x * (u >= self.rate).to(x.dtype) / (1 - self.rate)

    def extra_repr(self) -> str:
        return f"p={self.p}"


def seed_counters(module: "nn.Module", key: int) -> None:
    """Give one member's module its counter-based streams: the `Linear` layers, numbered 0, 1, ... in `module.modules()` order,
    draw their weights from (key, number), and the `Dropout` layers, numbered the same way, take `key` as their seed. `key` is
    the member's 32-bit key."""
    linear = dropout = 0
    for sub in module.modules():
        if isinstance(sub, Linear):
            sub.reset_from_key(key, linear)
            linear += 1
        elif isinstance(sub, Dropout):
            sub.number = dropout
            sub.seed.fill_(int(key) & _M32)
            dropout += 1


def set_dropout_step(module: "nn.Module", step: int) -> None:
    """Write the training step into every `Dropout` of `module` (the loop's form; the stack writes its buffers directly)."""
    for sub in module.modules():
        if isinstance(sub, Dropout):
            sub.step.fill_(int(step))


class MLP(nn.Module):
    """A fully connected network: `Linear`, `Activation` and (when `dropout` > 0) `Dropout` for each hidden width, then a
    `Linear` to `n_outputs`. With `hidden=(256, 128)`, `dropout=0.1` and an output per period of the day it is the Lago-style
    DNN of the research."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        hidden: Sequence[int] = (256, 128),
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"MLP(activation={activation!r}): use one of {sorted(ACTIVATIONS)}")
        dropout = float(dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"MLP(dropout) needs 0 <= dropout < 1, got {dropout}")
        widths = [int(h) for h in hidden]
        if any(h < 1 for h in widths):
            raise ValueError(f"MLP(hidden) needs positive widths, got {list(hidden)}")
        layers, width = [], int(n_inputs)
        for h in widths:
            layers += [Linear(width, h), Activation(activation)]
            if dropout > 0:
                layers.append(Dropout(dropout))
            width = h
        layers.append(Linear(width, int(n_outputs)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


def _fresh(t: "torch.Tensor") -> "torch.Tensor":
    return torch.clone(t, memory_format=torch.contiguous_format)


def _per_slot_linear(x: "torch.Tensor", weight: "torch.Tensor", bias: "torch.Tensor") -> "torch.Tensor":
    """bias + x @ weight for each slot of (S, m, k), (S, k, n), (S, 1, n), each slot's product on fresh copies of its operands.

    On the CPU a batched product (`torch.baddbmm`, the BLAS's gemm per slot on views into the stack) can take another kernel for
    a slot whose matrices don't start on the same boundary as the first's, which moves the last bits of that slot's result with
    its place in the stack (seen with 7 x 12 inputs and 6 outputs). Fresh copies give every slot the same alignment, so a member
    is the same in any slot; on the research's shapes (247-256-128-24, batches of 64) the result equals `baddbmm`'s, forward and
    backward, to the bit. The product by 1.0 (exact) is there for the backward: it hands the slot's product a fresh copy of its
    gradient instead of a view into the stack's."""
    outs = [
        torch.addmm(_fresh(b), _fresh(xs), _fresh(w)) * 1.0
        for xs, w, b in zip(x.unbind(0), weight.unbind(0), bias.unbind(0))
    ]
    return torch.stack(outs)


def _define_stacked_linear():
    """`epftoolbox2::stacked_linear(x, weight, bias)`: under vmap `bias + x @ weight` for every slot in one call, not vmap's own
    rule for a product and a sum (`bmm` and then an add, other bits than the stacked trainer's): one `torch.baddbmm` over the
    stack on a GPU, whose batched kernel computes every slot alike, and `_per_slot_linear` on the CPU. Outside vmap it is the
    plain product."""
    name = "epftoolbox2::stacked_linear"
    existing = getattr(getattr(torch.ops, "epftoolbox2", None), "stacked_linear", None)
    if existing is not None:  # this module imported again (a reload): keep the first registration
        return existing

    @torch.library.custom_op(name, mutates_args=())
    def stacked_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, weight) + bias

    @stacked_linear.register_fake
    def _(x, weight, bias):
        return x.new_empty(*x.shape[:-1], weight.shape[-1])

    def batched(info, in_dims, x, weight, bias):
        size = info.batch_size

        def front(t, dim):
            return t.unsqueeze(0).expand(size, *t.shape) if dim is None else t.movedim(dim, 0)

        x, weight, bias = front(x, in_dims[0]), front(weight, in_dims[1]), front(bias, in_dims[2])
        lead = x.shape[1:-1]
        flat = x.reshape(size, -1, x.shape[-1])
        if flat.device.type == "cpu":
            out = _per_slot_linear(flat, weight, bias)
        else:
            out = torch.baddbmm(bias, flat, weight)
        return out.reshape(size, *lead, weight.shape[-1]), 0

    stacked_linear.register_vmap(batched)
    return stacked_linear


_stacked_linear = _define_stacked_linear()


def _define_stacked_activation():
    """`epftoolbox2::stacked_activation(x, name)`: under vmap the activation of every slot in one call: on the CPU slot by slot
    on fresh copies (the product by 1.0, exact, hands each slot's backward a fresh copy of its gradient too), on a GPU, whose
    elementwise kernels compute every element alike, over the whole stack."""
    existing = getattr(getattr(torch.ops, "epftoolbox2", None), "stacked_activation", None)
    if existing is not None:
        return existing

    @torch.library.custom_op("epftoolbox2::stacked_activation", mutates_args=())
    def stacked_activation(x: torch.Tensor, name: str) -> torch.Tensor:
        return ACTIVATIONS[name](x)

    @stacked_activation.register_fake
    def _(x, name):
        return torch.empty_like(x)

    def batched(info, in_dims, x, name):
        fn = ACTIVATIONS[name]
        if in_dims[0] is None:  # the same tensor for every slot
            return fn(x), None
        x = x.movedim(in_dims[0], 0)
        if x.device.type == "cpu":
            out = torch.stack([fn(_fresh(slot)) * 1.0 for slot in x.unbind(0)])
        else:
            out = fn(x)
        return out, 0

    stacked_activation.register_vmap(batched)
    return stacked_activation


_stacked_activation = _define_stacked_activation()
