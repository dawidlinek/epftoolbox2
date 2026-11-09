"""Where and how torch runs a fit: the device, TF32 and thread counts, scoped, and the settings a manifest records.

- `resolve_device("auto")` is CUDA when a 1x1 matmul runs on it, else the CPU (decision 20: a GPU torch has no kernels for, such
  as the GTX 1050 Ti's sm_61 in today's wheels, fails that matmul with "no kernel image" and falls back). The probe runs in a
  short-lived subprocess, once per process, so this process never initialises CUDA: a pool forked from it can still use the GPU.
- `torch_scope(device, tf32=False, threads=1)` sets the TF32 flags (`torch.backends.cuda.matmul.allow_tf32`,
  `torch.backends.cudnn.allow_tf32`, `torch.set_float32_matmul_precision`) and torch's intra-op thread count for one fit and puts
  back what was there on exit, also after an exception, so one model's settings never reach the next model of the process (TF32
  moved Chronos-2 by up to 0.071 EUR/MWh). The precision state is read and restored through both of torch's interfaces (the
  flags above and, from torch 2.9, the `fp32_precision` attributes, which torch keeps separately), so a later reader of either
  sees what it saw before.
- `pin_torch_threads(n)` is the pool initialiser's (`_worker._pool_init` with `config["torch_threads"]`): environment variables set
  after a fork are too late, a forked worker inherits the parent's thread count.
- `runtime_fields(...)` are the settings that change a torch family's numbers, for `BaseModel._runtime_fields`: the manifest
  records them and a resume with other values is refused.
"""

import contextlib
import functools
import json
import os
import subprocess
import sys
from typing import Any, Dict, Iterator, Optional, Tuple

from ..._extras import MissingExtraError

# Stacks are padded to this many members unless `pad_to` says otherwise (`trainer.train_members`).
DEFAULT_PAD = 16
STACKS = ("auto", "loop", "vmap")
DTYPES = ("float32", "float64")

# torch >= 2.9's precision attributes under torch.backends, parents before the operations they set (setting a parent sets its
# children): restoring in this order gives back every value.
_PRECISION_PARENTS = ("fp32_precision", "cudnn.fp32_precision", "mkldnn.fp32_precision")
_PRECISION_LEAVES = (
    "cuda.matmul.fp32_precision",
    "cudnn.conv.fp32_precision",
    "cudnn.rnn.fp32_precision",
    "mkldnn.matmul.fp32_precision",
    "mkldnn.conv.fp32_precision",
    "mkldnn.rnn.fp32_precision",
)

_PROBE = r"""
import json, sys
try:
    import torch
    out = {"available": bool(torch.cuda.is_available())}
    if out["available"]:
        a = torch.ones(1, 1, device="cuda:%d")
        out["value"] = float((a @ a).item())
        out["name"] = torch.cuda.get_device_name(%d)
        out["capability"] = list(torch.cuda.get_device_capability(%d))
except Exception as exc:
    out = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
sys.stdout.write(json.dumps(out))
"""


def _torch():
    try:
        import torch
    except ImportError:
        raise MissingExtraError("the torch trainer", "torch") from None
    return torch


@functools.lru_cache(maxsize=None)
def cuda_probe(index: int = 0) -> Tuple[bool, Optional[str], str]:
    """(usable, GPU name, reason) for CUDA device `index`, from a 1x1 matmul in a subprocess. A torch built without CUDA (the CPU
    wheel) answers False without starting one."""
    torch = _torch()
    if not torch.backends.cuda.is_built():
        return False, None, "this torch is built without CUDA"
    code = _PROBE % (index, index, index)
    try:
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=dict(os.environ)
        )
        answer = json.loads(done.stdout.strip().splitlines()[-1])
    except (subprocess.SubprocessError, OSError, ValueError, IndexError) as exc:
        return False, None, f"the CUDA probe failed: {exc}"
    if answer.get("available") and answer.get("value") == 1.0:
        return True, answer.get("name"), "ok"
    return False, None, answer.get("error") or "torch.cuda.is_available() is False"


def resolve_device(device: Any = "auto"):
    """The `torch.device` a fit runs on: "auto" is CUDA when `cuda_probe` passes, else the CPU; "cuda" or "cuda:N" must pass the
    probe (else `RuntimeError` with its reason); anything else is passed to `torch.device` as given."""
    torch = _torch()
    if isinstance(device, torch.device):
        device = str(device)
    name = "auto" if device is None else str(device)
    if name == "auto":
        return torch.device("cuda:0") if cuda_probe(0)[0] else torch.device("cpu")
    resolved = torch.device(name)
    if resolved.type == "cuda":
        index = 0 if resolved.index is None else resolved.index
        usable, _, reason = cuda_probe(index)
        if not usable:
            raise RuntimeError(
                f"device={device!r}: CUDA device {index} can't run a matmul ({reason}). Use device='cpu'."
            )
        return torch.device("cuda", index)
    return resolved


def device_name(device: Any) -> str:
    """ "cpu", or the GPU's name: recorded where a change only warns (bits are not known to hold across GPU types)."""
    resolved = resolve_device(device)
    if resolved.type != "cuda":
        return resolved.type
    return cuda_probe(resolved.index or 0)[1] or "cuda"


def resolve_stack(stack: str, device) -> str:
    """ "loop" or "vmap": "auto" is a loop on the CPU (a pool of single-thread fits is the fast way there) and a padded stack on
    a GPU (decision 18)."""
    if stack not in STACKS:
        raise ValueError(f"stack={stack!r}: use one of {STACKS}")
    if stack != "auto":
        return stack
    return "vmap" if resolve_device(device).type == "cuda" else "loop"


def resolve_pad(stack: str, pad_to: Optional[int]) -> Optional[int]:
    """The stack size of a vmap fit (`pad_to`, or DEFAULT_PAD); None for the loop, which has no stack."""
    if stack == "loop":
        return None
    pad = DEFAULT_PAD if pad_to is None else int(pad_to)
    if pad < 1:
        raise ValueError(f"pad_to={pad_to!r}: a stack holds at least one member")
    return pad


def runtime_fields(
    *,
    device: Any = "auto",
    dtype: str = "float32",
    tf32: bool = False,
    threads_per_fit: int = 1,
    stack: str = "auto",
    pad_to: Optional[int] = None,
) -> Dict[str, Any]:
    """The settings of a torch fit that change its numbers, resolved, for the manifest's `runtime` (compared on resume): the
    device type, dtype, `tf32`, torch's thread count per fit, the stack ("loop" or "vmap") and its size. The GPU's name is not
    among them (`device_name`): a different GPU should warn, not refuse."""
    if dtype not in DTYPES:
        raise ValueError(f"dtype={dtype!r}: use one of {DTYPES}")
    resolved = resolve_device(device)
    kind = resolve_stack(stack, resolved)
    return {
        "device": resolved.type,
        "dtype": dtype,
        "tf32": bool(tf32),
        "threads_per_fit": int(threads_per_fit),
        "stack": kind,
        "pad_to": resolve_pad(kind, pad_to),
    }


def pin_torch_threads(threads: int) -> None:
    """torch's intra-op threads to `threads` and its inter-op threads to 1 in this process (a pool worker). The inter-op count
    can be set once per process, before any inter-op work: a worker forked from a parent that set it or used it keeps the
    parent's (torch raises, and this leaves it)."""
    torch = _torch()
    threads = int(threads)
    if threads < 1:
        raise ValueError(f"torch threads must be at least 1, got {threads}")
    if torch.get_num_threads() != threads:
        torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


# --- TF32 and threads ------------------------------------------------------------------------------------------------------


def _attribute(root, path: str):
    for part in path.split("."):
        root = getattr(root, part)
    return root


def _set_attribute(root, path: str, value) -> None:
    *parents, last = path.split(".")
    for part in parents:
        root = getattr(root, part)
    setattr(root, last, value)


def _has_attribute(root, path: str) -> bool:
    try:
        _attribute(root, path)
    except (AttributeError, RuntimeError):
        return False
    return True


def precision_state() -> Dict[str, Any]:
    """What `torch_scope` puts back: the legacy flags (None when torch refuses to read them because the two interfaces were
    mixed) and every `fp32_precision` attribute this torch has."""
    torch = _torch()
    try:
        legacy: Optional[Tuple[bool, bool, str]] = (
            bool(torch.backends.cuda.matmul.allow_tf32),
            bool(torch.backends.cudnn.allow_tf32),
            torch.get_float32_matmul_precision(),
        )
    except RuntimeError:
        legacy = None
    paths = [p for p in _PRECISION_PARENTS + _PRECISION_LEAVES if _has_attribute(torch.backends, p)]
    return {"legacy": legacy, "precision": {p: _attribute(torch.backends, p) for p in paths}}


def _restore_precision(state: Dict[str, Any]) -> None:
    torch = _torch()
    legacy = state["legacy"]
    if legacy is not None:
        torch.backends.cuda.matmul.allow_tf32 = legacy[0]
        torch.backends.cudnn.allow_tf32 = legacy[1]
        torch.set_float32_matmul_precision(legacy[2])
    for path, value in state["precision"].items():
        _set_attribute(torch.backends, path, value)


def _set_tf32(tf32: bool, legacy_readable: bool) -> None:
    torch = _torch()
    if legacy_readable:
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        return
    # Someone set the new interface in a way the legacy flags can't express: stay on the new one (as the legacy "high" does,
    # TF32 on reaches oneDNN's matmul too).
    value = "tf32" if tf32 else "ieee"
    for path in _PRECISION_LEAVES:
        if _has_attribute(torch.backends, path):
            _set_attribute(torch.backends, path, value)


@contextlib.contextmanager
def torch_scope(device: Any = "auto", tf32: bool = False, threads: Optional[int] = 1) -> Iterator[Any]:
    """Run a fit on `device` (resolved, yielded) with TF32 on or off and torch's intra-op threads at `threads` (None leaves
    them), and put back the precision flags and the thread count as they were on exit, after an exception too. The inter-op
    count is not touched (torch allows setting it once per process; `pin_torch_threads`)."""
    torch = _torch()
    resolved = resolve_device(device)
    saved = precision_state()
    saved_threads = torch.get_num_threads()
    try:
        _set_tf32(bool(tf32), saved["legacy"] is not None)
        if threads is not None and int(threads) != saved_threads:
            torch.set_num_threads(int(threads))
        yield resolved
    finally:
        _restore_precision(saved)
        if torch.get_num_threads() != saved_threads:
            torch.set_num_threads(saved_threads)
