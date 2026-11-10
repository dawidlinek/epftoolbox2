"""What the in-context tabular regressors share (step 6e): `TabPFNModel` and `TabICLModel` (`models/tabpfn.py`,
`models/tabicl.py`).

An in-context model is fitted by showing it its training rows (the context) and asked for the predictive distribution of new
rows, whose median is the forecast. Here each runs as a period unit on the refit calendar (`refit="7D"` by default): a fit
keeps the unit's rows (raw predictors, the target through epftoolbox's Invariant, `INVARIANT_TARGET`) and one call predicts
every issue row of its block (`_predict_block`, step 6d), padded to the block's nominal length, so the context is paid for
once a block and a forecast doesn't depend on which days a run holds. The median is taken in the transformed target space and
mapped back by the scaler's inverse: the transform is monotone, so the median maps exactly.

**The backend.** A family's package is imported only when a worker first fits (`_backend`), never when the class is imported,
built, saved or loaded: someone who only fetches data never imports torch or a model package. The backend is loaded once per
worker process from the resolved local weights file (`_cards.resolve_model_weights`: the cache `epf weights fetch` filled, or
`weights=`), inside `_weights.offline()`, with the path of the one file it needs, never a directory: the network is built
there, once, and every fit's regressor is handed it (TabPFN's `ModelSpecs`, TabICL's loaded model), so no fit reaches the
package's own loader and its download paths (tabpfn's include a direct URL that `HF_HUB_OFFLINE` doesn't stop); every fit and
prediction runs inside `offline()` too. `_load_backend(resolved, device)` is each family's; tests replace it with a numpy fake
(`tests/models/in_context_fakes.py`), or the real package's `fit`/`predict` with fakes in the extras job.

**Devices.** The parent resolves the run's devices once (`_run_devices`: "auto" is CUDA when GPU 0's probe passes, and then
every GPU of GPU 0's model, name and compute capability, whose probe passes, each probe in a subprocess, GPUs of another model
left out with a warning; else the CPU; "cuda:N" that GPU) and hands them to the workers (`config["in_context_devices"]`);
worker k fits on device k modulo their number (`cuda:{worker % n_gpus}`, k from the last part of its multiprocessing
identity), checks that it can use it, and stops the run if it can't, so a report never mixes CPU and GPU forecasts or two GPU
models. Nothing here starts CUDA in the parent before its pool forks. What a package would size by a GPU's free memory is
pinned (TabPFN's memory saving off, TabICL's batches at 50,000 and its outputs on the GPU) and recorded.

**Determinism.** Every fit and its predictions run in `torch_scope(device, tf32=False, threads=1)` (one torch thread, TF32
off, both put back after), and a run's workers pin torch to one thread (`config["torch_threads"]`, `_worker._pool_init`), so a
forecast is the same in any process; the seed (`random_state`) is the same for every fit, as the research ran them. The
manifest's `runtime` records the package's, torch's, numpy's, scipy's and scikit-learn's versions (the card's pins), the
device type, its kernels (`cpu_capability` or `cuda_capability`), the precision the fits compute in (`precision`) and the GPU's or
CPU's name (warn-only), all but the name compared on resume. On a CPU both run in float32, pinned (tabpfn would autocast to
bfloat16 on a CPU with AVX512-BF16 or AMX, whose capability reads "AVX512" too); on a GPU they autocast to float16 (TabICL for
contexts of 1,024 rows or 60 columns), so a CPU and a GPU forecast differ by more than rounding: verify on the CPU
(`verify(device="cpu")`).

**Errors.** Whatever the package raises in loading, a fit or a prediction stops the run as a RuntimeError naming the model,
never a `fit_error` skip that would leave windows silently without forecasts. A context whose targets are all equal (a period whose
target is always 0) predicts that value without calling the package.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import importlib.util
import multiprocessing
import numbers
import os
import re
import subprocess
import sys
import threading
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .._extras import MissingExtraError
from . import _worker
from ._cards import resolve_model_weights
from ._weights import ResolvedWeights, offline
from .base import BaseModel, _usable_cpu_count
from .hgb import INVARIANT_TARGET

# `device`: "auto" (CUDA when a 1x1 matmul runs on it, else the CPU), "cpu", "cuda" or "cuda:N".
DEVICE = re.compile(r"(auto|cpu|cuda(:[0-9]+)?)\Z")
# torch threads of one fit (`torch_scope`), recorded as `threads_per_fit` and compared on resume.
THREADS_PER_FIT = 1
# Processes a run starts per GPU by default (`MAX_PROCESSES` overrides): the research ran 4 to 6 per GPU.
GPU_PROCESSES = 6
# The precision a fit computes in, by device type, as `_runtime_fields` records it (a family's `_PRECISION`).
FLOAT32 = "float32"

# The loaded backends of this process: (class, weights file, its sha256, device, settings) -> backend.
_BACKENDS: Dict[tuple, Any] = {}
_BACKENDS_LOCK = threading.Lock()
# The devices this worker process checked it can use (`InContextModel._check_worker_device`).
_CHECKED_DEVICES: set = set()


def has_torch() -> bool:
    """Whether torch can be imported here (found, not imported)."""
    return importlib.util.find_spec("torch") is not None


@functools.lru_cache(maxsize=None)
def cuda_device_count() -> int:
    """How many GPUs torch sees, counted in a subprocess (this process must not start CUDA before its pool forks); 0 where
    torch or CUDA is missing or the count can't be read."""
    code = "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"
    try:
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        return int(done.stdout.strip().splitlines()[-1]) if done.returncode == 0 else 0
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 0


def worker_index() -> int:
    """This pool worker's number among the processes its parent started, counted from 0 (0 outside a pool): which of the run's
    devices it fits on (`InContextModel._device_text`). multiprocessing numbers a process by its ancestry, `(1,)`, `(2,)`, ...
    for a parent's children and `(3, 1)`, `(3, 2)`, ... for the children of its third child, so the last part is the one that
    tells a pool's workers apart (the first is the same for every worker of a pool started inside a child). A parent's count
    goes on across its pools, so a second pool's workers continue it; modulo the number of devices they still spread."""
    identity = multiprocessing.current_process()._identity
    return identity[-1] - 1 if identity else 0


def package(name: str, model: "InContextModel") -> Any:
    """`import name`, or `MissingExtraError` naming the model's extra where the package isn't installed (a run's pre-flight
    finds that first; this is for a direct call). Any other import error is the package's own and goes through as it is."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name is None or error.name.split(".")[0] != name.split(".")[0]:
            raise
        raise MissingExtraError(type(model).__name__, model._card.extra) from error


class _Constant:
    """A context whose targets are all equal: every row's median is that value."""

    def __init__(self, value: float):
        self.value = float(value)


class Fitted:
    """A unit's fit: the backend that holds the loaded model and what its `fit` returned (the context)."""

    __slots__ = ("backend", "context")

    def __init__(self, backend: Any, context: Any):
        self.backend = backend
        self.context = context


class InContextModel(BaseModel):
    """What `TabPFNModel` and `TabICLModel` share: the period unit on the refit calendar with one block call, the device and
    thread scope, the per-process backend and the card's checks. A family defines `_card`, `_family_kwargs()` (its settings
    that reach the workers), `_backend_settings()` (those its backend is built with) and `_load_backend(resolved, device)`."""

    _engine_ok = False
    # The precision a fit computes in on each device type, recorded in the manifest's `runtime` and compared on resume.
    _PRECISION: Dict[str, str] = {"cpu": FLOAT32}
    # A PooledModel's pool and a ResidualModel's learner stage would run without the torch thread pin (`_worker_inputs`).
    _pooled_ok = False
    _residual_learner_ok = False
    # `weights` is written only when set (a path), so a default model's YAML has no `weights: null`.
    _WRITTEN_WHEN_SET = {**BaseModel._WRITTEN_WHEN_SET, "weights": None}

    def _setup(
        self,
        refit: Optional[str],
        random_state: int,
        device: str,
        weights: Optional[str],
        block_predict: bool,
    ) -> None:
        self.random_state = random_state
        self.device = device
        self.weights = weights
        self.block_predict = block_predict
        # `refit` last: BaseModel's check reads the class's fit methods.
        self.refit = refit

    def __setattr__(self, key, value):
        if key == "random_state":
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise ValueError(f"random_state must be a whole number (the seed of every fit), got {value!r}")
            value = int(value)
        if key == "n_estimators":
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError(f"n_estimators must be a whole number from 1, got {value!r}")
            value = int(value)
        if key == "device" and not (isinstance(value, str) and DEVICE.match(value)):
            raise ValueError(f"device must be 'auto', 'cpu', 'cuda' or 'cuda:N', got {value!r}")
        if key == "block_predict" and not isinstance(value, bool):
            raise ValueError(f"block_predict must be True or False, got {value!r}")
        if key == "weights" and value is not None:
            if not isinstance(value, (str, os.PathLike)):
                raise ValueError(f"weights must be None, a path or '<owner>/<name>@<revision>', got {value!r}")
            value = os.fspath(value)
        super().__setattr__(key, value)

    # --- what reaches the workers and the manifest ------------------------------------------------------------------

    def _family_kwargs(self) -> Dict[str, Any]:  # pragma: no cover - every family defines it
        raise NotImplementedError

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        """The family's settings, the seed and the device, and `weights` where set: a worker rebuilds the model from these
        and resolves its weights itself. `block_predict` is the parent's (`config["block_predict"]`, the manifest's
        `model_params`)."""
        out = {**self._family_kwargs(), "random_state": self.random_state, "device": self.device}
        if self.weights is not None:
            out["weights"] = self.weights
        return out

    def _resolved_device(self) -> Any:
        """The `torch.device` the model's `device` names (`_torch.runtime.resolve_device`: "auto" probes CUDA in a subprocess,
        so this process never starts CUDA before its pool forks); None where torch isn't installed (a test's numpy fake)."""
        if not has_torch():
            return None
        from ._torch.runtime import resolve_device

        return resolve_device(self.device)

    def _run_devices(self) -> Tuple[str, ...]:
        """The devices a run's fits use, resolved once in this process for the model's `device` (and kept for it): "cpu";
        "cuda:N" for that GPU; for "auto" or "cuda", CUDA when GPU 0's probe passes (`_torch.runtime.resolve_device`), and then
        every GPU of GPU 0's model (its name and compute capability) whose probe passes (`cuda_device_count`, `cuda_probe`,
        `cuda_capability`, each in a subprocess), so worker k fits on GPU k modulo their number. GPUs of another model are
        left out with a warning: under float16 autocast two models give other numbers, and which unit runs where depends on
        the scheduling, so a run over both would not repeat (name them with `device="cuda:N"` or `CUDA_VISIBLE_DEVICES`).
        ("cpu",) without torch."""
        memo = vars(self).get("_devices_memo")
        if memo is not None and memo[0] == self.device:
            return memo[1]
        resolved = self._resolved_device()
        if resolved is None:
            devices: Tuple[str, ...] = ("cpu",)
        elif resolved.type == "cuda" and self.device in ("auto", "cuda"):
            devices = self._same_model_gpus(resolved.index or 0) or (str(resolved),)
        else:
            devices = (str(resolved),)
        self._devices_memo = (self.device, devices)
        return devices

    def _same_model_gpus(self, reference: int) -> Tuple[str, ...]:
        """Every GPU whose probe passes and whose (name, compute capability) is GPU `reference`'s, warning about the others."""
        from ._torch.runtime import cuda_capability, cuda_probe

        def model_of(index: int) -> Tuple[Optional[str], Optional[str]]:
            return cuda_probe(index)[1], cuda_capability(index)

        usable = [index for index in range(cuda_device_count()) if cuda_probe(index)[0]]
        if reference not in usable:
            return ()
        kept = [index for index in usable if model_of(index) == model_of(reference)]
        left = [index for index in usable if index not in kept]
        if left:
            others = ", ".join(f"cuda:{i} ({model_of(i)[0]}, capability {model_of(i)[1]})" for i in left)
            warnings.warn(
                f"{type(self).__name__}({self.name!r}): device={self.device!r} uses cuda:{reference}'s model "
                f"({model_of(reference)[0]}, capability {model_of(reference)[1]}) alone and leaves out {others}: on two GPU "
                "models the float16 forecasts differ, and which unit runs where depends on the scheduling. Name the GPUs "
                "with device='cuda:N' or CUDA_VISIBLE_DEVICES.",
                UserWarning,
                stacklevel=2,
            )
        return tuple(f"cuda:{index}" for index in kept)

    def _runtime_fields(self) -> Dict[str, Any]:
        """The device type and its kernels (`cpu_capability`, `cuda_capability`), the precision the fits compute in
        (`precision`), the package's memory settings that would otherwise follow free memory (`_memory_settings`), the torch
        threads of a fit, and the device's model name with every device the run uses (`device_name`, warn-only, as
        `"NVIDIA H100 80GB HBM3 x 2 (cuda:0, cuda:1)"`), next to the card's package versions: compared on resume, but the
        name. Read from the run's devices (`_run_devices`), all of one GPU model."""
        devices = self._run_devices()
        first = devices[0]
        kind = first.split(":")[0]
        fields: Dict[str, Any] = {
            "device": kind,
            "precision": self._PRECISION[kind],
            **self._memory_settings(kind),
            "threads_per_fit": THREADS_PER_FIT,
        }
        if not has_torch():
            return fields
        import torch

        from ._torch.runtime import cpu_name, cuda_capability, cuda_probe

        if kind == "cpu":
            fields["cpu_capability"] = torch.backends.cpu.get_cpu_capability()
            fields["device_name"] = cpu_name()
        else:
            index = int(first.split(":")[1]) if ":" in first else 0
            fields["cuda_capability"] = cuda_capability(index)
            name = cuda_probe(index)[1] or "cuda"
            fields["device_name"] = name if len(devices) == 1 else f"{name} x {len(devices)} ({', '.join(devices)})"
        return fields

    def _memory_settings(self, kind: str) -> Dict[str, Any]:
        """The family's settings that pin what its package would size by the free memory of the device (`kind`), recorded in
        the runtime fields; none here."""
        return {}

    def _executor_config(self) -> tuple:
        """One fit at a time per process (torch's thread count and precision flags are the process's), in `MAX_PROCESSES`
        processes: by default every usable CPU, or `GPU_PROCESSES` (6, the research's count) per GPU of the run."""
        if "MAX_PROCESSES" in os.environ:
            return 1, int(os.environ["MAX_PROCESSES"])
        devices = self._run_devices()
        cpus = max(1, _usable_cpu_count())
        if devices[0].startswith("cuda"):
            return 1, min(GPU_PROCESSES * len(devices), cpus)
        return 1, cpus

    def _worker_inputs(self, state: Any, threads_per_process: int) -> Tuple[Dict, Dict]:
        """BaseModel's, with the run's devices as this process resolved them (`config["in_context_devices"]`, each worker's
        `_device_text`) and torch pinned to one thread in each worker (`_worker._pool_init`) where torch is installed."""
        expanded, config = super()._worker_inputs(state, threads_per_process)
        config["in_context_devices"] = list(self._run_devices())
        if has_torch():
            from ._torch.runtime import check_torch_threads

            config["torch_threads"] = check_torch_threads(THREADS_PER_FIT, forked=not _worker._POOLS_SPAWN)
        return expanded, config

    # --- the backend ----------------------------------------------------------------------------------------------

    def _backend_settings(self) -> Tuple:
        """The settings the backend is built with (part of its key in this process)."""
        return tuple(sorted(self._family_kwargs().items())) + (("random_state", self.random_state),)

    def _load_backend(
        self, resolved: ResolvedWeights, device: str
    ) -> Any:  # pragma: no cover - every family defines it
        """The family's backend, loaded from `resolved` on `device`: an object with `fit(x, y) -> context`,
        `predict(context, rows) -> medians` and `scope()`, a context manager every call runs in."""
        raise NotImplementedError

    def _device_text(self) -> str:
        """The device this fit runs on. In a run's task (a fit in progress, `_worker.current_fit`), the parent's: worker k's is
        the run's devices' k-th modulo their number (`config["in_context_devices"]`), checked once (`_check_worker_device`);
        a fit called outside a run resolves the model's `device` here."""
        devices = _worker._W_CONFIG.get("in_context_devices") if _worker.current_fit() is not None else None
        if devices:
            device = devices[worker_index() % len(devices)]
            if device not in _CHECKED_DEVICES:
                self._check_worker_device(device)
                _CHECKED_DEVICES.add(device)
            return device
        return self._run_devices()[0]

    def _check_worker_device(self, device: str) -> None:
        """A RuntimeError, which stops the run, where this worker can't use the GPU its parent resolved (its own probe
        disagrees): its forecasts would come from another device than the manifest records."""
        if not device.startswith("cuda"):
            return
        import torch

        index = int(device.split(":")[1]) if ":" in device else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"{type(self).__name__}({self.name!r}): the run resolved {device} in its parent process, and this worker can't "
                f"use it (CUDA available: {torch.cuda.is_available()}, GPUs seen: {torch.cuda.device_count()}): its forecasts "
                "would come from another device than the manifest records. Make the GPU visible to the run's workers, or "
                "use device='cpu'."
            )

    def _backend(self) -> Any:
        """This process's backend for the model's weights, device and settings, loaded on first use under `offline()`."""
        resolved = resolve_model_weights(self)
        if resolved is None:  # pragma: no cover - every in-context card has weights
            raise RuntimeError(f"{type(self).__name__}: its card names no weights")
        device = self._device_text()
        key = (type(self), str(resolved.root), tuple(sorted(resolved.files.items())), device, self._backend_settings())
        with _BACKENDS_LOCK:
            backend = _BACKENDS.get(key)
            if backend is None:
                with offline():
                    backend = self._load_backend(resolved, device)
                _BACKENDS[key] = backend
        return backend

    @contextlib.contextmanager
    def _package_errors(self, what: str):
        """The package's errors as a RuntimeError naming the model, which stops the run (`MissingExtraError` goes through)."""
        try:
            yield
        except MissingExtraError:
            raise
        except Exception as error:
            first = (str(error).strip().splitlines() or [""])[0]
            raise RuntimeError(
                f"{type(self).__name__}({self.name!r}): {self._card.family} failed to {what}: {type(error).__name__}: {first}"
            ) from error

    # --- fits -----------------------------------------------------------------------------------------------------

    def _fit_model(self, train_x: np.ndarray, train_y: np.ndarray) -> Union[Fitted, _Constant]:
        """The context: the unit's rows, shown to the backend once (its `fit`). A context whose targets are all equal is that
        value, with no call."""
        y = np.asarray(train_y, dtype=np.float64).reshape(-1)
        if len(y) and np.all(y == y[0]):
            return _Constant(y[0])
        x = np.ascontiguousarray(np.asarray(train_x, dtype=np.float64).reshape(len(y), -1))
        with self._package_errors("load"):
            backend = self._backend()
        with self._package_errors("fit"), offline(), backend.scope():
            context = backend.fit(x, np.ascontiguousarray(y))
        return Fitted(backend, context)

    def _predict_block(self, fitted: Union[Fitted, _Constant], rows: np.ndarray) -> Tuple[np.ndarray, List[list]]:
        """The median of each row's predictive distribution, in the scaler's target space, from one call; no coefficients."""
        rows = np.ascontiguousarray(np.asarray(rows, dtype=np.float64))
        if isinstance(fitted, _Constant):
            return np.full(len(rows), fitted.value, dtype=np.float64), [[] for _ in range(len(rows))]
        with self._package_errors("predict"), offline(), fitted.backend.scope():
            medians = fitted.backend.predict(fitted.context, rows)
        medians = np.asarray(medians, dtype=np.float64).reshape(-1)
        if medians.shape != (len(rows),):
            raise RuntimeError(
                f"{type(self).__name__}: the backend returned {medians.shape} medians for {len(rows)} rows: one per row"
            )
        return medians, [[] for _ in range(len(rows))]

    def _predict_fitted(self, fitted: Union[Fitted, _Constant], test_x: np.ndarray) -> Tuple[float, list]:
        predictions, _ = self._predict_block(fitted, np.asarray(test_x, dtype=np.float64).reshape(1, -1))
        return float(predictions[0]), []

    def _fit_predict(self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> Tuple[float, list]:
        return self._predict_fitted(self._fit_model(train_x, train_y), test_x)


@contextlib.contextmanager
def torch_fit_scope(device: str):
    """A fit's scope on a real backend: `torch_scope(device, tf32=False, threads=THREADS_PER_FIT)`, put back after."""
    from ._torch.runtime import torch_scope

    with torch_scope(device, tf32=False, threads=THREADS_PER_FIT):
        yield


__all__ = ["InContextModel", "INVARIANT_TARGET", "DEVICE", "GPU_PROCESSES", "THREADS_PER_FIT", "Fitted", "package"]
