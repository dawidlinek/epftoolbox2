"""`TorchModel`: any `torch.nn.Module` class trained by the library's trainer on the library's rows (step 6l, plan section 3.11).

The user writes the module; the library owns everything else: the rows and the training cut (the refit calendar, never warm
started), input standardisation (binary columns kept) and an Invariant (asinh of median and MAD) of each output's target, the
loss, Adam, early stopping on a random share of each fit's own training dates with the best weights restored, the seeds and
their mean, fixed training and inference shapes, the device, TF32 and thread scope, and the manifest.

**The module contract.** With `predictors=[...]`: `__init__(n_inputs, n_outputs, **module_params)` and `forward(x)`, x of shape
(batch, n_inputs). With named blocks `inputs={"price_d-1": [24 names], ...}`: `__init__(input_sizes, n_outputs,
**module_params)`, `input_sizes` a dict of each block's width in the order given, and `forward(blocks)`, a dict of one (batch,
width) tensor per block. Either returns (batch, n_outputs) in the scaled target space. The module sees scaled tensors only,
never dates or targets. A block list is flattened into `predictors` for every check (the information set, the day-level check,
`verify`), and the blocks are views of the flattened columns.

**The two outputs.**
- `output="day"`: one fit per (horizon, refit anchor, window member) with an output for each period of the target day (a day
  unit, `models._units`): rows are local dates, so every predictor must be day-level (`price_h05_d-1`, `is_monday_d+1`; a
  column that changes within a day raises with its profile names as the fix). Needs `refit`.
- `output="period"`: one fit per (period, horizon, anchor) with one output, on that period's rows (today's period path):
  `_fit_model`/`_predict_fitted` with `refit`, `_fit_predict` per forecast with `refit=None`.

**Seeds.** `seeds=k` members per fit, member s seeded by `_calendar.unit_seed(random_state, unit, s)` (the unit's period or
-1, horizon, anchor, list and window member; for `refit=None` the issue day is the anchor), never by the model's name or place.
A forecast is the mean of its members' forecasts in target units (EUR/MWh), as a list of training windows is. A member that
diverged (no finite early-stopping loss, or weights that aren't finite: `TrainResult.failed`) makes every forecast of its fit
a `fit_error`: a mean of the others would change what `seeds=k` means, silently.

**Shapes.** Steps per epoch come from the nominal rows of a fit: its window's days (`max(W, rows)`; the rows themselves for
`"expanding"`), so a fit whose span starts before the data trains as many steps as a full one, wrapping its own rows. A day
unit predicts its whole block in one call of the block's nominal length (the worker pads it); a period unit one row at a time.

**Determinism.** Every fit runs in `torch_scope(device, tf32, threads=1)`: one torch thread per fit (a pool of single-thread
fits is the fast way on a CPU; the loop's numbers change with the thread count), TF32 off unless `tf32=True`, all of it put
back after the fit. The manifest's `runtime` records the torch version, device type, dtype, `tf32`, `threads_per_fit`, `stack`,
`pad_to` and the CPU kernels, compared on resume; `model_params` the module's import path, its source hash and `module_params`.
One fit at a time per process: `THREADS_PER_PROCESS` is not used for a torch model, `MAX_PROCESSES` (default every usable
CPU, one process on a GPU) is.

**Saving.** The module is saved by its import path (`package.module:Class`) and `module_hash`, a hash of the source file that
defines it and of its package's version (a change of a layer it imports from the same file shows too). Loading a file whose
hash differs from the module's warns: the forecasts will differ from the saved run's. `save()` raises for a class defined in
`__main__` or inside a function, which a later load can't import; such a class runs (forked workers inherit it).

Importing this module imports no torch: the extra is needed to fit, not to build, save or load the model.
"""

import contextlib
import functools
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import re
import sys
import threading
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .._extras import has_extra
from ..scalers.registry import ScalerSpec, scaler_spec
from ..scalers.target import MultiTargetAsinh
from ._calendar import EXPANDING, unit_seed
from ._cards import RUN_DATA, ModelCard
from ._units import DAY, PERIOD
from ._worker import _POOLS_SPAWN, current_fit, member_mean
from .base import BaseModel, _block_co_batch_items, _predicts_blocks, _usable_cpu_count

OUTPUTS = ("period", "day")
LOSS_NAMES = ("l1", "mse")
_LOSS_PATH = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*\Z")
# The float type of every torch fit (`runtime_fields`' `dtype`).
DTYPE = "float32"
# Torch's intra-op threads per fit (`runtime_fields`' `threads_per_fit`): one, in a pool of single-thread fits.
THREADS_PER_FIT = 1

TORCH_CARD = ModelCard(
    family="torch",
    extra="torch",
    licence="BSD-3-Clause",
    licence_url="https://github.com/pytorch/pytorch/blob/main/LICENSE",
    commercial_use=True,
    revenue_cap=None,
    attribution=None,
    weights=None,
    runtime=("torch>=2.9",),
    corpus=RUN_DATA,
    corpus_end=None,
    released=None,
)

# Classes with no import path (defined in `__main__` or a function), by the key `module_ref` gives them: a pool's forked
# workers rebuild the model from `_model_kwargs`, which holds the key.
_UNIMPORTABLE: Dict[str, type] = {}
_UNIMPORTABLE_LOCK = threading.Lock()


def module_ref(cls: type) -> str:
    """The text a model's `_model_kwargs` and manifest hold for its module class: its import path (`package.module:Class`), or
    for a class without one (`__main__`, a function's) `module:qualname` registered in this process (`#2`, `#3`, ... after it
    for another class of the same name), which `resolve_module` finds again here and in forked workers."""
    from ..pipelines._yaml import class_path

    try:
        return class_path(cls, "the module")
    except TypeError:
        pass
    base = f"{getattr(cls, '__module__', '?')}:{getattr(cls, '__qualname__', repr(cls))}"
    with _UNIMPORTABLE_LOCK:
        key, n = base, 1
        while key in _UNIMPORTABLE and _UNIMPORTABLE[key] is not cls:
            n += 1
            key = f"{base}#{n}"
        _UNIMPORTABLE[key] = cls
    return key


def resolve_module(module: Union[str, type], where: str) -> type:
    """The class `module` names: a class as it is, a key of `module_ref`, or an import path (importing it, and torch with it)."""
    if inspect.isclass(module):
        return module
    if not isinstance(module, str):
        raise TypeError(f"{where}: module must be a torch.nn.Module class or 'package.module:Class', got {module!r}")
    found = _UNIMPORTABLE.get(module)
    if found is not None:
        return found
    from ..pipelines._yaml import load_class

    return load_class(module, where)


@functools.lru_cache(maxsize=None)
def _distribution_version(top: str) -> Optional[str]:
    """`name version` of the installed distribution that provides the top-level package `top`, None for none (a script's
    folder, a test module)."""
    for distribution in importlib.metadata.packages_distributions().get(top, []):
        try:
            return f"{distribution} {importlib.metadata.version(distribution)}"
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


_FILE_HASHES: Dict[Tuple[str, int, int], str] = {}


def _file_digest(path: str) -> Optional[str]:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (path, stat.st_size, stat.st_mtime_ns)
    found = _FILE_HASHES.get(key)
    if found is None:
        with open(path, "rb") as f:
            found = hashlib.sha256(f.read()).hexdigest()
        _FILE_HASHES[key] = found
    return found


def module_hash(module: Union[str, type]) -> str:
    """A hash (16 hex digits) of the source file that defines the module class and of the version of the package it comes
    from: a change to the class, or to a layer it uses from the same file, or a new release of its package changes it. Found
    without importing the module when it isn't imported yet (`importlib.util.find_spec`). A class with no file (typed into an
    interpreter) is hashed by its source, or its name."""
    if inspect.isclass(module):
        name, qualname = module.__module__, module.__qualname__
    else:
        name, _, qualname = str(module).partition(":")
        qualname = qualname.split("#")[0]
    loaded = sys.modules.get(name)
    path = getattr(loaded, "__file__", None)
    if path is None and loaded is None:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        path = getattr(spec, "origin", None)
    source = _file_digest(path) if path else None
    if source is None and inspect.isclass(module):
        try:
            source = hashlib.sha256(inspect.getsource(module).encode()).hexdigest()
        except (OSError, TypeError):
            source = None
    payload = json.dumps(
        ["epftoolbox2-module-1", source or f"{name}:{qualname}", _distribution_version(name.split(".")[0])]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def check_contract(cls: type, blocks: bool, module_params: Dict[str, Any], where: str) -> None:
    """A TypeError unless `cls(n_inputs, n_outputs, **module_params)` (or with named blocks `cls(input_sizes, n_outputs,
    **module_params)`) fits its signature: those two parameter names first, every key of `module_params` taken, every other
    parameter without a default given. A class whose signature can't be read is taken as it is."""
    try:
        parameters = list(inspect.signature(cls).parameters.values())
    except (TypeError, ValueError):
        return
    need = ("input_sizes", "n_outputs") if blocks else ("n_inputs", "n_outputs")
    positional = [p.name for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if tuple(positional[:2]) != need:
        style = "inputs={block: [names]}" if blocks else "predictors=[names]"
        other = "predictors=[names]" if blocks else "inputs={block: [names]}"
        raise TypeError(
            f"{where}: with {style} the module is built as {cls.__name__}({', '.join(need)}, **module_params), but its "
            f"__init__ takes ({', '.join(positional[:2]) or 'nothing'}, ...): name its first two parameters {need[0]} and "
            f"{need[1]}, or give {other} if it takes {'n_inputs' if blocks else 'input_sizes'}"
        )
    rest = [p for p in parameters if p.name not in need]
    if not any(p.kind == p.VAR_KEYWORD for p in rest):
        taken = {p.name for p in rest if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        unknown = sorted(set(module_params) - taken)
        if unknown:
            raise TypeError(
                f"{where}: {cls.__name__}.__init__ takes no {unknown} (module_params); it takes {sorted(taken)}"
            )
    missing = [
        p.name
        for p in rest
        if p.default is p.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.name not in module_params
    ]
    if missing:
        raise TypeError(f"{where}: {cls.__name__}.__init__ needs {missing}: give them in module_params")


def _plain(value: Any, where: str) -> Any:
    """`value` as the JSON a manifest stores and reads back equal (tuples as lists, numpy scalars as numbers): what the
    manifest compares on resume must survive the round trip. A class or a function raises: a saved file would hold its
    import path, which the module would then get as a string."""
    from ..pipelines._yaml import to_yaml_value

    def check(item: Any, at: str) -> None:
        if inspect.isclass(item) or callable(item):
            raise TypeError(f"{at} must be a plain value (str, int, float, bool, None, list or dict), got {item!r}")
        if isinstance(item, dict):
            for key, sub in item.items():
                check(sub, f"{at}.{key}")
        elif isinstance(item, (list, tuple)):
            for sub in item:
                check(sub, at)

    check(value, where)
    return to_yaml_value(value, where)


class _Fitted:
    """A fit's members (`trainer.TrainResult`), the device they live on, and for a period unit the Invariant of its target."""

    def __init__(self, result: Any, device: Any, target: Optional[MultiTargetAsinh] = None):
        self.result = result
        self.device = device
        self.target = target


class TorchModel(BaseModel):
    """A `torch.nn.Module` class of your own, trained by the library (see the module docstring for the contract).

    Args:
        module: The module class, or its import path "package.module:Class" (saved so).
        predictors: Full predictor names (with `{horizon}` where a column depends on it), a `(horizon)` function or a
            function of `ctx` (`models.context`), for a module that takes one flat input. Give this or `inputs`.
        inputs: Named blocks of full predictor names, `{"block": [names], ...}`, for a module that takes `input_sizes` and a
            dict of tensors (weights shared across hours, a day as a sequence). Flattened into `predictors` for every check.
        module_params: Keyword arguments of the module's `__init__` (its architecture), as plain values.
        output: "day", one fit per (horizon, anchor) with an output per period of the target day (day-level predictors
            only); "period", one fit per (period, horizon, anchor) with one output.
        training_window: Days of rows per fit, "expanding", or a list of them averaged in target units.
        refit: The fixed refit calendar ("7D" on Mondays, "1MS", "1D", "auto"); None fits every forecast ("period" only).
        seeds: Members per fit, each with its own seed; a forecast is their mean in target units.
        loss: "l1", "mse" or an elementwise function `(prediction, target) -> loss` as "package.module:function", in the
            scaled target space; the library masks missing slots and averages.
        lr, max_epochs, patience, batch_size, early_stopping, weight_decay: Adam's learning rate and weight decay, at most
            `max_epochs` epochs of `batch_size` rows, early stopping after `patience` epochs without improvement on a random
            share `early_stopping` of each fit's own training dates (0 trains every epoch and keeps the last weights).
        stack: "auto" (a loop on a CPU, a padded vmap stack on a GPU), "loop" or "vmap" (needs `nets.Dropout`, no batch
            norm); `pad_to` the stack's size (default 16). Each changes the numbers and is compared on resume.
        device: "auto" (CUDA if a matmul runs on it, else the CPU), "cpu", "cuda" or "cuda:N".
        tf32: TF32 matrix products on a GPU for this model's fits (off by default; recorded and compared).
        random_state: The base seed of every member's seed.
        newest_target, binary_mask, scaler: As for every model; the scaler standardises the inputs ("standard"; "period"
            output takes only that, since it fits its target's Invariant itself).
        name: The model's name.
        module_hash: The hash a saved file holds (`module_hash`); a different one warns that the module changed.
    """

    _card = TORCH_CARD
    _engine_ok = False
    # `refit` is always written: its default here is "7D", not BaseModel's None, so a file without it would load another model.
    _WRITTEN_WHEN_SET = {key: value for key, value in BaseModel._WRITTEN_WHEN_SET.items() if key != "refit"}
    # A day model predicts each block in one call (`models._units`); a period model one row at a time, so neither for it.
    _predicts_together = property(_predicts_blocks)
    _co_batch_items = _block_co_batch_items

    def __init__(
        self,
        module: Union[str, type],
        predictors: Union[List[Union[str, Callable]], Callable, None] = None,
        inputs: Optional[Dict[str, List[str]]] = None,
        module_params: Optional[Dict[str, Any]] = None,
        output: str = "day",
        training_window: Union[int, str, List[Union[int, str]]] = 728,
        refit: Optional[str] = "7D",
        seeds: int = 4,
        loss: str = "l1",
        lr: float = 3e-4,
        max_epochs: int = 300,
        patience: int = 20,
        batch_size: int = 64,
        early_stopping: float = 0.2,
        weight_decay: float = 0.0,
        stack: str = "auto",
        pad_to: Optional[int] = None,
        device: str = "auto",
        tf32: bool = False,
        random_state: int = 0,
        newest_target: str = "d-1",
        binary_mask: Optional[str] = None,
        scaler: ScalerSpec = "standard",
        name: str = "torch",
        module_hash: Optional[str] = None,
    ):
        where = f"TorchModel({name!r})"
        flat = self._flatten(predictors, inputs, where)
        self._setup(
            flat,
            output=output,
            training_window=training_window,
            refit=refit,
            seeds=seeds,
            loss=loss,
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            early_stopping=early_stopping,
            weight_decay=weight_decay,
            stack=stack,
            pad_to=pad_to,
            device=device,
            tf32=tf32,
            random_state=random_state,
            newest_target=newest_target,
            binary_mask=binary_mask,
            scaler=scaler,
            name=name,
        )
        self.output = output
        self.inputs = None if inputs is None else {str(block): list(names) for block, names in inputs.items()}
        self.module_params = _plain(dict(module_params or {}), f"{where}.module_params")
        if not isinstance(self.module_params, dict):
            raise TypeError(f"{where}: module_params must be a dict, got {module_params!r}")
        self.module = self._resolved_or_path(module, where)
        if inspect.isclass(self.module):
            check_contract(self.module, self.inputs is not None, self.module_params, where)
        self.module_hash = self._checked_hash(module_hash, where)

    # --- construction ---------------------------------------------------------------------------------------------

    @staticmethod
    def _flatten(predictors: Any, inputs: Optional[Dict[str, Sequence[str]]], where: str) -> Any:
        """The predictors BaseModel checks and the workers read: `predictors`, or the blocks of `inputs` one after another."""
        if inputs is None:
            if predictors is None:
                raise ValueError(
                    f"{where}: give predictors=[...] (a flat module) or inputs={{block: [...]}} (named blocks)"
                )
            return predictors
        if not isinstance(inputs, dict) or not inputs:
            raise TypeError(
                f"{where}: inputs must be a non-empty dict {{block: [full predictor names]}}, got {inputs!r}"
            )
        flat: List[str] = []
        for block, names in inputs.items():
            if isinstance(names, str) or not isinstance(names, (list, tuple)) or not names:
                raise TypeError(
                    f"{where}: inputs[{block!r}] must be a non-empty list of predictor names, got {names!r}"
                )
            if not all(isinstance(name, str) for name in names):
                raise TypeError(
                    f"{where}: inputs[{block!r}] takes full predictor names; a predictor function goes in predictors= "
                    "with a flat module"
                )
            flat.extend(names)
        # A worker rebuilds the model with predictors=[] and the blocks (`task_inputs`); a saved file holds both, equal.
        given = predictors is not None and not (isinstance(predictors, list) and not predictors)
        if given and (callable(predictors) or list(predictors) != flat):
            raise ValueError(
                f"{where}: give predictors or inputs, not both: inputs are flattened into the predictors (a saved file holds "
                "both, the predictors being the blocks' names one after another; leave them out when you edit the blocks)"
            )
        return flat

    def _setup(
        self,
        predictors: Any,
        *,
        output: str,
        seeds: int,
        loss: str,
        lr: float,
        max_epochs: int,
        patience: int,
        batch_size: int,
        early_stopping: float,
        weight_decay: float,
        stack: str,
        pad_to: Optional[int],
        device: str,
        tf32: bool,
        random_state: int,
        scaler: ScalerSpec,
        name: str,
        **base: Any,
    ) -> None:
        """What TorchModel and its presets share: the checked training settings, then BaseModel's."""
        where = f"{type(self).__name__}({name!r})"
        if output not in OUTPUTS:
            raise ValueError(f"{where}: output must be one of {list(OUTPUTS)}, got {output!r}")
        # Before BaseModel's refit check, which reads the unit kind.
        self._unit_kind = DAY if output == "day" else PERIOD
        spec = scaler_spec(scaler)
        if output == "period" and spec != "standard":
            raise ValueError(
                f"{where}: output='period' standardises its inputs and fits its target's Invariant (asinh of median and "
                f"MAD) itself, so it takes scaler='standard' only, got {scaler!r}; output='day' takes any input scaler"
            )
        if isinstance(seeds, bool) or not isinstance(seeds, (int, np.integer)) or seeds < 1:
            raise ValueError(f"{where}: seeds must be a whole number of at least 1, got {seeds!r}")
        if not isinstance(loss, str) or (loss not in LOSS_NAMES and not _LOSS_PATH.match(loss)):
            raise ValueError(
                f"{where}: loss must be one of {list(LOSS_NAMES)} or an elementwise function as 'package.module:function', "
                f"got {loss!r}"
            )
        for label, value, low in (("lr", lr, 0.0), ("weight_decay", weight_decay, None)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{where}: {label} must be a finite number, got {value!r}")
            if (low is not None and value <= low) or value < 0:
                raise ValueError(
                    f"{where}: {label} must be {'positive' if low is not None else 'at least 0'}, got {value!r}"
                )
        for label, value in (("max_epochs", max_epochs), ("patience", patience), ("batch_size", batch_size)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{where}: {label} must be a whole number of at least 1, got {value!r}")
        if (
            isinstance(early_stopping, bool)
            or not isinstance(early_stopping, (int, float))
            or not 0 <= early_stopping < 1
        ):
            raise ValueError(
                f"{where}: early_stopping is a share of each fit's training dates in [0, 1), got {early_stopping!r}"
            )
        if stack not in ("auto", "loop", "vmap"):
            raise ValueError(f"{where}: stack must be 'auto', 'loop' or 'vmap', got {stack!r}")
        if pad_to is not None and (isinstance(pad_to, bool) or not isinstance(pad_to, (int, np.integer)) or pad_to < 1):
            raise ValueError(f"{where}: pad_to must be None or a whole number of at least 1, got {pad_to!r}")
        if not isinstance(device, str) or not re.fullmatch(r"auto|cpu|cuda(:[0-9]+)?", device):
            raise ValueError(f"{where}: device must be 'auto', 'cpu', 'cuda' or 'cuda:N', got {device!r}")
        if not isinstance(tf32, (bool, np.bool_)):
            raise ValueError(f"{where}: tf32 must be True or False, got {tf32!r}")
        if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)) or random_state < 0:
            raise ValueError(f"{where}: random_state must be a whole number of at least 0, got {random_state!r}")
        super().__init__(predictors, name=name, scaler=spec, **base)
        self.seeds = int(seeds)
        self.loss = loss
        self.lr = float(lr)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.early_stopping = float(early_stopping)
        self.weight_decay = float(weight_decay)
        self.stack = stack
        self.pad_to = None if pad_to is None else int(pad_to)
        self.device = device
        self.tf32 = bool(tf32)
        self.random_state = int(random_state)

    def __setattr__(self, key, value):
        if key == "output":  # the unit kind follows it
            if value not in OUTPUTS:
                raise ValueError(f"output must be one of {list(OUTPUTS)}, got {value!r}")
            super().__setattr__("_unit_kind", DAY if value == "day" else PERIOD)
        super().__setattr__(key, value)

    @staticmethod
    def _resolved_or_path(module: Any, where: str) -> Any:
        """The class `module` names, or the path as given while it can't be imported for want of torch: the model builds, saves
        and loads without the extra, and the pre-flight names it before any fit. With torch there, a path that doesn't
        import raises here."""
        if inspect.isclass(module):
            return module
        try:
            return resolve_module(module, where)
        except ImportError:
            if isinstance(module, str) and not has_extra("torch"):
                return module
            raise

    def _checked_hash(self, saved: Optional[str], where: str) -> str:
        current = module_hash(self._module_spec()[0])
        if saved is not None and str(saved) != current:
            warnings.warn(
                f"{where}: the module {self._module_ref()} changed since this configuration was saved (module_hash "
                f"{saved} -> {current}: its source file or its package's version): forecasts will differ from the saved "
                "run's, and a run resumed over its results refuses (the manifest compares the hash)",
                UserWarning,
                stacklevel=3,
            )
        return current

    # --- what a fit builds -----------------------------------------------------------------------------------------

    def _module_spec(self) -> Tuple[Any, Dict[str, Any]]:
        """(the module class or its path, its keyword arguments)."""
        return self.module, self.module_params

    def _module_hash(self) -> str:
        return self.module_hash

    def _module_ref(self) -> str:
        module = self._module_spec()[0]
        return module_ref(module) if inspect.isclass(module) else str(module)

    def _module_class(self) -> type:
        return resolve_module(self._module_spec()[0], f"{type(self).__name__}({self.name!r})")

    def _block_sizes(self) -> Optional[Dict[str, int]]:
        inputs = getattr(self, "inputs", None)
        return None if inputs is None else {block: len(names) for block, names in inputs.items()}

    def _block_slices(self) -> Optional[Dict[str, slice]]:
        sizes = self._block_sizes()
        if sizes is None:
            return None
        out, start = {}, 0
        for block, size in sizes.items():
            out[block] = slice(start, start + size)
            start += size
        return out

    def _factory(self, n_inputs: int, n_outputs: int) -> Callable[[], Any]:
        cls = self._module_class()
        params = dict(self._module_spec()[1])
        sizes = self._block_sizes()
        if sizes is not None:
            if sum(sizes.values()) != n_inputs:
                raise RuntimeError(f"{self.name}: blocks of {sum(sizes.values())} columns for a fit of {n_inputs}")
            return lambda: cls(dict(sizes), n_outputs, **params)
        return lambda: cls(n_inputs, n_outputs, **params)

    # --- the run --------------------------------------------------------------------------------------------------

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        return {
            "module": self._module_ref(),
            "module_hash": self._module_hash(),
            "module_params": self.module_params,
            "inputs": self.inputs,
            "output": self.output,
            **self._training_kwargs(),
        }

    def _training_kwargs(self) -> Dict[str, Any]:
        return {
            "seeds": self.seeds,
            "loss": self.loss,
            "lr": self.lr,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "early_stopping": self.early_stopping,
            "weight_decay": self.weight_decay,
            "stack": self.stack,
            "pad_to": self.pad_to,
            "device": self.device,
            "tf32": self.tf32,
            "random_state": self.random_state,
        }

    def _runtime_fields(self) -> Dict[str, Any]:
        from ._torch.runtime import runtime_fields

        return runtime_fields(
            device=self.device,
            dtype=DTYPE,
            tf32=self.tf32,
            threads_per_fit=THREADS_PER_FIT,
            stack=self.stack,
            pad_to=self.pad_to,
        )

    def _executor_config(self) -> tuple:
        """One fit at a time per process, at one torch thread (`torch_scope`): torch's thread count and precision flags are
        the process's, so task threads (`THREADS_PER_PROCESS`) would share them. `MAX_PROCESSES` processes, by default every
        usable CPU, or one on a GPU (each process would hold a CUDA context of its own)."""
        default = max(1, _usable_cpu_count())
        if "MAX_PROCESSES" not in os.environ:
            from ._torch.runtime import resolve_device

            if resolve_device(self.device).type == "cuda":
                default = 1
        return 1, int(os.environ.get("MAX_PROCESSES", default))

    def _worker_inputs(self, state: Any, threads_per_process: int) -> Tuple[Dict, Dict]:
        """BaseModel's, with torch pinned to one thread in each worker (`_worker._pool_init`), after a check in this process,
        before the pool starts, that the module builds and fits the stack and the loss imports: a mistake there would
        otherwise be every fit's `fit_error`."""
        from ._torch.runtime import check_torch_threads

        expanded, config = super()._worker_inputs(state, threads_per_process)
        config["torch_threads"] = check_torch_threads(THREADS_PER_FIT, forked=not _POOLS_SPAWN)
        lists = list(expanded.values())
        self._check_buildable(len(lists[0]) if lists else 0)
        return expanded, config

    def _check_buildable(self, n_inputs: int) -> None:
        """Build the module once on the CPU, from a generator of its own and on one thread (no parallel torch work in a process
        that forks), and check it: a `torch.nn.Module`, and fit for a vmap stack if the run stacks; the loss imports."""
        from ._torch.runtime import resolve_device, resolve_stack, torch_scope
        from ._torch.trainer import check_vmappable, resolve_loss

        import torch

        resolve_loss(self.loss)
        n_outputs = self._periods_per_day if self._unit_kind == DAY else 1
        sizes = self._block_sizes()
        width = sum(sizes.values()) if sizes is not None else n_inputs
        with torch_scope("cpu", tf32=False, threads=1), torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            module = self._factory(width, n_outputs)()
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"{self.name}: the module class built a {type(module).__name__}, not a torch.nn.Module")
        if resolve_stack(self.stack, resolve_device(self.device)) == "vmap":
            check_vmappable(module)

    # --- fits -----------------------------------------------------------------------------------------------------

    def _fit_rows(self, x: np.ndarray, y: np.ndarray, y_mask: Optional[np.ndarray]) -> Any:
        """Train `seeds` members on every row of (x, y): the fit in progress (`_worker.current_fit`) gives their seeds and the
        nominal rows. A member that diverged is a ValueError, every forecast of the fit's `fit_error`."""
        context = current_fit()
        if context is None:
            raise RuntimeError(
                f"{self.name}: a TorchModel fits on the library's fit units (its seeds are the unit's), and this fit has none: "
                "it runs through ModelPipeline.run or the model's run, not PooledModel or a direct call"
            )
        from ._torch.runtime import torch_scope
        from ._torch.trainer import MemberSpec, train_members

        n = len(x)
        window = context.window
        nominal = n if (isinstance(window, str) and window == EXPANDING) else max(int(window), n)
        rows = np.arange(n)
        members = [
            MemberSpec(seed_key=unit_seed(self.random_state, context.unit, index), train_rows=rows)
            for index in range(self.seeds)
        ]
        with torch_scope(self.device, tf32=self.tf32, threads=THREADS_PER_FIT) as device:
            result = train_members(
                self._factory(x.shape[1], y.shape[1]),
                x,
                y,
                y_mask,
                members,
                loss=self.loss,
                lr=self.lr,
                max_epochs=self.max_epochs,
                patience=self.patience,
                batch_size=self.batch_size,
                early_stopping=self.early_stopping,
                weight_decay=self.weight_decay,
                stack=self.stack,
                pad_to=self.pad_to,
                device=device,
                dtype=DTYPE,
                nominal_rows=nominal,
                blocks=self._block_slices(),
            )
        failed = np.flatnonzero(result.failed)
        if failed.size:
            raise ValueError(
                f"seed{'s' if failed.size > 1 else ''} {failed.tolist()} of {self.seeds} diverged (no finite early-stopping "
                "loss, or weights that aren't finite): the fit's forecasts are skipped, never a mean of the other seeds"
            )
        return _Fitted(result, device)

    def _members_out(self, fitted: _Fitted, rows: np.ndarray) -> np.ndarray:
        """(members, rows, outputs): each member's forecasts of `rows`, in the scope of the fit (a loop's numbers move with
        the thread count)."""
        with torch_scope_of(self, fitted.device):
            return fitted.result.predict(np.asarray(rows, dtype=np.float64))

    def _fit_days(self, x: np.ndarray, y: np.ndarray, y_mask: np.ndarray) -> _Fitted:
        return self._fit_rows(x, y, y_mask)

    def _predict_days(self, fitted: _Fitted, rows: np.ndarray) -> np.ndarray:
        """Each member's forecasts, (members, rows, P): the worker maps each back with its period's Invariant and averages
        them in target units."""
        return self._members_out(fitted, rows)

    def _fit_model(self, train_x: np.ndarray, train_y: np.ndarray) -> _Fitted:
        """A period unit: the targets as the scaler gave them (standardised), then their Invariant, which is the raw
        target's (median and MAD move with an affine map), so the module trains in asinh space as a day unit's does."""
        y = np.asarray(train_y, dtype=np.float64)[:, None]
        target = MultiTargetAsinh().fit(y)
        fitted = self._fit_rows(np.asarray(train_x, dtype=np.float64), target.transform(y), None)
        fitted.target = target
        return fitted

    def _predict_fitted(self, fitted: _Fitted, test_x: np.ndarray) -> Tuple[float, list]:
        """One row's forecast in the scaler's units: each member's mapped back by the Invariant, then their mean (the scaler's
        map back to target units is affine, so it is the mean in target units)."""
        out = self._members_out(fitted, np.asarray(test_x, dtype=np.float64)[:1])
        return member_mean(fitted.target.inverse(out[:, 0, 0], 0)), []

    def _fit_predict(self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> Tuple[float, list]:
        return self._predict_fitted(self._fit_model(train_x, train_y), test_x)


@contextlib.contextmanager
def torch_scope_of(model: TorchModel, device: Any):
    """The scope of `model`'s fits (`runtime.torch_scope`) on the device they ran on."""
    from ._torch.runtime import torch_scope

    with torch_scope(device, tf32=model.tf32, threads=THREADS_PER_FIT) as resolved:
        yield resolved
