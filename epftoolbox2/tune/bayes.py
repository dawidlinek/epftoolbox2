"""`BayesSearch`: Optuna's proposals and successive halving over the validation period, returning ordinary models.

How a search runs (`BayesSearch.run`):

1. The frame is cut after the validation end (local days), before the first trial, and `test=` is refused: no trial can read
   the test period, whatever the model does. The last validation days' forecasts for days after the end are made (every trial
   makes them) and never scored: their actuals are the test period's.
2. **Rung 0.** `n_trials` trials, each proposed by the sampler (TPE, or random) from the trials finished so far and scored on
   the first rung: `ModelPipeline.run(validation=..., validation_every=every_nth_day)` of the model `make(params)` builds, with
   the rung's `seeds` in place of the model's. The sampler of trial i is seeded from (`seed`, i), so with one worker a search
   is the same twice; workers running at once see each other's trials in a timing-dependent order.
3. **Rungs 1, 2, ….** Once every trial of a rung has finished, the best `keep` share of those it scored (by validation MAE,
   ties by trial number; at least one) goes on to the next rung, as Optuna trials enqueued with fixed settings in a study of
   that rung. 100 trials and `keep=1/3` promote 33, then 11.
4. The result (`SearchResult`) holds every trial with its settings, the rungs it reached and their scores, the ranking and
   what `best()` chose; `save()` writes it, and it is the record: parallel trials finish in a timing-dependent order, so a
   rerun of a search is not.

A trial is scored on identical keys with every other trial of its rung: the rung's planned forecasts (its issue days, horizons
and periods) less those the frame has no row to issue from (`missing_test_row`, the same for every model). A trial that skips
any other key (missing predictors, a failed fit, ...) or forecasts a non-finite value gets no score: it fails at that rung,
with its skip counts as the reason, so no model wins by leaving out its hard days. The MAE is over the keys with an actual.

Parallel workers (`epf tune search.py --workers N`, or a SLURM array of the same script) share the trials through `storage`,
Optuna's journal file. Studies: `epftoolbox2_search` (rung 0) and `epftoolbox2_search:rung<r>`. The process that creates a
study sets it up (settings, warm-start trials, the promoted trials) and the others wait for it; a process that joins a study
made with other settings or data raises. A worker killed during a trial leaves it running in the journal, and the others wait
for it at the end of the rung: `epf tune search.py --requeue` (with no worker running) marks such trials interrupted, their
slots are proposed again (rung 0) or their settings enqueued again (later rungs).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import inspect
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from .._extras import MissingExtraError
from ..logging import get_logger

logger = get_logger(__name__)

SAMPLERS = ("tpe", "random")
STUDY = "epftoolbox2_search"
RESULT_KEY = "epftoolbox2_search"
RESULT_VERSION = 1

# Set by `epf tune` for the processes it starts: how many share the storage, and which one this is.
ENV_WORKERS = "EPF_TUNE_WORKERS"
ENV_WORKER = "EPF_TUNE_WORKER"
# Seconds between two looks at the storage while waiting for other workers (a rung to finish, a study to be set up).
ENV_POLL = "EPF_TUNE_POLL"
# Seconds a process waits for the one that created a study to set it up (which takes well under a second) before it says that
# process must have stopped; and seconds between two notes naming the trials a rung still waits for.
ENV_SETUP_TIMEOUT = "EPF_TUNE_SETUP_TIMEOUT"
ENV_WAIT_NOTE = "EPF_TUNE_WAIT_NOTE"

# Trial user attributes.
_WARM = "warm_start"
_SURPLUS = "surplus"
_REJECTED = "rejected"  # a proposal of a Subset under its min_size: proposed again, never scored
_INTERRUPTED = "interrupted"
_PARENT = "trial"
_ERROR = "error"
_FORECASTS = "forecasts"
_PLANNED = "planned"
_MODEL = "model"  # the fingerprint of the model a trial scored (`model_fingerprint`)
# Study user attributes.
_SETTINGS = "search"
_READY = "ready"
_PROMOTED = "promoted"

_MISSING_TEST_ROW = (
    "missing_test_row"  # `models._worker.SKIP_MISSING_TEST_ROW`: no row to issue from, for every model alike
)


def _optuna():
    try:
        import optuna
    except ImportError as exc:
        raise MissingExtraError("BayesSearch", "tune") from exc
    return optuna


# ---------------------------------------------------------------------------------------------------------------------------
# The space
# ---------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Subset:
    """A subset of `items` as one setting of a search space: each item in or out (a yes/no choice per item for the
    sampler), given to `make` as the list of the items in, in their order here. A proposal with fewer than `min_size` items
    gets no score (the trial fails at rung 0 without a run), so `make` never sees one.

    ```python
    space = {"groups": Subset(["load", "wind", "solar"], min_size=1)}
    ```
    """

    items: Tuple[Any, ...]
    min_size: int = 0

    def __init__(self, items: Sequence[Any], min_size: int = 0):
        items = tuple(_native(list(items)))
        min_size = _native(min_size)
        if not items:
            raise ValueError("Subset needs at least one item")
        if len({_canonical(item) for item in items}) != len(items):
            raise ValueError(f"Subset items must be distinct, got {list(items)}")
        if isinstance(min_size, bool) or not isinstance(min_size, int) or not 0 <= min_size <= len(items):
            raise ValueError(f"Subset min_size must be a whole number from 0 to {len(items)}, got {min_size!r}")
        for item in items:
            _check_plain(item, "a Subset item")
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "min_size", min_size)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=repr)


def _check_plain(value: Any, where: str) -> None:
    """A value the saved result can hold as YAML and read back equal: None, bool, int, float, str, or lists and dicts of
    them."""
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{where} must be finite, got {value!r}")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _check_plain(item, where)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{where}: dict keys must be strings, got {key!r}")
            _check_plain(item, where)
        return
    raise TypeError(
        f"{where} must be None, a bool, number or string, or a list or dict of them (the saved result holds it), got "
        f"{type(value).__name__}: {value!r}"
    )


def _plain(value: Any) -> Any:
    """Tuples as lists, as YAML reads them back."""
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class _Dimension:
    key: str
    kind: str  # "choice", "float", "int", "subset"
    values: Tuple[Any, ...] = ()  # choices or subset items
    low: Any = None
    high: Any = None
    log: bool = False
    min_size: int = 0

    def record(self) -> Dict[str, Any]:
        if self.kind == "choice":
            return {"choice": _plain(list(self.values))}
        if self.kind == "subset":
            return {"subset": _plain(list(self.values)), "min_size": self.min_size}
        return {self.kind: [self.low, self.high], "log": self.log}

    def names(self) -> List[str]:
        """The sampler's parameter names of this dimension."""
        if self.kind == "subset":
            return [f"{self.key}[{i}]" for i in range(len(self.values))]
        return [self.key]

    def suggest(self, trial: Any) -> Any:
        if self.kind == "choice":
            return self.values[trial.suggest_categorical(self.key, list(range(len(self.values))))]
        if self.kind == "float":
            return float(trial.suggest_float(self.key, self.low, self.high, log=self.log))
        if self.kind == "int":
            return int(trial.suggest_int(self.key, self.low, self.high, log=self.log))
        flags = [trial.suggest_categorical(name, [False, True]) for name in self.names()]
        return [item for item, used in zip(self.values, flags) if used]

    def decode(self, params: Mapping[str, Any]) -> Any:
        """The value of this dimension from the sampler's parameters of a trial."""
        if self.kind == "choice":
            return self.values[int(params[self.key])]
        if self.kind == "float":
            return float(params[self.key])
        if self.kind == "int":
            return int(params[self.key])
        return [item for item, name in zip(self.values, self.names()) if params[name]]

    def encode(self, value: Any) -> Optional[Dict[str, Any]]:
        """The sampler's parameters of a value, or None if this dimension can't hold it."""
        if self.kind == "choice":
            wanted = _canonical(_plain(value))
            for index, choice in enumerate(self.values):
                if _canonical(_plain(choice)) == wanted:
                    return {self.key: index}
            return None
        if self.kind in ("float", "int"):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not self.low <= value <= self.high:
                return None
            if self.kind == "int" and int(value) != value:
                return None
            return {self.key: int(value) if self.kind == "int" else float(value)}
        if not isinstance(value, (list, tuple)):
            return None
        wanted = {_canonical(_plain(item)) for item in value}
        known = [_canonical(_plain(item)) for item in self.values]
        if not wanted <= set(known):
            return None
        return {name: key in wanted for name, key in zip(self.names(), known)}


def _native(value: Any) -> Any:
    """`value` with numpy scalars and arrays as Python ones (`np.linspace(...)` choices, `np.int64` bounds), which the saved
    result holds as YAML and reads back equal."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [_native(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_native(item) for item in value)
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    return value


def _dimension(key: Any, spec: Any) -> _Dimension:
    if not isinstance(key, str) or not key or "[" in key or "]" in key:
        raise ValueError(f"A space key must be a non-empty string without '[' or ']', got {key!r}")
    where = f"space[{key!r}]"
    if isinstance(spec, Subset):
        return _Dimension(key, "subset", spec.items, min_size=spec.min_size)
    if isinstance(spec, np.ndarray):
        spec = spec.tolist()
    spec = _native(spec)
    if isinstance(spec, list):
        if not spec:
            raise ValueError(f"{where}: a choice needs at least one value")
        for value in spec:
            _check_plain(value, f"{where} choice")
        return _Dimension(key, "choice", tuple(spec))
    if isinstance(spec, tuple):
        if len(spec) not in (2, 3) or (len(spec) == 3 and spec[2] != "log"):
            raise ValueError(f"{where}: a range is (low, high) or (low, high, 'log'), got {spec!r}")
        low, high = spec[0], spec[1]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (low, high)):
            raise ValueError(f"{where}: a range's bounds must be finite numbers, got {spec!r}")
        if not low < high:
            raise ValueError(f"{where}: a range needs low < high, got {spec!r}")
        log = len(spec) == 3
        if log and low <= 0:
            raise ValueError(f"{where}: a log range needs low > 0, got {spec!r}")
        kind = "int" if isinstance(low, int) and isinstance(high, int) else "float"
        cast = int if kind == "int" else float
        return _Dimension(key, kind, low=cast(low), high=cast(high), log=log)
    raise TypeError(
        f"{where} must be a list of choices, a range (low, high) or (low, high, 'log'), or a Subset, got "
        f"{type(spec).__name__}: {spec!r}"
    )


def _dimension_from_record(key: str, record: Mapping[str, Any]) -> _Dimension:
    if "choice" in record:
        return _Dimension(key, "choice", tuple(record["choice"]))
    if "subset" in record:
        return _Dimension(key, "subset", tuple(record["subset"]), min_size=int(record.get("min_size", 0)))
    kind = "int" if "int" in record else "float"
    low, high = record[kind]
    return _Dimension(key, kind, low=low, high=high, log=bool(record.get("log", False)))


class _Space:
    def __init__(self, dimensions: Sequence[_Dimension]):
        self.dimensions = list(dimensions)

    @classmethod
    def of(cls, space: Mapping[str, Any]) -> "_Space":
        if not isinstance(space, Mapping) or not space:
            raise ValueError("space must be a non-empty dict of setting -> choices, range or Subset")
        return cls([_dimension(key, spec) for key, spec in space.items()])

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "_Space":
        return cls([_dimension_from_record(key, spec) for key, spec in record.items()])

    def record(self) -> Dict[str, Any]:
        return {d.key: d.record() for d in self.dimensions}

    def suggest(self, trial: Any) -> Dict[str, Any]:
        return {d.key: d.suggest(trial) for d in self.dimensions}

    def decode(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        return {d.key: d.decode(params) for d in self.dimensions}

    def encode(self, values: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        out: Dict[str, Any] = {}
        for d in self.dimensions:
            if d.key not in values:
                return None
            encoded = d.encode(values[d.key])
            if encoded is None:
                return None
            out.update(encoded)
        return out

    def too_small(self, values: Mapping[str, Any]) -> Optional[str]:
        """Why a proposal can't be scored (a subset under its `min_size`), or None."""
        for d in self.dimensions:
            if d.kind == "subset" and len(values[d.key]) < d.min_size:
                return f"{d.key}: {len(values[d.key])} item(s), fewer than min_size={d.min_size}"
        return None


# ---------------------------------------------------------------------------------------------------------------------------
# Rungs and scores
# ---------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Rung:
    every_nth_day: int
    seeds: Optional[int] = None

    def record(self) -> Dict[str, Any]:
        return {"every_nth_day": self.every_nth_day, "seeds": self.seeds}


def _rungs(rungs: Optional[Sequence[Mapping[str, Any]]]) -> List[_Rung]:
    if rungs is None:
        return [_Rung(1)]
    if isinstance(rungs, (str, bytes, Mapping)) or not isinstance(rungs, Sequence) or not rungs:
        raise ValueError("rungs must be a non-empty list of {'every_nth_day': k, 'seeds': s} (seeds optional)")
    out = []
    for i, rung in enumerate(rungs):
        if not isinstance(rung, Mapping) or not set(rung) <= {"every_nth_day", "seeds"} or "every_nth_day" not in rung:
            raise ValueError(
                f"rungs[{i}] must be {{'every_nth_day': k}} or {{'every_nth_day': k, 'seeds': s}}, got {rung!r}"
            )
        every, seeds = rung["every_nth_day"], rung.get("seeds")
        if isinstance(every, bool) or not isinstance(every, int) or every < 1:
            raise ValueError(f"rungs[{i}]['every_nth_day'] must be a whole number of at least 1, got {every!r}")
        if seeds is not None and (isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 1):
            raise ValueError(f"rungs[{i}]['seeds'] must be a whole number of at least 1, got {seeds!r}")
        out.append(_Rung(every, seeds))
    return out


@dataclass(frozen=True)
class _Score:
    """A trial's outcome at a rung: its validation MAE, or None with the reason."""

    mae: Optional[float]
    forecasts: int = 0
    planned: int = 0
    error: Optional[str] = None
    model: Optional[str] = None  # `model_fingerprint` of the model `make` built


def _described(value: Any) -> Any:
    """A value of a model's settings as JSON can hold it, a function by its import path and the hash of its code."""
    if callable(value) and not isinstance(value, type):
        code = getattr(value, "__code__", None)
        digest = hashlib.sha256(code.co_code).hexdigest()[:12] if code is not None else ""
        return f"{getattr(value, '__module__', '')}:{getattr(value, '__qualname__', repr(value))}#{digest}"
    if isinstance(value, type):
        return f"{value.__module__}:{value.__qualname__}"
    return repr(value)


def model_fingerprint(model: Any) -> str:
    """What a search compares to know that `make` builds the model it scored: a hash of the model's class and its settings as
    a pipeline saves them (`ModelPipeline.to_dict`), or its public attributes where a setting can't be saved (a predictor
    function defined in the script)."""
    from ..pipelines.model_pipeline import ModelPipeline

    try:
        params = ModelPipeline()._serialize_component(model)["params"]
    except (TypeError, ValueError):
        params = {key: value for key, value in vars(model).items() if not key.startswith("_")}
        if getattr(model, "_ctx_fn", None) is not None:
            params["predictors"] = model._ctx_fn
    text = json.dumps(
        {"class": f"{type(model).__module__}:{type(model).__qualname__}", "params": params},
        sort_keys=True,
        default=_described,
    )
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _set_seeds(model: Any, seeds: int, where: str) -> None:
    """A rung's (or `best()`'s) `seeds` in place of the model's: only for a class that takes `seeds`."""
    if "seeds" not in inspect.signature(type(model)).parameters or not hasattr(model, "seeds"):
        raise ValueError(
            f"{where} sets seeds={seeds}, and {type(model).__name__} has no seeds parameter: leave seeds out for this model"
        )
    model.seeds = int(seeds)


def _seed_of(seed: int, rung: int, number: int) -> int:
    """The sampler seed of trial `number` of a rung: a pure function of the three, independent of the worker."""
    return int(np.random.SeedSequence([seed, rung, number]).generate_state(1)[0])


def _poll() -> float:
    try:
        return max(0.01, float(os.environ.get(ENV_POLL, "2")))
    except ValueError:
        return 2.0


def _seconds(name: str, default: float) -> float:
    try:
        return max(0.01, float(os.environ.get(name, default)))
    except ValueError:
        return default


def _workers() -> int:
    try:
        return max(1, int(os.environ.get(ENV_WORKERS, "1")))
    except ValueError:
        return 1


@contextlib.contextmanager
def _quiet_optuna() -> Iterator[None]:
    """Optuna's per-trial log lines and its experimental-feature warnings off, for this search alone."""
    optuna = _optuna()
    level = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
            yield
    finally:
        optuna.logging.set_verbosity(level)


@dataclass
class _Run:
    """What every trial of a search runs on."""

    frame: pd.DataFrame
    target: str
    horizons: Tuple[int, ...]
    freq: str
    validation: Tuple[str, str]
    pipeline: Dict[str, Any]
    storage: Any = None  # the search's Optuna storage, once open


# ---------------------------------------------------------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------------------------------------------------------


class BayesSearch:
    """A search over a model's settings on the validation period, with Optuna proposing them and successive halving
    dropping weak ones early (extra `[tune]`). Its result gives ordinary models (`SearchResult.best`).

    Args:
        make: A function of a dict of settings (one value per key of `space`) that returns one model, the same class every
            time. Its `name` names the search and the models `best()` returns (`{name}_bayes_00`, ...).
        space: Setting -> what it can be: a list of choices (any values: numbers, strings, lists such as `[256, 128]`), a
            range `(low, high)` (whole numbers when both are ints, else floats), `(low, high, "log")` on a log scale, or a
            `Subset(items, min_size=0)` of a list (each item in or out).
        sampler: "tpe" (Optuna's `TPESampler`: random for the first 10 trials, then where low validation MAE is likeliest)
            or "random".
        n_trials: How many settings are tried at the first rung.
        seed: Seeds the sampler, trial by trial (see the module docstring), so one worker gives the same table twice.
        rungs: The rungs of successive halving, cheapest first: `{"every_nth_day": k}` scores on every k-th validation issue
            day (`ModelPipeline.run(validation_every=k)`), with `"seeds": s` in place of the model's `seeds` (a model
            without that parameter raises). None is one rung on every day with the model as built.
        keep: The share of a rung's scored trials that goes on to the next rung (the best, at least one).
        warm_start: A saved result (`SearchResult.save`) or a `SearchResult`: its trials that reached its last rung, best
            first, are the first trials of this search (those this space can hold), as last year's best starting this year's.
        storage: A journal file (Optuna's `JournalFileBackend`) that workers in other processes share and a rerun resumes;
            None keeps the trials in memory. Needed for `epf tune --workers N`.
    """

    def __init__(
        self,
        make: Callable[[Dict[str, Any]], Any],
        space: Mapping[str, Any],
        sampler: str = "tpe",
        n_trials: int = 100,
        seed: int = 0,
        rungs: Optional[Sequence[Mapping[str, Any]]] = None,
        keep: float = 1 / 3,
        warm_start: Union[str, Path, "SearchResult", None] = None,
        storage: Union[str, Path, None] = None,
    ):
        if not callable(make):
            raise TypeError(f"make must be a function of the settings that returns a model, got {make!r}")
        if sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {SAMPLERS}, got {sampler!r}")
        if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
            raise ValueError(f"n_trials must be a whole number of at least 1, got {n_trials!r}")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"seed must be a whole number of at least 0, got {seed!r}")
        if isinstance(keep, bool) or not isinstance(keep, (int, float)) or not 0 < keep <= 1:
            raise ValueError(f"keep must be a share in (0, 1], got {keep!r}")
        if warm_start is not None and not isinstance(warm_start, (str, Path, SearchResult)):
            raise TypeError(f"warm_start must be a saved result's path or a SearchResult, got {warm_start!r}")
        if storage is not None and not isinstance(storage, (str, Path)):
            raise TypeError(f"storage must be the path of a journal file, or None, got {storage!r}")
        self.make = make
        self.space = dict(space)
        self.sampler = sampler
        self.n_trials = n_trials
        self.seed = seed
        self.rungs = None if rungs is None else [dict(rung) for rung in rungs]
        self.keep = float(keep)
        self.warm_start = warm_start
        self.storage = storage
        self._space = _Space.of(space)
        self._rungs = _rungs(rungs)

    # -- running -----------------------------------------------------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame,
        target: str = "price",
        horizon: Optional[int] = None,
        validation: Optional[Tuple[Any, Any]] = None,
        *,
        horizons: Optional[Sequence[int]] = None,
        freq: str = "1h",
        information_set: Any = None,
        issue_time: Optional[str] = None,
        issue_timezone: str = "Europe/Brussels",
        availability: Optional[str] = None,
        engine: str = "off",
        test: Any = None,
    ) -> "SearchResult":
        """Run the search on the validation period and return its result.

        `target`, `horizon`/`horizons`, `freq` and `validation=(start, end)` are as in `ModelPipeline.run`; `information_set`
        (or `issue_time` and `issue_timezone`), `availability` and `engine` as in `ModelPipeline(...)`. Each trial runs one
        model through a `ModelPipeline` of these on the frame cut after the validation end, and is scored on the forecasts for
        days up to it.
        `test=` raises: a search chooses on validation, and the test period is never read (`validation="previous_year"` too:
        run one search per year, each warm-started from the year before).
        """
        if test is not None:
            raise ValueError(
                "BayesSearch runs on the validation period alone, and never reads the test period: drop test=, add the "
                "models of result.best() to a pipeline and run that with test="
            )
        if isinstance(data, Mapping) or not isinstance(data, pd.DataFrame):
            raise TypeError("BayesSearch.run takes one frame (a pandas DataFrame); search each zone on its own frame")
        if validation is None or isinstance(validation, str):
            raise ValueError(
                f"BayesSearch needs validation=(start, end), got {validation!r}: with 'previous_year', run one search per "
                "year, warm-started from the year before"
            )
        from ..models._calendar import resolve_horizons
        from ..pipelines.model_pipeline import ModelPipeline
        from ..results.windows import to_period

        start, end = to_period(validation, "validation")
        _, planned_horizons = resolve_horizons(horizon, horizons, 7)
        pipeline = dict(
            information_set=information_set,
            issue_time=issue_time,
            issue_timezone=issue_timezone,
            availability=availability,
            engine=engine,
        )
        ModelPipeline(**pipeline)  # its arguments' errors before anything runs
        if _workers() > 1 and self.storage is None:
            raise ValueError(
                f"{_workers()} workers share a search's trials through storage=: pass a journal file, such as "
                "storage='runs/search.journal'"
            )
        frame = cut_frame(data, end)
        run = _Run(frame, target, tuple(planned_horizons), freq, (start, end), pipeline)
        optuna = _optuna()
        with _quiet_optuna():
            storage = run.storage = self._storage()
            settings = self._settings(run)
            study = self._open(storage, STUDY, optuna.samplers.RandomSampler(seed=self.seed), settings, warm=True)
            self._check_models(storage)
            self._work_rung0(study, run)
            for r in range(1, len(self._rungs)):
                self._wait_for_rung(storage, r - 1)
                self._work_rung(storage, r, run)
            self._wait_for_rung(storage, len(self._rungs) - 1)
            return SearchResult(self._record(storage, settings), make=self.make)

    def _storage(self) -> Any:
        optuna = _optuna()
        if self.storage is None:
            return optuna.storages.InMemoryStorage()
        from optuna.storages.journal import JournalFileBackend

        path = Path(self.storage)
        path.parent.mkdir(parents=True, exist_ok=True)
        return optuna.storages.JournalStorage(JournalFileBackend(str(path)))

    def _settings(self, run: _Run) -> Dict[str, Any]:
        """What workers of one search must share, and a resume must find again: compared as JSON."""
        from ..results.manifest import hash_frame

        info = run.pipeline["information_set"]
        return {
            "space": self._space.record(),
            "sampler": self.sampler,
            "n_trials": self.n_trials,
            "seed": self.seed,
            "rungs": [rung.record() for rung in self._rungs],
            "keep": self.keep,
            "target": run.target,
            "horizons": list(run.horizons),
            "freq": run.freq,
            "validation": list(run.validation),
            "data_through": run.validation[1],
            "data_hash": hash_frame(run.frame),
            "information_set": None if info is None else info.to_dict(),
            "issue_time": run.pipeline["issue_time"],
            "issue_timezone": run.pipeline["issue_timezone"] if run.pipeline["issue_time"] is not None else None,
            "availability": run.pipeline["availability"],
            "engine": run.pipeline["engine"],
        }

    def _open(self, storage: Any, name: str, sampler: Any, settings: Dict[str, Any], warm: bool = False) -> Any:
        """The study `name`, set up by the process that creates it (settings, and the warm-start trials of rung 0) and
        waited for by the others, which check that it was made with the same settings."""
        optuna = _optuna()
        try:
            study = optuna.create_study(
                storage=storage, study_name=name, direction="minimize", sampler=sampler, load_if_exists=False
            )
        except optuna.exceptions.DuplicatedStudyError:
            study = optuna.load_study(study_name=name, storage=storage, sampler=sampler)
            self._wait_for_setup(study, name)
            stored = study.user_attrs.get(_SETTINGS)
            if _canonical(stored) != _canonical(settings):
                differ = sorted(
                    key
                    for key in {*(stored or {}), *settings}
                    if _canonical((stored or {}).get(key)) != _canonical(settings.get(key))
                )
                raise ValueError(
                    f"The search in {self.storage} was made with other settings or data ({', '.join(differ)} differ): use "
                    "another storage file for this search"
                )
            return study
        study.set_user_attr(_SETTINGS, settings)
        if warm:
            for params in self._warm_params():
                study.enqueue_trial(params, user_attrs={_WARM: True})
        study.set_user_attr(_READY, True)
        return study

    def _warm_params(self) -> List[Dict[str, Any]]:
        """The sampler's parameters of the warm-start trials: the best of the earlier search that this space can hold."""
        if self.warm_start is None:
            return []
        earlier = self.warm_start if isinstance(self.warm_start, SearchResult) else SearchResult.load(self.warm_start)
        out, left_out = [], []
        for number in earlier.finalists():
            values = earlier.params(number)
            encoded = self._space.encode(values)
            if encoded is None:
                left_out.append(number)
            elif len(out) < self.n_trials:
                out.append(encoded)
        if left_out:
            warnings.warn(
                f"warm_start: trial(s) {left_out} of the earlier search have settings this space can't hold, and are left out",
                UserWarning,
                stacklevel=4,
            )
        return out

    def _sampler(self, number: int) -> Any:
        optuna = _optuna()
        seed = _seed_of(self.seed, 0, number)
        if self.sampler == "random":
            return optuna.samplers.RandomSampler(seed=seed)
        return optuna.samplers.TPESampler(seed=seed, constant_liar=True)

    def _work_rung0(self, study: Any, run: _Run) -> None:
        """Take trials of rung 0 (warm-start ones first) until `n_trials` exist."""
        optuna = _optuna()
        while True:
            trial = study.ask()
            trials = study.get_trials(deepcopy=False)
            # The trials of the search before this one (not surplus, not interrupted): rung 0 holds n_trials of them, whoever
            # took them.
            before = sum(1 for t in trials if t.number < trial.number and not _dropped(t))
            if before >= self.n_trials:
                trial.set_user_attr(_SURPLUS, True)
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                return
            study.sampler = self._sampler(trial.number)  # read when the settings are suggested, below
            values = self._space.suggest(trial)
            reason = self._space.too_small(values)
            if reason is not None:
                # A Subset under its min_size: no setting of the search, so it takes no slot and the sampler proposes again.
                # Told the worst value there is, so that TPE learns where they are (it ignores failed trials and would keep
                # proposing them when fewer items score better); every reader of the trials leaves it out (`_dropped`).
                trial.set_user_attr(_REJECTED, reason)
                study.tell(trial, math.inf)
                rejected = sum(1 for t in study.get_trials(deepcopy=False) if t.user_attrs.get(_REJECTED))
                if rejected > max(1000, 20 * self.n_trials):
                    raise ValueError(
                        f"{rejected} proposals were subsets under their min_size ({reason}): the sampler hardly ever proposes "
                        "enough items. Lower min_size, or search fewer items."
                    )
                continue
            self._evaluate(study, trial, values, 0, run)

    def _work_rung(self, storage: Any, r: int, run: _Run) -> None:
        """Set up rung `r` (if this process creates its study: the promoted trials, enqueued) and take its trials."""
        optuna = _optuna()
        name = f"{STUDY}:rung{r}"
        try:
            study = optuna.create_study(
                storage=storage,
                study_name=name,
                direction="minimize",
                sampler=optuna.samplers.RandomSampler(seed=self.seed),
                load_if_exists=False,
            )
            leader = True
        except optuna.exceptions.DuplicatedStudyError:
            study = optuna.load_study(
                study_name=name, storage=storage, sampler=optuna.samplers.RandomSampler(seed=self.seed)
            )
            leader = False
        if leader:
            _set_up_rung(storage, study, r, self.keep)
        else:
            self._wait_for_setup(study, name)
        while True:
            trial = study.ask()
            if _PARENT not in trial.user_attrs:  # nothing left in the queue
                trial.set_user_attr(_SURPLUS, True)
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                return
            values = self._space.suggest(trial)  # the enqueued settings
            self._evaluate(study, trial, values, r, run)

    def _evaluate(self, study: Any, trial: Any, values: Dict[str, Any], r: int, run: _Run) -> None:
        optuna = _optuna()
        try:
            score = self._score(values, r, run)
        except Exception as exc:
            trial.set_user_attr(_ERROR, f"{type(exc).__name__}: {exc}"[:1000])
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            raise
        except BaseException:  # a KeyboardInterrupt, a SystemExit: the trial runs again in the next run
            _interrupt(run.storage, study, trial._trial_id)
            raise
        trial.set_user_attr(_FORECASTS, score.forecasts)
        trial.set_user_attr(_PLANNED, score.planned)
        if score.model is not None:
            trial.set_user_attr(_MODEL, score.model)
        if score.mae is None:
            trial.set_user_attr(_ERROR, score.error)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        else:
            study.tell(trial, score.mae)

    def _model(self, values: Dict[str, Any], seeds: Optional[int], where: str) -> Tuple[Any, str]:
        """The model of these settings with `seeds` in place of its own, and the fingerprint of the model `make` built."""
        from ..models.base import BaseModel

        model = self.make(dict(values))
        if not isinstance(model, BaseModel):
            raise TypeError(f"make must return one model (a BaseModel), got {type(model).__name__}")
        fingerprint = model_fingerprint(model)
        if seeds is not None:
            _set_seeds(model, seeds, where)
        return model, fingerprint

    def _score(self, values: Dict[str, Any], r: int, run: _Run) -> _Score:
        """The validation MAE of the model of these settings at rung `r`, on the rung's keys; no score if it skips any."""
        from ..pipelines.model_pipeline import ModelPipeline

        reason = self._space.too_small(values)
        if reason is not None:
            return _Score(None, error=reason)
        rung = self._rungs[r]
        model, fingerprint = self._model(values, rung.seeds, f"rungs[{r}]")
        short = _reach_problem(model, run.frame, run.validation[0], run.horizons)
        if short is not None:
            raise ValueError(f"The trial of the settings {values}: {short}")
        if r == 0:  # every rung's seeds against this model, before the dear rungs run
            for i, other in enumerate(self._rungs):
                if other.seeds is not None:
                    _set_seeds(self.make(dict(values)), other.seeds, f"rungs[{i}]")
        pipeline = ModelPipeline(**run.pipeline)
        pipeline.add_model(model)
        report = pipeline.run(
            run.frame,
            target=run.target,
            horizons=run.horizons,
            freq=run.freq,
            validation=run.validation,
            validation_every=rung.every_nth_day,
        )
        return dataclasses.replace(score_ref(report, model.name, through=run.validation[1]), model=fingerprint)

    def _check_models(self, storage: Any) -> None:
        """A ValueError if a trial on file was scored with another model than `make` builds now from its settings (a resume
        after `make` was edited would compare two searches' trials)."""
        optuna = _optuna()
        built: Dict[str, str] = {}
        for name in optuna.get_all_study_names(storage):
            if name != STUDY and not name.startswith(f"{STUDY}:rung"):
                continue
            for t in optuna.load_study(study_name=name, storage=storage).get_trials(deepcopy=False):
                stored = t.user_attrs.get(_MODEL)
                if stored is None or not _has_params(t, self._space):
                    continue
                values = self._space.decode(t.params)
                key = _canonical(values)
                if key not in built:
                    built[key] = model_fingerprint(self.make(dict(values)))
                if built[key] != stored:
                    raise ValueError(
                        f"Trial {t.number} of {name!r} in {self.storage} was scored with another model than make builds now "
                        f"from its settings {values}: make was changed since. Use another storage file for this search."
                    )

    def _promote(self, storage: Any, r: int) -> List[int]:
        return _promote(storage, r, self.keep)

    def _rung_done(self, storage: Any, r: int) -> bool:
        optuna = _optuna()
        if r == 0:
            study = optuna.load_study(study_name=STUDY, storage=storage)
            live = sorted((t for t in study.get_trials(deepcopy=False) if not _dropped(t)), key=lambda t: t.number)[
                : self.n_trials
            ]
            return len(live) == self.n_trials and all(t.state.is_finished() for t in live)
        study = optuna.load_study(study_name=f"{STUDY}:rung{r}", storage=storage)
        if _READY not in study.user_attrs:
            return False
        done = {
            t.user_attrs[_PARENT]
            for t in study.get_trials(deepcopy=False)
            if not _dropped(t) and _PARENT in t.user_attrs and t.state.is_finished()
        }
        return set(study.user_attrs[_PROMOTED]) <= done

    def _wait_for_setup(self, study: Any, name: str) -> None:
        """Wait for the process that created `study` to set it up; a TimeoutError saying how to recover if it stopped."""
        timeout = _seconds(ENV_SETUP_TIMEOUT, 120.0)
        started = time.monotonic()
        while _READY not in study.user_attrs:
            if time.monotonic() - started > timeout:
                fix = (
                    "No trial ran: remove the journal file and start the search again."
                    if name == STUDY
                    else f"Kill the workers, then run `epf tune --requeue {self.storage}`, which finishes its setup."
                )
                raise TimeoutError(
                    f"The study {name!r} in {self.storage} was created and not set up within {timeout:.0f} s: the process that "
                    f"created it stopped before. {fix}"
                )
            time.sleep(_poll())

    def _wait_for_rung(self, storage: Any, r: int) -> None:
        """Wait until every trial of rung `r` has finished, naming the ones still running now and then: a worker killed
        during a trial leaves it running, and only `epf tune --requeue` (with no worker running) puts it back."""
        every = _seconds(ENV_WAIT_NOTE, 900.0)
        noted = time.monotonic()
        while not self._rung_done(storage, r):
            if time.monotonic() - noted > every:
                noted = time.monotonic()
                warnings.warn(
                    f"Still waiting for trial(s) {_running(storage, r)} of rung {r} in {self.storage}. If no worker runs them "
                    f"any more, kill the workers, then run `epf tune --requeue {self.storage}` and start them again.",
                    UserWarning,
                    stacklevel=2,
                )
            time.sleep(_poll())

    def _record(self, storage: Any, settings: Dict[str, Any]) -> Dict[str, Any]:
        """The result's record, from the storage once every rung has finished: the same in every worker."""
        optuna = _optuna()
        first = optuna.load_study(study_name=STUDY, storage=storage)
        live = sorted((t for t in first.get_trials(deepcopy=False) if not _dropped(t)), key=lambda t: t.number)
        live = live[: self.n_trials]
        by_rung = [_rung_results(storage, r) for r in range(len(self._rungs))]
        trials = []
        for t in live:
            rungs = []
            for r, results in enumerate(by_rung):
                if t.number not in results:
                    continue
                value, forecasts, planned, error = results[t.number]
                rungs.append(
                    {
                        "rung": r,
                        "every_nth_day": self._rungs[r].every_nth_day,
                        "seeds": self._rungs[r].seeds,
                        "state": "complete" if value is not None else "failed",
                        "validation_mae": value,
                        "forecasts": forecasts,
                        "planned": planned,
                        "error": error,
                    }
                )
            trials.append(
                {
                    "trial": t.number,
                    "params": _plain(self._space.decode(t.params)) if _has_params(t, self._space) else None,
                    "warm_start": bool(t.user_attrs.get(_WARM, False)),
                    "model": t.user_attrs.get(_MODEL),  # the fingerprint of the model it scored, None if it never ran
                    "rungs": rungs,
                }
            )
        promoted = []
        for r in range(1, len(self._rungs)):
            study = optuna.load_study(study_name=f"{STUDY}:rung{r}", storage=storage)
            promoted.append(list(study.user_attrs[_PROMOTED]))
        name = None
        complete = [trial for trial in trials if trial["params"] is not None]
        if complete:
            with contextlib.suppress(Exception):
                name = self.make(dict(complete[0]["params"])).name
        from ..results.manifest import _package_version

        return {
            RESULT_KEY: RESULT_VERSION,
            "name": None if name is None else f"{name}_bayes",
            "package_version": _package_version(),
            "search": {
                **settings,
                "warm_start": _warm_label(self.warm_start),
                "storage": None if self.storage is None else str(self.storage),
            },
            "trials": trials,
            # How many proposals were Subsets under their min_size: proposed again, never part of the trials.
            "rejected": sum(1 for t in first.get_trials(deepcopy=False) if t.user_attrs.get(_REJECTED)),
            "promoted": promoted,
            "chosen": [],
        }


def _warm_label(warm: Any) -> Optional[str]:
    """What the record says a search was warm-started from: the earlier result's file, or its name if it wasn't saved."""
    if warm is None:
        return None
    if isinstance(warm, SearchResult):
        return str(warm.path) if warm.path is not None else f"{warm.name} (not saved)"
    return str(warm)


def _dropped(trial: Any) -> bool:
    """A trial that is no trial of the search: one asked for past `n_trials` (or past a rung's queue), interrupted, or a
    rejected proposal (a Subset under its min_size)."""
    return bool(trial.user_attrs.get(_SURPLUS) or trial.user_attrs.get(_INTERRUPTED) or trial.user_attrs.get(_REJECTED))


def _has_params(trial: Any, space: _Space) -> bool:
    return all(name in trial.params for d in space.dimensions for name in d.names())


def _rung_results(storage: Any, r: int) -> Dict[int, Tuple[Optional[float], int, int, Optional[str]]]:
    """Rung-0 trial number -> (validation MAE or None, forecasts, planned, error) of the finished trials of rung `r`."""
    optuna = _optuna()
    name = STUDY if r == 0 else f"{STUDY}:rung{r}"
    study = optuna.load_study(study_name=name, storage=storage)
    out = {}
    for t in sorted(study.get_trials(deepcopy=False), key=lambda t: t.number):
        if _dropped(t) or not t.state.is_finished():
            continue
        number = t.number if r == 0 else t.user_attrs.get(_PARENT)
        if number is None:
            continue
        value = float(t.value) if t.state == optuna.trial.TrialState.COMPLETE else None
        out[number] = (
            value,
            int(t.user_attrs.get(_FORECASTS, 0)),
            int(t.user_attrs.get(_PLANNED, 0)),
            t.user_attrs.get(_ERROR),
        )
    return out


def _rung_study(r: int) -> str:
    return STUDY if r == 0 else f"{STUDY}:rung{r}"


def _promote(storage: Any, r: int, keep: float) -> List[int]:
    """The rung-0 trial numbers that go on from rung `r`: the best `keep` share of the trials it tried (scored or failed),
    the best by validation MAE among those scored, ties by trial number; at least one."""
    scored = _rung_results(storage, r)
    ranked = sorted((value, number) for number, (value, _, _, _) in scored.items() if value is not None)
    count = max(1, math.floor(len(scored) * keep + 1e-9))
    return [number for _, number in ranked[:count]]


def _set_up_rung(storage: Any, study: Any, r: int, keep: float) -> None:
    """Rung `r`'s study set up: the trials rung r-1 promotes, each enqueued once with its rung-0 settings, then READY. Also
    what `requeue` runs on a study whose creator stopped before it was set up: what is enqueued already isn't again."""
    promoted = list(study.user_attrs.get(_PROMOTED) or _promote(storage, r - 1, keep))
    params = _first_params(storage)
    there = {t.user_attrs.get(_PARENT) for t in study.get_trials(deepcopy=False)}
    if _PROMOTED not in study.user_attrs:
        study.set_user_attr(_PROMOTED, promoted)
    for number in promoted:
        if number not in there:
            study.enqueue_trial(params[number], user_attrs={_PARENT: number})
    study.set_user_attr(_READY, True)


def _first_params(storage: Any) -> Dict[int, Dict[str, Any]]:
    """Rung-0 trial number -> the sampler's parameters of its settings (an enqueued trial's fixed ones if it ran no further)."""
    optuna = _optuna()
    study = optuna.load_study(study_name=STUDY, storage=storage)
    return {
        t.number: dict(t.params or t.system_attrs.get("fixed_params") or {}) for t in study.get_trials(deepcopy=False)
    }


def _running(storage: Any, r: int) -> List[int]:
    """The trials of rung `r` still running, by rung-0 trial number."""
    optuna = _optuna()
    study = optuna.load_study(study_name=_rung_study(r), storage=storage)
    return [
        t.number if r == 0 else t.user_attrs.get(_PARENT)
        for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
        if not _dropped(t)
    ]


def _interrupt(storage: Any, study: Any, trial_id: int) -> None:
    """Mark a trial interrupted and put it back: a later rung's trial enqueued again with its parent's rung-0 settings (it may
    have been killed before it suggested them), a warm-start trial with its settings; any other rung-0 trial frees its slot,
    which the sampler proposes again."""
    optuna = _optuna()
    frozen = storage.get_trial(trial_id)
    storage.set_trial_user_attr(trial_id, _INTERRUPTED, True)
    if not frozen.state.is_finished():
        storage.set_trial_state_values(trial_id, optuna.trial.TrialState.FAIL)
    if frozen.user_attrs.get(_SURPLUS):
        return
    if study.study_name != STUDY:
        parent = frozen.user_attrs.get(_PARENT)
        if parent is not None:
            study.enqueue_trial(_first_params(storage)[parent], user_attrs={_PARENT: parent})
    elif frozen.user_attrs.get(_WARM):
        study.enqueue_trial(dict(frozen.system_attrs.get("fixed_params") or frozen.params), user_attrs={_WARM: True})


def requeue(journal: Union[str, Path]) -> Dict[str, Any]:
    """What `epf tune --requeue JOURNAL` does, with no worker running: every trial left running in the journal's search is
    marked interrupted and put back (`_interrupt`), and a later rung's study left unset by a killed worker is set up. Reads no
    data. Returns {"requeued": [(study, trial number)], "notes": [...]}."""
    optuna = _optuna()
    path = Path(journal)
    if not path.is_file():
        raise FileNotFoundError(f"No journal file {path}")
    from optuna.storages.journal import JournalFileBackend

    storage = optuna.storages.JournalStorage(JournalFileBackend(str(path)))
    requeued, notes = [], []
    with _quiet_optuna():
        names = optuna.get_all_study_names(storage)
        if STUDY not in names:
            raise ValueError(f"{path} holds no BayesSearch (no study {STUDY!r})")
        first = optuna.load_study(study_name=STUDY, storage=storage)
        if _READY not in first.user_attrs:
            notes.append(f"Rung 0 was never set up, so no trial ran: remove {path} and start the search again.")
            return {"requeued": requeued, "notes": notes}
        keep = float(first.user_attrs[_SETTINGS]["keep"])
        for name in sorted(names, key=lambda n: (n != STUDY, n)):
            if name != STUDY and not name.startswith(f"{STUDY}:rung"):
                continue
            study = optuna.load_study(study_name=name, storage=storage)
            for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
                _interrupt(storage, study, t._trial_id)
                requeued.append((name, t.number))
            if _READY not in study.user_attrs:
                _set_up_rung(storage, study, int(name.rsplit("rung", 1)[1]), keep)
                notes.append(f"Set up {name!r}, whose creator stopped before it did.")
    return {"requeued": requeued, "notes": notes}


def _reach_problem(model: Any, frame: pd.DataFrame, first_day: str, horizons: Sequence[int]) -> Optional[str]:
    """Why the frame is too short for a model of `refit=None` (a run checks a refit model's itself, `_check_reaches_back`):
    the first validation forecasts train on the rows of their window, back to `training_span`'s first row (the newest target
    on D-1), and a predictor's day lag reads that many days before it. A frame starting later trains them on fewer days
    than the window says, silently, so the windows of a search would compare on different terms. None if it reaches."""
    from ..models._calendar import EXPANDING, training_span
    from ..models._contract import wall_clock
    from ..models.base import _days_back

    if getattr(model, "refit", None) is not None or not len(frame.index):
        return None
    widths = [m for m in getattr(model, "_members", ()) if m != EXPANDING and isinstance(m, (int, np.integer))]
    if not widths:
        return None
    day = pd.Timestamp(first_day).date().toordinal()
    first_row, width, horizon = min(
        (int(training_span(day, int(w), int(h), -1)[0]), int(w), int(h)) for w in widths for h in horizons
    )
    predictors = getattr(model, "predictors", None)
    names = [
        name.replace("{horizon}", str(h))
        for name in (predictors if isinstance(predictors, list) else [])
        for h in horizons
    ]
    lag, lagged = _days_back(name for name in names if isinstance(name, str))
    needed = pd.Timestamp.fromordinal(first_row - lag).date()
    starts = wall_clock(frame.index).min().date()
    if starts <= needed:
        return None
    back = "" if not lag else f", and its {lagged} reads {lag} day{'s' if lag > 1 else ''} before that"
    return (
        f"{model.name}'s training window of {width} days at horizon {horizon} trains the first validation forecast "
        f"({pd.Timestamp(first_day).date()}) on rows from {pd.Timestamp.fromordinal(first_row).date()}{back}, but the frame "
        f"starts on {starts}: it would train on fewer days than its window says. Start the frame on {needed} or earlier, or "
        "search smaller windows."
    )


def cut_frame(data: pd.DataFrame, validation_end: Any) -> pd.DataFrame:
    """The rows of `data` on local days up to `validation_end`, in their order: what a search's trials read. A row after it is
    the test period's, or later, and so is the actual of a forecast for a day after it."""
    from ..models._contract import wall_clock

    through = pd.Timestamp(pd.Timestamp(validation_end).date())
    return data[np.asarray(wall_clock(data.index).normalize() <= through)]


def score_ref(report: Any, name: str, through: Optional[str] = None) -> _Score:
    """The validation MAE of model `name` in `report` (the validation report of its run), over the keys with an actual for a
    day up to `through` (every day without it), if it forecast every planned key the frame has a row to issue from with a
    finite value; else no score, with the reason."""
    counts = report.forecast_counts().set_index("model").loc[name]
    planned, produced = int(counts["planned"]), int(counts["produced"])
    skipped = {
        reason: int(count)
        for reason, count in counts.items()
        if reason not in ("planned", "produced", "missing") and count
    }
    predictions, actuals = [], []
    for record in report.refs[name].iter_records(cols=["target_date", "prediction", "actual"]):
        predictions.append(record.get("prediction"))
        # A target after `through` is the test period's: forecast, never scored.
        late = through is not None and record.get("target_date", "") > through
        actuals.append(None if late else record.get("actual"))
    predicted = np.asarray([np.nan if p is None else p for p in predictions], dtype=np.float64)
    actual = np.asarray([np.nan if a is None else a for a in actuals], dtype=np.float64)
    lacking = planned - produced - skipped.get(_MISSING_TEST_ROW, 0)
    if lacking > 0:
        shown = {reason: count for reason, count in skipped.items() if reason != _MISSING_TEST_ROW}
        return _Score(
            None, produced, planned, f"forecast {produced} of {planned} planned keys (skipped: {shown or lacking})"
        )
    if not np.isfinite(predicted).all():
        bad = int((~np.isfinite(predicted)).sum())
        return _Score(None, produced, planned, f"{bad} forecast(s) not finite")
    scored = np.isfinite(actual)
    if not scored.any():
        return _Score(None, 0, planned, "no forecast has an actual")
    # An exact sum: the same bits whatever order the pool finished the forecasts in.
    mae = math.fsum(np.abs(predicted[scored] - actual[scored]).tolist()) / int(scored.sum())
    return _Score(mae, int(scored.sum()), planned, None)


# ---------------------------------------------------------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------------------------------------------------------


class SearchResult:
    """The record of a search: its settings and space, every trial (settings, the rungs it reached with their validation
    MAE and forecast counts), the trials each rung promoted and what `best()` chose. `save()` writes it and `load()` reads it
    back; it, not a rerun, is the record of a search run by parallel workers."""

    def __init__(self, record: Dict[str, Any], make: Optional[Callable[[Dict[str, Any]], Any]] = None):
        if record.get(RESULT_KEY) != RESULT_VERSION:
            raise ValueError(f"Not a saved search result (version {RESULT_VERSION}): {list(record)[:5]}")
        self._record = record
        self._make = make
        self.path: Optional[Path] = None
        self._space = _Space.from_record(record["search"]["space"])
        self._handed_out: List[Dict[str, Any]] = []  # the provenance of best()'s models, given the file when saved

    @property
    def name(self) -> Optional[str]:
        """`{model name}_bayes`, from the model of the first trial."""
        return self._record.get("name")

    @property
    def record(self) -> Dict[str, Any]:
        """The record `save()` writes (a copy)."""
        return json.loads(json.dumps(self._record))

    @property
    def candidates(self) -> int:
        """How many settings the search tried (its rung-0 trials)."""
        return len(self._record["trials"])

    def _trial(self, number: int) -> Dict[str, Any]:
        for trial in self._record["trials"]:
            if trial["trial"] == number:
                return trial
        raise KeyError(f"No trial {number} in this search")

    def params(self, number: int) -> Dict[str, Any]:
        """The settings of trial `number`, as `make` got them."""
        params = self._trial(number)["params"]
        if params is None:
            raise ValueError(f"Trial {number} has no settings (it failed before its settings were drawn)")
        return dict(params)

    @staticmethod
    def _last(trial: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        """The highest rung a trial was scored on (complete), or None."""
        complete = [rung for rung in trial["rungs"] if rung["state"] == "complete"]
        return max(complete, key=lambda rung: rung["rung"]) if complete else None

    @classmethod
    def _failed_after(cls, trial: Mapping[str, Any]) -> bool:
        """Whether a trial failed a rung after the last one it was scored on: promoted, then no score on the dearer rung."""
        last = cls._last(trial)
        return last is not None and any(
            rung["state"] != "complete" and rung["rung"] > last["rung"] for rung in trial["rungs"]
        )

    def ranking(self) -> List[int]:
        """Trial numbers, best first: the highest rung scored first, then the validation MAE there, then the trial number.
        Trials never scored are left out, and so is a trial that failed a rung after its last score: it was promoted and then
        skipped keys or gave non-finite forecasts at the dearer rung, so its cheaper score doesn't stand."""
        keyed = []
        for trial in self._record["trials"]:
            last = self._last(trial)
            if last is not None and not self._failed_after(trial):
                keyed.append((-last["rung"], last["validation_mae"], trial["trial"]))
        return [number for _, _, number in sorted(keyed)]

    def _top(self) -> Optional[int]:
        """The highest rung a ranked trial was scored on."""
        ranked = self.ranking()
        return None if not ranked else self._last(self._trial(ranked[0]))["rung"]

    def finalists(self) -> List[int]:
        """The ranked trials scored on the highest rung any ranked trial reached (what a later search warm-starts from)."""
        top = self._top()
        return [number for number in self.ranking() if self._last(self._trial(number))["rung"] == top]

    def table(self) -> pd.DataFrame:
        """One row per trial, in trial order: `trial`, each setting, `warm_start`, `rung` (the highest rung it was scored on),
        `validation_mae` there, then per rung r `mae_r<r>` and `forecasts_r<r>` (empty where it didn't run or got no score),
        and `error` (why its last rung gave no score)."""
        keys = [d.key for d in self._space.dimensions]
        n_rungs = len(self._record["search"]["rungs"])
        rows = []
        for trial in self._record["trials"]:
            last = self._last(trial)
            row: Dict[str, Any] = {"trial": trial["trial"]}
            for key in keys:
                row[key] = None if trial["params"] is None else trial["params"][key]
            row["warm_start"] = trial["warm_start"]
            row["rung"] = None if last is None else last["rung"]
            row["validation_mae"] = np.nan if last is None else last["validation_mae"]
            by_rung = {rung["rung"]: rung for rung in trial["rungs"]}
            for r in range(n_rungs):
                rung = by_rung.get(r)
                row[f"mae_r{r}"] = np.nan if rung is None or rung["validation_mae"] is None else rung["validation_mae"]
                row[f"forecasts_r{r}"] = None if rung is None or rung["state"] != "complete" else rung["forecasts"]
            errors = [rung["error"] for rung in trial["rungs"] if rung["error"]]
            row["error"] = errors[-1] if errors else None
            rows.append(row)
        table = pd.DataFrame(rows)
        for column in ["rung", *[f"forecasts_r{r}" for r in range(n_rungs)]]:
            if column in table:
                table[column] = table[column].astype("Int64")
        return table

    def best(
        self, k: int = 1, seeds: Optional[int] = None, make: Optional[Callable[[Dict[str, Any]], Any]] = None
    ) -> List[Any]:
        """The `k` best trials' models (`ranking()`), built by `make` (the search's; pass it for a loaded result) and named
        `{name}_bayes_00`, `{name}_bayes_01`, ...: ordinary models with their full settings, to add to a pipeline that runs
        the test period. `seeds` replaces each model's `seeds` (a model without that parameter raises). Each model's
        manifest records where it came from (the result file once saved, the trial, its validation MAE and the number of
        candidates) as provenance, and `report.choices()` shows it with its validation-test gap. The choice goes into the
        record, and into the file if it was saved."""
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a whole number of at least 1, got {k!r}")
        if seeds is not None and (isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 1):
            raise ValueError(f"seeds must be a whole number of at least 1, got {seeds!r}")
        make = make or self._make
        if make is None:
            raise ValueError(
                "This result was loaded from a file: pass make=, the function that builds a model from settings"
            )
        ranked = self.ranking()
        if len(ranked) < k:
            raise ValueError(f"best({k}): the search scored {len(ranked)} trial(s)")
        top = self._top()
        below = [number for number in ranked[:k] if self._last(self._trial(number))["rung"] != top]
        if below:
            warnings.warn(
                f"best({k}) reaches below the {len(self.finalists())} finalist(s): "
                f"trial(s) {below} were ranked on a cheaper rung than rung {top}, on fewer days or seeds; their validation "
                "MAE isn't comparable with the finalists', and report.choices() shows none for them",
                UserWarning,
                stacklevel=2,
            )
        from ..models.base import BaseModel

        models, chosen = [], []
        for index, number in enumerate(ranked[:k]):
            trial = self._trial(number)
            last = self._last(trial)
            model = make(self.params(number))
            if not isinstance(model, BaseModel):
                raise TypeError(f"make must return one model (a BaseModel), got {type(model).__name__}")
            scored = trial.get("model")
            if scored is not None and model_fingerprint(model) != scored:
                raise ValueError(
                    f"best({k}): make builds another model from trial {number}'s settings {self.params(number)} than the one the "
                    "search scored: pass the make the search ran with"
                )
            if seeds is not None:
                _set_seeds(model, seeds, f"best(seeds={seeds})")
            model.name = f"{model.name}_bayes_{index:02d}"
            provenance = {
                "search": self.name,
                "file": None if self.path is None else str(self.path),
                "trial": number,
                "rank": index,
                "candidates": self.candidates,
                "rung": last["rung"],
                "every_nth_day": last["every_nth_day"],
                "seeds": last["seeds"],
                "validation": list(self._record["search"]["validation"]),
                # The last day whose data the search read: a test period of this model must start after it.
                "data_through": self._record["search"].get("data_through", self._record["search"]["validation"][1]),
                "validation_mae": last["validation_mae"],
                "validation_forecasts": last["forecasts"],
                "validation_share": last["forecasts"] / last["planned"] if last["planned"] else None,
                # The rung the finalists were scored on: a model from a cheaper rung has a score of another fidelity.
                "top_rung": top,
                # What the search scored: a final run that forecasts something else is warned about in report.choices().
                "target": self._record["search"].get("target"),
                "horizons": list(self._record["search"].get("horizons") or []),
                "freq": self._record["search"].get("freq"),
                "issue_days": None,
            }
            model._search = provenance
            self._handed_out.append(provenance)
            models.append(model)
            chosen.append({"name": model.name, "trial": number})
        self._record["chosen"].append({"k": k, "seeds": seeds, "models": chosen})
        if self.path is not None:
            self.save(self.path)
        return models

    def save(self, path: Union[str, Path]) -> None:
        """Write the record as YAML (written in full before the file is replaced). Models `best()` gave before get the file in
        their provenance, and a later `best()` adds its choice to the file."""
        from ..pipelines._yaml import dump_yaml

        path = Path(path)
        text = dump_yaml(self._record)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
        partial.write_text(text)
        os.replace(partial, path)
        self.path = path
        for provenance in self._handed_out:
            provenance["file"] = str(path)

    @classmethod
    def load(cls, path: Union[str, Path], make: Optional[Callable[[Dict[str, Any]], Any]] = None) -> "SearchResult":
        """A saved result; `make` lets `best()` build its models."""
        path = Path(path)
        record = yaml.safe_load(path.read_text())
        if not isinstance(record, dict):
            raise ValueError(f"{path} is not a saved search result")
        result = cls(record, make=make)
        result.path = path
        return result


__all__ = ["BayesSearch", "SearchResult", "Subset", "cut_frame", "requeue", "score_ref"]
