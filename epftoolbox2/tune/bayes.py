"""`BayesSearch`: Optuna's proposals and successive halving over the validation period, returning ordinary models.

How a search runs (`BayesSearch.run`):

1. The frame is cut after the validation end plus the largest horizon (local days), before the first trial, and `test=` is
   refused: no trial can read the test period, whatever the model does.
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

# Set by `epf tune` for the processes it starts: how many share the storage, and (alone) a run that only requeues.
ENV_WORKERS = "EPF_TUNE_WORKERS"
ENV_WORKER = "EPF_TUNE_WORKER"
ENV_REQUEUE = "EPF_TUNE_REQUEUE"
# Seconds between two looks at the storage while waiting for other workers (a rung to finish, a study to be set up).
ENV_POLL = "EPF_TUNE_POLL"
# How long a process waits for the one that created a study to set it up, in seconds.
SETUP_TIMEOUT = 600.0

# Trial user attributes.
_WARM = "warm_start"
_SURPLUS = "surplus"
_INTERRUPTED = "interrupted"
_PARENT = "trial"
_ERROR = "error"
_FORECASTS = "forecasts"
_PLANNED = "planned"
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
        items = tuple(items)
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


def _dimension(key: Any, spec: Any) -> _Dimension:
    if not isinstance(key, str) or not key or "[" in key or "]" in key:
        raise ValueError(f"A space key must be a non-empty string without '[' or ']', got {key!r}")
    where = f"space[{key!r}]"
    if isinstance(spec, Subset):
        return _Dimension(key, "subset", spec.items, min_size=spec.min_size)
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
        return _Dimension(key, kind, low=int(low) if kind == "int" else float(low), high=high, log=log)
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
        model through a `ModelPipeline` of these on the frame cut after the validation end plus the largest horizon.
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
        frame = cut_frame(data, end, max(planned_horizons))
        run = _Run(frame, target, tuple(planned_horizons), freq, (start, end), pipeline)
        optuna = _optuna()
        with _quiet_optuna():
            storage = self._storage()
            if os.environ.get(ENV_REQUEUE) == "1":
                found = self._requeue(storage)
                print(f"Requeued {found} interrupted trial(s) in {self.storage}")
                raise SystemExit(0)
            settings = self._settings(run)
            study = self._open(storage, STUDY, optuna.samplers.RandomSampler(seed=self.seed), settings, warm=True)
            self._work_rung0(study, run)
            for r in range(1, len(self._rungs)):
                self._wait(lambda: self._rung_done(storage, r - 1))
                self._work_rung(storage, r, run)
            self._wait(lambda: self._rung_done(storage, len(self._rungs) - 1))
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
            "data_through": _through(run.validation[1], max(run.horizons)),
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
            self._wait(
                lambda: _READY in study.user_attrs, timeout=SETUP_TIMEOUT, what=f"the study {name!r} to be set up"
            )
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
            before = sum(1 for t in trials if t.number < trial.number and not t.user_attrs.get(_INTERRUPTED))
            if before >= self.n_trials:
                trial.set_user_attr(_SURPLUS, True)
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                return
            study.sampler = self._sampler(trial.number)  # read when the settings are suggested, below
            values = self._space.suggest(trial)
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
            promoted = self._promote(storage, r - 1)
            first = optuna.load_study(study_name=STUDY, storage=storage)
            params = {t.number: t.params for t in first.get_trials(deepcopy=False)}
            study.set_user_attr(_PROMOTED, promoted)
            for number in promoted:
                study.enqueue_trial(params[number], user_attrs={_PARENT: number})
            study.set_user_attr(_READY, True)
        else:
            self._wait(
                lambda: _READY in study.user_attrs, timeout=SETUP_TIMEOUT, what=f"the study {name!r} to be set up"
            )
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
        except BaseException:
            trial.set_user_attr(_INTERRUPTED, True)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            raise
        trial.set_user_attr(_FORECASTS, score.forecasts)
        trial.set_user_attr(_PLANNED, score.planned)
        if score.mae is None:
            trial.set_user_attr(_ERROR, score.error)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        else:
            study.tell(trial, score.mae)

    def _model(self, values: Dict[str, Any], seeds: Optional[int], where: str) -> Any:
        from ..models.base import BaseModel

        model = self.make(dict(values))
        if not isinstance(model, BaseModel):
            raise TypeError(f"make must return one model (a BaseModel), got {type(model).__name__}")
        if seeds is not None:
            _set_seeds(model, seeds, where)
        return model

    def _score(self, values: Dict[str, Any], r: int, run: _Run) -> _Score:
        """The validation MAE of the model of these settings at rung `r`, on the rung's keys; no score if it skips any."""
        from ..pipelines.model_pipeline import ModelPipeline

        reason = self._space.too_small(values)
        if reason is not None:
            return _Score(None, error=reason)
        rung = self._rungs[r]
        model = self._model(values, rung.seeds, f"rungs[{r}]")
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
        return score_ref(report, model.name)

    def _promote(self, storage: Any, r: int) -> List[int]:
        """The rung-0 trial numbers that go on from rung `r`: the best `keep` share of those it scored, at least one."""
        scored = _rung_results(storage, r)
        evaluated = len(scored)
        ranked = sorted(
            ((value, number) for number, (value, _, _, _) in scored.items() if value is not None),
            key=lambda pair: (pair[0], pair[1]),
        )
        count = max(1, math.floor(evaluated * self.keep + 1e-9))
        return [number for _, number in ranked[:count]]

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

    def _wait(
        self, ready: Callable[[], bool], timeout: Optional[float] = None, what: str = "the other workers"
    ) -> None:
        started = time.monotonic()
        while not ready():
            if timeout is not None and time.monotonic() - started > timeout:
                raise TimeoutError(
                    f"Waited {timeout:.0f} s for {what} in {self.storage}: the process that created it may have stopped. "
                    "Start the search again with another storage file."
                )
            time.sleep(_poll())

    def _requeue(self, storage: Any) -> int:
        """Mark every running trial interrupted (with no worker running): its rung-0 slot is proposed again, a later rung's
        trial enqueued again."""
        optuna = _optuna()
        found = 0
        for name in optuna.get_all_study_names(storage):
            if name != STUDY and not name.startswith(f"{STUDY}:rung"):
                continue
            study = optuna.load_study(study_name=name, storage=storage)
            for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
                storage.set_trial_user_attr(t._trial_id, _INTERRUPTED, True)
                storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)
                if name != STUDY:
                    study.enqueue_trial(t.params, user_attrs={_PARENT: t.user_attrs[_PARENT]})
                found += 1
        return found

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
                "warm_start": self.warm_start
                if isinstance(self.warm_start, (str, type(None)))
                else str(self.warm_start),
                "storage": None if self.storage is None else str(self.storage),
            },
            "trials": trials,
            "promoted": promoted,
            "chosen": [],
        }


def _dropped(trial: Any) -> bool:
    """A trial that is no trial of the search: one asked for past `n_trials` (or past a rung's queue), or interrupted."""
    return bool(trial.user_attrs.get(_SURPLUS) or trial.user_attrs.get(_INTERRUPTED))


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


def _through(end: str, horizon: int) -> str:
    return (pd.Timestamp(end) + pd.Timedelta(days=horizon)).date().isoformat()


def cut_frame(data: pd.DataFrame, validation_end: Any, horizon: int) -> pd.DataFrame:
    """The rows of `data` on local days up to `validation_end` plus `horizon` (the last target day a validation forecast
    has), in their order: what a search's trials read. A row after it is the test period's, or later."""
    from ..models._contract import wall_clock

    through = pd.Timestamp(_through(str(pd.Timestamp(validation_end).date()), horizon))
    return data[np.asarray(wall_clock(data.index).normalize() <= through)]


def score_ref(report: Any, name: str) -> _Score:
    """The validation MAE of model `name` in `report` (the validation report of its run), over the keys with an actual, if it
    forecast every planned key the frame has a row to issue from with a finite value; else no score, with the reason."""
    counts = report.forecast_counts().set_index("model").loc[name]
    planned, produced = int(counts["planned"]), int(counts["produced"])
    skipped = {
        reason: int(count)
        for reason, count in counts.items()
        if reason not in ("planned", "produced", "missing") and count
    }
    predictions, actuals = [], []
    for record in report.refs[name].iter_records(cols=["prediction", "actual"]):
        predictions.append(record.get("prediction"))
        actuals.append(record.get("actual"))
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

    def ranking(self) -> List[int]:
        """Trial numbers, best first: the highest rung scored first, then the validation MAE there, then the trial number.
        Trials never scored are left out."""
        keyed = []
        for trial in self._record["trials"]:
            last = self._last(trial)
            if last is not None:
                keyed.append((-last["rung"], last["validation_mae"], trial["trial"]))
        return [number for _, _, number in sorted(keyed)]

    def finalists(self) -> List[int]:
        """The ranked trials scored on the highest rung any trial reached (what a later search warm-starts from)."""
        ranked = self.ranking()
        if not ranked:
            return []
        top = self._last(self._trial(ranked[0]))["rung"]
        return [number for number in ranked if self._last(self._trial(number))["rung"] == top]

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
        from ..models.base import BaseModel

        models, chosen = [], []
        for index, number in enumerate(ranked[:k]):
            trial = self._trial(number)
            last = self._last(trial)
            model = make(self.params(number))
            if not isinstance(model, BaseModel):
                raise TypeError(f"make must return one model (a BaseModel), got {type(model).__name__}")
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
                "validation_mae": last["validation_mae"],
                "validation_forecasts": last["forecasts"],
                "validation_share": last["forecasts"] / last["planned"] if last["planned"] else None,
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


__all__ = ["BayesSearch", "SearchResult", "Subset", "cut_frame", "score_ref"]
