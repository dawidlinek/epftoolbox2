"""Foundation models (step 6i): pretrained sequence models that forecast the target's next steps from its own history and its
covariates' (`T0Model`, t0-alpha; `ChronosModel`, Chronos-2), zero-shot or, for t0, fine-tuned from the base weights at each
anchor of a calendar. `FoundationModel` is their shared base, which 6j's families build on.

**What a forecast reads.** A forecast issued on D at horizon h reads series, not predictor columns, through the history
accessor (`models._history`): the target's last `context x periods` steps in physical time on the UTC index, ending where the
information set and `newest_target` say (`"d-1"`, the default, at D-1's last step; `"published"` at the newest step its rule
calls known at the issue time, D's prices at 11:45), each `known_ahead` series over the same steps and the steps ahead to the
end of the target day, each `past` series over the context, NaN after its own newest known step. So the context has one length
on every day, a DST target day has 23 or 25 steps ahead (decision 12), and every series is an input `verify` perturbs,
`report.availability()` lists (one row per series) and the manifest hashes. Calendar covariates are not offered (they always
hurt, `zoo_tsfm/REPORT.md:25`).

**Zero-shot** (`fine_tune_every=None`): each forecast is its own, with no fit and no anchor beyond its run date (`refit="1D"`,
internal). The calls are the issue days of a fixed calendar of `batch_size` days (`"<batch_size>D"` from Monday 1970-01-05),
every day of it that the frame has and whose series the backend can take, the run's or not, grouped by S and padded to
`batch_size` items with copies of the first, so a call has one shape and holds the same items alone, in a full run, after a
resume and whatever `test_start`: on a CPU a forecast is the same bits in all of them.

**Fine-tuned** (`fine_tune_every="12MS"`, t0's default): at each anchor of that calendar (the fixed refit calendar of step 3),
for each horizon and each of `seeds` seeds, the family fine-tunes a copy of the base weights (never the previous anchor's, plan
section 3.13) on the windows of the target days up to the anchor's cut, the newest full day of targets known at its issue time
(D-1 under `"d-1"`, the anchor itself under `"published"` with a 13:00 price rule), each window read as the forecast issued
`horizon` days before its target day; the model asserts that no window's target is after the cut before the trainer runs, and the
trainer asserts it again. The block's issue days are then forecast in calls of `batch_size`, and a forecast is the mean over the
seeds of each seed's median. The recipe is t0_ft.py's (`fine_tune`, a plain dict, YAML verbatim; see `FINE_TUNE_DEFAULTS`).
GPU fine-tunes are not repeatable (t0 repeats differed by up to 0.16 EUR/MWh), so each fine-tune is stored in
`save_dir/<name>.fits/` (decision 13) with a sidecar (its sha256, the data end, the best step, the digest of its windows and of
everything that decides it), and a resume or a re-forecast loads it instead of fine-tuning again; a missing one is fine-tuned
again, one whose file no longer has its sha256 raises.

**Where it runs** (decision 19): in this process, on `device` ("auto": CUDA when a matmul runs on it, else the CPU) with
`THREADS_PER_PROCESS` torch threads (default `min(8, usable CPUs)`), TF32 off, inside `torch_scope`, which restores the
process's settings afterwards. The backend is loaded once per process from the weights the pre-flight resolved
(`epf weights fetch`), offline, each file by name. The manifest records and compares the runtime packages (t0 0.5.0 against git
moved single hours by 161 EUR/MWh), the weight hashes, the device type, dtype, TF32 and the thread count.

**Outputs** are point forecasts: the 0.5 level of the family's quantiles (decision 24). Records are a day unit's: every planned
(period, horizon, run date), `refit_anchor` the run date (zero-shot) or the fine-tune's anchor, coefficients `[]`.
"""

from __future__ import annotations

import collections
import contextlib
import datetime
import hashlib
import json
import math
import os
import re
import warnings
from abc import abstractmethod
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._calendar import EXPANDING, FREQ_STEPS, block_days, check_refit, periods_per_day, refit_anchor, unit_seed
from ._cards import ModelCard, WeightsRef, resolve_model_weights, runtime_versions
from ._history import HistorySpec, history_batch, history_batches
from ._units import DAY, NO_TRANSFORM
from ._weights import offline, sha256_of
from ._worker import (
    SKIP_FIT_ERROR,
    SKIP_MISSING_TEST_PREDICTORS,
    SKIP_TOO_FEW_ROWS,
    TaskInputs,
    _issued,
    _unit_record,
    block_records,
    block_units,
    member_mean,
)
from .base import BaseModel, _usable_cpu_count

__all__ = [
    "prune_fine_tunes",
    "FoundationModel",
    "T0Model",
    "ChronosModel",
    "FINE_TUNE_DEFAULTS",
    "T0_CARD",
    "CHRONOS_CARD",
    "check_fine_tune",
]

# The research's t0 fine-tune (`zoo_tsfm/code/t0_ft.py`, `zoo_followup/code/t0_roll.py`): AdamW at lr 1e-5, batches of 16
# windows, at most 400 steps, the holdout the last 61 days of windows, evaluated every 25 steps, a stop after 4 evaluations
# without a gain, gradients clipped at norm 1; `window` is which target days train ("expanding": every one from the frame's
# first with a whole context, or the newest k days).
FINE_TUNE_DEFAULTS: Dict[str, Any] = {
    "lr": 1e-5,
    "max_steps": 400,
    "batch_size": 16,
    "holdout_days": 61,
    "eval_every": 25,
    "patience": 4,
    "grad_clip": 1.0,
    "window": EXPANDING,
}
# Keys a fine_tune dict may add, at the trainer's (and the research's) values when left out.
FINE_TUNE_OPTIONAL: Dict[str, Any] = {"weight_decay": 0.0, "min_delta": 1e-4, "eval_batch_size": 32}
# A zero-shot model's refit rule: every forecast its own, with no anchor beyond its run date.
ZERO_SHOT_REFIT = "1D"
# The float type of every foundation-model call (`runtime_fields`' `dtype`).
DTYPE = "float32"
# The most torch threads a CPU run uses by default (decision 19).
DEFAULT_THREADS = 8
_DEVICE = re.compile(r"auto|cpu|cuda(:[0-9]+)?")
# Loaded backends of this process, by (class, weight hashes, device): a loop of variants loads its family once.
_LOADED: "collections.OrderedDict[tuple, Any]" = collections.OrderedDict()
_MAX_LOADED = 2


def _whole(value: Any, what: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{what} is a whole number from {minimum}, got {value!r}")
    return int(value)


def _positive(value: Any, what: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{what} is a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{what} must be {'at least 0' if allow_zero else 'positive'}, got {value!r}")
    return value


def check_fine_tune(value: Any) -> Optional[Dict[str, Any]]:
    """A `fine_tune` dict merged over `FINE_TUNE_DEFAULTS` (None stays None), each value checked: `lr` > 0, `max_steps` >= 0,
    `batch_size`, `holdout_days`, `eval_every`, `patience` >= 1, `grad_clip` > 0 or None, `window` "expanding" or days >= 1,
    and the optional `weight_decay` >= 0, `min_delta` >= 0, `eval_batch_size` >= 1. Unknown keys raise."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"fine_tune is a dict of the fine-tune's settings, got {type(value).__name__}")
    allowed = {**FINE_TUNE_DEFAULTS, **FINE_TUNE_OPTIONAL}
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"fine_tune: unknown key(s) {unknown}; the keys are {sorted(allowed)}")
    out = {**FINE_TUNE_DEFAULTS, **dict(value)}
    out["lr"] = _positive(out["lr"], "fine_tune['lr']")
    out["max_steps"] = _whole(out["max_steps"], "fine_tune['max_steps']", 0)
    for key in ("batch_size", "holdout_days", "eval_every", "patience"):
        out[key] = _whole(out[key], f"fine_tune[{key!r}]", 1)
    if out["grad_clip"] is not None:
        out["grad_clip"] = _positive(out["grad_clip"], "fine_tune['grad_clip'] (None clips nothing)")
    if out["window"] != EXPANDING:
        out["window"] = _whole(out["window"], "fine_tune['window'] (or 'expanding')", 1)
    for key in ("weight_decay", "min_delta"):
        if key in out:
            out[key] = _positive(out[key], f"fine_tune[{key!r}]", allow_zero=True)
    if "eval_batch_size" in out:
        out["eval_batch_size"] = _whole(out["eval_batch_size"], "fine_tune['eval_batch_size']", 1)
    return out


class _Internal:
    """A BaseModel attribute a foundation model sets itself (`predictors`, `training_window`, `scaler`, `refit`, ...): stored
    under a private name, so it is no parameter a pipeline saves (`ModelPipeline._serialize_component` writes the public
    attributes), while BaseModel's validation in `__setattr__` still runs."""

    def __set_name__(self, owner: type, name: str) -> None:
        self.slot = f"_fm_{name}"
        self.name = name

    def __get__(self, obj: Any, owner: Optional[type] = None) -> Any:
        if obj is None:
            return self
        try:
            return obj.__dict__[self.slot]
        except KeyError:
            raise AttributeError(self.name) from None

    def __set__(self, obj: Any, value: Any) -> None:
        obj.__dict__[self.slot] = value


def _local_date(stamp: pd.Timestamp, tz: Any) -> datetime.date:
    return (stamp if tz is None else stamp.tz_convert(tz)).date()


def _created_at() -> str:
    """The wall-clock time a checkpoint's sidecar records (UTC, seconds): the one wall-clock read of this module, provenance
    only (nothing compares it)."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _digest(parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


class FoundationModel(BaseModel):
    """A pretrained sequence model of the target's history (see the module's docstring), the base of `T0Model`,
    `ChronosModel` and 6j's families. A family defines `_card`, `_load_backend(resolved, device)` (importing its package
    there, never at module level) and, where it differs, `_takes_known`, `_takes_past` (False refuses those covariates when the
    model is built: a univariate family), `_accepts_missing` (the backend takes NaN in a context) and `_can_fine_tune`.

    Args:
        context: Local days of the target's history a forecast reads, T = context x periods per day steps.
        known_ahead: Base column names (`"load_forecast"`, never a shifted one) read over the context and the steps ahead,
            which must be known through the end of the target day at the issue time.
        past: Base column names read over the context only, NaN after their own newest known step.
        batch_size: Items per backend call; every call is padded to it (and a zero-shot call holds the issue days of a fixed
            calendar of `batch_size` days). Compared on resume.
        device: "auto", "cpu", "cuda" or "cuda:N".
        weights: A file or directory of weights used as given (hashed, recorded, warned about when not the card's), or
            `"<repo>@<revision>"` of the cache; None for the card's, fetched by `epf weights fetch`.
        fine_tune_every: None (zero-shot), or a refit rule (`"12MS"` yearly, `"1MS"` monthly, `"kD"`) whose anchors fine-tune.
        fine_tune: The fine-tune's settings (`FINE_TUNE_DEFAULTS`, merged over them).
        seeds: Fine-tunes per anchor and horizon, their medians averaged (a zero-shot model runs once whatever it says).
        random_state: The base seed of every fine-tune (`unit_seed`).
        newest_target: `"d-1"` (the target's context and a fine-tune's targets end at D-1) or `"published"` (where its rule
            says at the issue time).
        name: The model's name.
    """

    _unit_kind = DAY
    _engine_ok = False
    # A foundation model's target is the series itself: no transform (the manifest's `target_transform`).
    target_transform = NO_TRANSFORM
    # Which inputs the family takes, and whether its backend takes NaN in a context (`HistorySpec.accepts_missing`).
    _takes_known = True
    _takes_past = True
    _accepts_missing = True
    _can_fine_tune = False
    # Whether the family runs on torch (its device, TF32 and threads scoped and recorded); a test's numpy fake doesn't.
    _uses_torch = True
    # The most context steps the family's backend reads (None: any): a longer context is cut to its newest steps by the
    # package, which a run warns about and the manifest's runtime records (`context_steps`).
    _max_context_steps: Optional[int] = None
    _default_name = "foundation"

    # BaseModel's attributes that a foundation model sets itself, kept out of its saved parameters (`_Internal`).
    predictors = _Internal()
    training_window = _Internal()
    scaler = _Internal()
    refit = _Internal()
    binary_mask = _Internal()
    normalise_by = _Internal()
    bounds = _Internal()
    zero_when = _Internal()

    def __init__(
        self,
        context: int = 364,
        known_ahead: Sequence[str] = (),
        past: Sequence[str] = (),
        batch_size: int = 64,
        device: str = "auto",
        weights: Optional[str] = None,
        fine_tune_every: Optional[str] = None,
        fine_tune: Optional[Dict[str, Any]] = None,
        seeds: int = 1,
        random_state: int = 0,
        newest_target: str = "d-1",
        name: Optional[str] = None,
    ):
        self.context = context
        self.known_ahead = known_ahead
        self.past = past
        self.batch_size = batch_size
        self.device = device
        self.weights = weights
        self.fine_tune_every = fine_tune_every
        self.fine_tune = fine_tune
        self.seeds = seeds
        self.random_state = random_state
        super().__init__(
            [],
            self._window(),
            name=self._default_name if name is None else name,
            newest_target=newest_target,
            refit=self._rule(),
        )
        self.__dict__["_fm_ready"] = True
        self._history_rules = None
        # Fine-tunes this model ran (a resume that loads its stored ones runs none) and checkpoints it loaded.
        self._fine_tunes = 0
        self._checkpoints_loaded = 0
        # Each fine-tune's windows by "<anchor>_h<horizon>" (kept, dropped and why) and the key its checkpoints went under.
        self._window_counts: Dict[str, Any] = {}
        self._fit_key_used: Optional[str] = None
        self._run_counts = (0, 0)

    # --- parameters --------------------------------------------------------------------------------------------------

    def __setattr__(self, key: str, value: Any) -> None:
        cls = type(self).__name__
        if key == "context":
            value = _whole(value, f"{cls}: context (local days)", 1)
        elif key in ("known_ahead", "past"):
            if isinstance(value, str):
                raise ValueError(f"{cls}: {key} is a list of column names, got the string {value!r}: write [{value!r}]")
            value = [] if value is None else list(value)
            if any(not isinstance(name, str) or not name for name in value):
                raise ValueError(f"{cls}: {key} holds column names, got {value!r}")
            takes = self._takes_known if key == "known_ahead" else self._takes_past
            if value and not takes:
                raise ValueError(
                    f"{cls} is univariate: its backend reads the target's history alone, so it takes no {key} ({value})"
                )
        elif key == "batch_size":
            value = _whole(value, f"{cls}: batch_size", 1)
        elif key == "device":
            if not isinstance(value, str) or not _DEVICE.fullmatch(value):
                raise ValueError(f"{cls}: device must be 'auto', 'cpu', 'cuda' or 'cuda:N', got {value!r}")
        elif key == "weights":
            value = None if value is None else os.fspath(value)
        elif key == "fine_tune_every":
            if value is not None:
                if value == "auto" or check_refit(value) is None:
                    raise ValueError(f"{cls}: fine_tune_every is None or a rule 'kD' or 'kMS', got {value!r}")
                if not self._can_fine_tune:
                    raise ValueError(f"{cls} doesn't fine-tune: fine_tune_every must be None (zero-shot)")
        elif key == "fine_tune":
            value = check_fine_tune(value)
        elif key == "seeds":
            value = _whole(value, f"{cls}: seeds", 1)
        elif key == "random_state":
            value = _whole(value, f"{cls}: random_state", 0)
        super().__setattr__(key, value)
        if key in ("context", "fine_tune_every", "fine_tune") and self.__dict__.get("_fm_ready"):
            super().__setattr__("training_window", self._window())
            super().__setattr__("refit", self._rule())

    def _settings(self) -> Dict[str, Any]:
        """The fine-tune's settings: `fine_tune` merged over the defaults."""
        return check_fine_tune(self.fine_tune if self.fine_tune is not None else {})

    def _window(self) -> Any:
        """The days of rows a run reads before an issue day (BaseModel's `training_window`): the context for zero-shot, with a
        fine-tune its windows' days and their contexts, or every row from the frame's first date ("expanding")."""
        if getattr(self, "fine_tune_every", None) is None:
            return self.context
        window = self._settings()["window"]
        return EXPANDING if window == EXPANDING else self.context + int(window)

    def _rule(self) -> str:
        every = getattr(self, "fine_tune_every", None)
        return ZERO_SHOT_REFIT if every is None else every

    @property
    def _call_rule(self) -> str:
        """The calendar a call's issue days come from: the fine-tune's blocks, or for zero-shot `batch_size` days."""
        return self.fine_tune_every if self.fine_tune_every is not None else f"{self.batch_size}D"

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        """What changes a forecast, compared on resume (with the runtime and the weights' hashes, which the manifest compares
        on its own: not the path or the device asked for)."""
        return {
            "context": self.context,
            "known_ahead": list(self.known_ahead),
            "past": list(self.past),
            "batch_size": self.batch_size,
            "fine_tune_every": self.fine_tune_every,
            "fine_tune": self.fine_tune,
            "seeds": self.seeds,
            "random_state": self.random_state,
        }

    def _history_spec(self, target: str) -> HistorySpec:
        return HistorySpec(self.context, self.known_ahead, self.past, target, self._accepts_missing, self.newest_target)

    def _co_batch_items(self, run_day: Any) -> List[datetime.date]:
        """The other issue days whose items a call that forecasts `run_day` may hold (`verify`'s co-batch check): the other days
        of its calendar block (`_call_rule`)."""
        day = pd.Timestamp(run_day).date()
        return [other for other in block_days(day, self._call_rule) if other != day]

    # A foundation model fits over history batches, never over rows: BaseModel's and a day unit's row methods never run.
    def _fit_predict(self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray):
        raise RuntimeError(f"{type(self).__name__} forecasts from history batches, not rows")

    def _fit_days(self, x: np.ndarray, y: np.ndarray, y_mask: np.ndarray):
        raise RuntimeError(f"{type(self).__name__} forecasts from history batches, not rows")

    def _predict_days(self, fitted: Any, rows: np.ndarray):
        raise RuntimeError(f"{type(self).__name__} forecasts from history batches, not rows")

    # --- runtime -----------------------------------------------------------------------------------------------------

    def _threads(self) -> int:
        """Torch threads of this process's calls (decision 19): `THREADS_PER_PROCESS`, else min(8, usable CPUs)."""
        given = os.environ.get("THREADS_PER_PROCESS")
        return max(1, int(given)) if given else max(1, min(DEFAULT_THREADS, _usable_cpu_count()))

    def _runtime_fields(self) -> Dict[str, Any]:
        """torch's version, the device type, dtype, TF32 (off), the thread count and the device's kernels
        (`_torch.runtime.runtime_fields`, without the trainer's stack), all compared on resume but the device's name."""
        if not self._uses_torch:
            fields = {"device": "cpu", "dtype": DTYPE, "threads_per_fit": self._threads()}
        else:
            from ._torch.runtime import runtime_fields

            fields = runtime_fields(
                device=self.device, dtype=DTYPE, tf32=False, threads_per_fit=self._threads(), stack="loop"
            )
            fields.pop("stack", None)
            fields.pop("pad_to", None)
        if self._max_context_steps is not None:
            # The context steps the backend reads, which a context beyond its limit doesn't change.
            fields["context_steps"] = min(self.context * self._periods_per_day, self._max_context_steps)
        return fields

    @abstractmethod
    def _load_backend(self, resolved: Any, device: Any, options: Mapping[str, Any]) -> Any:
        """The family's `_fm.Backend` on `device`, from `resolved` (the card's files, or None for a card without weights) with
        the load options `options` (`_backend_options`), importing the family's package here and nowhere else."""

    def _backend_options(self) -> Dict[str, Any]:
        """The model's settings a backend is built with (plain values): part of the key a loaded backend is shared under, so
        two models of a family share one only when these are equal. Empty here."""
        return {}

    def _predict_options(self) -> Dict[str, Any]:
        """The model's settings every backend call takes (`Backend.predict(call, options)`), such as a family's averaging or
        clipping switches. Empty here."""
        return {}

    @contextlib.contextmanager
    def _scope(self) -> Any:
        """The device of this run's calls, inside torch's scope (TF32 off, the thread count) for a torch family."""
        if not self._uses_torch:
            yield "cpu"
            return
        from ._torch.runtime import torch_scope

        with torch_scope(self.device, tf32=False, threads=self._threads()) as device:
            yield device

    def _backend(self, device: Any) -> Any:
        """This family's backend on `device`, loaded once per process from the weights the pre-flight resolved, offline."""
        resolved = resolve_model_weights(self)
        files = None if resolved is None else tuple(sorted(resolved.files.items()))
        options = self._backend_options()
        key = (type(self).__module__, type(self).__qualname__, files, str(device), json.dumps(options, sort_keys=True))
        backend = _LOADED.get(key)
        if backend is None:
            with offline():
                backend = self._load_backend(resolved, device, options)
            _LOADED[key] = backend
            while len(_LOADED) > _MAX_LOADED:
                _LOADED.popitem(last=False)
        _LOADED.move_to_end(key)
        return backend

    # --- the run -----------------------------------------------------------------------------------------------------

    def _check_inputs(
        self,
        data,
        target,
        horizon,
        freq,
        information_set=None,
        test_start=None,
        test_end=None,
        plan=None,
        availability="raise",
        windows=None,
        cut_rules=None,
        horizons=None,
        weekdays=None,
        stacklevel=4,
        shared=None,
    ):
        """BaseModel's checks (its series registered as inputs, a late one refused or listed), then the rules the series are cut
        by, kept for the run."""
        expanded = super()._check_inputs(
            data,
            target,
            horizon,
            freq,
            information_set,
            test_start,
            test_end,
            plan=plan,
            availability=availability,
            windows=windows,
            cut_rules=cut_rules,
            horizons=horizons,
            weekdays=weekdays,
            stacklevel=stacklevel + 1,
            shared=shared,
        )
        self._history_rules = self._cut_rules(information_set, cut_rules)
        steps = self.context * periods_per_day(freq)
        if self._max_context_steps is not None and steps > self._max_context_steps:
            read = self._max_context_steps / periods_per_day(freq)
            warnings.warn(
                f"{self.name}: context={self.context} days of {freq} periods is {steps} steps, and {type(self).__name__} reads "
                f"at most {self._max_context_steps}: the package reads the newest {self._max_context_steps} steps ({read:.1f} "
                "days), which the manifest records as context_steps",
                UserWarning,
                stacklevel=stacklevel,
            )
        return expanded

    def _call_jobs(self, tasks: Iterable[tuple], first_day: datetime.date) -> Dict[datetime.date, List[tuple]]:
        """The run's calls: the tasks of each block of `_call_rule` (a fine-tune's anchor, or a zero-shot block of
        `batch_size` days), in the order of `tasks`."""
        rule = self._call_rule
        jobs: Dict[datetime.date, List[tuple]] = {}
        for task in tasks:
            jobs.setdefault(refit_anchor(first_day + datetime.timedelta(days=task[2]), rule), []).append(task)
        return jobs

    def _run_tasks(self, tasks: List[tuple], state: Any, sink: Any) -> None:
        """Every call of the run in this process (decision 19): a block's issue days that the frame has, read through the
        history accessor, forecast by the backend (after the anchor's fine-tunes), and each block's records through `sink`."""
        from rich.progress import Progress

        hour_arrays = self._build_hour_arrays(max(state.horizons))
        expanded, config = self._worker_inputs(state, 1)
        inputs = TaskInputs(hour_arrays, expanded, config, self)
        rows = set(np.concatenate([arrays["days"] for arrays in hour_arrays.values()]).tolist())
        first_ordinal = config["refit"]["first_ordinal"]
        store = self._store(state)
        self._window_counts = {}
        self._run_counts = (self._fine_tunes, self._checkpoints_loaded)
        jobs = self._call_jobs(tasks, state.first_day)
        done = state.n_days - len({task[2] for task in tasks})
        with self._scope() as device, Progress() as progress:
            bar = progress.add_task(f"[cyan]{self.name}", total=state.n_days, completed=done)
            for anchor, job in jobs.items():
                days = [day for day in block_days(anchor, self._call_rule) if day.toordinal() - first_ordinal in rows]
                ran = self._run_block_calls(anchor, days, block_units(config, job), inputs, state.data, device, store)
                results, skipped = block_records(config, ran)
                self._collect(state, results, skipped, sink)
                progress.advance(bar, len({task[2] for task in job}))
        self._record_fine_tunes(state, store)

    def _run_block_calls(
        self,
        anchor: datetime.date,
        days: List[datetime.date],
        units: Dict[Any, List[tuple]],
        inputs: TaskInputs,
        frame: pd.DataFrame,
        device: Any,
        store: Optional["_Checkpoints"],
    ) -> List[List[Tuple[tuple, Any, int]]]:
        """(task, record or skip reason, member) of each unit of one call block."""
        spec = self._history_spec(self._target)
        horizons = sorted({unit.horizon for unit in units})
        batches = history_batches(spec, frame, days, self._history_rules, self._freq, horizons) if days else {}
        backend = None
        ran = []
        if self.fine_tune_every is None:
            forecasts: Dict[int, Dict] = {}
            for horizon in horizons:
                batch = batches.get(horizon)
                if batch is not None and any(not item.missing for item in batch.items):
                    backend = backend if backend is not None else self._backend(device)
                    backend.reset()
                    forecasts[horizon] = self._forecast(batch, backend)
                else:
                    forecasts[horizon] = {}
            for unit, tasks in units.items():
                ran.append(self._outcomes(inputs, unit, tasks, forecasts[unit.horizon]))
            return ran
        for unit, tasks in units.items():
            batch = batches.get(unit.horizon)
            if batch is None or not any(not item.missing for item in batch.items):
                ran.append(self._outcomes(inputs, unit, tasks, {}))
                continue
            backend = backend if backend is not None else self._backend(device)
            per_seed, skip = self._fine_tuned_forecasts(unit, batch, frame, spec, backend, store)
            if skip is not None:
                ran.append([(task, skip, unit.member) for task in tasks])
                continue
            keys = set.intersection(*(set(seed) for seed in per_seed))
            mean = {key: member_mean([seed[key] for seed in per_seed]) for key in keys}
            ran.append(self._outcomes(inputs, unit, tasks, mean))
        return ran

    def _forecast(self, batch: Any, backend: Any) -> Dict[Tuple[datetime.date, int], float]:
        """{(issue day, period): the point forecast} of every item of `batch` the backend can take, in calls of `batch_size`."""
        complete = [i for i, item in enumerate(batch.items) if not item.missing]
        point = backend.point_index
        options = self._predict_options()
        out: Dict[Tuple[datetime.date, int], float] = {}
        for call in batch.batches(self.batch_size, complete):
            quantiles = np.asarray(backend.predict(call, options), dtype=np.float64)
            expected = (self.batch_size, call.steps_ahead, len(backend.levels))
            if quantiles.shape != expected:
                raise RuntimeError(
                    f"{type(backend).__name__}.predict returned shape {quantiles.shape}, not {expected} (batch, steps, levels)"
                )
            values = quantiles[:, :, point]
            for row, i in enumerate(call.items):
                item = batch.items[i]
                for step, period in enumerate(item.slot_periods):
                    if period >= 0:
                        out[(item.issue_day, int(period))] = float(values[row, step])
        return out

    def _outcomes(
        self, inputs: TaskInputs, unit: Any, tasks: List[tuple], forecasts: Mapping[Tuple[datetime.date, int], float]
    ) -> List[Tuple[tuple, Any, int]]:
        """Each task's record from `forecasts`, or its skip: no issue row, a day the backend couldn't take
        (`missing_test_predictors`), a forecast that isn't finite (`fit_error`)."""
        outcomes, issued = _issued(inputs, tasks)
        start = datetime.date.fromisoformat(inputs.config["test_start"])
        for task, test in issued:
            value = forecasts.get((start + datetime.timedelta(days=task[2]), task[0]))
            if value is None:
                outcomes.append((task, SKIP_MISSING_TEST_PREDICTORS))
            elif not math.isfinite(value):
                outcomes.append((task, f"{SKIP_FIT_ERROR}: the backend's forecast is {value}"))
            else:
                outcomes.append((task, _unit_record(inputs, unit, task, test, value, [], None)))
        return [(task, outcome, unit.member) for task, outcome in outcomes]

    # --- fine-tunes --------------------------------------------------------------------------------------------------

    def _store(self, state: Any) -> Optional["_Checkpoints"]:
        """Where this run's fine-tunes are stored: `<save_dir>/<stem>.fits/` next to the results `<stem>.jsonl` (the model's name
        in lower case, spaces as underscores) and their manifest; None without a store or a fine-tune."""
        if self.fine_tune_every is None or not state.store:
            return None
        results = Path(state.store.path)
        return _Checkpoints(results.with_name(results.stem + ".fits"))

    def _record_fine_tunes(self, state: Any, store: Optional["_Checkpoints"]) -> None:
        """This run's fine-tunes into the manifest's provenance (`fine_tunes`): the key its checkpoints are stored under (what
        `prune_fine_tunes` keeps), how many it ran and loaded, and each anchor's windows (kept, dropped and why)."""
        if store is None or state.manifest is None or not self._window_counts:
            return
        from ..results.manifest import manifest_path, record_fine_tunes

        record_fine_tunes(
            manifest_path(state.store.path),
            state.manifest,
            {
                "key": self._fit_key_used,
                "fine_tuned": self._fine_tunes - self._run_counts[0],
                "loaded": self._checkpoints_loaded - self._run_counts[1],
                "windows": dict(sorted(self._window_counts.items())),
            },
        )

    def _anchor_cut(
        self, anchor: datetime.date, frame: pd.DataFrame, spec: HistorySpec
    ) -> Tuple[pd.Timestamp, datetime.date]:
        """(the newest target step a fine-tune at `anchor` may read, its newest whole local date): the target's cut of a forecast
        issued on the anchor at horizon 1 (`newest_target`'s: D-1's last step under "d-1", where the rule says under
        "published", at most the anchor's last step), and the date of the newest day it holds whole."""
        (item,) = history_batch(spec, frame, [anchor], self._history_rules, self._freq, 1, arrays=False).items
        cut = item.newest[spec.target]
        after = cut + FREQ_STEPS[self._freq]
        return cut, _local_date(after, frame.index.tz) - datetime.timedelta(days=1)

    def _fine_tune_data(
        self, unit: Any, frame: pd.DataFrame, spec: HistorySpec
    ) -> Tuple[Any, str, pd.Timestamp, Dict[str, Any]]:
        """(the windows of the unit's fine-tune, their digest, the cut instant, their counts): each target day up to the
        anchor's cut (every one from the frame's first, or the newest `window` days), read as the forecast issued `horizon`
        days before it. A window is dropped where its context starts before the frame's first row, where the backend couldn't
        forecast from it (`HistoryItem.missing`: a series with no value in the context, a known-ahead value missing ahead, any
        gap for a backend that takes none) and where its target day has no target; a missing value the backend takes (a gap
        in a context, masked) keeps it. The counts (`candidates`, `kept`, `with_gaps`, `dropped` by reason) warn where data
        dropped any. Asserted: no window's target is after the cut."""
        from ._fm import FineTuneData

        horizon, anchor = unit.horizon, unit.anchor
        cut, cut_date = self._anchor_cut(anchor, frame, spec)
        tz = frame.index.tz
        first_local = _local_date(frame.index[0], tz)
        window = self._settings()["window"]
        first = first_local + datetime.timedelta(days=self.context) if window == EXPANDING else cut_date
        if window != EXPANDING:
            first = cut_date - datetime.timedelta(days=int(window) - 1)
        targets = [first + datetime.timedelta(days=i) for i in range((cut_date - first).days + 1)]
        issue_days = [day - datetime.timedelta(days=horizon) for day in targets]
        items: List[Any] = []
        ys: List[np.ndarray] = []
        dropped: collections.Counter = collections.Counter()
        with_gaps = 0
        candidates = 0
        if issue_days:
            batch = history_batch(spec, frame, issue_days, self._history_rules, self._freq, horizon)
            candidates = len(batch.items)
            step = FREQ_STEPS[self._freq]
            series = frame[spec.target]
            utc = series.index.tz_convert("UTC") if tz is not None else series.index
            values = pd.Series(series.to_numpy(dtype=np.float64, na_value=np.nan), index=utc)
            for item in batch.items:
                if item.first_step < frame.index[0]:
                    dropped[DROPPED_BEFORE_FRAME] += 1
                    continue
                if item.missing:
                    dropped[f"missing values of {', '.join(item.missing)}"] += 1
                    continue
                first_ahead = item.first_step + len(item.target) * step
                instants = pd.date_range(first_ahead, periods=item.steps_ahead, freq=step)
                if tz is not None:
                    instants = instants.tz_convert("UTC")
                y = values.reindex(instants).to_numpy(dtype=np.float64).copy()
                y[np.asarray(item.slot_periods) < 0] = np.nan
                if not np.isfinite(y).any():
                    dropped[DROPPED_NO_TARGET] += 1
                    continue
                last = instants[np.flatnonzero(np.isfinite(y))[-1]]
                if (last.tz_convert(cut.tz) if tz is not None else last) > cut:
                    raise AssertionError(
                        f"{self.name}: the fine-tune at {anchor} (horizon {horizon}) would read a target at {last}, after its "
                        f"cut {cut}: a fine-tune may only read targets known at the anchor's issue time"
                    )
                y.setflags(write=False)
                with_gaps += any(gap[0] for gap in item.gaps.values())
                items.append(item)
                ys.append(y)
        dates = np.array([item.target_day for item in items], dtype="datetime64[D]")
        if len(dates) and dates.max() > np.datetime64(cut_date, "D"):
            raise AssertionError(
                f"{self.name}: a window of the fine-tune at {anchor} has its target on {dates.max()}, after its cut {cut_date}"
            )
        counts = {
            "candidates": candidates,
            "kept": len(items),
            "with_gaps": with_gaps,
            "dropped": dict(sorted(dropped.items())),
        }
        by_data = {reason: n for reason, n in dropped.items() if reason != DROPPED_BEFORE_FRAME}
        if by_data:
            shown = "; ".join(f"{n} for {reason}" for reason, n in sorted(by_data.items()))
            warnings.warn(
                f"{self.name}: the fine-tune at {anchor} (horizon {horizon}) keeps {len(items)} of {candidates} windows: "
                f"dropped {shown}"
                + (f"; {dropped[DROPPED_BEFORE_FRAME]} start before the frame" if dropped[DROPPED_BEFORE_FRAME] else "")
                + (f"; {with_gaps} kept hold gaps the backend masks" if with_gaps else ""),
                UserWarning,
                stacklevel=2,
            )
        data = FineTuneData(tuple(items), tuple(ys), dates, cut_date, tuple(spec.past), tuple(spec.known_ahead))
        digest = hashlib.sha256()
        digest.update(dates.astype("int64").tobytes())
        for item, y in zip(items, ys):
            for array in (
                item.target,
                *(item.past[n] for n in spec.past),
                *(item.known[n] for n in spec.known_ahead),
                y,
            ):
                digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
        return data, digest.hexdigest(), cut, counts

    def _fit_key(self, spec: HistorySpec) -> str:
        """The digest of everything that decides a fine-tune but its windows: the family, its series and context, the
        fine-tune's settings and base seed, the base weights' hashes, the runtime (packages, device type, dtype, TF32, threads)
        and the rules the series are cut by."""
        resolved = resolve_model_weights(self)
        runtime = {key: value for key, value in self._runtime_fields().items() if key != "device_name"}
        rules = None if self._history_rules is None else self._history_rules_of(spec, self._history_rules, self._freq)
        return _digest(
            {
                "class": f"{type(self).__module__}:{type(self).__qualname__}",
                "spec": [spec.context, list(spec.known_ahead), list(spec.past), spec.target, spec.newest_target],
                "freq": self._freq,
                "fine_tune": self._settings(),
                "random_state": self.random_state,
                "weights": None if resolved is None else dict(sorted(resolved.files.items())),
                "packages": runtime_versions(self._card) if self._card is not None else {},
                "runtime": runtime,
                "rules": rules,
            }
        )

    def _fine_tuned_forecasts(
        self,
        unit: Any,
        batch: Any,
        frame: pd.DataFrame,
        spec: HistorySpec,
        backend: Any,
        store: Optional["_Checkpoints"],
    ) -> Tuple[List[Dict], Optional[str]]:
        """Each seed's forecasts of the block from its fine-tune at the unit's anchor (stored, or run and stored), or the skip
        reason of every task of the unit when there are too few windows."""
        data, data_digest, cut, counts = self._fine_tune_data(unit, frame, spec)
        self._window_counts[f"{unit.anchor.isoformat()}_h{unit.horizon}"] = counts
        settings = self._settings()
        dates = data.dates
        if len(dates) == 0 or (dates <= dates.max() - np.timedelta64(settings["holdout_days"], "D")).sum() == 0:
            return [], (
                f"{SKIP_TOO_FEW_ROWS}: the fine-tune at {unit.anchor} has {len(dates)} window(s), none before its "
                f"{settings['holdout_days']}-day holdout"
            )
        key = None
        if store is not None:
            key = self._fit_key(spec)
            self._fit_key_used = key
        per_seed = []
        for index in range(self.seeds):
            seed = unit_seed(self.random_state, unit, index)
            path = None if store is None else store.find(unit, index, key, data_digest, seed)
            if path is not None:
                backend.load_state(path)
                self._checkpoints_loaded += 1
            else:
                backend.reset()
                outcome = backend.fine_tune(data, settings, seed)
                self._fine_tunes += 1
                if store is not None:
                    store.save(backend, unit, index, seed, key, data_digest, data, cut, outcome, counts)
            per_seed.append(self._forecast(batch, backend))
        backend.reset()
        return per_seed, None


# Why a window of a fine-tune is dropped where no data is to blame: its context starts before the frame's first row.
DROPPED_BEFORE_FRAME = "a context starting before the frame"
DROPPED_NO_TARGET = "no target on the target day"


class _Checkpoints:
    """A run's stored fine-tunes (decision 13). Per (anchor, horizon, seed index, key) a sidecar `<stem>.json` (anchor, horizon,
    seed, the key of what decides it, the file it names with its sha256 and size, the data end, the cut, the best step, the
    windows and their digest and counts, the trainer's statistics) and the checkpoint it names, `<stem>.<sha256 prefix>.ckpt`,
    a name of its content's. A save writes the checkpoint under a new name, then replaces the sidecar, then removes the stem's
    other checkpoints (the refit's predecessor, or a crash's leftover), so a crash anywhere leaves the old pair or the new one
    whole. A pair that doesn't match (no file, another sha256, key, seed or windows) is fine-tuned again, with a warning.
    Leftovers of an interrupted write (`*.partial`) are removed when the store opens: the results store's lock keeps two runs
    of one model from writing here at once."""

    def __init__(self, directory: Path):
        self.directory = directory
        if directory.is_dir():
            for leftover in directory.glob("*.partial"):
                leftover.unlink(missing_ok=True)

    @staticmethod
    def stem(unit: Any, index: int, key: str) -> str:
        return f"{unit.anchor.isoformat()}_h{unit.horizon}_s{index}_{key[:16]}"

    def _refit(self, sidecar: Path, why: str) -> None:
        warnings.warn(
            f"the stored fine-tune {sidecar.stem} {why}: fine-tuning again, which on a GPU gives other numbers than any "
            "forecast already stored from it",
            UserWarning,
            stacklevel=3,
        )

    def find(self, unit: Any, index: int, key: str, data_digest: str, seed: int) -> Optional[Path]:
        """The checkpoint of this unit, seed index, key and seed fine-tuned on these windows, None (fine-tune again) when there
        is none, or when what is stored doesn't match, which warns."""
        sidecar = self.directory / f"{self.stem(unit, index, key)}.json"
        if not sidecar.is_file():
            return None
        try:
            record = json.loads(sidecar.read_text())
            path = self.directory / record["file"]
        except (ValueError, KeyError, TypeError):
            self._refit(sidecar, "has a sidecar that can't be read")
            return None
        if not path.is_file():
            self._refit(sidecar, f"is gone ({path.name}, which its sidecar names)")
            return None
        if sha256_of(path) != record.get("sha256"):
            self._refit(sidecar, f"changed after it was stored ({path.name} has another sha256 than its sidecar's)")
            return None
        if record.get("key") != key or record.get("seed") != int(seed):
            self._refit(sidecar, "was made with another key or seed")
            return None
        if record.get("data_digest") != data_digest:
            self._refit(sidecar, "was made on other windows (the data changed since)")
            return None
        return path

    def save(
        self,
        backend: Any,
        unit: Any,
        index: int,
        seed: int,
        key: str,
        data_digest: str,
        data: Any,
        cut: pd.Timestamp,
        outcome: Any,
        counts: Optional[Dict[str, Any]] = None,
    ) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = self.stem(unit, index, key)
        partial = self.directory / f"{stem}.ckpt.partial"
        backend.save_state(partial)
        digest = sha256_of(partial, cached=False)
        path = self.directory / f"{stem}.{digest[:16]}.ckpt"
        os.replace(partial, path)  # a name of its own: the pair on file stays whole until the sidecar names this file
        record = {
            "anchor": unit.anchor.isoformat(),
            "horizon": int(unit.horizon),
            "seed_index": int(index),
            "seed": int(seed),
            "key": key,
            "file": path.name,
            "sha256": digest,
            "bytes": path.stat().st_size,
            "data_end": str(outcome.data_end),
            "cut": str(data.cut),
            "cut_step": str(cut),
            "best_step": int(outcome.best_step),
            "windows": len(data),
            "window_counts": counts,
            "data_digest": data_digest,
            "stats": json.loads(json.dumps(dict(outcome.stats), default=str)),
            "created_at": _created_at(),
        }
        written = self.directory / f"{stem}.json.partial"
        written.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(written, self.directory / f"{stem}.json")
        for other in self.directory.glob(f"{stem}.*.ckpt"):  # superseded by this refit, or left by a crash
            if other.name != path.name:
                other.unlink(missing_ok=True)
        return path


def prune_fine_tunes(save_dir: Any, name: str, keep_keys: Optional[Sequence[str]] = None) -> List[Path]:
    """Remove from the model `name`'s `<save_dir>/<stem>.fits/` what no current run reads: the stored fine-tunes (sidecar and checkpoint) whose
    key isn't in `keep_keys`, by default the key the model's last run stored under (its manifest's `fine_tunes`), checkpoint
    files no sidecar names, and leftovers of an interrupted write. Returns the paths removed. A t0 checkpoint is about 407 MB.

    Without `keep_keys` and with no key in the manifest (no run with a store fine-tuned there yet) it raises rather than
    remove everything."""
    from types import SimpleNamespace

    from ..pipelines.model_pipeline import ModelPipeline

    save_dir = Path(save_dir)
    stem = ModelPipeline.results_path(save_dir, SimpleNamespace(name=name), "jsonl").stem
    directory = save_dir / f"{stem}.fits"
    if keep_keys is None:
        manifest = save_dir / f"{stem}.manifest.json"
        entry = json.loads(manifest.read_text()).get("fine_tunes") if manifest.is_file() else None
        if not entry or not entry.get("key"):
            raise ValueError(
                f"{manifest} records no key of a run's fine-tunes, so nothing says which to keep: pass keep_keys"
            )
        keep_keys = [entry["key"]]
    keep = set(keep_keys)
    removed: List[Path] = []
    if not directory.is_dir():
        return removed
    named = set()
    for sidecar in sorted(directory.glob("*.json")):
        try:
            record = json.loads(sidecar.read_text())
        except ValueError:
            record = {}
        if record.get("key") in keep:
            named.add(record.get("file"))
            continue
        sidecar.unlink()
        removed.append(sidecar)
    for path in sorted([*directory.glob("*.ckpt"), *directory.glob("*.partial")]):
        if path.name not in named:
            path.unlink()
            removed.append(path)
    return removed


# --- the families ---------------------------------------------------------------------------------------------------------

T0_CARD = ModelCard(
    family="t0-alpha",
    extra="t0",
    licence="Apache-2.0",
    licence_url="https://huggingface.co/theforecastingcompany/t0-alpha/blob/main/LICENSE",
    commercial_use=True,
    revenue_cap=None,
    attribution=None,
    # The repository's history was squashed on 2026-10-15, so the files are pinned by their sha256 as well as the commit (the
    # hashes recorded by the first `epf weights fetch`, 2026-11-10, equal to the repository's LFS hashes).
    weights=WeightsRef(
        "theforecastingcompany/t0-alpha",
        "741927847fe02367624749bb9290b2f09f128e7d",
        {
            "config.json": "b2b545685283c579b99c774da0d7f07525683efca2b27dc4ffec99c109d5beba",
            "model.safetensors": "16c030d3fd70f06dc4238e9a8356e9b5a631d07f80f1bc76ba539991aed5897f",
        },
        bytes=406_601_742,
    ),
    runtime=("tfc-t0==0.5.0", "torch>=2.9"),
    corpus="GIFT-Eval pretraining split (real series, each ending before 2022) and synthetic corpora",
    corpus_end=date(2021, 12, 31),
    released=date(2026, 6, 9),
)

CHRONOS_CARD = ModelCard(
    family="chronos-2",
    extra="chronos",
    licence="Apache-2.0",
    licence_url="https://www.apache.org/licenses/LICENSE-2.0",
    commercial_use=True,
    revenue_cap=None,
    attribution=None,
    weights=WeightsRef(
        "amazon/chronos-2",
        "29ec3766d36d6f73f0696f85560a422f50e8498c",
        {
            "config.json": "ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031",
            "model.safetensors": "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42",
        },
        bytes=477_931_539,
    ),
    runtime=("chronos-forecasting>=2.3.2,<3", "torch>=2.9", "transformers"),
    # The paper (arXiv 2510.15821) trains on real univariate series of the Chronos and GIFT-Eval pretraining corpora (no
    # electricity prices among them) and synthetic multivariate data, and gives no end date: the verdict goes by the release.
    corpus="Chronos and GIFT-Eval pretraining corpora (real univariate series, no end date given) and synthetic data",
    corpus_end=None,
    released=date(2025, 10, 30),
)


class T0Model(FoundationModel):
    """t0-alpha (The Forecasting Company, 102M, Apache-2.0), zero-shot or fine-tuned yearly from the base weights (the
    research's monthly re-fine-tune gained 1.55 % on DE and 1.7 % on AT against none; yearly is untested). Past covariates are
    extra rows of the context, known-ahead ones its future covariates; the forecast is the median of its 19 levels. Needs the
    `[t0]` extra (`tfc-t0==0.5.0`) and `epf weights fetch T0Model`."""

    _card = T0_CARD
    _can_fine_tune = True
    _default_name = "t0"

    def __init__(
        self,
        context: int = 364,
        known_ahead: Sequence[str] = (),
        past: Sequence[str] = (),
        fine_tune_every: Optional[str] = "12MS",
        seeds: int = 3,
        fine_tune: Optional[Dict[str, Any]] = None,
        batch_size: int = 64,
        device: str = "auto",
        weights: Optional[str] = None,
        random_state: int = 0,
        newest_target: str = "d-1",
        name: str = "t0",
    ):
        super().__init__(
            context=context,
            known_ahead=known_ahead,
            past=past,
            batch_size=batch_size,
            device=device,
            weights=weights,
            fine_tune_every=fine_tune_every,
            fine_tune=dict(FINE_TUNE_DEFAULTS) if fine_tune is None else fine_tune,
            seeds=seeds,
            random_state=random_state,
            newest_target=newest_target,
            name=name,
        )

    def _load_backend(self, resolved: Any, device: Any, options: Mapping[str, Any]) -> Any:
        from ._fm.t0 import T0Backend

        return T0Backend(resolved, device)


class ChronosModel(FoundationModel):
    """Chronos-2 (Amazon, 120M, Apache-2.0), zero-shot: known-ahead covariates over the context and the steps ahead, past
    ones over the context; the forecast is its 0.5 quantile. It reads at most 8,192 steps (341 days of hours): a longer
    context is cut to its newest 8,192 steps by the package, which a run warns about. Needs the `[chronos]` extra and `epf
    weights fetch ChronosModel`."""

    _card = CHRONOS_CARD
    _default_name = "chronos2"
    # Chronos-2's `context_length` (its config.json): 341 days of hours, 85 of quarter-hours.
    _max_context_steps = 8192
    # Chronos-2 doesn't fine-tune here (its fine-tuning is out of this step): the base's fine-tune settings are no parameters
    # of this class, so a pipeline doesn't save them.
    fine_tune_every = _Internal()
    fine_tune = _Internal()
    seeds = _Internal()
    random_state = _Internal()

    def __init__(
        self,
        context: int = 364,
        known_ahead: Sequence[str] = (),
        past: Sequence[str] = (),
        batch_size: int = 64,
        device: str = "auto",
        weights: Optional[str] = None,
        newest_target: str = "d-1",
        name: str = "chronos2",
    ):
        super().__init__(
            context=context,
            known_ahead=known_ahead,
            past=past,
            batch_size=batch_size,
            device=device,
            weights=weights,
            newest_target=newest_target,
            name=name,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        return {
            "context": self.context,
            "known_ahead": list(self.known_ahead),
            "past": list(self.past),
            "batch_size": self.batch_size,
        }

    def _load_backend(self, resolved: Any, device: Any, options: Mapping[str, Any]) -> Any:
        from ._fm.chronos import ChronosBackend

        return ChronosBackend(resolved, device)
