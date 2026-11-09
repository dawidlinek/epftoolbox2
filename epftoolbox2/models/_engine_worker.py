"""The worker side of an engine group (`models/_engine.py`): one pool for every model of a chunk, each job one (day or refit
block, period) holding the pending tasks of every model of the chunk on that day or block and period.

Each task runs on its model's **own** inputs (`_worker.TaskInputs`: the chunk's union arrays, which a fit reads by column name,
and the model's expanded lists, config and rebuilt model), through the model path's functions: `_run_task` for a refit=None
task, `_run_unit` for each unit of a refit block (`_worker.block_units`), whose records `_worker.block_records` makes as
`_run_block` does. Those functions select every fit's rows, mask and skips, and build every record.

**Fits from shared statistics** (steps 5i and 5j). A model whose config names a fit path (`config["fit_path"]`,
`fit_path_of`: OLS, Ridge, LassoCV and LEAR with `StandardScaler` or one of the library's VSTs, LEAR with `solver="lars"`)
hands each fit to `EngineFits` once the model path has selected it (`_worker.StatsFit`), in place of the scaler and
`_fit_predict` / `_fit_model`:

- **StandardScaler** (5i): per **span** (period, horizon and the period's rows of `training_span`, before any filter: the same
  rows, by date, in any group), one `_sums.SpanSums` over the columns the job's fits of that span read (`_job_spans`), built
  when its first fit asks and let go after its last; one per task thread, never shared between threads. Per fit,
  `SpanSums.over(ρ, its columns)` (ρ the model path's finite rows; a downdate when at most `DOWNDATE_SHARE` (τ) of the span's
  rows are left out, else a direct sum: the same bits either way), `_stats.standard_stats` with the fit's mask and
  `scale_binary`, and each issue row through `params.transform` and `params.inverse`.
- **A VST** (5j: asinh, npit, mirror-log, polynomial, `ENGINE_VSTS`) is fitted on ρ and nonlinear, so no sums of the raw rows
  can give it: per **row set** (span, ρ and the scaler's spec, `_RowSet`) the scaler's own `fit_transform` runs once on the
  union of the columns its fits read, each column with its mask flag ((column, True) transformed, (column, False) as it is: a
  column two fits flag differently is two columns of the set), and a `SpanSums` of the transformed columns and target over
  ρ alone; per fit `over(every row, its columns)` and `_stats.transformed_stats` (no map: the rows are already scaled). The
  scaler transforms column by column, so each transformed column and the target are the model path's to the bit, whatever
  else the set holds (`tests/models/test_engine_vst_cv.py`); a fit with another ρ is another set, with its own transform. An
  issue row is transformed value by value with the set's column parameters (`_RowSet.transform`), its forecast mapped back
  through the set's scaler (`inverse`).
- **CV blocks** (5j: LassoCV, LEAR `"cv"`, `ScaledStats.blocks`): the `folds` KFold blocks of ρ in row order
  (`_cv_stats.block_rows`), each block's sums from the same `SpanSums` and `_cv_stats.block_stats` with the fit's own scaler
  parameters, built for each fit of a model that needs them and never shared with another fit's statistics, so a hand-back
  only a CV model's blocks call for leaves its neighbours alone.

Then `model._fit_stats` and each issue row through `model._predict_fitted` (one row at a time).

A fit of no column, one with fewer than `_stats.MIN_ROWS` rows in ρ (or than its CV folds), one whose sums overflow float64
(`_sums.SumsOverflow`) or that can't be scaled or cut into blocks from the sums (`StatsFallback`: cancellation, overflow), a
LassoCV or LEAR fit with two near-twin columns (`near_twin`: 1 - |cos| at most their `_ENGINE_NEAR_TWIN`, 1e-5), one
`_fit_stats` declines (None: OLS's rank and conditioning guards, Ridge's n <= p and rounding budget, LassoCV's n <= p in a
fold, an estimator that isn't the one the statistics reproduce) or raises a ValueError on, and one whose active columns' Gram
is too ill-conditioned for its class (`_ENGINE_CONDITION_MAX`, LEAR) is **handed back**: the model path fits it there, on the
same values, to the bit. So is the forecast of an issue row too far from the fit's training rows (`far_issue`): beyond its
model's `_ENGINE_ISSUE_Z` training standard deviations along the eigenvectors of the fit's Gram (`_stats.issue_distance`: OLS
30, LEAR 10), or where RidgeCV's own rounding at that row exceeds its budget (`_stats.issue_rounding_ok`; with a VST, in target
units through the slope of the scaler's inverse at the forecast), since an extrapolation multiplies the last bits in which
the two paths' coefficients differ; a refit unit forecasts its other rows from statistics. Each hand-back is decided from the
fit's own rows and span, and the issue row's alone, so it is the same in any group and whichever rows a unit forecasts.

Every element of the sums is a function of the two columns' values in ρ and of the span (or row set), and everything after
them is the fit's own, run on one BLAS thread (`_one_blas_thread`: LAPACK's own blocking moves bits with the thread count), so
a model's records are the same bits alone, in any group, order or chunking, at any process, task thread or BLAS thread count,
and whatever τ. Nothing stored names the engine but the manifest's `fit_path` (compared on resume) and its provenance counts
(`EngineCounts`, never compared).

**`engine="check"`** runs as "on", and a fit whose span's hash is 0 modulo `CHECK_EVERY` (`_sampled`: period, horizon and
the span's first and last dates, nothing of the group) is also fitted on the model path, on the same values: its forecasts
are compared with the engine's (the records keep the engine's) and counted (`EngineCounts.check_dict`: fits, the largest
absolute and relative gaps, and how many chose another kink or penalty), which the manifest records as `engine_check`.
"""

import contextlib
import datetime
import hashlib
import json
import math
import numbers
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from threadpoolctl import threadpool_limits

from ..scalers.registry import make_scaler, scaler_spec
from ..scalers.standard import StandardScaler
from ..scalers.vst import AsinhScaler, MirrorLogScaler, NPITScaler, PolyScaler
from . import _worker as worker
from ._cv_stats import block_rows, block_stats
from ._lars import CV_FOLDS
from ._stats import (
    MIN_ROWS,
    RIDGE_ROUNDING_MAX,
    StatsFallback,
    issue_distance,
    issue_metric,
    issue_rounding_ok,
    near_twins,
    standard_stats,
    transformed_stats,
)
from ._sums import SpanSums, SumsOverflow
from ._worker import (
    StatsFit,
    TaskFailed,
    TaskInputs,
    _limit_native_threads,
    _limit_openmp_in_task_thread,
    _run_task,
    _run_unit,
    block_records,
    block_units,
    finite_rows,
    issue_row,
    member_rows,
    task_inputs,
    task_list,
    task_newest,
    task_place,
    unit_rows,
)

# The numeric path of the fits from shared statistics, as a manifest records it (`fit_path`, compared on resume): a later
# release that changes their numbers gets another name. None runs every fit of a group on the model path (5h's plumbing, which
# `tests/models/test_engine_groups.py` checks to the bit against engine off).
ENGINE_FIT_PATH: Optional[str] = "engine-1"

# The VSTs a row set runs the own `fit_transform` of (`_RowSet`): the library's classes exactly, whose transform is column by
# column with the parameters `(params, mean, std)` per column (`scalers.vst._VSTScaler`); a subclass may transform otherwise.
ENGINE_VSTS = (AsinhScaler, NPITScaler, MirrorLogScaler, PolyScaler)

# τ: a fit's sums are the span's totals minus the rows ρ leaves out when those are at most this share of the span's rows, else
# summed over ρ directly. Both give the same bits (`_sums`); τ decides speed alone.
DOWNDATE_SHARE = 0.5

# How far out of its training values an issue row may be for the engine to forecast it, in training standard deviations
# (`_stats.issue_distance`), where a model class sets none (`_ENGINE_ISSUE_Z`): an extrapolation multiplies the last bits of
# a coefficient, in which the engine and the model path differ, so a row beyond its model's bound is forecast on the model path.
ISSUE_Z_MAX = 10.0

# A VST's forecast is forecast on the model path where its scaler's inverse is steeper there than this many times its median
# slope at the training targets (`FAR_ISSUE`): an error in scaled units is that much larger in target units there, and the
# bounds in scaled units (`_ENGINE_ISSUE_Z`, `_ENGINE_CONDITION_MAX`) were set where the slope is the target's standard
# deviation throughout. Set by the sweep `engine_vst_cv/sweep.py` (`rules.py`): at 3 the worst kept gaps were OLS 6.3e-9,
# Ridge 3.2e-10, LEAR (BIC, AIC; asinh, Invariant, npit, poly) 4.7e-10, and at most 1.3 % of the forecasts of a training row
# went back; at 10 LEAR's reached 3.2e-9, without it OLS's 2.5e-4. Out of sample (new designs, every bound applied by this
# code): OLS 3.7e-9, LassoCV 5.4e-9, Ridge 2.5e-10, LEAR 7.6e-10.
VST_SLOPE_MAX = 3.0

# `engine="check"`: the fits also run on the model path are those whose span hashes to 0 modulo this (`_sampled`).
CHECK_EVERY = 50

# Engine against the model path per class, in target units (the step 5 spec's tolerance table): `engine="check"` warns about a
# model whose largest gap is above its class's (`tolerance_of`).
ENGINE_TOLERANCE = {"OLSModel": 1e-8, "RidgeModel": 1e-9, "LassoCVModel": 1e-8, "LEARModel": 1e-8}

# Why a fit went back to the model path (`EngineCounts.fallbacks`): no column (the model path then skips, or forecasts as it
# does); an active Gram whose condition number is above the model's `_ENGINE_CONDITION_MAX` (LEAR, `ILL_CONDITIONED`);
# fewer than `MIN_ROWS` rows in ρ, or than the model's CV folds; sums past float64's range ("overflow", also
# `StatsFallback`'s) or `StatsFallback`'s other reason ("cancellation"); two near-twin columns (`NEAR_TWIN`, below);
# `_fit_stats` returned None; `_fit_stats` raised a ValueError. And per issue row: one beyond its model's bound (`FAR_ISSUE`, counted per row; the fit forecasts its other rows).
NO_COLUMNS = "no_columns"
ILL_CONDITIONED = "ill_conditioned"
FEW_ROWS = "few_rows"
OVERFLOW = "overflow"
DECLINED = "declined"
ERROR = "error"
FAR_ISSUE = "far_issue"
# Two columns of the fit near twins (`_stats.near_twins`) for a model that sets `_ENGINE_NEAR_TWIN` (LassoCV and LEAR, 1e-5).
NEAR_TWIN = "near_twin"

# What `_engine_init` loaded into a pool's worker: the chunk's models' inputs, by their place in the chunk.
_E_MODELS: List["ModelInputs"] = []


def cv_folds(model) -> Optional[int]:
    """The number of KFold blocks a model with an inner cross-validation is scored on: its `cv` where it has one (LassoCV: a
    whole number from 2, or None for scikit-learn's 5), else `_lars.CV_FOLDS` (LEAR `"cv"`); None for anything else (a
    splitter object, whose folds aren't KFold's blocks)."""
    cv = getattr(model, "cv", None)
    if cv is None:
        return CV_FOLDS
    if isinstance(cv, bool) or not isinstance(cv, numbers.Integral) or cv < 2:
        return None
    return int(cv)


def fit_path_of(model) -> Optional[str]:
    """The numeric path of a model's fits in an engine group (`ENGINE_FIT_PATH`), or None where they are the model path's: a
    model that declares no `_ENGINE_NEEDS` or needs something other than rows and CV blocks, one that says its fits can't come
    from statistics to the model path's digits (`_ENGINE_FROM_STATS` False: LEAR with `solver="cd"`), one that needs CV blocks
    of folds that aren't KFold's (`cv_folds` None), or one whose scaler is neither exactly `StandardScaler` nor one of
    `ENGINE_VSTS` keeps the model path."""
    if ENGINE_FIT_PATH is None:
        return None
    needs = getattr(model, "_ENGINE_NEEDS", None)
    if needs is None or set(needs) - {"rows", "blocks"} or not getattr(model, "_ENGINE_FROM_STATS", True):
        return None
    if "blocks" in needs and cv_folds(model) is None:
        return None
    try:
        scaler = make_scaler(getattr(model, "scaler", "standard"))
    except (TypeError, ValueError):
        return None
    return ENGINE_FIT_PATH if type(scaler) is StandardScaler or type(scaler) in ENGINE_VSTS else None


def tolerance_of(model) -> Optional[float]:
    """The engine's tolerance against the model path for a model's class (`ENGINE_TOLERANCE`, by the library class it is or
    derives from), or None for a class the table doesn't know."""
    for klass in type(model).__mro__:
        if klass.__module__.startswith("epftoolbox2.models.") and klass.__name__ in ENGINE_TOLERANCE:
            return ENGINE_TOLERANCE[klass.__name__]
    return None


class ModelInputs(NamedTuple):
    """One model of a chunk in a worker: its name (for errors), its `TaskInputs`, and with fits from shared statistics
    (`config["fit_path"]`) its `StandardScaler`'s `scale_binary` or its VST's spec (`vst`, as JSON: a row set's key) and the
    number of CV blocks its fits need (`folds`, or None); `check` with `engine="check"`. A model with neither `scale_binary`
    nor `vst` fits every fit on the model path."""

    name: str
    inputs: TaskInputs
    scale_binary: Optional[bool] = None
    vst: Optional[str] = None
    folds: Optional[int] = None
    check: bool = False

    @property
    def from_stats(self) -> bool:
        return self.scale_binary is not None or self.vst is not None


class EngineTaskFailed(TaskFailed):
    """A task of one model of an engine job raised: `model` is the model's place in the chunk, `message` the model path's
    `TaskFailed` text (the task and the worker's traceback)."""

    def __init__(self, model: int, message: str):
        super().__init__(model, message)
        self.model = model
        self.message = message

    def __str__(self) -> str:
        return self.message


class EngineCounts:
    """What one model's fits from shared statistics did, for the manifest's provenance (never compared): `fits` made from the
    statistics (that forecast at least one row), `fallbacks` handed back to the model path by reason (fits, and issue rows for
    `FAR_ISSUE`), and the largest distance of an issue row the engine forecast from its training values (`max_issue_z`, in
    training standard deviations along the eigenvectors of its fit's Gram, `_stats.issue_distance`). With `check`
    (`engine="check"`), the fits also run on the model path (`check_fits`), the largest absolute and relative gaps of their
    forecasts (`check_abs`, `check_rel`, finite gaps), how many forecasts' gaps weren't finite (`nonfinite`: one side NaN or
    inf, the largest gap of all, which no later gap and no merge can hide), how many of the fits chose another kink or penalty
    (`other_choice`), and how many fits the engine forecast that the model path couldn't fit (`model_path_errors`: its
    ValueError, a fit_error there, so the forecast counts would differ)."""

    __slots__ = (
        "fits",
        "fallbacks",
        "max_issue_z",
        "check_fits",
        "check_abs",
        "check_rel",
        "other_choice",
        "nonfinite",
        "model_path_errors",
    )

    def __init__(self, check: bool = False):
        self.fits = 0
        self.fallbacks: Counter = Counter()
        self.max_issue_z = 0.0
        self.check_fits: Optional[int] = 0 if check else None
        self.check_abs = 0.0
        self.check_rel = 0.0
        self.other_choice = 0
        self.nonfinite = 0
        self.model_path_errors = 0

    def __bool__(self) -> bool:
        return bool(self.fits or self.fallbacks or self.check_fits)

    def add(self, other: "EngineCounts") -> None:
        self.fits += other.fits
        self.fallbacks.update(other.fallbacks)
        self.max_issue_z = max(self.max_issue_z, other.max_issue_z)
        if other.check_fits is not None:
            self.check_fits = (self.check_fits or 0) + other.check_fits
            self.check_abs = max(self.check_abs, other.check_abs)
            self.check_rel = max(self.check_rel, other.check_rel)
            self.other_choice += other.other_choice
            self.nonfinite += other.nonfinite
            self.model_path_errors += other.model_path_errors

    def issue(self, z: float) -> None:
        self.max_issue_z = max(self.max_issue_z, z)

    def checked(self, pairs: Sequence[Tuple[float, float]], other_choice: bool) -> None:
        """One fit run on both paths: its forecasts (engine, model path) per row the engine forecast."""
        self.check_fits = (self.check_fits or 0) + 1
        self.other_choice += bool(other_choice)
        for mine, theirs in pairs:
            gap = abs(mine - theirs)
            if not math.isfinite(gap):  # NaN or inf on one side: counted, never compared away
                self.nonfinite += 1
                continue
            self.check_abs = max(self.check_abs, gap)
            size = max(abs(mine), abs(theirs))
            self.check_rel = max(self.check_rel, gap / size if size > 0 else 0.0)

    def model_path_error(self) -> None:
        """A checked fit the model path raised a ValueError on (it would skip it as a fit_error) and the engine forecast."""
        self.check_fits = (self.check_fits or 0) + 1
        self.model_path_errors += 1

    def to_dict(self) -> Dict:
        return {
            "fits": int(self.fits),
            "fallbacks": {reason: int(self.fallbacks[reason]) for reason in sorted(self.fallbacks)},
            "max_issue_z": float(self.max_issue_z),
        }

    def check_dict(self) -> Optional[Dict]:
        """`engine_check` of the manifest, or None without `check`."""
        if self.check_fits is None:
            return None
        return {
            "fits": int(self.check_fits),
            "max_abs": float(self.check_abs),
            "max_rel": float(self.check_rel),
            "other_choice": int(self.other_choice),
            "nonfinite": int(self.nonfinite),
            "model_path_errors": int(self.model_path_errors),
        }


def model_inputs(hour_arrays: Dict, models: Sequence[tuple]) -> List[ModelInputs]:
    """Each model's inputs from (name, expanded lists, config, class, `_model_kwargs`), the model rebuilt as a pool's worker
    rebuilds it (`_worker.task_inputs`), all on the chunk's one set of arrays."""
    out = []
    for name, expanded, config, model_cls, model_kwargs in models:
        inputs = task_inputs(hour_arrays, expanded, config, model_cls, model_kwargs)
        scale_binary, vst, folds = None, None, None
        if config.get("fit_path") is not None:
            spec = config.get("scaler", "standard")
            scaler = make_scaler(spec)
            if type(scaler) is StandardScaler:
                scale_binary = bool(scaler.scale_binary)
            else:
                vst = json.dumps(scaler_spec(spec), sort_keys=True)
            if "blocks" in inputs.model._ENGINE_NEEDS:
                folds = cv_folds(inputs.model)
        check = bool(config.get("engine_check")) and (scale_binary is not None or vst is not None)
        out.append(ModelInputs(name, inputs, scale_binary, vst, folds, check))
    return out


def _engine_init(hour_arrays: Dict, models: Sequence[tuple]) -> None:
    """Initialiser of an engine pool: native threads limited as `_worker._pool_init` limits them, then every model's inputs."""
    _limit_native_threads()
    _engine_load(hour_arrays, models)


def _engine_load(hour_arrays: Dict, models: Sequence[tuple]) -> None:
    """Load a chunk's models' inputs into this process (`_engine_init` without the thread limit, for a caller's process)."""
    global _E_MODELS
    _E_MODELS = model_inputs(hour_arrays, models)


# --- spans ------------------------------------------------------------------------------------------------------------

SpanKey = Tuple[
    int, int, int, int
]  # (period, horizon, first row, end row) of the period's arrays: a span's rows, by date
# A VST's row set: the span's key, ρ as `np.packbits` of the span's finite rows, and the scaler's spec as JSON.
RowSetKey = Tuple[int, int, int, int, bytes, str]


def _item_spans(what: tuple, unit_tasks: Optional[List[tuple]], inputs: TaskInputs) -> List[Tuple[SpanKey, List[int]]]:
    """The spans of a task's (refit=None: each member's) or a unit's fits, with the columns each reads, by the model path's own
    selection (`_fit_one_numpy`, `_fit_unit`): what a job's `SpanSums` must hold. A task with no issue row has none."""
    cfg = inputs.config
    if unit_tasks is None:
        hour, hz, day_in_test = what
        arrays = inputs.hour_arrays[hour]
        days = arrays["days"]
        day = cfg["offset"] + day_in_test
        if issue_row(days, day) is None:
            return []
        _, _, position = task_place(cfg, day_in_test)
        newest = task_newest(cfg, hour, position)
        _, predictors, _ = task_list(cfg, inputs.expanded, hour, hz, position)
        members = cfg.get("members")
        windows = [cfg["training_window"]] if members is None else members
        spans = [member_rows(days, day, member, hz, newest) for member in windows]
    else:
        hour, hz = what.period, what.horizon
        arrays = inputs.hour_arrays[hour]
        spans = [unit_rows(cfg, arrays["days"], what)[0]]
        predictors = inputs.expanded[(hour, hz) if what.list_id is None else what.list_id]
    columns = [arrays["col_index"][name] for name in predictors]
    return [((hour, hz, rows.start, rows.stop), columns) for rows in spans if rows.stop > rows.start]


def _rowset_key(period: int, horizon: int, rows: slice, finite: np.ndarray, spec: str) -> RowSetKey:
    return (period, horizon, rows.start, rows.stop, np.packbits(np.asarray(finite, dtype=bool)).tobytes(), spec)


def _planned_rowset(arrays: Dict, key: SpanKey, columns: List[int], spec: str) -> RowSetKey:
    """The row set of a fit of a VST model as the job plans it: ρ the span's rows whose target and every column of the fit are
    finite, as `_fit_member` and `_fit_unit` filter them (`_worker.finite_rows`)."""
    period, hz, start, stop = key
    finite = finite_rows(arrays["x"][start:stop][:, columns], arrays["y"][start:stop, hz - 1])
    return _rowset_key(period, hz, slice(start, stop), finite, spec)


class _Span:
    """One span's `SpanSums` over the columns its job's fits read (their places in the period's arrays, ascending)."""

    __slots__ = ("sums", "columns", "n")

    def __init__(self, arrays: Dict, key: SpanKey, columns: np.ndarray):
        _, hz, start, stop = key
        self.columns = columns
        self.n = stop - start
        self.sums = SpanSums(arrays["x"][start:stop][:, columns], arrays["y"][start:stop, hz - 1].copy())

    def local(self, columns: Sequence[int]) -> np.ndarray:
        """The places of a fit's columns among the span's; a column the job's plan didn't give the span is a bug, not a fit."""
        wanted = np.asarray(columns, dtype=np.int64)
        at = np.searchsorted(self.columns, wanted)
        if wanted.size and (at.max() >= self.columns.size or not np.array_equal(self.columns[at], wanted)):
            raise RuntimeError(
                f"a fit reads columns {wanted.tolist()} its span's sums don't hold {self.columns.tolist()}"
            )
        return at


def inverse_slopes(scaler, z) -> np.ndarray:
    """The slope of `scaler.inverse` (target units per scaled unit) at each scaled value of `z`, the larger of the one-sided
    ones with a step of 1e-6·(1 + |z|) (so at an npit knot the steeper side): how much a fit's error in scaled units there is
    in target units. inf where the inverse overflows."""
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    step = 1e-6 * (1.0 + np.abs(z))
    with np.errstate(over="ignore", invalid="ignore"):
        at = np.asarray(scaler.inverse(z), dtype=np.float64).reshape(-1)
        up = np.asarray(scaler.inverse(z + step), dtype=np.float64).reshape(-1)
        down = np.asarray(scaler.inverse(z - step), dtype=np.float64).reshape(-1)
        slope = np.maximum(np.abs(up - at), np.abs(at - down)) / step
    return np.where(np.isfinite(slope), slope, np.inf)


def inverse_slope(scaler, z) -> float:
    """The largest of `inverse_slopes` (0.0 for no value)."""
    slopes = inverse_slopes(scaler, z)
    return float(slopes.max()) if slopes.size else 0.0


class _RowSet:
    """One VST row set (see the module): a span's rows ρ for one scaler spec, its columns transformed by one call of the
    scaler's own `fit_transform` on ρ, each with its mask flag (`virtual`: (column, flag) pairs, ascending; flag False leaves
    the column as it is), the transformed target, and a `SpanSums` of them over ρ alone. `column_params` and `vst` are the
    scaler's per column (its `_column_params`, `_vst_mask`), `scaler` maps a forecast back (its target's parameters),
    `yscale` and `typical` the largest and the median slope of that inverse at the training targets (`inverse_slopes`)."""

    __slots__ = (
        "virtual",
        "index",
        "scaler",
        "column_params",
        "vst",
        "Z",
        "zy",
        "r",
        "sums",
        "yscale",
        "typical",
        "all",
        "values",
    )

    def __init__(self, arrays: Dict, key: RowSetKey, virtual: Sequence[Tuple[int, bool]], names: np.ndarray):
        _, hz, start, stop, packed, spec = key
        finite = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), count=stop - start).astype(bool)
        rho = start + np.flatnonzero(finite)
        cols = np.array([c for c, _ in virtual], dtype=np.int64)
        flags = np.array([flag for _, flag in virtual], dtype=bool)
        columns = [str(name) for name in names[cols]]
        scaler = make_scaler(json.loads(spec))
        x = arrays["x"][np.ix_(rho, cols)]
        y = arrays["y"][rho, hz - 1]
        Z, zy, _ = scaler.fit_transform(x, y, np.empty((0, cols.size)), scalable_mask=flags, columns=columns)
        self.virtual = frozenset(virtual)
        self.index = {pair: i for i, pair in enumerate(virtual)}
        self.scaler = scaler
        self.column_params = list(scaler._column_params)
        self.vst = np.asarray(scaler._vst_mask(cols.size, columns), dtype=bool)
        self.Z = Z
        self.zy = zy
        self.r = int(rho.size)
        self.sums = SpanSums(Z, zy)
        slopes = inverse_slopes(scaler, zy)
        self.yscale = float(slopes.max())
        self.typical = float(np.median(slopes))
        self.all = np.ones(self.r, dtype=bool)
        self.values: Dict[Tuple[int, int], float] = {}

    def local(self, columns: Sequence[int], mask: np.ndarray) -> np.ndarray:
        """The places of a fit's (column, mask flag) pairs in the set (KeyError where the set doesn't hold one)."""
        return np.array([self.index[(int(c), bool(flag))] for c, flag in zip(columns, mask)], dtype=np.int64)

    def transform(self, local: np.ndarray, x: np.ndarray) -> np.ndarray:
        """An issue row's raw values (`x`, the fit's columns in its order) as `fit_transform` transforms a test row: a VST
        column (`_forward`) then (· - mean) / std with its parameters, another scaled one (x - mean) / std, an unscaled one
        as it is. Value by value, each on an array of its own, so a value's bits depend on its column and itself alone,
        whichever other columns or rows go with it; kept per (column, value bits), as every fit of the set reads them."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        bits = x.view(np.int64)
        out = np.empty(x.size)
        for j, place in enumerate(local.tolist()):
            key = (place, int(bits[j]))
            value = self.values.get(key)
            if value is None:
                value = self.values[key] = self._value(place, x[j : j + 1])
            out[j] = value
        return out

    def _value(self, place: int, x: np.ndarray) -> float:
        params = self.column_params[place]
        if params is None:
            return float(x[0])
        fitted, mean, std = params
        column = self.scaler._forward(fitted, x) if self.vst[place] else x
        return float(((column - mean) / std)[0])


class _Spans:
    """The spans and row sets of one task thread: each span's `_Span` or row set's `_RowSet` built when a fit first asks for
    it, from the job's plan (`columns`, the columns each one's fits read), and let go once the thread ran the last item that
    reads it (`done`). A row set a fit asks for with a (column, flag) it doesn't hold (a column the fit's mask leaves alone
    that the set's first fit scaled, or the other way round) is built again with it, and every pair it held: each pair's
    transform and sums are its own, so no bit moves."""

    def __init__(self, hour_arrays: Dict, columns: Dict[tuple, np.ndarray], uses: Counter):
        self.hour_arrays = hour_arrays
        self.columns = columns
        self.uses = uses
        self.built: Dict[tuple, Union[_Span, _RowSet]] = {}
        self._names: Dict[int, np.ndarray] = {}

    def get(self, period: int, horizon: int, rows: slice) -> _Span:
        key = (period, horizon, rows.start, rows.stop)
        span = self.built.get(key)
        if span is None:
            if key not in self.columns:
                raise RuntimeError(f"a fit of span {key} that its job's plan doesn't have")
            span = self.built[key] = _Span(self.hour_arrays[period], key, self.columns[key])
        return span

    def rowset(self, fit: StatsFit, spec: str) -> _RowSet:
        key = _rowset_key(fit.period, fit.horizon, fit.rows, fit.finite, spec)
        wanted = {(int(c), bool(flag)) for c, flag in zip(fit.columns, np.asarray(fit.mask, dtype=bool))}
        rowset = self.built.get(key)
        if rowset is None or not wanted <= rowset.virtual:
            if key not in self.columns:
                raise RuntimeError(f"a fit of row set {key[:4]} that its job's plan doesn't have")
            virtual = {(int(c), True) for c in self.columns[key].tolist()} | wanted
            if rowset is not None:
                virtual |= rowset.virtual
            rowset = self.built[key] = _RowSet(
                self.hour_arrays[fit.period], key, sorted(virtual), self.names(fit.period)
            )
        return rowset

    def names(self, period: int) -> np.ndarray:
        """The column names of a period's arrays, by place."""
        names = self._names.get(period)
        if names is None:
            index = self.hour_arrays[period]["col_index"]
            names = np.empty(max(index.values(), default=-1) + 1, dtype=object)
            for name, place in index.items():
                names[place] = name
            self._names[period] = names
        return names

    def done(self, keys) -> None:
        for key in keys:
            self.uses[key] -= 1
            if self.uses[key] <= 0:
                self.built.pop(key, None)


class EngineFits:
    """A model's fits from shared statistics in one task thread: the `engine` its tasks' `_fit_member` and units' `_fit_unit`
    call with each fit (`_worker.StatsFit`), returning (prediction in target units, coefficients) per issue row, or None to hand
    the fit back to the model path (see the module)."""

    def __init__(self, spans: _Spans, inputs: ModelInputs, counts: EngineCounts):
        self.spans = spans
        self.inputs = inputs
        self.model = inputs.inputs.model
        self.scale_binary = inputs.scale_binary
        self.counts = counts
        self._first_day: Optional[int] = None

    def _back(self, reason: str) -> None:
        self.counts.fallbacks[reason] += 1
        return None

    def __call__(self, fit: StatsFit) -> Optional[List[Optional[Tuple[float, list]]]]:
        """The fit's forecast of each issue row (prediction in target units, coefficients) or None for a row the model path
        forecasts (`FAR_ISSUE`); None for the whole fit (a hand-back)."""
        if fit.x.shape[1] == 0:
            return self._back(NO_COLUMNS)  # the model path's own outcome: a fit_error, or LEAR's mean
        r = fit.x.shape[0]
        folds = self.inputs.folds
        if r < MIN_ROWS or (folds is not None and r < folds):
            return self._back(FEW_ROWS)  # the model path's descent or KFold refuses fewer rows than folds: a fit_error
        try:
            stats, transform, inverse, rowset = self._statistics(fit)
        except SumsOverflow:
            return self._back(OVERFLOW)
        except StatsFallback as exc:
            return self._back(exc.reason)
        near = getattr(self.model, "_ENGINE_NEAR_TWIN", None)
        if near is not None and near_twins(stats.G, near):
            return self._back(NEAR_TWIN)
        try:
            fitted = self.model._fit_stats(stats)
        except ValueError:
            return self._back(ERROR)
        if fitted is None:
            return self._back(DECLINED)
        bound = getattr(self.model, "_ENGINE_ISSUE_Z", ISSUE_Z_MAX)
        condition_max = getattr(self.model, "_ENGINE_CONDITION_MAX", None)
        metric = issue_metric(stats, fitted) if bound < np.inf or condition_max is not None else None
        if condition_max is not None and not metric.condition <= condition_max:
            return self._back(ILL_CONDITIONED)
        out: List[Optional[Tuple[float, list]]] = []
        for row in range(fit.test.shape[0]):
            # One row at a time, as the model path forecasts (decision 16 of the step 3 spec). Whether a row is too far out is
            # decided from the fit and that row alone, so the same whichever other rows a unit forecasts.
            z = transform(fit.test[row : row + 1])
            distance = 0.0 if metric is None else issue_distance(metric, z)
            if not (distance <= bound and issue_rounding_ok(fitted, stats, z)):
                self.counts.fallbacks[FAR_ISSUE] += 1
                out.append(None)
                continue
            prediction, coefs = self.model._predict_fitted(fitted, z)
            if rowset is not None and not _steep_ok(fitted, stats, z, rowset, float(prediction)):
                self.counts.fallbacks[FAR_ISSUE] += 1
                out.append(None)
                continue
            self.counts.issue(distance)
            out.append((inverse(float(prediction)), coefs))
        if all(row is None for row in out):
            return None
        self.counts.fits += 1
        if self.inputs.check and self._sampled(fit):
            self._check(fit, fitted, out)
        return out

    def _statistics(self, fit: StatsFit):
        """(`ScaledStats`, the issue rows' transform, the forecast's inverse, the row set or None) of a fit: from its span's
        sums and `standard_stats`, or from its VST row set and `transformed_stats`; with the CV blocks where the model needs
        them, from the same sums. Raises `SumsOverflow` or `StatsFallback` for a hand-back."""
        folds = self.inputs.folds
        if self.inputs.vst is None:
            span = self.spans.get(fit.period, fit.horizon, fit.rows)
            columns = span.local(fit.columns)
            r = fit.x.shape[0]
            method = "downdate" if span.n - r <= DOWNDATE_SHARE * span.n else "direct"
            raw = span.sums.over(fit.finite, columns, method=method)
            stats, params = standard_stats(raw, (fit.x, fit.y), fit.mask, self.scale_binary)
            if folds is not None:
                blocks = [span.sums.over(rows, columns) for rows in block_rows(np.flatnonzero(fit.finite), folds)]
                stats = stats._replace(blocks=block_stats(raw, blocks, params, (fit.x, fit.y)))
            return stats, params.transform, params.inverse, None
        rowset = self.spans.rowset(fit, self.inputs.vst)
        local = rowset.local(fit.columns, fit.mask)
        if isinstance(getattr(rowset.scaler, "apply_to", None), list):
            # The model path's own warning where the fit's patterns match none of its columns (`fit_transform` on its rows).
            rowset.scaler._vst_mask(local.size, [str(name) for name in self.spans.names(fit.period)[fit.columns]])
        raw = rowset.sums.over(rowset.all, local)

        def rows():
            return np.ascontiguousarray(rowset.Z[:, local]), rowset.zy

        stats, params = transformed_stats(raw, rows, rowset.yscale)
        if folds is not None:
            blocks = [rowset.sums.over(part, local) for part in block_rows(np.arange(rowset.r), folds)]
            made = block_stats(raw, blocks, params, rows(), absolute=True)
            stats = stats._replace(blocks=tuple(block._replace(yscale=rowset.yscale) for block in made))

        def transform(x):
            return rowset.transform(local, x)[None, :]

        return stats, transform, rowset.scaler.inverse, rowset

    def _sampled(self, fit: StatsFit) -> bool:
        """Whether `engine="check"` runs this fit on the model path too: the hash of its span (period, horizon, its first and
        last rows' dates), 0 modulo `CHECK_EVERY`. Nothing of the group, the model or the run's start reaches it."""
        if self._first_day is None:
            cfg = self.inputs.inputs.config
            self._first_day = datetime.date.fromisoformat(cfg["test_start"]).toordinal() - int(cfg["offset"])
        days = self.spans.hour_arrays[fit.period]["days"]
        first, last = int(days[fit.rows.start]) + self._first_day, int(days[fit.rows.stop - 1]) + self._first_day
        digest = hashlib.blake2b(f"{fit.period}:{fit.horizon}:{first}:{last}".encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") % CHECK_EVERY == 0

    def _check(self, fit: StatsFit, fitted, out: List[Optional[Tuple[float, list]]]) -> None:
        """The fit on the model path as well (its scaler and `_fit_model` on the same values, each row through
        `_predict_fitted`), counted against the engine's forecasts (`EngineCounts.checked`); one the model path can't fit
        (a ValueError, its fit_error, where the engine forecast) is counted as such (`model_path_errors`)."""
        names = [str(name) for name in self.spans.names(fit.period)[fit.columns]]
        scaler = make_scaler(self.inputs.inputs.config.get("scaler", "standard"))
        x, y, test = scaler.fit_transform(
            fit.x.copy(), fit.y.copy(), fit.test.copy(), scalable_mask=fit.mask, columns=names
        )
        try:
            own = self.model._fit_model(x, y)
        except ValueError:
            self.counts.model_path_error()
            return
        pairs = []
        for row, made in enumerate(out):
            if made is not None:
                prediction, _ = self.model._predict_fitted(own, test[row : row + 1])
                pairs.append((made[0], scaler.inverse(float(prediction))))
        self.counts.checked(pairs, _other_choice(fitted, own))


def _steep_ok(fitted, stats, z: np.ndarray, rowset: _RowSet, prediction: float) -> bool:
    """Whether a VST fit's forecast (`prediction`, scaled) is where its scaler's inverse is no steeper than `VST_SLOPE_MAX`
    times its median slope at the training targets; and for a fit that carries its model path's rounding (Ridge), whether
    that stays within `RIDGE_ROUNDING_MAX` in target units there: its budget is in target units through `yscale`, the
    inverse's largest slope at the training targets, so a forecast where the inverse is steeper scales it by that slope. A
    function of the fit and the row alone."""
    slope = inverse_slope(rowset.scaler, [prediction])
    if not slope <= VST_SLOPE_MAX * rowset.typical:
        return False
    if getattr(fitted, "rounding", None) is None or slope <= rowset.yscale:
        return True
    return issue_rounding_ok(fitted, stats, z, RIDGE_ROUNDING_MAX * rowset.yscale / slope)


def _other_choice(engine_fit, model_fit) -> bool:
    """Whether the engine's fit chose another penalty than the model path's: another kink (LEAR), or another α (Ridge,
    LassoCV: their grids' values agree to rounding, adjacent ones are percent apart)."""
    kink = getattr(engine_fit, "kink", None)
    if kink is not None:
        return int(kink) != int(getattr(model_fit, "kink", -1))
    mine, theirs = getattr(engine_fit, "alpha_", None), getattr(model_fit, "alpha_", None)
    if mine is None or theirs is None:
        return False
    return not math.isclose(float(mine), float(theirs), rel_tol=1e-9)


def _one_blas_thread(fits_from_statistics: bool):
    """The BLAS thread count a job with fits from statistics runs at: one, whatever the process's. The sums are exact at any
    count, but a fit's own solve isn't: LAPACK's `eigh` (Ridge) and Cholesky, and a threaded GEMM on the fit's rows, change their
    last bits with the thread count once a fit has about a hundred columns (n 400, p 120: Ridge's eigenvectors and `Zc @ V`;
    p 247: `dpotrf` too). A pool's worker runs at `NATIVE_THREADS_PER_TASK` = 1 already, so nothing changes there; set once per
    job, before its task threads start, since a BLAS library's thread count is the process's."""
    if not fits_from_statistics or (worker._W_THREADPOOLS is not None and worker.NATIVE_THREADS_PER_TASK == 1):
        return contextlib.nullcontext()
    return threadpool_limits(limits=1, user_api="blas")


def _job_spans(
    items: Sequence[Tuple[int, tuple, Optional[List[tuple]]]], models: Sequence[ModelInputs]
) -> Tuple[List[List[tuple]], Dict[tuple, np.ndarray]]:
    """Per item the spans (StandardScaler) or row sets (a VST, `_planned_rowset`) its engine fits read (none for a model on
    the model path), and per span or row set the columns of all of them."""
    keys: List[List[tuple]] = []
    columns: Dict[tuple, set] = {}
    for place, what, unit_tasks in items:
        model = models[place]
        if not model.from_stats:
            keys.append([])
            continue
        spans = _item_spans(what, unit_tasks, model.inputs)
        if model.vst is not None:
            arrays = model.inputs.hour_arrays
            spans = [(_planned_rowset(arrays[key[0]], key, cols, model.vst), cols) for key, cols in spans]
        keys.append(list(dict.fromkeys(key for key, _ in spans)))
        for key, cols in spans:
            columns.setdefault(key, set()).update(cols)
    return keys, {key: np.array(sorted(cols), dtype=np.int64) for key, cols in columns.items()}


def _batches(keys: List[List[tuple]], threads: int) -> List[List[int]]:
    """The items cut into at most `threads` batches, each run on one thread with its own spans: the items that read the same
    spans together, the largest such sets first onto the least loaded batch (ties: the earlier batch); each batch in item
    order."""
    together: Dict[tuple, List[int]] = {}
    for i, item_keys in enumerate(keys):
        together.setdefault(tuple(sorted(item_keys)), []).append(i)
    batches: List[List[int]] = [[] for _ in range(threads)]
    for _, indices in sorted(together.items(), key=lambda kv: (-len(kv[1]), kv[1][0])):
        least = min(range(threads), key=lambda b: (len(batches[b]), b))
        batches[least].extend(indices)
    return [sorted(batch) for batch in batches if batch]


Outcome = Tuple[int, List[Dict], List[Tuple[tuple, str]], Optional[EngineCounts]]


def run_job(job: Sequence[Tuple[int, List[tuple]]], models: Optional[Sequence[ModelInputs]] = None) -> List[Outcome]:
    """Run one engine job: (model's place in the chunk, its tasks of the job's day or block and period) for each model with
    some, on `models` (the pool's if None). Returns (place, records, [(task, skip reason)], its `EngineCounts` or None) per
    model, as `_run_day` or `_run_block` returns them for that model's tasks, its tasks or units in their order in the job.
    With `THREADS_PER_PROCESS` above 1 the job's tasks (refit=None) or units (refit) of every model run on that many threads,
    those that read exactly the same spans on one (`_batches`), each thread with its own spans: a span that items of two
    threads read (a window list's and a single window's) is built on both. A task that raises raises
    `EngineTaskFailed` naming its model."""
    models = _E_MODELS if models is None else models
    items: List[Tuple[int, tuple, Optional[List[tuple]]]] = []  # (place, task or unit, the unit's tasks or None)
    blocks: Dict[int, bool] = {}
    for place, tasks in job:
        cfg = models[place].inputs.config
        blocks[place] = cfg.get("refit") is not None
        if blocks[place]:
            items.extend((place, unit, unit_tasks) for unit, unit_tasks in block_units(cfg, tasks).items())
        else:
            items.extend((place, task, None) for task in tasks)
    threads = models[job[0][0]].inputs.config["threads_per_process"] if job else 1
    keys, columns = _job_spans(items, models)
    arrays = {id(models[place].inputs.hour_arrays) for place, _ in job if models[place].from_stats}
    if len(arrays) > 1:
        raise RuntimeError("the models of an engine job with fits from shared statistics must share one set of arrays")
    hour_arrays = models[job[0][0]].inputs.hour_arrays if job else {}

    def run(batch: List[int]) -> Tuple[List[Tuple[int, Union[Dict, str, list]]], Dict[int, EngineCounts]]:
        spans = _Spans(hour_arrays, columns, Counter(key for i in batch for key in keys[i]))
        counts: Dict[int, EngineCounts] = {}
        fitters: Dict[int, EngineFits] = {}
        out = []
        for i in batch:
            place, what, unit_tasks = items[i]
            model = models[place]
            engine = None
            if model.from_stats:
                engine = fitters.get(place)
                if engine is None:
                    counts[place] = EngineCounts(check=model.check)
                    engine = fitters[place] = EngineFits(spans, model, counts[place])
            try:
                if unit_tasks is None:
                    outcome = _run_task(what, model.inputs, engine)
                else:
                    outcome = _run_unit(what, unit_tasks, model.inputs, engine)
            except TaskFailed as exc:
                raise EngineTaskFailed(place, str(exc)) from None
            spans.done(keys[i])
            out.append((i, outcome))
        return out, counts

    with _one_blas_thread(bool(columns)):
        if len(items) <= 1 or threads == 1:
            parts = [run(list(range(len(items))))]
        else:
            with ThreadPoolExecutor(max_workers=threads, initializer=_limit_openmp_in_task_thread) as pool:
                futures = [pool.submit(run, batch) for batch in _batches(keys, threads)]
                parts = [future.result() for future in as_completed(futures)]
    ran = sorted(((i, outcome) for part, _ in parts for i, outcome in part), key=lambda done: done[0])
    totals: Dict[int, EngineCounts] = {}
    for _, counts in parts:
        for place, part_counts in counts.items():
            totals.setdefault(place, EngineCounts()).add(part_counts)

    # Per model, in the order of its tasks or units in the job, as `_run_day` and `_run_block` collect them on one thread.
    done: Dict[int, list] = {place: [] for place, _ in job}
    for i, outcome in ran:
        place, what, _ = items[i]
        done[place].append((what, outcome))
    out: List[Outcome] = []
    for place, outcomes in done.items():
        if blocks[place]:
            results, skipped = block_records(models[place].inputs.config, [outcome for _, outcome in outcomes])
        else:
            results = [outcome for _, outcome in outcomes if not isinstance(outcome, str)]
            skipped = [(task, outcome) for task, outcome in outcomes if isinstance(outcome, str)]
        out.append((place, results, skipped, totals.get(place)))
    return out
