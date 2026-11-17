"""Stored fits (step 8d of the forecasting layer): a model's fits made before an issue, kept as arrays in its live directory, and
the forecasts served from them.

    live/de_price/fits/<model>/<anchor>/
      units.npz   every unit of the anchor: its scaler's state (`BaseScaler.state`), its model's (`BaseModel._fit_state`) and,
                  with `ModelPipeline(engine="on")`, its fit from shared statistics; arrays only, read with allow_pickle=False
      fit.json    the anchor, the configuration digest, each unit (key, outcome, rows, newest target), the hash of the rows the
                  fits read, the fit path, the kernels, the package version, when it was made, and the sha256 of units.npz

**A unit** is what a backtest fits once: with `refit` a `FitUnit` (period, horizon, anchor, list, window member) fitted on the rows
of a forecast issued on its anchor (`_worker.unit_training`), without it one window member of one forecast, whose anchor is its
issue day (`_worker.member_training`). It is fitted by the backtest's own functions on the backtest's own rows, and forecasts
through them (`_worker._fit_unit(stored=)`, `_worker._fit_member(stored=)`): the issue row's skips, the scaler's transform of
it, `_predict_fitted` one row at a time, the inverse and the record are the backtest's lines, so a forecast served from a stored
fit is the backtest's record to the bit (tests/models/test_stored_fits.py). With the engine a unit keeps both its fit from shared
statistics (`_engine_worker.EngineUnitFit`) and the model path's, since whether an issue row is too far out for the engine is
the row's own (`EngineUnitFit.forecast`).

**Keys.** A unit is found by (period, horizon, the hash of its list, its window's place among the model's): the hash is the
record's `list_hash`, so a predictor function's list is found by its content, whichever id its plan gave it.

**Due.** `refit(if_due=True)` fits an anchor when no stored fit of it with the current configuration digest holds every unit the
issue day needs (`config_digest`: the live manifest's compared fields, the information set's availability, the predictor
function's fingerprint and this format). **A fit is refused while a training target it reads is not yet known** by the target's
rule at the clock's now (decision 5 of the step 8 spec, `check_published`): the error names the publication time.

**Serving** (`serve`, what `forecast()` of 8e calls): the anchor's stored fit, or with `fit_missing="previous"` the newest stored
older one of the same digest, its age recorded (decision 15); `"fit"` fits the missing anchor inline, `"skip"` leaves its
forecasts as `no_stored_fit` skips. A `refit=None` model without a stored fit for the issue day fits inline, as the backtest does.
Units the anchor's own stored fit lacks (a list of a predictor function that the refit's plan didn't give: an input it switches on
was out when the revision ran but not when the anchor was fitted) are fitted inline at the anchor, unless `"skip"`.

**Families.** A class stores fits when the class that fits and forecasts (`_fit_model`, `_predict_fitted`) also defines
`_fit_state` and `_from_state` (`stores_fits`): OLS, Ridge, LassoCV and LEAR. Every family of step 6 (torch, the trees, the
in-context and foundation models) brings no format yet and fits at serve time: no pickle (decision 4).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._clock import current_clock
from ..scalers.base import BaseScaler
from ._lars import LearFit
from ._stats import IssueMetric, LinearFit, SolveError, StandardParams

# The version of the files' layout and of what a unit holds: in every configuration digest, so a later format refits.
FORMAT = (
    3  # 2: the rows read from their first date, `incomplete`, `notes`; 3: each unit's `rows_hash` of the cells it reads
)
FITS_DIR = "fits"
UNITS_FILE = "units.npz"
FIT_FILE = "fit.json"
# The skip reason of a forecast whose anchor has no stored fit, served with `fit_missing="skip"`.
SKIP_NO_STORED_FIT = "no_stored_fit"
FIT_MISSING = ("previous", "fit", "skip")


class StoredFitError(ValueError):
    """A stored fit that can't be served: its `units.npz` isn't the file its `fit.json` names (tampered with, or written over),
    its `fit.json` can't be read, or it holds another unit than its list says."""


class FitTooEarlyError(ValueError):
    """A fit asked for before a training target it reads is known by the target's rule (decision 5 of the step 8 spec)."""


# ---------------------------------------------------------------------------------------------------------------------------
# States: fitted objects as arrays
# ---------------------------------------------------------------------------------------------------------------------------


def _text(value: str) -> np.ndarray:
    return np.array(str(value))


def _read_text(state: Mapping[str, np.ndarray], key: str) -> str:
    return str(state[key][()])


def _check_kind(state: Mapping[str, np.ndarray], kind: str, owner: str) -> None:
    found = _read_text(state, "kind") if "kind" in state else None
    if found != kind:
        raise StoredFitError(f"{owner}: a stored fit of kind {found!r}, not {kind!r}")


def estimator_state(estimator: Any) -> Dict[str, np.ndarray]:
    """A fitted scikit-learn linear estimator (`LinearRegression`, `LassoCV`, `RidgeCV`) as arrays: its class, `coef_`,
    `intercept_`, `n_features_in_` and `alpha_` where it has one. What its `predict` reads (`X @ coef_ + intercept_`), so the
    estimator `estimator_from_state` sets them on forecasts to the bit."""
    out = {
        "kind": _text("sklearn"),
        "class": _text(type(estimator).__name__),
        "coef": np.array(estimator.coef_, dtype=np.float64),
        "intercept": np.array(estimator.intercept_, dtype=np.float64),
        "n_features_in": np.array(int(estimator.n_features_in_), dtype=np.int64),
    }
    alpha = getattr(estimator, "alpha_", None)
    if alpha is not None:
        out["alpha"] = np.array(alpha, dtype=np.float64)
    return out


def estimator_from_state(estimator: Any, state: Mapping[str, np.ndarray]) -> Any:
    """`estimator` (unfitted, the model's own `_estimator()`) with the fitted attributes of `estimator_state` set on it."""
    _check_kind(state, "sklearn", type(estimator).__name__)
    if _read_text(state, "class") != type(estimator).__name__:
        raise StoredFitError(f"a stored {_read_text(state, 'class')} can't be read as a {type(estimator).__name__}")
    estimator.coef_ = np.array(state["coef"], dtype=np.float64)
    intercept = np.array(state["intercept"], dtype=np.float64)
    estimator.intercept_ = np.float64(intercept) if intercept.ndim == 0 else intercept
    estimator.n_features_in_ = int(state["n_features_in"])
    if "alpha" in state:
        estimator.alpha_ = np.float64(state["alpha"])
    return estimator


def linear_fit_state(fit: LinearFit) -> Dict[str, np.ndarray]:
    """An engine's `LinearFit` (OLS, Ridge, LassoCV from shared statistics) as arrays: coefficients, intercept, penalty, and what
    its issue-row checks read (Ridge's `rounding`, OLS's `SolveError`)."""
    out = {
        "kind": _text("linear_fit"),
        "coef": np.array(fit.coef_, dtype=np.float64),
        "intercept": np.array(fit.intercept_, dtype=np.float64),
        "centred": np.array(bool(fit.centred)),
    }
    if fit.alpha_ is not None:
        out["alpha"] = np.array(fit.alpha_, dtype=np.float64)
    if fit.rounding is not None:
        out["rounding"] = np.array(fit.rounding, dtype=np.float64)
    if fit.solve is not None:
        out["solve_live"] = np.array(fit.solve.live, dtype=np.int64)
        out["solve_factor"] = np.array(fit.solve.factor, dtype=np.float64, order="K")
        out["solve_correction"] = np.array(fit.solve.correction, dtype=np.float64)
        out["solve_h"] = np.array(fit.solve.h, dtype=np.float64)
    return out


def linear_fit_from_state(state: Mapping[str, np.ndarray]) -> LinearFit:
    _check_kind(state, "linear_fit", "LinearFit")
    solve = None
    if "solve_live" in state:
        solve = SolveError(
            np.array(state["solve_live"], dtype=np.int64),
            np.array(state["solve_factor"], dtype=np.float64, order="K"),
            np.array(state["solve_correction"], dtype=np.float64),
            np.array(state["solve_h"], dtype=np.float64),
        )
    return LinearFit(
        np.array(state["coef"], dtype=np.float64),
        float(state["intercept"]),
        float(state["alpha"]) if "alpha" in state else None,
        np.array(state["rounding"], dtype=np.float64) if "rounding" in state else None,
        bool(state["centred"]),
        solve,
    )


def lear_fit_state(fit: LearFit) -> Dict[str, np.ndarray]:
    """A `LearFit` as arrays: `w`, `xbar`, `ybar`, and the stage-1 `alpha`, `kink` and `converged` it was chosen with."""
    return {
        "kind": _text("lear"),
        "w": np.array(fit.w, dtype=np.float64),
        "xbar": np.array(fit.xbar, dtype=np.float64),
        "ybar": np.array(fit.ybar, dtype=np.float64),
        "alpha": np.array(fit.alpha, dtype=np.float64),
        "kink": np.array(int(fit.kink), dtype=np.int64),
        "converged": np.array(bool(fit.converged)),
    }


def lear_fit_from_state(state: Mapping[str, np.ndarray]) -> LearFit:
    _check_kind(state, "lear", "LEARModel")
    return LearFit(
        np.array(state["w"], dtype=np.float64),
        np.array(state["xbar"], dtype=np.float64),
        float(state["ybar"]),
        float(state["alpha"]),
        int(state["kink"]),
        bool(state["converged"]),
    )


def stores_fits(model: Any) -> Optional[str]:
    """None when `model`'s fits can be stored, else why they fit at serve time: a model that fits nothing, a family that brings
    no format of its own (`_fit_state` and `_from_state` not defined by the class that defines `_fit_model` and
    `_predict_fitted`, so a subclass with another estimator never stores its parent's), and what the stored path leaves out:
    day and pooled units, block calls."""
    from ._units import PERIOD  # the units module reads this one's callers
    from ._worker import block_calls

    cls = type(model)
    if getattr(model, "_fits_nothing", False):
        return "it fits nothing (each forecast is a value of its issue row)"
    if getattr(model, "_unit_kind", PERIOD) != PERIOD:
        return f"its units are {model._unit_kind} units, which store no fit yet: it fits at serve time"
    owners = {
        next((klass for klass in cls.__mro__ if name in vars(klass)), None)
        for name in ("_fit_model", "_predict_fitted", "_fit_state", "_from_state")
    }
    from .base import BaseModel  # base imports this module's callers

    if len(owners) != 1 or None in owners or BaseModel in owners:
        return (
            f"{cls.__name__} brings no stored format (one class must define _fit_model, _predict_fitted, _fit_state and "
            "_from_state): it fits at serve time"
        )
    if block_calls(model):
        return "its refit units predict their blocks in one call, which a stored fit doesn't serve yet: it fits at serve time"
    return None


def _prefixed(prefix: str, state: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {f"{prefix}.{key}": value for key, value in state.items()}


def _section(arrays: Mapping[str, np.ndarray], prefix: str) -> Dict[str, np.ndarray]:
    start = prefix + "."
    return {key[len(start) :]: value for key, value in arrays.items() if key.startswith(start)}


class _RowChecks(NamedTuple):
    """What an engine fit's issue-row checks read of its statistics (`issue_rounding_ok`, `solve_error_ok`, `_steep_ok`)."""

    zbar: np.ndarray
    yscale: Optional[float]


class _RowSetView(NamedTuple):
    """What `_engine_worker._steep_ok` reads of a VST row set: its scaler's inverse, and the median and largest slope of that
    inverse at the training targets."""

    scaler: BaseScaler
    typical: float
    yscale: float


def _vst_row_transform(scaler: BaseScaler, columns: List[Optional[Tuple[Any, Any, Any]]], vst: np.ndarray):
    """An issue row's transform of a VST engine fit, as `_engine_worker._RowSet.transform` makes it: each value on an array of
    its own, through its column's VST (`_forward`) and (· - mean) / std, a column the mask leaves alone as it is."""

    def transform(x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        out = np.empty(x.size)
        for j in range(x.size):
            params = columns[j]
            if params is None:
                out[j] = float(x[j : j + 1][0])
                continue
            fitted, mean, std = params
            column = scaler._forward(fitted, x[j : j + 1]) if vst[j] else x[j : j + 1]
            out[j] = float(((column - mean) / std)[0])
        return out[None, :]

    return transform


def engine_fit_state(efit: Any, model: Any) -> Dict[str, np.ndarray]:
    """An `_engine_worker.EngineUnitFit` as arrays: the model's fit (`_fit_state`), what the issue-row checks read (z̄, the
    target's scale, the `IssueMetric` and the bound), and the issue rows' transform and the forecast's inverse: StandardScaler's
    `StandardParams`, or a VST's column parameters of the fit's columns with its row set's scaler, typical and largest slope."""
    from ..scalers.vst import _packed

    out = _prefixed("fitted", model._fit_state(efit.fitted))
    out["zbar"] = np.array(efit.stats.zbar, dtype=np.float64)
    yscale = efit.stats.yscale
    out["has_yscale"] = np.array(yscale is not None)
    out["yscale"] = np.array(np.nan if yscale is None else float(yscale))
    out["bound"] = np.array(float(efit.bound))
    if efit.metric is not None:
        metric = efit.metric
        out["metric.columns"] = np.array(metric.columns, dtype=np.int64)
        out["metric.centre"] = np.array(metric.centre, dtype=np.float64)
        out["metric.column_scale"] = np.array(metric.column_scale, dtype=np.float64)
        out["metric.vectors"] = np.array(metric.vectors, dtype=np.float64, order="K")
        out["metric.scale"] = np.array(metric.scale, dtype=np.float64)
        out["metric.condition"] = np.array(float(metric.condition))
    if efit.params is not None:
        out["kind"] = _text("standard")
        for name in StandardParams._fields:
            out[f"params.{name}"] = np.array(getattr(efit.params, name))
        return out
    rowset, local = efit.rowset, np.asarray(efit.local, dtype=np.int64)
    columns = [rowset.column_params[place] for place in local.tolist()]
    out["kind"] = _text("vst")
    out.update(_prefixed("scaler", rowset.scaler.state()))
    out["vst"] = np.array(np.asarray(rowset.vst, dtype=bool)[local], dtype=bool)
    out["scaled"] = np.array([column is not None for column in columns], dtype=bool)
    out["mean"] = np.array([0.0 if column is None else column[1] for column in columns], dtype=np.float64)
    out["std"] = np.array([1.0 if column is None else column[2] for column in columns], dtype=np.float64)
    out.update(_packed("params", [column[0] for column in columns if column is not None and column[0] is not None]))
    out["typical"] = np.array(float(rowset.typical))
    out["rowset_yscale"] = np.array(float(rowset.yscale))
    return out


def engine_fit_from_state(state: Mapping[str, np.ndarray], model: Any) -> Any:
    """The `_engine_worker.EngineUnitFit` `engine_fit_state` stored, whose `forecast` of a row is the stored one's to the bit."""
    from ..scalers.vst import _unpacked
    from ._engine_worker import EngineUnitFit

    fitted = model._from_state(_section(state, "fitted"))
    stats = _RowChecks(
        np.array(state["zbar"], dtype=np.float64), float(state["yscale"]) if bool(state["has_yscale"]) else None
    )
    metric = None
    if "metric.columns" in state:
        metric = IssueMetric(
            np.array(state["metric.columns"], dtype=np.int64),
            np.array(state["metric.centre"], dtype=np.float64),
            np.array(state["metric.column_scale"], dtype=np.float64),
            np.array(state["metric.vectors"], dtype=np.float64, order="K"),
            np.array(state["metric.scale"], dtype=np.float64),
            float(state["metric.condition"]),
        )
    bound = float(state["bound"])
    kind = _read_text(state, "kind")
    if kind == "standard":
        values = {name: np.array(state[f"params.{name}"]) for name in StandardParams._fields}
        params = StandardParams(
            mean=values["mean"].astype(np.float64),
            scale=values["scale"].astype(np.float64),
            scaled=values["scaled"].astype(bool),
            constant=values["constant"].astype(bool),
            ymean=float(values["ymean"]),
            yscale=float(values["yscale"]),
            shift=values["shift"].astype(np.float64),
            offset=values["offset"].astype(np.float64),
        )
        return EngineUnitFit(fitted, stats, metric, bound, params.transform, params.inverse, None, params, None)
    if kind != "vst":
        raise StoredFitError(f"an engine fit of kind {kind!r}")
    scaler = BaseScaler.from_state(_section(state, "scaler"))
    vst = np.array(state["vst"], dtype=bool)
    scaled = np.array(state["scaled"], dtype=bool)
    mean = np.array(state["mean"], dtype=np.float64)
    std = np.array(state["std"], dtype=np.float64)
    params = iter(_unpacked(state, "params", int((scaled & vst).sum())))
    columns = [
        (next(params) if vst[j] else None, np.float64(mean[j]), np.float64(std[j])) if scaled[j] else None
        for j in range(scaled.size)
    ]
    rowset = _RowSetView(scaler, float(state["typical"]), float(state["rowset_yscale"]))
    transform = _vst_row_transform(scaler, columns, vst)
    return EngineUnitFit(fitted, stats, metric, bound, transform, scaler.inverse, rowset, None, None)


# ---------------------------------------------------------------------------------------------------------------------------
# One unit's stored fit
# ---------------------------------------------------------------------------------------------------------------------------

# (period, horizon, the hash of the unit's list, its window's place among the model's): how a unit is found.
UnitKey = Tuple[int, int, str, int]


def key_text(key: UnitKey) -> str:
    return f"{key[0]}/{key[1]}/{key[2]}/{key[3]}"


@dataclass
class UnitFit:
    """A unit's fit as stored: its key and anchor, what `fit.json` says of it (`meta`: its training skip or `fit_error`, the
    newest target relative to the anchor, a matching list's record fields, its rows), its scaler and its model's fit (None for a
    skip or a fit_error), and its fit from shared statistics (None without the engine, or where the engine handed it back).
    `_worker._fit_unit(stored=)` and `_fit_member(stored=)` forecast from it."""

    key: UnitKey
    anchor: datetime.date
    meta: Dict[str, Any]
    scaler: Optional[BaseScaler] = None
    fitted: Any = None
    engine: Any = None

    @property
    def skip(self) -> Optional[str]:
        return self.meta.get("skip")

    @property
    def fit_error(self) -> Optional[str]:
        return self.meta.get("fit_error")

    def training_of(self, arrays: Dict, predictors: Sequence[str]) -> Any:
        """What the stored unit says of its training rows, as `_worker.unit_training` gives it: its training skip, or a
        `Training` holding its newest target, its list's places in `arrays` and a matching list's record fields."""
        from ._worker import Training

        if self.skip is not None:
            return self.skip
        col_idx = [arrays["col_index"][name] for name in predictors]
        width = self.meta.get("columns")
        if width is not None and width != len(col_idx):
            raise StoredFitError(
                f"the stored unit {key_text(self.key)} of {self.anchor} was fitted on {width} columns, its list now has "
                f"{len(col_idx)}"
            )
        extra = self.meta.get("extra")
        return Training(None, int(self.meta["newest"]), list(predictors), col_idx, None, None, None, None, None, extra)

    def forecasts(self, model: Any, rows: np.ndarray, flat: bool = True) -> Union[str, List[Tuple[float, list]]]:
        """(prediction in target units, coefficients) of each raw issue row of `rows` (k, columns), as the backtest makes it: the
        engine's where it has a fit and the row isn't too far out for it (`EngineUnitFit.forecast`), the model path's for the
        others, every row through the scaler at once (`transform`, value by value) and `_predict_fitted` one row at a time, then
        the inverse. The stored `fit_error` where a row needs the model path and its fit failed, as `_fit_unit` gives it to every
        forecast of the call. `flat=False` keeps `_predict_fitted`'s coefficients as they are, as `_fit_member` does."""
        from ._worker import flat_coefficients

        rows = np.ascontiguousarray(
            rows, dtype=np.float64
        )  # as `_fit_unit`'s: a strided row takes another product kernel
        out: List[Optional[Tuple[float, list]]] = [None] * rows.shape[0]
        if self.engine is not None:
            out = [self.engine.forecast(model, rows[row : row + 1]) for row in range(rows.shape[0])]
        if all(row is not None for row in out):
            return out
        if self.fit_error is not None:
            return self.fit_error
        scaled = self.scaler.transform(rows)
        for row in range(rows.shape[0]):
            if out[row] is None:
                prediction, coefs = model._predict_fitted(self.fitted, scaled[row : row + 1])
                coefs = flat_coefficients(coefs, model, "_predict_fitted") if flat else coefs
                out[row] = (self.scaler.inverse(float(prediction)), coefs)
        return out


def unit_from_arrays(
    key: UnitKey, anchor: datetime.date, meta: Dict, arrays: Mapping[str, np.ndarray], model: Any
) -> UnitFit:
    """A `UnitFit` from what a fit job made or `units.npz` holds of it (`scaler.*`, `model.*`, `engine.*`)."""
    scaler = BaseScaler.from_state(_section(arrays, "scaler")) if "scaler.spec" in arrays else None
    fitted = model._from_state(_section(arrays, "model")) if "model.kind" in arrays else None
    engine = engine_fit_from_state(_section(arrays, "engine"), model) if "engine.kind" in arrays else None
    if meta.get("skip") is None and meta.get("fit_error") is None and (scaler is None or fitted is None):
        raise StoredFitError(f"the stored unit {key_text(key)} of {anchor} has no scaler or no fit, and no skip")
    return UnitFit(key, anchor, meta, scaler, fitted, engine)


# ---------------------------------------------------------------------------------------------------------------------------
# Fitting units: the worker's side
# ---------------------------------------------------------------------------------------------------------------------------


class FitItem(NamedTuple):
    """One unit a fit job fits: a refit model's `FitUnit` (`task` None), or one window member of a refit=None model's task (hour,
    hz, day_in_test), whose `unit` is `FitUnit(period, horizon, issue date, list id, member's place)`."""

    unit: Any
    task: Optional[tuple] = None


def _item_rows(item: FitItem, inputs: Any) -> Tuple[slice, List[int], int, Any, Optional[int], List[str], Any]:
    """(the span's rows before any filter, the list's places, the newest target, the member's window, the list id, the list, the
    parent's mask or None) of an item, as `_fit_unit` and `_fit_one_numpy` select them."""
    from ._worker import member_rows, task_list, task_mask, task_newest, task_place, unit_rows

    cfg = inputs.config
    unit = item.unit
    hour, hz = unit.period, unit.horizon
    arrays = inputs.hour_arrays[hour]
    members = cfg.get("members")
    if item.task is None:
        rows, newest = unit_rows(cfg, arrays["days"], unit)
        predictors = inputs.expanded[(hour, hz) if unit.list_id is None else unit.list_id]
        window = cfg["training_window"] if members is None else members[unit.member]
        mask = None
    else:
        window_name, _, position = task_place(cfg, item.task[2])
        newest = task_newest(cfg, hour, position)
        list_id, predictors, mask_of = task_list(cfg, inputs.expanded, hour, hz, position)
        window = cfg["training_window"] if members is None else members[unit.member]
        mask = task_mask(cfg, window_name, mask_of) if members is None else task_mask(cfg, window_name, mask_of, window)
        rows = member_rows(arrays["days"], cfg["offset"] + item.task[2], window, hz, newest)
    col_idx = [arrays["col_index"][name] for name in predictors]
    return rows, col_idx, newest, window, unit.list_id, list(predictors), mask


def _job_engine(items: Sequence[FitItem], inputs: Any) -> Tuple[Any, List[List[tuple]]]:
    """The engine's fits of a job's items (`_engine_worker.EngineFits` on spans or row sets planned for them, as `run_job`
    plans a job's) and the span keys of each item, or (None, []) without fits from shared statistics."""
    from collections import Counter

    from ._engine_worker import EngineCounts, EngineFits, _planned_rowset, _Spans, inputs_of_model

    if inputs.config.get("fit_path") is None:
        return None, []
    model_inputs = inputs_of_model("refit", inputs)
    if not model_inputs.from_stats:
        return None, []
    keys: List[List[tuple]] = []
    columns: Dict[tuple, set] = {}
    for item in items:
        rows, col_idx = _item_rows(item, inputs)[:2]
        if rows.stop <= rows.start:
            keys.append([])
            continue
        key = (item.unit.period, item.unit.horizon, rows.start, rows.stop)
        if model_inputs.vst is not None:
            key = _planned_rowset(inputs.hour_arrays[key[0]], key, col_idx, model_inputs.vst)
        keys.append([key])
        columns.setdefault(key, set()).update(col_idx)
    planned = {key: np.array(sorted(cols), dtype=np.int64) for key, cols in columns.items()}
    spans = _Spans(inputs.hour_arrays, planned, Counter(key for item_keys in keys for key in item_keys))
    return EngineFits(spans, model_inputs, EngineCounts()), keys


def _fit_item(item: FitItem, inputs: Any, engine: Any) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """(what `fit.json` keeps of the item, its arrays) of one unit's fit: the backtest's rows (`unit_training` or
    `member_training`), then the engine's fit from shared statistics where it makes one, and the model path's fit (its scaler on
    those rows and `_fit_model`), as `_fit_unit` and `_fit_member` fit them."""
    from ._worker import (
        SKIP_FIT_ERROR,
        StatsFit,
        _unit_list_hash,
        fit_scope,
        make_scaler,
        member_training,
        member_unit,
        unit_training,
    )

    cfg, model = inputs.config, inputs.model
    unit = item.unit
    hour, hz = unit.period, unit.horizon
    arrays = inputs.hour_arrays[hour]
    if item.task is None:
        training = unit_training(unit, inputs)
        scope = unit
        window = _item_rows(item, inputs)[3]
    else:
        _, _, newest, window, list_id, predictors, mask = _item_rows(item, inputs)
        day = cfg["offset"] + item.task[2]
        training = member_training(arrays, cfg, hour, hz, day, window, newest, list_id, predictors, mask)
        scope = member_unit(cfg, hour, hz, day, list_id, window)
    if isinstance(training, str):
        return {"skip": training, "fit_error": None, "engine": False}, {}
    first = datetime.date.fromisoformat(cfg["test_start"]).toordinal() - int(cfg["offset"])
    meta: Dict[str, Any] = {
        "skip": None,
        "fit_error": None,
        "engine": False,
        "columns": len(training.col_idx),
        "rows": int(training.x.shape[0]),
        "first_row": datetime.date.fromordinal(first + int(training.row_days.min())).isoformat(),
        "last_row": datetime.date.fromordinal(first + int(training.row_days.max())).isoformat(),
        "extra": training.extra,
    }
    out: Dict[str, np.ndarray] = {}
    empty = np.empty((0, len(training.col_idx)))
    if engine is not None:
        fit = StatsFit(
            hour, hz, training.rows, training.col_idx, training.finite, training.x, training.y, training.mask, empty
        )
        made = engine.fit_of(fit)
        if made is not None:
            out.update(_prefixed("engine", engine_fit_state(made, model)))
            meta["engine"] = True
    scaler = make_scaler(cfg.get("scaler", "standard"))
    x, y, _ = scaler.fit_transform(
        training.x.copy(), training.y.copy(), empty, scalable_mask=training.mask, columns=training.predictors
    )
    try:
        with fit_scope(scope, window, _unit_list_hash(cfg, unit.list_id), training.row_days):
            fitted = model._fit_model(x, y)
    except ValueError as exc:
        meta["fit_error"] = f"{SKIP_FIT_ERROR}: {exc}"
        return meta, out
    out.update(_prefixed("scaler", scaler.state()))
    out.update(_prefixed("model", model._fit_state(fitted)))
    return meta, out


def fit_items(items: Sequence[FitItem], inputs: Any = None) -> List[Tuple[int, Dict[str, Any], Dict[str, np.ndarray]]]:
    """Fit a job's items with `inputs` (the pool's if None): (the item's place in `items`, what `fit.json` keeps, its arrays) of
    each. A pool job: nothing is predicted, and nothing of a fit depends on another item of the job."""
    from ._worker import TaskFailed, _inputs_of

    inputs = _inputs_of(inputs)
    engine, keys = _job_engine(items, inputs)
    out = []
    for place, item in enumerate(items):
        try:
            meta, arrays = _fit_item(item, inputs, engine)
        except Exception as exc:
            import traceback

            raise TaskFailed(f"the fit of {item.unit} raised {exc!r}\n{traceback.format_exc()}") from None
        out.append((place, meta, arrays))
        if engine is not None:
            engine.spans.done(keys[place])
    return out


# ---------------------------------------------------------------------------------------------------------------------------
# The store: fits/<model>/<anchor>/{units.npz, fit.json}
# ---------------------------------------------------------------------------------------------------------------------------


def model_dir(save_dir: Union[str, Path], model_name: str) -> Path:
    """`fits/<model>` of a live directory, named as the model's results are (`ModelPipeline.results_path`)."""
    return Path(save_dir) / FITS_DIR / model_name.lower().replace(" ", "_")


def fit_dir(save_dir: Union[str, Path], model_name: str, anchor: datetime.date) -> Path:
    return model_dir(save_dir, model_name) / anchor.isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:  # a file system that can't sync a directory
        pass
    finally:
        os.close(fd)


def write_fit(
    directory: Path, meta: Dict[str, Any], units: Sequence[Tuple[Dict[str, Any], Dict[str, np.ndarray]]]
) -> None:
    """Write an anchor's fit: every unit's arrays into `units.npz` (`packed`: one array per name over the units, numpy's own
    format, no object array), then `fit.json` with `meta`, each unit's entry and the sha256 of `units.npz`, both synced, in a
    directory of their own that replaces `directory` whole (a reader finds the old fit, the new one or none, never a mix)."""
    arrays = packed([unit_arrays for _, unit_arrays in units])
    directory.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:12]
    staging = directory.parent / f".{directory.name}.new-{token}"
    staging.mkdir()
    try:
        with open(staging / UNITS_FILE, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        record = {
            **meta,
            "units": [unit_meta for unit_meta, _ in units],
            "artifacts": {UNITS_FILE: _sha256(staging / UNITS_FILE)},
        }
        with open(staging / FIT_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, indent=1, sort_keys=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        _fsync(staging)
        if directory.exists():
            old = directory.parent / f".{directory.name}.old-{token}"
            os.replace(directory, old)
            os.replace(staging, directory)
            shutil.rmtree(old, ignore_errors=True)
        else:
            os.replace(staging, directory)
        _fsync(directory.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def packed(units: Sequence[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """The arrays of every unit by name, one flat array per name (`<name>/data`, in unit order, each array raveled in its own
    memory order) with the units that have it (`/units`), where each one's values start (`/offsets`), its shape (`/shapes`, -1
    padded) and whether it is Fortran-ordered (`/fortran`), so that `unpacked` gives back each array with its values, dtype kind,
    shape and memory order (a BLAS call on a transposed copy could round otherwise): a few dozen arrays in the file whatever the
    number of units, which keeps loading one anchor's fit to milliseconds."""
    by_name: Dict[str, List[Tuple[int, np.ndarray]]] = {}
    for place, arrays in enumerate(units):
        for name, value in arrays.items():
            value = np.asarray(value)
            if value.dtype == object:
                raise TypeError(f"a stored fit holds arrays of numbers or text only, got an object array {name}")
            by_name.setdefault(name, []).append((place, value))
    out: Dict[str, np.ndarray] = {}
    for name, entries in by_name.items():
        kinds = {value.dtype.kind for _, value in entries}
        if len(kinds) != 1:
            raise TypeError(f"a stored fit's {name} has values of several kinds {sorted(kinds)}")
        fortran = [value.ndim > 1 and value.flags.f_contiguous and not value.flags.c_contiguous for _, value in entries]
        flat = [value.ravel(order="F" if f else "C") for (_, value), f in zip(entries, fortran)]
        ndim = max(value.ndim for _, value in entries)
        shapes = np.full((len(entries), max(ndim, 1)), -1, dtype=np.int64)
        for row, (_, value) in enumerate(entries):
            shapes[row, : value.ndim] = value.shape
        out[f"{name}/data"] = np.concatenate(flat) if flat else np.zeros(0)
        out[f"{name}/units"] = np.array([place for place, _ in entries], dtype=np.int64)
        out[f"{name}/offsets"] = np.concatenate([[0], np.cumsum([piece.size for piece in flat])]).astype(np.int64)
        out[f"{name}/shapes"] = shapes
        out[f"{name}/ndim"] = np.array([value.ndim for _, value in entries], dtype=np.int64)
        out[f"{name}/fortran"] = np.array(fortran, dtype=bool)
    return out


def unpacked(arrays: Mapping[str, np.ndarray], count: int) -> List[Dict[str, np.ndarray]]:
    """Each of `count` units' arrays, as `packed` stored them (views of the flat arrays, in their own memory order)."""
    out: List[Dict[str, np.ndarray]] = [{} for _ in range(count)]
    names = {key.rpartition("/")[0] for key in arrays if key.endswith("/data")}
    for name in names:
        data = arrays[f"{name}/data"]
        units = arrays[f"{name}/units"]
        offsets = arrays[f"{name}/offsets"]
        shapes = arrays[f"{name}/shapes"]
        ndims = arrays[f"{name}/ndim"]
        fortran = arrays[f"{name}/fortran"]
        if (
            not (len(units) == len(ndims) == len(fortran) == len(shapes) == len(offsets) - 1)
            or offsets[-1] != data.size
        ):
            raise StoredFitError(f"the stored arrays of {name} don't agree in length")
        for row, place in enumerate(units.tolist()):
            if not 0 <= place < count:
                raise StoredFitError(f"the stored arrays of {name} name unit {place} of {count}")
            shape = tuple(int(n) for n in shapes[row, : int(ndims[row])])
            piece = data[int(offsets[row]) : int(offsets[row + 1])]
            out[place][name] = piece.reshape(shape, order="F" if fortran[row] else "C")
    return out


def _npz_units(payload: Union[bytes, Path], count: int) -> List[Dict[str, np.ndarray]]:
    import io

    source = io.BytesIO(payload) if isinstance(payload, bytes) else payload
    with np.load(source, allow_pickle=False) as npz:
        arrays = {name: npz[name] for name in npz.files}
    return unpacked(arrays, count)


@dataclass
class StoredFit:
    """An anchor's stored fit, read and checked (`read_fit`): its `fit.json` and its units by key."""

    directory: Path
    anchor: datetime.date
    meta: Dict[str, Any]
    units: Dict[UnitKey, UnitFit] = field(default_factory=dict)


class UnreadableFitError(StoredFitError):
    """An anchor's `fit.json` that can't be read as JSON: what an interrupted copy or a disk error leaves. `refit` refits the
    anchor; serving treats it as no fit, with a warning."""


def read_meta(directory: Path) -> Optional[Dict[str, Any]]:
    """`fit.json` of an anchor's directory, None where there is none; `UnreadableFitError` where it isn't JSON of an object. Its
    format isn't judged here: `read_fit` takes another format for another configuration."""
    path = directory / FIT_FILE
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UnreadableFitError(f"{path} can't be read: {exc}") from None
    if not isinstance(meta, dict):
        raise UnreadableFitError(f"{path} isn't a stored fit's description")
    return meta


_ANY = object()


def read_fit(directory: Path, model: Any, digest: Optional[str] = None, kernels: Any = _ANY) -> Optional[StoredFit]:
    """The stored fit in `directory`, checked: None where there is none, or it is of another format, or (with `digest`) of
    another configuration, or (with `kernels`) made with another LEAR kernel backend, whose fits agree with this one's within
    1e-10, not to the bit; `UnreadableFitError` where `fit.json` can't be read; `StoredFitError` where `units.npz` isn't the file
    `fit.json` names (its sha256), or a unit's arrays don't build back. A fit made by another package version warns: its
    numbers are the code's of then, which a backtest by this one may not repeat."""
    from ..results.manifest import _package_version

    meta = read_meta(directory)
    if meta is None or meta.get("format") != FORMAT:
        return None
    if digest is not None and meta.get("config_digest") != digest:
        return None
    if kernels is not _ANY and meta.get("kernels") != kernels:
        return None
    version = _package_version()
    if meta.get("package_version") != version:
        warnings.warn(
            f"{directory}: the stored fit was made by epftoolbox2 {meta.get('package_version')}, this is {version}: a backtest "
            "by this version may not give its forecasts to the bit. Refit it (refit(..., if_due=False)) to be sure.",
            UserWarning,
            stacklevel=3,
        )
    path = directory / UNITS_FILE
    expected = (meta.get("artifacts") or {}).get(UNITS_FILE)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise StoredFitError(f"{path} can't be read: {exc}") from None
    if hashlib.sha256(payload).hexdigest() != expected:
        raise StoredFitError(
            f"{path} isn't the file {directory / FIT_FILE} names (sha256 {expected}): it was changed after the fit was stored. "
            "Refit this anchor (ModelPipeline.refit(..., if_due=False))."
        )
    entries = meta.get("units", [])
    try:
        groups = _npz_units(payload, len(entries))
    except (ValueError, KeyError, OSError) as exc:
        raise StoredFitError(f"{path} can't be read: {exc}") from None
    anchor = datetime.date.fromisoformat(meta["anchor"])
    stored = StoredFit(directory, anchor, meta)
    for place, unit_meta in enumerate(entries):
        key = tuple(unit_meta["key"])
        try:
            stored.units[key] = unit_from_arrays(key, anchor, unit_meta, groups[place], model)
        except (KeyError, ValueError, TypeError) as exc:
            raise StoredFitError(f"{directory}: unit {key_text(key)} doesn't build back: {exc}") from None
    return stored


def _left_by_a_write(root: Path, anchor: str) -> Optional[Path]:
    """The old fit of `anchor` that a write stopped between its two renames left (`write_fit`: `.<anchor>.old-<token>`), where
    the anchor's own directory is gone: the newest such copy with a `fit.json`, else None."""
    olds = [path for path in root.glob(f".{anchor}.old-*") if (path / FIT_FILE).exists()]
    return max(olds, key=lambda path: path.stat().st_mtime_ns) if olds else None


def anchor_dir(save_dir: Union[str, Path], model_name: str, anchor: datetime.date) -> Path:
    """Where the anchor's fit is read from: its directory, or where a write stopped between its two renames, the old fit it kept
    (read as it is: only `sweep`, under the live lock, puts it back)."""
    directory = fit_dir(save_dir, model_name, anchor)
    if directory.exists():
        return directory
    return _left_by_a_write(directory.parent, anchor.isoformat()) or directory


def sweep(root: Path) -> None:
    """What unfinished writes left in a model's `fits/<model>/`, cleared under the live directory's lock: an old fit whose
    anchor's directory is gone (a write stopped between its renames) is put back, and every other `.<anchor>.new-*` (a staging
    directory) and `.<anchor>.old-*` is removed."""
    if not root.is_dir():
        return
    leftovers = [
        path for path in root.iterdir() if path.name.startswith(".") and (".new-" in path.name or ".old-" in path.name)
    ]
    for anchor in sorted({path.name[1:].split(".", 1)[0] for path in leftovers}):
        if not (root / anchor).exists():
            old = _left_by_a_write(root, anchor)
            if old is not None:
                os.replace(old, root / anchor)
    for path in leftovers:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    _fsync(root)


def stored_anchors(save_dir: Union[str, Path], model_name: str) -> List[datetime.date]:
    """The anchors with a stored fit of the model in the live directory (or an old one a stopped write kept), in date order."""
    root = model_dir(save_dir, model_name)
    if not root.is_dir():
        return []
    out = set()
    for path in root.iterdir():
        name = path.name[1:].split(".", 1)[0] if path.name.startswith(".") and ".old-" in path.name else path.name
        try:
            day = datetime.date.fromisoformat(name)
        except ValueError:
            continue  # a staging directory of an unfinished write
        if (path / FIT_FILE).exists():
            out.add(day)
    return sorted(out)


# How much older than the anchor it stands in for a fit `fit_missing="previous"` serves, at least: two refit periods of the
# model's coarsest rule where those are longer (`previous_cap`). An older fit isn't served; its forecasts are `no_stored_fit`
# skips, and the anchor's `source` says the cap stopped it.
PREVIOUS_MAX_DAYS = 31


def previous_cap(model: Any) -> int:
    """The age in days, from the anchor it stands in for, up to which an older fit is served: `PREVIOUS_MAX_DAYS`, or two
    nominal periods of the model's coarsest refit rule (`_calendar.nominal_block_length`: k days of "kD", 31 k of "kMS") where
    that is more, so a model refitted every two months still has its last fit served after one missed refit."""
    from ._calendar import nominal_block_length

    rule = getattr(model, "_refit_block", None)
    return PREVIOUS_MAX_DAYS if rule is None else max(PREVIOUS_MAX_DAYS, 2 * nominal_block_length(rule))


def find_fit(
    save_dir: Union[str, Path],
    name: str,
    model: Any,
    anchor: datetime.date,
    digest: str,
    previous: bool = True,
    kernels: Any = _ANY,
    max_days: int = PREVIOUS_MAX_DAYS,
) -> Tuple[Optional[StoredFit], Optional[datetime.date]]:
    """(the stored fit of the model `name` of `anchor` with this configuration digest and kernel backend, or with `previous` the
    newest stored older one of the same at most `max_days` older (decision 15: a missed refit leaves the last fit in use, its age
    recorded, with a warning), or None; the newest older anchor of the same configuration passed over for being older than that,
    or None). A fit is never taken from an anchor after `anchor`, nor from another configuration. An unreadable `fit.json` is no
    fit, with a warning; an older anchor's tampered `units.npz` too, while the anchor's own raises `StoredFitError`. `model`
    builds the fits back (`_from_state`)."""
    try:
        found = read_fit(anchor_dir(save_dir, name, anchor), model, digest, kernels)
    except UnreadableFitError as exc:
        warnings.warn(f"{name}: no stored fit of {anchor}: {exc}", UserWarning, stacklevel=3)
        found = None
    if found is not None or not previous:
        return found, None
    beyond = None
    for older in reversed(stored_anchors(save_dir, name)):
        if older >= anchor:
            continue
        try:
            found = read_fit(anchor_dir(save_dir, name, older), model, digest, kernels)
        except StoredFitError as exc:
            warnings.warn(f"{name}: the stored fit of {older} is passed over: {exc}", UserWarning, stacklevel=3)
            continue
        if found is None:
            continue
        if (anchor - older).days > max_days:
            beyond = older
            break
        warnings.warn(
            f"{name}: no stored fit of {anchor}; the forecasts are served from the fit of {older}, {(anchor - older).days} "
            "day(s) older",
            UserWarning,
            stacklevel=3,
        )
        return found, None
    if beyond is not None:
        warnings.warn(
            f"{name}: no stored fit of {anchor}, and the newest older one, of {beyond}, is {(anchor - beyond).days} days older, "
            f"more than the {max_days} an older fit is served for: its forecasts are no_stored_fit skips",
            UserWarning,
            stacklevel=3,
        )
    return None, beyond


# ---------------------------------------------------------------------------------------------------------------------------
# The plan of an issue day: the run's own preparation, its units and their newest targets
# ---------------------------------------------------------------------------------------------------------------------------


@dataclass
class IssuePlan:
    """What a model fits and forecasts for one issue day: its run prepared as a backtest of that day alone
    (`BaseModel._prepare_run` with `test_start = test_end` the issue day: the plan, the checks, the masks, the newest targets),
    the worker inputs (`_worker.TaskInputs`, with `config["fit_path"]` when the engine fits it), every planned task and the
    pending ones (no `missing_test_row` or `not_published` decided before any fit), the units by anchor and key, the newest
    target of each unit relative to its anchor, and the configuration digest."""

    model: Any
    issue: datetime.date
    state: Any
    inputs: Any
    tasks: List[tuple]
    pending: List[tuple]
    items: Dict[datetime.date, Dict[UnitKey, FitItem]]
    newest: Dict[Tuple[datetime.date, UnitKey], int]
    digest: str
    fit_path: Optional[str]
    target: str
    freq: str
    information_set: Any
    hashes: Any
    used_columns: List[str]
    training_cut: Dict[str, Any]
    # The date of the first row of each unit's span, before any filter (None for an empty span), and the LEAR kernels' backend.
    first_rows: Dict[Tuple[datetime.date, UnitKey], Optional[datetime.date]] = field(default_factory=dict)
    kernels: Optional[str] = None


def _list_hash(cfg: Dict, expanded: Dict, hour: int, hz: int, list_id: Optional[int]) -> str:
    from ._worker import _list_hash_of

    return _list_hash_of(cfg, expanded, hour, hz, list_id)


def config_digest(
    model: Any, state: Any, data: pd.DataFrame, information_set: Any, fit_path: Optional[str], hashes: Any
) -> str:
    """The configuration a stored fit is of: the compared fields of the model's live manifest for this run (its manifest's
    `_COMPARED_FIELDS` without the data, `LIVE_UNCOMPARED`: the class, the lists' layout, the target, horizons, issue weekdays,
    freq, training window, parameters and scaler, the data's zone, the training cut with its refit rules and newest-target rule,
    the fit path, a carded family's runtime), with a predictor function's fingerprint, the chained columns' upstreams, what
    decides when an input is known in the information set, and this format, as a sha256."""
    from ..ensembles._compute import _what_decides_availability
    from ..results.manifest import _COMPARED_FIELDS, CHAINED_IDENTITY, LIVE_UNCOMPARED, build_manifest

    rules = model._cut_rules(information_set, None)
    manifest = build_manifest(
        model=model,
        data=data,
        used_columns=list(dict.fromkeys([state.target, *model._used_predictors])),
        expanded=state.expanded,
        target=state.target,
        horizon=state.horizon,
        horizons=state.horizons,
        freq=state.freq,
        test_start=state.test_start,
        test_end=state.test_end,
        information_set=information_set,
        predictor_function=None if state.plan is None else state.plan.function_record(),
        frame_hashes=hashes,
        training_cut=model._training_cut(rules, state.target, state.freq),
        issue_days=state.weekdays,
        fit_path=fit_path,
        chained_inputs=getattr(model, "_chained_record", None),
    ).to_dict()
    kept = {name: manifest.get(name) for name in _COMPARED_FIELDS if name not in LIVE_UNCOMPARED}
    kept["predictor_function"] = manifest.get("predictor_function")
    kept["chained_inputs"] = {
        column: {key: entry.get(key) for key in CHAINED_IDENTITY}
        for column, entry in sorted((manifest.get("chained_inputs") or {}).items())
    }
    kept["information_set"] = _what_decides_availability(manifest.get("information_set"))
    kept["format"] = FORMAT
    return hashlib.sha256(json.dumps(kept, sort_keys=True, default=str).encode()).hexdigest()


def issue_manifest(plan: "IssuePlan", data: pd.DataFrame) -> Any:
    """The manifest of the issue day `plan` plans (`results.manifest.build_manifest`, as `config_digest` builds it, with the frame's
    hashes): what a live call writes as the model's live manifest (`results.manifest.check_and_write_live`), which compares the
    configuration and records the data as provenance."""
    from ..results.manifest import build_manifest

    model, state = plan.model, plan.state
    rules = model._cut_rules(plan.information_set, None)
    return build_manifest(
        model=model,
        data=data,
        used_columns=list(plan.used_columns),
        expanded=state.expanded,
        target=state.target,
        horizon=state.horizon,
        horizons=state.horizons,
        freq=state.freq,
        test_start=state.test_start,
        test_end=state.test_end,
        information_set=plan.information_set,
        predictor_function=None if state.plan is None else state.plan.function_record(),
        frame_hashes=plan.hashes,
        training_cut=model._training_cut(rules, state.target, state.freq),
        issue_days=state.weekdays,
        fit_path=plan.fit_path,
        chained_inputs=getattr(model, "_chained_record", None),
    )


def issue_lists(plan: "IssuePlan") -> Dict[Tuple[int, int], List[str]]:
    """Each planned key's list of columns, by (period, horizon): what the issue's input snapshot holds of the model
    (`pipelines._issue_records.snapshot`)."""
    from ._worker import task_list, task_place

    cfg = plan.inputs.config
    out: Dict[Tuple[int, int], List[str]] = {}
    for hour, hz, day_in_test in plan.tasks:
        position = task_place(cfg, day_in_test)[2]
        out[(int(hour), int(hz))] = [str(name) for name in task_list(cfg, plan.inputs.expanded, hour, hz, position)[1]]
    return out


def _rows_sha256(plan: "IssuePlan", anchor: datetime.date, keys: Iterable[UnitKey]) -> str:
    """The sha256 of these units' `unit_rows_hash`es, in key order: what their fits read."""
    hashes = hashlib.sha256()
    for key in _sorted_keys(keys):
        hashes.update(f"{key_text(key)}={unit_rows_hash(plan, anchor, key)}\n".encode())
    return hashes.hexdigest()


def _fit_identity(
    plan: "IssuePlan",
    anchor: datetime.date,
    stored: Optional["StoredFit"],
    inline: Sequence[UnitKey] = (),
) -> Dict[str, Any]:
    """What a served forecast's fit was made from (the training data's identity of an issue's records): a stored fit's `fit.json`
    sha256 and configuration digest, with `inline_rows_sha256` over the units the stored anchor lacked and this call fitted
    (`inline`); an inline fit's rows (`anchor_rows`) and the sha256 of its units' `unit_rows_hash`es."""
    if stored is not None:
        path = stored.directory / FIT_FILE
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            digest = None
        out = {"fit_json_sha256": digest, "config_digest": stored.meta.get("config_digest")}
        if inline:
            out["inline_rows_sha256"] = _rows_sha256(plan, anchor, inline)
        return out
    return {
        "config_digest": plan.digest,
        "rows": anchor_rows(plan, anchor),
        "rows_sha256": _rows_sha256(plan, anchor, plan.items.get(anchor, {})),
    }


def plan_issue(
    model: Any,
    data: pd.DataFrame,
    issue: datetime.date,
    *,
    target: str,
    horizon: int,
    horizons: Sequence[int],
    freq: str,
    information_set: Any,
    availability: str,
    issue_days: Optional[Sequence[str]],
    engine: bool,
) -> IssuePlan:
    """The `IssuePlan` of `model` on `issue`. `engine` True fits it from shared statistics where an engine group would
    (`ModelPipeline(engine="on")`: `_engine_capable` and a fit path, `_engine_worker.fit_path_of`)."""
    from ..results.manifest import FrameHashes
    from ._calendar import FitUnit
    from ._engine_worker import fit_path_of
    from ._worker import block_units, task_inputs, task_newest, task_place, unit_rows

    day = issue.isoformat()
    _ = model._kernel_backend  # numba compiles here, before any pool forks
    # A backtest of the issue day alone, into no store: its plan, checks, masks and newest targets are that run's.
    state = model._prepare_run(
        data,
        day,
        day,
        target,
        None,
        freq,
        save_to=None,
        forecast_only=False,
        information_set=information_set,
        allow_manifest_mismatch=False,
        require_manifest=False,
        plan=None,
        availability=availability,
        warn_listing=False,
        windows=None,
        frame_hashes=None,
        cut_rules=None,
        horizons=list(horizons),
        issue_days=issue_days,
    )
    try:
        hour_arrays = model._build_hour_arrays(max(state.horizons))
        threads, _ = model._executor_config()
        expanded, config = model._worker_inputs(state, threads)
        fit_path = fit_path_of(model) if engine and model._engine_capable(state.plan) else None
        if fit_path is not None:
            config = {**config, "fit_path": fit_path}
        model_cls, model_kwargs = model._worker_model()
        inputs = task_inputs(hour_arrays, expanded, config, model_cls, model_kwargs)
        items: Dict[datetime.date, Dict[UnitKey, FitItem]] = {}
        newest: Dict[Tuple[datetime.date, UnitKey], int] = {}
        if config.get("refit") is None:
            members = config.get("members")
            for task in state.all_tasks:
                hour, hz, day_in_test = task
                position = task_place(config, day_in_test)[2]
                task_lists = config.get("task_lists")
                list_id = None if task_lists is None else int(task_lists[(hour, hz)][position])
                hashed = _list_hash(config, expanded, hour, hz, list_id)
                for place in range(1 if members is None else len(members)):
                    key = (hour, hz, hashed, place)
                    items.setdefault(issue, {})[key] = FitItem(FitUnit(hour, hz, issue, list_id, place), task)
                    newest[(issue, key)] = task_newest(config, hour, position)
        else:
            for unit in block_units(config, state.all_tasks):
                key = (
                    unit.period,
                    unit.horizon,
                    _list_hash(config, expanded, unit.period, unit.horizon, unit.list_id),
                    unit.member,
                )
                items.setdefault(unit.anchor, {})[key] = FitItem(unit)
                newest[(unit.anchor, key)] = unit_rows(config, hour_arrays[unit.period]["days"], unit)[1]
        first_ordinal = datetime.date.fromisoformat(config["test_start"]).toordinal() - int(config["offset"])
        first_rows: Dict[Tuple[datetime.date, UnitKey], Optional[datetime.date]] = {}
        for anchor, units in items.items():
            for key, item in units.items():
                rows = _item_rows(item, inputs)[0]
                days = hour_arrays[key[0]]["days"]
                first_rows[(anchor, key)] = (
                    datetime.date.fromordinal(first_ordinal + int(days[rows.start])) if rows.stop > rows.start else None
                )
        hashes = FrameHashes(data)
        digest = config_digest(model, state, data, information_set, fit_path, hashes)
        used = list(dict.fromkeys([target, *model._used_predictors]))
        training_cut = model._training_cut(model._cut_rules(information_set, None), target, freq)
    finally:
        model._release()
    return IssuePlan(
        model,
        issue,
        state,
        inputs,
        list(state.all_tasks),
        list(state.tasks),
        items,
        newest,
        digest,
        fit_path,
        target,
        freq,
        information_set,
        hashes,
        used,
        training_cut,
        first_rows,
        model._kernel_backend,
    )


def check_published(plan: IssuePlan, anchors: Iterable[datetime.date], now: pd.Timestamp, data: pd.DataFrame) -> None:
    """Refuse (`FitTooEarlyError`) a fit of these anchors made before what it reads is out at `now` (decision 5 of the step 8
    spec), the message giving the publication time:

    - a training target not yet known by the target's rule: for each unit the cell of its newest target (anchor + min(k, h - 1),
      its period) and the 31 days before it, or under a rule over time (`over_time`) every target of its span, which such a rule
      can publish later than a newer one;
    - an input of its newest training row that its rule puts out by the anchor's issue time but not yet at `now`: a value read
      earlier would be another vintage, or missing. An input late even at the issue time is the configuration's own choice
      (`availability`), and one without a rule can't be judged: neither refuses.

    Without an information set, or a rule for the target, nothing tells when a target is out: that raises too."""
    from ..information_set import _STEP_MINUTES, InformationSetError, over_time

    info = plan.information_set
    model = plan.model
    if info is None:
        raise FitTooEarlyError(
            f"{model.name}: a stored fit is made only where the target's rule says when its training targets are out, and the "
            "pipeline has no information set: pass issue_time= or information_set= to ModelPipeline"
        )
    step = _STEP_MINUTES[plan.freq]
    col = info._resolve_for(plan.target, step)
    if col.rule is None:
        raise InformationSetError(
            f"{model.name}: a stored fit is made only where the target's rule says when its training targets are out, and "
            f"'{plan.target}' has none. Declare one, e.g. InformationSet(rules={{'{plan.target}': "
            "Availability.published('13:00', 1)}})."
        )
    whole_span = over_time(col.rule)
    zone = info._issue_zone(data.index)

    def when(instant: pd.Timestamp) -> str:
        return f"{instant.tz_convert(zone):%Y-%m-%d %H:%M} {zone} ({instant.tz_convert('UTC'):%Y-%m-%d %H:%M} UTC)"

    for anchor in sorted(set(anchors)):
        units = plan.items.get(anchor, {})
        if not units:
            continue
        reach: Dict[int, int] = {}
        back: Dict[int, int] = {}
        horizon_of: Dict[int, int] = {}
        for key in units:
            period, hz = key[0], key[1]
            offset = min(int(plan.newest[(anchor, key)]), hz - 1)
            if offset >= reach.get(period, offset):
                reach[period], horizon_of[period] = offset, hz
            first = plan.first_rows.get((anchor, key))
            oldest = offset - 31
            if whole_span and first is not None:
                oldest = min(oldest, (first + datetime.timedelta(days=hz) - anchor).days)
            back[period] = min(back.get(period, oldest), oldest)
        clock = info._issue_clock(pd.DatetimeIndex([pd.Timestamp(anchor)]), data.index, plan.freq)
        periods = np.array(sorted(reach), dtype=np.int64)
        known: Dict[int, np.ndarray] = {}
        for offset in range(min(back.values()), max(reach.values()) + 1):
            known[offset] = np.asarray(clock.available(col, periods, step, row_day=offset)[:, 0], dtype=float)
        for place, period in enumerate(periods.tolist()):
            latest, day = -np.inf, None
            for offset in range(back[period], reach[period] + 1):
                minutes = known[offset][place]
                if np.isfinite(minutes) and minutes >= latest:
                    latest, day = minutes, anchor + datetime.timedelta(days=offset)
            if day is None:
                continue
            out = clock.ctx.origin + pd.Timedelta(minutes=float(latest))
            if out > now:
                raise FitTooEarlyError(
                    f"{model.name}: the fit anchored on {anchor} trains on '{plan.target}' of {day} (period {period}, horizon "
                    f"{horizon_of[period]}), which its rule ('{col.rule.describe()}') publishes at {when(out)}, after the "
                    f"clock's now ({now.tz_convert('UTC'):%Y-%m-%d %H:%M} UTC). A fit made now would train on fewer targets "
                    "than the backtest's: refit once it is out."
                )
        issue = float(clock.issue[0])
        cells: Dict[Tuple[str, int], set] = {}
        for key, item in units.items():
            period, hz = key[0], key[1]
            row = min(int(plan.newest[(anchor, key)]) - hz, -1)  # the newest training row, relative to the anchor
            for name in _item_rows(item, plan.inputs)[5]:
                cells.setdefault((name, row), set()).add(period)
        for (name, row), wanted in sorted(cells.items()):
            try:
                predictor = info._resolve_for(name, step)
            except InformationSetError:
                continue
            if predictor.rule is None:
                continue
            ps = np.array(sorted(wanted), dtype=np.int64)
            minutes = np.asarray(clock.available(predictor, ps, step, row_day=row)[:, 0], dtype=float)
            for period, value in zip(ps.tolist(), minutes.tolist()):
                if not np.isfinite(value) or value > issue:
                    continue
                out = clock.ctx.origin + pd.Timedelta(minutes=value)
                if out > now:
                    raise FitTooEarlyError(
                        f"{model.name}: the fit anchored on {anchor} reads '{name}' on its newest training row, "
                        f"{anchor + datetime.timedelta(days=row)} (period {period}), which its rule "
                        f"('{predictor.rule.describe()}') publishes at {when(out)}, after the clock's now "
                        f"({now.tz_convert('UTC'):%Y-%m-%d %H:%M} UTC), though by the issue time: a value read now would be "
                        "another vintage or missing. Refit once it is out."
                    )


def incomplete_days(plan: IssuePlan, anchor: datetime.date, data: pd.DataFrame) -> List[str]:
    """The newest target days of an anchor's units whose target the frame doesn't hold whole (a slot without a row, or a
    missing value), though the rule says they are out (`check_published` passed): a fit then trains on fewer rows than the
    backtest of a complete frame. The fit is still made, marked `incomplete` in `fit.json` with a warning, and its rows hash
    makes a refit on a frame that holds them due (`refit` reports it `stale`)."""
    from ._calendar import clock_periods

    days = sorted(
        {
            anchor + datetime.timedelta(days=min(int(plan.newest[(anchor, key)]), key[1] - 1))
            for key in plan.items[anchor]
        }
    )
    local = pd.Series(data[plan.target].to_numpy(), index=pd.Index(data.index.date))
    out = []
    for day in days:
        values = local[local.index == day]
        expected = 1 if plan.freq == "1D" else len(clock_periods(day, plan.freq, data.index.tz))
        if len(values) < expected or values.isna().any():
            out.append(day.isoformat())
    return out


def run_fits(plan: IssuePlan, items: Sequence[FitItem]) -> List[Tuple[Dict[str, Any], Dict[str, np.ndarray]]]:
    """Fit `items` of `plan` in a pool of processes (`_worker._pool_init`: one BLAS thread each, as a backtest's), one job per
    period; (meta, arrays) of each, in the order of `items`."""
    import sys

    from ._worker import _pool_init, pool_context, pool_native_threads

    if not items:
        return []
    jobs: Dict[int, List[int]] = {}
    for place, item in enumerate(items):
        jobs.setdefault(item.unit.period, []).append(place)
    model = plan.model
    _, processes = model._executor_config()
    processes = max(1, min(processes, len(jobs)))
    if sys.platform == "win32":
        processes = min(processes, 61)
    inputs = plan.inputs
    model_cls, model_kwargs = model._worker_model()
    out: List[Optional[Tuple[Dict, Dict]]] = [None] * len(items)
    with (
        pool_native_threads(),
        ProcessPoolExecutor(
            max_workers=processes,
            mp_context=pool_context(),
            initializer=_pool_init,
            initargs=(inputs.hour_arrays, inputs.expanded, inputs.config, model_cls, model_kwargs),
        ) as pool,
    ):
        futures = {pool.submit(fit_items, [items[place] for place in places]): places for places in jobs.values()}
        try:
            for future in as_completed(futures):
                places = futures.pop(future)
                for local, meta, arrays in future.result():
                    out[places[local]] = (meta, arrays)
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
    return out  # type: ignore[return-value]


def unit_rows_hash(plan: IssuePlan, anchor: datetime.date, key: UnitKey) -> str:
    """The sha256 of exactly the cells a unit's fit reads: the dates of its span's rows (before any filter), its list's values
    on them and their targets at its horizon (and a per-row target's scale column), NaN in one bit pattern. Another value there
    (a target filled in, a revision) moves it; a row or column the fit doesn't read, such as the inputs of the issue day's row or
    a revised forecast on a row past the span, doesn't."""
    from ._row_target import scale_column

    item = plan.items[anchor][key]
    rows, col_idx = _item_rows(item, plan.inputs)[:2]
    cfg = plan.inputs.config
    arrays = plan.inputs.hour_arrays[key[0]]
    columns = list(col_idx)
    scale = scale_column(arrays, cfg.get("row_target"), key[1])
    if scale is not None:
        columns.append(scale)
    first = datetime.date.fromisoformat(cfg["test_start"]).toordinal() - int(cfg["offset"])
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(arrays["days"][rows] + first, dtype="<i8").tobytes())
    for values in (arrays["x"][rows][:, columns], arrays["y"][rows, key[1] - 1]):
        values = np.array(values, dtype="<f8", order="C")
        values[np.isnan(values)] = np.nan
        digest.update(values.tobytes())
    return digest.hexdigest()


def _unit_meta(plan: IssuePlan, anchor: datetime.date, key: UnitKey, made: Dict[str, Any]) -> Dict[str, Any]:
    newest = int(plan.newest[(anchor, key)])
    item = plan.items[anchor][key]
    cfg = plan.inputs.config
    members = cfg.get("members")
    return {
        "key": list(key),
        "window": cfg["training_window"] if members is None else members[item.unit.member],
        "newest": newest,
        "newest_target_day": (anchor + datetime.timedelta(days=min(newest, key[1] - 1))).isoformat(),
        "rows_hash": unit_rows_hash(plan, anchor, key),
        **made,
    }


MASK_ISSUE_DAY_NOTE = (
    "mask: a refit=None unit's binary-column mask is read on its issue day, as a backtest of that day alone reads it; a backtest "
    "over a longer test reads it on the test's first day, and the two differ where a column turns binary or stops being "
    "binary between them (binary_mask='first_day' only: the default reads each fit's own)"
)
INCOMPLETE_NOTE = (
    "incomplete: the fit of {anchor} was made from a frame without every '{target}' value of {days}, which its rule said was out: "
    "it trains on fewer rows than a backtest of a complete frame"
)
STALE_NOTE = (
    "stale: the fit of {anchor} read cells the frame now holds otherwise (a value filled in or revised): served as it was made, "
    "not as a backtest of this frame"
)
ENGINE_FIT_ERROR_NOTE = (
    "engine: {n} unit(s) have a fit from shared statistics and a model-path fit_error: an issue row too far out for the engine "
    "is the fit_error of every forecast served with it in one call, as in a backtest of that call's days; a backtest of a whole "
    "block gives it to every day of the block"
)


def anchor_notes(plan: IssuePlan, units: Sequence[Dict[str, Any]]) -> List[str]:
    """Where a forecast served from these units can differ from a backtest over a longer test, though it is a backtest of its
    issue day to the bit: a refit=None model's mask read on the issue day, and engine units whose model path fails."""
    notes = []
    if plan.model.refit is None and not plan.model._masks_from_fits:
        notes.append(MASK_ISSUE_DAY_NOTE)
    failing = sum(1 for unit in units if unit.get("engine") and unit.get("fit_error"))
    if failing:
        notes.append(ENGINE_FIT_ERROR_NOTE.format(n=failing))
    return notes


def anchor_rows(plan: IssuePlan, anchor: datetime.date) -> Dict[str, Any]:
    """Where an anchor's fits read (provenance): the used columns, from the first row of any unit's span to the newest target day
    of any unit. What they read is hashed per unit (`unit_rows_hash`)."""
    firsts = [day for (a, _), day in plan.first_rows.items() if a == anchor and day is not None]
    first = min(firsts, default=anchor).isoformat()
    through = max(
        (
            anchor + datetime.timedelta(days=min(int(plan.newest[(anchor, key)]), key[1] - 1))
            for key in plan.items[anchor]
        ),
        default=anchor,
    ).isoformat()
    return {"from": first, "through": through, "columns": list(plan.used_columns)}


def anchor_meta(
    plan: IssuePlan,
    anchor: datetime.date,
    units: Sequence[Dict[str, Any]],
    data: pd.DataFrame,
    incomplete: Sequence[str] = (),
) -> Dict[str, Any]:
    """`fit.json` of an anchor, but its units and artifacts: what the fit is of, what it read, which newest target days the
    frame didn't hold whole (`incomplete`), and where its forecasts can differ from a longer backtest's (`notes`)."""
    from ..results.manifest import _package_version

    model = plan.model
    return {
        "format": FORMAT,
        "model": model.name,
        "model_class": type(model).__name__,
        "anchor": anchor.isoformat(),
        "issue": plan.issue.isoformat(),
        "config_digest": plan.digest,
        "fit_path": plan.fit_path,
        "refit": model.refit,
        "training_cut": plan.training_cut,
        "target": plan.target,
        "freq": plan.freq,
        "kernels": plan.kernels,
        "rows": anchor_rows(plan, anchor),
        "incomplete": list(incomplete),
        "notes": anchor_notes(plan, units),
        "package_version": _package_version(),
        "created_at": current_clock().now().isoformat(),
    }


def stale_rows(stored: "StoredFit", plan: IssuePlan, anchor: datetime.date) -> bool:
    """Whether the cells a stored fit of `anchor` read are no longer the frame's: a unit's hash (`unit_rows_hash`) of the units
    the plan has moved (a value filled in, revised, or gone), or a unit recorded none."""
    for key, unit in stored.units.items():
        if key not in plan.items.get(anchor, {}):
            continue  # a unit of a list this day doesn't use: nothing of the frame to compare it with
        if unit.meta.get("rows_hash") != unit_rows_hash(plan, anchor, key):
            return True
    return False


def fitted_units(
    plan: IssuePlan, anchor: datetime.date, keys: Sequence[UnitKey]
) -> Tuple[List[Tuple[Dict, Dict]], Dict[UnitKey, UnitFit]]:
    """Fit the units `keys` of `anchor` (`run_fits`): (each one's `fit.json` entry and arrays, in order; the `UnitFit`s by key)."""
    made = run_fits(plan, [plan.items[anchor][key] for key in keys])
    entries, units = [], {}
    for key, (meta, arrays) in zip(keys, made):
        entry = _unit_meta(plan, anchor, key, meta)
        entries.append((entry, arrays))
        units[key] = unit_from_arrays(key, anchor, entry, arrays, plan.inputs.model)
    return entries, units


def _stored_entries(stored: StoredFit) -> List[Tuple[Dict, Dict[str, np.ndarray]]]:
    """A stored fit's units as `write_fit` takes them, read back from its own `units.npz` (checked by `read_fit`)."""
    entries = stored.meta.get("units", [])
    groups = _npz_units(stored.directory / UNITS_FILE, len(entries))
    return [(unit_meta, groups[place]) for place, unit_meta in enumerate(entries)]


# ---------------------------------------------------------------------------------------------------------------------------
# refit(): fit what is due and store it
# ---------------------------------------------------------------------------------------------------------------------------


@dataclass
class RefitResult:
    """What `ModelPipeline.refit` did for one issue day: per model the anchors it fitted and stored (`fitted`), those already
    stored with this configuration and these rows (`not_due`), those refitted because the rows their fit read moved (`stale`),
    those stored from a frame missing a newest target (`incomplete`), the units it fitted, and how long each took;
    `not_stored`, why a model's fits aren't stored (it fits at serve time). `noop` is True when nothing was fitted."""

    issue: str
    fitted: Dict[str, List[str]] = field(default_factory=dict)
    not_due: Dict[str, List[str]] = field(default_factory=dict)
    not_stored: Dict[str, str] = field(default_factory=dict)
    units: Dict[str, int] = field(default_factory=dict)
    seconds: Dict[str, float] = field(default_factory=dict)
    # Anchors refitted whole because the rows their stored fit read moved, and anchors stored from a frame missing a newest
    # target the rule calls out (`incomplete_days`).
    stale: Dict[str, List[str]] = field(default_factory=dict)
    incomplete: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def noop(self) -> bool:
        return not any(self.fitted.values())


def _sorted_keys(keys: Iterable[UnitKey]) -> List[UnitKey]:
    return sorted(keys, key=lambda key: (key[0], key[1], key[2], key[3]))


def refit_models(
    models: Sequence[Any],
    data: pd.DataFrame,
    save_dir: Path,
    issue: datetime.date,
    *,
    if_due: bool,
    target: str,
    horizons: Sequence[int],
    freq: str,
    information_set: Any,
    availability: str,
    issue_days: Optional[Sequence[str]],
    engine: bool,
    before_write: Optional[Any] = None,
) -> RefitResult:
    """`ModelPipeline.refit`'s work once the live directory's lock is held: every model's issue-day plan, which anchors are due,
    the publication check of every due fit before any is made (decision 5), then `before_write()` (the live directory's
    identity), each model's due units fitted (`run_fits`) and each anchor written (`write_fit`), merged with the units of the
    same configuration already stored there where the rows they read are still the frame's, else refitted whole (`stale`). An
    anchor whose stored fit can't be read, or is of another format, kernel backend or configuration, is refitted whole."""
    import time

    result = RefitResult(issue.isoformat())
    planned = []
    for model in models:
        reason = stores_fits(model)
        if reason is not None:
            result.not_stored[model.name] = reason
            continue
        plan = plan_issue(
            model,
            data,
            issue,
            target=target,
            horizon=max(horizons),
            horizons=horizons,
            freq=freq,
            information_set=information_set,
            availability=availability,
            issue_days=issue_days,
            engine=engine,
        )
        due: Dict[datetime.date, Tuple[List[UnitKey], Optional[StoredFit]]] = {}
        sweep(model_dir(save_dir, model.name))
        for anchor, units in sorted(plan.items.items()):
            directory = fit_dir(save_dir, model.name, anchor)
            try:
                stored = read_fit(directory, plan.inputs.model, plan.digest, plan.kernels)
            except StoredFitError as exc:
                warnings.warn(
                    f"{model.name}: the stored fit of {anchor} is refitted whole: {exc}", UserWarning, stacklevel=4
                )
                stored = None
            if stored is not None and stale_rows(stored, plan, anchor):
                result.stale.setdefault(model.name, []).append(anchor.isoformat())
                stored = None  # the rows it read moved: refitted whole, nothing of it kept
            held = set() if stored is None else set(stored.units)
            if if_due and set(units) <= held:
                result.not_due.setdefault(model.name, []).append(anchor.isoformat())
                continue
            keys = _sorted_keys(set(units) - held if if_due else set(units))
            due[anchor] = (keys, stored)
        planned.append((plan, due))
    now = current_clock().now()
    for plan, due in planned:
        check_published(plan, due, now, data)
    if before_write is not None:
        before_write()
    for plan, due in planned:
        name = plan.model.name
        started = time.perf_counter()
        result.fitted[name] = []
        result.units[name] = 0
        for anchor, (keys, stored) in due.items():
            entries, _ = fitted_units(plan, anchor, keys)
            kept = (
                {}
                if stored is None
                else dict(zip((tuple(m["key"]) for m, _ in _stored_entries(stored)), _stored_entries(stored)))
            )
            merged = {**kept, **{tuple(entry["key"]): (entry, arrays) for entry, arrays in entries}}
            ordered = [merged[key] for key in _sorted_keys(merged)]
            incomplete = incomplete_days(plan, anchor, data)
            if incomplete:
                result.incomplete.setdefault(name, []).append(anchor.isoformat())
                warnings.warn(
                    f"{name}: the frame doesn't hold every '{plan.target}' value of {', '.join(incomplete)}, which its rule says is "
                    f"out by now: the fit of {anchor} is stored without them, marked incomplete, and a refit on a frame that "
                    "holds them refits it (stale rows)",
                    UserWarning,
                    stacklevel=4,
                )
            meta = anchor_meta(plan, anchor, [m for m, _ in ordered], data, incomplete)
            write_fit(fit_dir(save_dir, name, anchor), meta, ordered)
            result.fitted[name].append(anchor.isoformat())
            result.units[name] += len(keys)
        result.seconds[name] = time.perf_counter() - started
    return result


# ---------------------------------------------------------------------------------------------------------------------------
# Serving: an issue day's forecasts from stored fits
# ---------------------------------------------------------------------------------------------------------------------------


@dataclass
class Served:
    """One model's forecasts of an issue day served from stored fits (`serve`): its records, forecasts and skip records as a
    backtest's store holds them (`_worker.forecast_record`, `skip_record`), and per anchor the fit used: `source` "stored" (the
    anchor's own), "previous" (the newest older one of the same configuration), "inline" (fitted by this call) or "missing"
    (none: its forecasts are `no_stored_fit` skips), `used`, the anchor of the fit served, and `fit_age_days`, the issue day
    less that anchor (`inline_units`: units an anchor's stored fit lacked, fitted by this call). `status` is "served", or
    "not_stored" for a model whose fits aren't stored (`reason` says why): it has no records here and fits at serve time through
    its own run of the issue day, which `forecast()` routes it to. `notes` say where a forecast served can differ from a backtest
    over a longer test, though it is the backtest of its issue day to the bit (`anchor_notes`)."""

    model: str
    issue: str
    records: List[Dict[str, Any]]
    fits: Dict[str, Dict[str, Any]]
    status: str = "served"
    reason: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    # What a live call records of the issue (step 8e): the model's manifest of the issue day (`issue_manifest`) and each planned
    # key's list (`issue_lists`); None for a model whose fits aren't stored.
    manifest: Any = field(default=None, repr=False)
    lists: Optional[Dict[Tuple[int, int], List[str]]] = field(default=None, repr=False)
    # Per anchor what the fit used was made from (`_fit_identity`): the training data's identity of the issue's records.
    identity: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)


def serve(
    model: Any,
    data: pd.DataFrame,
    issue: datetime.date,
    save_dir: Optional[Union[str, Path]],
    *,
    target: str,
    horizons: Sequence[int],
    freq: str,
    information_set: Any,
    availability: str,
    issue_days: Optional[Sequence[str]],
    engine: bool,
    fit_missing: str = "previous",
) -> Served:
    """The forecasts of `model` for `issue` from its stored fits in the live directory `save_dir` (what `forecast()` of 8e calls
    per model): the issue day planned as a backtest of that day alone (`plan_issue`), each anchor's stored fit of the current
    configuration digest checked on load (`read_fit`), and each forecast made by the backtest's own lines from it
    (`_worker._fit_unit(stored=)`, `_fit_one_numpy(stored=)`, `block_records`), on one BLAS thread as a pool's worker. Where an
    anchor has no stored fit: `fit_missing="previous"` serves the newest stored older one (a refit model; its age recorded in
    `Served.fits`, the records' `refit_anchor` the fit's own), `"fit"` fits it inline (refused, as `refit`, while a training
    target it reads isn't out), `"skip"` gives its forecasts as `no_stored_fit` skips; a refit=None model fits inline unless
    `"skip"`, as the backtest does. An older fit is served only within `previous_cap` days of the anchor it stands in for, with a
    warning; one past it leaves the anchor "missing (older fit beyond cap)", with a warning. A stored fit made from a frame
    missing a newest target says so (`incomplete`, a note, a warning); one whose cells the frame now holds otherwise is fitted
    inline with `"fit"` (`replaced: "stale"`), else served as it is, marked `stale`, with a note and a warning. Units the anchor's own stored fit lacks (a predictor function's list it wasn't fitted for) are fitted inline at the
    anchor with "previous" and "fit" (`inline_units` in the anchor's entry), and are `no_stored_fit` skips with "skip" or where
    an older fit stands in for the anchor. A model whose fits aren't stored (`stores_fits`) gets `status`
    "not_stored", no records: it fits at serve time through its own run."""
    from threadpoolctl import threadpool_limits

    from ._calendar import refit_anchor
    from ._worker import (
        NATIVE_THREADS_PER_TASK,
        _fit_one_numpy,
        _fit_unit,
        block_records,
        block_units,
        issue_row,
        skip_record,
        task_place,
    )

    if fit_missing not in FIT_MISSING:
        raise ValueError(f"fit_missing must be one of {list(FIT_MISSING)}, got {fit_missing!r}")
    reason = stores_fits(model)
    if reason is not None:
        return Served(model.name, issue.isoformat(), [], {}, status="not_stored", reason=reason)
    plan = plan_issue(
        model,
        data,
        issue,
        target=target,
        horizon=max(horizons),
        horizons=horizons,
        freq=freq,
        information_set=information_set,
        availability=availability,
        issue_days=issue_days,
        engine=engine,
    )
    inputs = plan.inputs
    cfg = inputs.config
    previous = fit_missing == "previous" and cfg.get("refit") is not None
    inline = fit_missing == "fit" or (fit_missing != "skip" and cfg.get("refit") is None)
    lookup: Dict[datetime.date, Optional[Tuple[Dict[UnitKey, UnitFit], datetime.date]]] = {}
    fits: Dict[str, Dict[str, Any]] = {}
    identity: Dict[str, Dict[str, Any]] = {}
    notes: List[str] = []
    cap = previous_cap(model)
    for anchor in sorted(plan.items):
        stored, beyond = (
            (None, None)
            if save_dir is None
            else find_fit(save_dir, model.name, inputs.model, anchor, plan.digest, previous, plan.kernels, cap)
        )
        if stored is not None and stored.anchor == anchor and stale_rows(stored, plan, anchor):
            if fit_missing == "fit":
                warnings.warn(
                    f"{model.name}: the stored fit of {anchor} read cells the frame now holds otherwise (a value filled in or "
                    "revised): fitted inline from the frame",
                    UserWarning,
                    stacklevel=3,
                )
                check_published(plan, [anchor], current_clock().now(), data)
                made, units = fitted_units(plan, anchor, _sorted_keys(plan.items[anchor]))
                notes.extend(anchor_notes(plan, [meta for meta, _ in made]))
                lookup[anchor] = (units, anchor)
                fits[anchor.isoformat()] = {
                    "source": "inline",
                    "used": anchor.isoformat(),
                    "fit_age_days": (issue - anchor).days,
                    "replaced": "stale",
                }
                identity[anchor.isoformat()] = _fit_identity(plan, anchor, None)
                continue
            warnings.warn(
                f"{model.name}: the stored fit of {anchor} read cells the frame now holds otherwise (a value filled in or "
                "revised), and is served as it is: refit, or serve with fit_missing='fit', for the frame's",
                UserWarning,
                stacklevel=3,
            )
            notes.append(STALE_NOTE.format(anchor=anchor))
            stale = True
        else:
            stale = False
        if stored is not None:
            units = dict(stored.units)
            entry = {
                "source": "stored" if stored.anchor == anchor else "previous",
                "used": stored.anchor.isoformat(),
                "fit_age_days": (issue - stored.anchor).days,
            }
            if stale:
                entry["stale"] = True
            incomplete = list(stored.meta.get("incomplete") or [])
            if incomplete:
                entry["incomplete"] = incomplete
                notes.append(
                    INCOMPLETE_NOTE.format(anchor=stored.anchor, days=", ".join(incomplete), target=plan.target)
                )
                warnings.warn(
                    f"{model.name}: the stored fit of {stored.anchor} was made from a frame without every '{plan.target}' "
                    f"value of {', '.join(incomplete)}: it trains on fewer rows than a backtest of a complete frame",
                    UserWarning,
                    stacklevel=3,
                )
            notes.extend(stored.meta.get("notes", []))
            # Units the anchor's own stored fit lacks (a predictor function's list it wasn't fitted for: the refit planned the
            # anchor's issue day before an input was out, and today's revision uses it) are fitted inline at the anchor, the
            # backtest's fit of them, under "previous" too: no older fit stands in for an anchor that is stored.
            missing = _sorted_keys(set(plan.items[anchor]) - set(units)) if stored.anchor == anchor else []
            inline_keys: List[UnitKey] = []
            if missing and fit_missing != "skip":
                check_published(plan, [anchor], current_clock().now(), data)
                units.update(fitted_units(plan, anchor, missing)[1])
                entry["inline_units"] = len(missing)
                inline_keys = missing
            identity[anchor.isoformat()] = _fit_identity(plan, anchor, stored, inline_keys)
            lookup[anchor] = (units, stored.anchor)
            fits[anchor.isoformat()] = entry
        elif inline:
            check_published(plan, [anchor], current_clock().now(), data)
            made, units = fitted_units(plan, anchor, _sorted_keys(plan.items[anchor]))
            notes.extend(anchor_notes(plan, [meta for meta, _ in made]))
            lookup[anchor] = (units, anchor)
            fits[anchor.isoformat()] = {
                "source": "inline",
                "used": anchor.isoformat(),
                "fit_age_days": (issue - anchor).days,
            }
            identity[anchor.isoformat()] = _fit_identity(plan, anchor, None)
        else:
            lookup[anchor] = None
            fits[anchor.isoformat()] = {
                "source": "missing" if beyond is None else "missing (older fit beyond cap)",
                "used": None,
                "fit_age_days": None,
                **({} if beyond is None else {"beyond_cap": beyond.isoformat(), "cap_days": cap}),
            }

    results: List[Dict] = []
    skipped: List[Tuple[tuple, str]] = []
    older: Dict[tuple, str] = {}  # (hour, horizon, run_date) -> the oldest older fit a forecast was served from
    first_day = datetime.date.fromisoformat(cfg["test_start"])
    with threadpool_limits(limits=NATIVE_THREADS_PER_TASK):
        if cfg.get("refit") is None:
            members = cfg.get("members")
            for task in plan.pending:
                hour, hz, day_in_test = task
                position = task_place(cfg, day_in_test)[2]
                task_lists = cfg.get("task_lists")
                list_id = None if task_lists is None else int(task_lists[(hour, hz)][position])
                hashed = _list_hash(cfg, inputs.expanded, hour, hz, list_id)
                found = lookup.get(issue)
                keys = [(hour, hz, hashed, place) for place in range(1 if members is None else len(members))]
                if found is None or any(key not in found[0] for key in keys):
                    skipped.append((task, SKIP_NO_STORED_FIT))
                    continue
                outcome = _fit_one_numpy(hour, hz, day_in_test, inputs, None, {k[3]: found[0][k] for k in keys})
                if isinstance(outcome, str):
                    skipped.append((task, outcome))
                else:
                    results.append(outcome)
        else:
            ran = []
            for unit, unit_tasks in block_units(cfg, plan.pending).items():
                key = (
                    unit.period,
                    unit.horizon,
                    _list_hash(cfg, inputs.expanded, unit.period, unit.horizon, unit.list_id),
                    unit.member,
                )
                found = lookup.get(unit.anchor)
                if found is None or key not in found[0]:
                    ran.append([(task, SKIP_NO_STORED_FIT, unit.member) for task in unit_tasks])
                    continue
                units, used = found
                outcomes = _fit_unit(unit._replace(anchor=used), unit_tasks, inputs, stored=units[key])
                ran.append([(task, outcome, unit.member) for task, outcome in outcomes])
                if used != unit.anchor:
                    for task in unit_tasks:
                        place = (task[0], task[1], (first_day + datetime.timedelta(days=task[2])).isoformat())
                        older[place] = min(older.get(place, used.isoformat()), used.isoformat())
            results, skipped = block_records(cfg, ran)
            for record in results:
                place = (record["hour"], record["horizon"], record["run_date"])
                if place in older:
                    record["refit_anchor"] = older[place]

    block = model._refit_block
    for task in [task for task in plan.tasks if task not in set(plan.pending)]:
        row = issue_row(inputs.hour_arrays[task[0]]["days"], cfg["offset"] + task[2])
        skipped.append((task, "missing_test_row" if row is None else "not_published"))
    records = list(results)
    for (hour, hz, day_in_test), outcome in skipped:
        window, day_in_window, position = task_place(cfg, day_in_test)
        run_date = first_day + datetime.timedelta(days=day_in_test)
        arrays = inputs.hour_arrays[hour]
        row = issue_row(arrays["days"], cfg["offset"] + day_in_test)
        task_lists = cfg.get("task_lists")
        list_id = None if task_lists is None else int(task_lists[(hour, hz)][position])
        anchor = run_date if block is None else refit_anchor(run_date, block)
        records.append(
            skip_record(
                run_date,
                hour,
                hz,
                day_in_window,
                None if row is None else float(arrays["y"][row, hz - 1]),
                cfg.get("freq", "1h"),
                window,
                outcome,
                _list_hash(cfg, inputs.expanded, hour, hz, list_id),
                anchor.isoformat(),
            )
        )
    return Served(
        model.name,
        issue.isoformat(),
        records,
        fits,
        notes=list(dict.fromkeys(notes)),
        manifest=issue_manifest(plan, data),
        lists=issue_lists(plan),
        identity=identity,
    )


class ServedAlone:
    """A model whose `run` forecasts each issue day through the serve path (`serve`, with fits made inline from the frame it is
    given, as `forecast()` makes them without a stored fit): what `verify(through="serve")` probes, so that the leakage probe and
    its control cover the live path's planning, row selection, fits and forecasts. Everything else is the model's (attributes are
    read through). A model whose fits aren't stored runs through its own `run`, as `forecast()` runs it.

    `rules` gives the information set of each target the probes run on (the target, and the copy of it `verify` trains on when
    the target is also an input): the serve path plans a predictor function with it and refuses a fit whose training targets
    aren't known by its rule (`check_published`). `engine` is whether the pipeline's engine is on."""

    def __init__(self, model: Any, rules: Mapping[str, Any], engine: bool):
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_rules", dict(rules))
        object.__setattr__(self, "_engine", bool(engine))

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_model"), name)

    def run(
        self,
        data: pd.DataFrame,
        test_start: Any = None,
        test_end: Any = None,
        target: str = "price",
        horizon: Optional[int] = None,
        freq: str = "1h",
        horizons: Optional[Sequence[int]] = None,
        cut_rules: Any = None,
        plan: Any = None,
        **others: Any,
    ) -> Any:
        from ..results.ref import ModelResultRef
        from ._calendar import resolve_horizons

        model = object.__getattribute__(self, "_model")
        if stores_fits(model) is not None:
            given = {"horizon": horizon} if horizons is None else {"horizons": horizons}
            extra = {key: value for key, value in (("cut_rules", cut_rules), ("plan", plan)) if value is not None}
            return model.run(
                data, test_start=test_start, test_end=test_end, target=target, freq=freq, **given, **extra, **others
            )
        _, planned = resolve_horizons(horizon, horizons, 7)
        info = object.__getattribute__(self, "_rules").get(target, cut_rules)
        first, last = (datetime.date.fromisoformat(str(day)[:10]) for day in (test_start, test_end))
        records: List[Dict[str, Any]] = []
        for offset in range((last - first).days + 1):
            day = first + datetime.timedelta(days=offset)
            records.extend(
                serve(
                    model,
                    data,
                    day,
                    None,
                    target=target,
                    horizons=planned,
                    freq=freq,
                    information_set=info,
                    availability="none",
                    issue_days=None,
                    engine=object.__getattribute__(self, "_engine"),
                    fit_missing="fit",
                ).records
            )
        return ModelResultRef(
            name=model.name,
            path=None,
            count=len(records),
            test_start=first.isoformat(),
            test_end=last.isoformat(),
            horizon=max(planned),
            freq=freq,
            _results=records,
        )
