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
FORMAT = 1
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
    """Write an anchor's fit: every unit's arrays into `units.npz` (`u<i>.<section>.<name>`, numpy's own format, no object
    array), then `fit.json` with `meta`, each unit's entry and the sha256 of `units.npz`, both synced, in a directory of their
    own that replaces `directory` whole (a reader finds the old fit, the new one or none, never a mix)."""
    arrays: Dict[str, np.ndarray] = {}
    for place, (_, unit_arrays) in enumerate(units):
        for name, value in unit_arrays.items():
            if value.dtype == object:
                raise TypeError(f"a stored fit holds arrays of numbers or text only, got an object array {name}")
            arrays[f"u{place}.{name}"] = value
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


@dataclass
class StoredFit:
    """An anchor's stored fit, read and checked (`read_fit`): its `fit.json` and its units by key."""

    directory: Path
    anchor: datetime.date
    meta: Dict[str, Any]
    units: Dict[UnitKey, UnitFit] = field(default_factory=dict)


def read_meta(directory: Path) -> Optional[Dict[str, Any]]:
    """`fit.json` of an anchor's directory, None where there is none; StoredFitError where it can't be read."""
    path = directory / FIT_FILE
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StoredFitError(f"{path} can't be read: {exc}") from None
    if not isinstance(meta, dict) or meta.get("format") != FORMAT:
        raise StoredFitError(f"{path} is no stored fit of format {FORMAT}")
    return meta


def read_fit(directory: Path, model: Any, digest: Optional[str] = None) -> Optional[StoredFit]:
    """The stored fit in `directory`, checked: None where there is none or (with `digest`) it is another configuration's;
    `StoredFitError` where `units.npz` isn't the file `fit.json` names (its sha256), or a unit's arrays don't build back."""
    meta = read_meta(directory)
    if meta is None or (digest is not None and meta.get("config_digest") != digest):
        return None
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
    import io

    groups: Dict[int, Dict[str, np.ndarray]] = {}
    with np.load(io.BytesIO(payload), allow_pickle=False) as npz:
        for name in npz.files:
            place, _, rest = name.partition(".")
            groups.setdefault(int(place[1:]), {})[rest] = npz[name]
    anchor = datetime.date.fromisoformat(meta["anchor"])
    stored = StoredFit(directory, anchor, meta)
    for place, unit_meta in enumerate(meta.get("units", [])):
        key = tuple(unit_meta["key"])
        try:
            stored.units[key] = unit_from_arrays(key, anchor, unit_meta, groups.get(place, {}), model)
        except (KeyError, ValueError, TypeError) as exc:
            raise StoredFitError(f"{directory}: unit {key_text(key)} doesn't build back: {exc}") from None
    return stored


def stored_anchors(save_dir: Union[str, Path], model_name: str) -> List[datetime.date]:
    """The anchors with a stored fit of the model in the live directory, in date order."""
    root = model_dir(save_dir, model_name)
    if not root.is_dir():
        return []
    out = []
    for path in root.iterdir():
        try:
            day = datetime.date.fromisoformat(path.name)
        except ValueError:
            continue  # a staging directory of an unfinished write
        if (path / FIT_FILE).exists():
            out.append(day)
    return sorted(out)


def find_fit(
    save_dir: Union[str, Path], name: str, model: Any, anchor: datetime.date, digest: str, previous: bool = True
) -> Optional[Tuple[StoredFit, int]]:
    """(the stored fit of the model `name` of `anchor` with this configuration digest, 0), or with `previous` the newest stored
    older one of the same digest and its age in days (decision 15: a missed refit leaves the last fit in use, its age recorded);
    None where there is neither. A fit is never taken from an anchor after `anchor`, nor from another configuration. `model`
    builds the fits back (`_from_state`)."""
    found = read_fit(fit_dir(save_dir, name, anchor), model, digest)
    if found is not None:
        return found, 0
    if not previous:
        return None
    for older in reversed(stored_anchors(save_dir, name)):
        if older >= anchor:
            continue
        found = read_fit(fit_dir(save_dir, name, older), model, digest)
        if found is not None:
            return found, (anchor - older).days
    return None


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
    )


def check_published(plan: IssuePlan, anchors: Iterable[datetime.date], now: pd.Timestamp, data: pd.DataFrame) -> None:
    """Refuse (`FitTooEarlyError`) a fit of these anchors that reads a training target not yet known at `now` by the target's
    rule (decision 5 of the step 8 spec): for each unit the cell of its newest target, (anchor + min(k, h - 1), its period), and
    the 31 days before it, which a rule over time can publish later than a newer one. The message gives the publication time.
    Without an information set, or a rule for the target, nothing tells when a target is out: that raises too."""
    from ..information_set import _STEP_MINUTES, InformationSetError

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
    zone = info._issue_zone(data.index)
    for anchor in sorted(set(anchors)):
        units = plan.items.get(anchor, {})
        if not units:
            continue
        reach: Dict[int, int] = {}
        horizon_of: Dict[int, int] = {}
        for key in units:
            period, hz = key[0], key[1]
            offset = min(int(plan.newest[(anchor, key)]), hz - 1)
            if offset >= reach.get(period, offset):
                reach[period], horizon_of[period] = offset, hz
        clock = info._issue_clock(pd.DatetimeIndex([pd.Timestamp(anchor)]), data.index, plan.freq)
        periods = np.array(sorted(reach), dtype=np.int64)
        known: Dict[int, np.ndarray] = {}
        for offset in range(min(reach.values()) - 31, max(reach.values()) + 1):
            known[offset] = np.asarray(clock.available(col, periods, step, row_day=offset)[:, 0], dtype=float)
        for place, period in enumerate(periods.tolist()):
            newest = reach[period]
            latest, day = -np.inf, None
            for offset in range(newest - 31, newest + 1):
                minutes = known[offset][place]
                if np.isfinite(minutes) and minutes >= latest:
                    latest, day = minutes, anchor + datetime.timedelta(days=offset)
            if day is None:
                continue
            out = clock.ctx.origin + pd.Timedelta(minutes=float(latest))
            if out > now:
                raise FitTooEarlyError(
                    f"{model.name}: the fit anchored on {anchor} trains on '{plan.target}' of {day} (period {period}, horizon "
                    f"{horizon_of[period]}), which its rule ('{col.rule.describe()}') publishes at "
                    f"{out.tz_convert(zone):%Y-%m-%d %H:%M} {zone} ({out.tz_convert('UTC'):%Y-%m-%d %H:%M} UTC), after the "
                    f"clock's now ({now.tz_convert('UTC'):%Y-%m-%d %H:%M} UTC). A fit made now would train on fewer targets "
                    "than the backtest's: refit once it is out."
                )


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
        **made,
    }


def anchor_meta(plan: IssuePlan, anchor: datetime.date, units: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """`fit.json` of an anchor, but its units and artifacts: what the fit is of and what it read."""
    from ..results.manifest import _package_version

    model = plan.model
    through = max((unit["newest_target_day"] for unit in units), default=anchor.isoformat())
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
        "kernels": model._kernel_backend,
        "rows": {
            "through": through,
            "columns": plan.used_columns,
            "hash": plan.hashes.of_columns(through, plan.used_columns),
        },
        "package_version": _package_version(),
        "created_at": current_clock().now().isoformat(),
    }


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
    groups: Dict[int, Dict[str, np.ndarray]] = {}
    with np.load(stored.directory / UNITS_FILE, allow_pickle=False) as npz:
        for name in npz.files:
            place, _, rest = name.partition(".")
            groups.setdefault(int(place[1:]), {})[rest] = npz[name]
    return [(unit_meta, groups.get(place, {})) for place, unit_meta in enumerate(stored.meta.get("units", []))]


# ---------------------------------------------------------------------------------------------------------------------------
# refit(): fit what is due and store it
# ---------------------------------------------------------------------------------------------------------------------------


@dataclass
class RefitResult:
    """What `ModelPipeline.refit` did for one issue day: per model the anchors it fitted and stored (`fitted`), those already
    stored with this configuration (`not_due`), the units it fitted, and how long each took; `not_stored`, why a model's fits
    aren't stored (it fits at serve time). `noop` is True when nothing was fitted."""

    issue: str
    fitted: Dict[str, List[str]] = field(default_factory=dict)
    not_due: Dict[str, List[str]] = field(default_factory=dict)
    not_stored: Dict[str, str] = field(default_factory=dict)
    units: Dict[str, int] = field(default_factory=dict)
    seconds: Dict[str, float] = field(default_factory=dict)

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
) -> RefitResult:
    """`ModelPipeline.refit`'s work once the live directory's lock is held: every model's issue-day plan, which anchors are due,
    the publication check of every due fit before any is made (decision 5), then each model's due units fitted (`run_fits`) and
    each anchor written (`write_fit`), merged with the units of the same configuration already stored there."""
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
        for anchor, units in sorted(plan.items.items()):
            directory = fit_dir(save_dir, model.name, anchor)
            try:
                stored = read_fit(directory, plan.inputs.model, plan.digest)
            except StoredFitError:
                if if_due:
                    raise
                stored = None  # refitted whole below
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
            write_fit(fit_dir(save_dir, name, anchor), anchor_meta(plan, anchor, [m for m, _ in ordered]), ordered)
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
    anchor's own), "previous" (the newest older one of the same configuration, `fit_age_days` its age), "inline" (fitted by this
    call) or "missing" (none: its forecasts are `no_stored_fit` skips), and `used`, the anchor of the fit served."""

    model: str
    issue: str
    records: List[Dict[str, Any]]
    fits: Dict[str, Dict[str, Any]]


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
    `"skip"`, as the backtest does. A model whose fits aren't stored (`stores_fits`) raises: it fits at serve time through its own
    run."""
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
        raise ValueError(f"{model.name}: {reason}, through its own run of the issue day")
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
    lookup: Dict[datetime.date, Optional[Tuple[Dict[UnitKey, UnitFit], datetime.date]]] = {}
    fits: Dict[str, Dict[str, Any]] = {}
    for anchor in sorted(plan.items):
        found = (
            None if save_dir is None else find_fit(save_dir, model.name, inputs.model, anchor, plan.digest, previous)
        )
        if found is not None:
            stored, age = found
            lookup[anchor] = (stored.units, stored.anchor)
            fits[anchor.isoformat()] = {
                "source": "stored" if age == 0 else "previous",
                "used": stored.anchor.isoformat(),
                "fit_age_days": age,
            }
        elif fit_missing == "fit" or (fit_missing != "skip" and cfg.get("refit") is None):
            check_published(plan, [anchor], current_clock().now(), data)
            _, units = fitted_units(plan, anchor, _sorted_keys(plan.items[anchor]))
            lookup[anchor] = (units, anchor)
            fits[anchor.isoformat()] = {"source": "inline", "used": anchor.isoformat(), "fit_age_days": 0}
        else:
            lookup[anchor] = None
            fits[anchor.isoformat()] = {"source": "missing", "used": None, "fit_age_days": None}

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
    return Served(model.name, issue.isoformat(), records, fits)
