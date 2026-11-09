"""Gradient-boosted trees of LightGBM, XGBoost and CatBoost: `LightGBMModel`, `XGBoostModel`, `CatBoostModel` (step 6o).

Each runs as `HistGBModel` does (`models/hgb.py`): `periods="pooled"` (the default) fits one model per (horizon, refit day) on the
(date, period) rows of every period (`models._units`, the research's "global" rows), each predictor read on its period's own
row, the virtual `period` a feature where the list names it, and each period's target through its own Invariant (asinh of
median and MAD); `periods="each"` fits one per period, horizon and refit day. Both run on the refit calendar and predict each
block in one call (`_predict_block`): a tree's prediction of a row reads that row alone. Inputs go in as they are
(`INVARIANT_TARGET`: a tree splits on order, not scale).

Each package is an extra of its own (`[lightgbm]`, `[xgboost]`, `[catboost]`), imported when a model first fits, so the classes
import, save and load without it, and the pre-flight of a run names a missing one before any model fits. The card records the
package's version in the manifest's `runtime`, with the thread count of a fit (and XGBoost's device), all compared on resume.

Threads: a fit and its predictions run on `threads` OpenMP threads (one by default), the limit set around them alone
(`threadpoolctl`) and the count given to the package (`n_jobs`, `thread_count`). The pool's tasks hold OpenMP at one thread,
which XGBoost would otherwise keep to whatever `n_jobs` says; and without a limit XGBoost forms an OpenMP team even at
`n_jobs=1`, after which a forked child that runs OpenMP on several threads hangs (GNU OpenMP), so a fit on one thread leaves a
process safe to fork. The estimator is built, and its package with its OpenMP runtime loaded, before the scope, which limits
only what is loaded. The seed (`random_state`) is the same for every fit, so a forecast doesn't depend on which fits a run
makes. LightGBM runs with `deterministic=True` and `force_row_wise=True` (the research's settings, `zoo_classic/code/ztree.py`).
"""

import importlib
import numbers
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .._extras import MissingExtraError
from ._cards import RUN_DATA, ModelCard
from ._units import PERIOD, POOLED
from .base import BaseModel, _usable_cpu_count
from .hgb import INVARIANT_TARGET, PERIODS

# `objective`: the library's names and each package's.
OBJECTIVES = ("l1", "l2")
LIGHTGBM_OBJECTIVES = {"l1": "l1", "l2": "l2"}
XGBOOST_OBJECTIVES = {"l1": "reg:absoluteerror", "l2": "reg:squarederror"}
CATBOOST_OBJECTIVES = {"l1": "MAE", "l2": "RMSE"}
# XGBoost's `device`: the CPU, or a CUDA GPU (`"cuda"`, `"cuda:1"`). CPU and GPU forecasts differ (3.849 vs 3.878 MAE on
# Lago's DE in the research), so the device type is recorded in the manifest's `runtime` and compared on resume.
_DEVICE = re.compile(r"^(cpu|cuda(:[0-9]+)?)\Z")


def _card(family: str, extra: str, licence: str, licence_url: str, pin: str) -> ModelCard:
    return ModelCard(
        family=family,
        extra=extra,
        licence=licence,
        licence_url=licence_url,
        commercial_use=True,
        revenue_cap=None,
        attribution=None,
        weights=None,
        runtime=(pin,),
        corpus=RUN_DATA,
        corpus_end=None,
        released=None,
    )


LIGHTGBM_CARD = _card(
    "lightgbm", "lightgbm", "MIT", "https://github.com/microsoft/LightGBM/blob/master/LICENSE", "lightgbm>=4.5"
)
XGBOOST_CARD = _card(
    "xgboost", "xgboost", "Apache-2.0", "https://github.com/dmlc/xgboost/blob/master/LICENSE", "xgboost>=2.1"
)
CATBOOST_CARD = _card(
    "catboost", "catboost", "Apache-2.0", "https://github.com/catboost/catboost/blob/master/LICENSE", "catboost>=1.2.9"
)


def _thread_scope(threads: int):
    """A scope with OpenMP at `threads` threads in this thread, put back on exit."""
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=threads, user_api="openmp")


def _package(name: str, model: "_BoostedTrees") -> Any:
    """`import name`, or `MissingExtraError` naming the model's extra where the package isn't installed (the pre-flight of a run
    finds that first; this is for a direct call). Any other import error is the package's own and goes through as it is."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        raise MissingExtraError(type(model).__name__, model._card.extra) from error


class _BoostedTrees(BaseModel):
    """What the three tree families share: the unit kind from `periods`, the library's `objective`, the thread count of a fit,
    and the fit and predictions of a scikit-learn style estimator (`_estimator()`, each class's) inside the thread scope."""

    _engine_ok = False

    def __setattr__(self, key, value):
        if key == "periods" and value not in PERIODS:
            raise ValueError(f"periods must be one of {list(PERIODS)}, got {value!r}")
        if key == "periods" and "periods" in vars(self):
            raise ValueError(f"periods can't change after the model is made: build another {type(self).__name__}")
        if key == "objective" and value not in OBJECTIVES:
            raise ValueError(f"objective must be one of {list(OBJECTIVES)} (absolute or squared error), got {value!r}")
        if key == "threads":
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError(f"threads must be a whole number from 1 (OpenMP threads of one fit), got {value!r}")
            value = int(value)
        if key == "random_state":
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise ValueError(f"random_state must be a whole number (the seed of every fit), got {value!r}")
            value = int(value)
        super().__setattr__(key, value)

    @property
    def _unit_kind(self) -> str:
        """`"pooled"` with `periods="pooled"`, else `"period"`."""
        return POOLED if self.periods == "pooled" else PERIOD

    def _tree_kwargs(self) -> Dict[str, Any]:
        return {"periods": self.periods, "objective": self.objective}

    def _seed_kwargs(self) -> Dict[str, Any]:
        return {"random_state": self.random_state, "threads": self.threads}

    def _runtime_fields(self) -> Dict[str, Any]:
        """The thread count of a fit, recorded next to the package's version and compared on resume."""
        return {"threads_per_fit": self.threads}

    def _on_gpu(self) -> bool:
        return False

    def _executor_config(self) -> tuple:
        """(task threads per process, processes): `THREADS_PER_PROCESS` (1) and `MAX_PROCESSES`, by default as many processes
        as the usable CPUs hold fits of `threads` threads each, or one where the fits run on a GPU."""
        task_threads = int(os.environ.get("THREADS_PER_PROCESS", "1"))
        default = 1 if self._on_gpu() else max(1, _usable_cpu_count() // (task_threads * self.threads))
        return task_threads, int(os.environ.get("MAX_PROCESSES", default))

    def _estimator(self) -> Any:  # pragma: no cover - every family defines it
        raise NotImplementedError

    def _fit_model(self, train_x: np.ndarray, train_y: np.ndarray) -> Any:
        # Built (the package and its OpenMP runtime loaded) before the scope: `threadpool_limits` limits only what is loaded.
        estimator = self._estimator()
        with _thread_scope(self.threads):
            return estimator.fit(train_x, train_y)

    def _predict_block(self, fitted: Any, rows: np.ndarray) -> Tuple[np.ndarray, List[list]]:
        with _thread_scope(self.threads):
            predictions = fitted.predict(np.asarray(rows, dtype=np.float64))
        return np.asarray(predictions, dtype=np.float64), [[] for _ in range(len(rows))]

    def _predict_fitted(self, fitted: Any, test_x: np.ndarray) -> Tuple[float, list]:
        predictions, _ = self._predict_block(fitted, np.asarray(test_x, dtype=np.float64).reshape(1, -1))
        return float(predictions[0]), []

    def _fit_predict(self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> Tuple[float, list]:
        return self._predict_fitted(self._fit_model(train_x, train_y), test_x)


class LightGBMModel(_BoostedTrees):
    """LightGBM's `LGBMRegressor` on the refit calendar, by default over the pooled (date, period) rows (`[lightgbm]` extra).

    Args:
        predictors: Full column names (`{horizon}` allowed), saved verbatim. With `periods="pooled"` each is read on the row's
            own period (`price_d-1` is that period's price, `price_h05_d-1` the date's 05:00 price), so a list must be the same
            for every period, and `"period"` (0 to P-1) is a feature where listed. `presets.tree_predictors` gives the
            research's rows.
        training_window: Days of rows, `"expanding"` (the default: every row from the frame's first date), or an int.
        refit: How often a fit is made, on the fixed calendar (`"7D"`, Mondays, by default); a pooled fit needs one.
        periods: `"pooled"` (one fit over every period's rows) or `"each"` (one per period).
        objective: `"l1"` (absolute error, the default) or `"l2"` (squared error).
        n_estimators, learning_rate, num_leaves, min_child_samples, subsample, subsample_freq, colsample_bytree, reg_lambda:
            LightGBM's (scikit-learn API names), at the research's values. Every fit runs with `deterministic=True` and
            `force_row_wise=True`.
        random_state: The seed of the row and column samples, the same for every fit.
        threads: OpenMP threads of one fit (recorded in the manifest's `runtime` and compared on resume).
        scaler: The predictors as they are and the target as the Invariant, by default (`periods="each"`; a pooled fit gives
            each period's target its own Invariant whatever this says).
        newest_target, binary_mask: As for every model (see `BaseModel`).
    """

    _card = LIGHTGBM_CARD

    def __init__(
        self,
        predictors: List[str],
        training_window: Union[int, str] = "expanding",
        refit: Optional[str] = "7D",
        periods: str = "pooled",
        objective: str = "l1",
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        subsample_freq: int = 1,
        colsample_bytree: float = 0.5,
        reg_lambda: float = 1.0,
        random_state: int = 0,
        threads: int = 1,
        name: str = "lgbm",
        *,
        scaler: Any = INVARIANT_TARGET,
        newest_target: str = "d-1",
        binary_mask: Optional[str] = None,
    ):
        super().__init__(
            predictors,
            training_window,
            name,
            scaler=scaler,
            newest_target=newest_target,
            refit=None,
            binary_mask=binary_mask,
        )
        self.periods = periods
        self.objective = objective
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.threads = threads
        # `refit` last, whose check reads the unit kind (`periods`); it keeps its place among the parameters a file writes.
        self.refit = refit

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        return {
            **self._tree_kwargs(),
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "subsample_freq": self.subsample_freq,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            **self._seed_kwargs(),
        }

    def _estimator(self):
        lightgbm = _package("lightgbm", self)
        return lightgbm.LGBMRegressor(
            objective=LIGHTGBM_OBJECTIVES[self.objective],
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=self.subsample_freq,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.threads,
            verbose=-1,
            deterministic=True,
            force_row_wise=True,
        )


class XGBoostModel(_BoostedTrees):
    """XGBoost's `XGBRegressor` (`tree_method="hist"`) on the refit calendar, by default over the pooled (date, period) rows
    (`[xgboost]` extra).

    Args:
        predictors, training_window, refit, periods, objective, random_state, threads, scaler, newest_target, binary_mask: As
            for `LightGBMModel`.
        n_estimators, learning_rate, max_depth, subsample, colsample_bytree, min_child_weight, reg_lambda: XGBoost's
            (scikit-learn API names), at the research's values.
        device: `"cpu"` (the default) or a CUDA GPU (`"cuda"`, `"cuda:1"`). CPU and GPU forecasts differ, so the device type is
            recorded in the manifest's `runtime` and compared on resume; on a GPU a run takes one process unless
            `MAX_PROCESSES` says otherwise.
    """

    _card = XGBOOST_CARD

    def __init__(
        self,
        predictors: List[str],
        training_window: Union[int, str] = "expanding",
        refit: Optional[str] = "7D",
        periods: str = "pooled",
        objective: str = "l1",
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.5,
        min_child_weight: float = 5.0,
        reg_lambda: float = 1.0,
        random_state: int = 0,
        threads: int = 1,
        device: str = "cpu",
        name: str = "xgb",
        *,
        scaler: Any = INVARIANT_TARGET,
        newest_target: str = "d-1",
        binary_mask: Optional[str] = None,
    ):
        super().__init__(
            predictors,
            training_window,
            name,
            scaler=scaler,
            newest_target=newest_target,
            refit=None,
            binary_mask=binary_mask,
        )
        self.periods = periods
        self.objective = objective
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.threads = threads
        self.device = device
        self.refit = refit

    def __setattr__(self, key, value):
        if key == "device" and not (isinstance(value, str) and _DEVICE.match(value)):
            raise ValueError(f"device must be 'cpu' or a CUDA GPU ('cuda', 'cuda:1'), got {value!r}")
        super().__setattr__(key, value)

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        return {
            **self._tree_kwargs(),
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_lambda": self.reg_lambda,
            **self._seed_kwargs(),
            "device": self.device,
        }

    def _on_gpu(self) -> bool:
        return self.device != "cpu"

    def _runtime_fields(self) -> Dict[str, Any]:
        """The device type (`cpu`, `cuda`) and the thread count of a fit, compared on resume."""
        return {"device": self.device.split(":")[0], **super()._runtime_fields()}

    def _estimator(self):
        xgboost = _package("xgboost", self)
        return xgboost.XGBRegressor(
            objective=XGBOOST_OBJECTIVES[self.objective],
            tree_method="hist",
            device=self.device,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.threads,
        )


class CatBoostModel(_BoostedTrees):
    """CatBoost's `CatBoostRegressor` on the refit calendar, by default over the pooled (date, period) rows (`[catboost]` extra,
    which has no wheels for free-threaded Python or 3.15).

    Args:
        predictors, training_window, refit, periods, objective, random_state, threads, scaler, newest_target, binary_mask: As
            for `LightGBMModel`.
        iterations, learning_rate, depth, l2_leaf_reg: CatBoost's, at the research's values (`ztree.py`). CatBoost runs
            silently and writes no files (`allow_writing_files=False`).
    """

    _card = CATBOOST_CARD

    def __init__(
        self,
        predictors: List[str],
        training_window: Union[int, str] = "expanding",
        refit: Optional[str] = "7D",
        periods: str = "pooled",
        objective: str = "l1",
        iterations: int = 1000,
        learning_rate: float = 0.05,
        depth: int = 8,
        l2_leaf_reg: float = 5.0,
        random_state: int = 0,
        threads: int = 1,
        name: str = "catboost",
        *,
        scaler: Any = INVARIANT_TARGET,
        newest_target: str = "d-1",
        binary_mask: Optional[str] = None,
    ):
        super().__init__(
            predictors,
            training_window,
            name,
            scaler=scaler,
            newest_target=newest_target,
            refit=None,
            binary_mask=binary_mask,
        )
        self.periods = periods
        self.objective = objective
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state
        self.threads = threads
        self.refit = refit

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        return {
            **self._tree_kwargs(),
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            **self._seed_kwargs(),
        }

    def _estimator(self):
        catboost = _package("catboost", self)
        return catboost.CatBoostRegressor(
            loss_function=CATBOOST_OBJECTIVES[self.objective],
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_state,
            thread_count=self.threads,
            verbose=0,
            allow_writing_files=False,
        )


__all__ = [
    "LightGBMModel",
    "XGBoostModel",
    "CatBoostModel",
    "OBJECTIVES",
    "LIGHTGBM_CARD",
    "XGBOOST_CARD",
    "CATBOOST_CARD",
]
