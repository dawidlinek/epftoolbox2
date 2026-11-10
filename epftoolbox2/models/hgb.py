"""`HistGBModel`: scikit-learn's histogram gradient boosting, pooled over the periods of a day by default (step 6k).

`periods="pooled"` fits one model per (horizon, refit day) on the (date, period) rows of every period (`models._units`, the
research's "global" rows, `zoo_classic/code/ztree.py`), each predictor read on its period's own row, the virtual `period` a
feature where the list names it, and each period's target through its own Invariant (asinh of median and MAD), or another
`target_transform` ("pooled": one Invariant over every row; "none": the raw target); `periods="each"` fits one per period,
horizon and refit day, as every core model does. Both run on the refit calendar and predict each block in
one call (`_predict_block`): a tree's prediction of a row reads that row alone.

The defaults are the research's (`ztree.py:29-30`: 500 iterations, learning rate 0.05, 31 leaves, 20 rows a leaf, L2 1.0,
half the features per split, absolute error, no early stopping), and it is the default learner of `ResidualModel`. A fit and
its predictions run on one OpenMP thread (`threadpoolctl`), so that a forecast is the same in a worker, in the parent and at
any thread count: scikit-learn's histogram sums reduce over threads.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ._cards import RUN_DATA, ModelCard
from ._units import PER_PERIOD, PERIOD, POOLED, check_periods_target_transform
from .base import BaseModel

# How a run of this class fits the periods of a day: one fit over every period's rows, or one per period.
PERIODS = ("pooled", "each")
# `loss`: the name scikit-learn gives each.
LOSSES = {"l1": "absolute_error", "l2": "squared_error"}
# The target as epftoolbox's Invariant and the predictors as they are (a tree splits on order, not scale): a period unit's
# `scaler`; a pooled unit transforms its targets itself, by its `target_transform` (each period's Invariant by default,
# `scalers.target.MultiTargetAsinh`).
INVARIANT_TARGET = {"name": "asinh", "params": {"apply_to": "target", "standardise": False}}

CARD = ModelCard(
    family="hist-gradient-boosting",
    extra=None,
    licence="BSD-3-Clause",
    licence_url="https://github.com/scikit-learn/scikit-learn/blob/main/COPYING",
    commercial_use=True,
    revenue_cap=None,
    attribution=None,
    weights=None,
    # `max_features` came in 1.4 (the core requires 1.3): the pre-flight refuses an older one before any fit. The version is
    # recorded and compared on resume, since the algorithm moves between releases (1.8 and 1.9 differ, B1 of step 6k).
    runtime=("scikit-learn>=1.4",),
    corpus=RUN_DATA,
    corpus_end=None,
    released=None,
)


def _one_thread():
    """A scope with OpenMP held at one thread: scikit-learn's HGB reduces its histogram and gradient sums over OpenMP threads,
    so their count would reach the last bits of a fit."""
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=1, user_api="openmp")


class HistGBModel(BaseModel):
    """`sklearn.ensemble.HistGradientBoostingRegressor` on the refit calendar, by default over the pooled (date, period) rows.

    Args:
        predictors: Full column names (`{horizon}` allowed), saved verbatim. With `periods="pooled"` each is read on the row's
            own period (`price_d-1` is that period's price, `price_h05_d-1` the date's 05:00 price), so a list must be the same
            for every period, and `"period"` (0 to P-1) is a feature where listed. `presets.tree_predictors` gives the
            research's rows.
        training_window: Days of rows, `"expanding"` (the default: every row from the frame's first date), or an int.
        refit: How often a fit is made, on the fixed calendar (`"7D"`, Mondays, by default); a pooled fit needs one.
        periods: `"pooled"` (one fit over every period's rows) or `"each"` (one per period).
        loss: `"l1"` (absolute error, the default) or `"l2"` (squared error).
        max_iter, learning_rate, max_leaf_nodes, min_samples_leaf, l2_regularization, max_features: scikit-learn's, at the
            research's values. `early_stopping` is always off, so every row of the window trains.
        random_state: The seed of the feature subsets (`max_features`), the same for every fit, so a forecast doesn't depend on
            which fits a run makes.
        scaler: The predictors as they are and the target as the Invariant, by default (`periods="each"`; a pooled fit
            transforms its targets by `target_transform` whatever this says).
        target_transform: How a pooled fit transforms its targets: `"per_period"` (each period's own Invariant, the
            default), `"pooled"` (one Invariant over every row) or `"none"` (the raw target: the estimator fits the stacked
            targets as they are and its predictions are the forecasts). `periods="each"` takes the default alone.
        newest_target, binary_mask: As for every model (see `BaseModel`).
        normalise_by, bounds, zero_when: A per-row target, as for every model (see `BaseModel`), with either value of
            `periods`: a pooled fit divides each row's target by its own period row's scale.
    """

    _card = CARD
    _engine_ok = False
    # `target_transform` is written only when set, so files saved before it existed don't move.
    _WRITTEN_WHEN_SET = {**BaseModel._WRITTEN_WHEN_SET, "target_transform": PER_PERIOD}

    def __init__(
        self,
        predictors: List[str],
        training_window: Union[int, str] = "expanding",
        refit: Optional[str] = "7D",
        periods: str = "pooled",
        loss: str = "l1",
        max_iter: int = 500,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        l2_regularization: float = 1.0,
        max_features: float = 0.5,
        random_state: int = 0,
        name: str = "hgb",
        *,
        scaler: Any = INVARIANT_TARGET,
        target_transform: str = PER_PERIOD,
        newest_target: str = "d-1",
        binary_mask: Optional[str] = None,
        normalise_by: Optional[str] = None,
        bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
        zero_when: Optional[str] = None,
    ):
        # `refit` last, whose check reads the unit kind (`periods`); it keeps its place among the parameters a file writes.
        super().__init__(
            predictors,
            training_window,
            name,
            scaler=scaler,
            newest_target=newest_target,
            refit=None,
            binary_mask=binary_mask,
            normalise_by=normalise_by,
            bounds=bounds,
            zero_when=zero_when,
        )
        self.periods = periods
        self.target_transform = target_transform
        self.loss = loss
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_features = max_features
        self.random_state = random_state
        self.refit = refit

    def __setattr__(self, key, value):
        if key == "periods" and value not in PERIODS:
            raise ValueError(f"periods must be one of {list(PERIODS)}, got {value!r}")
        if key == "loss" and value not in LOSSES:
            raise ValueError(f"loss must be one of {list(LOSSES)}, got {value!r}")
        if key == "periods" and "periods" in vars(self):
            raise ValueError("periods can't change after the model is made: build another HistGBModel")
        if key == "target_transform":
            check_periods_target_transform(self, value, self.periods)
        super().__setattr__(key, value)

    @property
    def _unit_kind(self) -> str:
        """`"pooled"` with `periods="pooled"`, else `"period"`."""
        return POOLED if self.periods == "pooled" else PERIOD

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        return {
            "periods": self.periods,
            "loss": self.loss,
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "max_features": self.max_features,
            "random_state": self.random_state,
        }

    def _estimator(self):
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            loss=LOSSES[self.loss],
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            max_features=self.max_features,
            early_stopping=False,
            random_state=self.random_state,
        )

    def _fit_model(self, train_x: np.ndarray, train_y: np.ndarray) -> Any:
        # Built (and scikit-learn's OpenMP runtime loaded) before the scope: `threadpool_limits` limits only what is loaded.
        estimator = self._estimator()
        with _one_thread():
            return estimator.fit(train_x, train_y)

    def _predict_block(self, fitted: Any, rows: np.ndarray) -> Tuple[np.ndarray, List[list]]:
        with _one_thread():
            predictions = fitted.predict(np.asarray(rows, dtype=np.float64))
        return predictions, [[] for _ in range(len(rows))]

    def _predict_fitted(self, fitted: Any, test_x: np.ndarray) -> Tuple[float, list]:
        with _one_thread():
            return float(fitted.predict(np.asarray(test_x, dtype=np.float64).reshape(1, -1))[0]), []

    def _fit_predict(self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> Tuple[float, list]:
        return self._predict_fitted(self._fit_model(train_x, train_y), test_x)


__all__ = ["HistGBModel", "LOSSES", "PERIODS", "INVARIANT_TARGET"]
