"""LEAR, the Lasso-estimated autoregressive model of Lago, Marcjasz, De Schutter & Weron (2021), Applied Energy 293: one Lasso per
period, its penalty chosen on every fit by an information criterion on the LARS path, or by blocked cross-validation."""

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from . import _lars
from ._cv_stats import cv_stats_of
from ._lars import CRITERIA, CV_SOLVER_ERROR, SOLVERS, CvStats, GramStats, LearFit, cv_stats, gram_stats
from ._stats import ScaledStats
from .base import BaseModel

# Lago et al.'s calibration windows of 56, 84, 1092 and 1456 days as this library's training rows. The original keeps the last
# cw days before the target day and drops the first week of them for the lags, so a forecast trains on cw - 7 targets, the newest
# the issue day's; under `newest_target="published"` and a 13:00 rule for the price that is `training_window` cw - 7.
LAGO_WINDOWS = (49, 77, 1085, 1449)
# epftoolbox's "Invariant" scaler: median/MAD asinh on the target and every non-binary input, nothing standardised after it.
LAGO_SCALER = {"name": "asinh", "params": {"standardise": False}}


class LEARModel(BaseModel):
    # What the step-5 engine must put in the `ScaledStats` it hands `_fit_stats`: the sums alone for an information criterion.
    # A `criterion="cv"` instance needs the CV blocks too (`_cv_stats.block_stats`): `__setattr__` sets its own.
    _ENGINE_NEEDS = frozenset()

    def __init__(
        self,
        predictors,
        training_window=(56, 84, 728, "expanding"),
        criterion: str = "bic",
        solver: str = "lars",
        *,
        scaler="asinh",
        name: str = "LEAR",
        newest_target: str = "d-1",
        refit: Optional[str] = None,
        binary_mask: Optional[str] = None,
    ):
        """LEAR: per period a Lasso whose penalty is chosen on every fit in two stages, from the centred sums of the scaled
        training rows alone (`models._lars.lear_fit`, scikit-learn 0.24.2's rules, which Lago et al. ran).

        1. α* is the first minimum of `criterion` over the kinks of the LARS-lasso path on centred unit-norm columns: old
           `LassoLarsIC(normalize=True)`'s `n·MSE/var(y) + K·df`, K = 2 for `"aic"` and log n for `"bic"`.
        2. The Lasso at that α* on the centred columns as they are, not normalised: Lago's refit, about √n times less
           penalised than the path's own solution at α*. `solver="lars"` solves it exactly (a second LARS path, run just past
           α*); `"cd"` is Lago's `Lasso(max_iter=2500)`, coordinate descent in column order with tol 1e-4, which can stop
           before it converges and then depends on the order of the predictors.

        `criterion="cv"` is old `LassoLarsCV(cv=KFold(5), normalize=True, max_iter=2500)` instead (`_lars.lear_fit_cv`):
        the training rows in five contiguous blocks in date order, each fold's LARS path on its four blocks, centred and
        normalised on them, scored on the fifth; α* the first minimum of the folds' mean held-out squared error over their
        pooled kinks (at most 1,000 of them); the coefficients the full window's normalised path at α*, divided by the norms,
        with no refit. Each fold drops the exact twins and the columns constant up to rounding of its own training rows, as
        the window's fit drops them on the window. It takes `solver="lars"` (`"cd"` raises). A fit costs 5.0 to 6.8 times one of
        `"bic"` with the default `solver="lars"` and 3.5 to 4.8 times one with `"cd"` (from the 1449- to the 49-row window); a
        whole run 1.7 to 1.9 and 1.5 to 1.7 times as much.

        The intercept is the centring: a forecast is `ybar + (x - xbar) @ w` on the scaled row. A column the scaler made
        constant gets the coefficient 0 in both stages.

        The defaults are the recommended single model of the plan's D2: BIC, the exact refit, `AsinhScaler` and the average of
        four training windows, `[56, 84, 728, "expanding"]`. `newest_target="published"` trains on the issue day's prices
        where an information set says they are known; the default `"d-1"` needs none. `LEARModel.lago(...)` is the model of
        Lago et al. (2021) exactly.

        The kernels run in numpy, or compiled by numba with the `[fast]` extra (`pip install 'epftoolbox2[fast]'`), chosen by
        `EPFTOOLBOX2_KERNELS=auto|numpy|numba` (auto: numba when it imports). Exact twin columns are dropped before both stages
        but one (`_lars.without_twins`), and the two backends then agree within 1e-9 with the same kinks, not to the bit, wherever
        no other tie between columns decides an entry; the manifest records which ran."""
        super().__init__(
            predictors,
            training_window,
            name=name,
            scaler=scaler,
            newest_target=newest_target,
            refit=refit,
            binary_mask=binary_mask,
        )
        self.criterion = criterion  # checked in __setattr__
        self.solver = solver

    def __setattr__(self, key, value):
        if key == "criterion" and value not in CRITERIA:
            raise ValueError(f"criterion must be one of {list(CRITERIA)}, got {value!r}")
        if key == "solver" and value not in SOLVERS:
            raise ValueError(f"solver must be one of {list(SOLVERS)}, got {value!r}")
        if key in ("criterion", "solver"):
            other = self.__dict__.get("solver" if key == "criterion" else "criterion")
            if {value, other} == {"cv", "cd"}:
                raise ValueError(CV_SOLVER_ERROR)
        super().__setattr__(key, value)
        if key == "criterion":
            # Starts with "_", so YAML and the manifest never see it.
            super().__setattr__("_ENGINE_NEEDS", frozenset({"blocks"}) if value == "cv" else frozenset())

    @classmethod
    def lago(
        cls, exogenous: Sequence[str], *, price: str = "price", calendar_prefix: str = "", name: str = "LEAR"
    ) -> "LEARModel":
        """The LEAR of Lago, Marcjasz, De Schutter & Weron (2021), as the original epftoolbox runs it: a plain `LEARModel`.

        Its 247 inputs come in the original's order (`presets.lear_predictors(..., order="lago")`), `exogenous` in the order of
        the dataset's file, which coordinate descent that stops early depends on. For `EpftoolboxBenchmarkSource`'s data:

        - NP: `["load_forecast", "wind_forecast"]`
        - PJM: `["load_forecast", "zonal_load_forecast"]`
        - FR: `["generation_forecast", "load_forecast"]`
        - BE: `["generation_forecast_fr", "load_forecast_fr"]`
        - DE: `["load_forecast_amprion", "wind_solar_forecast"]`

        The rest: `criterion="aic"`, `solver="cd"` (the original's `Lasso(max_iter=2500)`), epftoolbox's Invariant scaler
        (`{"name": "asinh", "params": {"standardise": False}}`), `newest_target="published"` and the training windows
        `[49, 77, 1085, 1449]`, averaged with equal weights: Lago's calibration windows 56, 84, 1092 and 1456 days, of which the
        original drops the first week for its lags, so they hold that many targets, the newest the issue day's. That needs an
        information set in which the issue day's prices are known, as in the paper:
        `InformationSet("11:45", "Europe/Brussels", rules={"price": Availability.published("13:00", 1, timezone="Etc/GMT-1")})`
        for the European markets. PJM's frame is on `Etc/GMT+5`, where that rule dates the evening hours of a day on the next one
        and would train them on the day before: give the rule and the issue time on PJM's clock,
        `InformationSet("11:45", "Etc/GMT+5", rules={"price": Availability.published("13:00", 1, timezone="Etc/GMT+5")})`.

        The paper forecasts one day ahead, `horizon=1`. Other horizons run, with the inputs anchored at the issue day and
        `x_d+{horizon}`, which only a series published that far ahead can give (see Predictor Presets). Saved to YAML, the
        model is these settings, with no mark of where they came from."""
        from ..presets import lear_predictors

        return cls(
            lear_predictors(exogenous, price, calendar_prefix=calendar_prefix, order="lago"),
            training_window=list(LAGO_WINDOWS),
            criterion="aic",
            solver="cd",
            scaler=dict(LAGO_SCALER, params=dict(LAGO_SCALER["params"])),
            newest_target="published",
            name=name,
        )

    @property
    def _model_kwargs(self):
        return {"criterion": self.criterion, "solver": self.solver}

    @property
    def _kernel_backend(self) -> Optional[str]:
        # Compiles numba's kernels (or loads them from its cache) on the first call, so a run calls it before its pool starts.
        return _lars._kernels().BACKEND

    def _fit_predict(self, train_x, train_y, test_x):
        return self._predict_fitted(self._fit_model(train_x, train_y), test_x)

    def _fit_model(self, train_x, train_y) -> LearFit:
        if self.criterion == "cv":
            return self._fit_stats(cv_stats(train_x, train_y))
        return self._fit_stats(gram_stats(train_x, train_y))

    def _fit_stats(self, stats: Union[GramStats, CvStats, ScaledStats]) -> LearFit:
        """The fit from the centred sums of a unit's scaled training rows alone, `GramStats` for an information criterion and
        `CvStats` (those of the five blocks in date order) for `"cv"`: where step 5's engine hands in statistics it shares
        between models. The engine's `ScaledStats` are read through `.gram()`, and for `"cv"` through their blocks
        (`_cv_stats.cv_stats_of`; TypeError when they carry none)."""
        if isinstance(stats, ScaledStats):
            stats = cv_stats_of(stats) if self.criterion == "cv" else stats.gram()
        return _lars._kernels().lear_fit(stats, self.criterion, self.solver)

    def _predict_fitted(self, fitted: LearFit, test_x) -> Tuple[float, list]:
        row = np.asarray(test_x, dtype=np.float64).reshape(-1)
        if row.shape != fitted.w.shape:
            raise ValueError(f"LEARModel forecasts one row of {fitted.w.size} values, got {np.shape(test_x)}")
        return float(fitted.ybar + (row - fitted.xbar) @ fitted.w), fitted.w.tolist()


__all__ = ["LEARModel", "LAGO_WINDOWS", "LAGO_SCALER"]
