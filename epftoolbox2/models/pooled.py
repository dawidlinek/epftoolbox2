"""`PooledModel`: one model fitted over the zones of a run of several (`ModelPipeline.run(data={zone: frame})`).

A pooled unit is (period, issue day D, horizon), or with `refit` (period, refit anchor, horizon), and without the horizon when
`over` holds `"horizon"`. Its rows are the union of the member fits' rows: each zone's (and horizon's) training rows,
`training_span(D, W, h)` in that zone's own local dates and filtered as the model's own path filters them (`_worker.member_rows`,
`unit_rows`, `finite_rows`), **scaled by that member's own scaler fitted on them**, its target too, then stacked zone by zone (in
the order the run's `data` lists the zones), horizon by horizon, row by row in date order. So a market of large values can't
dominate the fit, and no scale is fitted on more rows than one forecast of that zone may read (a scale of the whole stack would
reach across zones and dates). The issue row of (zone, horizon) is scaled by its member's parameters and the forecast inverted
with them. With one zone a pooled model is the model itself, to the bit.

The fit itself is the model's `_fit_model`, each forecast its `_predict_fitted` on one row (`models/_pooled.py`): OLS, LassoCV,
Ridge and LEAR, whose two equal their `_fit_predict`. Each zone's forecasts go to that zone's store (`save_dir/<zone>/<name>.jsonl`)
under the model's keys `(hour, horizon, run_date)`, and each zone's manifest records `pooled` (the zones, `over`, `zone_feature`,
`missing` and the other zones' data hashes), which a resume compares: another zone's data moves this zone's forecasts.
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

from .base import BaseModel

ZONE = "zone"
HORIZON = "horizon"
# The axes a unit can pool over: the zones always, the horizons too when `over` holds "horizon".
POOL_AXES = (ZONE, HORIZON)
# What tells the rows of each zone apart in a fit: one 0/1 column per zone (`zone_<Z>`), one column of the zone's place in the
# run's order as a number (a categorical code, for tree learners that take one), or nothing.
ONEHOT = "onehot"
CODE = "code"
ZONE_FEATURES = (ONEHOT, CODE, None)
# A predictor a zone's frame lacks: refused before any fit, or a column of zeros, its standardised mean (as the S02 study did).
MISSING_RAISE = "raise"
MISSING_ZERO = "zero"
MISSING_POLICIES = (MISSING_RAISE, MISSING_ZERO)


def check_over(over: Union[str, Sequence[str]]) -> Tuple[str, ...]:
    """`over` as a tuple in `POOL_AXES` order; ValueError unless it holds "zone" and nothing outside `POOL_AXES`."""
    axes = (over,) if isinstance(over, str) else tuple(over)
    unknown = [axis for axis in axes if axis not in POOL_AXES]
    if unknown or ZONE not in axes or len(set(axes)) != len(axes):
        raise ValueError(
            f"over must hold 'zone', and may hold 'horizon' too (one fit for every horizon of a period and day), got {over!r}"
        )
    return tuple(axis for axis in POOL_AXES if axis in axes)


class PooledModel:
    """One model fitted over several zones: a forecast of zone Z is made from a fit on every zone's training rows.

    Args:
        model: The model fitted, with its predictors, training window, scaler and refit rule: an `OLSModel`, `LassoCVModel`,
            `RidgeModel` or `LEARModel` (a model with `_fit_model` and `_predict_fitted`). Each zone's rows are its own fit's
            rows, scaled by a scaler of that model's kind fitted on them alone.
        over: The axes one fit pools: `("zone",)` (the default), one fit per period, issue day and horizon over every zone; or
            `("zone", "horizon")`, one fit per period and issue day over every zone and horizon, the predictors of each horizon
            stacked by their place in the list, so each horizon's list must have the same length.
        zone_feature: What tells the zones apart in a fit: `"onehot"` (a 0/1 column `zone_<Z>` per zone, the default), `"code"`
            (one column of the zone's place in the run's order, for a tree learner that reads it as a category), or None.
            With `"horizon"` in `over` a one-hot horizon is added as well. None of these with a single zone or horizon.
        missing: A predictor a zone's frame lacks: `"raise"` (the default) refuses the run before any fit; `"zero"` gives it
            the value 0 in that zone, its standardised mean.
        name: The name of the pooled model, which names its results in each zone; the inner model's name if not given.

    Refused in this step: a predictor function of `ctx`, a list of training windows, `newest_target="published"` (each zone
    issues on its own clock, and one fit serves them all) and `NaiveModel`, which fits nothing. A shared fit is made by the
    earliest issue instant of its day over the zones, so `ModelPipeline` judges every zone's training rows as of it.
    """

    def __init__(
        self,
        model: BaseModel,
        *,
        over: Union[str, Sequence[str]] = (ZONE,),
        zone_feature: Optional[str] = ONEHOT,
        missing: str = MISSING_RAISE,
        name: Optional[str] = None,
    ):
        self.model = model
        self.over = over
        self.zone_feature = zone_feature
        self.missing = missing
        self.name = model.name if name is None else name

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "model":
            check_pooled_model(value)
        elif key == "over":
            value = check_over(value)
        elif key == "zone_feature" and value not in ZONE_FEATURES:
            raise ValueError(f"zone_feature must be one of {list(ZONE_FEATURES)}, got {value!r}")
        elif key == "missing" and value not in MISSING_POLICIES:
            raise ValueError(f"missing must be one of {list(MISSING_POLICIES)}, got {value!r}")
        super().__setattr__(key, value)

    @property
    def pools_horizons(self) -> bool:
        """Whether one fit serves every horizon of its period and day (`"horizon"` in `over`)."""
        return HORIZON in self.over

    def identity(self, zones: Sequence[str]) -> Dict[str, Any]:
        """What each zone's manifest records as `pooled`, without the other zones' data: the zones in their order (the order
        of the stack), `over`, `zone_feature` and `missing`."""
        return {
            "zones": list(zones),
            "over": list(self.over),
            "zone_feature": self.zone_feature,
            "missing": self.missing,
        }

    def __repr__(self) -> str:
        return (
            f"PooledModel({type(self.model).__name__}({self.model.name!r}), over={self.over!r}, "
            f"zone_feature={self.zone_feature!r}, missing={self.missing!r}, name={self.name!r})"
        )


def check_pooled_model(model: Any) -> None:
    """A ValueError (TypeError for something that isn't a model) unless `PooledModel` can fit `model` over zones in this step."""
    if not isinstance(model, BaseModel):
        raise TypeError(f"PooledModel pools a model such as OLSModel, got {type(model).__name__}")
    what = f"PooledModel({type(model).__name__}({model.name!r}))"
    if model._fits_nothing:
        raise ValueError(
            f"{what}: {type(model).__name__} fits nothing, so there is no fit to pool. Run it in each zone."
        )
    if not type(model)._can_refit():
        raise ValueError(
            f"{what}: a pooled fit is made once and forecasts each zone's row from it (`_fit_model`, `_predict_fitted`), "
            f"which {type(model).__name__} doesn't define in one class with its `_fit_predict`."
        )
    if model._ctx_fn is not None:
        raise ValueError(
            f"{what}: a predictor function of `ctx` can't be pooled in this step; give the predictors as a list."
        )
    if len(model._members) > 1:
        raise ValueError(f"{what}: a list of training windows can't be pooled in this step; pool each window's model.")
    if model.newest_target != "d-1":
        raise ValueError(
            f"{what}: newest_target={model.newest_target!r} is refused for a pooled model: each zone issues on its own clock, "
            "and one fit serves them all, so its newest targets would be another zone's. Use newest_target='d-1'."
        )
