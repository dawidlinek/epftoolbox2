from typing import Dict, List, Sequence, Tuple

from ._calendar import check_newest_target, check_refit
from .base import BaseModel


class NaiveModel(BaseModel):
    """A forecast that is one column's value on the row it is issued from: a published forecast taken as a benchmark, such as
    a TSO's week-ahead load forecast, or a naive "same as last week". Nothing is fitted, scaled or trained on, so a forecast
    needs no history and its record has no coefficients.

    Args:
        predictors: A list of one name, such as `["load_forecast_daily_max_d+{horizon}"]` (`{horizon}` and callables are
            expanded as for any model), or a function of `ctx` returning a list of one name for each forecast. More than one
            name raises.
        name: Display name for reports.
        known_only: With an information set, a forecast whose value its rule puts after the issue time is no forecast: the
            value wasn't published yet. Such a forecast is counted as `not_published` in `forecast_counts()`, and is neither
            refused nor listed, at every availability level and in `verify`. False judges the column as any model's input
            instead: refused at "raise", listed at "warn". Without an information set every forecast is made either way.

    `newest_target` and `refit` are a fit's: they aren't parameters, and setting either to anything but its default raises.
    """

    _WRITTEN_WHEN_SET = {**BaseModel._WRITTEN_WHEN_SET, "training_window": 0, "scaler": "standard"}
    _fits_nothing = True

    def __init__(self, predictors, name: str = "Naive", *, known_only: bool = True):
        if isinstance(predictors, str):
            raise TypeError(
                f"NaiveModel takes a list of one name, got the string {predictors!r}: write [{predictors!r}]"
            )
        if not callable(predictors) and len(predictors) > 1:
            raise ValueError(
                f"NaiveModel forecasts one column's value, and got {len(predictors)} names: {list(predictors)[:3]}"
            )
        if not isinstance(known_only, bool):
            raise TypeError(f"known_only must be True or False, got {known_only!r}")
        # No training rows: a forecast reads its issue row alone, so the frame needs no history before the test.
        super().__init__(predictors, training_window=0, name=name)
        self.known_only = known_only

    def __setattr__(self, key, value):
        if key == "newest_target" and check_newest_target(value) != "d-1":
            raise ValueError(
                f"NaiveModel fits nothing, so it has no newest training target: newest_target={value!r} has no meaning. "
                "Leave it 'd-1'."
            )
        if key == "refit" and check_refit(value) is not None:
            raise ValueError(f"NaiveModel fits nothing, so it has nothing to refit: refit={value!r}. Leave it None.")
        super().__setattr__(key, value)

    @property
    def _known_only(self) -> bool:
        return self.known_only

    def _precompute_expanded_predictors(self, horizons: Sequence[int]) -> Dict[Tuple[int, int], List[str]]:
        expanded = super()._precompute_expanded_predictors(horizons)
        for (period, horizon), names in expanded.items():
            if len(names) != 1:
                raise ValueError(
                    f"{self.name}: NaiveModel forecasts one column's value, and its predictors give {len(names)} names at "
                    f"period {period}, horizon {horizon}: {names[:3]}"
                )
        return expanded

    def _plan_predictors(self, *args, **kwargs):
        plan = super()._plan_predictors(*args, **kwargs)
        if plan is None:
            return None
        if plan.uses_matching:
            raise ValueError(
                f"{self.name}: NaiveModel trains on nothing, so a list with train_on='matching' has no meaning. Return a plain "
                "list of one name."
            )
        many = [names for names in plan.lists if len(names) != 1]
        if many:
            raise ValueError(
                f"{self.name}: NaiveModel forecasts one column's value, and its predictor function returned "
                f"{len(many[0])} names: {list(many[0])[:3]}"
            )
        return plan

    def _fit_predict(self, train_x, train_y, test_x):
        raise NotImplementedError(
            "NaiveModel fits nothing: a forecast is read from its issue row (`_worker._row_value`)"
        )
