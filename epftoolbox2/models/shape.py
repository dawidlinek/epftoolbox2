"""`ShapeModel`: the within-hour shape of a 15-minute target on top of a stored hourly forecast (step 7f).

The resolution study's best structure, B24_dyn (`epftoolbox2-research/resolution/REPORT.md`, R1: 24 hourly models plus a model of
each quarter-hour's deviation from its hour's mean, tied with 96 per-quarter-hour models at a quarter of the fits), as two
ordinary pipelines and this model. The hourly pipeline runs first and stores its forecasts; a `ForecastSource` reads them into the
15-minute frame (a chained column, step 7a), where `ResampleTransformer` steps each hour's value over its quarter-hours. A
ShapeModel in a 15-minute run then:

- trains its learner on y - ȳ: each training row's target minus the mean of the target over the quarter-hours of its hour on its
  target date (an auxiliary target built in `_preprocess` by the same calendar shift as the targets, so it holds nothing the
  training cut excludes; NaN unless the hour has every quarter-hour with a value, and then the row drops);
- drops a training row whose target date isn't natively 15-minute: `native="distinct"` keeps a date where some hour has 4
  quarter-hours with values that aren't all equal (a step-filled hourly date has 4 equal ones in every hour), `native="YYYY-MM-DD"`
  keeps the dates from that one on (the study's "shape model on 15-minute days only", which avoided the shape dilution of a
  step-filled history);
- forecasts the issue row's `base` value plus the learner's forecast. Both go through the per-row target seam
  (`models._row_target.offset_config`: the offset is ȳ on a training row and `base` on the issue row);
- warns where `base` isn't a chained column (`InformationSet.chained_of`, registered by `with_source_rules`): the audit found the
  study trained B24_dyn on actual hourly ramps and served forecast ones; regressors built from a chained base are out-of-sample
  forecasts on the training rows and on the issue row alike.

The learner is any model that fits, of period or pooled units, with a list of predictor names (the shape regressors: the hourly
ramp from the chained base by hour lags, within-hour deviations of 15-minute forecasts from `PeriodStatsTransformer`). Its
`training_window`, `refit`, `scaler`, `newest_target` and `binary_mask` are the ShapeModel's: properties that read the learner. A
learner with no predictors and an intercept forecasts the per-slot mean deviation over its window (B24_static); `LEARModel([],
scaler="standard")` is one (LEAR forecasts the training mean of a fit of no column).

With `newest_target="published"` the newest target of each period is the oldest of its hour's periods (a quarter-hour's deviation
reads every quarter-hour of its hour), and with `native="distinct"` the oldest of its target date's periods (the flag reads every
hour of the date): a training row never reads a target its rule doesn't call known by the issue time.

The records are the target's (`actual` the quarter-hour's), on the keys of any 15-minute model of the run: the spring switch's
skipped hour has no row, the autumn switch's repeated hour no target there, so no training row, actual or forecast. They carry
the learner's coefficients. Not engine-capable, not poolable over zones (`PooledModel`), and no per-row target of its own or its
learner's in this step.
"""

import datetime
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._date_utils import shift_calendar_days
from ..data.transformers.period_stats import hour_starts
from ..results.store import outside_stacklevel
from . import _chained, _contract, _row_target
from ._calendar import NewestTargets, listed_days, periods_per_day
from ._history import reads_history
from ._units import DAY, PER_PERIOD, unit_kind
from .base import BaseModel, _without_repeats, hour_arrays_of

# `native`: a target date is native when some hour of it has quarter-hours that aren't all equal.
NATIVE_DISTINCT = "distinct"
# The run frequency a ShapeModel forecasts at, and the period its shape is taken over.
SHAPE_FREQ = "15min"
_PERIODS = {pd.Timedelta(hours=1): "1h"}


def hour_mean_column(target: str, horizon: int) -> str:
    """The period frames' auxiliary column of the target's hour mean at `horizon` (the mean over the quarter-hours of the hour
    of `{target}_d+{horizon}`, on its target date). Internal: no transformer makes such a name, and the run refuses a frame
    column of it."""
    return f"{target}_d+{horizon}~hour_mean"


def native_column(target: str, horizon: int) -> str:
    """The period frames' auxiliary 0/1 column of whether the target date of `{target}_d+{horizon}` is native (`native`)."""
    return f"{target}_d+{horizon}~native"


def hour_means(target: pd.Series, slots: int) -> np.ndarray:
    """Each row's mean of `target` over the slots of its hour (`PeriodStatsTransformer`'s `_hour_mean`, the same groups and the
    same sums): NaN unless all `slots` of the hour have a row with a value. Hours of the local clock, each occurrence of the
    repeated autumn hour on its own (`hour_starts`)."""
    keys = hour_starts(target.index)
    values = pd.DataFrame({"value": target.to_numpy(dtype=np.float64)})
    groups = values.groupby(keys)
    complete = groups.count().eq(slots)
    return groups.mean().where(complete).reindex(keys)["value"].to_numpy()


def distinct_dates(target: pd.Series, slots: int) -> np.ndarray:
    """Whether each row's local date has an hour of `slots` quarter-hours with values that aren't all equal (`native="distinct"`):
    False on a date whose every hour is step-filled from one hourly value, or has a slot missing."""
    index = target.index
    keys = hour_starts(index)
    values = pd.Series(target.to_numpy(dtype=np.float64))
    groups = values.groupby(keys)
    distinct = groups.count().eq(slots) & groups.max().ne(groups.min())
    wall = index.tz_localize(None) if index.tz is not None else index
    dates = np.asarray(wall.normalize().asi8)
    by_date = pd.Series(distinct.reindex(keys).to_numpy(dtype=bool)).groupby(dates).any()
    return by_date.reindex(dates).to_numpy(dtype=bool)


def _checked_native(value: Any, where: str) -> str:
    """`native` as kept: `"distinct"`, or a date as its ISO text (a YAML file writes it so)."""
    if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
        return value.isoformat()
    if value == NATIVE_DISTINCT:
        return NATIVE_DISTINCT
    if isinstance(value, str):
        try:
            return datetime.date.fromisoformat(value).isoformat()
        except ValueError:
            pass
    raise ValueError(
        f"{where}: native is 'distinct' (a date counts where some hour has quarter-hours that aren't all equal) or the first "
        f"natively 15-minute date, such as '2025-10-01', got {value!r}"
    )


def _checked_period(value: Any, where: str) -> str:
    try:
        length = pd.Timedelta(value) if isinstance(value, str) else None
    except (TypeError, ValueError):
        length = None
    if length not in _PERIODS:
        raise ValueError(
            f"{where}: period is '1h', the hour whose quarter-hours the shape is taken over, got {value!r}"
        )
    return value


def _checked_base(value: Any, where: str) -> str:
    if value is None:
        raise ValueError(f"{where}: base names the column of the hourly forecast, such as 'price_1h_fc_d+{{horizon}}'")
    return _row_target.check_column("base", value)


class ShapeModel(BaseModel):
    """A stored hourly forecast plus a learner's forecast of each quarter-hour's deviation from its hour's mean (see the module
    docstring).

    Args:
        base: The column of the hourly forecast, on every quarter-hour of its hour (`{horizon}` allowed): a chained column, such
            as `price_1h_fc_d+{horizon}` from a `ForecastSource` stepped over the quarter-hours by `ResampleTransformer` and
            lagged by `LagTransformer`. Read on the issue row only: the forecast is its value plus the learner's.
        learner: The model of the deviation, such as `OLSModel(shape_regressors, training_window=182, fit_intercept=True)`: its
            estimator, predictors (a list of names), training window, refit rule, scaler, newest target and binary-column rule.
        period: The period whose mean the shape is taken from: "1h".
        native: Which target dates train: "distinct" (a date where some hour has quarter-hours that aren't all equal, so not a
            step-filled hourly one) or the first natively 15-minute date, such as "2025-10-01".
        name: The model's name, which names its results.
    """

    _engine_ok = False
    # A PooledModel fits its member's units over zones, without this model's offset.
    _pooled_ok = False
    # Read by `ModelPipeline._load_component`: the parameter that holds a model, saved as {class, params}.
    _NESTED_MODELS = ("learner",)

    def __init__(
        self,
        base: str,
        learner: BaseModel,
        *,
        period: str = "1h",
        native: Union[str, datetime.date] = NATIVE_DISTINCT,
        name: str = "Shape",
    ):
        self.name = name  # first: the checks below name the model
        self.learner = learner
        self.base = base
        self.period = period
        self.native = native
        super().__init__(
            learner.predictors,
            learner.training_window,
            name,
            scaler=learner.scaler,
            newest_target=learner.newest_target,
            refit=learner.refit,
            binary_mask=learner.binary_mask,
        )

    @property
    def _where(self) -> str:
        return f"ShapeModel({getattr(self, 'name', '?')!r})"

    def __setattr__(self, key, value):
        where = f"ShapeModel({getattr(self, 'name', '?')!r})"
        if key == "base":
            value = _checked_base(value, where)
        elif key == "learner":
            self._check_learner(value, where)
        elif key == "period":
            value = _checked_period(value, where)
        elif key == "native":
            value = _checked_native(value, where)
        elif key in _row_target.PARAMETERS and value is not None:
            raise ValueError(
                f"{where}: {key} (a per-row target) isn't supported on a ShapeModel in this step: its learner's target is "
                "already the target less its hour's mean, and its forecast the base's value plus the learner's"
            )
        super().__setattr__(key, value)

    @staticmethod
    def _check_learner(learner: Any, where: str) -> None:
        from .residual import ResidualModel  # it imports base, as this module does

        if not isinstance(learner, BaseModel) or isinstance(learner, (ShapeModel, ResidualModel)):
            raise TypeError(
                f"{where}: learner must be a model that fits, such as OLSModel(...), got {type(learner).__name__}"
            )
        if learner._fits_nothing:
            raise ValueError(f"{where}: {type(learner).__name__} fits nothing, so it can't learn the within-hour shape")
        if (
            learner._ctx_fn is not None
            or learner._predictors_fn is not None
            or not all(isinstance(name, str) for name in learner.predictors)
        ):
            raise ValueError(
                f"{where}: the learner's predictors must be a list of names in this step (with {{horizon}})"
            )
        if not getattr(type(learner), "_residual_learner_ok", True):
            raise ValueError(
                f"{where}: {type(learner).__name__} isn't supported as a learner in this step (the ShapeModel's pool doesn't "
                "pin torch's threads): use a linear model or a tree"
            )
        given = [key for key in _row_target.PARAMETERS if getattr(learner, key, None) is not None]
        if given:
            raise ValueError(
                f"{where}: the learner has {', '.join(given)} (a per-row target), which isn't supported in a ShapeModel in "
                "this step: the learner's target is the target less its hour's mean"
            )
        if unit_kind(learner) == DAY:
            raise ValueError(
                f"{where}: a learner of day units isn't supported in this step: its fit's targets aren't one row's, which "
                "the offset of each row needs. Use a per-period or pooled one"
            )
        if reads_history(learner):
            raise ValueError(
                f"{where}: a learner that reads series (`_history_spec`) isn't supported in this step: its fits read series, "
                "not the rows whose target the offset changes"
            )

    # -- what is the learner's ---------------------------------------------------------------------------------------------

    def _same_as_learner(self, key: str, value: Any, theirs: Any) -> None:
        if "learner" in vars(self) and value != theirs:
            raise ValueError(
                f"{self._where}: {key} is the learner's ({theirs!r}); set it on the learner, not to {value!r}"
            )

    @property
    def predictors(self) -> List[str]:
        """The learner's predictors."""
        return self.learner.predictors

    @predictors.setter
    def predictors(self, value: Any) -> None:
        self._same_as_learner("predictors", value, self.learner.predictors)

    @property
    def training_window(self) -> Any:
        """The learner's training window."""
        return self.learner.training_window

    @training_window.setter
    def training_window(self, value: Any) -> None:
        self._same_as_learner("training_window", value, self.learner.training_window)

    @property
    def _members(self) -> Tuple[Any, ...]:
        """The learner's training windows, each fitted on its own (`BaseModel.training_window`)."""
        return self.learner._members

    @_members.setter
    def _members(self, value: Any) -> None:
        self._same_as_learner("training windows", tuple(value), tuple(self.learner._members))

    @property
    def refit(self) -> Optional[str]:
        """The learner's refit rule."""
        return self.learner.refit

    @refit.setter
    def refit(self, value: Optional[str]) -> None:
        self._same_as_learner("refit", value, self.learner.refit)

    @property
    def scaler(self) -> Any:
        """The learner's scaler."""
        return self.learner.scaler

    @scaler.setter
    def scaler(self, value: Any) -> None:
        self._same_as_learner("scaler", value, self.learner.scaler)

    @property
    def newest_target(self) -> str:
        """The learner's newest training target."""
        return self.learner.newest_target

    @newest_target.setter
    def newest_target(self, value: str) -> None:
        self._same_as_learner("newest_target", value, self.learner.newest_target)

    @property
    def binary_mask(self) -> Optional[str]:
        """The learner's binary-column rule."""
        return self.learner.binary_mask

    @binary_mask.setter
    def binary_mask(self, value: Optional[str]) -> None:
        self._same_as_learner("binary_mask", value, self.learner.binary_mask)

    @property
    def _unit_kind(self) -> str:
        """The learner's unit kind: a pooled learner's units hold every period's rows."""
        return unit_kind(self.learner)

    @property
    def target_transform(self) -> str:
        return getattr(self.learner, "target_transform", PER_PERIOD)

    @property
    def _card(self) -> Any:
        """The learner's card: its runtime packages are compared on resume."""
        return getattr(self.learner, "_card", None)

    def _runtime_fields(self) -> Dict[str, Any]:
        return self.learner._runtime_fields()

    def _executor_config(self) -> tuple:
        """The learner's: the run's pool fits the learner (a tree of `threads` threads per fit takes fewer processes)."""
        return self.learner._executor_config()

    def _preflight(self, seen: Optional[Dict] = None) -> List[Any]:
        return self.learner._preflight(seen)

    @property
    def _kernel_backend(self) -> Optional[str]:
        return self.learner._kernel_backend

    @property
    def block_predict(self) -> bool:
        """Whether a per-period learner predicts each refit block in one call (`_worker.block_calls` of the learner)."""
        from ._worker import block_calls

        return block_calls(self.learner)

    def _worker_model(self) -> Tuple[type, Dict]:
        """The learner's class and keyword arguments: the workers fit the learner's estimator on the deviations."""
        return type(self.learner), self.learner._model_kwargs

    # The estimator is the learner's: these are what `refit`'s checks ask one class to define (the workers rebuild the learner).
    def _fit_predict(self, train_x, train_y, test_x):
        return self.learner._fit_predict(train_x, train_y, test_x)

    def _fit_model(self, train_x, train_y):
        return self.learner._fit_model(train_x, train_y)

    def _predict_fitted(self, fitted, test_x):
        return self.learner._predict_fitted(fitted, test_x)

    def _predict_block(self, fitted, rows):
        return self.learner._predict_block(fitted, rows)

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        """What the manifest's `model_params` records besides the scaler: the base, the period, the native rule and the learner
        (its class and its own keyword arguments; its list, window, refit, newest target and mask rule are recorded as this
        model's)."""
        return {
            "base": self.base,
            "period": self.period,
            "native": self.native,
            "learner": {"class": type(self.learner).__name__, "params": dict(self.learner._model_kwargs)},
        }

    # -- the frame -----------------------------------------------------------------------------------------------------------

    def _set_freq(self, freq: str) -> None:
        if freq != SHAPE_FREQ:
            raise ValueError(
                f"{self._where}: a ShapeModel forecasts the within-hour shape of a 15-minute run (freq='15min'), got "
                f"freq={freq!r}. Run the hourly model in a pipeline of its own and chain its forecasts in"
            )
        super()._set_freq(freq)

    @property
    def _slots(self) -> int:
        """Quarter-hours of an hour: the periods of one hour's shape."""
        return periods_per_day(SHAPE_FREQ) // 24

    def _native_rows(self, target: pd.Series) -> np.ndarray:
        """Whether each row's local date is a native 15-minute date by `native`."""
        if self.native == NATIVE_DISTINCT:
            return distinct_dates(target, self._slots)
        index = target.index
        wall = index.tz_localize(None) if index.tz is not None else index
        return np.asarray(wall.normalize() >= pd.Timestamp(self.native))

    def _preprocess(self, data: pd.DataFrame, horizon: int, target: str) -> pd.DataFrame:
        """BaseModel's frame, with the auxiliary columns of each horizon h: the hour mean of `{target}_d+{h}` (`hour_mean_column`)
        and whether its target date is native (`native_column`, 1 or 0), each by the calendar shift that builds `{target}_d+{h}`
        (NaN where the target is: the slot skipped in spring, the repeated autumn hour from a day that has it once)."""
        out = super()._preprocess(data, horizon, target)
        index = out.index
        means = pd.Series(hour_means(out[target], self._slots), index=index)
        native = pd.Series(self._native_rows(out[target]).astype(np.float64), index=index)
        columns = {}
        for h in range(1, horizon + 1):
            future = shift_calendar_days(index, h, ambiguous="NaT")
            for name, series in ((hour_mean_column(target, h), means), (native_column(target, h), native)):
                aligned = series.reindex(future)
                aligned.index = index
                columns[name] = aligned
        return pd.concat([out, pd.DataFrame(columns)], axis=1)

    def _build_hour_arrays(self, horizon: int) -> Dict[int, Dict]:
        """The period arrays with the auxiliary columns of every horizon after the inputs: what the worker's offset reads."""
        extra = [
            name
            for h in range(1, horizon + 1)
            for name in (hour_mean_column(self._target, h), native_column(self._target, h))
        ]
        return hour_arrays_of(
            self._hour_data,
            self._hour_days,
            self._target,
            horizon,
            [*self._used_predictors, *extra],
            self._periods_per_day,
        )

    def _worker_inputs(self, state: Any, threads_per_process: int) -> Tuple[Dict, Dict]:
        """BaseModel's, with the offset (`models._row_target.offset_config`): on a training row the target's hour mean, on the
        issue row `base`; a training row whose target date isn't native drops."""
        expanded, config = super()._worker_inputs(state, threads_per_process)
        horizons = sorted(state.horizons)
        target = self._target
        config["row_target"] = _row_target.offset_config(
            {h: hour_mean_column(target, h) for h in horizons},
            {h: _row_target.expand(self.base, h) for h in horizons},
            {h: native_column(target, h) for h in horizons},
        )
        return expanded, config

    def _newest_targets(
        self,
        data: pd.DataFrame,
        target: str,
        horizon: int,
        freq: str,
        rules: Any,
        issue_days: Optional[Sequence[datetime.date]],
    ) -> Optional[NewestTargets]:
        """BaseModel's newest targets (`newest_target="published"`), each period's the oldest of its hour's periods (the hour
        mean reads them all), or with `native="distinct"` of its target date's (the native flag reads every hour of the date)."""
        found = super()._newest_targets(data, target, horizon, freq, rules, issue_days)
        if found is None or not found.k.size:
            return found
        periods = found.k.shape[0]
        groups = (
            np.zeros(periods, dtype=np.int64) if self.native == NATIVE_DISTINCT else np.arange(periods) // self._slots
        )
        k = found.k.copy()
        for group in np.unique(groups):
            on = groups == group
            k[on] = found.k[on].min(axis=0)
        return NewestTargets(found.days, k)

    # -- the checks ----------------------------------------------------------------------------------------------------------

    def _check_inputs(
        self,
        data: pd.DataFrame,
        target: str,
        horizon: int,
        freq: str,
        information_set: Any = None,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
        plan: Any = None,
        availability: str = "raise",
        windows: Any = None,
        cut_rules: Any = None,
        horizons: Optional[Sequence[int]] = None,
        weekdays: Optional[Sequence[int]] = None,
        stacklevel: int = 4,
        shared: Any = None,
    ) -> Dict[Tuple[int, int], List[str]]:
        """The learner's lists as BaseModel checks a model's, then the base: a numeric frame column, an input the model reads
        (`_used_predictors`, what `verify` probes and the manifest hashes), judged by the information set on the issue row alone
        (no training row reads it), late-upstream-checked as a chained column (`models._chained`, without coverage: a missing
        base skips its forecast as `missing_test_predictors`), and a warning where it isn't a chained column."""
        if plan is not None:
            raise ValueError(f"{self.name}: the learner's predictors are a list; a ShapeModel takes no plan")
        # Again at the run: a learner changed after this model was built.
        self._check_learner(self.learner, self._where)
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
        planned = tuple(range(1, horizon + 1)) if horizons is None else tuple(horizons)
        bases = {
            (period, hz): [_row_target.expand(self.base, hz)]
            for period in range(self._periods_per_day)
            for hz in planned
        }
        names = list(dict.fromkeys(name for listed in bases.values() for name in listed))
        _contract.check_predictors(data, bases, target, horizon)
        internal = {
            name for h in range(1, horizon + 1) for name in (hour_mean_column(target, h), native_column(target, h))
        }
        clash = sorted(internal & {*self._used_predictors, *names, target})
        if clash:
            raise ValueError(
                f"{self.name}: {clash} are the ShapeModel's own auxiliary columns: rename the frame column"
            )
        self._used_predictors = list(dict.fromkeys([*self._used_predictors, *names]))
        issued = self._days_of(test_start, test_end, windows)
        found, by_window = self._judge_lists(
            bases,
            data,
            target,
            freq,
            information_set,
            test_start,
            test_end,
            availability,
            windows,
            weekdays,
            None,
            issued,
            check_target=False,
            issue_row_only=True,
        )
        if found is not None:
            self._availability_listing = _without_repeats(self._availability_listing, found)
            for window, rows in by_window.items():
                self._window_listings[window] = _without_repeats(self._window_listings.get(window, []), rows)
        self._check_base_chained(data, bases, names, information_set, cut_rules, availability, issued, weekdays)
        return expanded

    def _check_base_chained(
        self,
        data: pd.DataFrame,
        bases: Dict[Tuple[int, int], List[str]],
        names: List[str],
        information_set: Any,
        cut_rules: Any,
        availability: str,
        issued: Optional[Sequence[datetime.date]],
        weekdays: Optional[Sequence[int]],
    ) -> None:
        """A warning where a base name isn't read from another pipeline's stored forecasts; the late-upstream check of one that
        is (`_chained.check` without the coverage of training rows, which don't read the base), its record merged into the
        manifest's `chained_inputs` and its rows into the listing."""
        rules = information_set if information_set is not None else cut_rules
        unchained = [name for name in names if rules is None or rules.chained_of(name) is None]
        if unchained:
            what = (
                "can't be told to be a chained column without an information set"
                if rules is None
                else "is no chained column (a ForecastSource's, registered by information_set.with_source_rules(sources))"
            )
            warnings.warn(
                f"{self.name}: the base ({', '.join(unchained)}) {what}, so the forecast may not be built on out-of-sample "
                "forecasts of the hourly level: regressors built from the base may hold one thing on the training rows and "
                "another on the issue row. Chain the hourly model's stored forecasts in with a ForecastSource.",
                UserWarning,
                stacklevel=outside_stacklevel(),
            )
        if information_set is None:
            return
        days = None if issued is None else listed_days(issued, weekdays)
        found = _chained.check(self.name, data, bases, information_set, availability, days, None, self._freq)
        if found.record:
            self._chained_record = {**found.record, **(self._chained_record or {})}
        if found.listing:
            self._availability_listing = _without_repeats(self._availability_listing, found.listing)
            self._window_listings = {
                window: _without_repeats(listed, found.listing) for window, listed in self._window_listings.items()
            }

    def __repr__(self) -> str:
        return (
            f"ShapeModel(base={self.base!r}, learner={type(self.learner).__name__}({self.learner.name!r}), "
            f"native={self.native!r}, name={self.name!r})"
        )


__all__ = ["ShapeModel", "NATIVE_DISTINCT", "hour_mean_column", "native_column", "hour_means", "distinct_dates"]
