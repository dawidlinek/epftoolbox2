"""`ResidualModel`: a base model's out-of-sample forecasts corrected by a learner fitted on their residuals (step 6k).

The research's best classical model (`zoo_classic/REPORT.md`: LEAR-BIC or ridge w728 as the base, a boosted tree fitted weekly on
the base's out-of-sample residuals, 4 % better than LEAR's window average on DE and AT). A run has two stages:

1. **The base** runs as a plain run of its own over a warm-up before the run's first issue day through its last one, into
   `<name>.base.jsonl` next to the model's results (own manifest; in memory without a `save_dir`). Its records are what a plain
   run of the base over those days gives, bit for bit, and it resumes from its file as any run. Every day is an issue day of it,
   whatever `issue_days` says: the learner trains on every date's rows.
2. **The learner** trains, per refit unit, on the rows of the issue days in its span whose base forecast exists, with the
   target `y - base forecast` (the base's own record for that row's period, horizon and issue day: an out-of-sample forecast,
   never a fitted value), each period's residuals through their own Invariant (`ztree.py:97-105`); a forecast is the issue day's
   base forecast plus the learner's prediction mapped back. A row dated D' at horizon h has the target of D' + h, and the
   learner's span (`training_span`) ends where its cut does, so a base forecast enters a learner row only with a target known by
   then.

The learner's list may name the base's forecasts, issued on the row's date: `base_fc_d+{horizon}` (the row's own period) and
`base_fc_hHH_d+{horizon}` (period HH of the target day, the same on every row of a date; `base_fc_sNN_d+{horizon}` at a
sub-hourly freq). They are known at the issue time by construction, so the frame and information-set checks leave them out.

Records are the learner stage's, with `prediction` the corrected forecast, `actual` the target, and `base_prediction` (the base's
forecast it corrects) after every other key. A key with no base forecast on its issue day is skipped as `no_base_forecast`.

Warm-up (`ResidualModel._warmup_start`), the base's first issue day: `warmup="YYYY-MM-DD"` that date, whatever the run's days
(recommended: no forecast then depends on where the test starts); `warmup=k` k days before the first issue day (the research's
form: every forecast depends on that day); without `warmup=`, for an int learner window W the earliest row the learner's first
fit reads (the anchor of the first issue day minus W, the largest horizon and how far back a published target reaches), for
`"expanding"` the base's first day with its largest int window filled (the frame's first date plus that window, its deepest lag
and the largest horizon, on a refit day of the base's). An expanding base has no such day (its first fits train on a handful of
rows, and every later learner fit would include their residuals), so it needs `warmup=`. A warm-up the frame can't hold is
refused, naming the day the frame must start on. With `warmup=`, the manifest's compared `residual` field records the base's
start, so a resume from another one refuses.
"""

import copy
import datetime
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._date_utils import resolve_date
from ..results.availability import merge_listings
from ..results.manifest import ManifestMismatchError, _block_params, configuration_digest, manifest_path
from ..results.store import is_parquet_path, outside_stacklevel
from . import _contract
from ._cards import read_provenance, run_provenance
from ._calendar import (
    EXPANDING,
    FREQ_STEPS,
    horizon_arguments,
    periods_per_day,
    refit_anchor,
    refit_block_end,
    resolve_horizons,
)
from ._units import DAY, PER_PERIOD, PERIOD, unit_kind
from ._worker import block_calls, issue_row
from .base import BaseModel, _days_back, _test_days

# The names a learner's list gives the base's forecasts: `base_fc_d+{h}` (the row's own period) and `base_fc_hHH_d+{h}` or
# `base_fc_sNN_d+{h}` (period HH or NN of the target day, day-level), always at the list's own horizon.
BASE_PREFIX = "base_fc"
_BASE_NAME = re.compile(r"base_fc(?:_([hs])(\d{2}))?_d\+(\d+)")
# Where the base's results go, next to the model's: `<name>.base.jsonl` (and the base copy's name, `<name>.base`).
BASE_SUFFIX = ".base"
# A key whose issue day has no base forecast (the base skipped it): nothing to correct.
SKIP_NO_BASE_FORECAST = "no_base_forecast"


def is_base_name(name: str) -> bool:
    """Whether a predictor name is one of a base's forecasts (`base_fc...`), which a `ResidualModel` makes, not the frame."""
    return isinstance(name, str) and name.startswith(BASE_PREFIX)


def parse_base_name(name: str, freq: str) -> Tuple[Optional[int], int]:
    """(the period of the target day it reads, None for the row's own; the horizon) of a base forecast's name, at `freq`; a
    ValueError naming the forms for anything else: `base_fc_d+{horizon}`, `base_fc_h05_d+{horizon}` hourly, `base_fc_s20_d+...`
    at 15 or 30 minutes."""
    found = _BASE_NAME.fullmatch(name)
    hourly = FREQ_STEPS[freq] == pd.Timedelta(hours=1)
    forms = "base_fc_d+{horizon}" + (
        "" if periods_per_day(freq) == 1 else f" or base_fc_{'h05' if hourly else 's20'}_d+{{horizon}}"
    )
    if found is None:
        raise ValueError(f"{name!r} isn't a base forecast's name: a learner's list names them {forms}")
    letter, slot, horizon = found.groups()
    if letter is None:
        return None, int(horizon)
    period = int(slot)
    if periods_per_day(freq) == 1 or letter != ("h" if hourly else "s") or period >= periods_per_day(freq):
        raise ValueError(
            f"{name!r} names no period of a {freq} day: a learner's list names the base's forecasts {forms}"
        )
    return period, int(horizon)


def _date(value: Any) -> datetime.date:
    return pd.Timestamp(value).date()


def _later(day: datetime.date, days: int) -> datetime.date:
    return day + datetime.timedelta(days=int(days))


def _class_default(cls: type, key: str) -> Any:
    """The default of `key` in the first `__init__` of `cls`'s method resolution order that names it (inspect.Parameter.empty
    where none does)."""
    import inspect

    for klass in cls.__mro__:
        init = vars(klass).get("__init__")
        if init is None:
            continue
        parameters = inspect.signature(init).parameters
        if key in parameters:
            return parameters[key].default
    return inspect.Parameter.empty


def _normal_window(value: Any) -> Any:
    """A training window as a model keeps it (`resolve_training_window`), or `value` itself where it is none."""
    from ._calendar import resolve_training_window

    try:
        return resolve_training_window(value)[0]
    except (TypeError, ValueError):
        return value


def _model_description(model: BaseModel, calendar: bool = True) -> Dict[str, Any]:
    """What defines a nested model's forecasts, as plain JSON for a manifest's `model_params`: its class, list, newest target,
    binary-column rule, scaler and its own `_model_kwargs`, and with `calendar` (the base's) its name, window and refit rule, which
    the learner takes from the ResidualModel. A `(horizon)` function's list shows as an empty list here: the base's own manifest,
    whose configuration the `residual` field hashes, records its lists."""
    out = {"class": type(model).__name__}
    if calendar:
        out.update(name=model.name, training_window=model.training_window, refit=getattr(model, "refit", None))
    out.update(
        predictors=[name for name in model.predictors if isinstance(name, str)],
        newest_target=getattr(model, "newest_target", "d-1"),
        binary_mask=getattr(model, "binary_mask", None),
        scaler=model.scaler,
        params=dict(model._model_kwargs),
    )
    return out


def _check_base(base: Any, where: str) -> None:
    from .pooled import PooledModel  # it imports the models

    if isinstance(base, PooledModel):
        raise ValueError(
            f"{where}: a PooledModel can't be a base: it fits over the zones of a run of several and has no run of its own "
            "on one frame. Give the base it pools, which runs in each zone"
        )
    if not isinstance(base, BaseModel):
        raise TypeError(
            f"{where}: base must be a model, such as LEARModel(...), or the name of a model added to the pipeline before it, "
            f"got {type(base).__name__}"
        )
    if isinstance(base, ResidualModel):
        raise ValueError(f"{where}: a ResidualModel's base can't be a ResidualModel in this step")
    if base._ctx_fn is not None:
        raise ValueError(
            f"{where}: a base with a predictor function of `ctx` isn't supported in this step: the checks and `verify` read "
            "the base's inputs from its lists. Give the base's predictors as a list."
        )


class ResidualModel(BaseModel):
    """A base model's out-of-sample forecasts plus a learner's forecast of their residual (see the module docstring).

    Args:
        base: The model whose forecasts are corrected: a model (saved inside this one in YAML) or the name of a model added to
            the pipeline before this one, whose copy it runs (saved as the name). Its own forecasts go to `<name>.base.jsonl`.
        learner: The model fitted on the residuals, such as `HistGBModel(predictors=...)`: its estimator, predictors (a list of
            names, which may name the base's forecasts), scaler and newest target. Its `training_window`, `refit` and `name` are
            this model's.
        training_window: The learner's days of rows: `"expanding"` (every row with a base forecast, the default) or an int.
        refit: The learner's refit calendar (`"7D"`, Mondays, by default); a refit is required.
        warmup: Where the base's forecasts start: a date (`"2022-01-01"`, recommended: no forecast then depends on the test's
            first day), or a number of days before the first issue day (every forecast then depends on that day); None works it
            out, which an expanding base can't (see the module docstring).
        name: The model's name, which names its results; the base's are `<name>.base`.
    """

    _engine_ok = False
    # A PooledModel fits its members' units over zones and would skip the base stage.
    _pooled_ok = False
    # Read by `ModelPipeline._load_component`: parameters that hold a model, saved as {class, params}.
    _NESTED_MODELS = ("base", "learner")
    # What a nested model's YAML leaves out: the learner's own window, refit rule and name, which this model replaces.
    _NESTED_OMIT = {"learner": ("training_window", "refit", "name")}
    _WRITTEN_WHEN_SET = {**BaseModel._WRITTEN_WHEN_SET, "warmup": None}

    def __init__(
        self,
        base: Union[BaseModel, str],
        learner: BaseModel,
        training_window: Union[int, str] = "expanding",
        refit: Optional[str] = "7D",
        warmup: Union[int, str, datetime.date, None] = None,
        name: str = "residual",
    ):
        self.name = name  # first: the checks below name the model
        self._named_base: Optional[BaseModel] = None  # a pipeline's model `base` names (`bind_base`)
        # The base's forecasts during a run: (period, horizon, issue date ordinal) -> forecast.
        self._base_forecasts: Optional[Dict[Tuple[int, int, int], float]] = None
        self._residual: Optional[Dict[str, Any]] = None  # the manifest's `residual` field during a run
        self._actuals: Dict[int, np.ndarray] = {}  # each period's targets, as the learner's arrays had them
        self._actual_days: Dict[int, np.ndarray] = {}
        self._residual_base: Optional[Dict[str, Any]] = (
            None  # the base's card for the manifest's provenance during a run
        )
        self._base_check: Optional[Tuple[BaseModel, Any]] = None  # the base copy and its check, from `_check_inputs`
        self.base = base
        self.learner = learner
        self.warmup = warmup
        super().__init__(
            learner.predictors,
            training_window,
            name,
            scaler=learner.scaler,
            newest_target=learner.newest_target,
            refit=refit,
            binary_mask=learner.binary_mask,
        )
        self._warn_about_the_learner_s_own()

    # -- what is the learner's -------------------------------------------------------------------------------------------

    def __setattr__(self, key, value):
        where = f"ResidualModel({getattr(self, 'name', '?')!r})"
        if key == "base":
            if isinstance(value, str):
                if not value:
                    raise ValueError(f"{where}: base names no model")
            else:
                _check_base(value, where)
        elif key == "learner":
            self._check_learner(value, where)
        elif key == "warmup" and value is not None:
            value = self._checked_warmup(value, where)
        elif key == "training_window" and isinstance(value, (list, tuple, range)) and len(value) != 1:
            raise ValueError(
                f"{where}: a list of training windows isn't supported for a ResidualModel in this step: one int or 'expanding'"
            )
        elif key == "refit" and value is None:
            raise ValueError(
                f"{where}: the learner fits on the refit calendar: give refit ('7D' weekly, '1D' every day)"
            )
        super().__setattr__(key, value)

    @staticmethod
    def _checked_warmup(value: Any, where: str) -> Union[int, str]:
        """`warmup` as kept: a whole number of days from 1, or a date as its ISO text (a YAML file writes it so)."""
        if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
            return value.isoformat()
        if isinstance(value, str):
            try:
                return datetime.date.fromisoformat(value).isoformat()
            except ValueError:
                pass
        elif not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)) and value >= 1:
            return int(value)
        raise ValueError(
            f"{where}: warmup is the base's first issue day as a date ('2022-01-01'), a whole number of days from 1 before the "
            f"first issue day, or None, got {value!r}"
        )

    @staticmethod
    def _check_learner(learner: Any, where: str) -> None:
        if not isinstance(learner, BaseModel) or isinstance(learner, ResidualModel):
            raise TypeError(f"{where}: learner must be a model such as HistGBModel(...), got {type(learner).__name__}")
        if learner._fits_nothing:
            raise ValueError(f"{where}: {type(learner).__name__} fits nothing, so it can't learn the residuals")
        if (
            learner._ctx_fn is not None
            or learner._predictors_fn is not None
            or not all(isinstance(name, str) for name in learner.predictors)
        ):
            raise ValueError(
                f"{where}: the learner's predictors must be a list of names in this step (with {{horizon}})"
            )
        kind = unit_kind(learner)
        if kind == DAY:
            raise ValueError(
                f"{where}: a learner of day units isn't supported in this step: use a pooled or per-period one"
            )
        if not type(learner)._can_refit(kind):
            raise ValueError(f"{where}: {type(learner).__name__} can't fit on a refit calendar, which a learner does")

    def _warn_about_the_learner_s_own(self) -> None:
        """A warning where the learner was given a training window or refit rule of its own, other than its class's default,
        that this model replaces. The default is that of the first class of the learner's method resolution order whose
        `__init__` names the parameter (`OLSModel` takes `refit` through `**kw`: BaseModel's), compared normalised, so a
        learner left at its defaults, or loaded from a file that leaves them out, gives none."""
        for key in ("training_window", "refit"):
            mine, theirs = getattr(self, key), getattr(self.learner, key, None)
            default = _class_default(type(self.learner), key)
            if key == "training_window":
                theirs, default = _normal_window(theirs), _normal_window(default)
            if theirs != mine and theirs != default:
                warnings.warn(
                    f"ResidualModel({self.name!r}): the learner's {key}={theirs!r} is replaced by the ResidualModel's {mine!r}: "
                    f"the learner fits with this model's {key}; give it here",
                    UserWarning,
                    stacklevel=outside_stacklevel(),
                )

    def _same_as_learner(self, key: str, value: Any, theirs: Any) -> None:
        if "learner" in vars(self) and value != theirs:
            raise ValueError(
                f"ResidualModel: {key} is the learner's ({theirs!r}); set it on the learner, not to {value!r}"
            )

    @property
    def predictors(self) -> List[str]:
        """The learner's predictors."""
        return self.learner.predictors

    @predictors.setter
    def predictors(self, value: Any) -> None:
        self._same_as_learner("predictors", value, self.learner.predictors)

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
        """The learner's card: its runtime packages (scikit-learn for `HistGBModel`) are compared on resume."""
        return getattr(self.learner, "_card", None)

    def _runtime_fields(self) -> Dict[str, Any]:
        return self.learner._runtime_fields()

    def _preflight(self, seen: Optional[Dict] = None) -> List[Any]:
        """The learner's and the base's: the run fits both."""
        base = self.base if isinstance(self.base, BaseModel) else self._named_base
        return self.learner._preflight(seen) + ([] if base is None else base._preflight(seen))

    def _stage_learner(self) -> BaseModel:
        """A copy of the learner with this model's training window and refit rule: what the learner stage fits with."""
        stage = copy.deepcopy(self.learner)
        stage.training_window = self.training_window
        stage.refit = self.refit
        return stage

    @property
    def block_predict(self) -> bool:
        """Whether a per-period learner predicts each block in one call (`_worker.block_calls` of the learner); a pooled
        learner always does."""
        return block_calls(self._stage_learner())

    def _worker_model(self) -> Tuple[type, Dict]:
        """The learner's class and keyword arguments: the workers fit the learner's estimator on the residual rows."""
        stage = self._stage_learner()
        return type(stage), stage._model_kwargs

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
        out = {
            "base": _model_description(self._base_model()),
            "learner": _model_description(self.learner, calendar=False),
        }
        if self.warmup is not None:
            out["warmup"] = self.warmup
        return out

    def _check_settings(self) -> Tuple[Any, ...]:
        return (super()._check_settings(), self._base_model()._check_settings(), self.learner._check_settings())

    # -- the base ----------------------------------------------------------------------------------------------------------

    def bind_base(self, model: Any) -> None:
        """Run a copy of `model` as the base `base` names (what `ModelPipeline` does with `base="name"`)."""
        _check_base(
            model, f"ResidualModel({self.name!r}): base={self.base!r} names {type(model).__name__}({model.name!r})"
        )
        self._named_base = model

    def _base_model(self) -> BaseModel:
        """A copy of the base, named `<name>.base`: what the run fits."""
        source = self.base if isinstance(self.base, BaseModel) else self._named_base
        if source is None:
            raise ValueError(
                f"{self.name}: base={self.base!r} names a model of a pipeline: add this model to the ModelPipeline after "
                "that one, or pass the model itself as base"
            )
        base = copy.deepcopy(source)
        base.name = f"{self.name}{BASE_SUFFIX}"
        return base

    @staticmethod
    def base_path(save_to: Union[str, Path]) -> Path:
        """Where the base's results go for a model whose results are `save_to`: `<name>.base.jsonl` next to them."""
        path = Path(save_to)
        return path.with_name(path.stem + BASE_SUFFIX + path.suffix)

    def _issue_span(
        self, test_start: Any, test_end: Any, windows: Any, forecast_only: bool = False
    ) -> Optional[Tuple[datetime.date, datetime.date]]:
        """(first, last) issue day of a run, None for a check without a test period."""
        if windows is not None:
            return _date(windows.first_day), _date(windows.last_day)
        if forecast_only:
            return _date(resolve_date(test_start or "today")), _date(resolve_date(test_end or "today"))
        if test_start is None or test_end is None:
            return None
        return _date(resolve_date(test_start)), _date(resolve_date(test_end))

    def _base_full_window(
        self,
        base: BaseModel,
        data: pd.DataFrame,
        target: str,
        horizon: int,
        horizons: Sequence[int],
        freq: str,
        rules: Any,
        span: Tuple[datetime.date, datetime.date],
    ) -> datetime.date:
        """The first day the base forecasts on with its largest int window filled: the frame's first date plus that window (none
        for a model that fits nothing; an expanding base is refused before), the deepest lag of its lists, the largest horizon
        and how far back its newest target can be (`"published"`), on a refit day of the base's if it refits."""
        base._set_freq(freq)
        first = _contract.wall_clock(data.index).min().date()
        widths = [member for member in base._members if member != EXPANDING]
        window = 0 if base._fits_nothing or not widths else max(widths)
        lag, _ = _days_back(name for names in base._precompute_expanded_predictors(horizons).values() for name in names)
        newest = base._newest_targets(data, target, horizon, freq, rules, _test_days(*(d.isoformat() for d in span)))
        reach = 0 if newest is None else int(newest.reach)
        day = _later(first, window + lag + horizon + reach)
        block = getattr(base, "_refit_block", None) if getattr(base, "refit", None) is not None else None
        if block is not None and refit_anchor(day, block) < day:
            day = _later(refit_block_end(day, block), 1)
        return day

    def _warmup_start(
        self,
        base: BaseModel,
        data: pd.DataFrame,
        target: str,
        horizon: int,
        horizons: Sequence[int],
        freq: str,
        rules: Any,
        span: Tuple[datetime.date, datetime.date],
    ) -> datetime.date:
        """The base's first issue day (see the module docstring); a ValueError where the frame can't hold the warm-up."""
        first_issue = span[0]
        frame_first = _contract.wall_clock(data.index).min().date()
        if self.warmup is not None:
            dated = isinstance(self.warmup, str)
            start = datetime.date.fromisoformat(self.warmup) if dated else _later(first_issue, -self.warmup)
            if start < frame_first:
                raise ValueError(
                    f"{self.name}: warmup={self.warmup!r} starts the base's forecasts on {start}, before the frame's first date "
                    f"{frame_first}: start the frame on {start} or earlier, or give a later warmup"
                )
            if start > first_issue:
                raise ValueError(
                    f"{self.name}: warmup={self.warmup!r} starts the base's forecasts on {start}, after the first issue day "
                    f"{first_issue}: the learner would have no residual to train on there. Give an earlier date"
                )
            return start
        if not base._fits_nothing and all(member == EXPANDING for member in base._members):
            raise ValueError(
                f"{self.name}: an expanding base has no first day with a full training window: its first fits train on a "
                "handful of rows from the frame's first dates, and every later learner fit would include their residuals. Give "
                "warmup= the date its forecasts start on (one with as many days of rows before it as you want it to have)"
            )
        full = self._base_full_window(base, data, target, horizon, horizons, freq, rules, span)
        if self.training_window == EXPANDING:
            if full > first_issue:
                raise ValueError(
                    f"{self.name}: the base forecasts with its largest window filled from {full} on, after the first issue day "
                    f"{first_issue}, so the learner would have no residual to train on: start the frame earlier, the test "
                    "later, or give warmup= days of base forecasts to train on"
                )
            return full
        anchor = refit_anchor(first_issue, self._refit_block)
        newest = self._newest_targets(data, target, horizon, freq, rules, _test_days(*(d.isoformat() for d in span)))
        reach = 0 if newest is None else int(newest.reach)
        start = _later(anchor, -(self.training_window + horizon + reach))
        if start < full:
            raise ValueError(
                f"{self.name}: the learner's first fit (refit day {anchor}, training window {self.training_window}) trains on "
                f"rows from {start}, but the base forecasts with its largest window filled only from {full} on: start the frame "
                f"{(full - start).days} days earlier, the test later, or give warmup= to accept shorter base windows"
            )
        return start

    # -- the checks --------------------------------------------------------------------------------------------------------

    def _checked_lists(self, expanded: Dict[Tuple[int, int], List[str]]) -> Dict[Tuple[int, int], List[str]]:
        """The learner's lists without the base's forecasts, which this model makes: known at the issue time by construction."""
        return {
            key: [name for name in names if not is_base_name(name)]
            for key, names in super()._checked_lists(expanded).items()
        }

    def _check_base_names(self, data: pd.DataFrame, freq: str, horizons: Sequence[int]) -> List[str]:
        """The base forecasts the learner's lists name, each once; a ValueError for a name of another form, one at another
        horizon than its list's, and a frame column of such a name."""
        names = []
        for (_, horizon), listed in self._precompute_expanded_predictors(horizons).items():
            for name in listed:
                if not is_base_name(name):
                    continue
                _, own = parse_base_name(name, freq)
                if own != horizon:
                    raise ValueError(
                        f"{self.name}: {name!r} in the list of horizon {horizon} is the base's forecast of another horizon: "
                        "write base_fc_d+{horizon} (or base_fc_hHH_d+{horizon}), the forecast the row's own horizon targets"
                    )
                if name in data.columns:
                    raise ValueError(
                        f"{self.name}: the frame has a column named {name!r}, which the learner's list names as the base's "
                        "forecast: rename the column"
                    )
                names.append(name)
        return list(dict.fromkeys(names))

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
        """The base's inputs on its days (the warm-up through the last issue day, every day), then the learner's (without the
        base's forecasts): the frame columns of both are this model's (`_used_predictors`, what `verify` probes and the
        manifest hashes), and what either uses late is listed for this model."""
        if plan is not None:
            raise ValueError(f"{self.name}: the learner's predictors are a list; a ResidualModel takes no plan")
        self._set_freq(freq)
        planned = tuple(range(1, horizon + 1)) if horizons is None else tuple(horizons)
        self._check_base_names(data, freq, planned)
        span = self._issue_span(test_start, test_end, windows)
        base = self._base_model()
        rules = self._cut_rules(information_set, cut_rules)
        start = end = None
        if span is not None:
            start = self._warmup_start(base, data, target, horizon, planned, freq, rules, span).isoformat()
            end = span[1].isoformat()
        # The base copy and its check, which the run uses (`_run_base` with `checked=`) where it is the run checked: one check.
        base_check = base._checked_inputs(
            data,
            target,
            horizon,
            freq,
            information_set,
            start,
            end,
            availability=availability,
            cut_rules=cut_rules,
            horizons=horizons,
            shared=shared,
        )
        self._base_check = (base, base_check)
        base_used, base_listing = list(base_check.used_predictors), list(base_check.listing)
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
        self._used_predictors = list(dict.fromkeys([*self._used_predictors, *base_used]))
        if base_listing:
            self._availability_listing = merge_listings(self._availability_listing, base_listing)
            self._window_listings = {
                window: merge_listings(rows, base_listing) for window, rows in self._window_listings.items()
            }
        return expanded

    # -- the run -----------------------------------------------------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame,
        test_start: str = None,
        test_end: str = None,
        target: str = "price",
        horizon: Optional[int] = None,
        freq: str = "1h",
        save_to: str = None,
        forecast_only: bool = False,
        information_set: Any = None,
        allow_manifest_mismatch: bool = False,
        require_manifest: bool = False,
        plan: Any = None,
        availability: str = "raise",
        warn_listing: bool = True,
        windows: Any = None,
        frame_hashes: Any = None,
        cut_rules: Any = None,
        horizons: Optional[Sequence[int]] = None,
        issue_days: Optional[Sequence[str]] = None,
        checked: Any = None,
        save_coefficients: bool = True,
    ):
        """The base over its warm-up and the run's days, then the learner on its residuals (see the module docstring); the
        arguments are `BaseModel.run`'s. The base gets the same target, horizons, freq, information set, level and store options,
        and runs every day from its warm-up start to the last issue day."""
        if save_to is not None and is_parquet_path(save_to):
            raise ValueError(
                f"{self.name}: a ResidualModel's records hold `base_prediction`, which the Parquet results schema has no "
                "column for: write its results as JSONL (save_format='jsonl')"
            )
        span = self._issue_span(test_start, test_end, windows, forecast_only)
        if span is None:
            raise ValueError("test_start and test_end are required unless forecast_only=True")
        largest, planned = resolve_horizons(horizon, horizons, 7)
        self._set_freq(freq)
        rules = self._cut_rules(information_set, cut_rules)
        base, base_check = self._base_model(), None
        if self._base_check is not None and self._base_check[0]._check_settings() == base._check_settings():
            base, base_check = self._base_check  # the copy `_check_inputs` checked, for this base as it is now
        self._base_check = None
        start = self._warmup_start(base, data, target, largest, planned, freq, rules, span)
        if save_to is not None and not allow_manifest_mismatch:
            self._refuse_before_the_base(save_to, start)
        self._warn_about_the_base_s_masks(base)
        base_to = None if save_to is None else str(self.base_path(save_to))
        self._base_forecasts, field = self._run_base(
            base,
            data,
            test_start=start.isoformat(),
            test_end=span[1].isoformat(),
            target=target,
            freq=freq,
            save_to=base_to,
            information_set=information_set,
            allow_manifest_mismatch=allow_manifest_mismatch,
            require_manifest=require_manifest,
            availability=availability,
            warn_listing=False,  # this model's listing holds the base's: warned for both
            frame_hashes=frame_hashes,
            cut_rules=cut_rules,
            save_coefficients=save_coefficients,
            **({} if base_check is None else {"checked": base_check}),
            **horizon_arguments(largest, planned),
        )
        self._residual = self._residual_field(field, start)
        self._residual_base = self._base_provenance(base, base_to, start)
        try:
            return super().run(
                data,
                test_start,
                test_end,
                target,
                horizon,
                freq,
                save_to,
                forecast_only,
                information_set,
                allow_manifest_mismatch,
                require_manifest,
                plan,
                availability,
                warn_listing,
                windows,
                frame_hashes,
                cut_rules,
                horizons,
                issue_days,
                checked,
                save_coefficients,
            )
        finally:
            self._base_forecasts = None
            self._residual = None
            self._residual_base = None
            self._base_check = None

    def _residual_field(self, field: Dict[str, Any], start: datetime.date) -> Dict[str, Any]:
        """The manifest's compared `residual` field: what `_run_base` gave (the base's configuration), and with `warmup=` the
        base's first issue day, on which every forecast depends then (with an int `warmup` it moves with the test's start)."""
        return dict(field) if self.warmup is None else {**field, "base_start": start.isoformat()}

    def _refuse_before_the_base(self, save_to: Union[str, Path], start: datetime.date) -> None:
        """A ManifestMismatchError, before the base runs and writes anything, where this model's results were made with
        another base, learner or warm-up start: what its manifest's `model_params` and `residual` say, compared as the
        manifest's own check would. The base's configuration digest is compared after its run, by that check."""
        path = manifest_path(Path(save_to))
        if not path.exists():
            return
        existing = json.loads(path.read_text())
        if existing.get("kind") is not None or "model_params" not in existing:
            return  # an ensemble's, or no model's: the manifest's own check says so
        mine = json.loads(json.dumps({"scaler": self.scaler, **self._model_kwargs, **_block_params(self)}, default=str))
        diffs = []
        if existing["model_params"] != mine:
            diffs.append(f"model_params: {existing['model_params']!r} (existing results) vs {mine!r} (this run)")
        theirs = (existing.get("residual") or {}).get("base_start")
        ours = None if self.warmup is None else start.isoformat()
        if theirs != ours:
            diffs.append(f"residual.base_start: {theirs!r} (existing results) vs {ours!r} (this run)")
        # The learner's own window and refit rule, which the manifest records outside `model_params`.
        window = json.loads(json.dumps(self.training_window))
        if existing.get("training_window") != window:
            diffs.append(
                f"training_window: {existing.get('training_window')!r} (existing results) vs {window!r} (this run)"
            )
        refit = (existing.get("training_cut") or {}).get("refit")
        if refit != self.refit:
            diffs.append(f"training_cut.refit: {refit!r} (existing results) vs {self.refit!r} (this run)")
        if diffs:
            raise ManifestMismatchError(
                f"{self.name}: the results in {path.parent} were produced with a different configuration than this run "
                f"({path.name}):\n" + "\n".join(f"  - {d}" for d in diffs) + "\nNothing was run or written, the base "
                "included. Use a different save_dir, remove the existing results, or pass allow_manifest_mismatch=True to "
                "overwrite the manifest and resume anyway."
            )

    def _warn_about_the_base_s_masks(self, base: BaseModel) -> None:
        """A warning where the base reads its binary-column masks on its run's first day (`refit=None`, `binary_mask=None`):
        that day is the warm-up start, not a run of the base's own, so `base_prediction` can differ from the records of the
        same model run beside this one (`tests/test_known_issues.py`, a forecast depends on the day its test starts)."""
        if (
            base._fits_nothing
            or getattr(base, "refit", None) is not None
            or getattr(base, "binary_mask", None) is not None
        ):
            return
        warnings.warn(
            f"{self.name}: its base {type(base).__name__} has refit=None and binary_mask=None, so each base forecast's "
            "binary-column mask is read on the base run's first day, the warm-up start: base_prediction can then differ from "
            "the records of the same model run on its own over other days. Give the base binary_mask='fit' (or refit) for "
            "forecasts that don't depend on where a run starts.",
            UserWarning,
            stacklevel=outside_stacklevel(),
        )

    def _base_provenance(
        self, base: BaseModel, base_to: Optional[str], start: datetime.date
    ) -> Optional[Dict[str, Any]]:
        """The base's card, runtime and weights for this model's provenance (`report.provenance()` shows it as a row of its
        own, judged from the base's first issue day), its manifest's where it wrote one; None for a base without a card."""
        if getattr(base, "_card", None) is None:
            return None
        found = None if base_to is None else read_provenance(manifest_path(Path(base_to)))
        if found is None:
            found = run_provenance(base, None, None, start.isoformat())
        return {**found, "first_issue_day": start.isoformat()}

    def _finish_run(self, state: Any) -> Any:
        ref = super()._finish_run(state)
        # The manifest's after a resume (the earliest first issue day of the base over the file's runs), else this run's.
        base = getattr(state.manifest, "residual_base", None) or self._residual_base
        if base is not None:
            ref.provenance = {**(ref.provenance or {}), "base": dict(base)}
        return ref

    @staticmethod
    def _run_base(
        base: BaseModel, data: pd.DataFrame, **arguments: Any
    ) -> Tuple[Dict[Tuple[int, int, int], float], Dict]:
        """Run the base (`base.run(data, **arguments)`) and return its forecasts by (period, horizon, issue date ordinal), finite
        ones only, and the manifest's `residual` field: the base's own manifest's configuration (`configuration_digest`), which
        this model's manifest compares on resume, or nothing for a base run in memory. Tests stand in for it to give stored
        forecasts."""
        ref = base.run(data, **arguments)
        forecasts: Dict[Tuple[int, int, int], float] = {}
        for record in ref.iter_records(cols=["hour", "horizon", "run_date", "prediction"]):
            value = record.get("prediction")
            if value is not None and math.isfinite(value):
                ordinal = datetime.date.fromisoformat(record["run_date"]).toordinal()
                forecasts[(int(record["hour"]), int(record["horizon"]), ordinal)] = float(value)
        saved = arguments.get("save_to")
        if saved is None:
            return forecasts, {}
        return forecasts, {"base_manifest": configuration_digest(json.loads(manifest_path(Path(saved)).read_text()))}

    def _execute(self, state: Any, sink: Any = None) -> None:
        """The learner's tasks, less those whose issue day has no base forecast (`no_base_forecast`, decided before any fit)."""
        kept = []
        for task in state.tasks:
            day = state.first_day + datetime.timedelta(days=task[2])
            if (task[0], task[1], day.toordinal()) in self._base_forecasts:
                kept.append(task)
            else:
                state.skip(
                    task, SKIP_NO_BASE_FORECAST, f"period {task[0]}, horizon {task[1]}, run date {day}", state.store
                )
        state.tasks = kept
        super()._execute(state, sink)

    def _build_hour_arrays(self, horizon: int) -> Dict[int, Dict]:
        """The learner's period arrays: each row's target less the base's forecast of its period and horizon issued on its date
        (NaN where there is none, so the row doesn't train), and the base forecasts its list names as columns. The targets
        themselves are kept for the records' `actual`."""
        arrays = super()._build_hour_arrays(horizon)
        periods = self._periods_per_day
        horizons = range(1, horizon + 1)
        names = self._check_base_names(pd.DataFrame(), self._freq, horizons)
        days = max((int(a["days"].max()) + 1 for a in arrays.values() if len(a["days"])), default=0)
        table = np.full((periods, horizon, days), np.nan)
        first = self._first_ordinal
        for (period, h, ordinal), value in self._base_forecasts.items():
            day = ordinal - first
            if h <= horizon and 0 <= day < days and period < periods:
                table[period, h - 1, day] = value
        self._actuals, self._actual_days = {}, {}
        for period, a in arrays.items():
            rows = a["days"]
            own = table[period][:, rows].T  # (rows, horizons): the base's forecast of this period from each row's date
            self._actuals[period] = a["y"]
            self._actual_days[period] = rows
            a["y"] = a["y"] - own
            columns = []
            for name in names:
                slot, h = parse_base_name(name, self._freq)
                columns.append(table[period if slot is None else slot, h - 1, rows])
            if columns:
                index = dict(a["col_index"])
                for name in names:
                    index[name] = len(index)
                a["x"] = np.column_stack([a["x"], *columns])
                a["col_index"] = index
        return arrays

    def _worker_inputs(self, state: Any, threads_per_process: int) -> Tuple[Dict, Dict]:
        expanded, config = super()._worker_inputs(state, threads_per_process)
        # The learner's block calls, as the learner itself decides them (`_worker.block_calls`).
        config.pop("block_predict", None)
        if unit_kind(self) == PERIOD and block_calls(self._stage_learner()):
            config["block_predict"] = True
        return expanded, config

    def _collect(self, state: Any, results: List[Dict], skipped: List[Tuple[tuple, str]], sink: Any) -> None:
        """Each forecast as its key's base forecast plus the learner's residual forecast, with its target as `actual` and the
        base forecast as `base_prediction`, after every other key."""
        for record in results:
            period, horizon = record["hour"], record["horizon"]
            ordinal = datetime.date.fromisoformat(record["run_date"]).toordinal()
            base = self._base_forecasts[(period, horizon, ordinal)]
            row = issue_row(self._actual_days[period], ordinal - self._first_ordinal)
            record["prediction"] = record["prediction"] + base
            record["actual"] = float(self._actuals[period][row, horizon - 1])
            record["base_prediction"] = base
        super()._collect(state, results, skipped, sink)

    def _release(self) -> None:
        super()._release()
        self._actuals, self._actual_days = {}, {}

    def __deepcopy__(self, memo: Dict[int, Any]) -> "ResidualModel":
        """A copy without the base's check of the last `_check_inputs`, which holds the run's frame (`verify` copies models)."""
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        for key, value in vars(self).items():
            object.__setattr__(copied, key, None if key == "_base_check" else copy.deepcopy(value, memo))
        return copied

    def __repr__(self) -> str:
        base = repr(self.base) if isinstance(self.base, str) else f"{type(self.base).__name__}({self.base.name!r})"
        return f"ResidualModel(base={base}, learner={type(self.learner).__name__}({self.learner.name!r}), name={self.name!r})"


__all__ = ["ResidualModel", "BASE_PREFIX", "BASE_SUFFIX", "SKIP_NO_BASE_FORECAST", "is_base_name", "parse_base_name"]
