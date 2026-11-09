"""Per-row targets (step 7c): a model's `normalise_by`, `bounds` and `zero_when`, one seam in the worker.

- `normalise_by="<column>"`: each training row's target is divided by that row's value of the column, and the forecast is
  multiplied by the issue row's value. A training row whose value is missing or not positive drops out, as a row with a missing
  input does; on the issue row a missing value is a `missing_test_predictors` skip and one not positive a `non_positive_scale`
  skip.
- `bounds=(low, high)`: the forecast is clipped after it is scaled back, in multiples of the issue row's scale with
  `normalise_by` (a capacity factor's `(0, 1.05)`), in target units without. Either side may be None.
- `zero_when="<column>"`: where the issue row's value is 1 the forecast is 0, with no fit (`coefficients: []`, `detail:
  "zero_when"`, status ok); a missing value is a `missing_test_predictors` skip, and the run refuses any other value on an issue
  row before any fit (`check_zero_values`). It is decided after the skips of the training rows and of the issue row's inputs, and
  before the issue row's scale and the fit, so a forecast of 0 is made where the scale is missing or not positive (a clear-sky
  night) and where the fit would fail (`fit_error`): the two places where a model's keys differ from the model's without it.
  Training rows are untouched, and the column is judged by the information set on the issue row alone.

`{horizon}` is the only placeholder in either name, as in a predictor. So the learner sees (y - o) / s on its training rows and the
forecast is o + s * y_hat, clipped, with s and o the issue row's; o is 0 here (a model with an offset of its own, 7f's
`ShapeModel`, uses the same seam). y_hat is the learner's forecast mapped back through the model's scaler, which is fitted on the
normalised targets. Each training window of a list is scaled back and clipped on its own, so a member's forecast is that of the
model with its window alone, and the forecast is their mean.

The columns are inputs the model reads (`BaseModel._used_predictors`: loaded and hashed in the manifest), never regressors.
`normalise_by`'s is judged by the information set on the issue row and, for a rule that changes over time, on the training rows;
`zero_when`'s on the issue row alone, which is all it reads. `verify` perturbs the scale as a predictor and flips the unknown
cells of `zero_when`'s column (0 to 1 and back), runs its copies without `bounds` (the clip follows the unclipped forecast), and
doesn't ask a forecast of 0 to move under its control. A model with any of the three set is not engine-capable: its targets
differ per row from its group's. Periods and pooled units take them; day units, models that read history and models that fit
nothing (bar `bounds`) refuse them (`BaseModel._check_row_target_support`).
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# The `detail` of a forecast `zero_when` made without a fit (status ok).
ZERO_WHEN = "zero_when"
# The parameters, in the order a model takes them.
PARAMETERS = ("normalise_by", "bounds", "zero_when")
# The columns' parameters, and their keys in a worker's `config["row_target"]`.
_COLUMNS = {"normalise_by": "scale", "zero_when": "zero"}
# What `columns_at`, `column_lists` and `with_columns` read: both columns, or one of them.
BOTH, SCALE, ZERO = ("normalise_by", "zero_when"), ("normalise_by",), ("zero_when",)


def check_column(key: str, value: Any) -> Optional[str]:
    """`value` for `normalise_by` or `zero_when` (`key`): None, or a column name, with `{horizon}` where it depends on the
    horizon. Anything else is a ValueError (a TypeError for something that isn't a string)."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(
            f"{key} is a column name (with {{horizon}} where it depends on the horizon), or None, got {type(value).__name__}: "
            f"{value!r}"
        )
    if not value.strip() or value != value.strip():
        raise ValueError(f"{key} names a column, without spaces around it, got {value!r}")
    if "{" in value.replace("{horizon}", "") or "}" in value.replace("{horizon}", ""):
        raise ValueError(f"{key}={value!r}: {{horizon}} is the only placeholder a column name may hold")
    return value


def check_bounds(value: Any) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """`bounds` as kept: None, or (low, high) as floats, either side None for no bound, low <= high. `(None, None)` is None,
    so it is written nowhere."""
    if value is None:
        return None
    if isinstance(value, (str, bytes, dict)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(
            f"bounds is (low, high), either side a number or None, such as (0, 1.05) with normalise_by or (0, None), got "
            f"{value!r}"
        )
    sides = []
    for side, bound in zip(("low", "high"), value):
        if bound is None:
            sides.append(None)
            continue
        if isinstance(bound, (bool, np.bool_)) or not isinstance(bound, (int, float, np.integer, np.floating)):
            raise ValueError(f"bounds: {side} must be a number or None, got {bound!r}")
        if not math.isfinite(float(bound)):
            raise ValueError(f"bounds: {side} must be finite (None for no bound), got {bound!r}")
        sides.append(float(bound))
    low, high = sides
    if low is not None and high is not None and low > high:
        raise ValueError(f"bounds: low {low} is above high {high}")
    if low is None and high is None:
        return None
    return low, high


def expand(name: str, horizon: int) -> str:
    """A column name at `horizon`: `{horizon}` replaced, as a predictor's is (`BaseModel._expand_predictors`)."""
    return name.replace("{horizon}", str(horizon))


def is_set(model: Any) -> bool:
    """Whether any of `normalise_by`, `bounds` and `zero_when` is set on `model`."""
    return any(getattr(model, key, None) is not None for key in PARAMETERS)


def columns_at(model: Any, horizon: int, keys: Sequence[str] = BOTH) -> List[str]:
    """The columns `model`'s per-row target reads at `horizon` for the parameters `keys` (both: `normalise_by`'s, then
    `zero_when`'s), each once."""
    names = [getattr(model, key, None) for key in keys]
    return list(dict.fromkeys(expand(name, horizon) for name in names if name is not None))


def column_lists(
    model: Any, periods: int, horizons: Iterable[int], keys: Sequence[str] = BOTH
) -> Dict[Tuple[int, int], List[str]]:
    """(period, horizon) -> the columns the per-row target reads there for the parameters `keys`, for a check that reads lists
    (`_contract.check_predictors`, `InformationSet.check`, `late_uses`); empty without such a column."""
    horizons = list(horizons)
    if not any(getattr(model, key, None) is not None for key in keys):
        return {}
    return {(period, hz): columns_at(model, hz, keys) for period in range(periods) for hz in horizons}


def with_columns(
    model: Any, lists: Dict[Tuple[int, int], List[str]], keys: Sequence[str] = BOTH
) -> Dict[Tuple[int, int], List[str]]:
    """`lists` with the per-row target's columns of the parameters `keys` of each (period, horizon) after the list's own names,
    each once; `lists` itself (the same object) without such a column."""
    if not any(getattr(model, key, None) is not None for key in keys):
        return lists
    return {key: list(dict.fromkeys([*names, *columns_at(model, key[1], keys)])) for key, names in lists.items()}


def zero_columns(model: Any, horizons: Iterable[int]) -> List[str]:
    """`zero_when`'s column at each of `horizons`, each once; empty without it."""
    return list(dict.fromkeys(name for hz in horizons for name in columns_at(model, hz, ZERO)))


def check_zero_values(name: str, data: Any, columns: Iterable[str], issue_days: Optional[Sequence[Any]]) -> None:
    """A ValueError if a `zero_when` column of `columns` holds a finite value other than 0 or 1 on a row of the frame `data`
    dated on one of `issue_days` (local dates, ISO text or dates), the rows forecasts are issued from; nothing without
    `issue_days` (a check with no test period: its runs check theirs). A missing value stays allowed: it skips its forecast as
    `missing_test_predictors`."""
    columns = list(columns)
    if issue_days is None or not columns:
        return
    index = data.index
    wall = index.tz_localize(None) if getattr(index, "tz", None) is not None else index
    days = np.array([np.datetime64(str(day)[:10], "D") for day in issue_days])
    on = np.isin(np.asarray(wall.normalize(), dtype="datetime64[D]"), days)
    for column in columns:
        values = data[column].to_numpy(dtype=np.float64)[on]
        odd = np.isfinite(values) & (values != 0.0) & (values != 1.0)
        if odd.any():
            shown = ", ".join(f"{value:g}" for value in np.unique(values[odd])[:3])
            raise ValueError(
                f"{name}: zero_when={column!r} holds {shown} on an issue row; it must be 0 (forecast as usual), 1 (a forecast "
                "of 0, with no fit) or missing (no forecast). Turn the column into a 0/1 flag first"
            )


def manifest_params(model: Any) -> Dict[str, Any]:
    """What a manifest's `model_params` records of the per-row target, compared on resume: the parameters that are set, bounds as
    a list. Empty without one, so a manifest doesn't change."""
    out: Dict[str, Any] = {}
    for key in PARAMETERS:
        value = getattr(model, key, None)
        if value is not None:
            out[key] = list(value) if key == "bounds" else value
    return out


def config_of(model: Any, horizons: Iterable[int]) -> Optional[Dict[str, Any]]:
    """What a worker reads (`config["row_target"]`): `scale` and `zero`, horizon -> the column's name (None for a parameter not
    set), and `bounds`, (low, high) or None. None without any of them, so the config doesn't change."""
    if not is_set(model):
        return None
    horizons = list(horizons)
    out: Dict[str, Any] = {"bounds": getattr(model, "bounds", None)}
    for key, short in _COLUMNS.items():
        name = getattr(model, key, None)
        out[short] = None if name is None else {hz: expand(name, hz) for hz in horizons}
    return out


# ---------------------------------------------------------------------------------------------------------------------------
# In the worker
# ---------------------------------------------------------------------------------------------------------------------------


def scale_column(arrays: Dict, row_target: Optional[Dict], hz: int) -> Optional[int]:
    """The place of `normalise_by`'s column at horizon `hz` in a period's arrays, None without it."""
    if row_target is None or row_target["scale"] is None:
        return None
    return arrays["col_index"][row_target["scale"][hz]]


def usable_scale(values: Any) -> Any:
    """Which scale values a row may be divided by: finite and positive (a bool, or an array of them)."""
    return np.isfinite(values) & (values > 0)


def zero_outcome(arrays: Dict, row_target: Optional[Dict], hz: int, row: int) -> Optional[bool]:
    """What `zero_when` says on the issue row `row` at horizon `hz`: True for a forecast of 0 (the value is 1), False for a
    forecast as usual (any other finite value, or no `zero_when`), None for a missing value (a `missing_test_predictors` skip)."""
    if row_target is None or row_target["zero"] is None:
        return False
    value = arrays["x"][row, arrays["col_index"][row_target["zero"][hz]]]
    if not np.isfinite(value):
        return None
    return bool(value == 1)


def scaled_back(row_target: Dict, prediction: float, scale: Optional[float]) -> float:
    """The forecast from the learner's, in target units of the scaler (`prediction`, y_hat): s * y_hat with `normalise_by` (`scale`,
    the issue row's), then clipped to `bounds`, multiples of s with `normalise_by` and target units without."""
    value = float(prediction)
    if scale is not None:
        value = float(scale) * value
    bounds = row_target["bounds"]
    if bounds is not None:
        low, high = bounds
        if scale is not None:
            low = None if low is None else low * float(scale)
            high = None if high is None else high * float(scale)
        if low is not None and value < low:
            value = low
        if high is not None and value > high:
            value = high
    return value
