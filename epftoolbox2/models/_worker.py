from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union
import datetime
import hashlib
import math
import os
import traceback

import numpy as np
from threadpoolctl import ThreadpoolController

from ..scalers.registry import make_scaler
from ..scalers.standard import StandardScaler
from ._calendar import FitUnit, member_window, refit_anchor, training_span

_W_HOUR_ARRAYS: Dict = {}
_W_EXPANDED_PREDS: Dict = {}
_W_CONFIG: Dict = {}
_W_FIT_PREDICT = None
_W_MODEL = None  # the instance `_W_FIT_PREDICT` is bound to, whose `_fit_model` and `_predict_fitted` a refit uses

# Native thread pools (BLAS, OpenMP) used by one task. The pool already runs MAX_PROCESSES x THREADS_PER_PROCESS
# tasks at once, so more threads per task only oversubscribe the cores (an OLS run took 4-40x longer that way).
NATIVE_THREADS_PER_TASK = 1
# Read by these libraries when they load, so they cover a library a model imports after the pool started.
NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_W_THREADPOOLS: Optional[ThreadpoolController] = None
_W_NATIVE_LIMITS = None  # held for the worker's lifetime

# Reasons a task produces no forecast without being an error. They are counted per model and reported.
SKIP_TOO_FEW_ROWS = "too_few_rows"
SKIP_NO_FINITE_TRAINING_ROWS = "no_finite_training_rows"
SKIP_MISSING_TEST_PREDICTORS = "missing_test_predictors"
SKIP_FIT_ERROR = "fit_error"
SKIP_NO_MATCHING_DAYS = "no_matching_days"
# The issue day has no row for this period (a gap in the data); without the row there is nothing to forecast from.
SKIP_MISSING_TEST_ROW = "missing_test_row"
# A model that forecasts a published value (`NaiveModel`, `known_only`) whose value its rule puts after the issue time: the
# forecast didn't exist yet. Decided in the parent, before any task runs (`BaseModel._unpublished`).
SKIP_NOT_PUBLISHED = "not_published"

# In `PredictorPlan.day_lists`, next to list ids: a day no matching forecast of that (period, horizon) reads, and a day
# the function returned a list on that no forecast uses, which matches nothing.
NOT_PLANNED = -1
OTHER_LIST = -2


class TaskFailed(RuntimeError):
    """An unexpected exception inside a worker task; carries the task identity and the worker's traceback."""


def _limit_native_threads() -> None:
    """Limit BLAS and OpenMP to NATIVE_THREADS_PER_TASK threads in this worker process.

    Only for a pool worker: the limits are never restored, and the caller's process must keep its own.
    """
    global _W_THREADPOOLS, _W_NATIVE_LIMITS
    for name in NATIVE_THREAD_ENV_VARS:
        os.environ[name] = str(NATIVE_THREADS_PER_TASK)
    # Libraries already loaded ignore the variables: under fork that is everything the parent imported, under
    # spawn whatever importing this module loaded (numpy's BLAS at least). threadpoolctl sets them directly.
    _W_THREADPOOLS = ThreadpoolController()
    _W_NATIVE_LIMITS = _W_THREADPOOLS.limit(limits=NATIVE_THREADS_PER_TASK)


def _limit_openmp_in_task_thread() -> None:
    """The OpenMP limit is per thread (the BLAS ones are per process), so each task thread sets its own."""
    if _W_THREADPOOLS is not None:
        _W_THREADPOOLS.limit(limits=NATIVE_THREADS_PER_TASK, user_api="openmp")


def _pool_init(hour_arrays, expanded_preds, config, model_cls, model_kwargs):
    """Initialiser of the worker pool: limit native threads, then load the data and the model."""
    _limit_native_threads()
    _worker_init(hour_arrays, expanded_preds, config, model_cls, model_kwargs)


def _worker_init(hour_arrays, expanded_preds, config, model_cls, model_kwargs):
    global _W_HOUR_ARRAYS, _W_EXPANDED_PREDS, _W_CONFIG, _W_FIT_PREDICT, _W_MODEL
    _W_HOUR_ARRAYS = hour_arrays
    _W_EXPANDED_PREDS = expanded_preds
    _W_CONFIG = config
    # Built with refit=None whatever the model's: what a refit needs travels in `config["refit"]`.
    _W_MODEL = model_cls(predictors=[], **model_kwargs)
    _W_FIT_PREDICT = _W_MODEL._fit_predict


def training_rows(days: np.ndarray, day: int, training_window: int, hz: int, newest: int = -1) -> slice:
    """The rows the task (`hz`, `day`) may train on, in a period whose rows are on `days`, which are sorted: those whose date is in
    `training_span`, `newest` being the day of the newest target relative to `day`. Rows are chosen by date, since a date can hold
    two of them or none (see `epftoolbox2.models._calendar`), so a doubled autumn row has both occurrences in or out, each with the
    same target."""
    # newest_target="d-1" takes today's path, call for call.
    first, last = (
        training_span(day, training_window, hz) if newest == -1 else training_span(day, training_window, hz, newest)
    )
    return slice(int(np.searchsorted(days, first, side="left")), int(np.searchsorted(days, last, side="right")))


def issue_row(days: np.ndarray, day: int) -> Optional[int]:
    """The row a forecast issued on `day` is made from: the last one of that date, so the second occurrence of a repeated autumn
    hour. None if the date has no row in this period."""
    row = int(np.searchsorted(days, day, side="right")) - 1
    return row if row >= 0 and days[row] == day else None


def training_row_days(days: np.ndarray, day: int, training_window: int, hz: int, newest: int = -1) -> np.ndarray:
    """The day id of each training row of the task (`hz`, `day`) in a period whose rows are on `days`, before any filter."""
    if newest == -1:
        return days[training_rows(days, day, training_window, hz)]
    return days[training_rows(days, day, training_window, hz, newest)]


def matching_rows(row_days: np.ndarray, lists: np.ndarray, start: int, list_id: int) -> np.ndarray:
    """Which of these rows got list `list_id`, where `lists` has an entry per calendar day from day id `start` on. A doubled
    autumn row follows its date. A row the plan didn't evaluate would quietly match nothing, so it raises."""
    row_lists = lists[row_days - start]
    if (row_lists == NOT_PLANNED).any():
        raise RuntimeError(
            "A training row of a train_on='matching' forecast is on a day its plan has no list for: "
            f"{(row_days[row_lists == NOT_PLANNED] - start).tolist()[:3]} day(s) from the start of the plan's days"
        )
    return row_lists == list_id


def matching_digest(row_days: np.ndarray, first_ordinal: int) -> str:
    """A short hash of a set of days, as dates: day ids depend on where the frame starts, the dates don't."""
    days = np.unique(row_days).astype("<i8") + first_ordinal
    return hashlib.sha256(days.tobytes()).hexdigest()[:12]


def _fit_one_numpy(hour: int, hz: int, day_in_test: int) -> Union[Dict, str]:
    """Fit and forecast one task.

    Returns the result record, or a skip reason (one of the SKIP_* names, optionally followed by ": detail")
    when no forecast is possible.

    `day_in_test` counts days from the run's first day. In a run with validation `run_days` says what that day is: its
    window, the days since the start of that window (which the record's `day_in_test` holds) and its place in the plan
    of a predictor function, which has no days between the windows. Each window has its own binary-column masks. Nothing else
    depends on the day's window, so a forecast is what a plain run over its window alone gives.

    A model with a list of training windows (`config["members"]`) fits each member on its own rows, with its own mask and
    scaler, and the forecast is their mean in target units (`_combined`), made only if every member's is: else the task is
    skipped with the first failing member's reason, tagged with its window. Member i's forecast is that of the model with
    `training_window` its window, to the bit. With one window the body runs once, as it always did.
    """
    arrays = _W_HOUR_ARRAYS[hour]
    cfg = _W_CONFIG
    day = cfg["offset"] + day_in_test
    run_days = cfg.get("run_days")
    window, day_in_window, position = (None, day_in_test, day_in_test) if run_days is None else run_days[day_in_test]
    scalable_masks = cfg["scalable_masks"] if window is None else cfg["scalable_masks"][window]

    days = arrays["days"]
    # A model with newest_target="published" has the day of its newest target per issue day, read from the target's rule.
    newest_targets = cfg.get("newest")
    newest = -1 if newest_targets is None else int(newest_targets[hour][position])
    test = issue_row(days, day)
    if test is None:
        return SKIP_MISSING_TEST_ROW

    # A fixed list is keyed by (period, horizon). With a predictor function of `ctx` (see epftoolbox2.models.context)
    # each task has its own list, and each list its own binary-column mask per (period, horizon). The mask of a matching
    # list is that list's too, not one per set of days, while the scaler below is fitted on what the filter leaves.
    task_lists = cfg.get("task_lists")
    if task_lists is None:
        list_id = None
        predictors = _W_EXPANDED_PREDS[(hour, hz)]
        mask_of = (hour, hz)
    else:
        list_id = int(task_lists[(hour, hz)][position])
        predictors = _W_EXPANDED_PREDS[list_id]
        mask_of = (list_id, hour, hz)

    def fit(member, scalable_mask):
        return _fit_member(arrays, cfg, hour, hz, day, member, newest, test, list_id, predictors, scalable_mask)

    members = cfg.get("members")
    if members is None:  # one training window: masks keyed as they always were
        fitted = fit(cfg["training_window"], scalable_masks[mask_of])
        if isinstance(fitted, str):
            return fitted
        prediction, coefs, extra = fitted
    else:
        outcomes = []
        for member in members:
            fitted = fit(member, scalable_masks[(*mask_of, member)])
            if isinstance(fitted, str):
                return _tagged(fitted, member)
            outcomes.append(fitted)
        prediction, coefs, extra = None, [], {}

    run_date = datetime.date.fromisoformat(cfg["test_start"]) + datetime.timedelta(days=day_in_test)
    record = {
        "run_date": run_date.isoformat(),
        "target_date": (run_date + datetime.timedelta(days=hz)).isoformat(),
        "run_weekday": run_date.weekday(),
        "hour": hour,
        "horizon": hz,
        "day_in_test": day_in_window,
        "prediction": prediction,
        "actual": float(arrays["y"][test, hz - 1]),
        "coefficients": coefs,
        "freq": cfg.get("freq", "1h"),
    }
    if newest_targets is not None:
        # The newest target date of the training span, D + min(k, h-1); a row is never dated `day` or later (`training_span`).
        # Not necessarily a target that trained: it can be a slot the clock skips or a missing value, whose row drops.
        record["newest_target_day"] = (run_date + datetime.timedelta(days=min(newest, hz - 1))).isoformat()
    if list_id is not None:
        record["predictor_list"] = cfg["list_hashes"][list_id]
    record.update(extra)
    if window is not None:
        record["window"] = window
    if members is not None:
        return _combined(record, [prediction for prediction, _, _ in outcomes])
    return record


def _fit_member(
    arrays: Dict,
    cfg: Dict,
    hour: int,
    hz: int,
    day: int,
    member,
    newest: int,
    test: int,
    list_id,
    predictors,
    scalable_mask,
) -> Union[Tuple[float, list, Dict], str]:
    """One training window's forecast of the task (`hour`, `hz`, day id `day`) issued from row `test`: (the prediction in target
    units, the coefficients, a matching list's record fields), or a skip reason. Its rows are those of `training_span` with the
    member's window (`member_window`: `"expanding"` from the frame's first date), with the task's newest target."""
    days = arrays["days"]
    span = member_window(member, day)
    # newest_target="d-1" takes today's path, call for call.
    train = training_rows(days, day, span, hz) if newest == -1 else training_rows(days, day, span, hz, newest)

    col_idx = [arrays["col_index"][c] for c in predictors]
    train_x = arrays["x"][train][:, col_idx]
    train_y = arrays["y"][train][:, hz - 1]

    # No row dated inside the span, so nothing to train on for this (hour, horizon, day).
    if train_x.shape[0] == 0:
        return SKIP_TOO_FEW_ROWS

    test_x = arrays["x"][test : test + 1][:, col_idx]

    # train_on="matching" filters what the span holds, so the set can't reach past it. The plan evaluated the function on these
    # days, so nothing about it is computed here.
    matching = cfg.get("matching")
    matching_days = None
    train_days = None
    if matching is not None and list_id in matching["ids"]:
        train_days = days[train]
        same = matching_rows(train_days, matching["lists"][(hour, hz)], matching["start"], list_id)
        train_x, train_y, train_days = train_x[same], train_y[same], train_days[same]
        if train_days.size == 0:
            return SKIP_NO_MATCHING_DAYS
        matching_days = matching_digest(train_days, matching["first_ordinal"])

    # Drop rows with NaN/inf in either predictors or target to avoid model fit failures.
    finite_rows = np.isfinite(train_y)
    if train_x.size:
        finite_rows &= np.isfinite(train_x).all(axis=1)
    train_x = train_x[finite_rows]
    train_y = train_y[finite_rows]

    if train_x.shape[0] == 0:
        return SKIP_NO_FINITE_TRAINING_ROWS

    if not np.isfinite(test_x).all():
        return SKIP_MISSING_TEST_PREDICTORS

    scaler = make_scaler(cfg.get("scaler", "standard"))
    train_x, train_y, test_x = scaler.fit_transform(
        train_x.copy(),
        train_y.copy(),
        test_x.copy(),
        scalable_mask=scalable_mask,
        columns=predictors,
    )
    try:
        pred, coefs = _W_FIT_PREDICT(train_x, train_y, test_x)
    except ValueError as exc:
        # scikit-learn raises ValueError on degenerate input (e.g. fewer rows than CV folds); counted, not fatal.
        return f"{SKIP_FIT_ERROR}: {exc}"

    extra = {}
    if matching_days is not None:
        extra["training_days"] = int(np.unique(train_days[finite_rows]).size)
        extra["matching_days"] = matching_days
    return scaler.inverse(float(pred)), coefs, extra


def _tagged(reason: str, member) -> str:
    """A member's skip reason, with its window after the reason (`"no_finite_training_rows: window 56"`), before any detail. A
    missing issue row is the key's, whatever the window, and stays as it is."""
    if reason == SKIP_MISSING_TEST_ROW:
        return reason
    name, sep, detail = reason.partition(": ")
    return f"{name}: window {member}" + (f": {detail}" if sep else "")


def _combined(record: Dict, predictions: List[float]) -> Dict:
    """The record of a model with a list of training windows, from its first member's: the prediction is the members' mean in
    target units, `math.fsum` divided by their number (exactly rounded, so their order can't change a bit), `member_predictions`
    holds theirs in member order, and the coefficients are empty (decision 11 of the step 4 spec)."""
    out = {}
    for key, value in record.items():
        if key == "prediction":
            out["prediction"] = math.fsum(predictions) / len(predictions)
            out["member_predictions"] = [float(prediction) for prediction in predictions]
        elif key == "coefficients":
            out["coefficients"] = []
        else:
            out[key] = value
    return out


def _row_value(hour: int, hz: int, day_in_test: int) -> Union[Dict, str]:
    """Forecast one task of a model that fits nothing (`NaiveModel`): its one predictor's value on the row the forecast is
    issued from, the last row dated on the issue day, as it stands. No training row, scaler or mask is read, so a task needs
    no history. Returns the record, with no coefficients, or a skip reason, as `_fit_one_numpy` does."""
    arrays = _W_HOUR_ARRAYS[hour]
    cfg = _W_CONFIG
    day = cfg["offset"] + day_in_test
    run_days = cfg.get("run_days")
    window, day_in_window, position = (None, day_in_test, day_in_test) if run_days is None else run_days[day_in_test]
    test = issue_row(arrays["days"], day)
    if test is None:
        return SKIP_MISSING_TEST_ROW
    task_lists = cfg.get("task_lists")
    list_id = None if task_lists is None else int(task_lists[(hour, hz)][position])
    (name,) = _W_EXPANDED_PREDS[(hour, hz) if list_id is None else list_id]
    value = float(arrays["x"][test, arrays["col_index"][name]])
    if not np.isfinite(value):
        return SKIP_MISSING_TEST_PREDICTORS

    run_date = datetime.date.fromisoformat(cfg["test_start"]) + datetime.timedelta(days=day_in_test)
    record = {
        "run_date": run_date.isoformat(),
        "target_date": (run_date + datetime.timedelta(days=hz)).isoformat(),
        "run_weekday": run_date.weekday(),
        "hour": hour,
        "horizon": hz,
        "day_in_test": day_in_window,
        "prediction": value,
        "actual": float(arrays["y"][test, hz - 1]),
        "coefficients": [],
        "freq": cfg.get("freq", "1h"),
    }
    if list_id is not None:
        record["predictor_list"] = cfg["list_hashes"][list_id]
    if window is not None:
        record["window"] = window
    return record


def _run_task(task: tuple) -> Union[Dict, str]:
    try:
        # A model that fits nothing has `config["fits"]` False and forecasts from the issue row alone.
        return _fit_one_numpy(*task) if _W_CONFIG.get("fits", True) else _row_value(*task)
    except Exception as exc:
        hour, hz, day_in_test = task
        raise TaskFailed(
            f"task (period={hour}, horizon={hz}, day_in_test={day_in_test}) raised {exc!r}\n{traceback.format_exc()}"
        ) from None


def _run_day(tasks: list) -> Tuple[List[Dict], List[Tuple[tuple, str]]]:
    """Run one test day's tasks. Returns (results, [(task, skip reason)]); unexpected errors raise TaskFailed."""
    threads = _W_CONFIG["threads_per_process"]
    results, skipped = [], []

    def collect(task, outcome):
        if isinstance(outcome, str):
            skipped.append((task, outcome))
        else:
            results.append(outcome)

    if len(tasks) <= 1 or threads == 1:
        for t in tasks:
            collect(t, _run_task(t))
        return results, skipped

    with ThreadPoolExecutor(max_workers=threads, initializer=_limit_openmp_in_task_thread) as pool:
        futures = {pool.submit(_run_task, t): t for t in tasks}
        for future in as_completed(futures):
            collect(futures[future], future.result())
    return results, skipped


def _place_of(day_in_test: int) -> Tuple[Optional[str], int, int]:
    """(window, days since its start, place among the issue days) of a task's day, as `_fit_one_numpy` reads it."""
    run_days = _W_CONFIG.get("run_days")
    return (None, day_in_test, day_in_test) if run_days is None else run_days[day_in_test]


def _unit_of(task: tuple, member: int = 0) -> FitUnit:
    """The fit a task of a model with `refit` forecasts from: its period and horizon, the anchor of its issue day, its list, and
    the index of its training window among the model's (0 with one window)."""
    hour, hz, day_in_test = task
    cfg = _W_CONFIG
    task_lists = cfg.get("task_lists")
    list_id = None if task_lists is None else int(task_lists[(hour, hz)][_place_of(day_in_test)[2]])
    run_date = datetime.date.fromisoformat(cfg["test_start"]) + datetime.timedelta(days=day_in_test)
    return FitUnit(hour, hz, refit_anchor(run_date, cfg["refit"]["rule"]), list_id, member)


def _units_of(task: tuple) -> List[FitUnit]:
    """The units a task forecasts from: one, or with a list of training windows one per member, in member order."""
    members = _W_CONFIG.get("members")
    return [_unit_of(task)] if members is None else [_unit_of(task, member) for member in range(len(members))]


def _fit_unit(unit: FitUnit, tasks: List[tuple]) -> List[Tuple[tuple, Union[Dict, str]]]:
    """Fit one unit of a model with `refit` and forecast each of `tasks` from it, one row at a time. Returns (task, record or
    skip reason) of each.

    The rows are those of a forecast issued on the anchor: `training_span(anchor, W, h, newest)` with the anchor's newest target,
    filtered as `_fit_one_numpy` filters a task's. Which columns are binary is read from these rows, before any filter, so a
    forecast's scaling depends on its anchor alone, never on where the run starts (the mask of a forecast with `refit=None` is
    read on the window's first day). The scaler is fitted on the training rows and transforms the test rows of `tasks` together,
    value by value, so a forecast is the same whichever other forecasts of its unit are run with it: one cut short and resumed
    equals a fresh one.
    """
    cfg = _W_CONFIG
    refit = cfg["refit"]
    hour, hz = unit.period, unit.horizon
    arrays = _W_HOUR_ARRAYS[hour]
    days = arrays["days"]
    anchor = unit.anchor.toordinal() - refit["first_ordinal"]
    members = cfg.get("members")
    # The unit's training window: the model's, or with a list of them its member's (`"expanding"` from the frame's first date).
    span = member_window(cfg["training_window"] if members is None else members[unit.member], anchor)
    newest_targets = refit["newest"]
    if newest_targets is None:
        newest = -1
        train = training_rows(days, anchor, span, hz)
    else:
        # Read on the anchor: a target published between the anchor and the issue day doesn't reach the fit.
        newest = int(newest_targets.at(hour, unit.anchor.toordinal()))
        train = training_rows(days, anchor, span, hz, newest)

    predictors = _W_EXPANDED_PREDS[(hour, hz) if unit.list_id is None else unit.list_id]
    col_idx = [arrays["col_index"][c] for c in predictors]
    train_x = arrays["x"][train][:, col_idx]
    train_y = arrays["y"][train][:, hz - 1]
    scalable_mask = StandardScaler.get_scalable_mask(train_x)

    outcomes: List[Tuple[tuple, Union[Dict, str]]] = []
    issued = []
    for task in tasks:
        test = issue_row(days, cfg["offset"] + task[2])
        if test is None:
            outcomes.append((task, SKIP_MISSING_TEST_ROW))
        else:
            issued.append((task, test))

    def skip_all(reason: str) -> List[Tuple[tuple, Union[Dict, str]]]:
        return outcomes + [(task, reason) for task, _ in issued]

    if train_x.shape[0] == 0:
        return skip_all(SKIP_TOO_FEW_ROWS)

    matching = cfg.get("matching")
    matching_days = None
    train_days = None
    if matching is not None and unit.list_id in matching["ids"]:
        train_days = days[train]
        same = matching_rows(train_days, matching["lists"][(hour, hz)], matching["start"], unit.list_id)
        train_x, train_y, train_days = train_x[same], train_y[same], train_days[same]
        if train_days.size == 0:
            return skip_all(SKIP_NO_MATCHING_DAYS)
        matching_days = matching_digest(train_days, matching["first_ordinal"])

    finite_rows = np.isfinite(train_y)
    if train_x.size:
        finite_rows &= np.isfinite(train_x).all(axis=1)
    train_x = train_x[finite_rows]
    train_y = train_y[finite_rows]
    if train_x.shape[0] == 0:
        return skip_all(SKIP_NO_FINITE_TRAINING_ROWS)

    forecast = []
    for task, test in issued:
        if np.isfinite(arrays["x"][test, col_idx]).all():
            forecast.append((task, test))
        else:
            outcomes.append((task, SKIP_MISSING_TEST_PREDICTORS))
    if not forecast:
        return outcomes

    scaler = make_scaler(cfg.get("scaler", "standard"))
    train_x, train_y, test_x = scaler.fit_transform(
        train_x.copy(),
        train_y.copy(),
        arrays["x"][[test for _, test in forecast]][:, col_idx],
        scalable_mask=scalable_mask,
        columns=predictors,
    )
    try:
        fitted = _W_MODEL._fit_model(train_x, train_y)
    except ValueError as exc:
        return outcomes + [(task, f"{SKIP_FIT_ERROR}: {exc}") for task, _ in forecast]

    for row, (task, test) in enumerate(forecast):
        # One row at a time (decision 16 of the step 3 spec): a product over several rows can differ in the last bits.
        pred, coefs = _W_MODEL._predict_fitted(fitted, test_x[row : row + 1])
        window, day_in_window, _ = _place_of(task[2])
        run_date = datetime.date.fromisoformat(cfg["test_start"]) + datetime.timedelta(days=task[2])
        record = {
            "run_date": run_date.isoformat(),
            "target_date": (run_date + datetime.timedelta(days=hz)).isoformat(),
            "run_weekday": run_date.weekday(),
            "hour": hour,
            "horizon": hz,
            "day_in_test": day_in_window,
            "prediction": scaler.inverse(float(pred)),
            "actual": float(arrays["y"][test, hz - 1]),
            "coefficients": coefs,
            "freq": cfg.get("freq", "1h"),
            "refit_anchor": unit.anchor.isoformat(),
        }
        if newest_targets is not None:
            # The newest target date of the unit's span, anchor + min(k, h-1), as `_fit_one_numpy` writes it for its day.
            record["newest_target_day"] = (unit.anchor + datetime.timedelta(days=min(newest, hz - 1))).isoformat()
        if unit.list_id is not None:
            record["predictor_list"] = cfg["list_hashes"][unit.list_id]
        if matching_days is not None:
            record["training_days"] = int(np.unique(train_days[finite_rows]).size)
            record["matching_days"] = matching_days
        if window is not None:
            record["window"] = window
        outcomes.append((task, record))
    return outcomes


def _run_unit(unit: FitUnit, tasks: List[tuple]) -> List[Tuple[tuple, Union[Dict, str]]]:
    try:
        return [(task, outcome, unit.member) for task, outcome in _fit_unit(unit, tasks)]
    except Exception as exc:
        raise TaskFailed(
            f"fit unit (period={unit.period}, horizon={unit.horizon}, anchor={unit.anchor}, list={unit.list_id}, "
            f"member={unit.member}) of tasks {[task[2] for task in tasks]} raised {exc!r}\n{traceback.format_exc()}"
        ) from None


def _run_block(tasks: list) -> Tuple[List[Dict], List[Tuple[tuple, str]]]:
    """Run the tasks of one refit block, those whose issue days share an anchor: each unit's fit once, then its forecasts.
    Returns what `_run_day` does. Nothing carries over between units: each fits from scratch on its own rows.

    With a list of training windows a task has a unit per member, all in its block, and its record is made after every one of
    them ran: the members' mean if each forecast it, else the first failing member's reason (in member order), tagged with its
    window, as `_fit_one_numpy` does."""
    units: Dict[FitUnit, List[tuple]] = {}
    for task in tasks:
        for unit in _units_of(task):
            units.setdefault(unit, []).append(task)
    threads = _W_CONFIG["threads_per_process"]
    members = _W_CONFIG.get("members")
    by_task: Dict[tuple, Dict[int, Union[Dict, str]]] = {}
    results, skipped = [], []

    def collect(outcomes):
        for task, outcome, member in outcomes:
            if members is not None:
                by_task.setdefault(task, {})[member] = outcome
            elif isinstance(outcome, str):
                skipped.append((task, outcome))
            else:
                results.append(outcome)

    if len(units) <= 1 or threads == 1:
        for unit, unit_tasks in units.items():
            collect(_run_unit(unit, unit_tasks))
    else:
        with ThreadPoolExecutor(max_workers=threads, initializer=_limit_openmp_in_task_thread) as pool:
            futures = [pool.submit(_run_unit, unit, unit_tasks) for unit, unit_tasks in units.items()]
            for future in as_completed(futures):
                collect(future.result())
    for task, outcomes in by_task.items():
        ordered = [outcomes[member] for member in range(len(members))]
        failed = next(((member, outcome) for member, outcome in enumerate(ordered) if isinstance(outcome, str)), None)
        if failed is not None:
            skipped.append((task, _tagged(failed[1], members[failed[0]])))
        else:
            results.append(_combined(ordered[0], [record["prediction"] for record in ordered]))
    return results, skipped
