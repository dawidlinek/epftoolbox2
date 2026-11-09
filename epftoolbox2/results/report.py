from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..evaluators import stats
from ..evaluators._compare import (
    MIN_VERDICT_DAYS,
    SHORT_TEST_DAYS,
    LossPanel,
    ModelFrames,
    PairComparison,
    SetComparison,
    adjust_family,
    compare_pair,
    compare_set,
    confidence_set,
    short_test_message,
    too_few_distinct_models,
    verdict,
)
from ..evaluators._days import check_period_ids, periods_per_hour
from ..evaluators.base import FORECAST_KEYS, Evaluator, SetEvaluator
from ..evaluators.dm import DMEvaluator, check_test_settings
from ..evaluators.economic import EconomicEvaluator
from ..evaluators.mcs import check_mcs_settings
from ..logging import get_logger
from ..models._calendar import clock_periods_on
from ..models._cards import provenance_frame, window_starts
from .availability import listing_frame
from .breakdowns import (
    AGGREGATES,
    BLOCK_AGGREGATES,
    PEAK_HOURS,
    block_aggregate,
    check_peak_hours,
    check_source_zone,
    numbered_by_the_clock,
    daily_aggregate,
    hourly_aggregate,
    price_level,
    source_table,
    source_values,
    spike_threshold,
    switch_dates,
)
from .breakdowns import error_split as split_errors
from .matching import rows_of_refs, training_day_frame
from .ref import ModelResultRef
from .store import fingerprint_path
from .vintages import flag_mixed_classes, vintage_frame
from .windows import TEST, VALIDATION, RunWindows

logger = get_logger(__name__)

# (name in compute_all(), grouping columns). `year` is the year of target_date; `run_weekday` and `target_weekday`
# are the weekday of run_date / target_date.
GROUP_CONFIGS: List[Tuple[str, List[str]]] = [
    ("hour", ["hour"]),
    ("horizon", ["horizon"]),
    ("hour_horizon", ["hour", "horizon"]),
    ("year", ["year"]),
    ("year_horizon", ["year", "horizon"]),
    ("run_weekday_horizon", ["run_weekday", "horizon"]),
    ("target_weekday_horizon", ["target_weekday", "horizon"]),
]

# The order in which a model's records are evaluated and listed: the forecast key, then the values, so that two
# records with the same key (or without one) still fall in the same place. Workers write records as tasks finish,
# which varies with the number of processes, scheduling and resuming, and float sums depend on the order of their
# terms: a metric over the rows in file order can differ in the last bit between two runs of the same study.
RECORD_ORDER = ("run_date", "target_date", "horizon", "hour", "prediction", "actual", "day_in_test")

# The grouping columns of the breakdowns that are not in `compute_all()`: `by_quarter()` (with `year`), `by_day()` and
# `by_price_level()`.
_BREAKDOWN_KEYS = ("quarter", "target_date", "price_level")

# The columns of `summary()` and of the grouped tables that no evaluator wrote.
_TABLE_KEYS = {"model", "forecasts", *(key for _, keys in GROUP_CONFIGS for key in keys), *_BREAKDOWN_KEYS}

# The frequencies of a report made by `aggregate()`, and the periods an hour holds at each frequency of results.
DAILY = "1D"
HOURLY = "1h"
QUARTER_HOURLY = "15min"
_PERIODS_PER_HOUR = {"1h": 1, "15min": 4}

# What `aggregate()` writes in `skipped` for the periods it drops, by the frequency it builds.
_INCOMPLETE = {DAILY: "incomplete_day", HOURLY: "incomplete_hour"}

# The columns of `EvaluationReport.error_split()`.
ERROR_SPLIT_COLUMNS = ["model", "days", "forecasts", "mse", "level", "hour", "quarter"]

# Two-sided size of the `verdict` of `tests()`.
_TEST_ALPHA = 0.05

# The columns of `EvaluationReport.choices()`.
CHOICE_COLUMNS = [
    "ensemble",
    "segment",
    "chosen",
    "k",
    "validation_mae",
    "validation_forecasts",
    "validation_share",
    "validation_start",
    "validation_end",
    "test_start",
    "test_end",
]


def metric_columns(table: pd.DataFrame) -> List[str]:
    """The columns of a table of `summary()`, `compute_all()` or `by_*()` that evaluators wrote, in table order."""
    keys = _TABLE_KEYS | {table.attrs["by_column"]} if "by_column" in table.attrs else _TABLE_KEYS
    return [c for c in table.columns if c not in keys]


def _in_record_order(df: pd.DataFrame, first: Tuple[str, ...] = ()) -> pd.DataFrame:
    """`df` sorted by `RECORD_ORDER` (after the columns in `first`) with a fresh index; columns it lacks are skipped."""
    by = [c for c in (*first, *RECORD_ORDER) if c in df.columns]
    return df.sort_values(by, kind="stable").reset_index(drop=True) if by else df


class EvaluationReport:
    """Tables and comparisons over the forecasts of one window of a run (see `epftoolbox2.results.windows`).

    Every method reads through `ModelResultRef.iter_records`, so nothing of another window is reachable from a report. The
    report of a run is the test window, and `report.validation` the validation report. Without a test window the run
    returns the validation report itself. `report.window` says which one it is.
    """

    def __init__(
        self,
        results_or_refs: Dict[str, Union[ModelResultRef, List[Dict]]],
        evaluators: List[Evaluator],
        source_data: Optional[pd.DataFrame] = None,
        validation: Optional["EvaluationReport"] = None,
        choices: Optional[Mapping[str, List[Dict[str, Any]]]] = None,
    ):
        self.evaluators = evaluators
        self.source_data = source_data
        self.refs: Dict[str, ModelResultRef] = {}
        self._read_frames: Dict[str, Tuple[tuple, pd.DataFrame]] = {}
        self._validation = validation
        self._choices = dict(choices or {})

        validation_refs: Dict[str, ModelResultRef] = {}
        for name, v in results_or_refs.items():
            if isinstance(v, ModelResultRef):
                if v.by_window:  # the ref of a whole run with windows
                    if VALIDATION in v.by_window:
                        validation_refs[name] = v.by_window[VALIDATION]
                    v = v.by_window.get(TEST, v.by_window.get(VALIDATION))
                self.refs[name] = v
            else:
                self.refs[name] = ModelResultRef(
                    name=name,
                    path=None,
                    count=len(v),
                    test_start="",
                    test_end="",
                    horizon=0,
                    _results=v,
                )
        held = sorted({ref.window or TEST for ref in self.refs.values()})
        if len(held) > 1:
            shown = {name: ref.window or TEST for name, ref in self.refs.items()}
            raise ValueError(
                f"These results are on different windows ({shown}): one report would compare models on different forecasts "
                "whose roles differ. Build one report per window, from the runs with the same validation= and test=."
            )
        expected = sum(not ref.test_only for ref in self.refs.values())
        if validation is None and self.window == TEST and validation_refs and len(validation_refs) == expected:
            self._validation = EvaluationReport(
                validation_refs, evaluators, source_data=source_data, choices=self._choices
            )

    @property
    def window(self) -> str:
        """Which window this report is on: `"validation"` or `"test"`. A plain run has only the test window."""
        return next((ref.window for ref in self.refs.values() if ref.window is not None), TEST)

    @property
    def validation(self) -> "EvaluationReport":
        """The report on the validation window of this run: only forecasts whose target day is before the first test issue
        day, since a user picks from it by hand and a later target would be look-ahead. `forecast_counts()` counts the rest
        as `target_in_test`. A `ValueError` on a run with no validation."""
        if self.window == VALIDATION:
            return self
        if self._validation is None:
            raise ValueError(
                "This report has no validation window: run with validation=(start, end) or validation='previous_year' "
                "(aggregate() and join() keep it only where they can: call them on report.validation)."
            )
        return self._validation

    @property
    def windows(self) -> Optional[RunWindows]:
        """The windows and segments of the run behind this report, or None for a plain run."""
        return next((ref.windows for ref in self.refs.values() if ref.windows is not None), None)

    def summary(self) -> pd.DataFrame:
        """One row per model: `model, forecasts`, then the metrics. `forecasts` is the model's own number of forecasts."""
        return self._summary_table(self._model_frames())

    def by_hour(self) -> pd.DataFrame:
        """One row per (group, model): `model`, the grouping columns, `forecasts`, then the metrics.

        This is the shape of every `by_*()` table (`by_hour`, `by_horizon`, `by_hour_horizon`, `by_year`,
        `by_year_horizon`, `by_run_weekday_horizon`, `by_target_weekday_horizon`, `by_quarter`, `by_day`,
        `by_price_level`). `forecasts` is the model's own number of forecasts in that group, as in `summary()`. A model
        with none in a group has no row there.

        The metrics of a group are on the forecasts each model has in it, and models can have different ones (a model
        that skips keys, or a `NaiveModel(known_only=True)` whose source isn't published yet). When the models of a
        group have different counts, the table warns, and `df.attrs["unequal_counts"]` lists those groups (a model
        absent from a group counts 0). The relative metrics and tests are on the keys a pair shares; MAE, RMSE and the
        like are not. Equal counts do not prove equal keys: `forecast_counts()` and the `n_forecasts` columns of the
        comparisons say more.
        """
        return self._grouped_table(self._model_frames(), ["hour"])

    def by_horizon(self) -> pd.DataFrame:
        return self._grouped_table(self._model_frames(), ["horizon"])

    def by_hour_horizon(self) -> pd.DataFrame:
        return self._grouped_table(self._model_frames(), ["hour", "horizon"])

    def by_year(self) -> pd.DataFrame:
        return self._grouped_table(self._model_frames(), ["year"])

    def by_year_horizon(self) -> pd.DataFrame:
        return self._grouped_table(self._model_frames(), ["year", "horizon"])

    def by_run_weekday_horizon(self) -> pd.DataFrame:
        return self._grouped_table(self._model_frames(), ["run_weekday", "horizon"])

    def by_target_weekday_horizon(self) -> pd.DataFrame:
        return self._grouped_table(self._model_frames(), ["target_weekday", "horizon"])

    def by_quarter(self) -> pd.DataFrame:
        """One row per (calendar quarter of the target date, model): `model, year, quarter, forecasts`, then the metrics.

        `forecasts` is the model's own count in that quarter. Every evaluator runs on the quarter alone, so relative
        metrics and tests compare the models on the forecasts they share inside it. A quarter is short by nature: the
        tests print `n_days` and say nothing below 30 days.
        """
        return self._grouped_table(self._model_frames(), ["year", "quarter"])

    def by_day(self) -> pd.DataFrame:
        """One row per (target date, model): `model, target_date, forecasts`, then the metrics.

        All horizons that forecast the date are in its group. A day holds one run date per horizon, so the tests have
        almost nothing to run on and their columns stay empty (`n_days` says how many days they had).
        """
        return self._grouped_table(self._model_frames(), ["target_date"])

    def by_price_level(self, spike: Optional[float] = None) -> pd.DataFrame:
        """One row per (price level, model): `model, price_level, forecasts`, then the metrics.

        The level of a forecast comes from its actual price alone, never from a prediction: `negative` (actual below 0),
        `spike` (actual at or above the threshold) and `normal` (the rest). A forecast is in exactly one group; one
        without a finite actual is in none. The threshold is `min(300, 99th percentile)` of the actual prices of the
        periods any model forecast (each period once, however many horizons forecast it), unless `spike` gives it. It is
        one number for all models: `df.attrs` holds `spike_threshold`, `spike_rule` and `spike_quantile`, and `no_level`,
        the forecasts per model that are in no group for lack of a finite actual (so the groups add up to `summary()`).
        Every evaluator runs on the group alone, on the keys the models share inside it.

        Args:
            spike: The price (EUR/MWh) from which a period is a spike, a positive number; by default the rule above.
        """
        frames = self._model_frames()
        threshold, rule, quantile = spike_threshold(frames, spike)
        labels = {"price_level": lambda df: price_level(df["actual"], threshold)}
        table = self._grouped_table(frames, ["price_level"], labels=labels)
        table.attrs.update(
            spike_threshold=threshold,
            spike_rule=rule,
            spike_quantile=quantile,
            levels=f"negative: actual < 0; spike: actual >= {threshold:g}; normal: the rest",
            no_level={
                name: n
                for name, df in frames.items()
                if (n := int((~np.isfinite(df["actual"].to_numpy(dtype=float))).sum()))
            },
        )
        return table

    def aggregate(
        self, freq: str = "1D", agg: str = "max", peak_hours: Tuple[int, int] = PEAK_HOURS
    ) -> "EvaluationReport":
        """A report on daily or hourly aggregates of every model's forecasts and actuals, with the same evaluators.

        With `freq="1D"`, for each (run date, target date, horizon) the prediction and the actual are reduced by `agg` over
        the periods of the day, which turns hourly or 15-minute forecasts into forecasts of the day's maximum, minimum, mean or
        spread, or of the mean over a block of its hours: `"peak"` is the mean over the periods of `peak_hours` (local
        08:00 to 19:59 by default) and `"offpeak"` over the rest of the day. The new report has `hour` 0 and frequency
        `"1D"`. A day counts only when all its periods are there, with a finite prediction and actual: 23, 24 or 25 hours
        (92, 96 or 100 quarter-hours), judged by the period ids as `EconomicEvaluator` does, and against the clock of the
        data's time zone (`ModelResultRef.zone`): a day with fewer finite periods than `clock_periods` gives it is dropped.
        So a model that lacks 02:00 on an ordinary Tuesday is not aggregated over 23 hours while another is aggregated over
        24, which would score them against different actuals on one key. The spring switch day is whole over its 23 hours.
        The day after it, from a forecast issued on the switch day, lacks the hour no forecast was issued from and is dropped
        at every horizon, and the autumn switch day is dropped for its repeated hour, which has no `actual`. A block counts
        when every period of it that the day has is there and finite: the peak of both switch days counts, and so does the
        offpeak of the spring day, over the hours its clock has.

        With `freq="1h"`, which only 15-minute results take, the four quarter-hours of each hour are reduced by `agg` (the
        study's MAEh is the MAE of `aggregate("1h", agg="mean")`). An hour counts when all four quarter-hours are there
        with a finite prediction and actual: the repeated autumn hour, which has no actual, is dropped, and the hour the
        spring switch skips is no hour of that day. The new report has frequency `"1h"`, `hour` the local hour, and keeps
        the time zone, so `aggregate("1D")` of it judges days by the clock again.

        Without a zone, for plain lists of records, the periods of a target date are every period that any model has a record
        for, at any horizon. A period that no model has at any horizon is then taken at its word, since that is how a clock
        change looks in the ids (a 23-hour day lacks the hour it skips).

        Each model is judged on its own days, so a model missing a period loses that day (or hour) and no other model does.
        Dropped days are counted, never lost silently: `forecast_counts()` of the new report has them as `missing` and under
        `incomplete_day` (`incomplete_hour` at `"1h"`), and a warning names the models. `planned` there is the days (hours)
        with any forecast, so days without a single forecast are in the first report's `forecast_counts()`.

        The evaluators carry over, except `EconomicEvaluator` at `"1D"` (a schedule needs the periods of a day), and the
        evaluators that read source-data columns (`ScaledMAEEvaluator`), which are left out with a warning: the new report
        has no `source_data`. `summary()`, `by_*()`, `tests()`, `mcs()`, `ranks()`, `compare()` and `leaderboard()` work; on
        a daily report `tests(by="hour")` raises, and `compare()` has no per-hour columns.

        Args:
            freq: `"1D"` for daily series, `"1h"` for hourly ones from quarter-hours.
            agg: `"max"`, `"min"`, `"mean"` or `"spread"` (the maximum minus the minimum) at either frequency; `"peak"` or
                `"offpeak"` at `"1D"`.
            peak_hours: `(first, end)` local hours of the peak block, the end left out: `(8, 20)` is 08:00 to 19:59. Only
                for `agg="peak"` and `"offpeak"`.
        """
        if freq not in (DAILY, HOURLY):
            raise ValueError(
                f"aggregate() builds daily or hourly series: freq must be {DAILY!r} or {HOURLY!r}, got {freq!r}"
            )
        if agg not in AGGREGATES:
            raise ValueError(f"agg must be one of {list(AGGREGATES)}, got {agg!r}")
        block = agg in BLOCK_AGGREGATES
        if block and freq != DAILY:
            raise ValueError(f"agg={agg!r} is the mean over a block of a day's hours: it needs freq={DAILY!r}.")
        peak_hours = check_peak_hours(peak_hours)
        if not block and peak_hours != PEAK_HOURS:
            raise ValueError(
                f"peak_hours sets the block of agg='peak' and 'offpeak'; it means nothing for agg={agg!r}."
            )
        if agg == "offpeak" and peak_hours == (0, 24):
            raise ValueError("peak_hours=(0, 24) leaves no offpeak hours.")
        freqs = self._result_freqs()
        if DAILY in freqs.values():
            raise ValueError(f"These results are daily already ({freqs}): there is nothing to aggregate.")
        if freq == HOURLY and set(freqs.values()) - {QUARTER_HOURLY}:
            raise ValueError(
                f"aggregate('1h') turns quarter-hours into hours, so it needs 15-minute results; these are {freqs}."
            )
        self._require_one_frequency("Aggregating")
        if freq == DAILY:
            self._refuse_hours_dropped_without_a_zone()
        if block or freq == HOURLY:
            self._refuse_records_off_the_clock(agg)

        frames = self._model_frames()
        starts = [df["run_date"].min() for df in frames.values() if "run_date" in df and len(df)]
        start = min(starts) if starts else pd.Timestamp(0)
        # Without a zone, the periods a target date has: every period that any model has a record for, at any horizon.
        keyed = [
            df[["target_date", "hour"]] for df in frames.values() if len(df) and {"target_date", "hour"} <= set(df)
        ]
        periods = pd.concat(keyed).drop_duplicates().groupby("target_date").size() if keyed and freq == DAILY else None
        day_ids: Dict[pd.Timestamp, Tuple[int, ...]] = {}
        if block and keyed:
            for date, ids in pd.concat(keyed).drop_duplicates().groupby("target_date")["hour"]:
                day_ids[pd.Timestamp(date)] = tuple(sorted(int(i) for i in ids))
        results: Dict[str, ModelResultRef] = {}
        dropped: Dict[str, int] = {}
        for name, ref in self.refs.items():
            frame = frames[name]
            model_freq = freqs.get(name, ref.freq)
            if freq == HOURLY:
                rows, seen, missing = hourly_aggregate(frame, agg)
            else:
                per_hour = _PERIODS_PER_HOUR.get(model_freq)
                if per_hour is None:
                    per_hour = periods_per_hour(frame["hour"].to_numpy()) if "hour" in frame else 1
                clock = None
                if ref.zone is not None and "target_date" in frame:
                    clock = clock_periods_on(frame["target_date"].unique(), model_freq, ref.zone)
                if block:
                    if "hour" in frame:
                        check_period_ids(frame["hour"].to_numpy(), model_freq, f"aggregate({agg!r}) of {name!r}")
                    on_the_day = clock if clock is not None else numbered_by_the_clock(day_ids, per_hour)
                    rows, seen, missing = block_aggregate(frame, agg, per_hour, peak_hours, on_the_day)
                else:
                    rows, seen, missing = daily_aggregate(frame, agg, per_hour, periods, clock)
            records = _period_records(rows, start, ref.window, freq)
            results[name] = ModelResultRef(
                name=name,
                path=None,
                count=seen,
                # A plain ref says here what it covers; `windows` does for a ref of a window.
                test_start=ref.test_start if ref.window is None else "",
                test_end=ref.test_end if ref.window is None else "",
                horizon=ref.horizon,
                horizons=ref.horizons,
                issue_days=ref.issue_days,
                freq=freq,
                _results=records,
                produced=len(records),
                skipped={_INCOMPLETE[freq]: missing},
                availability_listing=ref.availability_listing,
                vintages=ref.vintages,
                provenance=ref.provenance,
                window=ref.window,
                target_before=ref.target_before,
                windows=ref.windows,
                **({"zone": ref.zone} if freq == HOURLY else {}),
            )
            if missing:
                dropped[name] = missing

        if dropped:
            if freq == HOURLY:
                what = "hours that lack some of their quarter-hours (hours"
            elif block:
                what = f"days whose {agg} hours lack some of their periods (days"
            else:
                what = "days that lack some of their periods (days"
            warnings.warn(
                f"aggregate({agg!r}) left out {what} per model: {dropped}). They are in forecast_counts() of the new "
                f"report as {_INCOMPLETE[freq]}.",
                stacklevel=2,
            )
        economic = [e for e in self.evaluators if isinstance(e, EconomicEvaluator)] if freq == DAILY else []
        if economic:
            warnings.warn(
                "aggregate() leaves EconomicEvaluator out of the new report: a schedule needs the periods of a day, and "
                "a daily series has one.",
                stacklevel=2,
            )
        sourced = [e for e in self.evaluators if _source_columns_of(e)]
        if sourced:
            warnings.warn(
                f"aggregate() leaves {[type(e).__name__ for e in sourced]} out of the new report: they read source-data "
                "columns at the periods of the results, and the new report has no source data.",
                stacklevel=2,
            )
        evaluators = [e for e in self.evaluators if not any(e is out for out in (*economic, *sourced))]
        return EvaluationReport(results, evaluators, choices=self._choices)

    def _refuse_records_off_the_clock(self, agg: str) -> None:
        """Results with a zone that hold records of slots the clock doesn't have on their date: the reader leaves those out
        (`ModelResultRef.off_the_clock`), so a day numbered through (0-22 on a 23-hour day, 0-91 at 15 minutes) would pass for
        the clock's day with every hour after the switch an hour off, and a block or an hour of it would be read shifted. A run
        never writes such a record; refused, never scored."""
        off = {name: n for name, ref in self.refs.items() if ref.zone is not None and (n := ref.off_the_clock())}
        if off:
            raise ValueError(
                f"aggregate({agg!r}) reads period ids as the local clock's, but these results hold records of slots their "
                f"clock doesn't have, which are off the clock and left out (per model: {off}): their days are numbered "
                "otherwise, so their hours after a switch would be read an hour off. Number the periods by the clock, as a "
                "run does, or pass the records without a zone."
            )

    def _refuse_hours_dropped_without_a_zone(self) -> None:
        """Hourly results of `aggregate("1h")` that dropped hours, with no zone to tell what a day has: a day without its
        dropped 02:00 (the repeated autumn hour, which has no actual) looks like a 23-hour spring day and would be aggregated
        over the hours left, while `aggregate("1D")` of the quarter-hours drops it. Refused, never scored."""
        dropped = {
            name: ref.skipped["incomplete_hour"]
            for name, ref in self.refs.items()
            if ref.zone is None and ref.skipped.get("incomplete_hour")
        }
        if dropped:
            raise ValueError(
                f"These hourly results come from aggregate('1h'), which left out hours (incomplete_hour per model: "
                f"{dropped}), and they have no time zone: a day without such an hour can't be told from a day whose clock "
                "skips it. Aggregate the quarter-hour report to '1D' directly, or give its results a zone "
                "(ModelResultRef.zone), which a report of a run has."
            )

    def error_split(self, period: str = "1h") -> pd.DataFrame:
        """How each model's squared error divides into the level of the day, its hour profile and the shape within the hour.

        One row per model: `model, days, forecasts, mse, level, hour, quarter`. On the model's whole non-switch days (a
        forecast day with all 96 quarter-hours finite, on a date whose clock doesn't change), with `e` the errors of a day,
        the mean squared error `mse` is exactly the sum of three parts, and the table gives each as a share of it: `level`,
        the square of the day's mean error; `hour`, each hour's mean error minus the day's; `quarter`, each quarter-hour's
        error minus its hour's. A forecast flat within the hour has a large `quarter` share, one with the daily level wrong a
        large `level` share. `days` and `forecasts` (quarter-hours) are what each share is on. Each model is judged on its
        own days, so a model with holes has fewer of them; when the counts differ a warning says so and
        `df.attrs["unequal_days"]` holds them.

        Args:
            period: The period the shape is measured within: `"1h"`, the only one so far.
        """
        if period != "1h":
            raise ValueError(f"error_split() splits the error at the hour: period must be '1h', got {period!r}")
        freqs = self._result_freqs()
        if set(freqs.values()) - {QUARTER_HOURLY}:
            raise ValueError(
                f"error_split() splits quarter-hour errors into the day, its hours and the quarter-hours within them, so "
                f"it needs 15-minute results; these are {freqs}."
            )
        frames = self._model_frames()
        rows = []
        for name, ref in self.refs.items():
            frame = frames[name]
            switches = None
            if ref.zone is not None and len(frame) and "target_date" in frame:
                switches = switch_dates(frame["target_date"].unique(), ref.zone)
            rows.append({"model": name, **split_errors(frame, switches)})
        table = pd.DataFrame(rows, columns=ERROR_SPLIT_COLUMNS)
        table.attrs["period"] = period
        days = dict(zip(table["model"], table["days"]))
        if len(set(days.values())) > 1:
            table.attrs["unequal_days"] = days
            warnings.warn(
                f"The models have different numbers of whole days ({days}), so their error splits are on different "
                "samples; see report.forecast_counts().",
                stacklevel=2,
            )
        return table

    def by_column(self, column: str, bins: Optional[Sequence[float]] = None) -> pd.DataFrame:
        """One row per (value of a source-data column, model): `model, <column>, forecasts`, then the metrics.

        Each forecast is grouped by the value of `column` in the report's `source_data` at its target row: the row of its
        target date and period, joined as `CsvExporter(extra_columns=)` joins (the repeated autumn hour takes its second
        row, the one a run forecasts from), so the source must be at the results' frequency. A 0/1 flag such as
        `is_night` gives two groups (daylight MAE is the row of 0); with `bins`, a numeric column is cut into the
        intervals between the edges, each closed on the left (`bins=[-inf, 0, inf]` puts prices below 0 apart). The value
        is read from the data, never from a forecast, so it is the same for every model. A forecast without a value (no
        source row, NaN, or outside the bins) is in no group: `df.attrs["no_value"]` counts them per model, so the groups
        add up to `summary()`. Every evaluator runs on the group alone, on the keys the models share inside it.

        Args:
            column: The source-data column to group by.
            bins: Increasing edges to cut a numeric column at; None groups by the values themselves.
        """
        if self.source_data is None:
            raise ValueError(
                "by_column() groups forecasts by a column of the report's source data, and this report has none (a report "
                "from aggregate() has none: group the report it came from)."
            )
        if column in _TABLE_KEYS or column in ("prediction", "actual"):
            raise ValueError(f"by_column({column!r}): the report's tables already use that name; rename the column.")
        edges = None
        if bins is not None:
            if isinstance(bins, (int, float, np.number)) or isinstance(bins, str):
                raise ValueError(
                    f"bins must be a list of edges, not {bins!r}: a number of bins would put each model's edges elsewhere."
                )
            edges = np.asarray(list(bins), dtype=float)
            if len(edges) < 2 or not (np.diff(edges) > 0).all():
                raise ValueError(f"bins must be at least two increasing edges, got {list(bins)}")
        freq = self._require_one_frequency("by_column()") or HOURLY
        frames = self._model_frames()
        table = self._source_table([column], frames, freq)

        def value_of(df: pd.DataFrame) -> pd.Series:
            if not len(df) or not {"target_date", "hour"} <= set(df):
                return pd.Series(np.nan, index=df.index, dtype=object)
            values = source_values(table, df)[column]
            if edges is None:
                return values
            cut = pd.cut(pd.to_numeric(values, errors="coerce"), edges, right=False)
            return pd.Series(cut.astype(object), index=df.index)

        no_value = {}
        for name, df in frames.items():
            if n := int(value_of(df).isna().sum()):
                no_value[name] = n
        grouped = self._grouped_table(frames, [column], labels={column: value_of})
        grouped.attrs.update(by_column=column, bins=None if edges is None else edges.tolist(), no_value=no_value)
        return grouped

    def _source_table(self, columns: Sequence[str], frames: Mapping[str, pd.DataFrame], freq: str) -> pd.DataFrame:
        """`breakdowns.source_table` of the report's source data, checked over the target dates of `frames`.

        The join reads the source's wall clock, so a source in another time zone than the results' would give each forecast
        another hour's value: with results that know their zone, the source's index must be in it."""
        check_source_zone(self.source_data, [ref.zone for ref in self.refs.values()], "Joining source-data columns")
        dates = [df["target_date"] for df in frames.values() if len(df) and "target_date" in df]
        days = None
        if dates:
            every = pd.DatetimeIndex(pd.concat(dates))
            days = (every.min(), every.max())
        return source_table(self.source_data, columns, freq, days)

    def join(self, other: "EvaluationReport") -> "EvaluationReport":
        """A report with the models of both, for results of separate runs that are to be compared.

        The references to the results are kept as they are, and the evaluators are the ones of this report: when `other`
        has different ones a warning says so, since the columns of its evaluators are lost and its models are scored by
        these. A model name in both reports raises `ValueError`, and so do results at different frequencies (period 5 of
        hourly and 15-minute results are different times of the day). Every table of the joined report is what one report
        over all the models would give, because all of them are computed on the forecast keys, not on file order.

        Reports on different windows raise `ValueError`, whether they differ in `window` or, for a run with validation, in
        the days of it. `.validation` of the joined report joins theirs, if both have one.

        Args:
            other: The report whose models are added after this report's.
        """
        if not isinstance(other, EvaluationReport):
            raise TypeError(f"join() takes an EvaluationReport, got {type(other).__name__}")
        clash = [name for name in {**other.refs, **other._choices} if name in {**self.refs, **self._choices}]
        if clash:
            raise ValueError(f"Both reports have a model named {clash}: rename one before joining.")
        self._require_same_window(other)
        validation = None
        if self.window == TEST and self._validation is not None and other._validation is not None:
            validation = self._validation.join(other._validation)
        joined = EvaluationReport(
            {**self.refs, **other.refs},
            list(self.evaluators),
            source_data=self.source_data if self.source_data is not None else other.source_data,
            validation=validation,
            choices={**self._choices, **other._choices},
        )
        joined._require_one_frequency("Joining reports")
        left, right = [_describe(e) for e in self.evaluators], [_describe(e) for e in other.evaluators]
        if left != right:
            warnings.warn(
                f"The reports have different evaluators, and the joined report uses the left one's: {left} (the other "
                f"report's: {right}).",
                stacklevel=2,
            )
        return joined

    def _require_same_window(self, other: "EvaluationReport") -> None:
        if self.window != other.window:
            raise ValueError(
                f"These reports are on different windows ({self.window!r} and {other.window!r}): their forecasts play "
                "different roles and can't be compared. Join report.validation with another report.validation."
            )
        refs = [*self.refs.values(), *other.refs.values()]
        if any(ref.window is not None for ref in refs):
            periods = {_period_of(ref) for ref in refs} - {None}
            if len(periods) > 1:
                raise ValueError(
                    f"These reports are on different {self.window} periods ({sorted(periods, key=str)}): compare forecasts "
                    "of one period, or run them with the same validation= and test=."
                )

    def tests(
        self,
        benchmark: Optional[str] = None,
        horizon: Union[int, str, None] = None,
        variance: str = "hac",
        lag: Optional[int] = None,
        correction: Optional[str] = "holm",
        by: Optional[str] = None,
        interval: Optional[float] = 0.95,
        block: int = 7,
        reps: int = 4000,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Each model against `benchmark`, on the forecasts the pair shares: MAE ratio, its interval, a DM test.

        One row per (model, horizon): `model, benchmark, horizon, n_days, n_forecasts, mae, mae_benchmark, ratio, ci_low,
        ci_high, statistic, p_value, p_holm, m, mde, verdict`. A positive statistic means the model is better. `ratio` is
        `mae / mae_benchmark`, both forecast-weighted on the pair's keys, and `mde` the difference (a share of
        `mae_benchmark`) this test length would likely detect, by the Newey-West variance whatever `variance` is. Holm runs
        over the `m` models tested at that horizon, and `verdict` reads `p_holm` at 5 %. Exact clones, models with the same
        predictions (an `AverageEnsemble` with `top=1` and what it chose), count once and share a `p_holm`, and a clone of the
        benchmark doesn't count. Keys count: two models are one test when the keys each shares with the benchmark, and their
        predictions there, are the same. `verdict` is empty below 30 days and when the test and the ratio point in opposite
        ways, the interval when the days fit in one block. `df.attrs` holds the settings, `clones`, the groups of clones per
        horizon of the table, and `partial_days`: the days per model with only part of their periods in common, with a warning.

        Args:
            benchmark: The reference model; by default the benchmark of the report's `DMEvaluator`.
            horizon: `None` for one set of rows per horizon, a number for one horizon, `"all"` to average the daily
                losses of all horizons per run date.
            variance: `"hac"` (Newey-West) or `"plain"` (epftoolbox's variance).
            lag: Newey-West lag; the default is `stats.default_lag` of the number of days (5 at 365).
            correction: `"holm"`, or `None` (`p_holm` empty, `verdict` from `p_value`).
            by: `"hour"` for one row per (model, horizon, hour), without `m`, `p_holm` and `verdict`: a per-hour DM is a
                heatmap, never a verdict (it called two seeds of one model different 37 % of the time).
            interval: Coverage of the bootstrap interval of the ratio (blocks of `block` issue days); `None` skips it.
            block: Block length of that bootstrap, in issue days (observations: seven Fridays are seven weeks).
            reps: Bootstrap repetitions.
            seed: Bootstrap seed.

        Warns when a comparison spans fewer than 182 calendar days from its first run date to its last; below 365
        `df.attrs["note"]` says so. `n_days` counts the days with forecasts: 104 Fridays over two years are 104.
        """
        check_test_settings(variance, lag, correction, seed, interval)
        if by not in (None, "hour"):
            raise ValueError(f"by must be None or 'hour', got {by!r}")
        if by == "hour" and self._is_daily():
            raise ValueError(
                "by='hour' needs periods within the day, but this report is daily (it comes from aggregate())."
            )
        panel = self._panel()
        benchmark = self._benchmark(benchmark, panel, required=True)
        models = [m for m in panel.models if m != benchmark]
        settings = dict(variance=variance, lag=lag, interval=interval, block=block, reps=reps, seed=seed)

        rows: List[Dict[str, Any]] = []
        lengths: List[Tuple[int, int]] = []  # (calendar span, days) of each row, for the test-length note
        partial: Dict[str, int] = {}
        clones: Dict[Union[int, str], List[List[str]]] = {}
        for label, keys in self._horizon_slices(panel, horizon, each_horizon=True):
            if by == "hour":
                clones[label] = panel.pair_clone_groups(keys, models, benchmark)
                hours = panel.hour[keys]
                for hour in np.unique(hours):
                    for model in models:
                        pair = compare_pair(panel, model, benchmark, keys[hours == hour], **settings)
                        rows.append(_pair_row(model, benchmark, label, pair, hour=int(hour)))
                        lengths.append((pair.span_days, pair.n_days))
                continue
            pairs = [compare_pair(panel, model, benchmark, keys, **settings) for model in models]
            lengths += [(pair.span_days, pair.n_days) for pair in pairs]
            p_values = [pair.p_value for pair in pairs]
            p_holm, m, clones[label] = adjust_family(panel, keys, models, benchmark, p_values, correction)
            for model, pair, adjusted in zip(models, pairs, p_holm):
                row = _pair_row(model, benchmark, label, pair)
                judged = float(adjusted) if correction == "holm" else pair.p_value
                row.update(p_holm=float(adjusted), m=m, mde=pair.mde, verdict=verdict(pair, judged, _TEST_ALPHA))
                rows.append(row)
                partial[model] = partial.get(model, 0) + pair.partial_days

        columns = list(_PAIR_COLUMNS)
        if by == "hour":
            columns.insert(columns.index("n_days"), "hour")
        else:
            columns += ["p_holm", "m"]
        columns += ["mde"] + ([] if by == "hour" else ["verdict"])
        table = pd.DataFrame(rows, columns=columns)
        table.attrs.update(
            benchmark=benchmark,
            variance=variance,
            lag=lag,
            correction=correction,
            interval=interval,
            alpha=_TEST_ALPHA,
            clones=clones,
        )
        if by == "hour":
            table.attrs["descriptive"] = (
                "Per-hour DM is a heatmap, never a verdict: it called two seeds of one model different 37 % of the time "
                "even after Holm, so this table has no p_holm and no verdict. Use it to see where in the day two models "
                "differ."
            )
        if len(table):
            # Per hour the switch hour has a day less than the others; the length of the test is what counts.
            span, n_days = max(lengths) if by == "hour" else min(lengths)
            _flag_test_length(table, n_days, span)
        _flag_partial_days(table, partial)
        self._flag_vintages(table, [(model, benchmark) for model in models])
        return table

    def mcs(
        self,
        alpha: float = 0.10,
        statistic: str = "R",
        block: int = 7,
        reps: int = 1000,
        seed: int = 0,
        models: Optional[Sequence[str]] = None,
        horizon: Union[int, str, None] = None,
    ) -> pd.DataFrame:
        """The model confidence set: which models the data cannot tell from the best one.

        One row per model, by MAE: `model, n_days, n_forecasts, mae, ratio_to_best, p_value, in_90, in_75`, with `in_90`
        the 90 % set (p-value above 0.10) and `in_75` the 75 % set (above 0.25). An `alpha` other than those adds its own
        column (`alpha=0.05`: `in_95`). Below 30 days the sets stay empty. The comparison here is the whole set of models,
        so everything is on the keys all of them hold: `df.attrs` holds `n_days`, `n_forecasts` and `limited_by`, the models
        that lack forecasts the others have, with how many. If one model cuts the sample, leave it out with `models`. Exact
        clones on those keys (an `AverageEnsemble` with `top=1` and what it chose) are one model to the set, each with its
        result, and count once among the 3: their groups are in `df.attrs["clones"]`.

        Args:
            alpha: Size of the set the result is built for.
            statistic: `"R"` (T_R, recommended: seed-stable) or `"max"` (T_max, which keeps models much worse than the
                best and changes with the seed).
            block: Mean block length of the stationary bootstrap, in issue days (observations).
            reps: Bootstrap repetitions (at least 1,000 recommended).
            seed: Bootstrap seed.
            models: The set of models; by default all of them. Fewer than 3, counting exact clones once, gives NaN with a
                warning.
            horizon: A number for one horizon; by default (or `"all"`) the daily losses of all horizons are averaged per
                run date.
        """
        check_mcs_settings(alpha, statistic, seed)
        panel = self._panel()
        names = self._chosen_models(panel, models)
        label, keys = self._pooled_slice(panel, horizon)
        compared = compare_set(panel, names, keys)
        daily = compared.daily
        levels = sorted({0.10, 0.25, float(alpha)})
        included: Dict[str, Any] = {_set_column(level): np.nan for level in levels}
        p_values: Any = np.nan
        if len(names) < 3:
            warnings.warn(
                f"The model confidence set needs at least 3 models, got {len(names)}: its columns are NaN.",
                stacklevel=2,
            )
        elif too_few_distinct_models(compared) is not None:
            warnings.warn(too_few_distinct_models(compared), stacklevel=2)
        elif daily.n_days < 2:
            warnings.warn(_too_few_days(compared), stacklevel=2)
        else:
            result = confidence_set(compared, alpha, statistic, block, reps, seed)
            p_values = result.p_values
            if daily.n_days >= MIN_VERDICT_DAYS:
                included = {_set_column(level): result.included(level) for level in levels}
        mae = daily.mae()
        best = np.nanmin(mae) if np.isfinite(mae).any() else np.nan
        table = pd.DataFrame(
            {
                "model": names,
                "n_days": daily.n_days,
                "n_forecasts": daily.n_forecasts,
                "mae": mae,
                "ratio_to_best": mae / best if best else np.nan,
                "p_value": p_values,
                **included,
            }
        )
        table = table.sort_values("mae", kind="stable").reset_index(drop=True)
        return self._finish_set_table(table, compared, label, alpha=alpha, statistic=statistic, reps=reps, seed=seed)

    def ranks(
        self,
        block: int = 7,
        reps: int = 4000,
        seed: int = 0,
        level: float = 0.90,
        horizon: Union[int, str, None] = None,
        models: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """How decisive the ranking is: every model's chance of being best and the interval of its rank.

        One row per model, by rank: `model, n_days, n_forecasts, mae, rank, p_best, rank_low, rank_high`. Days are resampled
        in blocks of `block` issue days and the models ranked in each sample. As in `mcs()`, everything is on the keys all the models hold
        (`df.attrs` holds `n_days`, `n_forecasts`, `limited_by`), and `rank` is by the mean daily loss. Exact clones on those
        keys (an `AverageEnsemble` with `top=1` and what it chose) are ranked as one model and have one rank and one `p_best`,
        so they share a row's numbers and the next model has the next rank: their groups are in `df.attrs["clones"]`. Other
        models with the same losses on every day are ranked by their order in the report, with a warning.

        Args:
            block: Block length of the bootstrap, in issue days (observations).
            reps: Bootstrap repetitions.
            seed: Bootstrap seed.
            level: Coverage of the rank interval.
            horizon: A number for one horizon; by default (or `"all"`) the daily losses of all horizons are averaged per
                run date.
            models: The models to rank; by default all of them.
        """
        panel = self._panel()
        names = self._chosen_models(panel, models)
        label, keys = self._pooled_slice(panel, horizon)
        compared = compare_set(panel, names, keys)
        daily = compared.daily
        columns: Dict[str, Any] = dict.fromkeys(("rank", "p_best", "rank_low", "rank_high"), np.nan)
        if daily.n_days < 2:
            warnings.warn(_too_few_days(compared), stacklevel=2)
        elif daily.n_days <= block:
            warnings.warn(
                f"{daily.n_days} days fit in one bootstrap block of {block}: every resample would hold all of them, so "
                "there is no rank uncertainty to report and its columns are NaN.",
                stacklevel=2,
            )
        else:
            kept, twin_of = compared.kept, compared.twin_of
            _warn_on_identical_losses(daily.losses[:, kept], [names[j] for j in kept])
            found = stats.rank_bootstrap(daily.losses[:, kept], block=block, reps=reps, seed=seed, level=level)
            columns = {
                "rank": found.rank[twin_of],
                "p_best": found.p_best[twin_of],
                "rank_low": found.rank_low[twin_of],
                "rank_high": found.rank_high[twin_of],
            }
        table = pd.DataFrame(
            {"model": names, "n_days": daily.n_days, "n_forecasts": daily.n_forecasts, "mae": daily.mae(), **columns}
        )
        table = table.sort_values(["rank", "mae"], kind="stable").reset_index(drop=True)
        return self._finish_set_table(table, compared, label, level=level, reps=reps, seed=seed)

    def compare(self, a: str, b: str, horizon: Union[int, str, None] = None) -> pd.DataFrame:
        """Two models on the forecasts they share, in one row. `ratio` is `mae_a / mae_b`: below 1, `a` is better.

        Columns: `a, b, horizon, n_days, n_forecasts, mae_a, mae_b, ratio, ci_low, ci_high, statistic, p_plain, p_hac,
        hours_a_better, hours_b_better, hours_tested, mde`. `statistic` is the HAC DM statistic (positive: `a` is better),
        `p_plain` and `p_hac` its two-sided p-values with the plain and the Newey-West variance. The `hours_*` columns count
        the hours where a per-hour DM test (plain variance, Holm over the hours) rejects: descriptive, see
        `df.attrs["descriptive"]`. `mde` is the difference this test length would likely detect, a share of `mae_b`.

        Args:
            a: One model (the one in the numerator of `ratio`).
            b: The other (the benchmark of the test).
            horizon: A number for one horizon; by default (or `"all"`) the daily losses of all horizons are averaged per
                run date.
        """
        panel = self._panel()
        self._chosen_models(panel, [a, b])
        label, keys = self._pooled_slice(panel, horizon)
        hac = compare_pair(panel, a, b, keys)
        plain = compare_pair(panel, a, b, keys, variance="plain", interval=None)

        shared = panel.common(keys, (a, b))
        hours = panel.hour[shared]
        statistics, p_values = [], []
        daily_series = self._is_daily()
        for hour in [] if daily_series else np.unique(hours):
            daily = panel.daily(shared[hours == hour], (a, b))
            if daily.n_days >= 2:
                result = stats.dm_test(daily.losses[:, 0], daily.losses[:, 1], variance="plain")
                statistics.append(result.statistic)
                p_values.append(result.p_value)
        rejected = stats.holm(p_values) < _TEST_ALPHA
        table = pd.DataFrame(
            [
                {
                    "a": a,
                    "b": b,
                    "horizon": label,
                    "n_days": hac.n_days,
                    "n_forecasts": hac.n_forecasts,
                    "mae_a": hac.mae,
                    "mae_b": hac.mae_benchmark,
                    "ratio": hac.ratio,
                    "ci_low": hac.ci_low,
                    "ci_high": hac.ci_high,
                    "statistic": hac.statistic,
                    "p_plain": plain.p_value,
                    "p_hac": hac.p_value,
                    # A daily series has no hours in a day: there is no per-hour test to count.
                    "hours_a_better": np.nan if daily_series else int((rejected & (np.asarray(statistics) > 0)).sum()),
                    "hours_b_better": np.nan if daily_series else int((rejected & (np.asarray(statistics) < 0)).sum()),
                    "hours_tested": np.nan if daily_series else len(p_values),
                    "mde": hac.mde,
                }
            ]
        )
        table.attrs.update(
            ratio="mae_a / mae_b: below 1, a is better; a positive statistic also means a is better.",
            descriptive=(
                "hours_a_better and hours_b_better are descriptive: per-hour DM called two seeds of one model different "
                "37 % of the time even after Holm. They show where in the day the models differ, not which one is better."
            ),
        )
        _flag_test_length(table, hac.n_days, hac.span_days)
        _flag_partial_days(table, {a: hac.partial_days})
        self._flag_vintages(table, [(a, b)])
        return table

    def leaderboard(self, benchmark: Optional[str] = None, horizon: Union[int, str, None] = None) -> pd.DataFrame:
        """One row per model, by MAE, with its comparison against `benchmark` on the forecasts the pair shares.

        `n_forecasts` and `MAE` are the model's own, so with different forecast sets the MAE column compares different
        samples (a warning says so): read `ratio`, which does not. `n_days` and `n_common` are the days and forecasts of the
        pair, and `ratio`, `ci_low`, `ci_high`, `p_value` and `p_holm` (Holm over the models compared, with exact clones
        counted once, as in `tests()`) are on them, with the defaults of `tests()`. Without a benchmark (none given, no
        `DMEvaluator`) only `n_days` is added. `df.attrs["clones"]` holds the groups of clones: of the tests with a benchmark,
        and of the models with the same forecasts on the keys of the table without one.

        Args:
            benchmark: The reference model; by default the benchmark of the report's `DMEvaluator`.
            horizon: A number for one horizon; by default (or `"all"`) the daily losses of all horizons are averaged per
                run date.
        """
        panel = self._panel()
        benchmark = self._benchmark(benchmark, panel, required=False)
        _, keys = self._pooled_slice(panel, horizon)
        clones = panel.clone_groups(keys, panel.models)
        rows: List[Dict[str, Any]] = []
        lengths: Dict[str, Tuple[int, int]] = {}  # model -> (calendar span, days) of its row, for the test-length note
        for model in panel.models:
            own = panel.common(keys, (model,))
            daily = panel.daily(own, (model,))
            mae = float(daily.mae()[0])
            rows.append({"model": model, "n_forecasts": len(own), "MAE": mae, "n_days": daily.n_days})
            lengths[model] = (daily.span_days, daily.n_days)
        held = ~np.isnan(panel.errors[keys])
        if (held != held[:, :1]).any():
            warnings.warn(
                "These models have different forecast sets, so their MAEs are on different samples. The ratio "
                "columns are on the keys of each pair; see report.forecast_counts().",
                stacklevel=2,
            )
        if benchmark is not None:
            others = [row for row in rows if row["model"] != benchmark]
            pairs = [compare_pair(panel, row["model"], benchmark, keys) for row in others]
            names = [row["model"] for row in others]
            p_holm, _, clones = adjust_family(panel, keys, names, benchmark, [pair.p_value for pair in pairs], "holm")
            partial = {row["model"]: pair.partial_days for row, pair in zip(others, pairs)}
            for row, pair, adjusted in zip(others, pairs, p_holm):
                lengths[row["model"]] = (pair.span_days, pair.n_days)
                row.update(
                    n_days=pair.n_days,
                    n_common=pair.n_forecasts,
                    ratio=pair.ratio,
                    ci_low=pair.ci_low,
                    ci_high=pair.ci_high,
                    p_value=pair.p_value,
                    p_holm=float(adjusted),
                )
            reference = next(row for row in rows if row["model"] == benchmark)
            reference.update(n_common=reference["n_forecasts"], ratio=1.0)
        columns = ["model", "n_forecasts", "MAE", "n_days"]
        if benchmark is not None:
            columns += ["n_common", "ratio", "ci_low", "ci_high", "p_value", "p_holm"]
        table = pd.DataFrame(rows, columns=columns).sort_values("MAE", kind="stable").reset_index(drop=True)
        table.attrs["benchmark"] = benchmark
        table.attrs["clones"] = clones
        if len(table):
            span, n_days = min(lengths.values())
            _flag_test_length(table, n_days, span)
        if benchmark is not None:
            _flag_partial_days(table, partial)
        self._flag_vintages(table, [panel.models])  # ranked by MAE all together
        return table

    # -- what the comparison tables share ---------------------------------------------------------------------------

    def _panel(self) -> LossPanel:
        freq = self._require_one_frequency()
        # A daily series has one period a day, and that is a whole day.
        return LossPanel(self._model_frames(), **({"whole_day_periods": (1,)} if freq == DAILY else {}))

    def _is_daily(self) -> bool:
        return DAILY in self._result_freqs().values()

    def _require_one_frequency(self, action: str = "Comparing models") -> Optional[str]:
        """Period 5 of hourly results and of quarter-hour results are different times of the day: they cannot be paired.

        Returns the one frequency, or None when no model has results."""
        freqs = self._result_freqs()
        if len(set(freqs.values())) > 1:
            raise ValueError(
                f"{action} needs one frequency, but these results have several: {freqs}. Their period ids "
                "(hour 0-23 vs quarter-hour 0-95) name different times of the day."
            )
        return next(iter(freqs.values()), None)

    def _check_set_evaluators(self) -> None:
        if any(isinstance(evaluator, SetEvaluator) for evaluator in self.evaluators):
            self._require_one_frequency()

    def _benchmark(self, benchmark: Optional[str], panel: LossPanel, required: bool) -> Optional[str]:
        if benchmark is None:
            benchmark = next((ev.benchmark for ev in self.evaluators if isinstance(ev, DMEvaluator)), None)
        if benchmark is None:
            if required:
                raise ValueError("No benchmark: pass benchmark=... or add a DMEvaluator to the pipeline.")
            return None
        if benchmark not in panel:
            raise ValueError(f"Unknown benchmark {benchmark!r}; models: {panel.models}")
        return benchmark

    @staticmethod
    def _chosen_models(panel: LossPanel, models: Optional[Sequence[str]]) -> List[str]:
        if models is None:
            return list(panel.models)
        names = list(models)
        unknown = [m for m in names if m not in panel]
        if unknown:
            raise ValueError(f"Unknown model(s) {unknown}; models: {panel.models}")
        if len(set(names)) != len(names):
            raise ValueError(f"Models are listed twice: {names}")
        return names

    @staticmethod
    def _horizon_slices(
        panel: LossPanel, horizon: Union[int, str, None], each_horizon: bool
    ) -> List[Tuple[Union[int, str], np.ndarray]]:
        """`[(label in the table, positions of the keys)]`, one per horizon or one for all of them together."""
        if horizon == "all" or (horizon is None and not each_horizon):
            return [("all", panel.rows())]
        if horizon is None:
            return [(h, panel.rows(h)) for h in panel.horizons]
        if isinstance(horizon, str) or horizon not in panel.horizons:
            raise ValueError(f"horizon must be 'all' or one of {panel.horizons}, got {horizon!r}")
        return [(int(horizon), panel.rows(horizon))]

    def _pooled_slice(self, panel: LossPanel, horizon: Union[int, str, None]) -> Tuple[Union[int, str], np.ndarray]:
        """All horizons together unless `horizon` names one; a report with one horizon is labelled with it."""
        if horizon is None and len(panel.horizons) == 1:
            horizon = panel.horizons[0]
        return self._horizon_slices(panel, horizon, each_horizon=False)[0]

    @staticmethod
    def _finish_set_table(
        table: pd.DataFrame, compared: SetComparison, label: Union[int, str], **settings
    ) -> pd.DataFrame:
        table.attrs.update(
            n_days=compared.daily.n_days,
            n_forecasts=compared.daily.n_forecasts,
            limited_by=compared.limited_by,
            clones=[list(group) for group in compared.clones],
            horizon=label,
            **settings,
        )
        _flag_test_length(table, compared.daily.n_days, compared.daily.span_days)
        _flag_partial_days(table, {"the set": compared.daily.partial_days})
        return table

    def predictions(self) -> pd.DataFrame:
        """Return a tidy DataFrame of predictions only (no actuals required)."""
        cols = ["run_date", "target_date", "hour", "horizon", "prediction"]
        rows = []
        for model_name, ref in self.refs.items():
            for r in self._iter_ref(ref, cols=cols):
                row = {k: r[k] for k in cols if k in r}
                row["model"] = model_name
                rows.append(row)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        # By model, horizon and hour as before; the date columns only settle rows that tie on those.
        by = [c for c in ("model", "horizon", "hour", "run_date", "target_date", "prediction") if c in df.columns]
        return df.sort_values(by, kind="stable").reset_index(drop=True)

    def forecast_counts(self) -> pd.DataFrame:
        """Planned vs produced forecasts per model, with one column per skip reason.

        Models are only comparable on the forecasts they all produced; check this table before comparing.
        `planned` equals `produced` for results passed in as plain lists.
        """
        reasons = sorted({reason for ref in self.refs.values() for reason in ref.skipped})
        rows = []
        for name, ref in self.refs.items():
            produced = ref.produced
            if produced is None:
                produced = sum(1 for _ in self._iter_ref(ref, cols=["hour"]))
            planned = max(ref.count, produced)
            row = {"model": name, "planned": planned, "produced": produced, "missing": planned - produced}
            row.update({reason: ref.skipped.get(reason, 0) for reason in reasons})
            rows.append(row)
        return pd.DataFrame(rows, columns=["model", "planned", "produced", "missing", *reasons])

    def availability(self) -> pd.DataFrame:
        """The inputs of each model that its rule puts after the issue time or that have no rule, as a run with
        `availability="warn"` found them, and every series a sequence model reads (`models._history`), late or not. Empty, with
        the same columns, when there is nothing to list.

        Columns: `model`, `predictor`, `role` (`predictor`, `training target`, or what a predictor function read it with:
        `ctx.value`, `ctx.published`; a sequence model's `target`, `past` or `known_ahead`), `status` (`late`, `no rule`, or
        `ok` for a sequence model's series read in time), `list` (the hash of a predictor function's list; empty for fixed lists
        and for what the function only read), `periods` and `horizons` (the exact uses of the row, such as `10-23` and `1-7`),
        `rule`, `known` and `issued` (relative to the issue day D, on the day and period it comes latest), and `note` (an
        assumption or setting that would make it known in time). With a sequence model's rows also `newest` (how far each series
        is read, relative to D) and `vintage` (its rule's class), empty on the other rows.

        A report built from `save_dir` shows what the manifests hold, for every test period the results were run on.
        """
        return listing_frame({name: ref.availability_listing for name, ref in self.refs.items()})

    def vintages(self) -> pd.DataFrame:
        """The vintage class of each input of each model: what the archive it was read from holds, by its rule's `vintage`.
        Empty, with the same columns, when no model read an input with a class (or a run had no information set).

        Columns: `model`, `predictor` (as the model read it, `price_d-1`), `series` (the column its rule is for, `price`),
        `class` (`snapshot`, `row_stamp`, `last_write_stamp`, `latest_revision`, or `unknown` for a rule without one; the classes
        of a column's rules over time joined by ", ") and `ctx` (whether a predictor function read it through `ctx`). An
        ensemble's rows are its models'.

        A class changes no number. A `latest_revision` or `last_write_stamp` series holds values written after the issue day,
        which no forecast saw then: a backtest reading it is more accurate than a live run would have been. Comparisons of
        models that read one series with different classes warn (`df.attrs["vintages"]`).
        """
        return vintage_frame({name: ref.vintages for name, ref in self.refs.items()})

    def provenance(self) -> pd.DataFrame:
        """What each model with a card is under and was trained on (`models._cards.ModelCard`): one row per such model, empty
        with the same columns when none has one (every core model).

        Columns: `model`, `family`, `licence`, `licence_url`, `commercial_use`, `revenue_cap` (a licence that loses its rights
        above a revenue), `attribution` (what a publication must credit), `weights` (`repo@revision` of the cache, or the path
        given as `weights=`), `sha256` (the first 12 hex digits of each weight file's hash), `runtime` (the version, and commit,
        of each runtime package, and the settings a resume compares), `corpus`, `corpus_end`, `released`, and the contamination
        verdict of each window of the run, `validation_verdict` and `test_verdict` (empty for a window the run hasn't):
        `clean` when the corpus ends before the window's first issue day or the model trains on the run's data, `clean
        (released before)` when the corpus end is undocumented and the weights were released before that day, `possibly
        contaminated` otherwise, or the card's stated verdict. Each model is judged by the windows of its own run, also in a
        report that joins runs over other windows. A report built from `save_dir` reads the manifests. See the docs page
        Licences and Weights."""
        # Each model by its own run's windows: a report can join the refs of runs over other windows (`join`, refs passed in).
        starts = {
            name: window_starts(ref.windows, None) if ref.windows is not None else {TEST: ref.test_start}
            for name, ref in self.refs.items()
        }
        return provenance_frame({name: ref.provenance for name, ref in self.refs.items()}, starts)

    def _flag_vintages(self, table: pd.DataFrame, groups: Sequence[Sequence[str]], stacklevel: int = 2) -> None:
        """Say so when models compared in `table` (each group compared together) read one series with different classes."""
        flag_mixed_classes(
            table, {name: ref.vintages for name, ref in self.refs.items()}, groups, stacklevel=stacklevel + 1
        )

    def _flag_dm_vintages(self, table: pd.DataFrame, models: Sequence[str], stacklevel: int = 2) -> None:
        """`_flag_vintages` for the pairs a `DMEvaluator` column compares: each model with the benchmark."""
        groups = [
            (model, evaluator.benchmark)
            for evaluator in self.evaluators
            if isinstance(evaluator, DMEvaluator)
            for model in models
            if model != evaluator.benchmark
        ]
        if groups:
            self._flag_vintages(table, groups, stacklevel=stacklevel + 1)

    def training_days(self) -> pd.DataFrame:
        """How many days the forecasts of each `train_on="matching"` list train on, from the records of this report's window.
        Empty, with the same columns, when no list has that rule.

        Columns: `model`, `list` (its hash), `predictors`, `forecasts` (how many of its forecasts exist), `periods` and
        `horizons` (where), `smallest_days`, `median_days` and `largest_days` (the days they train on: past days of the window
        the function returned this list for, after dropping rows with a missing value), and `thin_forecasts`
        (how many train on no more days than the list has predictors).
        """
        return training_day_frame(rows_of_refs(self.refs))

    def choices(self) -> pd.DataFrame:
        """What each ensemble chose on validation, one row per ensemble and test period served. Empty, with the same columns,
        when nothing chose anything: a plain average chooses nothing.

        Columns: `ensemble`, `segment` (its place in `report.windows.segments`), `chosen` (the members kept, in member order),
        `k`, `validation_mae` (of the average of `chosen`), `validation_forecasts` (how many it was scored on: the keys every
        member holds with a finite prediction and actual, issued in the validation period for a day before the first test
        issue day), `validation_share` (those keys as a share of the ones the fullest member holds), `validation_start`,
        `validation_end`, and `test_start`, `test_end`, the test period served. With no test period known yet those two are
        empty, `segment` is 0, and `report.windows.segments` is empty.

        `df.attrs` holds, per ensemble and segment, `member_mae` (each member's MAE on those keys), `k_mae` (with
        `top="validation"`, the MAE of the average of the best k members) and `limited_by` (the members whose holes cut the keys,
        with how many forecasts each lacks). A report of either window shows the same table.
        """
        rows = []
        member_mae: Dict[str, Dict[int, Dict[str, float]]] = {}
        k_mae: Dict[str, Dict[int, Dict[str, float]]] = {}
        limited_by: Dict[str, Dict[int, Dict[str, int]]] = {}
        for ensemble, chosen in self._choices.items():
            for row in chosen:
                test = row["test"] or (None, None)
                rows.append(
                    {
                        "ensemble": ensemble,
                        **{key: row[key] for key in ("segment", "chosen", "k")},
                        **{key: row[key] for key in ("validation_mae", "validation_forecasts", "validation_share")},
                        "validation_start": row["validation"][0],
                        "validation_end": row["validation"][1],
                        "test_start": test[0],
                        "test_end": test[1],
                    }
                )
                member_mae.setdefault(ensemble, {})[row["segment"]] = row["member_mae"]
                limited_by.setdefault(ensemble, {})[row["segment"]] = row["limited_by"]
                if "k_mae" in row:
                    k_mae.setdefault(ensemble, {})[row["segment"]] = row["k_mae"]
        table = pd.DataFrame(rows, columns=CHOICE_COLUMNS)
        table.attrs.update(member_mae=member_mae, k_mae=k_mae, limited_by=limited_by)
        return table

    def _result_freqs(self) -> Dict[str, str]:
        """The frequency each model's results were produced at, for the models that have results.

        Read from the result records, which carry it. `ref.freq` is only what the caller said (the default of
        `ModelPipeline.report` is "1h"), so it is the fallback for records without the key, which are hourly.
        """
        freqs: Dict[str, str] = {}
        for name, ref in self.refs.items():
            first = ref.first_record(cols=["freq"])  # one line of a file, not the whole of it
            if first is not None:
                freqs[name] = first.get("freq", ref.freq)
        return freqs

    def iter_details(self) -> Iterator[Tuple[str, pd.DataFrame]]:
        detail_cols = ["run_date", "target_date", "hour", "horizon", "day_in_test", "actual", "prediction"]
        for name, ref in self.refs.items():
            rows = [{k: r[k] for k in detail_cols if k in r} for r in self._iter_ref(ref, cols=detail_cols)]
            df = _in_record_order(pd.DataFrame(rows))
            yield name, df
            del df, rows

    def _apply_evaluators(
        self,
        model_dfs: Mapping[str, pd.DataFrame],
        group_keys: Sequence[str] = (),
        freqs: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """`{model: {column: value}}` for one group: each evaluator runs once, on every model's frame of that group.

        `freqs` (`_result_freqs()`) goes to the evaluators with the frames (`ModelFrames.freqs`): a within-hour shape exists
        only in quarter-hour results."""
        frames = model_dfs if isinstance(model_dfs, ModelFrames) else ModelFrames(model_dfs, group_keys, freqs)
        columns: Dict[str, Dict[str, Any]] = {name: {} for name in model_dfs}
        written_by: Dict[str, Tuple[Evaluator, bool]] = {}
        for evaluator in self.evaluators:
            for name, value in evaluator.compute_group(frames).items():
                several = isinstance(value, Mapping)
                for key, part in value.items() if several else ((None, value),):
                    column = evaluator.column_name(key) if several else evaluator.name
                    _claim_column(written_by, column, evaluator, several)
                    columns[name][column] = part
        return columns

    def _iter_ref(self, ref: ModelResultRef, cols: List[str] = None) -> Iterator[Dict]:
        return ref.iter_records(cols)

    def compute_all(self) -> Dict[str, pd.DataFrame]:
        model_dfs = self._model_frames()
        result: Dict[str, pd.DataFrame] = {"summary": self._summary_table(model_dfs)}
        for config_name, group_keys in GROUP_CONFIGS:
            result[config_name] = self._grouped_table(model_dfs, group_keys)
        return result

    def _model_frames(self) -> Dict[str, pd.DataFrame]:
        """One DataFrame per model, read once: `FORECAST_KEYS`, `prediction` and `actual`.

        This is what evaluators receive (whole, or cut into groups). `run_date` and `target_date` are
        datetime64 and `hour` / `horizon` integers, which keeps the frames compact. A key that no record of
        the model carries is left out of its frame; evaluators that need it (rMAE) say so. The rows are in
        `RECORD_ORDER`, not in the order the records were written, so every sum an evaluator takes is the same
        however the workers happened to finish.
        """
        return {name: self._cached_frame(name, ref) for name, ref in self.refs.items()}

    def _cached_frame(self, name: str, ref: ModelResultRef) -> pd.DataFrame:
        """`_model_frame(ref)`, read from a results file only when the file changed: parsing the JSONL is the slowest
        step of a report, and a comparison session calls several of its methods. A Parquet directory changes with its catalog,
        which every committed part appends to (`results.store.fingerprint_path`). Results held in memory are always read
        again, since nothing tells when a list changed."""
        path = None if ref.path is None else Path(ref.path)
        if path is None or not path.exists():
            return self._model_frame(ref)
        status = fingerprint_path(path).stat()
        fingerprint = (
            str(path),
            status.st_mtime_ns,
            status.st_size,
            ref.test_start,
            ref.test_end,
            ref.window,
            ref.target_before,
            str(ref.zone),
        )
        cached = self._read_frames.get(name)
        if cached is None or cached[0] != fingerprint:
            cached = (fingerprint, self._model_frame(ref))
            self._read_frames[name] = cached
        # A deep copy: an evaluator that edits its frame in place must not change what the next table reads.
        return cached[1].copy()

    def _model_frame(self, ref: ModelResultRef) -> pd.DataFrame:
        columns = [*FORECAST_KEYS, "prediction", "actual"]
        read = ref.frame(columns) if type(self)._iter_ref is EvaluationReport._iter_ref else None
        if read is not None and len(read):  # Parquet: the frame the records below would make, without a dict per record
            keep = [c for c in columns if c in ("prediction", "actual") or read[c].notna().any()]
            return _in_record_order(read[keep])
        values: Dict[str, list] = {c: [] for c in columns}
        for r in self._iter_ref(ref, cols=columns):
            for c in columns:
                values[c].append(r.get(c))
        frame: Dict[str, Union[list, pd.Series]] = {}
        for c in columns:
            if c in ("prediction", "actual"):
                frame[c] = values[c]
            elif any(v is not None for v in values[c]):
                frame[c] = (
                    pd.to_datetime(pd.Series(values[c], dtype="object"), format="ISO8601")
                    if c.endswith("_date")
                    else values[c]
                )
        return _in_record_order(pd.DataFrame(frame))

    def _summary_table(self, model_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        self._check_set_evaluators()
        model_dfs = self._with_source_columns(model_dfs)
        frames = ModelFrames(model_dfs, freqs=self._result_freqs())
        metrics = self._apply_evaluators(frames)
        table = pd.DataFrame(
            [{"model": name, "forecasts": len(df), **metrics.get(name, {})} for name, df in model_dfs.items()]
        )
        by_evaluator = {e.name: e.clones(frames) for e in self.evaluators if isinstance(e, SetEvaluator)}
        if by_evaluator:  # `DM` and `MCS` read them on other keys, so they needn't agree
            table.attrs["clones"] = by_evaluator
        self._flag_dm_vintages(table, list(model_dfs), stacklevel=3)
        return table

    def _grouped_table(
        self,
        model_dfs: Dict[str, pd.DataFrame],
        group_keys: List[str],
        labels: Optional[Mapping[str, Callable[[pd.DataFrame], pd.Series]]] = None,
    ) -> pd.DataFrame:
        """One row per (group, model): each model's evaluators run on its own rows of the group, with every
        model's rows of that group as `model_dfs`. Rows are ordered by group, then by model.

        `forecasts` is each model's number of forecasts in the group, after the group keys. A model with none in a group
        has no row there; when the models of a group differ in their counts (a model without a row counts 0) the table
        warns and `table.attrs["unequal_counts"]` lists those groups.

        `labels` gives a group key that is not a column of the frames or one `_split` knows: a function of a model's frame
        to the group of each row."""
        self._check_set_evaluators()
        model_dfs = self._with_source_columns(model_dfs)
        freqs = self._result_freqs()
        groups = {name: dict(self._split(df, group_keys, labels)) for name, df in model_dfs.items()}
        rows = []
        for key in sorted({k for g in groups.values() for k in g}):
            group_dfs = {name: g[key] for name, g in groups.items() if key in g}
            metrics = self._apply_evaluators(group_dfs, group_keys, freqs)
            for name, group in group_dfs.items():
                row = {"model": name, **dict(zip(group_keys, key)), "forecasts": len(group)}
                rows.append({**row, **metrics.get(name, {})})
        table = pd.DataFrame(rows)
        _flag_unequal_counts(table, group_keys, list(model_dfs))
        self._flag_dm_vintages(table, list(model_dfs), stacklevel=3)
        return table

    def _with_source_columns(self, model_dfs: Mapping[str, pd.DataFrame]) -> Mapping[str, pd.DataFrame]:
        """The frames with the source-data columns that evaluators read (`ScaledMAEEvaluator`), joined at each forecast's
        target date and period; the frames themselves when no evaluator reads any."""
        readers = [e for e in self.evaluators if _source_columns_of(e)]
        if not readers:
            return model_dfs
        columns = list(dict.fromkeys(c for e in readers for c in _source_columns_of(e)))
        names = [type(e).__name__ for e in readers]
        if self.source_data is None:
            raise ValueError(
                f"{names} read {columns} from the report's source data, and this report has none: build it with "
                "source_data (a report of ModelPipeline has it; one from aggregate() has none)."
            )
        clash = [c for c in columns if c in (*FORECAST_KEYS, "prediction", "actual")]
        if clash:
            raise ValueError(f"{names} read the source column(s) {clash}, whose names the forecast frames use.")
        freq = self._require_one_frequency("Joining source-data columns") or HOURLY
        table = self._source_table(columns, model_dfs, freq)
        joined = {}
        for name, df in model_dfs.items():
            if len(df) and {"target_date", "hour"} <= set(df):
                values = source_values(table, df)
                df = df.assign(**{c: values[c] for c in columns})
            joined[name] = df
        return joined

    @staticmethod
    def _split(
        df: pd.DataFrame,
        group_keys: List[str],
        labels: Optional[Mapping[str, Callable[[pd.DataFrame], pd.Series]]] = None,
    ) -> Iterator[Tuple[tuple, pd.DataFrame]]:
        """Cut one model's frame by `group_keys`. Rows without a value for a key take no part in the grouping."""
        columns = []
        for k in group_keys:
            if labels and k in labels:
                column = labels[k](df)
            elif k == "year":
                column = df["target_date"].dt.year if "target_date" in df else None
            elif k == "quarter":
                column = df["target_date"].dt.quarter if "target_date" in df else None
            elif k == "run_weekday":
                column = df["run_date"].dt.day_of_week if "run_date" in df else None
            elif k == "target_weekday":
                column = df["target_date"].dt.day_of_week if "target_date" in df else None
            else:
                column = df[k] if k in df else None
            if column is None:
                return
            columns.append(column.rename(k))
        for key, part in df.groupby(columns if len(columns) > 1 else columns[0], sort=False):
            key = key if isinstance(key, tuple) else (key,)
            yield tuple(v.item() if hasattr(v, "item") else v for v in key), part


_PAIR_COLUMNS = (
    "model",
    "benchmark",
    "horizon",
    "n_days",
    "n_forecasts",
    "mae",
    "mae_benchmark",
    "ratio",
    "ci_low",
    "ci_high",
    "statistic",
    "p_value",
)


def _pair_row(
    model: str, benchmark: str, horizon: Union[int, str], pair: PairComparison, **extra: Any
) -> Dict[str, Any]:
    return {
        "model": model,
        "benchmark": benchmark,
        "horizon": horizon,
        **extra,
        "n_days": pair.n_days,
        "n_forecasts": pair.n_forecasts,
        "mae": pair.mae,
        "mae_benchmark": pair.mae_benchmark,
        "ratio": pair.ratio,
        "ci_low": pair.ci_low,
        "ci_high": pair.ci_high,
        "statistic": pair.statistic,
        "p_value": pair.p_value,
        "mde": pair.mde,
    }


def _period_of(ref: ModelResultRef) -> Optional[Tuple[str, str, Optional[str]]]:
    """(first, last issue day, target cut) of what a ref reads, which `aggregate()` keeps in its windows; None if unknown."""
    if ref.windows is not None and ref.window is not None:
        return (*getattr(ref.windows, ref.window), ref.target_before)
    return (ref.test_start, ref.test_end, ref.target_before) if ref.test_start else None


def _period_records(
    rows: pd.DataFrame, start: pd.Timestamp, window: Optional[str] = None, freq: str = DAILY
) -> List[Dict[str, Any]]:
    """Result records for the daily or hourly forecasts of `aggregate()`; `day_in_test` counts days from the earliest run
    date, and `hour` is the period of the new frequency (0 for a day)."""
    if rows.empty:
        return []
    run, target = pd.to_datetime(rows["run_date"]), pd.to_datetime(rows["target_date"])
    columns = (
        run.dt.strftime("%Y-%m-%d").tolist(),
        target.dt.strftime("%Y-%m-%d").tolist(),
        run.dt.day_of_week.tolist(),
        rows["hour"].astype(np.int64).tolist(),
        rows["horizon"].tolist(),
        (run - start).dt.days.tolist(),
        rows["prediction"].tolist(),
        rows["actual"].tolist(),
    )
    return [
        {
            "run_date": run_date,
            "target_date": target_date,
            "run_weekday": run_weekday,
            "hour": hour,
            "horizon": horizon,
            "day_in_test": day_in_test,
            "prediction": prediction,
            "actual": actual,
            "freq": freq,
            **({} if window is None else {"window": window}),
        }
        for run_date, target_date, run_weekday, hour, horizon, day_in_test, prediction, actual in zip(*columns)
    ]


def _source_columns_of(evaluator: Evaluator) -> Tuple[str, ...]:
    """The source-data columns an evaluator reads from its frames (`ScaledMAEEvaluator`), which the report joins first."""
    return tuple(getattr(evaluator, "source_columns", ()))


def _describe(obj: Any) -> str:
    """`Class(public attributes)` of an evaluator and what it holds, to tell whether two reports use the same ones."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return repr(obj)
    if isinstance(obj, (list, tuple)):
        return "[" + ", ".join(_describe(v) for v in obj) + "]"
    if isinstance(obj, Mapping):
        return "{" + ", ".join(f"{k!r}: {_describe(v)}" for k, v in obj.items()) + "}"
    attributes = {k: v for k, v in getattr(obj, "__dict__", {}).items() if not k.startswith("_")}
    return f"{type(obj).__name__}({', '.join(f'{k}={_describe(v)}' for k, v in attributes.items())})"


def _set_column(alpha: float) -> str:
    """The table column of the `1 - alpha` confidence set: `in_90` for 0.10."""
    return f"in_{100 * (1 - alpha):.3g}"


def _too_few_days(compared: SetComparison) -> str:
    return (
        f"These models share only {compared.daily.n_days} day(s) of forecasts, too few to compare (models short of "
        f"forecasts: {compared.limited_by}): the result columns are NaN."
    )


def _flag_test_length(table: pd.DataFrame, n_days: int, span_days: Optional[int] = None) -> None:
    """Warn on a very short comparison and leave a note in `table.attrs` on one shorter than a year, by its calendar span
    (`span_days`, `n_days` if not given): 104 Fridays are two years of seasons, though `n_days` is 104."""
    note = short_test_message(n_days, span_days)
    if note is None:
        return
    table.attrs["note"] = note
    if (n_days if span_days is None else span_days) < SHORT_TEST_DAYS:
        warnings.warn(f"{note} Below {SHORT_TEST_DAYS} days read it as indicative only.", stacklevel=3)
    else:
        logger.info(note)


def _flag_unequal_counts(table: pd.DataFrame, group_keys: Sequence[str], models: Sequence[str]) -> None:
    """Say so when the models of a group of `table` have different numbers of forecasts: their metrics there are on
    different samples. A model with no row in a group counts 0 (it has no forecast there), so a model that is missing from
    a group where another has forecasts is as much a difference as a model with fewer. The groups go to
    `table.attrs["unequal_counts"]` (a table without any gets no attribute). The warning shows the group with the widest gap
    and says how many groups differ; it is raised by the public method that built the table (`stacklevel=4`)."""
    if len(models) < 2 or table.empty:
        return
    wide = table.pivot(index=list(group_keys), columns="model", values="forecasts").reindex(columns=list(models))
    wide = wide.fillna(0).astype(int)
    spread = wide.max(axis=1) - wide.min(axis=1)
    uneven = wide[spread > 0]
    if uneven.empty:
        return
    table.attrs["unequal_counts"] = [key if isinstance(key, tuple) else (key,) for key in uneven.index]
    widest = spread[spread > 0].idxmax()
    widest = widest if isinstance(widest, tuple) else (widest,)
    where = ", ".join(f"{k}={v}" for k, v in zip(group_keys, widest))
    counts = ", ".join(f"{model}: {n}" for model, n in wide.loc[widest if len(widest) > 1 else widest[0]].items())
    warnings.warn(
        f"The models have different numbers of forecasts in {len(uneven)} of {len(wide)} groups by {', '.join(group_keys)}, "
        f"so their metrics there are on different samples (widest gap at {where}: {counts}). Relative metrics and tests "
        "use the keys a pair shares, MAE and RMSE do not; see report.forecast_counts() and "
        'df.attrs["unequal_counts"].',
        stacklevel=4,
    )


def _flag_partial_days(table: pd.DataFrame, partial: Mapping[str, int]) -> None:
    """Say so when days of a comparison hold only part of their periods: `n_days` counts them like whole ones."""
    partial = {name: count for name, count in partial.items() if count}
    table.attrs["partial_days"] = partial
    if partial:
        warnings.warn(
            f"Days with only part of their periods in common (by model or set): {partial}. The DM test weighs every day "
            "the same and the ratio every forecast, so read them together; see report.forecast_counts().",
            stacklevel=3,
        )


def _warn_on_identical_losses(losses: np.ndarray, names: Sequence[str]) -> None:
    columns: Dict[bytes, str] = {}
    for name, column in zip(names, losses.T):
        first = columns.setdefault(column.tobytes(), name)
        if first != name:
            warnings.warn(
                f"{first!r} and {name!r} have the same losses on every day: their ranks and chances of being best are "
                "decided by their order in the report.",
                stacklevel=3,
            )


def _claim_column(
    written_by: Dict[str, Tuple[Evaluator, bool]], column: str, evaluator: Evaluator, several: bool
) -> None:
    """Refuse a second evaluator on a column an evaluator with several columns wrote, or the other way round.

    Two evaluators of the same name and one value each (two `rMAEEvaluator`s) have always shared their column, and the
    last one won. That stays, so that such a report keeps its numbers, but no longer silently.
    """
    if column in _TABLE_KEYS:
        raise ValueError(f"An evaluator cannot write the column {column!r}: the report's tables use it.")
    first, first_several = written_by.setdefault(column, (evaluator, several))
    if first is evaluator:
        return
    if several or first_several:
        raise ValueError(
            f"Two evaluators write the column {column!r}. Keep one in the pipeline; for a second benchmark call "
            "report.tests(benchmark=...)."
        )
    warnings.warn(
        f"Two evaluators write the column {column!r}, and only the last one shows. Give them different names.",
        stacklevel=4,
    )
