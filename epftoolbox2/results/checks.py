"""Checks before export (step 8f): `ForecastChecks`, what each check finds on a member's day, and records turned by them.

A **member-day** is one member's forecasts of one (issue day, horizon), so of one target day: the keys `(hour, horizon,
run_date)` of one `run_date` and `horizon`. `ForecastChecks` looks at the predictions of a member-day and at the clock alone,
never at an actual, so no check can leak an outcome into what it lets through:

- `periods`: every period the clock gives the day has a forecast. The periods are those `IssueCalendar` plans (`planned_periods`):
  the target day's, less a period the issue day lacks, which no row could be issued from. So a spring day has 23 hours (92
  quarter-hours), an autumn day 24 periods (96), the repeated hour being one period that covers its two occurrences (25 hours, 100
  quarter-hours: `day_products`), a normal day 24 (96), and a `"1D"` day one.
- `finite`: no prediction is NaN, infinite or null.
- `range=(low, high)`: no finite prediction below `low` or above `high` (both included; either may be None, no bound).
- `not_constant`: the finite predictions of a day of two or more periods aren't one value.
- `max_step`: no step between neighbouring periods of the day (in clock order, so 01:00 and 03:00 on a spring day) is larger than
  `max_step`.

A member-day failing any check has none of its forecasts kept: each becomes a skip record with status `check_failed` and the
failing checks in `detail` (`check_failed_record`), so a `FallbackEnsemble` moves to its next rung for that day and an average
counts it as `member_missing`. A key the member skipped for another reason keeps its own skip record. `check_records` does this
for records held in memory (a live revision, before it is written); a backtest does it as each member-day's jobs come back
(`models._checks.CheckGate`), so no failing forecast is ever written there.
"""

from __future__ import annotations

import datetime
import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import pandas as pd

from ..models._calendar import FREQ_STEPS, periods_per_day, skipped_periods
from .store import BACKTEST_REVISION, STATUS_OK, with_status

CHECK_FAILED = "check_failed"
# The checks in the order they are run and named in a record's `detail`.
CHECK_NAMES = ("periods", "finite", "range", "not_constant", "max_step")
# How many period ids a `detail` names before it says how many more there are.
_NAMED = 6


def _bound(value: Any, what: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"ForecastChecks: {what} must be a finite number or None, got {value!r}")
    return float(value)


def _number(value: float) -> str:
    return f"{value:.6g}"


def _ids(periods: Sequence[int]) -> str:
    shown = ", ".join(str(p) for p in periods[:_NAMED])
    return shown if len(periods) <= _NAMED else f"{shown} and {len(periods) - _NAMED} more"


class ForecastChecks:
    """What a member's forecasts of one target day must pass before they are exported, scored or served by a fallback.

    Set on a pipeline, `ModelPipeline(checks=ForecastChecks(...))`, they apply to every model of every run, backtests included
    (decision 13 of step 8), so a backtest counts each model's check failures (`forecast_counts()`) as a live run would. Its
    scores don't: a failed day has no forecast, so a model's MAE, and every comparison on common keys, is over the days that
    passed, conditional on passing, and a model's worst days can drop out of its own score (the comparison tables warn and
    hold `df.attrs["check_failed"]`). To compare models, score what is delivered (a `FallbackEnsemble`, `report.rungs()`) or
    run with `checks=None`. Without checks (the default) nothing is checked and nothing changes. They read predictions and the
    clock, never an actual.

    Args:
        periods: Every period the clock gives the day has a forecast: 23, 24 or 24 hours on spring, normal and autumn days (92, 96,
            96 quarter-hours; the repeated autumn hour is one period), one at `"1D"`; a forecast issued on a spring day lacks the
            skipped period on every target day, which is no failure (nothing could be issued from it).
        finite: No prediction is NaN, infinite or missing.
        range: `(low, high)`: every finite prediction within it, both bounds included. Either bound may be None. None checks no
            range.
        not_constant: The finite predictions of a day of two or more periods aren't all one value (a model stuck on its
            intercept, a frozen input).
        max_step: The largest step allowed between neighbouring periods of the day, in the target's units; None checks no steps.
    """

    def __init__(
        self,
        *,
        periods: bool = True,
        finite: bool = True,
        range: Optional[Sequence[Optional[float]]] = None,
        not_constant: bool = True,
        max_step: Optional[float] = None,
    ):
        for name, value in (("periods", periods), ("finite", finite), ("not_constant", not_constant)):
            if not isinstance(value, bool):
                raise TypeError(f"ForecastChecks: {name} must be True or False, got {value!r}")
        self.periods = periods
        self.finite = finite
        if range is not None:
            if isinstance(range, (str, bytes)) or not isinstance(range, Sequence) or len(range) != 2:
                raise ValueError(f"ForecastChecks: range must be (low, high), got {range!r}")
            low, high = _bound(range[0], "range's low"), _bound(range[1], "range's high")
            if low is not None and high is not None and not low < high:
                raise ValueError(f"ForecastChecks: range's low must be below its high, got {range!r}")
            if low is None and high is None:
                raise ValueError("ForecastChecks: range=(None, None) bounds nothing; leave range out")
            range = (low, high)
        self.range = range
        self.not_constant = not_constant
        max_step = _bound(max_step, "max_step")
        if max_step is not None and max_step <= 0:
            raise ValueError(f"ForecastChecks: max_step must be above 0, got {max_step!r}")
        self.max_step = max_step

    def to_dict(self) -> Dict[str, Any]:
        """The settings as plain values, as a manifest records them (`checks`) and `checks.json` names them."""
        return {
            "periods": self.periods,
            "finite": self.finite,
            "range": None if self.range is None else list(self.range),
            "not_constant": self.not_constant,
            "max_step": self.max_step,
        }

    @classmethod
    def from_dict(cls, settings: Mapping[str, Any]) -> "ForecastChecks":
        return cls(**dict(settings))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ForecastChecks) and self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash(repr(self))

    def __repr__(self) -> str:
        return "ForecastChecks(" + ", ".join(f"{key}={value!r}" for key, value in self.to_dict().items()) + ")"

    def failures(self, predictions: Mapping[int, Optional[float]], planned: Sequence[int]) -> List[Tuple[str, str]]:
        """The checks one member-day fails, as (name, what it found), in `CHECK_NAMES` order; empty when it passes.

        `predictions` holds the day's forecasts by period id (None for a forecast without a value), `planned` the periods the
        clock gives the day (`planned_periods`). A period outside `planned` is no period of the day and is read by no check."""
        order = sorted(set(int(p) for p in planned))
        held = [p for p in order if p in predictions]
        values = {p: (math.nan if predictions[p] is None else float(predictions[p])) for p in held}
        finite = [p for p in held if math.isfinite(values[p])]
        found: List[Tuple[str, str]] = []
        if self.periods and len(held) < len(order):
            missing = [p for p in order if p not in predictions]
            found.append(("periods", f"{len(held)} of {len(order)}, missing {_ids(missing)}"))
        if self.finite and len(finite) < len(held):
            bad = [p for p in held if not math.isfinite(values[p])]
            found.append(
                ("finite", f"{len(bad)} not finite ({', '.join(f'{p} at {values[p]}' for p in bad[:_NAMED])})")
            )
        if self.range is not None:
            low, high = self.range
            out = [
                p for p in finite if (low is not None and values[p] < low) or (high is not None and values[p] > high)
            ]
            if out:
                bounds = f"[{'-inf' if low is None else _number(low)}, {'inf' if high is None else _number(high)}]"
                shown = ", ".join(f"{p} at {_number(values[p])}" for p in out[:_NAMED])
                found.append(("range", f"{len(out)} outside {bounds} ({shown})"))
        if self.not_constant and len(finite) >= 2 and len({values[p] for p in finite}) == 1:
            found.append(("not_constant", f"all {len(finite)} at {_number(values[finite[0]])}"))
        if self.max_step is not None:
            steps = [
                (abs(values[b] - values[a]), a, b)
                for a, b in zip(order, order[1:])
                if a in values and b in values and math.isfinite(values[a]) and math.isfinite(values[b])
            ]
            over = [step for step in steps if step[0] > self.max_step]
            if over:
                step, a, b = max(over, key=lambda one: one[0])
                more = f" ({len(over)} steps above it)" if len(over) > 1 else ""
                found.append(
                    ("max_step", f"{_number(step)} from period {a} to {b}, above {_number(self.max_step)}{more}")
                )
        return found


def failure_detail(found: Sequence[Tuple[str, str]]) -> str:
    """The `detail` of a failing member-day's skip records: each failing check, `name: what it found`, joined by "; "."""
    return "; ".join(f"{name}: {what}" for name, what in found)


def _date(day: Any) -> datetime.date:
    return day if isinstance(day, datetime.date) else datetime.date.fromisoformat(str(day)[:10])


def planned_periods(freq: str, zone: Any) -> Callable[[Any, int], Tuple[int, ...]]:
    """The periods of a member-day, `(run_date, horizon) -> period ids`, as `IssueCalendar` plans them: those the clock has on
    the target day (`run_date + horizon`) and on the issue day. Every period without a zone."""
    periods = periods_per_day(freq)

    def planned(run_date: Any, horizon: int) -> Tuple[int, ...]:
        day = _date(run_date)
        gone = skipped_periods(day.isoformat(), freq, zone) | skipped_periods(
            (day + datetime.timedelta(days=int(horizon))).isoformat(), freq, zone
        )
        return tuple(period for period in range(periods) if period not in gone)

    return planned


def day_products(day: Any, freq: str, zone: Any) -> int:
    """How many market products (instants at `freq`) the clock gives a local date: 92, 96 or 100 quarter-hours on spring,
    normal and autumn days, 23, 24 or 25 hours, one at `"1D"`; every period without a zone."""
    if freq not in FREQ_STEPS:
        raise ValueError(f"Unsupported freq '{freq}'. Supported: {list(FREQ_STEPS)}")
    if zone is None or freq == "1D":
        return periods_per_day(freq)
    date = _date(day)
    midnights = pd.DatetimeIndex([pd.Timestamp(date), pd.Timestamp(date + datetime.timedelta(days=1))])
    start, end = midnights.tz_localize(zone, nonexistent="shift_forward", ambiguous=[True, True])
    return int((end - start) / FREQ_STEPS[freq])


def is_forecast(record: Mapping[str, Any]) -> bool:
    """Whether a record is a forecast (status "ok", or none: a record written before statuses), not a skip record."""
    return record.get("status", STATUS_OK) == STATUS_OK


def check_failed_record(record: Mapping[str, Any], detail: str) -> Dict[str, Any]:
    """The skip record that takes a failing forecast's place: its key fields in their order, `prediction` null, its `actual`,
    `coefficients` empty, `freq` and `window`, then status `check_failed`, its `list_hash` and `refit_anchor`, its `revision`
    and `detail`, as `models._worker.skip_record` lays out every skip record. A live store stamps its live keys after them."""
    skipped: Dict[str, Any] = {
        key: record[key]
        for key in ("run_date", "target_date", "run_weekday", "hour", "horizon", "day_in_test")
        if key in record
    }
    skipped.update(prediction=None, actual=record.get("actual"), coefficients=[], freq=record.get("freq"))
    if record.get("window") is not None:
        skipped["window"] = record["window"]
    with_status(skipped, CHECK_FAILED, record.get("list_hash"), record.get("refit_anchor"), detail)
    skipped["revision"] = record.get("revision", BACKTEST_REVISION)
    return skipped


class CheckOutcome(NamedTuple):
    """What `check_records` gives: each member's records with its failing member-days' forecasts replaced by `check_failed` skip
    records, the failing member-days (`findings`, what `checks.json` lists) and how many member-days were checked."""

    records: Dict[str, List[Dict[str, Any]]]
    findings: List[Dict[str, Any]]
    checked: int

    def to_dict(self, checks: ForecastChecks) -> Dict[str, Any]:
        """`checks.json` of a live revision: the settings, how many member-days were checked and each one that failed."""
        return {"checks": checks.to_dict(), "member_days": self.checked, "failed": list(self.findings)}


def check_records(
    records: Mapping[str, Iterable[Mapping[str, Any]]],
    checks: ForecastChecks,
    clock_periods: Callable[[Any, int], Sequence[int]],
    *,
    freq: Optional[str] = None,
    zone: Any = None,
) -> CheckOutcome:
    """Run `checks` over each member's records (member name -> its records) per member-day, `clock_periods(run_date, horizon)`
    giving the periods the clock gives that day (`planned_periods(freq, zone)`).

    Every forecast of a failing member-day is replaced in place by its `check_failed` skip record (`check_failed_record`); skip
    records and passing days stay as they are, in their order. A member-day without a forecast isn't checked. A finding is
    {member, run_date, target_date, horizon, periods (planned), forecasts, failed: {check: what it found}, detail}, and with
    `freq` given `products` (the target day's market products, `day_products`, so 100 on an autumn day at 15 minutes). Two
    forecasts of one key raise `ValueError`: which one would be delivered is undefined."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    findings: List[Dict[str, Any]] = []
    checked = 0
    for member, held in records.items():
        held = [dict(record) for record in held]
        days: Dict[Tuple[str, int], List[int]] = {}
        for at, record in enumerate(held):
            days.setdefault((str(record["run_date"]), int(record["horizon"])), []).append(at)
        for (run_date, horizon), places in days.items():
            forecasts: Dict[int, Optional[float]] = {}
            for at in places:
                if not is_forecast(held[at]):
                    continue
                period = int(held[at]["hour"])
                if period in forecasts:
                    raise ValueError(
                        f"{member}: two forecasts of period {period}, horizon {horizon}, run date {run_date}"
                    )
                forecasts[period] = held[at].get("prediction")
            if not forecasts:
                continue
            checked += 1
            planned = tuple(clock_periods(run_date, horizon))
            found = checks.failures(forecasts, planned)
            if not found:
                continue
            detail = failure_detail(found)
            for at in places:
                if is_forecast(held[at]):
                    held[at] = check_failed_record(held[at], detail)
            target = (_date(run_date) + datetime.timedelta(days=horizon)).isoformat()
            finding = {
                "member": member,
                "run_date": run_date,
                "target_date": target,
                "horizon": horizon,
                "periods": len(planned),
                **({} if freq is None else {"products": day_products(target, freq, zone)}),
                "forecasts": len(forecasts),
                "failed": dict(found),
                "detail": detail,
            }
            findings.append(finding)
        out[member] = held
    return CheckOutcome(out, findings, checked)
