"""One place for dates: which rows a forecast trains on, and which forecasts a run plans.

A forecast is a key (period, horizon h, issue day D). Everything here is about local calendar dates, because a period of a day
isn't always one row: the hour clocks repeat in autumn has two rows on its date, and the hour they skip in spring has none. So
rows are chosen by their date, never by counting them, and a key is planned only if the clock has its period on D and on D + h.

A period is on the clock of a date when any minute of it is: Lord Howe skips 02:00 to 02:29 in spring, so at 1 h its period 2 stays,
with its row at 02:30, and at 15 min periods 8 and 9 go. At `"1D"` a date has one period, 0, whose row is at one wall-clock time that
the clock has once on every date (`_contract.check_index`), so it is on the clock of every date, whatever the day's length.

Day ids are integers, one per local date, as `BaseModel._preprocess` numbers them. `zone` is what a frame's `index.tz` holds, or
the name of a zone, or None for a frame with no clock of its own (a naive index): such a frame, like a UTC one, has every period
on every date.
"""

import collections.abc
import datetime
import functools
import re
from typing import Any, Dict, FrozenSet, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .._date_utils import zone_name

# One row per local date, at one wall-clock time of day (`_contract.check_index`).
DAILY = "1D"
FREQ_STEPS = {"1h": pd.Timedelta(hours=1), "15min": pd.Timedelta(minutes=15), DAILY: pd.Timedelta(days=1)}


def periods_per_day(freq: str) -> int:
    """How many periods an ordinary day has at `freq`: 24, 96, or 1 at `"1D"`."""
    if freq not in FREQ_STEPS:
        raise ValueError(f"Unsupported freq '{freq}'. Supported: {list(FREQ_STEPS)}")
    return int(pd.Timedelta(days=1) / FREQ_STEPS[freq])


def training_span(anchor: Any, window: int, hz: int, newest: Any = -1) -> Tuple[Any, Any]:
    """The first and last row date a forecast from day id `anchor` at horizon `hz` may train on, both included.

    `newest` is the day of the newest training target relative to `anchor`, so with the default of -1 the last row is
    `anchor - 1 - hz`, whose target is dated `anchor - 1`, and the span holds `window` dates. No row is dated `anchor` or later,
    whatever `newest` says. `anchor` and `newest` may be arrays of day ids.
    """
    last = np.minimum(anchor + newest - hz, anchor - 1)
    return last - window + 1, last


def resolve_horizons(horizon: Any, horizons: Any, default: int) -> Tuple[Any, Tuple[int, ...]]:
    """(the largest horizon, every horizon planned, in order) of a run told `horizon=H`, which means 1 to H, or `horizons=`.

    `horizons` is any sequence of whole numbers from 1, such as `[1, 3, 7]` or `range(3, 10)`: it is sorted, and an empty one or
    one with a repeat raises. Giving both raises; giving neither is `horizon=default`. `range(1, H + 1)` is then exactly
    `horizon=H`. Whatever the list, targets are built for 1 to its largest, so `{target}_d+k` up to it stays refused as an input.
    """
    if horizons is None:
        horizon = default if horizon is None else horizon
        if isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError(f"horizon must be a whole number of days from 1, got {horizon!r}")
        return horizon, tuple(range(1, horizon + 1))
    if horizon is not None:
        raise ValueError(
            f"Pass horizon or horizons, not both: horizon={horizon!r} means every horizon from 1 to it, horizons={horizons!r} "
            "the ones it lists."
        )
    if isinstance(horizons, (str, bytes)) or not isinstance(horizons, collections.abc.Iterable):
        raise ValueError(
            f"horizons must be a sequence of whole numbers of days, such as [1, 3, 7] or range(3, 10), got {horizons!r}; "
            "horizon=H is every horizon from 1 to H"
        )
    listed = list(horizons)
    bad = [h for h in listed if isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer)) or h < 1]
    if bad:
        raise ValueError(f"horizons are whole numbers of days from 1, got {bad[:3]} in {listed}")
    if not listed:
        raise ValueError("horizons is empty: a run needs at least one horizon")
    found = [int(h) for h in listed]
    repeated = sorted({h for h in found if found.count(h) > 1})
    if repeated:
        raise ValueError(f"horizons lists {repeated} more than once: {found}")
    found.sort()
    return found[-1], tuple(found)


# The training window that holds every row from the frame's first local date (day id 0) up to a forecast's span's end.
EXPANDING = "expanding"


def resolve_training_window(value: Any) -> Tuple[Any, Tuple[Any, ...]]:
    """(the model's `training_window`, its members) of `training_window=value`: an int W from 1, `"expanding"`, or a list of
    them, whose members are fitted separately and averaged.

    A list is normalised like `horizons=`: ints ascending, then `"expanding"`; a repeat, an empty list, a bool, a float, another
    string or a value below 1 raises. A list of one is that one in everything (`[W]` is `W`), so the list form never moves a
    single-window run; any longer list stays a list, the form YAML and the manifest write.
    """
    listed = list(value) if isinstance(value, (list, tuple, range, np.ndarray)) else [value]
    if not listed:
        raise ValueError("training_window is empty: give an int from 1, 'expanding', or a list of them")
    found = []
    for member in listed:
        if isinstance(member, str) and member == EXPANDING:
            found.append(EXPANDING)
        elif isinstance(member, (bool, np.bool_)) or not isinstance(member, (int, np.integer)) or member < 1:
            raise ValueError(
                f"training_window takes whole numbers of days from 1 and 'expanding' (every row from the frame's first "
                f"date), or a list of them, such as [56, 84, 728, 'expanding'], got {member!r}"
                + ("" if len(listed) == 1 else f" in {listed}")
            )
        else:
            found.append(int(member))
    repeated = sorted({str(member) for member in found if found.count(member) > 1})
    if repeated:
        raise ValueError(f"training_window lists {repeated} more than once: {listed}")
    members = tuple(sorted(member for member in found if member != EXPANDING)) + (
        (EXPANDING,) if EXPANDING in found else ()
    )
    return (members[0] if len(members) == 1 else list(members)), members


def window_members(training_window: Any) -> Tuple[Any, ...]:
    """The members of a normalised `training_window` (what a manifest or a model holds): one for an int or `"expanding"`."""
    return tuple(training_window) if isinstance(training_window, list) else (training_window,)


def mask_members(training_window: Any) -> Tuple[Any, ...]:
    """What each binary-column mask of a model is kept under next to its (period, horizon): (None,) for one window, whose masks
    are keyed as they always were, else each member (`windows.mask_key(..., member)`)."""
    members = window_members(training_window)
    return (None,) if len(members) == 1 else members


def member_window(member: Any, day: Any) -> Any:
    """The int window of `training_span` that takes a member's rows for a fit on day id `day` (an array too): the member itself,
    or for `"expanding"` `day + 1`, whose span starts before day id 0, the frame's first local date, at every horizon and newest
    target, so that every row of the frame up to the span's end trains."""
    return day + 1 if isinstance(member, str) and member == EXPANDING else member


def every_horizon(horizon: Any, horizons: Sequence[int]) -> bool:
    """Whether `horizons` is every horizon from 1 to `horizon`, the run `horizon=H` plans."""
    return tuple(horizons) == tuple(range(1, horizon + 1))


def horizon_arguments(horizon: Any, horizons: Sequence[int]) -> Dict[str, Any]:
    """What to pass on to a `run(...)`: `horizon=H` for every horizon from 1 to H, as before `horizons` existed, else `horizons`."""
    return {"horizon": horizon} if every_horizon(horizon, horizons) else {"horizons": tuple(horizons)}


WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def resolve_issue_days(issue_days: Any) -> Optional[Tuple[int, ...]]:
    """The weekdays a run issues forecasts on, as sorted `date.weekday()` numbers (Monday 0), from `issue_days=`, a list of
    weekday names in any case such as `["friday"]`; None is every day, and so are all seven, which normalise to it.

    Names only: a number would be read as one of two weekday conventions. A bare string, an empty list, an unknown name or a
    repeated one raises.
    """
    if issue_days is None:
        return None
    if isinstance(issue_days, (str, bytes)) or not isinstance(issue_days, collections.abc.Iterable):
        raise ValueError(
            f"issue_days must be a list of weekday names, such as ['friday'] or ['monday', 'thursday'], got {issue_days!r}"
        )
    listed = list(issue_days)
    if not listed:
        raise ValueError("issue_days is empty: a run needs at least one weekday to issue forecasts on")
    found = []
    for name in listed:
        if not isinstance(name, str) or name.strip().lower() not in WEEKDAYS:
            raise ValueError(f"issue_days are weekday names, one of {list(WEEKDAYS)} in any case, got {name!r}")
        found.append(WEEKDAYS.index(name.strip().lower()))
    repeated = sorted({WEEKDAYS[day] for day in found if found.count(day) > 1})
    if repeated:
        raise ValueError(f"issue_days lists {repeated} more than once: {listed}")
    if len(found) == len(WEEKDAYS):
        return None
    return tuple(sorted(found))


def issue_day_names(weekdays: Optional[Iterable[int]]) -> Optional[List[str]]:
    """The names of these weekday numbers, in week order, as a manifest and YAML write them; None for every day."""
    return None if weekdays is None else [WEEKDAYS[day] for day in sorted(weekdays)]


def listed_days(days: Iterable[Any], weekdays: Optional[Iterable[int]]) -> List[Any]:
    """The days among `days` (dates, or ISO date strings) whose weekday is listed, in their order; all of them for None."""
    days = list(days)
    if weekdays is None:
        return days
    kept = frozenset(weekdays)
    return [day for day in days if _as_date(day).weekday() in kept]


def check_issue_days_in(start: Any, end: Any, weekdays: Optional[Iterable[int]], label: str = "test") -> None:
    """A ValueError if no day from `start` to `end` is one of `weekdays`: such a window would plan nothing."""
    if weekdays is None:
        return
    first, last = _as_date(start), _as_date(end)
    if not listed_days((first + datetime.timedelta(days=d) for d in range((last - first).days + 1)), weekdays):
        raise ValueError(
            f"The {label} period {first} to {last} has no {' or '.join(issue_day_names(weekdays))}, so with "
            f"issue_days={issue_day_names(weekdays)} nothing would be forecast in it"
        )


NEWEST_TARGETS = ("d-1", "published")


def check_newest_target(value: Any) -> str:
    """`value` if it is one of `NEWEST_TARGETS`, else a ValueError."""
    if not isinstance(value, str) or value not in NEWEST_TARGETS:
        raise ValueError(
            f"newest_target must be one of {list(NEWEST_TARGETS)}, got {value!r}: 'd-1' trains on targets up to the day "
            "before the issue day, 'published' on every target its rule says is known by the issue time"
        )
    return value


# Refit anchors count from here: a Monday, so that "7D" refits on Mondays. Never move it: every anchor would move with it.
_EPOCH = datetime.date(1970, 1, 5)
REFIT_EPOCH = _EPOCH.isoformat()
_REFIT_RULE = re.compile(r"([1-9][0-9]*)(D|MS)")

# `refit="auto"`: each training window refits on its own rule (`member_rules`), daily up to `AUTO_DAILY_UP_TO` days and weekly beyond
# (decision 16 of the step 4 spec). The rule reads the window alone: research Q5 found a weekly refit costs a short window 2.7 to
# 6.9 % and a long one at most 0.4 %. With [56, 84, 728, "expanding"] it makes (2 + 2/7) / 4 = 57 % of the fits; on lago() over DE
# the window average lost 0.08 % of its MAE at about 40 % of the compute (the short windows are about a third of a fit's cost there).
REFIT_AUTO = "auto"
AUTO_DAILY_UP_TO = 84
AUTO_SHORT_RULE, AUTO_LONG_RULE = "1D", "7D"


def check_refit(value: Any) -> Optional[str]:
    """`value` if it is None (a fit per forecast), `"kD"` or `"kMS"` with a whole k from 1, or `"auto"` (a rule per training
    window, `member_rules`), else a ValueError."""
    if value is None or value == REFIT_AUTO:
        return value
    if not isinstance(value, str) or _REFIT_RULE.fullmatch(value) is None:
        raise ValueError(
            f"refit must be None, 'kD' or 'kMS' with a whole k from 1, such as '7D' or '1MS', or 'auto', got {value!r}: None "
            "fits every forecast on its own, '7D' once every 7 days and '1MS' on the first of every month, counted from Monday "
            f"{REFIT_EPOCH}, and 'auto' a window of up to {AUTO_DAILY_UP_TO} days daily and a longer or 'expanding' one weekly"
        )
    return value


def member_rules(members: Sequence[Any], refit: Optional[str]) -> Tuple[Optional[str], ...]:
    """The refit rule of each training window of a model (`resolve_training_window`'s members), in their order: `refit` for every
    one of them, or with `"auto"` `"1D"` for a window of at most 84 days and `"7D"` for a longer one or `"expanding"`.

    Each member is then fitted on the anchors of its own rule, so a list such as `[56, 728]` fits its 56-day window every day and
    its 728-day window on Mondays, and a forecast averages the newest fit of each. A single window under `"auto"` is the model with
    its rule: `training_window=56, refit="auto"` forecasts as `refit="1D"` does, 728 as `"7D"`."""
    if refit != REFIT_AUTO:
        return tuple(refit for _ in members)
    return tuple(
        AUTO_SHORT_RULE if member != EXPANDING and member <= AUTO_DAILY_UP_TO else AUTO_LONG_RULE for member in members
    )


def _nests(fine: str, coarse: str) -> bool:
    """Whether every block of `fine` lies inside a block of `coarse`, so that each anchor of `coarse` is one of `fine`: `"1D"` in
    every rule, `"kD"` in `"KD"` and `"kMS"` in `"KMS"` when k divides K (both count from the same epoch)."""
    (k_fine, unit_fine), (k_coarse, unit_coarse) = (_REFIT_RULE.fullmatch(rule).groups() for rule in (fine, coarse))
    if unit_fine == unit_coarse:
        return int(k_coarse) % int(k_fine) == 0
    return unit_fine == "D" and int(k_fine) == 1


def block_rule(rules: Sequence[Optional[str]]) -> Optional[str]:
    """The coarsest of the members' rules (`member_rules`), whose blocks hold every other rule's: a pool job is one of its blocks,
    a record's `refit_anchor` its anchor (the oldest fit a forecast uses), and the rows a run reads go back from it. None without
    `refit`; a ValueError for rules whose blocks don't nest, which `"auto"` never gives."""
    distinct = sorted({rule for rule in rules if rule is not None})
    if not distinct:
        return None
    for coarse in distinct:
        if all(_nests(fine, coarse) for fine in distinct):
            return coarse
    raise ValueError(f"the refit rules {distinct} don't nest: no one of them holds the blocks of every other")


def newest_rule(rules: Sequence[Optional[str]]) -> Optional[str]:
    """The finest of the members' rules, whose anchor is the newest a forecast's fits are made on: the rows of its fits end there.
    None without `refit`."""
    distinct = sorted({rule for rule in rules if rule is not None})
    if not distinct:
        return None
    for fine in distinct:
        if all(_nests(fine, coarse) for coarse in distinct):
            return fine
    raise ValueError(f"the refit rules {distinct} don't nest: no one of them fits within every other")


def anchors_of_rules(days: Iterable[Any], rules: Sequence[Optional[str]]) -> List[datetime.date]:
    """The anchors of these issue days under any of `rules` (a model's `member_rules`), each once, in date order: every day a fit
    of a forecast on them is made on, so where its rows are cut. With one rule, `refit_anchors`."""
    days = list(days)
    return sorted({anchor for rule in dict.fromkeys(rules) for anchor in refit_anchors(days, rule)})


def refit_anchor(day: Any, refit: str) -> datetime.date:
    """The local date a forecast issued on `day` is fitted on, with `refit` `"kD"` or `"kMS"` (see `check_refit`).

    `"kD"`: the last date on or before `day` a whole number of k days from Monday 1970-01-05, so `"7D"` is the Monday on or before
    it and `"1D"` the day itself. `"kMS"`: the first of the last month on or before `day` a whole number of k months from January
    1970, so `"1MS"` is the first of its month, `"3MS"` January, April, July or October and `"12MS"` 1 January. It depends on the
    issue day alone, never on where a run starts, and a local date has no clock to change.
    """
    date = _as_date(day)
    k, unit = _REFIT_RULE.fullmatch(refit).groups()
    if unit == "D":
        return date - datetime.timedelta(days=(date - _EPOCH).days % int(k))
    months = (date.year - 1970) * 12 + date.month - 1
    first = months - months % int(k)
    return datetime.date(1970 + first // 12, first % 12 + 1, 1)


def refit_block_end(day: Any, refit: str) -> datetime.date:
    """The last issue day fitted on the anchor of `day`: the day before the next refit day."""
    anchor = refit_anchor(day, refit)
    k, unit = _REFIT_RULE.fullmatch(refit).groups()
    if unit == "D":
        return anchor + datetime.timedelta(days=int(k) - 1)
    months = (anchor.year - 1970) * 12 + anchor.month - 1 + int(k)
    return datetime.date(1970 + months // 12, months % 12 + 1, 1) - datetime.timedelta(days=1)


def refit_anchors(days: Iterable[Any], refit: str) -> List[datetime.date]:
    """The anchors of these issue days, each once, in date order."""
    return sorted({refit_anchor(day, refit) for day in days})


class FitUnit(NamedTuple):
    """One fit of a model with `refit`: every forecast of a period and horizon issued in one block, with one list.

    Its rows are `training_span(anchor, W, horizon, newest)`, with `newest` the newest target known by the issue time on the
    anchor (-1 with `newest_target="d-1"`), so the cut is relative to the refit day and nothing after it trains. `anchor` is a
    local date (`refit_anchor`), `list_id` the plan's id of the list (None for fixed lists). Each of its forecasts is predicted on
    its own from the one fit. `member` is the index of its training window in the model's members (`resolve_training_window`): a
    model with a list of windows fits one unit per member, and a forecast averages its members' units. With `refit="auto"` the
    anchor is that of the member's own rule (`member_rules`), so the members of one forecast can have different anchors.
    """

    period: int
    horizon: int
    anchor: datetime.date
    list_id: Optional[int]
    member: int = 0


class NewestTargets:
    """The newest training target of each forecast of a model with `newest_target="published"`.

    `k[period, i]` is the day of that target relative to the issue day `days[i]`, as `InformationSet.newest_targets` reads it
    from the target's rule, and the forecast (period, h, D) trains on the rows of `training_span(D, W, h, k)`. Days are local
    dates, looked up by their ordinal: day ids depend on where a frame starts, dates don't.
    """

    def __init__(self, days: Sequence[Any], k: np.ndarray):
        self.days = [_as_date(day) for day in days]
        self.k = np.asarray(k, dtype=np.int64)
        ordinals = np.array([day.toordinal() for day in self.days], dtype=np.int64)
        self._order = np.argsort(ordinals, kind="stable")
        self._ordinals = ordinals[self._order]

    @property
    def reach(self) -> int:
        """How many days before the first row of `training_span(D, W, h)` a span may start: more than none only where a target
        older than D-1 is the newest known."""
        return max(0, -1 - int(self.k.min())) if self.k.size else 0

    def at(self, period: int, ordinals: Any) -> Any:
        """k of `period` on these issue days, given by their date ordinals. A day that isn't one of `days` raises KeyError."""
        wanted = np.asarray(ordinals, dtype=np.int64)
        place = np.clip(np.searchsorted(self._ordinals, wanted), 0, max(len(self._ordinals) - 1, 0))
        if not len(self._ordinals) or (self._ordinals[place] != wanted).any():
            raise KeyError(f"no newest target was read for issue day ordinal(s) {wanted}")
        found = self.k[period, self._order[place]]
        return int(found) if np.ndim(found) == 0 else found

    def by_position(self, days: Optional[Sequence[Any]] = None) -> Dict[int, np.ndarray]:
        """period -> k on each of `days`, in their order: what a worker reads by a task's place among the issue days. `days`
        (dates) are some of the days this was read for, such as the listed weekdays of a run with `issue_days`; all of them if
        not given."""
        if days is None:
            return {period: self.k[period].copy() for period in range(self.k.shape[0])}
        ordinals = np.array([_as_date(day).toordinal() for day in days], dtype=np.int64)
        return {
            period: np.atleast_1d(np.asarray(self.at(period, ordinals))).copy() for period in range(self.k.shape[0])
        }


def _as_date(day: Any) -> datetime.date:
    return pd.Timestamp(day).date()


_DAY = pd.Timedelta(days=1).value
_MINUTE = pd.Timedelta(minutes=1).value


def _clock_gaps(
    first: datetime.date, last: datetime.date, freq: str, zone: Any
) -> Dict[datetime.date, Tuple[int, ...]]:
    """The periods the clock doesn't have on each date from `first` to `last`, for the dates that lack any: those with no minute
    on the clock."""
    if zone is None or last < first:
        return {}
    days = (last - first).days + 1
    periods = periods_per_day(freq)
    minutes_per_period = 1440 // periods
    midnights = pd.date_range(first, periods=days + 1, freq="D", unit="ns")
    # Only a date whose clock changes lasts other than 24 hours from its midnight to the next, so only those are read minute by
    # minute. A midnight the clock skips is read as the first minute after it, a repeated one as its first occurrence.
    starts = midnights.tz_localize(zone, nonexistent="shift_forward", ambiguous=np.ones(days + 1, dtype=bool))
    changing = np.flatnonzero(np.diff(starts.as_unit("ns").asi8) != _DAY)
    gaps: Dict[datetime.date, Tuple[int, ...]] = {}
    for day in changing.tolist():
        minutes = midnights.asi8[day] + np.arange(1440, dtype=np.int64) * _MINUTE
        wall = pd.DatetimeIndex(minutes.astype("datetime64[ns]"))
        exists = ~wall.tz_localize(zone, nonexistent="NaT", ambiguous=np.ones(1440, dtype=bool)).isna()
        skipped = np.flatnonzero(~exists.reshape(periods, minutes_per_period).any(axis=1))
        if len(skipped):
            gaps[first + datetime.timedelta(days=day)] = tuple(skipped.tolist())
    return gaps


def clock_periods(day: Any, freq: str, zone: Any) -> Tuple[int, ...]:
    """The period ids that exist on a local date, in order: the hour skipped in spring has none, and one repeated in autumn has one."""
    date = _as_date(day)
    skipped = _clock_gaps(date, date, freq, zone).get(date, ())
    return tuple(period for period in range(periods_per_day(freq)) if period not in skipped)


def skipped_periods(day: Any, freq: str, zone: Any) -> FrozenSet[int]:
    """The period ids a local date doesn't have on the clock: what `clock_periods` leaves out, for a look-up per record."""
    try:
        hash(zone)
    except TypeError:  # dateutil's zones, which are read by their name
        zone = zone_name(zone) or zone
        try:
            hash(zone)
        except TypeError:
            return _skipped_periods.__wrapped__(day, freq, zone)
    return _skipped_periods(day, freq, zone)


@functools.lru_cache(maxsize=4096)
def _skipped_periods(day: Any, freq: str, zone: Any) -> FrozenSet[int]:
    date = _as_date(day)
    return frozenset(_clock_gaps(date, date, freq, zone).get(date, ()))


def clock_periods_on(days: Iterable[Any], freq: str, zone: Any) -> Dict[pd.Timestamp, Tuple[int, ...]]:
    """`clock_periods` of each of these local dates, keyed by the date as a timestamp."""
    dates = pd.DatetimeIndex(pd.to_datetime(list(days))).unique().sort_values()
    if not len(dates):
        return {}
    gaps = _clock_gaps(dates[0].date(), dates[-1].date(), freq, zone)
    every = range(periods_per_day(freq))
    return {date: tuple(p for p in every if p not in gaps.get(date.date(), ())) for date in dates}


class IssueCalendar:
    """Which forecasts a run plans: every key whose period the clock has on its issue day and on its target day.

    A slot the clock doesn't have is no missing forecast: nothing could be issued from it, or scored against it. A real gap in the
    data is another matter: the slot is planned, and the run counts it as `missing_test_row` or the record has no `actual`.

    This is the one count of planned forecasts, so that `BaseModel.run`, `ModelPipeline.report` and the ensembles agree per window.
    `report` builds it from a manifest's `data_timezone`, with no data at hand. `zone` is as in the module docstring.

    `weekdays` are the weekdays a run issues on (`resolve_issue_days`), every day for None: a day of another weekday is no issue
    day, so its keys are neither planned nor missing.
    """

    def __init__(self, freq: str, zone: Any, horizons: Iterable[int], weekdays: Optional[Iterable[int]] = None):
        self.freq = freq
        self.zone = zone
        self.horizons = tuple(horizons)
        self.weekdays = None if weekdays is None else tuple(sorted(weekdays))
        self._periods = periods_per_day(freq)

    def issue_days(self, start: Any, end: Any) -> List[datetime.date]:
        """The issue days from `start` to `end`, both included: those of a listed weekday."""
        first, last = _as_date(start), _as_date(end)
        return listed_days(
            (first + datetime.timedelta(days=day) for day in range((last - first).days + 1)), self.weekdays
        )

    def _gaps(self, first: datetime.date, last: datetime.date) -> Dict[datetime.date, Tuple[int, ...]]:
        """The clock's gaps on these issue days and the days their forecasts are for."""
        return _clock_gaps(first, last + datetime.timedelta(days=max(self.horizons, default=0)), self.freq, self.zone)

    @staticmethod
    def _gone(gaps: Dict[datetime.date, Tuple[int, ...]], day: datetime.date, horizon: int) -> Set[int]:
        return {*gaps.get(day, ()), *gaps.get(day + datetime.timedelta(days=horizon), ())}

    def tasks(self, first_day: datetime.date, days: Sequence[int]) -> List[Tuple[int, int, int]]:
        """The planned `(period, horizon, day)` of these days, each a count of days from `first_day`, by day, then horizon, then
        period. A day of a weekday not listed has none."""
        if self.weekdays is not None:
            days = [day for day in days if (first_day + datetime.timedelta(days=day)).weekday() in self.weekdays]
        if not len(days):
            return []
        gaps = self._gaps(
            first_day + datetime.timedelta(days=min(days)), first_day + datetime.timedelta(days=max(days))
        )
        tasks = []
        for day in days:
            date = first_day + datetime.timedelta(days=day)
            for horizon in self.horizons:
                gone = self._gone(gaps, date, horizon)
                tasks.extend((period, horizon, day) for period in range(self._periods) if period not in gone)
        return tasks

    def count(self, start: Any, end: Any) -> int:
        """How many forecasts are planned on the issue days from `start` to `end`."""
        days = self.issue_days(start, end)
        if not days:
            return 0
        whole = len(days) * len(self.horizons) * self._periods
        gaps = self._gaps(days[0], days[-1])
        if not gaps:
            return whole
        return whole - sum(len(self._gone(gaps, day, horizon)) for day in days for horizon in self.horizons)

    def planned_cells(self, days: Sequence[datetime.date]) -> Dict[int, np.ndarray]:
        """horizon -> whether the key of each period and issue day is planned, as [period, position in days]."""
        grids = {horizon: np.ones((self._periods, len(days)), dtype=bool) for horizon in self.horizons}
        for period, horizon, position in self.unplanned(days):
            grids[horizon][period, position] = False
        return grids

    def unplanned(self, days: Sequence[datetime.date]) -> List[Tuple[int, int, int]]:
        """`(period, horizon, position in days)` of the keys on these issue days that the clock doesn't have."""
        if not len(days):
            return []
        gaps = self._gaps(min(days), max(days))
        if not gaps:
            return []
        return [
            (period, horizon, position)
            for position, day in enumerate(days)
            for horizon in self.horizons
            for period in sorted(self._gone(gaps, day, horizon))
        ]
