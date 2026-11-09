"""When each input becomes known, and checks that a forecast only uses what is known when it is issued.

A forecast issued on day D at the issue time (e.g. 11:45 Brussels, before the 12:00 day-ahead gate closure) may
only use values published by then. Each base column gets an `Availability` rule; predictor names are parsed
(`x`, `x_d-k`, `x_d+k`, `x_h-k`, ...) to find the base column and the delivery time of the value they hold (a
DailyProfileTransformer column, `x_h05` or `x_h05_d-1`, is delivered in the slot it names, not at the row, and a
DailyStatsTransformer column, `x_day_max` or `x_day_max_d-1`, in the last slot of the day it aggregates); and
`InformationSet.check` evaluates every expanded predictor for every (period, horizon) before any model is fitted.
`leakage_probe` tests the same boundary empirically: it perturbs every value that is not yet known at issue time
and checks that the forecasts don't move. `positive_control` tests the probe: it perturbs the freshest values that
are known, and the forecasts must move, or the perturbation isn't reaching the model at all.

Times are compared as instants, in minutes elapsed since midnight of the issue day D in the data's time zone, so a
DST switch on D can't make a value known after the issue read as known before it. Rules give wall-clock times in
their own zone (a publication time, a delivery day), and day shifts keep the wall-clock time; delays, period lengths
and hour or minute shifts are elapsed time.
"""

from __future__ import annotations

import contextlib
import datetime
import fnmatch
import hashlib
import math
import re
import threading
import warnings
from dataclasses import dataclass, field, replace
from functools import cached_property, lru_cache
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._date_utils import daily_anchor, zone_name
from .logging import get_logger

logger = get_logger(__name__)

DEFAULT_ISSUE_TIME = "11:45"
KINDS = ("static", "realised", "published", "weekly", "weather_run", "as_of", "weather_cutoff")
# What the archive of a column holds (`Availability(vintage=...)`): `snapshot`, each value as published at a stated time (an
# auction, a forecast's day-ahead vintage); `row_stamp`, every value with a trustworthy stamp of its first publication;
# `last_write_stamp`, a stamp of its last write, which can tell a leak but is no vintage; `latest_revision`, the newest value
# only. A class decides no number: it says what a backtest read, which `report.vintages()` lists.
VINTAGES = ("snapshot", "row_stamp", "last_write_stamp", "latest_revision")
# The classes whose archive holds values written after they were first out: a backtest reads what no forecast saw then.
REVISED_VINTAGES = ("latest_revision", "last_write_stamp")
# How a rule without a class (`vintage=None`, any rule of yours that doesn't say) is shown.
UNKNOWN_VINTAGE = "unknown"
# When a value that no rule of its column holds for is known: never, so it is always late.
NEVER = "never (no rule holds then)"
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
# `Availability(week_zone=DATA_ZONE)`: a weekly rule's week is one of the data's local dates (each market's own week).
DATA_ZONE = "data"
_UNKNOWN_POLICIES = ("raise", "warn", "ignore")
# What a run does with an input known after the issue time: nothing, list it and go on, or refuse before any fit.
AVAILABILITY_LEVELS = ("none", "warn", "raise")
_SUFFIX = re.compile(r"^(?P<base>.+)_(?P<unit>d|h|min|s)(?P<sign>[+-])(?P<n>\d+)$")
_UNIT_MINUTES = {"d": 1440.0, "h": 60.0, "min": 1.0, "s": 1.0 / 60.0}
# A period's length in minutes. A daily row ("1D") is no slot: its `h` shifts and its profiles and aggregates are hourly,
# so its step is an hour, and its periods (one) come from `models._calendar.periods_per_day`, its time from its index.
_STEP_MINUTES = {"1h": 60, "15min": 15, "1D": 60}
_DAILY = "1D"
_EPOCH = pd.Timestamp("1970-01-01")  # days and minutes counted from it, in `_zone_table`
# DailyProfileTransformer's slot suffixes (`data.transformers._daily.slot_names`): `_h00`..`_h23` for hourly data,
# `_s00`..`_s95` for 15-minute data. No sign, so they never parse as a shift.
_PROFILE_SLOT = re.compile(r"^(?P<column>.+)_(?P<unit>[hs])(?P<n>\d{2})$")
_SLOT_MINUTES = {"h": 60, "s": 15}
# DailyStatsTransformer's suffixes (`data.transformers.daily_stats`): a statistic of the column over the row's whole
# local day. No sign, so they never parse as a shift. `is_day_mean` and `warsaw_is_day_max` are not such columns but
# Open-Meteo's `is_day` aggregated over locations (`_open_meteo_common.aggregated_variable_names`), whoever's rule
# reads them, so a column named `is` or `..._is` can't be told apart and is read as a plain column.
_DAY_AGGREGATE = re.compile(r"^(?P<column>.+)(?<!_is)(?<!^is)_day_(?:min|max|mean|last)$")


# The note of a use that is late on a training row, not on the issue day's.
_TRAINING_ROWS_HINT = (
    "a training row holds a value known only then (under rules over time an older value can come later than a newer one, and "
    "a value outside every rule's period is never known): shorten the training window, or start the frame where the rule "
    "begins, since no row before the frame's first date is read; loosen the rule only if the data really was known earlier"
)
_LATE_ADVICE = (
    "Use values known by then (e.g. `x_d-1` instead of `x`), or, if a rule is too strict for your data, "
    "override it with InformationSet(rules={...})."
)


def _parse_time(value: str) -> int:
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", str(value))
    if not match or int(match[1]) > 24 or int(match[2]) > 59:
        raise ValueError(f"Expected a time 'HH:MM', got '{value}'")
    return int(match[1]) * 60 + int(match[2])


def _minutes(value: str) -> int:
    return _parse_time(value)


def _parse_delay(value: str) -> float:
    try:
        return pd.Timedelta(value).total_seconds() / 60.0
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Expected a delay such as '1h' or '30min', got '{value}'") from exc


def _iso_day(value: Any, name: str) -> Optional[str]:
    """A validity bound as an ISO date, "YYYY-MM-DD": a calendar date, not an instant."""
    if value is None:
        return None
    try:
        day = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a date such as '2026-10-08', got {value!r}") from exc
    if day.tz is not None or day != day.normalize():
        raise ValueError(f"{name} is a calendar date such as '2026-10-08', not an instant, got {value!r}")
    return day.date().isoformat()


def _market_clock(tz: Any) -> str:
    """The name of the zone `issue_timezone="data"` issues in on a frame in zone `tz`: a region's (`Europe/Warsaw`), as
    `zone_name` gives it. A naive index, UTC, an `Etc/` zone or a fixed offset is no market's clock, and a forecast issued "at
    12:00" there would be issued at another hour in every market east or west of it: a ValueError says to name the zone."""
    if tz is None:
        raise ValueError(
            "issue_timezone='data' issues each forecast on the clock of its frame's zone, and this frame's index has none: "
            "localise it (df.tz_localize('Europe/Warsaw')) or name the issue zone (issue_timezone='Europe/Warsaw')."
        )
    name = zone_name(tz) or str(tz)
    if name == "UTC" or name.startswith("Etc/") or "/" not in name:
        raise ValueError(
            f"issue_timezone='data' issues each forecast on the clock of its frame's zone, and this frame is indexed in {name}, "
            "no market's clock (a frame from DataPipeline(timezone=None) is in UTC): name the issue zone "
            "(issue_timezone='Europe/Warsaw'), or index the frame in the market's zone (df.tz_convert('Europe/Warsaw'))."
        )
    return name


def _check_zone(zone: str) -> None:
    try:
        pd.Timestamp("2000-01-01", tz=zone)
    except Exception as exc:
        raise ValueError(f"Unknown time zone '{zone}'") from exc


class InformationSetError(ValueError):
    """A predictor or training target uses a value that is not known at issue time."""


def check_level(level: str) -> str:
    """`level` if it is one of `AVAILABILITY_LEVELS`; a `ValueError` otherwise."""
    if level not in AVAILABILITY_LEVELS:
        raise ValueError(f"availability must be one of {list(AVAILABILITY_LEVELS)}, got {level!r}")
    return level


# ---------------------------------------------------------------------------
# Availability rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Availability:
    """When the value of a base column for a delivery period becomes known.

    Kinds:
        static:      known in advance (calendar features).
        realised:    known `delay` after the end of the delivery period (actual load, generation).
        published:   published at `time` (in `timezone`) on the day `days_before` days before the delivery day
                     (day-ahead prices: 13:00, days_before=1; a daily close: its close time, days_before=0;
                     a value known only after its delivery day: "00:00", days_before=-1). The delivery day is a local
                     date of `day_zone`: None, `timezone`'s; "data", the data's own (on a frame in another zone than the
                     issue time's, the later of the data's date and the issue zone's, so it is never early); or a zone such
                     as "Europe/Helsinki", the zone whose dates another pipeline's forecasts are for (`ForecastSource`).
                     The publication instant stays `time` in `timezone`, on the date `days_before` before that day.
        weekly:      published on `weekday` at `time` (in `timezone`) for the whole following Monday-Sunday week
                     (week-ahead load). The week is one of local dates of `week_zone`: None, `timezone`'s; "data", the
                     data's own, so each market's Monday to Sunday on a frame in its zone (on a frame in another zone than
                     the issue time's, the later of the data's week and the issue zone's, so it is never early); or a zone
                     such as "Europe/Helsinki". The publication instant stays `time` in `timezone` whatever the week's zone.
        weather_run: a weather forecast column `x_d+k` from the Open-Meteo previous-runs API. The value at row t
                     comes from the model run started about at t (lead time k days), published `delay` later, so
                     availability depends on the row time, not on k.
        as_of:       known at a fixed `time` (in `timezone`) on the issue day D, regardless of a leading `_d+k`
                     (a value fixed as of the source's own issue time, e.g. a reanalysis snapshot rather than a
                     rolling forecast run). A shift after the leading `_d+k` moves the day: `x_d+1_d+1` holds
                     the value of D+1 and is known as of D+1.
        weather_cutoff: an OpenMeteoSource column. Known `delay` after `run`:00 UTC on the issue day D itself,
                     regardless of a leading `_d+k` — the source's own leak-free cutoff is fixed to D's run, not
                     to the delivery day (see OpenMeteoSource and `publication_delay_hours`). A shift after the
                     leading `_d+k` moves the day the run is read from, as for `as_of`. A run is a UTC hour, so
                     `timezone` is "UTC" (None means it too), and any other zone raises.

    A clock time of a rule (`time`) is a wall-clock time of `timezone`, read at the instant it denotes: one the clocks
    repeat at its second occurrence and one they skip at the switch, never earlier than it may be.

    `vintage` is the rule's class, what the archive of the column holds (one of `VINTAGES`; None says nothing, shown as
    "unknown"). It decides no number. `valid_from` and `valid_until` (ISO dates, both included; `.during()` sets them) say on
    which days the rule holds, for a lag that changed: a value whose day is outside the period isn't known by this rule, and a
    value no rule of its column holds for is never known. The day is the one the value is delivered on, a date of `timezone`
    for `published` (of `day_zone` where it is set), of the week's zone (`week_zone`) for `weekly`, and of the data's zone for `realised`, `weather_run`
    and `static`; for `as_of` and `weather_cutoff` it is the day the value is read on (the issue day, moved by a shift after a leading `_d+k`). A column's
    rules over time are a list in `InformationSet(rules={name: [rule, rule]})`, with disjoint periods (`RuleChain`).
    """

    kind: str
    delay: str = "0min"
    time: str = "00:00"
    days_before: int = 0
    weekday: str = "friday"
    timezone: Optional[str] = "Europe/Brussels"
    note: str = ""
    vintage: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    week_zone: Optional[str] = None
    day_zone: Optional[str] = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Unknown availability kind '{self.kind}'. Kinds: {list(KINDS)}")
        _parse_delay(self.delay)
        _parse_time(self.time)
        if self.weekday.lower() not in _WEEKDAYS:
            raise ValueError(f"Unknown weekday '{self.weekday}'")
        if self.kind == "weather_cutoff":
            # A model run's time is a UTC hour: read in any other zone, "06" plus a delay lands hours off what `describe` says.
            if self.timezone is None:
                object.__setattr__(self, "timezone", "UTC")
            elif self.timezone != "UTC":
                raise ValueError(
                    f"A weather_cutoff rule's run time is a UTC hour, so its timezone is 'UTC' (or None), got "
                    f"{self.timezone!r}. Use Availability.weather_cutoff(run, delay)."
                )
        if self.timezone is not None:
            _check_zone(self.timezone)
        if self.week_zone is not None:
            if self.kind != "weekly":
                raise ValueError(f"week_zone is a weekly rule's, not a {self.kind} rule's: {self.week_zone!r}")
            if self.week_zone != DATA_ZONE:
                _check_zone(self.week_zone)
        if self.day_zone is not None:
            if self.kind != "published":
                raise ValueError(f"day_zone is a published rule's, not a {self.kind} rule's: {self.day_zone!r}")
            if self.day_zone != DATA_ZONE:
                _check_zone(self.day_zone)
        if self.vintage is not None and self.vintage not in VINTAGES:
            raise ValueError(f"Unknown vintage class {self.vintage!r}. Classes: {list(VINTAGES)}, or None")
        object.__setattr__(self, "valid_from", _iso_day(self.valid_from, "valid_from"))
        object.__setattr__(self, "valid_until", _iso_day(self.valid_until, "valid_until"))
        if self.valid_from is not None and self.valid_until is not None and self.valid_from > self.valid_until:
            raise ValueError(f"valid_from {self.valid_from} is after valid_until {self.valid_until}")

    @classmethod
    def static(cls, note: str = "", vintage: Optional[str] = None) -> "Availability":
        return cls("static", note=note, vintage=vintage)

    @classmethod
    def realised(cls, delay: str = "1h", note: str = "", vintage: Optional[str] = None) -> "Availability":
        return cls("realised", delay=delay, note=note, vintage=vintage)

    @classmethod
    def published(
        cls,
        time: str,
        days_before: int = 1,
        timezone: Optional[str] = "Europe/Brussels",
        note: str = "",
        vintage: Optional[str] = None,
        day_zone: Optional[str] = None,
    ) -> "Availability":
        """Published at `time` in `timezone` on the date `days_before` before the delivery day, a local date of `day_zone`:
        `timezone`'s with None, the data's own with "data", or a named zone's (another pipeline's data zone, `ForecastSource`)."""
        return cls(
            "published",
            time=time,
            days_before=int(days_before),
            timezone=timezone,
            note=note,
            vintage=vintage,
            day_zone=day_zone,
        )

    @classmethod
    def daily_close(
        cls,
        time: str,
        timezone: Optional[str] = "Europe/Brussels",
        note: str = "",
        vintage: Optional[str] = None,
        day_zone: Optional[str] = None,
    ) -> "Availability":
        """A value for trading day X known at its close on X (e.g. fuel or carbon settlement prices); X a local date of
        `day_zone` (`published`)."""
        return cls.published(time, days_before=0, timezone=timezone, note=note, vintage=vintage, day_zone=day_zone)

    @classmethod
    def weekly(
        cls,
        weekday: str = "friday",
        time: str = "10:00",
        timezone: Optional[str] = "Europe/Brussels",
        note: str = "",
        vintage: Optional[str] = None,
        week_zone: Optional[str] = None,
    ) -> "Availability":
        """Published on `weekday` at `time` in `timezone` for the following Monday-Sunday week of `week_zone`'s local dates:
        `timezone`'s with None, the data's own with "data" (the built-in A31 rule's), or a named zone's."""
        return cls(
            "weekly",
            weekday=weekday.lower(),
            time=time,
            timezone=timezone,
            note=note,
            vintage=vintage,
            week_zone=week_zone,
        )

    @classmethod
    def weather_run(cls, delay: str = "6h", note: str = "", vintage: Optional[str] = None) -> "Availability":
        return cls("weather_run", delay=delay, note=note, vintage=vintage)

    @classmethod
    def as_of(
        cls, time: str, timezone: Optional[str] = "Europe/Brussels", note: str = "", vintage: Optional[str] = None
    ) -> "Availability":
        """Known at a fixed `time` on the issue day D; a leading `_d+k` doesn't change that."""
        return cls("as_of", time=time, timezone=timezone, note=note, vintage=vintage)

    @classmethod
    def weather_cutoff(cls, run: str, delay: str, note: str = "", vintage: Optional[str] = None) -> "Availability":
        """An OpenMeteoSource column: known `delay` after `run` (a UTC hour "HH") on the issue day D, regardless
        of a leading `_d+k` — like `as_of`, but the fixed time is expressed as a UTC run hour plus a delay that
        may push it past 24:00 (into the next UTC calendar day), rather than a local "HH:MM" clock time."""
        return cls("weather_cutoff", time=f"{run}:00", delay=delay, timezone="UTC", note=note, vintage=vintage)

    def during(self, valid_from: Any = None, valid_until: Any = None) -> "Availability":
        """This rule holding only on the days from `valid_from` to `valid_until` (dates, both included; None leaves that
        side open). A list of such rules with disjoint periods is a column's rule over time:
        `[Availability.published("14:30", -4).during(valid_until="2026-10-07"),
        Availability.published("14:30", -1).during(valid_from="2026-10-08")]`."""
        return replace(self, valid_from=valid_from, valid_until=valid_until)

    @property
    def bounded(self) -> bool:
        """Whether the rule holds on some days only."""
        return self.valid_from is not None or self.valid_until is not None

    def _holds_on(self, days: np.ndarray, origin: pd.Timestamp) -> np.ndarray:
        """Whether each of `days` (days after the naive midnight `origin`) is in the validity period."""
        days = np.asarray(days, dtype=float)
        inside = np.ones(days.shape, dtype=bool)
        if self.valid_from is not None:
            inside &= days >= (pd.Timestamp(self.valid_from) - origin) // pd.Timedelta(days=1)
        if self.valid_until is not None:
            inside &= days <= (pd.Timestamp(self.valid_until) - origin) // pd.Timedelta(days=1)
        return inside

    def to_dict(self) -> Dict[str, Any]:
        keys = {
            "static": [],
            "realised": ["delay"],
            "published": ["time", "days_before", "timezone"],
            "weekly": ["weekday", "time", "timezone"],
            "weather_run": ["delay"],
            "as_of": ["time", "timezone"],
            "weather_cutoff": ["time", "delay", "timezone"],
        }[self.kind]
        out: Dict[str, Any] = {"kind": self.kind}
        out.update({k: getattr(self, k) for k in keys})
        for name in (
            "week_zone",
            "day_zone",
            "vintage",
            "valid_from",
            "valid_until",
        ):  # only when set: a file without them reads as it always did
            if getattr(self, name) is not None:
                out[name] = getattr(self, name)
        if self.note:
            out["note"] = self.note
        return out

    @classmethod
    def from_dict(cls, data: Union[Dict[str, Any], "Availability"]) -> "Availability":
        if isinstance(data, Availability):
            return data
        if data.get("kind") == "weather_cutoff" and "timezone" not in data:
            data = {**data, "timezone": "UTC"}  # a run is a UTC hour, not the class's default zone
        return cls(**data)

    def describe(self) -> str:
        if not self.bounded:
            return self._describe_timing()
        if self.valid_from is None:
            period = f"to {self.valid_until}"
        elif self.valid_until is None:
            period = f"from {self.valid_from}"
        else:
            period = f"{self.valid_from} to {self.valid_until}"
        return f"{self._describe_timing()} (valid {period})"

    def _describe_timing(self) -> str:
        zone = f" {self.timezone}" if self.timezone else ""
        if self.kind == "static":
            return "known in advance"
        if self.kind == "realised":
            return f"known {self.delay} after the end of the period"
        if self.kind == "published":
            # The delivery day's zone where it isn't `timezone`'s; nothing without one, so every description stays as it was.
            days = {None: "", DATA_ZONE: " (delivery days of the data's local dates)"}.get(
                self.day_zone, f" (delivery days of {self.day_zone} dates)"
            )
            if self.days_before == 0:
                return f"published {self.time}{zone} on the delivery day{days}"
            if self.days_before > 0:
                return f"published {self.time}{zone} on D-{self.days_before} of delivery{days}"
            return f"published {self.time}{zone} {-self.days_before} day(s) after delivery{days}"
        if self.kind == "weekly":
            week = {None: "", DATA_ZONE: " of the data's local dates"}.get(
                self.week_zone, f" of {self.week_zone} dates"
            )
            return f"published {self.weekday.capitalize()} {self.time}{zone} for the following week{week}"
        if self.kind == "as_of":
            return f"known {self.time}{zone} on the issue day"
        if self.kind == "weather_cutoff":
            run = self.time.split(":", 1)[0]
            return f"known {self.delay} after the {run} UTC run on the issue day"
        return f"weather model run published {self.delay} after its start (about the row time)"


@dataclass(frozen=True)
class RuleChain:
    """The rules of one column over time, each holding on the days of its validity period (`Availability.during`). The periods
    are disjoint, and a value no rule holds for is never known, so it is late, conservatively. `InformationSet(rules={name:
    [rule, rule]})` makes one of a list, and it is written back as the list, in the order of the periods.

    A gap between two periods is allowed: the values of those days are never known. Two periods that share a day, or two rules
    open on the same side, raise `ValueError`.
    """

    rules: Tuple[Availability, ...]

    def __post_init__(self):
        rules = [Availability.from_dict(rule) for rule in self.rules]
        if not rules:
            raise ValueError("A list of availability rules needs at least one rule")
        ordered = tuple(sorted(rules, key=lambda rule: rule.valid_from or ""))
        for earlier, later in zip(ordered, ordered[1:]):
            if earlier.valid_until is None or later.valid_from is None or earlier.valid_until >= later.valid_from:
                raise ValueError(
                    "The rules of one column must hold on disjoint periods, so that one rule decides each value: "
                    f"'{earlier.describe()}' and '{later.describe()}' overlap. Give each its own days with "
                    ".during(valid_from=..., valid_until=...)."
                )
        object.__setattr__(self, "rules", ordered)

    @property
    def kind(self) -> str:
        """The kind of its rules if they share one, else "chain"."""
        kinds = {rule.kind for rule in self.rules}
        return kinds.pop() if len(kinds) == 1 else "chain"

    def to_dict(self) -> List[Dict[str, Any]]:
        return [rule.to_dict() for rule in self.rules]

    def describe(self) -> str:
        return "; ".join(rule.describe() for rule in self.rules)

    def __iter__(self):
        return iter(self.rules)

    def __len__(self) -> int:
        return len(self.rules)


Rule = Union[Availability, RuleChain]


def _as_rule(value: Any) -> Rule:
    """A rule of `InformationSet(rules=...)`: an `Availability`, a dict of one, or a list of them (a `RuleChain`)."""
    if isinstance(value, (Availability, RuleChain)):
        return value
    if isinstance(value, (list, tuple)):
        return RuleChain(tuple(value))
    return Availability.from_dict(value)


def rule_elements(rule: Rule) -> Tuple[Availability, ...]:
    """The rules of a `RuleChain`, or a rule on its own."""
    return rule.rules if isinstance(rule, RuleChain) else (rule,)


def vintage_of(rule: Optional[Rule]) -> str:
    """The class of a column's rule as shown: its `vintage`, "unknown" without one or without a rule, and the classes of a
    `RuleChain`'s rules in the order of their periods, each once, joined by ", "."""
    if rule is None:
        return UNKNOWN_VINTAGE
    classes = [one.vintage or UNKNOWN_VINTAGE for one in rule_elements(rule)]
    return ", ".join(dict.fromkeys(classes))


def over_time(rule: Optional[Rule]) -> bool:
    """Whether a column's rule changes over time: a `RuleChain`, or a rule with a validity period. Such a rule needn't make an
    older value known before a newer one (a lag that shortens, a gap, a `valid_from`), so the check reads the training rows and
    every slot of a day aggregate too; with any other rule a known value implies that every older one is."""
    return isinstance(rule, RuleChain) or (rule is not None and rule.bounded)


@dataclass(frozen=True)
class TrainingRows:
    """The rows a forecast's fit reads, for checking a column whose rule changes over time (`over_time`) on them as well as on
    the issue day's row: `InformationSet.check(..., training_rows=TrainingRows(window))` and `late_uses`, which a model run passes.

    At horizon h, relative to the issue day D, rows dated from D - window - h - reach to D - 1 - h, or to D - 1 when the newest
    target is read from the target's rule (`to_issue`, `newest_target="published"`); `reach` is how far a newest target lies
    before D - 1. Their targets are dated from D - window - reach to D - 1. With `refit` the forecast of D uses the fit of its
    refit day A on or before D, whose rows are those of a forecast issued on A: the same, `D - A` days earlier. Either way they
    are judged as of D's issue time, as `leakage_probe` judges them.

    `members` is for a model whose training windows refit on different rules (`refit="auto"`): each window's (window, rule), and
    the rows judged are those of each window's own fit, from its own refit day, and no row between them that no window reads.
    The training targets of a day are then those of the windows refitted on it, since a refit model's target is checked on its
    refit days. Without `members`, `window` and `refit` are the one span.

    Rows dated before `first_day`, the frame's first local date, don't exist, so no fit reads them and they aren't judged: a
    window that reaches before the frame (an `"expanding"` one always does, `BaseModel._reach_window`) is judged on the rows the
    frame has. Without `first_day` every row day of the window is judged.

    Args:
        window: The model's `training_window`.
        reach: Days a newest target lies before D - 1 at most (`NewestTargets.reach`, read on the refit days with `refit`).
        to_issue: Rows reach D - 1, as with `newest_target="published"`.
        refit: The model's `refit` (a rule), whose refit days move the rows back.
        first_day: The frame's first local date (ISO), before which nothing is judged; None judges the whole window.
        members: (window, rule) of each training window, where the windows refit on different rules; empty otherwise.
    """

    window: int
    reach: int = 0
    to_issue: bool = False
    refit: Optional[str] = None
    first_day: Optional[str] = None
    members: Tuple[Tuple[int, Optional[str]], ...] = ()

    def __post_init__(self):
        # Pairs as tuples, however given (a list of lists reads the same), so the rows compare and hash by value.
        object.__setattr__(self, "members", tuple(tuple(member) for member in self.members))

    def predictor_days(self, horizon: int, lag: int = 0, window: Optional[int] = None) -> range:
        """The row days, relative to D, of the training rows at `horizon`, of a fit made `lag` days before D (its refit day), of
        `window` days (the model's if not given)."""
        window = self.window if window is None else window
        return range(-(window + horizon + self.reach + lag), -lag if self.to_issue else -(horizon + lag))

    def lags(self, days: pd.DatetimeIndex, rule: Optional[str] = None) -> np.ndarray:
        """How many days before each of `days` its fit is made: 0, or with `refit` (or `rule`) the days since its refit day."""
        rule = self.refit if rule is None else rule
        if rule is None:
            return np.zeros(len(days), dtype=np.int64)
        from .models._calendar import refit_anchor  # models import this module

        return np.array([(day.date() - refit_anchor(day, rule)).days for day in days], dtype=np.int64)

    def latest(self, known: "_RowsKnown", horizon: int) -> np.ndarray:
        """When the training rows at `horizon` of the forecast of each [period, issue day] of `known`'s clock are all known, of
        the rows dated on or after `first_day`; with `members`, the rows of every window's own fit."""
        days = known.days
        floors = self.floors(days)
        out = np.full((known.size, len(days)), -np.inf)
        for window, rule in self.members or ((self.window, self.refit),):
            lags = self.lags(days, rule)
            for lag in np.unique(lags):
                on = lags == lag
                span = self.predictor_days(horizon, int(lag), window)
                found = (known.latest(span) if floors is None else known.latest_since(span, floors))[:, on]
                out[:, on] = np.maximum(out[:, on], found) if self.members else found
        return out

    def target_days(self, window: Optional[int] = None) -> range:
        """The days, relative to D, of the training targets before D - 1, the one the check reads anyway, of `window` days (the
        model's if not given)."""
        return range(-((self.window if window is None else window) + self.reach), -1)

    def target_latest(self, known: "_RowsKnown") -> np.ndarray:
        """When the training targets of `target_days` are all known, for each [period, issue day], of those dated on or after
        `first_day` (a target is a row's: one dated before the frame's first date has no row there). With `members`, the days
        are refit days, and a day's targets are those of the longest window refitted on it; a day no window is refitted on has
        none."""
        floors = self.floors(known.days)
        if not self.members:
            return (
                known.latest(self.target_days()) if floors is None else known.latest_since(self.target_days(), floors)
            )
        days = known.days
        windows = np.zeros(len(days), dtype=np.int64)
        for window, rule in self.members:
            refitted = self.lags(days, rule) == 0
            windows[refitted] = np.maximum(windows[refitted], window)
        out = np.full((known.size, len(days)), -np.inf)
        for window in np.unique(windows[windows > 0]).tolist():
            on = windows == window
            span = self.target_days(window)
            out[:, on] = (known.latest(span) if floors is None else known.latest_since(span, floors))[:, on]
        return out

    def floors(self, days: pd.DatetimeIndex) -> Optional[np.ndarray]:
        """The first row day, relative to each of `days`, that the frame has: `first_day - D`; None without `first_day`."""
        if self.first_day is None:
            return None
        first = datetime.date.fromisoformat(self.first_day)
        return np.array([(first - day.date()).days for day in days], dtype=np.int64)


def is_revised(rule: Optional[Rule]) -> bool:
    """Whether some rule of a column has a class whose archive holds revised values (`REVISED_VINTAGES`)."""
    return rule is not None and any(one.vintage in REVISED_VINTAGES for one in rule_elements(rule))


def rule_dict(rule: Rule, leave_out: Sequence[str] = ()) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """`rule.to_dict()` without the keys `leave_out`, a list of them for a `RuleChain`: what a manifest compares of a rule
    leaves out its note and its class, which decide no number."""
    if isinstance(rule, RuleChain):
        return [rule_dict(one, leave_out) for one in rule.rules]
    return {key: value for key, value in rule.to_dict().items() if key not in leave_out}


# Assumptions a user may accept explicitly. Each replaces the default rule for one pattern.
ASSUMPTIONS: Dict[str, Tuple[str, Availability, str]] = {
    "entsoe_a69_by_10": (
        "generation_*_forecast",
        Availability.published(
            "10:00",
            1,
            note="assumed: A69 wind/solar day-ahead forecast published by 10:00 D-1",
            vintage="latest_revision",
        ),
        "ENTSO-E A69 day-ahead wind and solar forecasts are published by 10:00 Brussels on the day before delivery, "
        "like the day-ahead load forecast. The legal deadline is 18:00 (Reg. 543/2013 Art. 14(2)(d)), after gate "
        "closure, and ENTSO-E gives no publication time per value.",
    ),
}


# The class of the ENTSO-E series whose archive holds the newest value only (A65, A69, A75).
_REVISED = "latest_revision"
_A31_NOTE = "A65/A31 week-ahead load forecast: Fridays for the next week (Art. 6(2)(c))"
# The week-ahead load forecast: one value per local day of each market's Monday-to-Sunday week, published on the Friday before
# at 10:00 Brussels, two hours before the 12:00 CET day-ahead gate closure, in every zone (Reg. 543/2013 Art. 6(2)(c)).
_A31 = Availability.weekly("friday", "10:00", note=_A31_NOTE, vintage=_REVISED, week_zone=DATA_ZONE)


_EXAA_NOTE = (
    "A44 position 2, the EXAA auction (gate 10:15 on D-1): legally within 1 h after the gate (Art. 12.1.d), so "
    "11:15; first seen about 10:35 in the one check made, not yet measured, so 11:15 is kept until it is"
)
_SWISS_NOTE = (
    "A44 for CH, the Swiss day-ahead auction (CH is not in SDAC; gate 11:00 on D-1): legally by 12:00; the practical "
    "first-seen time (about 11:05 to 11:30) is unmeasured, so measure first-seen times before issuing at 11:45"
)


def _exaa_rule(zone: str) -> Availability:
    return Availability.published("11:15", 1, timezone=zone, note=_EXAA_NOTE, vintage="snapshot")


_EXAA_RULE_VIENNA = _exaa_rule("Europe/Vienna")


def _entsoe_price_rules(source: Any, prefix: str) -> Dict[str, Availability]:
    """The rules `EntsoeSource` prices need beyond the built-in `price`: the EXAA column of AT and DE_LU, and CH's `price`,
    which is the Swiss auction that closes before SDAC (11:00, so published by 12:00 and not 13:00)."""
    rules: Dict[str, Availability] = {}
    if "price" not in source.type:
        return rules
    if getattr(source, "price_exaa", False):
        zone = "Europe/Vienna" if source._area_name == "AT" else "Europe/Berlin"  # DE_LU, DE_AT_LU
        rules[f"{prefix}price_exaa"] = _exaa_rule(zone)
    if source._area_name == "CH":
        rules[f"{prefix}price"] = Availability.published(
            "12:00", 1, timezone="Europe/Zurich", note=_SWISS_NOTE, vintage="snapshot"
        )
    return rules


# Energy-Charts (EnergyChartsSource). The vintages are the TSOs' own, matched to ENTSO-E's by name and by their errors against
# actuals (DE 2025, RMSE against the day-ahead forecast: solar intraday -13.8 %, current -29.1 %; wind onshore -22.7 %, -45.1 %),
# not from documentation. For a past date the API returns a separate, stable series per forecast type, but a stable answer today
# does not show that no value was rewritten after the time of its rule, so day-ahead and intraday are `latest_revision`, like
# ENTSO-E's TSO forecasts (`snapshot` needs a value saved at the rule's time that a refetch weeks later still matches). `current`
# is filled in a batch after the day, at a time nobody measured (live at 10:42 CET on 2026-11-08 it held no value for any of the
# day's quarter-hours, delivered ones included), and may be recomputed: no issue time knows it.
_EC_ZONE = "Europe/Berlin"
_EC_DAY_AHEAD = Availability.published(
    "18:00",
    1,
    _EC_ZONE,
    note="Energy-Charts day-ahead forecast: the TSOs' evening-before vintage, assumed out by 18:00 on D-1",
    vintage=_REVISED,
)
_EC_INTRADAY = Availability.published(
    "08:00",
    0,
    _EC_ZONE,
    note="Energy-Charts intraday forecast: the morning-of-delivery vintage, assumed out by 08:00 on the delivery day",
    vintage=_REVISED,
)
# A rule that holds on no day: every use of the column, at any shift, is "never (no rule holds then)", so it is refused.
_EC_CURRENT = Availability.published(
    "00:00",
    -1,
    _EC_ZONE,
    note="Energy-Charts current forecast: filled in a batch after the day, at an unmeasured time, and possibly recomputed; "
    "no issue time knows it, so no rule holds",
    vintage=_REVISED,
).during(valid_until="1900-01-01")
_EC_ACTUAL = Availability.realised(
    "1h", note="Energy-Charts actuals (public_power, cbet, cbpf): 1 h after the period, later revised", vintage=_REVISED
)
_EC_PRICE = Availability.published(
    "13:00",
    1,
    _EC_ZONE,
    note="Energy-Charts day-ahead price: SDAC results within about 1 h after the 12:00 gate closure on D-1",
    vintage="snapshot",
)
_EC_PRICE_CH = Availability.published(
    "12:00",
    1,
    _EC_ZONE,
    note="Energy-Charts CH price: Switzerland's own day-ahead auction (not SDAC), results around noon on D-1; "
    "the time the archive first held a value is unmeasured",
    vintage="snapshot",
)
_EC_NOT_SDAC = ("ME", "RS", "UA-IPS", "UA-BEI", "IE(SEM)")  # their own auctions: no rule here (CH has one of its own)
_EC_FORECASTS = {"day-ahead": _EC_DAY_AHEAD, "intraday": _EC_INTRADAY, "current": _EC_CURRENT}


def energy_charts_rules(
    prefix: str = "",
    forecast_types: Sequence[str] = tuple(_EC_FORECASTS),
    production_types: Sequence[str] = ("solar", "wind_onshore", "wind_offshore", "load"),
    zone: Optional[str] = None,
) -> Dict[str, Availability]:
    """The rules of the columns an `EnergyChartsSource` emits, under `prefix`: `{x}_forecast_day_ahead` (published 18:00 on
    D-1, `latest_revision`), `{x}_forecast_intraday` (08:00 on the delivery day), `{x}_forecast_current` (no rule holds on any day: never
    known, so refused at any shift), `actual_*`, `cbet_*` and `cbpf_*` (realised, 1 h), and `price_day_ahead` (13:00 on D-1; CH 12:00;
    none for a zone outside SDAC other than CH, which has to be given a rule)."""
    stem = f"{prefix}_" if prefix else ""
    rules: Dict[str, Availability] = {}
    for production in production_types:
        for forecast in forecast_types:
            if production != "load" or forecast == "day-ahead":
                rules[f"{stem}{production}_forecast_{forecast.replace('-', '_')}"] = _EC_FORECASTS[forecast]
    for name in ("actual_*", "cbet_*", "cbpf_*"):
        rules[f"{stem}{name}"] = _EC_ACTUAL
    if zone not in _EC_NOT_SDAC:
        rules[f"{stem}price_day_ahead"] = _EC_PRICE_CH if zone == "CH" else _EC_PRICE
    return rules


def default_rules() -> Dict[str, Availability]:
    """Built-in rules for the columns the library's ENTSO-E and calendar sources emit (unprefixed names).

    ENTSO-E publication times follow Reg. 543/2013 and SDAC practice; they are the latest times the data
    should be available, so a check against them is conservative.

    Weather columns have no rule here: a `*_temperature_2m`-style suffix alone doesn't say whether a column
    is a forecast or an observed value. `source_rules()` assigns one to exactly the columns an
    `OpenMeteoSource` emits.

    The dict is a copy, yours to change.
    """
    return dict(_built_in_rules())


@lru_cache(maxsize=1)
def _built_in_rules() -> Dict[str, Availability]:
    """What `default_rules` copies, built once: every resolution of a name reads it."""
    rules: Dict[str, Availability] = {
        # ENTSO-E
        # An auction's result is published once: what the archive holds is what was out.
        "price": Availability.published(
            "13:00",
            1,
            note="A44 day-ahead prices: SDAC results within about 1 h after the 12:00 gate closure on D-1",
            vintage="snapshot",
        ),
        # The EXAA auction (EntsoeSource(price_exaa=True), AT and DE_LU): see `_EXAA_RULE`.
        "price_exaa": _EXAA_RULE_VIENNA,
        # Prefixed, as AT and DE_LU need to be told apart (`de_price_exaa`): a bare InformationSet has no source rules.
        "*_price_exaa": _EXAA_RULE_VIENNA,
        # ENTSO-E keeps the newest value of these only: A65 was revised after the gate on 22 of 42 days measured.
        "load_actual": Availability.realised(
            "1h", note="A65/A16 actual load: 1 h after the period (Reg. 543/2013 Art. 6(2)(a))", vintage=_REVISED
        ),
        "load_forecast": Availability.published(
            "10:00",
            1,
            note="A65/A01 day-ahead load forecast: 2 h before gate closure on D-1 (Art. 6(2)(b))",
            vintage=_REVISED,
        ),
        # A31 holds one value per local day of the market's own week, published 2 h before the 12:00 CET gate closure in
        # every zone: the week is read in the data's local dates, the instant stays 10:00 Brussels.
        "load_forecast_daily_min": _A31,
        "load_forecast_daily_max": _A31,
        "generation_*_forecast": Availability.published(
            "18:00",
            1,
            note="A69 day-ahead wind/solar forecast: legally by 18:00 Brussels on D-1 (Art. 14(2)(d)); "
            "accept assumption 'entsoe_a69_by_10' to assume 10:00",
            vintage=_REVISED,
        ),
        "generation_*": Availability.realised(
            "1h", note="A75 actual generation per type: 1 h after the period", vintage=_REVISED
        ),
    }
    # No Energy-Charts names here: an EnergyChartsSource prefixes its columns ("ec_") so that none can take an ENTSO-E
    # rule, and its rules come from `source_rules` only. Outside a source-aware run its columns have no rule (refused).
    calendar = Availability.static(note="calendar (CalendarSource)")
    for name in ["is_holiday", "holiday_name", "weekday", "weekday_name", "month", "month_name"]:
        rules[name] = calendar
    for name in ["sunrise", "sunset", "daylight_hours"]:
        rules[name] = calendar
    for day in _WEEKDAYS:
        rules[f"is_{day}"] = calendar
    for month in [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]:
        rules[f"is_{month}"] = calendar
    rules["holiday_*"] = calendar
    rules["is_[0-9]*"] = calendar
    return rules


def source_rules(sources: Iterable[Any]) -> Dict[str, Availability]:
    """Rules for the columns a DataPipeline's sources emit, following their prefixes and column choices."""
    from .data.sources.benchmark import DATASETS, EpftoolboxBenchmarkSource
    from .data.sources.calendar import CalendarSource
    from .data.sources.energy_charts import EnergyChartsSource
    from .data.sources.entsoe import EntsoeSource
    from .data.sources.forecast import ForecastSource
    from .data.sources.open_meteo import OpenMeteoSource
    from .data.sources.pse import PseSource

    rules: Dict[str, Availability] = {}
    defaults = default_rules()
    for source in sources:
        if isinstance(source, CalendarSource) and getattr(source, "prefix", ""):
            prefix = source.prefix
            for key, rule in defaults.items():
                if rule.kind == "static":
                    rules[f"{prefix}{key}"] = rule
        elif isinstance(source, EntsoeSource):
            prefix = f"{source.prefix}_" if getattr(source, "prefix", "") else ""
            rules.update(_entsoe_price_rules(source, prefix))
            if "load" not in source.type:
                continue
            # A31 week-ahead daily values are stamped at the market's own local midnight (EntsoeSource's
            # bound zone, see `_daily_zone`), not necessarily this area's own when it is a neighbour combined into
            # another market's pipeline. The week is read in that zone, which holds on a frame in any zone (the
            # built-in rule reads the data's); the publication instant stays 10:00 Brussels, 2 h before the gate.
            rule = Availability.weekly(
                "friday",
                "10:00",
                note=f"{_A31_NOTE}, anchored to the market's own local day",
                vintage=_REVISED,
                week_zone=source._daily_zone(),
            )
            for name in ("load_forecast_daily_min", "load_forecast_daily_max"):
                rules[f"{prefix}{name}"] = rule
        elif isinstance(source, EnergyChartsSource):
            rules.update(
                energy_charts_rules(source.prefix, source._forecast_types, source._production_types, source._zone)
            )
        elif isinstance(source, OpenMeteoSource):
            rule = _weather_availability(source)
            for name in _weather_column_names(source):
                rules[name] = rule
        elif isinstance(source, PseSource):
            rules.update(_pse_rules(source))
        elif isinstance(source, ForecastSource):
            # Another pipeline's forecasts: known at its issue time on the day `horizon` before their target date, a date of its
            # data zone (`day_zone`), whatever the zone of the frame they are read on.
            rules[source.column] = source.availability()
        elif isinstance(source, EpftoolboxBenchmarkSource):
            # The files are local wall-clock time, and `index_timezone` only labels that clock (one hour off the true
            # instants in summer). So a row's delivery day is the day on that same clock: the file's 23:00 on D is
            # delivered on D, which Brussels rules would read as 00:00 on D+1 all summer.
            prefix = f"{source.prefix}_" if source.prefix else ""
            zone = source.index_timezone
            rules[f"{prefix}price"] = Availability.published(
                "13:00",
                1,
                timezone=zone,
                note="day-ahead price of the benchmark files, known the day before delivery",
                vintage="snapshot",  # an auction's result
            )
            for name in DATASETS[source.dataset]["exogenous"].values():
                rules[f"{prefix}{name}"] = Availability.published(
                    "10:00", 1, timezone=zone, note="day-ahead forecast of the benchmark files, available on d-1"
                )
    return rules


@dataclass(frozen=True)
class ChainedInput:
    """A column read from another pipeline's stored forecasts (`ForecastSource`), as `InformationSet.with_source_rules` registers
    it: what the downstream's checks (`models._chained`) and manifest (`chained_inputs`) read of its upstream.

    `late` are the inputs the upstream's leaf models listed as late or without a rule, `unchecked` the leaf models that ran at
    level "none": either makes every downstream use of the column a `late upstream` one. `digest` is the upstream manifest's
    `manifest_digest`."""

    column: str
    path: str
    model: str
    horizon: int
    issue_time: str
    issue_timezone: str
    data_timezone: str
    available: Optional[str]
    digest: str
    late: Tuple[str, ...] = ()
    unchecked: Tuple[str, ...] = ()
    freq: str = "1h"

    @property
    def late_upstream(self) -> bool:
        """Whether nothing vouches for the upstream's inputs at its issue time: some were late, or it didn't look."""
        return bool(self.late or self.unchecked)

    def record(self) -> Dict[str, Any]:
        """What a downstream manifest records of it (`chained_inputs[column]`), coverage aside."""
        return {
            "path": self.path,
            "model": self.model,
            "horizon": self.horizon,
            "issue_time": self.issue_time,
            "issue_timezone": self.issue_timezone,
            "data_timezone": self.data_timezone,
            "available": self.available,
            "digest": self.digest,
        }

    def describe(self) -> str:
        """`model`'s forecasts at `horizon` in `path`, for a table's rule column."""
        return f"forecasts of {self.model} (horizon {self.horizon}) in {self.path}"


def chained_sources(sources: Iterable[Any]) -> Dict[str, ChainedInput]:
    """Column -> its `ChainedInput`, for the `ForecastSource`s among `sources`."""
    from .data.sources.forecast import ForecastSource

    out: Dict[str, ChainedInput] = {}
    for source in sources:
        if isinstance(source, ForecastSource):
            out[source.column] = source.chained_input()
    return out


_PSE_ZONE = "Europe/Warsaw"
_PSE_STAMP = (
    "PSE `publication_ts` is a stamp of the last write, so the table can show a rewrite, not the first publication"
)


def _pse_rules(source: Any) -> Dict[str, Availability]:
    """The rules of a PseSource's columns, in Warsaw time (the local day of a PSE business date).

    A delivery day D is known: the `kse-load` forecast at 10:00 on D-1, like ENTSO-E's day-ahead load forecast (not measured:
    the conservative choice until first-seen times are); the day-ahead prices (`csdac-pln`, `rce-pln`) at 15:00 on D-1 (365
    days measured: median 13:47, 99th percentile 14:59, latest 18:00; never rewritten); `pdgsz` at the source's
    `pdgsz_cutoff` on D-1 (the source keeps the version stamped by then, so the rule is built from the parameter); the actuals
    (`kse-load`, `his-wlk-cal`, `przeplywy-mocy`) at 13:00 on D+1. Those tables are stamped by their last write (rows reloaded in
    bulk carry the reload time), so none has a vintage: `last_write_stamp`."""
    snapshots = {
        "csdac_pln": (
            "15:00",
            "PSE csdac-pln SDAC price in PLN: published on D-1 (latest 18:00 in a year), never rewritten",
        ),
        "rce_pln": (
            "15:00",
            "PSE rce-pln prosumer net-billing price, not the SDAC price: published on D-1, never rewritten",
        ),
    }
    rules: Dict[str, Availability] = {}
    for dataset in source.datasets:
        for column in source._columns_of(dataset):
            name = source._named(column)
            if dataset == "pdgsz":
                rules[name] = Availability.published(
                    source.pdgsz_cutoff,
                    1,
                    _PSE_ZONE,
                    note=f"PSE pdgsz: the version of an hour stamped by {source.pdgsz_cutoff} on D-1, as of that time; "
                    "an hour with none is NaN",
                    vintage="snapshot",
                )
            elif column in snapshots:
                time, note = snapshots[column]
                rules[name] = Availability.published(time, 1, _PSE_ZONE, note=note, vintage="snapshot")
            elif column == "load_fcst":
                rules[name] = Availability.published(
                    "10:00",
                    1,
                    _PSE_ZONE,
                    note=f"PSE kse-load day-ahead load forecast: assumed 10:00 on D-1 like ENTSO-E's (not measured). {_PSE_STAMP}",
                    vintage="last_write_stamp",
                )
            else:
                # The actuals, `his-wlk-cal` and the flows: known on D+1 at 13:00.
                rules[name] = Availability.published(
                    "13:00",
                    -1,
                    _PSE_ZONE,
                    note=f"PSE {dataset}: known about 13:00 on D+1. {_PSE_STAMP}",
                    vintage="last_write_stamp",
                )
    return rules


def _weather_column_names(source: Any) -> List[str]:
    """The unshifted column names an OpenMeteoSource emits (each is also emitted as `{name}_d+{i}`)."""
    from .data.sources._open_meteo_common import aggregated_variable_names

    prefix = f"{source.prefix}_" if getattr(source, "prefix", "") else ""
    # A source built with `locations` (see OpenMeteoSource.locations) emits aggregated columns
    # (`{prefix}{variable}_{kind}` per entry — see `aggregated_variable_names`) only when `agg` is set;
    # it also, or instead, emits per-location columns `{prefix}{name}_{variable}`, with `agg` only when
    # `keep_locations`, always otherwise (they are then the only output). Both column shapes follow the
    # same cutoff rule.
    locations = getattr(source, "_locations", [])
    agg = getattr(source, "_agg", None)
    if not locations:
        return [f"{prefix}{variable}" for variable in source.columns]
    names = [f"{prefix}{variable}" for variable in aggregated_variable_names(source.columns, agg)]
    if agg is None or getattr(source, "keep_locations", False):
        names += [f"{prefix}{location['name']}_{variable}" for location in locations for variable in source.columns]
    return names


def _source_market_zones(sources: Iterable[Any]) -> Dict[str, str]:
    """Rule key -> the market zone of the source whose rule it is, for rules that only mean something on an index
    in that zone. An OpenMeteoSource's cutoff is `run`:00 UTC of the local market day D, so the issue day D of a row
    is a day of the market's zone: on an index in another zone (UTC, say) a row can belong to another day D.
    Rules that carry their own zone and convert exactly (the week-ahead load rule, calendar columns) need none."""
    from .data.sources.open_meteo import OpenMeteoSource

    zones: Dict[str, str] = {}
    for source in sources:
        zone = getattr(source, "market_timezone", None)
        if isinstance(source, OpenMeteoSource) and zone:
            for name in _weather_column_names(source):
                zones[name] = zone
    return zones


def _weather_availability(source: Any) -> Availability:
    """An OpenMeteoSource column is available `delay` after `D` at `run`:00 UTC (the source's own leak-free
    cutoff, see OpenMeteoSource), regardless of `api` or a leading `_d+k`. `delay` is a conservative, model-side
    estimate of the real-world publication lag (see `publication_delay_hours`): the source itself has no notion
    of it, only of the nominal run instant.

    `run` and `model` are read with `getattr` and default when a source doesn't set them, so this works with
    any `OpenMeteoSource`-like object.
    """
    from .data.sources.open_meteo import publication_delay_hours

    run = getattr(source, "run", "00")
    delay_h = publication_delay_hours(getattr(source, "model", ""))
    return Availability.weather_cutoff(
        run,
        f"{delay_h}h",
        note="Open-Meteo: known after the model run's typical publication delay",
        vintage="snapshot",  # each issue day's own run, as the previous-runs archive holds it
    )


# ---------------------------------------------------------------------------
# Predictor names
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shift:
    """A suffix such as `_d-1` (unit "d", offset -1) or `_h-24`; positive offsets point to the future."""

    unit: str
    offset: int

    @property
    def minutes(self) -> float:
        return self.offset * _UNIT_MINUTES[self.unit]


@dataclass(frozen=True)
class ResolvedColumn:
    name: str
    base: str
    shifts: Tuple[Shift, ...]  # innermost (closest to the base name) first
    rule: Optional[Rule]  # an `Availability` or a `RuleChain`
    key: Optional[str] = None  # the rule's key (exact name or pattern)
    layer: Optional[str] = None
    built_from: Optional["ResolvedColumn"] = None  # what `base` was built from, if a transformer's feature (see `slot`)

    @property
    def slot(self) -> Optional[Tuple[int, int]]:
        """(start, length) in minutes of the DailyProfileTransformer slot that `base` names, or None.

        `price_h05` holds the price of 05:00-06:00 on its row's local day, the same value on every row of that day,
        so its availability is that of slot (300, 60), whatever the row's period. `_sNN` is a 15-minute slot.
        """
        return _profile_slot(self.base)

    @property
    def day_aggregate(self) -> bool:
        """Whether `base` names a DailyStatsTransformer column (`x_day_max`, `x_d-1_day_mean`).

        Such a column holds a statistic of its row's whole local day, the same value on every row of that day, so it is
        known when the last slot of that day is, whatever the row's period.
        """
        return _DAY_AGGREGATE.match(self.base) is not None

    @property
    def series(self) -> str:
        """The base column at the bottom of what this was built from: the one a rule would be declared for."""
        return self.base if self.built_from is None else self.built_from.series


def parse_predictor(name: str) -> List[Tuple[str, Tuple[Shift, ...]]]:
    """All readings of a column name as base + suffixes, from the full name to the fully stripped base.

    `price_d-1` -> [("price_d-1", ()), ("price", (d-1,))]; shifts are listed innermost first.
    """
    chain = [(name, ())]
    current, shifts = name, []
    while True:
        match = _SUFFIX.match(current)
        if not match:
            break
        sign = 1 if match["sign"] == "+" else -1
        shifts.insert(0, Shift(match["unit"], sign * int(match["n"])))
        current = match["base"]
        chain.append((current, tuple(shifts)))
    return chain


def _profile_slot(base: str) -> Optional[Tuple[int, int]]:
    """(start, length) in minutes of the slot of day a DailyProfileTransformer column `base` names, or None."""
    match = _PROFILE_SLOT.match(base)
    if match is None:
        return None
    length = _SLOT_MINUTES[match["unit"]]
    start = int(match["n"]) * length
    return (start, length) if start < 1440 else None


def _built_from_name(base: str) -> Optional[str]:
    """The column a DailyProfileTransformer or DailyStatsTransformer column `base` was built from, with its own shifts
    (`x_d-1` for `x_d-1_h05` and `x_d-1_day_max`), or None for a column that is neither."""
    match = _PROFILE_SLOT.match(base) if _profile_slot(base) is not None else _DAY_AGGREGATE.match(base)
    return None if match is None else match["column"]


def _find_rule(name: str, chain: List[Tuple[str, Tuple[Shift, ...]]], rules, layer: str) -> Optional[ResolvedColumn]:
    """The rule of one layer for a column: exact names from the full name down to the base, then glob patterns from the
    base up."""
    for candidate, shifts in chain:
        if candidate in rules:
            return ResolvedColumn(name, candidate, shifts, _as_rule(rules[candidate]), candidate, layer)
    patterns = [k for k in rules if any(ch in k for ch in "*?[")]
    for candidate, shifts in reversed(chain):
        for pattern in patterns:
            if fnmatch.fnmatchcase(candidate, pattern):
                return ResolvedColumn(name, candidate, shifts, _as_rule(rules[pattern]), pattern, layer)
    return None


# ---------------------------------------------------------------------------
# Information set
# ---------------------------------------------------------------------------

ROLE_PREDICTOR = "predictor"
ROLE_TARGET = "training target"
# How far back `InformationSet.newest_targets` looks for a known training target, in days relative to the issue day.
NEWEST_TARGET_FROM = -31
# Tables `newest_targets` computed, by what they were computed from, the oldest first, and the lock that guards them: models
# can be planned in threads, and a free-threaded build runs them at once.
_NEWEST_TABLES: Dict[Tuple[Any, ...], np.ndarray] = {}
_NEWEST_TABLES_KEPT = 16
_NEWEST_TABLES_LOCK = threading.Lock()
# The verdicts `violations` reached for one column (`InformationSet._late_of`), by everything they are computed from, the
# oldest first, and their lock: a sweep of variants checks the same columns on the same days once per variant. A verdict is
# a few late periods, so many are kept; 0 keeps none, and every check computes its own, as before (the tests compare the two).
_VERDICTS: Dict[Tuple[Any, ...], Tuple[Tuple[Any, ...], Tuple[Any, ...]]] = {}
_VERDICTS_KEPT = 65_536
_VERDICTS_LOCK = threading.Lock()
ROLE_VALUE = "ctx.value"
ROLE_PUBLISHED = "ctx.published"


@dataclass
class Violation:
    """One use of a value not known at the issue time (`available` > `issue`), or, for a column without a rule, one use
    that can't be judged (`column.rule` is None and both times are NaN)."""

    predictor: str
    period: int
    horizon: int
    column: ResolvedColumn
    available: float  # an instant: minutes elapsed since midnight of the issue day in the data's zone (see `_Context`)
    issue: float
    role: str = ROLE_PREDICTOR
    hint: str = ""
    # `available` and `issue` as shown: the data zone's wall clock, with the UTC offset where that wall time occurs twice
    available_label: str = ""
    issue_label: str = ""


class InformationSet:
    """The information available when a forecast is issued, and the rules that say when each column is known.

    Args:
        issue_time: Local time the forecast is issued on day D, "HH:MM".
        issue_timezone: Zone of `issue_time`, or "data" (`DATA_ZONE`): each frame's own zone, so that every market of a run
            over several (`ModelPipeline.run(data={zone: frame})`) issues at `issue_time` on its own clock. `ModelPipeline`
            resolves it for each frame (`in_data_zone`) before anything reads it, so its manifests record the zone; given to a
            model's own `run` or a check directly, it is read in the zone of the frame it judges (`_issue_zone`).
        rules: Overrides and additions, `{column name or glob pattern: Availability or dict}`, or a list of them with
            disjoint validity periods for a lag that changed over time (`Availability.during`, `RuleChain`). They take
            precedence over accepted assumptions, rules derived from data sources, and the built-in defaults.
        assumptions: Names from `ASSUMPTIONS` the user accepts explicitly (e.g. "entsoe_a69_by_10").
        unknown: What to do with a predictor whose base column has no rule: "raise" (default), "warn" or
            "ignore". A check that skips columns it knows nothing about would report a pass it cannot vouch
            for, so declare rules for your own columns.
        use_defaults: Include `default_rules()`.

    Example:
        >>> info = InformationSet("11:45", "Europe/Brussels", rules={"load": Availability.realised("1h")})
        >>> ModelPipeline(information_set=info)
    """

    def __init__(
        self,
        issue_time: str = DEFAULT_ISSUE_TIME,
        issue_timezone: str = "Europe/Brussels",
        rules: Optional[Dict[str, Union[Availability, Dict[str, Any], Sequence[Any]]]] = None,
        assumptions: Optional[Sequence[str]] = None,
        unknown: str = "raise",
        use_defaults: bool = True,
    ):
        _parse_time(issue_time)
        if issue_timezone != DATA_ZONE:
            _check_zone(issue_timezone)
        if unknown not in _UNKNOWN_POLICIES:
            raise ValueError(f"unknown must be one of {list(_UNKNOWN_POLICIES)}, got '{unknown}'")
        unknown_assumptions = [a for a in (assumptions or []) if a not in ASSUMPTIONS]
        if unknown_assumptions:
            raise ValueError(f"Unknown assumption(s) {unknown_assumptions}. Available: {list(ASSUMPTIONS)}")
        self.issue_time = issue_time
        self.issue_timezone = issue_timezone
        self.rules: Dict[str, Rule] = {k: _as_rule(v) for k, v in (rules or {}).items()}
        self.assumptions: List[str] = list(assumptions or [])
        self.unknown = unknown
        self.use_defaults = use_defaults
        self._source_rules: Dict[str, Rule] = {}
        self._source_zones: Dict[str, str] = {}
        self._chained: Dict[str, ChainedInput] = {}

    # -- configuration --------------------------------------------------------

    def with_source_rules(self, sources: Iterable[Any]) -> "InformationSet":
        """A copy that also knows the columns of these data sources (prefixes, Open-Meteo variables), and, of a `ForecastSource`,
        its upstream (`chained_inputs`)."""
        sources = list(sources)
        out = self.copy()
        out._source_rules = {**self._source_rules, **{k: _as_rule(v) for k, v in source_rules(sources).items()}}
        out._source_zones = {**self._source_zones, **_source_market_zones(sources)}
        out._chained = {**self._chained, **chained_sources(sources)}
        return out

    def chained_inputs(self) -> Dict[str, ChainedInput]:
        """Column -> its upstream, for each column read from another pipeline's stored forecasts (`ForecastSource`), as
        `with_source_rules` found them."""
        return dict(self._chained)

    def chained_of(self, name: str) -> Optional[ChainedInput]:
        """The upstream of the chained column a predictor name reads (`solar_fc_d+1`, `solar_fc_h05_d-1`,
        `solar_fc_day_max`), through its shifts and what it was built from; None for any other column."""
        if not self._chained:
            return None
        chain = parse_predictor(name)
        for candidate, _ in chain:
            if candidate in self._chained:
                return self._chained[candidate]
        built = _built_from_name(chain[-1][0])
        return None if built is None else self.chained_of(built)

    def copy(self) -> "InformationSet":
        out = InformationSet(
            self.issue_time,
            self.issue_timezone,
            dict(self.rules),
            list(self.assumptions),
            self.unknown,
            self.use_defaults,
        )
        out._source_rules = dict(self._source_rules)
        out._source_zones = dict(self._source_zones)
        out._chained = dict(self._chained)
        return out

    def in_data_zone(self, index: pd.DatetimeIndex) -> "InformationSet":
        """This information set for a frame with this index: itself, or with `issue_timezone="data"` a copy issued in the
        index's zone (`zone_name`, as a manifest records it), which a run with the zone written out gives to the bit. A naive
        index has no zone to issue in, and neither has one in UTC or at a fixed offset (`_market_clock`): both raise."""
        if self.issue_timezone != DATA_ZONE:
            return self
        out = self.copy()
        out.issue_timezone = _market_clock(getattr(index, "tz", None))
        _check_zone(out.issue_timezone)
        return out

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"issue_time": self.issue_time, "issue_timezone": self.issue_timezone}
        if self.rules:
            out["rules"] = {k: v.to_dict() for k, v in self.rules.items()}
        if self.assumptions:
            out["assumptions"] = list(self.assumptions)
        out["unknown"] = self.unknown
        if not self.use_defaults:
            out["use_defaults"] = False
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InformationSet":
        return cls(**data)

    def _layers(self) -> List[Tuple[str, Dict[str, Availability]]]:
        layers = [("rules", self.rules)]
        accepted = {ASSUMPTIONS[a][0]: ASSUMPTIONS[a][1] for a in self.assumptions}
        if accepted:
            layers.append(("assumptions", accepted))
        if self._source_rules:
            layers.append(("sources", self._source_rules))
        if self.use_defaults:
            layers.append(("defaults", _built_in_rules()))
        return layers

    def resolve(self, name: str) -> ResolvedColumn:
        """Find the rule for a column: exact names from the full name down to the base, then glob patterns
        from the base up; layers in order rules > accepted assumptions > sources > defaults.

        A DailyProfileTransformer or DailyStatsTransformer column first looks for a user's rule on its own name, which
        describes the series it was built from; otherwise it takes the whole resolution of the column it was built from,
        so `generation_solar_forecast_day_max` is judged by `generation_*_forecast`, not caught by `generation_*`. Either
        way the value is read at its slot or the day's last slot, and the built-from column's source zone guard applies.
        An exact rule on a shifted name (`x_h05_d+1`) describes that very column, as for any other column.
        """
        chain = parse_predictor(name)
        base, shifts = chain[-1]
        column = _built_from_name(base)
        if column is None:
            for layer, rules in self._layers():
                found = _find_rule(name, chain, rules, layer)
                if found is not None:
                    return found
            return ResolvedColumn(name, base, shifts, None)
        user = _find_rule(name, chain, self.rules, "rules")
        if user is not None and user.base != base:
            return user
        source = self.resolve(column)
        ruling = source if user is None else user
        return ResolvedColumn(name, base, shifts, ruling.rule, ruling.key, ruling.layer, source)

    def _resolve_for(self, name: str, step: int) -> ResolvedColumn:
        """`resolve` for data with slots of `step` minutes.

        Only on 15-minute data is a two-digit `_sNN` column surely a quarter-hour (another slot length is off the grid or
        has three digits). On hourly data it can be a slot of any length, and reading `load_actual_s30` of a 30-minute
        profile (15:00) as a quarter-hour (07:30) by the rule of `load_actual` would turn the error for a column without
        a rule into a silent look-ahead, so there it needs a user's rule on its own name.
        """
        col = self.resolve(name)
        link = col
        while step != 15 and link.built_from is not None:
            own = _find_rule(link.name, parse_predictor(link.name), self.rules, "rules")
            if link.slot is not None and link.slot[1] == _SLOT_MINUTES["s"] and own is None:
                return ResolvedColumn(name, col.base, col.shifts, None)
            link = link.built_from
        return col

    # -- the check ------------------------------------------------------------

    def check(
        self,
        expanded: Dict[Tuple[int, int], List[str]],
        target: str,
        freq: str,
        index: pd.DatetimeIndex,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
        model_name: str = "Model",
        issue_days: Optional[Sequence[str]] = None,
        check_target: bool = True,
        training_rows: Optional[TrainingRows] = None,
    ) -> None:
        """Raise InformationSetError if a predictor (or the newest training target) is not known at issue time.

        `issue_days` checks these days instead of [test_start, test_end]: a run with validation forecasts no day between
        its windows. `check_target=False` leaves the training target out, for a model whose newest target is read from its
        rule (`newest_target="published"`, see `newest_targets`): what it trains on is known by construction. `training_rows`
        are the rows the fits read, on which a column whose rule changes over time is checked too (see `violations`)."""
        violations, unknown = self.violations(
            expanded,
            target,
            freq,
            index,
            test_start,
            test_end,
            issue_days,
            check_target=check_target,
            training_rows=training_rows,
        )
        self._report_unknown(unknown, model_name)
        if violations:
            raise InformationSetError(self._format(violations, model_name, zone=self._issue_zone(index)))
        self._log_assumptions(model_name)

    def late_uses(
        self,
        expanded: Dict[Tuple[int, int], List[str]],
        target: str,
        freq: str,
        index: pd.DatetimeIndex,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
        model_name: str = "Model",
        issue_days: Optional[Sequence[str]] = None,
        check_target: bool = True,
        training_rows: Optional[TrainingRows] = None,
    ) -> List[Violation]:
        """What `availability="warn"` lists: every use `check` refuses, and every use of a column without a rule, whatever
        `unknown` says, instead of an error. A column of a data source on an index in another zone still raises: neither
        the check nor the listing could say anything true about it. `check_target` and `training_rows` are as in `check`."""
        found, _ = self.violations(
            expanded,
            target,
            freq,
            index,
            test_start,
            test_end,
            issue_days,
            check_target=check_target,
            training_rows=training_rows,
        )
        step = _STEP_MINUTES[freq]
        resolved: Dict[str, ResolvedColumn] = {}
        targets = [(target, ROLE_TARGET)] if check_target else []

        def no_rule(name: str) -> Optional[ResolvedColumn]:
            col = resolved.get(name)
            if col is None:
                col = resolved[name] = self._resolve_for(name, step)
            return col if col.rule is None else None

        for (period, hz), predictors in sorted(expanded.items()):
            for name, role in [(p, ROLE_PREDICTOR) for p in predictors] + targets:
                col = no_rule(name)
                if col is not None:
                    found.append(Violation(name, period, hz, col, math.nan, math.nan, role))
        self._log_assumptions(model_name)
        return found

    def _report_unknown(self, unknown: Iterable[str], model_name: str) -> None:
        """Follow the `unknown` policy for columns without a rule."""
        if not unknown:
            return
        message = (
            f"{model_name}: no availability rule for column(s) {sorted(unknown)}. Declare when they are known, "
            "e.g. InformationSet(rules={'my_col': Availability.realised('1h')}) or Availability.static(), "
            "or pass unknown='warn'."
        )
        if self.unknown == "raise":
            raise InformationSetError(message)
        if self.unknown == "warn":
            warnings.warn(message, stacklevel=3)

    def _log_assumptions(self, model_name: str) -> None:
        if self.assumptions:
            logger.info(f"{model_name}: information set accepts assumption(s) {self.assumptions}")

    def violations(
        self,
        expanded: Dict[Tuple[int, int], List[str]],
        target: str,
        freq: str,
        index: pd.DatetimeIndex,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
        issue_days: Optional[Sequence[str]] = None,
        check_target: bool = True,
        training_rows: Optional[TrainingRows] = None,
    ) -> Tuple[List[Violation], set]:
        """All (predictor, period, horizon) uses of values not known at issue time, and unknown base columns.

        Every issue day D of the test period (or of `issue_days`) is checked, and each zone conversion uses the UTC offsets at the instant
        it converts: when the data zone and a rule's zone don't switch DST together (a UTC or fixed-offset index
        against Brussels rules), a value delivered or published across a switch from D has other offsets than D.
        A violation reports the issue day on which its value comes latest after the issue time.

        `check_target=False` has no use of the training target, for a model with `newest_target="published"`: its newest
        target is the newest its rule calls known (`newest_targets`), so it is known by construction.

        A known value of the issue day's row says that every older one is known, unless the column's rule changes over time
        (`over_time`): a lag that shortens, a gap or a `valid_from` can leave an older value unknown. Given `training_rows`, such a
        column is also checked on every row the fits read at each horizon, and the training target on every target they read,
        as `leakage_probe` perturbs them; any other column is checked as before.
        """
        step = _STEP_MINUTES[freq]
        resolved: Dict[str, ResolvedColumn] = {}
        unknown = set()
        # The newest training target is the value delivered on D-1 in the same period.
        uses = {ROLE_PREDICTOR: 0, ROLE_TARGET: -1}  # role -> the row's day relative to D
        periods: Dict[Tuple[str, str], set] = {}  # the periods reading each (name, role); the horizon doesn't matter
        by_horizon: Dict[Tuple[str, int], set] = {}  # the periods reading each (name, horizon), for the training rows
        for (period, hz), predictors in expanded.items():
            for name in predictors:
                periods.setdefault((name, ROLE_PREDICTOR), set()).add(period)
                by_horizon.setdefault((name, hz), set()).add(period)
            if check_target:
                periods.setdefault((target, ROLE_TARGET), set()).add(period)
        targets = [(target, ROLE_TARGET)] if check_target else []
        for name in ([target] if check_target else []) + [name for name, _ in periods]:
            if name not in resolved:
                resolved[name] = self._resolve_for(name, step)
                if resolved[name].rule is None:
                    unknown.add(resolved[name].series)
                else:
                    self._check_source_zone(resolved[name], index)

        changing = sorted({col.series for col in resolved.values() if over_time(col.rule)})
        if changing and training_rows is None:
            warnings.warn(
                f"The rules of {changing} change over time, and only the issue day's row was judged: with such a rule an older "
                "row a fit reads can hold a value known later than the issue day's. Pass training_rows=TrainingRows(window, "
                "...), as a model run does, to judge those rows too.",
                UserWarning,
                stacklevel=3,
            )
        days = pd.DatetimeIndex(issue_days) if issue_days is not None else self._issue_days(index, test_start, test_end)
        if not len(days):
            return [], unknown
        clocks: List[_IssueClock] = []

        def clock() -> "_IssueClock":
            """The issue clock of these days, made when a verdict isn't kept already."""
            if not clocks:
                clocks.append(self._issue_clock(days, index, freq))
            return clocks[0]

        # What a column's verdict is computed from besides the column itself (`_verdict`): the issue clock's inputs (the issue
        # time and its zone, the data's zone, a daily frame's row time, the issue days) and the assumptions, whose hints it says.
        # The issue days by a digest of their instants: a key holds a few hundred bytes, however many days a check reads.
        clock_key = (
            self.issue_time,
            self.issue_timezone,
            self._data_zone(index),
            getattr(index, "tz", None) is None,
            freq,
            _row_anchor(index, freq),
            str(days.tz),
            str(days.dtype),
            len(days),
            hashlib.sha256(days.asi8.tobytes()).hexdigest(),
            tuple(self.assumptions),
        )

        late: Dict[Tuple[str, str, int], _LatePeriod] = {}  # (name, role, period) -> its worst day
        # (name, period, horizon) -> its worst day, of a predictor whose rule changes over time: its training rows depend on h.
        late_at: Dict[Tuple[str, int, int], _LatePeriod] = {}
        for (name, role), rows in periods.items():
            col = resolved[name]
            if col.rule is None:
                continue
            judged = training_rows if training_rows is not None and over_time(col.rule) else None
            # A predictor whose rule changes over time is judged on each horizon's training rows, of the periods reading it there.
            horizons = (
                tuple(sorted((hz, tuple(sorted(at))) for (other, hz), at in by_horizon.items() if other == name))
                if judged is not None and role == ROLE_PREDICTOR
                else ()
            )
            found, found_at = self._verdict(
                col, role, tuple(sorted(rows)), uses[role], step, clock, clock_key, judged, horizons
            )
            for one in found:
                late[(name, role, one.period)] = one
            for hz, one in found_at:
                late_at[(name, one.period, hz)] = one

        out: List[Violation] = []
        for (period, hz), predictors in sorted(expanded.items()):
            for name, role in [(p, ROLE_PREDICTOR) for p in predictors] + targets:
                found = late_at.get((name, period, hz)) if role == ROLE_PREDICTOR else None
                found = found or late.get((name, role, period))
                if found is not None:
                    col = resolved[name]
                    out.append(found.use(name, hz, col, role, self._weather_cutoff_hint(col)))
        return out, unknown

    def _verdict(
        self,
        col: ResolvedColumn,
        role: str,
        rows: Tuple[int, ...],
        row_day: int,
        step: int,
        clock: Callable[[], "_IssueClock"],
        clock_key: Tuple[Any, ...],
        training_rows: Optional[TrainingRows],
        horizons: Tuple[Tuple[int, Tuple[int, ...]], ...],
    ) -> Tuple[Tuple[_LatePeriod, ...], Tuple[Tuple[int, _LatePeriod], ...]]:
        """`_late_of`, kept by everything it is computed from (`_VERDICTS`): the column as resolved (its rule included), the
        role, the periods reading it, the row's day, the step, the clock's inputs (`clock_key`), and for a rule over time the
        training rows and each horizon's periods. So the models of a sweep reading one column on the same days judge it once."""
        key = (clock_key, repr(col), role, rows, row_day, step, training_rows, horizons)
        try:
            hash(key)
        except TypeError:  # rows given in a form that can't be a key: judged as they are, and nothing kept
            key = None
        if _VERDICTS_KEPT <= 0 or key is None:
            return self._late_of(col, role, rows, row_day, step, clock(), training_rows, horizons)
        with _VERDICTS_LOCK:
            found = _VERDICTS.get(key)
        if found is None:
            found = self._late_of(col, role, rows, row_day, step, clock(), training_rows, horizons)
            with _VERDICTS_LOCK:
                _VERDICTS[key] = found
                while len(_VERDICTS) > _VERDICTS_KEPT:
                    del _VERDICTS[next(iter(_VERDICTS))]
        return found

    def _late_of(
        self,
        col: ResolvedColumn,
        role: str,
        rows: Tuple[int, ...],
        row_day: int,
        step: int,
        clock: "_IssueClock",
        training_rows: Optional[TrainingRows],
        horizons: Tuple[Tuple[int, Tuple[int, ...]], ...],
    ) -> Tuple[Tuple[_LatePeriod, ...], Tuple[Tuple[int, _LatePeriod], ...]]:
        """The late periods of one column `violations` reads in the periods `rows` (sorted) as `role`, on the row of D +
        `row_day`: (each late period, on its worst issue day; for a predictor whose rule changes over time, judged on its
        training rows at each horizon of `horizons` instead, (horizon, late period) of each). `training_rows` is given for a rule
        over time alone."""
        rows = np.array(rows)
        avail = clock.available(col, rows, step, row_day)
        own = None
        if training_rows is not None:
            if role == ROLE_PREDICTOR:
                known = _RowsKnown(col, rows, clock, step)
                found_at = []
                for hz, at in horizons:
                    at = np.array(at)
                    where = np.searchsorted(rows, at)
                    own = avail[where]
                    trained = training_rows.latest(known, hz)[where]
                    found_at.extend(
                        (hz, found)
                        for found in self._late_periods(col, at, np.maximum(own, trained), clock, step, own=own)
                    )
                return (), tuple(found_at)
            own = avail
            avail = np.maximum(avail, training_rows.target_latest(_RowsKnown(col, rows, clock, step)))
        return tuple(self._late_periods(col, rows, avail, clock, step, row_day, own=own)), ()

    @staticmethod
    def _latest_of_rows(
        col: ResolvedColumn, rows: np.ndarray, clock: "_IssueClock", step: int, row_days: Iterable[int]
    ) -> np.ndarray:
        """The latest instant a value of `col` on the rows of these periods dated D + each of `row_days` is known, for every
        [period, issue day D]; -inf without a row day."""
        return _RowsKnown(col, rows, clock, step).latest(row_days)

    def newest_targets(
        self,
        target: str,
        freq: str,
        index: pd.DatetimeIndex,
        days: Sequence[Any],
        horizon: int,
        training_window: Optional[int] = None,
    ) -> np.ndarray:
        """The newest training target of each forecast of a model with `newest_target="published"`, as `[period, issue day]`:
        the largest k such that the target at (D + k, period) and every older one down to (D - 31, period) are known by the
        issue time on D, by the target's own rule.

        It walks up from k = -31 and stops at the first unknown cell, or at `horizon - 1`, beyond which no row moves (a row
        is never dated D or later: see `models._calendar.training_span`). "Every older one" matters when a rule makes a newer
        value known before an older one: rows of such older targets would be a look-ahead. Each cell is read through
        `_IssueClock.available`, the path `violations` and `ctx.published` read through, and a cell the clock skips counts as
        known, having no row. Only the rule decides: a missing value at the newest cell drops that row from a fit, as any
        missing value does, and nothing older takes its place.

        `days` are the issue days, in order. Raises InformationSetError if the target has no rule, whatever `unknown` says
        (it decides numbers, which no setting may leave to chance), or if its value of D-31 isn't known on some issue day.

        A known D-31 says every older target is known, unless the rule changes over time (`over_time`). Then, given the
        `training_window` W, every target the window holds, back to D + k - W + 1, must be known too, or this raises.
        """
        step = _STEP_MINUTES[freq]
        col = self._resolve_for(target, step)
        if col.rule is None:
            raise InformationSetError(
                f"newest_target='published' reads the training target's rule, and '{target}' has none. Declare when it is "
                f"known, e.g. InformationSet(rules={{'{target}': Availability.published('13:00', 1)}})."
            )
        self._check_source_zone(col, index)
        days = pd.DatetimeIndex(pd.to_datetime([str(pd.Timestamp(day).date()) for day in days]))
        # A pipeline asks for the same table several times per model (planning, the input check, the run), and a sweep of
        # variants once per variant, so tables are kept by everything they are computed from, a daily row's time included.
        key = (
            repr(col),
            self.issue_time,
            self.issue_timezone,
            self._data_zone(index),
            getattr(index, "tz", None) is None,
            freq,
            _row_anchor(index, freq) if len(days) else None,
            tuple(days.asi8.tolist()),
            int(horizon),
            training_window,
            # Under a rule over time the check reads back to the frame's first date, so a frame from another day is another table.
            frame_first_day(index),
        )
        with _NEWEST_TABLES_LOCK:
            found = _NEWEST_TABLES.get(key)
        if found is None:
            found = self._newest_targets(col, target, freq, index, days, int(horizon), training_window)
            found.setflags(write=False)
            with _NEWEST_TABLES_LOCK:
                _NEWEST_TABLES[key] = found
                while len(_NEWEST_TABLES) > _NEWEST_TABLES_KEPT:
                    del _NEWEST_TABLES[next(iter(_NEWEST_TABLES))]
        return found

    def _newest_targets(
        self,
        col: ResolvedColumn,
        target: str,
        freq: str,
        index: pd.DatetimeIndex,
        days: pd.DatetimeIndex,
        horizon: int,
        training_window: Optional[int] = None,
    ) -> np.ndarray:
        step = _STEP_MINUTES[freq]
        periods = np.arange(_periods_per_day(freq))
        newest = np.full((len(periods), len(days)), max(horizon - 1, NEWEST_TARGET_FROM), dtype=np.int64)
        if not len(days):
            return newest
        clock = self._issue_clock(days, index, freq)
        open_cells = np.ones(newest.shape, dtype=bool)  # every cell from D-31 up to k is known
        for k in range(NEWEST_TARGET_FROM, horizon):
            known = clock.available(col, periods, step, row_day=k) <= clock.issue
            if k == NEWEST_TARGET_FROM and not known.all():
                period, day = (int(i[0]) for i in np.nonzero(~known))
                raise InformationSetError(
                    f"newest_target='published' looks back {-NEWEST_TARGET_FROM} days at most, and by its rule "
                    f"('{col.rule.describe()}') the training target '{target}' of {-NEWEST_TARGET_FROM} days before the issue "
                    f"day isn't known at the issue time, e.g. period {period} for the issue day {days[day].date()}. Use "
                    "newest_target='d-1' with a rule this slow, and check the rule."
                )
            stopped = open_cells & ~known
            newest[stopped] = k - 1
            open_cells &= known
            if not open_cells.any():
                break
        if training_window is not None and over_time(col.rule):
            # The window's oldest target is D + min(k, 0) - W + 1: a row is never dated D or later, so at a horizon h <= k the
            # newest row is D-1 and its target D-1+h, back W-1 days from there (`models._calendar.training_span`), h=1 the oldest.
            # Under rules over time a known D-31 says nothing of those before it. A target dated before the frame's first local date
            # has no row there, so it trains nothing and isn't judged.
            oldest = np.maximum(np.minimum(newest, 0) - training_window + 1, _frame_floors(index, days)[None, :])
            for k in range(int(oldest.min()), NEWEST_TARGET_FROM):
                unknown = (oldest <= k) & (clock.available(col, periods, step, row_day=k) > clock.issue)
                if unknown.any():
                    period, day = (int(i[0]) for i in np.nonzero(unknown))
                    raise InformationSetError(
                        f"newest_target='published' trains on the targets of a window of {training_window} days, and by its "
                        f"rules over time ('{col.rule.describe()}') the training target '{target}' of {-k} days before the "
                        f"issue day isn't known at the issue time, e.g. period {period} for the issue day {days[day].date()}. "
                        "Shorten the training_window, start the frame where the rule begins (no target before the frame's "
                        "first date is read, and an 'expanding' window reads back to it), or start the test later."
                    )
        return newest

    def _issue_clock(
        self, days: pd.DatetimeIndex, index: pd.DatetimeIndex, freq: Optional[str] = None
    ) -> "_IssueClock":
        """The issue times of these issue days (naive midnights, the first one first) on an index like `index`, whose rows are
        at `freq`: a daily frame's row is at the time of day its index says (`_date_utils.daily_anchor`), any other's at its
        period."""
        zone = self._issue_zone(index)
        ctx = _Context(days[0], self._data_zone(index), naive=getattr(index, "tz", None) is None, issue_zone=zone)
        offset = np.asarray((days - days[0]) // pd.Timedelta(days=1), dtype=np.int64)
        issue = _issue_minutes(offset * 1440.0 + _minutes(self.issue_time), zone, ctx)
        midnight = _to_instants(offset * 1440.0, ctx.data_zone, ctx)
        return _IssueClock(days, ctx, offset, issue, midnight, _row_anchor(index, freq))

    def _late_periods(
        self,
        col: ResolvedColumn,
        rows: np.ndarray,
        avail: np.ndarray,
        clock: "_IssueClock",
        step: int,
        row_day: int = 0,
        used: Optional[np.ndarray] = None,
        own: Optional[np.ndarray] = None,
    ) -> List["_LatePeriod"]:
        """Each period with a value known after the issue time, on the issue day it is latest; its times are minutes since
        midnight of that day, as a context for it alone would give them. `avail` is `clock.available(col, rows, step,
        row_day)`. `used` limits the search to the cells [period, issue day] a forecast actually reads, for lists that
        hold on some days only. `own`, for a column whose rule changes over time, is the checked row's alone, `avail` then
        holding the training rows' latest too: where those come later the use says so."""
        excess = avail - clock.issue
        if used is not None:
            excess = np.where(used, excess, -np.inf)
        worst = excess.argmax(axis=1)
        late = [(r, period, i) for r, (period, i) in enumerate(zip(rows.tolist(), worst)) if excess[r, i] > 0]
        hints = self._assumption_hints(col, rows, clock, step, row_day) if late else []
        out = []
        for r, period, i in late:
            day_ctx = clock.context_of(i)
            hint = next((text for known_by, text in hints if known_by[r, i]), "")
            if own is not None and avail[r, i] > own[r, i]:
                hint = _TRAINING_ROWS_HINT
            known = clock.minute_of_day(avail[r, i], i)
            issue = day_ctx.issue_minute
            out.append(
                _LatePeriod(period, known, issue, hint, _fmt_instant(known, day_ctx), _fmt_instant(issue, day_ctx))
            )
        return out

    def _check_source_zone(self, col: ResolvedColumn, index: pd.DatetimeIndex) -> None:
        """Raise if `col` is a data source's column whose rule needs the market zone and the index is in another.

        The check reads each row's issue day D as a local day of the index's zone. An Open-Meteo column took its
        run from D in the source's market zone, so on a UTC index a row at 22:00Z belongs to the market's next
        day, whose run is published only the next morning: neither the check nor the probe could see that.
        Zones with the same wall clock on every row (Europe/Berlin and Europe/Warsaw) are the same here. A column built
        from such a column can't be judged either, whatever rule its own name has: only a user's rule for the column
        itself lifts this.
        """
        if col.built_from is not None:
            self._check_source_zone(col.built_from, index)
            return
        zone = self._source_zones.get(col.key) if col.layer == "sources" and col.key else None
        if zone is None or getattr(index, "tz", None) is None or not len(index):
            return
        if (index.tz_convert(zone).tz_localize(None) == index.tz_localize(None)).all():
            return
        raise InformationSetError(
            f"Column '{col.name}' comes from a data source for the market zone '{zone}', but the data is indexed "
            f"in '{index.tz}'. The source cut each local day D of the market off at D's own model run, which a "
            f"'{index.tz}' index cannot tell: a row late in the evening already belongs to the next day of "
            f"'{zone}', whose run is published only the next morning. Put the index in the market zone first: add "
            f"a TimezoneTransformer to the DataPipeline (it takes its zone from DataPipeline(timezone='{zone}'))."
        )

    def _assumption_hints(
        self, col: ResolvedColumn, rows: np.ndarray, clock: "_IssueClock", step: int, row_day: int
    ) -> List[Tuple[np.ndarray, str]]:
        """For each assumption not accepted whose rule would replace `col`'s: the cells [row, issue day] it would make known
        in time, and the hint that says so."""
        out = []
        for name, (pattern, rule, description) in ASSUMPTIONS.items():
            if name in self.assumptions or col.key != pattern:
                continue
            known = clock.available_under(col, rule, rows, step, row_day) <= clock.issue
            out.append((known, f"known in time if you accept assumption '{name}': {description}"))
        return out

    @staticmethod
    def _weather_cutoff_hint(col: ResolvedColumn) -> str:
        """An OpenMeteoSource column's own cutoff is a source-side setting, not a data fact: point at it."""
        cutoffs = [] if col.rule is None else [one for one in rule_elements(col.rule) if one.kind == "weather_cutoff"]
        if not cutoffs:
            return ""
        rule = cutoffs[-1]  # the newest period's, of a column's rules over time
        # The shifts after a profile's slot or a day aggregate always shift the row; only the column it was built from
        # has a lead.
        if col.built_from is not None:
            row_shifts = list(col.shifts) + _shifts_after_source_lead(col.built_from.shifts)
        else:
            row_shifts = _shifts_after_source_lead(col.shifts)
        if any(s.offset > 0 for s in row_shifts):
            return (
                "a forward shift after the source's own `_d+k` reads the run of a later issue day, which starts "
                "after the issue time whatever `run` is; drop the forward shift"
            )
        run = rule.time.split(":", 1)[0]
        return (
            f"OpenMeteoSource(run={run!r}, ...) is known {rule.delay} after that UTC run; suggest an "
            "earlier run or a later issue time"
        )

    @staticmethod
    def _issue_days(index, test_start, test_end) -> pd.DatetimeIndex:
        """The issue days D (naive midnights) of the test period, else of the whole index."""
        if test_start is not None and test_end is not None:
            return pd.date_range(pd.Timestamp(test_start), pd.Timestamp(test_end), freq="D")
        if len(index):
            local = index.tz_localize(None) if index.tz is not None else index
            return pd.date_range(local.min().normalize(), local.max().normalize(), freq="D")
        return pd.date_range("2024-01-01", periods=7, freq="D")

    def _data_zone(self, index) -> str:
        return str(index.tz) if getattr(index, "tz", None) is not None else self._issue_zone(index)

    def _issue_zone(self, index) -> str:
        """The zone the issue time is read in on a frame with this index: `issue_timezone`, or with `"data"` (`DATA_ZONE`) the
        index's own zone, as `in_data_zone` resolves it. An index that is naive, in UTC or at a fixed offset has no market's
        clock to issue on, and raises (`_market_clock`)."""
        if self.issue_timezone != DATA_ZONE:
            return self.issue_timezone
        try:
            _market_clock(getattr(index, "tz", None))
        except ValueError as exc:
            raise InformationSetError(str(exc)) from None
        return str(index.tz)

    def _format(
        self, violations: List[Violation], model_name: str, advice: bool = True, zone: Optional[str] = None
    ) -> str:
        """The message of a refusal; `zone` is the issue zone as resolved for the frame (`_issue_zone`), `issue_timezone` if
        not given."""
        # One line per base column and rule, listing the predictors that read it.
        grouped: Dict[Tuple[str, str, Optional[str]], List[Violation]] = {}
        for v in violations:
            grouped.setdefault((v.role, v.column.base, v.column.key), []).append(v)
        lines = [
            f"{model_name}: {len(violations)} use(s) of inputs not known at the issue time "
            f"({self.issue_time} {zone or self.issue_timezone} on the issue day D):"
        ]
        for (role, base, key), items in grouped.items():
            worst = max(items, key=lambda v: v.available - v.issue)
            names = list(dict.fromkeys(v.predictor for v in items))
            shown = ", ".join(names[:4]) + (f", ... ({len(names)} names)" if len(names) > 4 else "")
            periods = _ranges(sorted({v.period for v in items}))
            horizons = _ranges(sorted({v.horizon for v in items}))
            rule = f"'{base}'" + (f" (rule '{key}')" if key and key != base else "")
            lines.append(
                f"  - {role} {shown}: {rule} is {worst.column.rule.describe()}. Periods {periods}, horizons "
                f"{horizons}; e.g. {worst.predictor} for period {worst.period} is known "
                f"{worst.available_label or _fmt_minute(worst.available)}, issued "
                f"{worst.issue_label or _fmt_minute(worst.issue)}."
            )
            hinted = [v.predictor for v in items if v.hint]
            if hinted:
                first = next(v for v in items if v.hint)
                lines.append(f"      {', '.join(dict.fromkeys(hinted))}: {first.hint}")
        if advice:
            lines.append(_LATE_ADVICE)
        return "\n".join(lines)


@dataclass(frozen=True)
class _Context:
    """Times are minutes elapsed since `origin`, midnight of the naive day `day` in `data_zone`: instants, so two of
    them compare correctly across a DST switch. Wall-clock times of any zone are converted at the very instant they
    denote (see `_localize`/`_wall`), not with a fixed offset of `day`."""

    day: pd.Timestamp  # the issue day D, or (in `violations`) the first issue day of the test period
    data_zone: str
    issue_minute: float = 0.0  # the issue instant on `day`; `violations` keeps one per issue day instead
    naive: bool = False  # the data's index has no zone (see `_shift_rows`)
    # The issue time's zone: a week of the data's dates is read in it too (`_data_weeks`).
    issue_zone: Optional[str] = None

    @property
    def weekday(self) -> int:
        return self.day.weekday()

    @cached_property
    def origin(self) -> pd.Timestamp:
        return pd.Timestamp(self.day).tz_localize(self.data_zone, ambiguous=True, nonexistent="shift_forward")

    @cached_property
    def epoch_day(self) -> int:
        """`day` in days since 1970-01-01."""
        return int((pd.Timestamp(self.day) - _EPOCH) // pd.Timedelta(days=1))

    @cached_property
    def origin_minutes(self) -> float:
        """`origin` in minutes since the epoch."""
        return float((self.origin.tz_convert("UTC").tz_localize(None) - _EPOCH) / pd.Timedelta(minutes=1))


@dataclass(frozen=True)
class _IssueClock:
    """Issue days and their issue times. `violations` and the `ctx` of a predictor function read availability through
    it, so they can't disagree about a cell."""

    days: pd.DatetimeIndex
    ctx: _Context  # its day is the first issue day
    offset: np.ndarray  # each issue day as days after the first
    issue: np.ndarray  # its issue instant in minutes since the first's midnight (`ctx.origin`)
    midnight: np.ndarray  # its own midnight in the same minutes: elapsed time, so days of 23 and 25 hours count as such
    # A daily frame's row time, wall-clock minutes after midnight, the same on every date (`_contract.check_index`); None for
    # rows at their period's start.
    anchor: Optional[float] = None
    _skipped: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict, compare=False, repr=False)

    def row_minutes(self, periods: np.ndarray, step: int) -> np.ndarray:
        """The wall-clock minute of the row of each period: its start, or on a daily frame the time its rows are stamped at."""
        periods = np.asarray(periods)
        if self.anchor is not None:
            return np.full(len(periods), self.anchor, dtype=float)
        return (periods * step).astype(float)

    def available(self, col: ResolvedColumn, periods: np.ndarray, step: int, row_day: int = 0) -> np.ndarray:
        """The instant the value of `col` at the row (D + row_day, period) is known, in minutes since the first issue
        day's midnight, for every [period, issue day D]. A repeated autumn hour is read at its second occurrence: the row
        the model forecasts from (the last of its period that day), and the later of the two, so the first is covered.

        A row the data zone's clock skips (02:00 on the day it goes forward) has no value, so it gets no verdict: -inf, as if
        known in advance, which `published` still answers False for, since the cell is empty."""
        return self.available_under(col, col.rule, periods, step, row_day)

    def available_under(
        self, col: ResolvedColumn, rule: Rule, periods: np.ndarray, step: int, row_day: int = 0
    ) -> np.ndarray:
        """`available` with `col` read through `rule` instead of its own."""
        periods = np.asarray(periods)
        issue_day = np.tile(self.offset, len(periods))
        minute = np.repeat(self.row_minutes(periods, step), len(self.days))
        later = np.zeros(len(minute), dtype=bool)
        daily = self.anchor is not None
        avail = _available_array(col, rule, issue_day + row_day, minute, step, self.ctx, later, issue_day, daily)
        avail = avail.reshape(len(periods), len(self.days))
        if not self.ctx.naive:
            skipped = self.skipped(periods, step, row_day)
            if skipped.any():
                # A skipped slot has no value to be late. So `newest_targets` walks past it: on the day before a spring
                # switch the switch hour's newest target is the switch day itself, whose row has no target and drops, and
                # the window of W dates holds W-1 rows. Deliberate, not stopping the walk one day short: since rows are chosen
                # by date a window is W dates, and "d-1" gives that slot W-1 rows on the day after the switch in the same
                # way, so "published" equals "d-1" bit for bit wherever the rule gives -1.
                avail = np.where(skipped, -np.inf, avail)
        return avail

    def skipped(self, periods: np.ndarray, step: int, row_day: int = 0) -> np.ndarray:
        """Whether the clock skips the row (D + row_day, period), for every [period, issue day D]: the slots
        `models._calendar` doesn't plan. A naive index has every slot, so none."""
        grid = self._skipped.get((step, row_day))
        if grid is None:
            grid = self._skipped[(step, row_day)] = self._skipped_slots(step, row_day)
        return grid[np.asarray(periods)]

    def _skipped_slots(self, step: int, row_day: int) -> np.ndarray:
        from .models._calendar import clock_periods_on  # models import this module

        # A daily row's period, 0, is on the clock of every date, as for the planning.
        freq = (
            _DAILY
            if self.anchor is not None
            else next(name for name, minutes in _STEP_MINUTES.items() if minutes == step)
        )
        grid = np.zeros((_periods_per_day(freq), len(self.days)), dtype=bool)
        if self.ctx.naive:
            return grid
        days = self.days + pd.Timedelta(days=row_day)
        exists = clock_periods_on(days, freq, self.ctx.data_zone)
        for j, day in enumerate(days):
            grid[:, j] = True
            grid[list(exists[day]), j] = False
        return grid

    def minute_of_day(self, minutes: float, i: int) -> float:
        """Minutes since the first issue day's midnight -> since the midnight of issue day `i`."""
        return float(minutes - self.midnight[i])

    def context_of(self, i: int) -> _Context:
        """The context of issue day `i` alone."""
        return replace(self.ctx, day=self.days[i], issue_minute=self.minute_of_day(self.issue[i], i))

    def label(self, minutes: float, i: int) -> str:
        """Minutes since the first issue day's midnight as they read on issue day `i`, as `D+1 01:00`."""
        return _fmt_instant(self.minute_of_day(minutes, i), self.context_of(i))


class _RowsKnown:
    """When the values of a column on the rows of some periods are known, by the row's day relative to each issue day D, each
    day read once however many windows ask for it (a check reads one window per horizon)."""

    def __init__(self, col: ResolvedColumn, rows: np.ndarray, clock: "_IssueClock", step: int):
        self._args = (col, rows, step)
        self._clock = clock
        self._by_day: Dict[int, np.ndarray] = {}
        self.days = clock.days
        self.size = len(rows)

    def latest(self, row_days: Iterable[int]) -> np.ndarray:
        """The latest instant over these row days, for every [period, issue day]; -inf without a row day."""
        col, rows, step = self._args
        latest = np.full((len(rows), len(self._clock.days)), -np.inf)
        for row_day in row_days:
            np.maximum(latest, self._known(row_day), out=latest)
        return latest

    def latest_since(self, row_days: range, floors: np.ndarray) -> np.ndarray:
        """`latest` of the row days of `row_days` (a range of step 1) from `floors[i]` on, for the issue day i: a running maximum
        from the last row day back, so each day is read once; -inf where no row day is left."""
        col, rows, step = self._args
        latest = np.full((len(rows), len(self._clock.days)), -np.inf)
        if not len(row_days):
            return latest
        start = np.clip(floors, row_days.start, row_days.stop)  # row_days.stop: no row day left
        running = np.full_like(latest, -np.inf)
        for row_day in range(row_days.stop - 1, int(start.min()) - 1, -1):
            np.maximum(running, self._known(row_day), out=running)
            on = start == row_day
            latest[:, on] = running[:, on]
        return latest

    def _known(self, row_day: int) -> np.ndarray:
        known = self._by_day.get(row_day)
        if known is None:
            col, rows, step = self._args
            known = self._by_day[row_day] = self._clock.available(col, rows, step, row_day)
        return known


class _LatePeriod(NamedTuple):
    """A period whose value is known after the issue time, on the issue day it is latest (`InformationSet._late_periods`)."""

    period: int
    known: float  # minutes since midnight of that day
    issue: float
    hint: str
    known_label: str
    issue_label: str

    def use(
        self, name: str, horizon: int, col: ResolvedColumn, role: str = ROLE_PREDICTOR, hint: str = ""
    ) -> Violation:
        """This period as a use of `name`; `hint` is said only if the period has no hint of its own."""
        return Violation(
            name,
            self.period,
            horizon,
            col,
            self.known,
            self.issue,
            role,
            self.hint or hint,
            self.known_label,
            self.issue_label,
        )


def _frame_floors(index: pd.DatetimeIndex, days: pd.DatetimeIndex) -> np.ndarray:
    """The frame's first local date relative to each of `days`, in days (`first - D`): no row day before it exists."""
    if not len(index):
        return np.full(len(days), np.iinfo(np.int64).min // 2, dtype=np.int64)
    wall = index.tz_localize(None) if getattr(index, "tz", None) is not None else index
    first = wall.min().date()
    return np.array([(first - day.date()).days for day in days], dtype=np.int64)


def frame_first_day(index: pd.DatetimeIndex) -> Optional[str]:
    """The frame's first local date as ISO, what `TrainingRows(first_day=)` takes; None for an empty frame."""
    if not len(index):
        return None
    wall = index.tz_localize(None) if getattr(index, "tz", None) is not None else index
    return wall.min().date().isoformat()


def _periods_per_day(freq: str) -> int:
    from .models._calendar import periods_per_day  # models import this module

    return periods_per_day(freq)


def _row_anchor(index: pd.DatetimeIndex, freq: Optional[str]) -> Optional[float]:
    """The time of day a daily frame's rows stand for, in minutes after midnight; None at any other `freq`."""
    return daily_anchor(index) if freq == _DAILY else None


def _localize(
    day: np.ndarray, minute: Union[float, np.ndarray], zone: str, ctx: _Context, first: Optional[np.ndarray] = None
) -> pd.DatetimeIndex:
    """The instants of wall-clock time `minute` on the days `ctx.day + day` in `zone`.

    A wall time skipped by a spring switch moves forward to the switch; a repeated autumn one is read as its first
    occurrence, or as its second where `first` is False.
    """
    minutes = np.asarray(day, dtype=float) * 1440.0 + np.asarray(minute, dtype=float)
    naive = pd.DatetimeIndex(ctx.day + pd.to_timedelta(np.atleast_1d(minutes), unit="min"))
    first = np.ones(len(naive), dtype=bool) if first is None else first
    return naive.tz_localize(zone, ambiguous=first, nonexistent="shift_forward")


def _minutes_since(instants: pd.DatetimeIndex, ctx: _Context) -> np.ndarray:
    return np.asarray((instants - ctx.origin) / pd.Timedelta(minutes=1), dtype=float)


def _instants(minutes: np.ndarray, ctx: _Context) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(ctx.origin + pd.to_timedelta(np.asarray(minutes, dtype=float), unit="min"))


def _wall(instants: pd.DatetimeIndex, zone: str, ctx: _Context) -> np.ndarray:
    """Wall-clock minutes of `instants` in `zone` after the naive midnight `ctx.day` (D+1 01:00 is 1500)."""
    return np.asarray((instants.tz_convert(zone).tz_localize(None) - ctx.day) / pd.Timedelta(minutes=1), dtype=float)


def _issue_minutes(minutes: np.ndarray, zone: str, ctx: _Context) -> np.ndarray:
    """Issue times, wall-clock minutes after midnight of `ctx.day` in `zone` -> their instants (see `_Context`).

    Unlike a publication time, which a skipped wall time moves forward to the switch, an issue time skipped by a
    spring switch is read with the offset after it: 02:30 Brussels on 2024-03-31 is 00:30Z (01:30 CET), its earliest
    reading, so the issue is never later than asked. This holds within one zone too; a repeated autumn issue time
    is its first occurrence.
    """
    forward = _localize(np.zeros(len(np.atleast_1d(minutes))), minutes, zone, ctx)
    naive = pd.DatetimeIndex(ctx.day + pd.to_timedelta(np.atleast_1d(np.asarray(minutes, dtype=float)), unit="min"))
    earliest = forward - (forward.tz_localize(None) - naive)  # back by as much as the switch moved it forward
    return _minutes_since(earliest, ctx)


# Times convert by arithmetic: a wall time is its instant plus the zone's UTC offset, which is one number on most days, and on
# a day the clocks change, one before the change and one after. The offsets come from a table per zone, built with pandas once,
# and the arithmetic follows pandas' own rules for a time the clocks repeat or skip, so the result is the same to the bit.
# A day the table can't describe (two changes in a day, an offset in seconds) goes through pandas.

_ZONE_TABLES: Dict[str, Tuple[Any, ...]] = {}  # zone -> what `_zone_table` returns


def _zone_table(zone: str, lo: int, hi: int) -> Tuple[Any, ...]:
    """(first, midnight, before, after, change, exact, odd) for consecutive days counted since 1970-01-01, `first` the first
    of them, covering at least `lo` ... `hi + 1`. Per day: its local midnight in `zone` in minutes since the epoch (UTC), the
    zone's UTC offset in minutes before and after the one change of offset that reaches the day's wall times, the change's
    instant (inf without one), whether the day is described exactly so, and whether it needs more than `before`."""
    held = _ZONE_TABLES.get(zone)
    if held is not None and held[0] <= lo and hi + 1 < held[0] + len(held[1]):
        return held
    first, last = lo - 366, hi + 367
    if held is not None:
        first, last = min(first, held[0]), max(last, held[0] + len(held[1]) - 1)
    days = np.arange(first, last + 1)
    naive = pd.DatetimeIndex(_EPOCH + pd.to_timedelta(days, unit="D"))
    local = naive.tz_localize(zone, ambiguous=np.ones(len(naive), dtype=bool), nonexistent="shift_forward")
    midnight = np.asarray((local.tz_convert("UTC").tz_localize(None) - _EPOCH) / pd.Timedelta(minutes=1), dtype=float)
    wall = local.tz_localize(None)
    whole = np.asarray(wall == naive) & np.asarray(wall.tz_localize(zone, ambiguous="NaT", nonexistent="NaT").notna())
    steady = np.append((np.diff(midnight) == 1440.0) & whole[:-1] & whole[1:], False)
    before = days * 1440.0 - midnight  # exact on a steady day; the others are read minute by minute below
    after, change, exact = before.copy(), np.full(len(days), np.inf), steady.copy()
    for k in np.flatnonzero(~steady):
        # Every UTC minute whose wall time can fall on this day, whatever the offset (at most 14 h either way).
        utc = np.arange(days[k] * 1440 - 900, (days[k] + 1) * 1440 + 900)
        clock = pd.DatetimeIndex(_EPOCH + pd.to_timedelta(utc, unit="min")).tz_localize("UTC").tz_convert(zone)
        offset = np.asarray((clock.tz_localize(None) - _EPOCH) / pd.Timedelta(minutes=1)) - utc
        moves = np.flatnonzero(np.diff(offset))
        if len(moves) > 1 or not np.array_equal(offset, np.round(offset)):
            continue  # `exact` stays False: pandas reads this day
        before[k], after[k] = offset[0], offset[-1]
        change[k] = utc[moves[0] + 1] if len(moves) else np.inf
        exact[k] = True
    odd = ~exact | np.isfinite(change)  # the days one offset doesn't describe
    for array in (midnight, before, after, change, exact, odd):
        array.setflags(write=False)
    _ZONE_TABLES[zone] = table = (first, midnight, before, after, change, exact, odd)
    return table


def _to_instants(
    wall: Union[float, np.ndarray], zone: str, ctx: _Context, first: Optional[np.ndarray] = None
) -> np.ndarray:
    """Wall-clock minutes after the naive midnight `ctx.day` in `zone` -> their instants, in minutes since `ctx.origin`.

    As `_localize` reads them: a wall time skipped by a spring switch moves forward to the next whole hour after it (the
    switch, for a whole-hour switch), and a repeated autumn one is its first occurrence, or its second where `first` is False.
    """
    wall = np.atleast_1d(np.asarray(wall, dtype=float))
    if not len(wall):
        return wall.copy()
    absolute = wall + ctx.epoch_day * 1440.0
    lo, hi = int(absolute.min() // 1440.0), int(absolute.max() // 1440.0)
    start, _, before, after, change, exact, odd = _zone_table(zone, lo, hi)
    k = np.floor(absolute / 1440.0).astype(np.int64)
    k -= start
    out = absolute - before[k]
    rows = np.flatnonzero(odd[k])  # on a day of a change: which side of it, or both, or neither
    if len(rows):
        a, kk = absolute[rows], k[rows]
        early, late, at = a - before[kk], a - after[kk], change[kk]
        in_early, in_late = early < at, late >= at  # both: the clocks repeat it; neither: they skip it
        twice = in_early & in_late
        if first is not None:
            twice &= np.broadcast_to(np.asarray(first, dtype=bool), wall.shape)[rows]
        fixed = np.where((in_early & ~in_late) | twice, early, late)
        gap = ~in_early & ~in_late
        # As pandas' shift_forward: the next whole hour of the wall clock, read with the offset after the change.
        fixed[gap] = a[gap] - np.mod(a[gap], 60.0) + 60.0 - after[kk][gap]
        out[rows] = fixed
        slow = rows[~exact[kk] | (gap & (fixed < at))]
        if len(slow):
            which = None if first is None else np.broadcast_to(np.asarray(first, dtype=bool), wall.shape)[slow]
            out[slow] = _minutes_since(_localize(0, wall[slow], zone, ctx, which), ctx) + ctx.origin_minutes
    return out - ctx.origin_minutes


def _to_wall(instants: np.ndarray, zone: str, ctx: _Context, wall: Optional[np.ndarray] = None) -> np.ndarray:
    """Instants in minutes since `ctx.origin` -> their wall-clock minutes in `zone` after the naive midnight `ctx.day`.

    `wall`, the wall times in `zone` these instants were made from, is the answer on every day of one offset, and only the
    others are converted: a skipped time was moved, and a day of a change needs its side of it."""
    instants = np.atleast_1d(np.asarray(instants, dtype=float))
    if not len(instants):
        return instants.copy()
    if wall is not None:
        wall = np.asarray(wall, dtype=float)
        absolute = wall + ctx.epoch_day * 1440.0
        start, *_, odd = _zone_table(zone, int(absolute.min() // 1440.0), int(absolute.max() // 1440.0))
        k = np.floor(absolute / 1440.0).astype(np.int64)
        k -= start
        rows = np.flatnonzero(odd[k])
        out = wall.copy()
        if len(rows):
            out[rows] = _to_wall(instants[rows], zone, ctx)
        return out
    absolute = instants + ctx.origin_minutes
    lo, hi = int(absolute.min() // 1440.0), int(absolute.max() // 1440.0)
    start, midnight, before, after, change, exact, odd = _zone_table(zone, lo - 2, hi + 2)
    # The local day: that of the UTC day's offset, or the one before or after it next to a change. The table reaches a year
    # past either end, so k and k + 1 are in it.
    k = np.floor(absolute / 1440.0).astype(np.int64)
    k -= start
    k = np.floor((absolute + before[k]) / 1440.0).astype(np.int64)
    k -= start
    k -= absolute < midnight[k]
    k += absolute >= midnight[k + 1]
    out = absolute + before[k]
    rows = np.flatnonzero(odd[k])
    if len(rows):
        kk = k[rows]
        a = absolute[rows]
        out[rows] = a + np.where(a < change[kk], before[kk], after[kk])
        slow = rows[~exact[kk]]
        if len(slow):
            out[slow] = _wall(_instants(instants[slow], ctx), zone, ctx) + ctx.epoch_day * 1440.0
    return out - ctx.epoch_day * 1440.0


# ---------------------------------------------------------------------------
# Leakage probe
# ---------------------------------------------------------------------------

# Read through this rule, a column gives the end of the delivery period of its value, through the same shifts, slots and
# day aggregates as under its own rule.
_DELIVERED = Availability.realised("0min")


@dataclass
class ProbeResult:
    """Outcome of `leakage_probe`: forecasts whose prediction moved when unknown values were perturbed."""

    run_day: str
    forecasts: int
    changed: List[Tuple[int, int]]  # (period, horizon)
    max_abs_change: float
    perturbed: Dict[str, int]  # column -> number of perturbed cells
    unchecked: List[str]  # columns without a rule, left unperturbed
    missing: List[Tuple[int, int]] = field(default_factory=list)  # forecasts present in only one of the runs
    # Column -> cells a predictor function read through `ctx` that hold another value in the perturbed frame. A forecast
    # whose function read one is in `changed`, whichever list the noise happened to leave it with.
    moved_reads: Dict[str, int] = field(default_factory=dict)
    # Column -> cells a predictor function read with `ctx.value` though they are known only after the issue time, which
    # only happens when it was planned with `availability` "warn" or "none". Their forecasts are in `changed` too.
    late_reads: Dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.changed and not self.missing

    def assert_ok(self) -> None:
        if not self.ok:
            raise AssertionError(
                f"Forecasts issued on {self.run_day} depend on values not known at issue time: "
                f"{len(self.changed)} of {self.forecasts} changed (max |change| {self.max_abs_change:.6g}), "
                f"e.g. (period, horizon) {self.changed[:10]}; forecasts present in only one run: {self.missing[:10]}. "
                + (f"Read through ctx and moved: {self.moved_reads}. " if self.moved_reads else "")
                + (f"Read through ctx though known later: {self.late_reads}. " if self.late_reads else "")
                + f"Perturbed cells per column: {self.perturbed}"
            )


def leakage_probe(
    model: Any,
    data: pd.DataFrame,
    run_day: str,
    information_set: InformationSet,
    target: str = "price",
    horizon: Optional[int] = None,
    freq: str = "1h",
    columns: Optional[Sequence[str]] = None,
    atol: float = 1e-9,
    seed: int = 0,
    rebuild: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    check_target_boundary: bool = False,
    availability: str = "raise",
    horizons: Optional[Sequence[int]] = None,
    span: Optional[Tuple[str, str]] = None,
    flip: bool = False,
) -> ProbeResult:
    """Perturb every value not known at issue time on `run_day` and check that the forecasts don't change.

    Each column is resolved to its base column and delivery time (so `price_d-1` holds the price of D-1),
    and every cell whose value is known only after `information_set.issue_time` on `run_day` is replaced by
    a large random number. The model is run for `run_day` on the original and on the perturbed data; any
    forecast that moves used information from after the issue time. Columns without a rule are left as
    they are and listed in `unchecked` (or raise, following `information_set.unknown`).

    `model` is a model instance (its `run` is called twice) or a zero-argument factory returning one.

    With `rebuild`, `data` is treated as *source* data (before feature engineering) instead of the model's
    already-built input frame: perturbation is applied to `data` directly, by the raw availability of its
    own columns, and `rebuild(frame) -> features` reproduces the transformer pipeline that builds the
    model's input from it. This catches a leak a transformer bakes into a derived column (e.g. an
    elapsed-time shift that crosses a DST boundary), which perturbing the already-built column by its
    parsed name cannot see: a wrong shift moves the *right* raw value into the *wrong* row, and that row's
    own name still looks fine. At `freq="1D"` the source's rows are read by their hours, and the daily frame `rebuild` makes
    of the perturbed source is then perturbed as daily rows too (what `violations` calls unknown on it), or a whole date's
    value the source repeats on every hour, `actual_daily_max` say, would reach the daily row from an hour known in time.

    `check_target_boundary` additionally perturbs `target` for every row on or after `run_day`, regardless
    of its own Availability rule: the boundary of a model with `newest_target="d-1"` (training targets end at D-1) is a
    deliberate, conservative choice that can be stricter than a column's factual publication time. A model with
    `newest_target="published"` trains on what the target's rule says is known, which may be dated `run_day`, so for it the
    rule alone is the boundary: `check_target_boundary=True` raises, and its run reads its newest targets from
    `information_set`, as `run(information_set=...)` would, without anything else of it.

    A predictor function of `ctx` is planned on each frame, so a value it reads can move its list. A cell it read that
    differs between the frames counts as a change of the forecasts that read it, in `changed` and `moved_reads`, because
    the noise may well leave a list where it was. So does a cell it read on a training day of a `train_on="matching"` list:
    the function answers there as of that day's own issue time, so no such cell is unknown on `run_day` unless it reads
    something it can't know.

    `availability` is the level the function is planned at. With "raise" a late `ctx.value()` stops the probe with
    `InformationSetError`; with "warn" or "none" it returns the value, and the forecasts that read it are in `changed` and
    `late_reads`, whatever the noise did. The boundary tested is the same at every level.

    `horizon=H` runs every horizon from 1 to H, 1 if neither is given, and `horizons` the ones it lists, as `run` does.

    A model with `refit` runs from `run_day` to the end of its refit block (or of the frame) and only `run_day`'s forecasts are
    compared, since a fit serves every issue day of its block and a leak can need the later ones in the run to show. `span`
    (first, last), ISO dates, widens the runs to cover them too, as `co_batch_probe` does with the other items of a call: the
    perturbation is the same, every cell unknown at `run_day`'s issue time on every row.

    `flip` turns each unknown value v of `columns` into 1 - v instead of noise (`perturb_unknown`): the probe of a 0/1 flag such
    as a model's `zero_when` column, which noise would only ever make "not 1", so a flag that is 0 on the issue day would move
    nothing. Without `flip` a model's `zero_when` columns get no noise at all (a run refuses a flag that isn't 0 or 1), and with
    `rebuild` they are put back as `rebuild(data)` builds them, in the daily pass too: probe them with `flip=True`, as `verify`
    does.
    """
    run_day = str(pd.Timestamp(run_day).date())
    horizon = _horizon_arguments(horizon, horizons, 1)
    if check_target_boundary and _newest_target_of(model) == "published":
        raise ValueError(
            "check_target_boundary=True perturbs the target from the issue day on, which a model with "
            "newest_target='published' may train on where its rule says the value is known by the issue time: for such a "
            "model the target's rule is the boundary, and the probe perturbs what it calls unknown. Leave it False."
        )
    target_from_issue = target if check_target_boundary else None
    flags = [] if flip else _flags_of(model, horizon)
    columns = _without(columns, data, flags)
    perturbed, counts, unchecked = (
        perturb_unknown(
            data, run_day, information_set, freq, columns, seed, target_from_issue=target_from_issue, flip=True
        )
        if flip
        else perturb_unknown(data, run_day, information_set, freq, columns, seed, target_from_issue=target_from_issue)
    )
    base = data
    if rebuild is not None:
        base, perturbed = rebuild(data), rebuild(perturbed)
        if _daily_rows(perturbed.index, freq):
            # The source was read by its hours, so a value of a whole date it already holds (a daily maximum repeated on every
            # hour) was perturbed only on the hours known late. The daily rows are read here as `violations` reads them.
            ruled = [name for name in perturbed.columns if _has_rule(information_set, name, freq) and name not in flags]
            perturbed, more, _ = perturb_unknown(perturbed, run_day, information_set, freq, ruled, seed)
            counts = {**counts, **{name: max(counts.get(name, 0), n) for name, n in more.items()}}
        perturbed = _flags_put_back(perturbed, base, flags)

    common = (run_day, target, horizon, freq, information_set, availability, span)
    original, read, late = _issue_day_forecasts(model, base, *common)
    probed, read_probed, _ = _issue_day_forecasts(model, perturbed, *common)
    changed, max_abs_change, missing = _compare_forecasts(original, probed, atol)
    read_by, moved_reads = _moved_reads(read, read_probed, atol)
    late_reads: Dict[str, int] = {}
    for use in late:
        late_reads[use.predictor] = late_reads.get(use.predictor, 0) + 1
    return ProbeResult(
        run_day=run_day,
        forecasts=len(original),
        changed=sorted({*changed, *read_by, *((use.period, use.horizon) for use in late)}),
        max_abs_change=max_abs_change,
        perturbed=counts,
        unchecked=unchecked,
        missing=missing,
        moved_reads=moved_reads,
        late_reads=late_reads,
    )


def _issue_day_forecasts(
    model: Any,
    frame: pd.DataFrame,
    run_day: str,
    target: str,
    horizon: Dict[str, Any],
    freq: str,
    information_set: InformationSet,
    availability: str = "raise",
    span: Optional[Tuple[str, str]] = None,
    zeros: Optional[set] = None,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray]], List[Violation]]:
    """The model's forecasts issued on `run_day` from `frame`, by (period, horizon), the cells a predictor function of
    `ctx` read (see `PredictorPlan.read_cells`), and its reads of values known only later (`PredictorPlan.late_reads`).
    `horizon` is what to pass on to the run: `{"horizon": H}` or `{"horizons": (...)}` (`_horizon_arguments`). `span`
    (first, last), ISO dates, widens the run to cover them as well; the forecasts and reads returned are still `run_day`'s.

    Such a function is planned on this very frame, so a value it reads can move its list, and it answers through the
    information set. The run itself gets none, so a late list isn't refused and the forecasts show it. A model with
    `newest_target="published"` reads its training cut from it (`cut_rules`), and nothing else; so does a `known_only` model
    (`NaiveModel`) which of its forecasts are published, since one that isn't has no forecast in any run, and a model that reads
    history (`models._history`) where its series are cut, whatever its `newest_target`.

    A model with `refit` is run from `run_day` to the end of its refit block (or of the frame), and only `run_day`'s forecasts are
    returned: a unit fits once for every issue day of its block, so a fit that read one of the later days would show only in a
    run that has them, never in a run of one day. The reads are `run_day`'s, from a plan of that day alone.
    """
    m = model() if callable(model) and not hasattr(model, "run") else model
    plan_predictors = getattr(m, "_plan_predictors", None)
    first_day, last_day = run_day, _last_day_of_block(m, frame, run_day)
    if span is not None:  # `co_batch_probe`: the run covers the other items of the issue day's call too
        first_day, last_day = min(first_day, span[0]), max(last_day, span[1])
    plan = run_plan = None
    if plan_predictors is not None:
        horizons = horizon.get("horizons")

        def planned(start: str, end: str) -> Any:
            return plan_predictors(
                frame,
                target,
                horizon.get("horizon") or max(horizons),
                freq,
                information_set,
                start,
                end,
                warn_unpublished=False,
                availability=availability,
                **({} if horizons is None else {"horizons": horizons}),
            )

        plan = run_plan = planned(run_day, run_day)
        if plan is not None and (first_day, last_day) != (run_day, run_day):
            run_plan = planned(first_day, last_day)
    extra = {} if run_plan is None else {"plan": run_plan}
    from .models._history import reads_history  # models import this module

    # A sequence model cuts its series by the rules (`models._history`), whatever its newest target, as such a model's training
    # cut does.
    if getattr(m, "newest_target", "d-1") == "published" or getattr(m, "_known_only", False) or reads_history(m):
        extra["cut_rules"] = information_set
    read = {} if plan is None else plan.read_cells()  # the run lets go of them
    late = [] if plan is None else plan.late_reads()
    ref = m.run(frame, test_start=first_day, test_end=last_day, target=target, freq=freq, **horizon, **extra)
    issued = [r for r in ref._results if r["run_date"] == run_day]
    if zeros is not None:  # the forecasts of 0 a `zero_when` made without a fit (`models._row_target`)
        zeros.update((r["hour"], r["horizon"]) for r in issued if r.get("detail") == "zero_when")
    return {(r["hour"], r["horizon"]): r["prediction"] for r in issued}, read, late


def _last_day_of_block(model: Any, frame: pd.DataFrame, run_day: str) -> str:
    """The last day a probe runs `model` to from `run_day`: the day itself, or with `refit` the last issue day of its block,
    at the latest the frame's last local date. With `refit="auto"` the block is the coarsest of its windows' rules
    (`BaseModel._refit_block`), whose fit serves the most issue days."""
    refit = getattr(model, "_refit_block", getattr(model, "refit", None))
    if refit is None:
        return run_day
    from .models._calendar import refit_block_end  # models import this module

    index = frame.index.tz_localize(None) if getattr(frame.index, "tz", None) is not None else frame.index
    last = min(refit_block_end(run_day, refit), index.max().date())
    return max(last, datetime.date.fromisoformat(run_day)).isoformat()


def _moved_reads(
    original: Dict[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray]],
    other: Dict[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray]],
    atol: float,
) -> Tuple[List[Tuple[int, int]], Dict[str, int]]:
    """The forecasts (period, horizon) that read a cell through `ctx` holding another value in the other frame, and how
    many such cells each column has.

    The forecasts alone can't show this. A function choosing its list by `ctx.value("x") > 0` keeps the list when the
    probe's large positive noise lands on the side the value was already on, so its forecasts stay as they are on most
    days though the choice depends on a value it can't know.
    """
    forecasts, counts = set(), {}
    for key, (cells, values) in original.items():
        if key not in other:
            continue
        name, horizon = key[:2]  # a cell read on a training day has one more part: it is on other days than the others
        other_cells, other_values = other[key]
        moved = cells & other_cells & _cells_differ(values, other_values, atol)
        if moved.any():
            forecasts.update((int(period), horizon) for period in np.nonzero(moved)[0])
            counts[name] = counts.get(name, 0) + int(moved.sum())
    return sorted(forecasts), counts


def _cells_differ(a: np.ndarray, b: np.ndarray, atol: float) -> np.ndarray:
    both_missing = pd.isna(a) & pd.isna(b)
    if a.dtype == object or b.dtype == object:
        return ~(both_missing | (a == b))
    with np.errstate(invalid="ignore"):
        return ~(both_missing | (a == b) | (np.abs(a - b) <= atol))


def _compare_forecasts(
    original: Dict[Tuple[int, int], float], other: Dict[Tuple[int, int], float], atol: float
) -> Tuple[List[Tuple[int, int]], float, List[Tuple[int, int]]]:
    """The forecasts that differ by more than `atol`, the largest difference, and those present in only one run."""
    diffs = {k: abs(original[k] - other[k]) for k in original.keys() & other.keys()}
    changed = sorted(k for k, d in diffs.items() if not d <= atol)
    return changed, max(diffs.values(), default=0.0), sorted(original.keys() ^ other.keys())


@dataclass
class ControlResult:
    """Outcome of `positive_control`: forecasts that moved when the freshest known values were perturbed."""

    run_day: str
    forecasts: int
    moved: List[Tuple[int, int]]  # (period, horizon)
    max_abs_change: float
    perturbed: Dict[str, int]  # column -> number of perturbed cells
    unchecked: List[str]  # columns without a rule, left unperturbed
    unmoved: List[Tuple[int, int]] = field(default_factory=list)  # forecasts of both runs that did not move
    # A model that reads history: the series whose own fresh values were perturbed and moved no forecast. The probe can't vouch
    # that what the model reads of them is the frame's (a copy of its own, a series it ignores): a blind spot.
    blind: List[str] = field(default_factory=list)
    # Forecasts of 0 a `zero_when` made without a fit in both runs (`models._row_target`), kept out of `unmoved`: they read their
    # flag alone, which `verify` probes by flipping it, so not moving under the control is no blind spot.
    exempt: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.moved) and not self.blind


def positive_control(
    model: Any,
    data: pd.DataFrame,
    run_day: str,
    information_set: InformationSet,
    target: str = "price",
    horizon: Optional[int] = None,
    freq: str = "1h",
    columns: Optional[Sequence[str]] = None,
    atol: float = 1e-9,
    seed: int = 0,
    rebuild: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    availability: str = "raise",
    horizons: Optional[Sequence[int]] = None,
    span: Optional[Tuple[str, str]] = None,
) -> ControlResult:
    """Perturb the freshest values known at issue time on `run_day` and check that the forecasts do change.

    A `leakage_probe` that finds nothing is only evidence if its perturbation reaches the model. The control perturbs
    what the forecast may legally use from the last day before the issue (see `perturb_recent`), the same way and
    through the same `rebuild`, and reports which forecasts moved. If none did, the probe's silence means nothing: the
    model has no such input, or it doesn't read the frame it is given.

    What the model fits on stays as it is. Every model trains on the target, so a control that perturbed it would move
    every model, whatever its inputs, and could never fail. `target` is therefore left out of `columns`. With `rebuild`
    the raw target has to be perturbed to reach `price_d-1`, so the rebuilt frame gets its original `target` column
    back, and its rows older than D-1 too, where a raw value of D-1 lands in `x_d+1`. (The exception is the look-ahead
    after a fall-back switch, a known issue, which trains the switch hour on D-1's row.)

    A model that reads history (`models._history`) reads its target's context as an input: for it the target is perturbed too, as
    any input, and its training target, if it has one, is the context's own. Without `rebuild` and `span` each series of its spec
    is perturbed on its own, the target too, one run each: `moved` is what any of them moved, and `blind` the series whose fresh
    values were perturbed and moved nothing, which the probe can't vouch for (a series read from a copy of the model's own would
    pass every probe). A series with no fresh value to perturb (a static rule) says nothing either way and isn't blind. Over runs
    widened by `span` (the co-batch check's, which asks only whether they reach the model) every series is perturbed at once.

    With `rebuild`, days are those of the frame it returns, so a source in UTC is read in the market's zone.

    A model with `newest_target="published"` may train on the row of D-1 (at horizon 1 when the target of D is known by the
    issue time), so for it only rows dated D or later change: an outlier in a training row would move the forecast whatever
    its inputs, and the control could never fail. With `rebuild` that is a row of the built frame: the raw values of D-1 still
    change, so that what row D builds from them (`x_d-1`) does, and every built row before D is put back.

    `model` is a model instance or a zero-argument factory, and `availability` the level a predictor function is planned
    at, and `horizon` or `horizons` what it runs, as in `leakage_probe`. A model with `refit` runs to the end of its block, as there,
    and `span` widens the runs as there.
    """
    from .models._history import reads_history  # models import this module

    run_day = str(pd.Timestamp(run_day).date())
    horizon = _horizon_arguments(horizon, horizons, 1)
    first_row_day = 0 if _newest_target_of(model) == "published" else -1
    # A model's `zero_when` flags get no noise (a run refuses a flag that isn't 0 or 1); its forecasts of 0 are `exempt`.
    flags = _flags_of(model, horizon)
    names = list(data.columns) if columns is None else list(columns)
    names = [name for name in names if name not in flags]
    # A model that reads history reads its target's context as an input (`models._history`): its fresh values are perturbed too.
    history = reads_history(model() if callable(model) and not hasattr(model, "run") else model)
    if rebuild is None and not history:
        names = [name for name in names if name != target]
    if rebuild is None and history and span is None:
        return _history_control(
            model,
            data,
            run_day,
            information_set,
            target,
            horizon,
            freq,
            names,
            atol,
            seed,
            availability,
            span,
            first_row_day,
        )
    if rebuild is None:
        base = data
        perturbed, counts, unchecked = perturb_recent(
            data, run_day, information_set, freq, names, seed, first_row_day=first_row_day
        )
    else:
        base = rebuild(data)
        both_aware = data.index.tz is not None and base.index.tz is not None
        in_market_zone = data.tz_convert(base.index.tz) if both_aware else data
        perturbed, counts, unchecked = perturb_recent(in_market_zone, run_day, information_set, freq, names, seed)
        perturbed.index = data.index
        perturbed = _with_training_data_of(
            base, rebuild(perturbed), run_day, None if history else target, first_row_day
        )
        perturbed = _flags_put_back(perturbed, base, flags)
    common = (run_day, target, horizon, freq, information_set, availability, span)
    zeros_before: set = set()
    zeros_after: set = set()
    original, *_ = _issue_day_forecasts(model, base, *common, zeros=zeros_before)
    other, *_ = _issue_day_forecasts(model, perturbed, *common, zeros=zeros_after)
    moved, max_abs_change, _ = _compare_forecasts(original, other, atol)
    exempt = (zeros_before & zeros_after) - set(moved)
    return ControlResult(
        run_day=run_day,
        forecasts=len(original),
        moved=moved,
        max_abs_change=max_abs_change,
        perturbed=counts,
        unchecked=unchecked,
        unmoved=sorted((original.keys() & other.keys()) - set(moved) - exempt),
        exempt=sorted(exempt),
    )


def _history_control(
    model: Any,
    data: pd.DataFrame,
    run_day: str,
    information_set: InformationSet,
    target: str,
    horizon: Dict[str, Any],
    freq: str,
    names: Sequence[str],
    atol: float,
    seed: int,
    availability: str,
    span: Optional[Tuple[str, str]],
    first_row_day: int,
) -> ControlResult:
    """`positive_control` of a model that reads history: each series of its spec that `names` holds perturbed on its own (one run
    each, against one unperturbed run), the moves of all of them, and the series whose own perturbation moved nothing."""
    from .models._history import history_spec_of  # models import this module

    m = model() if callable(model) and not hasattr(model, "run") else model
    spec = history_spec_of(m, target)
    series = [name for name in spec.names if name in names]
    common = (run_day, target, horizon, freq, information_set, availability, span)
    original, *_ = _issue_day_forecasts(model, data, *common)
    moved: set = set()
    both = set(original)
    largest = 0.0
    counts: Dict[str, int] = {}
    unchecked: List[str] = []
    blind = []
    for name in series:
        perturbed, found, missed = perturb_recent(
            data, run_day, information_set, freq, [name], seed, first_row_day=first_row_day
        )
        counts.update(found)
        unchecked.extend(missed)
        if not found.get(name):
            continue  # nothing fresh to perturb: no evidence either way
        other, *_ = _issue_day_forecasts(model, perturbed, *common)
        changed, change, _ = _compare_forecasts(original, other, atol)
        moved.update(changed)
        both &= set(other)
        largest = max(largest, change)
        if not changed:
            blind.append(name)
    return ControlResult(
        run_day=run_day,
        forecasts=len(original),
        moved=sorted(moved),
        max_abs_change=largest,
        perturbed=counts,
        unchecked=sorted(set(unchecked)),
        unmoved=sorted(both - moved),
        blind=blind,
    )


@dataclass
class CoBatchResult:
    """Outcome of `co_batch_probe`: the leakage probe (and control) over a run widened to the other items of the call that
    predicts the issue day's forecasts, so that the call holds them."""

    run_day: str
    # The other issue days of the call, as the model names them and the frame has them; None for a model that names none
    # (no `_co_batch_items`), whose check could not run.
    items: Optional[List[str]]
    probe: Optional[ProbeResult] = None  # None when nothing ran: no other item the frame has
    control: Optional[ControlResult] = None  # the positive control over the same widened run, when it ran
    # The issue day's forecasts on the frame cut after its date against the whole frame's (`issue_day_cut`), for a model that
    # predicts its refit blocks in one call (`cuts_at_issue_day`); None when it didn't run.
    cut: Optional["CutResult"] = None

    @property
    def ok(self) -> bool:
        return (self.probe is None or self.probe.ok) and (self.cut is None or self.cut.ok)

    @property
    def moved(self) -> List[Tuple[int, int]]:
        """(period, horizon) of the issue day's forecasts that moved under the widened probe or with the cut frame, or exist in
        only one of their runs, each once."""
        keys = set()
        for result in (self.probe, self.cut):
            if result is not None:
                keys.update(result.changed, result.missing)
        return sorted(keys)


@dataclass
class CutResult:
    """Outcome of `issue_day_cut` (and `target_day_cut`): the issue day's forecasts on the frame cut after its local date (after
    what each forecast reads), as a run on that day has it, against those on the whole frame, compared at atol 0 (equal, or NaN in
    both)."""

    run_day: str
    forecasts: int
    changed: List[Tuple[int, int]]  # (period, horizon)
    max_abs_change: float
    missing: List[Tuple[int, int]] = field(default_factory=list)  # forecasts present in only one of the runs
    # Where the cut frames end, as a message says it: "on" the issue day's date, or after what each forecast of a model that
    # reads history reads.
    ends: str = ""

    @property
    def ok(self) -> bool:
        return not self.changed and not self.missing


def cuts_at_issue_day(model: Any) -> bool:
    """Whether the co-batch check also compares the issue day's forecasts on the frame cut after its date (`issue_day_cut`):
    for a model that predicts its refit blocks in one call (`_predict_block`, and `_predicts_together` True). Its call holds the
    issue rows of the block's later days that the frame has, which a run on the issue day doesn't have, so a forecast that
    depends on which rows its call holds, not only on their values, would differ between a backtest and that run. A row of a
    later day enters such a call only as an issue row, so the cut frame takes away nothing the forecast may read. A model of day
    or pooled units (`_unit_kind`, `models._units`) predicts its blocks in one call the same way, and is cut too. A model that
    reads history (`models._history.reads_history`) never is, whatever else it declares: its forecasts read the target day's
    known-ahead values on later rows, which this cut would take away, and `target_day_cut` cuts its frame instead, after what
    each forecast of the day reads (which takes a later day's rows out of a row-block call too: they hold no input there)."""
    from .models._history import reads_history  # models import this module

    m = model() if callable(model) and not hasattr(model, "run") else model
    if reads_history(m):
        return False
    blocks = getattr(type(m), "_predict_block", None) is not None or getattr(m, "_unit_kind", "period") != "period"
    return blocks and bool(getattr(m, "_predicts_together", False))


def cuts_after_target_day(model: Any) -> bool:
    """Whether the co-batch check compares the issue day's forecasts at each horizon on the frame of what they read
    (`target_day_cut`): for every model that reads history (`models._history.reads_history`), whatever else it declares."""
    from .models._history import reads_history  # models import this module

    m = model() if callable(model) and not hasattr(model, "run") else model
    return reads_history(m)


def issue_day_cut(
    model: Any,
    frame: pd.DataFrame,
    run_day: str,
    information_set: InformationSet,
    target: str = "price",
    horizon: Optional[int] = None,
    freq: str = "1h",
    availability: str = "raise",
    horizons: Optional[Sequence[int]] = None,
    runs_in: Optional[Callable[[], Any]] = None,
) -> CutResult:
    """`run_day`'s forecasts on `frame` cut after `run_day`'s local date (the rows a run on that day has) against those on the
    whole `frame`, each run as the probe runs it (to the end of a refit block, or of the frame), at atol 0. A model whose block
    call has one shape, with each forecast taken from its own row and the earlier days in the same places, gives the same bits
    on both; one whose forecast depends on how many rows, or which, its call holds doesn't. `runs_in`, a zero-argument factory
    of a context manager, wraps each run (`verify` seeds the global generators alike for both). `model` is a model instance or
    a zero-argument factory, a fresh copy for each run."""
    run_day = str(pd.Timestamp(run_day).date())
    local = frame.index.tz_localize(None) if frame.index.tz is not None else frame.index
    cut = frame[np.asarray(local.normalize() <= pd.Timestamp(run_day))]
    common = (run_day, target, _horizon_arguments(horizon, horizons, 1), freq, information_set, availability)
    forecasts = []
    for each in (frame, cut):
        if runs_in is None:
            forecasts.append(_issue_day_forecasts(model, each, *common)[0])
        else:
            with runs_in():
                forecasts.append(_issue_day_forecasts(model, each, *common)[0])
    whole, alone = forecasts
    keys = sorted(whole.keys() & alone.keys())
    changed = [key for key in keys if not (whole[key] == alone[key] or (pd.isna(whole[key]) and pd.isna(alone[key])))]
    finite = [abs(whole[key] - alone[key]) for key in changed if np.isfinite(whole[key]) and np.isfinite(alone[key])]
    max_abs = max(finite, default=float("inf") if changed else 0.0)
    return CutResult(run_day, len(whole), changed, max_abs, sorted(whole.keys() ^ alone.keys()), f"on {run_day}")


@contextlib.contextmanager
def _without_shortfall_warnings():
    """No warning of planned forecasts not produced, for a cut frame's runs (`target_day_cut`), whose later days have no data
    by design: `models.base.quiet_shortfall`, in this thread or task alone; a run that produces nothing still raises."""
    from .models.base import quiet_shortfall  # models import this module

    with quiet_shortfall():
        yield


def _history_cut(
    model: Any, frame: pd.DataFrame, run_day: str, information_set: InformationSet, target: str, freq: str, horizon: int
) -> pd.DataFrame:
    """The frame a run on `run_day` has for the forecast at `horizon` of a model that reads history: every row up to D, and after
    D only what that forecast reads (`models._history`): its known-ahead series to the end of D + h, its target and past series
    to the target's cut (D + 1 for a late issue time), every other value NaN, and no row after them. So the call of D's forecast
    holds no later day whose series aren't all there as of D: a later day's context and its known-ahead values beyond what D's
    forecast reads are gone, and a later day's row has no input."""
    from .models._history import history_batch, history_spec_of  # models import this module

    m = model() if callable(model) and not hasattr(model, "run") else model
    spec = history_spec_of(m, target)
    (item,) = history_batch(spec, frame, [run_day], information_set, freq, horizon, arrays=False).items
    day = pd.Timestamp(run_day)
    reach = pd.Timestamp(item.newest[spec.target].date())  # a step of the frame's zone: its local date
    last = day + pd.Timedelta(days=horizon) if spec.known_ahead else max(day, reach)
    local = frame.index.tz_localize(None) if frame.index.tz is not None else frame.index
    dates = local.normalize()
    cut = frame[np.asarray(dates <= last)].copy()
    after = np.asarray(dates[dates <= last] > day)
    later = np.asarray(dates[dates <= last] > reach)
    for name in cut.columns:
        column = cut[name]
        if name in spec.known_ahead or not pd.api.types.is_numeric_dtype(column) or pd.api.types.is_bool_dtype(column):
            continue
        mask = after & later if name in (spec.target, *spec.past) else after
        if mask.any():
            cut[name] = column.astype(float).where(~mask)
    return cut


def target_day_cut(
    model: Any,
    frame: pd.DataFrame,
    run_day: str,
    information_set: InformationSet,
    target: str = "price",
    horizon: Optional[int] = None,
    freq: str = "1h",
    availability: str = "raise",
    horizons: Optional[Sequence[int]] = None,
    runs_in: Optional[Callable[[], Any]] = None,
) -> CutResult:
    """`run_day`'s forecasts at each horizon h of a model that reads history on the frame a run on D has for them (`_history_cut`:
    the rows up to D, and after D only what the forecast at h reads), against those of the whole `frame`, at atol 0, each run as
    the probe runs it (`issue_day_cut`'s comparison). A forecast that moves there depends on where the frame ends (the other days
    of its call, a statistic over them), so a backtest and a run on the day differ. A univariate model's frame ends on D (or
    the target's cut), so no later day of its block keeps a complete context in its call. `runs_in` wraps each run, as there;
    the cut runs report no shortfall of forecasts, their later days having no data by design."""
    from .models._calendar import horizon_arguments, resolve_horizons  # models import this module

    run_day = str(pd.Timestamp(run_day).date())
    _, chosen = resolve_horizons(horizon, horizons, 1)

    def forecasts(each: pd.DataFrame, arguments: Dict[str, Any]) -> Dict[Tuple[int, int], float]:
        common = (run_day, target, arguments, freq, information_set, availability)
        with contextlib.ExitStack() as stack:
            if runs_in is not None:
                stack.enter_context(runs_in())
            if each is not frame:
                stack.enter_context(_without_shortfall_warnings())
            return _issue_day_forecasts(model, each, *common)[0]

    whole = forecasts(frame, horizon_arguments(*resolve_horizons(horizon, horizons, 1)))
    alone: Dict[Tuple[int, int], float] = {}
    for h in chosen:
        cut = _history_cut(model, frame, run_day, information_set, target, freq, h)
        alone.update({key: value for key, value in forecasts(cut, {"horizons": (h,)}).items() if key[1] == h})
    keys = sorted(whole.keys() & alone.keys())
    changed = [key for key in keys if not (whole[key] == alone[key] or (pd.isna(whole[key]) and pd.isna(alone[key])))]
    finite = [abs(whole[key] - alone[key]) for key in changed if np.isfinite(whole[key]) and np.isfinite(alone[key])]
    max_abs = max(finite, default=float("inf") if changed else 0.0)
    ends = "after what each forecast reads"
    return CutResult(run_day, len(whole), changed, max_abs, sorted(whole.keys() ^ alone.keys()), ends)


def frame_cut(
    model: Any,
    frame: pd.DataFrame,
    run_day: str,
    information_set: InformationSet,
    target: str = "price",
    horizon: Optional[int] = None,
    freq: str = "1h",
    availability: str = "raise",
    horizons: Optional[Sequence[int]] = None,
    runs_in: Optional[Callable[[], Any]] = None,
) -> Optional[CutResult]:
    """The co-batch check's cut frames for `model`, in one result: `issue_day_cut` where `cuts_at_issue_day`, `target_day_cut`
    where `cuts_after_target_day`, both for a model of both; None for a model of neither."""
    args = (model, frame, run_day, information_set, target, horizon, freq, availability)
    found = []
    if cuts_at_issue_day(model):
        found.append(issue_day_cut(*args, horizons=horizons, runs_in=runs_in))
    if cuts_after_target_day(model):
        found.append(target_day_cut(*args, horizons=horizons, runs_in=runs_in))
    if not found:
        return None
    if len(found) == 1:
        return found[0]
    first, second = found
    return CutResult(
        first.run_day,
        max(first.forecasts, second.forecasts),
        sorted({*first.changed, *second.changed}),
        max(first.max_abs_change, second.max_abs_change),
        sorted({*first.missing, *second.missing}),
        f"{first.ends} and {second.ends}",
    )


def co_batch_items(model: Any, run_day: str, frame: Optional[pd.DataFrame] = None) -> Optional[List[str]]:
    """The issue days whose items share the call that predicts `run_day`'s forecasts, as the model names them
    (`model._co_batch_items(run_day)`), as ISO dates, sorted, without `run_day` itself, and with `frame` only the dates it has
    rows on; None for a model without the hook. `model` is a model instance or a zero-argument factory."""
    m = model() if callable(model) and not hasattr(model, "run") else model
    hook = getattr(m, "_co_batch_items", None)
    if hook is None:
        return None
    run_day = str(pd.Timestamp(run_day).date())
    named = {str(pd.Timestamp(day).date()) for day in hook(run_day)} - {run_day}
    if frame is not None:
        local = frame.index.tz_localize(None) if frame.index.tz is not None else frame.index
        named &= {str(day.date()) for day in local.normalize().unique()}
    return sorted(named)


def co_batch_probe(
    model: Any,
    data: pd.DataFrame,
    run_day: str,
    information_set: InformationSet,
    target: str = "price",
    horizon: Optional[int] = None,
    freq: str = "1h",
    columns: Optional[Sequence[str]] = None,
    atol: float = 1e-9,
    seed: int = 0,
    rebuild: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    check_target_boundary: bool = False,
    availability: str = "raise",
    horizons: Optional[Sequence[int]] = None,
    items: Optional[Iterable[Any]] = None,
    control: bool = False,
) -> CoBatchResult:
    """The leakage probe over the issue day's whole call: `run_day`'s forecasts must not depend on anything unknown at its issue
    time, whatever else is predicted with them.

    A model that predicts several items in one call (the issue rows of a refit block, a batch of issue days) puts other days' rows
    into the call that predicts `run_day`'s forecasts, and can let those forecasts read them: an in-context model attending to
    the rows it is asked about, a batch statistic. The plain probe runs `run_day` alone (or to the end of its refit block), so a
    call made of the run's own days holds `run_day` alone there and such a read can't show. Here the model names the other items
    (`_co_batch_items(run_day)`, or `items`), the runs are widened to cover them (`leakage_probe(span=...)`) and the perturbation
    is the probe's: every cell the information set calls unknown at `run_day`'s issue time, each row read as of its own issue
    day where that is later (`perturb_unknown`), so the other items' targets, their late inputs and a later row's value from its
    own weather run (`weather_cutoff`, `as_of`) change, and with `check_target_boundary` the target from `run_day` on. What
    the information set calls known at D's issue time for that row stays: a forecast published days ahead on another item's
    row, a block's earlier days, training rows. Only `run_day`'s forecasts are compared. `control` runs the positive control over the same widened runs as well.

    For a model that predicts its refit blocks in one call (`cuts_at_issue_day`) it also compares `run_day`'s forecasts on the
    frame cut after its date with the whole frame's (`issue_day_cut`, in `cut`): its call holds the later days the frame has,
    which a run on `run_day` doesn't. For a model that reads history (`cuts_after_target_day`) the frame is cut after each
    target day instead (`target_day_cut`, `frame_cut`).

    A model without the hook gives `items=None` and runs nothing; one that names no other day the frame has gives `items=[]`
    and runs nothing either. The other arguments are `leakage_probe`'s.
    """
    run_day = str(pd.Timestamp(run_day).date())
    frame = data if rebuild is None else rebuild(data)
    if items is None:
        others = co_batch_items(model, run_day, frame)
    else:
        local = frame.index.tz_localize(None) if frame.index.tz is not None else frame.index
        present = {str(day.date()) for day in local.normalize().unique()}
        others = sorted({str(pd.Timestamp(day).date()) for day in items} & present - {run_day})
    if not others:
        return CoBatchResult(run_day, others)
    span = (others[0], others[-1])
    common = dict(freq=freq, columns=columns, atol=atol, seed=seed, rebuild=rebuild, availability=availability)
    common.update(horizon=horizon, horizons=horizons, span=span)
    probe = leakage_probe(
        model, data, run_day, information_set, target, check_target_boundary=check_target_boundary, **common
    )
    moved = positive_control(model, data, run_day, information_set, target, **common) if control else None
    cut = frame_cut(model, frame, run_day, information_set, target, horizon, freq, availability, horizons=horizons)
    return CoBatchResult(run_day, others, probe, moved, cut)


def _horizon_arguments(horizon: Optional[int], horizons: Optional[Sequence[int]], default: int) -> Dict[str, Any]:
    """`{"horizon": H}` for every horizon from 1 to H, else `{"horizons": (...)}`: what a probe passes on to `run`."""
    from .models._calendar import horizon_arguments, resolve_horizons  # models import this module

    return horizon_arguments(*resolve_horizons(horizon, horizons, default))


def _newest_target_of(model: Any) -> str:
    """The `newest_target` of a model instance or of one a zero-argument factory makes; `"d-1"` for one that has none."""
    m = model() if callable(model) and not hasattr(model, "run") else model
    return getattr(m, "newest_target", "d-1")


def _flags_of(model: Any, horizon: Dict[str, Any]) -> List[str]:
    """The `zero_when` columns (`models._row_target`) of a model instance or of one a zero-argument factory makes, at each horizon
    a probe runs (`horizon`, what `_horizon_arguments` gives); empty for a model without `zero_when`. A 0/1 flag gets no noise: a
    run refuses any other value on an issue row, and only flipping it (`flip`) probes it."""
    from .models._row_target import zero_columns  # models import this module

    m = model() if callable(model) and not hasattr(model, "run") else model
    if getattr(m, "zero_when", None) is None:
        return []
    horizons = horizon.get("horizons") or range(1, horizon["horizon"] + 1)
    return zero_columns(m, horizons)


def _without(columns: Optional[Sequence[str]], data: pd.DataFrame, flags: Sequence[str]) -> Optional[List[str]]:
    """`columns` (every column of `data` if None) less `flags`; `columns` as given without flags."""
    if not flags:
        return columns
    return [name for name in (data.columns if columns is None else columns) if name not in flags]


def _flags_put_back(perturbed: pd.DataFrame, base: pd.DataFrame, flags: Sequence[str]) -> pd.DataFrame:
    """`perturbed` with each of `flags` it has as `base` holds it: a flag `rebuild` built from perturbed source columns is put
    back, so no noise reaches it."""
    present = [name for name in flags if name in perturbed.columns and name in base.columns]
    if not present:
        return perturbed
    out = perturbed.copy()
    for name in present:
        out[name] = base[name].to_numpy()
    return out


def _with_training_data_of(
    base: pd.DataFrame, perturbed: pd.DataFrame, run_day: str, target: Optional[str], first_row_day: int = -1
) -> pd.DataFrame:
    """A copy of `perturbed` with `target` (unless None) and every row dated before D + `first_row_day` (D-1 by default) taken
    from `base`."""
    out = perturbed.copy()
    local = out.index.tz_localize(None) if out.index.tz is not None else out.index
    older = np.asarray(local.normalize() < pd.Timestamp(run_day) + pd.Timedelta(days=first_row_day))
    out.loc[older, base.columns] = base.loc[older, base.columns]
    if target is not None and target in base.columns:
        out[target] = base[target]
    return out


def perturb_unknown(
    data: pd.DataFrame,
    run_day: str,
    information_set: InformationSet,
    freq: str = "1h",
    columns: Optional[Sequence[str]] = None,
    seed: int = 0,
    target_from_issue: Optional[str] = None,
    flip: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
    """Copy of `data` where every numeric value not known at issue time on `run_day` is replaced by noise.

    `flip` replaces each such value v by 1 - v instead (a missing one stays missing): the perturbation of a 0/1 flag, whose
    every value noise would make "not 1" (a model's `zero_when`, `models._row_target`).

    Each row is read as of its own issue day where that is later than `run_day`: an `as_of` or `weather_cutoff` value on a row
    after D comes from that row's own run, so it is unknown at D's issue time (rows up to D are read on D). Every other kind
    gives the same verdict either way.

    `target_from_issue`, when given a column name, additionally perturbs every row of that column dated
    `run_day` or later, regardless of its own Availability rule: the model's own boundary (training targets
    end at D-1) is a deliberate, conservative choice, and can be stricter than the column's factual
    publication time (e.g. a day-ahead price is, in reality, already published before its delivery day).
    """
    index = data.index
    step = _STEP_MINUTES[freq]
    daily = _daily_rows(index, freq)
    ctx, day, minute, first = _rows_around_issue(index, run_day, information_set)
    rng = np.random.default_rng(seed)
    # A row after D holds what was out at its own issue day: an `as_of` or `weather_cutoff` value comes from that day's
    # run, so a later row's is unknown on D though the same value read on D's row through a shift would be known. Rows up
    # to D are read on D, as they always were.
    own_day = np.maximum(day, 0)

    out = data.copy()
    counts: Dict[str, int] = {}
    unchecked: List[str] = []
    for name in columns if columns is not None else data.columns:
        if not pd.api.types.is_numeric_dtype(data[name]) or pd.api.types.is_bool_dtype(data[name]):
            continue
        col = information_set._resolve_for(name, step)
        if col.rule is None:
            unchecked.append(name)
            continue
        information_set._check_source_zone(col, index)
        available = _available_array(col, col.rule, day, minute, step, ctx, first, own_day, daily=daily)
        mask = available > ctx.issue_minute
        if target_from_issue is not None and name == target_from_issue:
            mask = mask | (day >= 0)
        if mask.any():
            values = out[name].to_numpy(dtype=float, copy=True)
            if flip:
                values[mask] = 1.0 - values[mask]
            else:
                values[mask] = 1e6 + rng.normal(0.0, 1e4, int(mask.sum()))
            out[name] = values
            counts[name] = int(mask.sum())
    _report_unchecked(unchecked, information_set)
    return out, counts, unchecked


def perturb_recent(
    data: pd.DataFrame,
    run_day: str,
    information_set: InformationSet,
    freq: str = "1h",
    columns: Optional[Sequence[str]] = None,
    seed: int = 0,
    first_row_day: int = -1,
) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
    """Copy of `data` where the freshest values known at issue time on `run_day` are replaced by noise.

    The perturbation of `positive_control`, on the other side of the boundary from `perturb_unknown`. That boundary
    moves back one local day, to the start of D-1: a value is replaced if it is known at the issue time on D, and it is
    delivered on D-1 or later, or first known on D-1 or later. So `price_d-1` is replaced on D's row, although the price
    of D-1 was published on D-2, before an issue time simply moved back a day: its delivery day puts it in. So is
    today's `load_forecast_d+1`, and a slow series' newest value (`x_d-3` published on D-1). `x_d-2` of a day-ahead
    series is not, nor anything older.

    Only rows dated D-1 or later change: older rows are what the forecast is trained on, and an outlier there would
    move a fit whatever the forecast's own inputs are. `first_row_day=0` changes only rows dated D or later, for a model with
    `newest_target="published"`, which may train on the row of D-1. A column with a `static` rule is left alone: a calendar value
    can't be known late, so perturbing it says nothing about the boundary. A missing value stays missing. Columns
    without a rule are left alone and listed, as in `perturb_unknown`.
    """
    index = data.index
    step = _STEP_MINUTES[freq]
    daily = _daily_rows(index, freq)
    ctx, day, minute, first = _rows_around_issue(index, run_day, information_set)
    # Midnight of D-1 in minutes since midnight of D: -1440, or -1380 and -1500 if D-1 had a clock change.
    yesterday = float(_to_instants(-1440.0, ctx.data_zone, ctx)[0])
    rng = np.random.default_rng(seed)

    out = data.copy()
    counts: Dict[str, int] = {}
    unchecked: List[str] = []
    for name in columns if columns is not None else data.columns:
        if not pd.api.types.is_numeric_dtype(data[name]) or pd.api.types.is_bool_dtype(data[name]):
            continue
        col = information_set.resolve(name)
        if col.rule is None:
            unchecked.append(name)
            continue
        if col.rule.kind == "static":
            continue
        information_set._check_source_zone(col, index)
        # Known at the issue time on D, a row after D read at its own issue day, as in `perturb_unknown`.
        known = _available_array(col, col.rule, day, minute, step, ctx, first, np.maximum(day, 0), daily=daily)
        known = known <= ctx.issue_minute
        # `issue_day=day` reads an `as_of` or `weather_cutoff` value at its own row's run, not at D's, which is all
        # `perturb_unknown` needs to know whether a run is out yet: here it tells yesterday's run from older ones.
        published = _available_array(col, col.rule, day, minute, step, ctx, first, issue_day=day, daily=daily)
        delivered = _available_array(col, _DELIVERED, day, minute, step, ctx, first, daily=daily)
        values = out[name].to_numpy(dtype=float, copy=True)
        fresh = (day >= first_row_day) & ((published > yesterday) | (delivered > yesterday))
        mask = known & fresh & np.isfinite(values)
        if mask.any():
            values[mask] = 1e6 + rng.normal(0.0, 1e4, int(mask.sum()))
            out[name] = values
            counts[name] = int(mask.sum())
    _report_unchecked(unchecked, information_set)
    return out, counts, unchecked


class StepsReader:
    """The steps of a frame's index (instants, each a row's) read for several issue days at once: when the value of a column at
    a step is known, and each day's issue time, on the scale of minutes since the first issue day's midnight in the data's zone.

    This is the per-cell reading `perturb_unknown` (and so `verify`) perturbs by, for many days in one call: a step is read as of
    its own issue day where that is later than the day's (an `as_of` or `weather_cutoff` value of a later row comes from that
    row's own run), every time through `_available_array`, on the issue clock `violations` reads (`_IssueClock`). Times are
    whole minutes or fractions with a short binary expansion, so a common origin changes no comparison. A sequence model's
    context is cut here (`models._history`)."""

    def __init__(self, information_set: InformationSet, steps: pd.DatetimeIndex, days: Sequence[Any], freq: str = "1h"):
        if not len(days):
            raise ValueError("a StepsReader reads steps for one issue day at least")
        self._info = information_set
        self._steps = steps
        self._step = _STEP_MINUTES[freq]
        self._daily = _daily_rows(steps, freq)
        days = pd.DatetimeIndex([pd.Timestamp(day) for day in days])
        self._clock = information_set._issue_clock(days, steps, freq)
        _, self._day, self._minute, self._first = _rows_around_issue(steps, str(days[0].date()), information_set)
        self._columns: Dict[str, ResolvedColumn] = {}

    def column(self, name: str) -> ResolvedColumn:
        """`name` resolved; InformationSetError for a column without a rule, whatever `unknown` says: the cut decides numbers."""
        col = self._columns.get(name)
        if col is None:
            col = self._info._resolve_for(name, self._step)
            if col.rule is None:
                raise InformationSetError(
                    f"'{name}' has no availability rule, and a sequence model's context is cut where its rule says the values "
                    f"are known: declare it, e.g. InformationSet(rules={{'{name}': Availability.realised('1h')}})."
                )
            self._info._check_source_zone(col, self._steps)
            self._columns[name] = col
        return col

    def available(self, name: str, rows: np.ndarray, days: np.ndarray) -> np.ndarray:
        """The instant the value of `name` at each step `steps[rows]` is known, read for the issue day `days` (positions in the
        reader's days) of each, in minutes since the first issue day's midnight."""
        col = self.column(name)
        rows, days = np.asarray(rows, dtype=np.int64), np.asarray(days, dtype=np.int64)
        day = self._day[rows]
        issue_day = np.maximum(day, self._clock.offset[days])  # a row after the issue day: its own day's reading
        out = _available_array(
            col,
            col.rule,
            day,
            self._minute[rows],
            self._step,
            self._clock.ctx,
            self._first[rows],
            issue_day,
            self._daily,
        )
        return np.asarray(out, dtype=float)

    def issue(self, days: np.ndarray) -> np.ndarray:
        """The issue instant of each of these issue days (positions), on the same scale."""
        return self._clock.issue[np.asarray(days, dtype=np.int64)]

    def since_midnight(self, minutes: float, day: int) -> float:
        """An instant on the reader's scale as minutes since the midnight of issue day `day` (a position)."""
        return self._clock.minute_of_day(minutes, day)

    def label(self, minutes: float, day: int) -> str:
        """An instant on the reader's scale as the data zone's clock reads it from issue day `day`, as `D+1 18:00`."""
        return self._clock.label(float(minutes), day)


@dataclass(frozen=True, eq=False)
class StepsKnown:
    """When the values of one column at some steps are known, and the issue time, on one issue day (`steps_known`): minutes since
    that day's midnight in the data's zone."""

    available: np.ndarray
    issue: float
    _reader: StepsReader = field(repr=False, compare=False)

    @property
    def known(self) -> np.ndarray:
        """Whether each value is known at the issue time: exactly the cells `perturb_unknown` leaves as they are."""
        return ~(self.available > self.issue)

    def newest(self) -> int:
        """The position of the newest step known with every step before it: one before the first unknown step, -1 if the first
        is unknown, the last if none is. Under a rule over time a newer value can be out before an older one; it stays after the
        cut."""
        unknown = np.flatnonzero(~self.known)
        return len(self.available) - 1 if not len(unknown) else int(unknown[0]) - 1

    def label(self, minutes: float) -> str:
        """Minutes since the issue day's midnight as the data zone's clock reads them, as `D+1 18:00`."""
        return self._reader.label(float(minutes), 0)


def steps_known(
    information_set: InformationSet, name: str, steps: pd.DatetimeIndex, run_day: Any, freq: str = "1h"
) -> StepsKnown:
    """When the value of the column `name` at each of `steps` (instants of the data's index) is known, and the issue time on
    `run_day`: `StepsReader` for one day, the reading of `perturb_unknown` for these steps. Raises InformationSetError for a
    column without a rule."""
    reader = StepsReader(information_set, steps, [str(pd.Timestamp(run_day).date())], freq)
    rows = np.arange(len(steps))
    available = reader.available(name, rows, np.zeros(len(steps), dtype=np.int64))
    return StepsKnown(available, float(reader.issue(np.zeros(1, dtype=np.int64))[0]), reader)


def _has_rule(information_set: InformationSet, name: Any, freq: str) -> bool:
    return isinstance(name, str) and information_set._resolve_for(name, _STEP_MINUTES[freq]).rule is not None


def _daily_rows(index: pd.DatetimeIndex, freq: str) -> bool:
    """Whether the rows of a frame perturbed for a `freq` run are a daily frame's, each a whole local date (decision 12 reads
    their plain `realised` values so). With `rebuild` a daily model's probe perturbs the source it is built from, whose rows
    are hourly slots, read as such, and then the daily frame `rebuild` makes of it, read as daily rows (`leakage_probe`)."""
    if freq != _DAILY or not len(index):
        return False
    local = index.tz_localize(None) if index.tz is not None else index
    return not local.normalize().has_duplicates


def _rows_around_issue(
    index: pd.DatetimeIndex, run_day: str, information_set: InformationSet
) -> Tuple[_Context, np.ndarray, np.ndarray, np.ndarray]:
    """The context with the issue time on `run_day`, and each row's local day after `run_day`, minute of its day and
    whether it is not the second occurrence of a repeated autumn wall time."""
    # Every zone conversion reads the UTC offsets at the instant it converts (a value's own shifted delivery instant
    # for its delivery day, its publication instant for the way back), not at run_day's noon or at the row's own
    # date. They differ across a DST switch whenever the data zone and a rule's zone don't switch together, e.g. a
    # UTC or fixed-offset (benchmark source) index against Europe/Brussels rules.
    zone = information_set._issue_zone(index)
    ctx = _Context(pd.Timestamp(run_day), information_set._data_zone(index), naive=index.tz is None, issue_zone=zone)
    issue = _issue_minutes(np.array([float(_minutes(information_set.issue_time))]), zone, ctx)
    ctx = replace(ctx, issue_minute=float(issue[0]))
    local = index.tz_localize(None) if index.tz is not None else index
    day = np.asarray((local.normalize() - pd.Timestamp(run_day)) // pd.Timedelta(days=1), dtype=np.int64)
    minute = np.asarray(local.hour * 60 + local.minute + local.second / 60.0, dtype=float)
    # Which occurrence of a repeated autumn wall time each row is; a day shift keeps it, like shift_calendar_days.
    first = np.ones(len(index), dtype=bool)
    if index.tz is not None:
        first = index.asi8 == local.tz_localize(index.tz, ambiguous=first, nonexistent="NaT").asi8
    return ctx, day, minute, first


def _report_unchecked(unchecked: List[str], information_set: InformationSet) -> None:
    if not unchecked:
        return
    message = f"leakage_probe: no availability rule for {unchecked}; left unperturbed"
    if information_set.unknown == "raise":
        raise InformationSetError(message + ". Declare rules or pass unknown='warn'.")
    if information_set.unknown == "warn":
        warnings.warn(message, stacklevel=3)


def _available_array(
    col: ResolvedColumn,
    rule: Rule,
    day: np.ndarray,
    minute: np.ndarray,
    step: int,
    ctx: _Context,
    first: np.ndarray,
    issue_day: Union[int, np.ndarray] = 0,
    daily: bool = False,
) -> np.ndarray:
    """Vectorised `_available` for rows at (ctx.day + day, minute); the value is `col` read through `rule`.

    `day` and `minute` are the row's local day and wall-clock minute in the data zone; the result is the instant the
    value is known, in minutes since midnight of `ctx.day` (see `_Context`). `issue_day` is each row's issue day D
    in days after `ctx.day` (`violations` checks many issue days at once; `perturb_unknown` one, `ctx.day` itself).
    `first` marks rows that are not the second occurrence of a repeated autumn wall time.

    `daily` says the rows are a daily frame's (`freq="1D"`), each a whole local date stamped at one time of day. A plain
    `realised` value of such a row is the date's (a daily maximum, say), known `delay` after the end of the local day the
    shifts reach, not after an hour from the stamp: else `actual_daily_max` would read as known at 13:00 on its own date. A
    plain `published` or `weekly` one is read at that date's last slot, or at the stamp if it is later, so a date spanning two
    of the rule's delivery days isn't read on the earlier. One with an `h`, `min` or `s` shift is an hourly slot, `x_h-1` the hour before the stamp. This
    holds for the column named alone; what a profile or a day aggregate is built from is read at its slot, as on hourly rows.

    A rule with a validity period, or a `RuleChain`, holds for the values whose day is in its period: the day the value is
    delivered (a date of the rule's zone, or of `day_zone`, for `published`, of its week's zone for `weekly`, of the data's
    zone for `realised`, `weather_run` and `static`), and for `as_of` and `weather_cutoff` the day it is read on. A value no rule holds for is never known: `inf`.
    Each check, `ctx`, `explain`, the probes and `newest_targets` read through here, so they agree on every value. A rule with
    no period is read as it always was.
    """
    if col.built_from is not None:
        # A profile column holds, on every row of a local day, the column it was built from at its slot of that day:
        # `x_h05_d-1` on row D is `x` at 05:00 on D-1 (the shifts after the slot pick the day, and none of them is a
        # source's lead), `x_d+1_h05` is `x_d+1` at 05:00 on D. A repeated autumn slot is its first occurrence, the
        # one DailyProfileTransformer copies; a slot skipped in spring moves forward to the switch.
        # A day aggregate holds a statistic of that column over every slot of the day (23, 24 or 25 hours on a DST
        # index), so it is known when the latest of them is: the day's last slot (23:00 for hourly data), on the day
        # the shifts after the aggregate reach. Each rule's availability never decreases with the slot's time, so no
        # earlier slot is known later; a repeated last slot is its second occurrence.
        reached = _shifted_day(col.shifts, day, minute, first, ctx)
        if col.slot is None and over_time(rule):
            # Under rules over time an earlier slot can be known after the last (a lag that shortens, a gap, a day of one zone
            # that reaches two of another's): the aggregate is known when every slot of its day is.
            latest = np.full(len(reached), -np.inf)
            second = np.zeros(len(reached), dtype=bool)
            for slot in np.arange(0.0, 1440.0, step):
                read = np.full(len(reached), slot)
                latest = np.maximum(
                    latest,
                    _available_array(col.built_from, rule, reached, read, step, ctx, second, issue_day + reached - day),
                )
            return latest
        slot, length = col.slot or (1440.0 - step, step)
        slots = np.full(len(reached), float(slot))
        occurrence = np.full(len(reached), col.slot is not None)
        return _available_array(
            col.built_from, rule, reached, slots, length, ctx, occurrence, issue_day + reached - day
        )
    rules = rule_elements(rule)
    if len(rules) == 1 and not rules[0].bounded:
        return _rule_available(col, rules[0], day, minute, step, ctx, first, issue_day, daily)
    out = _over_time_available(col, rules, day, minute, step, ctx, first, issue_day, daily)
    if daily and all(shift.unit == "d" for shift in col.shifts) and any(one.kind in _WHOLE_DATE for one in rules):
        # A plain published or weekly value of a daily row is the whole date's, read at its last slot. Under rules over time
        # an earlier slot can be known later (the date reaching two delivery days of the rule's zone, one under a slower rule):
        # it is known when every slot is.
        for slot in np.arange(0.0, 1440.0, step):
            read = np.full(len(day), slot)
            out = np.maximum(
                out, _over_time_available(col, rules, day, minute, step, ctx, first, issue_day, daily, read)
            )
    return out


# The kinds whose value on a daily row is the whole date's, delivered in every slot of it.
_WHOLE_DATE = ("published", "weekly")


def _over_time_available(
    col: ResolvedColumn,
    rules: Sequence[Availability],
    day: np.ndarray,
    minute: np.ndarray,
    step: int,
    ctx: _Context,
    first: np.ndarray,
    issue_day: Union[int, np.ndarray],
    daily: bool,
    slot: Optional[np.ndarray] = None,
) -> np.ndarray:
    """`_available_array` of a plain column under rules with validity periods: each holds for the values whose day is in its
    period, and a value none holds for is never known (`inf`). `slot` (wall-clock minutes) reads a published or weekly rule's
    value there, as an hourly slot at its second occurrence, for one slot of a daily row; the other kinds read the row as it is."""
    # A week (or delivery day) of the data's dates can be read in two zones (`_data_weeks`): each reading picks its rules, and
    # the later counts.
    data_week = any(_reads_data_dates(one) for one in rules)
    latest = None
    for week in _data_weeks(ctx) if data_week else (None,):
        out = np.full(len(day), np.inf)  # what no rule holds for is never known
        held = np.zeros(len(day), dtype=bool)
        for one in rules:
            if slot is not None and one.kind in _WHOLE_DATE:
                read = (slot, np.zeros(len(day), dtype=bool), False)
            else:
                read = (minute, first, daily)
            known, on = _rule_available(
                col, one, day, read[0], step, ctx, read[1], issue_day, read[2], with_day=True, data_week=week
            )
            inside = one._holds_on(on, ctx.day)
            # Periods are disjoint dates, but two rules in zones a day apart could both claim a value: the later time counts.
            out = np.where(inside, np.where(held, np.maximum(out, known), known), out)
            held |= inside
        latest = out if latest is None else np.maximum(latest, out)
    return latest


def _rule_available(
    col: ResolvedColumn,
    rule: Availability,
    day: np.ndarray,
    minute: np.ndarray,
    step: int,
    ctx: _Context,
    first: np.ndarray,
    issue_day: Union[int, np.ndarray] = 0,
    daily: bool = False,
    with_day: bool = False,
    data_week: Optional[str] = None,
) -> Any:
    """`_available_array` of a plain column (no profile, no day aggregate) under one rule, its validity period aside. With
    `with_day` also the day the period is read on, in days after `ctx.day`: the value's delivery day, or for `as_of` and
    `weather_cutoff` the day it is read on. `data_week` reads a week (or a delivery day) of the data's dates in that one zone,
    not in each of `_data_weeks`."""
    if rule.kind == "static":
        known = np.full(len(day), -np.inf)
        return (known, _shifted_day(col.shifts, day, minute, first, ctx)) if with_day else known
    if rule.kind == "as_of":
        # Fixed to a time on the day the value is read from: the issue day D itself, moved by every shift after a
        # leading `_d+k` (see `_read_day_offset`), but not by the leading `_d+k`, the row's own day or minute.
        read_day = issue_day + _read_day_offset(col.shifts, day, minute, ctx, first)
        at = np.asarray(read_day, dtype=float) * 1440.0 + _minutes(rule.time)
        known = _to_instants(at, rule.timezone or ctx.data_zone, ctx, _second(at))
        return (known, np.asarray(read_day, dtype=float)) if with_day else known
    if rule.kind == "weather_cutoff":
        # Fixed to `run`:00 UTC plus `delay` on the day the value is read from, same as `as_of` above; the delay is
        # elapsed time after the run and may run into the next day. The run is a UTC hour (`Availability` sees to it).
        read_day = issue_day + _read_day_offset(col.shifts, day, minute, ctx, first)
        run = _to_instants(np.asarray(read_day, dtype=float) * 1440.0 + _minutes(rule.time), "UTC", ctx)
        known = run + _parse_delay(rule.delay)
        return (known, np.asarray(read_day, dtype=float)) if with_day else known
    shifts = list(col.shifts)
    if rule.kind == "weather_run" and shifts and shifts[0].unit == "d" and shifts[0].offset > 0:
        shifts = shifts[1:]
    point, wall = _shift_rows(day, minute, shifts, ctx, first)  # delivery instant of the value (weather: the row time)
    if rule.kind in ("realised", "weather_run"):
        if rule.kind == "weather_run":
            known = point + _parse_delay(rule.delay)
        elif daily and all(shift.unit == "d" for shift in shifts):
            # The value of a whole local date: known when that date ends, at the next midnight of the data zone.
            end = (np.floor(wall / 1440.0) + 1.0) * 1440.0
            known = _to_instants(end, ctx.data_zone, ctx, _second(end)) + _parse_delay(rule.delay)
        else:
            known = point + step + _parse_delay(rule.delay)
        return (known, np.floor(wall / 1440.0)) if with_day else known  # delivered on a day of the data's zone
    if daily and all(shift.unit == "d" for shift in shifts):
        # The value of a whole local date is delivered in every slot of it, so it is known when its last slot is: no earlier
        # slot is published later. A date that spans two of the rule's delivery days (Helsinki's against Brussels') is known
        # when the later is, wherever the row is stamped. A stamp after the last slot's start (23:45) is later still: the rule
        # zone's midnight can fall between them (Kathmandu's against Kolkata's), so the later of the two is read.
        last = np.floor(wall / 1440.0) * 1440.0 + 1440.0 - step
        last_point = _to_instants(last, ctx.data_zone, ctx, _second(last))
        later = last_point > point
        point = np.where(later, last_point, point)
        wall = np.where(later, last, wall)
    # The delivery day is a calendar day in the rule's own (market) zone, not in the data's: a value at
    # 23:00 UTC in winter (CET, UTC+1) delivers at 00:00 CET the next day, even though the data zone's own
    # wall clock hasn't crossed midnight yet. A weekly rule's week may be another zone's (`week_zone`): the built-in
    # A31's is the data's, each market's Monday to Sunday, while it is published at 10:00 Brussels. A published rule's delivery
    # day may be too (`day_zone`): another pipeline's forecast is for a date of its data zone, issued on its issue zone's clock.
    zone = rule.timezone or ctx.data_zone
    known = delivery_day = None
    for day_zone in _delivery_zones(rule, ctx, data_week):
        day_of = np.floor(_to_wall(point, day_zone, ctx, wall if day_zone == ctx.data_zone else None) / 1440.0)
        if rule.kind == "published":
            publication_day = day_of - rule.days_before
        else:  # weekly: published on `weekday` of the week before the delivery week
            delivery_weekday = np.mod(ctx.weekday + day_of, 7)
            publication_day = day_of - delivery_weekday - 7 + _WEEKDAYS.index(rule.weekday)
        at = publication_day * 1440.0 + _minutes(rule.time)
        reading = _to_instants(at, zone, ctx, _second(at))
        if known is None:
            known, delivery_day = reading, day_of
        else:  # two readings of a week (or day) of the data's dates: the later counts, never an early one
            later = reading > known
            known, delivery_day = np.where(later, reading, known), np.where(later, day_of, delivery_day)
    return (known, delivery_day) if with_day else known


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _delivery_zones(rule: Availability, ctx: _Context, data_week: Optional[str] = None) -> Tuple[str, ...]:
    """The zones whose local dates a rule's delivery day is read in: a weekly rule's week in `week_zone`, a published rule's day
    in `day_zone`, else in the rule's own zone; for "data" those of `_data_weeks`, or `data_week` alone. The one place either is
    read, so that a validity period's day is the same day."""
    own = rule.week_zone if rule.kind == "weekly" else rule.day_zone
    if own != DATA_ZONE:
        return (own or rule.timezone or ctx.data_zone,)
    return (data_week,) if data_week is not None else _data_weeks(ctx)


def _reads_data_dates(rule: Availability) -> bool:
    """Whether a rule's delivery day is a date of the data's zone ("data" as `week_zone` or `day_zone`)."""
    return DATA_ZONE in (rule.week_zone, rule.day_zone)


def _data_weeks(ctx: _Context) -> Tuple[str, ...]:
    """The zones a week of the data's local dates is read in. The data's zone is the market's on a frame kept in it, but a frame
    converted to another zone (UTC, a fixed offset) still holds each value of the market's date: a Warsaw hour stamped 23:00Z on
    Sunday is Monday's. So on a frame whose zone isn't the issue's, the issue zone's dates are read too and the later week counts:
    exact when the issue zone is the market's, never early, and too strict only where neither zone is the market's."""
    if ctx.issue_zone is None or ctx.issue_zone == ctx.data_zone:
        return (ctx.data_zone,)
    return (ctx.data_zone, ctx.issue_zone)


def _shifts_after_source_lead(shifts: Sequence[Shift]) -> List[Shift]:
    """`shifts` without a leading `_d+k`, which is a source's own lead (the forecast for k days ahead) and not a
    shift of the row."""
    rest = list(shifts)
    if rest and rest[0].unit == "d" and rest[0].offset > 0:
        rest = rest[1:]
    return rest


def _shift_rows(
    day: np.ndarray, minute: np.ndarray, shifts: Sequence[Shift], ctx: _Context, first: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """The rows `shifts` read the rows (ctx.day + day, minute) from: their instants (minutes since `ctx.origin`) and their
    data-zone wall-clock minutes after midnight of `ctx.day`.

    The outermost shift moves the row first (`x_d+1_h-24` is `x_d+1` 24 hours earlier). A day shift keeps the wall-clock
    time and the occurrence of a repeated autumn hour, as shift_calendar_days does, so its wall time may be one the clocks
    skip, which has no row: its instant moves forward to the switch. An hour or minute shift is elapsed time: on a 25-hour
    day 12 hours before 11:00 is 00:00 on that day, on a 23-hour day 12 hours before 12:00 is 23:00 on the day before. A
    naive index (DSTNormalizeTransformer's 24 slots a day) is shifted on its own wall clock, as LagTransformer shifts it,
    and its repeated autumn slot, which averages both hours, is complete only at the second.
    """
    wall = np.asarray(day, dtype=float) * 1440.0 + np.asarray(minute, dtype=float)
    if ctx.naive:
        wall = wall + sum(s.minutes for s in shifts)
        return _to_instants(wall, ctx.data_zone, ctx, _second(wall)), wall
    instants = _to_instants(wall, ctx.data_zone, ctx, first)
    for shift in reversed(shifts):
        if shift.unit == "d":
            occurrence = _to_instants(wall, ctx.data_zone, ctx) == instants
            wall = wall + shift.minutes
            instants = _to_instants(wall, ctx.data_zone, ctx, occurrence)
        else:
            instants = instants + shift.minutes
            wall = _to_wall(instants, ctx.data_zone, ctx)
    return instants, wall


def _second(wall: np.ndarray) -> np.ndarray:
    """Read every wall time at its second occurrence where the clocks repeat it: the later of the two."""
    return np.zeros(np.shape(np.atleast_1d(wall)), dtype=bool)


def _shifted_wall(
    shifts: Sequence[Shift], day: np.ndarray, minute: np.ndarray, first: np.ndarray, ctx: _Context
) -> np.ndarray:
    """The wall-clock minutes of `_shift_rows`; day shifts alone need no conversion."""
    if ctx.naive or all(s.unit == "d" for s in shifts):
        return np.asarray(day, dtype=float) * 1440.0 + np.asarray(minute, dtype=float) + sum(s.minutes for s in shifts)
    return _shift_rows(day, minute, shifts, ctx, first)[1]


def _shifted_day(
    shifts: Sequence[Shift], day: np.ndarray, minute: np.ndarray, first: np.ndarray, ctx: _Context
) -> np.ndarray:
    """The local day (after `ctx.day`) of the row that `shifts` read a row at (`day`, `minute`) from, every shift moving
    it: the day a profile or a day aggregate is read on."""
    return np.floor(_shifted_wall(shifts, day, minute, first, ctx) / 1440.0)


def _delivered_span(
    col: ResolvedColumn, day: np.ndarray, minute: np.ndarray, step: int, ctx: _Context
) -> Tuple[np.ndarray, np.ndarray]:
    """The first and last slot the value of `col` at the row (`day`, `minute`) was delivered in, as wall-clock minutes
    after midnight of `ctx.day`. `col` needs no rule: this is what its name says, read the way `_available_array` reads it.

    A plain column is delivered in one slot, the row's with its shifts applied. A profile holds the column it was built
    from at its slot, on the day the shifts after the slot reach, and a day aggregate that column over every slot of that
    day, so `x_day_max_d-1` runs from 00:00 to the last slot of D-1 on 23-, 24- and 25-hour days alike, and an aggregate
    of an hour lag (`x_h-12_day_max`) over two days.
    """
    day, minute = np.asarray(day), np.asarray(minute, dtype=float)
    later = np.zeros(len(day), dtype=bool)  # no rows, only (day, period): read as the row the model forecasts from
    if col.built_from is None:
        wall = _shifted_wall(col.shifts, day, minute, later, ctx)
        return wall, wall
    reached = _shifted_day(col.shifts, day, minute, later, ctx)
    if col.slot is not None:
        return _delivered_span(col.built_from, reached, np.full(len(reached), float(col.slot[0])), col.slot[1], ctx)
    start, _ = _delivered_span(col.built_from, reached, np.zeros(len(reached)), step, ctx)
    _, end = _delivered_span(col.built_from, reached, np.full(len(reached), 1440.0 - step), step, ctx)
    return start, end


def _read_day_offset(
    shifts: Sequence[Shift], day: np.ndarray, minute: np.ndarray, ctx: _Context, first: np.ndarray
) -> np.ndarray:
    """Days after (negative: before) the issue day D that an `as_of` or `weather_cutoff` column is read from.

    Such a rule is fixed to the issue day of the row a value is taken from. A leading `_d+k` is the source's own
    lead and keeps the row. Any shift after it moves the row: `x_d+1_d+1` on day D holds what `x_d+1` had on D+1
    (its run, issued on D+1), `x_d+1_d-1` what it had on D-1, and an hour or minute shift that crosses midnight
    moves it to the next or the previous day (see `_shift_rows`).
    """
    return _shifted_day(_shifts_after_source_lead(shifts), day, minute, first, ctx) - np.asarray(day)


def _fmt_minute(value: float) -> str:
    if value == -math.inf:
        return "in advance"
    if value == math.inf:
        return NEVER
    day, minute = divmod(int(round(value)), 1440)
    label = "D" if day == 0 else f"D{day:+d}"
    return f"{label} {minute // 60:02d}:{minute % 60:02d}"


@lru_cache(maxsize=4096)  # a listing asks again per horizon and list, for the same few times
def _fmt_instant(value: float, ctx: _Context) -> str:
    """`_fmt_minute` for an instant (minutes since `ctx.origin`) on the data zone's wall clock, with its UTC offset
    where that wall time occurs twice, so a value known in the second 01:00 doesn't read as before an issue at 01:30."""
    if value == -math.inf:
        return "in advance"
    if value == math.inf:
        return NEVER
    instant = _instants(np.array([value]), ctx).tz_convert(ctx.data_zone)
    text = _fmt_minute(float(_wall(instant, ctx.data_zone, ctx)[0]))
    local = instant.tz_localize(None)
    earlier = local.tz_localize(ctx.data_zone, ambiguous=np.ones(1, dtype=bool), nonexistent="NaT")
    if not earlier.equals(local.tz_localize(ctx.data_zone, ambiguous=np.zeros(1, dtype=bool), nonexistent="NaT")):
        offset = instant[0].strftime("%z")
        text += f" (UTC{offset[:3]}:{offset[3:]})"
    return text


def _ranges(values: Sequence[int]) -> str:
    if not values:
        return "-"
    parts, start, prev = [], values[0], values[0]
    for v in list(values[1:]) + [None]:
        if v is not None and v == prev + 1:
            prev = v
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        if v is not None:
            start = prev = v
    return ", ".join(parts)


__all__ = [
    "ASSUMPTIONS",
    "AVAILABILITY_LEVELS",
    "Availability",
    "CoBatchResult",
    "ControlResult",
    "CutResult",
    "InformationSet",
    "InformationSetError",
    "ProbeResult",
    "RuleChain",
    "TrainingRows",
    "VINTAGES",
    "co_batch_probe",
    "cuts_at_issue_day",
    "default_rules",
    "issue_day_cut",
    "leakage_probe",
    "parse_predictor",
    "perturb_recent",
    "perturb_unknown",
    "positive_control",
    "source_rules",
]
