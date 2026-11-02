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

import fnmatch
import math
import re
import threading
import warnings
from dataclasses import dataclass, field, replace
from functools import cached_property, lru_cache
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .logging import get_logger

logger = get_logger(__name__)

DEFAULT_ISSUE_TIME = "11:45"
KINDS = ("static", "realised", "published", "weekly", "weather_run", "as_of", "weather_cutoff")
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_UNKNOWN_POLICIES = ("raise", "warn", "ignore")
# What a run does with an input known after the issue time: nothing, list it and go on, or refuse before any fit.
AVAILABILITY_LEVELS = ("none", "warn", "raise")
_SUFFIX = re.compile(r"^(?P<base>.+)_(?P<unit>d|h|min|s)(?P<sign>[+-])(?P<n>\d+)$")
_UNIT_MINUTES = {"d": 1440.0, "h": 60.0, "min": 1.0, "s": 1.0 / 60.0}
_STEP_MINUTES = {"1h": 60, "15min": 15}
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
                     a value known only after its delivery day: "00:00", days_before=-1).
        weekly:      published on `weekday` at `time` for the whole following Monday-Sunday week (week-ahead load).
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
    """

    kind: str
    delay: str = "0min"
    time: str = "00:00"
    days_before: int = 0
    weekday: str = "friday"
    timezone: Optional[str] = "Europe/Brussels"
    note: str = ""

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

    @classmethod
    def static(cls, note: str = "") -> "Availability":
        return cls("static", note=note)

    @classmethod
    def realised(cls, delay: str = "1h", note: str = "") -> "Availability":
        return cls("realised", delay=delay, note=note)

    @classmethod
    def published(
        cls, time: str, days_before: int = 1, timezone: Optional[str] = "Europe/Brussels", note: str = ""
    ) -> "Availability":
        return cls("published", time=time, days_before=int(days_before), timezone=timezone, note=note)

    @classmethod
    def daily_close(cls, time: str, timezone: Optional[str] = "Europe/Brussels", note: str = "") -> "Availability":
        """A value for trading day X known at its close on X (e.g. fuel or carbon settlement prices)."""
        return cls.published(time, days_before=0, timezone=timezone, note=note)

    @classmethod
    def weekly(
        cls, weekday: str = "friday", time: str = "10:00", timezone: Optional[str] = "Europe/Brussels", note: str = ""
    ) -> "Availability":
        return cls("weekly", weekday=weekday.lower(), time=time, timezone=timezone, note=note)

    @classmethod
    def weather_run(cls, delay: str = "6h", note: str = "") -> "Availability":
        return cls("weather_run", delay=delay, note=note)

    @classmethod
    def as_of(cls, time: str, timezone: Optional[str] = "Europe/Brussels", note: str = "") -> "Availability":
        """Known at a fixed `time` on the issue day D; a leading `_d+k` doesn't change that."""
        return cls("as_of", time=time, timezone=timezone, note=note)

    @classmethod
    def weather_cutoff(cls, run: str, delay: str, note: str = "") -> "Availability":
        """An OpenMeteoSource column: known `delay` after `run` (a UTC hour "HH") on the issue day D, regardless
        of a leading `_d+k` — like `as_of`, but the fixed time is expressed as a UTC run hour plus a delay that
        may push it past 24:00 (into the next UTC calendar day), rather than a local "HH:MM" clock time."""
        return cls("weather_cutoff", time=f"{run}:00", delay=delay, timezone="UTC", note=note)

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
        zone = f" {self.timezone}" if self.timezone else ""
        if self.kind == "static":
            return "known in advance"
        if self.kind == "realised":
            return f"known {self.delay} after the end of the period"
        if self.kind == "published":
            if self.days_before == 0:
                return f"published {self.time}{zone} on the delivery day"
            if self.days_before > 0:
                return f"published {self.time}{zone} on D-{self.days_before} of delivery"
            return f"published {self.time}{zone} {-self.days_before} day(s) after delivery"
        if self.kind == "weekly":
            return f"published {self.weekday.capitalize()} {self.time}{zone} for the following week"
        if self.kind == "as_of":
            return f"known {self.time}{zone} on the issue day"
        if self.kind == "weather_cutoff":
            run = self.time.split(":", 1)[0]
            return f"known {self.delay} after the {run} UTC run on the issue day"
        return f"weather model run published {self.delay} after its start (about the row time)"


# Assumptions a user may accept explicitly. Each replaces the default rule for one pattern.
ASSUMPTIONS: Dict[str, Tuple[str, Availability, str]] = {
    "entsoe_a69_by_10": (
        "generation_*_forecast",
        Availability.published("10:00", 1, note="assumed: A69 wind/solar day-ahead forecast published by 10:00 D-1"),
        "ENTSO-E A69 day-ahead wind and solar forecasts are published by 10:00 Brussels on the day before delivery, "
        "like the day-ahead load forecast. The legal deadline is 18:00 (Reg. 543/2013 Art. 14(2)(d)), after gate "
        "closure, and ENTSO-E gives no publication time per value.",
    ),
}


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
        "price": Availability.published(
            "13:00", 1, note="A44 day-ahead prices: SDAC results within about 1 h after the 12:00 gate closure on D-1"
        ),
        "load_actual": Availability.realised(
            "1h", note="A65/A16 actual load: 1 h after the period (Reg. 543/2013 Art. 6(2)(a))"
        ),
        "load_forecast": Availability.published(
            "10:00", 1, note="A65/A01 day-ahead load forecast: 2 h before gate closure on D-1 (Art. 6(2)(b))"
        ),
        "load_forecast_daily_min": Availability.weekly(
            "friday", "10:00", note="A65/A31 week-ahead load forecast: Fridays for the next week (Art. 6(2)(c))"
        ),
        "load_forecast_daily_max": Availability.weekly(
            "friday", "10:00", note="A65/A31 week-ahead load forecast: Fridays for the next week (Art. 6(2)(c))"
        ),
        "generation_*_forecast": Availability.published(
            "18:00",
            1,
            note="A69 day-ahead wind/solar forecast: legally by 18:00 Brussels on D-1 (Art. 14(2)(d)); "
            "accept assumption 'entsoe_a69_by_10' to assume 10:00",
        ),
        "generation_*": Availability.realised("1h", note="A75 actual generation per type: 1 h after the period"),
    }
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
    from .data.sources.entsoe import EntsoeSource
    from .data.sources.open_meteo import OpenMeteoSource

    rules: Dict[str, Availability] = {}
    defaults = default_rules()
    for source in sources:
        if isinstance(source, CalendarSource) and getattr(source, "prefix", ""):
            prefix = source.prefix
            for key, rule in defaults.items():
                if rule.kind == "static":
                    rules[f"{prefix}{key}"] = rule
        elif isinstance(source, EntsoeSource) and "load" in source.type:
            # A31 week-ahead daily values are stamped at the market's own local midnight (EntsoeSource's
            # bound zone, see `_daily_zone`), not necessarily Brussels', and not necessarily this area's own
            # when it is a neighbour combined into another market's pipeline. `default_rules()`'s weekly rule
            # for these columns assumes Brussels, which is wrong for any other market — see `_daily_zone`.
            prefix = f"{source.prefix}_" if getattr(source, "prefix", "") else ""
            rule = Availability.weekly(
                "friday",
                "10:00",
                timezone=source._daily_zone(),
                note="A65/A31 week-ahead load forecast: Fridays for the next week (Art. 6(2)(c)), anchored to "
                "the market's own local day",
            )
            for name in ("load_forecast_daily_min", "load_forecast_daily_max"):
                rules[f"{prefix}{name}"] = rule
        elif isinstance(source, OpenMeteoSource):
            rule = _weather_availability(source)
            for name in _weather_column_names(source):
                rules[name] = rule
        elif isinstance(source, EpftoolboxBenchmarkSource):
            # The files are local wall-clock time, and `index_timezone` only labels that clock (one hour off the true
            # instants in summer). So a row's delivery day is the day on that same clock: the file's 23:00 on D is
            # delivered on D, which Brussels rules would read as 00:00 on D+1 all summer.
            prefix = f"{source.prefix}_" if source.prefix else ""
            zone = source.index_timezone
            rules[f"{prefix}price"] = Availability.published(
                "13:00", 1, timezone=zone, note="day-ahead price of the benchmark files, known the day before delivery"
            )
            for name in DATASETS[source.dataset]["exogenous"].values():
                rules[f"{prefix}{name}"] = Availability.published(
                    "10:00", 1, timezone=zone, note="day-ahead forecast of the benchmark files, available on d-1"
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
        run, f"{delay_h}h", note="Open-Meteo: known after the model run's typical publication delay"
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
    rule: Optional[Availability]
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
            return ResolvedColumn(name, candidate, shifts, rules[candidate], candidate, layer)
    patterns = [k for k in rules if any(ch in k for ch in "*?[")]
    for candidate, shifts in reversed(chain):
        for pattern in patterns:
            if fnmatch.fnmatchcase(candidate, pattern):
                return ResolvedColumn(name, candidate, shifts, rules[pattern], pattern, layer)
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
        issue_timezone: Zone of `issue_time`.
        rules: Overrides and additions, `{column name or glob pattern: Availability or dict}`. They take
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
        rules: Optional[Dict[str, Union[Availability, Dict[str, Any]]]] = None,
        assumptions: Optional[Sequence[str]] = None,
        unknown: str = "raise",
        use_defaults: bool = True,
    ):
        _parse_time(issue_time)
        _check_zone(issue_timezone)
        if unknown not in _UNKNOWN_POLICIES:
            raise ValueError(f"unknown must be one of {list(_UNKNOWN_POLICIES)}, got '{unknown}'")
        unknown_assumptions = [a for a in (assumptions or []) if a not in ASSUMPTIONS]
        if unknown_assumptions:
            raise ValueError(f"Unknown assumption(s) {unknown_assumptions}. Available: {list(ASSUMPTIONS)}")
        self.issue_time = issue_time
        self.issue_timezone = issue_timezone
        self.rules: Dict[str, Availability] = {k: Availability.from_dict(v) for k, v in (rules or {}).items()}
        self.assumptions: List[str] = list(assumptions or [])
        self.unknown = unknown
        self.use_defaults = use_defaults
        self._source_rules: Dict[str, Availability] = {}
        self._source_zones: Dict[str, str] = {}

    # -- configuration --------------------------------------------------------

    def with_source_rules(self, sources: Iterable[Any]) -> "InformationSet":
        """A copy that also knows the columns of these data sources (prefixes, Open-Meteo variables)."""
        sources = list(sources)
        out = self.copy()
        out._source_rules = {**self._source_rules, **source_rules(sources)}
        out._source_zones = {**self._source_zones, **_source_market_zones(sources)}
        return out

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
    ) -> None:
        """Raise InformationSetError if a predictor (or the newest training target) is not known at issue time.

        `issue_days` checks these days instead of [test_start, test_end]: a run with validation forecasts no day between
        its windows. `check_target=False` leaves the training target out, for a model whose newest target is read from its
        rule (`newest_target="published"`, see `newest_targets`): what it trains on is known by construction."""
        violations, unknown = self.violations(
            expanded, target, freq, index, test_start, test_end, issue_days, check_target=check_target
        )
        self._report_unknown(unknown, model_name)
        if violations:
            raise InformationSetError(self._format(violations, model_name))
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
    ) -> List[Violation]:
        """What `availability="warn"` lists: every use `check` refuses, and every use of a column without a rule, whatever
        `unknown` says, instead of an error. A column of a data source on an index in another zone still raises: neither
        the check nor the listing could say anything true about it. `check_target` is as in `check`."""
        found, _ = self.violations(
            expanded, target, freq, index, test_start, test_end, issue_days, check_target=check_target
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
    ) -> Tuple[List[Violation], set]:
        """All (predictor, period, horizon) uses of values not known at issue time, and unknown base columns.

        Every issue day D of the test period (or of `issue_days`) is checked, and each zone conversion uses the UTC offsets at the instant
        it converts: when the data zone and a rule's zone don't switch DST together (a UTC or fixed-offset index
        against Brussels rules), a value delivered or published across a switch from D has other offsets than D.
        A violation reports the issue day on which its value comes latest after the issue time.

        `check_target=False` has no use of the training target, for a model with `newest_target="published"`: its newest
        target is the newest its rule calls known (`newest_targets`), so it is known by construction.
        """
        step = _STEP_MINUTES[freq]
        resolved: Dict[str, ResolvedColumn] = {}
        unknown = set()
        # The newest training target is the value delivered on D-1 in the same period.
        uses = {ROLE_PREDICTOR: 0, ROLE_TARGET: -1}  # role -> the row's day relative to D
        periods: Dict[Tuple[str, str], set] = {}  # the periods reading each (name, role); the horizon doesn't matter
        for (period, _), predictors in expanded.items():
            for name in predictors:
                periods.setdefault((name, ROLE_PREDICTOR), set()).add(period)
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

        days = pd.DatetimeIndex(issue_days) if issue_days is not None else self._issue_days(index, test_start, test_end)
        if not len(days):
            return [], unknown
        clock = self._issue_clock(days, index)

        late: Dict[Tuple[str, str, int], _LatePeriod] = {}  # (name, role, period) -> its worst day
        for (name, role), rows in periods.items():
            col = resolved[name]
            if col.rule is None:
                continue
            rows = np.array(sorted(rows))
            avail = clock.available(col, rows, step, uses[role])
            for found in self._late_periods(col, rows, avail, clock, step, uses[role]):
                late[(name, role, found.period)] = found

        out: List[Violation] = []
        for (period, hz), predictors in sorted(expanded.items()):
            for name, role in [(p, ROLE_PREDICTOR) for p in predictors] + targets:
                if (name, role, period) in late:
                    col = resolved[name]
                    out.append(late[(name, role, period)].use(name, hz, col, role, self._weather_cutoff_hint(col)))
        return out, unknown

    def newest_targets(
        self, target: str, freq: str, index: pd.DatetimeIndex, days: Sequence[Any], horizon: int
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
        # variants once per variant, so tables are kept by everything they are computed from.
        key = (
            repr(col),
            self.issue_time,
            self.issue_timezone,
            self._data_zone(index),
            getattr(index, "tz", None) is None,
            freq,
            tuple(days.asi8.tolist()),
            int(horizon),
        )
        with _NEWEST_TABLES_LOCK:
            found = _NEWEST_TABLES.get(key)
        if found is None:
            found = self._newest_targets(col, target, freq, index, days, int(horizon))
            found.setflags(write=False)
            with _NEWEST_TABLES_LOCK:
                _NEWEST_TABLES[key] = found
                while len(_NEWEST_TABLES) > _NEWEST_TABLES_KEPT:
                    del _NEWEST_TABLES[next(iter(_NEWEST_TABLES))]
        return found

    def _newest_targets(
        self, col: ResolvedColumn, target: str, freq: str, index: pd.DatetimeIndex, days: pd.DatetimeIndex, horizon: int
    ) -> np.ndarray:
        step = _STEP_MINUTES[freq]
        periods = np.arange(1440 // step)
        newest = np.full((len(periods), len(days)), max(horizon - 1, NEWEST_TARGET_FROM), dtype=np.int64)
        if not len(days):
            return newest
        clock = self._issue_clock(days, index)
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
        return newest

    def _issue_clock(self, days: pd.DatetimeIndex, index: pd.DatetimeIndex) -> "_IssueClock":
        """The issue times of these issue days (naive midnights, the first one first) on an index like `index`."""
        ctx = _Context(days[0], self._data_zone(index), naive=getattr(index, "tz", None) is None)
        offset = np.asarray((days - days[0]) // pd.Timedelta(days=1), dtype=np.int64)
        issue = _issue_minutes(offset * 1440.0 + _minutes(self.issue_time), self.issue_timezone, ctx)
        return _IssueClock(days, ctx, offset, issue, _to_instants(offset * 1440.0, ctx.data_zone, ctx))

    def _late_periods(
        self,
        col: ResolvedColumn,
        rows: np.ndarray,
        avail: np.ndarray,
        clock: "_IssueClock",
        step: int,
        row_day: int = 0,
        used: Optional[np.ndarray] = None,
    ) -> List["_LatePeriod"]:
        """Each period with a value known after the issue time, on the issue day it is latest; its times are minutes since
        midnight of that day, as a context for it alone would give them. `avail` is `clock.available(col, rows, step,
        row_day)`. `used` limits the search to the cells [period, issue day] a forecast actually reads, for lists that
        hold on some days only."""
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
        if col.rule is None or col.rule.kind != "weather_cutoff":
            return ""
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
        run = col.rule.time.split(":", 1)[0]
        return (
            f"OpenMeteoSource(run={run!r}, ...) is known {col.rule.delay} after that UTC run; suggest an "
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
        return str(index.tz) if getattr(index, "tz", None) is not None else self.issue_timezone

    def _format(self, violations: List[Violation], model_name: str, advice: bool = True) -> str:
        # One line per base column and rule, listing the predictors that read it.
        grouped: Dict[Tuple[str, str, Optional[str]], List[Violation]] = {}
        for v in violations:
            grouped.setdefault((v.role, v.column.base, v.column.key), []).append(v)
        lines = [
            f"{model_name}: {len(violations)} use(s) of inputs not known at the issue time "
            f"({self.issue_time} {self.issue_timezone} on the issue day D):"
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
    _skipped: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict, compare=False, repr=False)

    def available(self, col: ResolvedColumn, periods: np.ndarray, step: int, row_day: int = 0) -> np.ndarray:
        """The instant the value of `col` at the row (D + row_day, period) is known, in minutes since the first issue
        day's midnight, for every [period, issue day D]. A repeated autumn hour is read at its second occurrence: the row
        the model forecasts from (the last of its period that day), and the later of the two, so the first is covered.

        A row the data zone's clock skips (02:00 on the day it goes forward) has no value, so it gets no verdict: -inf, as if
        known in advance, which `published` still answers False for, since the cell is empty."""
        return self.available_under(col, col.rule, periods, step, row_day)

    def available_under(
        self, col: ResolvedColumn, rule: Availability, periods: np.ndarray, step: int, row_day: int = 0
    ) -> np.ndarray:
        """`available` with `col` read through `rule` instead of its own."""
        periods = np.asarray(periods)
        issue_day = np.tile(self.offset, len(periods))
        minute = np.repeat(periods * step, len(self.days)).astype(float)
        later = np.zeros(len(minute), dtype=bool)
        avail = _available_array(col, rule, issue_day + row_day, minute, step, self.ctx, later, issue_day)
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

        grid = np.zeros((1440 // step, len(self.days)), dtype=bool)
        if self.ctx.naive:
            return grid
        days = self.days + pd.Timedelta(days=row_day)
        freq = next(name for name, minutes in _STEP_MINUTES.items() if minutes == step)
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
    own name still looks fine.

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
    perturbed, counts, unchecked = perturb_unknown(
        data, run_day, information_set, freq, columns, seed, target_from_issue=target_from_issue
    )
    base = data
    if rebuild is not None:
        base, perturbed = rebuild(data), rebuild(perturbed)

    common = (run_day, target, horizon, freq, information_set, availability)
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
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray]], List[Violation]]:
    """The model's forecasts issued on `run_day` from `frame`, by (period, horizon), the cells a predictor function of
    `ctx` read (see `PredictorPlan.read_cells`), and its reads of values known only later (`PredictorPlan.late_reads`).
    `horizon` is what to pass on to the run: `{"horizon": H}` or `{"horizons": (...)}` (`_horizon_arguments`).

    Such a function is planned on this very frame, so a value it reads can move its list, and it answers through the
    information set. The run itself gets none, so a late list isn't refused and the forecasts show it. A model with
    `newest_target="published"` reads its training cut from it (`cut_rules`), and nothing else.
    """
    m = model() if callable(model) and not hasattr(model, "run") else model
    plan_predictors = getattr(m, "_plan_predictors", None)
    plan = None
    if plan_predictors is not None:
        horizons = horizon.get("horizons")
        plan = plan_predictors(
            frame,
            target,
            horizon.get("horizon") or max(horizons),
            freq,
            information_set,
            run_day,
            run_day,
            warn_unpublished=False,
            availability=availability,
            **({} if horizons is None else {"horizons": horizons}),
        )
    extra = {} if plan is None else {"plan": plan}
    if getattr(m, "newest_target", "d-1") == "published":
        extra["cut_rules"] = information_set
    read = {} if plan is None else plan.read_cells()  # the run lets go of them
    late = [] if plan is None else plan.late_reads()
    ref = m.run(frame, test_start=run_day, test_end=run_day, target=target, freq=freq, **horizon, **extra)
    return {(r["hour"], r["horizon"]): r["prediction"] for r in ref._results}, read, late


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

    @property
    def ok(self) -> bool:
        return bool(self.moved)


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

    With `rebuild`, days are those of the frame it returns, so a source in UTC is read in the market's zone.

    A model with `newest_target="published"` may train on the row of D-1 (at horizon 1 when the target of D is known by the
    issue time), so for it only rows dated D or later change: an outlier in a training row would move the forecast whatever
    its inputs, and the control could never fail. With `rebuild` that is a row of the built frame: the raw values of D-1 still
    change, so that what row D builds from them (`x_d-1`) does, and every built row before D is put back.

    `model` is a model instance or a zero-argument factory, and `availability` the level a predictor function is planned
    at, and `horizon` or `horizons` what it runs, as in `leakage_probe`.
    """
    run_day = str(pd.Timestamp(run_day).date())
    horizon = _horizon_arguments(horizon, horizons, 1)
    first_row_day = 0 if _newest_target_of(model) == "published" else -1
    names = list(data.columns) if columns is None else list(columns)
    if rebuild is None:
        names = [name for name in names if name != target]
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
        perturbed = _with_training_data_of(base, rebuild(perturbed), run_day, target, first_row_day)
    original, *_ = _issue_day_forecasts(model, base, run_day, target, horizon, freq, information_set, availability)
    other, *_ = _issue_day_forecasts(model, perturbed, run_day, target, horizon, freq, information_set, availability)
    moved, max_abs_change, _ = _compare_forecasts(original, other, atol)
    return ControlResult(
        run_day=run_day,
        forecasts=len(original),
        moved=moved,
        max_abs_change=max_abs_change,
        perturbed=counts,
        unchecked=unchecked,
        unmoved=sorted((original.keys() & other.keys()) - set(moved)),
    )


def _horizon_arguments(horizon: Optional[int], horizons: Optional[Sequence[int]], default: int) -> Dict[str, Any]:
    """`{"horizon": H}` for every horizon from 1 to H, else `{"horizons": (...)}`: what a probe passes on to `run`."""
    from .models._calendar import horizon_arguments, resolve_horizons  # models import this module

    return horizon_arguments(*resolve_horizons(horizon, horizons, default))


def _newest_target_of(model: Any) -> str:
    """The `newest_target` of a model instance or of one a zero-argument factory makes; `"d-1"` for one that has none."""
    m = model() if callable(model) and not hasattr(model, "run") else model
    return getattr(m, "newest_target", "d-1")


def _with_training_data_of(
    base: pd.DataFrame, perturbed: pd.DataFrame, run_day: str, target: str, first_row_day: int = -1
) -> pd.DataFrame:
    """A copy of `perturbed` with `target` and every row dated before D + `first_row_day` (D-1 by default) taken from `base`."""
    out = perturbed.copy()
    local = out.index.tz_localize(None) if out.index.tz is not None else out.index
    older = np.asarray(local.normalize() < pd.Timestamp(run_day) + pd.Timedelta(days=first_row_day))
    out.loc[older, base.columns] = base.loc[older, base.columns]
    if target in base.columns:
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
) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
    """Copy of `data` where every numeric value not known at issue time on `run_day` is replaced by noise.

    `target_from_issue`, when given a column name, additionally perturbs every row of that column dated
    `run_day` or later, regardless of its own Availability rule: the model's own boundary (training targets
    end at D-1) is a deliberate, conservative choice, and can be stricter than the column's factual
    publication time (e.g. a day-ahead price is, in reality, already published before its delivery day).
    """
    index = data.index
    step = _STEP_MINUTES[freq]
    ctx, day, minute, first = _rows_around_issue(index, run_day, information_set)
    rng = np.random.default_rng(seed)

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
        available = _available_array(col, col.rule, day, minute, step, ctx, first)
        mask = available > ctx.issue_minute
        if target_from_issue is not None and name == target_from_issue:
            mask = mask | (day >= 0)
        if mask.any():
            values = out[name].to_numpy(dtype=float, copy=True)
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
        known = _available_array(col, col.rule, day, minute, step, ctx, first) <= ctx.issue_minute
        # `issue_day=day` reads an `as_of` or `weather_cutoff` value at its own row's run, not at D's, which is all
        # `perturb_unknown` needs to know whether a run is out yet: here it tells yesterday's run from older ones.
        published = _available_array(col, col.rule, day, minute, step, ctx, first, issue_day=day)
        delivered = _available_array(col, _DELIVERED, day, minute, step, ctx, first)
        values = out[name].to_numpy(dtype=float, copy=True)
        fresh = (day >= first_row_day) & ((published > yesterday) | (delivered > yesterday))
        mask = known & fresh & np.isfinite(values)
        if mask.any():
            values[mask] = 1e6 + rng.normal(0.0, 1e4, int(mask.sum()))
            out[name] = values
            counts[name] = int(mask.sum())
    _report_unchecked(unchecked, information_set)
    return out, counts, unchecked


def _rows_around_issue(
    index: pd.DatetimeIndex, run_day: str, information_set: InformationSet
) -> Tuple[_Context, np.ndarray, np.ndarray, np.ndarray]:
    """The context with the issue time on `run_day`, and each row's local day after `run_day`, minute of its day and
    whether it is not the second occurrence of a repeated autumn wall time."""
    # Every zone conversion reads the UTC offsets at the instant it converts (a value's own shifted delivery instant
    # for its delivery day, its publication instant for the way back), not at run_day's noon or at the row's own
    # date. They differ across a DST switch whenever the data zone and a rule's zone don't switch together, e.g. a
    # UTC or fixed-offset (benchmark source) index against Europe/Brussels rules.
    ctx = _Context(pd.Timestamp(run_day), information_set._data_zone(index), naive=index.tz is None)
    issue = _issue_minutes(np.array([float(_minutes(information_set.issue_time))]), information_set.issue_timezone, ctx)
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
    rule: Availability,
    day: np.ndarray,
    minute: np.ndarray,
    step: int,
    ctx: _Context,
    first: np.ndarray,
    issue_day: Union[int, np.ndarray] = 0,
) -> np.ndarray:
    """Vectorised `_available` for rows at (ctx.day + day, minute); the value is `col` read through `rule`.

    `day` and `minute` are the row's local day and wall-clock minute in the data zone; the result is the instant the
    value is known, in minutes since midnight of `ctx.day` (see `_Context`). `issue_day` is each row's issue day D
    in days after `ctx.day` (`violations` checks many issue days at once; `perturb_unknown` one, `ctx.day` itself).
    `first` marks rows that are not the second occurrence of a repeated autumn wall time.
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
        slot, length = col.slot or (1440.0 - step, step)
        slots = np.full(len(reached), float(slot))
        occurrence = np.full(len(reached), col.slot is not None)
        return _available_array(
            col.built_from, rule, reached, slots, length, ctx, occurrence, issue_day + reached - day
        )
    if rule.kind == "static":
        return np.full(len(day), -np.inf)
    if rule.kind == "as_of":
        # Fixed to a time on the day the value is read from: the issue day D itself, moved by every shift after a
        # leading `_d+k` (see `_read_day_offset`), but not by the leading `_d+k`, the row's own day or minute.
        read_day = issue_day + _read_day_offset(col.shifts, day, minute, ctx, first)
        at = np.asarray(read_day, dtype=float) * 1440.0 + _minutes(rule.time)
        return _to_instants(at, rule.timezone or ctx.data_zone, ctx, _second(at))
    if rule.kind == "weather_cutoff":
        # Fixed to `run`:00 UTC plus `delay` on the day the value is read from, same as `as_of` above; the delay is
        # elapsed time after the run and may run into the next day. The run is a UTC hour (`Availability` sees to it).
        read_day = issue_day + _read_day_offset(col.shifts, day, minute, ctx, first)
        run = _to_instants(np.asarray(read_day, dtype=float) * 1440.0 + _minutes(rule.time), "UTC", ctx)
        return run + _parse_delay(rule.delay)
    shifts = list(col.shifts)
    if rule.kind == "weather_run" and shifts and shifts[0].unit == "d" and shifts[0].offset > 0:
        shifts = shifts[1:]
    point, wall = _shift_rows(day, minute, shifts, ctx, first)  # delivery instant of the value (weather: the row time)
    if rule.kind == "realised":
        return point + step + _parse_delay(rule.delay)
    if rule.kind == "weather_run":
        return point + _parse_delay(rule.delay)
    # The delivery day is a calendar day in the rule's own (market) zone, not in the data's: a value at
    # 23:00 UTC in winter (CET, UTC+1) delivers at 00:00 CET the next day, even though the data zone's own
    # wall clock hasn't crossed midnight yet.
    zone = rule.timezone or ctx.data_zone
    delivery_day = np.floor(_to_wall(point, zone, ctx, wall if zone == ctx.data_zone else None) / 1440.0)
    if rule.kind == "published":
        publication_day = delivery_day - rule.days_before
    else:  # weekly: published on `weekday` of the week before the delivery week
        delivery_weekday = np.mod(ctx.weekday + delivery_day, 7)
        publication_day = delivery_day - delivery_weekday - 7 + _WEEKDAYS.index(rule.weekday)
    at = publication_day * 1440.0 + _minutes(rule.time)
    return _to_instants(at, zone, ctx, _second(at))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


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
    day, minute = divmod(int(round(value)), 1440)
    label = "D" if day == 0 else f"D{day:+d}"
    return f"{label} {minute // 60:02d}:{minute % 60:02d}"


@lru_cache(maxsize=4096)  # a listing asks again per horizon and list, for the same few times
def _fmt_instant(value: float, ctx: _Context) -> str:
    """`_fmt_minute` for an instant (minutes since `ctx.origin`) on the data zone's wall clock, with its UTC offset
    where that wall time occurs twice, so a value known in the second 01:00 doesn't read as before an issue at 01:30."""
    if value == -math.inf:
        return "in advance"
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
    "ControlResult",
    "InformationSet",
    "InformationSetError",
    "ProbeResult",
    "default_rules",
    "leakage_probe",
    "parse_predictor",
    "perturb_recent",
    "perturb_unknown",
    "positive_control",
    "source_rules",
]
