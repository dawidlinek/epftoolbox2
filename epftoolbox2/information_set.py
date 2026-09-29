"""When each input becomes known, and checks that a forecast only uses what is known when it is issued.

A forecast issued on day D at the issue time (e.g. 11:30 Brussels, before the 12:00 day-ahead gate closure) may
only use values published by then. Each base column gets an `Availability` rule; predictor names are parsed
(`x`, `x_d-k`, `x_d+k`, `x_h-k`, ...) to find the base column and the delivery time of the value they hold; and
`InformationSet.check` evaluates every expanded predictor for every (period, horizon) before any model is fitted.
`leakage_probe` tests the same boundary empirically: it perturbs every value that is not yet known at issue time
and checks that the forecasts don't move.

Times are compared as local wall-clock times of the data's time zone, relative to midnight of the issue day D.
"""

from __future__ import annotations

import fnmatch
import math
import re
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .logging import get_logger

logger = get_logger(__name__)

KINDS = ("static", "realised", "published", "weekly", "weather_run", "as_of", "weather_cutoff")
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_UNKNOWN_POLICIES = ("raise", "warn", "ignore")
_SUFFIX = re.compile(r"^(?P<base>.+)_(?P<unit>d|h|min|s)(?P<sign>[+-])(?P<n>\d+)$")
_UNIT_MINUTES = {"d": 1440.0, "h": 60.0, "min": 1.0, "s": 1.0 / 60.0}
_STEP_MINUTES = {"1h": 60, "15min": 15}


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
                     rolling forecast run).
        weather_cutoff: an OpenMeteoSource column. Known `delay` after `run`:00 UTC on the issue day D itself,
                     regardless of a leading `_d+k` — the source's own leak-free cutoff is fixed to D's run, not
                     to the delivery day (see OpenMeteoSource and `publication_delay_hours`).
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
    """
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
            from .data.sources._open_meteo_common import aggregated_variable_names

            prefix = f"{source.prefix}_" if getattr(source, "prefix", "") else ""
            rule = _weather_availability(source)
            # A source built with `locations` (see OpenMeteoSource.locations) emits aggregated columns
            # (`{prefix}{variable}` for a bare `agg` string, `{prefix}{variable}_{kind}` per entry for a list
            # of them — see `aggregated_variable_names`) only when `agg` is set; it also, or instead, emits
            # per-location columns `{prefix}{name}_{variable}` — with `agg` only when `keep_locations`, always
            # otherwise (they are then the only output). Both column shapes follow the same cutoff rule.
            locations = getattr(source, "_locations", [])
            agg = getattr(source, "_agg", None)
            if not locations:
                for variable in source.columns:
                    rules[f"{prefix}{variable}"] = rule
            else:
                suffixed = isinstance(getattr(source, "agg", None), (list, tuple))
                for variable in aggregated_variable_names(source.columns, agg, suffixed):
                    rules[f"{prefix}{variable}"] = rule
                if agg is None or getattr(source, "keep_locations", False):
                    for location in locations:
                        for variable in source.columns:
                            rules[f"{prefix}{location['name']}_{variable}"] = rule
    return rules


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


# ---------------------------------------------------------------------------
# Information set
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    predictor: str
    period: int
    horizon: int
    column: ResolvedColumn
    available: float  # minutes after midnight of the issue day, data-local time
    issue: float
    role: str = "predictor"
    hint: str = ""


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
        >>> info = InformationSet("11:30", "Europe/Brussels", rules={"load": Availability.realised("1h")})
        >>> ModelPipeline(information_set=info)
    """

    def __init__(
        self,
        issue_time: str = "11:30",
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

    # -- configuration --------------------------------------------------------

    def with_source_rules(self, sources: Iterable[Any]) -> "InformationSet":
        """A copy that also knows the columns of these data sources (prefixes, Open-Meteo variables)."""
        out = self.copy()
        out._source_rules = {**self._source_rules, **source_rules(sources)}
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
            layers.append(("defaults", default_rules()))
        return layers

    def resolve(self, name: str) -> ResolvedColumn:
        """Find the rule for a column: exact names from the full name down to the base, then glob patterns
        from the base up; layers in order rules > accepted assumptions > sources > defaults."""
        chain = parse_predictor(name)
        for layer, rules in self._layers():
            for candidate, shifts in chain:
                if candidate in rules:
                    return ResolvedColumn(name, candidate, shifts, rules[candidate], candidate, layer)
            patterns = [k for k in rules if any(ch in k for ch in "*?[")]
            for candidate, shifts in reversed(chain):
                for pattern in patterns:
                    if fnmatch.fnmatchcase(candidate, pattern):
                        return ResolvedColumn(name, candidate, shifts, rules[pattern], pattern, layer)
        base, shifts = chain[-1]
        return ResolvedColumn(name, base, shifts, None)

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
    ) -> None:
        """Raise InformationSetError if a predictor (or the newest training target) is not known at issue time."""
        violations, unknown = self.violations(expanded, target, freq, index, test_start, test_end)
        if unknown:
            message = (
                f"{model_name}: no availability rule for column(s) {sorted(unknown)}. Declare when they are known, "
                "e.g. InformationSet(rules={'my_col': Availability.realised('1h')}) or Availability.static(), "
                "or pass unknown='warn'."
            )
            if self.unknown == "raise":
                raise InformationSetError(message)
            if self.unknown == "warn":
                warnings.warn(message, stacklevel=2)
        if violations:
            raise InformationSetError(self._format(violations, model_name))
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
    ) -> Tuple[List[Violation], set]:
        """All (predictor, period, horizon) uses of values not known at issue time, and unknown base columns.

        Every issue day D of the test period is checked, and each zone conversion uses the UTC offsets at the instant
        it converts: when the data zone and a rule's zone don't switch DST together (a UTC or fixed-offset index
        against Brussels rules), a value delivered or published across a switch from D has other offsets than D.
        A violation reports the issue day on which its value comes latest after the issue time.
        """
        step = _STEP_MINUTES[freq]
        resolved: Dict[str, ResolvedColumn] = {}
        unknown = set()
        # The newest training target is the value delivered on D-1 in the same period.
        uses = {"predictor": 0, "training target": -1}  # role -> the row's day relative to D
        periods: Dict[Tuple[str, str], set] = {}  # the periods reading each (name, role); the horizon doesn't matter
        for (period, _), predictors in expanded.items():
            for name in predictors:
                periods.setdefault((name, "predictor"), set()).add(period)
            periods.setdefault((target, "training target"), set()).add(period)
        for name in [target] + [name for name, _ in periods]:
            if name not in resolved:
                resolved[name] = self.resolve(name)
                if resolved[name].rule is None:
                    unknown.add(resolved[name].base)

        days = self._issue_days(index, test_start, test_end)
        if not len(days):
            return [], unknown
        ctx = _Context(days[0], self._data_zone(index))
        # Each issue day as days after the first, and its issue time in data-zone minutes after the first's midnight.
        offset = np.asarray((days - days[0]) // pd.Timedelta(days=1), dtype=np.int64)
        issue = _issue_minutes(offset * 1440.0 + _minutes(self.issue_time), self.issue_timezone, ctx)

        late: Dict[Tuple[str, str, int], Tuple[float, float, str]] = {}  # (name, role, period) -> its worst day
        for (name, role), rows in periods.items():
            col = resolved[name]
            if col.rule is None:
                continue
            rows = np.array(sorted(rows))
            issue_day = np.tile(offset, len(rows))
            minute = np.repeat(rows * step, len(days)).astype(float)
            avail = _available_array(col, col.rule, issue_day + uses[role], minute, step, ctx, issue_day=issue_day)
            avail = avail.reshape(len(rows), len(days))
            worst = (avail - issue).argmax(axis=1)
            for period, i, known in zip(rows.tolist(), worst, avail[np.arange(len(rows)), worst]):
                if known <= issue[i]:
                    continue
                # Minutes after midnight of that issue day, as a context for that day alone would give them.
                day_ctx = _Context(days[i], ctx.data_zone, float(issue[i] - offset[i] * 1440.0))
                hint = self._assumption_hint(col, uses[role], period * step, step, day_ctx)
                late[(name, role, period)] = (float(known - offset[i] * 1440.0), day_ctx.issue_minute, hint)

        out: List[Violation] = []
        for (period, hz), predictors in sorted(expanded.items()):
            for name, role in [(p, "predictor") for p in predictors] + [(target, "training target")]:
                if (name, role, period) in late:
                    known, issue_minute, hint = late[(name, role, period)]
                    col = resolved[name]
                    hint = hint or self._weather_cutoff_hint(col)
                    out.append(Violation(name, period, hz, col, known, issue_minute, role, hint))
        return out, unknown

    def _assumption_hint(self, col, day, row_minute, step, ctx) -> str:
        for name, (pattern, rule, description) in ASSUMPTIONS.items():
            if name in self.assumptions or col.key != pattern:
                continue
            if _available(col, rule, day, row_minute, step, ctx) <= ctx.issue_minute:
                return f"known in time if you accept assumption '{name}': {description}"
        return ""

    @staticmethod
    def _weather_cutoff_hint(col: ResolvedColumn) -> str:
        """An OpenMeteoSource column's own cutoff is a source-side setting, not a data fact: point at it."""
        if col.rule is None or col.rule.kind != "weather_cutoff":
            return ""
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

    def _format(self, violations: List[Violation], model_name: str) -> str:
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
                f"{_fmt_minute(worst.available)}, issued {_fmt_minute(worst.issue)}."
            )
            hinted = [v.predictor for v in items if v.hint]
            if hinted:
                first = next(v for v in items if v.hint)
                lines.append(f"      {', '.join(dict.fromkeys(hinted))}: {first.hint}")
        lines.append(
            "Use values known by then (e.g. `x_d-1` instead of `x`), or, if a rule is too strict for your data, "
            "override it with InformationSet(rules={...})."
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class _Context:
    """Times are wall-clock minutes of `data_zone` after the naive midnight `day`; converting them to or from another
    zone uses the UTC offsets at the very instant converted (see `_to_zone`), not a fixed offset of `day`."""

    day: pd.Timestamp  # the issue day D, or (in `violations`) the first issue day of the test period
    data_zone: str
    issue_minute: float = 0.0  # the issue time on `day`; `violations` keeps one per issue day instead

    @property
    def weekday(self) -> int:
        return self.day.weekday()


def _available(col: ResolvedColumn, rule: Availability, day: int, row_minute: float, step: int, ctx: _Context) -> float:
    """Minutes after midnight of D (data-local) when the value of `col` at the row (D + day, row_minute) is known."""
    return float(_available_array(col, rule, np.array([day]), np.array([row_minute], dtype=float), step, ctx)[0])


def _to_zone(minutes: np.ndarray, zone: Optional[str], ctx: _Context, first: Optional[np.ndarray] = None) -> np.ndarray:
    """Data-zone wall-clock minutes after midnight of `ctx.day` -> the same instants' wall clock in `zone`.

    `first` says, per value, which occurrence a repeated autumn wall time is (see `_convert_wall_clock`).
    """
    if zone is None or zone == ctx.data_zone:
        return minutes
    return _convert_wall_clock(minutes, ctx.day, ctx.data_zone, zone, first)


def _from_zone(minutes: np.ndarray, zone: Optional[str], ctx: _Context) -> np.ndarray:
    """Inverse of `_to_zone`: wall-clock minutes after midnight of `ctx.day` in `zone` -> the data zone's."""
    if zone is None or zone == ctx.data_zone:
        return minutes
    return _convert_wall_clock(minutes, ctx.day, zone, ctx.data_zone)


def _issue_minutes(minutes: np.ndarray, zone: str, ctx: _Context) -> np.ndarray:
    """Issue times, wall-clock minutes after midnight of `ctx.day` in `zone` -> the data zone's.

    Unlike a publication time, which a skipped wall time moves forward to the switch, an issue time skipped by a
    spring switch is read with the offset after it: 02:30 Brussels on 2024-03-31 is 00:30Z (01:30 CET), its earliest
    reading, so the issue is never later than asked. This holds within one zone too; a repeated autumn issue time
    is its first occurrence.
    """
    naive = pd.DatetimeIndex(ctx.day + pd.to_timedelta(np.asarray(minutes, dtype=float), unit="min"))
    forward = naive.tz_localize(zone, ambiguous=np.ones(len(naive), dtype=bool), nonexistent="shift_forward")
    earliest = forward - (forward.tz_localize(None) - naive)  # back by as much as the switch moved it forward
    return np.asarray((earliest.tz_convert(ctx.data_zone).tz_localize(None) - ctx.day) / pd.Timedelta(minutes=1))


# ---------------------------------------------------------------------------
# Leakage probe
# ---------------------------------------------------------------------------


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

    @property
    def ok(self) -> bool:
        return not self.changed and not self.missing

    def assert_ok(self) -> None:
        if not self.ok:
            raise AssertionError(
                f"Forecasts issued on {self.run_day} depend on values not known at issue time: "
                f"{len(self.changed)} of {self.forecasts} changed (max |change| {self.max_abs_change:.6g}), "
                f"e.g. (period, horizon) {self.changed[:10]}; forecasts present in only one run: {self.missing[:10]}. "
                f"Perturbed cells per column: {self.perturbed}"
            )


def leakage_probe(
    model: Any,
    data: pd.DataFrame,
    run_day: str,
    information_set: InformationSet,
    target: str = "price",
    horizon: int = 1,
    freq: str = "1h",
    columns: Optional[Sequence[str]] = None,
    atol: float = 1e-9,
    seed: int = 0,
    rebuild: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    check_target_boundary: bool = False,
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
    of its own Availability rule: the model's own boundary (training targets end at D-1) is a deliberate,
    conservative choice that can be stricter than a column's factual publication time.
    """
    run_day = str(pd.Timestamp(run_day).date())
    target_from_issue = target if check_target_boundary else None
    perturbed, counts, unchecked = perturb_unknown(
        data, run_day, information_set, freq, columns, seed, target_from_issue=target_from_issue
    )
    base = data
    if rebuild is not None:
        base, perturbed = rebuild(data), rebuild(perturbed)

    def forecasts(frame):
        m = model() if callable(model) and not hasattr(model, "run") else model
        ref = m.run(frame, test_start=run_day, test_end=run_day, target=target, horizon=horizon, freq=freq)
        return {(r["hour"], r["horizon"]): r["prediction"] for r in ref._results}

    original, probed = forecasts(base), forecasts(perturbed)
    shared = original.keys() & probed.keys()
    diffs = {k: abs(original[k] - probed[k]) for k in shared}
    changed = sorted(k for k, d in diffs.items() if not d <= atol)
    missing = sorted(original.keys() ^ probed.keys())
    return ProbeResult(
        run_day=run_day,
        forecasts=len(original),
        changed=changed,
        max_abs_change=max(diffs.values(), default=0.0),
        perturbed=counts,
        unchecked=unchecked,
        missing=missing,
    )


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
    # Every zone conversion reads the UTC offsets at the instant it converts (a value's own shifted delivery instant
    # for its delivery day, its publication instant for the way back), not at run_day's noon or at the row's own
    # date. They differ across a DST switch whenever the data zone and a rule's zone don't switch together, e.g. a
    # UTC or fixed-offset (benchmark source) index against Europe/Brussels rules.
    ctx = _Context(pd.Timestamp(run_day), information_set._data_zone(index))
    issue = _issue_minutes(np.array([float(_minutes(information_set.issue_time))]), information_set.issue_timezone, ctx)
    ctx = replace(ctx, issue_minute=float(issue[0]))
    local = index.tz_localize(None) if index.tz is not None else index
    day = np.asarray((local.normalize() - pd.Timestamp(run_day)) // pd.Timedelta(days=1), dtype=np.int64)
    minute = np.asarray(local.hour * 60 + local.minute + local.second / 60.0, dtype=float)
    # Which occurrence of a repeated autumn wall time each row is; a day shift keeps it, like shift_calendar_days.
    first = np.ones(len(index), dtype=bool)
    if index.tz is not None:
        first = index.asi8 == local.tz_localize(index.tz, ambiguous=first, nonexistent="NaT").asi8
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
        available = _available_array(col, col.rule, day, minute, step, ctx, first)
        mask = available > ctx.issue_minute
        if target_from_issue is not None and name == target_from_issue:
            mask = mask | (day >= 0)
        if mask.any():
            values = out[name].to_numpy(dtype=float, copy=True)
            values[mask] = 1e6 + rng.normal(0.0, 1e4, int(mask.sum()))
            out[name] = values
            counts[name] = int(mask.sum())
    if unchecked:
        message = f"leakage_probe: no availability rule for {unchecked}; left unperturbed"
        if information_set.unknown == "raise":
            raise InformationSetError(message + ". Declare rules or pass unknown='warn'.")
        if information_set.unknown == "warn":
            warnings.warn(message, stacklevel=2)
    return out, counts, unchecked


def _available_array(
    col: ResolvedColumn,
    rule: Availability,
    day: np.ndarray,
    minute: np.ndarray,
    step: int,
    ctx: _Context,
    first: Optional[np.ndarray] = None,
    issue_day: Union[int, np.ndarray] = 0,
) -> np.ndarray:
    """Vectorised `_available` for rows at (ctx.day + day, minute); the value is `col` read through `rule`.

    Every time is in data-zone minutes after midnight of `ctx.day`, and times in a rule's own zone are converted
    exactly, at each converted instant (`_to_zone`/`_from_zone`). `issue_day` is each row's issue day D in days
    after `ctx.day` (`violations` checks many issue days at once; `perturb_unknown` one, `ctx.day` itself).
    `first` marks rows that are not the second occurrence of a repeated autumn wall time (default: all).
    """
    if rule.kind == "static":
        return np.full(len(day), -np.inf)
    if rule.kind == "as_of":
        # Fixed to a time on the issue day D itself: independent of the row's own day/minute or any shift,
        # so a leading `_d+k` doesn't change it.
        known = np.full(len(day), _minutes(rule.time), dtype=float) + issue_day * 1440.0
        return _from_zone(known, rule.timezone, ctx)
    if rule.kind == "weather_cutoff":
        # Fixed to `run`:00 UTC plus `delay` on the issue day D itself, same independence as `as_of` above. The
        # run+delay total is minutes after UTC midnight of D and may exceed 1440 (a late run plus its delay can
        # land on D+1 in UTC); that is fine here, this is a plain offset, not a wall-clock time to re-parse.
        cutoff = np.full(len(day), _minutes(rule.time) + _parse_delay(rule.delay), dtype=float) + issue_day * 1440.0
        return _from_zone(cutoff, rule.timezone, ctx)
    shifts = list(col.shifts)
    if rule.kind == "weather_run" and shifts and shifts[0].unit == "d" and shifts[0].offset > 0:
        shifts = shifts[1:]
    # Delivery time of the value (for weather: the row time), in minutes after midnight of D.
    point = (day + sum(s.offset for s in shifts if s.unit == "d")) * 1440.0 + minute
    point = point + sum(s.minutes for s in shifts if s.unit != "d")
    if rule.kind == "realised":
        return point + step + _parse_delay(rule.delay)
    if rule.kind == "weather_run":
        return point + _parse_delay(rule.delay)
    # The delivery day is a calendar day in the rule's own (market) zone, not in the data's: a value at
    # 23:00 UTC in winter (CET, UTC+1) delivers at 00:00 CET the next day, even though `point` (the data
    # zone's own wall clock) hasn't crossed midnight yet.
    delivery_day = np.floor(_to_zone(point, rule.timezone, ctx, first) / 1440.0)
    if rule.kind == "published":
        publication_day = delivery_day - rule.days_before
    else:  # weekly: published on `weekday` of the week before the delivery week
        delivery_weekday = np.mod(ctx.weekday + delivery_day, 7)
        publication_day = delivery_day - delivery_weekday - 7 + _WEEKDAYS.index(rule.weekday)
    return _from_zone(publication_day * 1440.0 + _minutes(rule.time), rule.timezone, ctx)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _convert_wall_clock(
    minutes: np.ndarray, day: pd.Timestamp, from_zone: str, to_zone: str, first: Optional[np.ndarray] = None
) -> np.ndarray:
    """Wall-clock minutes after midnight of `day` in `from_zone` -> the same instants' wall clock in `to_zone`.

    A wall time skipped by a spring switch moves forward to the switch; a repeated autumn one is read as its
    first occurrence, or as its second where `first` is False.
    """
    naive = pd.DatetimeIndex(day + pd.to_timedelta(np.asarray(minutes, dtype=float), unit="min"))
    first = np.ones(len(naive), dtype=bool) if first is None else first
    local = naive.tz_localize(from_zone, ambiguous=first, nonexistent="shift_forward")
    return np.asarray((local.tz_convert(to_zone).tz_localize(None) - day) / pd.Timedelta(minutes=1), dtype=float)


def _fmt_minute(value: float) -> str:
    if value == -math.inf:
        return "in advance"
    day, minute = divmod(int(round(value)), 1440)
    label = "D" if day == 0 else f"D{day:+d}"
    return f"{label} {minute // 60:02d}:{minute % 60:02d}"


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
    "Availability",
    "InformationSet",
    "InformationSetError",
    "ProbeResult",
    "default_rules",
    "leakage_probe",
    "parse_predictor",
    "perturb_unknown",
    "source_rules",
]
