"""Energy-Charts (Fraunhofer ISE) data source: forecasts, actual generation and load, day-ahead prices, cross-border flows.

Data: Energy-Charts.info, Fraunhofer Institute for Solar Energy Systems ISE, licensed CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). Using it requires attribution to Energy-Charts.info. Day-ahead prices
come from Bundesnetzagentur | SMARD.de (CC BY 4.0) for the zones in ``CC_BY_ZONES`` only; the prices of every other zone
are for private and internal use (see ``EnergyChartsSource``).
"""

import hashlib
import json
import os
import re
import threading
import weakref
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests

from epftoolbox2._clock import Clock, current_clock, settled_now
from epftoolbox2.logging import get_logger
from .base import DataSource

# Bump whenever the output for the same parameters changes, so that cached frames are rebuilt instead of served.
SCHEMA_VERSION = 1

ATTRIBUTION = "Energy-Charts.info (Fraunhofer ISE), CC BY 4.0"

TYPES = ("forecast", "power", "price", "cbet", "cbpf")

# `forecast_type` of /public_power_forecast -> the vintage word in the column name.
FORECAST_TYPES = {"day-ahead": "day_ahead", "intraday": "intraday", "current": "current"}
FORECAST_PRODUCTION_TYPES = ("solar", "wind_onshore", "wind_offshore", "load")
# The production types of the forecast endpoint that are published as a day-ahead forecast only.
DAY_AHEAD_ONLY = ("load",)

COUNTRIES = (
    "ch eu all at ba be bg cy cz de dk ee es fi fr ge gr hr hu ie it lt lu lv md me mk nl no pl pt ro rs se si sk ua uk xk"
).split()

# The bidding zones /price knows (the OpenAPI document's BiddingZoneV2).
BIDDING_ZONES = (
    "DE-LU AT BE BG CH CZ DK1 DK2 EE ES FI FR GR HR HU IT-Calabria IT-Centre-North IT-Centre-South IT-North IT-SACOAC "
    "IT-SACODC IT-Sardinia IT-Sicily IT-South LT LV ME NL NO1 NO2 NO2NSL NO3 NO4 NO5 PL PT RO RS SE1 SE2 SE3 SE4 SI SK "
    "DE-AT-LU IE(SEM) UA-IPS UA-BEI IT-Brindisi IT-Foggia IT-GR IT-North-AT IT-North-CH IT-North-SI IT-North-FR "
    "IT-Priolo IT-Rossano"
).split()
# Prices of these zones are CC BY 4.0 (from Bundesnetzagentur | SMARD.de); the other zones' are for private and internal
# use only, which a published result or a shared dataset is not.
CC_BY_ZONES = (
    "AT",
    "BE",
    "CH",
    "CZ",
    "DE-LU",
    "DE-AT-LU",
    "DK1",
    "DK2",
    "FR",
    "HU",
    "IT-North",
    "NL",
    "NO2",
    "PL",
    "SE4",
    "SI",
)
# Zones whose day-ahead auction is not SDAC (no 12:00 CET coupling gate): the built-in availability rule of `price`
# covers SDAC zones and CH only.
NON_SDAC_ZONES = ("CH", "ME", "RS", "UA-IPS", "UA-BEI", "IE(SEM)")
# A country code's bidding zone when it has exactly one.
_ZONE_OF_COUNTRY = {"de": "DE-LU", "ie": "IE(SEM)"}
_AMBIGUOUS_COUNTRIES = ("dk", "it", "no", "se", "lu", "uk", "eu", "all", "ua")
_ZONE_LOOKUP = {zone.upper().replace("_", "-"): zone for zone in BIDDING_ZONES}

BASE_URL = "https://api.energy-charts.info"
# Per endpoint, per client IP: 2 requests a minute with a burst of 4; /price 2 a minute with a burst of 2. The server may
# lower these at times, hence the Retry-After handling as well.
RATE_PER_MINUTE = 2.0
BURST = {"price": 2}
DEFAULT_BURST = 4


# Usual series of /public_power for Germany, to check prefixes against when `power` is not given.
_USUAL_POWER = (
    "load residual_load solar wind_onshore wind_offshore biomass fossil_gas fossil_hard_coal "
    "fossil_brown_coal_lignite hydro_run_of_river hydro_pumped_storage nuclear waste others"
).split()


class EnergyChartsApiError(RuntimeError):
    """The Energy-Charts API refused a request or kept failing."""


def snake_case(name: str) -> str:
    """``"Fossil brown coal / lignite"`` -> ``"fossil_brown_coal_lignite"``."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def forecast_column(production_type: str, forecast_type: str, prefix: str = "") -> str:
    """The column of a forecast: ``solar_forecast_day_ahead``, ``wind_onshore_forecast_intraday``..."""
    stem = f"{production_type}_forecast_{FORECAST_TYPES[forecast_type]}"
    return f"{prefix}_{stem}" if prefix else stem


class _TokenBucket:
    """A token bucket for one endpoint, shared by every source in the process (the limit is per client IP)."""

    def __init__(self, capacity: float, per_second: float):
        self.capacity = capacity
        self.per_second = per_second
        self.tokens = capacity
        self.stamp: Optional[float] = None
        self.blocked_until = 0.0
        self.lock = threading.Lock()

    def acquire(self, clock: Callable[[], float], sleep: Callable[[float], None]) -> None:
        while True:
            with self.lock:
                now = clock()
                if self.stamp is not None:
                    self.tokens = min(self.capacity, self.tokens + (now - self.stamp) * self.per_second)
                self.stamp = now
                if self.blocked_until > now:
                    wait = self.blocked_until - now
                elif self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                else:
                    wait = (1.0 - self.tokens) / self.per_second
            sleep(wait)

    def block(self, now: float, seconds: float) -> None:
        """The server said to wait: nothing leaves for `seconds`, and the burst is spent."""
        with self.lock:
            self.blocked_until = max(self.blocked_until, now + seconds)
            self.tokens = 0.0
            self.stamp = now


# By endpoint and clock (its `id`): a bucket's stamps are seconds of one clock's `monotonic()` (a FakeClock's start at 0).
# A clock's buckets go when it does (`_CLOCKS`: a finalizer, or the clock itself where it can't be weakly referenced, so that
# its id isn't reused while its buckets are kept); the key is no clock, which may be unhashable (a dataclass with `__eq__`).
_BUCKETS: Dict[Tuple[str, int], _TokenBucket] = {}
_CLOCKS: Dict[int, Any] = {}
_BUCKETS_LOCK = threading.RLock()  # a finalizer can run, and take it, while this thread holds it


def _forget_clock(ident: int) -> None:
    with _BUCKETS_LOCK:
        _CLOCKS.pop(ident, None)
        for key in [key for key in _BUCKETS if key[1] == ident]:
            del _BUCKETS[key]


def _bucket(endpoint: str, clock: Clock) -> _TokenBucket:
    with _BUCKETS_LOCK:
        ident = id(clock)
        if ident not in _CLOCKS:
            try:
                _CLOCKS[ident] = weakref.finalize(clock, _forget_clock, ident)
            except TypeError:  # no weak reference to it (__slots__ without __weakref__): kept, with its buckets
                _CLOCKS[ident] = clock
        key = (endpoint, ident)
        if key not in _BUCKETS:
            _BUCKETS[key] = _TokenBucket(BURST.get(endpoint, DEFAULT_BURST), RATE_PER_MINUTE / 60.0)
        return _BUCKETS[key]


def _retry_after_seconds(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, (parsedate_to_datetime(value) - current_clock().now()).total_seconds())
    except (TypeError, ValueError):
        return None


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join("~", ".cache")
    return Path(base).expanduser() / "epftoolbox2" / "energy-charts"


def _user_agent() -> str:
    try:
        from importlib.metadata import version

        return f"epftoolbox2/{version('epftoolbox2')} (research client)"
    except Exception:  # noqa: BLE001 - not installed as a distribution
        return "epftoolbox2 (research client)"


def _to_list(value: Union[None, str, List[str], tuple]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


class EnergyChartsSource(DataSource):
    """
    Energy-Charts (Fraunhofer ISE) data source, no API key

    Supports fetching (``type``):
    - ``"forecast"``: ``/public_power_forecast``, the TSOs' solar, wind onshore, wind offshore and load forecasts in MW,
      one column per production type and forecast type: ``solar_forecast_day_ahead``, ``wind_onshore_forecast_intraday``,
      ``wind_offshore_forecast_current``, ``load_forecast_day_ahead``. The load forecast exists as ``day-ahead`` only.
    - ``"power"``: ``/public_power``, actual net generation and load in MW (shares in %): ``actual_solar``,
      ``actual_load``, ``actual_residual_load``, ``actual_fossil_gas``... (the series the API returns for ``country``)
    - ``"price"``: ``/price``, the day-ahead price of one bidding zone in EUR/MWh: ``price_day_ahead``
    - ``"cbet"`` / ``"cbpf"``: ``/cbet``, ``/cbpf``, cross-border electricity trading and physical flows between
      ``country`` and each neighbour in GW (positive = import): ``cbet_france``, ``cbpf_sum``...

    The three forecast types are different vintages of one forecast: ``day-ahead`` (``forecast_types=["day-ahead"]``,
    the default) is published on the evening before, ``intraday`` on the morning of the delivery day, and ``current`` is
    the newest revision, filled in a batch after the day at a time nobody measured (and possibly recomputed). Only
    ``current`` is never usable in a backtest: its availability rule holds on no day, so it is never known at any shift
    and ``availability="raise"`` refuses it. ``day-ahead`` and ``intraday`` are ``latest_revision`` like ENTSO-E's TSO
    forecasts: the archive does not show that no value was rewritten after its rule's time. That the three names are the TSOs' vintages is an inference from their names
    and their errors against actuals (ordered day-ahead > intraday > current), not from documentation.

    Timestamps of the API mark the start of the period and are unix seconds, so both DST switches come out right. The
    grid is the API's own: 15-minute for forecasts and generation, 15-minute or hourly for prices (CH is hourly), and
    an older stretch can be hourly where a newer one is 15-minute; ``fetch`` does not resample (use
    ``ResampleTransformer``).

    **Licence.** Data: Energy-Charts.info (Fraunhofer ISE), CC BY 4.0, attribution to Energy-Charts.info required. Day-ahead
    prices are CC BY 4.0 from Bundesnetzagentur | SMARD.de for AT, BE, CH, CZ, DE-LU, DE-AT-LU, DK1, DK2, FR, HU,
    IT-North, NL, NO2, PL, SE4 and SI. The prices of any other zone are for private and internal use only: asking for one
    raises ``ValueError`` unless ``allow_restricted_prices=True``, and then logs a warning.

    **Rate limits.** The public API allows 2 requests a minute per IP and endpoint (burst 4, ``/price`` burst 2) and answers
    HTTP 429 with ``Retry-After`` beyond that. The source throttles itself to that rate (shared by all sources in the
    process), waits ``Retry-After`` on a 429, fetches at most a year per request and caches each answer for a period
    that is over for more than two days on disk (``cache_dir``, ``use_cache``). A year of the default forecast types costs
    4 forecast requests (about 2 minutes once the burst is spent), power and price one each.

    Args:
        country: country code of the forecast, power and cross-border endpoints (``"de"``, ``"at"``...).
        type: any of ``"forecast"``, ``"power"``, ``"price"``, ``"cbet"``, ``"cbpf"``. ``None`` is
            ``["forecast", "power", "price"]``.
        bzn: bidding zone of the price (``"DE-LU"``, ``"CH"``...). ``None`` takes the country's, which only countries with
            one zone have (``"de"`` is ``DE-LU``); Denmark, Italy, Norway and Sweden need it.
        forecast_types: ``"day-ahead"``, ``"intraday"``, ``"current"``.
        production_types: ``"solar"``, ``"wind_onshore"``, ``"wind_offshore"``, ``"load"``. ``None`` is all four.
        power: series of ``/public_power`` to keep, as in the column names without ``actual_`` (``"solar"``,
            ``"fossil_gas"``) or a comma-separated string. ``None`` keeps every series the API returns.
        allow_restricted_prices: accept a zone whose prices are for private use only.
        prefix: prepended to every column (``f"{prefix}_{column}"``). The default ``"ec"`` keeps every column off the built-in
            availability rules of other sources; ``""`` is allowed but then a name can fall to a default rule.
        cache_dir: where the answers are kept (``~/.cache/epftoolbox2/energy-charts`` by default).
        use_cache: keep and reuse the answers on disk.

    Example:
        >>> source = EnergyChartsSource("de", type=["forecast", "price"], production_types=["solar", "wind_onshore"])
        >>> df = source.fetch(pd.Timestamp("2025-03-01", tz="UTC"), pd.Timestamp("2025-03-08", tz="UTC"))
    """

    REQUEST_TIMEOUT = 60
    MAX_RETRIES = 6
    RETRY_BACKOFF_SECONDS = 4.0
    MAX_RETRY_WAIT_SECONDS = 900.0
    CHUNK_DAYS = 366
    # A period that ended less than this long ago may still change, and is not written to the disk cache.
    UNSETTLED = pd.Timedelta(days=2)

    def __init__(
        self,
        country: str = "de",
        type: Optional[List[str]] = None,
        bzn: Optional[str] = None,
        forecast_types: Optional[List[str]] = None,
        production_types: Optional[List[str]] = None,
        power: Optional[Union[List[str], str]] = None,
        allow_restricted_prices: bool = False,
        prefix: str = "ec",
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ):
        self.country = country
        self.type = _to_list(type)
        self.bzn = bzn
        self.forecast_types = _to_list(forecast_types)
        self.production_types = _to_list(production_types)
        self.power = _to_list(power)
        self.allow_restricted_prices = allow_restricted_prices
        self.prefix = prefix if prefix is not None else ""
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.session = requests.Session()
        self.logger = get_logger(__name__)

        self._validate_config()

    def __del__(self):
        if hasattr(self, "session") and self.session:
            self.session.close()

    # The current clock (`epftoolbox2.clock`): the rate limit's seconds and its waits; the response cache settles against
    # `settled_now` (never later than the machine's now).
    def _clock(self) -> float:
        return current_clock().monotonic()

    def _sleep(self, seconds: float) -> None:
        current_clock().sleep(seconds)

    def _now(self) -> pd.Timestamp:
        return settled_now()

    # -- configuration --------------------------------------------------------

    @property
    def _types(self) -> List[str]:
        return list(self.type) if self.type else ["forecast", "power", "price"]

    @property
    def _forecast_types(self) -> List[str]:
        return list(self.forecast_types) if self.forecast_types else ["day-ahead"]

    @property
    def _production_types(self) -> List[str]:
        return list(self.production_types) if self.production_types else list(FORECAST_PRODUCTION_TYPES)

    @property
    def _zone(self) -> Optional[str]:
        """The bidding zone of the price, in the API's spelling, or None without a price."""
        if "price" not in self._types:
            return None
        if self.bzn:
            return _ZONE_LOOKUP[str(self.bzn).upper().replace("_", "-")]
        return _ZONE_OF_COUNTRY.get(self.country.lower(), self.country.upper())

    def _validate_config(self) -> bool:
        if not isinstance(self.country, str) or self.country.lower() not in COUNTRIES:
            raise ValueError(f"Unknown country {self.country!r}. Countries: {COUNTRIES}")
        for t in self._types:
            if t not in TYPES:
                raise ValueError(f"Invalid type '{t}'. Must be one of: {list(TYPES)}")
        for f in self._forecast_types:
            if f not in FORECAST_TYPES:
                raise ValueError(f"Invalid forecast type '{f}'. Must be one of: {list(FORECAST_TYPES)}")
        for p in self._production_types:
            if p not in FORECAST_PRODUCTION_TYPES:
                raise ValueError(f"Invalid production type '{p}'. Must be one of: {list(FORECAST_PRODUCTION_TYPES)}")
        if "forecast" in self._types and all(p in DAY_AHEAD_ONLY for p in self._production_types):
            if "day-ahead" not in self._forecast_types:
                raise ValueError("The load forecast exists as 'day-ahead' only; add it to forecast_types")
        if not isinstance(self.prefix, str):
            raise ValueError(f"prefix must be a string, got {self.prefix!r}")
        if self.bzn is not None and str(self.bzn).upper().replace("_", "-") not in _ZONE_LOOKUP:
            raise ValueError(f"Unknown bidding zone {self.bzn!r}. Zones: {BIDDING_ZONES}")
        self._validate_prefix()
        if "price" in self._types:
            self._validate_price_zone()
        if "current" in self._forecast_types and "forecast" in self._types:
            self.logger.warning(
                "EnergyCharts: forecast_type 'current' is filled in after the day, not a vintage a forecast "
                "issued earlier could have used; its availability rule refuses it at every shift."
            )
        return True

    def _output_names(self) -> List[str]:
        """Representative output columns of the selected types (the series of /public_power and the neighbours of
        /cbet, /cbpf are the API's, so a few usual ones stand for them)."""
        names: List[str] = []
        for t in self._types:
            if t == "forecast":
                for production in self._production_types:
                    for forecast in self._forecast_types:
                        if production not in DAY_AHEAD_ONLY or forecast == "day-ahead":
                            names.append(forecast_column(production, forecast))
            elif t == "power":
                series = [snake_case(p).removeprefix("actual_") for p in self.power] if self.power else _USUAL_POWER
                names += [f"actual_{name}" for name in series]
            elif t == "price":
                names.append("price_day_ahead")
            else:
                names += [f"{t}_{name}" for name in ("sum", "austria", "france")]
        return [f"{self.prefix}_{name}" if self.prefix else name for name in names]

    def _validate_prefix(self) -> None:
        """A prefix that makes a column take a built-in default rule of another source (``holiday_*`` is known in
        advance, ``generation_*`` is ENTSO-E's) would give it that rule silently outside a source-aware run."""
        from epftoolbox2.information_set import InformationSet

        info = InformationSet()
        for name in self._output_names():
            for form in (name, f"{name}_d-1"):
                found = info.resolve(form)
                if found.rule is not None and found.layer == "defaults":
                    raise ValueError(
                        f"prefix {self.prefix!r} makes the column {form!r} take the built-in default rule "
                        f"'{found.key}' ({found.rule.describe()}) instead of an Energy-Charts one. Use another prefix, "
                        f"such as the default 'ec'."
                    )

    def _validate_price_zone(self) -> None:
        country = self.country.lower()
        if not self.bzn and country in _AMBIGUOUS_COUNTRIES:
            raise ValueError(f"country {self.country!r} has several bidding zones; give bzn=..., e.g. 'DK1'")
        if not self.bzn and country not in _ZONE_OF_COUNTRY and country.upper() not in _ZONE_LOOKUP:
            raise ValueError(f"country {self.country!r} has no bidding zone of its own; give bzn=...")
        zone = self._zone
        if zone not in CC_BY_ZONES:
            if not self.allow_restricted_prices:
                raise ValueError(
                    f"The Energy-Charts prices of zone {zone!r} are for private and internal use only (CC BY 4.0 covers "
                    f"{', '.join(CC_BY_ZONES)}). Pass allow_restricted_prices=True to use them anyway; a published result "
                    f"or a shared dataset may then break their terms."
                )
            self.logger.warning(
                f"EnergyCharts: the prices of zone {zone!r} are for private and internal use only; do not publish or "
                "share them or anything derived from them."
            )

    # -- fetch ------------------------------------------------------------------

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
        end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")
        if end <= start:
            raise ValueError(f"End timestamp ({end}) must be after start timestamp ({start})")

        began = current_clock().monotonic()
        self.logger.info(f"EnergyCharts [{self.country}]: downloading {', '.join(self._types)}; data {ATTRIBUTION}")
        chunks = self._chunks(start, end)
        parts: List[pd.DataFrame] = []
        for chunk_start, chunk_end in chunks:
            for endpoint, params, parse in self._requests():
                payload = self._get(endpoint, params, chunk_start, chunk_end)
                frame = parse(payload)
                if frame is not None and not frame.empty:
                    parts.append(frame)

        if not parts:
            self.logger.warning(f"EnergyCharts [{self.country}]: no data between {start} and {end}")
            return pd.DataFrame()

        # Chunks and answers are disjoint in time per column; concatenate each column's pieces, then join the columns.
        columns: Dict[str, List[pd.Series]] = {}
        for frame in parts:
            for name in frame.columns:
                columns.setdefault(name, []).append(frame[name])
        series = {}
        for name, pieces in columns.items():
            joined = pd.concat(pieces).sort_index(kind="stable")
            series[name] = joined[~joined.index.duplicated(keep="first")]
        result = pd.DataFrame(series).sort_index()
        result = result[(result.index >= start) & (result.index < end)]
        if result.empty:
            self.logger.warning(f"EnergyCharts [{self.country}]: no data between {start} and {end}")
            return pd.DataFrame()
        if self.prefix:
            result.columns = [f"{self.prefix}_{column}" for column in result.columns]
        self.logger.info(f"EnergyCharts [{self.country}]: done in {current_clock().monotonic() - began:.2f} sec")
        return result

    def _chunks(self, start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        step = pd.Timedelta(days=self.CHUNK_DAYS)
        chunks = []
        current = start
        while current < end:
            chunks.append((current, min(current + step, end)))
            current += step
        return chunks

    def _requests(self) -> List[Tuple[str, Dict[str, str], Callable[[Dict[str, Any]], Optional[pd.DataFrame]]]]:
        """(endpoint, query without the time range, parser) of every request one chunk takes."""
        out: List[Tuple[str, Dict[str, str], Callable[[Dict[str, Any]], Optional[pd.DataFrame]]]] = []
        country = self.country.lower()
        for t in self._types:
            if t == "forecast":
                for production in self._production_types:
                    for forecast in self._forecast_types:
                        if production in DAY_AHEAD_ONLY and forecast != "day-ahead":
                            continue
                        params = {"country": country, "production_type": production, "forecast_type": forecast}
                        name = forecast_column(production, forecast)
                        out.append(("public_power_forecast", params, self._single(name, "forecast_values")))
            elif t == "power":
                out.append(("public_power", {"country": country}, self._listed("production_types", "actual_")))
            elif t == "price":
                out.append(("price", {"bzn": str(self._zone)}, self._single("price_day_ahead", "price")))
            else:
                out.append((t, {"country": country}, self._listed("countries", f"{t}_")))
        return out

    # -- parsing ----------------------------------------------------------------

    @staticmethod
    def _index(payload: Dict[str, Any]) -> Optional[pd.DatetimeIndex]:
        seconds = payload.get("unix_seconds")
        if not seconds:
            return None
        return pd.DatetimeIndex(pd.to_datetime(pd.Series(seconds, dtype="int64"), unit="s", utc=True))

    def _single(self, column: str, key: str) -> Callable[[Dict[str, Any]], Optional[pd.DataFrame]]:
        def parse(payload: Dict[str, Any]) -> Optional[pd.DataFrame]:
            index = self._index(payload)
            if index is None:
                return None
            values = payload.get(key)
            if values is None or len(values) != len(index):
                raise EnergyChartsApiError(
                    f"EnergyCharts: '{key}' has {len(values or [])} values for {len(index)} stamps"
                )
            return pd.DataFrame({column: pd.Series(values, dtype="float64").to_numpy()}, index=index)

        return parse

    def _listed(self, key: str, stem: str) -> Callable[[Dict[str, Any]], Optional[pd.DataFrame]]:
        def parse(payload: Dict[str, Any]) -> Optional[pd.DataFrame]:
            index = self._index(payload)
            if index is None:
                return None
            data: Dict[str, Any] = {}
            for entry in payload.get(key) or []:
                column = f"{stem}{snake_case(entry['name'])}"
                values = entry.get("data")
                if values is None or len(values) != len(index):
                    raise EnergyChartsApiError(f"EnergyCharts: series '{entry['name']}' does not match the time stamps")
                data.setdefault(column, pd.Series(values, dtype="float64").to_numpy())
            frame = pd.DataFrame(data, index=index)
            if key == "production_types" and self.power:
                wanted = [f"actual_{snake_case(name).removeprefix('actual_')}" for name in self.power]
                missing = [column for column in wanted if column not in frame.columns]
                if missing:
                    raise ValueError(f"Series {missing} not in the {self.country} power data: {sorted(frame.columns)}")
                frame = frame[wanted]
            return frame

        return parse

    # -- HTTP ---------------------------------------------------------------------

    def _get(self, endpoint: str, params: Dict[str, str], start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, Any]:
        """One endpoint over [start, end): from the disk cache, or the API (throttled, 429 and 5xx retried)."""
        # The API's `end` is inclusive and unix seconds are unambiguous: the last second before `end`.
        query = dict(params, start=str(int(start.timestamp())), end=str(int(end.timestamp()) - 1))
        path = self._cache_path(endpoint, query)
        if path is not None and path.exists():
            try:
                return json.loads(path.read_text())
            except (OSError, ValueError):
                pass  # an unreadable file is a miss

        payload = self._request(endpoint, query, f"{endpoint} {start:%Y-%m-%d} to {end:%Y-%m-%d} {params}")
        if path is not None and end <= self._now() - self.UNSETTLED:
            self._cache_write(path, payload)
        return payload

    def _request(self, endpoint: str, query: Dict[str, str], label: str) -> Dict[str, Any]:
        bucket = _bucket(endpoint, current_clock())
        error = ""
        for attempt in range(self.MAX_RETRIES + 1):
            wait: Optional[float] = None
            bucket.acquire(self._clock, self._sleep)
            try:
                response = self.session.get(
                    f"{BASE_URL}/{endpoint}",
                    params=query,
                    headers={"User-Agent": _user_agent(), "Accept": "application/json"},
                    timeout=self.REQUEST_TIMEOUT,
                )
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            except requests.RequestException as exc:
                raise EnergyChartsApiError(f"EnergyCharts: {label} failed: {exc}") from None
            else:
                status = response.status_code
                if status == 200:
                    try:
                        payload = response.json()
                    except ValueError:
                        raise EnergyChartsApiError(f"EnergyCharts: {label}: the answer is not JSON") from None
                    if not isinstance(payload, dict):
                        raise EnergyChartsApiError(f"EnergyCharts: {label}: unexpected answer {str(payload)[:200]}")
                    return payload
                error = f"HTTP {status}: {(response.reason or response.text or '')[:200]}"
                if status != 429 and status < 500:
                    raise EnergyChartsApiError(f"EnergyCharts: {label} failed with {error}")
                wait = _retry_after_seconds(response.headers.get("Retry-After"))
                if status == 429:
                    wait = wait if wait is not None else 60.0
                    if wait > self.MAX_RETRY_WAIT_SECONDS:
                        raise EnergyChartsApiError(f"EnergyCharts: {label}: rate limited, Retry-After {wait:.0f} s")
                    bucket.block(self._clock(), wait)

            if attempt == self.MAX_RETRIES:
                break
            if wait is None:
                wait = self.RETRY_BACKOFF_SECONDS * 2**attempt
            wait = min(wait, self.MAX_RETRY_WAIT_SECONDS)
            self.logger.warning(
                f"EnergyCharts: {label}: {error}; retrying in {wait:.0f} s ({attempt + 1}/{self.MAX_RETRIES})"
            )
            self._sleep(wait)
        raise EnergyChartsApiError(f"EnergyCharts: {label} failed after {self.MAX_RETRIES + 1} attempts: {error}")

    # -- disk cache -----------------------------------------------------------------

    def _cache_path(self, endpoint: str, query: Dict[str, str]) -> Optional[Path]:
        if not self.use_cache:
            return None
        digest = hashlib.sha256(json.dumps([endpoint, sorted(query.items())]).encode()).hexdigest()[:32]
        base = Path(os.path.expanduser(os.path.expandvars(self.cache_dir))) if self.cache_dir else default_cache_dir()
        return base / f"{endpoint}-{digest}.json"

    def _cache_write(self, path: Path, payload: Dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, path)
        except OSError as exc:
            self.logger.warning(f"EnergyCharts: the answer was not cached: {exc}")

    def get_cache_config(self) -> dict:
        config: Dict[str, Any] = {
            "source_type": "energy_charts",
            "country": self.country.lower(),
            "types": sorted(self._types),
            "schema_version": SCHEMA_VERSION,
        }
        if "forecast" in self._types:
            config["forecast_types"] = sorted(self._forecast_types)
            config["production_types"] = sorted(self._production_types)
        if "price" in self._types:
            config["bzn"] = self._zone
        if "power" in self._types and self.power:
            config["power"] = sorted(self.power)
        if self.prefix:
            config["prefix"] = self.prefix
        return config
