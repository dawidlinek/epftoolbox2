import re
import unicodedata
from typing import Any, Dict, List, Optional, Union

import holidays as holidays_lib
import numpy as np
import pandas as pd
import requests
from astral import LocationInfo
from astral.sun import sun
from pandas.tseries.frequencies import to_offset

from epftoolbox2.logging import get_logger
from . import _openholidays
from .._market_timezone import resolve as _resolve_timezone
from .base import DataSource

logger = get_logger(__name__)

COUNTRY_DATA = {
    "PL": {"timezone": "Europe/Warsaw", "lat": 52.2297, "lon": 21.0122},
    "DE": {"timezone": "Europe/Berlin", "lat": 52.5200, "lon": 13.4050},
    "FR": {"timezone": "Europe/Paris", "lat": 48.8566, "lon": 2.3522},
    "ES": {"timezone": "Europe/Madrid", "lat": 40.4168, "lon": -3.7038},
    "IT": {"timezone": "Europe/Rome", "lat": 41.9028, "lon": 12.4964},
    "NL": {"timezone": "Europe/Amsterdam", "lat": 52.3676, "lon": 4.9041},
    "BE": {"timezone": "Europe/Brussels", "lat": 50.8503, "lon": 4.3517},
    "AT": {"timezone": "Europe/Vienna", "lat": 48.2082, "lon": 16.3738},
    "CH": {"timezone": "Europe/Zurich", "lat": 46.9480, "lon": 7.4474},
    "CZ": {"timezone": "Europe/Prague", "lat": 50.0755, "lon": 14.4378},
    "SK": {"timezone": "Europe/Bratislava", "lat": 48.1486, "lon": 17.1077},
    "HU": {"timezone": "Europe/Budapest", "lat": 47.4979, "lon": 19.0402},
    "RO": {"timezone": "Europe/Bucharest", "lat": 44.4268, "lon": 26.1025},
    "BG": {"timezone": "Europe/Sofia", "lat": 42.6977, "lon": 23.3219},
    "GR": {"timezone": "Europe/Athens", "lat": 37.9838, "lon": 23.7275},
    "PT": {"timezone": "Europe/Lisbon", "lat": 38.7223, "lon": -9.1393},
    "DK": {"timezone": "Europe/Copenhagen", "lat": 55.6761, "lon": 12.5683},
    "SE": {"timezone": "Europe/Stockholm", "lat": 59.3293, "lon": 18.0686},
    "NO": {"timezone": "Europe/Oslo", "lat": 59.9139, "lon": 10.7522},
    "FI": {"timezone": "Europe/Helsinki", "lat": 60.1699, "lon": 24.9384},
    "GB": {"timezone": "Europe/London", "lat": 51.5074, "lon": -0.1278},
    "IE": {"timezone": "Europe/Dublin", "lat": 53.3498, "lon": -6.2603},
}

WEEKDAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTH_NAMES = [
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
]

# Holiday names are requested in this language so that `holiday_name` values and one-hot column names do not
# depend on the process locale (without it, `holidays` falls back to LANG/LC_ALL) or the country's own language.
HOLIDAY_LANGUAGE = "en_US"

# Bump when this source's own computation changes (parsing, column layout, ...): every existing cache entry is
# then ignored. The installed `holidays` package's own version is part of the cache key too (see
# get_cache_config), since its holiday data changes between releases independently of this source's code.
SCHEMA_VERSION = 1

# Market-structure changes, policy interventions and shocks catalogued for `regimes=`. "start" is the first
# delivery day (local date); a regime dummy is 1 from that date on (through "end", inclusive, when given) for the
# listed countries only. "ex_ante" records whether the *end* date (open-ended entries: the date itself) was known
# in advance, at the time it would have been used by a forecaster — not identified afterwards from the data.
# "default" controls `regimes=True`: only entries with `default: True` whose `countries` include this source's
# `country` are added. It's `True` exactly when `ex_ante` is `True`, except for a few entries that are ex ante
# but still excluded from the default set by name below (`baltic_sync`: a system/frequency change with only an
# indirect link to day-ahead prices). Every entry is reachable by name regardless of `default`, via
# `regimes=["name", ...]` or `list_regimes()`.
DEFAULT_REGIMES = {
    # --- already shipped ---
    "de_at_split": {
        "start": "2018-10-01",
        "countries": ["DE", "AT"],
        "ex_ante": True,
        "default": True,
        "description": "Split of the German-Austrian bidding zone (DE-LU and AT)",
        "source": "https://www.entsoe.eu/",
    },
    "brexit_decoupling": {
        "start": "2021-01-01",
        "countries": ["GB", "IE"],
        "ex_ante": True,
        "default": True,
        "description": "GB and the Irish SEM leave implicit day-ahead coupling with SDAC",
        "source": "https://www.ofgem.gov.uk/",
    },
    "core_fb": {
        "start": "2022-06-09",
        "countries": "AT BE HR CZ FR DE HU LU NL PL RO SK SI".split(),
        "ex_ante": True,
        "default": True,
        "description": "Core flow-based day-ahead market coupling (first auction 2022-06-08)",
        "source": "https://www.jao.eu/news/core-flow-based-market-coupling-project-go-live",
    },
    "nordic_fb": {
        "start": "2024-10-30",
        "countries": ["DK", "FI", "NO", "SE"],
        "ex_ante": True,
        "default": True,
        "description": "Nordic flow-based day-ahead market coupling (first auction 2024-10-29)",
        "source": "https://www.nordpoolgroup.com/",
    },
    "sdac_15min_mtu": {
        "start": "2025-10-01",
        "countries": "AT BE BG CZ DE DK ES FI FR GR HU IT NL NO PL PT RO SE SK".split(),
        "ex_ante": True,
        "default": True,
        "description": "15-minute market time unit in SDAC (first auction 2025-09-30)",
        "source": "https://www.nemo-committee.eu/assets/files/successful-implementation-of-15-minute-market-time-unit-(mtu)-in-sdac.pdf",
    },
    "sem_30min_mtu": {
        "start": "2025-10-01",
        "countries": ["IE"],
        "ex_ante": True,
        "default": True,
        "description": "30-minute market time unit in the Irish SEM day-ahead market",
        "source": "https://www.semopx.com/market-messages/mcsc-confirms-sdac-15min-mtusem-30min-mtu-go-live-30-sept-2025",
    },
    # --- NWE/CWE/SWE/Nordic coupling, GB, I-SEM ---
    "cwe_market_coupling": {
        "start": "2010-11-09",
        "countries": ["BE", "FR", "DE", "NL"],
        "ex_ante": True,
        "default": True,
        "description": "Central Western Europe day-ahead coupling go-live (Trilateral FR-BE-NL coupling extended to DE)",
        "source": "https://www.epexspot.com/en/news/10-years-first-market-coupling-initiative-tlc",
    },
    "nwe_price_coupling": {
        "start": "2014-02-05",
        "countries": ["BE", "FR", "DE", "NL", "DK", "SE", "NO", "FI", "PL"],
        "ex_ante": True,
        "default": True,
        "description": (
            "North-Western Europe day-ahead price coupling go-live (first live PCR/Euphemia run); "
            "PL included only via the SwePol HVDC link"
        ),
        "source": "https://www.nordpoolgroup.com/en/message-center-container/newsroom/exchange-message-list/2014/Q1/No-052014---/",
    },
    "gb_nwe_coupling": {
        "start": "2014-02-05",
        "end": "2020-12-31",
        "countries": ["GB"],
        "ex_ante": True,
        "default": True,
        "description": "GB implicitly coupled to NWE/SDAC via IFA/BritNed/Moyle, ending with Brexit (see brexit_decoupling)",
        "source": "https://www.nordpoolgroup.com/en/message-center-container/newsroom/exchange-message-list/2014/Q1/No-052014---/",
    },
    "swe_market_coupling": {
        "start": "2014-05-14",
        "countries": ["ES", "PT", "FR"],
        "ex_ante": True,
        "default": True,
        "description": "South-Western Europe day-ahead market coupling go-live (Iberia joins via the FR border)",
        "source": "https://www.epexspot.com/en/news/launch-full-swe-coupling-13-may-2014",
    },
    "cwe_flow_based": {
        "start": "2015-05-21",
        "countries": ["BE", "FR", "DE", "NL"],
        "ex_ante": True,
        "default": True,
        "description": "Flow-based (rather than ATC/NTC) day-ahead capacity allocation on CWE-internal borders",
        "source": "https://www.transnetbw.de/en/newsroom/press-releases/flow-based-methodology-for-cwe-market-coupling-successfully-launched",
    },
    "isem_start": {
        "start": "2018-10-01",
        "countries": ["IE"],
        "ex_ante": True,
        "default": True,
        "description": "I-SEM go-live: Integrated Single Electricity Market replaces SEM, couples to SDAC via interconnectors",
        "source": "https://www.semcommittee.com/news/new-all-island-wholesale-electricity-market-goes-live",
    },
    "baltic_sync": {
        "start": "2025-02-09",
        "countries": ["EE", "LV", "LT"],
        "ex_ante": True,
        "default": False,
        "description": (
            "Baltic states desynchronise from the Russia/Belarus BRELL ring and synchronise with the Continental "
            "European grid; primarily a frequency/system change, not a day-ahead coupling change (the Baltics "
            "were already SDAC-coupled). Not in COUNTRY_DATA yet, so name-only"
        ),
        "source": "https://www.entsoe.eu/news/2025/02/09/entso-e-confirms-successful-synchronization-of-the-continental-european-electricity-system-with-the-systems-of-the-baltic-countries/",
    },
    # --- CEE/SEE coupling and market launches ---
    "cz_sk_hu_coupling_2012": {
        "start": "2012-09-12",
        "countries": ["CZ", "SK", "HU"],
        "ex_ante": True,
        "default": True,
        "description": "Hungary joins the CZ-SK day-ahead coupling (new SK-HU border)",
        "source": "https://www.okte.sk/media/27706/20120816_%20mc%20sg%20meeting%20press%20release_final.pdf",
    },
    "cee_4m_coupling": {
        "start": "2014-11-20",
        "end": "2021-06-17",
        "countries": ["CZ", "SK", "HU", "RO"],
        "ex_ante": True,
        "default": True,
        "description": (
            "4M Market Coupling (RO joins CZ/SK/HU): own PCR session, gate closure 11:00 CET, limits -500/+3000; "
            "ends when it merges into the single SDAC session (interim_coupling_project)"
        ),
        "source": "https://old.okte.sk/en/information/news/2014/11/14/4m-market-coupling-launch/",
    },
    "gr_dam_launch": {
        "start": "2020-11-01",
        "countries": ["GR"],
        "ex_ante": True,
        "default": True,
        "description": "HEnEx day-ahead market replaces the DAS mandatory pool run since 2005",
        "source": "https://www.enexgroup.gr/web/guest/dam-idm-archive",
    },
    "it_gr_coupling": {
        "start": "2020-12-16",
        "countries": ["GR", "IT"],
        "ex_ante": True,
        "default": True,
        "description": "Greece joins SDAC through its only coupled border (500 MW IT-GR HVDC)",
        "source": "https://www.nemo-committee.eu/assets/files/extension-of-single-day-ahead-coupling-(sdac)-to-greece-.pdf",
    },
    "bg_dam_launch": {
        "start": "2016-01-20",
        "end": "2021-05-11",
        "countries": ["BG"],
        "ex_ante": False,
        "default": False,
        "description": "First Bulgarian day-ahead market (isolated, MRC tools but no coupled border) until GR-BG coupling",
        "source": "http://umm.ibex.bg/en/announcements/news/first-successful-real-trading-session.html",
    },
    "gr_bg_coupling": {
        "start": "2021-05-12",
        "countries": ["BG", "GR"],
        "ex_ante": True,
        "default": True,
        "description": "Bulgaria's isolation ends; GR-BG becomes implicit NTC",
        "source": "https://www.nemo-committee.eu/assets/files/210511%20Press%20release%20successful%20go-live%20BG-GR%20coupling-3b29443c09f164416f6b8bd9eda896e2.pdf",
    },
    "interim_coupling_project": {
        "start": "2021-06-18",
        "countries": ["PL", "CZ", "SK", "HU", "AT", "DE", "RO"],
        "ex_ante": True,
        "default": True,
        "description": "Poland's synchronous borders coupled for the first time; 4M MC merges into one SDAC session",
        "source": "https://www.entsoe.eu/news/2021/06/18/successful-go-live-of-interim-coupling-reaching-the-sdac-enduring-phase/",
    },
    "bg_ro_coupling": {
        "start": "2021-10-28",
        "countries": ["BG", "RO"],
        "ex_ante": True,
        "default": True,
        "description": "Closes the SEE coupling loop (NTC)",
        "source": "https://www.nemo-committee.eu/assets/files/211027%20PR%20successful%20go-live%20BG-RO%20border-281d241324d5ce9a4049381c71f4021d.pdf",
    },
    # --- bidding zones ---
    "se_bidding_zones": {
        "start": "2011-11-01",
        "countries": ["SE"],
        "ex_ante": True,
        "default": True,
        "description": "Sweden split into four bidding zones SE1-SE4",
        "source": "https://www.nordpoolgroup.com/4a7b04/globalassets/download-center/day-ahead/elspot-area-change-log.pdf",
    },
    "it_zone_calabria": {
        "start": "2021-01-01",
        "countries": ["IT"],
        "ex_ante": True,
        "default": True,
        "description": "New Calabria (IT-CALA) bidding zone split from South; first Terna zonal review since 2009",
        "source": "https://lightbox.terna.it/en/insight/new-electricity-market-zones",
    },
    # --- harmonised SDAC price limits ---
    "hmmcp_4000": {
        "start": "2022-05-11",
        "countries": "AT BE BG CZ DE DK ES FI FR GR HU IT NL NO PL PT RO SE SK".split(),
        "ex_ante": True,
        "default": True,
        "description": "Harmonised maximum clearing price for SDAC raised from 3000 to 4000 EUR/MWh",
        "source": "https://www.epexspot.com/en/news/harmonised-maximum-clearing-price-sdac-be-set-4000-eurmwh-10th-may-2022",
    },
    "hmmcp_min600": {
        "start": "2026-05-29",
        "countries": "AT BE BG CZ DE DK ES FI FR GR HU IT NL NO PL PT RO SE SK".split(),
        "ex_ante": True,
        "default": True,
        "description": "Harmonised minimum clearing price for SDAC lowered from -500 to -600 EUR/MWh",
        "source": "https://www.nemo-committee.eu/assets/files/harmonised-minimum-clearing-price-for-sdac-to-be-set-to-600-eur-per-mwh-starting-from-the-28th-may-2026-(trading-date).pdf",
    },
    # --- policy interventions ---
    "iberian_exception": {
        "start": "2022-06-15",
        "end": "2023-12-31",
        "countries": ["ES", "PT"],
        "ex_ante": False,
        "default": False,
        "description": (
            "Gas-price cap on the marginal fuel entering the Iberian wholesale market (RDL 10/2022); the final "
            "end date was only fixed by RDL 3/2023 on 2023-03-28, well after the mechanism started"
        ),
        "source": "https://www.energiaysociedad.es/regulaciones/en-que-consiste-la-excepcion-iberica-real-decreto-ley-rdl-10-2022/",
    },
    "de_nuclear_phaseout": {
        "start": "2023-04-15",
        "countries": ["DE"],
        "ex_ante": True,
        "default": True,
        "description": (
            "Last 3 German nuclear reactors (Isar 2, Neckarwestheim 2, Emsland) shut down for good; extension "
            "from 2022-12-31 decided by the Bundestag 2022-11-11"
        ),
        "source": "https://www.base.bund.de/en/nuclear-safety/nuclear-phase-out/nuclear-phase-out.html",
    },
    "be_nuclear_doel3_tihange2": {
        "start": "2023-02-01",
        "countries": ["BE"],
        "ex_ante": True,
        "default": True,
        "description": "Doel 3 (closed 2022-09-23) and Tihange 2 (closed 2023-01-31) permanently shut down, ~2 GW leaves the mix",
        "source": "https://afcn.fgov.be/",
    },
    "arenh": {
        "start": "2011-07-01",
        "end": "2025-12-31",
        "countries": ["FR"],
        "ex_ante": True,
        "default": True,
        "description": (
            "Regulated Access to Incumbent Nuclear Electricity: EDF sells nuclear output to competitors at a "
            "regulated price, capping the effective wholesale price; replaced 2026-01-01 by the Versement "
            "Nucleaire Universel/CAPN mechanism"
        ),
        "source": "https://www.8advisory.com/en/2026/03/31/nuclear-power-navigating-frances-post-arenh-market-transformation/",
    },
    "pl_capacity_market": {
        "start": "2021-01-01",
        "countries": ["PL"],
        "ex_ante": True,
        "default": True,
        "description": "Polish capacity market (Act of 2017-12-08): first capacity delivery year",
        "source": "https://www.forum-energii.eu/en/the-capacity-market-in-poland-more-expensive-than-ever",
    },
    # --- shocks (open-ended only) ---
    "war_ukraine": {
        "start": "2022-02-24",
        "countries": "AT BE BG CZ DE DK ES FI FR GR HU IT NL NO PL PT RO SE SK GB IE CH".split(),
        "ex_ante": False,
        "default": False,
        "description": "Russian invasion of Ukraine; open-ended shock marker for the European gas/power crisis onset",
        "source": "well-known date, not from a single citable market document",
    },
}


def list_regimes(country: Optional[str] = None) -> pd.DataFrame:
    """`DEFAULT_REGIMES` as a DataFrame (name, start, end, countries, ex_ante, default, description, source), one
    row per catalog entry, in catalog order. With `country`, only entries whose `countries` include it (case
    insensitive)."""
    country = country.upper() if country is not None else None
    rows = [
        {
            "name": name,
            "start": spec["start"],
            "end": spec.get("end"),
            "countries": list(spec["countries"]),
            "ex_ante": spec["ex_ante"],
            "default": spec["default"],
            "description": spec["description"],
            "source": spec["source"],
        }
        for name, spec in DEFAULT_REGIMES.items()
        if country is None or country in spec["countries"]
    ]
    columns = ["name", "start", "end", "countries", "ex_ante", "default", "description", "source"]
    return pd.DataFrame(rows, columns=columns)


def _sanitise_name(name: str) -> str:
    """'New Year's Day' -> 'new_years_day': ASCII, lower case, words joined by underscores."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    ascii_name = ascii_name.replace("'", "")
    return re.sub(r"[^0-9a-z]+", "_", ascii_name.lower()).strip("_") or "unnamed"


def _grid_step(freq: str) -> pd.Timedelta:
    """Row spacing of a calendar grid: a divisor of one hour, or whole hours dividing a day."""
    try:
        step = pd.Timedelta(to_offset(freq))
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid freq {freq!r}: must be a fixed frequency such as '1h' or '15min'") from e
    hour, day, zero = pd.Timedelta(hours=1), pd.Timedelta(days=1), pd.Timedelta(0)
    divides_hour = zero < step <= hour and hour % step == zero
    whole_hours = step >= hour and step % hour == zero and day % step == zero
    if not (divides_hour or whole_hours):
        raise ValueError(f"Invalid freq {freq!r}: must divide one hour (e.g. '15min') or be whole hours dividing a day")
    return step


def _local_grid(start: pd.Timestamp, end: pd.Timestamp, step: pd.Timedelta, timezone: str):
    """UTC grid over [start, end), its local-time view, and the local date of each row (codes into dates)."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
    end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")
    if end <= start:
        raise ValueError(f"End timestamp ({end}) must be after start timestamp ({start})")
    index = pd.date_range(start.ceil(step), end, freq=step, inclusive="left")
    local_index = index.tz_convert(timezone)
    codes, dates = pd.factorize(local_index.tz_localize(None).normalize())
    return index, local_index, codes, pd.DatetimeIndex(dates)


class CalendarSource(DataSource):
    """Calendar features on a UTC grid, computed from local dates and local wall-clock times.

    Every feature is known in advance, so its `_d+k` versions are legal predictors.

    With `school_holidays="national"` or `"all"`, it also downloads school holiday flags from the OpenHolidays API
    (https://www.openholidaysapi.org, ODbL 1.0 licence: credit it when you publish results or share a dataset
    built from it, e.g. `epftoolbox2.data.sources._openholidays.ATTRIBUTION`). This is the only case where
    CalendarSource makes network calls; without it, it stays offline and deterministic. Values are NaN before
    the API's history starts for the country, or for school years not yet published, and a warning names the
    affected dates; entries marked as half days are ignored (not counted as a holiday).

    Example:
        >>> source = CalendarSource(country="DE", region="BY", holidays="binary", weekday="onehot")
        >>> df = source.fetch(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))

    Example:
        >>> source = CalendarSource(country="PL", school_holidays="all")
        >>> # columns include: school_holiday_share (fraction of Poland's 16 voivodeships on school holiday)
    """

    # Set by DataPipeline._bind_component_timezones() when `timezone` is left unset; never written by users.
    _pipeline_timezone: Optional[str] = None

    def __init__(
        self,
        country: str,
        timezone: Optional[str] = None,
        holidays: Union[str, bool] = "binary",
        weekday: Union[str, bool] = "number",
        hour: Union[str, bool] = False,
        month: Union[str, bool] = False,
        daylight: bool = False,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        prefix: str = "",
        freq: str = "1h",
        region: Optional[str] = None,
        categories: Optional[Union[str, List[str]]] = None,
        bridge_days: bool = False,
        christmas_period: bool = False,
        holiday_distance: int = 0,
        dst_switch: bool = False,
        regimes: Union[bool, List[str], Dict[str, Any]] = False,
        school_holidays: Union[bool, str] = False,
    ):
        """
        Args:
            country: ISO country code (see COUNTRY_DATA).
            timezone: Time zone whose local dates and wall-clock times define the features. None (default)
                inherits `DataPipeline.timezone` when this source is part of a pipeline; standalone, it falls
                back to the country's own zone.
            holidays: False | "binary" | "onehot" | "name".
            weekday: False | "number" | "onehot" | "name".
            hour: False | "number" | "onehot". Hourly or coarser `freq`: local hour of day (`hour`, `is_{h}`).
                Sub-hourly `freq`: index of the period within the local day (`period`, `is_period_{p}`),
                e.g. 0-95 at 15 minutes, numbered like the models' own period index.
            month: False | "number" | "onehot" | "name".
            daylight: Add sunrise, sunset (local decimal hours) and daylight_hours.
            lat, lon: Location for daylight (default: the country's capital).
            prefix: Prefix for column names (no separator is added).
            freq: Row spacing. Must divide one hour, or be a whole number of hours that divides one day.
            region: Subdivision code passed to the `holidays` package for regional public holidays, e.g. "BY"
                or "DE-BY" (Bavaria), "ENG" (England), "ZH" (Zurich).
            categories: Holiday categories of the `holidays` package, e.g. ["public", "bank"]
                (default: the package default, public holidays only).
            bridge_days: Add `is_bridge_day`: a working day whose previous and next days are both a weekend
                day or a holiday (e.g. the Friday after a Thursday holiday).
            christmas_period: Add `is_christmas_period`: 24 December to 1 January inclusive.
            holiday_distance: If N > 0, add `days_to_holiday` and `days_since_holiday` (0 on a holiday), capped at N.
            dst_switch: Add `is_dst_switch_day`: local days with 23 or 25 hours.
            regimes: Dummies `regime_{name}`, 1 on local dates start <= D <= end and 0 otherwise (end inclusive;
                open-ended when there is no end). False: none. True: every `DEFAULT_REGIMES` entry with
                `default: True` whose `countries` include `country`. A list of names: exactly those catalog
                entries (an unknown name raises, listing the known ones; a country not in an entry's
                `countries` only logs a warning, the column is still added). A dict replaces the catalog
                entirely with custom periods, one entry per name: a date string (open-ended,
                `{name: "YYYY-MM-DD"}`), a `(start, end)` pair, or `{name: {"start": ..., "end": ...}}`
                (`"end"` optional). A bounded period is only leak-free if its end date was announced in
                advance, not identified afterwards from the data (an ex-post label like "energy crisis
                2021-2023" is look-ahead in a backtest). `DEFAULT_REGIMES` also carries this per entry as
                `ex_ante`; see `list_regimes()` to inspect the catalog.
            school_holidays: False (default) | True | "national" | "all". Download school holiday flags from
                the OpenHolidays API (network calls happen only when this is not False). True is the same as
                "national": column `is_school_holiday`, 1.0 on a nationwide school-holiday day, 0.0 otherwise.
                "all": column `school_holiday_share`, the unweighted fraction of the country's top-level
                regions on school holiday that day (a nationwide period counts for every region).
        """
        self.country = country.upper()
        self._validate_country()

        country_info = COUNTRY_DATA[self.country]
        self.timezone = timezone
        self.lat = lat if lat is not None else country_info["lat"]
        self.lon = lon if lon is not None else country_info["lon"]

        self.holidays = holidays
        self.weekday = weekday
        self.hour = hour
        self.month = month
        self.daylight = daylight
        self.prefix = prefix
        self.freq = freq
        self.region = self._normalise_region(region)
        self.categories = [categories] if isinstance(categories, str) else categories
        if self.categories is not None:
            self.categories = list(self.categories)
        self.bridge_days = bridge_days
        self.christmas_period = christmas_period
        self.holiday_distance = self._validate_count("holiday_distance", holiday_distance, maximum=366)
        self.dst_switch = dst_switch
        self.regimes = self._normalise_regimes(regimes)
        self.school_holidays = self._normalise_school_holidays(school_holidays)

        self.session = requests.Session()

        self._validate_config()

    @staticmethod
    def _normalise_school_holidays(value) -> Union[bool, str]:
        if value is True:
            return "national"
        if value is False or value in ("national", "all"):
            return value
        raise ValueError(f"school_holidays must be False, True, 'national' or 'all', got {value!r}")

    @property
    def market_timezone(self) -> str:
        """Resolved zone: `timezone` if set, else the one a `DataPipeline` inherited it from, else the
        country's own zone (COUNTRY_DATA) for standalone use."""
        return _resolve_timezone(self.timezone, self) or COUNTRY_DATA[self.country]["timezone"]

    @staticmethod
    def _validate_count(name: str, value, maximum: Optional[int] = None) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        if value < 0 or (maximum is not None and value > maximum):
            raise ValueError(f"{name} must be between 0 and {maximum if maximum is not None else 'inf'}, got {value}")
        return int(value)

    @staticmethod
    def _normalise_regimes(regimes) -> Union[bool, List[str], Dict[str, Any]]:
        if isinstance(regimes, bool):
            return regimes
        if isinstance(regimes, (list, tuple)):
            names = list(regimes)
            if not names:
                raise ValueError("regimes: a list must name at least one DEFAULT_REGIMES entry")
            unknown = [n for n in names if n not in DEFAULT_REGIMES]
            if unknown:
                raise ValueError(f"regimes: unknown name(s) {unknown}. Known: {sorted(DEFAULT_REGIMES)}")
            return names
        if not isinstance(regimes, dict) or not regimes:
            raise ValueError(
                "regimes must be True, False, a non-empty list of DEFAULT_REGIMES names, or a non-empty dict of "
                f"{{name: date}}, {{name: (start, end)}} or {{name: {{'start': ..., 'end': ...}}}}, got {regimes!r}"
            )
        normalised = {}
        for name, spec in regimes.items():
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name):
                raise ValueError(f"regimes: name {name!r} must contain only letters, digits and underscores")
            start, end = CalendarSource._parse_regime_period(name, spec)
            normalised[name] = start if end is None else {"start": start, "end": end}
        return normalised

    @staticmethod
    def _parse_regime_period(name: str, spec) -> tuple:
        """A regime spec (a date, a `(start, end)` pair, or `{"start": ..., "end": ...}`) as (start, end) ISO
        date strings; `end` is None when the period is open-ended."""
        if isinstance(spec, dict):
            unknown = set(spec) - {"start", "end"}
            if unknown or "start" not in spec:
                raise ValueError(
                    f"regimes: {name!r} as a dict must be {{'start': ..., 'end': ...}} ('end' optional), got {spec!r}"
                )
            start, end = spec["start"], spec.get("end")
        elif isinstance(spec, (tuple, list)):
            if len(spec) != 2:
                raise ValueError(f"regimes: {name!r} as a pair must be (start, end), got {spec!r}")
            start, end = spec
        else:
            start, end = spec, None

        start_iso = CalendarSource._regime_date(name, start, "start")
        end_iso = None if end is None else CalendarSource._regime_date(name, end, "end")
        if end_iso is not None and end_iso < start_iso:
            raise ValueError(f"regimes: {name!r} end {end_iso} is before its start {start_iso}")
        return start_iso, end_iso

    @staticmethod
    def _regime_date(name: str, value: Any, which: str) -> str:
        try:
            return pd.Timestamp(value).date().isoformat()
        except (ValueError, TypeError) as e:
            raise ValueError(f"regimes: {name!r} has an invalid {which} date {value!r}") from e

    def _validate_country(self):
        if self.country not in COUNTRY_DATA:
            raise ValueError(f"Unsupported country: {self.country}. Supported: {list(COUNTRY_DATA.keys())}")

    def _normalise_region(self, region: Optional[str]) -> Optional[str]:
        if region is None:
            return None
        code = str(region).strip()
        if code.upper().startswith(f"{self.country}-"):
            code = code[len(self.country) + 1 :]
        known = getattr(type(holidays_lib.country_holidays(self.country)), "subdivisions", ())
        return code.upper() if code.upper() in known else code

    def _validate_config(self):
        valid_holiday = {False, "binary", "onehot", "name"}
        valid_weekday = {False, "number", "onehot", "name"}
        valid_hour = {False, "number", "onehot"}
        valid_month = {False, "number", "onehot", "name"}

        if self.holidays not in valid_holiday:
            raise ValueError(f"Invalid holidays value: {self.holidays}. Valid: {valid_holiday}")
        if self.weekday not in valid_weekday:
            raise ValueError(f"Invalid weekday value: {self.weekday}. Valid: {valid_weekday}")
        if self.hour not in valid_hour:
            raise ValueError(f"Invalid hour value: {self.hour}. Valid: {valid_hour}")
        if self.month not in valid_month:
            raise ValueError(f"Invalid month value: {self.month}. Valid: {valid_month}")

        self._step = _grid_step(self.freq)

        cls = type(holidays_lib.country_holidays(self.country))
        if self.region is not None:
            try:
                holidays_lib.country_holidays(self.country, subdiv=self.region)
            except NotImplementedError as e:
                raise ValueError(
                    f"Invalid region {self.region!r} for {self.country}. Valid: {list(getattr(cls, 'subdivisions', ()))}"
                ) from e
        if self.categories is not None:
            supported = set(getattr(cls, "supported_categories", ()))
            unknown = [c for c in self.categories if c not in supported]
            if unknown or not self.categories:
                raise ValueError(f"Invalid categories {unknown} for {self.country}. Valid: {sorted(supported)}")

        for name in ("bridge_days", "christmas_period", "dst_switch"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be True or False, got {getattr(self, name)!r}")

    def _holiday_language(self) -> Optional[str]:
        cls = type(holidays_lib.country_holidays(self.country))
        # Countries without translations (GB, IE) only have English names.
        return HOLIDAY_LANGUAGE if HOLIDAY_LANGUAGE in (getattr(cls, "supported_languages", ()) or ()) else None

    def _holiday_calendar(self, years: List[int]) -> holidays_lib.HolidayBase:
        kwargs = {"years": years, "subdiv": self.region, "language": self._holiday_language()}
        if self.categories is not None:
            kwargs["categories"] = tuple(self.categories)
        if self.country == "SE":
            # holidays.Sweden lists every Sunday as a holiday unless told not to.
            return holidays_lib.Sweden(include_sundays=False, **kwargs)
        return holidays_lib.country_holidays(self.country, **kwargs)

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Rows on the `freq` grid in [start, end), indexed in UTC."""
        # Features that depend only on the local date are computed once per date and broadcast to its rows.
        index, local_index, codes, local_dates = _local_grid(start, end, self._step, self.market_timezone)

        df = pd.DataFrame(index=index)

        if self.holidays:
            df = self._add_holidays(df, local_dates, codes)
        if self.weekday:
            df = self._add_weekday(df, local_index)
        if self.hour:
            df = self._add_hour(df, local_index)
        if self.month:
            df = self._add_month(df, local_index)
        if self.daylight:
            df = self._add_daylight(df, local_dates, codes)
        if self.bridge_days or self.holiday_distance:
            df = self._add_holiday_context(df, local_dates, codes)
        if self.christmas_period:
            christmas = ((local_dates.month == 12) & (local_dates.day >= 24)) | (
                (local_dates.month == 1) & (local_dates.day == 1)
            )
            df[f"{self.prefix}is_christmas_period"] = np.asarray(christmas, dtype=int)[codes]
        if self.dst_switch:
            df = self._add_dst_switch(df, local_dates, codes)
        if self.regimes:
            for name, spec in self._resolved_regimes().items():
                start, end = (spec, None) if isinstance(spec, str) else (spec["start"], spec["end"])
                flag = local_dates >= pd.Timestamp(start)
                if end is not None:
                    flag &= local_dates <= pd.Timestamp(end)
                df[f"{self.prefix}regime_{name}"] = np.asarray(flag, dtype=int)[codes]
        if self.school_holidays:
            df = self._add_school_holidays(df, local_dates, codes)

        return df

    def _holiday_flags(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """True where the local date is a holiday of the configured calendar (country, region, categories)."""
        if not len(dates):
            return np.zeros(0, dtype=bool)
        calendar = self._holiday_calendar(list(range(dates.min().year, dates.max().year + 1)))
        return np.array([date in calendar for date in dates.date], dtype=bool)

    def _add_holiday_context(self, df: pd.DataFrame, local_dates: pd.DatetimeIndex, codes: np.ndarray) -> pd.DataFrame:
        # Look far enough beyond the range that neighbours and the nearest holidays of its edge days are known.
        margin = pd.Timedelta(days=max(1, self.holiday_distance))
        if len(local_dates):
            span = pd.date_range(local_dates.min() - margin, local_dates.max() + margin, freq="D")
        else:
            span = pd.DatetimeIndex([])
        holiday = self._holiday_flags(span)
        rows = span.get_indexer(local_dates)[codes]

        if self.bridge_days:
            day_off = holiday | (span.dayofweek >= 5)
            bridge = np.zeros(len(span), dtype=bool)
            bridge[1:-1] = ~day_off[1:-1] & day_off[:-2] & day_off[2:]
            df[f"{self.prefix}is_bridge_day"] = bridge.astype(int)[rows]

        if self.holiday_distance:
            cap = self.holiday_distance
            positions = np.arange(len(span))
            holidays_at = np.flatnonzero(holiday)
            to_next = np.full(len(span), cap)
            since_last = np.full(len(span), cap)
            if len(holidays_at):
                nxt = np.searchsorted(holidays_at, positions, side="left")
                has_next = nxt < len(holidays_at)
                to_next[has_next] = holidays_at[nxt[has_next]] - positions[has_next]
                prv = np.searchsorted(holidays_at, positions, side="right") - 1
                has_prev = prv >= 0
                since_last[has_prev] = positions[has_prev] - holidays_at[prv[has_prev]]
            df[f"{self.prefix}days_to_holiday"] = np.minimum(to_next, cap)[rows]
            df[f"{self.prefix}days_since_holiday"] = np.minimum(since_last, cap)[rows]
        return df

    def _add_dst_switch(self, df: pd.DataFrame, local_dates: pd.DatetimeIndex, codes: np.ndarray) -> pd.DataFrame:
        # A local day has 23 or 25 hours when its UTC offset at noon differs from the previous day's.
        def utc_offset(naive: pd.DatetimeIndex) -> np.ndarray:
            aware = naive.tz_localize(
                self.market_timezone, ambiguous=np.zeros(len(naive), dtype=bool), nonexistent="shift_forward"
            )
            return np.asarray(aware.tz_localize(None) - aware.tz_convert("UTC").tz_localize(None))

        noon = local_dates + pd.Timedelta(hours=12)
        switch = utc_offset(noon) != utc_offset(noon - pd.Timedelta(days=1))
        df[f"{self.prefix}is_dst_switch_day"] = switch.astype(int)[codes]
        return df

    def _resolved_regimes(self) -> Dict[str, Any]:
        """Each entry as `_normalise_regimes` would store a custom one: a bare ISO date, or {"start", "end"}."""
        if self.regimes is True:
            return {
                name: self._catalog_period(spec)
                for name, spec in DEFAULT_REGIMES.items()
                if spec["default"] and self.country in spec["countries"]
            }
        if isinstance(self.regimes, list):
            result = {}
            for name in self.regimes:
                spec = DEFAULT_REGIMES[name]
                if self.country not in spec["countries"]:
                    logger.warning(
                        f"regimes: {name!r} does not list {self.country} among its countries "
                        f"({spec['countries']}); adding the column anyway"
                    )
                result[name] = self._catalog_period(spec)
            return result
        return dict(self.regimes) if self.regimes else {}

    @staticmethod
    def _catalog_period(spec: dict) -> Any:
        return {"start": spec["start"], "end": spec["end"]} if spec.get("end") else spec["start"]

    def _add_holidays(self, df: pd.DataFrame, local_dates: pd.DatetimeIndex, codes: np.ndarray) -> pd.DataFrame:
        years = list(range(local_dates.min().year, local_dates.max().year + 1)) if len(local_dates) else []
        calendar = self._holiday_calendar(years)
        names_per_date = [calendar.get_list(d) for d in local_dates.date]
        is_holiday = np.array([len(names) > 0 for names in names_per_date], dtype=int)

        if self.holidays in ("binary", "onehot"):
            df[f"{self.prefix}is_holiday"] = is_holiday[codes] if len(codes) else np.array([], dtype=int)
        if self.holidays == "name":
            joined = np.array(["; ".join(names) if names else None for names in names_per_date], dtype=object)
            df[f"{self.prefix}holiday_name"] = joined[codes] if len(codes) else np.array([], dtype=object)
        elif self.holidays == "onehot":
            # One column per holiday of the years covered, so the column set does not depend on the exact range.
            columns = sorted({_sanitise_name(n) for d in calendar for n in calendar.get_list(d)})
            position = {name: i for i, name in enumerate(columns)}
            per_date = np.zeros((len(local_dates), len(columns)), dtype=int)
            for i, names in enumerate(names_per_date):
                for name in names:
                    per_date[i, position[_sanitise_name(name)]] = 1
            dummies = pd.DataFrame(
                per_date[codes], index=df.index, columns=[f"{self.prefix}holiday_{c}" for c in columns]
            )
            df = pd.concat([df, dummies], axis=1)
        return df

    def _add_weekday(self, df: pd.DataFrame, local_index: pd.DatetimeIndex) -> pd.DataFrame:
        weekday_series = pd.Series(local_index.weekday, index=df.index)

        if self.weekday == "number":
            df[f"{self.prefix}weekday"] = weekday_series
        elif self.weekday == "name":
            df[f"{self.prefix}weekday_name"] = weekday_series.map(lambda x: WEEKDAY_NAMES[x])
        elif self.weekday == "onehot":
            for i, name in enumerate(WEEKDAY_NAMES):
                df[f"{self.prefix}is_{name}"] = (weekday_series == i).astype(int)
        return df

    def _add_hour(self, df: pd.DataFrame, local_index: pd.DatetimeIndex) -> pd.DataFrame:
        if self._step < pd.Timedelta(hours=1):
            # Sub-hourly: index of the period within the local day on the wall-clock grid (0-95 at 15 minutes).
            seconds = local_index.hour * 3600 + local_index.minute * 60 + local_index.second
            values = seconds // int(self._step.total_seconds())
            name, onehot_prefix = "period", "is_period_"
            n_values = int(pd.Timedelta(days=1) / self._step)
        else:
            values = local_index.hour
            name, onehot_prefix = "hour", "is_"
            n_values = 24
        series = pd.Series(np.asarray(values), index=df.index)

        if self.hour == "number":
            df[f"{self.prefix}{name}"] = series
        elif self.hour == "onehot":
            dummies = {f"{self.prefix}{onehot_prefix}{v}": (series == v).astype(int) for v in range(n_values)}
            df = pd.concat([df, pd.DataFrame(dummies, index=df.index)], axis=1)
        return df

    def _add_month(self, df: pd.DataFrame, local_index: pd.DatetimeIndex) -> pd.DataFrame:
        month_series = pd.Series(local_index.month, index=df.index)

        if self.month == "number":
            df[f"{self.prefix}month"] = month_series
        elif self.month == "name":
            df[f"{self.prefix}month_name"] = month_series.map(lambda x: MONTH_NAMES[x - 1])
        elif self.month == "onehot":
            for i, name in enumerate(MONTH_NAMES, start=1):
                df[f"{self.prefix}is_{name}"] = (month_series == i).astype(int)
        return df

    def _add_daylight(self, df: pd.DataFrame, local_dates: pd.DatetimeIndex, codes: np.ndarray) -> pd.DataFrame:
        location = LocationInfo(latitude=self.lat, longitude=self.lon, timezone=self.market_timezone)

        per_date = np.full((len(local_dates), 3), np.nan)
        missing = []
        for i, date in enumerate(local_dates.date):
            try:
                s = sun(location.observer, date=date, tzinfo=location.timezone)
            except ValueError:  # polar day or night: the sun never rises or never sets
                missing.append(date)
                continue
            sunrise, sunset = s["sunrise"], s["sunset"]
            per_date[i] = (
                sunrise.hour + sunrise.minute / 60,
                sunset.hour + sunset.minute / 60,
                (sunset - sunrise).total_seconds() / 3600,
            )
        if missing:
            logger.warning(
                f"CalendarSource: no sunrise or sunset at ({self.lat}, {self.lon}) on {len(missing)} date(s) "
                f"from {missing[0]} to {missing[-1]}; daylight columns are NaN there"
            )

        values = per_date[codes] if len(codes) else np.empty((0, 3))
        df[f"{self.prefix}sunrise"] = values[:, 0]
        df[f"{self.prefix}sunset"] = values[:, 1]
        df[f"{self.prefix}daylight_hours"] = values[:, 2]
        return df

    def _add_school_holidays(self, df: pd.DataFrame, local_dates: pd.DatetimeIndex, codes: np.ndarray) -> pd.DataFrame:
        """`is_school_holiday` (nationwide flag) or `school_holiday_share` (unweighted fraction of the country's
        top-level regions on school holiday) from the OpenHolidays API, depending on `school_holidays`."""
        national = self.school_holidays == "national"
        column = f"{self.prefix}is_school_holiday" if national else f"{self.prefix}school_holiday_share"
        if not len(local_dates):
            df[column] = np.array([], dtype=float)
            return df

        margin = pd.Timedelta(days=_openholidays.QUERY_MARGIN_DAYS)
        days = pd.date_range(local_dates.min() - margin, local_dates.max() + margin, freq="D")
        entries = _openholidays.fetch_entries(self.session, self.country, days[0], days[-1])
        logger.info(
            f"OpenHolidays [{self.country}]: {len(entries)} school holiday periods. {_openholidays.ATTRIBUTION}"
        )
        positions = days.get_indexer(local_dates)

        if national:
            per_day = _openholidays.daily_flags(entries, None, days)
        else:
            regions = _openholidays.top_level_subdivisions(self.session, self.country)
            if not regions:
                raise ValueError(
                    f"OpenHolidays lists no subdivisions for {self.country}; school_holidays='all' needs some"
                )
            # Unweighted average across top-level regions; a NaN region-day makes the whole day's share NaN (a
            # partial average would look like fewer regions on holiday than is actually known).
            per_region_day = np.column_stack([_openholidays.daily_flags(entries, region, days) for region in regions])
            per_day = per_region_day.mean(axis=1)

        per_date = per_day[positions]
        df[column] = per_date[codes]
        if np.isnan(per_date).any():
            self._warn_missing_school_holidays(column, local_dates[np.isnan(per_date)])
        return df

    def _warn_missing_school_holidays(self, column: str, missing: pd.DatetimeIndex) -> None:
        logger.warning(
            f"OpenHolidays [{self.country}]: no school holiday data for {len(missing)} local date(s) "
            f"from {missing.min().date()} to {missing.max().date()}; {column} is NaN there"
        )

    def get_cache_config(self) -> dict:
        """Every output-affecting parameter, the resolved market time zone, the installed `holidays` package
        version (its holiday data changes between releases, independently of this source's own code) and a
        `schema_version`. This also makes `school_holidays` output (downloaded from the OpenHolidays API)
        cached like any other source; see the Caching docs for the same revision caution that applies to
        ENTSO-E data."""
        config = {
            "source_type": "calendar",
            "schema_version": SCHEMA_VERSION,
            "holidays_version": holidays_lib.__version__,
            "country": self.country,
            "timezone": self.market_timezone,
            "holidays": self.holidays,
            "weekday": self.weekday,
            "hour": self.hour,
            "month": self.month,
            "daylight": self.daylight,
            "lat": self.lat,
            "lon": self.lon,
            "freq": self.freq,
            "region": self.region,
            "categories": self.categories,
            "bridge_days": self.bridge_days,
            "christmas_period": self.christmas_period,
            "holiday_distance": self.holiday_distance,
            "dst_switch": self.dst_switch,
            "regimes": self.regimes,
            "school_holidays": self.school_holidays,
        }
        if self.prefix:
            config["prefix"] = self.prefix
        return config
