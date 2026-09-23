import html
import re
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from epftoolbox2.logging import get_logger
from .base import DataSource

# Bump whenever the output for the same parameters changes, so that cached frames from older parsers are not served.
# 2: every price period kept on a uniform grid, SDAC sequence only, curveType A03 filled, A75 consumption split,
#    daily values on the local calendar, [start, end) range, PsrType names.
SCHEMA_VERSION = 2


class NoMatchingDataError(Exception):
    """Raised when ENTSOE API returns no matching data"""

    pass


class EntsoeApiError(RuntimeError):
    """The ENTSO-E API refused a request or kept failing. The message never contains the API key."""


AREAS = {
    "DE_50HZ": "10YDE-VE-------2",
    "AL": "10YAL-KESH-----5",
    "DE_AMPRION": "10YDE-RWENET---I",
    "AT": "10YAT-APG------L",
    "BY": "10Y1001A1001A51S",
    "BE": "10YBE----------2",
    "BA": "10YBA-JPCC-----D",
    "BG": "10YCA-BULGARIA-R",
    "CZ_DE_SK": "10YDOM-CZ-DE-SKK",
    "HR": "10YHR-HEP------M",
    "CWE": "10YDOM-REGION-1V",
    "CY": "10YCY-1001A0003J",
    "CZ": "10YCZ-CEPS-----N",
    "DE_AT_LU": "10Y1001A1001A63L",
    "DE_LU": "10Y1001A1001A82H",
    "DK": "10Y1001A1001A65H",
    "DK_1": "10YDK-1--------W",
    "DK_1_NO_1": "46Y000000000007M",
    "DK_2": "10YDK-2--------M",
    "DK_CA": "10Y1001A1001A796",
    "EE": "10Y1001A1001A39I",
    "FI": "10YFI-1--------U",
    "MK": "10YMK-MEPSO----8",
    "FR": "10YFR-RTE------C",
    "DE": "10Y1001A1001A83F",
    "GR": "10YGR-HTSO-----Y",
    "HU": "10YHU-MAVIR----U",
    "IS": "IS",
    "IE_SEM": "10Y1001A1001A59C",
    "IE": "10YIE-1001A00010",
    "IT": "10YIT-GRTN-----B",
    "IT_SACO_AC": "10Y1001A1001A885",
    "IT_CALA": "10Y1001C--00096J",
    "IT_SACO_DC": "10Y1001A1001A893",
    "IT_BRNN": "10Y1001A1001A699",
    "IT_CNOR": "10Y1001A1001A70O",
    "IT_CSUD": "10Y1001A1001A71M",
    "IT_FOGN": "10Y1001A1001A72K",
    "IT_GR": "10Y1001A1001A66F",
    "IT_MACRO_NORTH": "10Y1001A1001A84D",
    "IT_MACRO_SOUTH": "10Y1001A1001A85B",
    "IT_MALTA": "10Y1001A1001A877",
    "IT_NORD": "10Y1001A1001A73I",
    "IT_NORD_AT": "10Y1001A1001A80L",
    "IT_NORD_CH": "10Y1001A1001A68B",
    "IT_NORD_FR": "10Y1001A1001A81J",
    "IT_NORD_SI": "10Y1001A1001A67D",
    "IT_PRGP": "10Y1001A1001A76C",
    "IT_ROSN": "10Y1001A1001A77A",
    "IT_SARD": "10Y1001A1001A74G",
    "IT_SICI": "10Y1001A1001A75E",
    "IT_SUD": "10Y1001A1001A788",
    "RU_KGD": "10Y1001A1001A50U",
    "LV": "10YLV-1001A00074",
    "LT": "10YLT-1001A0008Q",
    "LU": "10YLU-CEGEDEL-NQ",
    "LU_BZN": "10Y1001A1001A82H",
    "MT": "10Y1001A1001A93C",
    "ME": "10YCS-CG-TSO---S",
    "GB": "10YGB----------A",
    "GE": "10Y1001A1001B012",
    "GB_IFA": "10Y1001C--00098F",
    "GB_IFA2": "17Y0000009369493",
    "GB_ELECLINK": "11Y0-0000-0265-K",
    "UK": "10Y1001A1001A92E",
    "NL": "10YNL----------L",
    "NO_1": "10YNO-1--------2",
    "NO_1A": "10Y1001A1001A64J",
    "NO_2": "10YNO-2--------T",
    "NO_2_NSL": "50Y0JVU59B4JWQCU",
    "NO_2A": "10Y1001C--001219",
    "NO_3": "10YNO-3--------J",
    "NO_4": "10YNO-4--------9",
    "NO_5": "10Y1001A1001A48H",
    "NO": "10YNO-0--------C",
    "PL_CZ": "10YDOM-1001A082L",
    "PL": "10YPL-AREA-----S",
    "PT": "10YPT-REN------W",
    "MD": "10Y1001A1001A990",
    "RO": "10YRO-TEL------P",
    "RU": "10Y1001A1001A49F",
    "SE_1": "10Y1001A1001A44P",
    "SE_2": "10Y1001A1001A45N",
    "SE_3": "10Y1001A1001A46L",
    "SE_4": "10Y1001A1001A47J",
    "RS": "10YCS-SERBIATSOV",
    "SK": "10YSK-SEPS-----K",
    "SI": "10YSI-ELES-----O",
    "GB_NIR": "10Y1001A1001A016",
    "ES": "10YES-REE------0",
    "SE": "10YSE-1--------K",
    "CH": "10YCH-SWISSGRIDZ",
    "DE_TENNET": "10YDE-EON------1",
    "DE_TRANSNET": "10YDE-ENBW-----N",
    "TR": "10YTR-TEIAS----W",
    "UA": "10Y1001C--00003F",
    "UA_DOBTPP": "10Y1001A1001A869",
    "UA_BEI": "10YUA-WEPS-----0",
    "UA_IPS": "10Y1001C--000182",
    "XK": "10Y1001C--00100H",
    "DE_AMP_LU": "10Y1001C--00002H",
}

# Areas that are not bidding zones. Day-ahead prices (A44) exist only per bidding zone, and the API answers
# "No matching data" for these codes.
NOT_BIDDING_ZONES = {
    "DE": "DE_LU (DE_AT_LU before 2018-10-01)",
    "DE_50HZ": "DE_LU",
    "DE_AMPRION": "DE_LU",
    "DE_TENNET": "DE_LU",
    "DE_TRANSNET": "DE_LU",
    "DE_AMP_LU": "DE_LU",
    "LU": "DE_LU",
    "DK": "DK_1 or DK_2",
    "DK_CA": "DK_1 or DK_2",
    "NO": "one of NO_1 ... NO_5",
    "SE": "one of SE_1 ... SE_4",
    "IT": "one of the Italian zones (IT_NORD, IT_CNOR, IT_CSUD, IT_SUD, IT_CALA, IT_SICI, IT_SARD)",
    "IT_MACRO_NORTH": "one of the Italian zones (IT_NORD, IT_CNOR, ...)",
    "IT_MACRO_SOUTH": "one of the Italian zones (IT_CSUD, IT_SUD, IT_CALA, IT_SICI, IT_SARD)",
    "IE": "IE_SEM",
    "GB_NIR": "IE_SEM",
    "UK": "GB",
    "CWE": "a single bidding zone such as FR, NL, BE, AT or DE_LU",
    "PL_CZ": "PL or CZ",
}

# Local time of each area, used to put daily (and longer) positions on local calendar days. ENTSO-E periods for
# these resolutions start at local midnight; a period that does not is stepped in UTC instead.
_DEFAULT_TIMEZONE = "Europe/Brussels"
AREA_TIMEZONES = {
    "GB": "Europe/London",
    "UK": "Europe/London",
    "GB_IFA": "Europe/London",
    "GB_IFA2": "Europe/London",
    "GB_ELECLINK": "Europe/London",
    "GB_NIR": "Europe/London",
    "IE": "Europe/Dublin",
    "IE_SEM": "Europe/Dublin",
    "PT": "Europe/Lisbon",
    "IS": "Atlantic/Reykjavik",
    "FI": "Europe/Helsinki",
    "EE": "Europe/Tallinn",
    "LV": "Europe/Riga",
    "LT": "Europe/Vilnius",
    "BG": "Europe/Sofia",
    "GR": "Europe/Athens",
    "RO": "Europe/Bucharest",
    "CY": "Asia/Nicosia",
    "MD": "Europe/Chisinau",
    "UA": "Europe/Kiev",
    "UA_DOBTPP": "Europe/Kiev",
    "UA_BEI": "Europe/Kiev",
    "UA_IPS": "Europe/Kiev",
    "TR": "Europe/Istanbul",
    "BY": "Europe/Minsk",
    "RU": "Europe/Moscow",
    "RU_KGD": "Europe/Kaliningrad",
    "GE": "Asia/Tbilisi",
}

# ENTSO-E PsrType codes. Column names are the snake_case of these names, e.g. B11 -> hydro_run_of_river_and_pondage.
PSRTYPE_MAPPINGS = {
    "A03": "Mixed",
    "A04": "Generation",
    "A05": "Load",
    "B01": "Biomass",
    "B02": "Fossil Brown coal/Lignite",
    "B03": "Fossil Coal-derived gas",
    "B04": "Fossil Gas",
    "B05": "Fossil Hard coal",
    "B06": "Fossil Oil",
    "B07": "Fossil Oil shale",
    "B08": "Fossil Peat",
    "B09": "Geothermal",
    "B10": "Hydro Pumped Storage",
    "B11": "Hydro Run-of-river and pondage",
    "B12": "Hydro Water Reservoir",
    "B13": "Marine",
    "B14": "Nuclear",
    "B15": "Other renewable",
    "B16": "Solar",
    "B17": "Waste",
    "B18": "Wind Offshore",
    "B19": "Wind Onshore",
    "B20": "Other",
    "B21": "AC Link",
    "B22": "DC Link",
    "B23": "Substation",
    "B24": "Transformer",
    "B25": "Energy storage",
}

# Columns holding one value per day (or longer), stamped at the start of that day. A value whose day began before
# the requested start but still covers it is reported at the start, so that clipping to [start, end) keeps it.
_PERIOD_COLUMNS = {
    "load_forecast_daily_min": "P1D",
    "load_forecast_daily_max": "P1D",
}

_RESOLUTION_RE = re.compile(r"^P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?$")


def lookup_area(code: str) -> tuple[str, str]:
    code_upper = code.upper()
    if code_upper in AREAS:
        return (code_upper, AREAS[code_upper])
    raise ValueError(f"Invalid country code: {code}")


def psr_column_name(psr_type: Optional[str]) -> str:
    """snake_case column stem for a PsrType code; unknown codes are kept as their lower-case code."""
    name = PSRTYPE_MAPPINGS.get(psr_type, psr_type or "unknown")
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def parse_resolution(text: str) -> Union[pd.Timedelta, pd.DateOffset]:
    """ISO 8601 duration of an ENTSO-E resolution: a Timedelta for PTnM/PTnH, a calendar DateOffset for PnD/PnM/PnY."""
    match = _RESOLUTION_RE.match(text or "")
    if not match or text in ("P", "PT") or not any(match.groups()):
        raise NotImplementedError(f"Unknown resolution format: {text}")
    years, months, days, hours, minutes = (int(g) if g else 0 for g in match.groups())
    if years or months or days:
        return pd.DateOffset(years=years, months=months, days=days, hours=hours, minutes=minutes)
    return pd.Timedelta(hours=hours, minutes=minutes)


def _child_text(element: Optional[ET.Element], tag: str) -> Optional[str]:
    if element is None:
        return None
    child = element.find(tag)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def _combine(parts: List[pd.Series]) -> pd.Series:
    """Concatenate series; where timestamps repeat (overlapping periods or chunks), the first one wins."""
    if not parts:
        return pd.Series(dtype=float)
    combined = pd.concat(parts).sort_index(kind="stable")
    return combined[~combined.index.duplicated(keep="first")]


def _retry_after_seconds(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, (parsedate_to_datetime(value) - pd.Timestamp.now(tz="UTC")).total_seconds())
    except (TypeError, ValueError):
        return None


def _acknowledgement_reason(text: str) -> Optional[str]:
    """The <Reason><text> of an ENTSO-E Acknowledgement document."""
    match = re.search(r"<Reason>.*?<text>(.*?)</text>", text or "", re.S)
    return html.unescape(match.group(1).strip()) if match else None


class EntsoeSource(DataSource):
    """
    ENTSOE data source for electricity market data

    Supports fetching:
    - Load: actual load, day-ahead forecast, week-ahead forecast (daily min/max)
    - Generation: actual generation and consumption by production type, wind/solar day-ahead forecasts
    - Price: day-ahead (SDAC) market prices

    ``fetch(start, end)`` returns the half-open range [start, end) with a UTC index. Prices come on a uniform grid:
    hourly (quarter-hours averaged into hours) with ``prefer_15min=False``, 15-minute (hourly periods step-filled)
    with ``prefer_15min=True``. Other series keep the resolution ENTSO-E publishes.

    Example:
        >>> source = EntsoeSource('PL', api_key='your-api-key', type=['load', 'generation', 'price'])
        >>> df = source.fetch(start=pd.Timestamp('2024-01-01', tz='Europe/Warsaw'),
        ...                   end=pd.Timestamp('2024-01-07', tz='Europe/Warsaw'))
        >>> # df contains all columns: load_actual, load_forecast, generation_*, price, etc.
    """

    API_URL = "https://web-api.tp.entsoe.eu/api"
    REQUEST_TIMEOUT = 60
    MAX_RETRIES = 5
    RETRY_BACKOFF_SECONDS = 2.0
    MAX_RETRY_WAIT_SECONDS = 120.0

    def __init__(
        self,
        country_code: str,
        api_key: str,
        type: List[str],
        prefer_15min: bool = False,
    ):
        self.country_code = country_code
        self.api_key = api_key
        self.type = type
        self.prefer_15min = prefer_15min
        self._area_name, self._area_code = lookup_area(country_code)
        self._timezone = AREA_TIMEZONES.get(self._area_name, _DEFAULT_TIMEZONE)
        self.session = requests.Session()

        self.console = Console()
        self.logger = get_logger(__name__)

        self._validate_config()

    def __del__(self):
        """Cleanup session on object destruction"""
        if hasattr(self, "session") and self.session:
            self.session.close()

    def _validate_config(self):
        if not self.api_key:
            raise ValueError("API key cannot be empty")
        if not self.type:
            raise ValueError("At least one data type must be specified")

        valid_types = {"load", "generation", "price"}
        for t in self.type:
            if t not in valid_types:
                raise ValueError(f"Invalid type '{t}'. Must be one of: {valid_types}")

        if "price" in self.type and self._area_name in NOT_BIDDING_ZONES:
            raise ValueError(
                f"'{self.country_code}' is not a bidding zone; day-ahead prices are published per bidding zone. "
                f"Use {NOT_BIDDING_ZONES[self._area_name]}."
            )

        return True

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
        end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")

        if end <= start:
            raise ValueError(f"End timestamp ({end}) must be after start timestamp ({start})")

        start_time = time.time()

        chunks = self._generate_chunks(start, end, months=3)

        self.logger.info(f"ENTSOE [{self._area_name}]: Start downloading {', '.join(self.type)} data")

        all_results = {dtype: [] for dtype in self.type}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]ENTSOE [{self._area_name}]: Downloading {', '.join(self.type)}...",
                total=len(chunks),
            )

            for chunk_start, chunk_end in chunks:
                date_range = f"{chunk_start.date()} to {chunk_end.date()}"
                progress.update(task, description=f"[cyan]ENTSOE [{self._area_name}]: {date_range}")

                chunk_data = self._fetch_chunk(chunk_start, chunk_end)

                for dtype in self.type:
                    chunk_df = chunk_data.get(dtype)
                    if chunk_df is not None and not chunk_df.empty:
                        all_results[dtype].append(chunk_df)
                progress.advance(task)

        dataframes = []
        for dtype in self.type:
            if not all_results[dtype]:
                continue
            # Chunks overlap where the API returns whole days or weeks; the overlapping values are identical.
            df = pd.concat(all_results[dtype]).sort_index(kind="stable")
            if df.index.has_duplicates:
                df = df.groupby(level=0).first()
            df = self._clip(df, start, end)
            if df.empty:
                continue
            self._warn_on_late_start(dtype, df, start)
            dataframes.append(df)

        elapsed = time.time() - start_time
        if not dataframes:
            self.logger.warning(
                f"ENTSOE [{self._area_name}]: no {', '.join(self.type)} data between {start} and {end} "
                f"({elapsed:.2f} sec)"
            )
            return pd.DataFrame()

        self._log_success(elapsed)
        return pd.concat(dataframes, axis=1)

    def _generate_chunks(self, start: pd.Timestamp, end: pd.Timestamp, months: int = 3) -> list:
        if months <= 0:
            raise ValueError(f"months parameter must be positive, got {months}")

        chunks = []
        current = start

        while current < end:
            next_chunk = current + pd.DateOffset(months=months)
            chunk_end = min(next_chunk, end)
            chunks.append((current, chunk_end))
            current = next_chunk

        return chunks

    def _fetch_chunk(self, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        result = {}

        if "load" in self.type:
            result["load"] = self._fetch_load(start, end)

        if "generation" in self.type:
            result["generation"] = self._fetch_generation(start, end)

        if "price" in self.type:
            result["price"] = self._fetch_price(start, end)

        return result

    def _clip(self, df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Keep [start, end). A daily (or longer) value whose period covers ``start`` is moved to ``start``."""
        carried = {}
        for column, resolution in _PERIOD_COLUMNS.items():
            if column not in df.columns:
                continue
            values = df[column].dropna()
            if values.empty or start in values.index:
                continue
            before = values[values.index < start]
            if before.empty:
                continue
            period_start = before.index[-1]
            period_end = (period_start.tz_convert(self._timezone) + parse_resolution(resolution)).tz_convert("UTC")
            if period_end > start:
                carried[column] = before.iloc[-1]

        df = df[(df.index >= start) & (df.index < end)].copy()
        if carried:
            if start not in df.index:
                df = df.reindex(df.index.union([start]))
            for column, value in carried.items():
                df.loc[start, column] = value
        return df

    def _warn_on_late_start(self, dtype: str, df: pd.DataFrame, start: pd.Timestamp):
        first = df.index.min()
        if first - start > pd.Timedelta(days=1):
            self.logger.warning(
                f"ENTSOE [{self._area_name}]: {dtype} data starts at {first}, {first - start} after the requested "
                f"start {start}; earlier periods are not published for this area"
                + (" (DE_LU starts on 2018-10-01; use DE_AT_LU before)" if self._area_name == "DE_LU" else "")
            )

    def _fetch_load(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        dfs = []

        xml = self._api_request(
            {
                "documentType": "A65",
                "processType": "A16",
                "outBiddingZone_Domain": self._area_code,
                "out_Domain": self._area_code,
            },
            start,
            end,
        )
        df_actual = self._parse_loads(xml, "A16")
        if not df_actual.empty:
            dfs.append(df_actual)

        xml = self._api_request(
            {
                "documentType": "A65",
                "processType": "A01",
                "outBiddingZone_Domain": self._area_code,
            },
            start,
            end,
        )
        df_forecast = self._parse_loads(xml, "A01")
        if not df_forecast.empty:
            dfs.append(df_forecast)

        xml = self._api_request(
            {
                "documentType": "A65",
                "processType": "A31",
                "outBiddingZone_Domain": self._area_code,
            },
            start,
            end,
        )
        df_week = self._parse_loads(xml, "A31")
        if not df_week.empty:
            dfs.append(df_week)
        elif xml is not None:
            self.logger.warning(
                f"ENTSOE [{self._area_name}]: A31 (week-ahead forecast) data available but no min/max values found"
            )

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, axis=1)

    def _fetch_generation(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        dfs = []

        xml = self._api_request(
            {
                "documentType": "A75",
                "processType": "A16",
                "in_Domain": self._area_code,
            },
            start,
            end,
        )
        dfs.append(self._parse_generation(xml).add_prefix("generation_"))

        xml = self._api_request(
            {
                "documentType": "A69",
                "processType": "A01",
                "in_Domain": self._area_code,
            },
            start,
            end,
        )
        dfs.append(self._parse_generation(xml).add_prefix("generation_").add_suffix("_forecast"))

        dfs = [df for df in dfs if not df.empty]
        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

    def _fetch_price(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        xml = self._api_request(
            {
                "documentType": "A44",
                "in_Domain": self._area_code,
                "out_Domain": self._area_code,
            },
            start,
            end,
        )
        series = self._prices_to_grid(self._parse_prices(xml))
        if series.empty:
            return pd.DataFrame()
        return pd.DataFrame({"price": series})

    def _api_request(self, params: Dict, start: pd.Timestamp, end: pd.Timestamp) -> Optional[str]:
        """GET one document. Returns None for "No matching data"; retries 429/5xx and network errors; raises
        EntsoeApiError otherwise. The key travels in the SECURITY_TOKEN header, never in the URL."""
        query = dict(params, periodStart=self._format_time(start), periodEnd=self._format_time(end))
        label = "/".join(v for v in (params.get("documentType"), params.get("processType")) if v)
        label = f"{label} {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC"
        error = ""

        for attempt in range(self.MAX_RETRIES + 1):
            wait = None
            try:
                response = self.session.get(
                    url=self.API_URL,
                    params=query,
                    headers={"SECURITY_TOKEN": self.api_key},
                    timeout=self.REQUEST_TIMEOUT,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                error = f"{type(exc).__name__}: {exc}"
            except requests.RequestException as exc:
                raise EntsoeApiError(self._redact(f"ENTSOE [{self._area_name}]: {label} failed: {exc}")) from None
            else:
                status = response.status_code
                text = response.text
                if "No matching data found" in text:
                    self.logger.warning(f"ENTSOE [{self._area_name}]: no matching data for {label}")
                    return None
                if status < 400 and "Acknowledgement_MarketDocument" not in text[:1000]:
                    return text
                reason = _acknowledgement_reason(text) or response.reason or text[:200]
                error = f"HTTP {status}: {reason}"
                if status < 400 or (status != 429 and status < 500):
                    raise EntsoeApiError(self._redact(f"ENTSOE [{self._area_name}]: {label} failed with {error}"))
                wait = _retry_after_seconds(response.headers.get("Retry-After"))

            if attempt == self.MAX_RETRIES:
                break
            if wait is None:
                wait = self.RETRY_BACKOFF_SECONDS * 2**attempt
            wait = min(wait, self.MAX_RETRY_WAIT_SECONDS)
            self.logger.warning(
                self._redact(
                    f"ENTSOE [{self._area_name}]: {label}: {error}; retrying in {wait:.0f} s "
                    f"({attempt + 1}/{self.MAX_RETRIES})"
                )
            )
            time.sleep(wait)

        raise EntsoeApiError(
            self._redact(f"ENTSOE [{self._area_name}]: {label} failed after {self.MAX_RETRIES + 1} attempts: {error}")
        )

    def _redact(self, text: str) -> str:
        return text.replace(self.api_key, "***") if self.api_key else text

    @staticmethod
    def _format_time(ts: pd.Timestamp) -> str:
        return ts.tz_convert("UTC").strftime("%Y%m%d%H%M")

    def _parse_loads(self, xml_text: Optional[str], process_type: str) -> pd.DataFrame:
        parts = self._parse_timeseries(xml_text, "quantity")
        if not parts:
            return pd.DataFrame()
        if process_type in ["A01", "A16"]:
            col_name = "load_forecast" if process_type == "A01" else "load_actual"
            return pd.DataFrame({col_name: _combine([s for _, s in parts])})

        columns = {"A60": "load_forecast_daily_min", "A61": "load_forecast_daily_max"}
        result = {}
        for business_type, column in columns.items():
            selected = [s for meta, s in parts if meta["business_type"] == business_type]
            if selected:
                result[column] = _combine(selected)
        return pd.DataFrame(result) if result else pd.DataFrame()

    def _parse_generation(self, xml_text: Optional[str]) -> pd.DataFrame:
        """One column per production type; series reported in the out-bidding-zone domain (pumping and other
        consumption) get a ``_consumption`` suffix instead of being mixed into generation."""
        columns: Dict[str, List[pd.Series]] = {}
        for meta, series in self._parse_timeseries(xml_text, "quantity"):
            name = psr_column_name(meta["psr_type"])
            if meta["out_bidding_zone"] and not meta["in_bidding_zone"]:
                name += "_consumption"
            columns.setdefault(name, []).append(series)
        if not columns:
            return pd.DataFrame()
        return pd.DataFrame({name: _combine(parts) for name, parts in columns.items()}).sort_index()

    def _parse_prices(self, xml_text: Optional[str]) -> Dict[pd.Timedelta, pd.Series]:
        """Day-ahead prices by native resolution. Only the SDAC series is kept: classificationSequence position 1,
        or no classificationSequence. DE_LU and AT also publish a separate 15-minute auction as position 2."""
        by_resolution: Dict[pd.Timedelta, List[pd.Series]] = {}
        skipped = set()
        for meta, series in self._parse_timeseries(xml_text, "price.amount"):
            if meta["sequence"] not in (None, "1"):
                skipped.add(meta["sequence"])
                continue
            step = parse_resolution(meta["resolution"])
            if not isinstance(step, pd.Timedelta):
                continue
            by_resolution.setdefault(step, []).append(series)
        if skipped and not by_resolution:
            self.logger.warning(
                f"ENTSOE [{self._area_name}]: only price series with classificationSequence {sorted(skipped)} were "
                "published; they are not the day-ahead coupling price and were ignored"
            )
        return {step: _combine(parts) for step, parts in by_resolution.items()}

    def _prices_to_grid(self, by_resolution: Dict[pd.Timedelta, pd.Series]) -> pd.Series:
        """Put every period on one grid: 1 h (sub-hourly periods averaged) or 15 min (longer periods step-filled).
        Where a timestamp is published at several resolutions, the one closest to the grid wins."""
        target = pd.Timedelta("15min") if self.prefer_15min else pd.Timedelta("1h")
        result = pd.Series(dtype=float)
        for step in sorted(by_resolution, key=lambda s: abs(s - target)):
            series = by_resolution[step]
            if step < target:
                groups = series.groupby(series.index.floor(target))
                means = groups.mean()
                # an hour with a missing quarter-hour would be a biased mean; leave it missing instead
                series = means[groups.count() == int(target / step)]
            elif step > target:
                k = int(step / target)
                offsets = np.tile(pd.TimedeltaIndex([target * i for i in range(k)]).to_numpy(), len(series))
                series = pd.Series(np.repeat(series.to_numpy(), k), index=series.index.repeat(k) + offsets)
            result = series if result.empty else result.combine_first(series)
        return result.sort_index()

    def _parse_timeseries(self, xml_text: Optional[str], value_tag: str) -> List[Tuple[dict, pd.Series]]:
        """Decode every TimeSeries/Period into (metadata, values indexed by UTC period start).

        Positions are mapped from the Period's ``timeInterval`` and ``resolution``. Under curveType A03 ENTSO-E omits
        a point that repeats the previous value, so omitted positions (including trailing ones up to the end of the
        Period) are forward-filled; under A01 they stay missing.
        """
        if not xml_text:
            return []
        root = self._parse_xml(xml_text)
        out = []
        for ts in root.iter("TimeSeries"):
            meta = {
                "business_type": _child_text(ts, "businessType"),
                "curve_type": _child_text(ts, "curveType") or "A01",
                "psr_type": _child_text(ts.find("MktPSRType"), "psrType"),
                "in_bidding_zone": _child_text(ts, "inBiddingZone_Domain.mRID"),
                "out_bidding_zone": _child_text(ts, "outBiddingZone_Domain.mRID"),
                "in_domain": _child_text(ts, "in_Domain.mRID"),
                "out_domain": _child_text(ts, "out_Domain.mRID"),
                "sequence": _child_text(ts, "classificationSequence_AttributeInstanceComponent.position"),
            }
            for period in ts.iter("Period"):
                interval = period.find("timeInterval")
                start_text = _child_text(interval, "start")
                end_text = _child_text(interval, "end")
                resolution = _child_text(period, "resolution")
                if not start_text or not end_text or not resolution:
                    continue
                index = self._period_index(pd.Timestamp(start_text), pd.Timestamp(end_text), resolution)
                values = np.full(len(index), np.nan)
                for point in period.iter("Point"):
                    position = _child_text(point, "position")
                    value = _child_text(point, value_tag)
                    if position is None or value is None:
                        continue
                    position = int(position)
                    if 1 <= position <= len(index):
                        values[position - 1] = float(value.replace(",", ""))
                series = pd.Series(values, index=index)
                if meta["curve_type"] == "A03":
                    series = series.ffill()
                out.append((dict(meta, resolution=resolution), series.dropna()))
        return out

    def _period_index(self, start: pd.Timestamp, end: pd.Timestamp, resolution: str) -> pd.DatetimeIndex:
        """UTC start of every position of a Period. Daily and longer resolutions follow the local calendar of the
        area (a week with the autumn DST switch lasts 169 hours but still has 7 daily positions)."""
        start = start.tz_convert("UTC").as_unit("ns")
        end = end.tz_convert("UTC").as_unit("ns")
        step = parse_resolution(resolution)
        if isinstance(step, pd.Timedelta):
            return pd.date_range(start, periods=int((end - start) / step), freq=step)

        local_start = start.tz_convert(self._timezone)
        if local_start == local_start.normalize():
            stamps = []
            current, local_end = local_start, end.tz_convert(self._timezone)
            while current < local_end:
                stamps.append(current)
                current = current + step
            return pd.DatetimeIndex(stamps).tz_convert("UTC")

        # Not aligned with local midnight of the assumed timezone: step in UTC, rounding the number of positions
        nominal = (start + step) - start
        count = max(1, round((end - start) / nominal))
        return pd.DatetimeIndex([start + step * i for i in range(count)])

    def _parse_xml(self, xml_text: str) -> ET.Element:
        if "<!DOCTYPE" in xml_text or "<!ENTITY" in xml_text:
            raise EntsoeApiError(f"ENTSOE [{self._area_name}]: refusing to parse a response that declares a DTD")
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise EntsoeApiError(f"ENTSOE [{self._area_name}]: response is not valid XML ({exc})") from None
        for element in root.iter():
            element.tag = element.tag.rpartition("}")[2]
        return root

    def _log_success(self, elapsed: float):
        self.logger.info(f"ENTSOE [{self._area_name}]: Download completed successfully in {elapsed:.2f} sec")

    def get_cache_config(self) -> dict:
        return {
            "source_type": "entsoe",
            "area_code": self._area_code,
            "types": sorted(self.type),
            "prefer_15min": self.prefer_15min,
            "schema_version": SCHEMA_VERSION,
        }
