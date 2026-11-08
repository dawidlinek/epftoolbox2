import time
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional

import pandas as pd
import requests

from epftoolbox2.logging import get_logger
from .base import DataSource

MARKET_ZONE = "Europe/Warsaw"
SCHEMA_VERSION = 1

# The 15-minute tables stamp a row with the END of its period (`dtime` local, `dtime_utc` UTC; `period` and `period_utc`
# give the start and the end). The index of the frame is the period START, like every other source. `pdgsz` is hourly
# and stamps the hour's start.
QUARTER = pd.Timedelta(minutes=15)

# Output columns of each dataset (the API's own field names, except where a table has a long format). `his-wlk-cal`
# has more fields than it is worth to load by default; `DEFAULT_COLUMNS` are the ones loaded when `columns` is not set.
FLOW_SECTIONS = (
    "CZ-PL",
    "DE-PL",
    "LT-PL",
    "PL-CZ",
    "PL-DE",
    "PL-LT",
    "PL-SE",
    "PL-SK",
    "PL-UA",
    "SE-PL",
    "SK-PL",
    "UA-PL",
)
FLOW_COLUMNS = tuple(f"flow_{section.lower().replace('-', '_')}" for section in FLOW_SECTIONS)
_FLOW_COLUMN_OF_SECTION = dict(zip(FLOW_SECTIONS, FLOW_COLUMNS))

DATASET_COLUMNS: Dict[str, tuple] = {
    "kse-load": ("load_actual", "load_fcst"),
    "his-wlk-cal": (
        "demand",
        "wi",
        "pv",
        "jg",
        "jga",
        "jgm",
        "jgo",
        "jgm1",
        "jgm2",
        "jgw1",
        "jgw2",
        "jgz1",
        "jgz2",
        "jgz3",
        "jnwrb",
        "swm_p",
        "swm_np",
    ),
    "csdac-pln": ("csdac_pln",),
    "rce-pln": ("rce_pln",),
    "pdgsz": ("usage_fcst",),
    "przeplywy-mocy": FLOW_COLUMNS,
}
DEFAULT_COLUMNS: Dict[str, tuple] = {**DATASET_COLUMNS, "his-wlk-cal": ("demand", "wi", "pv", "jg")}
HOURLY_DATASETS = ("pdgsz",)
DEFAULT_DATASETS = ("kse-load", "csdac-pln")


class PseApiError(RuntimeError):
    """The PSE reports API refused a request or answered with something that is not a table."""


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


class PseSource(DataSource):
    """
    Polish TSO (PSE) reports API, ``https://api.raporty.pse.pl``: load, day-ahead price, national demand and wind and PV
    generation, cross-border flows and the recommended-usage periods. No API key. Source: PSE. The API's terms of use
    are not published; do not redistribute the data in bulk.

    The index is the UTC **start** of each period, on the 15-minute grid of the tables (a 23-hour day has 92 rows, a
    25-hour day 100). PSE stamps a 15-minute row with the end of its period; the source reads ``dtime_utc`` and
    shifts it back by 15 minutes. ``pdgsz`` is hourly and is repeated over the four quarter-hours of each hour.

    Datasets (``datasets``) and their columns:

    - ``kse-load``: ``load_actual`` and ``load_fcst`` (MW), the national load and PSE's day-ahead forecast of it.
    - ``his-wlk-cal``: ``demand``, ``wi`` (wind), ``pv`` (solar) and ``jg`` (PSE's generation field) by
      default, and with ``columns`` the other fields of the table under PSE's names (``jga``, ``jgm``, ``jgo``,
      ``jgm1``, ``jgm2``, ``jgw1``, ``jgw2``, ``jgz1``, ``jgz2``, ``jgz3``, ``jnwrb``, ``swm_p``, ``swm_np``).
    - ``csdac-pln``: ``csdac_pln``, the SDAC price of the Polish bidding zone in PLN/MWh (the EUR price times the
      day's EUR/PLN rate).
    - ``rce-pln``: ``rce_pln``, the market energy price (RCE) of the prosumer net-billing scheme in PLN/MWh. It is
      another series than the SDAC price (correlation 0.98, mean absolute difference 43 PLN/MWh in a comparison): do not use it
      as the day-ahead price.
    - ``pdgsz``: ``usage_fcst``, PSE's recommended-usage code of each hour. The table keeps every version of an hour;
      the source takes the **first** one, by its UTC publication stamp, because a later version may be written on the
      delivery day.
    - ``przeplywy-mocy``: one column per cross-border section, ``flow_de_pl``, ``flow_pl_de``, ... (see
      ``FLOW_COLUMNS``).

    PSE's ``publication_ts`` is a stamp of the last write of a row, not of its first publication (rows reloaded in
    bulk carry the reload time), so the source does not use it, except to order the versions of ``pdgsz``. What was
    known when is in the built-in rules of ``InformationSet`` (``source_rules``), see the documentation page.

    History starts in June 2024 for load, SDAC, RCE and generation, a month earlier or later for other tables; an
    earlier range returns no rows (a warning is logged), and rows PSE has not published yet are NaN.

    Example:
        >>> source = PseSource(datasets=["kse-load", "csdac-pln"], prefix="pse")
        >>> df = source.fetch(pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-01-08", tz="UTC"))
        >>> # columns: pse_load_actual, pse_load_fcst, pse_csdac_pln
    """

    API_URL = "https://api.raporty.pse.pl/api"
    USER_AGENT = "epftoolbox2"
    PAGE_SIZE = 10000
    CHUNK_DAYS = 31
    REQUEST_TIMEOUT = 60
    MAX_RETRIES = 5
    RETRY_BACKOFF_SECONDS = 2.0
    MAX_RETRY_WAIT_SECONDS = 120.0

    def __init__(
        self,
        datasets: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        prefix: str = "",
    ):
        """
        Args:
            datasets: Datasets to load, from ``DATASET_COLUMNS`` (default: ``kse-load`` and ``csdac-pln``).
            columns: Columns to keep (default: every column of the datasets, for ``his-wlk-cal`` ``demand``,
                ``wi``, ``pv`` and ``jg``). A column of a dataset that is not in ``datasets`` raises.
            prefix: Prefix for the column names, ``{prefix}_{column}`` (default: none).
        """
        self.datasets = list(datasets) if datasets is not None else list(DEFAULT_DATASETS)
        self.columns = list(columns) if columns is not None else None
        self.prefix = prefix

        self.session = requests.Session()
        self.logger = get_logger(__name__)

        self._validate_config()

    def __del__(self):
        if hasattr(self, "session") and self.session:
            self.session.close()

    # ------------------------------------------------------------------ configuration

    def _validate_config(self) -> bool:
        if not self.datasets:
            raise ValueError("PseSource needs at least one dataset")
        unknown = [d for d in self.datasets if d not in DATASET_COLUMNS]
        if unknown:
            raise ValueError(f"Unknown PSE dataset(s) {unknown}. Available: {sorted(DATASET_COLUMNS)}")
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError(f"Duplicate datasets in {self.datasets}")
        if self.columns is not None:
            offered = {c for d in self.datasets for c in DATASET_COLUMNS[d]}
            unknown = [c for c in self.columns if c not in offered]
            if unknown:
                raise ValueError(
                    f"Column(s) {unknown} are not in the datasets {self.datasets}. Available: {sorted(offered)}"
                )
            if len(set(self.columns)) != len(self.columns):
                raise ValueError(f"Duplicate columns in {self.columns}")
            for dataset in self.datasets:
                if not self._columns_of(dataset):
                    raise ValueError(f"No column of the dataset {dataset!r} is selected in columns={self.columns}")
        return True

    def _columns_of(self, dataset: str) -> List[str]:
        """The output columns (unprefixed) of one dataset, in the dataset's order."""
        if self.columns is None:
            return list(DEFAULT_COLUMNS[dataset])
        return [c for c in DATASET_COLUMNS[dataset] if c in self.columns]

    def output_columns(self) -> List[str]:
        """The names of the columns ``fetch`` returns, prefix included."""
        return [self._named(c) for dataset in self.datasets for c in self._columns_of(dataset)]

    def _named(self, column: str) -> str:
        return f"{self.prefix}_{column}" if self.prefix else column

    def get_cache_config(self) -> dict:
        return {
            "source_type": "pse",
            "datasets": list(self.datasets),
            "columns": [c for d in self.datasets for c in self._columns_of(d)],
            "prefix": self.prefix,
            "schema_version": SCHEMA_VERSION,
        }

    # ------------------------------------------------------------------ fetching

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        if end <= start:
            return self._empty()

        frames = [self._fetch_dataset(dataset, start, end) for dataset in self.datasets]
        df = pd.concat(frames, axis=1).sort_index()
        df = df.loc[(df.index >= start) & (df.index < end)]
        df.index.name = None
        return df[self.output_columns()]

    def _empty(self) -> pd.DataFrame:
        return pd.DataFrame(columns=self.output_columns(), index=pd.DatetimeIndex([], tz="UTC"), dtype=float)

    def _fetch_dataset(self, dataset: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """One dataset on [start, end): the local business days that hold those instants, in chunks of CHUNK_DAYS."""
        first = start.tz_convert(MARKET_ZONE).normalize().tz_localize(None)
        last = (end - pd.Timedelta(1, "ns")).tz_convert(MARKET_ZONE).normalize().tz_localize(None)
        records: List[dict] = []
        day = first
        while day <= last:
            chunk_end = min(day + pd.Timedelta(days=self.CHUNK_DAYS - 1), last)
            records += self._request_table(dataset, day, chunk_end)
            day = chunk_end + pd.Timedelta(days=1)
        if not records:
            self.logger.warning(f"PSE [{dataset}]: no rows for {start:%Y-%m-%d} to {end:%Y-%m-%d} UTC")
        return self._parse(dataset, records)

    @staticmethod
    def _filter(first: pd.Timestamp, last: pd.Timestamp) -> str:
        """``$filter`` on the business date, which is the local date of a period's start."""
        return f"business_date ge '{first:%Y-%m-%d}' and business_date le '{last:%Y-%m-%d}'"

    def _request_table(self, dataset: str, first: pd.Timestamp, last: pd.Timestamp) -> List[dict]:
        """All rows of the business days first..last, following ``nextLink`` from page to page."""
        url: Optional[str] = f"{self.API_URL}/{dataset}"
        params: Optional[dict] = {"$filter": self._filter(first, last), "$first": self.PAGE_SIZE}
        label = f"{dataset} {first:%Y-%m-%d} to {last:%Y-%m-%d}"
        rows: List[dict] = []
        seen = set()
        while url is not None:
            body = self._get_json(url, params, label)
            values = body.get("value")
            if not isinstance(values, list):
                raise PseApiError(f"PSE [{label}]: the answer has no 'value' list: {str(body)[:200]}")
            rows += values
            url, params = body.get("nextLink") or None, None
            if url is not None:
                if not url.startswith(self.API_URL + "/") or url in seen:
                    raise PseApiError(f"PSE [{label}]: unexpected nextLink {url[:120]!r}")
                seen.add(url)
        return rows

    def _get_json(self, url: str, params: Optional[dict], label: str) -> dict:
        """GET one page. Retries 429/5xx and network errors; anything else raises PseApiError."""
        error = ""
        for attempt in range(self.MAX_RETRIES + 1):
            wait = None
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers={"User-Agent": self.USER_AGENT, "Accept": "application/json"},
                    timeout=self.REQUEST_TIMEOUT,
                )
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            except requests.RequestException as exc:
                raise PseApiError(f"PSE [{label}]: request failed: {exc}") from None
            else:
                status = response.status_code
                if status < 400:
                    try:
                        body = response.json()
                    except ValueError:
                        raise PseApiError(f"PSE [{label}]: the answer is not JSON: {response.text[:200]!r}") from None
                    if not isinstance(body, dict):
                        raise PseApiError(f"PSE [{label}]: unexpected answer {str(body)[:200]}")
                    return body
                error = f"HTTP {status}: {self._reason(response)}"
                if status != 429 and status < 500:
                    raise PseApiError(f"PSE [{label}] failed with {error}")
                wait = _retry_after_seconds(response.headers.get("Retry-After"))

            if attempt == self.MAX_RETRIES:
                break
            if wait is None:
                wait = self.RETRY_BACKOFF_SECONDS * 2**attempt
            wait = min(wait, self.MAX_RETRY_WAIT_SECONDS)
            self.logger.warning(f"PSE [{label}]: {error}; retrying in {wait:.0f} s ({attempt + 1}/{self.MAX_RETRIES})")
            time.sleep(wait)
        raise PseApiError(f"PSE [{label}] failed after {self.MAX_RETRIES + 1} attempts: {error}")

    @staticmethod
    def _reason(response) -> str:
        try:
            return str(response.json()["error"]["message"])
        except (ValueError, KeyError, TypeError):
            return (getattr(response, "reason", "") or getattr(response, "text", "") or "")[:200]

    # ------------------------------------------------------------------ parsing

    def _parse(self, dataset: str, records: List[dict]) -> pd.DataFrame:
        columns = self._columns_of(dataset)
        names = [self._named(c) for c in columns]
        if not records:
            return pd.DataFrame(columns=names, index=pd.DatetimeIndex([], tz="UTC"), dtype=float)
        df = pd.DataFrame.from_records(records)
        if "dtime_utc" not in df.columns:
            raise PseApiError(f"PSE [{dataset}]: rows without dtime_utc: {sorted(df.columns)}")
        instants = pd.to_datetime(df["dtime_utc"], utc=True, format="mixed")

        if dataset == "przeplywy-mocy":
            return self._parse_flows(df, instants, columns)
        if dataset in HOURLY_DATASETS:
            return self._parse_hourly(dataset, df, instants, columns)

        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise PseApiError(f"PSE [{dataset}]: fields {missing} missing from the answer: {sorted(df.columns)}")
        out = df[columns].apply(pd.to_numeric, errors="coerce")
        out.index = instants - QUARTER
        return self._finish(dataset, out, names)

    def _parse_flows(self, df: pd.DataFrame, instants: pd.Series, columns: List[str]) -> pd.DataFrame:
        for field in ("section_code", "value"):
            if field not in df.columns:
                raise PseApiError(f"PSE [przeplywy-mocy]: field {field!r} missing: {sorted(df.columns)}")
        df = df.assign(start=instants - QUARTER, column=df["section_code"].map(_FLOW_COLUMN_OF_SECTION))
        unknown = sorted(df.loc[df["column"].isna(), "section_code"].unique())
        if unknown:
            self.logger.warning(f"PSE [przeplywy-mocy]: sections {unknown} are not in FLOW_SECTIONS and are left out")
        df = df.dropna(subset=["column"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        if df.duplicated(["start", "column"]).any():
            raise PseApiError("PSE [przeplywy-mocy]: two rows for one section and period")
        wide = df.pivot(index="start", columns="column", values="value")
        wide = wide.reindex(columns=columns)
        return self._finish("przeplywy-mocy", wide, [self._named(c) for c in columns])

    def _parse_hourly(self, dataset: str, df: pd.DataFrame, instants: pd.Series, columns: List[str]) -> pd.DataFrame:
        """pdgsz: one row per hour and version. Keep each hour's first version (by its UTC publication stamp) and
        repeat the hour over its four quarter-hours."""
        for field in ("publication_ts_utc", *columns):
            if field not in df.columns:
                raise PseApiError(f"PSE [{dataset}]: field {field!r} missing: {sorted(df.columns)}")
        df = df.assign(hour=instants, stamp=pd.to_datetime(df["publication_ts_utc"], utc=True, format="mixed"))
        first = df.sort_values(["hour", "stamp"], kind="stable").drop_duplicates("hour", keep="first")
        hourly = first.set_index("hour")[columns].apply(pd.to_numeric, errors="coerce")
        quarters = [hourly.set_axis(hourly.index + k * QUARTER) for k in range(4)]
        return self._finish(dataset, pd.concat(quarters), [self._named(c) for c in columns])

    @staticmethod
    def _finish(dataset: str, out: pd.DataFrame, names: List[str]) -> pd.DataFrame:
        out = out.sort_index()
        if out.index.has_duplicates:
            raise PseApiError(f"PSE [{dataset}]: two rows for one period")
        out.columns = names
        out.index.name = None
        return out.astype(float)


__all__ = ["PseSource", "PseApiError", "DATASET_COLUMNS", "FLOW_COLUMNS", "FLOW_SECTIONS", "MARKET_ZONE"]
