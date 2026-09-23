import datetime
import re
import time
from typing import Dict, List, Optional, Union
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from epftoolbox2.logging import get_logger
from .base import DataSource

# name -> (API entity, field, publication kind)
# "day_ahead": published the day before delivery (market results).
# "realised": measured values, published after delivery (next day in the current regime).
SERIES = {
    "rce": ("rce-pln", "rce_pln", "day_ahead"),
    "csdac": ("csdac-pln", "csdac_pln", "day_ahead"),
    "demand": ("his-wlk-cal", "demand", "realised"),
    "wind": ("his-wlk-cal", "wi", "realised"),
    "solar": ("his-wlk-cal", "pv", "realised"),
    "generation_scheduled": ("his-wlk-cal", "jg", "realised"),
    "generation_non_balancing": ("his-wlk-cal", "jnwrb", "realised"),
    "exchange_parallel": ("his-wlk-cal", "swm_p", "realised"),
    "exchange_non_parallel": ("his-wlk-cal", "swm_np", "realised"),
}

# Nominal availability used by as_of="schedule": (days relative to the business day, local time). An assumption,
# checked only where first publications survive (September 2026): market results appeared on B-1 around 13:50
# (latest 17:05); realised values on B+1 around 03:00 in Feb-Apr and Jul-Sep 2026 (a few days up to 19:00).
# All realised records before 2026-01-18 were re-published then, so their first publication times are unknown.
SCHEDULE = {
    "day_ahead": (-1, "18:00"),
    "realised": (1, "06:00"),
}

# PSE business days and publication times are Polish local time.
PSE_TZ = "Europe/Warsaw"
# First business day in API v2 (checked 2026-09 for rce-pln, csdac-pln and his-wlk-cal).
HISTORY_START = pd.Timestamp("2024-06-14")


class PseSource(DataSource):
    """
    PSE (Polish TSO) data source using the public reporting API v2 (https://api.raporty.pse.pl).

    Every record carries a `publication_ts`, so values can be filtered "as of" an issue time: with
    `issue_time` set, the value placed on a row of local day D is kept only if it was published at or before
    `issue_time` on D. The API keeps only the last version of each record, so `publication_ts` is the time of
    the last (re-)publication; the default `as_of="publication"` rule therefore never uses a value before it
    existed in its current form, but drops values that were re-published after the issue time.

    Output columns (prefix "pse"): raw mode (`issue_time=None`) gives `pse_{series}` at delivery time.
    As-of mode gives one column per lag k, relative to the row's local day D and the same local wall-clock time:
    `pse_{series}` (k=0, value for D), `pse_{series}_d-{k}` (value for D-k) and `pse_{series}_d+{h}`
    (k=-h, value for D+h), each as known at the issue time on D.

    Example:
        >>> source = PseSource(series=["rce", "demand"], issue_time="11:30", lags=[0, 1, 7])
        >>> df = source.fetch(pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-02-01", tz="UTC"))
        >>> # columns: pse_rce, pse_rce_d-1, pse_rce_d-7, pse_demand (empty), pse_demand_d-1, pse_demand_d-7
    """

    API_URL = "https://api.raporty.pse.pl/api"
    PAGE_SIZE = 20000
    VALID_AS_OF = ("publication", "schedule")
    VALID_FREQ = ("15min", "1h")

    def __init__(
        self,
        series: List[str],
        issue_time: Optional[str] = "11:30",
        lags: Union[int, List[int]] = 0,
        as_of: str = "publication",
        timezone: str = "Europe/Warsaw",
        freq: str = "1h",
        prefix: str = "pse",
    ):
        """
        Args:
            series: Series to fetch, see `SERIES` ("rce", "csdac", "demand", "wind", "solar",
                "generation_scheduled", "generation_non_balancing", "exchange_parallel", "exchange_non_parallel").
            issue_time: Local time "HH:MM" (in `timezone`) at which forecasts are issued on each row's day D.
                None returns raw values at delivery time without any availability filtering.
            lags: Calendar-day offset(s) relative to D (positive = past, negative = future, 0 = D itself).
                Default 0. Only 0 is allowed without `issue_time`.
            as_of: "publication" (value published at or before the issue time according to `publication_ts`)
                or "schedule" (value assumed available at the nominal time in `SCHEDULE`, ignoring re-publications).
            timezone: Time zone defining the local day D and the issue time; must match the TimezoneTransformer.
            freq: Output resolution, "1h" (mean of the quarter-hours) or "15min" (native).
            prefix: Column prefix (default "pse"); "" for none.
        """
        if isinstance(series, str):
            series = [series]
        if isinstance(lags, int):
            lags = [lags]
        self.series = list(series or [])
        self.issue_time = issue_time
        self.lags = list(lags) if lags is not None else []
        self.as_of = as_of
        self.timezone = timezone
        self.freq = freq
        self.prefix = prefix

        self.session = requests.Session()
        self.logger = get_logger(__name__)

        self._validate_config()

    def __del__(self):
        if hasattr(self, "session") and self.session:
            self.session.close()

    def _validate_config(self) -> bool:
        if not self.series:
            raise ValueError("At least one series must be specified")
        unknown = [s for s in self.series if s not in SERIES]
        if unknown:
            raise ValueError(f"Unknown PSE series {unknown}. Available: {sorted(SERIES)}")
        if len(set(self.series)) != len(self.series):
            raise ValueError(f"Duplicate series in {self.series}")
        if not self.lags or not all(isinstance(k, (int, np.integer)) and not isinstance(k, bool) for k in self.lags):
            raise ValueError(f"lags must be a non-empty list of integers, got {self.lags!r}")
        if len(set(self.lags)) != len(self.lags):
            raise ValueError(f"Duplicate lags in {self.lags}")
        if self.as_of not in self.VALID_AS_OF:
            raise ValueError(f"as_of must be one of {self.VALID_AS_OF}, got {self.as_of!r}")
        if self.freq not in self.VALID_FREQ:
            raise ValueError(f"freq must be one of {self.VALID_FREQ}, got {self.freq!r}")
        try:
            ZoneInfo(self.timezone)
        except Exception:
            raise ValueError(f"Invalid timezone: {self.timezone!r}")
        if self.issue_time is None:
            if self.lags != [0]:
                raise ValueError(
                    "lags other than [0] need issue_time: without it values are not filtered by publication time"
                )
        else:
            self._issue_offset()
        return True

    def _issue_offset(self) -> pd.Timedelta:
        # Only strings: unquoted 17:00 in YAML is the integer 1020, which fromisoformat would read as 10:20.
        if not isinstance(self.issue_time, str) or not re.fullmatch(r"\d{1,2}:\d{2}", self.issue_time):
            raise ValueError(f"issue_time must be a string 'HH:MM', got {self.issue_time!r}")
        try:
            t = datetime.time.fromisoformat(self.issue_time.zfill(5))
        except ValueError:
            raise ValueError(f"issue_time must be a string 'HH:MM', got {self.issue_time!r}")
        return pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)

    # ------------------------------------------------------------------ fetching

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
        end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")
        if end <= start:
            raise ValueError(f"End timestamp ({end}) must be after start timestamp ({start})")

        index = pd.date_range(start.ceil(self.freq), end.floor(self.freq), freq=self.freq, tz="UTC", name="datetime")
        if index.empty:
            return pd.DataFrame(index=index)

        local_days = index.tz_convert(self.timezone).tz_localize(None).normalize()
        first_day = local_days.min() - pd.Timedelta(days=max(max(self.lags), 0) + 1)
        last_day = local_days.max() + pd.Timedelta(days=max(-min(self.lags), 0) + 1)
        if local_days.min() - pd.Timedelta(days=max(max(self.lags), 0)) < HISTORY_START:
            self.logger.warning(f"PSE: the API has no data before {HISTORY_START.date()}; earlier rows will be empty")

        tables = self._fetch_tables(first_day, last_day)

        columns = {}
        if self.issue_time is None:
            for name in self.series:
                values, _ = tables[name]
                columns[self._column_name(name, None)] = values.reindex(index).to_numpy()
        else:
            issue_utc = self._issue_times(index)
            for name in self.series:
                values, available = tables[name]
                for lag in self.lags:
                    col = self._column_name(name, lag)
                    v, a = self._align(values, available, index, lag)
                    keep = a <= issue_utc
                    columns[col] = np.where(keep, v, np.nan)
                    self._log_dropped(col, np.isfinite(v), keep)
                    self._log_missing(col, np.isfinite(v))

        return pd.DataFrame(columns, index=index)

    def _fetch_tables(self, first_day: pd.Timestamp, last_day: pd.Timestamp) -> Dict[str, tuple]:
        by_entity: Dict[str, List[str]] = {}
        for name in self.series:
            entity, field, _ = SERIES[name]
            by_entity.setdefault(entity, [])
            if field not in by_entity[entity]:
                by_entity[entity].append(field)

        tables = {}
        for entity, fields in by_entity.items():
            self.logger.info(f"PSE: downloading {entity} ({', '.join(fields)}) {first_day.date()} to {last_day.date()}")
            records = self._fetch_entity(entity, fields, first_day, last_day)
            frame = self._parse_records(records, fields)
            for name in self.series:
                s_entity, field, kind = SERIES[name]
                if s_entity == entity:
                    tables[name] = self._series_table(frame, field, kind)
        return tables

    def _fetch_entity(
        self, entity: str, fields: List[str], first_day: pd.Timestamp, last_day: pd.Timestamp
    ) -> List[dict]:
        select = ["business_date", "dtime_utc", "period_utc", "publication_ts_utc"] + fields
        params = {
            "$filter": f"business_date ge '{first_day:%Y-%m-%d}' and business_date le '{last_day:%Y-%m-%d}'",
            "$select": ",".join(select),
            "$first": self.PAGE_SIZE,
        }
        url = f"{self.API_URL}/{entity}"
        records: List[dict] = []
        while url:
            data = self._get_json(url, params)
            records.extend(data.get("value", []))
            # nextLink already carries the filter and the continuation token
            url, params = data.get("nextLink"), None
        return records

    def _get_json(self, url: str, params: Optional[dict], max_retries: int = 3) -> dict:
        for attempt in range(1, max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=120)
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {response.status_code}", response=response)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                if attempt == max_retries:
                    raise
                self.logger.warning(f"PSE: request failed (attempt {attempt}/{max_retries}): {e}")
                time.sleep(5 * attempt)
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------ parsing

    @staticmethod
    def _period_minutes(period_utc: pd.Series) -> pd.Series:
        """Duration in minutes from 'HH:MM - HH:MM' (end may be '24:00'); 15 when unparseable."""
        parts = period_utc.astype(str).str.extract(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")
        start = parts[0].astype(float) * 60 + parts[1].astype(float)
        end = parts[2].astype(float) * 60 + parts[3].astype(float)
        minutes = (end - start) % 1440
        return minutes.where(minutes > 0).fillna(15).astype(int)

    def _parse_records(self, records: List[dict], fields: List[str]) -> pd.DataFrame:
        if not records:
            empty = pd.Series([], dtype="datetime64[ns]")
            frame = pd.DataFrame(
                {"start": empty.dt.tz_localize("UTC"), "minutes": [], "business_date": empty, "published": empty}
            )
            return frame.assign(**{field: np.array([], dtype=float) for field in fields})
        raw = pd.DataFrame.from_records(records)
        end = pd.to_datetime(raw["dtime_utc"], format="ISO8601").dt.tz_localize("UTC")
        minutes = self._period_minutes(raw["period_utc"])
        frame = pd.DataFrame(
            {
                "start": end - pd.to_timedelta(minutes, unit="min"),
                "minutes": minutes,
                "business_date": pd.to_datetime(raw["business_date"]),
                # naive UTC, like every availability/issue time below
                "published": pd.to_datetime(raw["publication_ts_utc"], format="ISO8601"),
            }
        )
        for field in fields:
            frame[field] = pd.to_numeric(raw[field], errors="coerce") if field in raw else np.nan
        return frame

    def _availability(self, frame: pd.DataFrame, kind: str) -> pd.Series:
        """Time (naive UTC) from which each record counts as known."""
        if self.as_of == "publication":
            return frame["published"]
        days, hhmm = SCHEDULE[kind]
        t = datetime.time.fromisoformat(hhmm)
        local = frame["business_date"] + pd.Timedelta(days=days, hours=t.hour, minutes=t.minute)
        local = local.dt.tz_localize(PSE_TZ, ambiguous=True, nonexistent="shift_forward")
        return local.dt.tz_convert("UTC").dt.tz_localize(None)

    def _series_table(self, frame: pd.DataFrame, field: str, kind: str) -> tuple:
        """(values, availability) indexed by UTC slot start at `freq`."""
        data = pd.DataFrame(
            {
                "start": frame["start"],
                "minutes": frame["minutes"],
                "value": frame[field],
                "available": self._availability(frame, kind),
            }
        ).dropna(subset=["value"])
        data = data.sort_values("start")
        data = data[~data["start"].duplicated(keep="last")]

        if self.freq == "1h":
            grouped = data.groupby(data["start"].dt.floor("1h"))
            hourly = pd.DataFrame(
                {
                    "value": grouped["value"].mean(),
                    "available": grouped["available"].max(),
                    "minutes": grouped["minutes"].sum(),
                }
            )
            # Keep only complete hours so a partial hour isn't passed off as an hourly mean.
            hourly = hourly[hourly["minutes"] == 60]
            values, available = hourly["value"], hourly["available"]
        else:
            data = data[data["minutes"] == 15].set_index("start")
            values, available = data["value"], data["available"]

        values.index.name = available.index.name = "datetime"
        return values.astype(float), available

    # ------------------------------------------------------------------ as-of alignment

    def _issue_times(self, index: pd.DatetimeIndex) -> np.ndarray:
        local_midnight = index.tz_convert(self.timezone).tz_localize(None).normalize()
        issue_local = local_midnight + self._issue_offset()
        issue = issue_local.tz_localize(self.timezone, ambiguous=True, nonexistent="shift_forward")
        return issue.tz_convert("UTC").tz_localize(None).to_numpy()

    def _align(self, values: pd.Series, available: pd.Series, index: pd.DatetimeIndex, lag: int) -> tuple:
        """Values and availability for delivery (D - lag, same local wall-clock time) on each row of `index`."""
        if lag == 0:
            v = values.reindex(index).to_numpy(dtype=float)
            a = available.reindex(index).to_numpy()
            return v, a

        def by_local_time(s: pd.Series) -> pd.Series:
            s = s.copy()
            s.index = s.index.tz_convert(self.timezone).tz_localize(None)
            # The repeated hour on the autumn DST day: keep the first (summer-time) occurrence.
            return s[~s.index.duplicated(keep="first")]

        keys = index.tz_convert(self.timezone).tz_localize(None) - pd.Timedelta(days=lag)
        v = by_local_time(values).reindex(keys).to_numpy(dtype=float)
        a = by_local_time(available).reindex(keys).to_numpy()
        return v, a

    def _log_dropped(self, column: str, has_value: np.ndarray, keep: np.ndarray) -> None:
        n_values = int(has_value.sum())
        n_dropped = int((has_value & ~keep).sum())
        if n_values == 0 or n_dropped == 0:
            return
        if n_dropped == n_values:
            self.logger.warning(
                f"PSE: {column} is empty: none of its {n_values} values was published by {self.issue_time} "
                f"on the row's day (as_of={self.as_of!r}); use a larger lag"
            )
        else:
            self.logger.warning(
                f"PSE: {column}: {n_dropped} of {n_values} values were published after {self.issue_time} on the "
                f"row's day and were set to NaN (as_of={self.as_of!r}; PSE keeps only the last version, "
                f"so re-published values count as late)"
            )

    def _log_missing(self, column: str, has_value: np.ndarray) -> None:
        n_missing = int((~has_value).sum())
        if n_missing and n_missing < len(has_value):
            self.logger.info(
                f"PSE: {column}: {n_missing} of {len(has_value)} rows have no source value "
                f"(no data, or the hour missing on a spring DST day)"
            )

    def _column_name(self, series: str, lag: Optional[int]) -> str:
        base = f"{self.prefix}_{series}" if self.prefix else series
        if not lag:
            return base
        return f"{base}_d-{lag}" if lag > 0 else f"{base}_d+{-lag}"

    def get_cache_config(self) -> dict:
        return {
            "source_type": "pse",
            "schema_version": 1,
            "series": sorted(self.series),
            "issue_time": self.issue_time,
            "lags": sorted(int(k) for k in self.lags),
            "as_of": self.as_of,
            "timezone": self.timezone,
            "freq": self.freq,
            "prefix": self.prefix,
        }
