import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from zoneinfo import ZoneInfo

import holidays as holidays_lib
import numpy as np
import pandas as pd

from epftoolbox2.logging import get_logger
from .base import DataSource


class SettlementPriceSource(DataSource):
    """
    Daily settlement/close prices (e.g. TTF gas, EUA carbon, API2 coal, Brent) from your own CSV or Parquet files,
    placed on each row as they were known at the issue time of the row's local day D.

    For a series `x`, the prevailing close on calendar day T is the close of the last trading day <= T
    (forward-filled over weekends and holidays only). Column `x_d-k` on every row of local day D holds the
    prevailing close on D-k, kept only if that close was known (its trading day at `close_time`) by `issue_time`
    on D. With the default `lags=1` that is D-1's close, Friday's close on a Monday. A missing close on a trading
    day is never filled from an earlier day.

    Nothing is downloaded: settlement prices are licensed and may not be redistributed.

    Example:
        >>> source = SettlementPriceSource(
        ...     file_path="data/fuels.csv",
        ...     columns={"TTF": "ttf", "EUA": "eua"},
        ...     timezone="Europe/Warsaw",
        ...     close_time="19:30",
        ...     close_timezone="Europe/London",
        ...     trading_calendar="IFEU",
        ... )
        >>> df = source.fetch(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-06-01", tz="UTC"))
        >>> # columns: ttf_d-1, eua_d-1
    """

    VALID_EXTENSIONS = (".csv", ".parquet", ".pq")

    def __init__(
        self,
        file_path: Union[str, List[str]],
        columns: Union[List[str], Dict[str, str]],
        timezone: str,
        issue_time: str = "11:30",
        close_time: str = "23:59",
        close_timezone: Optional[str] = None,
        date_column: str = "date",
        date_format: Optional[str] = None,
        trading_calendar: str = "weekdays",
        lags: Union[int, List[int]] = 1,
        freq: str = "1h",
        separator: str = ",",
    ):
        """
        Args:
            file_path: CSV or Parquet file, or a list of them (merged on the date column).
            columns: File columns to use, as a list (names kept) or a dict {file column: output name}.
            timezone: Time zone of the forecast day D and of `issue_time`; must match the TimezoneTransformer.
            issue_time: Local time "HH:MM" at which forecasts are issued on day D (default "11:30").
            close_time: Time "HH:MM" on the trading day from which its close counts as known, e.g. the settlement
                publication time. The default "23:59" never uses a same-day close.
            close_timezone: Time zone of `close_time` (default: `timezone`).
            date_column: Column with the trading date (default "date").
            date_format: strftime format of the dates, e.g. "%d.%m.%Y". None lets pandas infer it (ISO dates are safe).
            trading_calendar: Expected trading days: "weekdays" (Mon-Fri), a `holidays` financial market code
                ("IFEU" = ICE Futures Europe, "ECB"/"TAR", "NYSE", ...: weekdays without that market's holidays),
                or "file" (every date in the file is a trading day and every other date is not).
            lags: Day offset(s) k >= 0: column `x_d-k` holds the prevailing close on D-k (default 1).
            freq: Row frequency of the output (default "1h"); every row of a day gets that day's value.
            separator: CSV separator (default ",").
        """
        self.file_path = str(file_path) if isinstance(file_path, (str, Path)) else [str(p) for p in file_path]
        if isinstance(columns, str):
            columns = [columns]
        self.columns = dict(columns) if isinstance(columns, dict) else list(columns or [])
        self.timezone = timezone
        self.issue_time = issue_time
        self.close_time = close_time
        self.close_timezone = close_timezone
        self.date_column = date_column
        self.date_format = date_format
        self.trading_calendar = trading_calendar
        self.lags = [lags] if isinstance(lags, int) else list(lags)
        self.freq = freq
        self.separator = separator

        self.logger = get_logger(__name__)

        self._validate_config()

    @property
    def _paths(self) -> List[Path]:
        paths = [self.file_path] if isinstance(self.file_path, str) else self.file_path
        return [Path(p).expanduser() for p in paths]

    @property
    def _mapping(self) -> Dict[str, str]:
        return self.columns if isinstance(self.columns, dict) else {c: c for c in self.columns}

    def _validate_config(self) -> bool:
        if not self._paths:
            raise ValueError("At least one file_path must be given")
        for path in self._paths:
            if path.suffix.lower() not in self.VALID_EXTENSIONS:
                raise ValueError(f"File must be one of {self.VALID_EXTENSIONS}, got: {path}")
            if not path.exists():
                raise FileNotFoundError(f"Settlement price file not found: {path}")
        if not self.columns:
            raise ValueError("columns must list at least one file column")
        names = list(self._mapping.values())
        if len(set(names)) != len(names):
            raise ValueError(f"Output column names must be unique, got {names}")
        if not self.date_column:
            raise ValueError("date_column cannot be empty")
        for tz in (self.timezone, self.close_timezone or self.timezone):
            try:
                ZoneInfo(tz)
            except Exception:
                raise ValueError(f"Invalid timezone: {tz!r}")
        self._time_offset(self.issue_time, "issue_time")
        self._time_offset(self.close_time, "close_time")
        if self.trading_calendar not in ("weekdays", "file") and self.trading_calendar not in self._financial_markets():
            raise ValueError(
                f"trading_calendar must be 'weekdays', 'file' or one of {sorted(self._financial_markets())}, "
                f"got {self.trading_calendar!r}"
            )
        if not self.lags or not all(isinstance(k, int) and not isinstance(k, bool) and k >= 0 for k in self.lags):
            raise ValueError(f"lags must be non-negative integers (a future close is never known), got {self.lags!r}")
        if len(set(self.lags)) != len(self.lags):
            raise ValueError(f"Duplicate lags in {self.lags}")
        try:
            step = pd.Timedelta(self.freq)
        except ValueError:
            raise ValueError(f"Invalid freq: {self.freq!r}")
        if not pd.Timedelta(0) < step <= pd.Timedelta(days=1):
            raise ValueError(f"freq must be positive and at most one day, got {self.freq!r}")
        return True

    @staticmethod
    def _financial_markets() -> List[str]:
        return list(holidays_lib.list_supported_financial())

    @staticmethod
    def _time_offset(value: str, name: str) -> pd.Timedelta:
        try:
            t = datetime.time.fromisoformat(str(value))
        except ValueError:
            raise ValueError(f"{name} must be 'HH:MM', got {value!r}")
        return pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)

    # ------------------------------------------------------------------ loading

    def _read_file(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, sep=self.separator)
        try:
            return pd.read_parquet(path)
        except ImportError as e:
            raise ImportError(f"Reading {path} needs a Parquet engine: pip install pyarrow") from e

    def _load_closes(self) -> pd.DataFrame:
        """Closes indexed by trading date (naive, sorted), one column per output name. NaN = no close."""
        mapping = self._mapping
        parts = []
        found = set()
        for path in self._paths:
            raw = self._read_file(path)
            if self.date_column not in raw.columns:
                raise ValueError(f"Date column '{self.date_column}' not found in {path}. Columns: {list(raw.columns)}")
            present = [c for c in mapping if c in raw.columns]
            found.update(present)
            dates = pd.to_datetime(raw[self.date_column], format=self.date_format)
            if dates.dt.tz is not None:
                dates = dates.dt.tz_convert(self.close_timezone or self.timezone).dt.tz_localize(None)
            for column in present:
                values = pd.to_numeric(raw[column], errors="coerce")
                unparsed = int(
                    (values.isna() & raw[column].notna() & (raw[column].astype(str).str.strip() != "")).sum()
                )
                if unparsed:
                    self.logger.warning(f"Settlement: {unparsed} non-numeric values in '{column}' of {path} ignored")
                parts.append(pd.DataFrame({"date": dates.dt.normalize(), "name": mapping[column], "value": values}))
        missing = [c for c in mapping if c not in found]
        if missing:
            raise ValueError(f"Columns {missing} not found in {[str(p) for p in self._paths]}")

        long = pd.concat(parts, ignore_index=True).dropna(subset=["date", "value"])
        conflicts = long.groupby(["date", "name"])["value"].nunique()
        conflicts = conflicts[conflicts > 1]
        if not conflicts.empty:
            examples = ", ".join(f"{name} on {date.date()}" for date, name in conflicts.index[:5])
            raise ValueError(f"Conflicting closes for the same trading date: {examples}")
        closes = long.drop_duplicates(["date", "name"]).pivot(index="date", columns="name", values="value")
        return closes.reindex(columns=list(mapping.values())).sort_index()

    def _trading_days(self, with_data: pd.DatetimeIndex, last_needed: pd.Timestamp) -> pd.DatetimeIndex:
        """Expected trading days from the first close to `last_needed`, plus every day that has a close."""
        if self.trading_calendar == "file" or with_data.empty:
            return with_data
        days = pd.bdate_range(with_data.min(), max(last_needed, with_data.max()))
        if self.trading_calendar != "weekdays":
            closed = holidays_lib.financial_holidays(
                self.trading_calendar, years=range(days.year.min(), days.year.max() + 1)
            )
            days = days[~days.isin(pd.DatetimeIndex(list(closed.keys())))]
        return days.union(with_data)

    # ------------------------------------------------------------------ fetching

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
        end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")
        if end <= start:
            raise ValueError(f"End timestamp ({end}) must be after start timestamp ({start})")

        index = pd.date_range(start.ceil(self.freq), end.floor(self.freq), freq=self.freq, tz="UTC", name="datetime")
        if index.empty:
            return pd.DataFrame(index=index)

        row_days = index.tz_convert(self.timezone).tz_localize(None).normalize()
        days = row_days.unique()
        issue = self._localize(days + self._time_offset(self.issue_time, "issue_time"), self.timezone)

        closes = self._load_closes()
        close_tz = self.close_timezone or self.timezone
        close_offset = self._time_offset(self.close_time, "close_time")

        columns = {}
        for name in closes.columns:
            series = closes[name].dropna()
            trading = self._trading_days(series.index, days.max())
            gaps = []
            for lag in self.lags:
                col = name if lag == 0 else f"{name}_d-{lag}"
                # Last trading day on or before D - lag, for every local day D of the output.
                pos = trading.searchsorted(days - pd.Timedelta(days=lag), side="right") - 1
                valid = pos >= 0
                if trading.empty:
                    used = pd.DatetimeIndex([pd.NaT] * len(days))
                else:
                    used = trading[np.clip(pos, 0, None)].where(valid)
                values = series.reindex(used).to_numpy(dtype=float)
                known_at = self._localize(used + close_offset, close_tz)
                keep = known_at <= issue
                day_values = np.where(keep, values, np.nan)
                gaps.extend(used[valid & np.isnan(values)])
                self._log_late(col, np.isfinite(values), keep)
                columns[col] = pd.Series(day_values, index=days).reindex(row_days).to_numpy()
            self._log_gaps(name, pd.DatetimeIndex(gaps).unique())

        return pd.DataFrame(columns, index=index)

    @staticmethod
    def _localize(naive: pd.DatetimeIndex, tz: str) -> np.ndarray:
        """Naive local wall-clock times -> naive UTC datetime64 (NaT stays NaT)."""
        local = naive.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")
        return local.tz_convert("UTC").tz_localize(None).to_numpy()

    def _log_late(self, column: str, has_value: np.ndarray, keep: np.ndarray) -> None:
        n_dropped = int((has_value & ~keep).sum())
        if n_dropped:
            self.logger.warning(
                f"Settlement: {column}: {n_dropped} of {int(has_value.sum())} days set to NaN because the close was "
                f"known only after {self.issue_time} on the row's day; use a larger lag"
            )

    def _log_gaps(self, name: str, gaps: pd.DatetimeIndex) -> None:
        if gaps.empty:
            return
        shown = ", ".join(str(d.date()) for d in gaps[:5]) + (", ..." if len(gaps) > 5 else "")
        self.logger.warning(
            f"Settlement: {name}: no close for {len(gaps)} expected trading day(s) ({shown}); rows that need them are "
            f"NaN, not filled from an earlier day. If they are holidays, choose a matching trading_calendar"
        )

    def get_cache_config(self) -> Optional[dict]:
        # Local files are re-read on every run, so edits to them are never hidden by a cache.
        return None
