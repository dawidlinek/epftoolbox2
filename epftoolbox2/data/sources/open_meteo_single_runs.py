import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn

from ._open_meteo_common import OpenMeteoBase, normalise_derived, parse_hourly, shift_local_days


class OpenMeteoSingleRunsSource(OpenMeteoBase):
    """
    Open-Meteo weather forecasts from exactly the model run that was available at the issue time.

    Uses the Open-Meteo Single Runs API, which stores every model run separately. For each local day D (in
    ``timezone``) the source takes the latest run on the ``run_interval_h`` cycle initialised at least
    ``publication_delay_h`` hours before ``issue_time`` on D, and fills ``{variable}_d+i`` on every row of D with
    that run's forecast for the same local wall-clock hour on D+i. If the archive lacks that run, or the run is too
    short for a lead, up to ``MAX_FALLBACK_RUNS`` older runs fill the gaps; runs published after the issue time are
    never used.

    Archive start (probed against the API in September 2026, latitude/longitude do not matter): ECMWF IFS HRES
    9 km (``ecmwf_ifs``) from 2024-03-14 00Z (only 00/12 UTC runs at first; other cycles follow later), ICON and
    GFS (``icon_seamless``, ``gfs_seamless``, ...) from 2026-04-02 00Z, JMA GSM (``jma_gsm``, ``jma_seamless``)
    from 2026-05-13 00Z. A ``model`` not covered here should be checked the same way before relying on it.

    Example:
        >>> source = OpenMeteoSingleRunsSource(latitude=52.23, longitude=21.01, horizon=7, prefix="warsaw")
        >>> df = source.fetch(pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-02-01", tz="UTC"))
    """

    API_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
    SCHEMA_VERSION = 1
    MAX_HORIZON = 14
    MAX_FALLBACK_RUNS = 4
    UNAVAILABLE_RUN = "requested model run is not available"

    def __init__(
        self,
        latitude: float,
        longitude: float,
        horizon: int = 7,
        model: str = "ecmwf_ifs",
        columns: Optional[List[str]] = None,
        prefix: str = "",
        proxy_url: Optional[str] = None,
        timezone: str = "Europe/Warsaw",
        issue_time: str = "10:00",
        publication_delay_h: float = 10,
        run_interval_h: int = 6,
        derived: Optional[List[str]] = None,
        hdd_base_c: float = 18.0,
        cdd_base_c: float = 21.0,
        strict: bool = False,
    ):
        """
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            horizon: Days ahead (d+1 .. d+horizon), 1-14; leads beyond the run's length come back empty.
            model: Weather model (default "ecmwf_ifs", the model with the longest single-run archive)
            columns: Weather variables (default: DEFAULT_COLUMNS)
            prefix: Prefix for column names
            proxy_url: HTTP(S) proxy URL
            timezone: Market time zone of the local day D, the d+i wall-clock hour and ``issue_time``
            issue_time: Local time ("HH:MM") on D at which the forecast is issued
            publication_delay_h: Hours between a run's initialisation and its availability (default 10)
            run_interval_h: Hours between model runs, starting at 00 UTC (6 for global models, 3 for ICON-EU/D2)
            derived: Extra columns computed from the forecast: "hdd", "cdd" (need temperature_2m) and
                "wind_speed_100m_cubed" (needs wind_speed_100m). Off by default.
            hdd_base_c: Base temperature of "hdd" (°C)
            cdd_base_c: Base temperature of "cdd" (°C)
            strict: Raise instead of warning when a returned column has missing values
        """
        self.latitude = latitude
        self.longitude = longitude
        self.horizon = horizon
        self.model = model
        self.columns = list(columns) if columns else list(self.DEFAULT_COLUMNS)
        self.prefix = prefix
        self.proxy_url = proxy_url
        self.timezone = timezone
        self.issue_time = issue_time
        self.publication_delay_h = publication_delay_h
        self.run_interval_h = run_interval_h
        self.derived = normalise_derived(derived)
        self.hdd_base_c = hdd_base_c
        self.cdd_base_c = cdd_base_c
        self.strict = strict

        self._setup()
        self._validate_config()

    @property
    def _label(self) -> str:
        return f"Open-Meteo single runs [{self.latitude}, {self.longitude}]"

    def _validate_config(self) -> bool:
        self._validate_common(max_horizon=self.MAX_HORIZON)
        interval = self.run_interval_h
        if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0 or 24 % interval:
            raise ValueError(f"run_interval_h must be a divisor of 24 (1, 2, 3, 4, 6, 8, 12, 24), got {interval!r}")
        return True

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch hourly weather forecast features for rows in [start, end)

        Returns:
            DataFrame with a UTC hourly index covering [start, end) and columns ``{prefix}_{variable}_d+{i}``.
            Trailing rows without any value are not returned.
        """
        start, end = self._to_utc(start), self._to_utc(end)
        if end <= start:
            raise ValueError(f"End timestamp ({end}) must be after start timestamp ({start})")

        rows = pd.date_range(start.ceil("1h"), end, freq="1h", inclusive="left")
        if rows.empty:
            return pd.DataFrame()

        started = time.time()
        issue, final = self._issue_instants(rows)
        interval = pd.Timedelta(hours=self.run_interval_h)
        # Latest run on the cycle that was published by the issue instant.
        chosen_init = (issue - pd.Timedelta(hours=self.publication_delay_h)).floor(f"{self.run_interval_h}h")
        targets = {lead: shift_local_days(rows, self.timezone, lead) for lead in range(1, self.horizon + 1)}
        values = {
            f"{column}_d+{lead}": np.full(len(rows), np.nan)
            for lead in range(1, self.horizon + 1)
            for column in self.columns
        }

        runs: Dict[pd.Timestamp, Optional[pd.DataFrame]] = {}
        groups = pd.Series(np.arange(len(rows))).groupby(chosen_init.asi8).indices
        self.logger.info(f"{self._label}: Start downloading {len(groups)} model runs")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"[cyan]{self._label}: Downloading...", total=len(groups))
            for init_ns, positions in groups.items():
                init = pd.Timestamp(init_ns, tz="UTC")
                progress.update(task, description=f"[cyan]{self._label}: run {init:%Y-%m-%d %H}z")
                self._fill_from_runs(init, interval, positions, targets, values, runs)
                progress.advance(task)

        result = pd.DataFrame(values, index=rows)
        result.index.name = "datetime"
        result = self._finish(result, final)
        self.logger.info(f"{self._label}: Download completed in {time.time() - started:.2f} sec")
        return result

    def _fill_from_runs(self, init, interval, positions, targets, values, runs) -> None:
        """Fill the rows at `positions` from run `init`; older runs only fill valid times it doesn't cover.

        A valid time inside an available run's range is settled by that run, even where one variable is null:
        falling back would hide a gap in that variable behind an older forecast.
        """
        settled = {lead: np.zeros(len(positions), dtype=bool) for lead in targets}
        for age in range(self.MAX_FALLBACK_RUNS + 1):
            run_init = init - age * interval
            if run_init not in runs:
                runs[run_init] = self._fetch_run(run_init)
            run = runs[run_init]
            if run is None or run.empty:
                continue
            has_value = run.notna().any(axis=1).to_numpy()
            if not has_value.any():
                continue
            covered_until = run.index[has_value].max()
            used = False
            for lead, target in targets.items():
                valid = target[positions]
                idx = run.index.get_indexer(valid)
                take = (idx >= 0) & np.asarray(valid <= covered_until) & ~settled[lead]
                if not take.any():
                    continue
                used = True
                for column in self.columns:
                    if column in run.columns:
                        values[f"{column}_d+{lead}"][positions[take]] = run[column].to_numpy()[idx[take]]
                settled[lead] |= take
            if used and age:
                self.logger.info(f"{self._label}: run {run_init:%Y-%m-%d %H}z fills in for run {init:%Y-%m-%d %H}z")
            # Stop when everything is settled, or when an available run added nothing: older runs reach even less
            # far ahead (this only continues past runs that are shorter than their predecessors).
            if all(mask.all() for mask in settled.values()) or (age and not used):
                return

    def _fetch_run(self, init: pd.Timestamp) -> Optional[pd.DataFrame]:
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": list(self.columns),
            "models": self.model,
            "timezone": "GMT",
            "run": init.strftime("%Y-%m-%dT%H:%M"),
            "forecast_days": min(16, self.horizon + 3),
        }
        data = self._request(self.API_URL, params, unavailable=self.UNAVAILABLE_RUN)
        if data is None:
            self.logger.info(f"{self._label}: run {init:%Y-%m-%d %H}z of {self.model} is not in the archive")
            return None
        return parse_hourly(data)

    def get_cache_config(self) -> dict:
        return {
            "source_type": "open_meteo_single_runs",
            "schema_version": self.SCHEMA_VERSION,
            **self._common_cache_config(),
            "run_interval_h": self.run_interval_h,
        }
