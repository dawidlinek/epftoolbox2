import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from epftoolbox2.logging import get_logger
from .base import DataSource

# The five open-access datasets of Lago, Marcjasz, De Schutter & Weron (2021), Applied Energy 293, 116983.
# As in epftoolbox's read_data, the first column is the date, the second the price, then two exogenous series;
# both are day-ahead forecasts for the row's (day, hour), available the day before. The exogenous columns are
# identified by a keyword in their header (file headers e.g. "Ampirion Load Forecast", "PV+Wind Forecast").
# "timezone" is the fixed UTC offset of the market's standard time: the files are local wall-clock time with
# 24 values on every day (DST days interpolated/averaged), which a fixed offset represents without gaps or repeats.
DATASETS = {
    "NP": {
        "market": "Nord Pool",
        "timezone": "Etc/GMT-1",  # CET, UTC+1
        "exogenous": {"load": "load_forecast", "wind": "wind_forecast"},
        "test_period": ("2016-12-27", "2018-12-24"),
    },
    "PJM": {
        "market": "PJM (COMED zone)",
        "timezone": "Etc/GMT+5",  # EST, UTC-5
        "exogenous": {"system": "load_forecast", "zonal": "zonal_load_forecast"},
        "test_period": ("2016-12-27", "2018-12-24"),
    },
    "FR": {
        "market": "EPEX-FR",
        "timezone": "Etc/GMT-1",
        "exogenous": {"load": "load_forecast", "generation": "generation_forecast"},
        "test_period": ("2015-01-04", "2016-12-31"),
    },
    "BE": {
        "market": "EPEX-BE",
        "timezone": "Etc/GMT-1",
        # Lago et al. use French load and generation forecasts as the Belgian exogenous inputs
        "exogenous": {"load": "load_forecast_fr", "generation": "generation_forecast_fr"},
        "test_period": ("2015-01-04", "2016-12-31"),
    },
    "DE": {
        "market": "EPEX-DE",
        "timezone": "Etc/GMT-1",
        "exogenous": {"load": "load_forecast_amprion", "wind": "wind_solar_forecast"},
        "test_period": ("2016-01-04", "2017-12-31"),
    },
}


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join("~", ".cache")
    return Path(base).expanduser() / "epftoolbox2" / "epftoolbox-benchmark"


class EpftoolboxBenchmarkSource(DataSource):
    """
    The epftoolbox benchmark datasets (NP, PJM, FR, BE, DE) of Lago et al. (2021), for comparing results with
    the original epftoolbox. Downloaded on first use from Zenodo record 4624805 into a local cache directory and
    verified against the md5 published in the record's metadata. The data are not shipped with epftoolbox2:
    the record's licence is "other-open" and its authors state that they don't own the data.

    The files are local wall-clock time with exactly 24 values per day (the spring DST hour interpolated, the two
    autumn hours averaged). The source maps them onto a UTC index through the market's fixed standard-time offset
    (`index_timezone`, e.g. "Etc/GMT-1" = UTC+1 for the European markets): every day keeps 24 hourly rows. Convert
    with `TimezoneTransformer(target_tz=source.index_timezone)`, not with a DST zone such as "Europe/Berlin".

    Columns: `price` and two exogenous series named by meaning (see `DATASETS`), e.g. `load_forecast_amprion`
    and `wind_solar_forecast` for DE. Each exogenous value at (day d, hour h) is the day-ahead forecast for
    (d, h), available on d-1.

    Example:
        >>> source = EpftoolboxBenchmarkSource("DE")
        >>> df = source.fetch(pd.Timestamp("2011-12-31", tz="UTC"), pd.Timestamp("2018-01-01", tz="UTC"))
        >>> # columns: price, load_forecast_amprion, wind_solar_forecast
    """

    RECORD_ID = "4624805"
    RECORD_API = "https://zenodo.org/api/records/4624805"

    def __init__(self, dataset: str, cache_dir: Optional[str] = None, prefix: str = ""):
        """
        Args:
            dataset: "NP", "PJM", "FR", "BE" or "DE".
            cache_dir: Directory for the downloaded CSV files (default: $XDG_CACHE_HOME or ~/.cache, then
                epftoolbox2/epftoolbox-benchmark).
            prefix: Prefix for the column names, "{prefix}_{column}" (default: none).
        """
        self.dataset = dataset.upper() if isinstance(dataset, str) else dataset
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.prefix = prefix

        self.session = requests.Session()
        self.logger = get_logger(__name__)

        self._validate_config()

    def __del__(self):
        if hasattr(self, "session") and self.session:
            self.session.close()

    def _validate_config(self) -> bool:
        if self.dataset not in DATASETS:
            raise ValueError(f"Unknown dataset {self.dataset!r}. Available: {sorted(DATASETS)}")
        return True

    @property
    def index_timezone(self) -> str:
        """Fixed-offset zone that turns the UTC index back into the files' wall-clock time."""
        return DATASETS[self.dataset]["timezone"]

    @property
    def test_period(self) -> tuple:
        """(first, last) day of the test period in Lago et al. (2021), Table 1: the last 104 weeks."""
        return DATASETS[self.dataset]["test_period"]

    @property
    def _path(self) -> Path:
        directory = Path(self.cache_dir).expanduser() if self.cache_dir else default_cache_dir()
        return directory / f"{self.dataset}.csv"

    # ------------------------------------------------------------------ download and verification

    @staticmethod
    def _md5(path: Path) -> str:
        digest = hashlib.md5(usedforsecurity=False)
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    def _record_file(self) -> dict:
        """The file entry of this dataset in the Zenodo record metadata: {"md5": ..., "url": ...}."""
        response = self.session.get(self.RECORD_API, timeout=60)
        response.raise_for_status()
        key = f"{self.dataset}.csv"
        for entry in response.json().get("files", []):
            if entry.get("key") == key:
                algorithm, _, checksum = str(entry.get("checksum", "")).partition(":")
                if algorithm != "md5" or not checksum:
                    raise ValueError(
                        f"Zenodo record {self.RECORD_ID}: no md5 checksum for {key}: {entry.get('checksum')!r}"
                    )
                url = (entry.get("links") or {}).get("self") or f"{self.RECORD_API}/files/{key}/content"
                return {"md5": checksum.lower(), "url": url}
        raise ValueError(f"Zenodo record {self.RECORD_ID} has no file {key}")

    def _ensure_file(self) -> Path:
        path = self._path
        sidecar = path.with_name(path.name + ".md5")
        if path.exists() and sidecar.exists():
            if self._md5(path) == sidecar.read_text().strip():
                return path
            self.logger.warning(f"Benchmark: {path} doesn't match its recorded md5; downloading it again")

        record = self._record_file()
        if path.exists() and self._md5(path) == record["md5"]:
            sidecar.write_text(record["md5"])
            return path

        self.logger.info(f"Benchmark: downloading {self.dataset}.csv from Zenodo record {self.RECORD_ID}")
        response = self.session.get(record["url"], timeout=120)
        response.raise_for_status()
        content = response.content
        checksum = hashlib.md5(content, usedforsecurity=False).hexdigest()
        if checksum != record["md5"]:
            raise ValueError(
                f"md5 mismatch for {self.dataset}.csv from Zenodo record {self.RECORD_ID}: "
                f"got {checksum}, the record publishes {record['md5']}. Nothing was saved."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(content)
        os.replace(tmp, path)
        sidecar.write_text(checksum)
        return path

    # ------------------------------------------------------------------ parsing

    def _columns(self, header: list) -> dict:
        """{file column: output name}: first value column = price, the others matched by keyword."""
        if len(header) != 3:
            raise ValueError(f"{self.dataset}.csv: expected a date and 3 value columns, got {header}")
        mapping = {header[0]: "price"}
        for keyword, name in DATASETS[self.dataset]["exogenous"].items():
            matches = [c for c in header[1:] if keyword in str(c).lower()]
            if len(matches) != 1:
                raise ValueError(f"{self.dataset}.csv: can't identify the '{keyword}' column in {header[1:]}")
            mapping[matches[0]] = name
        if len(mapping) != 3:
            raise ValueError(f"{self.dataset}.csv: ambiguous exogenous columns {header[1:]}")
        return mapping

    def _load(self) -> pd.DataFrame:
        path = self._ensure_file()
        raw = pd.read_csv(path, index_col=0)
        mapping = self._columns(list(raw.columns))
        # Same parsing as epftoolbox's read_data
        local = pd.DatetimeIndex(pd.to_datetime(raw.index))
        if local.tz is not None:
            raise ValueError(f"{path}: expected naive local timestamps, got time zone {local.tz}")
        if local.has_duplicates or not local.is_monotonic_increasing:
            raise ValueError(f"{path}: timestamps are not strictly increasing; the dates were not parsed as expected")

        per_day = pd.Series(1, index=local).groupby(local.normalize()).sum()
        steps = np.diff(local.asi8)
        if (per_day != 24).any() or (steps != pd.Timedelta("1h").value).any():
            self.logger.warning(
                f"Benchmark: {self.dataset}.csv is not a gap-free series of 24 hourly values per day; "
                f"days with other counts: {list(per_day[per_day != 24].index.date[:5])}"
            )

        df = raw.rename(columns=mapping)[["price", *DATASETS[self.dataset]["exogenous"].values()]].astype(float)
        df.index = local.tz_localize(self.index_timezone).tz_convert("UTC").rename("datetime")
        if self.prefix:
            df.columns = [f"{self.prefix}_{c}" for c in df.columns]
        return df

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Rows with ``start <= index < end`` (the source contract's half-open range; ``df.loc[start:end]``
        would also include a row exactly at ``end``)."""
        start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
        end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")
        if end <= start:
            raise ValueError(f"End timestamp ({end}) must be after start timestamp ({start})")
        df = self._load()
        result = df[(df.index >= start) & (df.index < end)].copy()
        if result.empty:
            self.logger.warning(
                f"Benchmark: {self.dataset} covers {df.index[0]} to {df.index[-1]} (UTC); nothing in {start} to {end}"
            )
        return result

    def get_cache_config(self) -> Optional[dict]:
        # The source keeps its own verified copy of the file; a second cache would only duplicate it.
        return None
