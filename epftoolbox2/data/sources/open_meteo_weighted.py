import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from epftoolbox2.logging import get_logger
from .base import DataSource

_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
# Variables that cannot be averaged arithmetically.
_CIRCULAR = re.compile(r"direction")
_CATEGORICAL = ("weather_code",)


def _source_classes() -> Dict[str, type]:
    from .open_meteo import OpenMeteoSource
    from .open_meteo_single_runs import OpenMeteoSingleRunsSource

    return {"OpenMeteoSource": OpenMeteoSource, "OpenMeteoSingleRunsSource": OpenMeteoSingleRunsSource}


def parse_locations(locations: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Normalise locations to [{"name", "latitude", "longitude", "weight"}].

    Accepts a list of dicts or the compact string "name:lat:lon:weight, name:lat:lon:weight".
    """
    if isinstance(locations, str):
        parsed = []
        for item in filter(None, (part.strip() for part in locations.split(","))):
            parts = item.split(":")
            if len(parts) != 4:
                raise ValueError(f'location {item!r} must be "name:latitude:longitude:weight"')
            name, lat, lon, weight = parts
            try:
                parsed.append(
                    {"name": name.strip(), "latitude": float(lat), "longitude": float(lon), "weight": float(weight)}
                )
            except ValueError:
                raise ValueError(f'location {item!r} must be "name:latitude:longitude:weight" with numbers')
        locations = parsed
    if not isinstance(locations, list) or not locations:
        raise ValueError("locations must be a non-empty list of {name, latitude, longitude, weight}")

    normalised, names = [], set()
    for location in locations:
        if not isinstance(location, dict) or set(location) != {"name", "latitude", "longitude", "weight"}:
            raise ValueError(f"each location needs exactly name, latitude, longitude and weight, got {location!r}")
        name, weight = location["name"], location["weight"]
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise ValueError(f"location name must use letters, digits and underscores, got {name!r}")
        if name in names:
            raise ValueError(f"duplicate location name {name!r}")
        names.add(name)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight <= 0:
            raise ValueError(f"weight of {name!r} must be a positive number, got {weight!r}")
        normalised.append(dict(location))
    return normalised


class OpenMeteoWeightedSource(DataSource):
    """
    Weighted average of Open-Meteo forecasts over several locations, e.g. population-weighted temperature or
    capacity-weighted wind speed for a bidding zone.

    Every location is fetched by the same underlying source (``OpenMeteoSource`` or ``OpenMeteoSingleRunsSource``)
    with the same ``source_params``, so all locations follow the same vintage rule. For each output column
    ``{prefix}_{variable}_d+{i}`` the source returns sum(w_l * x_l) / sum(w_l) over the locations; a row is missing
    if any location is missing (weights are never re-normalised over the available ones). Wind directions use a
    weighted circular mean; ``weather_code`` is categorical and is only kept per location.

    Weights are fixed parameters and apply to the whole sample, so they must be values known at every issue day D:
    use capacities or populations of a fixed reference date before the sample starts, not end-of-sample figures.

    Example:
        >>> source = OpenMeteoWeightedSource(
        ...     locations=[
        ...         {"name": "warsaw", "latitude": 52.23, "longitude": 21.01, "weight": 1.86},
        ...         {"name": "krakow", "latitude": 50.06, "longitude": 19.94, "weight": 0.80},
        ...     ],
        ...     source_params={"model": "icon_seamless", "horizon": 5, "columns": ["temperature_2m"]},
        ...     prefix="pl",
        ... )
    """

    VALID_SOURCES = ("OpenMeteoSource", "OpenMeteoSingleRunsSource")
    SCHEMA_VERSION = 1

    def __init__(
        self,
        locations: Union[str, List[Dict[str, Any]]],
        source: str = "OpenMeteoSource",
        source_params: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_locations: bool = False,
    ):
        """
        Args:
            locations: List of {"name", "latitude", "longitude", "weight"} dicts, or the string
                "name:lat:lon:weight, ...". Names become column prefixes with ``keep_locations``.
            source: "OpenMeteoSource" (Previous Runs API) or "OpenMeteoSingleRunsSource" (Single Runs API)
            source_params: Parameters for the underlying source, shared by all locations (anything but latitude,
                longitude and prefix), e.g. {"model": "icon_seamless", "horizon": 5, "derived": ["hdd"]}.
            prefix: Prefix for the weighted columns
            keep_locations: Also return each location's columns as ``{prefix}_{name}_{variable}_d+{i}``
        """
        self.locations = locations
        self.source = source
        self.source_params = dict(source_params) if source_params else {}
        self.prefix = prefix
        self.keep_locations = keep_locations

        self.logger = get_logger("epftoolbox2.data.sources.open_meteo")
        self._locations: List[Dict[str, Any]] = []
        self._sources: List[DataSource] = []
        self._validate_config()

    def _validate_config(self) -> bool:
        self._locations = parse_locations(self.locations)
        if self.source not in self.VALID_SOURCES:
            raise ValueError(f"source must be one of {self.VALID_SOURCES}, got {self.source!r}")
        reserved = {"latitude", "longitude", "prefix"} & set(self.source_params)
        if reserved:
            raise ValueError(f"source_params must not set {sorted(reserved)}; they come from locations")
        if not isinstance(self.keep_locations, bool):
            raise ValueError(f"keep_locations must be True or False, got {self.keep_locations!r}")

        cls = _source_classes()[self.source]
        self._sources = [
            cls(latitude=loc["latitude"], longitude=loc["longitude"], **self.source_params) for loc in self._locations
        ]
        return True

    @property
    def _weights(self) -> np.ndarray:
        weights = np.array([loc["weight"] for loc in self._locations], dtype=float)
        return weights / weights.sum()

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Fetch every location with the underlying source and return the weighted columns for [start, end)."""
        frames = []
        for location, source in zip(self._locations, self._sources):
            self.logger.info(f"Open-Meteo weighted: location {location['name']} (weight {location['weight']})")
            frames.append(source.fetch(start, end))
        if all(frame.empty for frame in frames):
            return pd.DataFrame()

        index = frames[0].index
        for frame in frames[1:]:
            index = index.union(frame.index)
        frames = [frame.reindex(index) for frame in frames]
        columns = [c for c in frames[0].columns]

        weights = self._weights
        weighted = {}
        for column in columns:
            variable = column.rsplit("_d+", 1)[0]
            if variable in _CATEGORICAL:
                continue
            stack = np.column_stack([frame[column].to_numpy(dtype=float) for frame in frames])
            if _CIRCULAR.search(variable):
                radians = np.deg2rad(stack)
                sin = (np.sin(radians) * weights).sum(axis=1)
                cos = (np.cos(radians) * weights).sum(axis=1)
                weighted[column] = np.mod(np.rad2deg(np.arctan2(sin, cos)), 360.0)
            else:
                weighted[column] = (stack * weights).sum(axis=1)
            # NaN at any location makes the weighted value NaN (sum() above would treat it as 0 otherwise).
            weighted[column][np.isnan(stack).any(axis=1)] = np.nan
        skipped = [c for c in columns if c.rsplit("_d+", 1)[0] in _CATEGORICAL]
        if skipped and not self.keep_locations:
            self.logger.info(
                f"Open-Meteo weighted: {', '.join(sorted(set(c.rsplit('_d+', 1)[0] for c in skipped)))} "
                "is categorical and is not averaged; use keep_locations=True to keep it per location"
            )

        result = pd.DataFrame(weighted, index=index)
        if self.prefix:
            result.columns = [f"{self.prefix}_{col}" for col in result.columns]
        if self.keep_locations:
            base = f"{self.prefix}_" if self.prefix else ""
            per_location = [
                frame.add_prefix(f"{base}{location['name']}_") for location, frame in zip(self._locations, frames)
            ]
            result = pd.concat([result, *per_location], axis=1)

        filled = result.notna().any(axis=1).to_numpy()
        keep = int(np.flatnonzero(filled)[-1]) + 1 if filled.any() else 0
        result = result.iloc[:keep]
        result.index.name = "datetime"
        return result

    def get_cache_config(self) -> dict:
        inner = self._sources[0].get_cache_config()
        for key in ("latitude", "longitude", "prefix"):
            inner.pop(key, None)
        return {
            "source_type": "open_meteo_weighted",
            "schema_version": self.SCHEMA_VERSION,
            "locations": [
                {key: loc[key] for key in ("name", "latitude", "longitude", "weight")} for loc in self._locations
            ],
            "source": self.source,
            "source_config": inner,
            "prefix": self.prefix,
            "keep_locations": self.keep_locations,
        }
