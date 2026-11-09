"""The sun's position at the middle of each period, a clear-sky irradiance and a night flag, from the rows' instants."""

import math
import numbers

import numpy as np
import pandas as pd

from ._daily import check_new_columns, check_spacing, parse_step, slot_of_day
from .base import Transformer

# The columns it adds (before `prefix`). `information_set` gives each a static rule (and `*_<name>` for a prefixed one).
COLUMNS = ("sun_elevation", "sun_sin_elevation", "clear_sky_ghi", "is_night")
ALGORITHMS = ("meeus", "spencer")
_DAY_NS = 86_400 * 10**9
# Days from the Unix epoch to J2000.0 (Julian day 2451545.0; the epoch is Julian day 2440587.5).
_J2000_DAYS = 10_957.5


def _pairs(locations) -> list:
    """`locations` as a list of [latitude, longitude] lists: one `(lat, lon)` pair or a list of them."""
    if isinstance(locations, (str, bytes)) or not hasattr(locations, "__iter__"):
        raise ValueError(f"locations must be (lat, lon) pairs, got {locations!r}")
    items = list(locations)
    if len(items) == 2 and all(isinstance(v, numbers.Real) and not isinstance(v, bool) for v in items):
        items = [items]
    if not items:
        raise ValueError("locations must hold at least one (lat, lon) pair")
    out = []
    for pair in items:
        values = list(pair) if hasattr(pair, "__iter__") and not isinstance(pair, (str, bytes)) else [pair]
        if len(values) != 2 or not all(isinstance(v, numbers.Real) and not isinstance(v, bool) for v in values):
            raise ValueError(f"each location is a (lat, lon) pair of numbers, got {pair!r}")
        lat, lon = float(values[0]), float(values[1])
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            raise ValueError(f"a location's latitude is in [-90, 90] and its longitude in [-180, 180], got {pair!r}")
        out.append([lat, lon])
    return out


def _sine_meeus(ns: np.ndarray, lat: float, lon: float) -> np.ndarray:
    """Sine of the sun's geometric elevation at UTC instants (nanoseconds since the epoch): the NOAA Solar Calculator's
    equations (Meeus, Astronomical Algorithms), as `astral.sun.elevation(..., with_refraction=False)` computes them."""
    t = (ns / _DAY_NS - _J2000_DAYS) / 36525.0  # Julian centuries since J2000.0
    l0 = np.mod(280.46646 + t * (36000.76983 + 0.0003032 * t), 360.0)  # geometric mean longitude, degrees
    m = np.radians(357.52911 + t * (35999.05029 - 0.0001537 * t))  # geometric mean anomaly
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)  # eccentricity of the Earth's orbit
    centre = (
        np.sin(m) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + np.sin(2.0 * m) * (0.019993 - 0.000101 * t)
        + np.sin(3.0 * m) * 0.000289
    )
    omega = np.radians(125.04 - 1934.136 * t)
    apparent = np.radians(l0 + centre - 0.00569 - 0.00478 * np.sin(omega))  # apparent longitude
    seconds = 21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))
    obliquity = np.radians(23.0 + (26.0 + seconds / 60.0) / 60.0 + 0.00256 * np.cos(omega))
    declination = np.arcsin(np.sin(obliquity) * np.sin(apparent))
    y = np.tan(obliquity / 2.0) ** 2
    l0r = np.radians(l0)
    equation_of_time = 4.0 * np.degrees(
        y * np.sin(2.0 * l0r)
        - 2.0 * e * np.sin(m)
        + 4.0 * e * y * np.sin(m) * np.cos(2.0 * l0r)
        - 0.5 * y * y * np.sin(4.0 * l0r)
        - 1.25 * e * e * np.sin(2.0 * m)
    )  # minutes
    return _sine(ns, lat, lon, declination, equation_of_time)


def _sine_spencer(ns: np.ndarray, lat: float, lon: float) -> np.ndarray:
    """Sine of the sun's elevation from NOAA's General Solar Position Calculations: the fractional year's Fourier series
    (Spencer 1971) for the declination and the equation of time, a 365-day year in every year, as the renewables study."""
    day_of_year = pd.DatetimeIndex(ns.astype("datetime64[ns]")).dayofyear.to_numpy()
    hours = np.mod(ns, _DAY_NS) / 3.6e12
    g = 2.0 * np.pi / 365.0 * (day_of_year - 1 + (hours - 12.0) / 24.0)
    declination = (
        0.006918
        - 0.399912 * np.cos(g)
        + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2.0 * g)
        + 0.000907 * np.sin(2.0 * g)
        - 0.002697 * np.cos(3.0 * g)
        + 0.00148 * np.sin(3.0 * g)
    )
    equation_of_time = 229.18 * (
        0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2.0 * g) - 0.040849 * np.sin(2.0 * g)
    )
    return _sine(ns, lat, lon, declination, equation_of_time)


def _sine(ns: np.ndarray, lat: float, lon: float, declination: np.ndarray, equation_of_time: np.ndarray) -> np.ndarray:
    """The sine of the elevation from the declination (radians) and the equation of time (minutes) at UTC instants."""
    true_solar_time = np.mod(ns, _DAY_NS) / 6e10 + equation_of_time + 4.0 * lon  # minutes
    hour_angle = np.radians(true_solar_time / 4.0 - 180.0)
    phi = math.radians(lat)
    return math.sin(phi) * np.sin(declination) + math.cos(phi) * np.cos(declination) * np.cos(hour_angle)


_SINE = {"meeus": _sine_meeus, "spencer": _sine_spencer}


def haurwitz(sine: np.ndarray) -> np.ndarray:
    """Clear-sky global horizontal irradiance in W/m² (Haurwitz 1945): 1098 cos z exp(-0.059 / cos z) with cos z the sine of
    the elevation, 0 where the sun is at or below the horizon."""
    sine = np.asarray(sine, dtype=float)
    up = sine > 0.0
    out = np.zeros(sine.shape)
    out[up] = 1098.0 * sine[up] * np.exp(-0.059 / sine[up])
    return out


class SolarGeometryTransformer(Transformer):
    """The sun's elevation at the middle of each period, its sine, a clear-sky irradiance and a night flag.

    Adds `sun_elevation` (degrees, geometric: no refraction), `sun_sin_elevation`, `clear_sky_ghi` (W/m², Haurwitz) and
    `is_night` (1 where the sine of the elevation is below `night_below`, else 0). Each is computed at the middle of the
    row's period, the row's instant plus half of `freq`, read as a UTC instant: so a clock change can't shift it, and the
    two rows of the repeated autumn hour, an hour apart, get their own values. With several `locations` each column is the
    weighted mean of the sites' values, and `is_night` reads the mean sine.

    They depend on nothing but the row's instant, so they are known at any time: the built-in information-set rules make
    them `static`, unshifted and as `_d+k` alike, and `is_night_d+{horizon}` can be a model's `zero_when`.

    Args:
        locations: A `(latitude, longitude)` pair in degrees, or a list of them (north and east positive).
        weights: One positive weight per location (normalised to sum 1), or None for equal weights.
        freq: Period length of the rows ("1h", "15min"): each value is at the row's instant plus half of it. Shorter than
            a day.
        prefix: Prepended to every column name with an underscore (`"de"` gives `de_sun_elevation`), for sites of
            several zones in one frame. Default none.
        night_below: The sine of the elevation below which a row is night. Default -0.05 (about -2.9 degrees, the
            renewables study's threshold for a solar forecast of 0).
        algorithm: "meeus" (default), the NOAA Solar Calculator's equations (Meeus), as `astral` computes them; or
            "spencer", NOAA's General Solar Position Calculations (a Fourier series in the fractional year), as the
            renewables study computed its elevation, up to about 0.5 degrees off near the equinoxes.

    Example:
        >>> df = SolarGeometryTransformer(locations=[(51.0, 10.0)]).transform(df)
    """

    def __init__(
        self,
        locations,
        weights=None,
        freq: str = "1h",
        prefix: str = "",
        night_below: float = -0.05,
        algorithm: str = "meeus",
    ):
        self.locations = _pairs(locations)
        if weights is None:
            self.weights = None
        else:
            values = list(weights)
            if len(values) != len(self.locations) or not all(
                isinstance(w, numbers.Real) and not isinstance(w, bool) and math.isfinite(w) and w > 0 for w in values
            ):
                raise ValueError(
                    f"weights must be one positive number per location ({len(self.locations)}), got {weights!r}"
                )
            self.weights = [float(w) for w in values]
        self.freq = freq
        self._step = parse_step(freq)
        if self._step >= pd.Timedelta(days=1):
            raise ValueError(
                f"freq must be shorter than a day: the sun's position is at the middle of a period, got {freq!r}"
            )
        if not isinstance(prefix, str):
            raise ValueError(f"prefix must be a string, got {prefix!r}")
        self.prefix = prefix
        if isinstance(night_below, bool) or not isinstance(night_below, numbers.Real) or not -1.0 <= night_below <= 1.0:
            raise ValueError(f"night_below is a sine of the elevation, in [-1, 1], got {night_below!r}")
        self.night_below = float(night_below)
        if algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {list(ALGORITHMS)}, got {algorithm!r}")
        self.algorithm = algorithm

    def column_names(self) -> list:
        """The names of the columns it adds, in order."""
        return [f"{self.prefix}_{name}" if self.prefix else name for name in COLUMNS]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        index = df.index
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("SolarGeometryTransformer needs a DatetimeIndex")
        if index.tz is None:
            raise ValueError(
                "SolarGeometryTransformer needs a time-zone-aware index: the sun's position is read at each row's UTC "
                "instant, which a naive wall clock doesn't give. Run it after TimezoneTransformer and before "
                "DSTNormalizeTransformer."
            )
        slot_of_day(index, self._step, self.freq)  # rejects data off the freq grid
        check_spacing(index, self._step, self.freq, type(self).__name__)  # and data further apart than freq
        names = self.column_names()
        check_new_columns(df, names, type(self).__name__)

        middle = index.tz_convert("UTC").tz_localize(None).as_unit("ns") + self._step / 2
        ns = middle.asi8  # int64 nanoseconds since the epoch, UTC
        weights = np.ones(len(self.locations)) if self.weights is None else np.asarray(self.weights)
        weights = weights / weights.sum()
        sine = np.zeros(len(index))
        elevation = np.zeros(len(index))
        ghi = np.zeros(len(index))
        for (lat, lon), weight in zip(self.locations, weights):
            one = np.clip(_SINE[self.algorithm](ns, lat, lon), -1.0, 1.0)
            sine += weight * one
            elevation += weight * np.degrees(np.arcsin(one))
            ghi += weight * haurwitz(one)
        values = {
            names[0]: elevation,
            names[1]: sine,
            names[2]: ghi,
            names[3]: (sine < self.night_below).astype(float),
        }
        return pd.concat([df, pd.DataFrame(values, index=index)], axis=1)
