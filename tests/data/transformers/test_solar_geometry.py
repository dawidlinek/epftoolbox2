"""SolarGeometryTransformer (step 7e): the sun's elevation at the middle of each period against astral and against the
renewables study's own formula, at several latitudes, over both clock changes, hourly and every 15 minutes."""

import math

import numpy as np
import pandas as pd
import pytest
from astral import Observer
from astral.sun import elevation as astral_elevation

from epftoolbox2.data.transformers import SolarGeometryTransformer
from epftoolbox2.data.transformers.solar_geometry import COLUMNS, haurwitz
from epftoolbox2.information_set import _SOLAR_COLUMNS
from epftoolbox2.pipelines import DataPipeline

# Far below the spec's 0.01 degrees: the transformer computes astral's own equations, so a dropped term shows.
ASTRAL_DEGREES = 1e-6
STUDY = 1e-9
SWITCHES = {"Europe/Warsaw": ("2024-03-31", "2024-10-27"), "Europe/Lisbon": ("2025-03-30", "2025-10-26")}


def astral_at(instants, lat, lon):
    return np.array([astral_elevation(Observer(lat, lon), t.to_pydatetime(), with_refraction=False) for t in instants])


def frame(index):
    return pd.DataFrame({"x": np.arange(len(index), dtype=float)}, index=index)


def switch_rows(zone, freq):
    """Every row of both switch days of `zone`: the spring day without the hour the clocks skip, the autumn day with the
    repeated hour twice."""
    spring, autumn = SWITCHES[zone]
    parts = [
        pd.date_range(day, pd.Timestamp(day) + pd.Timedelta(days=1), freq=freq, tz=zone, inclusive="left")
        for day in (spring, autumn)
    ]
    return parts[0].append(parts[1])


@pytest.mark.parametrize("freq", ["1h", "15min"])
def test_elevation_is_astral_s_at_the_middle_of_each_period_on_random_instants_and_places(freq):
    """1,000 (instant, place) pairs: 40 places from 60 S to 70 N and 180 W to 180 E, each with 25 random rows of 2000-2040 (and
    one neighbour) on
    the freq grid of a zone (Warsaw, Lisbon, Helsinki or UTC), and every row of both switch days of Warsaw or Lisbon. Each row
    is read at its UTC instant plus half a period."""
    rng = np.random.default_rng(7)
    step = pd.Timedelta(freq)
    worst = 0.0
    for place in range(40):
        lat, lon = rng.uniform(-60, 70), rng.uniform(-180, 180)
        zone = ["Europe/Warsaw", "Europe/Lisbon", "Europe/Helsinki", "UTC"][place % 4]
        start = pd.Timestamp("2000-01-01", tz="UTC").value // step.value
        stop = pd.Timestamp("2040-01-01", tz="UTC").value // step.value
        steps = rng.integers(start, stop, 25)
        index = pd.DatetimeIndex(pd.to_datetime(steps * step.value, utc=True)).tz_convert(zone)
        index = index.append(index[:1] + step)  # two neighbouring rows: the data is at `freq`
        if zone in SWITCHES:
            index = index.append(switch_rows(zone, freq))
        index = index.sort_values().unique()
        out = SolarGeometryTransformer([(lat, lon)], freq=freq).transform(frame(index))
        expected = astral_at(index + step / 2, lat, lon)
        worst = max(worst, float(np.abs(out["sun_elevation"].to_numpy() - expected).max()))
        np.testing.assert_allclose(out["sun_sin_elevation"], np.sin(np.radians(expected)), rtol=0, atol=1e-8)
    assert worst < ASTRAL_DEGREES


@pytest.mark.parametrize("zone", sorted(SWITCHES))
def test_each_row_of_a_switch_day_has_the_sun_of_its_own_instant(zone):
    """The two rows of the repeated autumn hour are an hour apart, and each is the sun at its own instant plus 30 minutes;
    the spring day has no 02:00 row, and its 03:00 row is the sun at 03:30 summer time. A 24-hour offset from the day before
    would put the sun an hour off on every row after the switch."""
    spring, autumn = SWITCHES[zone]
    out = SolarGeometryTransformer([(50.0, 15.0)]).transform(frame(switch_rows(zone, "1h")))
    autumn_rows = out.loc[autumn]
    assert len(out.loc[spring]) == 23 and len(autumn_rows) == 25
    repeated = autumn_rows[autumn_rows.index.hour == (2 if zone == "Europe/Warsaw" else 1)]
    assert len(repeated) == 2 and repeated["sun_elevation"].iloc[0] != repeated["sun_elevation"].iloc[1]
    np.testing.assert_allclose(
        repeated["sun_elevation"], astral_at(repeated.index + pd.Timedelta(minutes=30), 50.0, 15.0), atol=ASTRAL_DEGREES
    )
    later = out.loc[f"{spring} 12:00"]
    assert later["sun_elevation"] == pytest.approx(
        astral_at([pd.Timestamp(f"{spring} 12:30", tz=zone)], 50.0, 15.0)[0], abs=ASTRAL_DEGREES
    )


def study_sun_sin_elev(days, tz, lat, lon):
    """The renewables study's `build_data.sun_sin_elev`, verbatim: sin(solar elevation) at the middle of each local wall-clock
    hour of each day (n x 24)."""
    out = np.zeros((len(days), 24))
    for h in range(24):
        wall = days + pd.Timedelta(hours=h, minutes=30)
        t = wall.tz_localize(tz, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")
        doy = t.dayofyear.to_numpy()
        utc_h = (t.hour + t.minute / 60).to_numpy()
        g = 2 * np.pi / 365 * (doy - 1 + (utc_h - 12) / 24)
        decl = (
            0.006918
            - 0.399912 * np.cos(g)
            + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g)
            + 0.00148 * np.sin(3 * g)
        )
        eqt = 229.18 * (
            0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g)
        )
        tst = utc_h * 60 + eqt + 4 * lon
        ha = np.deg2rad(tst / 4 - 180)
        la = np.deg2rad(lat)
        out[:, h] = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    return out


@pytest.mark.parametrize(
    "zone, lat, lon", [("Europe/Berlin", 51.0, 10.0), ("Europe/Vienna", 47.8, 14.5)], ids=["DE_LU", "AT"]
)
def test_spencer_is_the_renewables_study_s_sine_on_its_sites(zone, lat, lon):
    """`algorithm="spencer"` gives the study's `SINEL` on its two sites over its whole span, 2023-07-01 to 2026-10-10, within
    1e-9 on every row the study has: not the hour the spring switch skips (the study reads it at 03:00) nor the second of the
    repeated autumn hour (the study reads the first)."""
    days = pd.date_range("2023-07-01", "2026-10-10", freq="D")
    grid = study_sun_sin_elev(days, zone, lat, lon)
    index = pd.date_range(days[0], days[-1] + pd.Timedelta(days=1), freq="1h", tz=zone, inclusive="left")
    out = SolarGeometryTransformer([(lat, lon)], algorithm="spencer").transform(frame(index))
    wall = index.tz_localize(None)
    first = ~pd.Series(wall).duplicated(keep="first").to_numpy()
    rows = np.flatnonzero(first)
    study = grid[days.get_indexer(wall.normalize()[rows]), wall.hour[rows]]
    assert np.abs(out["sun_sin_elevation"].to_numpy()[rows] - study).max() < STUDY
    assert len(rows) == len(days) * 24 - 3  # every hour of the study's grid but the three the spring switches skip


def test_the_two_algorithms_differ_by_the_spencer_series_error():
    """The fractional-year series is up to about half a degree off near the equinoxes, so the default is the NOAA Solar
    Calculator's equations; the study's formula is kept to reproduce its numbers."""
    index = pd.date_range("2024-03-15", "2024-03-25", freq="1h", tz="UTC", inclusive="left")
    meeus = SolarGeometryTransformer([(51.0, 10.0)]).transform(frame(index))["sun_elevation"]
    spencer = SolarGeometryTransformer([(51.0, 10.0)], algorithm="spencer").transform(frame(index))["sun_elevation"]
    gap = float((meeus - spencer).abs().max())
    assert 0.05 < gap < 0.6


def test_clear_sky_ghi_is_haurwitz_and_0_with_the_sun_down():
    index = pd.date_range("2024-06-21", "2024-06-22", freq="1h", tz="Europe/Berlin", inclusive="left")
    out = SolarGeometryTransformer([(51.0, 10.0)]).transform(frame(index))
    sine = out["sun_sin_elevation"].to_numpy()
    expected = np.where(sine > 0, 1098.0 * sine * np.exp(-0.059 / np.where(sine > 0, sine, 1.0)), 0.0)
    np.testing.assert_allclose(out["clear_sky_ghi"], expected, rtol=1e-14, atol=0)
    assert (out["clear_sky_ghi"][sine <= 0] == 0).all() and (out["clear_sky_ghi"][sine > 0] > 0).all()
    assert haurwitz(np.array([1.0]))[0] == pytest.approx(1098.0 * math.exp(-0.059))


@pytest.mark.parametrize("night_below", [-0.05, 0.0, 0.2])
def test_is_night_is_1_where_the_sine_is_below_the_threshold(night_below):
    index = pd.date_range("2024-01-10", "2024-01-12", freq="15min", tz="Europe/Warsaw", inclusive="left")
    transformer = SolarGeometryTransformer([(52.2, 21.0)], freq="15min", night_below=night_below)
    out = transformer.transform(frame(index))
    assert out["is_night"].isin([0.0, 1.0]).all()
    assert (out["is_night"] == (out["sun_sin_elevation"] < night_below)).all()
    assert out["is_night"].sum() > 0 and (out["is_night"] == 0).sum() > 0


def test_several_locations_give_the_weighted_mean_and_the_night_of_the_mean_sine():
    index = pd.date_range("2024-05-01", "2024-05-03", freq="1h", tz="Europe/Berlin", inclusive="left")
    sites = [(54.0, 9.0), (48.0, 12.0)]
    alone = [SolarGeometryTransformer([site]).transform(frame(index)) for site in sites]
    both = SolarGeometryTransformer(sites, weights=[3, 1]).transform(frame(index))
    for column in ("sun_elevation", "sun_sin_elevation", "clear_sky_ghi"):
        np.testing.assert_allclose(both[column], 0.75 * alone[0][column] + 0.25 * alone[1][column], rtol=1e-12)
    assert (both["is_night"] == (both["sun_sin_elevation"] < -0.05)).all()
    equal = SolarGeometryTransformer(sites).transform(frame(index))
    np.testing.assert_allclose(
        equal["sun_sin_elevation"], (alone[0]["sun_sin_elevation"] + alone[1]["sun_sin_elevation"]) / 2
    )


def test_a_prefix_names_the_columns_and_one_pair_is_a_location():
    index = pd.date_range("2024-05-01", periods=4, freq="1h", tz="Europe/Vienna")
    transformer = SolarGeometryTransformer((47.8, 14.5), prefix="at")
    assert transformer.locations == [[47.8, 14.5]]
    out = transformer.transform(frame(index))
    assert list(out.columns) == ["x", "at_sun_elevation", "at_sun_sin_elevation", "at_clear_sky_ghi", "at_is_night"]
    assert transformer.column_names() == [f"at_{name}" for name in COLUMNS]


def test_its_columns_are_the_ones_the_information_set_has_static_rules_for():
    assert tuple(COLUMNS) == _SOLAR_COLUMNS


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"locations": []}, "at least one"),
        ({"locations": [(91.0, 0.0)]}, "latitude"),
        ({"locations": [(50.0, 200.0)]}, "latitude"),
        ({"locations": [(50.0,)]}, "pair"),
        ({"locations": "51,10"}, "pairs"),
        ({"locations": [(50.0, 10.0)], "weights": [1, 2]}, "one positive number per location"),
        ({"locations": [(50.0, 10.0)], "weights": [0]}, "one positive number per location"),
        ({"locations": [(50.0, 10.0)], "freq": "1D"}, "shorter than a day"),
        ({"locations": [(50.0, 10.0)], "freq": "7min"}, "whole slots"),
        ({"locations": [(50.0, 10.0)], "night_below": 2.0}, "night_below"),
        ({"locations": [(50.0, 10.0)], "algorithm": "astral"}, "algorithm"),
    ],
)
def test_values_that_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SolarGeometryTransformer(**kwargs)


def test_frames_that_are_refused():
    transformer = SolarGeometryTransformer([(50.0, 10.0)])
    naive = pd.date_range("2024-01-01", periods=3, freq="1h")
    with pytest.raises(ValueError, match="time-zone-aware"):
        transformer.transform(frame(naive))
    quarter = pd.date_range("2024-01-01", periods=8, freq="15min", tz="Europe/Warsaw")
    with pytest.raises(ValueError, match="grid"):
        transformer.transform(frame(quarter))
    done = transformer.transform(frame(pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")))
    with pytest.raises(ValueError, match="already exist"):
        transformer.transform(done)


@pytest.mark.parametrize("freq", ["15min", "30min"])
def test_rows_further_apart_than_freq_are_refused(freq):
    """Hourly rows read with `freq="15min"` would be the sun 7.5 minutes into each hour, not 30: refused."""
    index = pd.date_range("2024-06-01", periods=48, freq="1h", tz="Europe/Berlin")
    with pytest.raises(ValueError, match="no two rows are closer"):
        SolarGeometryTransformer([(51.0, 10.0)], freq=freq).transform(frame(index))
    # A gap in hourly data at its own freq is no reason to refuse.
    SolarGeometryTransformer([(51.0, 10.0)]).transform(frame(index.delete([3, 4, 5])))


def test_a_utc_index_gives_the_same_values_as_a_local_one():
    """The values are of the rows' instants: the index's zone only labels them."""
    local = pd.date_range("2024-10-26", "2024-10-28", freq="1h", tz="Europe/Warsaw", inclusive="left")
    transformer = SolarGeometryTransformer([(52.0, 19.0)])
    a = transformer.transform(frame(local))
    b = transformer.transform(frame(local.tz_convert("UTC")))
    np.testing.assert_array_equal(a.drop(columns="x").to_numpy(), b.drop(columns="x").to_numpy())


def test_yaml_round_trip(tmp_path):
    pipeline = DataPipeline().add_transformer(
        SolarGeometryTransformer(
            [(51.0, 10.0), (48.0, 11.5)],
            weights=[2.0, 1.0],
            freq="15min",
            prefix="de",
            night_below=0.0,
            algorithm="spencer",
        )
    )
    path = tmp_path / "solar.yaml"
    pipeline.save(path)
    loaded = DataPipeline.load(path).transformers[0]
    assert isinstance(loaded, SolarGeometryTransformer)
    assert vars(loaded) == vars(pipeline.transformers[0])
