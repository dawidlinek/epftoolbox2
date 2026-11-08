"""EnergyChartsSource against the live Energy-Charts API (no key). Tiny ranges, and the source's own throttle keeps within the
public limit of 2 requests a minute per endpoint (burst 4, /price burst 2), so the whole module takes a few minutes at most.
Run with `uv run pytest -m integration tests/data/sources/test_energy_charts_live.py`."""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.sources import EnergyChartsSource

pytestmark = pytest.mark.integration

AUTUMN_DAY = (pd.Timestamp("2025-10-25 22:00", tz="UTC"), pd.Timestamp("2025-10-26 23:00", tz="UTC"))  # 25 h
SPRING_DAY = (pd.Timestamp("2025-03-29 23:00", tz="UTC"), pd.Timestamp("2025-03-30 22:00", tz="UTC"))  # 23 h


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("energy-charts"))


@pytest.fixture(scope="module")
def spring_forecasts(cache_dir):
    source = EnergyChartsSource(
        "de",
        type=["forecast", "power"],
        forecast_types=["day-ahead", "intraday", "current"],
        production_types=["solar"],
        power=["solar"],
        cache_dir=cache_dir,
        prefix="",
    )
    return source.fetch(*SPRING_DAY)


def test_forecast_stamps_mark_the_start_of_the_period_on_a_spring_day(spring_forecasts):
    df = spring_forecasts
    assert len(df) == 92  # 23 h at 15 min
    assert df.index[0] == SPRING_DAY[0] and df.index[-1] == SPRING_DAY[1] - pd.Timedelta(minutes=15)
    local = df.index.tz_convert("Europe/Berlin")
    assert local[0].hour == 0 and 2 not in set(local.hour)


def test_the_three_forecast_types_are_three_series(spring_forecasts):
    df = spring_forecasts
    assert df[["solar_forecast_day_ahead", "solar_forecast_intraday", "solar_forecast_current"]].notna().all().all()
    day_ahead, intraday, current = (df[f"solar_forecast_{v}"] for v in ("day_ahead", "intraday", "current"))
    assert not np.allclose(day_ahead, intraday) and not np.allclose(intraday, current)
    actual = df["actual_solar"]
    rmse = {
        name: float(np.sqrt(((series - actual) ** 2).mean()))
        for name, series in [("day_ahead", day_ahead), ("intraday", intraday), ("current", current)]
    }
    assert rmse["day_ahead"] > rmse["intraday"] > rmse["current"]  # the ordering the vintage names rest on


def test_price_on_the_autumn_switch_day_is_15_minute_for_de_lu_and_hourly_for_ch(cache_dir):
    de = EnergyChartsSource("de", type=["price"], cache_dir=cache_dir, prefix="").fetch(*AUTUMN_DAY)
    assert len(de) == 100 and de["price_day_ahead"].notna().all()
    assert de.index.to_series().diff().dropna().eq(pd.Timedelta(minutes=15)).all()
    ch = EnergyChartsSource("ch", type=["price"], cache_dir=cache_dir, prefix="").fetch(*AUTUMN_DAY)
    assert len(ch) == 25
    assert list(ch.index.tz_convert("Europe/Berlin").hour).count(2) == 2


def test_a_sub_range_matches_the_larger_range(cache_dir):
    source = EnergyChartsSource("de", type=["price"], use_cache=False, prefix="")
    whole = source.fetch(*AUTUMN_DAY)
    start = AUTUMN_DAY[0] + pd.Timedelta(hours=6)
    part = source.fetch(start, start + pd.Timedelta(hours=3))
    pd.testing.assert_frame_equal(part, whole.loc[part.index])


def test_a_restricted_zone_is_refused_before_any_request():
    with pytest.raises(ValueError, match="private and internal use"):
        EnergyChartsSource("de", type=["price"], bzn="RO")
