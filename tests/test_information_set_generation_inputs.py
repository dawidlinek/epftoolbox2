"""The information set and the generation inputs of step 7e: built-in static rules for SolarGeometryTransformer's columns and
ENTSO-E's installed capacity, and a RollingStatsTransformer column judged through the column it was built from at the last slot
of its window's newest day, every day of the window under rules over time."""

import warnings

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.sources.entsoe import EntsoeSource
from epftoolbox2.data.transformers import LagTransformer, RollingStatsTransformer
from epftoolbox2.information_set import (
    Availability,
    InformationSet,
    InformationSetError,
    perturb_unknown,
    source_rules,
)
from tests.test_information_set import _refused_periods, _verdicts_against_held_deliveries, expand, violations

SOLAR = ["sun_elevation", "sun_sin_elevation", "clear_sky_ghi", "is_night"]


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prefix", ["", "de_"])
def test_solar_geometry_columns_are_static_at_any_shift(prefix):
    """A function of the row's instant: known at any time, so `is_night_d+1` can be a model's `zero_when` and a lead of a week
    is as known as the column itself."""
    info = InformationSet()
    names = [f"{prefix}{name}{shift}" for name in SOLAR for shift in ("", "_d+1", "_d+7", "_d-1", "_h+30")]
    for name in names:
        col = info.resolve(name)
        assert col.rule is not None and col.rule.kind == "static", name
    assert violations(names, horizon=7) == (set(), set())


def test_no_pattern_gives_another_sun_column_a_rule():
    """Exact names and `*_<name>`: a series of yours named `sun_power` or `pv_sun_forecast` gets no rule from them."""
    info = InformationSet()
    for name in ["sun_power", "pv_sun_forecast_d+1", "is_nightly", "sun_elevation_max", "clear_sky"]:
        assert info.resolve(name).rule is None, name


def test_installed_capacity_is_static_with_its_archive_s_newest_revision():
    """Decision 8: A68 is due a week before the year it is for begins, so every row holds a value out before its year did."""
    info = InformationSet()
    for name in ["installed_capacity_solar", "installed_capacity_wind_onshore_d+1", "installed_capacity_solar_d+7"]:
        col = info.resolve(name)
        assert (col.key, col.rule.kind, col.rule.vintage) == ("installed_capacity_*", "static", "latest_revision")
    assert violations(["installed_capacity_solar_d+{horizon}"], horizon=7) == (set(), set())
    # A plain `installed_capacity` (no production type) is no ENTSO-E column: no rule.
    assert info.resolve("installed_capacity").rule is None


def test_a_prefixed_entsoe_source_s_capacity_gets_the_rule_from_its_source():
    prefixed = EntsoeSource("PL", "key", type=["installed_capacity"], prefix="pl")
    rules = source_rules([prefixed])
    assert rules["pl_installed_capacity_*"].kind == "static"
    assert rules["pl_installed_capacity_*"].vintage == "latest_revision"
    info = InformationSet().with_source_rules([prefixed])
    assert info.resolve("pl_installed_capacity_solar_d+1").rule.kind == "static"
    assert InformationSet().resolve("pl_installed_capacity_solar").rule is None  # outside a source-aware run, no rule
    # A source without the type, or without a prefix, adds nothing.
    assert "installed_capacity_*" not in source_rules([EntsoeSource("PL", "key", type=["installed_capacity"])])
    assert not any("capacity" in k for k in source_rules([EntsoeSource("PL", "key", type=["price"], prefix="pl")]))


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------


def test_rolling_names():
    info = InformationSet(rules={"x*": Availability.static()})
    for name, window, built in [
        ("x_roll7max", 7, "x"),
        ("x_roll365max_d-1", 365, "x"),
        ("x_roll30mean_d-2", 30, "x"),
        ("x_d-1_roll7min", 7, "x_d-1"),
        ("x_roll1max", 1, "x"),
        ("x_h05_roll7max_d-1", 7, "x_h05"),
    ]:
        col = info.resolve(name)
        assert (col.window, col.built_from.name) == (window, built), name
    for name in ["x_roll0max", "x_rollmax", "x_roll7median", "x_roll7max_h05", "x_roll07max"]:
        col = info.resolve(name)
        assert col.built_from is None or col.window == 1, name


def test_the_capacity_proxy_is_known_at_the_issue_and_one_without_the_lag_is_late_at_every_period():
    """The built-in `generation_*` (realised 1 h after its period): the window ending D-1 is out at 01:00 on D, the one ending D
    at 01:00 on D+1, so the second is refused at every period, also those whose own hour on D is out by the issue (00:00 to
    09:00). Matched by `generation_*` as a plain column, it was judged at its row's hour and let those through."""
    names = ["generation_solar_roll365max_d-1", "generation_solar_roll365max", "generation_solar_roll7mean_d-1"]
    found, unknown = violations(names, horizon=1)
    assert unknown == set()
    assert _refused_periods(found) == {"generation_solar_roll365max": set(range(24))}
    with pytest.raises(InformationSetError, match=r"generation_solar_roll365max\b"):
        InformationSet().check(
            expand(["generation_solar_roll365max"], horizon=1),
            "price",
            "1h",
            pd.date_range("2024-01-01", periods=24 * 30, freq="h", tz="Europe/Warsaw"),
            "2024-01-20",
            "2024-01-20",
        )


def test_a_rolling_statistic_of_a_published_series_follows_its_rule():
    """The day-ahead price of D is out at 13:00 on D-1: a window ending D is known at the issue on D, one ending D+1 (a lag of
    the column one day ahead) is not."""
    found, unknown = violations(["price_roll7max", "price_roll7max_d+1", "price_roll7mean_d-1"], horizon=1)
    assert unknown == set()
    assert _refused_periods(found) == {"price_roll7max_d+1": set(range(24))}


@pytest.mark.parametrize(
    "data_zone, rule_zone, switch, realised, freq",
    [
        ("Europe/Warsaw", "Europe/Brussels", "2024-03-31", False, "1h"),
        ("Europe/Warsaw", "Atlantic/Cape_Verde", "2024-10-27", True, "1h"),
        ("Etc/GMT-1", "Europe/Brussels", "2024-10-27", False, "15min"),
        ("Etc/GMT-1", "Atlantic/Cape_Verde", "2024-03-31", True, "15min"),
        ("Europe/Warsaw", "Europe/Brussels", "2024-10-27", True, "15min"),
        ("Etc/GMT-1", "Atlantic/Cape_Verde", "2024-10-27", False, "1h"),
    ],
)
def test_rolling_verdicts_match_the_newest_delivery_each_cell_holds(data_zone, rule_zone, switch, realised, freq):
    """Known answer from the data: `px` holds its own instant, so the maximum over a window holds its newest delivery, and the
    probe and the check must call a cell unknown exactly when that delivery is (the oracle of the day aggregates), for windows
    ending on the row's day and two days before it, and their day lags either way, over a switch, hourly and every 15 minutes
    (each level of each axis, and every pair of the zone, switch and freq levels)."""
    start = pd.Timestamp(switch) - pd.Timedelta(days=6)
    index = pd.date_range(start, start + pd.Timedelta(days=11), freq=freq, tz=data_zone, inclusive="left")
    data = pd.DataFrame({"px": (index.asi8 // 60_000_000_000).astype(float)}, index=index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # a fixed-offset index's days
        for lag in (0, 2):
            data = RollingStatsTransformer("px", window_days=3, stat="max", lag_days=lag, freq=freq).transform(data)
    rolling = [c for c in data.columns if c != "px"]
    data = LagTransformer(columns=rolling, lags=[1, -1], freq="day").transform(data).drop(columns="px")

    assert _verdicts_against_held_deliveries(data, rule_zone, switch, realised, freq) == {}


# Rules over time. Until BOUND a value of day X is out at 13:00 on X+2 (a slow archive), after it 1 h after its period.
BOUND = "2024-04-10"
SLOW_THEN_FAST = [
    Availability.published("13:00", -2, "Europe/Warsaw").during(valid_until=BOUND),
    Availability.realised("1h").during(valid_from="2024-04-11"),
]
# A gap: no rule holds for 11 and 12 April, whose values are never known.
WITH_A_GAP = [
    Availability.realised("1h").during(valid_until=BOUND),
    Availability.realised("1h").during(valid_from="2024-04-13"),
]
# A snapshot as of 09:00 on the day it is read on, each day of a window on its own day, with the same gap in the days read.
AS_OF_GAP = [
    Availability.as_of("09:00", "Europe/Warsaw").during(valid_until=BOUND),
    Availability.as_of("09:00", "Europe/Warsaw").during(valid_from="2024-04-13"),
]


def _known_at(instant: pd.Timestamp, rules) -> pd.Timestamp:
    """When the value of the period starting at `instant` is known, by hand (hourly periods)."""
    day = instant.tz_convert("Europe/Warsaw").tz_localize(None).normalize()
    for rule in rules:
        inside = (rule.valid_from is None or day >= pd.Timestamp(rule.valid_from)) and (
            rule.valid_until is None or day <= pd.Timestamp(rule.valid_until)
        )
        if not inside:
            continue
        if rule.kind == "realised":
            return instant + pd.Timedelta(hours=2)  # the end of its hour, plus 1 h
        if rule.kind == "as_of":
            return (day + pd.Timedelta(hours=9)).tz_localize("Europe/Warsaw")  # read on its own day
        return (day + pd.Timedelta(days=2, hours=13)).tz_localize("Europe/Warsaw")
    return pd.Timestamp.max.tz_localize("UTC")


@pytest.mark.filterwarnings("ignore:The rules of")
@pytest.mark.parametrize(
    "rules", [SLOW_THEN_FAST, WITH_A_GAP, AS_OF_GAP], ids=["slow then fast", "a gap", "a gap in the days read"]
)
@pytest.mark.parametrize("lag", [0, 1])
def test_under_rules_over_time_every_day_of_the_window_must_be_known(rules, lag):
    """A window whose newest day is under the fast rule can still hold an older day under the slow one, out later (or a day
    no rule holds for, never out): it is known only when every value of every day it summarises is. Each cell of the probe
    and each verdict of the check against the latest of its window's values by hand, on the issue days around the bound."""
    index = pd.date_range("2024-03-25", "2024-04-25", freq="h", tz="Europe/Warsaw", inclusive="left")
    info = InformationSet("11:45", "Europe/Warsaw", rules={"x": rules})
    name = RollingStatsTransformer("x", window_days=4, lag_days=lag).output_column
    data = pd.DataFrame({name: 1.0}, index=index)
    wall = index.tz_localize(None).normalize()
    by_day = {day: index[wall == day] for day in wall.unique()}
    latest = {
        day: max(
            _known_at(t, rules)
            for d in pd.date_range(day - pd.Timedelta(days=lag + 3), day - pd.Timedelta(days=lag))
            for t in by_day.get(d, [])
        )
        for day in wall.unique()
        if day - pd.Timedelta(days=lag + 3) >= wall[0]
    }
    late_days = set()
    for run_day in pd.date_range("2024-04-08", "2024-04-18"):
        issue = pd.Timestamp(f"{run_day.date()} 11:45", tz="Europe/Warsaw")
        late = latest[run_day] > issue
        if late:
            late_days.add(str(run_day.date()))
        perturbed, _, _ = perturb_unknown(data, str(run_day.date()), info)
        rows = wall == run_day
        assert (perturbed.loc[rows, name] != 1.0).tolist() == [late] * int(rows.sum()), run_day
        found, _ = info.violations(
            expand([name], horizon=1), "price", "1h", index, str(run_day.date()), str(run_day.date())
        )
        assert {v.period for v in found} == (set(range(24)) if late else set()), run_day
    # The newest day alone would have let some of these through.
    newest_alone = {
        str(day.date())
        for day in pd.date_range("2024-04-08", "2024-04-18")
        if max(_known_at(t, rules) for t in by_day[day - pd.Timedelta(days=lag)])
        > pd.Timestamp(f"{day.date()} 11:45", tz="Europe/Warsaw")
    }
    assert newest_alone <= late_days
    if lag:  # a window ending D-1: its newest day is out at 01:00 on D after the bound, while an older day isn't yet
        assert newest_alone < late_days


def test_explain_spans_the_whole_window():
    """`x_roll7max_d-1` at horizon 1 holds deliveries from 00:00 on D-7 to 23:00 on D-1: T-8 00:00 to T-2 23:00."""
    from epftoolbox2.information_set import _Context, _delivered_span

    info = InformationSet(rules={"x": Availability.realised("1h")})
    ctx = _Context(day=pd.Timestamp("2024-01-10"), data_zone="Europe/Warsaw")
    start, end = _delivered_span(info.resolve("x_roll7max_d-1"), np.array([0]), np.array([720.0]), 60, ctx)
    assert (start[0], end[0]) == (-7 * 1440.0, -1 * 1440.0 + 23 * 60.0)
    start, end = _delivered_span(info.resolve("x_day_max_d-1"), np.array([0]), np.array([720.0]), 60, ctx)
    assert (start[0], end[0]) == (-1440.0, -1440.0 + 23 * 60.0)
