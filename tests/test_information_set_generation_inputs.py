"""The information set and the generation inputs of step 7e: static rules for SolarGeometryTransformer's columns (built in for
its own names, from `source_rules` for a prefixed one), ENTSO-E's installed capacity known a week ahead, and a
RollingStatsTransformer column judged through the column it was built from at the last slot of its window's newest day, every
day of the window under rules over time."""

import warnings

import numpy as np
import pandas as pd
import pytest

import yaml

from epftoolbox2.data.sources.calendar import CalendarSource
from epftoolbox2.data.sources.entsoe import EntsoeSource
from epftoolbox2.data.transformers import LagTransformer, RollingStatsTransformer, SolarGeometryTransformer
from epftoolbox2.information_set import (
    Availability,
    InformationSet,
    InformationSetError,
    perturb_unknown,
    source_rules,
)
from epftoolbox2.pipelines import Workflow
from tests.test_information_set import _refused_periods, _verdicts_against_held_deliveries, expand, violations

SOLAR = ["sun_elevation", "sun_sin_elevation", "clear_sky_ghi", "is_night"]


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prefix", ["", "de_"])
def test_solar_geometry_columns_are_static_at_any_shift(prefix):
    """A function of the row's instant: known at any time, so `is_night_d+1` can be a model's `zero_when` and a lead of a week
    is as known as the column itself. The transformer's own names are built in; a prefixed one's come from `source_rules`, which
    has none without it."""
    names = [f"{prefix}{name}{shift}" for name in SOLAR for shift in ("", "_d+1", "_d+7", "_d-1", "_h+30")]
    info = InformationSet()
    if prefix:
        assert all(info.resolve(name).rule is None for name in names)
        info = info.with_source_rules([SolarGeometryTransformer([(51.0, 10.0)], prefix=prefix.rstrip("_"))])
    for name in names:
        col = info.resolve(name)
        assert col.rule is not None and col.rule.kind == "static", name
    assert violations(names, info=info, horizon=7) == (set(), set())


def test_no_pattern_gives_another_series_a_solar_rule():
    """Exact names only: a series of yours named `sun_power`, a measured night flag or another clear-sky model's irradiance gets
    no rule from them, also beside a prefixed transformer (which names its own columns)."""
    names = ["sun_power", "pv_sun_forecast_d+1", "is_nightly", "sun_elevation_max", "clear_sky", "cams_clear_sky_ghi"]
    names += ["measured_is_night_d-1", "de_sun_elevation", "at_is_night"]
    info = InformationSet().with_source_rules([SolarGeometryTransformer([(47.8, 14.5)], prefix="atx")])
    for name in names:
        assert info.resolve(name).rule is None, name
    assert info.resolve("atx_is_night_d+1").rule.kind == "static"


def test_a_prefixed_calendar_copies_the_calendar_s_rules_alone():
    """`CalendarSource(prefix="pl_")` gets the calendar's static rules under its prefix, not every static default: no
    `pl_sun_elevation` or `pl_is_night`."""
    rules = source_rules([CalendarSource("PL", prefix="pl_")])
    assert {"pl_is_holiday", "pl_is_monday", "pl_sunrise", "pl_holiday_*", "pl_is_[0-9]*"} <= set(rules)
    assert not any(word in key for key in rules for word in ("sun_", "night", "clear_sky", "capacity"))
    assert all(rule.note == "calendar (CalendarSource)" for rule in rules.values())


def test_a_workflow_knows_a_prefixed_solar_transformer_s_columns(tmp_path):
    """`Workflow` passes the data pipeline's transformers to `with_source_rules`, so `pl_is_night_d+1` is static at "raise"."""
    index = pd.date_range("2023-01-01", "2023-04-01", freq="1h", tz="UTC", inclusive="left")
    rng = np.random.default_rng(0)
    pd.DataFrame({"datetime": index.strftime("%Y-%m-%d %H:%M:%S"), "price": rng.normal(100, 10, len(index))}).to_csv(
        tmp_path / "prices.csv", index=False
    )
    config = {
        "data_pipeline": {
            "start": "2023-01-01",
            "end": "2023-03-31",
            "sources": [{"class": "CsvSource", "params": {"file_path": str(tmp_path / "prices.csv")}}],
            "transformers": [
                {"class": "TimezoneTransformer", "params": {"target_tz": "Europe/Warsaw"}},
                {"class": "SolarGeometryTransformer", "params": {"locations": [[52.2, 21.0]], "prefix": "pl"}},
                {"class": "LagTransformer", "params": {"columns": ["price"], "lags": [1], "freq": "day"}},
                {"class": "LagTransformer", "params": {"columns": ["pl_clear_sky_ghi"], "lags": [-1], "freq": "day"}},
            ],
        },
        "model_pipeline": {
            "test_start": "2023-03-01",
            "test_end": "2023-03-01",
            "horizon": 1,
            "information_set": {"issue_time": "11:45", "issue_timezone": "Europe/Brussels"},
            "availability": "raise",
            "models": [
                {
                    "class": "OLSModel",
                    "params": {
                        "predictors": ["price_d-1", "pl_clear_sky_ghi_d+1"],
                        "training_window": 30,
                        "name": "OLS",
                    },
                }
            ],
        },
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    report = Workflow.load(path).run()
    assert report.forecast_counts().loc[0, "produced"] == 24


def test_installed_capacity_is_known_a_week_ahead_with_its_archive_s_newest_revision():
    """Decision 8: A68 is due a week before the year it is for begins, so every row holds a value out before its year did, and
    a lead of up to 7 days is known (on 31 December, next year's value, due on 25 December). A longer lead can read next
    year's value before it is due: late at every period, refused at "raise". Read as out at 00:00 seven days before each date,
    both of the data's zone, so in Lisbon as in Warsaw."""
    info = InformationSet()
    for name in ["installed_capacity_solar", "installed_capacity_wind_onshore_d+1", "installed_capacity_solar_d+7"]:
        col = info.resolve(name)
        assert (col.key, col.rule.kind, col.rule.vintage) == ("installed_capacity_*", "published", "latest_revision")
    names = ["installed_capacity_solar_d+7", "installed_capacity_solar_d-1", "installed_capacity_solar_d+8"]
    for index in (
        pd.date_range("2024-12-01", "2025-01-20", freq="h", tz=zone) for zone in ("Europe/Warsaw", "Europe/Lisbon")
    ):
        found, unknown = violations(names, horizon=1, start="2024-12-20", end="2025-01-05", index=index)
        assert unknown == set()
        assert _refused_periods(found) == {"installed_capacity_solar_d+8": set(range(24))}
    with pytest.raises(InformationSetError, match="installed_capacity_solar_d\\+8"):
        info.check(
            expand(["installed_capacity_solar_d+8"], horizon=1), "price", "1h", index, "2024-12-27", "2024-12-27"
        )
    # A plain `installed_capacity` (no production type) is no ENTSO-E column: no rule.
    assert info.resolve("installed_capacity").rule is None


def test_a_prefixed_entsoe_source_s_capacity_gets_the_rule_from_its_source():
    prefixed = EntsoeSource("PL", "key", type=["installed_capacity"], prefix="pl")
    rules = source_rules([prefixed])
    assert rules["pl_installed_capacity_*"].kind == "published"
    assert rules["pl_installed_capacity_*"].vintage == "latest_revision"
    info = InformationSet().with_source_rules([prefixed])
    assert info.resolve("pl_installed_capacity_solar_d+1").rule.kind == "published"
    assert InformationSet().resolve("pl_installed_capacity_solar").rule is None  # outside a source-aware run, no rule
    # Without a prefix the source's rule replaces the built-in one, in its market's dates; without the type it adds nothing.
    plain = EntsoeSource("PL", "key", type=["installed_capacity"])
    assert source_rules([plain])["installed_capacity_*"].timezone == plain._daily_zone()
    assert not any("capacity" in k for k in source_rules([EntsoeSource("PL", "key", type=["price"], prefix="pl")]))


@pytest.mark.parametrize("prefix", ["", "pl"])
@pytest.mark.parametrize("zone", ["UTC", "America/New_York"])
def test_an_entsoe_source_s_capacity_is_read_on_its_market_s_dates(zone, prefix):
    """A68 for year Y is due at 00:00 on 25 December of Y-1 on the market's clock. On a frame west of the market (UTC, New
    York) the hours of 24 December from 23:00 UTC (18:00 in New York) are 1 January in Warsaw: their `_d+7` reads next
    year's value a day before it is due. The rule in data-zone dates (no source) let 1 and 6 of them through; the
    `EntsoeSource`'s rule reads the dates on its market's clock (`_daily_zone`), so none passes before it is due."""
    source = EntsoeSource("PL", "key", type=["installed_capacity"], prefix=prefix)
    rule = source_rules([source])[f"{prefix}_installed_capacity_*" if prefix else "installed_capacity_*"]
    assert rule.timezone == source._daily_zone() and rule.vintage == "latest_revision"
    info = InformationSet("11:45", "Europe/Brussels").with_source_rules([source])
    index = pd.date_range("2024-12-01", "2025-01-20", freq="h", tz=zone, inclusive="left")
    market = source._daily_zone()
    early = []
    for k in (7, 8):
        name = f"{prefix}_installed_capacity_solar_d+{k}" if prefix else f"installed_capacity_solar_d+{k}"
        for run_day in pd.date_range("2024-12-22", "2024-12-27"):
            day = str(run_day.date())
            flagged = {v.period for v in info.violations(expand([name], horizon=1), "price", "1h", index, day, day)[0]}
            issue = pd.Timestamp(f"{day} 11:45", tz="Europe/Brussels")
            for period in range(24):
                instant = (run_day + pd.Timedelta(days=k, hours=period)).tz_localize(zone)
                due = pd.Timestamp(f"{instant.tz_convert(market).year - 1}-12-25 00:00", tz=market)
                if issue < due and period not in flagged:
                    early.append((name, day, period))
    assert early == []


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
    late_days, known_days = set(), []
    for run_day in pd.date_range("2024-04-08", "2024-04-18"):
        issue = pd.Timestamp(f"{run_day.date()} 11:45", tz="Europe/Warsaw")
        late = latest[run_day] > issue
        (late_days.add if late else known_days.append)(str(run_day.date()))
        perturbed, _, _ = perturb_unknown(data, str(run_day.date()), info)
        rows = wall == run_day
        assert (perturbed.loc[rows, name] != 1.0).tolist() == [late] * int(rows.sum()), run_day
    # The check: each late day on its own, every known day in one call (any late one among them would be found).
    for day in sorted(late_days):
        found, _ = info.violations(expand([name], horizon=1), "price", "1h", index, day, day)
        assert {v.period for v in found} == set(range(24)), day
    found, _ = info.violations(expand([name], horizon=1), "price", "1h", index, issue_days=known_days)
    assert found == [], known_days
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


# The window of a lag, or of a window, reads further back than its own days: x_d-5_roll10max holds x on D-14 .. D-5, as
# x_roll10max_d-5 does, and x_roll7max_roll10max_d-1 x on D-16 .. D-1, as x_roll16max_d-1 does.
PAIRS = {
    # Until 10 April a value of day X is out at 13:00 on X+8, from 11 April 1 h after its period.
    "a lag": (
        ("x_d-5_roll10max", "x_roll10max_d-5"),
        [
            Availability.published("13:00", -8, "Europe/Warsaw").during(valid_until=BOUND),
            Availability.realised("1h").during(valid_from="2024-04-11"),
        ],
        "2024-04-19",  # x of 10 April is out at 13:00 on 18 April
    ),
    "a window": (
        ("x_roll7max_roll10max_d-1", "x_roll16max_d-1"),
        [
            Availability.published("13:00", -15, "Europe/Warsaw").during(valid_until=BOUND),
            Availability.realised("1h").during(valid_from="2024-04-11"),
        ],
        "2024-04-26",  # x of 10 April is out at 13:00 on 25 April
    ),
}


@pytest.mark.filterwarnings("ignore:The rules of")
@pytest.mark.parametrize("case", sorted(PAIRS))
def test_a_window_of_a_lag_or_of_a_window_reads_every_value_it_holds(case):
    """Under a rule that changes over time the two names of each pair hold the same values and get the same verdicts, in the
    check and the probe, on the issue days around the bound: late until the slow day is out, known from the day after. Read
    at the window's own days, the outer window missed the slow day its column reaches."""
    (name, same), rules, first_known = PAIRS[case]
    info = InformationSet("11:45", "Europe/Warsaw", rules={"x": rules})
    index = pd.date_range("2024-03-01", "2024-05-15", freq="h", tz="Europe/Warsaw", inclusive="left")
    wall = index.tz_localize(None).normalize()
    data = pd.DataFrame({name: 1.0, same: 1.0}, index=index)
    days = pd.date_range(pd.Timestamp(first_known) - pd.Timedelta(days=3), periods=6)
    for day in days:
        late = day < pd.Timestamp(first_known)
        verdicts = []
        for column in (name, same):
            found, _ = info.violations(
                expand([column], horizon=1), "price", "1h", index, str(day.date()), str(day.date())
            )
            verdicts.append({v.period for v in found})
        assert verdicts[0] == verdicts[1] == (set(range(24)) if late else set()), (case, day)
        perturbed, _, _ = perturb_unknown(data, str(day.date()), info)
        moved = (perturbed.loc[wall == day] != 1.0).all()
        assert moved[name] == moved[same] == late, (case, day)


# A rule in another zone than the data's: the delivery day is a Helsinki date (a Warsaw 23:00 is 00:00 of the next one) and an
# as_of value is read at 09:00 Helsinki. The probe judges every row of the frame, each against D's issue time.
HELSINKI = "Europe/Helsinki"
NEVER = pd.Timestamp.max.tz_localize("UTC")
ZONED = {
    "published in Helsinki": [
        Availability.published("13:00", -3, HELSINKI).during(valid_until=BOUND),
        Availability.realised("1h").during(valid_from="2024-04-11"),
    ],
    "as_of in Helsinki": [
        Availability.as_of("09:00", HELSINKI).during(valid_until=BOUND),
        Availability.as_of("09:00", HELSINKI).during(valid_from="2024-04-13"),
    ],
}


def _inside(rule, day: pd.Timestamp) -> bool:
    return (rule.valid_from is None or day >= pd.Timestamp(rule.valid_from)) and (
        rule.valid_until is None or day <= pd.Timestamp(rule.valid_until)
    )


def _slot_known(instant: pd.Timestamp, rules) -> pd.Timestamp:
    """The published or realised value of the hour starting at `instant`: published on its Helsinki date + 3 at 13:00 there, or
    realised 1 h after it ends on its Warsaw date; the later where both hold, never where none does."""
    known = []
    for rule in rules:
        if rule.kind == "published":
            day = instant.tz_convert(HELSINKI).tz_localize(None).normalize()
            if _inside(rule, day):
                known.append((day + pd.Timedelta(days=-rule.days_before, hours=13)).tz_localize(HELSINKI))
        elif _inside(rule, instant.tz_convert("Europe/Warsaw").tz_localize(None).normalize()):
            known.append(instant + pd.Timedelta(hours=2))
    return max(known) if known else NEVER


def _read_known(day: pd.Timestamp, rules) -> pd.Timestamp:
    """An as_of value read on `day`: at 09:00 Helsinki, if a rule holds for that day."""
    held = any(_inside(rule, day) for rule in rules)
    return (day + pd.Timedelta(hours=9)).tz_localize(HELSINKI) if held else NEVER


@pytest.mark.filterwarnings("ignore:The rules of")
@pytest.mark.parametrize("case", sorted(ZONED))
@pytest.mark.parametrize("run_day", ["2024-04-11", "2024-04-14", "2024-04-17"])
def test_rolling_cells_of_every_row_under_a_rule_of_another_zone(case, run_day):
    """Every row of the frame, not only the issue day's: under the published rule a row's window is known when every hour of
    its Warsaw days is (the Warsaw 23:00 of 10 April is 11 April in Helsinki, where neither rule holds: never known); under the
    as_of rule a row up to D reads its window's days on D - 1 back (a row after D on its own day - 1 back), each at 09:00
    Helsinki, so the rows before D share D's verdict."""
    rules = ZONED[case]
    name = "x_roll4max_d-1"
    info = InformationSet("11:45", "Europe/Warsaw", rules={"x": rules})
    index = pd.date_range("2024-03-28", "2024-04-24", freq="h", tz="Europe/Warsaw", inclusive="left")
    wall = index.tz_localize(None).normalize()
    D = pd.Timestamp(run_day)
    issue = pd.Timestamp(f"{run_day} 11:45", tz="Europe/Warsaw")
    expected = {}
    for day in wall.unique():
        if rules[0].kind == "published":
            hours = [
                t
                for d in pd.date_range(day - pd.Timedelta(days=4), day - pd.Timedelta(days=1))
                for t in pd.date_range(d, d + pd.Timedelta(days=1), freq="h", tz="Europe/Warsaw", inclusive="left")
            ]
            latest = max(_slot_known(t, rules) for t in hours)
        else:
            newest = max(day, D) - pd.Timedelta(days=1)  # read on the issue day's, or a later row's own, day - 1
            latest = max(_read_known(newest - pd.Timedelta(days=k), rules) for k in range(4))
        expected[day] = latest > issue
    perturbed, _, _ = perturb_unknown(pd.DataFrame({name: 1.0}, index=index), run_day, info)
    got = pd.Series((perturbed[name] != 1.0).to_numpy()).groupby(wall).agg(["all", "any"])
    assert (got["all"] == got["any"]).all()  # one verdict a day
    wrong = {str(day.date()) for day in got.index if bool(got.loc[day, "all"]) != expected[day]}
    assert not wrong, (case, run_day, sorted(wrong))
    if rules[0].kind == "as_of" and run_day == "2024-04-14":
        # D's window is read on 13 to 10 April, two of them in the gap: 1 April's row is late, though its own days aren't.
        assert expected[pd.Timestamp("2024-04-01")]


def test_explain_spans_the_whole_window():
    """`x_roll7max_d-1` at horizon 1 holds deliveries from 00:00 on D-7 to 23:00 on D-1: T-8 00:00 to T-2 23:00."""
    from epftoolbox2.information_set import _Context, _delivered_span

    info = InformationSet(rules={"x": Availability.realised("1h")})
    ctx = _Context(day=pd.Timestamp("2024-01-10"), data_zone="Europe/Warsaw")
    start, end = _delivered_span(info.resolve("x_roll7max_d-1"), np.array([0]), np.array([720.0]), 60, ctx)
    assert (start[0], end[0]) == (-7 * 1440.0, -1 * 1440.0 + 23 * 60.0)
    start, end = _delivered_span(info.resolve("x_day_max_d-1"), np.array([0]), np.array([720.0]), 60, ctx)
    assert (start[0], end[0]) == (-1440.0, -1440.0 + 23 * 60.0)
