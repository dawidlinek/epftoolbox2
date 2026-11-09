"""`Availability.published(day_zone=)`: the delivery day of a value as a date of another zone, and the chained columns it serves.

`day_zone` is the counterpart of `weekly(week_zone=)` for a delivery day: None (the default, and every rule written before it)
reads it in `timezone`, as before; "data" in the data's zone; a zone's name in that zone. Another pipeline's forecasts
(`ForecastSource`) are for dates of the upstream's data zone and issued on its issue zone's clock, so their rule is
`published(issue time, days_before=horizon, timezone=issue zone, day_zone=data zone)`.

The oracle writes down when each cell of a chained column is known from that description alone, in UTC instants: the value at
instant u is the forecast for u issued `horizon` days before u's date in the upstream data zone, at the issue time in the issue
zone, and a row of D reads `col_d+k` at the same wall-clock time k days later in the frame's zone (`shift_calendar_days`). It is
compared with every cell the check reads, for every triple of (upstream data zone, upstream issue zone, downstream frame zone)
among Warsaw, Helsinki, Lisbon and UTC, over both switches, for `col`, `col_d+1`, `col_d+2` and `col_d-1` of horizon-1 and
horizon-2 sources: equal to the minute, so no cell is early and none is late that isn't.
"""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from epftoolbox2._date_utils import shift_calendar_days
from epftoolbox2.data.sources import ForecastSource
from epftoolbox2.information_set import _STEP_MINUTES, Availability, InformationSet, InformationSetError, RuleChain
from epftoolbox2.models._calendar import periods_per_day
from epftoolbox2.pipelines import ModelPipeline

ZONES = ["Europe/Warsaw", "Europe/Helsinki", "Europe/Lisbon", "UTC"]
# Issue days of each window: the EU switches of 2024 and the days around them.
WINDOWS = {"spring": ("2024-03-28", "2024-04-02"), "autumn": ("2024-10-24", "2024-10-29")}
NAMES = {"col": 0, "col_d+1": 1, "col_d+2": 2, "col_d-1": -1}
ISSUE = "09:30"


def upstream(tmp_path, data_zone: str, issue_zone: str, freq: str = "1h", name: str = "up", issue: str = ISSUE) -> None:
    """An upstream manifest (and one record) written by hand: what `ForecastSource` reads to state its rule."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    record = {"run_date": "2024-01-01", "target_date": "2024-01-02", "hour": 0, "horizon": 1, "prediction": 1.0}
    (tmp_path / f"{name}.jsonl").write_text(json.dumps(record) + "\n")
    manifest = {
        "freq": freq,
        "horizon": 2,
        "data_timezone": data_zone,
        "information_set": {"issue_time": issue, "issue_timezone": issue_zone},
        "availability": {"level": "raise", "times": "instants", "listing": []},
    }
    (tmp_path / f"{name}.manifest.json").write_text(json.dumps(manifest))


def frame_index(zone: str, freq: str, window: str) -> pd.DatetimeIndex:
    first = pd.Timestamp(WINDOWS[window][0]) - pd.Timedelta(days=3)
    last = pd.Timestamp(WINDOWS[window][1]) + pd.Timedelta(days=4)
    return pd.date_range(first.tz_localize(zone), last.tz_localize(zone), freq=freq, inclusive="left")


def library_known(info: InformationSet, name: str, index: pd.DatetimeIndex, freq: str, window: str):
    """[period, issue day] -> when the check says the value of `name` on the row of the issue day is known (UTC), the issue
    instants, and which cells the clock skips."""
    days = pd.date_range(*WINDOWS[window], freq="D")
    clock = info._issue_clock(days, index, freq)
    step = _STEP_MINUTES[freq]
    periods = np.arange(periods_per_day(freq))
    avail = clock.available(info._resolve_for(name, step), periods, step)
    origin = clock.ctx.origin.tz_convert("UTC").tz_localize(None)  # instants as naive UTC datetime64
    minutes = np.where(np.isfinite(avail), avail, np.nan).ravel()  # a skipped cell has no verdict (-inf): NaT
    known = (origin + pd.to_timedelta(minutes, unit="min")).to_numpy()
    issued = (origin + pd.to_timedelta(clock.issue, unit="min")).to_numpy()
    return known.reshape(avail.shape), issued, clock.skipped(periods, step), days


def oracle(name: str, frame_zone: str, data_zone: str, issue_zone: str, horizon: int, freq: str, window: str):
    """[period, issue day] -> when the value is known by the description alone (UTC), NaT where the row reads nothing."""
    days = pd.date_range(*WINDOWS[window], freq="D")
    step = _STEP_MINUTES[freq]
    periods = np.arange(periods_per_day(freq))
    wall = pd.DatetimeIndex([day + pd.Timedelta(minutes=int(p * step)) for p in periods for day in days])
    # The row of a period: the second occurrence of a repeated wall time (the one a model forecasts from); none if skipped.
    rows = wall.tz_localize(frame_zone, ambiguous=np.zeros(len(wall), dtype=bool), nonexistent="NaT")
    value_at = shift_calendar_days(rows, NAMES[name]) if NAMES[name] else rows
    delivered = value_at.tz_convert(data_zone).tz_localize(None).normalize()
    issued_on = delivered - pd.Timedelta(days=horizon) + pd.Timedelta(ISSUE + ":00")
    known = issued_on.tz_localize(issue_zone, ambiguous=np.zeros(len(wall), dtype=bool), nonexistent="NaT")
    return known.tz_convert("UTC").tz_localize(None).to_numpy().reshape(len(periods), len(days))


TRIPLES = [(data, issue, frame) for data in ZONES for issue in ZONES for frame in ZONES]


@pytest.mark.parametrize(("data_zone", "issue_zone", "frame_zone"), TRIPLES)
def test_a_chained_cell_is_known_exactly_at_its_upstream_issue_instant(tmp_path, data_zone, issue_zone, frame_zone):
    upstream(tmp_path, data_zone, issue_zone)
    freq = "15min" if (ZONES.index(data_zone) + ZONES.index(issue_zone) + ZONES.index(frame_zone)) % 4 == 0 else "1h"
    for horizon in (1, 2):
        info = InformationSet("11:45", "Europe/Brussels").with_source_rules(
            [ForecastSource(tmp_path, "up", "col", horizon=horizon)]
        )
        for window in WINDOWS:
            index = frame_index(frame_zone, freq, window)
            for name in NAMES:
                known, _, skipped, days = library_known(info, name, index, freq, window)
                truth = oracle(name, frame_zone, data_zone, issue_zone, horizon, freq, window)
                judged = ~skipped & ~np.isnat(truth)
                early = judged & (known < truth)
                late = judged & (known > truth)
                assert not early.any(), (name, horizon, window, "early", np.argwhere(early)[:3])
                assert not late.any(), (name, horizon, window, "late", np.argwhere(late)[:3])
                assert judged.sum() > 0.9 * judged.size


def test_a_source_read_one_horizon_beyond_its_own_is_late_on_every_cell(tmp_path):
    upstream(tmp_path, "Europe/Helsinki", "Europe/Brussels")
    for horizon in (1, 2):
        info = InformationSet("11:45", "Europe/Brussels").with_source_rules(
            [ForecastSource(tmp_path, "up", "col", horizon=horizon)]
        )
        for window in WINDOWS:
            # On a frame in the upstream's zone a row of D reads the forecasts for one date of it: D + k, issued on D + k - h.
            index = frame_index("Europe/Helsinki", "1h", window)
            beyond = f"col_d+{horizon + 1}"
            known, issued, skipped, _ = library_known(info, beyond, index, "1h", window)
            assert (known[~skipped] > np.broadcast_to(issued, known.shape)[~skipped]).all()
            inside = f"col_d+{horizon}"
            known, issued, skipped, _ = library_known(info, inside, index, "1h", window)
            assert (known[~skipped] <= np.broadcast_to(issued, known.shape)[~skipped]).all()


def _rule_cells(rule: Availability, name: str, frame_zone: str, issue: str, window: str = "autumn"):
    info = InformationSet(issue, "Europe/Brussels", rules={"col": rule})
    known, issued, skipped, _ = library_known(info, name, frame_index(frame_zone, "1h", window), "1h", window)
    return known <= np.broadcast_to(issued, known.shape), skipped


def test_helsinki_an_upstream_read_in_brussels_dates_passes_a_forecast_issued_a_day_late(tmp_path):
    """The prototype's open finding: a Helsinki upstream issued at 09:30 Brussels. The forecast for Helsinki 00:00 of D+1 is
    issued at 09:30 on D; read in Brussels dates (23:00 of D) it would count as issued on D-1, so a downstream issued at 09:00
    on D could read it. In the upstream's own dates every hour of D+1 is late at 09:00, as it is."""
    upstream(tmp_path, "Europe/Helsinki", "Europe/Brussels")
    rule = ForecastSource(tmp_path, "up", "col").availability()
    assert rule.day_zone == "Europe/Helsinki"
    known, skipped = _rule_cells(rule, "col_d+1", "Europe/Helsinki", "09:00")
    assert not known[~skipped].any()
    brussels_dates = replace(rule, day_zone=None)
    known, skipped = _rule_cells(brussels_dates, "col_d+1", "Europe/Helsinki", "09:00")
    assert known[0].all() and not known[1:][~skipped[1:]].any()  # 00:00 Helsinki: a day early, every day


def test_lisbon_an_upstream_read_in_brussels_dates_refuses_a_forecast_issued_in_time(tmp_path):
    """Lisbon's 23:00 of D+1 is Brussels' 00:00 of D+2: read in Brussels dates the forecast for it would count as issued on
    D+1 and be refused at 11:45 on D, though it was issued at 09:30 on D."""
    upstream(tmp_path, "Europe/Lisbon", "Europe/Brussels")
    rule = ForecastSource(tmp_path, "up", "col").availability()
    known, skipped = _rule_cells(rule, "col_d+1", "Europe/Lisbon", "11:45")
    assert known[~skipped].all()
    known, skipped = _rule_cells(replace(rule, day_zone=None), "col_d+1", "Europe/Lisbon", "11:45")
    assert (
        not known[23].any() and known[:23][~skipped[:23]].all()
    )  # 0 false-late cells with the day zone, 23:00 without


@pytest.mark.parametrize("zone", ["Europe/Brussels", "Europe/Helsinki", "Europe/Lisbon", "UTC"])
@pytest.mark.parametrize("freq", ["1h", "15min", "1D"])
def test_none_reads_the_delivery_day_in_the_rule_s_own_zone_bitwise(zone, freq):
    """None is every rule as it was: equal, cell for cell, to `day_zone` set to the rule's own zone, on frames in other zones."""
    for frame_zone in ("Europe/Warsaw", "Europe/Helsinki", "UTC"):
        for window in WINDOWS:
            if freq == "1D":
                first = pd.Timestamp(WINDOWS[window][0]) - pd.Timedelta(days=3)
                index = (pd.date_range(first, periods=12, freq="D") + pd.Timedelta(hours=12)).tz_localize(frame_zone)
            else:
                index = frame_index(frame_zone, freq, window)
            for name in ("col", "col_d+1", "col_d-1", "col_h-3", "col_h05_d+1", "col_day_max_d-1"):
                found = []
                for day_zone in (None, zone):
                    rule = Availability.published("13:00", 1, timezone=zone, day_zone=day_zone)
                    info = InformationSet("11:45", "Europe/Brussels", rules={"col": rule})
                    days = pd.date_range(*WINDOWS[window], freq="D")
                    clock = info._issue_clock(days, index, freq)
                    step = _STEP_MINUTES[freq]
                    found.append(clock.available(info._resolve_for(name, step), np.arange(periods_per_day(freq)), step))
                assert np.array_equal(found[0], found[1]), (name, frame_zone, window)


def test_data_reads_the_data_s_dates_and_on_a_frame_in_another_zone_the_later_of_both():
    rule = Availability.published("10:00", 1, timezone="Europe/Brussels", day_zone="data")
    named = Availability.published("10:00", 1, timezone="Europe/Brussels", day_zone="Europe/Helsinki")
    index = frame_index("Europe/Helsinki", "1h", "autumn")
    days = pd.date_range(*WINDOWS["autumn"], freq="D")
    found = []
    for one in (rule, named):
        info = InformationSet("11:45", "Europe/Helsinki", rules={"col": one})
        found.append(
            info._issue_clock(days, index, "1h").available(info._resolve_for("col_d+1", 60), np.arange(24), 60)
        )
    assert np.array_equal(found[0], found[1])  # on a frame kept in its zone, "data" is that zone
    # On a UTC frame issued on Helsinki's clock, a Helsinki 00:00 (22:00 UTC the day before) is read in both dates: the later.
    utc = frame_index("UTC", "1h", "autumn")
    info = InformationSet("11:45", "Europe/Helsinki", rules={"col": rule})
    data = info._issue_clock(days, utc, "1h").available(info._resolve_for("col_d+1", 60), np.arange(24), 60)
    singles = []
    for zone in ("UTC", "Europe/Helsinki"):
        one = InformationSet("11:45", "Europe/Helsinki", rules={"col": replace(rule, day_zone=zone)})
        singles.append(one._issue_clock(days, utc, "1h").available(one._resolve_for("col_d+1", 60), np.arange(24), 60))
    assert np.array_equal(data, np.maximum(*singles)) and not np.array_equal(singles[0], singles[1])


def test_a_validity_period_holds_for_the_delivery_days_of_day_zone():
    """`valid_from` is a date of `day_zone`: Helsinki's 00:00 of the first day (Brussels' 23:00 of the day before) is held by
    the rule; read in Brussels dates it is the day before the period, held by none, and never known."""
    start = "2024-10-26"
    rules = [
        Availability.published("09:30", 1, timezone="Europe/Brussels", day_zone=zone).during(valid_from=start)
        for zone in ("Europe/Helsinki", None)
    ]
    index = frame_index("Europe/Helsinki", "1h", "autumn")
    days = pd.date_range("2024-10-24", "2024-10-24", freq="D")  # D: col_d+2 at 00:00 is Helsinki 00:00 on the 26th
    found = []
    for rule in rules:
        info = InformationSet("11:45", "Europe/Brussels", rules={"col": RuleChain((rule,))})
        found.append(
            info._issue_clock(days, index, "1h").available(info._resolve_for("col_d+2", 60), np.arange(24), 60)
        )
    assert np.isfinite(found[0]).all()
    assert np.isinf(found[1][0]).all() and np.isfinite(found[1][1:]).all()


def test_day_zone_belongs_to_published_rules_and_is_written_only_when_set():
    with pytest.raises(ValueError, match="day_zone is a published rule's"):
        Availability("weekly", day_zone="UTC")
    with pytest.raises(ValueError, match="day_zone is a published rule's"):
        Availability("realised", day_zone="data")
    with pytest.raises(Exception):
        Availability.published("10:00", day_zone="Mars/Olympus")
    assert "day_zone" not in Availability.published("13:00").to_dict()
    close = Availability.daily_close("17:30", day_zone="America/New_York")
    assert close.to_dict()["day_zone"] == "America/New_York" and close.days_before == 0
    assert "delivery days of America/New_York dates" in close.describe()
    assert Availability.published("13:00").describe() == "published 13:00 Europe/Brussels on D-1 of delivery"


def test_a_rule_with_day_zone_round_trips_through_yaml(tmp_path):
    rule = Availability.published("09:30", 2, timezone="Europe/Berlin", day_zone="Europe/Helsinki", vintage="snapshot")
    chain = [
        Availability.published("09:30", 1, day_zone="data").during(valid_until="2024-06-30"),
        Availability.published("10:30", 1, day_zone="data").during(valid_from="2024-07-01"),
    ]
    info = InformationSet("11:45", "Europe/Brussels", rules={"a": rule, "b": chain})
    assert InformationSet.from_dict(info.to_dict()).rules == info.rules
    path = tmp_path / "mp.yaml"
    ModelPipeline(information_set=info).save(path)
    assert "day_zone: Europe/Helsinki" in path.read_text()
    assert ModelPipeline.load(path).information_set.rules == info.rules


def test_the_chained_column_is_registered_through_its_shifts_and_what_it_was_built_from(tmp_path):
    upstream(tmp_path, "Europe/Warsaw", "Europe/Warsaw")
    info = InformationSet().with_source_rules([ForecastSource(tmp_path, "up", "solar_fc", horizon=1)])
    assert set(info.chained_inputs()) == {"solar_fc"}
    for name in ("solar_fc", "solar_fc_d+1", "solar_fc_h05_d-1", "solar_fc_d-1_day_max", "solar_fc_day_max_d+1"):
        assert info.chained_of(name).model == "up", name
    for name in ("solar", "price_d+1", "solar_fc2"):
        assert info.chained_of(name) is None
    assert InformationSet().chained_of("solar_fc") is None
    assert info.copy().chained_inputs() == info.chained_inputs()
    assert info.in_data_zone(pd.DatetimeIndex([], tz="UTC")).chained_inputs() == info.chained_inputs()


def test_without_its_source_a_chained_column_has_no_rule():
    """Outside a `Workflow` only `with_source_rules` knows it: the docs pass the sources every time."""
    assert InformationSet().resolve("solar_fc_d+1").rule is None


def test_a_validity_period_of_the_data_s_dates_holds_in_both_readings_on_a_frame_in_another_zone():
    """On a UTC frame issued on Helsinki's clock, "data" is read in both zones' dates and each reading picks its own rule, as
    a weekly rule's week does: Helsinki's first hours of the period's first day are still the day before in UTC, which no rule
    holds for, so they are never known, rather than known by the Helsinki reading alone."""
    rule = Availability.published("10:00", 1, timezone="Europe/Helsinki", day_zone="data").during(
        valid_from="2024-10-26"
    )
    info = InformationSet("11:45", "Europe/Helsinki", rules={"col": RuleChain((rule,))})
    index = frame_index("UTC", "1h", "autumn")
    days = pd.date_range("2024-10-25", "2024-10-26", freq="D")
    known = info._issue_clock(days, index, "1h").available(info._resolve_for("col", 60), np.arange(24), 60)
    assert np.isinf(known[:, 0]).all()  # 25 October in UTC: before the period in one reading at least
    assert np.isfinite(known[:, 1]).all()  # 26 October in UTC is 26 or 27 October in Helsinki: in both


@pytest.mark.parametrize(
    "info",
    [
        InformationSet("10:15", "Europe/Brussels"),
        InformationSet("10:15", "Europe/Brussels", assumptions=["entsoe_a69_by_10"]),
        InformationSet("10:15", "Europe/Brussels", rules={"generation_*": Availability.published("10:00", 1)}),
        InformationSet("10:15", "Europe/Brussels", rules={"generation_solar_*": Availability.static()}),
    ],
    ids=["sources", "assumption", "user glob", "static glob"],
)
def test_no_assumption_or_pattern_replaces_a_chained_column_s_rule(tmp_path, info):
    """An upstream issued at 10:30 under a name an assumption (`generation_*_forecast`, 10:00 on D-1) or a user's pattern
    matches: its forecast for D+1 is still out at 10:30 on D, after a 10:15 issue, through its shifts, profiles and day
    aggregates, in the check and in the probes alike (both read `resolve`)."""
    upstream(tmp_path, "Europe/Berlin", "Europe/Brussels", name="solar_ridge", issue="10:30")
    source = ForecastSource(tmp_path, "solar_ridge", "generation_solar_forecast")
    info = info.with_source_rules([source])
    for name in (
        "generation_solar_forecast_d+1",
        "generation_solar_forecast_h12_d+1",
        "generation_solar_forecast_d+1_day_max",
    ):
        col = info.resolve(name)
        assert (col.layer, col.key, col.rule) == ("sources", "generation_solar_forecast", source.availability()), name
        known, issued, skipped, _ = library_known(
            info, name, frame_index("Europe/Berlin", "1h", "spring"), "1h", "spring"
        )
        assert (known > np.broadcast_to(issued, known.shape))[~skipped].all(), name


@pytest.mark.parametrize(
    "key",
    [
        "generation_solar_forecast",
        "generation_solar_forecast_d+1",
        "generation_solar_forecast_h05",
        "generation_solar_forecast_day_max",
        "generation_solar_forecast_day_max_d+1",
        "generation_solar_forecast_d-1_h05",
    ],
)
def test_a_rule_naming_a_chained_column_is_refused(tmp_path, key):
    upstream(tmp_path, "Europe/Berlin", "Europe/Brussels", name="solar_ridge")
    source = ForecastSource(tmp_path, "solar_ridge", "generation_solar_forecast")
    info = InformationSet("10:15", "Europe/Brussels", rules={key: Availability.published("08:00", 1)})
    with pytest.raises(
        InformationSetError,
        match="names 'generation_solar_forecast' .or a column built from it., a column of forecasts",
    ):
        info.with_source_rules([source])
    added_later = InformationSet("10:15", "Europe/Brussels").with_source_rules([source])
    added_later.rules[key] = Availability.published("08:00", 1)
    with pytest.raises(InformationSetError, match="nothing replaces it"):
        added_later.refuse_rules_on_chained_columns()
    assert added_later.resolve("generation_solar_forecast_d+1").layer == "sources"  # and ignored meanwhile
