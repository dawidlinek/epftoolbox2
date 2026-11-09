"""Hour aggregates (`PeriodStatsTransformer`'s `_hour_mean`, `_hour_dev`) and the day's peak (`_day_peak`) in the information set.

An hour aggregate holds a statistic of the column it was built from over every slot of its row's hour, so it is known when the
hour's last slot is (10:45 for 10:00 to 11:00 on 15-minute data), on the day the shifts after it reach, in the occurrence of a
repeated autumn hour that its row is in. A deviation holds the row's own value too, one of the hour's slots. The day's peak is a
day aggregate, judged at the day's last slot like `_day_max`.

The known-answer oracle builds the data from its own instants: `px` holds the minutes since the epoch of its row, so the latest
delivery an hour aggregate holds is its hour's last slot, read off its value (the mean of the hour's slots plus half the hour
less one slot). The probe must perturb exactly the cells delivered after the issue time, and the check refuse exactly those
(predictor, issue day) pairs, over four data zones, both switches, hourly and 15-minute data, under a published and a realised
rule.
"""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.transformers import (
    DailyStatsTransformer,
    LagTransformer,
    PeriodStatsTransformer,
    ResampleTransformer,
)
from epftoolbox2.information_set import (
    _STEP_MINUTES,
    Availability,
    InformationSet,
    InformationSetError,
    _built_from_name,
    _Context,
    _delivered_span,
    perturb_unknown,
)
from epftoolbox2.models._calendar import periods_per_day

ISSUE = ("11:45", "Europe/Brussels")


def verdicts(info: InformationSet, name: str, index: pd.DatetimeIndex, freq: str, days) -> np.ndarray:
    """[period, issue day] -> the instant the check says the value of `name` on the row of the issue day is known, in minutes
    since the first issue day's midnight in the data zone."""
    clock = info._issue_clock(pd.DatetimeIndex(pd.to_datetime(days)), index, freq)
    step = _STEP_MINUTES[freq]
    return clock.available(info._resolve_for(name, step), np.arange(periods_per_day(freq)), step)


def quarter_hours(days: int = 30, zone: str = "Europe/Warsaw", start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=96 * days, freq="15min", tz=zone)


# --------------------------------------------------------------------------- names


def test_hour_aggregate_names():
    info = InformationSet(rules={"x*": Availability.static()})
    for name in [
        "x_hour_mean",
        "x_hour_dev",
        "x_hour_mean_d-1",
        "x_d-1_hour_dev",
        "x_h05_hour_mean",
        "x_day_max_hour_dev",
    ]:
        col = info.resolve(name)
        assert col.hour_aggregate and not col.day_aggregate and col.built_from is not None, name
    for name in ["x", "x_hour", "x_hour_median", "x_hours_mean", "x_hour_mean_h05", "x_hourly_mean", "hour_mean"]:
        assert not info.resolve(name).hour_aggregate, name
    assert _built_from_name("load_actual_d-1_hour_mean") == "load_actual_d-1"
    assert _built_from_name("price_hour_dev") == "price"
    for name in ["x_day_peak", "x_day_peak_d-1", "x_d-1_day_peak", "x_hour_mean_day_peak"]:
        assert info.resolve(name).day_aggregate, name
    assert not info.resolve("is_day_peak").day_aggregate  # Open-Meteo's is_day, as for the other statistics


# --------------------------------------------------------------------------- the spec's oracle


def test_a_deviation_at_10_15_is_known_with_the_10_45_slot_realised_or_with_the_day_ahead_forecast():
    """`price_hour_dev` at 10:15 of D holds the price at 10:15 less the mean of 10:00 to 10:45. Realised 30 minutes after its
    quarter-hour ends, it is known at 11:30 on D, when the 10:45 slot is, not at 10:45 when its own is; published 13:00 on
    D-1 (a day-ahead price), the day before."""
    index = quarter_hours()
    realised = InformationSet(*ISSUE, rules={"price": Availability.realised("30min")})
    published = InformationSet(*ISSUE, rules={"price": Availability.published("13:00", 1)})
    days = ["2024-01-10"]
    at_1015, at_1045 = 41, 43
    for name in ("price_hour_dev", "price_hour_mean"):
        known = verdicts(realised, name, index, "15min", days)[:, 0]
        assert known[40] == known[at_1015] == known[at_1045] == 11 * 60 + 30  # 10:45 + 15 min + 30 min
        assert known[44] == 12 * 60 + 30
        assert verdicts(published, name, index, "15min", days)[at_1015, 0] == -11 * 60  # 13:00 on D-1
    plain = verdicts(realised, "price", index, "15min", days)[:, 0]
    assert plain[at_1015] == 11 * 60  # its own slot: half an hour earlier than the aggregate's


def test_a_lead_after_the_aggregate_reads_d_plus_1_and_one_before_it_too():
    """`x_hour_mean_d+1` is the hour mean of D+1, judged on D+1: the day-ahead prices of D+1 are out at 13:00 on D, after the
    11:45 issue, at every period; the hour means of D are out at 13:00 on D-1."""
    index = quarter_hours()
    info = InformationSet(*ISSUE, rules={"price": Availability.published("13:00", 1)})
    for name, known in [
        ("price_hour_mean_d+1", 13 * 60),
        ("price_d+1_hour_mean", 13 * 60),
        ("price_hour_dev_d+1", 13 * 60),
        ("price_hour_mean", -11 * 60),
        ("price_hour_mean_d-1", -35 * 60),
    ]:
        assert (verdicts(info, name, index, "15min", ["2024-01-10"]) == known).all(), name


def test_an_hour_aggregate_of_a_realised_series_is_refused_from_its_hour_s_first_slot():
    """Look-ahead. Realised 1 h after its slot, the hour from 10:00 is known at 12:00 on D (its 10:45 slot), after the 11:45
    issue: refused from period 40 (10:00) on, while the plain column is refused only from 10:45. Judged at the row's slot or
    the hour's first, periods 40 to 42 would pass."""
    index = quarter_hours()
    info = InformationSet(*ISSUE, rules={"load_actual*": Availability.realised("1h")})
    names = ["load_actual_hour_mean", "load_actual_hour_dev", "load_actual"]
    expanded = {(period, 1): names for period in range(96)}
    found, unknown = info.violations(expanded, "price", "15min", index, "2024-01-10", "2024-01-12")
    refused = {}
    for v in found:
        refused.setdefault(v.predictor, set()).add(v.period)
    assert unknown == set()
    assert refused == {
        "load_actual_hour_mean": set(range(40, 96)),
        "load_actual_hour_dev": set(range(40, 96)),
        "load_actual": set(range(43, 96)),
    }
    with pytest.raises(InformationSetError, match="load_actual_hour_dev"):
        info.check({(41, 1): ["load_actual_hour_dev"]}, "price", "15min", index, "2024-01-10", "2024-01-10")


def test_the_probe_perturbs_every_row_of_an_hour_whose_last_slot_is_unknown():
    index = pd.date_range("2024-01-08", "2024-01-13", freq="15min", tz="Europe/Warsaw", inclusive="left")
    raw = pd.DataFrame({"load_actual": np.arange(len(index), dtype=float)}, index=index)
    data = PeriodStatsTransformer("load_actual").transform(raw).drop(columns="load_actual")
    info = InformationSet(*ISSUE, rules={"load_actual*": Availability.realised("1h")})
    perturbed, _, unchecked = perturb_unknown(data, "2024-01-10", info, freq="15min")
    assert unchecked == []
    wall = index.tz_localize(None)
    on_day = np.asarray(wall.normalize() == pd.Timestamp("2024-01-10"))
    changed = (perturbed != data).loc[on_day]
    hour = wall[on_day].hour
    assert changed.loc[hour >= 10].all().all() and not changed.loc[hour < 10].any().any()


def test_each_occurrence_of_the_repeated_autumn_hour_is_its_own_hour():
    """Warsaw repeats 02:00 to 02:59 on 2024-10-27: its first occurrence ends at 01:00Z, its second at 02:00Z. Issued at
    01:30Z, the hour mean of the first is known and that of the second is not: the probe perturbs the second occurrence's four
    rows alone (and every later row), and the check, which reads a period at its second occurrence (the row a model forecasts
    from), refuses from period 8 on."""
    index = pd.date_range("2024-10-25", "2024-10-30", freq="15min", tz="Europe/Warsaw", inclusive="left")
    raw = pd.DataFrame({"x": np.arange(len(index), dtype=float)}, index=index)
    data = PeriodStatsTransformer("x").transform(raw).drop(columns="x")
    info = InformationSet("01:30", "UTC", rules={"x": Availability.realised("0min")})
    perturbed, _, _ = perturb_unknown(data, "2024-10-27", info, freq="15min")
    changed = (perturbed != data).any(axis=1)
    utc = index.tz_convert("UTC")
    assert list(utc[changed.to_numpy()][:5]) == list(
        pd.date_range("2024-10-27 01:00", periods=5, freq="15min", tz="UTC")
    )
    assert not changed[utc < pd.Timestamp("2024-10-27 01:00", tz="UTC")].any()
    expanded = {(period, 1): ["x_hour_mean"] for period in range(96)}
    found, _ = info.violations(expanded, "price", "15min", index, "2024-10-27", "2024-10-27")
    assert {v.period for v in found} == set(range(8, 96))


def test_day_peak_is_judged_at_the_day_s_last_slot_not_at_the_peak_s():
    """`load_actual_day_peak` averages 08:00 to 19:59, but it is NaN unless the whole day is there, so it is known when the
    day's last slot is: 01:00 on D+1, refused at every period; that of D-1 at none."""
    index = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="Europe/Warsaw")
    info = InformationSet(*ISSUE, rules={"load_actual*": Availability.realised("1h")})
    expanded = {(period, 1): ["load_actual_day_peak", "load_actual_day_peak_d-1"] for period in range(24)}
    found, unknown = info.violations(expanded, "price", "1h", index, "2024-01-10", "2024-01-23")
    assert unknown == set()
    assert {(v.predictor, v.period) for v in found} == {("load_actual_day_peak", p) for p in range(24)}
    assert (verdicts(info, "load_actual_day_peak", index, "1h", ["2024-01-10"]) == 25 * 60).all()


def test_an_hour_aggregate_no_rule_holds_for_is_never_known():
    """A rule that holds from D on: an hour aggregate of D-1 is never known (no rule holds for its slots), one of D as the
    rule says, at the hour's last slot."""
    index = quarter_hours()
    rule = Availability.realised("1h").during(valid_from="2024-01-10")
    info = InformationSet(*ISSUE, rules={"x": rule})
    known = verdicts(info, "x_hour_mean", index, "15min", ["2024-01-10"])[:, 0]
    assert known[40] == 12 * 60 and known[0] == 2 * 60
    assert np.isinf(verdicts(info, "x_hour_mean_d-1", index, "15min", ["2024-01-10"])).all()


def test_under_rules_over_time_an_hour_aggregate_waits_for_every_slot_of_its_hour():
    """In Kolkata (UTC+05:30) the hour from 05:00 spans two UTC days: 05:00 and 05:15 are delivered on the UTC day before,
    05:30 and 05:45 on the next. Under a rule whose lag shortened at that UTC midnight, the hour's first slots come out after
    its last one, so the aggregate is known when every slot is: the latest of the plain column's verdicts over its hour, on
    every period of every issue day. Read at the last slot alone it would pass at a 12:00 issue on 2024-01-09 (06:30Z),
    though 05:00 of 2024-01-10 is out only at 10:00Z."""
    index = pd.date_range("2024-01-01", periods=96 * 20, freq="15min", tz="Asia/Kolkata")
    rules = [
        Availability.published("10:00", 0, timezone="UTC").during(valid_until="2024-01-09"),
        Availability.published("00:00", 1, timezone="UTC").during(valid_from="2024-01-10"),
    ]
    info = InformationSet("12:00", "Asia/Kolkata", rules={"x": rules})
    days = ["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"]
    for shift in ("", "_d+1", "_d-1"):
        plain = verdicts(info, f"x{shift}", index, "15min", days)
        latest = plain.reshape(24, 4, len(days)).max(axis=1).repeat(4, axis=0)
        for stat in ("mean", "dev"):
            assert np.array_equal(verdicts(info, f"x_hour_{stat}{shift}", index, "15min", days), latest), (stat, shift)
    # Minutes since midnight of D in Kolkata (18:30Z the day before): 10:00Z on D is 930, after the 12:00 issue (720), at
    # 05:00 to 05:45; the last slot alone is out at 00:00Z on D, 330.
    late = verdicts(info, "x_hour_mean_d+1", index, "15min", ["2024-01-09"])[20:24, 0]
    assert (late == 930).all()
    assert verdicts(info, "x_d+1", index, "15min", ["2024-01-09"])[23, 0] == 330


def test_on_a_daily_row_an_hour_aggregate_is_judged_over_its_whole_date():
    """Look-ahead. A daily frame of hour aggregates (`ResampleTransformer("1D", agg="max")` of `load_actual_hour_mean`) holds
    every hour of D on its noon-stamped row: realised 1 h after its slot, it is known at 01:00 on D+1 like the day's maximum,
    refused at 11:45 and perturbed on D. Judged at the stamp's hour it passed (known 01:00 on D, the stamp being midnight), and
    `verify` couldn't see it. On a Helsinki daily frame under a Brussels rule, `px_hour_mean_d+1` is refused like `px_d+1`."""
    index = pd.date_range("2024-01-01", periods=96 * 30, freq="15min", tz="Europe/Warsaw")
    frame = pd.DataFrame({"load_actual": np.random.default_rng(0).normal(100, 10, len(index))}, index=index)
    frame = PeriodStatsTransformer("load_actual").transform(frame)
    frame = DailyStatsTransformer("load_actual", stats=["max"], freq="15min").transform(frame)
    daily = ResampleTransformer("1D", agg="max").transform(frame)
    info = InformationSet(*ISSUE, rules={"load_actual*": Availability.realised("1h")})
    names = ["load_actual_hour_mean", "load_actual_hour_dev", "load_actual_day_max", "load_actual_hour_mean_d-1"]
    found, unknown = info.violations({(0, 1): names}, "price", "1D", daily.index, "2024-01-10", "2024-01-12")
    assert unknown == set()
    assert {v.predictor for v in found} == {"load_actual_hour_mean", "load_actual_hour_dev", "load_actual_day_max"}
    known = verdicts(info, "load_actual_hour_mean", daily.index, "1D", ["2024-01-10"])
    assert (known == verdicts(info, "load_actual_day_max", daily.index, "1D", ["2024-01-10"])).all()
    assert (known == 25 * 60).all()
    perturbed, _, _ = perturb_unknown(daily, "2024-01-10", info, freq="1D")
    on_day = np.asarray(daily.index.tz_localize(None).normalize() == pd.Timestamp("2024-01-10"))
    changed = (perturbed != daily).loc[on_day]
    assert changed[["load_actual_hour_mean", "load_actual_hour_dev", "load_actual_day_max"]].all().all()

    helsinki = pd.date_range("2024-01-01", periods=40, freq="D", tz="Europe/Helsinki")
    published = InformationSet(*ISSUE, rules={"px*": Availability.published("13:00", 1, "Europe/Brussels")})
    names = ["px_d+1", "px_hour_mean_d+1", "px_d+1_hour_mean", "px_hour_mean"]
    found, _ = published.violations({(0, 1): names}, "price", "1D", helsinki, "2024-01-20", "2024-01-22")
    assert {v.predictor for v in found} == {"px_d+1", "px_hour_mean_d+1", "px_d+1_hour_mean"}


def test_delivered_span_of_an_hour_aggregate_is_its_hour():
    ctx = _Context(day=pd.Timestamp("2024-01-10"), data_zone="Europe/Warsaw")
    info = InformationSet(rules={"x": Availability.static()})
    day, minute = np.array([0.0, 0.0]), np.array([615.0, 630.0])
    start, end = _delivered_span(info.resolve("x_hour_mean"), day, minute, 15, ctx)
    assert list(start) == [600.0, 600.0] and list(end) == [645.0, 645.0]
    start, end = _delivered_span(info.resolve("x_hour_dev_d-1"), day, minute, 15, ctx)
    assert list(start) == [600.0 - 1440, 600.0 - 1440] and list(end) == [645.0 - 1440, 645.0 - 1440]
    # On a daily row (step 60, stamped at noon) its whole date, as a day aggregate's.
    start, end = _delivered_span(info.resolve("x_hour_mean"), day[:1], np.array([720.0]), 60, ctx, daily=True)
    assert (list(start), list(end)) == ([0.0], [1380.0])
    assert _delivered_span(info.resolve("x_day_max"), day[:1], np.array([720.0]), 60, ctx) == (start, end)


# --------------------------------------------------------------------------- known answer from the data itself


def held_deliveries(zone: str, switch: str, freq: str) -> pd.DataFrame:
    """Hour aggregates of `px` (its row's minutes since the epoch) and their day and hour lags, each cell holding the latest
    delivery it aggregates: the last slot of its hour (NaN where the transformer leaves the hour incomplete)."""
    start = pd.Timestamp(switch) - pd.Timedelta(days=4)
    index = pd.date_range(start, start + pd.Timedelta(days=9), freq=freq, tz=zone, inclusive="left")
    step = _STEP_MINUTES[freq]
    to_last = (60 // step - 1) / 2 * step  # from the mean of an hour's slots to its last
    data = pd.DataFrame({"px": (index.asi8 // 60_000_000_000).astype(float)}, index=index)
    data = LagTransformer(columns="px", lags=[1], freq="day").transform(data)
    data = PeriodStatsTransformer(["px", "px_d-1"], freq=freq).transform(data)
    for column in ("px", "px_d-1"):
        data[f"{column}_hour_dev"] = data[column] - data[f"{column}_hour_dev"] + to_last  # = the mean + to_last
        data[f"{column}_hour_mean"] = data[f"{column}_hour_mean"] + to_last
    aggregates = ["px_hour_mean", "px_hour_dev"]
    data = LagTransformer(columns=aggregates, lags=[1, -1], freq="day").transform(data)
    data = LagTransformer(columns="px_hour_mean", lags=[1], freq="hour").transform(data)
    return data.drop(columns=["px", "px_d-1"])


def wrong_verdicts(data: pd.DataFrame, rule: Availability, switch: str, freq: str) -> dict:
    """Where the probe and the check disagree with the delivery each cell holds, on the four issue days around `switch`. A
    cell is unknown at the 11:45 Brussels issue exactly when its delivery is published (realised) after it; the check judges a
    (predictor, period) by the period's last row that day (the second of a repeated hour), and a column known, or unknown, on
    every row of the day at every period with a row. NaN cells hold no delivery."""
    index = data.index
    step = _STEP_MINUTES[freq]
    info = InformationSet(*ISSUE, rules={"px*": rule})
    held = data.notna().to_numpy()
    delivery = pd.to_datetime(data.to_numpy()[held] * 60, unit="s", utc=True)
    if rule.kind == "realised":
        known = delivery + pd.Timedelta(minutes=step) + pd.Timedelta(rule.delay)
    else:
        day = delivery.tz_convert(rule.timezone).normalize().tz_localize(None)
        known = (day - pd.Timedelta(days=rule.days_before) + pd.Timedelta(rule.time + ":00")).tz_localize(rule.timezone)
    wall = index.tz_localize(None)
    width = data.shape[1]
    on_day = np.repeat(wall.normalize().to_numpy()[:, None], width, axis=1)[held]
    period = np.repeat(np.asarray((wall.hour * 60 + wall.minute) // step)[:, None], width, axis=1)[held]
    names = np.repeat(np.asarray(data.columns)[None, :], len(index), axis=0)[held]
    every = range(1440 // step)
    expanded = {(p, 1): list(data.columns) for p in every}

    wrong = {}
    for run_day in pd.date_range(pd.Timestamp(switch) - pd.Timedelta(days=2), periods=4, freq="D"):
        day = str(run_day.date())
        unknown = np.asarray(known > pd.Timestamp(f"{day} {ISSUE[0]}", tz=ISSUE[1]))
        perturbed, _, _ = perturb_unknown(data, day, info, freq=freq)
        mismatch = (perturbed.to_numpy() != data.to_numpy())[held] != unknown
        if mismatch.any():
            wrong[(day, "probe")] = sorted(set(names[mismatch]))
        found, _ = info.violations(expanded, "price", freq, index, day, day)
        refused = {}
        for v in found:
            refused.setdefault(v.predictor, set()).add(v.period)
        today = on_day == run_day.to_datetime64()
        on_wall = wall[wall.normalize() == run_day]
        with_row = set(((on_wall.hour * 60 + on_wall.minute) // step).tolist())
        for name in np.unique(names[today]):
            chosen = today & (names == name)
            by_period = dict(zip(period[chosen].tolist(), unknown[chosen].tolist()))  # a later row overrides
            got = refused.get(name, set())
            if len(set(by_period.values())) == 1:
                want = with_row if unknown[chosen][0] else set()
            else:
                want, got = {p for p, late in by_period.items() if late}, got & set(by_period)
            if got != want:
                wrong[(day, "check", str(name))] = sorted(got ^ want)[:5]
    return wrong


PUBLISHED = Availability.published("13:00", 1, "Europe/Brussels")
REALISED = Availability.realised("1h")
# Every level of every axis and every pair: (data zone, switch, freq, rule), 8 cells.
GRID = [
    ("Europe/Warsaw", "2024-03-31", "15min", PUBLISHED),
    ("Europe/Warsaw", "2024-10-27", "1h", REALISED),
    ("Europe/Helsinki", "2024-03-31", "1h", PUBLISHED),
    ("Europe/Helsinki", "2024-10-27", "15min", REALISED),
    ("Europe/Lisbon", "2024-03-31", "15min", REALISED),
    ("Europe/Lisbon", "2024-10-27", "1h", PUBLISHED),
    ("UTC", "2024-03-31", "1h", REALISED),
    ("UTC", "2024-10-27", "15min", PUBLISHED),
]


@pytest.mark.parametrize(("zone", "switch", "freq", "rule"), GRID, ids=[f"{z}-{s}-{f}-{r.kind}" for z, s, f, r in GRID])
def test_hour_aggregate_verdicts_match_the_latest_delivery_each_cell_holds(zone, switch, freq, rule):
    data = held_deliveries(zone, switch, freq)
    assert set(data.columns) == {
        "px_hour_mean",
        "px_hour_dev",
        "px_d-1_hour_mean",
        "px_d-1_hour_dev",
        "px_hour_mean_d-1",
        "px_hour_mean_d+1",
        "px_hour_dev_d-1",
        "px_hour_dev_d+1",
        "px_hour_mean_h-1",
    }
    assert data.notna().to_numpy().mean() > 0.9
    assert wrong_verdicts(data, rule, switch, freq) == {}


def test_the_peak_s_verdicts_match_the_latest_delivery_it_holds_on_a_dst_day():
    """`px_day_peak` holds no delivery of its own (a mean of 08:00 to 19:59), but it is whole only with the whole day, so the
    latest delivery it depends on is the day's last slot, as `px_day_max`'s and `px_day_last`'s: the three agree everywhere."""
    for switch in ("2024-03-31", "2024-10-27"):
        start = pd.Timestamp(switch) - pd.Timedelta(days=4)
        index = pd.date_range(start, start + pd.Timedelta(days=9), freq="15min", tz="Europe/Warsaw", inclusive="left")
        data = pd.DataFrame({"px": (index.asi8 // 60_000_000_000).astype(float)}, index=index)
        data = DailyStatsTransformer("px", ["peak", "max"], freq="15min").transform(data)
        data["px_day_peak"] = data["px_day_max"]  # the delivery the peak waits for: the day's last
        data = LagTransformer(["px_day_peak"], [1, -1], "day").transform(data).drop(columns=["px", "px_day_max"])
        for rule in (PUBLISHED, REALISED):
            assert wrong_verdicts(data, rule, switch, "15min") == {}


# --------------------------------------------------------------------------- verify on a model


def test_verify_is_ok_with_its_control_on_hour_aggregates_and_finds_one_known_after_the_issue(monkeypatch):
    """A 15-minute model reading the hour mean and deviations of a day-ahead load forecast for D+1 (out at 10:00 on D) is ok,
    its positive control moving the forecasts; one reading the hour mean of the wind realised on D is a leak at warn and
    refused at raise from the 10:00 hour on (10:45 is out at 12:00, after the 11:45 issue)."""
    from epftoolbox2.models import OLSModel
    from epftoolbox2.pipelines import ModelPipeline
    from tests.synthetic import make_market_data

    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    df = make_market_data(days=30, freq="15min", horizon=1, noise=1.0, seed=3)
    df = PeriodStatsTransformer(["load", "wind"]).transform(df)
    df = LagTransformer(["load_hour_mean", "load_hour_dev"], [-1], "day").transform(df)
    rules = {
        "load": Availability.published("10:00", 1),
        "wind": Availability.realised("1h"),
        "is_week*": Availability.static(),
        "price": Availability.published("13:00", 1),
    }
    info = InformationSet("11:45", "Europe/Warsaw", rules=rules)
    known = ["load_hour_mean_d+1", "load_hour_dev_d+1", "is_weekend_d+1", "is_weekday_d+1"]
    test = ("2023-01-25", "2023-01-28")

    def pipeline(predictors, availability="raise"):
        out = ModelPipeline(information_set=info, availability=availability)
        return out.add_model(OLSModel(predictors, training_window=14, name="m"))

    report = pipeline(known).verify(df, *test, horizon=1, freq="15min", days=2)
    assert report.table().set_index("model").loc["m", "status"] == "ok"
    with pytest.raises(InformationSetError, match="wind_hour_mean"):
        pipeline(known + ["wind_hour_mean"]).run(df, *test, horizon=1, freq="15min")
    report = pipeline(known + ["wind_hour_mean"], availability="warn").verify(
        df, *test, horizon=1, freq="15min", days=2
    )
    assert report.table().set_index("model").loc["m", "status"] == "leak"
