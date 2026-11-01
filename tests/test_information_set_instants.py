"""The information set compares instants. An oracle on UTC datetimes, which shares no arithmetic with the code, says when
each cell is known, and the check, the `ctx` of a predictor function and the probe must all say the same, cell by cell."""

import re
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.transformers import LagTransformer
from epftoolbox2.information_set import (
    ROLE_PREDICTOR,
    Availability,
    InformationSet,
    InformationSetError,
    _available_array,
    _Context,
    leakage_probe,
    perturb_recent,
    perturb_unknown,
)
from epftoolbox2.models import OLSModel
from epftoolbox2.models.context import _Cells
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.availability import listing_rows
from tests.synthetic import PREDICTORS, make_market_data

UTC = timezone.utc
MINUTE = timedelta(minutes=1)
SPRING = date(2024, 3, 31)  # Warsaw: 02:00 CET -> 03:00 CEST, at 01:00Z
AUTUMN = date(2024, 10, 27)  # Warsaw: 03:00 CEST -> 02:00 CET, at 01:00Z

RULES = {
    "pub": Availability.published("13:00", 1),
    "lead": Availability.published("10:00", 1, timezone="America/New_York"),
    "after": Availability.published("00:00", -1, timezone="Etc/GMT+1"),
    "close": Availability.published("03:15", 0, timezone=None),
    "week": Availability.weekly("friday", "10:00"),
    "act": Availability.realised("1h"),
    "slow": Availability.realised("11h"),
    "run": Availability.weather_run("6h"),
    "asof": Availability.as_of("11:00", "Europe/Warsaw"),
    "local": Availability.as_of("02:30", None),
    "cut": Availability.weather_cutoff("00", "6h"),
    "cal": Availability.static(),
}
SUFFIXES = [
    "",
    "_d-1",
    "_d+1",
    "_d+2",
    "_h-1",
    "_h-24",
    "_h+8",
    "_min-30",
    "_d+1_d-1",
    "_d+1_h-24",
    "_h-24_d+1",
    "_h05",
    "_h02",
    "_h02_d-1",
    "_d+1_h05",
    "_day_max",
    "_day_max_d-1",
    "_d-1_day_max",
    "_h-24_day_max",
    "_h-1_h02",
]
NAMES = [f"{base}{suffix}" for base in RULES for suffix in SUFFIXES]
# The check, `ctx` and the probe read through one path, so fewer names do, on rows the data has.
AGREEMENT_NAMES = [
    f"{base}{suffix}"
    for base in ("pub", "close", "week", "act", "run", "asof", "cut")
    for suffix in ("", "_d+1", "_h-1", "_h+8", "_d+1_h-24", "_h02_d-1", "_h-24_day_max")
]
QUARTER_NAMES = [f"{base}{suffix}" for base in ("pub", "act", "run", "cut") for suffix in ["", "_h-1", "_d+1", "_s09", "_s09_d-1", "_day_max", "_h-24_day_max", "_d+1_h-24"]]  # fmt: skip

WARSAW_DAYS = [SPRING + timedelta(days=k) for k in (-2, -1, 0, 1, 2)] + [AUTUMN + timedelta(days=k) for k in (-2, -1, 0, 1, 2)]  # fmt: skip
NEW_YORK_DAYS = [date(2024, 3, 9), date(2024, 3, 10), date(2024, 3, 11), date(2024, 11, 2), date(2024, 11, 3), date(2024, 11, 4)]  # fmt: skip
HAVANA_DAYS = [date(2024, 3, 9), date(2024, 3, 10), date(2024, 3, 11), date(2024, 11, 2), date(2024, 11, 3), date(2024, 11, 4)]  # fmt: skip  # at midnight
ASUNCION_DAYS = [date(2024, 3, 22), date(2024, 3, 23), date(2024, 3, 24)]  # 23:00 repeats on the 23rd

CASES = {
    "warsaw 1h": ("Europe/Warsaw", "1h", "11:45", "Europe/Brussels", WARSAW_DAYS),
    "warsaw 1h, issue in the switch": ("Europe/Warsaw", "1h", "02:30", "Europe/Warsaw", WARSAW_DAYS),
    "warsaw 1h, issue after the switch": ("Europe/Warsaw", "1h", "03:30", "Europe/Warsaw", WARSAW_DAYS),
    "warsaw 15min": ("Europe/Warsaw", "15min", "11:45", "Europe/Brussels", WARSAW_DAYS),
    "warsaw 15min, issue in the switch": ("Europe/Warsaw", "15min", "02:30", "Europe/Warsaw", WARSAW_DAYS),
    "new york, issue in the switch": ("America/New_York", "1h", "01:30", "America/New_York", NEW_YORK_DAYS),
    "new york, issue in brussels": ("America/New_York", "1h", "11:45", "Europe/Brussels", NEW_YORK_DAYS),
    "havana, a midnight switch": ("America/Havana", "1h", "00:30", "America/Havana", HAVANA_DAYS),
    "asuncion, a repeated last hour": ("America/Asuncion", "1h", "23:30", "America/Asuncion", ASUNCION_DAYS),
    "utc 1h": ("UTC", "1h", "11:45", "Europe/Brussels", WARSAW_DAYS),
    "utc 1h, issue in the switch": ("UTC", "1h", "02:30", "Europe/Brussels", WARSAW_DAYS),
    "utc 15min": ("UTC", "15min", "11:45", "Europe/Brussels", WARSAW_DAYS),
    "fixed offset 1h": ("Etc/GMT-1", "1h", "11:45", "Europe/Brussels", WARSAW_DAYS),
    "fixed offset 15min": (timezone(timedelta(hours=2)), "15min", "02:30", "Europe/Brussels", WARSAW_DAYS),
    "naive 1h": (None, "1h", "11:45", "Europe/Brussels", WARSAW_DAYS),
    "naive 1h, issue in the switch": (None, "1h", "02:30", "Europe/Warsaw", WARSAW_DAYS),
    "naive 15min": (None, "15min", "03:30", "Europe/Warsaw", WARSAW_DAYS),
}


def _zone(name):
    match = re.fullmatch(r"UTC([+-])(\d\d):(\d\d)", str(name))
    if match:
        sign = 1 if match[1] == "+" else -1
        return timezone(sign * timedelta(hours=int(match[2]), minutes=int(match[3])))
    return ZoneInfo(str(name))


def _utc(wall, zone, second=False):
    """The instant of a wall-clock time: its second occurrence if asked, and the switch itself for a time it skips."""
    tz = _zone(zone)
    while True:
        first, last = (wall.replace(tzinfo=tz, fold=fold).astimezone(UTC) for fold in (0, 1))
        if first <= last:
            return last if second else first
        wall = wall + MINUTE  # fold 0 reads a skipped time with the old offset, past the switch, and fold 1 before it


def _is_second(wall, zone, instant):
    return instant != _utc(wall, zone) and instant == _utc(wall, zone, second=True)


def _parse(name, step):
    """(base, shifts inside the slot, the slot as (start, length), whether it is a day aggregate's last, shifts after it);
    each list of shifts has its outermost last, as the name does."""
    match = re.fullmatch(r"([a-z]+)((?:_(?:d|h|min)[+-]\d+)*)(?:_h(\d\d)|_s(\d\d)|(_day_max))?((?:_(?:d|h|min)[+-]\d+)*)", name)  # fmt: skip
    base, inner, hour, quarter, aggregate, outer = match.groups()

    def shifts(text):
        return [(unit, int(n)) for unit, n in re.findall(r"_(d|h|min)([+-]\d+)", text)]

    slot = None
    if hour is not None:
        slot = (int(hour) * 60, 60)
    elif quarter is not None:
        slot = (int(quarter) * 15, 15)
    elif aggregate is not None:
        slot = (1440 - step, step)
    return base, shifts(inner), slot, aggregate is not None, shifts(outer)


class Oracle:
    """When each cell is known, as the docs say it, on datetimes: `_d` keeps the wall clock and the occurrence, `_h` and
    `_min` are elapsed, the outermost shift comes first, and clock times in a rule are read at the instant they denote."""

    def __init__(self, data_zone, naive, step):
        self.zone, self.naive, self.step = data_zone, naive, step

    def _moved(self, wall, second, shifts):
        for unit, n in reversed(shifts):
            minutes = n * {"d": 1440, "h": 60, "min": 1}[unit]
            if self.naive:
                wall = wall + timedelta(minutes=minutes)
            elif unit == "d":
                second = _is_second(wall, self.zone, _utc(wall, self.zone, second))
                wall = wall + timedelta(minutes=minutes)
            else:
                moved = (_utc(wall, self.zone, second) + timedelta(minutes=minutes)).astimezone(_zone(self.zone))
                wall, second = moved.replace(tzinfo=None, fold=0), bool(moved.fold)
        return wall, second

    def known(self, name, issue_day, row_day, minute):
        """The instant the column holds its value at the row (`row_day`, `minute`), read at its second occurrence; None if
        it is known in advance."""
        base, inner, slot, aggregate, outer = _parse(name, self.step)
        wall = datetime.combine(row_day, datetime.min.time()) + timedelta(minutes=minute)
        rule = RULES[base]
        if slot is None:
            return self._cell(rule, inner, wall, True, issue_day, self.step)
        reached = self._moved(wall, True, outer)[0].date()
        issue_day = issue_day + (reached - row_day)
        start = datetime.combine(reached, datetime.min.time()) + timedelta(minutes=slot[0])
        return self._cell(rule, inner, start, aggregate, issue_day, slot[1])

    def _cell(self, rule, shifts, wall, second, issue_day, length):
        if rule.kind == "static":
            return None
        if rule.kind in ("as_of", "weather_cutoff"):
            rest = shifts[1:] if shifts and shifts[0][0] == "d" and shifts[0][1] > 0 else shifts
            read = issue_day + (self._moved(wall, second, rest)[0].date() - wall.date())
            at = datetime.combine(read, datetime.min.time()) + timedelta(minutes=_clock(rule.time))
            known = _utc(at, rule.timezone or self.zone)
            return known + _delay(rule) if rule.kind == "weather_cutoff" else known
        if rule.kind == "weather_run" and shifts and shifts[0][0] == "d" and shifts[0][1] > 0:
            shifts = shifts[1:]
        moved, second = self._moved(wall, second, shifts)
        point = _utc(moved, self.zone, second or self.naive)
        if rule.kind == "realised":
            return point + timedelta(minutes=length) + _delay(rule)
        if rule.kind == "weather_run":
            return point + _delay(rule)
        zone = rule.timezone or self.zone
        delivered = point.astimezone(_zone(zone)).date()
        if rule.kind == "published":
            day = delivered - timedelta(days=rule.days_before)
        else:
            day = delivered - timedelta(days=delivered.weekday() + 7) + timedelta(days=4)  # the Friday before that week
        return _utc(datetime.combine(day, datetime.min.time()) + timedelta(minutes=_clock(rule.time)), zone)


def _clock(text):
    return int(text[:2]) * 60 + int(text[3:])


def _delay(rule):
    return pd.Timedelta(rule.delay).to_pytimedelta()


def _issue_instant(day, issue_time, issue_zone):
    """A repeated issue time is its first occurrence, and a skipped one its earliest reading: the smaller of the two."""
    wall = datetime.combine(day, datetime.min.time()) + timedelta(minutes=_clock(issue_time))
    return min(wall.replace(tzinfo=_zone(issue_zone), fold=fold).astimezone(UTC) for fold in (0, 1))


def _index(tz, freq, days):
    first = pd.Timestamp(days[0]) - pd.Timedelta(days=4)
    last = pd.Timestamp(days[-1]) + pd.Timedelta(days=5)
    if tz is None:
        return pd.date_range(first, last, freq=freq, inclusive="left")
    return pd.date_range(first.tz_localize(tz), last.tz_localize(tz), freq=freq, inclusive="left")


def _setup(case):
    tz, freq, issue_time, issue_zone, days = CASES[case]
    index = _index(tz, freq, days)
    info = InformationSet(issue_time, issue_zone, rules=RULES, unknown="ignore")
    names = QUARTER_NAMES if freq == "15min" else NAMES
    step = 15 if freq == "15min" else 60
    data_zone = str(index.tz) if index.tz is not None else issue_zone
    return index, info, freq, names, step, data_zone, days, issue_time, issue_zone


def _wall(index):
    return index.tz_localize(None) if index.tz is not None else index


def _period_of(index, step):
    wall = _wall(index)
    return np.asarray(wall.hour * 60 + wall.minute) // step


def _frame(index, names):
    return pd.DataFrame(
        np.arange(1.0, len(index) + 1.0)[:, None] * np.ones((1, len(names))), index=index, columns=names
    )


@pytest.mark.parametrize("case", list(CASES))
def test_every_cell_is_known_when_the_oracle_says_in_the_check_and_in_ctx(case):
    index, info, freq, names, step, data_zone, days, issue_time, issue_zone = _setup(case)
    oracle = Oracle(data_zone, index.tz is None, step)
    origin = _utc(datetime.combine(days[0], datetime.min.time()), data_zone)
    issue = [_issue_instant(day, issue_time, issue_zone) for day in days]
    cells = _Cells(pd.DataFrame(index=index), info, freq, pd.DatetimeIndex(days))
    assert cells._clock.issue.tolist() == [(at - origin) / MINUTE for at in issue]

    wrong = []
    for name in names:
        available = cells.available(name)
        known = cells.known(name)
        for j, day in enumerate(days):
            for period in range(1440 // step):
                at = oracle.known(name, day, day, period * step)
                minutes = -np.inf if at is None else (at - origin) / MINUTE
                if available[period, j] != minutes or known[period, j] != (at is None or at <= issue[j]):
                    wrong.append((name, str(day), period, float(available[period, j]), float(minutes)))
    assert wrong == []


@pytest.mark.parametrize("case", list(CASES))
def test_ctx_the_check_and_the_probe_agree_cell_by_cell(case):
    """The grid of a whole window, the check of each day alone and the perturbation of each day's rows: a row the data has
    is noise exactly where the other two call it late, on every issue day."""
    index, info, freq, names, step, _, days, *_ = _setup(case)
    names = names if freq == "15min" else AGREEMENT_NAMES
    periods = 1440 // step
    frame = _frame(index, names)
    cells = _Cells(frame, info, freq, pd.DatetimeIndex(days))
    expanded = {(period, 1): names for period in range(periods)}
    grid = np.stack([cells.known(name) for name in names])
    local = _wall(index)

    for j, day in enumerate(days):
        found, _ = info.violations(expanded, "cal", freq, index, issue_days=[pd.Timestamp(day)])
        late = {(use.predictor, use.period) for use in found if use.role == ROLE_PREDICTOR}
        from_check = {(name, p) for i, name in enumerate(names) for p in range(periods) if not grid[i, p, j]}
        assert late == from_check, day

        window = frame[np.abs((local.normalize() - pd.Timestamp(day)) / pd.Timedelta(days=1)) <= 4]
        perturbed, _, _ = perturb_unknown(window, str(day), info, freq)
        on_day = _wall(window.index).normalize() == pd.Timestamp(day)
        # The last row of each period, what the model forecasts from.
        rows = pd.Series(np.flatnonzero(on_day), index=_period_of(window.index[on_day], step)).groupby(level=0).last()
        moved = perturbed.to_numpy()[rows.to_numpy()] != window.to_numpy()[rows.to_numpy()]
        assert moved.T.tolist() == (~grid[:, rows.index.to_numpy(), j]).tolist(), day


def test_the_clock_counts_each_days_real_length():
    """Elapsed minutes from the first issue day's midnight, and each day's own midnight on the same scale."""
    info = InformationSet("11:45", "Europe/Warsaw")
    index = pd.date_range("2024-10-20", "2024-11-02", freq="1h", tz="Europe/Warsaw")
    clock = info._issue_clock(pd.date_range("2024-10-26", "2024-10-28"), index)
    assert clock.midnight.tolist() == [0.0, 1440.0, 1440.0 + 1500.0]
    # 11:45 is 11:45 elapsed, then 12:45 on the 25-hour day, where the clocks went back before it.
    assert clock.issue.tolist() == [705.0, 1440.0 + 765.0, 1440.0 + 1500.0 + 705.0]
    assert clock.minute_of_day(clock.issue[2], 2) == 705.0
    spring = info._issue_clock(pd.date_range("2024-03-30", "2024-04-01"), index)
    assert spring.midnight.tolist() == [0.0, 1440.0, 1440.0 + 1380.0]


@pytest.mark.parametrize("day", ["2024-10-26", "2024-10-27", "2024-10-28", "2024-03-30", "2024-03-31", "2024-04-01"])
def test_a_time_reads_as_its_clock_on_every_day_of_a_window(day):
    """`load_forecast_d+2` is known at 10:00 on D+1, on every day. Across a window that starts a switch earlier, the label
    must count from that day's own midnight, or it reads an hour off after the switch."""
    first = "2024-10-24" if day.startswith("2024-10") else "2024-03-28"
    index = pd.date_range("2024-03-20", "2024-11-05", freq="1h", tz="Europe/Warsaw")
    days = pd.date_range(first, day)
    cells = _Cells(pd.DataFrame(index=index), InformationSet("11:45", "Europe/Warsaw"), "1h", days)
    assert cells.known_at("load_forecast_d+2", 0, len(days) - 1) == "D+1 10:00"


def test_the_worst_day_is_the_one_with_the_most_elapsed_time():
    """`load_forecast_d+2` is known 10:00 on D+1, after an 11:45 issue on D: 22 h 15 of elapsed time on an ordinary day, 23 h 15
    on the day before the 25-hour day and 21 h 15 on the day before the 23-hour one. The clock reads the same on all of them."""
    index = pd.date_range("2024-03-20", "2024-11-05", freq="1h", tz="Europe/Warsaw")
    info = InformationSet("11:45", "Europe/Warsaw")

    def latest(start, end):
        found, _ = info.violations({(12, 1): ["load_forecast_d+2"]}, "load_actual", "1h", index, start, end)
        return next(use for use in found if use.predictor == "load_forecast_d+2")

    assert (use := latest("2024-06-03", "2024-06-05")).available - use.issue == 22 * 60 + 15
    assert (use := latest("2024-03-30", "2024-03-30")).available - use.issue == 21 * 60 + 15
    assert (use := latest("2024-03-28", "2024-04-02")).available - use.issue == 22 * 60 + 15
    assert (use := latest("2024-10-24", "2024-10-28")).available - use.issue == 23 * 60 + 15
    assert (use.available_label, use.issue_label) == ("D+1 10:00", "D 11:45")
    row = listing_rows([use])[0]
    assert (row["known"], row["issued"], row["minutes_late"]) == ("D+1 10:00", "D 11:45", 23 * 60 + 15)


def test_a_repeated_hour_read_in_a_zone_whose_midnight_falls_between_its_occurrences():
    """Warsaw's 02:00 repeats at 00:00Z and 01:00Z on 2024-10-27, and midnight in UTC-01:00 falls at 01:00Z. A value published
    13:00 on its delivery day in that zone is delivered on the 26th at the first 02:00, on the 27th at the second, the row
    the model forecasts from: 13:00 there is 14:00Z, after an 11:45 Warsaw issue. Read at the first it passed."""
    info = InformationSet(
        "11:45", "Europe/Warsaw", rules={"x": Availability.published("13:00", 0, timezone="Etc/GMT+1")}
    )
    index = pd.date_range("2024-10-20", "2024-11-02", freq="1h", tz="Europe/Warsaw")
    found, _ = info.violations({(p, 1): ["x"] for p in range(24)}, "x_d-3", "1h", index, "2024-10-27", "2024-10-27")
    assert {use.period for use in found if use.role == ROLE_PREDICTOR} == set(range(2, 24))


def test_a_day_shift_between_two_switch_days_keeps_the_occurrence():
    """2023-10-29 and 2024-10-27 both repeat 02:00, 364 days apart, and LagTransformer maps each occurrence to its own. So
    the two rows of x_d+364 hold values delivered an hour apart, and are known an hour apart."""
    index = pd.date_range("2023-10-20", "2024-11-05", freq="1h", tz="Europe/Warsaw")
    raw = pd.DataFrame({"x": (index - pd.Timestamp("2023-01-01", tz="UTC")) / pd.Timedelta(minutes=1)}, index=index)
    built = LagTransformer(columns=["x"], lags=[-364], freq="day").transform(raw)
    held = built.loc[built.index.tz_localize(None) == pd.Timestamp("2023-10-29 02:00"), "x_d+364"].to_numpy()
    assert held[1] - held[0] == 60.0  # the transformer's own reading

    rule = Availability.realised("0min")
    col = InformationSet("11:45", "Europe/Warsaw", rules={"x": rule}).resolve("x_d+364")
    ctx = _Context(pd.Timestamp("2023-10-29"), "Europe/Warsaw")
    known = _available_array(col, rule, np.zeros(2), np.array([120.0, 120.0]), 60, ctx, np.array([True, False]))
    midnight = (ctx.origin - pd.Timestamp("2023-01-01", tz="UTC")) / pd.Timedelta(minutes=1)
    assert (known + midnight).tolist() == (held + 60.0).tolist()


def test_an_hour_lag_inside_a_day_aggregate_is_elapsed_time():
    """x_h-24_day_max on the 25-hour 2024-10-27: at its last slot, 23:00 CET (22:00Z), the lag reads 22:00Z the day before,
    00:00 CEST on the 27th itself, so the as_of value comes from D and is known at 12:00, after the 11:45 issue. Counting
    the 24 hours on the wall clock read it from D-1. The next day has no such hour."""
    rules = {"x": Availability.as_of("12:00", "Europe/Warsaw")}
    info = InformationSet("11:45", "Europe/Warsaw", rules=rules)
    index = pd.date_range("2024-10-20", "2024-11-02", freq="1h", tz="Europe/Warsaw")
    expanded = {(p, 1): ["x_h-24_day_max"] for p in range(24)}
    for day, late in (("2024-10-26", set()), ("2024-10-27", set(range(24))), ("2024-10-28", set())):
        found, _ = info.violations(expanded, "x_d-3", "1h", index, day, day)
        assert {use.period for use in found if use.role == ROLE_PREDICTOR} == late


def test_each_occurrence_of_a_repeated_hour_is_perturbed_at_its_own_instant():
    """Issued 02:30 Warsaw on 2024-10-27, read at its first occurrence (00:30Z). A weather run published at its row time is
    out at the first 02:00 (00:00Z) and not at the second (01:00Z). The row the model forecasts from is the second, and the
    check calls it late."""
    index = pd.date_range("2024-10-25", "2024-10-30", freq="1h", tz="Europe/Warsaw")
    frame = pd.DataFrame({"x": np.arange(1.0, len(index) + 1.0)}, index=index)
    info = InformationSet("02:30", "Europe/Warsaw", rules={"x": Availability.weather_run("0min")})
    perturbed, _, _ = perturb_unknown(frame, "2024-10-27", info)
    moved = index[perturbed["x"] != frame["x"]]
    assert moved[0] == pd.Timestamp("2024-10-27 01:00Z").tz_convert("Europe/Warsaw")
    found, _ = info.violations({(2, 1): ["x"]}, "x_d-3", "1h", index, "2024-10-27", "2024-10-27")
    assert [(use.predictor, use.period) for use in found] == [("x", 2)]


@pytest.mark.parametrize(
    "run_day, column, row, fresh",
    [
        # The day before D has 25 hours. Its first hour ends 1440 minutes before D's midnight, 60 after its own.
        ("2024-10-28", "x", "2024-10-27 00:00", True),
        ("2024-10-28", "x_d-1", "2024-10-27 23:00", False),
        # The day before D has 23 hours. x_d-1 at its last row holds the hour that ended at its midnight, 1380 minutes back.
        ("2024-04-01", "x", "2024-03-31 00:00", True),
        ("2024-04-01", "x_d-1", "2024-03-31 23:00", False),
    ],
)
def test_the_control_reaches_back_to_the_real_midnight_of_the_day_before(run_day, column, row, fresh):
    index = pd.date_range("2024-03-20", "2024-11-05", freq="1h", tz="Europe/Warsaw")
    frame = pd.DataFrame({column: np.arange(1.0, len(index) + 1.0)}, index=index)
    info = InformationSet("11:45", "Europe/Warsaw", rules={"x": Availability.realised("0min")})
    perturbed, _, _ = perturb_recent(frame, run_day, info)
    stamp = pd.Timestamp(row, tz="Europe/Warsaw")
    assert bool(perturbed.loc[stamp, column] != frame.loc[stamp, column]) == fresh


def test_a_value_known_across_a_spring_switch_counts_its_period_and_delay_as_elapsed_time():
    """Issued 03:30 Warsaw on 2024-03-31 (01:30Z). Hour 1, 01:00 CET (00:00Z), ends at 01:00Z and is known at 02:00Z: shown as
    04:00. As a wall clock it was 03:00, before the issue."""
    index = pd.date_range("2024-03-25", "2024-04-05", freq="1h", tz="Europe/Warsaw")
    info = InformationSet("03:30", "Europe/Warsaw")
    expanded = {(1, 1): ["load_actual"]}
    with pytest.raises(InformationSetError, match=r"is known D 04:00, issued D 03:30"):
        info.check(expanded, "load_actual_d-3", "1h", index, "2024-03-31", "2024-03-31")


def test_a_delay_across_the_repeated_hour_is_elapsed_time():
    """Issued 11:45 Warsaw on 2024-10-27 (10:45Z). Period 0, 00:00 CEST (22:00Z the day before), through a 12-hour delay is
    known at 11:00Z, 12:00 on the clock. Added to the wall clock it read 13:00, an hour later than it is."""
    index = pd.date_range("2024-10-20", "2024-11-02", freq="1h", tz="Europe/Warsaw")
    info = InformationSet("11:45", "Europe/Warsaw", rules={"x": Availability.realised("12h")})
    found, _ = info.violations({(0, 1): ["x"]}, "x_d-3", "1h", index, "2024-10-27", "2024-10-27")
    assert [(use.available_label, use.issue_label, use.available - use.issue) for use in found] == [
        ("D 12:00", "D 11:45", 15.0)
    ]


def _load_ahead_if_out(ctx):
    base = ["load_d+1", "wind_d+1", "is_weekend_d+1", "is_weekday_d+1"]
    return [*base, "load_d-1"] if ctx.published("load_actual_h+8") else base


def test_a_predictor_function_answers_at_the_real_instant():
    """Issued 11:45 Warsaw on 2024-03-31 (09:45Z). load_actual_h+8 at 00:00 (23:00Z) holds 07:00Z, which ends 08:00Z and is
    known 09:00Z: in time. At 01:00 (00:00Z) it holds 08:00Z, known 10:00Z: late, though the wall clock said 11:00. So the
    function's list changes between these two periods on this day alone: on an ordinary day 01:00 is known at 11:00."""
    data = make_market_data(start="2024-02-20", days=60, noise=2.0, seed=3).assign(**{"load_actual_h+8": 1.0})
    info = InformationSet("11:45", "Europe/Warsaw")
    model = OLSModel(predictors=_load_ahead_if_out, training_window=14, name="m")
    plan = model._plan_predictors(
        data, "price", 1, "1h", info, "2024-03-30", "2024-04-01", warn_unpublished=False, availability="none"
    )
    days = [str(day) for day in plan.issue_days]
    with_load = {
        day: [period for period in range(5) if "load_d-1" in plan.lists[plan.assignment[(period, 1)][i]]]
        for i, day in enumerate(days)
    }
    assert with_load == {"2024-03-30": [0, 1], "2024-03-31": [0], "2024-04-01": [0, 1]}


def test_the_probe_sees_a_look_ahead_the_wall_clock_let_through(monkeypatch):
    """The same cell through a real model: at 01:00 on 2024-03-31 load_actual_h+8 is known after the issue (see above), so
    noise goes there and the forecast moves. On the wall clock it was known, unperturbed, and the probe said nothing. At
    00:00 it is known and stays."""
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    data = make_market_data(start="2024-02-20", days=45, noise=2.0, seed=3)
    data["load_actual_h+8"] = data["load"].shift(-8)
    rules = {name: Availability.static() for name in ("load", "wind", "is_weekend", "is_weekday")}
    info = InformationSet("11:45", "Europe/Warsaw", rules=rules)
    model = OLSModel(predictors=[*PREDICTORS, "load_actual_h+8"], training_window=14)
    probe = leakage_probe(model, data, "2024-03-31", info, columns=["load_actual_h+8"])
    assert (1, 1) in probe.changed and (0, 1) not in probe.changed


def test_a_repeated_hour_is_shown_with_its_offset_in_explain():
    """Hour 1, 01:00 CEST on 2024-10-27 (23:00Z the day before), ends at 00:00Z and is known at 01:00Z: the second 02:00. In
    spring hour 1 (00:00Z) is known at 02:00Z, 04:00 CEST."""
    data = make_market_data(start="2024-01-01", days=320, noise=2.0, seed=3).assign(load_actual=1.0)
    pipeline = ModelPipeline(information_set=InformationSet("11:45", "Europe/Warsaw"), availability="warn")
    pipeline.add_model(OLSModel(predictors=["load_actual"], training_window=14, name="m"))
    for day, expected in (("2024-10-27", "D 02:00 (UTC+01:00)"), ("2024-03-31", "D 04:00")):
        tables = pipeline.explain(data, model="m", test=(day, day), horizon=1, hour=1, target="price", echo=False)
        assert tables["predictors"]["known"].tolist() == [expected]


def test_explain_reads_what_a_name_holds_from_the_second_occurrence():
    """The row of hour 2 on 2024-10-27 the model forecasts from is the second 02:00 (01:00Z). An hour before it is 00:00Z, the
    first 02:00: T-1 02:00 at horizon 1, on the wall clock. From the first 02:00 it would read 01:00."""
    data = make_market_data(start="2024-08-01", days=120, noise=2.0, seed=3).assign(**{"load_actual_h-1": 1.0})
    pipeline = ModelPipeline(information_set=InformationSet("11:45", "Europe/Warsaw"), availability="warn")
    pipeline.add_model(OLSModel(predictors=["load_actual_h-1"], training_window=14, name="m"))
    day = "2024-10-27"
    tables = pipeline.explain(data, model="m", test=(day, day), horizon=1, hour=2, target="price", echo=False)
    assert tables["predictors"]["target-relative"].tolist() == ["T-1 02:00"]
