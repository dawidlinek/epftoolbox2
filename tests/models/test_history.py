"""The history accessor (step 6h): what a sequence model reads for a forecast, cut by the information set.

The arrays are held against a reference built by hand from the frame in UTC: the EU's switch rule (the last Sundays of March
and October at 01:00 UTC) written out, and each rule of the step-3 grid (the built-in rules: prices published 13:00 on D-1,
the load forecast 10:00 on D-1, a wind forecast 18:00 on D-1, the actual load 1 h after its period, at the default issue time
11:45 Brussels) as the comparison it makes, with no code of the library's. Any cell the rules put after the issue time set to
1e9 moves no array.

`FakeFoundation` stands in for 6i's foundation models: a zero-shot day unit on the refit calendar whose forecast of each step
ahead is the context's last day (persistence) plus 0.1 x the known-ahead series, through history batches instead of rows. It
runs its units in this process, as a foundation model runs in one process per device (decision 19), with the day unit's
blocks, records and skips. With `newest_target="d-1"`, every model's default, its target's context ends at D-1's last step and
`verify` holds it to that boundary (the target perturbed from D on); with `"published"`, where the rule says.
"""

import dataclasses
import datetime
import io
import json
import warnings
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rich.console import Console

from epftoolbox2.exporters import TerminalExporter
from epftoolbox2.information_set import (
    Availability,
    InformationSet,
    InformationSetError,
    RuleChain,
    StepsReader,
    cuts_after_target_day,
    cuts_at_issue_day,
    perturb_unknown,
    steps_known,
)
from epftoolbox2.models._calendar import block_days
from epftoolbox2.models._history import HistorySpec, history_batch, history_batches, history_spec_of, reads_history
from epftoolbox2.models._worker import (
    SKIP_MISSING_TEST_PREDICTORS,
    TaskInputs,
    _issued,
    _unit_record,
    _worker_init,
    block_records,
    block_units,
)
from epftoolbox2.models.base import BaseModel
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path
from epftoolbox2.results.availability import (
    COLUMNS,
    HISTORY_COLUMNS,
    LATE,
    ON_TIME,
    history_rows,
    listing_frame,
    merge_listings,
    warn_about,
)
from tests.models import adapter_suite as suite

ZONE = "Europe/Warsaw"  # switches with Brussels, the rules' zone: one local date for both
STEP = {"1h": 60, "15min": 15}
CONTEXT = 7
INFO = InformationSet()  # 11:45 Brussels, the built-in rules: the step-3 grid
# A load forecast out two days ahead, for forecasts at horizon 2 that read it in time.
INFO_AHEAD = InformationSet(rules={"load_forecast": Availability.published("10:00", 2, vintage="latest_revision")})
SPRING = [datetime.date(2023, 3, 16) + datetime.timedelta(days=i) for i in range(20)]  # 26 March is the switch
AUTUMN = [datetime.date(2023, 10, 19) + datetime.timedelta(days=i) for i in range(20)]  # 29 October is the switch
DAYS = SPRING + AUTUMN
# Cut where the rules say (decision 12's counts); each test of "d-1" says so.
PRICE = HistorySpec(
    CONTEXT, ("load_forecast",), ("load_actual",), "price", accepts_missing=True, newest_target="published"
)
LOAD = HistorySpec(CONTEXT, ("load_forecast",), ("generation_wind_forecast",), "load_actual", newest_target="published")
SPECS = {"price": PRICE, "load": LOAD}


class InProcessPool:
    """Stands in for the worker pool, as in tests/models/test_day_units.py (no model here uses it: a safeguard)."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        _worker_init(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, task, *args):
        future = Future()
        try:
            future.set_result(task(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)


def market(freq="1h", start="2023-02-01", end="2023-12-01", seed=0):
    """A local frame of the grid's series. Each step's value is its position plus a series' offset and a little noise, so a
    value read from a step one off is another value."""
    index = pd.date_range(pd.Timestamp(start, tz=ZONE), pd.Timestamp(end, tz=ZONE), freq=freq, inclusive="left")
    rng = np.random.default_rng(seed)
    position = np.arange(len(index), dtype=float)
    columns = {"price": 100.0, "load_forecast": 1e4, "generation_wind_forecast": 2e4, "load_actual": 3e4}
    data = {name: offset + position + rng.uniform(0, 0.5, len(index)) for name, offset in columns.items()}
    return pd.DataFrame(data, index=index)


@pytest.fixture(scope="module")
def hourly():
    return market("1h")


@pytest.fixture(scope="module")
def quarter():
    return market("15min")


# ---------------------------------------------------------------------------------------------------------------------------
# The reference, by hand
# ---------------------------------------------------------------------------------------------------------------------------


def last_sunday(year, month):
    end = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
    return end - datetime.timedelta(days=(end.weekday() - 6) % 7)


def offset(utc):
    """The zone's UTC offset at a naive UTC instant: summer time from the last Sunday of March to that of October, 01:00 UTC."""
    start = pd.Timestamp(last_sunday(utc.year, 3)) + pd.Timedelta(hours=1)
    end = pd.Timestamp(last_sunday(utc.year, 10)) + pd.Timedelta(hours=1)
    return pd.Timedelta(hours=2 if start <= utc < end else 1)


def to_utc(day, minute):
    """A local wall time the clocks neither skip nor repeat, as a naive UTC instant."""
    wall = pd.Timestamp(day) + pd.Timedelta(minutes=minute)
    (found,) = [
        wall - pd.Timedelta(hours=h) for h in (1, 2) if offset(wall - pd.Timedelta(hours=h)) == pd.Timedelta(hours=h)
    ]
    return found


def unknown_by_hand(utc, day, freq):
    """{column: whether the value at each step (a naive UTC index of their starts) is out only after 11:45 on `day`}, each rule
    written out: prices 13:00 on the day before their delivery day, the load forecast 10:00 on the day before, the wind
    forecast 18:00 on the day before, the actual load an hour after its period ends."""
    values = utc.to_numpy()
    shift = np.full(len(utc), np.timedelta64(1, "h"))
    for year in set(utc.year):
        start = np.datetime64(pd.Timestamp(last_sunday(year, 3)) + pd.Timedelta(hours=1))
        end = np.datetime64(pd.Timestamp(last_sunday(year, 10)) + pd.Timedelta(hours=1))
        shift[(values >= start) & (values < end)] = np.timedelta64(2, "h")
    local = (values + shift).astype("datetime64[D]")
    issue = np.datetime64(to_utc(day, 11 * 60 + 45))
    d = np.datetime64(day, "D")
    return {
        "price": local > d,
        "load_forecast": local > d + 1,
        "generation_wind_forecast": local > d,
        "load_actual": values + np.timedelta64(STEP[freq] + 60, "m") > issue,
    }


def by_hand(utc_frame, spec, day, horizon, freq):
    """What the accessor must return for one issue day, from the frame in UTC."""
    step = pd.Timedelta(minutes=STEP[freq])
    steps = CONTEXT * (1440 // STEP[freq])
    end = to_utc(day + datetime.timedelta(days=horizon + 1), 0) - step  # the target day's last step

    back = pd.DatetimeIndex([end - i * step for i in range(4 * 1440 // STEP[freq])])  # four days back from the end
    unknown = unknown_by_hand(back, day, freq)

    def newest(name):  # the first known step back from the target day's end: every grid rule is monotone in time
        return back[np.flatnonzero(~unknown[name])[0]]

    cut = newest(spec.target)
    if spec.newest_target == "d-1":  # at the latest D-1's last step
        cut = min(cut, to_utc(day, 0) - step)
    ahead = int((end - cut) / step)
    context = pd.DatetimeIndex([cut - (steps - 1 - i) * step for i in range(steps)])
    later = pd.DatetimeIndex([cut + (i + 1) * step for i in range(ahead)])

    def values(name, instants):
        return utc_frame[name].reindex(instants).to_numpy()

    past = {}
    for name in spec.past:
        held = values(name, context)
        if spec.accepts_missing:
            held[context > newest(name)] = np.nan
        past[name] = held
    target_day = day + datetime.timedelta(days=horizon)
    walls = [u + offset(u) for u in later]  # local wall times
    slots = []
    for i, wall in enumerate(walls):
        period = (wall.hour * 60 + wall.minute) // STEP[freq]
        again = any(w.date() == target_day and (w.hour * 60 + w.minute) // STEP[freq] == period for w in walls[i + 1 :])
        slots.append((target_day, period) if wall.date() == target_day and not again else None)
    late = {name for name in spec.known_ahead if newest(name) < end}
    late |= {name for name in spec.past if not spec.accepts_missing and newest(name) < cut}
    return {
        "target": values(spec.target, context),
        "past": past,
        "known": {name: values(name, context.append(later)) for name in spec.known_ahead},
        "slots": tuple(slots),
        "first": context[0],
        "late": late,
    }


def same(a, b):
    np.testing.assert_array_equal(a, b)  # NaN equals NaN here, and nothing else is loose


def assert_item_is(item, expected):
    same(item.target, expected["target"])
    assert item.past.keys() == expected["past"].keys() and item.known.keys() == expected["known"].keys()
    for name in expected["past"]:
        same(item.past[name], expected["past"][name])
    for name in expected["known"]:
        same(item.known[name], expected["known"][name])
    assert item.slots == expected["slots"]
    assert item.first_step.tz_convert("UTC").tz_localize(None) == expected["first"]
    assert set(item.late) == expected["late"]


def utc_of(frame):
    return frame.tz_convert("UTC").tz_localize(None)


# ---------------------------------------------------------------------------------------------------------------------------
# The arrays
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("freq", ["1h", "15min"])
@pytest.mark.parametrize("which", ["price", "load"])
@pytest.mark.parametrize("horizon", [1, 2])
@pytest.mark.parametrize("newest", ["published", "d-1"])
def test_the_arrays_equal_a_reference_built_by_hand_from_the_utc_frame(hourly, quarter, freq, which, horizon, newest):
    """40 issue days around both switches: the target's context ends at its newest step known at 11:45 (with "d-1" at D-1's
    last step at the latest), the past series on the same steps (NaN after the actual load's newest known step, for a backend
    that takes missing values), the known-ahead series to the end of the target day, and each step ahead's (date, period), the
    repeated hour's second step forecasting it."""
    frame = hourly if freq == "1h" else quarter
    spec = dataclasses.replace(SPECS[which], newest_target=newest)
    batch = history_batch(spec, frame, DAYS, INFO, freq, horizon)
    assert [item.issue_day for item in batch.items] == DAYS
    classes = {"price": "snapshot", "load_forecast": "latest_revision", "generation_wind_forecast": "latest_revision"}
    classes["load_actual"] = "latest_revision"
    assert {series.name: series.vintage for series in batch.series} == {name: classes[name] for name in spec.names}
    utc = utc_of(frame)
    for item in batch.items:
        assert_item_is(item, by_hand(utc, spec, item.issue_day, horizon, freq))


def perturbed_by_hand(frame, day, freq, value=1e9):
    """`frame` with every cell the rules put after the issue time on `day` set to `value`, by the rules written out."""
    out = frame.copy()
    for name, late in unknown_by_hand(utc_of(frame).index, day, freq).items():
        out.loc[late, name] = value
    return out


@pytest.mark.parametrize("freq", ["1h", "15min"])
def test_any_cell_known_after_the_issue_time_set_to_1e9_moves_no_array(hourly, quarter, freq):
    """On each of the 40 days, every cell the rules put after the issue time is 1e9, and every array of that day's item is what
    it was, but that of a series the item lists as late (the load forecast of D+2, out at 10:00 on D+1, which a forecast at
    horizon 2 reads): a run lists that (availability "warn") or refuses it ("raise"). The cells are the very ones
    `perturb_unknown` (the probe) replaces, so the cut and `verify` agree on each."""
    frame = hourly if freq == "1h" else quarter
    days = DAYS
    specs = list(SPECS.values()) + [dataclasses.replace(spec, newest_target="d-1") for spec in SPECS.values()]
    for spec in specs if freq == "1h" else SPECS.values():
        for horizon in (1, 2):
            batch = history_batch(spec, frame, days, INFO, freq, horizon)
            for item in batch.items:
                moved = perturbed_by_hand(frame, item.issue_day, freq)
                (again,) = history_batch(spec, moved, [item.issue_day], INFO, freq, horizon).items
                same(again.target, item.target)
                for name in spec.past:
                    same(again.past[name], item.past[name])
                for name in spec.known_ahead:
                    if name in item.late:
                        assert (
                            horizon == 2
                            and name == "load_forecast"
                            and not np.array_equal(again.known[name], item.known[name])
                        )
                    else:
                        same(again.known[name], item.known[name])
                assert (again.slots, again.newest, again.late.keys()) == (item.slots, item.newest, item.late.keys())
    for day in days:
        moved = perturbed_by_hand(frame, day, freq, value=np.inf)
        probed, _, _ = perturb_unknown(frame, day.isoformat(), INFO, freq)
        same(np.isinf(moved.to_numpy()), probed.to_numpy() != frame.to_numpy())


@pytest.mark.parametrize("freq", ["1h", "15min"])
def test_t_is_fixed_and_s_is_counted_by_hand(hourly, quarter, freq):
    """T is the context's steps on every day, a switch inside the context or not. S: a price issued at 11:45 forecasts the target
    day, 24 steps an hour (23 for the spring target day, 25 for the autumn one), 96 a quarter-hour (92, 100); a load whose
    actuals are out to 10:00 (10:45 at 15 minutes) also forecasts the rest of D: 14 + 24 = 38 steps (53 + 96 = 149)."""
    frame = hourly if freq == "1h" else quarter
    per_hour = 60 // STEP[freq]
    steps = CONTEXT * 24 * per_hour
    rest_of_d = {"1h": 14, "15min": 53}[freq]  # 10:00 to 23:00, or 10:45 to 23:45
    for which, spec in SPECS.items():
        batch = history_batch(spec, frame, DAYS, INFO, freq, 1)
        assert batch.steps == steps
        for item in batch.items:
            assert len(item.target) == steps and all(len(v) == steps for v in item.past.values())
            target_day = item.issue_day + datetime.timedelta(days=1)
            hours = {datetime.date(2023, 3, 26): 23, datetime.date(2023, 10, 29): 25}.get(target_day, 24)
            expected = hours * per_hour + (0 if which == "price" else rest_of_d)  # the switch is before 10:00 on D
            assert item.steps_ahead == expected, (which, item.issue_day)
            assert all(len(v) == steps + expected for v in item.known.values())
            assert sum(slot is not None for slot in item.slots) == hours * per_hour - (per_hour if hours == 25 else 0)
    ordinary = history_batch(LOAD, hourly, [datetime.date(2023, 6, 14)], INFO, "1h", 1).items[0]
    assert ordinary.steps_ahead == 38 and ordinary.slots[:14] == (None,) * 14
    # With "d-1" both targets' contexts end at D-1's last step: D's 24 steps forecast nothing, D+1's 24 do.
    for spec in SPECS.values():
        (item,) = history_batch(dataclasses.replace(spec, newest_target="d-1"), hourly, ["2023-06-14"], INFO).items
        assert item.steps_ahead == 48 and item.slots[:24] == (None,) * 24 and item.newest_minutes[spec.target] == -60.0


def test_the_repeated_hour_is_forecast_by_its_second_step_and_the_skipped_one_by_none(hourly):
    spring, autumn = history_batch(PRICE, hourly, ["2023-03-25", "2023-10-28"], INFO, "1h", 1).items
    assert [slot[1] for slot in spring.slots] == [0, 1, *range(3, 24)]
    assert autumn.slots[2] is None and autumn.slots[3] == (datetime.date(2023, 10, 29), 2)
    assert [slot[1] for slot in autumn.slots if slot is not None] == list(range(24))
    same(autumn.slot_periods, [0, 1, -1, *range(2, 24)])


def test_a_known_ahead_series_not_known_through_the_target_day_is_late(hourly):
    """The wind forecast of D+1 is out at 18:00 on D, after the issue: late as a known-ahead series, with when it is known; as a
    past series of a price it is known as far as the target, D's last hour."""
    spec = HistorySpec(CONTEXT, ("generation_wind_forecast", "load_forecast"), (), "price", newest_target="published")
    (item,) = history_batch(spec, hourly, ["2023-06-14"], INFO, "1h", 1).items
    assert set(item.late) == {"generation_wind_forecast"}
    late = item.late["generation_wind_forecast"]
    assert (late.known_label, late.issue_label, late.known - late.issue) == ("D 18:00", "D 11:45", 375.0)
    assert item.newest_minutes == {"price": 1380.0, "generation_wind_forecast": 1380.0, "load_forecast": 2820.0}
    past = HistorySpec(CONTEXT, (), ("generation_wind_forecast",), newest_target="published")
    (item,) = history_batch(past, hourly, ["2023-06-14"], INFO).items
    assert not item.late and np.isfinite(item.past["generation_wind_forecast"]).all()


def test_a_past_series_ends_at_its_newest_known_step(hourly):
    """The actual load is out to 09:00-10:00 at 11:45: with a backend that takes missing values its last 14 hours of a price's
    context (10:00 to 23:00) are NaN; with one that doesn't, it is late, known when its 23:00 value is, at 01:00 on D+1."""
    (item,) = history_batch(PRICE, hourly, ["2023-06-14"], INFO, "1h", 1).items
    load = item.past["load_actual"]
    assert np.isnan(load[-14:]).all() and np.isfinite(load[:-14]).all() and not item.late and not item.missing
    strict = dataclasses.replace(PRICE, accepts_missing=False)
    (item,) = history_batch(strict, hourly, ["2023-06-14"], INFO, "1h", 1).items
    assert np.isfinite(item.past["load_actual"]).all()
    late = item.late["load_actual"]
    assert (late.known_label, late.issue_label) == ("D+1 01:00", "D 11:45")


def test_missing_values_are_counted_and_only_what_a_backend_can_t_take_keeps_an_item_out(hourly):
    """Two missing load forecasts in the context: counted, and the item is whole for a backend that takes missing values, not
    for one that doesn't; one missing on the target day is a missing step ahead, for both."""
    frame = hourly.copy()
    frame.loc["2023-06-12 05:00":"2023-06-12 06:00", "load_forecast"] = np.nan
    strict = dataclasses.replace(PRICE, accepts_missing=False, past=())
    (item,) = history_batch(PRICE, frame, ["2023-06-14"], INFO, "1h", 1).items
    assert item.gaps["load_forecast"] == (2, 0) and item.gaps["price"] == (0, 0) and item.missing == ()
    (item,) = history_batch(strict, frame, ["2023-06-14"], INFO, "1h", 1).items
    assert item.missing == ("load_forecast",)
    frame.loc["2023-06-15 03:00", "load_forecast"] = np.nan
    for spec in (PRICE, strict):
        (item,) = history_batch(spec, frame, ["2023-06-14"], INFO, "1h", 1).items
        assert item.gaps["load_forecast"] == (2, 1) and item.missing == ("load_forecast",)
    (item,) = history_batch(PRICE, hourly, ["2023-06-14"], INFO, "1h", 1).items
    assert item.missing == () and set(item.gaps.values()) == {(0, 0)}


def test_a_past_series_known_only_before_its_context_is_missing(hourly):
    """A past series out only five days after delivery, with a context of two days: NaN on every step, so missing, whatever
    the backend takes."""
    slow = InformationSet(rules={"load_actual": Availability.published("10:00", -5)})
    spec = HistorySpec(2, (), ("load_actual",), "price", accepts_missing=True, newest_target="published")
    (item,) = history_batch(spec, hourly, ["2023-06-14"], slow, "1h", 1).items
    assert np.isnan(item.past["load_actual"]).all() and item.missing == ("load_actual",) and not item.late


def test_batches_group_by_s_and_are_padded_with_copies_of_their_first_item(hourly):
    batch = history_batch(PRICE, hourly, DAYS, INFO, "1h", 1)
    calls = batch.batches(4)
    seen = []
    for call in calls:
        items = [batch.items[i] for i in call.items]
        assert {item.steps_ahead for item in items} == {call.steps_ahead}
        assert list(call.items) == sorted(call.items) and 1 <= len(call.items) <= 4
        assert call.target.shape == (4, batch.steps)
        assert call.known["load_forecast"].shape == (4, batch.steps + call.steps_ahead)
        for row in range(4):
            source = items[row] if row < len(items) else items[0]
            same(call.target[row], source.target)
            same(call.past["load_actual"][row], source.past["load_actual"])
            same(call.known["load_forecast"][row], source.known["load_forecast"])
        seen.extend(call.items)
    assert sorted(seen) == list(range(len(DAYS)))
    assert [call.steps_ahead for call in calls] == sorted(call.steps_ahead for call in calls)
    assert {call.steps_ahead for call in calls} == {23, 24, 25}
    (only,) = [call for call in calls if call.steps_ahead == 23]
    assert [batch.items[i].issue_day for i in only.items] == [datetime.date(2023, 3, 25)]
    assert [len(call.items) for call in batch.batches(64)] == [1, 38, 1]
    assert [call.items for call in batch.batches(64, which=[5, 3])] == [(3, 5)]


def test_items_are_read_only_views_and_horizons_are_read_at_once(hourly):
    """An item's arrays are read-only (a backend that writes in place raises rather than changes another item), and each
    horizon of `history_batches` is what `history_batch` gives it alone; `arrays=False` gives the same cuts without arrays."""
    days = DAYS[:10]
    batches = history_batches(PRICE, hourly, days, INFO, "1h", (1, 2, 3))
    for horizon, batch in batches.items():
        alone = history_batch(PRICE, hourly, days, INFO, "1h", horizon)
        cuts = history_batch(PRICE, hourly, days, INFO, "1h", horizon, arrays=False)
        for a, b, c in zip(batch.items, alone.items, cuts.items):
            same(a.target, b.target)
            same(a.known["load_forecast"], b.known["load_forecast"])
            same(a.past["load_actual"], b.past["load_actual"])
            assert (a.slots, a.newest, a.late.keys(), a.gaps) == (b.slots, b.newest, b.late.keys(), b.gaps)
            assert c.target is None and (c.slots, c.newest_minutes) == (a.slots, a.newest_minutes) and not c.gaps
    with pytest.raises(ValueError):
        batches[1].items[0].target[0] = 0.0
    with pytest.raises(ValueError, match="cuts alone"):
        history_batch(PRICE, hourly, days, INFO, "1h", 1, arrays=False).batches(4)


def test_a_rule_over_time_is_read_on_every_step_the_context_holds(hourly):
    """A load forecast out 900 hours after delivery up to 2023-05-10 and the day before from then on (a lag that shortened):
    issued on 2023-06-14 its values of 8 to 10 May aren't out yet, though every value from D-31 on is. A known value there says
    nothing of older ones under rules over time, so a context of 40 days reads them: late for a known-ahead series, an error
    for the target; one of 10 days doesn't reach them."""
    rules = InformationSet(
        rules={
            "load_forecast": RuleChain(
                [
                    Availability.realised("900h").during(valid_until="2023-05-10"),
                    Availability.published("10:00", 1).during(valid_from="2023-05-11"),
                ]
            )
        }
    )
    spec = HistorySpec(40, ("load_forecast",), (), "price", newest_target="published")
    (item,) = history_batch(spec, hourly, ["2023-06-14"], rules, "1h", 1).items
    assert "load_forecast" in item.late
    (fine,) = history_batch(dataclasses.replace(spec, context=10), hourly, ["2023-06-14"], rules).items
    assert not fine.late
    target = HistorySpec(40, (), (), "load_forecast", newest_target="published")
    with pytest.raises(InformationSetError, match="rules over time"):
        history_batch(target, hourly, ["2023-06-14"], rules, "1h", 1)


def test_the_reader_is_the_probe_s_reading_on_every_day_at_once(hourly):
    """`steps_known` calls known exactly the cells `perturb_unknown` leaves alone, column by column, an `as_of` value of a row
    after D (that row's own run's) unknown on D though its time of day is before the issue time; and `StepsReader` reading
    many days at once (the accessor's) gives each day's verdicts as reading it alone does."""
    rules = InformationSet(rules={"wx": Availability.as_of("09:00")})
    frame = hourly.assign(wx=hourly["price"] + 7.0)
    days = ["2023-03-25", "2023-03-26", "2023-06-14", "2023-10-28", "2023-10-29"]
    for day in days:
        probed, _, _ = perturb_unknown(frame, day, rules)
        for name in frame.columns:
            read = steps_known(rules, name, frame.index, day)
            same(read.known, probed[name].to_numpy() == frame[name].to_numpy())
    reader = StepsReader(rules, frame.index, days, "1h")
    rows = np.arange(len(frame))
    for i, day in enumerate(days):
        owners = np.full(len(frame), i)
        for name in frame.columns:
            known = ~(reader.available(name, rows, owners) > reader.issue(owners))
            same(known, steps_known(rules, name, frame.index, day).known)
    (item,) = history_batch(HistorySpec(CONTEXT, ("wx",), (), "price"), frame, ["2023-06-14"], rules).items
    assert item.newest_minutes["wx"] == 1380.0 and "wx" in item.late
    newest = steps_known(rules, "price", frame.index, "2023-06-14").newest()
    assert frame.index[newest] == pd.Timestamp("2023-06-14 23:00", tz=ZONE)


def test_what_the_accessor_refuses(hourly):
    with pytest.raises(ValueError, match="shifted column"):
        HistorySpec(CONTEXT, ("load_forecast_d+1",))
    with pytest.raises(ValueError, match="named twice"):
        HistorySpec(CONTEXT, ("load_forecast",), ("load_forecast",))
    with pytest.raises(ValueError, match="a list of column names"):
        HistorySpec(CONTEXT, "load_forecast")
    with pytest.raises(ValueError, match="newest_target"):
        HistorySpec(CONTEXT, newest_target="d")
    for bad in (0, 1.5, True):
        with pytest.raises(ValueError, match="context"):
            HistorySpec(bad)
    with pytest.raises(InformationSetError, match="need one"):
        history_batch(PRICE, hourly, ["2023-06-14"], None)
    with pytest.raises(ValueError, match="freq"):
        history_batch(PRICE, hourly, ["2023-06-14"], INFO, "1D")
    with pytest.raises(InformationSetError, match="no availability rule"):
        history_batch(HistorySpec(CONTEXT, ("mystery",)), hourly.assign(mystery=1.0), ["2023-06-14"], INFO)
    with pytest.raises(ValueError, match="no column"):
        history_batch(HistorySpec(CONTEXT, ("absent",)), hourly, ["2023-06-14"], INFO)
    off = hourly.copy()
    off.index = off.index + pd.Timedelta(minutes=5) * (off.index == pd.Timestamp("2023-06-10 05:00", tz=ZONE))
    with pytest.raises(ValueError, match="off the 1h steps"):
        history_batch(PRICE, off, ["2023-06-14"], INFO)
    slow = InformationSet(rules={"price": Availability.published("13:00", -40)})
    with pytest.raises(InformationSetError, match="a rule this slow"):
        history_batch(PRICE, hourly, ["2023-06-14"], slow)
    with pytest.raises(ValueError, match="has a zone"):
        history_batch(PRICE, hourly.tz_localize(None), [pd.Timestamp("2023-06-14 10:00", tz="UTC")], INFO)
    assert history_batch(PRICE, hourly, [], INFO).items == ()


def test_an_issue_day_with_a_zone_is_read_on_the_frame_s_clock(hourly):
    """00:30 on 15 June in Warsaw is 22:30 UTC on the 14th: the issue day is the 15th, as the frame's clock reads it."""
    stamps = [pd.Timestamp("2023-06-15 00:30", tz=ZONE).tz_convert("UTC"), "2023-06-14", datetime.date(2023, 6, 14)]
    batch = history_batch(PRICE, hourly, stamps, INFO, "1h", 1)
    assert [item.issue_day for item in batch.items] == [datetime.date(2023, 6, 14), datetime.date(2023, 6, 15)]


def test_a_frame_shorter_than_the_context_holds_nan_where_it_has_no_row(hourly):
    """T is fixed whatever the frame has: steps before its first row are NaN, counted as missing values of the context, which
    a backend that takes them can be given; and a frame of ten years costs what a call reads, not the whole frame."""
    frame = hourly.loc["2023-06-10":]
    (item,) = history_batch(PRICE, frame, ["2023-06-14"], INFO).items
    assert len(item.target) == CONTEXT * 24 and np.isnan(item.target[:48]).all()
    assert item.gaps["price"] == (48, 0) and "price" not in item.missing
    (item,) = history_batch(
        dataclasses.replace(PRICE, accepts_missing=False, past=()), frame, ["2023-06-14"], INFO
    ).items
    assert "price" in item.missing


# ---------------------------------------------------------------------------------------------------------------------------
# A fake foundation model through a day unit
# ---------------------------------------------------------------------------------------------------------------------------


class FakeFoundation(BaseModel):
    """Zero-shot: each step ahead is the target at the same step of the context's last day, plus 0.1 x the known-ahead series
    there. `by_batch` adds that much times the mean of the last context value over its call's rows (a model that reads the
    other items of its call); `offset` adds a constant (another model, for a loop)."""

    _unit_kind = "day"
    _engine_ok = False

    def __init__(
        self,
        context=CONTEXT,
        known_ahead=("load_forecast",),
        past=(),
        batch_size=64,
        accepts_missing=True,
        by_batch=0.0,
        offset=0.0,
        name="FakeFM",
        refit="7D",
        **kw,
    ):
        super().__init__(kw.pop("predictors", []), kw.pop("training_window", context), name=name, refit=refit, **kw)
        self.context = context
        self.known_ahead = list(known_ahead)
        self.past = list(past)
        self.batch_size = batch_size
        self.accepts_missing = accepts_missing
        self.by_batch = by_batch
        self.offset = offset
        self._history_rules = None
        self._calls = []

    @property
    def _model_kwargs(self):
        return {
            "context": self.context,
            "known_ahead": list(self.known_ahead),
            "past": list(self.past),
            "batch_size": self.batch_size,
            "accepts_missing": self.accepts_missing,
            "by_batch": self.by_batch,
            "offset": self.offset,
        }

    def _history_spec(self, target):
        return HistorySpec(self.context, self.known_ahead, self.past, target, self.accepts_missing, self.newest_target)

    def _fit_predict(self, train_x, train_y, test_x):  # BaseModel's, which a unit never calls
        raise RuntimeError("a history model fits over history batches, not rows")

    def _fit_days(self, x, y, y_mask):  # a day unit's methods, which its units, made of history, never call
        raise RuntimeError("a history model fits over history batches, not rows")

    def _predict_days(self, fitted, rows):
        raise RuntimeError("a history model forecasts from history batches, not rows")

    def _check_inputs(
        self,
        data,
        target,
        horizon,
        freq,
        information_set=None,
        test_start=None,
        test_end=None,
        plan=None,
        availability="raise",
        windows=None,
        cut_rules=None,
        horizons=None,
        weekdays=None,
        stacklevel=4,
        shared=None,
    ):
        """BaseModel's checks, which register the series as inputs and refuse or list a late one (`_check_history`), then the
        cut's rules kept for the run: a foundation model's run plumbing, 6i's."""
        expanded = super()._check_inputs(
            data,
            target,
            horizon,
            freq,
            information_set,
            test_start,
            test_end,
            plan=plan,
            availability=availability,
            windows=windows,
            cut_rules=cut_rules,
            horizons=horizons,
            weekdays=weekdays,
            stacklevel=stacklevel + 1,
            shared=shared,
        )
        self._history_rules = self._cut_rules(information_set, cut_rules)
        return expanded

    def _run_tasks(self, tasks, state, sink):
        """The day unit's blocks, in this process: each unit's forecasts from the history batches of its block's days."""
        hour_arrays = self._build_hour_arrays(max(state.horizons))
        expanded, config = self._worker_inputs(state, 1)
        inputs = TaskInputs(hour_arrays, expanded, config, self)
        for job in self._jobs(tasks, state.first_day).values():
            ran = [
                [
                    (task, outcome, unit.member)
                    for task, outcome in self._history_unit(unit, unit_tasks, inputs, state.data)
                ]
                for unit, unit_tasks in block_units(config, job).items()
            ]
            results, skipped = block_records(config, ran)
            self._collect(state, results, skipped, sink)

    def _history_unit(self, unit, tasks, inputs, frame):
        """One unit's forecasts: the call holds every day of its block that the frame has with series the backend can take, the
        run's tasks or not, so a forecast is the same alone, in its block and after a resume."""
        cfg = inputs.config
        outcomes, issued = _issued(inputs, tasks)
        first_ordinal = cfg["refit"]["first_ordinal"]
        rows = set(np.concatenate([arrays["days"] for arrays in inputs.hour_arrays.values()]).tolist())
        days = [
            day
            for day in block_days(unit.anchor, cfg["refit"]["rules"][unit.member])
            if day.toordinal() - first_ordinal in rows
        ]
        spec = self._history_spec(self._target)
        batch = history_batch(spec, frame, days, self._history_rules, self._freq, unit.horizon)
        complete = [i for i, item in enumerate(batch.items) if not item.missing]
        forecasts = {}
        for call in batch.batches(self.batch_size, complete):
            out = self._backend(call)
            assert out.shape == (self.batch_size, call.steps_ahead)
            for row, i in enumerate(call.items):
                item = batch.items[i]
                for step, slot in enumerate(item.slots):
                    if slot is not None:
                        forecasts[(item.issue_day, slot[1])] = float(out[row, step])
        start = datetime.date.fromisoformat(cfg["test_start"])
        for task, test in issued:
            forecast = forecasts.get((start + datetime.timedelta(days=task[2]), task[0]))
            if forecast is None:
                outcomes.append((task, SKIP_MISSING_TEST_PREDICTORS))
            else:
                outcomes.append((task, _unit_record(inputs, unit, task, test, forecast, [], None)))
        return outcomes

    def _backend(self, call):
        periods = self._periods_per_day
        steps = call.target.shape[1]
        out = call.target[:, steps - periods + np.arange(call.steps_ahead) % periods]
        for name in self.known_ahead:
            out = out + 0.1 * call.known[name][:, steps:]
        if self.by_batch:
            out = out + self.by_batch * float(call.target[:, -1].mean())
        self._calls.append((call.steps_ahead, len(call.items)))
        return out + self.offset


class ReadsTheFramesEnd(FakeFoundation):
    """Adds 1e-3 x the mean over its call of each context's first value: values every item may read, but how many items a
    call holds depends on where the frame ends."""

    def _backend(self, call):
        return super()._backend(call) + 1e-3 * float(call.target[:, 0].mean())


class Rows(FakeFoundation):
    """History and row blocks (`_predict_block`, declared, never called: its units read history alone)."""

    def _predict_block(self, fitted, rows):
        raise RuntimeError("never called")


class CountsItsRowDays(Rows):
    """A hybrid whose row part sees its call: 1e-3 x how many days of its block have a finite wind forecast on every row (an
    input of its rows, no series of its history), as a row block holds the days whose inputs are finite."""

    def _history_unit(self, unit, tasks, inputs, frame):
        days = block_days(unit.anchor, inputs.config["refit"]["rules"][unit.member])
        local = frame.index.tz_localize(None).normalize()
        wind = frame["generation_wind_forecast"]
        self._row_days = sum(
            bool(len(wind[local == pd.Timestamp(day)])) and bool(np.isfinite(wind[local == pd.Timestamp(day)]).all())
            for day in days
        )
        return super()._history_unit(unit, tasks, inputs, frame)

    def _backend(self, call):
        return super()._backend(call) + 1e-3 * self._row_days


class ReadsDsTarget(FakeFoundation):
    """A "d-1" model whose series are cut as a "published" one's: its context holds D's prices."""

    def _history_spec(self, target):
        return dataclasses.replace(super()._history_spec(target), newest_target="published")


def make(name, **params):
    return FakeFoundation(name=name, **params)


def fake_by_hand(frame, day, horizon, freq="1h", context=CONTEXT, newest_target="d-1"):
    """The fake's forecasts of one issue day, {period: value}, from the reference by hand."""
    spec = HistorySpec(context, ("load_forecast",), (), "price", True, newest_target)
    ref = by_hand(utc_of(frame), spec, day, horizon, freq)
    periods = 1440 // STEP[freq]
    steps = len(ref["target"])
    out = {}
    for i, slot in enumerate(ref["slots"]):
        if slot is not None:
            out[slot[1]] = ref["target"][steps - periods + i % periods] + 0.1 * ref["known"]["load_forecast"][steps + i]
    return out


@pytest.fixture
def registered(monkeypatch):
    from epftoolbox2.pipelines.model_pipeline import COMPONENT_REGISTRY

    monkeypatch.setitem(COMPONENT_REGISTRY["models"], "FakeFoundation", __name__)


def pipeline(*models, information_set=INFO, **settings):
    out = ModelPipeline(information_set=information_set, **settings)
    for model in models:
        out.add_model(model)
    return out


def test_the_fake_reads_history_and_is_cut_after_what_it_reads(hourly):
    """A model reads history when its class defines `_history_spec`. `verify` cuts its frame after what each forecast reads
    (`target_day_cut`), never after the issue day, whose next rows hold its known-ahead values, even with row blocks; a day unit
    without history is cut after the issue day, as before. Its spec carries its newest target, "d-1" by default."""
    model = make("FM")
    assert model.newest_target == "d-1"
    assert reads_history(model) and history_spec_of(model, "price") == HistorySpec(
        CONTEXT, ("load_forecast",), (), "price", True, "d-1"
    )
    assert model._predicts_together and not cuts_at_issue_day(model) and cuts_after_target_day(model)
    assert not model._engine_capable()

    class Day(FakeFoundation):
        _history_spec = None

    assert not reads_history(Day()) and cuts_at_issue_day(Day()) and not cuts_after_target_day(Day())

    assert not cuts_at_issue_day(Rows()) and cuts_after_target_day(Rows())
    from epftoolbox2.models import OLSModel

    assert not reads_history(OLSModel(["price_d-1"])) and history_spec_of(OLSModel(["price_d-1"]), "price") is None


@pytest.mark.filterwarnings("ignore:Availability")  # the load forecast of D+2, late at horizon 2: listed, and used
@pytest.mark.parametrize("freq", ["1h", "15min"])
@pytest.mark.parametrize("newest", ["d-1", "published"])
def test_the_fake_forecasts_what_its_reference_by_hand_says(hourly, quarter, freq, newest):
    """Under "d-1" the fake's persistence is of D-1; under "published" of D, whose prices are out at 11:45."""
    frame = hourly if freq == "1h" else quarter
    days = ["2023-03-24", "2023-03-28"] if freq == "1h" else ["2023-03-25", "2023-03-26"]
    report = pipeline(make("FM", newest_target=newest)).run(frame, *days, horizon=2, freq=freq, forecast_only=True)
    counts = report.forecast_counts().iloc[0]
    assert counts["produced"] == counts["planned"] > 0, counts.to_dict()
    records = report.refs["FM"]._results
    expected = {}
    for record in records:
        key = (datetime.date.fromisoformat(record["run_date"]), record["horizon"])
        if key not in expected:
            expected[key] = fake_by_hand(frame, *key, freq, newest_target=newest)
        assert record["prediction"] == expected[key][record["hour"]], record


@pytest.mark.filterwarnings("ignore:overflow")
def test_verify_is_ok_with_its_control_rerun_co_batch_and_cut_frames(hourly):
    """`adapter_suite.check_verify` on a block's first, middle and last days and both switches, for the default "d-1" fake (the
    target perturbed from D on) and a "published" one: the six other days of the "7D" block named, nothing moved by noise in
    them, nor by the frame cut after each target day, and every forecast moved by the control."""
    days = ["2023-03-20", "2023-03-22", "2023-03-25", "2023-03-26", "2023-10-28", "2023-10-29"]
    suite.check_verify(make, hourly, INFO, days, items=6)
    suite.check_verify(lambda name: make(name, newest_target="published"), hourly, INFO, days[1:3], items=6)
    suite.check_verify(make, hourly, INFO_AHEAD, ["2023-10-28"], horizon=2, items=6)
    report = pipeline(make("FM")).verify(hourly, "2023-03-22", "2023-03-22", horizon=2, days=1, co_batch=True)
    cut = report.checks[0].co_batch.cut
    assert cut is not None and cut.ok and cut.forecasts == 48 and cut.ends == "after what each forecast reads"


@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.parametrize(
    "leaky",
    [
        pytest.param(lambda name: make(name, by_batch=1e-3), id="reads its call's other items"),
        pytest.param(lambda name: ReadsTheFramesEnd(name=name), id="depends on where the frame ends"),
        pytest.param(
            lambda name: ReadsTheFramesEnd(name=name, known_ahead=()), id="univariate, depends on the frame's end"
        ),
    ],
)
def test_verify_catches_what_a_history_model_must_not_read(hourly, leaky):
    with pytest.raises(AssertionError):
        suite.check_verify(leaky, hourly, INFO, ["2023-06-14"])


@pytest.mark.filterwarnings("ignore:overflow")
def test_a_univariate_model_s_frame_ends_where_its_forecast_reads(hourly):
    """A univariate `ReadsTheFramesEnd` on the Saturday of a "7D" block at h=1: cut after the target day, Sunday's context is
    complete and stays in the call, which a run on Saturday doesn't have (0.00225 apart). Its frame ends on D, after what its
    forecast reads, so it is a leak."""
    report = pipeline(ReadsTheFramesEnd(name="FM", known_ahead=())).verify(hourly, "2023-06-17", "2023-06-17", days=1)
    assert not report.ok and not report.checks[0].co_batch.cut.ok


def test_a_d1_model_must_cut_its_history_as_d1(hourly, monkeypatch):
    """A `_history_spec` with another newest target than the model's is refused before any fit: verify holds a model to its
    own. With that check taken away, the "d-1" model reading D's prices is a leak under the target's boundary."""
    with pytest.raises(ValueError, match="newest_target='published' and the model has newest_target='d-1'"):
        pipeline(ReadsDsTarget(name="FM")).run(hourly, "2023-06-14", "2023-06-14", horizon=1, forecast_only=True)
    import epftoolbox2.models.base as base_module

    def as_checked(model, target):  # what the check sees: the model's own newest target; the run still reads D's prices
        spec = history_spec_of(model, target)
        return None if spec is None else dataclasses.replace(spec, newest_target=model.newest_target)

    monkeypatch.setattr(base_module, "history_spec_of", as_checked)
    with pytest.raises(AssertionError):
        suite.check_verify(lambda name: ReadsDsTarget(name=name), hourly, INFO, ["2023-06-14"])


@pytest.mark.filterwarnings("ignore:overflow")
def test_the_frame_cut_after_each_target_day_catches_what_the_probe_can_t(hourly):
    """`ReadsTheFramesEnd` reads only values known at the issue time, so the probe and the control pass it; the frame cut after
    the target day takes the block's later days out of its call, and its forecasts move."""
    report = pipeline(ReadsTheFramesEnd(name="FM")).verify(hourly, "2023-06-14", "2023-06-14", horizon=1, days=1)
    check = report.checks[0]
    assert check.probe.ok and check.co_batch.probe.ok and not check.co_batch.cut.ok
    assert "differ when the frame ends after what each forecast reads" in "\n".join(report.problems())


@pytest.mark.filterwarnings("ignore:overflow")
def test_verify_catches_a_cut_one_step_late(hourly, monkeypatch):
    """With every series cut one step after its newest known one, a "published" model of the actual load reads its value of
    10:00 on D, out at 12:00, which the persistence of the context's last day reads: a leak."""
    from epftoolbox2.models import _history

    real = _history._Cuts.__init__

    def late(self, *args, **kwargs):
        real(self, *args, **kwargs)
        last = self.end[max(self.end)]
        self.newest = {name: np.minimum(newest + 1, last) for name, newest in self.newest.items()}

    monkeypatch.setattr(_history._Cuts, "__init__", late)
    model = make("FM", newest_target="published")
    report = pipeline(model).verify(hourly, "2023-03-22", "2023-03-22", target="load_actual", horizon=1, days=1)
    assert report.table().iloc[0]["changed"] > 0
    with pytest.raises(AssertionError):
        report.assert_ok()


def test_a_1d_model_names_no_other_item_of_its_call():
    assert make("FM", refit="1D")._co_batch_items("2023-03-22") == []
    assert len(make("FM")._co_batch_items("2023-03-22")) == 6


def test_a_backend_that_takes_missing_values_forecasts_over_a_missing_hour(hourly):
    """An hour missing early in the context: a fake that takes missing values forecasts every period anyway, the same values as
    without the gap (persistence reads the last day); one that doesn't skips the day's forecasts as missing inputs."""
    frame = hourly.copy()
    frame.loc["2023-06-09 05:00", "price"] = np.nan

    def forecasts(model, data):
        ref = pipeline(model).run(data, "2023-06-14", "2023-06-14", horizon=1, forecast_only=True).refs["FM"]
        return {r["hour"]: r["prediction"] for r in ref._results}

    whole = forecasts(make("FM"), hourly)
    assert len(whole) == 24 and forecasts(make("FM"), frame) == whole
    with pytest.raises(RuntimeError, match="missing_test_predictors x24"):
        forecasts(make("FM", accepts_missing=False), frame)


def records_of(path):
    return sorted(Path(path).read_text().splitlines())


def test_alone_in_a_batch_of_64_and_resumed_are_the_same_bits(tmp_path):
    """Under "3MS" a block holds up to 93 days: the forecasts of a run over both switches with `batch_size=64` (calls of 64
    real items), each of some days run alone with `batch_size=1`, and a run resumed after 40 % of its records were deleted
    give the same bits."""
    frame = market("1h", start="2022-12-20", end="2024-01-10")
    test = ("2023-03-20", "2023-11-05")
    full = tmp_path / "full"
    model = make("FM", refit="3MS")
    pipeline(model).run(frame, *test, horizon=1, save_dir=full, forecast_only=True)
    assert (48, 64) in model._calls and {47, 49} <= {s for s, _ in model._calls}, model._calls
    lines = records_of(suite.results(full, "FM"))
    by_key = {}
    for line in lines:
        record = json.loads(line)
        if "prediction" in record:
            by_key[(record["run_date"], record["hour"])] = record["prediction"]
    for day in ("2023-03-20", "2023-03-25", "2023-03-26", "2023-06-30", "2023-10-28", "2023-11-05"):
        alone = pipeline(make("FM", refit="3MS", batch_size=1)).run(frame, day, day, horizon=1, forecast_only=True)
        produced = alone.refs["FM"]._results
        assert produced
        for record in produced:
            assert record["prediction"] == by_key[(day, record["hour"])]
    resumed = tmp_path / "resumed"
    pipeline(make("FM", refit="3MS")).run(frame, *test, horizon=1, save_dir=resumed, forecast_only=True)
    path = suite.results(resumed, "FM")
    kept = [line for i, line in enumerate(records_of(path)) if i % 5 not in (1, 3)]
    path.write_text("".join(f"{line}\n" for line in kept))
    pipeline(make("FM", refit="3MS")).run(frame, *test, horizon=1, save_dir=resumed, forecast_only=True)
    assert records_of(path) == lines
    # The batch size is the model's: a resume with another refuses, as one with any other parameter would.
    with pytest.raises(ManifestMismatchError, match="batch_size"):
        pipeline(make("FM", refit="3MS", batch_size=7)).run(
            frame, *test, horizon=1, save_dir=resumed, forecast_only=True
        )


def test_the_adapter_suite(hourly, quarter, tmp_path, monkeypatch, registered):
    """YAML round trip, a loop and its shards, a resume, `engine="on"` (which takes no such model), the DST counts (23 and 24
    forecasts, the repeated hour's actual NaN) and 96 quarter-hours; no card to pre-flight or show."""
    test = ("2023-03-15", "2023-03-28")
    info = INFO_AHEAD  # horizon 2 reads the load forecast of D+2
    suite.check_yaml_round_trip(make, hourly, tmp_path, test, horizon=2, information_set=info)
    variants = ({}, {"context": 5}, {"batch_size": 1, "offset": 1.0}, {"newest_target": "published"})
    suite.check_loop_and_shards(
        make, hourly, tmp_path / "loop", test, horizon=2, variants=variants, information_set=info
    )
    suite.check_resume(make, hourly, tmp_path / "resume", test, horizon=2, information_set=info)
    suite.check_engine(make, hourly, test, horizon=2, information_set=info)
    suite.check_dst_counts(make, hourly, "2023-03-26", "2023-10-29", information_set=INFO)
    suite.check_quarter_hours(make, quarter, ("2023-02-20", "2023-02-21"), information_set=INFO)
    suite.check_preflight_without_extra(make, monkeypatch, hourly, test)
    suite.check_provenance(make, hourly, tmp_path / "provenance", test, information_set=info)


# ---------------------------------------------------------------------------------------------------------------------------
# The listing, the terminal, explain and vintages
# ---------------------------------------------------------------------------------------------------------------------------


def test_availability_has_one_row_per_series_with_its_newest_step_and_class(hourly, tmp_path):
    """At "warn", one row per series, not per lagged step: a "published" model's target to D 23:00, the actual load (NaN after
    it) to 09:00, the load forecast to the end of each target day, each with its rule's class; the wind forecast, late as a
    known-ahead series, with when it is out. No warning for the series read in time; one for the late one. A report read back
    from the files shows the same. A "d-1" model's target reads to D-1 23:00."""

    def model():
        known = ["load_forecast", "generation_wind_forecast"]
        return make("FM", past=["load_actual"], known_ahead=known, newest_target="published")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = pipeline(model(), information_set=INFO_AHEAD).run(
            hourly, "2023-06-12", "2023-06-14", horizon=2, save_dir=tmp_path, forecast_only=True
        )
    listed = [w for w in caught if "Availability:" in str(w.message)]
    assert len(listed) == 1 and "generation_wind_forecast is late" in str(listed[0].message)
    assert "load_forecast is" not in str(listed[0].message)
    table = report.availability()
    assert list(table.columns) == COLUMNS + HISTORY_COLUMNS
    shown = {row["predictor"]: row for row in table.to_dict("records")}
    assert set(shown) == {"price", "load_actual", "load_forecast", "generation_wind_forecast"}
    assert [shown[name]["role"] for name in shown] == ["target", "past", "known_ahead", "known_ahead"]
    assert {name: (row["status"], row["newest"], row["vintage"]) for name, row in shown.items()} == {
        "price": (ON_TIME, "D 23:00", "snapshot"),
        "load_actual": (ON_TIME, "D 09:00", "latest_revision"),
        "load_forecast": (ON_TIME, "D+1 23:00 … D+2 23:00", "latest_revision"),  # read to each target day's end
        "generation_wind_forecast": (LATE, "D 23:00", "latest_revision"),
    }
    wind = shown["generation_wind_forecast"]
    assert (wind["known"], wind["issued"], wind["horizons"], wind["periods"]) == ("D+1 18:00", "D 11:45", "1-2", "0-23")
    with pytest.warns(UserWarning, match="generation_wind_forecast is late"):
        read_back = pipeline(model(), information_set=INFO_AHEAD).report(
            tmp_path, "2023-06-12", "2023-06-14", horizon=2, forecast_only=True
        )
    pd.testing.assert_frame_equal(read_back.availability(), table)
    with pytest.raises(InformationSetError, match="generation_wind_forecast"):
        pipeline(make("FM", known_ahead=["generation_wind_forecast"]), availability="raise").run(
            hourly, "2023-06-12", "2023-06-14", horizon=1, forecast_only=True
        )
    d1 = pipeline(make("FM")).run(hourly, "2023-06-12", "2023-06-13", horizon=1, forecast_only=True).availability()
    assert d1.set_index("predictor").loc["price", "newest"] == "D-1 23:00"


@pytest.mark.filterwarnings("ignore:Availability")  # what it lists, the late `price_d0`
def test_availability_of_a_model_of_predictors_keeps_its_columns(hourly):
    """Without a sequence model a listing's frame has the columns it always had, and its rows the keys they had."""
    from epftoolbox2.models import OLSModel

    frame = hourly.assign(price_d0=hourly["price"])
    rules = InformationSet(rules={"price_d0": Availability.published("13:00", 0)})
    report = pipeline(OLSModel(["load_forecast", "price_d0"], 28, name="OLS"), information_set=rules).run(
        frame, "2023-06-12", "2023-06-13", horizon=1, forecast_only=True
    )
    table = report.availability()
    assert list(table.columns) == COLUMNS and len(table) and set(table["status"]) == {LATE}
    (row,) = report.refs["OLS"].availability_listing
    assert set(row) == {"predictor", "role", "status", "list", "periods", "horizons", *COLUMNS[7:], "minutes_late"}
    assert listing_frame({"m": []}).columns.tolist() == COLUMNS


def test_the_rows_of_two_windows_at_one_horizon_combine(hourly):
    """A wind forecast out at 18:00 on D-1 up to March and at 20:00 from April: a July window and a February one at the same
    horizon give one late row, latest by the July days (D 20:00), as one batch of both does, in any order."""
    rules = InformationSet(
        rules={
            "generation_wind_forecast": RuleChain(
                [
                    Availability.published("18:00", 1).during(valid_until="2023-03-31"),
                    Availability.published("20:00", 1).during(valid_from="2023-04-01"),
                ]
            )
        }
    )
    spec = HistorySpec(CONTEXT, ("generation_wind_forecast",), (), "price")
    july = history_batch(spec, hourly, ["2023-07-03", "2023-07-04"], rules, "1h", 1)
    feb = history_batch(spec, hourly, ["2023-02-20", "2023-02-21"], rules, "1h", 1)
    both = history_batch(spec, hourly, ["2023-02-20", "2023-02-21", "2023-07-03", "2023-07-04"], rules, "1h", 1)
    expected = history_rows([both])
    late = [row for row in expected if row["status"] == LATE]
    assert [(row["predictor"], row["known"], row["minutes_late"]) for row in late] == [
        ("generation_wind_forecast", "D 20:00", 495.0)
    ]
    assert history_rows([july, feb]) == expected and history_rows([feb, july]) == expected


def test_merging_listings_spans_both_runs_newest_steps(hourly):
    """Two runs into one directory, of other days and horizons: the merged rows hold the range of both runs' newest steps,
    and a series late in one run and read in time in the other is late there."""
    utc = hourly.tz_convert("UTC")
    feb = [datetime.date(2023, 2, 20) + datetime.timedelta(days=i) for i in range(5)]
    july = [datetime.date(2023, 7, 1) + datetime.timedelta(days=i) for i in range(5)]
    early = history_rows([history_batch(PRICE, utc, feb, INFO, "1h", 1)])
    later = history_rows([history_batch(PRICE, utc, feb + july, INFO, "1h", 1)])
    fresh = {row["predictor"]: row["newest"] for row in later}
    merged = {row["predictor"]: row["newest"] for row in merge_listings(early, later)}
    # On a frame in UTC the Brussels delivery day ends at 21:00Z in July and 22:00Z in February: the range of both runs.
    assert merged == fresh and fresh["price"] == "D 21:00 … D 22:00" and early[0]["newest"] != fresh["price"]
    assert merge_listings(early, []) == early
    h1 = history_rows([history_batch(PRICE, hourly, ["2023-06-14"], INFO_AHEAD, "1h", 1)])
    h2 = history_rows([history_batch(PRICE, hourly, ["2023-06-14"], INFO_AHEAD, "1h", 2)])
    both = history_rows([history_batch(PRICE, hourly, ["2023-06-14"], INFO_AHEAD, "1h", h) for h in (1, 2)])
    newest = {(row["predictor"], row["horizons"]): row["newest"] for row in merge_listings(h1, h2)}
    assert newest == {(row["predictor"], row["horizons"]): row["newest"] for row in both}
    assert newest[("load_forecast", "1-2")] == "D+1 23:00 … D+2 23:00"
    strict = dataclasses.replace(PRICE, accepts_missing=False)
    read = history_rows([history_batch(PRICE, hourly, ["2023-06-14"], INFO, "1h", 1)])
    late = history_rows([history_batch(strict, hourly, ["2023-06-15"], INFO, "1h", 1)])
    status = {(row["predictor"], row["status"]): row for row in merge_listings(read, late)}
    assert ("load_actual", ON_TIME) not in status and status[("load_actual", LATE)]["newest"] == "D 09:00"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_about({"FM": read})  # nothing late: no warning


def test_the_terminal_lists_late_series_as_late_and_the_others_on_their_own(hourly):
    """The availability table holds what is late (and what has no rule); a sequence model's series read in time come in a table
    of their own, with how far each is read. A report with nothing late prints no availability table."""

    def printed(rows):
        out = io.StringIO()
        TerminalExporter()._print_availability(Console(file=out, width=250), listing_frame({"FM": rows}))
        return out.getvalue()

    spec = HistorySpec(CONTEXT, ("generation_wind_forecast", "load_forecast"), (), "price", newest_target="published")
    text = printed(history_rows([history_batch(spec, hourly, ["2023-06-14"], INFO, "1h", 1)]))
    late, _, series = text.partition("History:")
    assert "not known to be out" in late and "generation_wind_forecast" in late and "load_forecast" not in late
    assert "load_forecast" in series and "D+1 23:00" in series and "generation_wind" not in series
    text = printed(history_rows([history_batch(PRICE, hourly, ["2023-06-14"], INFO, "1h", 1)]))
    assert "not known to be out" not in text and "History:" in text
    assert printed([]) == ""


def test_explain_shows_one_row_per_series(hourly):
    model = make(
        "FM", past=["load_actual"], known_ahead=["load_forecast", "generation_wind_forecast"], newest_target="published"
    )
    out = pipeline(model).explain(hourly, "FM", test=("2023-03-24", "2023-03-26"), horizon=1, hour=5, echo=False)
    series = out["series"].set_index("series")
    assert series.loc["price", ["role", "steps", "newest", "status", "vintage"]].tolist() == [
        "target",
        str(CONTEXT * 24),
        "D 23:00",
        "ok",
        "snapshot",
    ]
    assert series.loc["load_forecast", "steps"] == f"{CONTEXT * 24} + 23…24"
    assert series.loc["generation_wind_forecast", "status"] == "late"
    assert series.loc["load_actual", "newest"] == "D 09:00"
    assert out["predictors"].empty
    d1 = pipeline(make("FM")).explain(hourly, "FM", test=("2023-03-24", "2023-03-26"), horizon=1, hour=5, echo=False)
    assert d1["series"].set_index("series").loc["price", "newest"] == "D-1 23:00"
    assert d1["series"].set_index("series").loc["load_forecast", "steps"] == f"{CONTEXT * 24} + 47…48"


def test_explain_shows_a_series_without_a_rule_as_a_predictor_without_one(hourly):
    frame = hourly.assign(mystery=1.0)
    out = pipeline(make("FM", known_ahead=["mystery", "load_forecast"])).explain(
        frame, "FM", test=("2023-06-12", "2023-06-14"), horizon=1, hour=5, echo=False
    )
    status = out["series"].set_index("series")["status"]
    assert (status["mystery"], status["load_forecast"], status["price"]) == ("no rule", "ok", "ok")
    out = pipeline(make("FM")).explain(
        frame.assign(price_x=frame["price"]), "FM", target="mystery", horizon=1, echo=False
    )
    assert set(out["series"]["status"]) == {"no rule", "not cut (the target has no rule)"}


def test_vintages_list_the_series(hourly):
    model = make("FM", past=["load_actual"])
    report = pipeline(model).run(hourly, "2023-06-12", "2023-06-12", horizon=1, forecast_only=True)
    table = report.vintages()
    assert {"load_actual", "load_forecast"} <= set(table["predictor"])


# ---------------------------------------------------------------------------------------------------------------------------
# What BaseModel does for a model that reads history: its checks, the manifest, a ResidualModel's base
# ---------------------------------------------------------------------------------------------------------------------------


class Minimal(BaseModel):
    """The least a model that reads history is: BaseModel's checks of it, with a list of its own."""

    def __init__(self, known_ahead=("load_forecast",), past=(), name="Minimal"):
        super().__init__(["load_forecast"], 28, name=name)
        self.known_ahead, self.past = list(known_ahead), list(past)

    def _fit_predict(self, train_x, train_y, test_x):
        return float(train_y.mean()), []

    def _history_spec(self, target):
        return HistorySpec(CONTEXT, self.known_ahead, self.past, target)


def test_basemodel_registers_a_history_model_s_series_and_refuses_or_lists_a_late_one(hourly):
    """`BaseModel._check_inputs` of a model that reads history: its series are inputs it reads; a known-ahead series out after
    the issue time is refused at "raise", naming it, and listed at "warn" with one row per series; at "none", and without an
    information set, nothing is judged. A model of predictors is checked as before."""
    test = ("2023-06-12", "2023-06-14")
    model = Minimal(known_ahead=["load_forecast", "generation_wind_forecast"], past=["load_actual"])
    model._check_inputs(hourly, "price", 1, "1h", INFO, *test, availability="warn")
    assert model._used_predictors == ["load_forecast", "load_actual", "generation_wind_forecast"]
    rows = {(row["predictor"], row["status"]) for row in model._availability_listing}
    assert rows == {
        ("price", ON_TIME),
        (
            "load_actual",
            ON_TIME,
        ),  # under "d-1" the context ends at D-1's last step, and the actual load is out to D 09:00
        ("load_forecast", ON_TIME),
        ("generation_wind_forecast", LATE),
    }
    with pytest.raises(InformationSetError, match="generation_wind_forecast on 3 issue day"):
        model._check_inputs(hourly, "price", 1, "1h", INFO, *test, availability="raise")
    model._check_inputs(hourly, "price", 1, "1h", INFO, *test, availability="none")
    assert model._availability_listing == [] and "generation_wind_forecast" in model._used_predictors
    Minimal()._check_inputs(hourly, "price", 1, "1h", INFO, *test, availability="raise")  # nothing late
    from epftoolbox2.models import OLSModel

    ols = OLSModel(["load_forecast"], 28)
    ols._check_inputs(hourly, "price", 1, "1h", INFO, *test, availability="warn")
    assert ols._used_predictors == ["load_forecast"] and ols._availability_listing == []


class ReadsPast(FakeFoundation):
    """Also reads its past series: 1e-3 x their sum over the context."""

    def _backend(self, call):
        return super()._backend(call) + 1e-3 * np.nansum(call.past["load_actual"], axis=1)[:, None]


@pytest.mark.parametrize("newest", ["d-1", "published"])
def test_a_resume_with_another_rule_of_a_series_refuses(hourly, tmp_path, newest):
    """A past series' rule moves a history model's numbers whatever its newest target: a resume with `realised("4h")` for it
    refuses on `training_cut`, and so does one with another issue time; a level alone still resumes."""
    test = ("2023-06-05", "2023-06-18")

    def run(info, **settings):
        model = ReadsPast(name="FM", past=["load_actual"], newest_target=newest)
        return pipeline(model, information_set=info, **settings).run(
            hourly, *test, horizon=1, save_dir=tmp_path, forecast_only=True
        )

    run(INFO)
    path = suite.results(tmp_path, "FM")
    lines = records_of(path)
    path.write_text("".join(f"{line}\n" for i, line in enumerate(lines) if i % 2 == 0))
    for other in (InformationSet(rules={"load_actual": Availability.realised("4h")}), InformationSet("11:00")):
        with pytest.raises(ManifestMismatchError, match="training_cut"):
            run(other)
    run(INFO, availability="none")
    assert records_of(path) == lines
    stored = json.loads(manifest_path(path).read_text())["training_cut"]["history"]
    assert stored["issue_time"] == "11:45" and set(stored["rules"]) == {"price", "load_actual", "load_forecast"}
    assert "note" not in stored["rules"]["price"] and "vintage" not in stored["rules"]["price"]


def test_a_resume_over_other_known_ahead_values_of_the_last_target_day_refuses(hourly, tmp_path):
    """The last issue day's forecasts read the load forecast of the day after `test_end`: the manifest hashes the known-ahead
    series through `test_end` plus the largest horizon, so a frame with other values there refuses to resume."""
    test = ("2023-06-05", "2023-06-18")
    pipeline(make("FM")).run(hourly, *test, horizon=1, save_dir=tmp_path, forecast_only=True)
    path = suite.results(tmp_path, "FM")
    (ahead,) = json.loads(manifest_path(path).read_text())["history_ahead"]
    assert ahead["through"] == "2023-06-19" and ahead["columns"] == ["load_forecast"]
    lines = records_of(path)
    path.write_text("".join(f"{line}\n" for i, line in enumerate(lines) if i % 2 == 0))
    changed = hourly.copy()
    changed.loc["2023-06-19", "load_forecast"] += 500.0
    with pytest.raises(ManifestMismatchError, match="history_ahead"):
        pipeline(make("FM")).run(changed, *test, horizon=1, save_dir=tmp_path, forecast_only=True)
    later = hourly.copy()
    later.loc["2023-06-20", "load_forecast"] += 500.0  # after what any forecast reads: resumes
    pipeline(make("FM")).run(later, *test, horizon=1, save_dir=tmp_path, forecast_only=True)
    assert records_of(path) == lines


def test_a_model_of_predictors_keeps_its_manifest_and_digest(hourly, tmp_path):
    """No `history_ahead` key and no history in `training_cut` for a model of predictors, and the configuration digest a
    `ResidualModel` records of its base doesn't hold the new field."""
    from epftoolbox2.models import OLSModel
    from epftoolbox2.results.manifest import configuration_digest

    pipeline(OLSModel(["load_forecast"], 28, name="OLS")).run(
        hourly, "2023-06-12", "2023-06-13", horizon=1, save_dir=tmp_path
    )
    stored = json.loads(manifest_path(suite.results(tmp_path, "OLS")).read_text())
    assert "history_ahead" not in stored and "history" not in stored["training_cut"]
    assert configuration_digest(stored) == configuration_digest({**stored, "history_ahead": None})


def test_a_residual_model_over_a_history_base_refuses(hourly):
    """`verify` would run a ResidualModel without the rules its base's series are cut by, and cut its frame after the issue
    day: a base that reads history is refused when it is given, by object or by name."""
    from epftoolbox2.models import OLSModel, ResidualModel

    with pytest.raises(ValueError, match="reads series"):
        ResidualModel(make("FM"), OLSModel(["load_forecast"]))
    model = ResidualModel("FM", OLSModel(["load_forecast"]))
    with pytest.raises(ValueError, match="reads series"):
        model.bind_base(make("FM"))


# ---------------------------------------------------------------------------------------------------------------------------
# The re-review: a hybrid's cut, a late issue time, the control of a univariate model, verify's noise
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:overflow")
def test_a_history_model_with_row_blocks_is_cut_after_what_it_reads(hourly):
    """A history model that also declares row blocks used to be cut after the issue day, which took its target day's known-ahead
    values away: no forecast, and verify crashed. It is cut after what each forecast reads, and is ok."""
    report = pipeline(Rows(name="R")).verify(hourly, "2023-06-14", "2023-06-14", days=1, co_batch=True)
    report.assert_ok()
    assert report.checks[0].co_batch.cut.ok


@pytest.mark.filterwarnings("ignore:overflow")
def test_a_hybrid_s_row_days_after_the_issue_day_have_no_input_in_its_cut(hourly):
    """On the block's Saturday at h=1 the cut frame holds Sunday's rows for the target day's load forecast; every other value
    after D is NaN there, so a row part that counts its block's days with finite inputs sees six, not the backtest's seven:
    a leak. (Keeping Sunday's wind forecast would hide it.)"""
    report = pipeline(CountsItsRowDays(name="R")).verify(hourly, "2023-06-17", "2023-06-17", days=1, co_batch=True)
    assert not report.ok and not report.checks[0].co_batch.cut.ok
    pipeline(Rows(name="R")).verify(hourly, "2023-06-17", "2023-06-17", days=1, co_batch=True).assert_ok()


LATE_ISSUE = InformationSet("14:00")  # D+1's prices are out at 13:00 on D


def test_a_late_issue_time_still_forecasts_the_whole_target_day(hourly):
    """At 14:00 D+1's prices are out, and a "published" target's context ends at the day before the target day at the latest:
    D's last hour at h=1 (S = 24, every hour of D+1 forecast), D+1's at h=2, as a row model's newest target is capped at h-1."""
    spec = HistorySpec(CONTEXT, (), (), "price", newest_target="published")
    batches = history_batches(spec, hourly, ["2023-06-14"], LATE_ISSUE, "1h", (1, 2))
    for horizon, newest in ((1, "2023-06-14 23:00"), (2, "2023-06-15 23:00")):
        (item,) = batches[horizon].items
        assert item.newest["price"] == pd.Timestamp(newest, tz=ZONE)
        assert item.steps_ahead == 24 and all(slot is not None for slot in item.slots)
    model = make("FM", newest_target="published", known_ahead=())
    report = pipeline(model, information_set=LATE_ISSUE).run(
        hourly, "2023-06-14", "2023-06-14", horizon=2, forecast_only=True
    )
    counts = report.forecast_counts().iloc[0]
    assert counts["produced"] == counts["planned"] == 48


@pytest.mark.filterwarnings("ignore:overflow")
def test_a_late_issue_model_is_verified_on_what_it_reads(hourly):
    """At 14:00 the forecast at h=2 reads D+1's prices: the frame its cut keeps holds them, and the model is ok."""
    model = make("FM", newest_target="published", known_ahead=())
    report = pipeline(model, information_set=LATE_ISSUE).verify(hourly, "2023-06-14", "2023-06-14", horizon=2, days=1)
    report.assert_ok()
    assert report.checks[0].co_batch.cut.ok and report.checks[0].co_batch.cut.forecasts == 48


def test_the_manifest_hashes_a_late_issue_target_after_test_end(hourly, tmp_path):
    """The last issue day's context reads the price of the day after `test_end` at 14:00 (h=2): `history_ahead` hashes the target
    through it, so a resume over another price there refuses; at 11:45 nothing is read after `test_end`."""
    test = ("2023-06-05", "2023-06-18")
    model = make("FM", newest_target="published", known_ahead=())
    pipeline(model, information_set=LATE_ISSUE).run(hourly, *test, horizon=2, save_dir=tmp_path, forecast_only=True)
    path = suite.results(tmp_path, "FM")
    assert [(a["through"], a["columns"]) for a in json.loads(manifest_path(path).read_text())["history_ahead"]] == [
        ("2023-06-19", ["price"])
    ]
    lines = records_of(path)
    path.write_text("".join(f"{line}\n" for i, line in enumerate(lines) if i % 2 == 0))
    changed = hourly.copy()
    changed.loc["2023-06-19", "price"] += 50.0
    with pytest.raises(ManifestMismatchError, match="history_ahead"):
        model = make("FM", newest_target="published", known_ahead=())
        pipeline(model, information_set=LATE_ISSUE).run(
            changed, *test, horizon=2, save_dir=tmp_path, forecast_only=True
        )
    early = tmp_path / "early"
    pipeline(make("FM", newest_target="published", known_ahead=())).run(
        hourly, *test, horizon=2, save_dir=early, forecast_only=True
    )
    assert "history_ahead" not in json.loads(manifest_path(suite.results(early, "FM")).read_text())


@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.parametrize("newest", ["d-1", "published"])
def test_the_control_moves_a_univariate_model(hourly, newest):
    """A univariate model reads only its target's context, which the control now perturbs as any input (from D-1 on under
    "d-1", from D on under "published"): ok, not inconclusive."""
    report = pipeline(make("FM", known_ahead=(), newest_target=newest)).verify(
        hourly, "2023-06-14", "2023-06-14", days=1
    )
    row = report.table().iloc[0]
    assert (row["status"], row["changed"]) == ("ok", 0) and row["control moved"] == 24


@pytest.mark.filterwarnings("ignore:overflow")
def test_verify_s_cut_runs_report_no_shortfall(hourly):
    """The runs of verify's cut frames skip the later days they took away: they log no shortfall of forecasts."""
    import logging

    caught = []

    class Keep(logging.Handler):
        def emit(self, record):
            caught.append(record.getMessage())

    log = logging.getLogger("epftoolbox2.models.base")
    handler = Keep(level=logging.WARNING)
    log.addHandler(handler)
    try:
        pipeline(make("FM")).verify(hourly, "2023-06-14", "2023-06-14", horizon=2, days=1, co_batch=True)
    finally:
        log.removeHandler(handler)
    assert not [message for message in caught if "planned forecasts" in message], caught


class ReadsACopy(FakeFoundation):
    """Reads its load forecast from a copy of its own (the actual load, a look-ahead) instead of the frame it is given: the
    probe can't reach it."""

    copy = market("1h")["load_actual"]

    def _history_unit(self, unit, tasks, inputs, frame):
        own = frame.assign(load_forecast=self.copy.reindex(frame.index))
        return super()._history_unit(unit, tasks, inputs, own)


@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.parametrize("newest", ["d-1", "published"])
def test_the_control_runs_per_series_and_finds_a_series_it_can_t_reach(hourly, newest):
    """Each series is perturbed on its own: a model that reads its load forecast from a copy of its own moves under the target's
    perturbation alone, and the load forecast's moves nothing, a blind spot: inconclusive, as before the target was perturbed.
    The fake reading both from the frame is ok."""
    report = pipeline(ReadsACopy(name="FM", newest_target=newest)).verify(hourly, "2023-06-14", "2023-06-14", days=1)
    row = report.table().iloc[0]
    assert (row["status"], row["changed"], row["control moved"]) == ("inconclusive", 0, 24)
    assert report.checks[0].control.blind == ["load_forecast"]
    assert "positive control of the series ['load_forecast']" in "\n".join(report.problems())
    plain = pipeline(make("FM", newest_target=newest)).verify(hourly, "2023-06-14", "2023-06-14", days=1)
    assert plain.table().iloc[0]["status"] == "ok" and plain.checks[0].control.blind == []


def test_a_quiet_block_hides_this_context_s_shortfalls_alone_in_any_order():
    """`quiet_shortfall` is a depth of this thread or task, not the logger's level: two blocks ending out of order leave runs after
    the first one's end quiet only while the other lasts, and loud after both; a thread already running elsewhere reports its
    runs' shortfalls all along. (A thread started inside a block inherits its context on a free-threaded build, as a run's own
    threads should.)"""
    import logging
    import threading
    from collections import Counter

    from epftoolbox2.models.base import quiet_shortfall

    caught = []

    class Keep(logging.Handler):
        def emit(self, record):
            caught.append(record.getMessage())

    log = logging.getLogger("epftoolbox2.models.base")
    handler = Keep(level=logging.WARNING)
    log.addHandler(handler)
    model = make("FM")

    def report(tag):
        model.name = tag
        model._report_counts(48, 24, Counter({"missing_test_predictors": 24}), {})

    go = threading.Event()

    def elsewhere():
        go.wait()
        report("another thread")

    other = threading.Thread(target=elsewhere)
    other.start()  # before any block: its context is its own
    try:
        first, second = quiet_shortfall(), quiet_shortfall()
        first.__enter__()
        second.__enter__()
        report("inside both")
        go.set()
        other.join()
        first.__exit__(None, None, None)
        report("inside the second")
        second.__exit__(None, None, None)
        report("after both")
    finally:
        go.set()
        log.removeHandler(handler)
    assert [message.split(":")[0] for message in caught] == ["another thread", "after both"]
    assert log.level == logging.INFO
