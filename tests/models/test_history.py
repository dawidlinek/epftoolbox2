"""The history accessor (step 6h): what a sequence model reads for a forecast, cut by the information set.

The arrays are held against a reference built by hand from the frame in UTC: the EU's switch rule (the last Sundays of March
and October at 01:00 UTC) written out, and each rule of the step-3 grid (the built-in rules: prices published 13:00 on D-1,
the load forecast 10:00 on D-1, a wind forecast 18:00 on D-1, the actual load 1 h after its period, at the default issue time
11:45 Brussels) as the comparison it makes, with no code of the library's. Any cell the rules put after the issue time set to
1e9 moves no array.

`FakeFoundation` stands in for 6i's foundation models: a zero-shot day unit on the refit calendar whose forecast of each step
ahead is the context's last day (persistence) plus 0.1 x the known-ahead series, through history batches instead of rows. It
runs its units in this process, as a foundation model runs in one process per device (decision 19), with the day unit's
blocks, records and skips. Its context is an input the rules cut, never a training target, so `verify` reads it as
`newest_target="published"`: the rule alone is the boundary.
"""

import datetime
import json
import warnings
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.information_set import (
    Availability,
    InformationSet,
    InformationSetError,
    RuleChain,
    cuts_at_issue_day,
    newest_known_step,
    perturb_unknown,
    steps_known,
)
from epftoolbox2.models._calendar import block_days
from epftoolbox2.models._history import HistorySpec, history_batch, history_spec_of, reads_history
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
from epftoolbox2.results.manifest import ManifestMismatchError
from epftoolbox2.results.availability import (
    COLUMNS,
    HISTORY_COLUMNS,
    LATE,
    READ,
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
PRICE = HistorySpec(CONTEXT, ("load_forecast",), ("load_actual",), "price", accepts_missing=True)
LOAD = HistorySpec(CONTEXT, ("load_forecast",), ("generation_wind_forecast",), "load_actual")
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
def test_the_arrays_equal_a_reference_built_by_hand_from_the_utc_frame(hourly, quarter, freq, which, horizon):
    """40 issue days around both switches: the target's context ends at its newest step known at 11:45, the past series on the
    same steps (NaN after the actual load's newest known step, for a backend that takes missing values), the known-ahead
    series to the end of the target day, and each step ahead's (date, period), the repeated hour's second step forecasting it."""
    frame = hourly if freq == "1h" else quarter
    spec = SPECS[which]
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
    for which, spec in SPECS.items():
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


def test_the_repeated_hour_is_forecast_by_its_second_step_and_the_skipped_one_by_none(hourly):
    spring, autumn = history_batch(PRICE, hourly, ["2023-03-25", "2023-10-28"], INFO, "1h", 1).items
    assert [slot[1] for slot in spring.slots] == [0, 1, *range(3, 24)]
    assert autumn.slots[2] is None and autumn.slots[3] == (datetime.date(2023, 10, 29), 2)
    assert [slot[1] for slot in autumn.slots if slot is not None] == list(range(24))
    same(autumn.periods, [0, 1, -1, *range(2, 24)])


def test_a_known_ahead_series_not_known_through_the_target_day_is_late(hourly):
    """The wind forecast of D+1 is out at 18:00 on D, after the issue: late as a known-ahead series, with when it is known; as a
    past series of a price it is known as far as the target, D's last hour."""
    spec = HistorySpec(CONTEXT, ("generation_wind_forecast", "load_forecast"), (), "price")
    (item,) = history_batch(spec, hourly, ["2023-06-14"], INFO, "1h", 1).items
    assert set(item.late) == {"generation_wind_forecast"}
    late = item.late["generation_wind_forecast"]
    assert (late.known_label, late.issue_label, late.known - late.issue) == ("D 18:00", "D 11:45", 375.0)
    assert item.newest_minutes == {"price": 1380.0, "generation_wind_forecast": 1380.0, "load_forecast": 2820.0}
    (item,) = history_batch(HistorySpec(CONTEXT, (), ("generation_wind_forecast",)), hourly, ["2023-06-14"], INFO).items
    assert not item.late and np.isfinite(item.past["generation_wind_forecast"]).all()


def test_a_past_series_ends_at_its_newest_known_step(hourly):
    """The actual load is out to 09:00-10:00 at 11:45: with a backend that takes missing values its last 14 hours of a price's
    context (10:00 to 23:00) are NaN; with one that doesn't, it is late, known when its 23:00 value is, at 01:00 on D+1."""
    (item,) = history_batch(PRICE, hourly, ["2023-06-14"], INFO, "1h", 1).items
    load = item.past["load_actual"]
    assert np.isnan(load[-14:]).all() and np.isfinite(load[:-14]).all() and not item.late and not item.missing
    strict = HistorySpec(CONTEXT, ("load_forecast",), ("load_actual",), "price")
    (item,) = history_batch(strict, hourly, ["2023-06-14"], INFO, "1h", 1).items
    assert np.isfinite(item.past["load_actual"]).all()
    late = item.late["load_actual"]
    assert (late.known_label, late.issue_label) == ("D+1 01:00", "D 11:45")


def test_missing_values_of_the_data_are_listed_and_the_cut_nans_are_not(hourly):
    frame = hourly.copy()
    frame.loc["2023-06-12 05:00":"2023-06-12 06:00", "load_forecast"] = np.nan
    (item,) = history_batch(PRICE, frame, ["2023-06-14"], INFO, "1h", 1).items
    assert item.missing == ("load_forecast",)
    (item,) = history_batch(PRICE, hourly, ["2023-06-14"], INFO, "1h", 1).items
    assert item.missing == ()


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
    by_s = {call.steps_ahead for call in calls}
    assert by_s == {23, 24, 25}
    (only,) = [call for call in calls if call.steps_ahead == 23]
    assert [batch.items[i].issue_day for i in only.items] == [datetime.date(2023, 3, 25)]
    assert [len(call.items) for call in batch.batches(64)] == [1, 38, 1]
    assert [call.items for call in batch.batches(64, which=[5, 3])] == [(3, 5)]


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
    spec = HistorySpec(40, ("load_forecast",), (), "price")
    (item,) = history_batch(spec, hourly, ["2023-06-14"], rules, "1h", 1).items
    assert "load_forecast" in item.late
    (fine,) = history_batch(HistorySpec(10, ("load_forecast",), (), "price"), hourly, ["2023-06-14"], rules).items
    assert not fine.late
    target = HistorySpec(40, (), (), "load_forecast")
    with pytest.raises(InformationSetError, match="rules over time"):
        history_batch(target, hourly, ["2023-06-14"], rules, "1h", 1)


def test_steps_known_is_the_probe_s_reading_including_an_as_of_value_of_a_later_row(hourly):
    """`steps_known` calls known exactly the cells `perturb_unknown` leaves alone, column by column. An `as_of` value of a row
    after D is that row's own run's, so unknown on D though its time of day is before the issue time."""
    rules = InformationSet(rules={"wx": Availability.as_of("09:00")})
    frame = hourly.assign(wx=hourly["price"] + 7.0)
    for day in ("2023-03-25", "2023-03-26", "2023-06-14", "2023-10-29"):
        probed, _, _ = perturb_unknown(frame, day, rules)
        for name in frame.columns:
            read = steps_known(rules, name, frame.index, day)
            same(read.known, probed[name].to_numpy() == frame[name].to_numpy())
    (item,) = history_batch(HistorySpec(CONTEXT, ("wx",), (), "price"), frame, ["2023-06-14"], rules).items
    assert item.newest_minutes["wx"] == 1380.0 and "wx" in item.late
    position = newest_known_step(rules, "price", frame.index, "2023-06-14")
    assert frame.index[position] == pd.Timestamp("2023-06-14 23:00", tz=ZONE)


def test_what_the_accessor_refuses(hourly):
    with pytest.raises(ValueError, match="shifted column"):
        HistorySpec(CONTEXT, ("load_forecast_d+1",))
    with pytest.raises(ValueError, match="named twice"):
        HistorySpec(CONTEXT, ("load_forecast",), ("load_forecast",))
    with pytest.raises(ValueError, match="a list of column names"):
        HistorySpec(CONTEXT, "load_forecast")
    for bad in (0, 1.5, True):
        with pytest.raises(ValueError, match="context"):
            HistorySpec(bad)
    with pytest.raises(ValueError, match="freq"):
        history_batch(PRICE, hourly, ["2023-06-14"], INFO, "1D")
    with pytest.raises(InformationSetError, match="no availability rule"):
        history_batch(HistorySpec(CONTEXT, ("mystery",)), hourly.assign(mystery=1.0), ["2023-06-14"], INFO)
    with pytest.raises(ValueError, match="no column"):
        history_batch(HistorySpec(CONTEXT, ("absent",)), hourly, ["2023-06-14"], INFO)
    off = hourly.copy()
    off.index = off.index + pd.Timedelta(minutes=5) * (np.arange(len(off)) == 10)
    with pytest.raises(ValueError, match="off the 1h steps"):
        history_batch(PRICE, off, ["2023-06-14"], INFO)
    slow = InformationSet(rules={"price": Availability.published("13:00", -40)})
    with pytest.raises(InformationSetError, match="a rule this slow"):
        history_batch(PRICE, hourly, ["2023-06-14"], slow)
    assert history_batch(PRICE, hourly, [], INFO).items == ()


def test_a_frame_shorter_than_the_context_holds_nan_where_it_has_no_row(hourly):
    """T is fixed whatever the frame has: steps before its first row are NaN, and the item says the target is missing."""
    frame = hourly.loc["2023-06-10":]
    (item,) = history_batch(PRICE, frame, ["2023-06-14"], INFO).items
    assert len(item.target) == CONTEXT * 24 and np.isnan(item.target[:48]).all() and "price" in item.missing


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
        # Its context is an input the rules cut: `verify` reads such a model's boundary from the rule, as for "published".
        kw.setdefault("newest_target", "published")
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
        return HistorySpec(self.context, self.known_ahead, self.past, target, self.accepts_missing)

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
        """The day unit's checks, then the series': the cut's rules kept for the run, the series as the inputs it reads (the
        manifest hashes them, `report.vintages()` lists them, the probe perturbs them), and each series late on an issue day
        refused at "raise" or listed, one row per series, at "warn"."""
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
        spec = self._history_spec(target)
        self._used_predictors = list(dict.fromkeys([*self._used_predictors, *spec.past, *spec.known_ahead]))
        if information_set is None or availability == "none" or test_start is None:
            return expanded
        horizons = tuple(range(1, horizon + 1)) if horizons is None else tuple(horizons)

        def batches(start, end):
            days = [day for day in self._days_of(start, end, None) if weekdays is None or day.weekday() in weekdays]
            return [history_batch(spec, data, days, information_set, freq, h) for h in horizons]

        found = batches(test_start, test_end) if windows is None else []
        for _, (start, end) in windows.periods if windows is not None else ():
            found += batches(start, end)
        if availability == "raise":
            late = sorted({name for batch in found for name in batch.late()})
            if late:
                raise InformationSetError(
                    f"{self.name}: {late} not known at the issue time as far as the forecasts read"
                )
            return expanded
        self._availability_listing = [*self._availability_listing, *history_rows(found)]
        for name, (start, end) in windows.periods if windows is not None else ():
            self._window_listings[name] = [*self._window_listings.get(name, []), *history_rows(batches(start, end))]
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
        """One unit's forecasts: the call holds every day of its block that the frame has with complete series, the run's
        tasks or not, so a forecast is the same alone, in its block and after a resume."""
        cfg = inputs.config
        outcomes, issued = _issued(inputs, tasks)
        first_ordinal = cfg["refit"]["first_ordinal"]
        rows = set(np.concatenate([arrays["days"] for arrays in inputs.hour_arrays.values()]).tolist())
        days = [
            day
            for day in block_days(unit.anchor, cfg["refit"]["rules"][unit.member])
            if day.toordinal() - first_ordinal in rows
        ]
        batch = history_batch(
            self._history_spec(self._target), frame, days, self._history_rules, self._freq, unit.horizon
        )
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


def make(name, **params):
    return FakeFoundation(name=name, **params)


def fake_by_hand(frame, day, horizon, freq="1h", context=CONTEXT):
    """The fake's forecasts of one issue day, {period: value}, from the reference by hand."""
    spec = HistorySpec(context, ("load_forecast",), (), "price", True)
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


def test_the_fake_reads_history_and_is_never_cut(hourly):
    """A model reads history when its class defines `_history_spec`; `verify` never runs it on the frame cut after the issue
    day, whose next rows hold its known-ahead values. A day unit without history is cut, as before."""
    model = make("FM")
    assert reads_history(model) and history_spec_of(model, "price") == HistorySpec(
        CONTEXT, ("load_forecast",), (), "price", True
    )
    assert model._predicts_together and not cuts_at_issue_day(model)
    assert not model._engine_capable()

    class Day(FakeFoundation):
        _history_spec = None

    assert not reads_history(Day()) and cuts_at_issue_day(Day())
    from epftoolbox2.models import OLSModel

    assert not reads_history(OLSModel(["price_d-1"])) and history_spec_of(OLSModel(["price_d-1"]), "price") is None


@pytest.mark.filterwarnings("ignore:Availability")  # the load forecast of D+2, late at horizon 2: listed, and used
@pytest.mark.parametrize("freq", ["1h", "15min"])
def test_the_fake_forecasts_what_its_reference_by_hand_says(hourly, quarter, freq):
    frame = hourly if freq == "1h" else quarter
    days = ["2023-03-24", "2023-03-28"] if freq == "1h" else ["2023-03-25", "2023-03-26"]
    report = pipeline(make("FM")).run(frame, *days, horizon=2, freq=freq, forecast_only=True)
    counts = report.forecast_counts().iloc[0]
    assert counts["produced"] == counts["planned"] > 0, counts.to_dict()
    records = report.refs["FM"]._results
    expected = {}
    for record in records:
        key = (datetime.date.fromisoformat(record["run_date"]), record["horizon"])
        if key not in expected:
            expected[key] = fake_by_hand(frame, *key, freq)
        assert record["prediction"] == expected[key][record["hour"]], record


@pytest.mark.filterwarnings("ignore:overflow")
def test_verify_is_ok_with_its_control_rerun_and_co_batch(hourly):
    """`adapter_suite.check_verify` on a block's first, middle and last days and both switches: the six other days of the
    "7D" block named, nothing moved by noise in them, every forecast moved by the control. A fake that reads its call's other
    rows (`by_batch`) is a leak, and so is the cut one step late."""
    days = ["2023-03-20", "2023-03-22", "2023-03-25", "2023-03-26", "2023-10-28", "2023-10-29"]
    suite.check_verify(make, hourly, INFO, days, items=6)
    suite.check_verify(make, hourly, INFO_AHEAD, ["2023-10-28"], horizon=2, items=6)
    with pytest.raises(AssertionError):
        suite.check_verify(lambda name: make(name, by_batch=1e-3), hourly, INFO, ["2023-03-22"])


@pytest.mark.filterwarnings("ignore:overflow")
def test_verify_catches_a_cut_one_step_late(hourly, monkeypatch):
    """With every series cut one step after its newest known one, the actual load's context ends at 10:00 on D, out at 12:00,
    which the persistence of its last day reads: a leak."""
    from epftoolbox2.information_set import StepsKnown

    real = StepsKnown.newest
    monkeypatch.setattr(StepsKnown, "newest", lambda self: min(real(self) + 1, len(self.available) - 1))
    report = pipeline(make("FM")).verify(hourly, "2023-03-22", "2023-03-22", target="load_actual", horizon=1, days=1)
    assert report.table().iloc[0]["changed"] > 0
    with pytest.raises(AssertionError):
        report.assert_ok()


def test_a_1d_model_names_no_other_item_of_its_call():
    assert make("FM", refit="1D")._co_batch_items("2023-03-22") == []
    assert len(make("FM")._co_batch_items("2023-03-22")) == 6


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
    assert (24, 64) in model._calls and {23, 25} <= {s for s, _ in model._calls}, model._calls
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
    variants = ({}, {"context": 5}, {"batch_size": 1, "offset": 1.0})
    suite.check_loop_and_shards(
        make, hourly, tmp_path / "loop", test, horizon=2, variants=variants, information_set=info
    )
    suite.check_resume(make, hourly, tmp_path / "resume", test, horizon=2, information_set=info)
    suite.check_engine(make, hourly, test, horizon=2, information_set=info)
    suite.check_dst_counts(make, hourly, "2023-03-26", "2023-10-29", information_set=INFO)
    suite.check_quarter_hours(make, quarter, ("2023-02-20", "2023-02-21"), information_set=INFO)
    suite.check_preflight_without_extra(make, monkeypatch, hourly, test)
    suite.check_provenance(make, hourly, tmp_path / "provenance", test, information_set=INFO)


# ---------------------------------------------------------------------------------------------------------------------------
# The listing, explain and vintages
# ---------------------------------------------------------------------------------------------------------------------------


def test_availability_has_one_row_per_series_with_its_newest_step_and_class(hourly, tmp_path):
    """At "warn", one row per series, not per lagged step: the target to D 23:00, the actual load (NaN after it) to 09:00, the
    load forecast to D+1 23:00, each with its rule's class; the wind forecast, late as a known-ahead series, with when it is
    out. No warning for the series read in time; one for the late one. A report read back from the files shows the same."""
    model = make("FM", past=["load_actual"], known_ahead=["load_forecast", "generation_wind_forecast"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = pipeline(model, information_set=INFO_AHEAD).run(
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
        "price": (READ, "D 23:00", "snapshot"),
        "load_actual": (READ, "D 09:00", "latest_revision"),
        "load_forecast": (READ, "D+1 23:00 / D+2 23:00", "latest_revision"),  # read to the target day's end
        "generation_wind_forecast": (LATE, "D 23:00", "latest_revision"),
    }
    wind = shown["generation_wind_forecast"]
    assert (wind["known"], wind["issued"], wind["horizons"], wind["periods"]) == ("D+1 18:00", "D 11:45", "1-2", "0-23")
    again = pipeline(
        make("FM", past=["load_actual"], known_ahead=["load_forecast", "generation_wind_forecast"]),
        information_set=INFO_AHEAD,
    )
    with pytest.warns(UserWarning, match="generation_wind_forecast is late"):
        read_back = again.report(tmp_path, "2023-06-12", "2023-06-14", horizon=2, forecast_only=True)
    pd.testing.assert_frame_equal(read_back.availability(), table)
    with pytest.raises(InformationSetError, match="generation_wind_forecast"):
        pipeline(make("FM", known_ahead=["generation_wind_forecast"]), availability="raise").run(
            hourly, "2023-06-12", "2023-06-14", horizon=1, forecast_only=True
        )


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
    assert not set(HISTORY_COLUMNS) & set(report.refs["OLS"].availability_listing[0])
    assert listing_frame({"m": []}).columns.tolist() == COLUMNS


def test_merging_listings_keeps_a_series_row_s_newest_step_and_class(hourly):
    batches = [history_batch(PRICE, hourly, ["2023-06-14"], INFO, "1h", 1)]
    rows = history_rows(batches)
    assert merge_listings(rows, []) == rows
    later = history_rows([history_batch(PRICE, hourly, ["2023-06-15"], INFO, "1h", 2)])
    merged = {(row["predictor"], row["horizons"]): row for row in merge_listings(rows, later)}
    assert merged[("price", "1-2")]["newest"] == "D 23:00" and merged[("price", "1-2")]["vintage"] == "snapshot"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_about({"FM": rows})  # nothing late: no warning


def test_explain_shows_one_row_per_series(hourly):
    model = make("FM", past=["load_actual"], known_ahead=["load_forecast", "generation_wind_forecast"])
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


def test_vintages_list_the_series(hourly):
    model = make("FM", past=["load_actual"])
    report = pipeline(model).run(hourly, "2023-06-12", "2023-06-12", horizon=1, forecast_only=True)
    table = report.vintages()
    assert {"load_actual", "load_forecast"} <= set(table["predictor"])
