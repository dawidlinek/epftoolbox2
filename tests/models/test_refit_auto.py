"""`refit="auto"` (step 4e): each training window refits on its own rule, `"1D"` up to 84 days and `"7D"` for a longer one or
`"expanding"` (`_calendar.member_rules`).

Member i of a list under `"auto"` is the single-window model with member i's window and its resolved rule, to the bit: its rows are
cut on the anchor of its own rule, its scaler and mask come from those rows, and the forecast is the members' `fsum / M`. A pool job
is a block of the coarsest rule, holding the long members' one fit and the short members' daily ones; a record's `refit_anchor` is
the coarsest member's anchor, the oldest fit it uses, and its `newest_target_day` the newest of its members'. These tests hold that
key by key, a single window under `"auto"` to its rule in every record, `[56, 728]` to hand fits over a leap day and both clock
changes, each member to 3g's leak cases on its own anchor, and the run to the probes, the pool and a resume. Mutants of the
implementation are in `epftoolbox2-research/library_check_step4/refit_auto/`.
"""

import datetime
import json
import math
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
import yaml

from epftoolbox2 import tui
from epftoolbox2.information_set import Availability, InformationSet, InformationSetError, positive_control
from epftoolbox2.models import BaseModel, LassoCVModel, LEARModel, OLSModel, RidgeModel
from epftoolbox2.models import _worker as worker
from epftoolbox2.models._calendar import (
    anchors_of_rules,
    block_rule,
    check_refit,
    member_rules,
    newest_rule,
)
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path
from epftoolbox2.results.windows import RunWindows
from tests.synthetic import PREDICTORS, make_market_data

RULES = {
    "load": Availability.published("10:00", 10),
    "wind": Availability.published("10:00", 10),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "aux": Availability.static(),
    "price": Availability.published("13:00", 1),  # D's price is known at 11:45 on D: what "published" trains on
}
INFO = InformationSet("11:45", "Europe/Brussels", rules=RULES)
PROVENANCE = ("package_version", "git_sha", "created_at", "updated_at")
SWITCH_DAYS = {"spring": "2023-03-26", "autumn": "2023-10-29"}  # Sundays, so a run around one crosses two "7D" blocks
SHORT, LONG = 14, 85  # on either side of 84 days: "1D" and "7D" under "auto"
MEMBERS = [SHORT, LONG, "expanding"]
RESOLVED = ["1D", "7D", "7D"]


class InProcessPool:
    """Stands in for the worker pool, whose start is most of the time these runs would take."""

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


def day(text):
    return datetime.date.fromisoformat(text)


def shift(text, days):
    return (day(text) + datetime.timedelta(days=days)).isoformat()


def local_dates(data):
    return np.asarray(pd.DatetimeIndex(data.index.tz_localize(None)).normalize())


def keyed(records):
    return {(r["run_date"], r["hour"], r["horizon"]): r for r in records}


def texts(records, *drop):
    """Key -> the record as JSON text without `drop`: numbers compare as text, so a NaN `actual` equals itself."""
    return {
        key: json.dumps({k: v for k, v in r.items() if k not in drop}, sort_keys=True)
        for key, r in keyed(records).items()
    }


def lines(path):
    """A result file's records as sorted text: the order a pool finishes them in isn't fixed."""
    return sorted(line for line in Path(path).read_text().splitlines() if line.strip())


def bits(value):
    return float(value).hex()


def market(switch, freq="1h", seed=3):
    """110 days before a switch and 20 after, drivers out to 3 days: enough for an 85-day window refitted on Mondays from a
    validation window 9 days before the switch. A daily frame is the hourly one at noon."""
    data = make_market_data(
        start=shift(SWITCH_DAYS[switch], -110),
        days=130,
        freq="1h" if freq == "1D" else freq,
        horizon=3,
        noise=5.0,
        seed=seed,
    )
    return data.at_time("12:00") if freq == "1D" else data


def move(data, rows, columns, seed=1):
    """A copy of `data` with every finite value of `columns` on `rows` moved by noise."""
    out = data.copy()
    values = out[columns].to_numpy(dtype=float, copy=True)
    where = rows[:, None] & np.isfinite(values)
    values[where] += np.random.default_rng(seed).normal(0, 3, size=int(where.sum()))
    out[columns] = values
    return out


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def test_auto_gives_each_window_its_rule():
    """Decision 16 of the step 4 spec: a window of up to 84 days refits daily, a longer one and "expanding" weekly. Any other
    `refit` is every member's rule. The coarsest rule holds every other's blocks, and the finest is in every other's."""
    assert check_refit("auto") == "auto"
    assert member_rules((56, 84, 85, 728, "expanding"), "auto") == ("1D", "1D", "7D", "7D", "7D")
    assert member_rules((1,), "auto") == ("1D",)
    assert member_rules((56, 728), "7D") == ("7D", "7D")
    assert member_rules((56, 728), None) == (None, None)
    assert block_rule(("1D", "7D", "7D")) == "7D" and newest_rule(("1D", "7D", "7D")) == "1D"
    assert block_rule(("1MS",)) == newest_rule(("1MS",)) == "1MS"
    assert (
        block_rule(("2D", "14D")) == "14D"
        and block_rule(("1MS", "3MS")) == "3MS"
        and block_rule(("1D", "1MS")) == "1MS"
    )
    assert block_rule((None, None)) is None
    with pytest.raises(ValueError, match="don't nest"):
        block_rule(("3D", "7D"))
    with pytest.raises(ValueError, match="don't nest"):
        block_rule(("7D", "1MS"))
    issued = ["2024-02-28", "2024-02-29", "2024-03-05"]
    assert anchors_of_rules(issued, ("1D", "7D")) == [
        day(d) for d in ("2024-02-26", "2024-02-28", "2024-02-29", "2024-03-04", "2024-03-05")
    ]
    model = OLSModel(PREDICTORS, training_window=[728, 56, "expanding"], refit="auto")
    assert model._refit_rules == ("1D", "7D", "7D") and model._refit_block == "7D"
    model.training_window = 56
    assert model._refit_rules == ("1D",) and model._refit_block == "1D"
    for cls in (OLSModel, LassoCVModel, RidgeModel, LEARModel):
        assert cls(PREDICTORS, refit="auto").refit == "auto"


def test_yaml_and_the_tui_keep_auto(tmp_path):
    pipeline = ModelPipeline()
    pipeline.add_model(LEARModel(PREDICTORS, refit="auto", name="lear"))
    path = tmp_path / "pipeline.yaml"
    pipeline.save(path)
    assert yaml.safe_load(path.read_text())["models"][0]["params"]["refit"] == "auto"
    loaded = ModelPipeline.load(path).models[0]
    assert loaded.refit == "auto" and loaded._refit_rules == ("1D", "1D", "7D", "7D")
    for name in ("OLSModel", "LassoCVModel", "RidgeModel", "LEARModel"):
        field = next(f for f in tui.PARAM_SCHEMAS[name] if f["name"] == "refit")
        assert "auto" in field["description"], name
        model = tui._instantiate(name, {"predictors": PREDICTORS, "training_window": [56, 728], "refit": "auto"})
        assert model._refit_rules == ("1D", "7D"), name


# ---------------------------------------------------------------------------
# A single window under "auto" is the model with its rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("window", "rule"), [(SHORT, "1D"), (84, "1D"), (LONG, "7D"), ("expanding", "7D")])
def test_a_single_window_under_auto_is_its_rule_in_every_record(tmp_path, window, rule):
    """`training_window=W, refit="auto"` writes the records of `refit=rule` byte for byte, `refit_anchor` included. The manifest
    differs in `training_cut` alone, which names "auto" and the rule it resolved to, so one doesn't resume the other."""
    data = market("spring")
    paths = {name: tmp_path / name / "m.jsonl" for name in ("auto", "rule")}
    for name, refit in (("auto", "auto"), ("rule", rule)):
        OLSModel(PREDICTORS, training_window=window, refit=refit, name="m").run(
            data, "2023-03-24", "2023-03-28", horizon=2, save_to=str(paths[name])
        )
    assert lines(paths["auto"]) == lines(paths["rule"])
    anchors = {json.loads(line)["refit_anchor"] for line in lines(paths["auto"])}
    assert anchors == (
        {"2023-03-24", "2023-03-25", "2023-03-26", "2023-03-27", "2023-03-28"}
        if rule == "1D"
        else {"2023-03-20", "2023-03-27"}
    )
    manifests = {}
    for name, path in paths.items():
        manifests[name] = json.loads(manifest_path(path).read_text())
        for field in PROVENANCE:
            manifests[name].pop(field)
    auto_cut, rule_cut = manifests["auto"].pop("training_cut"), manifests["rule"].pop("training_cut")
    assert manifests["auto"] == manifests["rule"]
    assert rule_cut["refit"] == rule and "refit_rules" not in rule_cut
    assert auto_cut == {**rule_cut, "refit": "auto", "refit_rules": [rule]}
    with pytest.raises(ManifestMismatchError, match="training_cut"):
        OLSModel(PREDICTORS, training_window=window, refit=rule, name="m").run(
            data, "2023-03-24", "2023-03-28", horizon=2, save_to=str(paths["auto"])
        )


# ---------------------------------------------------------------------------
# Member i of a mixed list is the single-window model with its resolved rule, to the bit
# ---------------------------------------------------------------------------


class Cell(NamedTuple):
    model: str  # "ols", "lasso", "ridge" or "lear"
    freq: str  # "1h", "15min" or "1D"
    switch: str  # "spring" or "autumn"
    horizons: tuple
    friday: bool  # issue_days=["friday"]
    published: bool  # newest_target="published"
    validation: bool  # a validation window and a test window


# Every level of every axis, each model at least once; LassoCV on one hourly cell of one horizon, which keeps its fits few.
CELLS = [
    Cell("ols", "1h", "spring", (1, 3), False, False, False),
    Cell("ols", "15min", "autumn", (1,), True, True, False),
    Cell("ols", "1D", "autumn", (1, 3), True, False, True),
    Cell("ridge", "1D", "spring", (1,), False, True, True),
    Cell("ridge", "15min", "spring", (1, 3), False, False, True),
    Cell("lasso", "1h", "autumn", (1,), False, False, False),
    Cell("lear", "1h", "spring", (1, 3), True, True, False),
    Cell("lear", "1h", "autumn", (1,), False, True, True),
]


def run_cell(cell, window, refit, data):
    options = {"refit": refit, "newest_target": "published" if cell.published else "d-1", "name": "m"}
    if cell.model == "ols":
        model = OLSModel(PREDICTORS, training_window=window, **options)
    elif cell.model == "ridge":
        model = RidgeModel(PREDICTORS, training_window=window, **options)
    elif cell.model == "lear":
        # Without is_weekday, minus is_weekend after LEAR's centring: an exact twin, whose tie in LARS turns on a last bit.
        lear = [p for p in PREDICTORS if not p.startswith("is_weekday")]
        model = LEARModel(lear, training_window=window, criterion="aic", solver="cd", **options)
    else:
        model = LassoCVModel(PREDICTORS, training_window=window, cv=3, max_iter=1000, **options)
    switch = SWITCH_DAYS[cell.switch]
    kwargs = {
        "freq": cell.freq,
        "horizons": cell.horizons,
        "issue_days": ["friday"] if cell.friday else None,
        "information_set": INFO if cell.published else None,
    }
    if cell.validation:
        windows = RunWindows.fixed((shift(switch, -9), shift(switch, -8)), (shift(switch, -2), shift(switch, 1)))
        return model.run(data, windows=windows, **kwargs)._results
    return model.run(data, shift(switch, -2), shift(switch, 5 if cell.friday else 1), **kwargs)._results


def check_mixed(listed, singles, coarse):
    """The list model's records against its single-window models', each with its resolved rule: the keys every member
    forecast, each member's prediction to the bit, the mean as `fsum / M` to the bit, `refit_anchor` that of the coarsest member
    (`coarse`, its place in the list), `newest_target_day` the newest of the members', and everything else the first member's."""
    lst, each = keyed(listed), [keyed(single) for single in singles]
    assert set(lst) == set.intersection(*(set(single) for single in each))
    assert lst, "nothing to compare"
    for key, record in lst.items():
        assert [bits(p) for p in record["member_predictions"]] == [
            bits(single[key]["prediction"]) for single in each
        ], key
        assert bits(record["prediction"]) == bits(math.fsum(record["member_predictions"]) / len(each))
        assert record["coefficients"] == []
        assert record["refit_anchor"] == each[coarse][key]["refit_anchor"], key
        if "newest_target_day" in record:
            assert record["newest_target_day"] == max(single[key]["newest_target_day"] for single in each), key
        own = ("prediction", "member_predictions", "coefficients", "refit_anchor", "newest_target_day")
        rest = {k: v for k, v in record.items() if k not in own}
        first = {k: v for k, v in each[0][key].items() if k not in own}
        assert json.dumps(rest, sort_keys=True) == json.dumps(first, sort_keys=True), key
    return lst


@pytest.mark.parametrize("cell", CELLS, ids=["-".join(map(str, cell)) for cell in CELLS])
def test_each_member_is_its_single_window_model_with_its_own_rule(cell):
    """`member_predictions[i]` is the prediction of the model with `training_window` its i-th member and `refit` that member's
    rule, written out ("1D" or "7D"), to the bit; the single-window models under "auto" are those, record for record. The
    short member is anchored on each issue day and the long ones on its Monday, which the record's `refit_anchor` names."""
    data = market(cell.switch, cell.freq)
    listed = run_cell(cell, MEMBERS, "auto", data)
    singles = [run_cell(cell, member, rule, data) for member, rule in zip(MEMBERS, RESOLVED)]
    found = check_mixed(listed, singles, coarse=1)
    assert {day(r["refit_anchor"]).weekday() for r in found.values()} == {0}
    assert any(r["run_date"] != r["refit_anchor"] for r in found.values())
    for member, single in zip(MEMBERS, singles):
        assert texts(run_cell(cell, member, "auto", data)) == texts(single), member
    if cell.validation:
        assert {record["window"] for record in found.values()} == {"validation", "test"}


def test_a_list_of_one_rule_is_unchanged_by_the_rules_of_its_members():
    """A list under one rule (here "7D") is what it was: every member on the block's anchor, the record the first member's."""
    data = market("spring")
    cell = CELLS[0]
    listed = run_cell(cell, MEMBERS, "7D", data)
    singles = [run_cell(cell, member, "7D", data) for member in MEMBERS]
    check_mixed(listed, singles, coarse=0)


# ---------------------------------------------------------------------------
# Hand fits: [56, 728] over a leap day and both clock changes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_years():
    """From 1 February 2022 to the end of October 2024 at 1 h and 15 min: 728 days before a Monday anchor of the leap day's week
    and of both 2024 switches (31 March, 27 October in Europe/Warsaw). Each frequency is built when a test first asks for it."""
    frames = {}

    def frame(freq):
        if freq not in frames:
            frames[freq] = make_market_data(start="2022-02-01", days=1005, freq=freq, horizon=2, noise=5.0, seed=11)
        return frames[freq]

    return frame


def period_of(wall, freq):
    return np.asarray(wall.hour * 4 + wall.minute // 15 if freq == "15min" else wall.hour)


def hand_fit(data, freq, period, anchor, issue, horizon, names, window):
    """OLS without intercept on the standardised target, from the rows of `period` dated anchor-W-h to anchor-1-h with every
    value, both rows of a repeated hour included; columns 0/1 on those rows stay as they are, others are standardised on the rows
    used. A target in a repeated hour has no value. Forecast from the last row of `period` on `issue`."""
    wall = data.index.tz_localize(None)
    dates = wall.normalize()
    here = period_of(wall, freq) == period
    a = pd.Timestamp(anchor)
    rows = np.flatnonzero(
        here
        & np.asarray(dates >= a - pd.Timedelta(days=window + horizon))
        & np.asarray(dates <= a - pd.Timedelta(days=1 + horizon))
    )
    prices = pd.Series(data["price"].to_numpy(), index=wall)
    prices = prices[~prices.index.duplicated(keep=False)]
    target = prices.reindex(wall[rows] + pd.Timedelta(days=horizon)).to_numpy()
    x = data[names].to_numpy(dtype=float)
    binary = np.array([set(np.unique(column[np.isfinite(column)])) <= {0.0, 1.0} for column in x[rows].T])
    usable = np.isfinite(target) & np.isfinite(x[rows]).all(axis=1)
    y, train = target[usable], x[rows][usable]
    means = np.where(binary, 0.0, train.mean(axis=0))
    deviations = np.where(binary, 1.0, train.std(axis=0, ddof=1))
    coefficients, *_ = np.linalg.lstsq((train - means) / deviations, (y - y.mean()) / y.std(ddof=1), rcond=None)
    test_row = np.flatnonzero(here & np.asarray(dates == pd.Timestamp(issue)))[-1]
    return float(((x[test_row] - means) / deviations) @ coefficients * y.std(ddof=1) + y.mean())


@pytest.mark.parametrize("freq", ["1h", "15min"])
@pytest.mark.parametrize(
    ("issues", "periods"),
    [
        (("2024-02-28", "2024-03-01"), {"1h": (0, 7, 18), "15min": (0, 30, 75)}),
        (("2024-03-30", "2024-04-01"), {"1h": (1, 2, 3, 18), "15min": (7, 8, 11, 12, 75)}),
        (("2024-10-26", "2024-10-28"), {"1h": (1, 2, 3, 18), "15min": (7, 8, 11, 12, 75)}),
    ],
    ids=["leap-day", "spring", "autumn"],
)
def test_56_and_728_are_hand_fits_on_their_own_anchors(two_years, freq, issues, periods):
    """`[56, 728]` under "auto": the 56-day member of each forecast is one OLS on the rows cut on its issue day, the 728-day member
    one on the rows cut on the Monday on or before it, scaler included, and the forecast their mean. Over the leap day, and over
    the 2024 switches, where the hour skipped in spring has no forecast on its day and the repeated autumn hour two training rows;
    the spring and autumn runs end on a Monday, which starts a block."""
    data = two_years(freq)
    names = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "is_weekday_d+{horizon}"]
    model = OLSModel(names, training_window=[56, 728], refit="auto")
    records = keyed(model.run(data, *issues, horizon=2, freq=freq)._results)
    issued = pd.date_range(*issues, freq="D").date
    skipped = {"1h": {2}, "15min": {8, 9, 10, 11}}[freq]  # 02:00 to 02:59 on 31 March 2024 isn't on the clock
    checked = 0
    for issue in (d.isoformat() for d in issued):
        for period in periods[freq]:
            for horizon in (1, 2):
                gone = "2024-03-31" in (issue, shift(issue, horizon)) and period in skipped
                assert ((issue, period, horizon) in records) != gone, (issue, period, horizon)
                if gone:  # a slot the clock skips: not planned
                    continue
                record = records[(issue, period, horizon)]
                monday = (day(issue) - datetime.timedelta(days=day(issue).weekday())).isoformat()
                assert record["refit_anchor"] == monday
                expanded = [name.replace("{horizon}", str(horizon)) for name in names]
                short = hand_fit(data, freq, period, issue, issue, horizon, expanded, 56)
                long = hand_fit(data, freq, period, monday, issue, horizon, expanded, 728)
                assert record["member_predictions"] == pytest.approx([short, long], abs=1e-9)
                assert record["prediction"] == pytest.approx((short + long) / 2, abs=1e-9)
                checked += 1
    assert checked >= len(issued) * len(periods[freq]) * 2 - 6


# ---------------------------------------------------------------------------
# What a run of "auto" fits, and where its jobs are
# ---------------------------------------------------------------------------


def test_a_block_holds_the_long_member_s_one_fit_and_the_short_member_s_daily_ones(monkeypatch):
    """Two weeks at horizon 1: the 14-day member is fitted on each of the 14 issue days, the 85-day one on each of the two
    Mondays, per period; a pool job is a Monday-to-Sunday week. "7D" on both would fit each twice, "1D" on both 14 times."""
    data = market("spring")
    fitted, jobs = [], []
    real_unit, real_block = worker._fit_unit, worker._run_block

    def count_unit(unit, tasks):
        fitted.append((unit.member, unit.anchor))
        return real_unit(unit, tasks)

    def count_block(tasks):
        jobs.append(sorted({task[2] for task in tasks}))
        return real_block(tasks)

    monkeypatch.setattr(worker, "_fit_unit", count_unit)
    monkeypatch.setattr("epftoolbox2.models.base._run_block", count_block)
    ref = OLSModel(PREDICTORS, training_window=[SHORT, LONG], refit="auto").run(
        data, "2023-03-27", "2023-04-09", horizon=1
    )
    assert ref.produced == 14 * 24
    short = [anchor for member, anchor in fitted if member == 0]
    long = [anchor for member, anchor in fitted if member == 1]
    assert len(short) == 14 * 24 and len(set(short)) == 14
    assert len(long) == 2 * 24 and set(long) == {day("2023-03-27"), day("2023-04-03")}
    assert sorted(jobs) == [list(range(0, 7)), list(range(7, 14))]


def test_the_frame_must_reach_back_from_each_member_s_own_anchor():
    """The 85-day member is fitted on Monday 3 April for the forecasts of Wednesday 5 April, so its rows start on 3 April - 85 - 1,
    6 days before the 14-day member's 5 April - 14 - 1 would ask for nothing earlier: a frame from that day gives the long frame's
    records, one from the day after is refused, naming the member. Daily, the long member would ask 2 days less."""
    data = market("spring")
    needed = shift("2023-04-03", -86)

    def run(frame, refit="auto"):
        model = OLSModel(PREDICTORS, training_window=[SHORT, LONG], refit=refit)
        return texts(model.run(frame, "2023-04-05", "2023-04-06", horizon=1)._results)

    def starting_on(first):
        return data[local_dates(data) >= np.datetime64(first)]

    assert run(starting_on(needed)) == run(data)
    with pytest.raises(
        ValueError, match=rf"training window 85, refit '7D'\), and the earliest row they train on is dated {needed}"
    ):
        run(starting_on(shift(needed, 1)))
    run(starting_on(shift(needed, 2)), refit="1D")
    with pytest.raises(ValueError, match="earliest row they train on is dated"):
        run(starting_on(shift(needed, 3)), refit="1D")


# ---------------------------------------------------------------------------
# Leakage: nothing after a_m - 1 - h trains member m
# ---------------------------------------------------------------------------

# (freq, block's Monday, its Sunday, frame start): blocks over each switch at 1 h and 15 min, and a daily one.
LEAK_CASES = {
    "1h-spring": ("1h", "2023-03-20", "2023-03-26", "2022-12-01"),
    "1h-autumn": ("1h", "2023-10-23", "2023-10-29", "2023-07-01"),
    "15min-spring": ("15min", "2023-03-20", "2023-03-26", "2022-12-20"),
    "15min-autumn": ("15min", "2023-10-23", "2023-10-29", "2023-07-20"),
    "1D-autumn": ("1D", "2023-10-23", "2023-10-29", "2023-07-01"),
}
INPUTS = [c for c in make_market_data(days=1, horizon=2).columns if c != "price"]


def leak_frame(freq, start, end):
    days = (day(end) - day(start)).days + 4
    hourly = make_market_data(
        start=start, days=days, freq="15min" if freq == "15min" else "1h", horizon=2, noise=5.0, seed=7
    )
    return hourly.at_time("12:00") if freq == "1D" else hourly


def members_of(records):
    """Key -> each member's prediction as hex, and the anchor."""
    return {key: ([bits(p) for p in r["member_predictions"]], r["refit_anchor"]) for key, r in keyed(records).items()}


@pytest.mark.parametrize("case", list(LEAK_CASES))
def test_nothing_after_a_m_minus_1_minus_h_trains_member_m(case):
    """For issue days D of a block whose Monday is a: the 14-day member is anchored on D and the 85-day one on a. Moving every input
    of the rows dated after a_m - 1 - h but D's own, and every target from a_m on, leaves member m's prediction of D at h bit for
    bit. Positive controls: the target of a - 1 moves the long member of every forecast of the block, and the target of D - 1 moves
    the short member of D and not its long one. The runs start the day after the Monday, so no forecast of them is issued on it."""
    freq, monday, sunday, frame_start = LEAK_CASES[case]
    data = leak_frame(freq, frame_start, sunday)
    dates = local_dates(data)
    first = shift(monday, 1)

    def run(frame):
        model = OLSModel(PREDICTORS, training_window=[SHORT, LONG], refit="auto")
        return members_of(model.run(frame, first, sunday, horizon=2, freq=freq)._results)

    base = run(data)
    assert {anchor for _, anchor in base.values()} == {monday}
    issues = sorted({key[0] for key in base})
    a = np.datetime64(monday)
    for issue in [issues[0], issues[len(issues) // 2], issues[-1]]:
        d = np.datetime64(issue)
        not_d = dates != d
        for horizon in (1, 2):
            on_d = {key: fit for key, fit in base.items() if key[0] == issue and key[2] == horizon}
            assert on_d
            for member, anchor in ((0, d), (1, a)):
                frame = move(data, (dates > anchor - np.timedelta64(1 + horizon, "D")) & not_d, INPUTS)
                frame = move(frame, (dates >= anchor) & not_d, ["price"], seed=2)
                moved = run(frame)
                assert {key: moved[key][0][member] for key in on_d} == {
                    key: fit[0][member] for key, fit in on_d.items()
                }, (
                    issue,
                    horizon,
                    member,
                )

    before_monday = run(move(data, dates == a - np.timedelta64(1, "D"), ["price"]))
    assert all(before_monday[key][0][1] != fit[0][1] for key, fit in base.items()), (
        "a - 1 trains every long fit of the block"
    )
    issue = issues[len(issues) // 2]
    day_before = run(move(data, dates == np.datetime64(issue) - np.timedelta64(1, "D"), ["price"]))
    for key, fit in base.items():
        if key[0] == issue:
            assert day_before[key][0][0] != fit[0][0] and day_before[key][0][1] == fit[0][1], key


def test_verify_is_ok_for_auto_on_a_block_s_last_day_with_its_control():
    """`verify` on the last day of a block, a Sunday, and over a sample, for d-1 and published: ok, with the control moving."""
    data = make_market_data(start="2022-12-01", days=150, horizon=2, noise=5.0, seed=5)
    pipeline = ModelPipeline(information_set=INFO)
    pipeline.add_model(OLSModel(PREDICTORS, training_window=MEMBERS, refit="auto", name="auto"))
    pipeline.add_model(
        OLSModel(
            PREDICTORS + ["price_d-1"],
            training_window=[SHORT, LONG],
            refit="auto",
            newest_target="published",
            name="pub",
        )
    )
    pipeline.verify(data, "2023-04-09", "2023-04-09", horizon=2, days=1).assert_ok()
    pipeline.verify(data, "2023-03-27", "2023-04-16", horizon=2, days=3).assert_ok()


def test_the_probes_run_auto_to_the_end_of_its_coarsest_block(monkeypatch):
    """A probe of Monday runs the model to Sunday, the end of the long member's block, whose one fit serves the week."""
    data = market("spring")
    monday = "2023-03-27"
    runs = []
    original = BaseModel.run

    def spy(self, frame, test_start=None, test_end=None, *args, **kwargs):
        runs.append((test_start, test_end))
        return original(self, frame, test_start, test_end, *args, **kwargs)

    monkeypatch.setattr(BaseModel, "run", spy)
    control = positive_control(
        lambda: OLSModel(PREDICTORS, training_window=[SHORT, LONG], refit="auto"), data, monday, INFO, horizon=1
    )
    assert control.ok
    assert set(runs) == {(monday, "2023-04-02")}


# ---------------------------------------------------------------------------
# The information set reads every member's anchors and rows
# ---------------------------------------------------------------------------


def test_the_training_target_is_checked_on_every_member_s_anchors(monkeypatch):
    """The 14-day member's rows are cut on each issue day and the 85-day member's on its Monday, so the newest training target is
    checked as of the issue time of both: the issue days and the Monday before the first."""
    data = market("spring")
    calls = []
    original = InformationSet.violations

    def spy(
        self, expanded, target, freq, index, test_start=None, test_end=None, issue_days=None, check_target=True, **more
    ):
        calls.append((tuple(issue_days or ()), check_target))
        return original(
            self, expanded, target, freq, index, test_start, test_end, issue_days, check_target=check_target, **more
        )

    monkeypatch.setattr(InformationSet, "violations", spy)
    OLSModel(PREDICTORS, training_window=[SHORT, LONG], refit="auto").run(
        data, "2023-03-29", "2023-03-31", horizon=1, information_set=INFO
    )
    assert ("2023-03-27", "2023-03-29", "2023-03-30", "2023-03-31") in [days for days, check in calls if check]
    monkeypatch.setattr(InformationSet, "violations", original)
    late = InformationSet("11:45", "Europe/Brussels", rules={**RULES, "price": Availability.realised("12h")})
    with pytest.raises(InformationSetError, match="training target"):
        OLSModel(PREDICTORS, training_window=[SHORT, LONG], refit="auto").run(
            data, "2023-03-29", "2023-03-31", horizon=1, information_set=late
        )


def test_a_rule_with_a_gap_is_checked_on_the_short_member_s_newest_rows():
    """`aux` is never known on 5 and 6 April (a rule that stops on the 4th and starts again on the 7th). A forecast on Sunday 9
    April at horizon 1 trains its 14-day member on rows up to 7 April, cut on the 9th, and its 85-day member on rows up to 1 April,
    cut on Monday the 3rd: only the short member reads the gap, so the list is refused as the short window alone is, and the long
    window alone passes. Rows judged from the long member's anchor to its own end would have missed it."""
    data = market("spring")
    data = data.assign(aux=np.random.default_rng(2).normal(0, 1, len(data)))
    gap = [
        Availability.static().during(valid_until="2023-04-04"),
        Availability.static().during(valid_from="2023-04-07"),
    ]
    info = InformationSet("11:45", "Europe/Brussels", rules={**RULES, "aux": gap})

    def run(window, refit):
        model = OLSModel([*PREDICTORS, "aux"], training_window=window, refit=refit)
        return model.run(data, "2023-04-09", "2023-04-09", horizon=1, information_set=info)

    assert run(LONG, "7D").produced == 24
    for window, refit in ((SHORT, "1D"), ([SHORT, LONG], "auto")):
        with pytest.raises(ValueError, match="aux"):
            run(window, refit)


def test_published_reads_each_member_s_newest_target_on_its_own_anchor():
    """D's price is known at 11:45 on D, so the 14-day member of a forecast issued on D trains on targets up to D and the 85-day
    one up to its Monday: the record's `newest_target_day` is the newer, D's at horizon 1, and the prices of the days after Monday,
    known by D's issue time, move the short member alone."""
    data = market("spring")

    def run(frame):
        model = OLSModel(PREDICTORS, training_window=[SHORT, LONG], refit="auto", newest_target="published")
        return model.run(frame, "2023-03-28", "2023-04-02", horizon=2, information_set=INFO)._results

    records = run(data)
    assert {r["newest_target_day"] == r["run_date"] for r in records if r["horizon"] == 1} == {True}
    assert {r["refit_anchor"] for r in records} == {"2023-03-27"}
    base = members_of(records)
    dates = local_dates(data)
    moved = members_of(
        run(move(data, (dates > np.datetime64("2023-03-27")) & (dates < np.datetime64("2023-03-30")), ["price"]))
    )
    for key, fit in base.items():
        assert moved[key][0][1] == fit[0][1], key
        if key[0] >= "2023-03-29":
            assert moved[key][0][0] != fit[0][0], key


# ---------------------------------------------------------------------------
# The pool and a resume
# ---------------------------------------------------------------------------


def test_one_two_or_three_processes_and_a_resume_mid_block_give_a_fresh_run(monkeypatch, tmp_path):
    """1, 2 and 3 processes give the same records. A file holding about half of each block's forecasts, many of whose units are
    stored in part, resumes to a fresh run line for line, fitting each unit with pending forecasts again. A stored record whose
    anchor a fresh run wouldn't give it (the short member's, the issue day) is refused."""
    data = market("spring")
    period = ("2023-03-22", "2023-04-09")  # three blocks, the first cut by the start

    def model():
        return OLSModel(PREDICTORS, training_window=MEMBERS, refit="auto", name="m")

    expected = texts(model().run(data, *period, horizon=2)._results)
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", ProcessPoolExecutor)
    for processes in ("1", "2", "3"):
        monkeypatch.setenv("MAX_PROCESSES", processes)
        assert texts(model().run(data, *period, horizon=2)._results) == expected, processes

    fresh, cut = tmp_path / "fresh" / "m.jsonl", tmp_path / "cut" / "m.jsonl"
    model().run(data, *period, horizon=2, save_to=str(fresh))
    model().run(data, *period, horizon=2, save_to=str(cut))
    stored = lines(cut)
    kept = [line for line, keep in zip(stored, np.random.default_rng(4).random(len(stored)) < 0.5) if keep]
    blocks = {}
    for line in stored:
        record = json.loads(line)
        blocks.setdefault((record["refit_anchor"], record["hour"], record["horizon"]), []).append(line in kept)
    assert sum(any(v) and not all(v) for v in blocks.values()) > 50  # many long units are stored in part
    cut.write_text("\n".join(kept) + "\n")
    model().run(data, *period, horizon=2, save_to=str(cut))
    assert lines(cut) == lines(fresh)

    records = [json.loads(line) for line in stored]
    records[3]["refit_anchor"] = records[3]["run_date"]
    assert records[3]["run_date"] != json.loads(stored[3])["refit_anchor"]
    cut.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    with pytest.raises(ManifestMismatchError, match="another refit anchor"):
        model().run(data, *period, horizon=2, save_to=str(cut))


def test_explain_shows_the_newest_of_the_members_targets():
    """`explain()` of a published "auto" list: the newest target of each forecast is its short member's, read on the issue day."""
    data = market("spring")
    pipeline = ModelPipeline(information_set=INFO)
    pipeline.add_model(
        OLSModel(PREDICTORS, training_window=[SHORT, LONG], refit="auto", newest_target="published", name="m")
    )
    out = pipeline.explain(data, "m", test=("2023-03-28", "2023-04-02"), horizon=1, echo=False)
    table = out["newest_target"]
    assert table["issue days"].sum() == 6
    assert list(table["newest target"]) == ["D"]
