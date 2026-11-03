"""`training_window=`: an int, `"expanding"`, or a list of them averaged inside the model (step 4b).

A member of a list is fitted on its own rows (`training_span` with its window, an expanding one from the frame's first local
date), with its own scaler and binary-column mask, exactly as the model with that window alone, and the forecast is the members'
mean in target units, `math.fsum(...) / M`, made only when every member's is. These tests hold each member to that model key by
key and bit for bit, `[W]` to `W` in every byte, an expanding member to a hand fit, the rule "all or nothing" to its skip counts,
and a list to the probes, the pool and a resume. Mutants of the implementation are in
`epftoolbox2-research/library_check_step4/window_lists/`.
"""

import itertools
import json
import math
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
import pytest

from epftoolbox2 import tui
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import LassoCVModel, NaiveModel, OLSModel, Use
from epftoolbox2.models import _worker as worker
from epftoolbox2.models._calendar import resolve_training_window
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.models.base import BaseModel
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path
from epftoolbox2.results.windows import RunWindows
from tests.synthetic import PREDICTORS, make_market_data

SWITCH_DAYS = {"spring": "2023-03-26", "autumn": "2023-10-29"}
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


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)


def day_of(day, offset=0):
    return (pd.Timestamp(day) + pd.Timedelta(days=offset)).date().isoformat()


def market(switch, freq="1h", seed=3):
    """45 days before a switch and 15 after, with drivers out to 3 days; a daily frame is the hourly one at noon."""
    data = make_market_data(
        start=day_of(SWITCH_DAYS[switch], -45),
        days=60,
        freq="1h" if freq == "1D" else freq,
        horizon=3,
        noise=5.0,
        seed=seed,
    )
    return data.at_time("12:00") if freq == "1D" else data


def local_dates(data):
    return np.asarray(pd.DatetimeIndex(data.index.tz_localize(None)).normalize())


def with_paused_column(data, mask_day):
    """`aux`: 0/1 until 20 days before `mask_day`, missing from then to the day before it, other values from it on. On the mask
    day a 14-day window's rows have no value of it, so that member reads its mask again on a later day, and a 28-day or expanding
    one's rows have the 0/1 values: the masks of the members differ, and only the short one re-reads."""
    data = data.copy()
    dates = local_dates(data)
    day = np.datetime64(mask_day)
    rng = np.random.default_rng(7)
    aux = (np.arange(len(data)) % 3 == 0).astype(float)
    aux[dates >= day - np.timedelta64(20, "D")] = np.nan
    later = dates >= day
    aux[later] = rng.normal(0, 1, int(later.sum()))
    data["aux"] = aux
    return data


def keyed(records):
    return {(r["run_date"], r["hour"], r["horizon"]): r for r in records}


def lines(path):
    """A result file's records as sorted text: the order a pool finishes them in isn't fixed."""
    return sorted(line for line in Path(path).read_text().splitlines() if line.strip())


def bits(value):
    return float(value).hex()


def check_members(listed, singles, members):
    """The list model's records against its single-window models': the same keys as those every member forecast, each member's
    prediction to the bit, the mean as `fsum / M` to the bit, and everything else as the first member's record."""
    lst, each = keyed(listed), [keyed(single) for single in singles]
    assert set(lst) == set.intersection(*(set(single) for single in each))
    assert lst, "nothing to compare"
    for key, record in lst.items():
        assert [bits(p) for p in record["member_predictions"]] == [bits(single[key]["prediction"]) for single in each]
        assert bits(record["prediction"]) == bits(math.fsum(record["member_predictions"]) / len(members))
        assert record["coefficients"] == []
        rest = {k: v for k, v in record.items() if k not in ("prediction", "member_predictions", "coefficients")}
        first = {k: v for k, v in each[0][key].items() if k not in ("prediction", "coefficients")}
        assert json.dumps(rest, sort_keys=True) == json.dumps(first, sort_keys=True), key
    return lst


# ---------------------------------------------------------------------------
# What training_window= takes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "attribute", "members"),
    [
        (365, 365, (365,)),
        (np.int64(56), 56, (56,)),
        ("expanding", "expanding", ("expanding",)),
        ([56], 56, (56,)),
        (["expanding"], "expanding", ("expanding",)),
        ([84, 56], [56, 84], (56, 84)),
        (["expanding", 728, 56, 84], [56, 84, 728, "expanding"], (56, 84, 728, "expanding")),
        ((56, "expanding"), [56, "expanding"], (56, "expanding")),
        (range(7, 9), [7, 8], (7, 8)),
    ],
)
def test_the_window_is_normalised_like_horizons(value, attribute, members):
    assert resolve_training_window(value) == (attribute, members)
    model = OLSModel(PREDICTORS, training_window=value)
    assert model.training_window == attribute and type(model.training_window) is type(attribute)
    assert model._members == members


@pytest.mark.parametrize(
    "value", [0, -7, 56.0, True, None, "56", "Expanding", [], [56, 56], ["expanding", "expanding"], [56, False], [0, 7]]
)
def test_a_bad_window_raises(value):
    with pytest.raises(ValueError, match="training_window"):
        resolve_training_window(value)
    with pytest.raises(ValueError, match="training_window"):
        OLSModel(PREDICTORS, training_window=value)


def test_setting_the_window_after_construction_normalises_it():
    model = LassoCVModel(PREDICTORS)
    model.training_window = [84, "expanding", 56]
    assert (model.training_window, model._members) == ([56, 84, "expanding"], (56, 84, "expanding"))
    model.training_window = [28]
    assert (model.training_window, model._members) == (28, (28,))


def test_a_naive_model_has_no_training_window():
    """It fits nothing, so a window, and a list of them, would average fits that don't exist: refused, and 0 stays."""
    naive = NaiveModel(["load_d+{horizon}"])
    assert (naive.training_window, naive._members) == (0, (0,))
    for value in ([56], 56, "expanding", [56, "expanding"]):
        with pytest.raises(ValueError, match="no training window"):
            naive.training_window = value
    with pytest.raises(TypeError):
        NaiveModel(["load_d+{horizon}"], training_window=[56, 84])


def test_the_tui_reads_a_window_list(monkeypatch):
    field = next(f for f in tui.PARAM_SCHEMAS["OLSModel"] if f["name"] == "training_window")
    answers = iter(["56, 84,728,expanding", "expanding", "365", "56,56", "28"])

    class Answer:
        def ask(self):
            return next(answers)

    monkeypatch.setattr(tui.questionary, "text", lambda *args, **kwargs: Answer())
    assert tui._ask_field(field) == [56, 84, 728, "expanding"]
    assert tui._ask_field(field) == "expanding"
    assert tui._ask_field(field) == 365
    assert tui._ask_field(field) == 28  # "56,56" is refused and asked again
    assert tui._instantiate("OLSModel", {"predictors": PREDICTORS, "training_window": [56, "expanding"]})._members == (
        56,
        "expanding",
    )


def test_yaml_writes_the_normalised_list_and_reads_it_back(tmp_path):
    pipeline = ModelPipeline()
    pipeline.add_model(OLSModel(PREDICTORS, training_window=["expanding", 728, 56, 84], name="cwa"))
    pipeline.add_model(OLSModel(PREDICTORS, training_window=[84], name="one"))
    path = tmp_path / "p.yaml"
    pipeline.save(path)
    text = path.read_text()
    assert "training_window:\n    - 56\n    - 84\n    - 728\n    - expanding\n" in text
    assert "training_window: 84\n" in text
    loaded = ModelPipeline.load(path)
    assert [m.training_window for m in loaded.models] == [[56, 84, 728, "expanding"], 84]
    assert loaded.models[0]._members == (56, 84, 728, "expanding")


# ---------------------------------------------------------------------------
# [W] is W, in every byte
# ---------------------------------------------------------------------------


def manifest_without_provenance(directory, name):
    manifest = json.loads(manifest_path(Path(directory) / f"{name}.jsonl").read_text())
    return {key: value for key, value in manifest.items() if key not in PROVENANCE}


@pytest.mark.parametrize("window", [28, "expanding"])
@pytest.mark.parametrize("model", ["ols", "lasso"])
def test_a_list_of_one_window_is_that_window_in_everything(tmp_path, model, window):
    data = market("spring")

    def build(training_window):
        if model == "ols":
            return OLSModel(PREDICTORS, training_window=training_window, name="m")
        return LassoCVModel(PREDICTORS, training_window=training_window, cv=3, max_iter=1000, name="m")

    for form, folder in ((window, "plain"), ([window], "listed")):
        build(form).run(data, "2023-03-25", "2023-03-26", horizon=2, save_to=str(tmp_path / folder / "m.jsonl"))
        pipeline = ModelPipeline()
        pipeline.add_model(build(form))
        pipeline.save(tmp_path / f"{folder}.yaml")
    assert lines(tmp_path / "plain" / "m.jsonl") == lines(tmp_path / "listed" / "m.jsonl")
    assert manifest_without_provenance(tmp_path / "plain", "m") == manifest_without_provenance(tmp_path / "listed", "m")
    assert (tmp_path / "plain.yaml").read_text() == (tmp_path / "listed.yaml").read_text()
    record = json.loads(lines(tmp_path / "plain" / "m.jsonl")[0])
    assert "member_predictions" not in record and record["coefficients"]


def test_a_reordered_list_resumes_and_another_list_refuses(tmp_path):
    """Results cut short by a list in one order resume with it in another, to a fresh run's file; another list is refused."""
    data = market("spring")
    path = tmp_path / "m.jsonl"
    OLSModel(PREDICTORS, training_window=[28, "expanding", 14], name="m").run(
        data, "2023-03-24", "2023-03-28", horizon=2, save_to=str(path)
    )
    manifest = json.loads(manifest_path(path).read_text())
    assert manifest["training_window"] == [14, 28, "expanding"]
    assert manifest["training_cut"] == {"rows": "dates", "expanding_from": "2023-02-09"}
    fresh = lines(path)
    path.write_text("\n".join(fresh[::3]) + "\n")
    OLSModel(PREDICTORS, training_window=["expanding", 14, 28], name="m").run(
        data, "2023-03-24", "2023-03-28", horizon=2, save_to=str(path)
    )
    assert lines(path) == fresh
    for other in ([14, 28], [14, 21, "expanding"], 14, "expanding"):
        with pytest.raises(ManifestMismatchError, match="training_window"):
            OLSModel(PREDICTORS, training_window=other, name="m").run(
                data, "2023-03-24", "2023-03-28", horizon=2, save_to=path
            )


# ---------------------------------------------------------------------------
# Member i of a list is the model with window i, to the bit
# ---------------------------------------------------------------------------


class Cell(NamedTuple):
    model: str  # "ols" or "lasso"
    freq: str  # "1h", "15min" or "1D"
    switch: str  # "spring" or "autumn"
    refit: Optional[str]  # None or "7D"
    horizons: tuple  # (1,) or (1, 3)
    friday: bool  # issue_days=["friday"]
    published: bool  # newest_target="published"
    validation: bool  # a validation window and a test window
    paused: bool  # `aux`, whose mask a short member reads again on a later day (`with_paused_column`)


# Every level of every axis and every pair of levels, not every cell: 8 cells, checked by the test below. LassoCV runs on 15-minute
# data in one cell alone, a day of one horizon under "7D", which keeps its fits few.
CELLS = [
    Cell("ols", "1D", "autumn", "7D", (1,), True, False, False, True),
    Cell("lasso", "1D", "spring", None, (1, 3), False, True, True, False),
    Cell("ols", "1h", "spring", "7D", (1,), False, False, True, False),
    Cell("lasso", "1h", "autumn", None, (1, 3), True, False, False, True),
    Cell("ols", "15min", "autumn", "7D", (1, 3), True, True, False, False),
    Cell("ols", "15min", "spring", None, (1,), True, True, True, True),
    Cell("lasso", "15min", "spring", "7D", (1,), False, False, False, False),
    Cell("ols", "1h", "autumn", None, (1,), False, True, True, True),
]
LEVELS = {
    "model": {"ols", "lasso"},
    "freq": {"1h", "15min", "1D"},
    "switch": {"spring", "autumn"},
    "refit": {None, "7D"},
    "horizons": {(1,), (1, 3)},
    "friday": {False, True},
    "published": {False, True},
    "validation": {False, True},
    "paused": {False, True},
}


def test_the_cells_cover_every_pair_of_levels():
    for a, b in itertools.combinations(Cell._fields, 2):
        found = {(getattr(cell, a), getattr(cell, b)) for cell in CELLS}
        assert found == set(itertools.product(LEVELS[a], LEVELS[b])), (a, b)


def run_cell(cell, window, data):
    predictors = PREDICTORS + (["aux"] if cell.paused else [])
    options = {"refit": cell.refit, "newest_target": "published" if cell.published else "d-1", "name": "m"}
    if cell.model == "ols":
        model = OLSModel(predictors, training_window=window, **options)
    else:
        model = LassoCVModel(predictors, training_window=window, cv=3, max_iter=1000, **options)
    switch = SWITCH_DAYS[cell.switch]
    kwargs = {
        "freq": cell.freq,
        "horizons": cell.horizons,
        "issue_days": ["friday"] if cell.friday else None,
        "information_set": INFO if cell.published else None,
    }
    if cell.validation:
        windows = RunWindows.fixed((day_of(switch, -9), day_of(switch, -8)), (day_of(switch, -2), day_of(switch, 1)))
        return model.run(data, windows=windows, **kwargs)._results
    if cell.model == "lasso" and cell.freq == "15min":
        return model.run(data, switch, switch, **kwargs)._results
    return model.run(data, day_of(switch, -2), day_of(switch, 5 if cell.friday else 1), **kwargs)._results


@pytest.mark.parametrize("cell", CELLS, ids=["-".join(map(str, cell)) for cell in CELLS])
def test_each_member_is_its_single_window_model(cell):
    """`member_predictions[i]` is the prediction of the same model with `training_window` its i-th member, to the bit, and the
    forecast is their `fsum` divided by their number; the keys are those every member forecast. The paused column's mask is read
    again on a later day by the short member only, so a mask taken from another member would show."""
    members = [14, "expanding"] if cell.model == "lasso" else [14, 28, "expanding"]
    data = market(cell.switch, cell.freq)
    if cell.paused:
        switch = SWITCH_DAYS[cell.switch]
        data = with_paused_column(data, day_of(switch, -9 if cell.validation else -2))
    listed = run_cell(cell, members, data)
    singles = [run_cell(cell, member, data) for member in members]
    found = check_members(listed, singles, members)
    if cell.refit is not None:
        assert all("refit_anchor" in record for record in found.values())
    if cell.validation:
        # With the paused column the short member has no row with a value on the validation's Friday: none of its keys is made.
        windows = {"test"} if cell.paused and cell.friday else {"validation", "test"}
        assert {record["window"] for record in found.values()} == windows


def weekday_lists(ctx):
    """Another list on weekdays, so that a list's mask and its first test day matter."""
    return PREDICTORS + (["price_d-1"] if ctx.issue_day.weekday() < 5 else [])


def test_a_predictor_function_s_members_are_its_single_window_models(tmp_path):
    """The masks of a function's lists are per member too ((list id, period, horizon, member)), and its manifest keeps each
    member's digest under the key with `@` and the member, which a resume compares: a changed digest of one member refuses."""
    data = with_paused_column(market("spring"), "2023-03-24")
    names = [*PREDICTORS, "aux"]

    def function(ctx):
        return names + (["price_d-1"] if ctx.issue_day.weekday() < 5 else [])

    members = [14, 28, "expanding"]
    path = tmp_path / "m.jsonl"
    listed = OLSModel(function, training_window=members, name="m").run(
        data, "2023-03-24", "2023-03-29", horizon=2, save_to=str(path)
    )
    singles = [
        OLSModel(function, training_window=member, name="m").run(data, "2023-03-24", "2023-03-29", horizon=2)._results
        for member in members
    ]
    check_members([json.loads(line) for line in lines(path)], singles, members)
    assert listed.produced > 0
    manifest = json.loads(manifest_path(path).read_text())
    keys = {key for entry in manifest["predictor_lists"] for key in entry["binary_masks"]}
    assert keys and all(key.rsplit("@", 1)[-1] in {"14", "28", "expanding"} for key in keys)
    # The short member re-reads `aux`: its digest differs from the others' on the same key.
    entry = manifest["predictor_lists"][0]["binary_masks"]
    assert entry["0-1@14"] != entry["0-1@28"] == entry["0-1@expanding"]

    OLSModel(function, training_window=members, name="m").run(
        data, "2023-03-24", "2023-03-29", horizon=2, save_to=str(path)
    )
    manifest["predictor_lists"][0]["binary_masks"]["0-1@28"] = "00000000"
    manifest_path(path).write_text(json.dumps(manifest))
    with pytest.raises(ManifestMismatchError, match="other binary columns"):
        OLSModel(function, training_window=members, name="m").run(
            data, "2023-03-24", "2023-03-29", horizon=2, save_to=str(path)
        )


def test_a_fixed_list_with_windows_keeps_a_digest_per_member(tmp_path):
    data = with_paused_column(market("spring"), "2023-03-17")
    path = tmp_path / "m.jsonl"
    windows = RunWindows.fixed(("2023-03-17", "2023-03-18"), ("2023-03-24", "2023-03-27"))
    model = OLSModel([*PREDICTORS, "aux"], training_window=[14, "expanding"], name="m")
    model.run(data, windows=windows, horizon=1, save_to=str(path))
    manifest = json.loads(manifest_path(path).read_text())
    assert {"validation/0-1@14", "validation/0-1@expanding", "0-1@14", "0-1@expanding"} <= set(manifest["binary_masks"])
    manifest["binary_masks"]["0-1@expanding"] = "00000000"
    manifest_path(path).write_text(json.dumps(manifest))
    with pytest.raises(ManifestMismatchError, match="other binary columns"):
        OLSModel([*PREDICTORS, "aux"], training_window=[14, "expanding"], name="m").run(
            data, windows=windows, horizon=1, save_to=str(path)
        )


# ---------------------------------------------------------------------------
# "expanding": every row from the frame's first local date
# ---------------------------------------------------------------------------


def hand_expanding(data, period, issue, horizon, names):
    """OLS without intercept on the standardised target, from every row of `period` dated from the frame's first date to
    issue - 1 - h with every value; 0/1 columns stay as they are, others are standardised on the rows used."""
    wall = data.index.tz_localize(None)
    dates = wall.normalize()
    here = np.asarray(wall.hour == period)
    day = pd.Timestamp(issue)
    rows = np.flatnonzero(here & np.asarray(dates <= day - pd.Timedelta(days=1 + horizon)))
    prices = pd.Series(data["price"].to_numpy(), index=wall)
    target = prices.reindex(wall[rows] + pd.Timedelta(days=horizon)).to_numpy()
    x = data[names].to_numpy(dtype=float)
    binary = np.array([set(np.unique(column[np.isfinite(column)])) <= {0.0, 1.0} for column in x[rows].T])
    usable = np.isfinite(target) & np.isfinite(x[rows]).all(axis=1)
    y, train = target[usable], x[rows][usable]
    means = np.where(binary, 0.0, train.mean(axis=0))
    deviations = np.where(binary, 1.0, train.std(axis=0, ddof=1))
    coefficients, *_ = np.linalg.lstsq((train - means) / deviations, (y - y.mean()) / y.std(ddof=1), rcond=None)
    test_row = np.flatnonzero(here & np.asarray(dates == day))[-1]
    return float(((x[test_row] - means) / deviations) @ coefficients * y.std(ddof=1) + y.mean())


def test_expanding_is_a_hand_fit_on_every_row_from_the_frame_s_first_date():
    data = market("spring")
    names = ["load_d+1", "wind_d+1", "is_weekend_d+1", "is_weekday_d+1"]
    found = keyed(
        OLSModel(PREDICTORS, training_window="expanding").run(data, "2023-03-24", "2023-03-27", horizon=2)._results
    )
    assert found
    for (issue, period, horizon), record in found.items():
        if horizon == 1 and period in (0, 2, 13, 23):
            assert record["prediction"] == pytest.approx(hand_expanding(data, period, issue, 1, names), abs=1e-9)


def test_dropping_the_frame_s_first_day_moves_every_expanding_forecast_and_no_other(tmp_path):
    data = market("autumn")
    trimmed = data[local_dates(data) > local_dates(data)[0]]

    def run(frame, **kwargs):
        model = OLSModel(PREDICTORS, training_window=[14, "expanding"], name="m")
        return keyed(model.run(frame, "2023-10-27", "2023-10-30", horizon=2, **kwargs)._results)

    full, cut = run(data), run(trimmed)
    assert full.keys() == cut.keys()
    for key in full:
        assert full[key]["member_predictions"][0] == cut[key]["member_predictions"][0]
        assert full[key]["member_predictions"][1] != cut[key]["member_predictions"][1]
    path = str(tmp_path / "m.jsonl")
    OLSModel(PREDICTORS, training_window="expanding", name="m").run(
        data, "2023-10-27", "2023-10-28", horizon=2, save_to=path
    )
    with pytest.raises(ManifestMismatchError, match="expanding_from"):
        OLSModel(PREDICTORS, training_window="expanding", name="m").run(
            trimmed, "2023-10-27", "2023-10-30", horizon=2, save_to=path
        )


def test_expanding_reads_the_whole_frame_where_an_int_window_reads_its_days():
    """15-minute data with hourly history: a 56-day window never reads the hourly part and runs; an expanding one trains on it,
    so the run is refused, also inside a list."""
    quarter = make_market_data(days=150, freq="15min")
    data = quarter[(quarter.index >= pd.Timestamp("2023-04-01", tz=quarter.index.tz)) | (quarter.index.minute == 0)]
    OLSModel(PREDICTORS, training_window=14)._check_inputs(data, "price", 1, "15min", None, "2023-05-20", "2023-05-20")
    for window in ("expanding", [14, "expanding"]):
        with pytest.raises(ValueError, match="coarser"):
            OLSModel(PREDICTORS, training_window=window)._check_inputs(
                data, "price", 1, "15min", None, "2023-05-20", "2023-05-20"
            )


@pytest.mark.parametrize(
    ("window", "needed"), [([14, 28], "2023-02-19"), ([14, 28, "expanding"], "2023-02-19"), ("expanding", None)]
)
def test_a_refit_list_needs_the_frame_its_longest_int_member_reads(window, needed):
    """With refit each member's units must find every row they train on: the longest int member decides, from the first anchor
    (Monday 20 March, so 20 March - 28 - 1 at horizon 1), and an expanding member asks for nothing. A frame from that day gives
    the int members' forecasts of a long frame; one from the day after is refused."""
    data = market("spring")

    def run(frame):
        model = OLSModel(PREDICTORS, training_window=window, refit="7D")
        records = keyed(model.run(frame, "2023-03-21", "2023-03-22", horizon=1)._results)
        return {
            key: [bits(p) for p in r.get("member_predictions", [r["prediction"]])[:2]] for key, r in records.items()
        }

    if needed is None:
        assert run(data[local_dates(data) >= np.datetime64("2023-03-01")])
        return
    long = run(data)
    assert run(data[local_dates(data) >= np.datetime64(needed)]) == long
    with pytest.raises(ValueError, match=f"earliest row they train on is dated {needed}"):
        run(data[local_dates(data) >= np.datetime64(needed) + np.timedelta64(1, "D")])


# ---------------------------------------------------------------------------
# All or nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("refit", [None, "7D"])
def test_a_key_is_forecast_only_when_every_member_is(monkeypatch, refit):
    """`aux` has no value in the 20 days before the test at periods 0-11: the 14-day member has no finite row there, so those
    keys are skipped with its reason and window, once each, and the expanding member's forecasts of them are dropped."""
    data = market("spring")
    dates, hours = local_dates(data), data.index.hour
    aux = np.random.default_rng(1).normal(0, 1, len(data))
    aux[(dates >= np.datetime64("2023-03-07")) & (dates < np.datetime64("2023-03-27")) & (hours < 12)] = np.nan
    data = data.assign(aux=aux)
    examples = {}
    original = BaseModel._report_counts

    def spy(self, planned, produced, skipped, found):
        examples.update(found)
        return original(self, planned, produced, skipped, found)

    monkeypatch.setattr(BaseModel, "_report_counts", spy)
    names = [*PREDICTORS, "aux"]
    ref = OLSModel(names, training_window=[14, "expanding"], refit=refit).run(
        data, "2023-03-27", "2023-03-28", horizon=1
    )
    assert ref.skipped == {"no_finite_training_rows": 24} and ref.produced == 24
    assert examples["no_finite_training_rows"].endswith(": window 14")
    assert {record["hour"] for record in ref._results} == set(range(12, 24))
    short = OLSModel(names, training_window=14, refit=refit).run(data, "2023-03-27", "2023-03-28", horizon=1)
    assert short.skipped == {"no_finite_training_rows": 24}
    whole = OLSModel(names, training_window="expanding", refit=refit).run(data, "2023-03-27", "2023-03-28", horizon=1)
    assert whole.produced == 48


# ---------------------------------------------------------------------------
# What a list doesn't support yet
# ---------------------------------------------------------------------------


def matching_on_weekends(ctx):
    return Use(PREDICTORS, train_on="matching") if ctx.target_day.weekday() >= 5 else PREDICTORS


@pytest.mark.parametrize("window", [[14, 28], "expanding"])
def test_train_on_matching_with_a_window_list_raises_naming_the_model(window):
    data = market("spring")
    with pytest.raises(ValueError, match="cwa: train_on='matching' isn't supported yet"):
        OLSModel(matching_on_weekends, training_window=window, name="cwa").run(
            data, "2023-03-24", "2023-03-27", horizon=1
        )
    pipeline = ModelPipeline()
    pipeline.add_model(OLSModel(matching_on_weekends, training_window=window, name="cwa"))
    with pytest.raises(ValueError, match="cwa: train_on='matching'"):
        pipeline.run(data, "2023-03-24", "2023-03-27", horizon=1)


# ---------------------------------------------------------------------------
# The probes, the pool and a resume
# ---------------------------------------------------------------------------


def test_verify_is_ok_for_a_window_list_with_its_positive_control():
    data = make_market_data(start="2021-01-01", days=800, horizon=2, noise=5.0, seed=5)
    pipeline = ModelPipeline(information_set=INFO)
    pipeline.add_model(OLSModel(PREDICTORS, training_window=[56, 728, "expanding"], name="cwa"))
    pipeline.add_model(
        OLSModel(
            PREDICTORS + ["price_d-1"], training_window=[56, 728, "expanding"], newest_target="published", name="pub"
        )
    )
    report = pipeline.verify(data, "2023-02-20", "2023-03-05", horizon=2, days=2)
    report.assert_ok()
    assert report.models == ["cwa", "pub"]


def stop_once_stored(path):
    """Raise in a worker once the parent has written forecasts to `path`: a pool reports the jobs that are done when it is
    asked in no fixed order, so a job that failed at once could come first and leave nothing stored."""
    deadline = time.monotonic() + 120
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    raise RuntimeError("stopped")


def stop_on(day, real, path):
    """A worker body that raises on one issue day: the run stops there with what it stored so far."""

    def stopping(arrays, cfg, hour, hz, today, *args):
        if today - cfg["offset"] == day:
            stop_once_stored(path)
        return real(arrays, cfg, hour, hz, today, *args)

    return stopping


@pytest.mark.parametrize("refit", [None, "7D"])
def test_a_list_is_the_same_in_any_pool_and_after_a_stopped_run(monkeypatch, tmp_path, refit):
    """1, 2 and 3 processes give the same records; a run stopped in the real pool at 2 processes and resumed at 3 is a fresh one,
    line for line."""
    data = with_paused_column(market("spring"), "2023-03-13")
    names = [*PREDICTORS, "aux"]
    period = ("2023-03-13", "2023-03-29")  # three "7D" blocks: the stopped one is the last, so one before it is stored

    def model():
        return OLSModel(names, training_window=[14, 28, "expanding"], refit=refit, name="m")

    def texts(records):
        return {key: json.dumps(record, sort_keys=True) for key, record in keyed(records).items()}

    expected = texts(model().run(data, *period, horizon=2)._results)
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", ProcessPoolExecutor)
    for processes in ("1", "2", "3"):
        monkeypatch.setenv("MAX_PROCESSES", processes)
        assert texts(model().run(data, *period, horizon=2)._results) == expected, processes

    fresh, stopped = tmp_path / "fresh" / "m.jsonl", tmp_path / "stopped" / "m.jsonl"
    model().run(data, *period, horizon=2, save_to=str(fresh))
    monkeypatch.setenv("MAX_PROCESSES", "2")
    if refit is None:
        monkeypatch.setattr(worker, "_fit_member", stop_on(14, worker._fit_member, stopped))
    else:
        real = worker._fit_unit

        def stopping(unit, tasks):
            if unit.anchor.isoformat() == "2023-03-27" and unit.member == 1:
                stop_once_stored(stopped)
            return real(unit, tasks)

        monkeypatch.setattr(worker, "_fit_unit", stopping)
    with pytest.raises(RuntimeError, match="forecasting failed"):
        model().run(data, *period, horizon=2, save_to=str(stopped))
    monkeypatch.undo()
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", ProcessPoolExecutor)
    monkeypatch.setenv("MAX_PROCESSES", "3")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    partial = len(lines(stopped))
    assert 0 < partial < len(lines(fresh))
    model().run(data, *period, horizon=2, save_to=str(stopped))
    assert lines(stopped) == lines(fresh)
