"""`FallbackEnsemble`, `EvaluationReport.rungs()` and `ModelPipeline.frozen()`.

The results are written by hand, as `tests/pipelines/test_ensembles.py` writes them, and `ModelPipeline.report()` computes the
ensembles from them with the code a run uses. The fallback is checked against a pandas oracle that knows nothing of the library:
which hours a local date has comes from `pd.date_range` in the zone, a day is served by the first member with a finite forecast of
each of them, and everything is compared bit for bit.
"""

import datetime
import json
import shutil
import warnings
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.ensembles import AverageEnsemble, FallbackEnsemble
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.models import HistGBModel, LassoCVModel, NaiveModel, OLSModel, ResidualModel
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.pipelines._frozen import _model_params as model_params
from epftoolbox2.pipelines.model_pipeline import COMPONENT_REGISTRY, write_zone_index
from epftoolbox2.results import ManifestMismatchError, RunWindows
from epftoolbox2.results.manifest import hash_predictor_layout, manifest_path
from tests.synthetic import PREDICTORS, make_market_data

ZONE = "Europe/Warsaw"
KEY = ("run_date", "target_date", "hour", "horizon")
# A fixed validation and test over the spring switch of 2024 (Sunday 31 March): issued on it, every horizon lacks 02:00, and
# a horizon-1 forecast issued the day before is for a 23-hour day.
SPRING = RunWindows.fixed(("2024-03-26", "2024-03-29"), ("2024-03-30", "2024-04-02"))
# And over the autumn one (Sunday 27 October), whose repeated hour is one period.
AUTUMN = RunWindows.fixed(("2024-10-22", "2024-10-25"), ("2024-10-26", "2024-10-29"))
PLAIN = RunWindows.fixed(("2024-05-01", "2024-05-04"), ("2024-05-05", "2024-05-09"))
SKIP_STATUSES = ("missing_test_predictors", "check_failed")


# The model every hand-written member stands for (`pipeline()` adds one per name), whose configuration its manifest records.
MEMBER = OLSModel(["x"], name="a")


class InProcessPool:
    """Stands in for the worker pool in the few real runs here, whose start is most of the time they would take."""

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
def offline(no_network, monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)
    yield


# ---------------------------------------------------------------------------------------------------------------------
# What a run leaves behind, by hand
# ---------------------------------------------------------------------------------------------------------------------


def plus_days(day, count):
    return (datetime.date.fromisoformat(day) + datetime.timedelta(days=count)).isoformat()


def local_periods(day, freq):
    """The period ids a local date has, by the oracle: from the instants of the day in the zone, not the library's clock."""
    start = pd.Timestamp(day).tz_localize(ZONE)
    end = pd.Timestamp(plus_days(day, 1)).tz_localize(ZONE)
    stamps = pd.date_range(start, end, freq="h" if freq == "1h" else "15min", inclusive="left")
    per_hour = 1 if freq == "1h" else 4
    return set((stamps.hour * per_hour + stamps.minute // 15 * (per_hour > 1)).tolist())


def planned_keys(windows, horizons, freq):
    """Every planned key of every issue day on file, with the window it is in: the periods its issue day and target day have."""
    keys = []
    for run_date in windows.issue_days():
        for horizon in horizons:
            target_date = plus_days(run_date, horizon)
            for period in sorted(local_periods(run_date, freq) & local_periods(target_date, freq)):
                keys.append((run_date, target_date, period, horizon))
    return keys


def actual_of(target_date, period):
    return 100.0 + datetime.date.fromisoformat(target_date).toordinal() % 50 + period


def record(key, prediction, windows, freq="1h"):
    run_date, target_date, period, horizon = key
    window = windows.window_of(run_date)
    first = dict(windows.periods)[window][0]
    return {
        "run_date": run_date,
        "target_date": target_date,
        "run_weekday": datetime.date.fromisoformat(run_date).weekday(),
        "hour": period,
        "horizon": horizon,
        "day_in_test": (datetime.date.fromisoformat(run_date) - datetime.date.fromisoformat(first)).days,
        "prediction": prediction,
        "actual": actual_of(target_date, period),
        "coefficients": [1.0],
        "freq": freq,
        "window": window,
    }


def skip_record(key, status, windows, freq="1h"):
    """A key without a forecast, as the store writes it: prediction null and the reason as `status`."""
    skipped = {**record(key, None, windows, freq), "coefficients": [], "status": status, "revision": 1}
    if status == "check_failed":
        skipped["detail"] = "not_constant"
    return skipped


def manifest(windows, horizons, freq):
    """What a run of `MEMBER` records, in the fields `frozen()` compares with the pipeline's model."""
    periods = 24 if freq == "1h" else 96
    return {
        "model_class": "OLSModel",
        "predictors": ["x"],
        "predictor_layout_hash": hash_predictor_layout({(p, h): ["x"] for p in range(periods) for h in horizons}),
        "training_window": MEMBER.training_window,
        # As a run writes it (the real runs below check the rule against the manifests runs write).
        "model_params": model_params(MEMBER),
        "target": "price",
        "horizon": max(horizons),
        **({} if tuple(horizons) == tuple(range(1, max(horizons) + 1)) else {"horizons": list(horizons)}),
        "freq": freq,
        "data_timezone": ZONE,
        "test_start": windows.first_day,
        "test_end": windows.last_day,
        "windows": windows.to_dict(),
        "column_hashes": {"price": "one target"},
        "frame_hash": "one frame",
        "data_columns_hash": "same data",
    }


def write_member(directory, name, records, windows, horizons=(1,), freq="1h"):
    Path(directory).mkdir(parents=True, exist_ok=True)
    path = Path(directory) / f"{name}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    manifest_path(path).write_text(json.dumps(manifest(windows, horizons, freq)))


def random_members(directory, windows, horizons, freq, seed, names=("a", "b", "c")):
    """Members with holes: whole days missing, single keys missing or skipped (`missing_test_predictors`, `check_failed`), and
    NaN or infinite predictions. Returns ({name: {key: prediction}} of the forecasts they hold, NaN and all, the planned keys).

    In each window, whatever the random holes, the first day is whole for the first of `names`, the second lacks it and is whole
    for the second, the third lacks the first two and is whole for the third, and the fourth lacks all of them, so every rung
    serves something and some day is missing."""
    rng = np.random.default_rng(seed)
    keys = planned_keys(windows, horizons, freq)
    days = sorted({(k[0], k[1], k[3]) for k in keys})
    planted = {}
    for window, _ in windows.periods:
        in_window = [day for day in days if windows.window_of(day[0]) == window]
        planted.update({in_window[0]: 0, in_window[1]: 1, in_window[2]: 2, in_window[3]: 3})
    # Per key, so that about 60 % of a member's days are whole at either frequency, before whole days go missing.
    hole = 0.35 / (24 if freq == "1h" else 96)
    held = {}
    for j, name in enumerate(names):
        gone_days = {day for day in days if planted.get(day, 0) > j or (day not in planted and rng.random() < 0.15)}
        records, predictions = [], {}
        for key in keys:
            whole = planted.get((key[0], key[1], key[3])) == j
            if (key[0], key[1], key[3]) in gone_days or (not whole and rng.random() < hole):
                if rng.random() < 0.5:
                    records.append(skip_record(key, SKIP_STATUSES[int(rng.integers(2))], windows, freq))
                continue
            draw = 1.0 if whole else rng.random()
            value = (
                np.nan
                if draw < hole / 2
                else np.inf
                if draw < hole
                else actual_of(key[1], key[2]) + (j + 1) * rng.normal()
            )
            predictions[key] = value
            records.append(record(key, value, windows, freq))
        rng.shuffle(records)  # the order a pool finishes in
        write_member(directory, name, records, windows, horizons, freq)
        held[name] = predictions
    return held, keys


def pipeline(*ensembles, models=("a", "b", "c")):
    built = ModelPipeline()
    for name in models:
        built.add_model(OLSModel(["x"], name=name))
    for ensemble in ensembles:
        built.add_ensemble(ensemble)
    return built


def report_of(directory, *ensembles, models=("a", "b", "c")):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the holes are meant: the shortfall warnings say so
        return pipeline(*ensembles, models=models).report(directory)


def stored(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def key_of(r):
    return tuple(r[k] for k in KEY)


def served_by(records):
    """{key: (prediction, rung, member)} of a fallback's records, the prediction as its bits."""
    return {key_of(r): (np.float64(r["prediction"]).tobytes(), r["rung"], r["member"]) for r in records}


# ---------------------------------------------------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------------------------------------------------


def finite(predictions):
    return {key: value for key, value in predictions.items() if np.isfinite(value)}


def oracle(held, members, keys, per):
    """{key: (prediction, rung, member)} by the rule written out: per (issue day, target day, horizon) the first member that has
    a finite forecast of each of its planned keys, or per key the first that has it. `held` holds each member's forecasts."""
    ok = {name: finite(held[name]) for name in members}
    frame = pd.DataFrame(keys, columns=list(KEY))
    served = {}
    for (run_date, target_date, horizon), day in frame.groupby(["run_date", "target_date", "horizon"], sort=True):
        day_keys = [tuple(row) for row in day[list(KEY)].itertuples(index=False)]
        if per == "day":
            whole = [j for j, name in enumerate(members) if all(key in ok[name] for key in day_keys)]
            if whole:
                j = whole[0]
                served.update({key: (np.float64(ok[members[j]][key]).tobytes(), j, members[j]) for key in day_keys})
        else:
            for key in day_keys:
                having = [j for j, name in enumerate(members) if key in ok[name]]
                if having:
                    j = having[0]
                    served[key] = (np.float64(ok[members[j]][key]).tobytes(), j, members[j])
    return served


def average(held, members, keys):
    """An `AverageEnsemble` by hand: `np.mean` in member order on the keys all of them have finite."""
    ok = [finite(held[name]) for name in members]
    return {
        key: float(np.mean(np.array([[m[key] for m in ok]]), axis=1)[0]) for key in keys if all(key in m for m in ok)
    }


def expected_counts(served, keys, windows, window):
    """(planned, produced, target_in_test, member_missing) of one window by the oracle."""
    mine = [key for key in keys if windows.window_of(key[0]) == window]
    first_test = windows.test[0]
    in_test = [key for key in mine if key in served and window == "validation" and key[1] >= first_test]
    produced = [key for key in mine if key in served and not (window == "validation" and key[1] >= first_test)]
    return len(mine), len(produced), len(in_test), len(mine) - len(produced) - len(in_test)


def counts(report, name):
    row = report.forecast_counts().set_index("model").loc[name]
    get = lambda column: int(row[column]) if column in row and not pd.isna(row[column]) else 0  # noqa: E731
    return get("planned"), get("produced"), get("target_in_test"), get("member_missing")


# ---------------------------------------------------------------------------------------------------------------------
# Against the oracle
# ---------------------------------------------------------------------------------------------------------------------

# Every level of each axis (switch, frequency, horizons, ladder order) and every pair of levels is in some cell.
CELLS = [
    (SPRING, "1h", (1, 2), ("a", "b", "c"), 1),
    (SPRING, "15min", (1,), ("c", "a", "b"), 2),
    (AUTUMN, "1h", (1,), ("c", "a", "b"), 3),
    (AUTUMN, "15min", (1, 2), ("a", "b", "c"), 4),
    (PLAIN, "1h", (1, 3), ("b", "c", "a"), 5),
]


@pytest.mark.parametrize("per", ["day", "key"])
@pytest.mark.parametrize("windows, freq, horizons, ladder, seed", CELLS)
def test_the_fallback_is_the_oracle_bit_for_bit_with_its_counts(tmp_path, windows, freq, horizons, ladder, seed, per):
    held, keys = random_members(tmp_path, windows, horizons, freq, seed, names=ladder)
    report = report_of(tmp_path, FallbackEnsemble(list(ladder), per=per, name="live"))
    want = oracle(held, ladder, keys, per)
    records = stored(tmp_path / "live.jsonl")
    assert served_by(records) == want
    assert len(records) == len(want)
    for r in records:
        assert r["status"] == "ok" and r["actual"] == actual_of(r["target_date"], r["hour"])
        assert r["window"] == windows.window_of(r["run_date"])
    # Some of each: days from each rung, keys from below rung 0, and missing days, or the cell checks less than it says.
    assert {rung for _, rung, _ in want.values()} == {0, 1, 2}
    assert len(want) < len(keys)
    for window, part in (("test", report), ("validation", report.validation)):
        assert counts(part, "live") == expected_counts(want, keys, windows, window)
        in_window = {key: value for key, value in want.items() if windows.window_of(key[0]) == window}
        if window == "validation":
            in_window = {key: value for key, value in in_window.items() if key[1] < windows.test[0]}
        assert served_by(part.refs["live"].iter_records()) == in_window


def test_per_day_never_stitches_a_day_and_per_key_does(tmp_path):
    held, keys = random_members(tmp_path, SPRING, (1, 2), "1h", seed=7)
    report_of(
        tmp_path,
        FallbackEnsemble(["a", "b", "c"], name="day"),
        FallbackEnsemble(["a", "b", "c"], per="key", name="key"),
    )

    def members_per_day(name):
        frame = pd.DataFrame(stored(tmp_path / f"{name}.jsonl"))
        return frame.groupby(["run_date", "target_date", "horizon"])["member"].nunique()

    assert (members_per_day("day") == 1).all()
    assert (members_per_day("key") > 1).any()
    # A served day is whole: as many keys as the clock plans for it.
    day = pd.DataFrame(stored(tmp_path / "day.jsonl")).groupby(["run_date", "target_date", "horizon"]).size()
    planned = pd.DataFrame(keys, columns=list(KEY)).groupby(["run_date", "target_date", "horizon"]).size()
    assert (day == planned.loc[day.index]).all()
    assert set(planned.loc[day.index]) == {23, 24}


def test_a_key_no_member_has_leaves_its_day_to_no_one(tmp_path):
    keys = planned_keys(PLAIN, (1,), "1h")
    lacking = ("2024-05-06", "2024-05-07", 9, 1)
    for j, name in enumerate(("a", "b")):
        write_member(tmp_path, name, [record(k, 100.0 + j, PLAIN) for k in keys if k != lacking], PLAIN)
    report = report_of(tmp_path, FallbackEnsemble(["a", "b"], name="live"), models=("a", "b"))
    records = stored(tmp_path / "live.jsonl")
    assert not [r for r in records if (r["run_date"], r["horizon"]) == ("2024-05-06", 1)]
    assert counts(report, "live") == (5 * 24, 4 * 24, 0, 24)
    assert report.rungs("live").attrs["missing"] == 24
    key = report_of(tmp_path, FallbackEnsemble(["a", "b"], per="key", name="live"), models=("a", "b"))
    assert counts(key, "live") == (5 * 24, 5 * 24 - 1, 0, 1)


def test_a_one_member_fallback_is_the_member(tmp_path):
    held, keys = random_members(tmp_path, AUTUMN, (1,), "1h", seed=11, names=("a",))
    report = report_of(tmp_path, FallbackEnsemble(["a"], per="key", name="live"), models=("a",))
    member = {key_of(r): r for r in stored(tmp_path / "a.jsonl") if "status" not in r}
    member = {key: r for key, r in member.items() if np.isfinite(r["prediction"])}
    records = stored(tmp_path / "live.jsonl")
    assert {key_of(r): r["prediction"] for r in records} == {key: r["prediction"] for key, r in member.items()}
    assert {(r["rung"], r["member"]) for r in records} == {(0, "a")}
    # A model's keys in a model's order, empty coefficients, then the rung and member before D4's keys.
    assert list(records[0]) == [*record(keys[0], 0.0, AUTUMN), "rung", "member", "status", "revision"]
    assert records[0]["coefficients"] == []
    assert counts(report, "a")[1] == counts(report, "live")[1]


def test_a_nested_fallback_of_averages_is_the_oracle(tmp_path):
    held, keys = random_members(tmp_path, SPRING, (1, 2), "1h", seed=13)
    report = report_of(
        tmp_path,
        AverageEnsemble(["a", "b"], name="ab"),
        AverageEnsemble(["b", "c"], name="bc"),
        FallbackEnsemble(["ab", "bc", "c"], name="live"),
        FallbackEnsemble(["live", "a"], per="key", name="outer"),
    )
    held = {**held, "ab": average(held, ["a", "b"], keys), "bc": average(held, ["b", "c"], keys)}
    assert {key_of(r): r["prediction"] for r in stored(tmp_path / "ab.jsonl")} == held["ab"]
    want = oracle(held, ["ab", "bc", "c"], keys, "day")
    assert served_by(stored(tmp_path / "live.jsonl")) == want
    assert {rung for _, rung, _ in want.values()} == {0, 1, 2}
    held["live"] = {key: np.frombuffer(bits)[0] for key, (bits, _, _) in want.items()}
    assert served_by(stored(tmp_path / "outer.jsonl")) == oracle(held, ["live", "a"], keys, "key")
    rungs = report.rungs("live")
    assert list(rungs["member"]) == ["ab", "bc", "c"]
    assert rungs["forecasts"].sum() == counts(report, "live")[1]


def test_a_fallback_over_a_top_ensemble_has_test_forecasts_only(tmp_path):
    """`best` has no validation forecasts, so neither has the fallback: c would serve every validation day otherwise."""
    keys = planned_keys(PLAIN, (1,), "1h")
    held = {}
    for name, error in (("a", 1.0), ("b", 2.0), ("c", 3.0)):
        lacking = ("2024-05-06", "2024-05-07", 3, 1) if name == "a" else None
        held[name] = {k: actual_of(k[1], k[2]) + error for k in keys if k != lacking}
        write_member(tmp_path, name, [record(k, value, PLAIN) for k, value in held[name].items()], PLAIN)
    report = report_of(
        tmp_path, AverageEnsemble(["a", "b"], top=1, name="best"), FallbackEnsemble(["best", "c"], name="live")
    )
    assert report.choices().iloc[0]["chosen"] == ["a"]
    records = stored(tmp_path / "live.jsonl")
    assert {r["window"] for r in records} == {"test"}
    assert "live" not in report.validation.refs
    test_keys = [k for k in keys if PLAIN.window_of(k[0]) == "test"]
    held["best"] = {key: value for key, value in held["a"].items() if key in test_keys}
    want = oracle(held, ["best", "c"], test_keys, "day")
    assert served_by(records) == want
    assert {member for _, _, member in want.values()} == {"best", "c"}
    assert counts(report, "live") == (len(test_keys), len(test_keys), 0, 0)


# ---------------------------------------------------------------------------------------------------------------------
# rungs(): by hand
# ---------------------------------------------------------------------------------------------------------------------


def hand_members(directory):
    """Five test days of 24 hours (2024-05-05 to 09). `a` errs +1 and lacks 05:00 of the 6th and all of the 7th; `b` errs -2 and
    lacks 00:00 of the 7th and 07:00 of the 9th; `c` errs +3 and lacks 01:00 of the 7th, so no one has all of the 7th."""
    keys = planned_keys(PLAIN, (1,), "1h")
    lacks = {
        "a": lambda run, hour: (run, hour) == ("2024-05-06", 5) or run == "2024-05-07",
        "b": lambda run, hour: (run, hour) in (("2024-05-07", 0), ("2024-05-09", 7)),
        "c": lambda run, hour: (run, hour) == ("2024-05-07", 1),
    }
    errors = {"a": 1.0, "b": -2.0, "c": 3.0}
    for name in ("a", "b", "c"):
        records = [record(k, actual_of(k[1], k[2]) + errors[name], PLAIN) for k in keys if not lacks[name](k[0], k[2])]
        write_member(directory, name, records, PLAIN)


def test_rungs_per_day_against_hand_counts(tmp_path):
    hand_members(tmp_path)
    rungs = report_of(tmp_path, FallbackEnsemble(["a", "b", "c"], name="live")).rungs("live")
    assert list(rungs.columns) == [
        "rung",
        "member",
        "days",
        "forecasts",
        "mae",
        "common_forecasts",
        "common_mae",
        "rung0_mae",
        "member_forecasts",
        "member_mae",
    ]
    # a serves the 5th, 8th and 9th; b the 6th, where a has 23 hours; nobody the 7th.
    rows = rungs.set_index("member").to_dict("index")
    assert rows["a"] == {
        "rung": 0,
        "days": 3,
        "forecasts": 72,
        "mae": 1.0,
        "common_forecasts": 72,
        "common_mae": 1.0,
        "rung0_mae": 1.0,
        "member_forecasts": 5 * 24 - 25,
        "member_mae": 1.0,
    }
    assert rows["b"] == {
        "rung": 1,
        "days": 1,
        "forecasts": 24,
        "mae": 2.0,
        "common_forecasts": 23,
        "common_mae": 2.0,
        "rung0_mae": 1.0,
        "member_forecasts": 5 * 24 - 2,
        "member_mae": 2.0,
    }
    assert rows["c"]["days"] == rows["c"]["forecasts"] == rows["c"]["common_forecasts"] == 0
    assert np.isnan([rows["c"][k] for k in ("mae", "common_mae", "rung0_mae")]).all()
    assert (rows["c"]["member_forecasts"], rows["c"]["member_mae"]) == (5 * 24 - 1, 3.0)
    assert rungs.attrs["missing"] == 24


def test_rungs_per_key_against_hand_counts(tmp_path):
    hand_members(tmp_path)
    report = report_of(tmp_path, FallbackEnsemble(["a", "b", "c"], per="key", name="live"))
    rows = report.rungs("live").set_index("member")
    # a: 95 keys; b: 05:00 of the 6th and 23 hours of the 7th; c: 00:00 of the 7th.
    assert list(rows["forecasts"]) == [95, 24, 1]
    assert list(rows["days"]) == [4, 2, 1]
    assert list(rows["common_forecasts"]) == [95, 0, 0]
    assert list(rows["mae"]) == [1.0, 2.0, 3.0]
    assert report.rungs("live").attrs["missing"] == 0


def test_rungs_compare_on_identical_keys_with_counts(tmp_path):
    """b serves the 6th, where a lacks 05:00. b errs 24 at 05:00 and 2 elsewhere, a 5 on the 6th and 1 elsewhere: b's MAE on
    what it served, on the keys it shares with a, a's on those keys and a's on all of its own are four different numbers."""
    keys = planned_keys(PLAIN, (1,), "1h")
    sixth = "2024-05-06"
    a = [
        record(k, actual_of(k[1], k[2]) + (5.0 if k[0] == sixth else 1.0), PLAIN) for k in keys if k[:3:2] != (sixth, 5)
    ]
    write_member(tmp_path, "a", a, PLAIN)
    b = [record(k, actual_of(k[1], k[2]) + (24.0 if k[2] == 5 else 2.0), PLAIN) for k in keys]
    write_member(tmp_path, "b", b, PLAIN)
    rows = report_of(tmp_path, FallbackEnsemble(["a", "b"], name="live"), models=("a", "b")).rungs("live")
    a, b = rows.set_index("member").loc["a"], rows.set_index("member").loc["b"]
    assert (b["days"], b["forecasts"], b["common_forecasts"]) == (1, 24, 23)
    assert b["mae"] == (23 * 2.0 + 24.0) / 24
    assert (b["common_mae"], b["rung0_mae"]) == (2.0, 5.0)
    assert (a["member_forecasts"], a["member_mae"]) == (5 * 24 - 1, (4 * 24 * 1.0 + 23 * 5.0) / (5 * 24 - 1))
    assert (a["forecasts"], a["mae"], a["rung0_mae"]) == (4 * 24, 1.0, 1.0)


def test_rungs_refuses_what_is_not_a_fallback(tmp_path):
    hand_members(tmp_path)
    report = report_of(tmp_path, AverageEnsemble(["a", "b"], name="avg"), FallbackEnsemble(["a", "b"], name="live"))
    with pytest.raises(ValueError, match="not a FallbackEnsemble"):
        report.rungs("avg")
    with pytest.raises(ValueError, match="No model or ensemble"):
        report.rungs("nothing")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the days the holes leave incomplete
        daily = report.aggregate()
    with pytest.raises(ValueError, match="not a FallbackEnsemble"):
        daily.rungs("live")
    alone = type(report)({"live": report.refs["live"], "a": report.refs["a"]}, [])
    with pytest.raises(ValueError, match=r"\['b'\] are not in this report"):
        alone.rungs("live")


# ---------------------------------------------------------------------------------------------------------------------
# frozen()
# ---------------------------------------------------------------------------------------------------------------------


def yearly_members(directory, test):
    """Three members over a previous_year study of two segments: on segment 0's validation days a and b err least, on segment
    1's b and c, so `top=2` chooses [a, b] and then [b, c]."""
    windows = RunWindows.of_previous_year(test)
    keys = planned_keys(windows, (1,), "1h")
    keys = [k for k in keys if k[2] in (0, 12)]  # two hours a day keep the files small

    def segment_of(run_date):
        return next(
            i for i, s in enumerate(windows.segments) if s.validation[0] <= run_date <= s.validation[1] or i == 1
        )

    errors = {"a": (1.0, 3.0), "b": (1.5, 1.5), "c": (3.0, 1.0)}
    for name, (first, second) in errors.items():
        records = [
            record(k, actual_of(k[1], k[2]) + (first if segment_of(k[0]) == 0 else second), windows) for k in keys
        ]
        Path(directory).mkdir(parents=True, exist_ok=True)
        path = Path(directory) / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        manifest_path(path).write_text(json.dumps(manifest(windows, (1,), "1h")))
    return windows


def study(*extra):
    return pipeline(
        AverageEnsemble(["a", "b", "c"], top=2, name="best"),
        FallbackEnsemble(["best", "c"], name="live"),
        *extra,
    )


def test_frozen_takes_the_newest_segments_members_and_equals_the_top_ensemble_there(tmp_path):
    test = ("2024-01-05", "2025-12-26")
    windows = yearly_members(tmp_path / "study", test)
    assert len(windows.segments) == 2
    original = study()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = original.report(tmp_path / "study")
        frozen = original.frozen(tmp_path / "study")
    assert list(report.choices()["chosen"]) == [["a", "b"], ["b", "c"]]

    best = frozen.ensembles[0]
    assert type(best) is AverageEnsemble and best.top is None and best.name == "best" and best.members == ["b", "c"]
    # The last day whose actual the newest choice read: the day before its test period.
    assert best.chosen_through == "2025-01-04"
    assert frozen.ensembles[1] is original.ensembles[1]
    assert original.ensembles[0].top == 2 and original.frozen_choices == {}
    newest = windows.segments[1]
    assert frozen.frozen_choices["best"] == {
        "members": ["a", "b", "c"],
        "top": 2,
        "chosen": ["b", "c"],
        "chosen_through": "2025-01-04",
        "segment": 1,
        "validation": list(newest.validation),
        "test": list(newest.test),
        "validation_mae": 1.25,
        "validation_forecasts": report.choices().iloc[1]["validation_forecasts"],
        "save_dir": str(tmp_path / "study"),
    }
    # The frozen average over a copy of the members' results equals the top ensemble on the newest segment, bit for bit.
    shutil.copytree(tmp_path / "study", tmp_path / "copy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frozen.report(tmp_path / "copy")

    def on_newest(path):
        lines = [
            json.dumps(r, sort_keys=True) for r in stored(path) if newest.test[0] <= r["run_date"] <= newest.test[1]
        ]
        return sorted(lines)

    assert on_newest(tmp_path / "copy" / "best.jsonl") == on_newest(tmp_path / "study" / "best.jsonl")
    assert len(on_newest(tmp_path / "study" / "best.jsonl")) == 2 * len(RunWindows(test=newest.test).issue_days())
    # and so does the fallback over it.
    assert on_newest(tmp_path / "copy" / "live.jsonl") == on_newest(tmp_path / "study" / "live.jsonl")


def test_a_frozen_pipeline_saved_reloaded_and_reported_forecasts_nothing_in_sample(tmp_path):
    """The reviewer's case: freeze, save, reload and report the study's own days. The frozen average makes no forecast issued
    on or before `chosen_through` (its MAE there would be the in-sample 1.25 of the days it was chosen on, against the honest
    1.757 of the top ensemble on the same 1444 keys), counts them as `chosen_in_sample` and says so; what it makes is the top
    ensemble's forecasts of the newest segment."""
    windows = yearly_members(tmp_path / "study", ("2024-01-05", "2025-12-26"))
    original = study().add_evaluator(MAEEvaluator())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        honest = original.report(tmp_path / "study")
        original.frozen(tmp_path / "study").save(tmp_path / "live.yaml")
    assert "chosen_through: '2025-01-04'" in (tmp_path / "live.yaml").read_text()
    reloaded = ModelPipeline.load(tmp_path / "live.yaml")
    assert reloaded.ensembles[0].chosen_through == "2025-01-04"
    shutil.copytree(tmp_path / "study", tmp_path / "copy")
    with pytest.warns(UserWarning, match="chosen_in_sample"):
        report = reloaded.report(tmp_path / "copy")

    first, second = windows.segments
    # Every period of a day is planned and the members hold two hours a day, so most of the planned keys are member_missing.
    planned_in, planned_out = (len(planned_keys(RunWindows(test=s.test), (1,), "1h")) for s in (first, second))
    in_sample = 2 * len(RunWindows(test=first.test).issue_days())
    out_of_sample = 2 * len(RunWindows(test=second.test).issue_days())
    counts = report.forecast_counts().set_index("model").loc["best"]
    assert (counts["planned"], counts["produced"], counts["chosen_in_sample"], counts["member_missing"]) == (
        planned_in + planned_out,
        out_of_sample,
        planned_in,
        planned_out - out_of_sample,
    )
    validation = report.validation.forecast_counts().set_index("model").loc["best"]
    assert validation["produced"] == 0 and validation["chosen_in_sample"] == validation["planned"] > 0
    assert {r["run_date"] > "2025-01-04" for r in stored(tmp_path / "copy" / "best.jsonl")} == {True}
    mae = report.summary().set_index("model").loc["best"]
    assert (mae["forecasts"], mae["MAE"]) == (out_of_sample, 1.25)
    # The top ensemble on the same keys: the same forecasts, so the same MAE; on all its own keys, the honest score.
    top = {key_of(r): r for r in honest.refs["best"].iter_records()}
    kept = [top[key_of(r)] for r in report.refs["best"].iter_records()]
    assert np.mean([abs(r["prediction"] - r["actual"]) for r in kept]) == 1.25
    assert honest.summary().set_index("model").loc["best", "forecasts"] == in_sample + out_of_sample == 1444
    assert round(honest.summary().set_index("model").loc["best", "MAE"], 3) == 1.757


def test_chosen_through_is_the_last_day_the_choice_read(tmp_path):
    """Fixed windows with a gap and horizons to 3: the validation days serve targets up to their end plus 3, before the test
    start; a validation-only study's choice reads every validation target."""
    gap = RunWindows.fixed(("2024-05-01", "2024-05-04"), ("2024-05-10", "2024-05-12"))
    keys = planned_keys(gap, (1, 2, 3), "1h")
    for name, error in (("a", 2.0), ("b", 1.0), ("c", 3.0)):
        records = [record(k, actual_of(k[1], k[2]) + error, gap) for k in keys]
        write_member(tmp_path / "gap", name, records, gap, horizons=(1, 2, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frozen = pipeline(AverageEnsemble(["a", "b", "c"], top=1, name="best")).frozen(tmp_path / "gap")
    assert frozen.frozen_choices["best"]["chosen_through"] == "2024-05-07" == frozen.ensembles[0].chosen_through
    near = RunWindows.fixed(("2024-05-01", "2024-05-04"), ("2024-05-06", "2024-05-08"))
    keys = planned_keys(near, (1, 2, 3), "1h")
    for name, error in (("a", 2.0), ("b", 1.0), ("c", 3.0)):
        records = [record(k, actual_of(k[1], k[2]) + error, near) for k in keys]
        write_member(tmp_path / "near", name, records, near, horizons=(1, 2, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frozen = pipeline(AverageEnsemble(["a", "b", "c"], top=1, name="best")).frozen(tmp_path / "near")
    assert frozen.ensembles[0].chosen_through == "2024-05-05"


def test_frozen_chooses_from_the_members_not_from_a_stale_ensemble_file(tmp_path):
    yearly_members(tmp_path, ("2024-01-05", "2025-12-26"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study().report(tmp_path)
        stale = manifest_path(tmp_path / "best.jsonl")
        held = json.loads(stale.read_text())
        held["choices"][1]["chosen"] = ["a"]
        stale.write_text(json.dumps(held))
        frozen = study().frozen(tmp_path)
    assert frozen.ensembles[0].members == ["b", "c"]


def test_frozen_of_a_validation_only_study_takes_its_one_choice(tmp_path):
    windows = RunWindows.fixed(("2024-05-01", "2024-05-08"), None)
    keys = planned_keys(windows, (1,), "1h")
    for name, error in (("a", 2.0), ("b", 1.0), ("c", 3.0)):
        write_member(tmp_path, name, [record(k, actual_of(k[1], k[2]) + error, windows) for k in keys], windows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frozen = pipeline(AverageEnsemble(["a", "b", "c"], top=1, name="best")).frozen(tmp_path)
    assert frozen.ensembles[0].members == ["b"]
    assert frozen.frozen_choices["best"]["test"] is None and frozen.frozen_choices["best"]["segment"] == 0
    # Every validation key counts with no test period: the last target is the validation end plus the horizon.
    assert frozen.ensembles[0].chosen_through == "2024-05-09"


def test_frozen_without_a_top_ensemble_is_a_copy_and_refuses_zones(tmp_path):
    plain = pipeline(FallbackEnsemble(["a", "b"], name="live"))
    copy = plain.frozen(tmp_path / "nothing_here")
    assert copy is not plain and copy.ensembles == plain.ensembles and copy.frozen_choices == {}
    yearly_members(tmp_path / "study", ("2024-01-05", "2025-12-26"))
    (tmp_path / "study" / "c.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="No results for model"):
        study().frozen(tmp_path / "study")
    write_zone_index(tmp_path, ["DE", "PL"])
    with pytest.raises(ValueError, match="each zone's folder"):
        study().frozen(tmp_path)


@pytest.mark.parametrize(
    "rebuilt, field",
    [
        (OLSModel(["something_else_d+{horizon}", "z"], training_window=7, name="b"), "predictors"),  # the reviewer's
        (OLSModel(["x", "y"], name="b"), "predictors"),
        (OLSModel(["x"], training_window=MEMBER.training_window + 1, name="b"), "training_window"),
        (OLSModel(["x"], scaler="asinh", name="b"), "model_params"),
        (LassoCVModel(["x"], name="b"), "model_class"),
    ],
)
def test_frozen_refuses_members_that_are_other_models_than_the_results(tmp_path, rebuilt, field):
    yearly_members(tmp_path, ("2024-01-05", "2025-12-26"))
    p = ModelPipeline().add_model(OLSModel(["x"], name="a")).add_model(rebuilt).add_model(OLSModel(["x"], name="c"))
    p.add_ensemble(AverageEnsemble(["a", "b", "c"], top=2, name="best"))
    with pytest.raises(ManifestMismatchError, match=rf"b: {field}"):
        p.frozen(tmp_path)
    assert not (tmp_path / "best.jsonl").exists()  # refused before the report


def horizon_list(horizon):
    return [f"load_d+{horizon}", f"is_weekend_d+{horizon}"]


def ctx_list(ctx):
    return PREDICTORS


def other_ctx_list(ctx):
    return PREDICTORS[:2]


def real_study(**changed):
    """Models of every kind of list a manifest records (a fixed one, a function of the horizon, a function of `ctx`), a
    `NaiveModel`, and a `ResidualModel` whose base isn't a member of the `top` ensemble; `changed` swaps one of them."""
    models = {
        "fixed": OLSModel(PREDICTORS, training_window=30, name="fixed"),
        "by_horizon": OLSModel(horizon_list, training_window=30, name="by_horizon"),
        "function": OLSModel(ctx_list, training_window=30, name="function"),
        "naive": NaiveModel(["load_d+{horizon}"], name="naive"),
        "base_ols": OLSModel([*PREDICTORS, "price_d-1"], training_window=21, name="base_ols"),
        "res": ResidualModel(
            "base_ols",
            HistGBModel(["load_d+{horizon}", "base_fc_d+{horizon}"], max_iter=12, target_transform="none"),
            training_window=28,
            name="res",
        ),
        **changed,
    }
    built = ModelPipeline()
    for model in models.values():
        built.add_model(model)
    return built.add_ensemble(AverageEnsemble(["fixed", "by_horizon", "function", "naive", "res"], top=2, name="best"))


def test_frozen_accepts_the_models_a_real_run_wrote_and_refuses_others(tmp_path):
    data = make_market_data(horizon=1, noise=5.0, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        real_study().run(data, validation=("2023-03-28", "2023-03-31"), test=("2023-04-03", "2023-04-05"), horizon=1, save_dir=str(tmp_path))  # fmt: skip
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frozen = real_study().frozen(tmp_path)
    assert not [w for w in caught if "frozen()" in str(w.message)]  # every field compared, none differing
    assert frozen.frozen_choices["best"]["chosen_through"] == "2023-04-01"
    # The residual's base is no member: it stays bound, and its results aren't read.
    (tmp_path / "base_ols.jsonl").unlink()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert real_study().frozen(tmp_path).ensembles[0].members == frozen.ensembles[0].members
    others = {
        "fixed": OLSModel(list(reversed(PREDICTORS)), training_window=30, name="fixed"),
        "by_horizon": OLSModel(lambda horizon: [f"wind_d+{horizon}"], training_window=30, name="by_horizon"),
        "function": OLSModel(other_ctx_list, training_window=30, name="function"),
        "naive": NaiveModel(["wind_d+{horizon}"], name="naive"),
        "res": ResidualModel(
            "base_ols",
            HistGBModel(["load_d+{horizon}", "base_fc_d+{horizon}"], max_iter=13, target_transform="none"),
            training_window=28,
            name="res",
        ),
    }
    for name, model in others.items():
        with pytest.raises(ManifestMismatchError, match=rf"{name}: (predictors|predictor_function|model_params)"):
            real_study(**{name: model}).frozen(tmp_path)


# ---------------------------------------------------------------------------------------------------------------------
# Construction and YAML
# ---------------------------------------------------------------------------------------------------------------------


def test_the_parameters_are_checked():
    with pytest.raises(ValueError, match="per must be one of"):
        FallbackEnsemble(["a", "b"], per="hour", name="live")
    with pytest.raises(TypeError):
        FallbackEnsemble(["a", "b"], "day", name="live")
    with pytest.raises(TypeError):
        FallbackEnsemble(["a", "b"])
    with pytest.raises(ValueError, match="listed twice"):
        FallbackEnsemble(["a", "a"], name="live")
    with pytest.raises(TypeError, match="chooses nothing"):
        FallbackEnsemble(["a"], name="live").choose(np.zeros((1, 1)), np.zeros(1))
    with pytest.raises(ValueError, match="can't have the ensembles"):
        pipeline(
            FallbackEnsemble(["a", "b"], name="live"), AverageEnsemble(["live", "c"], top=1, name="best")
        )._check_ensembles(None)


def test_yaml_round_trip(tmp_path):
    assert COMPONENT_REGISTRY["ensembles"]["FallbackEnsemble"] == "epftoolbox2.ensembles.fallback"
    original = pipeline(FallbackEnsemble(["a", "b"], per="key", name="live"), AverageEnsemble(["a", "b"], name="avg"))
    original.save(tmp_path / "pipeline.yaml")
    loaded = ModelPipeline.load(tmp_path / "pipeline.yaml")
    fallback = loaded.ensembles[0]
    assert type(fallback) is FallbackEnsemble
    assert (fallback.members, fallback.per, fallback.name) == (["a", "b"], "key", "live")
    assert loaded.to_dict() == original.to_dict()
