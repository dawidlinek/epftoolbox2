"""D4's record keys (step 5a): `status`, `list_hash`, `refit_anchor` and `revision` on every record, skip records, the readers that
give forecasts alone, revisions and repeated keys, a half-written last line, and the writer lock.

Every forecast's record keeps the keys it had before, in their order, and ends with the four; the bitwise claim against the
commit before them (records with the new keys removed, tables, counts, manifests without provenance) is the equivalence check in
`epftoolbox2-research/library_check_step5/record_status/`. Here: what the keys hold, and that a table, a count and a resume read
the forecasts alone, whatever else the file holds.
"""

import errno
import json
import os
import subprocess
import sys
import textwrap
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from filelock import FileLock

import epftoolbox2
from epftoolbox2.ensembles import AverageEnsemble
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import NaiveModel, OLSModel, RidgeModel
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.models.context import list_hash
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import manifest_path
from epftoolbox2.results.ref import ModelResultRef
from epftoolbox2.results.store import ResultsLockedError, ResultStore, read_records
from tests.synthetic import PREDICTORS, make_market_data

# The keys a forecast's record had before D4, in their order, and what it ends with now.
PLAIN_KEYS = [
    "run_date",
    "target_date",
    "run_weekday",
    "hour",
    "horizon",
    "day_in_test",
    "prediction",
    "actual",
    "coefficients",
    "freq",
]
NEW_KEYS = ["status", "list_hash", "refit_anchor", "revision"]
DAY = "2023-04-12"


class InProcessPool:
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


def market(**kwargs):
    return make_market_data(days=120, noise=5.0, seed=4, **kwargs)


def lines(path):
    return sorted(line for line in Path(path).read_text(encoding="utf-8").splitlines() if line)


def records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def without_new_keys(record):
    return {k: v for k, v in record.items() if k not in NEW_KEYS or (k == "refit_anchor" and "refit" in record)}


def by_hour(horizon, hour):
    """A list that depends on the period and the horizon, as a `(horizon, hour)` callable gives it."""
    if hour % 2:
        return PREDICTORS
    return PREDICTORS[:3] + ["price_d-1"] if horizon == 1 else PREDICTORS[:2]


def expand(names, horizon):
    return [name.format(horizon=horizon) for name in names]


def ctx_list(ctx):
    """A predictor function returning the fixed list `PREDICTORS` expands to."""
    return expand(PREDICTORS, ctx.horizon)


def run(data, model, save_dir, start=DAY, end=DAY, **kwargs):
    pipeline = ModelPipeline().add_model(model).add_evaluator(MAEEvaluator())
    return pipeline.run(data, test_start=start, test_end=end, horizon=2, save_dir=str(save_dir), **kwargs)


def tables(report):
    return {
        "summary": report.summary(),
        "by_hour": report.by_hour(),
        "by_horizon": report.by_horizon(),
        "counts": report.forecast_counts(),
    }


def assert_same_tables(a, b):
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name], check_exact=True, obj=name)


# --- the four keys ---------------------------------------------------------------------------------------------------------


def test_every_forecast_ends_with_the_four_keys_after_the_ones_it_had(tmp_path):
    data = market()
    run(data, OLSModel(PREDICTORS, 45, name="ols"), tmp_path)
    stored = records(tmp_path / "ols.jsonl")
    assert len(stored) == 48
    for r in stored:
        assert list(r) == PLAIN_KEYS + NEW_KEYS
        assert (r["status"], r["refit_anchor"], r["revision"]) == ("ok", r["run_date"], 1)
        assert r["list_hash"] == list_hash(
            expand(PREDICTORS, r["horizon"])
        )  # the hash a function returning that list gets
    assert len({r["list_hash"] for r in stored}) == 2  # one list per horizon
    assert json.loads(manifest_path(tmp_path / "ols.jsonl").read_text())["record_schema"] == 2


def test_a_fixed_list_has_its_own_hash_per_period_and_horizon(tmp_path):
    data = market()
    run(data, OLSModel(by_hour, 45, name="ols"), tmp_path)
    for r in records(tmp_path / "ols.jsonl"):
        assert r["list_hash"] == list_hash(expand(by_hour(r["horizon"], r["hour"]), r["horizon"])), (
            r["hour"],
            r["horizon"],
        )
    assert len({r["list_hash"] for r in records(tmp_path / "ols.jsonl")}) == 4


def test_a_function_record_hashes_its_list_as_a_fixed_list_of_the_same_names_does(tmp_path):
    data = market()
    run(data, OLSModel(PREDICTORS, 45, name="fixed"), tmp_path / "fixed")
    run(data, OLSModel(ctx_list, 45, name="function"), tmp_path / "function")
    fixed = {(r["hour"], r["horizon"]): r for r in records(tmp_path / "fixed" / "fixed.jsonl")}
    function = records(tmp_path / "function" / "function.jsonl")
    assert len(function) == len(fixed) == 48
    for r in function:
        assert list(r) == PLAIN_KEYS + ["predictor_list"] + NEW_KEYS
        assert r["list_hash"] == r["predictor_list"] == fixed[(r["hour"], r["horizon"])]["list_hash"]


def test_a_refit_record_keeps_its_anchor_where_it_was(tmp_path):
    data = market()
    run(data, RidgeModel(PREDICTORS, 40, refit="7D", name="ridge"), tmp_path, end="2023-04-13")
    for r in records(tmp_path / "ridge.jsonl"):
        assert list(r) == PLAIN_KEYS + ["refit_anchor", "status", "list_hash", "revision"]
        assert r["refit_anchor"] == "2023-04-10" and r["status"] == "ok" and r["revision"] == 1


def test_a_window_list_and_an_ensemble(tmp_path):
    data = market()
    pipeline = (
        ModelPipeline()
        .add_model(OLSModel(PREDICTORS, [30, 45], name="members"))
        .add_model(OLSModel(PREDICTORS, 45, name="single"))
        .add_ensemble(AverageEnsemble(["members", "single"], name="average"))
    )
    pipeline.run(data, test_start=DAY, test_end=DAY, horizon=2, save_dir=str(tmp_path))
    for r in records(tmp_path / "members.jsonl"):
        assert list(r) == PLAIN_KEYS[:7] + ["member_predictions"] + PLAIN_KEYS[7:] + NEW_KEYS
        assert r["coefficients"] == [] and r["refit_anchor"] == r["run_date"]
    for r in records(tmp_path / "average.jsonl"):  # no list or fit of its own
        assert list(r) == PLAIN_KEYS + ["status", "revision"]


def test_a_naive_record(tmp_path):
    data = market()
    run(data, NaiveModel(["load_d+{horizon}"], name="naive"), tmp_path)
    for r in records(tmp_path / "naive.jsonl"):
        assert list(r) == PLAIN_KEYS + NEW_KEYS
        assert r["list_hash"] == list_hash([f"load_d+{r['horizon']}"])


# --- skip records ----------------------------------------------------------------------------------------------------------


def holes(data):
    """Load missing on the issue rows of hours 3 and 4 (`missing_test_predictors`), and the 07:00 row gone
    (`missing_test_row`)."""
    data = data.copy()
    issue = (data.index.date == pd.Timestamp(DAY).date()) & np.isin(data.index.hour, [3, 4])
    data.loc[issue, "load_d+1"] = np.nan
    return data.drop(index=pd.Timestamp(f"{DAY} 07:00", tz=data.index.tz))


def skips_of(path):
    return {(r["hour"], r["horizon"]): r for r in records(path) if r["status"] != "ok"}


def test_a_skipped_key_has_one_skip_record_with_its_reason_and_actual(tmp_path):
    data = holes(market())
    report = run(data, OLSModel(PREDICTORS, 45, name="ols"), tmp_path)
    path = tmp_path / "ols.jsonl"
    skips = skips_of(path)
    assert set(skips) == {(3, 1), (4, 1), (7, 1), (7, 2)}
    for (hour, horizon), r in skips.items():
        assert list(r) == PLAIN_KEYS + NEW_KEYS
        assert r["prediction"] is None and r["coefficients"] == [] and "detail" not in r
        assert r["list_hash"] == list_hash(expand(PREDICTORS, horizon))
        assert (r["refit_anchor"], r["revision"], r["run_date"], r["day_in_test"]) == (DAY, 1, DAY, 0)
    assert skips[(3, 1)]["status"] == skips[(4, 1)]["status"] == "missing_test_predictors"
    assert skips[(7, 1)]["status"] == skips[(7, 2)]["status"] == "missing_test_row"
    target = data.loc[pd.Timestamp("2023-04-13 03:00", tz=data.index.tz), "price"]
    assert skips[(3, 1)]["actual"] == target  # the issue row's target, as a forecast's record has it
    assert skips[(7, 1)]["actual"] is None  # no row to read it from

    ref = report.refs["ols"]
    assert (ref.count, ref.produced) == (48, 44)
    assert ref.skipped == {"missing_test_predictors": 2, "missing_test_row": 2}
    assert len(list(ref.iter_records())) == 44 and all(r["status"] == "ok" for r in ref.iter_records())
    assert report.summary().set_index("model").loc["ols", "forecasts"] == 44
    store = ResultStore(path, test_start=DAY)
    assert len(store.load_all()) == 44 and len(list(store.iter_lines(statuses=None))) == 48
    assert json.loads(manifest_path(path).read_text())["skipped"] == {
        "test": {"missing_test_predictors": 2, "missing_test_row": 2}
    }


def test_the_detail_of_a_member_skip(tmp_path):
    data = market()
    days = pd.Index(data.index.date)
    # Load missing on every row of the 3-day window of the issue day's forecasts, so the short member has no finite row.
    short = (days >= pd.Timestamp("2023-04-08").date()) & (days <= pd.Timestamp("2023-04-11").date())
    data.loc[short & (data.index.hour == 5), "load_d+1"] = np.nan
    run(data, OLSModel(PREDICTORS, [3, 45], name="members"), tmp_path)
    skip = skips_of(tmp_path / "members.jsonl")[(5, 1)]
    assert (skip["status"], skip["detail"]) == ("no_finite_training_rows", "window 3")
    assert list(skip) == PLAIN_KEYS + NEW_KEYS + ["detail"]


def test_a_not_published_key_is_a_skip_record_from_the_parent(tmp_path):
    zone = "Europe/Warsaw"
    dates = pd.date_range("2024-03-18", periods=40, freq="D")
    frame = pd.DataFrame(
        {"actual_daily_max": 20000 + 50 * np.arange(40, dtype=float)},
        index=(dates + pd.Timedelta(hours=12)).tz_localize(zone),
    )
    for k in range(1, 8):
        frame[f"load_forecast_daily_max_d+{k}"] = 1000.0 * k + np.arange(40)
    info = InformationSet("12:00", zone, rules={"actual_daily_max": Availability.realised("1h")})
    pipeline = ModelPipeline(information_set=info).add_model(
        NaiveModel(["load_forecast_daily_max_d+{horizon}"], name="tso")
    )
    report = pipeline.run(
        frame, "2024-03-25", "2024-04-07", target="actual_daily_max", horizon=7, freq="1D", save_dir=str(tmp_path)
    )
    skips = [r for r in records(tmp_path / "tso.jsonl") if r["status"] != "ok"]
    assert len(skips) == 20 and {r["status"] for r in skips} == {"not_published"}
    assert (
        report.refs["tso"].produced == 78 and report.forecast_counts().set_index("model").loc["tso", "produced"] == 78
    )
    local = frame.index.tz_localize(None).normalize()
    for r in skips:
        assert r["freq"] == "1D" and r["hour"] == 0 and r["prediction"] is None
        target = pd.Timestamp(r["target_date"])
        assert r["actual"] == frame.loc[local == target, "actual_daily_max"].iloc[0]


def test_a_resume_retries_skips_and_writes_one_only_when_the_reason_changed(tmp_path):
    data = holes(market())
    model = OLSModel(PREDICTORS, 45, name="ols")
    run(data, model, tmp_path)
    path = tmp_path / "ols.jsonl"
    first = path.read_bytes()

    again = run(data, model, tmp_path)  # the same reasons: nothing written
    assert path.read_bytes() == first
    assert again.refs["ols"].skipped == {"missing_test_predictors": 2, "missing_test_row": 2}  # retried, as before

    # Hour 3 has its load back (a forecast now), and hour 4 has none on any row, which no window can train on (another reason).
    fixed = data.copy()
    issue = (fixed.index.date == pd.Timestamp(DAY).date()) & (fixed.index.hour == 3)
    fixed.loc[issue, "load_d+1"] = market().loc[fixed.index[issue], "load_d+1"].to_numpy()
    fixed.loc[fixed.index.hour == 4, "load_d+1"] = np.nan
    resumed = run(fixed, model, tmp_path, allow_manifest_mismatch=True)
    added = [json.loads(line) for line in path.read_bytes()[len(first) :].decode().splitlines()]
    assert sorted((r["hour"], r["horizon"], r["status"]) for r in added) == [
        (3, 1, "ok"),
        (4, 1, "no_finite_training_rows"),
    ]
    ref = resumed.refs["ols"]
    assert ref.produced == 45 and (3, 1) in {(r["hour"], r["horizon"]) for r in ref.iter_records()}
    assert ref.skipped == {"no_finite_training_rows": 1, "missing_test_row": 2}
    winners = {(r["hour"], r["horizon"]): r["status"] for r in ResultStore(path).iter_lines(statuses=None)}
    assert winners[(3, 1)] == "ok" and winners[(4, 1)] == "no_finite_training_rows"

    fresh = run(fixed, model, tmp_path / "fresh")
    ok = [line for line in lines(path) if '"status": "ok"' in line]
    assert ok == [line for line in lines(tmp_path / "fresh" / "ols.jsonl") if '"status": "ok"' in line]
    assert_same_tables(tables(resumed), tables(fresh))

    size = path.stat().st_size
    run(fixed, model, tmp_path, allow_manifest_mismatch=True)
    assert path.stat().st_size == size


def test_records_held_in_memory_are_the_forecasts_alone():
    data = holes(market())
    report = ModelPipeline().add_model(OLSModel(PREDICTORS, 45, name="ols")).run(data, DAY, DAY, horizon=2)
    ref = report.refs["ols"]
    assert ref.produced == len(ref._results) == 44 and {r["status"] for r in ref._results} == {"ok"}


# --- old files -------------------------------------------------------------------------------------------------------------


def as_before(path):
    """The file as the code before D4 left it: no skip records, no new keys, and a manifest without `record_schema`."""
    kept = [without_new_keys(r) for r in records(path) if r["status"] == "ok"]
    path.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
    manifest = json.loads(manifest_path(path).read_text())
    del manifest["record_schema"]
    manifest_path(path).write_text(json.dumps(manifest, indent=2) + "\n")
    return kept


def test_an_old_file_resumes_and_reports_as_before(tmp_path):
    data = holes(market())
    model = OLSModel(PREDICTORS, 45, name="ols")
    run(data, model, tmp_path / "old", end="2023-04-13")
    path = tmp_path / "old" / "ols.jsonl"
    old = as_before(path)
    old_lines = lines(path)
    assert len(old) == 92 and all("status" not in r for r in old)

    pipeline = ModelPipeline().add_model(model).add_evaluator(MAEEvaluator())
    report = pipeline.report(str(tmp_path / "old"), DAY, "2023-04-13", horizon=2)
    fresh = run(data, model, tmp_path / "fresh", end="2023-04-13")
    assert_same_tables(tables(report), tables(fresh))

    resumed = run(data, model, tmp_path / "old", start="2023-04-11", end="2023-04-13")  # a day more, into the old file
    assert set(old_lines) <= set(lines(path))  # the old records stay as they were
    added = [r for r in records(path) if "status" in r]
    assert {r["run_date"] for r in added if r["status"] == "ok"} == {"2023-04-11"}
    assert {(r["run_date"], r["status"]) for r in added if r["status"] != "ok"} == {
        (DAY, "missing_test_predictors"),
        (DAY, "missing_test_row"),
    }  # the old run's skipped keys, retried
    both = run(data, model, tmp_path / "both", start="2023-04-11", end="2023-04-13")
    assert_same_tables(tables(resumed), tables(both))  # a file mixing old and new records reads as one

    def plain(ref):  # `day_in_test` counts from the start each record was written under, as in any resume
        return sorted(
            json.dumps({k: v for k, v in without_new_keys(r).items() if k != "day_in_test"}) for r in ref.iter_records()
        )

    assert plain(resumed.refs["ols"]) == plain(both.refs["ols"])
    manifest = json.loads(manifest_path(path).read_text())
    assert manifest["record_schema"] == 2 and manifest["skipped"] == {
        "test": {"missing_test_predictors": 2, "missing_test_row": 2}
    }


# --- the last line ---------------------------------------------------------------------------------------------------------


def test_a_file_cut_mid_line_reads_with_a_warning_and_resumes_to_a_fresh_run(tmp_path):
    data = holes(market())
    model = OLSModel(PREDICTORS, 45, name="ols")
    fresh = run(data, model, tmp_path / "fresh")
    run(data, model, tmp_path / "cut")
    path = tmp_path / "cut" / "ols.jsonl"
    whole = path.read_bytes()
    last = whole.rstrip(b"\n").rfind(b"\n") + 1
    path.write_bytes(whole[: last + 37])  # a crash in the middle of the last line

    with pytest.warns(UserWarning, match="last line .* is not a whole record"):
        read = list(ModelResultRef("ols", path, 48, DAY, DAY, 2).iter_records())
    assert len(read) == len([r for r in records(tmp_path / "fresh" / "ols.jsonl") if r["status"] == "ok"]) - (
        json.loads(whole[last:])["status"] == "ok"
    )

    with pytest.warns(UserWarning, match=rf"cut off its last 37 bytes \(from byte {last}\)"):
        resumed = run(data, model, tmp_path / "cut")
    assert lines(path) == lines(tmp_path / "fresh" / "ols.jsonl")
    assert_same_tables(tables(resumed), tables(fresh))


def test_a_malformed_line_before_the_last_raises_with_its_number(tmp_path):
    data = market()
    run(data, OLSModel(PREDICTORS, 45, name="ols"), tmp_path)
    path = tmp_path / "ols.jsonl"
    text = path.read_text().splitlines()
    text[2] = text[2][:50]
    path.write_text("\n".join(text) + "\n")
    with pytest.raises(ValueError, match="line 3 is not a result record"):
        list(ModelResultRef("ols", path, 48, DAY, DAY, 2).iter_records())
    with pytest.raises(ValueError, match="line 3 is not a result record"):
        run(data, OLSModel(PREDICTORS, 45, name="ols"), tmp_path)
    with pytest.raises(ValueError, match="line 3"):
        ResultStore(path, test_start=DAY, lock=True)
    free = FileLock(str(tmp_path / "ols.lock"))
    free.acquire(timeout=0)  # neither kept the lock
    free.release()


def test_a_whole_last_record_without_its_newline_is_kept(tmp_path):
    path = tmp_path / "m.jsonl"
    record = {"run_date": DAY, "hour": 0, "horizon": 1, "prediction": 1.0, "actual": 2.0}
    path.write_text(json.dumps(record), encoding="utf-8")
    store = ResultStore(path, test_start=DAY, lock=True)
    store.save({**record, "hour": 1})
    store.close()
    assert [r["hour"] for r in records(path)] == [0, 1] and path.read_bytes().endswith(b"}\n")


# --- repeated keys and revisions -------------------------------------------------------------------------------------------


def test_repeats_with_the_same_forecast_warn_once_and_are_read_once(tmp_path):
    """The same forecast twice, once as an old-format line: equal but for D4's keys, so one forecast, read once."""
    data = market()
    report = run(data, OLSModel(PREDICTORS, 45, name="ols"), tmp_path)
    clean = tables(report)
    path = tmp_path / "ols.jsonl"
    stored = records(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(stored[0]) + "\n" + json.dumps(without_new_keys(stored[1])) + "\n")
    pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, 45, name="ols")).add_evaluator(MAEEvaluator())
    with pytest.warns(
        UserWarning, match=r"2 records repeat a forecast key .*, with the same forecast: each key is read once"
    ):
        assert_same_tables(tables(pipeline.report(str(tmp_path), DAY, DAY, horizon=2)), clean)
    with pytest.warns(UserWarning, match="repeat a forecast key"):
        assert ResultStore(path).load_all() == stored


def test_a_repeat_with_another_forecast_raises_naming_the_file_and_a_key(tmp_path):
    """Two configurations' forecasts on one key: no table, count, ensemble or resume may pick one."""
    data = market()
    model = OLSModel(PREDICTORS, 45, name="ols")
    run(data, model, tmp_path)
    path = tmp_path / "ols.jsonl"
    stored = records(path)
    changed = {**stored[5], "prediction": stored[5]["prediction"] + 1.0}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(stored[0]) + "\n" + json.dumps(changed) + "\n")
    key = f"period {stored[5]['hour']}, horizon {stored[5]['horizon']}, run date {DAY}"
    message = rf"{path.name} holds forecast keys twice with different forecasts \(1 records, e.g. {key}; 1 more"
    pipeline = ModelPipeline().add_model(model).add_evaluator(MAEEvaluator())
    with pytest.raises(ValueError, match=message):
        pipeline.report(str(tmp_path), DAY, DAY, horizon=2)
    with pytest.raises(ValueError, match=message):
        run(data, model, tmp_path)  # a resume
    with pytest.raises(ValueError, match=message):
        list(ModelResultRef("ols", path, 48, DAY, DAY, 2).iter_records())
    with pytest.warns(UserWarning, match=r"2 records repeat .* \(1 with another forecast, the first read\)"):
        assert list(read_records(path, duplicates="first")) == stored  # for a repair tool alone


def hand_records(path, prediction, n=10):
    rows = [
        {"run_date": DAY, "target_date": "2023-04-13", "hour": hour, "horizon": 1, "prediction": prediction + hour}
        for hour in range(n)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return rows


def swap_after_the_first_pass(monkeypatch, path, then):
    """Call `then()` once the first pass of `read_records` has read the last line of `path`, before the second pass."""
    from epftoolbox2.results import store

    lines = len(path.read_text().splitlines())
    real, calls = store._key_fields, []

    def counting(*args):
        out = real(*args)
        calls.append(1)
        if len(calls) == lines:
            then()
        return out

    monkeypatch.setattr(store, "_key_fields", counting)


def test_a_file_replaced_between_the_passes_is_read_as_it_was(tmp_path, monkeypatch):
    """An ensemble's file is `os.replace`d on every report: a read that spans it gives the file it opened, never a mix."""
    path, other = tmp_path / "e.jsonl", tmp_path / "other.jsonl"
    original = hand_records(path, 1.0)
    hand_records(other, 1000.25, n=7)  # other numbers, other line lengths
    swap_after_the_first_pass(monkeypatch, path, lambda: os.replace(other, path))
    assert list(read_records(path)) == original


@pytest.mark.parametrize("kind", ["other lines", "the same lengths, other keys", "the same keys, other numbers"])
def test_a_file_written_in_place_between_the_passes_raises(tmp_path, monkeypatch, kind):
    """Written in place, the offsets of the first pass point into other records: every chosen line is checked against what the
    first pass read off it, so this raises rather than reading the other file's numbers under the first's keys."""
    path, other = tmp_path / "e.jsonl", tmp_path / "other.jsonl"
    hand_records(path, 1.0)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if kind == "other lines":
        hand_records(other, 1000.25)  # ten records, longer lines: the first pass itself reads past where the file ended
    elif kind == "the same lengths, other keys":
        other.write_text("".join(json.dumps({**r, "hour": 9 - r["hour"]}) + "\n" for r in rows))
    else:
        other.write_text("".join(json.dumps({**r, "prediction": r["prediction"] + 4.0}) + "\n" for r in rows))
    swap_after_the_first_pass(monkeypatch, path, lambda: path.write_bytes(other.read_bytes()))
    with pytest.raises(ValueError, match="changed while it was read|is not a result record"):
        list(read_records(path))


def test_the_highest_revision_wins_wherever_it_stands(tmp_path):
    path = tmp_path / "m.jsonl"
    key = {"run_date": DAY, "target_date": "2023-04-13", "hour": 0, "horizon": 1}
    rows = [
        {**key, "prediction": 1.0, "status": "ok", "revision": 1},
        {**key, "prediction": 2.0, "status": "ok", "revision": 2},
        {**key, "hour": 1, "prediction": 3.0, "status": "ok", "revision": 2},
        {**key, "hour": 1, "prediction": 4.0, "status": "ok", "revision": 1},
        {**key, "hour": 2, "prediction": 5.0, "status": "fit_error", "revision": 1},
        {
            **key,
            "hour": 2,
            "prediction": 6.0,
            "status": "ok",
        },  # no revision: 1, and ok wins over a skip of its revision
        {**key, "hour": 3, "prediction": 7.0},  # an old record: ok
        {**key, "hour": 3, "prediction": None, "status": "fit_error", "revision": 1},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    read = {r["hour"]: r["prediction"] for r in read_records(path)}
    assert read == {0: 2.0, 1: 3.0, 2: 6.0, 3: 7.0}
    ref = ModelResultRef("m", path, 4, DAY, DAY, 1)
    assert sorted(r["prediction"] for r in ref.iter_records()) == [2.0, 3.0, 6.0, 7.0]
    store = ResultStore(path, test_start=DAY)
    assert [store.is_done(hour, 1, 0) for hour in range(4)] == [True] * 4


def test_a_forecast_of_a_key_at_revision_2_is_written_at_revision_2_and_read(tmp_path):
    """A forecast at revision 1 and a skip record at revision 2: the key is the skip's, so not done. The resume's forecast is
    written at revision 2, where a reader takes it, and the next resume writes nothing; `produced` is what the tables count."""
    data = market()
    model = OLSModel(PREDICTORS, 45, name="ols")
    run(data, model, tmp_path)
    path = tmp_path / "ols.jsonl"
    stored = records(path)
    later = {**stored[3], "prediction": None, "coefficients": [], "status": "fit_error", "revision": 2}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(later) + "\n")
    key = (stored[3]["hour"], stored[3]["horizon"], DAY)
    assert key not in {(r["hour"], r["horizon"], r["run_date"]) for r in read_records(path)}

    resumed = run(data, model, tmp_path)
    added = records(path)[len(stored) + 1 :]
    assert [(r["hour"], r["horizon"], r["status"], r["revision"]) for r in added] == [(*key[:2], "ok", 2)]
    assert added[0]["prediction"] == stored[3]["prediction"]  # the same fit, at the key's revision
    ref = resumed.refs["ols"]
    assert ref.produced == sum(1 for _ in ref.iter_records()) == 48
    assert resumed.summary().set_index("model").loc["ols", "forecasts"] == 48
    size = path.stat().st_size
    run(data, model, tmp_path)
    assert path.stat().st_size == size


def test_the_first_pass_reads_a_line_s_key_as_json_does(tmp_path):
    """The first pass of `read_records` reads key, revision and status off the line without parsing it: other spacing and key
    order, a quoted "status" or "revision" inside a string, and values not as the store writes them give what JSON gives."""
    path = tmp_path / "m.jsonl"
    rows = [
        '{"hour":0,"horizon":1,"run_date":"2023-04-12","prediction":1.0,"status":"ok"}',
        '{"run_date": "2023-04-12", "horizon": 1, "hour": 0, "prediction": 2.0, "status": "fit_error", "revision": 2,'
        ' "detail": "x \\"status\\": \\"ok\\", \\"revision\\": 9"}',
        '{"hour": 1.0, "horizon": 1, "run_date": "2023-04-12", "prediction": 3.0}',  # a float hour: parsed
        '{ "hour" : 2 , "horizon" : 1 , "run_date" : "2023-04-12" , "prediction" : 4.0 , "revision" : 1 }',
        '{"hour": 3, "horizon": 1, "run_date": "2023-04-12", "prediction": 5.0, "status": "too_few_rows"}',
        '{"hour": 3, "horizon": 1, "run_date": "2023-04-12", "prediction": 6.0, "status": "ok"}',
    ]
    path.write_text("".join(row + "\n" for row in rows), encoding="utf-8")
    chosen = {(r["hour"], r["horizon"]): (r["prediction"], r.get("status")) for r in read_records(path, statuses=None)}
    assert chosen == {(0, 1): (2.0, "fit_error"), (1, 1): (3.0, None), (2, 1): (4.0, None), (3, 1): (6.0, "ok")}
    assert [r["prediction"] for r in read_records(path)] == [3.0, 4.0, 6.0]


def test_skip_records_alone_are_not_done_and_the_last_is_the_keys(tmp_path):
    path = tmp_path / "m.jsonl"
    key = {"run_date": DAY, "target_date": "2023-04-13", "hour": 0, "horizon": 1, "prediction": None}
    rows = [{**key, "status": "too_few_rows", "revision": 1}, {**key, "status": "fit_error", "revision": 1}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    store = ResultStore(path, test_start=DAY)
    assert not store.is_done(0, 1, 0) and not store.has_results() and store.load_all() == []
    assert [r["status"] for r in store.iter_lines(statuses=None)] == ["fit_error"]
    assert not store.save_skip({**key, "status": "fit_error", "revision": 1})
    assert store.save_skip({**key, "status": "too_few_rows", "revision": 1})
    store.close()
    assert [r["status"] for r in records(path)] == ["too_few_rows", "fit_error", "too_few_rows"]


# --- the lock --------------------------------------------------------------------------------------------------------------

HOLD = textwrap.dedent(
    """
    import sys
    from filelock import FileLock
    lock = FileLock(sys.argv[1])
    lock.acquire()
    print("held", flush=True)
    sys.stdin.read()
    """
)


def test_a_second_writer_from_another_process_raises_naming_the_lock(tmp_path):
    data = market()
    lock = tmp_path / "ols.lock"
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLD, str(lock)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(ResultsLockedError, match=str(lock).replace("\\", "\\\\")):
            run(data, OLSModel(PREDICTORS, 45, name="ols"), tmp_path)
        assert not (tmp_path / "ols.jsonl").exists() and not manifest_path(tmp_path / "ols.jsonl").exists()
    finally:
        holder.communicate("")
    report = run(data, OLSModel(PREDICTORS, 45, name="ols"), tmp_path)  # released when the other run ends
    assert report.refs["ols"].produced == 48


def test_a_second_writer_in_one_process_raises_and_the_lock_goes_with_the_run(tmp_path):
    data = market()
    holding = ResultStore(tmp_path / "ols.jsonl", lock=True)
    with pytest.raises(ResultsLockedError, match="ols.lock"):
        run(data, OLSModel(PREDICTORS, 45, name="ols"), tmp_path)
    holding.close()
    run(data, OLSModel(PREDICTORS, 45, name="ols"), tmp_path)
    ResultStore(tmp_path / "ols.jsonl", lock=True).close()  # the run let it go


ORPHAN = textwrap.dedent(
    """
    import os, signal, sys, time
    sys.path.insert(0, sys.argv[2])  # the epftoolbox2 the test imported, whatever the working directory
    from epftoolbox2.results.store import ResultStore
    store = ResultStore(sys.argv[1], lock=True)
    worker = os.fork()  # as a pool worker of a run is
    if worker == 0:
        time.sleep(60)
        os._exit(0)
    print(worker, flush=True)
    os.kill(os.getpid(), signal.SIGKILL)  # the run killed, its worker not
    """
)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork")
def test_a_worker_outliving_a_killed_run_does_not_keep_the_lock(tmp_path):
    path = tmp_path / "ols.jsonl"
    root = str(Path(epftoolbox2.__file__).resolve().parents[1])
    run_ = subprocess.Popen([sys.executable, "-c", ORPHAN, str(path), root], stdout=subprocess.PIPE, text=True)
    worker = int(run_.stdout.readline())
    try:
        run_.wait(timeout=60)
        os.kill(worker, 0)  # the worker is still there
        ResultStore(path, lock=True).close()  # and the lock is free
    finally:
        os.kill(worker, 9)


@pytest.mark.skipif(sys.platform == "win32", reason="flock")
@pytest.mark.parametrize("code", [errno.ENOSYS, errno.ENOLCK, errno.EOPNOTSUPP])
def test_a_file_system_that_refuses_locks_warns_and_the_run_goes_on(tmp_path, monkeypatch, code):
    import fcntl

    def refuse(fd, operation):
        if operation & fcntl.LOCK_UN:
            return None
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(fcntl, "flock", refuse)
    with pytest.warns(UserWarning, match="the file system refused a lock"):
        report = run(market(), OLSModel(PREDICTORS, 45, name="ols"), tmp_path)
    assert report.refs["ols"].produced == 48


def test_results_are_read_as_utf8(tmp_path, monkeypatch):
    """Whatever the locale says: a hand-written record with a non-ASCII value reads back as written."""
    monkeypatch.setattr("locale.getpreferredencoding", lambda *a: "cp1252")
    path = tmp_path / "m.jsonl"
    path.write_bytes('{"run_date": "2023-04-12", "hour": 0, "horizon": 1, "note": "pr\u00f3ba"}\n'.encode("utf-8"))
    assert ResultStore(path).load_all()[0]["note"] == "pr\u00f3ba"
