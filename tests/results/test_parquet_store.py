"""Parquet results (step 5b): `run(save_format="parquet")`, the catalog, `epf results convert|compact|verify`, the readers' rules
on a directory, `ModelResultRef.frame`, `StoreWriter`, and `save_coefficients`.

JSONL stays the default and defines the record: a Parquet directory holds the same records, so a save_dir converted to Parquet
and back gives its lines byte for byte, and every table and count of a report is bitwise the same from either.
"""

import json
import os
import shutil
import subprocess
import sys
import re
import textwrap
import time
import warnings
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epftoolbox2 import cli
from epftoolbox2.ensembles import AverageEnsemble
from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator, rMAEEvaluator
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import NaiveModel, OLSModel, Use
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline, Workflow
from epftoolbox2.results import parquet_store
from epftoolbox2.results.parquet_store import CATALOG, ParquetResultStore, compact, convert, read_catalog, verify
from epftoolbox2.results.report import EvaluationReport
from epftoolbox2.results.store import (
    ResultsFormatError,
    ResultsLockedError,
    ResultStore,
    StoreWriter,
    iter_file_records,
    open_store,
    read_records,
)
from tests.synthetic import PREDICTORS, make_market_data


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


def lines(path):
    return sorted(line for line in Path(path).read_text(encoding="utf-8").splitlines() if line)


def every_line(save_dir):
    return {p.name: lines(p) for p in sorted(Path(save_dir).glob("*.jsonl"))}


def manifests(save_dir):
    return {p.name: p.read_bytes() for p in sorted(Path(save_dir).glob("*.manifest.json"))}


def bits(frame):
    """The float columns' bits, so a table is compared to the last bit (NaN included)."""
    return {
        c: frame[c].to_numpy(dtype=np.float64).view(np.int64).tolist()
        for c in frame.columns
        if pd.api.types.is_float_dtype(frame[c])
    }


def tables(report):
    out = {f"compute_all.{name}": table for name, table in report.compute_all().items()}
    out["forecast_counts"] = report.forecast_counts()
    out["by_quarter"] = report.by_quarter()
    out["by_day"] = report.by_day()
    out["details"] = pd.concat([df.assign(model=name) for name, df in report.iter_details()], ignore_index=True)
    out["predictions"] = report.predictions()
    return out


def assert_same_tables(a, b):
    assert a.keys() == b.keys()
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name], check_exact=True, obj=name)
        assert bits(a[name]) == bits(b[name]), name
        assert a[name].attrs == b[name].attrs, name


# --- scenarios: every kind of record a model writes ------------------------------------------------------------------------


def matching_list(ctx):
    """Weekend targets train on weekend days alone (two predictors), the others on four."""
    names = [f"load_d+{ctx.horizon}", f"wind_d+{ctx.horizon}"]
    if ctx.target_day.weekday() >= 5:
        return Use(names, train_on="matching")
    return names + [f"is_weekend_d+{ctx.horizon}", f"is_weekday_d+{ctx.horizon}"]


def holes(data, day, hours=(3, 4), drop_hour=7, column="load_d+1"):
    """A predictor missing on issue rows (`missing_test_predictors`) and a row gone (`missing_test_row`)."""
    data = data.copy()
    issue = (data.index.date == pd.Timestamp(day).date()) & np.isin(data.index.hour, list(hours))
    data.loc[issue, column] = np.nan
    gone = data.index[(data.index.date == pd.Timestamp(day).date()) & (data.index.hour == drop_hour)]
    return data.drop(index=gone)


WINDOWS = dict(validation=("2023-03-20", "2023-03-23"), test=("2023-03-25", "2023-03-28"))  # over the spring switch


def windowed_pipeline():
    return (
        ModelPipeline()
        .add_model(OLSModel(PREDICTORS, 30, name="ols"))
        .add_model(OLSModel(matching_list, 30, name="matching"))
        .add_model(OLSModel(PREDICTORS, [14, 28], refit="7D", name="refit_list"))
        .add_model(OLSModel(PREDICTORS, [3, 28], name="members"))
        .add_model(NaiveModel(["price_d-1"], name="naive"))
        .add_ensemble(AverageEnsemble(["ols", "refit_list"], name="average"))
        .add_ensemble(AverageEnsemble(["ols", "matching"], top=1, name="top1"))
        .add_evaluator(MAEEvaluator())
        .add_evaluator(RMSEEvaluator())
        .add_evaluator(rMAEEvaluator("naive"))
    )


def windows_data():
    data = make_market_data(days=100, noise=5.0, seed=5, horizon=2)
    data = holes(data, "2023-03-27")
    data = holes(data, "2023-03-22", hours=(10,), drop_hour=12, column="wind_d+2")
    days = pd.Index(data.index.date)
    # Load missing on the rows of the 3-day member's window of 2023-03-28 at 05:00: a member skip with a detail.
    short = (days >= pd.Timestamp("2023-03-24").date()) & (days <= pd.Timestamp("2023-03-27").date())
    data.loc[short & (data.index.hour == 5), "load_d+1"] = np.nan
    gap = (days == pd.Timestamp("2023-03-28").date()) & (data.index.hour == 15)
    data.loc[gap, "price"] = np.nan  # a NaN actual
    return data


PUBLISHED_INFO = InformationSet("11:45", "Europe/Brussels", rules={"price": Availability.published("13:00", 1)})


def published_pipeline():
    return (
        ModelPipeline(information_set=PUBLISHED_INFO)
        .add_model(OLSModel(PREDICTORS + ["price_d-1"], 30, name="published", newest_target="published"))
        .add_model(OLSModel(PREDICTORS, 30, name="plain"))
        .add_evaluator(MAEEvaluator())
        .add_evaluator(rMAEEvaluator("plain"))
    )


def quarter_pipeline():
    return (
        ModelPipeline()
        .add_model(OLSModel(PREDICTORS, 20, name="ols"))
        .add_model(NaiveModel(["price_d-1"], name="naive"))
        .add_evaluator(MAEEvaluator())
        .add_evaluator(rMAEEvaluator("naive"))
    )


def daily_frame(first="2024-02-19", days=60):
    zone = "Europe/Warsaw"
    dates = pd.date_range(first, periods=days, freq="D")
    index = (dates + pd.Timedelta(hours=12)).tz_localize(zone)
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {"actual_daily_max": 20000 + 50 * np.arange(days, dtype=float) + rng.normal(0, 30, days)}, index=index
    )
    for k in range(1, 8):
        frame[f"load_forecast_daily_max_d+{k}"] = 1000.0 * k + np.arange(days) + rng.normal(0, 5, days)
    frame.iloc[40, frame.columns.get_loc("load_forecast_daily_max_d+3")] = np.nan
    return frame


DAILY_INFO = InformationSet("12:00", "Europe/Warsaw", rules={"actual_daily_max": Availability.realised("1h")})


def daily_pipeline():
    return (
        ModelPipeline(information_set=DAILY_INFO)
        .add_model(NaiveModel(["load_forecast_daily_max_d+{horizon}"], name="tso"))
        .add_model(OLSModel(["load_forecast_daily_max_d+{horizon}"], 14, name="ols"))
        .add_evaluator(MAEEvaluator())
        .add_evaluator(rMAEEvaluator("tso"))
    )


# name -> (pipeline, run arguments, report arguments)
SCENARIOS = {
    "windows": (windowed_pipeline, dict(horizons=[1, 2], **WINDOWS), dict(horizons=[1, 2], **WINDOWS)),
    "fridays": (
        published_pipeline,
        dict(test_start="2023-03-10", test_end="2023-03-31", horizon=2, issue_days=["friday"]),
        dict(test_start="2023-03-10", test_end="2023-03-31", horizon=2),
    ),
    "quarter": (
        quarter_pipeline,
        dict(test_start="2023-03-25", test_end="2023-03-26", horizon=1, freq="15min"),
        dict(test_start="2023-03-25", test_end="2023-03-26", horizon=1),
    ),
    "daily": (
        daily_pipeline,
        dict(
            test_start="2024-03-18",
            test_end="2024-04-12",
            target="actual_daily_max",
            horizon=7,
            freq="1D",
            issue_days=["monday", "friday"],
        ),
        dict(test_start="2024-03-18", test_end="2024-04-12", horizon=7),
    ),
}


def scenario_data(name):
    if name == "windows":
        return windows_data()
    if name == "fridays":
        return make_market_data(days=95, noise=5.0, seed=3, horizon=2)
    if name == "quarter":
        data = make_market_data(start="2023-02-20", days=36, noise=5.0, seed=6, horizon=1, freq="15min")
        issue = (data.index.date == pd.Timestamp("2023-03-25").date()) & (data.index.hour == 5)
        data.loc[issue & (data.index.minute == 15), "load_d+1"] = np.nan
        return data.drop(index=data.index[issue & (data.index.minute == 45)])
    return daily_frame()


def revision_two(save_dir):
    """Hand-made lines in `ols.jsonl`: a revision-2 forecast of a key (read in place of its revision-1 one), a revision-2 skip
    record of another (which leaves that key without a forecast), and an identical repeat of a forecast (read once, warned)."""
    path = Path(save_dir) / "ols.jsonl"
    stored = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    ok = [r for r in stored if r.get("status") == "ok"]
    newer = dict(ok[0], prediction=ok[0]["prediction"] + 1.0, revision=2)
    skipped = {
        **ok[1],
        "prediction": None,
        "coefficients": [],
        "status": "fit_error",
        "revision": 2,
        "detail": "by hand",
    }
    with open(path, "a", encoding="utf-8") as f:
        for record in (newer, skipped, ok[2]):
            f.write(json.dumps(record) + "\n")


@pytest.fixture(scope="module")
def jsonl_runs(tmp_path_factory):
    """Each scenario run into a JSONL save_dir, with the reports of `run` and of `report` over the files."""
    root = tmp_path_factory.mktemp("jsonl_runs")
    out = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MAX_PROCESSES", "1")
        patch.setenv("THREADS_PER_PROCESS", "1")
        patch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)
        for name, (pipeline, run_args, _) in SCENARIOS.items():
            save_dir = root / name
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline().run(scenario_data(name), save_dir=str(save_dir), **run_args)
            out[name] = save_dir
        revision_two(out["windows"])
    return out


def report_of(name, save_dir):
    pipeline, _, report_args = SCENARIOS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = pipeline().report(str(save_dir), **report_args)
    return report


def all_tables(report):
    out = {f"test.{k}": v for k, v in tables(report).items()}
    if report.windows is not None:
        out.update({f"validation.{k}": v for k, v in tables(report.validation).items()})
    out["choices"] = report.choices()
    return out


def twin(save_dir, tmp_path, name):
    copy = tmp_path / name
    shutil.copytree(save_dir, copy)
    return copy


# --- round trip ------------------------------------------------------------------------------------------------------------


def test_the_scenarios_hold_every_record_kind(jsonl_runs):
    """What the round trips below cover: window lists, functions, matching lists, refit, validation windows, `"published"`,
    Fridays, 1D, 15-min, skip records with and without a detail, revision 2, NaN actuals."""
    keys, statuses, revisions, freqs, nan_actual = set(), set(), set(), set(), False
    for save_dir in jsonl_runs.values():
        for path in Path(save_dir).glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                keys |= set(record)
                statuses.add(record.get("status"))
                revisions.add(record.get("revision"))
                freqs.add(record.get("freq"))
                nan_actual |= isinstance(record.get("actual"), float) and np.isnan(record["actual"])
    assert {
        "member_predictions",
        "predictor_list",
        "training_days",
        "matching_days",
        "window",
        "newest_target_day",
        "refit_anchor",
        "detail",
    } <= keys
    assert {"ok", "missing_test_predictors", "missing_test_row", "no_finite_training_rows", "not_published"} <= statuses
    assert {1, 2} <= revisions and {"1h", "15min", "1D"} <= freqs and nan_actual
    refit = [json.loads(line) for line in (jsonl_runs["windows"] / "refit_list.jsonl").read_text().splitlines()]
    forecast = next(r for r in refit if r["status"] == "ok")
    assert list(forecast).index("refit_anchor") == list(forecast).index("freq") + 1  # where a refit record has it
    skip = next(r for r in refit if r["status"] != "ok")
    assert list(skip).index("refit_anchor") == list(skip).index("list_hash") + 1  # where any other record has it


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_jsonl_to_parquet_and_back_gives_the_lines_byte_for_byte(jsonl_runs, tmp_path, name):
    original = jsonl_runs[name]
    copy = twin(original, tmp_path, name)
    before = every_line(original)
    models = [
        p.stem for p in copy.glob("*.jsonl") if "kind" not in json.loads((copy / f"{p.stem}.manifest.json").read_text())
    ]

    converted = dict(convert(copy, "parquet"))
    assert sorted(converted) == sorted(models)
    for stem in models:
        assert not (copy / f"{stem}.jsonl").exists() and (copy / f"{stem}.parquet").is_dir()
        assert json.loads((copy / f"{stem}.manifest.json").read_text())["save_format"] == "parquet"
        # Every record, as a dict and as JSON text (floats as their repr, so their bits), in the order of the original's lines.
        stored = [json.dumps(r) for r in iter_file_records(copy / f"{stem}.parquet")]
        assert sorted(stored) == before[f"{stem}.jsonl"]
        assert len(stored) == converted[stem]

    convert(copy, "jsonl")
    assert every_line(copy) == before
    assert manifests(copy) == manifests(original)  # the format key gone again, every other byte as it was
    assert not list(copy.glob("*.parquet"))


def test_parquet_records_equal_the_jsonl_dicts_and_float_bits(jsonl_runs, tmp_path):
    copy = twin(jsonl_runs["windows"], tmp_path, "windows")
    convert(copy, "parquet")
    for stem in ("ols", "members", "refit_list", "matching", "naive"):
        jsonl = [json.loads(line) for line in (jsonl_runs["windows"] / f"{stem}.jsonl").read_text().splitlines()]
        parquet = list(iter_file_records(copy / f"{stem}.parquet"))
        key = lambda r: json.dumps(r)  # noqa: E731
        for a, b in zip(sorted(jsonl, key=key), sorted(parquet, key=key)):
            assert list(a) == list(b)  # the keys in their order
            assert a == b or json.dumps(a) == json.dumps(b)  # NaN != NaN as a dict, not as text
            for field in ("prediction", "actual"):
                if isinstance(a[field], float):
                    assert np.float64(a[field]).view(np.int64) == np.float64(b[field]).view(np.int64)
            assert [np.float64(c).view(np.int64) for c in a["coefficients"]] == [
                np.float64(c).view(np.int64) for c in b["coefficients"]
            ]


# --- reports from either format --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_every_table_and_count_is_bitwise_from_the_parquet_twin(jsonl_runs, tmp_path, name):
    original = twin(jsonl_runs[name], tmp_path, f"{name}_jsonl")
    copy = twin(jsonl_runs[name], tmp_path, f"{name}_parquet")
    convert(copy, "parquet")
    from_jsonl = all_tables(report_of(name, original))
    from_parquet = all_tables(report_of(name, copy))
    assert_same_tables(from_jsonl, from_parquet)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_the_frame_path_equals_the_dict_path(jsonl_runs, tmp_path, name, monkeypatch):
    copy = twin(jsonl_runs[name], tmp_path, name)
    convert(copy, "parquet")
    report = report_of(name, copy)
    for part in [report, *([report.validation] if report.windows is not None else [])]:
        for model, ref in part.refs.items():
            if ref.path is None or ref.path.suffix != ".parquet":
                continue
            fast = part._model_frame(ref)
            with monkeypatch.context() as patch:
                patch.setattr(type(ref), "frame", lambda self, columns: None)
                slow = part._model_frame(ref)
            assert list(fast.dtypes) == list(slow.dtypes), model
            pd.testing.assert_frame_equal(fast, slow, check_exact=True, obj=model)
            assert bits(fast) == bits(slow)
            assert ref.count_records() == sum(1 for _ in ref.iter_records(cols=["hour"]))


def test_a_report_s_cache_reads_a_parquet_directory_again_when_its_catalog_changes(tmp_path):
    path = tmp_path / "m.parquet"
    store = ParquetResultStore(path, test_start="2023-04-12")
    store.save(record(hour=0))
    store.flush()
    from epftoolbox2.results.ref import ModelResultRef

    ref = ModelResultRef(name="m", path=path, count=2, test_start="2023-04-12", test_end="2023-04-12", horizon=1)
    report = EvaluationReport({"m": ref}, [MAEEvaluator()])
    assert len(report._model_frames()["m"]) == 1
    store.save(record(hour=1))
    store.close()
    assert len(report._model_frames()["m"]) == 2


# --- the readers' rules on a directory -------------------------------------------------------------------------------------


def record(hour=3, horizon=1, run_date="2023-04-12", prediction=72.5, status="ok", revision=1, **extra):
    r = {
        "run_date": run_date,
        "target_date": (pd.Timestamp(run_date) + pd.Timedelta(days=horizon)).date().isoformat(),
        "run_weekday": pd.Timestamp(run_date).weekday(),
        "hour": hour,
        "horizon": horizon,
        "day_in_test": 0,
        "prediction": None if status != "ok" else prediction,
        "actual": 70.0,
        "coefficients": [] if status != "ok" else [0.5, -0.25],
        "freq": "1h",
        "status": status,
        "list_hash": "abcdef012345",
        "refit_anchor": run_date,
        "revision": revision,
    }
    r.update(extra)
    return r


def write_parts(path, *parts):
    """Each list of records as its own committed parts (one a year), in order, as they are: not through a store, which would
    write a record at its key's highest revision."""
    for number, records in enumerate(parts):
        by_year = {}
        for r in records:
            by_year.setdefault(r["run_date"][:4], []).append(r)
        for year, chunk in by_year.items():
            entry = parquet_store._write_part(Path(path), year, chunk, f"part-test-{number:03d}-{year}.parquet")
            parquet_store._append_entry(Path(path), entry)


def both_formats(tmp_path, *parts):
    """The same records as a JSONL file and as a directory of parts."""
    jsonl = tmp_path / "m.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for records in parts:
            for r in records:
                f.write(json.dumps(r) + "\n")
    write_parts(tmp_path / "p.parquet", *parts)
    return jsonl, tmp_path / "p.parquet"


def read_both(jsonl, directory, **kwargs):
    with warnings.catch_warnings(record=True) as caught_jsonl:
        warnings.simplefilter("always")
        a = list(read_records(jsonl, **kwargs))
    with warnings.catch_warnings(record=True) as caught_parquet:
        warnings.simplefilter("always")
        b = list(read_records(directory, **kwargs))
    return a, b, len(caught_jsonl), len(caught_parquet)


def test_the_highest_revision_wins_wherever_it_stands(tmp_path):
    jsonl, directory = both_formats(
        tmp_path,
        [record(prediction=1.0, revision=2), record(hour=4)],
        [record(prediction=2.0, revision=1), record(hour=4, prediction=3.0, revision=3)],
    )
    a, b, _, _ = read_both(jsonl, directory)
    assert sorted(map(json.dumps, a)) == sorted(map(json.dumps, b))
    assert {(r["hour"], r["prediction"]) for r in b} == {(3, 1.0), (4, 3.0)}


def test_a_forecast_beats_skip_records_and_skip_records_alone_are_not_read(tmp_path):
    jsonl, directory = both_formats(
        tmp_path,
        [record(status="too_few_rows"), record(hour=5, status="too_few_rows")],
        [record(prediction=9.0), record(hour=5, status="fit_error")],
    )
    a, b, _, _ = read_both(jsonl, directory)
    assert a == b == [record(prediction=9.0)]
    a, b, _, _ = read_both(jsonl, directory, statuses=None)
    assert sorted(map(json.dumps, a)) == sorted(map(json.dumps, b))
    assert {r["hour"]: r["status"] for r in b} == {3: "ok", 5: "fit_error"}  # of skip records alone, the last
    store = ParquetResultStore(directory, test_start="2023-04-12")
    assert store.is_done(3, 1, 0) and not store.is_done(5, 1, 0)
    store.close()


def test_a_repeat_with_the_same_forecast_warns_once_and_is_read_once(tmp_path):
    old_format = {k: v for k, v in record().items() if k not in ("status", "list_hash", "refit_anchor", "revision")}
    jsonl, directory = both_formats(tmp_path, [record(), record(hour=4)], [old_format, record()])
    a, b, warned_jsonl, warned_parquet = read_both(jsonl, directory)
    assert a == b == [record(), record(hour=4)]
    assert warned_jsonl == warned_parquet == 1


def test_a_repeat_with_another_forecast_raises_naming_the_directory_and_a_key(tmp_path):
    jsonl, directory = both_formats(tmp_path, [record()], [record(prediction=1.5)])
    for path in (jsonl, directory):
        with pytest.raises(ValueError, match=r"different forecasts .*period 3, horizon 1, run date 2023-04-12"):
            list(read_records(path))
    with pytest.warns(UserWarning, match="another forecast, the first read"):
        assert [r["prediction"] for r in read_records(directory, duplicates="first")] == [72.5]


def test_uncommitted_parts_are_not_read(tmp_path):
    directory = tmp_path / "m.parquet"
    write_parts(directory, [record()])
    orphan = directory / "year=2023" / "part-orphan-000001.parquet"
    shutil.copy(next(directory.glob("year=2023/part-*.parquet")), orphan)
    assert len(list(read_records(directory))) == 1  # the copy is listed nowhere
    assert any("orphan" in str(f) for f in verify(tmp_path))


def test_a_half_written_catalog_line_is_skipped_with_a_warning_and_cut_by_the_next_writer(tmp_path):
    directory = tmp_path / "m.parquet"
    write_parts(directory, [record()], [record(hour=4)])
    catalog = directory / CATALOG
    content = catalog.read_bytes()
    catalog.write_bytes(content[: len(content) - 30])  # the second line torn
    with pytest.warns(UserWarning, match="not a whole catalog entry"):
        assert [r["hour"] for r in read_records(directory)] == [3]
    with pytest.warns(UserWarning, match="cut off its last"):
        store = ParquetResultStore(directory, test_start="2023-04-12", lock=True)
    assert not store.is_done(4, 1, 0)
    store.close()
    assert catalog.read_bytes().endswith(b"\n")


def test_a_damaged_catalog_line_before_the_last_raises_with_its_number(tmp_path):
    directory = tmp_path / "m.parquet"
    write_parts(directory, [record()], [record(hour=4)])
    catalog = directory / CATALOG
    first, second = catalog.read_bytes().splitlines(keepends=True)
    catalog.write_bytes(first.replace(b'"rows":1', b'"rows":2') + second)  # the CRC no longer matches
    with pytest.raises(ValueError, match="line 1 is not a catalog entry"):
        list(read_records(directory))
    assert [f.level for f in verify(tmp_path)] == ["error"]


def test_verify_finds_a_changed_part_and_reports_nothing_on_clean_results(tmp_path):
    directory = tmp_path / "m.parquet"
    write_parts(directory, [record(), record(hour=4)])
    assert verify(tmp_path) == []
    part = next(directory.glob("year=*/part-*.parquet"))
    data = bytearray(part.read_bytes())
    data[len(data) // 2] ^= 0xFF
    part.write_bytes(bytes(data))
    found = verify(tmp_path)
    assert [f.level for f in found] == ["error"] and "bytes aren't those" in found[0].message


def test_verify_reports_repeated_keys(tmp_path):
    write_parts(tmp_path / "a.parquet", [record()], [record()], [record(hour=4, revision=2)], [record(hour=4)])
    write_parts(tmp_path / "b.parquet", [record()], [record(prediction=0.0)])
    found = {(f.model, f.level, f.message.split(": ", 1)[1].split(" ")[0]) for f in verify(tmp_path)}
    assert found == {("a", "note", "1"), ("b", "error", "1")}  # a: one identical repeat, one record of a lower revision
    assert {f.level for f in verify(tmp_path, model="a")} == {"note"}


# --- the store: resume, crash, compact, formats -----------------------------------------------------------------------------


def market():
    return make_market_data(days=120, noise=5.0, seed=4, horizon=2)


def plain_pipeline():
    return ModelPipeline().add_model(OLSModel(PREDICTORS, 45, name="ols")).add_evaluator(MAEEvaluator())


def run(save_dir, start="2023-04-10", end="2023-04-12", data=None, **kwargs):
    data = market() if data is None else data
    return plain_pipeline().run(data, test_start=start, test_end=end, horizon=2, save_dir=str(save_dir), **kwargs)


def stored_lines(path):
    return sorted(json.dumps(r) for r in iter_file_records(path))


def test_a_run_writes_parquet_and_reports_what_jsonl_reports(tmp_path):
    jsonl = run(tmp_path / "j")
    parquet = run(tmp_path / "p", save_format="parquet")
    assert not (tmp_path / "p" / "ols.jsonl").exists()
    assert [p.name for p in sorted((tmp_path / "p").iterdir())] == ["ols.lock", "ols.manifest.json", "ols.parquet"]
    assert stored_lines(tmp_path / "p" / "ols.parquet") == lines(tmp_path / "j" / "ols.jsonl")
    assert_same_tables(tables(jsonl), tables(parquet))
    manifest = json.loads((tmp_path / "p" / "ols.manifest.json").read_text())
    assert manifest["save_format"] == "parquet"
    assert "save_format" not in json.loads((tmp_path / "j" / "ols.manifest.json").read_text())
    # The same configuration otherwise: the manifests differ in provenance alone.
    other = json.loads((tmp_path / "j" / "ols.manifest.json").read_text())
    drop = ("created_at", "updated_at", "save_format")
    assert {k: v for k, v in manifest.items() if k not in drop} == {k: v for k, v in other.items() if k not in drop}


def test_resume_from_parquet_equals_a_fresh_run(tmp_path, monkeypatch):
    monkeypatch.setenv("PARQUET_ROWS_PER_PART", "40")
    fresh = run(tmp_path / "fresh", save_format="parquet")
    shutil.copytree(tmp_path / "fresh", tmp_path / "resumed")
    directory = tmp_path / "resumed" / "ols.parquet"
    catalog = (directory / CATALOG).read_bytes().splitlines(keepends=True)
    assert len(catalog) == 4  # 144 records in parts of 40
    (directory / CATALOG).write_bytes(b"".join(catalog[:2]))  # the last two parts never committed
    for entry in catalog[2:]:
        (directory / json.loads(entry)["file"]).unlink()
    resumed = run(tmp_path / "resumed", save_format="parquet")
    assert stored_lines(directory) == stored_lines(tmp_path / "fresh" / "ols.parquet")
    assert [e["rows"] for e in read_catalog(directory).entries] == [40, 40, 40, 24]  # what was pending, in parts of 40
    assert_same_tables(tables(resumed), tables(fresh))
    again = run(tmp_path / "resumed", save_format="parquet")  # nothing pending: nothing written
    assert len(read_catalog(directory).entries) == 4
    assert_same_tables(tables(again), tables(fresh))


def test_a_resume_into_parquet_writes_what_a_resume_into_jsonl_writes(tmp_path):
    """A run over a later start, then over the whole period: the resumed records keep the `day_in_test` of the run that wrote
    them, in either format."""
    for fmt in ("jsonl", "parquet"):
        run(tmp_path / fmt, start="2023-04-11", save_format=fmt)
        run(tmp_path / fmt, save_format=fmt)
    assert stored_lines(tmp_path / "parquet" / "ols.parquet") == lines(tmp_path / "jsonl" / "ols.jsonl")
    reports = [
        plain_pipeline().report(str(tmp_path / fmt), "2023-04-10", "2023-04-12", horizon=2)
        for fmt in ("jsonl", "parquet")
    ]
    assert_same_tables(tables(reports[0]), tables(reports[1]))


KILLED_RUN = """
import os, sys
os.environ["MAX_PROCESSES"] = "1"
os.environ["THREADS_PER_PROCESS"] = "1"
os.environ["PARQUET_ROWS_PER_PART"] = "20"
sys.path.insert(0, sys.argv[2])
import epftoolbox2.models.base as base
from epftoolbox2.results import parquet_store
from tests.results.test_parquet_store import InProcessPool, run

base.ProcessPoolExecutor = InProcessPool  # no pool worker outliving the kill, holding the pipes open

committed = [0]
original = parquet_store._append_entry


def append(directory, entry):
    committed[0] += 1
    if committed[0] == 3:  # the third part is renamed into place, and the run dies before its catalog line
        os._exit(9)
    original(directory, entry)


parquet_store._append_entry = append
run(sys.argv[1], save_format="parquet")
"""


def test_a_writer_killed_between_a_part_s_rename_and_its_catalog_line_resumes_to_the_same_records(tmp_path):
    root = Path(__file__).resolve().parents[2]
    killed = subprocess.run(
        [sys.executable, "-c", KILLED_RUN, str(tmp_path / "killed"), str(root)], capture_output=True, text=True
    )
    assert killed.returncode == 9, killed.stderr
    directory = tmp_path / "killed" / "ols.parquet"
    assert len(read_catalog(directory).entries) == 2 and len(list(directory.glob("year=*/part-*.parquet"))) == 3
    orphans = [f for f in verify(tmp_path / "killed") if "orphan" in f.message]
    assert len(orphans) == 1 and orphans[0].level == "note"

    resumed = run(tmp_path / "killed", save_format="parquet")
    fresh = run(tmp_path / "fresh", save_format="parquet")
    assert stored_lines(directory) == stored_lines(tmp_path / "fresh" / "ols.parquet")
    assert_same_tables(tables(resumed), tables(fresh))
    # Three live parts (two of the killed run, one of the resume) into one, and the orphan gone.
    assert compact(tmp_path / "killed")["ols"] == {"parts_merged": 3, "parts_written": 1, "files_removed": 4}
    assert verify(tmp_path / "killed") == []
    assert stored_lines(directory) == stored_lines(tmp_path / "fresh" / "ols.parquet")


def test_compact_keeps_every_record_and_is_idempotent(tmp_path):
    directory = tmp_path / "m.parquet"
    write_parts(
        directory,
        [record(), record(hour=4, status="too_few_rows"), record(run_date="2024-01-02")],
        [record(hour=4), record(hour=5, prediction=1.0)],
        [record(), record(hour=5, prediction=2.0, revision=2), record(run_date="2024-01-02", hour=1)],
    )
    every = stored_lines(directory)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the identical repeat of hour 3
        read = [json.dumps(r) for r in read_records(directory, statuses=None)]
    summary = compact(tmp_path)
    assert summary == {"m": {"parts_merged": 5, "parts_written": 2, "files_removed": 5}}
    assert stored_lines(directory) == every
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert sorted(json.dumps(r) for r in read_records(directory, statuses=None)) == sorted(read)
    assert len(list(directory.glob("year=*/part-*.parquet"))) == 2
    catalog, files = (directory / CATALOG).read_bytes(), sorted(directory.rglob("*"))
    assert compact(tmp_path) == {"m": {"parts_merged": 0, "parts_written": 0, "files_removed": 0}}
    assert (directory / CATALOG).read_bytes() == catalog and sorted(directory.rglob("*")) == files
    assert [f.level for f in verify(tmp_path)] == ["note", "note"]  # the repeat and the lower revision, still there


def test_compact_keeps_a_key_s_records_in_their_order(tmp_path):
    """The first forecast of a key is the one read with `duplicates="first"`: a compaction keeps which one that is."""
    directory = tmp_path / "m.parquet"
    write_parts(directory, [record(prediction=1.0)], [record(prediction=2.0)], [record(prediction=3.0)])
    compact(tmp_path)
    with pytest.warns(UserWarning):
        assert [r["prediction"] for r in read_records(directory, duplicates="first")] == [1.0]


@pytest.mark.parametrize("first, second", [("jsonl", "parquet"), ("parquet", "jsonl")])
def test_a_run_in_the_other_format_raises_naming_convert(tmp_path, first, second):
    run(tmp_path, save_format=first)
    with pytest.raises(ResultsFormatError, match=re.escape(f"epf results convert {tmp_path} --to {second}")):
        run(tmp_path, save_format=second)
    with pytest.raises(ResultsFormatError, match="epf results convert"):  # a model run on its own is refused too
        OLSModel(PREDICTORS, 45, name="ols").run(
            market(), "2023-04-10", "2023-04-12", horizon=2, save_to=str(tmp_path / f"ols.{second}")
        )
    convert(tmp_path, second)
    run(tmp_path, save_format=second)  # converted: resumes, nothing pending
    assert json.loads((tmp_path / "ols.manifest.json").read_text()).get("save_format", "jsonl") == second


def test_results_in_both_formats_raise_in_a_report(tmp_path):
    run(tmp_path, save_format="parquet")
    (tmp_path / "ols.jsonl").write_text("")
    with pytest.raises(ResultsFormatError, match="both formats"):
        plain_pipeline().report(str(tmp_path), "2023-04-10", "2023-04-12", horizon=2)


def test_a_second_writer_of_a_parquet_directory_raises_naming_the_lock(tmp_path):
    store = open_store(tmp_path / "ols.parquet", lock=True)
    with pytest.raises(ResultsLockedError, match="ols.lock"):
        open_store(tmp_path / "ols.parquet", lock=True)
    with pytest.raises(ResultsLockedError, match="ols.lock"):
        run(tmp_path, save_format="parquet")
    store.close()
    run(tmp_path, save_format="parquet")


def test_a_record_the_schema_has_no_column_for_raises_when_saved(tmp_path):
    store = ParquetResultStore(tmp_path / "m.parquet")
    with pytest.raises(ValueError, match="'note'"):
        store.save(record(note="x"))
    store.close()
    jsonl = tmp_path / "n.jsonl"
    jsonl.write_text(json.dumps(record(note="x")) + "\n")
    with pytest.raises(ValueError, match="'note'"):
        convert(tmp_path, "parquet")
    assert jsonl.exists() and not (tmp_path / "n.parquet").exists()  # nothing changed


def test_save_coefficients_false_writes_empty_coefficients_and_nothing_else_changes(tmp_path):
    with_them = run(tmp_path / "with")
    without = run(tmp_path / "without", save_coefficients=False)
    parquet = run(tmp_path / "parquet", save_coefficients=False, save_format="parquet")
    a = [json.loads(line) for line in lines(tmp_path / "with" / "ols.jsonl")]
    b = [json.loads(line) for line in lines(tmp_path / "without" / "ols.jsonl")]
    assert any(r["coefficients"] for r in a) and all(r["coefficients"] == [] for r in b)
    assert [dict(r, coefficients=[]) for r in a] == b
    assert [list(r) for r in a] == [list(r) for r in b]
    assert stored_lines(tmp_path / "parquet" / "ols.parquet") == lines(tmp_path / "without" / "ols.jsonl")
    assert_same_tables(tables(with_them), tables(without))
    assert_same_tables(tables(with_them), tables(parquet))
    manifest = lambda d: {  # noqa: E731
        k: v
        for k, v in json.loads((tmp_path / d / "ols.manifest.json").read_text()).items()
        if k not in ("created_at", "updated_at")
    }
    assert manifest("with") == manifest("without")


def test_the_workflow_yaml_keys_round_trip(tmp_path):
    workflow = Workflow(
        data_start="2023-01-01",
        data_end="2023-02-01",
        model_save_dir="results",
        model_save_format="parquet",
        model_save_coefficients=False,
    )
    saved = workflow.to_dict()["model_pipeline"]
    assert (saved["save_format"], saved["save_coefficients"]) == ("parquet", False)
    workflow.save(tmp_path / "w.yaml")
    loaded = Workflow.load(tmp_path / "w.yaml")
    assert (loaded.model_save_format, loaded.model_save_coefficients) == ("parquet", False)
    plain = Workflow(data_start="2023-01-01", data_end="2023-02-01").to_dict()["model_pipeline"]
    assert "save_format" not in plain and "save_coefficients" not in plain
    with pytest.raises(ValueError, match="save_format"):
        Workflow(data_start="2023-01-01", data_end="2023-02-01", model_save_format="csv")


def test_the_cli_converts_compacts_and_verifies(tmp_path, monkeypatch, capsys):
    run(tmp_path)

    def epf(*args):
        monkeypatch.setattr(sys, "argv", ["epf", "results", *args])
        try:
            cli.main()
        except SystemExit as exc:
            return exc.code
        return 0

    original = lines(tmp_path / "ols.jsonl")
    assert epf("convert", str(tmp_path), "--to", "parquet") == 0
    assert "ols: " in capsys.readouterr().out and (tmp_path / "ols.parquet").is_dir()
    assert epf("compact", str(tmp_path)) == 0 and epf("verify", str(tmp_path)) == 0
    assert "Results verified" in capsys.readouterr().out
    assert epf("convert", str(tmp_path), "--to", "jsonl", "--model", "OLS") == 0
    assert lines(tmp_path / "ols.jsonl") == original
    (tmp_path / "ols.jsonl").write_text(
        original[0] + "\n" + json.dumps(dict(json.loads(original[0]), prediction=0.5)) + "\n"
    )
    assert epf("verify", str(tmp_path)) == 1
    assert "different forecasts" in capsys.readouterr().out


# --- StoreWriter ------------------------------------------------------------------------------------------------------------


def open_descriptors():
    return len(os.listdir("/proc/self/fd"))


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="Linux /proc")
def test_ten_thousand_stores_on_one_writer_keep_at_most_64_files_open(tmp_path):
    before = open_descriptors()
    most = 0
    with StoreWriter() as writer:
        stores = []
        for i in range(10_000):
            fmt = "parquet" if i % 2 else "jsonl"
            stores.append(writer.open(tmp_path / f"m{i:05d}.{fmt}", test_start="2023-04-12", lock=True))
            if i % 500 == 0:
                most = max(most, open_descriptors() - before)
        for i, store in enumerate(stores):
            store.save(record(hour=i % 24, prediction=float(i)))
            if i % 500 == 0:
                most = max(most, open_descriptors() - before)
        writer.flush()
        most = max(most, open_descriptors() - before)
        # Every model is held: a writer of its own is refused, from this process and from another one.
        with pytest.raises(ResultsLockedError):
            ResultStore(tmp_path / "m00000.jsonl", lock=True)
        with pytest.raises(ResultsLockedError):
            ParquetResultStore(tmp_path / "m00001.parquet", lock=True)
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                PROBE_LOCK,
                str(tmp_path / "m00002.jsonl"),
                str(Path(__file__).resolve().parents[2]),
            ],
            capture_output=True,
            text=True,
        )
        assert probe.stdout.strip() == "locked", probe.stderr
    assert most <= 64, most
    assert open_descriptors() - before <= 0
    for i in (0, 1, 4_999, 9_998, 9_999):
        fmt = "parquet" if i % 2 else "jsonl"
        assert [r["prediction"] for r in read_records(tmp_path / f"m{i:05d}.{fmt}")] == [float(i)]
    ResultStore(tmp_path / "m00000.jsonl", lock=True).close()  # free once the writer closed
    ParquetResultStore(tmp_path / "m00001.parquet", lock=True).close()


PROBE_LOCK = """
import sys
sys.path.insert(0, sys.argv[2])
from epftoolbox2.results.store import ResultStore, ResultsLockedError
try:
    ResultStore(sys.argv[1], lock=True).close()
    print("free")
except ResultsLockedError:
    print("locked")
"""


def test_a_writer_s_stores_write_what_stores_of_their_own_write(tmp_path):
    records = [record(hour=h, horizon=k, prediction=h + k / 10) for h in range(24) for k in (1, 2)]
    alone_jsonl, alone_parquet = ResultStore(tmp_path / "a.jsonl"), ParquetResultStore(tmp_path / "a.parquet")
    with StoreWriter() as writer:
        shared_jsonl = writer.open(tmp_path / "b.jsonl")
        shared_parquet = writer.open(tmp_path / "b.parquet")
        with pytest.raises(ValueError, match="already open"):
            writer.open(tmp_path / "b.jsonl")
        for r in records:
            for store in (alone_jsonl, alone_parquet, shared_jsonl, shared_parquet):
                store.save(dict(r))
        alone_jsonl.close()
        alone_parquet.close()
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert stored_lines(tmp_path / "a.parquet") == stored_lines(tmp_path / "b.parquet") == lines(tmp_path / "a.jsonl")


def wait_for_parts(directory, n, seconds=10.0):
    deadline = time.monotonic() + seconds
    while len(read_catalog(directory).entries) < n and time.monotonic() < deadline:
        time.sleep(0.01)
    return [e["rows"] for e in read_catalog(directory).entries]


def test_a_parquet_buffer_becomes_a_part_at_its_row_limit_and_its_age(tmp_path, monkeypatch):
    monkeypatch.setenv("PARQUET_ROWS_PER_PART", "10")
    monkeypatch.setenv("PARQUET_FLUSH_SECONDS", "0.5")
    directory = tmp_path / "m.parquet"
    store = ParquetResultStore(directory)
    for hour in range(10):
        store.save(record(hour=hour))
    assert wait_for_parts(directory, 1) == [10]  # full: written at once
    for hour in range(10, 13):
        store.save(record(hour=hour))
    time.sleep(0.1)
    assert wait_for_parts(directory, 1, 0) == [10]  # not yet half a second old
    assert wait_for_parts(directory, 2) == [10, 3]  # then written without a flush
    store.save(record(hour=13))
    store.close()
    assert wait_for_parts(directory, 3, 0) == [10, 3, 1]


# --- import ------------------------------------------------------------------------------------------------------------------


def test_importing_the_package_loads_no_parquet_code():
    """pandas loads `pyarrow` itself; the Parquet reader and writer (`pyarrow.parquet`) and the store's modules load only when
    a Parquet store is used."""
    code = textwrap.dedent(
        """
        import sys
        import epftoolbox2
        from epftoolbox2.pipelines import ModelPipeline, Workflow
        from epftoolbox2.results.report import EvaluationReport
        print(sorted(m for m in sys.modules if m in ("pyarrow.parquet", "pyarrow._parquet",
              "epftoolbox2.results.parquet_store", "epftoolbox2.results.parquet_schema")))
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
    assert parquet_store.limits() == (200_000, 300.0, 1_000_000)
