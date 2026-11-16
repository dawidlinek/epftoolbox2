"""Checks before export in runs (step 8f): `ModelPipeline(checks=ForecastChecks(...))` in backtests, engine off and on, at 1h and
15 minutes, with refit blocks, resumes, windows, zones, the YAML and the manifest; a `FallbackEnsemble` moving past failed days.

Each check is compared with an oracle: the same pipeline run without checks gives every forecast, and a member-day (model,
issue day, horizon) fails exactly when the oracle's own reading of its predictions fails (the periods of a local date from
`pd.date_range` in the zone, each check written out with pandas). A checked run must leave every other forecast as the
unchecked run wrote it, line for line, and give each forecast of a failing day a `check_failed` skip record with the check in
`detail`, counted in `forecast_counts()` and the manifest.
"""

import datetime
import hashlib
import warnings
import json
import math
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from epftoolbox2.ensembles import FallbackEnsemble
from epftoolbox2.models import NaiveModel, OLSModel, _engine
from epftoolbox2.models import _engine_worker
from epftoolbox2.models._worker import _worker_init, current_fit
from epftoolbox2.pipelines import ModelPipeline, Workflow
from epftoolbox2.results import ManifestMismatchError
from epftoolbox2.results.checks import CHECK_FAILED, ForecastChecks
from epftoolbox2.results import manifest as manifest_module
from epftoolbox2.results.manifest import configuration_digest, manifest_path
from epftoolbox2.results.store import read_records
from tests.synthetic import PREDICTORS, calendar_shift, make_market_data

ZONE = "Europe/Warsaw"
TEST = ("2023-03-21", "2023-03-28")  # the spring switch, Sunday 26 March, inside
KEY = ("hour", "horizon", "run_date")
NAN_DAY = "2023-03-22"  # NanOLS forecasts period 5 of this issue day as NaN


class InProcessPool:
    """A model's pool in the test's own process: the records are what a worker returns."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        initializer(*initargs)

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
    monkeypatch.setattr(_engine, "ProcessPoolExecutor", InProcessPool)
    assert _worker_init and _engine_worker._engine_init  # the initialisers the pools call
    yield


class NanOLS(OLSModel):
    """OLS whose forecast of period 5 issued on `NAN_DAY` is NaN: what `finite` is for (no core model gives one)."""

    def _fit_predict(self, train_x, train_y, test_x):
        prediction, coefficients = super()._fit_predict(train_x, train_y, test_x)
        unit = current_fit().unit
        if unit.anchor.isoformat() == NAN_DAY and unit.period == 5:
            return math.nan, coefficients
        return prediction, coefficients


def frame(freq="1h", start="2023-02-20", days=45):
    """Synthetic data with a day for each check: `flat` (a smooth curve) one value all of 23 March and with a jump on 25 March,
    `wind_d+1` 30 times its value on the rows of 24 March (an OLS forecast of that day out of range), and `load_d+1` missing at
    09:00 of 27 March (that forecast skipped, so its day lacks a period)."""
    data = make_market_data(start=start, days=days, freq=freq, noise=1.0, seed=5, horizon=2)
    index = data.index
    hours = index.hour + index.minute / 60
    flat = pd.Series(100 + 10 * np.sin(2 * np.pi * hours / 24), index=index)
    local = pd.Series(index.tz_localize(None).normalize(), index=index)
    flat[local == pd.Timestamp("2023-03-23")] = 100.0
    flat[(local == pd.Timestamp("2023-03-25")) & (hours >= 12)] += 50.0
    data["flat"] = flat
    for horizon in (1, 2):
        data[f"flat_d+{horizon}"] = calendar_shift(flat, horizon)
    data.loc[local == pd.Timestamp("2023-03-24"), "wind_d+1"] *= 30
    data.loc[(local == pd.Timestamp("2023-03-27")) & (hours == 9), "load_d+1"] = np.nan
    return data


def models(freq):
    return [
        OLSModel(PREDICTORS, training_window=21, name="ols"),
        OLSModel(PREDICTORS, training_window=21, refit="7D", name="ols_7d"),
        NaiveModel(["flat_d+{horizon}"], name="naive"),
    ]


def pipeline(freq, engine, checks, extra=()):
    mp = ModelPipeline(engine=engine, checks=checks)
    for model in [*models(freq), *extra]:
        mp.add_model(model)
    return mp


def lines(path):
    return sorted(Path(path).read_text().splitlines())


def records(path, statuses=None):
    return {tuple(r[k] for k in KEY): r for r in read_records(Path(path), statuses=statuses)}


def plus(day, days):
    return (datetime.date.fromisoformat(day) + datetime.timedelta(days=days)).isoformat()


def wall_periods(day, freq):
    start, end = pd.Timestamp(day).tz_localize(ZONE), pd.Timestamp(plus(day, 1)).tz_localize(ZONE)
    stamps = pd.date_range(start, end, freq="h" if freq == "1h" else "15min", inclusive="left")
    per_hour = 1 if freq == "1h" else 4
    return set((stamps.hour * per_hour + stamps.minute // 15 * (per_hour > 1)).tolist())


def failing_days(unchecked, freq, settings):
    """The oracle: (run_date, horizon) -> the checks it fails, from the forecasts a run without checks wrote."""
    days = {}
    for (hour, horizon, run_date), record in unchecked.items():
        if record.get("status", "ok") == "ok":
            days.setdefault((run_date, horizon), {})[hour] = record["prediction"]
    out = {}
    for (run_date, horizon), values in days.items():
        planned = sorted(wall_periods(run_date, freq) & wall_periods(plus(run_date, horizon), freq))
        series = pd.Series({p: (np.nan if values.get(p) is None else values[p]) for p in planned if p in values})
        finite = series[np.isfinite(series)]
        failed = []
        if settings.get("periods", True) and len(series) < len(planned):
            failed.append("periods")
        if settings.get("finite", True) and len(finite) < len(series):
            failed.append("finite")
        low, high = settings.get("range") or (-np.inf, np.inf)
        if ((finite < low) | (finite > high)).any():
            failed.append("range")
        if settings.get("not_constant", True) and len(finite) >= 2 and finite.nunique() == 1:
            failed.append("not_constant")
        step = settings.get("max_step")
        if step is not None:
            pairs = [(a, b) for a, b in zip(planned, planned[1:]) if a in finite.index and b in finite.index]
            if any(abs(finite[b] - finite[a]) > step for a, b in pairs):
                failed.append("max_step")
        if failed:
            out[(run_date, horizon)] = failed
    return out


# Every check alone, and all together; each with engine off and on, and both frequencies over the cells (every pair of levels).
OFF_ALL = {"periods": False, "finite": False, "not_constant": False}
CELLS = [
    ("periods", {**OFF_ALL, "periods": True}, "off", "1h"),
    ("periods", {**OFF_ALL, "periods": True}, "on", "15min"),
    ("finite", {**OFF_ALL, "finite": True}, "off", "15min"),
    ("finite", {**OFF_ALL, "finite": True}, "on", "1h"),
    ("range", {**OFF_ALL, "range": (0.0, 400.0)}, "off", "1h"),
    ("range", {**OFF_ALL, "range": (0.0, 400.0)}, "on", "15min"),
    ("not_constant", {**OFF_ALL, "not_constant": True}, "off", "15min"),
    ("not_constant", {**OFF_ALL, "not_constant": True}, "on", "1h"),
    ("max_step", {**OFF_ALL, "max_step": 30.0}, "off", "1h"),
    ("max_step", {**OFF_ALL, "max_step": 30.0}, "on", "15min"),
    ("all", {"range": (0.0, 400.0), "max_step": 30.0}, "on", "1h"),
    ("all", {"range": (0.0, 400.0), "max_step": 30.0}, "off", "15min"),
]


@pytest.mark.parametrize("name, settings, engine, freq", CELLS)
def test_each_check_turns_exactly_the_failing_member_days(tmp_path, name, settings, engine, freq):
    data = frame(freq)
    extra = [NanOLS(PREDICTORS, training_window=21, name="nan_ols")] if name in ("finite", "all") else []
    unchecked = pipeline(freq, engine, None, extra).run(
        data, *TEST, horizon=2, freq=freq, save_dir=str(tmp_path / "plain")
    )
    checks = ForecastChecks(**settings)
    report = pipeline(freq, engine, checks, extra).run(
        data, *TEST, horizon=2, freq=freq, save_dir=str(tmp_path / "checked")
    )
    counts = report.forecast_counts().set_index("model")
    plain_counts = unchecked.forecast_counts().set_index("model")
    failed_somewhere = set()
    for model in counts.index:
        before = records(tmp_path / "plain" / f"{model}.jsonl")
        after = records(tmp_path / "checked" / f"{model}.jsonl")
        assert set(after) == set(before)
        failing = failing_days(before, freq, settings)
        failed_somewhere.update(check for found in failing.values() for check in found)
        turned = 0
        for key, record in before.items():
            hour, horizon, run_date = key
            if record["status"] == "ok" and (run_date, horizon) in failing:
                got = after[key]
                assert got["status"] == CHECK_FAILED and got["prediction"] is None, key
                assert [part.split(":")[0] for part in got["detail"].split("; ")] == failing[(run_date, horizon)]
                assert got["actual"] == record["actual"] and got["revision"] == 1
                turned += 1
            else:
                assert after[key] == record, key  # what passed, and every other skip, as the unchecked run wrote it
        assert counts.loc[model, "produced"] == plain_counts.loc[model, "produced"] - turned
        assert counts.loc[model].get(CHECK_FAILED, 0) == turned
        skipped = json.loads(manifest_path(tmp_path / "checked" / f"{model}.jsonl").read_text()).get("skipped", {})
        assert skipped.get("test", {}).get(CHECK_FAILED, 0) == turned
        assert counts.loc[model, "planned"] == plain_counts.loc[model, "planned"]
    expected = {name} if name != "all" else {"periods", "finite", "range", "not_constant", "max_step"}
    assert failed_somewhere == expected  # each check fails some day here, and with the others off nothing else does


def test_checks_off_change_nothing(tmp_path):
    data = frame()
    for engine in ("off", "on"):
        default = ModelPipeline(engine=engine)
        explicit = ModelPipeline(engine=engine, checks=None)
        for mp in (default, explicit):
            for model in models("1h"):
                mp.add_model(model)
        default.run(data, *TEST, horizon=2, save_dir=str(tmp_path / f"default_{engine}"))
        explicit.run(data, *TEST, horizon=2, save_dir=str(tmp_path / f"explicit_{engine}"))
        for model in ("ols", "ols_7d", "naive"):
            assert lines(tmp_path / f"default_{engine}" / f"{model}.jsonl") == lines(
                tmp_path / f"explicit_{engine}" / f"{model}.jsonl"
            )
            manifest = json.loads(manifest_path(tmp_path / f"explicit_{engine}" / f"{model}.jsonl").read_text())
            assert "checks" not in manifest
        assert "checks" not in explicit.to_dict()


def test_the_manifest_records_the_checks_and_a_resume_with_others_raises(tmp_path):
    data = frame()
    checks = ForecastChecks(range=(0.0, 400.0))
    mp = ModelPipeline(checks=checks).add_model(OLSModel(PREDICTORS, training_window=21, name="ols"))
    mp.run(data, *TEST, horizon=2, save_dir=str(tmp_path))
    manifest = json.loads(manifest_path(tmp_path / "ols.jsonl").read_text())
    assert manifest["checks"] == checks.to_dict()
    without = {key: value for key, value in manifest.items() if key != "checks"}
    assert configuration_digest(without) != configuration_digest(manifest)
    assert configuration_digest({**without, "checks": None}) == configuration_digest(without)
    # Without checks a digest is what it was before the field: the hash of the other compared fields alone.
    fields = [name for name in manifest_module._COMPARED_FIELDS if name != "checks"]
    kept = {name: without.get(name) for name in fields if name != "history_ahead" or without.get(name) is not None}
    before = hashlib.sha256(json.dumps(kept, sort_keys=True, default=str).encode()).hexdigest()
    assert configuration_digest(without) == before
    mp.run(data, *TEST, horizon=2, save_dir=str(tmp_path))  # the same checks resume
    for other in (ForecastChecks(range=(0.0, 300.0)), None):
        again = ModelPipeline(checks=other).add_model(OLSModel(PREDICTORS, training_window=21, name="ols"))
        with pytest.raises(ManifestMismatchError, match="checks"):
            again.run(data, *TEST, horizon=2, save_dir=str(tmp_path))


def test_a_resume_checks_a_day_with_its_forecasts_on_file_and_overrules_them(tmp_path):
    """A day half on file (a crash) with a forecast on file out of range: the resume runs the other half, checks the whole day,
    writes `check_failed` for the new half and over the old half one revision up. The next resume runs the day again, which
    now passes, and its forecasts are read again."""
    data = frame()
    checks = ForecastChecks(range=(0.0, 400.0))

    def run():
        mp = ModelPipeline(checks=checks).add_model(OLSModel(PREDICTORS, training_window=21, name="ols"))
        return mp.run(data, *TEST, horizon=2, save_dir=str(tmp_path))

    counts = run().forecast_counts().set_index("model").loc["ols"]
    first = counts[CHECK_FAILED]  # 24 March out of range, 27 March short of 09:00
    path = tmp_path / "ols.jsonl"
    clean = records(path)
    day = ("2023-03-22", 1)
    kept = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if (record["run_date"], record["horizon"]) == day:
            if record["hour"] < 12:
                continue  # never written
            if record["hour"] == 15:
                record["prediction"] = 500.0
        kept.append(json.dumps(record))
    path.write_text("\n".join(kept) + "\n")
    report = run()
    after = records(path)
    for hour in range(24):
        got = after[(hour, 1, day[0])]
        assert got["status"] == CHECK_FAILED and got["detail"] == "range: 1 outside [0, 400] (15 at 500)"
        assert got["revision"] == (2 if hour >= 12 else 1)
    # A resume counts what it skipped: the days that failed before, retried as every skipped key is, and the overruled day.
    assert report.forecast_counts().set_index("model").loc["ols", CHECK_FAILED] == first + 24
    others = {key: record for key, record in clean.items() if (key[2], key[1]) != day}
    assert {key: after[key] for key in others} == others
    run()
    healed = records(path)
    for hour in range(24):
        key = (hour, 1, day[0])
        assert healed[key]["status"] == "ok" and healed[key]["prediction"] == clean[key]["prediction"]
        assert healed[key]["revision"] == (2 if hour >= 12 else 1)
    size = len(path.read_text().splitlines())
    final = run()  # the failed days retried, failing as before: no line more
    assert len(path.read_text().splitlines()) == size
    assert final.forecast_counts().set_index("model").loc["ols", "produced"] == counts["produced"]


def test_a_fallback_moves_to_its_next_rung_on_a_failed_day(tmp_path):
    data = frame()
    without_wind = ["load_d+{horizon}", "is_weekend_d+{horizon}", "is_weekday_d+{horizon}"]

    def run(checks, where):
        mp = ModelPipeline(checks=checks)
        mp.add_model(OLSModel(PREDICTORS, training_window=21, name="a"))
        mp.add_model(OLSModel(without_wind, training_window=21, name="b"))
        mp.add_ensemble(FallbackEnsemble(["a", "b"], name="fb"))
        return mp.run(data, *TEST, horizon=1, save_dir=str(tmp_path / where))

    unchecked = run(None, "plain")
    checked = run(ForecastChecks(range=(0.0, 400.0)), "checked")
    b = records(tmp_path / "checked" / "b.jsonl")
    served = records(tmp_path / "checked" / "fb.jsonl")
    plain = records(tmp_path / "plain" / "fb.jsonl")
    assert set(served) == set(plain)  # 27 March, where both lack 09:00, is served by neither, checked or not
    assert not any(key[2] == "2023-03-27" for key in served)
    for key, record in served.items():
        if key[2] == "2023-03-24":  # a's day out of range: b's whole day
            assert record["rung"] == 1 and record["member"] == "b" and record["prediction"] == b[key]["prediction"]
            assert plain[key]["rung"] == 0  # unchecked, a's out-of-range day was served
        else:
            assert record == plain[key]
    assert len([key for key in served if key[2] == "2023-03-24"]) == 24
    rungs = checked.rungs("fb").set_index("rung")
    assert rungs.loc[1, "days"] == 1 and rungs.loc[1, "forecasts"] == 24
    assert unchecked.rungs("fb").set_index("rung").loc[1, "days"] == 0


def test_validation_windows_count_check_failures_per_window(tmp_path):
    data = frame(days=60)
    mp = ModelPipeline(checks=ForecastChecks(range=(0.0, 400.0)))
    mp.add_model(OLSModel(PREDICTORS, training_window=14, name="ols"))
    report = mp.run(
        data,
        validation=("2023-03-22", "2023-03-25"),
        test=("2023-03-26", "2023-03-29"),
        horizon=1,
        save_dir=str(tmp_path),
    )
    skipped = json.loads(manifest_path(tmp_path / "ols.jsonl").read_text())["skipped"]
    assert skipped["validation"] == {CHECK_FAILED: 24}  # 24 March, out of range
    assert skipped["test"] == {CHECK_FAILED: 23, "missing_test_predictors": 1}  # 27 March, short of 09:00
    assert report.validation.forecast_counts().set_index("model").loc["ols", CHECK_FAILED] == 24
    assert report.forecast_counts().set_index("model").loc["ols", CHECK_FAILED] == 23


def test_a_run_in_memory_and_over_zones_checks_too(tmp_path):
    data = frame()
    checks = ForecastChecks(range=(0.0, 400.0))
    mp = ModelPipeline(checks=checks).add_model(OLSModel(PREDICTORS, training_window=21, name="ols"))
    report = mp.run(data, *TEST, horizon=1)
    ref = report.refs["ols"]
    assert ref.skipped[CHECK_FAILED] == 24 + 23  # 24 March out of range, 27 March short of 09:00
    assert not any(r["run_date"] in ("2023-03-24", "2023-03-27") for r in ref.iter_records())
    assert ref.produced == ref.count - 48
    panel = mp.run({"PL": data, "CZ": data.copy()}, *TEST, horizon=1, save_dir=str(tmp_path))
    for zone in ("PL", "CZ"):
        counts = panel.zones[zone].forecast_counts().set_index("model")
        assert counts.loc["ols", CHECK_FAILED] == 47


def test_checks_are_saved_to_yaml_and_set_by_a_workflow(tmp_path):
    checks = ForecastChecks(periods=False, range=(-500, 4000), max_step=300)
    mp = ModelPipeline(checks=checks).add_model(OLSModel(PREDICTORS, name="ols"))
    mp.save(tmp_path / "mp.yaml")
    saved = yaml.safe_load((tmp_path / "mp.yaml").read_text())
    assert saved["checks"] == {"class": "ForecastChecks", "params": checks.to_dict()}
    assert ModelPipeline.load(tmp_path / "mp.yaml").checks == checks
    with pytest.raises(TypeError, match="ForecastChecks"):
        ModelPipeline(checks={"range": [0, 1]})
    data = frame()
    data.to_csv(tmp_path / "data.csv")
    workflow = {
        "data_pipeline": {
            "start": "2023-02-20",
            "end": "2023-04-05",
            "sources": [{"class": "CsvSource", "params": {"file_path": str(tmp_path / "data.csv")}}],
        },
        "model_pipeline": {"path": "mp.yaml", "checks": {"class": "ForecastChecks", "params": {"max_step": 5.0}}},
    }
    (tmp_path / "wf.yaml").write_text(yaml.safe_dump(workflow))
    built = Workflow.load(tmp_path / "wf.yaml")._build_model_pipeline()
    assert built.checks == ForecastChecks(max_step=5.0)
    workflow["model_pipeline"]["checks"] = None
    (tmp_path / "wf.yaml").write_text(yaml.safe_dump(workflow))
    assert Workflow.load(tmp_path / "wf.yaml")._build_model_pipeline().checks is None
    reloaded = Workflow.load(tmp_path / "wf.yaml")
    assert reloaded.to_dict()["model_pipeline"]["checks"] is None


@pytest.mark.parametrize("freq", ["1h", "15min", "1D"])
def test_switch_days_are_whole_by_the_clock_and_a_missing_repeated_hour_fails_its_day(tmp_path, freq):
    """Over the autumn switch (Sunday 29 October 2023) and the spring one: every day passes `periods` with the clock's
    periods (24 or 96 on the autumn day, the repeated hour one period; 23 or 92 in spring; one at 1D), and the autumn day issued
    without its repeated 02:00 (its second occurrence, the row it is issued from, has no input) fails."""
    hourly = freq != "15min"
    data = make_market_data(
        start="2023-03-01", days=245, freq="1h" if hourly else "15min", noise=1.0, seed=7, horizon=2
    )
    if freq == "1D":
        data = data.at_time("12:00")
    checks = ForecastChecks()
    mp = ModelPipeline(checks=checks).add_model(OLSModel(PREDICTORS, training_window=14, name="ols"))
    for test in (("2023-03-24", "2023-03-27"), ("2023-10-26", "2023-10-30")):
        report = mp.run(data, *test, horizon=2, freq=freq, save_dir=str(tmp_path / test[0]))
        assert CHECK_FAILED not in report.forecast_counts().columns
    if freq == "1D":
        return
    wall = data.index.tz_localize(None)
    second = (wall >= pd.Timestamp("2023-10-29 02:00")) & (wall < pd.Timestamp("2023-10-29 03:00"))
    second &= data.index.tz_convert("UTC") >= pd.Timestamp("2023-10-29 01:00", tz="UTC")  # the CET occurrence
    broken = data.copy()
    broken.loc[second, "load_d+1"] = np.nan
    report = mp.run(broken, "2023-10-26", "2023-10-30", horizon=2, freq=freq, save_dir=str(tmp_path / "broken"))
    failed = [r for r in read_records(tmp_path / "broken" / "ols.jsonl", statuses=(CHECK_FAILED,))]
    per_day = 24 if freq == "1h" else 96
    missing = 1 if freq == "1h" else 4
    assert {(r["run_date"], r["horizon"]) for r in failed} == {("2023-10-29", 1)}
    assert len(failed) == per_day - missing
    assert failed[0]["detail"].startswith(f"periods: {per_day - missing} of {per_day}, missing ")


def test_a_residual_model_s_own_forecasts_are_checked_not_its_base_s(tmp_path):
    from tests.models.test_residual import frame as residual_frame
    from tests.models.test_residual import residual

    data = residual_frame()
    test = ("2023-03-13", "2023-03-19")
    plain = ModelPipeline().add_model(residual()).run(data, *test, horizon=1, save_dir=str(tmp_path / "plain"))
    forecasts = list(records(tmp_path / "plain" / "residual.jsonl", statuses=("ok",)).values())
    peaks = pd.DataFrame(forecasts).groupby(["run_date", "horizon"])["prediction"].max()
    high = float(peaks.median())  # about half of the days peak above it
    checks = ForecastChecks(range=(None, high))
    report = (
        ModelPipeline(checks=checks)
        .add_model(residual())
        .run(data, *test, horizon=1, save_dir=str(tmp_path / "checked"))
    )
    days = {(r["run_date"], r["horizon"]) for r in forecasts if r["prediction"] > high}
    turned = sum(1 for r in forecasts if (r["run_date"], r["horizon"]) in days)
    assert 0 < turned < len(forecasts)
    assert report.forecast_counts().set_index("model").loc["residual", CHECK_FAILED] == turned
    assert plain.forecast_counts().set_index("model").loc["residual", "produced"] == len(forecasts)
    manifests = {path.name: json.loads(path.read_text()) for path in (tmp_path / "checked").rglob("*.manifest.json")}
    assert manifests.pop("residual.manifest.json")["checks"] == checks.to_dict()
    assert manifests and all("checks" not in manifest for manifest in manifests.values())  # the base's: unchecked


def selection_reports(tmp_path, *evaluators):
    """The review's case of a selection effect: OLS with an input corrupted on three days against a naive benchmark, run
    without checks ("off") and with `range=(0, 400)` ("on"), with `evaluators`."""
    data = make_market_data(start="2023-01-01", days=90, freq="1h", noise=1.0, seed=5, horizon=1)
    local = pd.Series(data.index.tz_localize(None).normalize(), index=data.index)
    for day in ("2023-03-05", "2023-03-12", "2023-03-19"):
        data.loc[local == pd.Timestamp(day), "wind_d+1"] *= 30
    reports = {}
    for tag, checks in (("off", None), ("on", ForecastChecks(range=(0.0, 400.0)))):
        mp = ModelPipeline(checks=checks)
        mp.add_model(OLSModel(PREDICTORS, training_window=28, name="ols"))
        mp.add_model(NaiveModel(["price_d-1"], name="naive"))
        mp.add_evaluator(*evaluators)
        reports[tag] = mp.run(data, "2023-02-15", "2023-03-28", horizon=1, save_dir=str(tmp_path / tag))
    return reports


def flagged(call):
    """`call()`'s table and whether it warned of check_failed days."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table = call()
    return table, any("check_failed" in str(w.message) for w in caught)


def test_comparisons_on_common_keys_say_when_checks_left_days_out(tmp_path):
    """A failed day is out of the common keys, so a model's worst days drop out of its own score (the review's case: OLS with an
    input corrupted on three days against a naive benchmark). The comparison tables warn and hold the counts in
    `df.attrs["check_failed"]`; without checks they hold no such attribute."""
    from epftoolbox2.evaluators import MAEEvaluator, rMAEEvaluator

    reports = selection_reports(tmp_path, MAEEvaluator(), rMAEEvaluator("naive"))
    off, on = reports["off"], reports["on"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plain = off.compare("ols", "naive")
        plain_summary = off.summary()
    assert not [w for w in caught if "check_failed" in str(w.message)]  # nothing to say without checks
    assert "check_failed" not in plain.attrs and "check_failed" not in plain_summary.attrs
    with pytest.warns(UserWarning, match="conditional on passing the checks"):
        checked = on.compare("ols", "naive")
    with pytest.warns(UserWarning, match="check_failed"):
        summary = on.summary()
    # The corrupted days OLS forecasts out of range (two of the three), and naive's Monday after the spring switch (its 02:00
    # reads the hour the clock skipped: `periods`).
    counts = on.forecast_counts().set_index("model")[CHECK_FAILED].astype(int).to_dict()
    assert counts == {"ols": 48, "naive": 23}
    assert checked.attrs["check_failed"] == counts == summary.attrs["check_failed"]
    with pytest.warns(UserWarning, match="check_failed"):
        assert on.leaderboard("naive").attrs["check_failed"] == counts
    with pytest.warns(UserWarning, match="check_failed"):
        assert on.tests("naive").attrs["check_failed"] == counts
    # What the warning is about: the corrupted days leave the common keys, and OLS's relative score collapses.
    rmae = {tag: report.summary().set_index("model").loc["ols", "rMAE"] for tag, report in reports.items()}
    assert rmae["off"] > 1.0 and rmae["on"] < 0.5
    common = {tag: int(table["n_forecasts"].iloc[0]) for tag, table in (("off", plain), ("on", checked))}
    assert common["on"] < common["off"]


def test_an_aggregated_report_keeps_the_check_failed_counts_and_counts_no_day_twice(tmp_path):
    """`aggregate()` builds new refs: the failed days are in none of their days, and still out of the common keys, so the
    models keep their `check_failed` counts (of forecasts) and the comparisons of the daily report flag them."""
    from epftoolbox2.evaluators import MAEEvaluator, rMAEEvaluator

    reports = selection_reports(tmp_path, MAEEvaluator(), rMAEEvaluator("naive"))
    counts = {"ols": 48, "naive": 23}
    daily = reports["on"].aggregate("1D", "mean")
    for call in (lambda: daily.tests("naive"), lambda: daily.summary(), lambda: daily.leaderboard("naive")):
        table, warned = flagged(call)
        assert warned and table.attrs["check_failed"] == counts
    table = daily.forecast_counts().set_index("model")
    assert table[CHECK_FAILED].astype(int).to_dict() == counts
    assert (table["missing"] == table["incomplete_day"]).all()  # the failed days in no count of days
    assert (table["planned"] == table["produced"] + table["incomplete_day"]).all()
    plain = reports["off"].aggregate("1D", "mean")
    table, warned = flagged(lambda: plain.summary())
    assert not warned and "check_failed" not in table.attrs and CHECK_FAILED not in plain.forecast_counts().columns
