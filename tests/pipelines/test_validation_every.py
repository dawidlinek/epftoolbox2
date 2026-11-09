"""`run(..., validation_every=k)`: every k-th day of the validation window, each forecast the one a run of every day gives it.

A thinned run must be a subset of the full run, bit for bit: the trap is a day count that reads a list, a mask or a newest
target by the place of a day among the run's issue days, which thinning changes. So every test that compares numbers compares
records as text, against the records of the full run on the kept days.
"""

import json
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from epftoolbox2.ensembles import AverageEnsemble
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import OLSModel, RidgeModel
from epftoolbox2.models import _engine
from epftoolbox2.models._calendar import IssueCalendar
from epftoolbox2.models._engine_worker import _engine_load
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results import RunWindows, read_windows
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path
from tests.synthetic import PREDICTORS, make_market_data, with_daily_columns

RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "price": Availability.published("13:00", 1),
    "flag": Availability.static(),
    "load_actual": Availability.realised("1h"),
    "actual_daily_*": Availability.realised("1h"),
}
INFO = InformationSet("11:45", "Europe/Warsaw", rules=RULES)


class InProcessPool:
    """The worker pool in the test's own process: a fit's bits are its own, whichever process runs it."""

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


class InProcessEnginePool(InProcessPool):
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        _engine_load(*initargs)


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)
    monkeypatch.setattr(_engine, "ProcessPoolExecutor", InProcessEnginePool)
    with threadpool_limits(limits=1):  # what a pool's worker runs on
        yield


def with_flag(frame, day):
    """A 0/1 column until `day`, other values after it: a list's binary mask depends on the rows it is read from."""
    out = frame.copy()
    flag = (np.random.default_rng(5).random(len(frame)) > 0.5).astype(float)
    rows = out.index.tz_localize(None).normalize() >= pd.Timestamp(day)
    flag[rows] = flag[rows] * 0.5 + 0.25
    out["flag"] = flag
    return out


def by_day_and_value(ctx):
    """A list that depends on the issue day and on a value it reads."""
    names = list(PREDICTORS)
    if ctx.issue_day.day % 2:
        names.append("flag")
    if ctx.published("load_d+{horizon}") and ctx.value("load_d+{horizon}") > 1000:
        names.remove("wind_d+{horizon}")
    return names


def lines(records, *drop):
    """Records as text, which compares them bit for bit (and NaN equal to NaN), whatever order they finished in."""
    return sorted(json.dumps({k: v for k, v in r.items() if k not in drop}, sort_keys=True) for r in records)


def stored(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


SWITCHES = {
    # validation over the switch day, test after it
    "spring": dict(
        start="2023-01-01", days=120, validation=("2023-03-14", "2023-04-02"), test=("2023-04-10", "2023-04-12")
    ),
    "autumn": dict(
        start="2023-08-01", days=120, validation=("2023-10-15", "2023-11-03"), test=("2023-11-10", "2023-11-12")
    ),
}


def frame_of(switch, freq):
    spec = SWITCHES[switch]
    hourly = make_market_data(start=spec["start"], days=spec["days"], horizon=3, noise=5.0, seed=3)
    if freq == "1D":
        return with_flag(with_daily_columns(hourly), "2023-03-01").at_time("12:00")
    frame = hourly if freq == "1h" else make_market_data(start=spec["start"], days=spec["days"], freq=freq, horizon=3, noise=5.0, seed=3)  # fmt: skip
    return with_flag(frame, "2023-03-25" if switch == "spring" else "2023-10-25")


DAILY = PREDICTORS + ["price_d-1", "actual_daily_max_d-1"]


def model_of(kind, predictors, refit, window, published, freq):
    if freq == "1D" and predictors is not by_day_and_value:
        predictors = DAILY
    arguments = dict(training_window=window, refit=refit, name="m")
    if published:
        arguments["newest_target"] = "published"
    if kind == "ridge":
        return RidgeModel(predictors, binary_mask="fit", **arguments)
    return OLSModel(predictors, **arguments)


# Cells over (switch) x (freq 1h/15min/1D) x (refit None/"7D") x (list fixed/function) x (k 2/3/7) x (engine off/on) x
# (newest target d-1/published) x (window int/list) x (OLS / Ridge with binary_mask="fit"): every level of every axis is in
# at least two cells and a DST switch is inside every validation window, so a day count off by the switch would show.
CELLS = [
    ("spring", "1h", None, PREDICTORS + ["flag"], 2, "off", False, 30, "ols"),
    ("spring", "15min", "7D", by_day_and_value, 3, "on", True, 30, "ridge"),
    ("spring", "1D", None, DAILY, 7, "off", True, [20, 30], "ridge"),
    ("spring", "1h", "7D", by_day_and_value, 7, "on", False, [20, 30], "ols"),
    ("autumn", "1h", "7D", PREDICTORS + ["flag"], 3, "off", True, [20, 30], "ridge"),
    ("autumn", "15min", None, PREDICTORS, 2, "on", False, [20, 30], "ols"),
    ("autumn", "1D", "7D", DAILY, 2, "on", False, 30, "ols"),
    ("autumn", "1h", None, by_day_and_value, 3, "off", True, 30, "ridge"),
]


def cell_id(value):
    if callable(value):
        return "function"
    if isinstance(value, list) and value and isinstance(value[0], str):
        return "fixed"
    return str(value)


@pytest.mark.parametrize("switch, freq, refit, predictors, every, engine, published, window, kind", CELLS, ids=cell_id)
def test_a_thinned_runs_records_are_the_full_runs_on_its_days(
    switch, freq, refit, predictors, every, engine, published, window, kind
):
    frame = frame_of(switch, freq)
    spec = SWITCHES[switch]

    def run(k):
        pipeline = ModelPipeline(information_set=INFO, engine=engine)
        pipeline.add_model(model_of(kind, predictors, refit, window, published, freq))
        report = pipeline.run(
            frame, validation=spec["validation"], test=spec["test"], horizon=2, freq=freq, validation_every=k
        )
        return report

    full, thinned = run(1), run(every)
    first = pd.Timestamp(spec["validation"][0]).date()
    last = pd.Timestamp(spec["validation"][1]).date()
    kept = {(first + pd.Timedelta(days=d)).isoformat() for d in range(0, (last - first).days + 1, every)}
    full_records = full.refs["m"]._results if full.refs["m"]._results is not None else []
    thin_records = thinned.refs["m"]._results
    on_kept = [r for r in full_records if r["window"] == "test" or r["run_date"] in kept]
    assert {r["run_date"] for r in thin_records if r["window"] == "validation"} == kept
    assert lines(thin_records) == lines(on_kept)
    # the reports: the validation window's numbers are the kept days', the test window's untouched
    assert lines(thinned.validation.refs["m"].iter_records()) == lines(
        r for r in full.validation.refs["m"].iter_records() if r["run_date"] in kept
    )
    assert lines(thinned.refs["m"].iter_records()) == lines(full.refs["m"].iter_records())
    counts = thinned.validation.forecast_counts().iloc[0]
    assert counts["planned"] == counts["produced"] > 0
    calendar = IssueCalendar(freq, frame.index.tz, (1, 2))
    assert counts["planned"] == calendar.count(*spec["validation"], every)
    assert thinned.forecast_counts().to_dict("records") == full.forecast_counts().to_dict("records")


def test_the_kept_days_count_from_the_first_validation_day_and_the_test_is_never_thinned():
    windows = RunWindows.resolve(("2023-03-01", "2023-03-10"), ("2023-03-20", "2023-03-22"), validation_every=4)
    assert windows.issue_days() == ["2023-03-01", "2023-03-05", "2023-03-09", "2023-03-20", "2023-03-21", "2023-03-22"]
    assert [windows.window_of(day) for day in ("2023-03-01", "2023-03-02", "2023-03-20")] == [
        "validation",
        None,
        "test",
    ]
    assert windows.issues_on("2023-03-05") and not windows.issues_on("2023-03-06") and windows.issues_on("2023-03-21")
    assert RunWindows.from_dict(windows.to_dict()) == windows
    assert windows.to_dict()["validation_every"] == 4
    every_day = RunWindows.resolve(("2023-03-01", "2023-03-10"), ("2023-03-20", "2023-03-22"))
    assert "validation_every" not in every_day.to_dict()  # every other run's manifest keeps its bytes
    assert every_day == RunWindows.resolve(
        ("2023-03-01", "2023-03-10"), ("2023-03-20", "2023-03-22"), validation_every=1
    )


@pytest.fixture(scope="module")
def df():
    return with_flag(make_market_data(horizon=2, noise=5.0, seed=3, days=110), "2023-03-25")


V = ("2023-03-14", "2023-04-02")
T = ("2023-04-10", "2023-04-12")


def ols(name="m", window=30):
    return OLSModel(PREDICTORS, training_window=window, name=name)


def built(*models, evaluators=(), ensembles=()):
    pipeline = ModelPipeline()
    for model in models:
        pipeline.add_model(model)
    for ensemble in ensembles:
        pipeline.add_ensemble(ensemble)
    pipeline.add_evaluator(*evaluators)
    return pipeline


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(validation_every=0, validation=V), "at least 1"),
        (dict(validation_every=True, validation=V), "at least 1"),
        (dict(validation_every=2.0, validation=V), "at least 1"),
        (dict(validation_every=2, test=T), "pass validation"),
        (dict(validation_every=2, validation="previous_year", test=T), "fixed validation window"),
        (dict(validation_every=2, validation=V, issue_days=["friday"]), "pass one of them"),
    ],
)
def test_what_is_refused_before_anything_runs(df, kwargs, message):
    with pytest.raises(ValueError, match=message):
        built(ols()).run(df, horizon=1, **kwargs)
    with pytest.raises(ValueError, match=message):  # over zones too
        built(ols()).run({"PL": df}, horizon=1, **kwargs)


def test_the_manifest_records_it_and_a_report_from_the_files_is_the_runs(df, tmp_path):
    report = built(ols(), evaluators=[MAEEvaluator()]).run(
        df, validation=V, test=T, horizon=2, save_dir=str(tmp_path), validation_every=5
    )
    assert read_windows(manifest_path(tmp_path / "m.jsonl")).validation_every == 5
    assert json.loads(manifest_path(tmp_path / "m.jsonl").read_text())["windows"]["validation_every"] == 5
    again = built(ols(), evaluators=[MAEEvaluator()]).report(tmp_path, horizon=2)
    for a, b in ((report, again), (report.validation, again.validation)):
        pd.testing.assert_frame_equal(a.summary(), b.summary())
        pd.testing.assert_frame_equal(a.forecast_counts(), b.forecast_counts())
    alone = built(ols(), evaluators=[MAEEvaluator()]).report(tmp_path, horizon=2, validation=V)
    pd.testing.assert_frame_equal(alone.forecast_counts(), report.validation.forecast_counts())
    pd.testing.assert_frame_equal(alone.summary(), report.validation.summary())


def test_a_plain_runs_manifest_has_nothing_new(df, tmp_path):
    built(ols()).run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path))
    assert "validation_every" not in json.loads(manifest_path(tmp_path / "m.jsonl").read_text())["windows"]


def test_a_resume_with_another_value_raises_and_the_other_days_stay_out_of_the_report(df, tmp_path):
    built(ols()).run(df, validation=V, horizon=1, save_dir=str(tmp_path), validation_every=1)
    with pytest.raises(ManifestMismatchError, match="validation_every=1"):
        built(ols()).run(df, validation=V, horizon=1, save_dir=str(tmp_path), validation_every=4)
    report = built(ols()).run(
        df, validation=V, horizon=1, save_dir=str(tmp_path), validation_every=4, allow_manifest_mismatch=True
    )
    days = {r["run_date"] for r in report.refs["m"].iter_records()}
    assert days == {"2023-03-14", "2023-03-18", "2023-03-22", "2023-03-26", "2023-03-30"}
    assert len({r["run_date"] for r in stored(tmp_path / "m.jsonl")}) == 20  # the file keeps the other days' records
    counts = report.forecast_counts().iloc[0]
    assert counts["planned"] == counts["produced"]
    again = built(ols()).report(tmp_path, horizon=1)
    assert lines(again.refs["m"].iter_records()) == lines(report.refs["m"].iter_records())


def test_the_same_value_resumes(df, tmp_path):
    first = built(ols()).run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path), validation_every=3)
    second = built(ols()).run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path), validation_every=3)
    assert lines(first.validation.refs["m"].iter_records()) == lines(second.validation.refs["m"].iter_records())


def test_parquet_results_read_the_kept_days_alone(df, tmp_path):
    jsonl = built(ols()).run(df, validation=V, horizon=1, save_dir=str(tmp_path / "j"), validation_every=1)
    built(ols()).run(df, validation=V, horizon=1, save_dir=str(tmp_path / "p"), save_format="parquet")
    thinned = built(ols()).run(
        df,
        validation=V,
        horizon=1,
        save_dir=str(tmp_path / "p"),
        save_format="parquet",
        validation_every=6,
        allow_manifest_mismatch=True,
    )
    kept = {"2023-03-14", "2023-03-20", "2023-03-26", "2023-04-01"}
    assert lines(thinned.refs["m"].iter_records()) == lines(
        r for r in jsonl.refs["m"].iter_records() if r["run_date"] in kept
    )
    report = built(ols(), evaluators=[MAEEvaluator()]).report(tmp_path / "p", horizon=1)
    assert report.summary()["forecasts"].tolist() == [len(lines(thinned.refs["m"].iter_records()))]


def test_an_ensemble_chooses_on_the_kept_days(df):
    models = (ols("a", 20), ols("b", 40))
    full = built(*models, ensembles=[AverageEnsemble(["a", "b"], top=1, name="e")]).run(
        df, validation=V, test=T, horizon=1
    )
    thinned = built(*models, ensembles=[AverageEnsemble(["a", "b"], top=1, name="e")]).run(
        df, validation=V, test=T, horizon=1, validation_every=4
    )
    kept = {"2023-03-14", "2023-03-18", "2023-03-22", "2023-03-26", "2023-03-30"}
    choice = thinned.choices().iloc[0]
    expected = {
        name: float(
            np.mean(
                [
                    abs(r["prediction"] - r["actual"])
                    for r in full.validation.refs[name].iter_records()
                    if r["run_date"] in kept
                ]
            )
        )
        for name in ("a", "b")
    }
    assert (
        choice["validation_forecasts"] == thinned.validation.forecast_counts().set_index("model").loc["a", "produced"]
    )
    assert thinned.choices().attrs["member_mae"]["e"][0] == pytest.approx(expected, abs=1e-12)
    assert list(choice["chosen"]) == [min(expected, key=expected.get)]
    assert "member_missing" not in thinned.forecast_counts().columns


def test_a_run_over_zones_thins_each_zone_as_a_run_alone(df):
    alone = built(ols()).run(df, validation=V, test=T, horizon=1, validation_every=3)
    zones = built(ols()).run({"PL": df}, validation=V, test=T, horizon=1, validation_every=3)
    assert lines(zones.zones["PL"].validation.refs["m"].iter_records()) == lines(
        alone.validation.refs["m"].iter_records()
    )
    assert lines(zones.zones["PL"].refs["m"].iter_records()) == lines(alone.refs["m"].iter_records())
