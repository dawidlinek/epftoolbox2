"""`freq="1D"` through the pipeline: issue days and horizons, windows, ensembles, the report, the exporters, YAML and the command
line, each held against a plain daily run or a run of the hourly frame it was cut from."""

import json
import sys
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from epftoolbox2 import cli
from epftoolbox2._frame_csv import load_frame_csv, save_frame_csv
from epftoolbox2.ensembles import AverageEnsemble
from epftoolbox2.evaluators import DMEvaluator, EconomicEvaluator, MAEEvaluator
from epftoolbox2.exporters import CsvExporter
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import OLSModel
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline, Workflow
from epftoolbox2.results.report import EvaluationReport
from tests.synthetic import PREDICTORS, make_market_data, with_daily_columns

WINDOW = 40
INPUTS = PREDICTORS + ["price_d-1", "load_actual_h-2", "actual_daily_max_d-1"]
TEST = ("2023-03-20", "2023-04-02")  # over the spring switch
RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "price": Availability.published("13:00", 1),
    "load_actual": Availability.realised("1h"),
    "actual_daily_*": Availability.realised("1h"),
}


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


@pytest.fixture(scope="module")
def frames():
    """Hourly rows from February to mid-April 2023, and the daily frame of their noons."""
    hourly = with_daily_columns(make_market_data(start="2023-02-01", days=75, horizon=3, noise=5.0, seed=11))
    return hourly, hourly.at_time("12:00")


def lines(records, *drop):
    """Records as text, which compares them bit for bit (and NaN equal to NaN), whatever order they finished in."""
    return sorted(json.dumps({k: v for k, v in r.items() if k not in drop}, sort_keys=True) for r in records)


def file_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def pipeline(*models, information_set=None):
    out = ModelPipeline(information_set=information_set)
    for name, predictors in models or [("ols", INPUTS)]:
        out.add_model(OLSModel(predictors, training_window=WINDOW, name=name))
    return out


def test_issue_days_and_horizons_together_are_the_fridays_and_horizons_of_a_plain_daily_run(frames, tmp_path):
    _, daily = frames
    every = pipeline().run(daily, test=TEST, horizon=3, freq="1D", save_dir=tmp_path / "every")
    some = pipeline().run(
        daily, test=TEST, horizons=[2, 3], issue_days=["friday"], freq="1D", save_dir=tmp_path / "some"
    )
    of_every = file_records(tmp_path / "every" / "ols.jsonl")
    kept = [r for r in of_every if r["run_weekday"] == 4 and r["horizon"] in (2, 3)]
    assert len(kept) == 2 * 2  # 24 and 31 March
    assert lines(file_records(tmp_path / "some" / "ols.jsonl")) == lines(kept)
    counts = some.forecast_counts().iloc[0]
    assert (counts["planned"], counts["produced"]) == (4, 4)
    assert every.forecast_counts().iloc[0]["planned"] == 14 * 3
    manifest = json.loads((tmp_path / "some" / "ols.manifest.json").read_text())
    assert (manifest["freq"], manifest["horizons"], manifest["issue_days"]) == ("1D", [2, 3], ["friday"])


def test_with_validation_each_window_of_a_daily_run_is_a_plain_run_over_it(frames, tmp_path):
    _, daily = frames
    validation = ("2023-03-10", "2023-03-19")
    pipeline().run(daily, validation=validation, test=TEST, horizon=2, freq="1D", save_dir=tmp_path / "both")
    pipeline().run(daily, test=TEST, horizon=2, freq="1D", save_dir=tmp_path / "test")
    pipeline().run(daily, test=validation, horizon=2, freq="1D", save_dir=tmp_path / "validation")
    both = file_records(tmp_path / "both" / "ols.jsonl")
    for window in ("test", "validation"):
        alone = file_records(tmp_path / window / "ols.jsonl")
        assert lines([r for r in both if r["window"] == window], "window") == lines(alone)


def test_an_average_of_daily_models_averages_them_key_by_key(frames, tmp_path):
    _, daily = frames
    pipe = pipeline(("a", INPUTS), ("b", PREDICTORS)).add_ensemble(AverageEnsemble(members=["a", "b"], name="avg"))
    report = pipe.run(daily, test=TEST, horizon=2, freq="1D", save_dir=tmp_path)
    members = {
        name: {(r["run_date"], r["horizon"]): r for r in file_records(tmp_path / f"{name}.jsonl")} for name in "ab"
    }
    averaged = file_records(tmp_path / "avg.jsonl")
    assert len(averaged) == 14 * 2
    for record in averaged:
        key = (record["run_date"], record["horizon"])
        assert record["prediction"] == np.mean([members["a"][key]["prediction"], members["b"][key]["prediction"]])
        assert (record["hour"], record["freq"]) == (0, "1D")
    counts = report.forecast_counts().set_index("model")
    assert counts.loc["avg", "planned"] == counts.loc["avg", "produced"] == 28


def test_economic_evaluator_raises_before_any_fit(frames, tmp_path, monkeypatch):
    _, daily = frames

    def no_fit(*args, **kwargs):
        raise AssertionError("a model ran")

    monkeypatch.setattr(OLSModel, "run", no_fit)
    monkeypatch.setattr(OLSModel, "_plan_predictors", no_fit)
    pipe = pipeline().add_evaluator(MAEEvaluator(), EconomicEvaluator())
    with pytest.raises(ValueError, match="EconomicEvaluator schedules a trading strategy.*freq='1D'"):
        pipe.run(daily, test=TEST, horizon=2, freq="1D")
    with pytest.raises(ValueError, match="EconomicEvaluator"):
        pipe.run(daily, test=TEST, horizon=2, freq="1D", forecast_only=True)
    with pytest.raises(ValueError, match="EconomicEvaluator"):
        pipe.report(tmp_path, test=TEST, horizon=2, freq="1D")


def test_a_report_of_daily_results_reads_them_back_and_compares_daily_losses(frames, tmp_path):
    _, daily = frames
    pipe = pipeline(("a", INPUTS), ("b", PREDICTORS)).add_evaluator(MAEEvaluator(), DMEvaluator(benchmark="b"))
    ran = pipe.run(daily, test=TEST, horizon=2, freq="1D", save_dir=tmp_path)
    read = pipe.report(tmp_path, test=TEST, horizon=2, freq="1D")
    pd.testing.assert_frame_equal(ran.summary(), read.summary())
    pd.testing.assert_frame_equal(ran.forecast_counts(), read.forecast_counts())
    tests = read.tests()
    assert set(tests["n_days"]) == {14} and set(tests["n_forecasts"]) == {14}
    with pytest.raises(ValueError, match="daily"):
        read.tests(by="hour")
    with pytest.raises(ValueError, match="daily already"):
        read.aggregate()


def test_csv_extra_columns_join_a_daily_source_once_per_forecast(frames, tmp_path):
    hourly, daily = frames
    path = tmp_path / "forecasts.csv"
    pipe = pipeline().add_exporter(CsvExporter(str(path), extra_columns=["load_actual_h-2", "is_weekend"]))
    report = pipe.run(daily, test=TEST, horizon=2, freq="1D")
    table = pd.read_csv(path, float_precision="round_trip")
    assert len(table) == 14 * 2 and set(table["hour"]) == {0}
    by_date = pd.Series(daily["load_actual_h-2"].to_numpy(), index=daily.index.date.astype(str))
    assert np.array_equal(table["load_actual_h-2"].to_numpy(), by_date.reindex(table["target_date"]).to_numpy())
    hourly_source = EvaluationReport(report.refs, [], source_data=hourly)
    with pytest.raises(ValueError, match=r"extra_columns need source data at the resolution of the 1D results"):
        CsvExporter(str(tmp_path / "again.csv"), extra_columns=["load_actual_h-2"]).export(hourly_source)


def test_a_daily_run_joins_the_daily_maximum_of_an_hourly_run(frames):
    """`aggregate("max")` of an hourly price forecast and a daily model of the price's daily maximum stamped at noon are scored
    on the same keys and the same actuals: one report, compared on their common days."""
    hourly, daily = frames
    plain = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "is_weekday_d+{horizon}"]
    hourly_report = pipeline(("hourly", plain)).add_evaluator(MAEEvaluator()).run(hourly, test=TEST, horizon=2)
    daily_report = (
        pipeline(("daily", plain))
        .add_evaluator(MAEEvaluator())
        .run(daily, test=TEST, target="price_daily_max", horizon=2, freq="1D")
    )
    with pytest.warns(UserWarning, match="left out days"):  # the spring day's targets of run S, by rule
        aggregated = hourly_report.aggregate(agg="max")
    for joined in (aggregated.join(daily_report), daily_report.join(aggregated)):
        frames_by_model = joined._model_frames()
        a, b = frames_by_model["hourly"], frames_by_model["daily"]
        common = a.merge(b, on=["run_date", "target_date", "hour", "horizon"], suffixes=("_h", "_d"))
        assert len(common) == len(a) and len(b) == 28
        assert np.array_equal(common["actual_h"].to_numpy(), common["actual_d"].to_numpy())
        assert set(joined.summary()["model"]) == {"hourly", "daily"}
        tests = joined.tests(benchmark="hourly")
        assert tests["n_forecasts"].tolist() == [len(common[common["horizon"] == h]) for h in (1, 2)]


def test_a_workflow_keeps_freq_1d_through_yaml_and_runs_a_daily_csv(frames, tmp_path, monkeypatch):
    """The data pipeline reads a daily CSV of UTC instants and puts it in the market's zone, where it is noon every date."""
    _, daily = frames
    workflow = Workflow(data_start="2023-02-01", data_end="2023-04-05", model_test=list(TEST), model_freq="1D")
    workflow.save(tmp_path / "experiment.yaml")
    assert yaml.safe_load((tmp_path / "experiment.yaml").read_text())["model_pipeline"]["freq"] == "1D"
    assert Workflow.load(tmp_path / "experiment.yaml").to_dict() == workflow.to_dict()

    columns = ["price", *[f"{c}_d+{h}" for h in (1, 2) for c in ("load", "wind", "is_weekend", "is_weekday")]]
    utc = daily[columns].tz_convert("UTC")
    utc.assign(datetime=utc.index.strftime("%Y-%m-%d %H:%M:%S")).to_csv(tmp_path / "daily.csv", index=False)
    config = {
        "data_pipeline": {
            "start": "2023-02-01",
            "end": "2023-04-05",
            "sources": [{"class": "CsvSource", "params": {"file_path": str(tmp_path / "daily.csv")}}],
            "transformers": [{"class": "TimezoneTransformer", "params": {"target_tz": "Europe/Warsaw"}}],
        },
        "model_pipeline": {
            "test": list(TEST),
            "horizon": 2,
            "freq": "1D",
            "save_dir": "results",
            "models": [{"class": "OLSModel", "params": {"predictors": PREDICTORS, "training_window": WINDOW}}],
        },
    }
    (tmp_path / "workflow.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    loaded = Workflow.load(tmp_path / "workflow.yaml")
    loaded.run()
    read = loaded._run_data()  # the CSV's numbers, which a parser may round in the last bit
    assert len(read) == 64 and np.array_equal(read.index, daily.index[:64])  # 1 February to 5 April, at noon
    in_process = OLSModel(PREDICTORS, training_window=WINDOW).run(read, *TEST, horizon=2, freq="1D")._results
    assert lines(file_records(tmp_path / "results" / "model.jsonl")) == lines(in_process)
    assert len(in_process) == 14 * 2


def run_cli(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["epf", *argv])
    try:
        cli.main()
    except SystemExit as stop:
        return stop.code
    return 0


def test_epf_model_and_verify_take_freq_1d(frames, tmp_path, monkeypatch, capsys):
    _, daily = frames
    save_frame_csv(daily, tmp_path / "daily.csv")
    loaded = load_frame_csv(tmp_path / "daily.csv")
    assert loaded.index.equals(daily.index)
    pipeline().save(tmp_path / "mp.yaml")
    argv = ["model", str(tmp_path / "mp.yaml"), "--data", str(tmp_path / "daily.csv"), "--test-start", TEST[0]]
    argv += ["--test-end", TEST[1], "--horizon", "2", "--freq", "1D", "--save-dir", str(tmp_path / "out")]
    assert run_cli(monkeypatch, argv) == 0
    in_process = OLSModel(INPUTS, training_window=WINDOW, name="ols").run(daily, *TEST, horizon=2, freq="1D")._results
    assert lines(file_records(tmp_path / "out" / "ols.jsonl")) == lines(in_process)

    pipeline(information_set=InformationSet("12:00", "Europe/Warsaw", rules=RULES)).save(tmp_path / "verify.yaml")
    argv = ["verify", str(tmp_path / "verify.yaml"), "--data", str(tmp_path / "daily.csv"), "--test-start", TEST[0]]
    argv += ["--test-end", TEST[1], "--days", "2", "--freq", "1D"]
    assert run_cli(monkeypatch, argv) == 0
    assert "ok" in capsys.readouterr().out


def test_report_reads_the_freq_its_runs_had_and_refuses_another(frames, tmp_path):
    """Told no freq, `report()` plans what the manifests record: 14 days of 2 horizons of daily results, not 24 periods each. A
    freq the manifests contradict raises, both ways, and so do models run at different frequencies."""
    hourly, daily = frames
    pipe = pipeline(("a", INPUTS), ("b", PREDICTORS)).add_evaluator(MAEEvaluator())
    pipe.run(daily, test=TEST, horizon=2, freq="1D", save_dir=tmp_path / "daily")
    counts = pipe.report(tmp_path / "daily", test=TEST).forecast_counts().set_index("model")
    assert counts["planned"].tolist() == [28, 28] and counts["missing"].tolist() == [0, 0]
    with pytest.raises(ValueError, match=r"report\(freq='1h'\), but the results .* were run at freq='1D'"):
        pipe.report(tmp_path / "daily", test=TEST, freq="1h")
    with pytest.raises(ValueError, match="EconomicEvaluator"):
        pipeline(("a", INPUTS), ("b", PREDICTORS)).add_evaluator(EconomicEvaluator()).report(
            tmp_path / "daily", test=TEST
        )

    plain = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "is_weekday_d+{horizon}"]
    hourly_pipe = pipeline(("a", plain))
    hourly_pipe.run(hourly, test=("2023-03-20", "2023-03-22"), horizon=1, save_dir=tmp_path / "hourly")
    assert hourly_pipe.report(tmp_path / "hourly", test=("2023-03-20", "2023-03-22")).forecast_counts()[
        "planned"
    ].tolist() == [72]
    with pytest.raises(ValueError, match=r"report\(freq='1D'\), but the results .* were run at freq='1h'"):
        hourly_pipe.report(tmp_path / "hourly", test=("2023-03-20", "2023-03-22"), freq="1D")

    pipeline(("b", plain)).run(daily, test=TEST, horizon=1, freq="1D", save_dir=tmp_path / "hourly")
    with pytest.raises(ValueError, match="run at different frequencies"):
        pipeline(("a", plain), ("b", plain)).report(tmp_path / "hourly", test=("2023-03-20", "2023-03-22"))


def test_report_refuses_hourly_reading_of_15_minute_results(tmp_path):
    quarter = make_market_data(start="2023-02-01", days=40, freq="15min", horizon=1, noise=5.0, seed=3)
    pipe = ModelPipeline().add_model(OLSModel(PREDICTORS, training_window=20, name="q"))
    pipe.run(quarter, test=("2023-03-01", "2023-03-02"), horizon=1, freq="15min", save_dir=tmp_path)
    assert pipe.report(tmp_path, test=("2023-03-01", "2023-03-02")).forecast_counts()["planned"].tolist() == [192]
    with pytest.raises(ValueError, match="were run at freq='15min'"):
        pipe.report(tmp_path, test=("2023-03-01", "2023-03-02"), freq="1h")
