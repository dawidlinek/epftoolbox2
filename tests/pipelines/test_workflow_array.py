"""Workflow in SLURM-array mode: one model per task, durable per-model results, one merge step (review H2)."""

import numpy as np
import pandas as pd
import pytest
import yaml

from epftoolbox2 import cli
from epftoolbox2.pipelines import Workflow


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


@pytest.fixture
def experiment(tmp_path):
    """A workflow YAML over a local CSV with two models, MAE + rMAE and a CSV exporter."""
    index = pd.date_range("2023-01-01", "2023-04-01", freq="1h", tz="UTC", inclusive="left")
    rng = np.random.default_rng(0)
    t = np.arange(len(index))
    price = 100 + 20 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 5, len(index))
    pd.DataFrame({"datetime": index.strftime("%Y-%m-%d %H:%M:%S"), "price": price}).to_csv(
        tmp_path / "prices.csv", index=False
    )
    predictors = ["price_d-1", "price_d-7"]
    config = {
        "data_pipeline": {
            "start": "2023-01-01",
            "end": "2023-03-31",
            "sources": [{"class": "CsvSource", "params": {"file_path": str(tmp_path / "prices.csv")}}],
            "transformers": [
                {"class": "TimezoneTransformer", "params": {"target_tz": "Europe/Warsaw"}},
                {"class": "LagTransformer", "params": {"columns": ["price"], "lags": [1, 7], "freq": "day"}},
            ],
        },
        "model_pipeline": {
            "test_start": "2023-03-01",
            "test_end": "2023-03-02",
            "target": "price",
            "horizon": 1,
            "save_dir": "results",
            "models": [
                {"class": "OLSModel", "params": {"predictors": predictors, "training_window": 30, "name": "OLS"}},
                {
                    "class": "LassoCVModel",
                    "params": {"predictors": predictors, "training_window": 30, "name": "Lasso", "cv": 3},
                },
            ],
            "evaluators": [
                {"class": "MAEEvaluator", "params": {}},
                {"class": "rMAEEvaluator", "params": {"base_model": "OLS"}},
            ],
            "exporters": [{"class": "CsvExporter", "params": {"path": str(tmp_path / "results.csv")}}],
        },
    }
    path = tmp_path / "experiment.yaml"

    def write(**environment):
        cfg = dict(config, environment=environment) if environment else config
        path.write_text(yaml.safe_dump(cfg))
        return path

    return write


def test_array_tasks_write_per_model_results_and_skip_exporters(tmp_path, experiment):
    for index in (0, 1):
        report = Workflow.load(experiment(model_index=index)).run()
        assert len(report.refs) == 1

    assert (tmp_path / "results" / "ols.jsonl").exists()
    assert (tmp_path / "results" / "lasso.jsonl").exists()
    assert not (tmp_path / "results.csv").exists()


def test_report_merges_all_models_and_matches_a_single_run(tmp_path, experiment):
    for index in (0, 1):
        Workflow.load(experiment(model_index=index)).run()
    merged = Workflow.load(experiment()).report()

    summary = merged.summary().set_index("model")
    assert set(summary.index) == {"OLS", "Lasso"}
    assert summary.loc["OLS", "rMAE"] == pytest.approx(1.0)
    assert np.isfinite(summary.loc["Lasso", "rMAE"])
    counts = merged.forecast_counts().set_index("model")
    assert (counts["planned"] == 48).all() and (counts["produced"] == 48).all()
    exported = pd.read_csv(tmp_path / "results.csv")
    assert {"OLS_prediction", "Lasso_prediction"} <= set(exported.columns)

    (tmp_path / "results").rename(tmp_path / "array_results")
    single = Workflow.load(experiment()).run()
    pd.testing.assert_frame_equal(single.summary(), merged.summary(), check_like=True)


def test_report_fails_when_a_model_has_no_results(tmp_path, experiment):
    Workflow.load(experiment(model_index=0)).run()
    wf = Workflow.load(experiment())

    with pytest.raises(FileNotFoundError, match="Lasso"):
        wf.report()
    partial = wf.report(allow_missing=True)
    assert list(partial.refs) == ["OLS"]


@pytest.mark.parametrize("index", [-1, 2])
def test_model_index_out_of_range_raises(experiment, index):
    with pytest.raises(ValueError, match="out of range"):
        Workflow.load(experiment(model_index=index)).run()


def test_array_mode_needs_a_save_dir(tmp_path, experiment, monkeypatch):
    path = experiment(model_index=0)
    config = yaml.safe_load(path.read_text())
    del config["model_pipeline"]["save_dir"]
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(Workflow, "_run_data", lambda self: pytest.fail("data pipeline ran before the check"))

    with pytest.raises(ValueError, match="save_dir"):
        Workflow.load(path).run()


def test_save_dir_round_trips_and_resolves_from_the_yaml_directory(tmp_path, experiment):
    wf = Workflow.load(experiment())
    assert wf.model_save_dir == "results"
    assert wf._save_dir() == tmp_path / "results"
    wf.save(tmp_path / "copy.yaml")
    assert yaml.safe_load((tmp_path / "copy.yaml").read_text())["model_pipeline"]["save_dir"] == "results"


def test_epf_report_command(tmp_path, experiment, capsys):
    for index in (0, 1):
        Workflow.load(experiment(model_index=index)).run()
    args = cli.build_parser().parse_args(["report", str(experiment())])
    cli.cmd_report(args)
    assert (tmp_path / "results.csv").exists()
