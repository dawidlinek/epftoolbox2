"""Array tasks writing Parquet (step 5b): 64 tasks of a `Workflow` over 640 OLS models, `models_per_task: 10`,
`save_format: parquet`, merge in `epf report` to the tables of one run, which wrote JSONL. On a daily frame (`freq: 1D`), so that
1,280 model runs take about a minute: what is checked is where records go and how they are read back, which a period more or
less doesn't change."""

import itertools
import json
from concurrent.futures import Future
from pathlib import Path

import pandas as pd
import pytest
import yaml

from epftoolbox2 import cli
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import Workflow
from epftoolbox2.results.store import iter_file_records
from tests.synthetic import make_market_data

T = ("2023-04-03", "2023-04-04")
COLUMNS = ["load_d+1", "wind_d+1", "is_weekend_d+1", "is_weekday_d+1"]
SUBSETS = [list(c) for r in range(1, len(COLUMNS) + 1) for c in itertools.combinations(COLUMNS, r)]
# 15 lists x 43 windows, the first 640: every model its own, so a model read from another's directory shows in the tables.
MODELS = [(f"m{i:03d}", s, w) for i, (s, w) in enumerate(itertools.product(SUBSETS, range(7, 50)))][:640]
PER_TASK = 10


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


def write_experiment(directory, frame_csv, save_format=None, **environment):
    model_pipeline = {
        "test": list(T),
        "target": "price",
        "horizon": 1,
        "freq": "1D",
        "save_dir": "results",
        **({"save_format": save_format} if save_format else {}),
        "models": [
            {"class": "OLSModel", "params": {"predictors": predictors, "training_window": window, "name": name}}
            for name, predictors, window in MODELS
        ],
        "evaluators": [
            {"class": "MAEEvaluator", "params": {}},
            {"class": "rMAEEvaluator", "params": {"base_model": "m000"}},
        ],
        "exporters": [{"class": "CsvExporter", "params": {"path": str(Path(directory) / "table.csv")}}],
    }
    config = {
        **({"environment": environment} if environment else {}),
        "data_pipeline": {
            "start": "2023-02-01",
            "end": "2023-04-05",
            "sources": [{"class": "CsvSource", "params": {"file_path": str(frame_csv)}}],
            "transformers": [{"class": "TimezoneTransformer", "params": {"target_tz": "Europe/Warsaw"}}],
        },
        "model_pipeline": model_pipeline,
    }
    path = Path(directory) / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def frame_csv(tmp_path_factory):
    data = make_market_data(start="2023-02-01", days=64, noise=5.0, seed=7, horizon=1).at_time("12:00")
    csv = data[["price", *COLUMNS]].tz_convert("UTC").reset_index(names="datetime")
    csv["datetime"] = csv["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    path = tmp_path_factory.mktemp("frame") / "frame.csv"
    csv.to_csv(path, index=False)
    return path


def tables(report):
    """Every table as text: `to_csv` writes each float so that it reads back as the same bits."""
    return {
        "summary": report.summary().to_csv(),
        "by_horizon": report.by_horizon().to_csv(),
        "counts": report.forecast_counts().to_csv(),
    }


def test_64_array_tasks_writing_parquet_merge_in_epf_report_to_one_run(tmp_path, frame_csv, monkeypatch):
    one = tmp_path / "one"
    one.mkdir()
    reference = Workflow.load(write_experiment(one, frame_csv)).run()  # one run, JSONL

    array = tmp_path / "array"
    array.mkdir()
    experiment = write_experiment(array, frame_csv, "parquet", model_index=0, models_per_task=PER_TASK)
    workflow = Workflow.load(experiment)
    assert workflow.array_size() == 64
    for index in range(64):
        workflow.model_index = index
        workflow.run()
    results = array / "results"
    assert sorted(p.name for p in results.glob("*.parquet")) == [f"{name}.parquet" for name, _, _ in MODELS]
    assert not list(results.glob("*.jsonl"))

    monkeypatch.setattr("sys.argv", ["epf", "report", str(experiment)])
    cli.main()
    merged = Workflow.load(experiment).report()
    assert tables(merged) == tables(reference)
    assert (array / "table.csv").read_bytes() == (one / "table.csv").read_bytes()
    counts = merged.forecast_counts()
    assert (counts["produced"] == 2).all() and len(counts) == 640
    for name, _, _ in MODELS[::64]:  # the records themselves, a model of each tenth of the list
        stored = sorted(json.dumps(r) for r in iter_file_records(results / f"{name}.parquet"))
        assert stored == sorted((one / "results" / f"{name}.jsonl").read_text().splitlines())
    manifest = json.loads((results / "m000.manifest.json").read_text())
    other = json.loads((one / "results" / "m000.manifest.json").read_text())
    drop = ("created_at", "updated_at", "save_format")
    assert manifest["save_format"] == "parquet"
    assert {k: v for k, v in manifest.items() if k not in drop} == {k: v for k, v in other.items() if k not in drop}
    pd.testing.assert_frame_equal(merged.summary(), reference.summary(), check_exact=True)
