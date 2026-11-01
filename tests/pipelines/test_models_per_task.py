"""`environment.models_per_task`: a SLURM array task runs a block of models, and the block size is no part of any result."""

import json
import warnings
from pathlib import Path

import pytest
import yaml

from epftoolbox2 import cli
from epftoolbox2.models.base import BaseModel
from epftoolbox2.pipelines import Workflow
from tests.synthetic import make_market_data

V = ("2023-03-28", "2023-03-29")
T = ("2023-04-03", "2023-04-04")
SHARED = ["is_weekend_d+1", "is_weekday_d+1"]
MODELS = {
    "m0": ["load_d+1", "wind_d+1", *SHARED],
    "m1": ["load_d+1", *SHARED],
    "m2": ["wind_d+1", *SHARED],
    "m3": ["load_d+1", "wind_d+1", "is_weekend_d+1"],
    "m4": SHARED,
}
NAMES = list(MODELS)
# Written out by hand, so that the tests don't share the code's arithmetic.
BLOCKS = {
    1: [["m0"], ["m1"], ["m2"], ["m3"], ["m4"]],
    2: [["m0", "m1"], ["m2", "m3"], ["m4"]],
    3: [["m0", "m1", "m2"], ["m3", "m4"]],
    5: [["m0", "m1", "m2", "m3", "m4"]],
    7: [["m0", "m1", "m2", "m3", "m4"]],
}


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


@pytest.fixture(scope="module")
def frame_csv(tmp_path_factory):
    data = make_market_data(start="2023-03-01", days=42, noise=5.0, seed=7)
    columns = ["price", "load_d+1", "wind_d+1", "is_weekend_d+1", "is_weekday_d+1"]
    csv = data[columns].tz_convert("UTC").reset_index(names="datetime")
    csv["datetime"] = csv["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    path = tmp_path_factory.mktemp("frame") / "frame.csv"
    csv.to_csv(path, index=False)
    return path


def write_experiment(directory, frame_csv, *, windows=True, **environment):
    """A workflow YAML with five models, an average and a `top=1` ensemble, and evaluators and an exporter that need them all."""
    model_pipeline = {
        **({"validation": list(V), "test": list(T)} if windows else {"test": list(T)}),
        "target": "price",
        "horizon": 1,
        "save_dir": "results",
        "models": [
            {"class": "OLSModel", "params": {"predictors": predictors, "training_window": 14, "name": name}}
            for name, predictors in MODELS.items()
        ],
        "ensembles": [
            *(
                [{"class": "AverageEnsemble", "params": {"members": NAMES, "top": 1, "name": "best"}}]
                if windows
                else []
            ),
            {"class": "AverageEnsemble", "params": {"members": ["m0", "m1", "m4"], "name": "average"}},
        ],
        "evaluators": [
            {"class": "MAEEvaluator", "params": {}},
            {"class": "rMAEEvaluator", "params": {"base_model": "m0"}},
        ],
        "exporters": [{"class": "CsvExporter", "params": {"path": str(Path(directory) / "table.csv")}}],
    }
    config = {
        **({"environment": environment} if environment else {}),
        "data_pipeline": {
            "start": "2023-03-01",
            "end": "2023-04-09",
            "sources": [{"class": "CsvSource", "params": {"file_path": str(frame_csv)}}],
            "transformers": [{"class": "TimezoneTransformer", "params": {"target_tz": "Europe/Warsaw"}}],
        },
        "model_pipeline": model_pipeline,
    }
    path = Path(directory) / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def manifest_without_timestamps(path):
    return {
        key: value for key, value in json.loads(path.read_text()).items() if key not in ("created_at", "updated_at")
    }


def tables(report):
    """Every table as text: `to_csv` writes each float so that it reads back as the same bits."""
    held = {
        "summary": report.summary().to_csv(),
        "counts": report.forecast_counts().to_csv(),
        "choices": report.choices().to_csv(),
    }
    if report.windows is not None:
        held["validation summary"] = report.validation.summary().to_csv()
        held["chosen on"] = json.dumps(report.choices().attrs["member_mae"], sort_keys=True)
    return held


class Snapshot:
    """What a results directory holds, and the tables and the export of the report on it."""

    def __init__(self, directory, report=None):
        results = Path(directory) / "results"
        self.records = {p.name: p.read_bytes() for p in sorted(results.glob("*.jsonl"))}
        self.manifests = {p.name: manifest_without_timestamps(p) for p in sorted(results.glob("*.manifest.json"))}
        self.tables = tables(report) if report is not None else None
        self.export = (Path(directory) / "table.csv").read_bytes()

    def sorted_records(self):
        """A resume appends what is pending, so the lines of a member's file come in another order: the lines are what count."""
        return {name: sorted(content.splitlines()) for name, content in self.records.items()}


@pytest.fixture(scope="module")
def one_run(tmp_path_factory, frame_csv):
    """What one run of the whole pipeline, with no array, leaves behind."""
    directory = tmp_path_factory.mktemp("one_run")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MAX_PROCESSES", "1")
        patch.setenv("THREADS_PER_PROCESS", "1")
        report = Workflow.load(write_experiment(directory, frame_csv)).run()
    return Snapshot(directory, report)


def run_array(directory, frame_csv, k, indices=None):
    for index in indices if indices is not None else range(len(BLOCKS[k])):
        Workflow.load(write_experiment(directory, frame_csv, model_index=index, models_per_task=k)).run()


def merge(directory, frame_csv, via="Workflow.report", **experiment):
    """Merged with no `model_index` or `models_per_task`, as if by a separate job."""
    path = write_experiment(directory, frame_csv, **experiment)
    if via == "epf report":
        cli.cmd_report(cli.build_parser().parse_args(["report", str(path)]))
        merged = Snapshot(directory)  # before anything else rewrites the files
        merged.tables = tables(Workflow.load(path).report())  # the command only prints them
        return merged
    return Snapshot(directory, Workflow.load(path).report())


def workflow_with(directory, frame_csv, **environment):
    return Workflow.load(write_experiment(directory, frame_csv, **environment))


def names_of(pipeline):
    return [model.name for model in pipeline.models]


# ---------------------------------------------------------------------------------------------------------------------
# Which models a task runs
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3, 7])
def test_each_index_selects_its_block_and_together_they_hold_every_model_once_in_order(tmp_path, frame_csv, k):
    blocks = [
        names_of(workflow_with(tmp_path, frame_csv, model_index=index, models_per_task=k)._build_model_pipeline())
        for index in range(len(BLOCKS[k]))
    ]
    assert blocks == BLOCKS[k]
    assert [name for block in blocks for name in block] == NAMES


def test_without_models_per_task_an_index_selects_one_model_as_it_always_did(tmp_path, frame_csv):
    assert names_of(workflow_with(tmp_path, frame_csv, model_index=3)._build_model_pipeline()) == ["m3"]


@pytest.mark.parametrize("k", [1, 2, 3, 7])
def test_a_task_runs_its_block_and_no_other_model(tmp_path, frame_csv, k, monkeypatch):
    ran = []
    monkeypatch.setattr(BaseModel, "run", lambda model, *args, **kwargs: ran.append(model.name))
    monkeypatch.setattr("epftoolbox2.pipelines.model_pipeline.EvaluationReport", lambda *args, **kwargs: None)
    for index, block in enumerate(BLOCKS[k]):
        workflow_with(tmp_path, frame_csv, model_index=index, models_per_task=k).run()
        assert ran[-len(block) :] == block
    assert ran == NAMES


@pytest.mark.parametrize(("k", "valid"), [(None, "0-4"), (1, "0-4"), (2, "0-2"), (3, "0-1"), (7, "0-0")])
def test_an_index_past_the_last_block_raises_naming_the_valid_range_before_reading_data(
    tmp_path, frame_csv, k, valid, monkeypatch
):
    monkeypatch.setattr(Workflow, "_run_data", lambda self: pytest.fail("the data pipeline ran before the check"))
    environment = {"models_per_task": k} if k else {}
    past_the_end = int(valid.split("-")[1]) + 1
    for index in (past_the_end, -1):
        with pytest.raises(ValueError, match=f"model_index {index} is out of range.*valid indices are {valid}"):
            workflow_with(tmp_path, frame_csv, model_index=index, **environment).run()


def test_epf_run_model_index_selects_a_block_of_the_yaml(tmp_path, frame_csv):
    path = write_experiment(tmp_path, frame_csv, models_per_task=2)
    cli.cmd_run(cli.build_parser().parse_args(["run", str(path), "--model-index", "2"]))
    assert [p.name for p in (tmp_path / "results").glob("*.jsonl")] == ["m4.jsonl"]


# ---------------------------------------------------------------------------------------------------------------------
# The size of the array
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("k", "size"), [(None, 5), (1, 5), (2, 3), (3, 2), (5, 1), (7, 1)])
def test_the_array_size_covers_every_model_whatever_the_index_is(tmp_path, frame_csv, k, size):
    environment = {"models_per_task": k} if k else {}
    assert workflow_with(tmp_path, frame_csv, **environment).array_size() == size
    assert workflow_with(tmp_path, frame_csv, model_index=0, **environment).array_size() == size


def test_a_pipeline_with_no_models_has_no_array(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump({"data_pipeline": {"start": "a", "end": "b"}, "model_pipeline": {}}))
    with pytest.raises(ValueError, match="At least one model"):
        Workflow.load(path).array_size()


# ---------------------------------------------------------------------------------------------------------------------
# Bit for bit
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "via"),
    [(1, "epf report"), (2, "epf report"), (2, "Workflow.report"), (5, "Workflow.report")],
)
def test_a_merged_array_is_bitwise_one_run(tmp_path, frame_csv, one_run, k, via):
    results = tmp_path / "results"
    for index, block in enumerate(BLOCKS[k]):
        before = set(results.glob("*.jsonl"))
        shard = Workflow.load(write_experiment(tmp_path, frame_csv, model_index=index, models_per_task=k)).run()
        assert list(shard.refs) == block
        assert sorted(p.stem for p in set(results.glob("*.jsonl")) - before) == block
    assert not (tmp_path / "table.csv").exists(), "evaluators and exporters wait for the merge"
    assert not (results / "best.jsonl").exists(), "ensembles wait for the merge"

    merged = merge(tmp_path, frame_csv, via)

    assert merged.records == one_run.records
    assert merged.manifests == one_run.manifests
    assert merged.tables == one_run.tables
    assert merged.export == one_run.export
    assert set(merged.records) == {f"{name}.jsonl" for name in [*NAMES, "best", "average"]}
    assert one_run.manifests["best.manifest.json"]["kind"] == "ensemble", "the manifests hold the ensemble's too"
    for path in results.iterdir():
        assert "per_task" not in path.read_text(), f"{path.name} records the block size"


def test_a_merged_array_without_windows_is_bitwise_one_run(tmp_path, frame_csv):
    """The plain path counts its forecasts in another way than a run with windows does."""
    single = tmp_path / "single"
    array = tmp_path / "array"
    single.mkdir()
    array.mkdir()
    one_run = Snapshot(single, Workflow.load(write_experiment(single, frame_csv, windows=False)).run())
    for index in range(len(BLOCKS[2])):
        Workflow.load(write_experiment(array, frame_csv, windows=False, model_index=index, models_per_task=2)).run()

    merged = merge(array, frame_csv, windows=False)

    assert merged.records == one_run.records
    assert merged.manifests == one_run.manifests
    assert merged.tables == one_run.tables
    assert merged.export == one_run.export


@pytest.mark.parametrize(("first", "then"), [(2, 3), (2, 1), (1, 2)])
def test_a_shard_resumed_at_another_block_size_is_bitwise_a_fresh_run(tmp_path, frame_csv, one_run, first, then):
    run_array(tmp_path, frame_csv, first, indices=[0])
    stopped = tmp_path / "results" / f"{BLOCKS[first][0][-1]}.jsonl"
    lines = stopped.read_text().splitlines(keepends=True)
    stopped.write_text("".join(lines[: len(lines) // 3]))  # an interrupted task

    run_array(tmp_path, frame_csv, then)
    merged = merge(tmp_path, frame_csv)

    assert merged.sorted_records() == one_run.sorted_records()
    assert merged.manifests == one_run.manifests
    assert merged.tables == one_run.tables
    assert merged.export == one_run.export


# ---------------------------------------------------------------------------------------------------------------------
# With no model_index
# ---------------------------------------------------------------------------------------------------------------------


def test_models_per_task_without_an_index_warns_and_runs_the_whole_pipeline(tmp_path, frame_csv, one_run):
    workflow = workflow_with(tmp_path, frame_csv, models_per_task=2)
    with pytest.warns(UserWarning, match="models_per_task=2 has no effect without model_index"):
        report = workflow.run()
    assert list(report.refs) == [*NAMES, "best", "average"]
    assert Snapshot(tmp_path, report).records == one_run.records


@pytest.mark.parametrize("environment", [{"models_per_task": 3}, {"models_per_task": 3, "model_index": 1}])
def test_a_merge_takes_every_model_whatever_the_yaml_says_and_it_does_not_warn(
    tmp_path, frame_csv, one_run, environment
):
    run_array(tmp_path, frame_csv, 3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = workflow_with(tmp_path, frame_csv, **environment).report()
    assert not [w for w in caught if "no effect" in str(w.message)]
    assert tables(report) == one_run.tables


# ---------------------------------------------------------------------------------------------------------------------
# Invalid values and YAML
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, -1, 1.5, 2.0, "2", "$K", True, [2]])
def test_an_invalid_block_size_is_refused_at_load(tmp_path, frame_csv, k):
    path = write_experiment(tmp_path, frame_csv, models_per_task=k)
    with pytest.raises(ValueError, match="models_per_task must be a positive integer"):
        Workflow.load(path)


@pytest.mark.parametrize("k", [0, -1, True])
def test_an_invalid_block_size_is_refused_when_set_in_python_too(k):
    with pytest.raises(ValueError, match="positive integer"):
        Workflow("2023-01-01", "2023-02-01", models_per_task=k)
    workflow = Workflow("2023-01-01", "2023-02-01")
    with pytest.raises(ValueError, match="positive integer"):
        workflow.models_per_task = k
    assert workflow.models_per_task is None


@pytest.mark.parametrize("k", [1, 2])
def test_models_per_task_round_trips_through_yaml_when_set(tmp_path, frame_csv, k):
    workflow = workflow_with(tmp_path, frame_csv, models_per_task=k, model_index=1)
    assert workflow.models_per_task == k
    workflow.save(tmp_path / "copy.yaml")
    saved = yaml.safe_load((tmp_path / "copy.yaml").read_text())["environment"]
    assert saved == {"model_index": 1, "models_per_task": k}
    assert Workflow.load(tmp_path / "copy.yaml").models_per_task == k


def test_a_yaml_without_the_key_loads_and_saves_as_it_always_did(tmp_path, frame_csv):
    path = write_experiment(tmp_path, frame_csv, model_index=1)
    workflow = Workflow.load(path)
    assert workflow.models_per_task is None
    assert names_of(workflow._build_model_pipeline()) == ["m1"]
    assert workflow.to_dict()["environment"] == {"model_index": 1}

    without_environment = Workflow.load(write_experiment(tmp_path, frame_csv))
    assert "environment" not in without_environment.to_dict()


def test_null_is_no_block_size(tmp_path, frame_csv):
    workflow = workflow_with(tmp_path, frame_csv, models_per_task=None, model_index=2)
    assert workflow.models_per_task is None
    assert names_of(workflow._build_model_pipeline()) == ["m2"]


# ---------------------------------------------------------------------------------------------------------------------
# What a merge refuses, and a shard before any fit
# ---------------------------------------------------------------------------------------------------------------------


def test_a_missing_block_is_refused_at_report_naming_its_models(tmp_path, frame_csv):
    run_array(tmp_path, frame_csv, 2, indices=[0, 2])  # m2 and m3 never ran
    path = write_experiment(tmp_path, frame_csv)

    with pytest.raises(FileNotFoundError, match="m2.*m3") as refused:
        Workflow.load(path).report()
    assert "m0" not in str(refused.value) and "m1" not in str(refused.value) and "m4" not in str(refused.value)

    with pytest.raises(FileNotFoundError, match="m2.*m3"):
        cli.cmd_report(cli.build_parser().parse_args(["report", str(path)]))
    assert not (tmp_path / "table.csv").exists(), "a refused merge exports nothing"
    assert not (tmp_path / "results" / "best.jsonl").exists(), "a refused merge computes no ensemble"


def test_allow_missing_leaves_out_the_block_and_its_ensembles(tmp_path, frame_csv):
    run_array(tmp_path, frame_csv, 2, indices=[0, 2])
    with pytest.warns(UserWarning, match="ensemble 'best' is left out"):
        report = Workflow.load(write_experiment(tmp_path, frame_csv)).report(allow_missing=True)
    assert list(report.refs) == ["m0", "m1", "m4", "average"], "'average' has m0, m1 and m4; 'best' lacks m2 and m3"


def test_a_late_block_can_complete_a_refused_merge(tmp_path, frame_csv, one_run):
    run_array(tmp_path, frame_csv, 2, indices=[0, 2])
    with pytest.raises(FileNotFoundError):
        merge(tmp_path, frame_csv)
    run_array(tmp_path, frame_csv, 2, indices=[1])
    merged = merge(tmp_path, frame_csv)
    assert merged.records == one_run.records and merged.tables == one_run.tables


def test_a_task_stopped_inside_a_model_is_warned_about_at_the_merge(tmp_path, frame_csv):
    run_array(tmp_path, frame_csv, 2)
    stopped = tmp_path / "results" / "m3.jsonl"
    lines = stopped.read_text().splitlines(keepends=True)
    stopped.write_text("".join(lines[:-10]))
    with pytest.warns(UserWarning, match=r"missing forecasts.*m3"):
        Workflow.load(write_experiment(tmp_path, frame_csv)).report()


@pytest.mark.parametrize("index", [0, 2])
def test_a_top_ensemble_with_no_validation_fails_any_shard_before_any_fit(tmp_path, frame_csv, index, monkeypatch):
    path = write_experiment(tmp_path, frame_csv, model_index=index, models_per_task=2)
    config = yaml.safe_load(path.read_text())
    del config["model_pipeline"]["validation"]
    path.write_text(yaml.safe_dump(config))
    fits = []
    monkeypatch.setattr(BaseModel, "run", lambda model, *args, **kwargs: fits.append(model.name))
    monkeypatch.setattr(Workflow, "_run_data", lambda self: pytest.fail("the data pipeline ran before the check"))

    with pytest.raises(ValueError, match="no validation window"):
        Workflow.load(path).run()
    assert fits == []
    assert not (tmp_path / "results").exists()


def test_an_ensemble_with_a_member_nobody_has_fails_any_shard_before_any_fit(tmp_path, frame_csv, monkeypatch):
    path = write_experiment(tmp_path, frame_csv, model_index=0, models_per_task=2)
    config = yaml.safe_load(path.read_text())
    config["model_pipeline"]["ensembles"][1]["params"]["members"] = ["m0", "typo"]
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(Workflow, "_run_data", lambda self: pytest.fail("the data pipeline ran before the check"))

    with pytest.raises(ValueError, match="unknown member 'typo'"):
        Workflow.load(path).run()


def test_a_shard_reports_its_block_with_no_evaluator(tmp_path, frame_csv):
    shard = workflow_with(tmp_path, frame_csv, model_index=1, models_per_task=2).run()
    assert list(shard.forecast_counts()["model"]) == ["m2", "m3"]
    assert "MAE" not in shard.summary().columns
