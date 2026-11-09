"""Step 5's done criterion at CI scale (step 5l): a model's forecasts are the same bits alone, in its engine group and over
the array tasks of a `Workflow`.

200 `OLSModel` and 50 `LEARModel` variants (BASE plus 3 to 12 of a 20-column weather union with 2 % of its values and a further
0.5 % missing per column, so that each variant has rows of its own; `aux` 0/1 before 2023-04-08 and continuous after), three issue
days, `engine: on`, fitted from shared statistics (OLS standard, LEAR asinh, both handing some fits back). On a daily frame
(`freq: 1D`, one row per date at noon), so that the 2 x 250 model runs take seconds: what is checked is that nothing a model
stores depends on its company, which the number of periods doesn't change (the hourly runs are the research ones below). The
group is one Workflow run of all 250; the array is 8 tasks of `models_per_task: 32` over the list, each run as `Workflow.run()`
runs it, then `Workflow.report()`; alone is a task of one model. Every model's results (sorted lines) and manifest (without
provenance) are bitwise the group's in both, and the merged report's tables are the group's.

The runs at scale (10,000 OLS variants over 64 tasks, 1,000 LEAR and Ridge, 200 LassoCV; 1 against 12 processes; engine off
within tolerance; `binary_mask="fit"` across test starts) are in `epftoolbox2-research/library_check_step5/engine_scale/`.
"""

import json
import random
from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest
import yaml
from threadpoolctl import threadpool_limits

from epftoolbox2.models import _engine, _engine_worker
from epftoolbox2.pipelines import Workflow
from epftoolbox2.results.manifest import _PROVENANCE_FIELDS
from tests.synthetic import make_market_data

UNION = 20
BASE = ["price_d-1", "load_d+1", "wind_d+1", "is_weekend_d+1", "aux"]
TEST = ("2023-04-10", "2023-04-12")
N_OLS, N_LEAR, PER_TASK = 200, 50, 32
ALONE = 6


class InProcessPool:
    """An engine pool in the test's own process: the bits are a fit's own, whichever process runs it."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        _engine_worker._engine_load(*initargs)

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
def in_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.delenv("ENGINE_MAX_MODELS", raising=False)
    monkeypatch.delenv("ENGINE_MAX_COLUMNS", raising=False)
    monkeypatch.setattr(_engine, "ProcessPoolExecutor", InProcessPool)
    with threadpool_limits(limits=1):  # what a pool's worker runs on
        yield


def gappy_frame():
    data = make_market_data(start="2023-01-01", days=110, noise=5.0, seed=11, horizon=1).at_time("12:00")
    rng = np.random.default_rng(12)
    n = len(data)
    load, wind = data["load"].to_numpy(), data["wind"].to_numpy()
    weather = {}
    for i in range(UNION):
        a, b = rng.normal(0, 1, 2)
        column = a * (load - 1000) / 100 + b * (wind - 100) / 50 + rng.normal(0, 1, n) + 5.0 * i
        column[rng.choice(n, max(1, int(0.02 * n)), replace=False)] = np.nan
        column[rng.random(n) < 0.005] = np.nan
        weather[f"w{i:02d}"] = column
    after = data.index >= pd.Timestamp("2023-04-08", tz=data.index.tz)
    aux = data["is_weekend"].to_numpy(dtype=float).copy()
    aux[after] = rng.normal(0.5, 0.3, after.sum())
    return pd.concat([data, pd.DataFrame({**weather, "aux": aux}, index=data.index)], axis=1)


def model_specs():
    rng = random.Random(13)
    specs = []
    for i in range(N_OLS + N_LEAR):
        weather = [f"w{j:02d}" for j in sorted(rng.sample(range(UNION), rng.randint(3, 12)))]
        if i < N_OLS:
            params = {"predictors": BASE + weather, "training_window": rng.choice([28, 56]), "name": f"ols{i:03d}"}
            specs.append({"class": "OLSModel", "params": params})
        else:
            params = {"predictors": BASE + weather, "training_window": [28, 56], "name": f"lear{i - N_OLS:02d}"}
            specs.append({"class": "LEARModel", "params": params})
    rng.shuffle(specs)  # each task's block holds both classes
    return specs


SPECS = model_specs()
NAMES = [spec["params"]["name"] for spec in SPECS]


@pytest.fixture(scope="module")
def frame_csv(tmp_path_factory):
    csv = gappy_frame().tz_convert("UTC").reset_index(names="datetime")
    csv["datetime"] = csv["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    path = tmp_path_factory.mktemp("frame") / "frame.csv"
    csv.to_csv(path, index=False)
    return path


def write_experiment(directory, frame_csv, **environment):
    config = {
        **({"environment": environment} if environment else {}),
        "data_pipeline": {
            "start": "2023-01-01",
            "end": "2023-04-21",
            "sources": [{"class": "CsvSource", "params": {"file_path": str(frame_csv)}}],
            "transformers": [{"class": "TimezoneTransformer", "params": {"target_tz": "Europe/Warsaw"}}],
        },
        "model_pipeline": {
            "test": list(TEST),
            "target": "price",
            "horizon": 1,
            "freq": "1D",
            "save_dir": "results",
            "engine": "on",
            "models": SPECS,
            "evaluators": [{"class": "MAEEvaluator", "params": {}}],
        },
    }
    directory.mkdir()
    path = directory / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def lines(path):
    return sorted(path.read_text(encoding="utf-8").splitlines())


def manifest(path):
    return {k: v for k, v in json.loads(path.read_text()).items() if k not in _PROVENANCE_FIELDS}


def tables(report):
    return {name: getattr(report, name)().to_csv() for name in ("summary", "by_horizon", "forecast_counts")}


# Variants with gaps of their own have different forecast counts, which the reports warn about.
@pytest.mark.filterwarnings("ignore:Report.*missing forecasts:UserWarning")
@pytest.mark.filterwarnings("ignore:The models have different numbers of forecasts:UserWarning")
def test_a_model_is_the_same_bits_alone_in_its_group_and_over_8_array_tasks(tmp_path, frame_csv):
    group = Workflow.load(write_experiment(tmp_path / "group", frame_csv)).run()

    experiment = write_experiment(tmp_path / "array", frame_csv, model_index=0, models_per_task=PER_TASK)
    workflow = Workflow.load(experiment)
    assert workflow.array_size() == 8
    for index in range(8):
        workflow.model_index = index
        workflow.run()
    merged = Workflow.load(experiment).report()

    alone_dir = write_experiment(tmp_path / "alone", frame_csv, model_index=0, models_per_task=1)
    alone = Workflow.load(alone_dir)
    places = sorted(random.Random(14).sample(range(len(SPECS)), ALONE))
    for place in places:
        alone.model_index = place
        alone.run()

    one, array = tmp_path / "group" / "results", tmp_path / "array" / "results"
    for name in NAMES:
        assert lines(array / f"{name}.jsonl") == lines(one / f"{name}.jsonl"), name
        assert manifest(array / f"{name}.manifest.json") == manifest(one / f"{name}.manifest.json"), name
    for place in places:
        name = NAMES[place]
        assert lines(tmp_path / "alone" / "results" / f"{name}.jsonl") == lines(one / f"{name}.jsonl"), name
    assert tables(merged) == tables(group)

    # The fits came from shared statistics (OLS standard, LEAR asinh), and some went back to the model path.
    engine = {name: json.loads((one / f"{name}.manifest.json").read_text())["engine"] for name in NAMES}
    assert all(e["fits"] > 0 for e in engine.values())
    assert sum(sum(e["fallbacks"].values()) for e in engine.values()) > 0
    assert {json.loads((one / f"{name}.manifest.json").read_text())["fit_path"] for name in NAMES} == {"engine-2"}
    counts = group.forecast_counts()
    assert len(counts) == len(NAMES) and (counts["produced"] > 0).all()
