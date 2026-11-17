"""The live commands of `epf` (step 8i): `epf forecast`, `refit`, `mark-delivered`, `status`, `monitor` and `snapshot`, in process
through `cli.main(argv, clock=...)`, and the `live:` section of the workflow YAML they read.

An experiment is a workflow YAML over a CSV of synthetic market data (no network), with two OLS models and a fallback
ensemble, issued 11:45 Europe/Warsaw on 2023-03-07. Every call runs on a `FakeClock`: sleeps pass at once.
"""

from concurrent.futures import Future

import pytest
import yaml

from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import _engine_worker
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import Workflow
from tests.synthetic import make_market_data

pytestmark = pytest.mark.usefixtures("no_network")

RULES = {
    "load": Availability.published("10:00", 10),
    "wind": Availability.published("10:00", 10),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "price": Availability.published("13:00", 1),
    "extra": Availability.published("11:15", 1),
}
INFO = InformationSet("11:45", "Europe/Warsaw", rules=RULES)
PREDICTORS = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "is_weekday_d+{horizon}"]
ISSUE = "2023-03-07"
BEFORE_DEADLINE = "2023-03-07T11:30+01:00"  # 11:30 in Warsaw: before the 11:45 issue time and the 11:55 deadline
AFTER_DEADLINE = "2023-03-07T11:58+01:00"


class InProcessPool:
    """Stands in for the worker pools, whose start is most of these tests' time."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        if initializer is _engine_worker._engine_init:
            _engine_worker._engine_load(*initargs)
        else:
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
    for module in ("epftoolbox2.models.base", "epftoolbox2.models._engine", "epftoolbox2.models._fits"):
        monkeypatch.setattr(f"{module}.ProcessPoolExecutor", InProcessPool)


LIVE = {
    "save_dir": "live",
    "deadline": "11:55",
    "poll": "5min",
    "budget": "120s",
    "timeout": "30s",
    "retries": 2,
    "on_failure": "cache",
    "deliver": "live",
    "export": {"path": "out/{issue}.csv", "format": "csv", "grid": "products"},
}


def market_csv(path, days=60):
    data = make_market_data(start="2023-01-20", days=days, horizon=2, noise=2.0)
    data.tz_convert("UTC").rename_axis("datetime").reset_index().to_csv(path, index=False)


def model(name, predictors=PREDICTORS):
    return {"class": "OLSModel", "params": {"predictors": predictors, "training_window": 21, "name": name}}


def experiment_config(tmp_path, live=LIVE, **model_pipeline):
    market_csv(tmp_path / "market.csv")
    return {
        "data_pipeline": {
            "cache": False,
            "sources": [{"class": "CsvSource", "params": {"file_path": str(tmp_path / "market.csv")}}],
            "transformers": [{"class": "TimezoneTransformer", "params": {"target_tz": "Europe/Warsaw"}}],
        },
        "model_pipeline": {
            "target": "price",
            "horizon": 2,
            "information_set": INFO.to_dict(),
            "engine": "off",
            "models": [model("main"), model("base", ["is_weekend_d+{horizon}", "is_weekday_d+{horizon}"])],
            "ensembles": [{"class": "FallbackEnsemble", "params": {"members": ["main", "base"], "name": "live"}}],
            **model_pipeline,
        },
        **({} if live is None else {"live": live}),
    }


@pytest.fixture
def experiment(tmp_path):
    """The path of a workflow YAML with a live section, over a CSV in `tmp_path`."""
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(experiment_config(tmp_path)))
    return path


# --- the live: section ------------------------------------------------------------------------------------------------------


def test_a_workflow_with_a_live_section_needs_no_data_range(experiment):
    workflow = Workflow.load(experiment)
    assert workflow.data_start is None and workflow.data_end is None
    assert workflow.live.save_dir == "live" and workflow.live.deadline == "11:55"
    assert workflow.live.export == LIVE["export"]


def test_the_live_section_survives_a_save_and_a_load(experiment, tmp_path):
    first = Workflow.load(experiment)
    again = tmp_path / "again.yaml"
    first.save(again)
    second = Workflow.load(again)
    assert second.live == first.live
    assert yaml.safe_load(again.read_text())["live"] == LIVE
    assert second.to_dict() == first.to_dict()
    assert "start" not in second.to_dict()["data_pipeline"]


def test_a_workflow_without_live_still_needs_its_data_range(tmp_path):
    config = experiment_config(tmp_path, live=None)
    path = tmp_path / "plain.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(KeyError, match="start"):
        Workflow.load(path)


def test_a_live_only_workflow_refuses_a_backtest(experiment):
    with pytest.raises(ValueError, match="live-only"):
        Workflow.load(experiment).run()


@pytest.mark.parametrize(
    "live, message",
    [
        ({"deadline": "11:55"}, "needs save_dir"),
        ({"save_dir": "live", "deadlin": "11:55"}, "unknown key"),
        ({"save_dir": "live", "deadline": 715}, "quote it"),
        ({"save_dir": "live", "wait_for": {"load": 645}}, "quote it"),
        ({"save_dir": "live", "wait_for": "later"}, "wait_for"),
        ({"save_dir": "live", "retries": -1}, "retries"),
        ({"save_dir": "live", "export": {"format": "csv"}}, "export.path"),
        ({"save_dir": "live", "export": {"path": "x", "format": "xlsx"}}, "export.format"),
        ({"save_dir": "live", "export": {"path": "x", "grid": "days"}}, "export.grid"),
        ({"save_dir": "live", "export": {"path": "x", "mode": "a"}}, "unknown key"),
    ],
)
def test_a_mistake_in_the_live_section_is_refused_when_the_file_is_loaded(tmp_path, live, message):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(experiment_config(tmp_path, live=live)))
    with pytest.raises(ValueError, match=message):
        Workflow.load(path)


def test_forecast_only_into_a_save_dir_is_still_refused_next_to_a_live_section(tmp_path):
    config = experiment_config(tmp_path, forecast_only=True, save_dir="backtest")
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="forecast_only"):
        Workflow.load(path)
