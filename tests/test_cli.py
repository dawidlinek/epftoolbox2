"""`epf data` → CSV → `epf model` must give the same forecasts as running in-process (review H6)."""

import json

import numpy as np
import pandas as pd
import pytest

from epftoolbox2 import cli
from epftoolbox2._frame_csv import load_frame_csv, save_frame_csv
from epftoolbox2.models import OLSModel
from epftoolbox2.pipelines import ModelPipeline
from tests.synthetic import PREDICTORS, make_market_data

MODEL_PREDICTORS = PREDICTORS + ["price_d-1"]


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


@pytest.fixture
def local_data():
    df = make_market_data(start="2023-01-01", days=120, noise=5.0, seed=7)
    df["holiday_name"] = np.where(df.index.dayofweek == 6, "Sunday", "none")  # string column, unused by the model
    return df


def model_args(tmp_path, data_path, **overrides):
    yaml_path = tmp_path / "mp.yaml"
    ModelPipeline().add_model(OLSModel(MODEL_PREDICTORS, training_window=45, name="OLS")).save(yaml_path)
    parser = cli.build_parser()
    argv = ["model", str(yaml_path), "--data", str(data_path), "--test-start", "2023-03-24", "--test-end"]
    argv += ["2023-03-27", "--horizon", "2", "--save-dir", str(tmp_path / "out")]
    for key, value in overrides.items():
        argv += [f"--{key.replace('_', '-')}", value]
    return parser.parse_args(argv)


def keyed(records):
    return {(r["run_date"], r["hour"], r["horizon"]): (r["prediction"], r["actual"]) for r in records}


def in_process(df):
    ref = OLSModel(MODEL_PREDICTORS, training_window=45, name="OLS").run(
        df, test_start="2023-03-24", test_end="2023-03-27", horizon=2
    )
    return keyed(ref._results)


def from_cli(tmp_path):
    lines = (tmp_path / "out" / "ols.jsonl").read_text().splitlines()
    return keyed(json.loads(line) for line in lines)


def test_csv_round_trip_keeps_the_local_time_zone_and_exact_values(tmp_path, local_data):
    path = tmp_path / "data.csv"
    save_frame_csv(local_data, path)
    assert path.read_text().splitlines()[0].startswith("datetime[Europe/Warsaw],")

    loaded = load_frame_csv(path)
    assert str(loaded.index.tz) == "Europe/Warsaw"
    pd.testing.assert_frame_equal(loaded, local_data, check_names=False, check_freq=False)


def test_cli_round_trip_equals_the_in_process_run(tmp_path, local_data):
    path = tmp_path / "data.csv"
    save_frame_csv(local_data, path)

    cli.cmd_model(model_args(tmp_path, path))

    expected = in_process(local_data)
    assert len(expected) > 180  # the window spans the spring DST switch, where a few inputs are missing
    assert from_cli(tmp_path) == expected


def test_cmd_data_writes_the_zone(tmp_path, local_data, monkeypatch):
    class StubPipeline:
        def run(self, start, end, cache):
            return local_data

    monkeypatch.setattr("epftoolbox2.pipelines.data_pipeline.DataPipeline.load", lambda path: StubPipeline())
    out = tmp_path / "data.csv"
    args = cli.build_parser().parse_args(["data", "dp.yaml", "--start", "a", "--end", "b", "--output", str(out)])
    cli.cmd_data(args)

    assert str(load_frame_csv(out).index.tz) == "Europe/Warsaw"


def test_old_csv_with_offsets_needs_the_zone(tmp_path, local_data):
    path = tmp_path / "old.csv"
    local_data.to_csv(path)  # what `epf data` wrote before: offsets, no zone name

    with pytest.raises(ValueError, match="--tz"):
        load_frame_csv(path)
    with pytest.raises(SystemExit):
        cli.cmd_model(model_args(tmp_path, path))

    cli.cmd_model(model_args(tmp_path, path, tz="Europe/Warsaw"))
    assert from_cli(tmp_path) == in_process(local_data)


def test_utc_and_naive_files_load_without_a_zone(tmp_path):
    utc = pd.DataFrame({"price": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"))
    utc.to_csv(tmp_path / "utc.csv")
    assert str(load_frame_csv(tmp_path / "utc.csv").index.tz) == "UTC"

    naive = utc.tz_localize(None)
    naive.to_csv(tmp_path / "naive.csv")
    assert load_frame_csv(tmp_path / "naive.csv").index.tz is None
    assert str(load_frame_csv(tmp_path / "naive.csv", tz="Europe/Warsaw").index.tz) == "Europe/Warsaw"


def test_unknown_zone_raises(tmp_path, local_data):
    save_frame_csv(local_data, tmp_path / "data.csv")
    with pytest.raises(ValueError, match="Unknown time zone"):
        load_frame_csv(tmp_path / "data.csv", tz="Mars/Olympus")


def test_model_command_passes_freq(tmp_path, monkeypatch):
    quarter = make_market_data(start="2023-01-01", days=60, freq="15min")
    path = tmp_path / "q.csv"
    save_frame_csv(quarter, path)
    seen = {}
    monkeypatch.setattr(ModelPipeline, "run", lambda self, **kwargs: seen.update(kwargs))

    cli.cmd_model(model_args(tmp_path, path, freq="15min"))
    assert seen["freq"] == "15min"
    assert seen["data"].index.tz is not None


def test_model_horizon_help_says_days():
    parser = cli.build_parser()
    model_parser = parser._subparsers._group_actions[0].choices["model"]
    horizon = next(a for a in model_parser._actions if a.dest == "horizon")
    assert "days" in horizon.help and "hours" not in horizon.help


def test_tui_asks_for_the_zone_of_an_old_csv(tmp_path, local_data, monkeypatch):
    from epftoolbox2 import tui

    path = tmp_path / "old.csv"
    local_data.to_csv(path)
    asked = []

    class Prompt:
        def ask(self):
            return "Europe/Warsaw"

    monkeypatch.setattr(tui.questionary, "text", lambda label, **kwargs: asked.append(label) or Prompt())

    loaded = tui._load_data_csv(path)
    assert len(asked) == 1
    assert str(loaded.index.tz) == "Europe/Warsaw"
    assert loaded.index.equals(local_data.index)
