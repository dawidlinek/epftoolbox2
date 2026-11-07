"""D2's recommended LEAR (`models/lear.mdx`, "Recommended settings"): one `LEARModel` per VST (Lago's Invariant asinh,
mirror-log, polynomial, N-PIT) and their `AverageEnsemble`, run for real on synthetic data.

The windows are scaled down from the docs' `[56, 84, 728, "expanding"]` to fit a 120-day frame; everything else is the docs'
loop. The ensemble is the mean of its four members bit for bit on the keys they share, and a key one member lacks is missing
for it and counted as `member_missing`, never averaged over the other three.
"""

import json
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest

from epftoolbox2.ensembles import AverageEnsemble
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import LEARModel
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline
from tests.synthetic import calendar_shift, make_market_data

# The docs' members: Lago's Invariant asinh (the one the research measured), then the three standardised VSTs.
VSTS = {"asinh": {"name": "asinh", "params": {"standardise": False}}, "mlog": "mlog", "poly": "poly", "npit": "npit"}
WINDOWS = [20, 40, "expanding"]
PREDICTORS = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "price_d-1", "price_d-2", "noise_a"]
RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "noise_a": Availability.static(),
    "price": Availability.published("13:00", 1),
}
TEST = ("2023-04-03", "2023-04-06")
KEY = ("run_date", "target_date", "hour", "horizon")


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


@pytest.fixture(scope="module")
def frame():
    df = make_market_data(days=120, noise=5.0, seed=5, horizon=2)
    df["noise_a"] = np.random.default_rng(2).normal(0, 1, len(df))
    df["price_d-2"] = calendar_shift(df["price"], -2)
    return df


def recommended(windows=WINDOWS):
    """The docs' loop, with its windows scaled down."""
    pipeline = ModelPipeline(
        information_set=InformationSet("11:45", "Europe/Brussels", rules=RULES), availability="raise"
    )
    for vst, scaler in VSTS.items():
        pipeline.add_model(
            LEARModel(
                predictors=PREDICTORS,
                training_window=windows,
                criterion="bic",
                scaler=scaler,
                newest_target="published",
                name=f"LEAR_{vst}",
            )
        )
    return pipeline.add_ensemble(AverageEnsemble(members=[f"LEAR_{v}" for v in VSTS], name="LEAR"))


def records(path):
    return {tuple(r[k] for k in KEY): r for r in map(json.loads, Path(path).read_text().splitlines())}


def counts_of(report):
    return report.forecast_counts().set_index("model")


@pytest.fixture(scope="module")
def run_dir(frame, tmp_path_factory):
    directory = tmp_path_factory.mktemp("recommended")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MAX_PROCESSES", "1")
        patch.setenv("THREADS_PER_PROCESS", "1")
        patch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)
        report = recommended().run(frame, test=TEST, horizon=2, save_dir=str(directory))
    return directory, report


def test_the_members_are_plain_lear_models_that_differ_only_in_their_scaler():
    pipeline = recommended()
    members = pipeline.models
    assert [m.name for m in members] == [f"LEAR_{v}" for v in VSTS]
    assert [m.scaler for m in members] == [
        {"name": "asinh", "params": {"apply_to": "all", "standardise": False}},
        "mlog",
        "poly",
        "npit",
    ]
    assert {(m.criterion, m.solver, m.newest_target, m.refit, tuple(m.training_window)) for m in members} == {
        ("bic", "lars", "published", None, tuple(WINDOWS))
    }
    assert pipeline.ensembles[0].members == [f"LEAR_{v}" for v in VSTS] and pipeline.ensembles[0].top is None


def test_the_ensemble_is_the_mean_of_its_four_members_bit_for_bit(run_dir):
    directory, report = run_dir
    members = [records(directory / f"lear_{v}.jsonl") for v in VSTS]
    ensemble = records(directory / "lear.jsonl")
    common = set.intersection(*(set(m) for m in members))
    assert set(ensemble) == common and len(common) == 4 * 24 * 2
    for key, record in ensemble.items():
        assert record["prediction"] == float(np.mean([m[key]["prediction"] for m in members]))
        assert record["actual"] == members[0][key]["actual"]
    # four different transformations: no two members agree on every key
    for i in range(4):
        for j in range(i + 1, 4):
            assert any(members[i][k]["prediction"] != members[j][k]["prediction"] for k in common)
    counts = counts_of(report)
    assert set(counts.loc[[f"LEAR_{v}" for v in VSTS] + ["LEAR"], "produced"]) == {len(common)}
    assert "member_missing" not in counts or (counts["member_missing"].fillna(0) == 0).all()


def test_a_key_one_member_lacks_is_missing_for_the_ensemble_and_counted(run_dir, tmp_path):
    """One record taken out of the polynomial member's file, as a task that crashed would leave it: the ensemble has no forecast
    there and counts it, and every other key is still the mean of the four."""
    directory, _ = run_dir
    for path in directory.iterdir():
        (tmp_path / path.name).write_bytes(path.read_bytes())
    poly = tmp_path / "lear_poly.jsonl"
    lines = poly.read_text().splitlines()
    dropped = tuple(json.loads(lines[17])[k] for k in KEY)
    poly.write_text("".join(line + "\n" for i, line in enumerate(lines) if i != 17))

    with pytest.warns(UserWarning, match="member_missing 1"):
        report = recommended().report(str(tmp_path), test=TEST, horizon=2)
    counts = counts_of(report)
    assert counts.loc["LEAR", "member_missing"] == 1
    assert list(counts.loc["LEAR", ["planned", "produced", "missing"]]) == [4 * 24 * 2, 4 * 24 * 2 - 1, 1]
    assert counts.loc["LEAR_poly", "produced"] == 4 * 24 * 2 - 1
    assert counts.loc["LEAR_asinh", "produced"] == 4 * 24 * 2

    members = [records(tmp_path / f"lear_{v}.jsonl") for v in VSTS]
    ensemble = records(tmp_path / "lear.jsonl")
    assert dropped not in ensemble and len(ensemble) == 4 * 24 * 2 - 1
    for key, record in ensemble.items():
        assert record["prediction"] == float(np.mean([m[key]["prediction"] for m in members]))
