"""Known methodology issues, pinned with xfail(strict=True).

Each test asserts the *correct* behaviour and currently fails. When an issue is fixed its test starts
passing, strict mode turns that into a failure, and the marker must be removed (move the test next to
the code it covers). Details: .claude/rules/{models,data,evaluation}.md.
"""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.transformers import DailyProfileTransformer
from epftoolbox2.information_set import Availability, InformationSet


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="`_sNN` profile names don't say their slot length; the information set reads each as a quarter-hour",
)
def test_a_half_hour_profile_slot_is_judged_at_its_own_time():
    """DailyProfileTransformer names the slots of every length but one hour `_sNN`. With freq="30min" on hourly data,
    `load_actual_s30` is the load at 15:00 on D, realised at 16:30 with a 1 h rule, but the information set reads it
    as quarter-hour 30 (07:30-07:45, known 08:45) and accepts it at every period of an 11:45 issue: a look-ahead."""
    index = pd.date_range("2024-01-01", periods=24 * 30, freq="h", tz="Europe/Warsaw")
    raw = pd.DataFrame({"load_actual": np.arange(len(index), dtype=float)}, index=index)
    data = DailyProfileTransformer(columns="load_actual", freq="30min").transform(raw)
    day = data.loc["2024-01-10"]
    assert (day["load_actual_s30"] == raw.loc["2024-01-10 15:00", "load_actual"]).all()  # 15:00 on every row
    info = InformationSet("11:45", "Europe/Brussels", rules={"load_actual_s*": Availability.realised("1h")})
    expanded = {(period, 1): ["load_actual_s30"] for period in range(24)}

    found, _ = info.violations(expanded, "price", "1h", index, "2024-01-10", "2024-01-16")

    assert {v.period for v in found} == set(range(24))


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="criterion='cv' cuts its five blocks by rows: the two rows of a doubled autumn hour can fall on either side of an edge",
)
def test_cv_keeps_both_rows_of_a_doubled_autumn_hour_in_one_block(monkeypatch):
    """On a Warsaw frame 2023-10-29 02:00 happens twice: two training rows of period 2 with one date and one target. Issued on
    2023-11-08 with a 49-row window, `LEARModel(criterion="cv")` cuts those rows at 10, 20, 30 and 40 (KFold's sizes) and the
    pair sits at rows 39 and 40, so the fold that holds out the last block trains on one copy and is scored on the other. Over
    September to December that is 4 of 2928 fits at W 49 and 56 and 3 at 84 (about 0.05 % of hourly keys); moving the edge
    changed 2 of the 4 W-49 forecasts, by 0.03 and 0.07 EUR/MWh. Frames through DSTNormalizeTransformer have no doubled rows."""
    import epftoolbox2.models.lear as lear_module
    from epftoolbox2.models import LEARModel
    from tests.models.test_lear import NAMES, InProcessPool, with_noise
    from tests.synthetic import make_market_data

    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)
    blocks_of = lear_module.cv_stats
    seen = []

    def noting(x, y, folds=5):
        stats = blocks_of(x, y, folds)
        edges = np.cumsum([block.n for block in stats.blocks])[:-1]
        pairs = np.flatnonzero(np.asarray(y)[1:] == np.asarray(y)[:-1]) + 1  # row i and i - 1: the doubled hour
        seen.extend((int(i), tuple(int(e) for e in edges)) for i in pairs)
        return stats

    monkeypatch.setattr(lear_module, "cv_stats", noting)
    frame = with_noise(make_market_data(start="2023-07-01", days=190, noise=5.0, seed=3, horizon=1))
    LEARModel(NAMES, training_window=49, criterion="cv").run(frame, "2023-11-08", "2023-11-08", horizon=1)
    if not seen:
        pytest.fail("no fit held both rows of the doubled hour: the case isn't exercised")
    assert all(i not in edges for i, edges in seen), seen


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="binary_mask=None reads a fixed list's binary columns on the test's first day, so a forecast depends on test_start",
)
def test_a_forecast_is_the_same_whatever_day_the_test_starts_on(monkeypatch):
    """Night-time solar: `solar` is 0 at 06:00 and 18:00 all winter and has values from 12 March. The default reads which columns
    are binary (and stay unscaled) on the test's first day and keeps that for every forecast of the test: from 1 March `solar` at
    06:00 is all 0 in the rows (binary), from 18 March it has values (scaled). So the forecasts at 06:00 and 18:00 issued from 18
    March differ between the test starts 1 and 18 March (all 56 such keys of a test to 31 March, by up to 6.1).
    `binary_mask="fit"` reads each fit's mask from its own rows and is independent of the start (`tests/models/test_binary_mask.py`);
    making it the default flips this test, and moves the forecasts whose own rows disagree with the first day's (decision 13 of the
    step 5 spec)."""
    from epftoolbox2.models import OLSModel
    from tests.models.test_binary_mask import SOLAR, SOLAR_PREDICTORS, InProcessPool, fits, solar_frame

    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)
    data = solar_frame()

    def run(start):
        return fits(OLSModel(SOLAR_PREDICTORS, training_window=21).run(data, start, SOLAR[1], horizon=2)._results)

    early, late = run(SOLAR[0]), run("2023-03-18")
    assert late.keys() <= early.keys()
    assert {key: early[key] for key in late} == late


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="a plain run resumed over an earlier test start mixes masks read on two first days in one file, without an error",
)
def test_a_plain_run_resumed_over_an_earlier_start_equals_the_fresh_run(monkeypatch, tmp_path):
    """Default `binary_mask=None`, fixed list, no validation or predictor function: a run over [S+5, E] and then one over [S, E]
    into the same `save_to` resumes without a word. The stored records read their mask on S+5, the new ones on S, so with
    `turning` 0/1 on S and continuous from S+3 (`OLSModel(MASKED, 21)`, spring frame) 286 of 812 records over [S, E] differ from
    a fresh [S, E] run. Correct is the fresh run's records or a refusal. `binary_mask="fit"` gives 0 differences
    (`tests/models/test_binary_mask.py`)."""
    from tests.models.test_binary_mask import InProcessPool, resumed_and_fresh

    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)

    resumed, fresh = resumed_and_fresh(None, tmp_path)

    assert len(fresh) > 0
    assert resumed == fresh, f"{len(set(resumed) ^ set(fresh)) // 2} of {len(fresh)} records differ from the fresh run"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="a 'published' row model's last issue days train on targets after test_end, which data_columns_hash doesn't cover",
)
def test_a_resume_over_another_target_after_test_end_gives_the_fresh_run(monkeypatch, tmp_path):
    """`newest_target="published"` issued at 14:00, when the prices of D+1 are out: at h=2 the newest training target of the
    last issue day is the price of the day after `test_end` (k = 1), which the manifest's `data_columns_hash` (rows up to
    `test_end`) doesn't cover. The h=1 records are deleted and the run resumed over a frame whose price of that day moved by 50:
    the resume is accepted, and the last day's 24 records at h=2, kept, aren't what a fresh run on that frame gives. Correct is
    the fresh run's records or a refusal (`ManifestMismatchError`). A model that reads history hashes what it reads after
    `test_end` (`Manifest.history_ahead`)."""
    import json

    from epftoolbox2.models import OLSModel
    from epftoolbox2.pipelines import ModelPipeline
    from epftoolbox2.results.manifest import ManifestMismatchError
    from tests.models.test_newest_target import InProcessPool
    from tests.synthetic import PREDICTORS, make_market_data

    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)
    frame = make_market_data(start="2023-01-01", days=120, horizon=2)
    rules = {name: Availability.published("10:00", 7) for name in ("load", "wind")}
    rules.update(is_weekend=Availability.static(), is_weekday=Availability.static())
    info = InformationSet("14:00", rules=rules)
    changed = frame.copy()
    changed.loc["2023-03-11", "price"] += 50.0
    test = ("2023-03-01", "2023-03-10")

    def run(data, save_dir):
        out = ModelPipeline(information_set=info)
        out.add_model(OLSModel(PREDICTORS, 28, newest_target="published", name="OLS"))
        out.run(data, *test, horizon=2, save_dir=save_dir)
        path = ModelPipeline.results_path(save_dir, type("Ref", (), {"name": "OLS"}), "jsonl")
        return path, {
            (r["run_date"], r["hour"], r["horizon"]): r.get("prediction")
            for r in map(json.loads, path.read_text().splitlines())
        }

    _, fresh = run(changed, tmp_path / "fresh")
    path, _ = run(frame, tmp_path / "resumed")
    path.write_text("".join(f"{line}\n" for line in path.read_text().splitlines() if json.loads(line)["horizon"] != 1))
    try:
        _, resumed = run(changed, tmp_path / "resumed")
    except ManifestMismatchError:
        return
    stale = sorted(key for key in fresh if resumed.get(key) != fresh[key])
    assert resumed == fresh, f"{len(stale)} of {len(fresh)} records aren't the fresh run's, e.g. {stale[:3]}"
