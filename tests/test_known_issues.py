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
    reason="a first_day run resumed over an earlier test start mixes masks read on two first days in one file, without an error",
)
def test_a_first_day_run_resumed_over_an_earlier_start_equals_the_fresh_run(monkeypatch, tmp_path):
    """`binary_mask="first_day"` (the default up to 2.3), fixed list, no validation or predictor function: a run over [S+5, E]
    and then one over [S, E] into the same `save_to` resumes without a word. The stored records read their mask on S+5, the new
    ones on S, so with `turning` 0/1 on S and continuous from S+3 (`OLSModel(MASKED, 21)`, spring frame) 286 of 812 records over
    [S, E] differ from a fresh [S, E] run. Correct is the fresh run's records or a refusal. The default, `"fit"`, gives 0
    differences (`tests/models/test_binary_mask.py`)."""
    from tests.models.test_binary_mask import InProcessPool, resumed_and_fresh

    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)

    resumed, fresh = resumed_and_fresh("first_day", tmp_path)

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


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="an information set that wasn't told a profile's slot length (no `with_source_rules` of its transformer: "
    "`ModelPipeline.load`, the CLI, a hand-built set) reads each `_sNN` as a quarter-hour",
)
def test_a_half_hour_profile_slot_is_judged_at_its_own_time_without_being_told():
    """The same as the judged-at-its-own-time test next to `InformationSet`, with the information set not told the slot length:
    `load_actual_s30` (15:00, an hourly row known at 17:00) is read as quarter-hour 30 (07:30, known 08:45) and accepted at every
    period of an 11:45 issue, and `_s20` (10:00, known 12:00) as 05:00. A warning says so (`_warn_untold_slot`); what would
    judge it is `with_source_rules([..., transformer])`."""
    index = pd.date_range("2024-01-01", periods=24 * 30, freq="h", tz="Europe/Warsaw")
    raw = pd.DataFrame({"load_actual": np.arange(len(index), dtype=float)}, index=index)
    DailyProfileTransformer(columns="load_actual", freq="30min").transform(raw)
    info = InformationSet("11:45", "Europe/Brussels", rules={"load_actual_s*": Availability.realised("1h")})
    expanded = {(period, 1): ["load_actual_s20", "load_actual_s30"] for period in range(24)}

    found, _ = info.violations(expanded, "price", "1h", index, "2024-01-10", "2024-01-16")

    assert {(v.predictor, v.period) for v in found} == {
        (name, p) for name in ("load_actual_s20", "load_actual_s30") for p in range(24)
    }


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="the built-in A31 rule (week_zone='data') on a market's frame converted to UTC and issued on UTC's clock reads the "
    "week in UTC dates alone: Warsaw's Monday 00:00 (23:00Z on Sunday) falls in the week before, published a week early",
)
def test_a31_on_a_utc_frame_issued_in_utc_is_never_read_before_its_friday():
    """A Warsaw frame converted to UTC, issued at 12:00 UTC: `_data_weeks` reads UTC's dates and the issue zone's, UTC's too, so
    the hour from 23:00Z on Sunday is dated to the week before and `load_forecast_daily_max_d+3`, issued on Thursday, is called
    known though Friday's publication of its week is a day later (15 cells in three January weeks). The A65/A69 rules read
    Brussels' dates too there (`LATER_OF_DATA_AND_RULE`); A31 has no such floor. Found in review of
    fix/load-wind-solar-local-days, not fixed there."""
    first, last = "2025-01-06", "2025-01-26"
    index = pd.date_range(
        (pd.Timestamp(first) - pd.Timedelta(days=3)).tz_localize("Europe/Warsaw"),
        (pd.Timestamp(last) + pd.Timedelta(days=12)).tz_localize("Europe/Warsaw"),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    info = InformationSet("12:00", "UTC")
    days = pd.date_range(first, last, freq="D")
    clock = info._issue_clock(days, index, "1h")
    early = []
    for k in range(8):
        name = f"load_forecast_daily_max_d+{k}" if k else "load_forecast_daily_max"
        found = clock.available(info._resolve_for(name, 60), np.arange(24), 60)
        for d, day in enumerate(days):
            for p in range(24):
                if found[p, d] > clock.issue[d]:
                    continue
                local = (day + pd.Timedelta(days=k, hours=p)).tz_localize("UTC").tz_convert("Europe/Warsaw")
                date = local.tz_localize(None).normalize()
                friday = date - pd.Timedelta(days=date.weekday() + 3) + pd.Timedelta(hours=10)
                if friday.tz_localize("Europe/Brussels") > (day + pd.Timedelta(hours=12)).tz_localize("UTC"):
                    early.append((str(day.date()), p, name))
    assert early == [], f"{len(early)} cells known before their week was published, e.g. {early[:3]}"
