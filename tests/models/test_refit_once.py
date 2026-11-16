"""`refit_from=` and `refit="once"`: the refit calendar counted from a chosen local date, and one fit on that date for every later
forecast.

`refit_from=F` moves the epoch the blocks count from (`"7D"` refits on F's weekday, `"kMS"` every k months from F's month,
before F too). `refit="once"` needs it: one fit on F, trained as a forecast issued on F would be (rows up to F - 1, or the newest
targets `"published"` calls known by F's issue time), serves every forecast issued on F or later, and a forecast issued before
F raises. So a forecast still depends on its issue day and the configuration alone. A model that predicts a block in one call
calls it on 7-day stretches from F, each padded to 7 days as a `"7D"` block is. Without `refit_from` nothing runs differently.
"""

import datetime
import json
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from threadpoolctl import threadpool_limits

from epftoolbox2.information_set import Availability, InformationSet, leakage_probe
from epftoolbox2.models import LEARModel, OLSModel
from epftoolbox2.models import _engine_worker
from epftoolbox2.models import _worker as worker
from epftoolbox2.models._calendar import (
    ONCE_CALL_DAYS,
    block_days,
    block_rule,
    check_refit,
    check_refit_from,
    member_rules,
    next_refit_day,
    nominal_block_length,
    refit_anchor,
    refit_block_end,
)
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path
from tests.models.test_block_predict import SEES_CALL, KernelModel
from tests.models.test_day_units import DayRidge, PooledOLS
from tests.models.test_day_units import frame as profile_frame
from tests.synthetic import PREDICTORS, make_market_data

WINDOW = 21
F = "2023-03-26"  # the spring switch in Warsaw: the date has 23 hours
RULES = {
    "load": Availability.published("10:00", 10),
    "wind": Availability.published("10:00", 10),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "price": Availability.published(
        "13:00", 1
    ),  # D's price is out at 13:00 on D - 1: known at 11:45 on D, D + 1's isn't
}
INFO = InformationSet("11:45", "Europe/Warsaw", rules=RULES)
POOLED = ("epftoolbox2.models.base", "epftoolbox2.models._engine", "epftoolbox2.models._fits")


class InProcessPool:
    """Stands in for the worker pools (a backtest's, an engine group's and refit's), as in tests/models/test_stored_fits.py."""

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
    for module in POOLED:
        monkeypatch.setattr(f"{module}.ProcessPoolExecutor", InProcessPool)
    for model in (KernelModel, DayRidge, PooledOLS):
        model.calls.clear()
    with threadpool_limits(limits=1):  # what a pool's worker runs on
        yield
    for model in (KernelModel, DayRidge, PooledOLS):
        model.calls.clear()


@pytest.fixture(scope="module")
def market():
    """1 January to 11 May 2023 in Warsaw, over the spring switch of 26 March."""
    return make_market_data(start="2023-01-01", days=130, horizon=2, noise=5.0, seed=3)


def day(text):
    return datetime.date.fromisoformat(text)


def shift(text, days):
    return (day(text) + datetime.timedelta(days=days)).isoformat()


def local_dates(data):
    return np.asarray(pd.DatetimeIndex(data.index.tz_localize(None)).normalize())


def ols(refit="once", refit_from=F, **kwargs):
    return OLSModel(PREDICTORS, training_window=WINDOW, refit=refit, refit_from=refit_from, **kwargs)


def keyed(records, *drop):
    """Key -> the record as JSON text without `drop`; numbers compare as text, so a NaN `actual` equals itself."""
    return {
        (r["run_date"], r["hour"], r["horizon"]): json.dumps(
            {k: v for k, v in r.items() if k not in drop}, sort_keys=True
        )
        for r in records
    }


def fits(records):
    """Key -> what the fit made of it: prediction, coefficients and anchor."""
    return {
        (r["run_date"], r["hour"], r["horizon"]): (r["prediction"], r["coefficients"], r.get("refit_anchor"))
        for r in records
    }


def lines(path):
    return sorted(line for line in Path(path).read_text().splitlines() if line.strip())


def move(data, rows, columns, seed=1):
    out = data.copy()
    values = out[columns].to_numpy(dtype=float, copy=True)
    where = rows[:, None] & np.isfinite(values)
    values[where] += np.random.default_rng(seed).normal(0, 3, size=int(where.sum()))
    out[columns] = values
    return out


# ---------------------------------------------------------------------------
# The calendar and the parameters
# ---------------------------------------------------------------------------


def test_refit_from_counts_the_blocks_from_its_date_before_it_too():
    wednesday = "2024-01-03"
    assert refit_anchor("2024-01-09", f"7D@{wednesday}") == day("2024-01-03")
    assert refit_anchor("2024-01-10", f"7D@{wednesday}") == day("2024-01-10")
    assert refit_anchor("2023-12-31", f"7D@{wednesday}") == day("2023-12-27")
    assert refit_block_end("2024-01-09", f"7D@{wednesday}") == day("2024-01-09")
    assert refit_anchor("2024-06-15", "3MS@2024-02-01") == day("2024-05-01")
    assert refit_anchor("2024-01-15", "3MS@2024-02-01") == day("2023-11-01")
    assert refit_block_end("2024-06-15", "3MS@2024-02-01") == day("2024-07-31")
    assert refit_anchor("2025-01-31", "12MS@2024-07-01") == day("2024-07-01")
    origin = day("2023-03-26")
    for k in (1, 2, 5, 7, 10):
        for offset in range(-40, 400, 3):
            issue = day("2023-01-01") + datetime.timedelta(days=offset)
            anchor = refit_anchor(issue, f"{k}D@{origin}")
            assert 0 <= (issue - anchor).days < k and (anchor - origin).days % k == 0
            assert next_refit_day(issue, f"{k}D@{origin}") == (
                anchor if anchor == issue else refit_block_end(issue, f"{k}D@{origin}") + datetime.timedelta(days=1)
            )
    # From the epochs themselves, the anchors without refit_from.
    for offset in range(-40, 900, 11):
        issue = day("2023-01-01") + datetime.timedelta(days=offset)
        assert refit_anchor(issue, "7D@1970-01-05") == refit_anchor(issue, "7D")
        assert refit_anchor(issue, "5MS@1970-01-01") == refit_anchor(issue, "5MS")
        assert refit_block_end(issue, "5MS@1970-01-01") == refit_block_end(issue, "5MS")


def test_once_is_one_anchor_from_refit_from_on_and_none_before():
    rule = "once@2024-01-03"
    assert refit_anchor("2024-01-03", rule) == refit_anchor("2026-07-01", rule) == day("2024-01-03")
    with pytest.raises(ValueError, match="refit_from=2024-01-03 .* issued on 2024-01-02"):
        refit_anchor("2024-01-02", rule)
    assert refit_block_end("2025-05-01", rule) == datetime.date.max
    assert next_refit_day("2023-12-01", rule) == next_refit_day("2024-01-03", rule) == day("2024-01-03")
    assert next_refit_day("2024-01-04", rule) is None
    # A block call is a stretch of 7 days from refit_from.
    assert ONCE_CALL_DAYS == nominal_block_length(rule) == 7
    assert block_days("2024-01-03", rule) == [day("2024-01-03") + datetime.timedelta(days=i) for i in range(7)]
    assert (
        block_days("2024-01-16", rule)
        == block_days("2024-01-10", rule)
        == [day("2024-01-10") + datetime.timedelta(days=i) for i in range(7)]
    )
    with pytest.raises(ValueError, match="refit_from"):
        block_days("2024-01-02", rule)
    with pytest.raises(ValueError, match="nominal length"):
        nominal_block_length("once")


def test_the_rules_a_model_resolves_to():
    assert member_rules([21], "7D") == ("7D",) and member_rules([56, 728], "auto") == ("1D", "7D")
    assert member_rules([21, 28], "once", F) == (f"once@{F}", f"once@{F}")
    assert member_rules([56, 728], "auto", "2024-01-03") == ("1D@2024-01-03", "7D@2024-01-03")
    assert block_rule(member_rules([56, 728], "auto", "2024-01-03")) == "7D@2024-01-03"
    assert block_rule(member_rules([21, 28], "once", F)) == f"once@{F}"
    assert ols()._refit_rules == (f"once@{F}",) and ols("7D", None)._refit_rules == ("7D",)


@pytest.mark.parametrize(
    "bad",
    [
        datetime.datetime(2024, 1, 3),
        pd.Timestamp("2024-01-03"),
        "2024-1-3",
        "20240103",
        "2024-01-03T00:00",
        20240103,
        "",
    ],
)
def test_refit_from_is_a_local_date(bad):
    with pytest.raises(ValueError, match="refit_from must be"):
        check_refit_from(bad)
    with pytest.raises(ValueError, match="refit_from must be"):
        ols("7D", bad)
    model = ols("7D", None)
    with pytest.raises(ValueError, match="refit_from must be"):
        model.refit_from = bad
    assert model.refit_from is None


@pytest.mark.parametrize(
    "stamp",
    [
        datetime.datetime(2024, 1, 3),
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-03", tz="Europe/Warsaw"),
        np.datetime64("2024-01-03"),
    ],
)
def test_a_time_stamp_is_refused_with_how_to_pass_its_date(stamp):
    """A stamp names an instant, and a tz-aware one's local date depends on the zone: refused, with the way to give the date."""
    with pytest.raises(ValueError, match=r"time stamp .*ts\.date\(\) or str\(ts\.date\(\)\)"):
        check_refit_from(stamp)


def test_refit_from_is_kept_as_its_iso_date():
    assert check_refit("once") == "once"
    assert check_refit_from(datetime.date(2024, 1, 3)) == "2024-01-03" == check_refit_from("2024-01-03")
    assert ols("7D", datetime.date(2024, 1, 3)).refit_from == "2024-01-03"


@pytest.mark.parametrize(
    ("refit", "refit_from", "match"),
    [
        ("once", None, "refit='once' needs refit_from"),
        (None, F, "needs refit"),
        ("1MS", "2023-03-15", "first of a month"),
    ],
)
def test_a_run_refuses_a_refit_and_refit_from_that_don_t_go_together(market, tmp_path, refit, refit_from, match):
    """The pair is checked when a run resolves the rules, before anything is written: a constructor may set them in either
    order."""
    model = ols(refit, refit_from)
    path = tmp_path / "m.jsonl"
    with pytest.raises(ValueError, match=match):
        model.run(market, "2023-04-03", "2023-04-04", horizon=1, save_to=str(path))
    assert not path.exists() and not manifest_path(path).exists()


# ---------------------------------------------------------------------------
# What a forecast is
# ---------------------------------------------------------------------------


def run_pipeline(models, data, start, end, save_dir, engine="off", **kwargs):
    pipeline = ModelPipeline(engine=engine)
    for model in models:
        pipeline.add_model(model)
    pipeline.run(data, start, end, horizon=2, save_dir=str(save_dir), **kwargs)
    return {model.name: lines(Path(save_dir) / f"{model.name.lower()}.jsonl") for model in models}


@pytest.mark.parametrize("engine", ["off", "on"])
def test_once_is_a_block_that_starts_on_refit_from_and_covers_the_test(market, tmp_path, engine):
    """Over a month from the spring switch: OLS and LEAR with `"once"` give the records of `"60D"` counted from the same date,
    bit for bit, on the model path and in an engine group. Every record's anchor is refit_from."""
    end = shift(F, 30)

    def models(refit):
        return [
            ols(refit, name="ols"),
            LEARModel(PREDICTORS, training_window=WINDOW, refit=refit, refit_from=F, name="lear"),
        ]

    once = run_pipeline(models("once"), market, F, end, tmp_path / "once", engine)
    block = run_pipeline(models("60D"), market, F, end, tmp_path / "block", engine)
    assert once == block
    for name, records in once.items():
        assert len(records) == 31 * 24 * 2 - 2, name  # 26 March has no 02:00 at either horizon's target day
        assert {json.loads(line)["refit_anchor"] for line in records} == {F}


def test_refit_from_on_the_epoch_s_weekday_moves_nothing(market):
    """`"7D"` from a Monday is `"7D"` from the epoch: the same anchors, so the same records bit for bit."""
    plain = ols("7D", None).run(market, "2023-03-29", "2023-04-18", horizon=2)._results
    monday = ols("7D", "2023-03-27").run(market, "2023-03-29", "2023-04-18", horizon=2)._results
    assert keyed(plain) == keyed(monday)


def test_one_fit_per_unit_whatever_the_test_s_length(market, monkeypatch):
    units = []
    real = worker._fit_unit

    def spy(unit, tasks, *args, **kwargs):
        units.append((unit, len(tasks)))
        return real(unit, tasks, *args, **kwargs)

    monkeypatch.setattr(worker, "_fit_unit", spy)
    records = ols().run(market, F, shift(F, 40), horizon=2)._results
    assert len(units) == 24 * 2 and {unit.anchor for unit, _ in units} == {day(F)}
    assert sum(count for _, count in units) == len(records)
    units.clear()
    ols("7D", F).run(market, F, shift(F, 40), horizon=2)
    assert len(units) == 6 * 24 * 2  # a fit every week from refit_from


def test_a_forecast_is_the_same_whatever_the_run_around_it(market):
    full = keyed(ols().run(market, F, shift(F, 30), horizon=2)._results, "day_in_test")
    part = keyed(ols().run(market, shift(F, 17), shift(F, 19), horizon=2)._results, "day_in_test")
    assert part and all(full[key] == text for key, text in part.items())
    days = keyed(ols().run(market, F, shift(F, 30), horizon=2, issue_days=["thursday"])._results, "day_in_test")
    assert days and all(full[key] == text for key, text in days.items())


def same_list(ctx):
    return [name.replace("{horizon}", str(ctx.horizon)) for name in PREDICTORS]


def test_a_predictor_function_plans_from_refit_from(market):
    """A function of `ctx` is planned on the days the one fit's span reaches, so its forecasts are the fixed list's."""
    fixed = fits(ols().run(market, shift(F, 10), shift(F, 16), horizon=2)._results)
    planned = fits(
        OLSModel(same_list, training_window=WINDOW, refit="once", refit_from=F)
        .run(market, shift(F, 10), shift(F, 16), horizon=2)
        ._results
    )
    assert planned == fixed and {fit[2] for fit in planned.values()} == {F}


def test_a_forecast_issued_before_refit_from_is_refused_before_anything_is_written(market, tmp_path):
    path = tmp_path / "m.jsonl"
    with pytest.raises(ValueError, match=f"refit_from={F}.*issued on {shift(F, -1)}"):
        ols().run(market, shift(F, -1), shift(F, 5), horizon=1, save_to=str(path))
    assert not path.exists() and not manifest_path(path).exists()
    pipeline = ModelPipeline()
    pipeline.add_model(ols(name="m"))
    with pytest.raises(ValueError, match=f"refit_from={F}"):
        pipeline.run(market, validation=(shift(F, -3), shift(F, -1)), test=(F, shift(F, 5)), horizon=1)
    with pytest.raises(ValueError, match=f"refit_from={F}"):
        leakage_probe(ols, market, shift(F, -1), INFO, horizon=1)


def test_the_frame_must_reach_back_from_refit_from(market):
    """The one fit's rows start W + h days before refit_from, whatever the test's first day."""
    frame = market[local_dates(market) >= np.datetime64(shift(F, -WINDOW - 2))]
    ols().run(frame, shift(F, 20), shift(F, 21), horizon=2)
    later = market[local_dates(market) >= np.datetime64(shift(F, -WINDOW - 1))]
    with pytest.raises(ValueError, match="earliest row"):
        ols().run(later, shift(F, 20), shift(F, 21), horizon=2)


# ---------------------------------------------------------------------------
# No look-ahead: the one fit's rows end where a forecast issued on refit_from's do
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("newest", "planted_from", "control_on"),
    [("d-1", F, shift(F, -1)), ("published", shift(F, 1), F)],
    ids=["d-1", "published"],
)
def test_a_target_planted_on_and_after_refit_from_moves_no_forecast(market, newest, planted_from, control_on):
    """Every price from `planted_from` on moved (with `"d-1"` refit_from's own; with `"published"` the day after it, F's price
    being out at 11:45 on F): no forecast of five weeks moves, only the actuals. The positive control moves the newest target
    the one fit may read, and every forecast moves but those of 02:00 under `"published"`: refit_from has no 02:00, so that
    period's newest target is a day older."""
    dates = local_dates(market)
    end = shift(F, 34)

    def run(frame):
        model = ols(newest_target=newest)
        return model.run(frame, F, end, horizon=2, information_set=INFO)._results

    base = run(market)
    planted = run(move(market, dates >= np.datetime64(planted_from), ["price"], seed=2))
    assert fits(planted) == fits(base)
    assert keyed(planted) != keyed(base)  # the actuals did move
    newest_days = {r.get("newest_target_day") for r in base}
    assert newest_days == ({None} if newest == "d-1" else {F})
    control = fits(run(move(market, dates == np.datetime64(control_on), ["price"], seed=2)))
    still = {key[1] for key, fit in fits(base).items() if control[key][0] == fit[0]}
    assert still == (set() if newest == "d-1" else {2})


def test_verify_is_ok_for_a_once_model(market):
    pipeline = ModelPipeline(information_set=INFO)
    pipeline.add_model(ols(name="once"))
    pipeline.verify(market, shift(F, 9), shift(F, 9), horizon=2, days=1).assert_ok()


# ---------------------------------------------------------------------------
# Manifests, resume and YAML
# ---------------------------------------------------------------------------


def test_the_manifest_records_once_and_its_date_and_a_resume_with_another_raises(market, tmp_path):
    path = tmp_path / "m.jsonl"
    ols().run(market, F, shift(F, 6), horizon=1, save_to=str(path))
    manifest = json.loads(manifest_path(path).read_text())
    assert manifest["training_cut"] == {"rows": "dates", "refit": "once", "refit_epoch": F}
    for refit, refit_from in (("once", shift(F, -1)), ("7D", F), ("60D", F)):
        with pytest.raises(ManifestMismatchError, match="training_cut"):
            ols(refit, refit_from).run(market, F, shift(F, 6), horizon=1, save_to=str(path))
    weekly = tmp_path / "weekly.jsonl"
    ols("7D", None).run(market, "2023-04-03", "2023-04-04", horizon=1, save_to=str(weekly))
    assert json.loads(manifest_path(weekly).read_text())["training_cut"]["refit_epoch"] == "1970-01-05"
    with pytest.raises(ManifestMismatchError, match="training_cut"):
        ols("7D", "2023-03-27").run(market, "2023-04-03", "2023-04-04", horizon=1, save_to=str(weekly))


def test_a_run_cut_short_resumes_to_a_fresh_one(market, tmp_path):
    """A crash leaves some forecasts of the one fit stored and others not: the resume fits it again and writes the others
    alone, so the file is a fresh run's, line for line."""
    fresh, cut = tmp_path / "fresh.jsonl", tmp_path / "cut.jsonl"
    ols().run(market, F, shift(F, 20), horizon=2, save_to=str(fresh))
    stored = lines(fresh)
    kept = [line for line, keep in zip(stored, np.random.default_rng(2).random(len(stored)) < 0.6) if keep]
    ols().run(market, F, shift(F, 20), horizon=2, save_to=str(cut))
    cut.write_text("\n".join(kept) + "\n")
    ols().run(market, F, shift(F, 20), horizon=2, save_to=str(cut))
    assert lines(cut) == stored


def test_yaml_writes_refit_from_only_when_set_and_loads_it_back(tmp_path):
    pipeline = ModelPipeline()
    pipeline.add_model(OLSModel(PREDICTORS, name="plain"))
    pipeline.add_model(ols(name="once"))
    pipeline.add_model(LEARModel(PREDICTORS, refit="7D", refit_from=datetime.date(2023, 3, 22), name="lear"))
    saved = pipeline.to_dict()["models"]
    assert "refit_from" not in saved[0]["params"]
    assert (saved[1]["params"]["refit"], saved[1]["params"]["refit_from"]) == ("once", F)
    assert saved[2]["params"]["refit_from"] == "2023-03-22"
    path = tmp_path / "pipeline.yaml"
    pipeline.save(path)
    assert yaml.safe_load(path.read_text())["models"][1]["params"]["refit_from"] == F
    loaded = ModelPipeline.load(path)
    assert [(m.refit, m.refit_from) for m in loaded.models] == [(None, None), ("once", F), ("7D", "2023-03-22")]
    assert loaded.to_dict() == pipeline.to_dict()


# ---------------------------------------------------------------------------
# Block calls: 7-day stretches from refit_from, one fit
# ---------------------------------------------------------------------------

WEDNESDAY = "2023-03-22"  # stretches from it don't start on Mondays, and the first holds the spring switch


def kernel(refit, **kwargs):
    return KernelModel(refit=refit, refit_from=WEDNESDAY, **SEES_CALL, **kwargs)


def day_ridge(refit, **kwargs):
    return DayRidge(refit=refit, refit_from=WEDNESDAY, **SEES_CALL, **kwargs)


def pooled_ols(refit, **kwargs):
    return PooledOLS(refit=refit, refit_from=WEDNESDAY, **SEES_CALL, **kwargs)


@pytest.fixture(scope="module")
def profiles():
    return profile_frame()


@pytest.mark.parametrize(
    ("make", "fit_method", "units", "rows_per_day"),
    [(kernel, "_fit_model", 24 * 2, 1), (day_ridge, "_fit_days", 2, 1), (pooled_ols, "_fit_model", 2, 24)],
    ids=["period", "day", "pooled"],
)
def test_a_block_call_under_once_is_a_seven_day_stretch_of_the_one_fit(
    market, profiles, monkeypatch, make, fit_method, units, rows_per_day
):
    """Three stretches from a Wednesday, the first over the spring switch: one fit per unit, calls of 7 days (each the shape of
    a `"7D"` call), a forecast the same in a run of its day alone (the models see their call's rows), and the first stretch
    the records of `"7D"` from the same Wednesday, whose first block is that fit and that call."""
    data = profiles if make is not kernel else market
    cls = type(make("once"))
    fitted = []
    real = getattr(cls, fit_method)

    def spy(self, *args):
        fitted.append(1)
        return real(self, *args)

    monkeypatch.setattr(cls, fit_method, spy)
    end = shift(WEDNESDAY, 20)
    full = keyed(make("once").run(data, WEDNESDAY, end, horizon=2)._results, "day_in_test")
    assert len(fitted) == units
    assert {len(call) for call in cls.calls} == {7 * rows_per_day}
    assert len(cls.calls) == 3 * units if make is kernel else 3 * 2
    alone = keyed(make("once").run(data, shift(WEDNESDAY, 10), shift(WEDNESDAY, 10), horizon=2)._results, "day_in_test")
    assert alone and all(full[key] == text for key, text in alone.items())
    weekly = keyed(make("7D").run(data, WEDNESDAY, shift(WEDNESDAY, 6), horizon=2)._results, "day_in_test")
    assert weekly == {key: text for key, text in full.items() if key[0] <= shift(WEDNESDAY, 6)}
    later = keyed(make("7D").run(data, shift(WEDNESDAY, 7), shift(WEDNESDAY, 13), horizon=2)._results, "day_in_test")
    assert any(full[key] != text for key, text in later.items())  # a weekly refit is another fit


# ---------------------------------------------------------------------------
# Stored fits: the stored fit is the one fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ["off", "on"])
def test_the_stored_fit_of_once_is_the_one_fit_and_serves_the_backtest_s_records(tmp_path, engine):
    from tests.models.test_stored_fits import backtest, days_of, served

    data = make_market_data(start="2023-01-20", days=80, horizon=2, noise=2.0)
    first, window = "2023-03-24", ("2023-03-24", "2023-03-29")
    pipeline = ModelPipeline(information_set=INFO, engine=engine)
    pipeline.add_model(OLSModel(PREDICTORS, WINDOW, refit="once", refit_from=first, name="m"))
    live = tmp_path / "live"
    made = pipeline.refit(data, live, issue=first, horizon=1)
    assert made.fitted == {"m": [first]}
    later = pipeline.refit(data, live, issue="2023-03-29", horizon=1)
    assert later.noop and later.not_due == {"m": [first]}
    expected = backtest(pipeline, data, window, "1h", 1, tmp_path / "backtest")["m"]
    got = served(pipeline, data, live, days_of(window), "1h", 1)["m"]
    assert set(got) == set(expected) and all(got[key] == expected[key] for key in expected)
    assert {json.loads(text)["refit_anchor"] for text in got.values()} == {first}
    assert sorted(path.name for path in (live / "fits" / "m").iterdir()) == [first]
    with pytest.raises(ValueError, match=f"refit_from={first}"):
        pipeline.refit(data, live, issue="2023-03-23", horizon=1)


# ---------------------------------------------------------------------------
# ResidualModel, a frozen choice and the torch stacks
# ---------------------------------------------------------------------------


def test_a_residual_model_s_base_and_learner_take_once():
    """A base that fits once forecasts from refit_from on: an expanding learner's warm-up starts there when the base's window
    is filled by then, and raises when it isn't (the one fit would train on fewer days). A learner that fits once: one anchor."""
    from tests.models.test_residual import TEST, base, residual, warm_start
    from tests.models.test_residual import frame as residual_frame

    data = residual_frame()
    once_base = base(refit="once", refit_from="2023-02-15")
    assert warm_start(residual(training_window="expanding", base_model=once_base), data) == day("2023-02-15")
    early = base(refit="once", refit_from="2023-01-20")  # its 21-day window is filled from 25 January
    with pytest.raises(ValueError, match="fits once"):
        warm_start(residual(training_window="expanding", base_model=early), data)
    learner = residual(training_window=28, refit="once", refit_from=TEST[0])
    assert learner._stage_learner().refit_from == TEST[0]
    records = ModelPipeline().add_model(learner).run(data, *TEST, horizon=1).refs["residual"]._results
    assert records and {r["refit_anchor"] for r in records} == {TEST[0]}
    saved = ModelPipeline().add_model(learner).to_dict()["models"][0]["params"]
    assert saved["refit_from"] == TEST[0] and "refit_from" not in saved["learner"]["params"]


def test_a_frozen_choice_compares_refit_from_as_the_run_writes_it(market, tmp_path):
    from epftoolbox2.pipelines._frozen import _model_cut

    path = tmp_path / "m.jsonl"
    for model in (ols(), ols("7D", None), ols("auto", "2023-03-27")):
        model.run(market, "2023-04-03", "2023-04-04", horizon=1, save_to=str(path))
        cut = json.loads(manifest_path(path).read_text())["training_cut"]
        assert _model_cut(model) == {k: v for k, v in cut.items() if k != "rows"}, model.refit
        path.unlink()
        manifest_path(path).unlink()


def test_the_torch_stack_key_holds_refit_from_and_an_mlp_fits_once():
    pytest.importorskip("torch")
    from epftoolbox2.models import MLPDailyModel
    from epftoolbox2.models._torch.groups import stack_key

    def mlp(refit="once", refit_from=F, **kwargs):
        return MLPDailyModel(
            ["price_h08_d-1", "load_h08_d+{horizon}", "is_weekend_d+{horizon}"],
            hidden=(4,),
            training_window=WINDOW,
            refit=refit,
            refit_from=refit_from,
            seeds=1,
            max_epochs=2,
            device="cpu",
            stack="vmap",
            loss="l1",
            **kwargs,
        )

    keys = {stack_key(mlp()), stack_key(mlp(refit_from=shift(F, 1))), stack_key(mlp("60D"))}
    assert len(keys) == 3 and None not in keys
    data = profile_frame()
    end = shift(F, 9)
    once = keyed(mlp(name="a").run(data, F, end, horizon=1)._results, "day_in_test")
    block = keyed(mlp("60D", name="a").run(data, F, end, horizon=1)._results, "day_in_test")
    assert once == block and {json.loads(text)["refit_anchor"] for text in once.values()} == {F}
    # Two of them stacked (one key): each unit predicts once per stretch through the stack's session, as alone.
    pipeline = ModelPipeline()
    pipeline.add_model(mlp(name="a")).add_model(mlp(name="b", random_state=1))
    stacked = pipeline.run(data, F, end, horizon=1)
    assert keyed(stacked.refs["a"]._results, "day_in_test") == once
    alone = keyed(mlp(name="b", random_state=1).run(data, F, end, horizon=1)._results, "day_in_test")
    assert keyed(stacked.refs["b"]._results, "day_in_test") == alone


def test_a_pooled_model_over_one_zone_fits_once_as_the_model_does(tmp_path):
    from epftoolbox2.models import PooledModel
    from tests.models.test_pooled import HOURLY_TEST, PREDICTORS as POOLED_PREDICTORS
    from tests.models.test_pooled import hourly

    frame = hourly("PL")
    first = HOURLY_TEST[0]

    def model(refit):
        return OLSModel(POOLED_PREDICTORS, training_window=30, refit=refit, refit_from=first, name="m")

    run = dict(test=HOURLY_TEST, horizon=2, freq="1h", target="price")
    ModelPipeline().add_model(model("once")).run(frame, save_dir=str(tmp_path / "plain"), **run)
    ModelPipeline().add_model(PooledModel(model("once"))).run({"PL": frame}, save_dir=str(tmp_path / "once"), **run)
    ModelPipeline().add_model(PooledModel(model("60D"))).run({"PL": frame}, save_dir=str(tmp_path / "block"), **run)
    once = lines(tmp_path / "once" / "PL" / "m.jsonl")
    assert once == lines(tmp_path / "plain" / "m.jsonl") == lines(tmp_path / "block" / "PL" / "m.jsonl")
    assert {json.loads(line)["refit_anchor"] for line in once} == {first}
