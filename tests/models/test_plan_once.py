"""Planning once (step 5k): the models of one pipeline run share what each would compute alike, and every listing, manifest,
record, table and warning is what each model computing its own gives, to the bit, engine on and off.

What is shared: the frame contract per (frame, freq, read days, target, horizon) (`base._RunChecks`), availability verdicts per
column (`information_set._VERDICTS`, kept by everything they are computed from), and on one set of period frames each column's
binary-mask verdict per (period, training rows) and the test days' missing rows (`_Periods.memo`). `base._SHARED = False` and
`information_set._VERDICTS_KEPT = 0` turn the sharing off, which is the computation of before.

The grid of `test_shared_checks_move_no_bit` covers every level of every axis and every pair of levels across its six cells,
not every cell (tests.md): freq 1h and 1D; the spring and the autumn switch; a plain run, validation + test and Fridays; the
availability levels "warn" and "raise"; each cell with engine off and on. Every cell's information set has vintage classes, a
rule over time (`RuleChain`) judged on the training rows, a weekly rule, a column without a rule, and a `published` target rule
that a model with `newest_target="published"` reads; its models are OLS, Ridge with `refit`, LEAR with a list of windows, a
predictor function reading `ctx.published`, `binary_mask="fit"` and NaiveModel, which share columns. The 1,000-model
measurements are in `epftoolbox2-research/library_check_step5/plan_once/`.
"""

import json
import warnings
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from epftoolbox2 import information_set as info_module
from epftoolbox2.information_set import Availability, InformationSet, TrainingRows
from epftoolbox2.models import LEARModel, NaiveModel, OLSModel, RidgeModel
from epftoolbox2.models import _contract, _engine, _engine_worker
from epftoolbox2.models import base as base_module
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.scalers.standard import StandardScaler
from tests.synthetic import calendar_shift, make_market_data

PROVENANCE = ("package_version", "git_sha", "created_at", "updated_at", "record_schema", "kernels")


class _InProcess:
    """Stands in for a worker pool: its start is most of what these small runs would take, and nothing here is about pools."""

    load = None

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        type(self).load(*initargs)

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


class ModelPool(_InProcess):
    load = staticmethod(_worker_init)


class EnginePool(_InProcess):
    load = staticmethod(_engine_worker._engine_load)


@pytest.fixture(autouse=True)
def in_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr(base_module, "ProcessPoolExecutor", ModelPool)
    monkeypatch.setattr(_engine, "ProcessPoolExecutor", EnginePool)
    with threadpool_limits(limits=1):
        yield


def sharing(monkeypatch, on):
    """Sharing on (with no verdict kept from before, so the run fills its own) or off: each model computes its own."""
    monkeypatch.setattr(base_module, "_SHARED", on)
    monkeypatch.setattr(info_module, "_VERDICTS_KEPT", 65_536 if on else 0)
    info_module._VERDICTS.clear()


# --- frames and rules -------------------------------------------------------------------------------------------------


def market(start, days, seed, daily=False):
    """`make_market_data` with a weekly-ruled column, a 0/1 column without a rule, a column without a value on some days
    (a mask read again) and a row missing on an issue day; daily at 12:00 for "1D"."""
    data = make_market_data(start=start, days=days, noise=5.0, seed=seed, horizon=2)
    data["week_load_d+1"] = 0.5 * data["load_d+1"] + np.random.default_rng(seed).normal(0, 1, len(data))
    data["aux"] = data["is_weekend"].to_numpy(dtype=float)
    data["settle_d-1"] = calendar_shift(data["load"] + data["wind"], -1)
    if daily:
        data = data.at_time("12:00").copy()
    return data


def with_gaps(data, empty_until, gone):
    data = data.copy()
    data.loc[data.index < pd.Timestamp(empty_until, tz=data.index.tz), "week_load_d+1"] = np.nan
    return data.drop(index=data.index[(data.index.date == pd.Timestamp(gone).date()) & (data.index.hour == 12)])


@pytest.fixture(scope="module")
def frames():
    spring = market("2023-01-01", 110, 61)
    autumn = market("2023-08-01", 110, 62)
    return {
        ("1h", "spring"): with_gaps(spring, "2023-02-25", "2023-03-24"),
        ("1h", "autumn"): with_gaps(autumn, "2023-09-30", "2023-10-31"),
        ("1D", "spring"): with_gaps(market("2023-01-01", 110, 63, daily=True), "2023-02-25", "2023-03-29"),
        ("1D", "autumn"): with_gaps(market("2023-08-01", 110, 64, daily=True), "2023-09-30", "2023-10-27"),
    }


def information_set(switch):
    """Vintage classes, a weekly rule, static and published rules (`wind_d+1` late on D), and `settle` as rules over time with
    a gap up to `switch`: no value of the gap is ever known, a later one the day after it, so its `_d-1` on D is known but a
    training row's from the gap isn't. `aux` has no rule (`unknown="warn"`)."""
    after = (pd.Timestamp(switch) + pd.Timedelta(days=1)).date().isoformat()
    return InformationSet(
        "11:45",
        "Europe/Brussels",
        rules={
            "load": Availability.published("10:00", 1, vintage="latest_revision"),
            "wind": Availability.published("18:00", 1),
            "settle": [
                Availability.published("10:00", -5).during(valid_until="2022-01-01"),
                Availability.published("10:00", -1, vintage="latest_revision").during(valid_from=after),
            ],
            "week_load": Availability.weekly("friday", "10:00", vintage="latest_revision"),
            "is_weekend": Availability.static(vintage="snapshot"),
            "is_weekday": Availability.static(),
            "price": Availability.published("13:00", 1, vintage="snapshot"),
        },
        unknown="warn",
        use_defaults=False,
    )


def reads_published(ctx):
    names = [f"load_d+{ctx.horizon}" if ctx.horizon == 1 else "load_d+1", "is_weekend_d+1", "price_d-1"]
    return names + ["wind_d+1"] if ctx.published("wind_d+1") else names + ["week_load_d+1"]


def models(level):
    """The models of a cell. At "warn" some read a value known after the issue time (`load_d+2`, `wind_d+1`, the training
    rows of `settle_d-1`); at "raise" none does (the rule over time is judged on the training rows, and none is late)."""
    late = ["load_d+{horizon}", "wind_d+1"] if level == "warn" else ["load_d+1"]
    return [
        OLSModel(
            [*late, "is_weekend_d+{horizon}", "is_weekday_d+{horizon}", "price_d-1", "settle_d-1"], 28, name="ols"
        ),
        OLSModel(["week_load_d+1", "is_weekend_d+1", "aux", "price_d-1"], 28, name="ols_weekly"),
        RidgeModel([*late, "price_d-1", "is_weekend_d+1", "settle_d-1"], 28, refit="7D", name="ridge_7d"),
        LEARModel(["load_d+1", "is_weekend_d+1", "price_d-1", "week_load_d+1"], [21, 28], name="lear_list"),
        OLSModel(reads_published, 28, name="function"),
        OLSModel(["load_d+1", "is_weekday_d+1", "price_d-1", "aux"], 28, newest_target="published", name="published"),
        OLSModel(["load_d+1", "aux", "is_weekend_d+1", "week_load_d+1"], 28, binary_mask="fit", name="mask_fit"),
        NaiveModel(["price_d-1"], name="naive"),
    ]


# (freq, switch, windows, level): every pair of levels of the four axes is in some cell.
CELLS = [
    ("1h", "spring", "plain", "warn"),
    ("1h", "autumn", "validation", "raise"),
    ("1h", "spring", "fridays", "raise"),
    ("1D", "autumn", "plain", "raise"),
    ("1D", "spring", "validation", "warn"),
    ("1D", "autumn", "fridays", "warn"),
]
WINDOWS = {
    ("spring", "plain"): {"test": ("2023-03-24", "2023-03-29")},
    ("spring", "validation"): {"validation": ("2023-03-20", "2023-03-22"), "test": ("2023-03-27", "2023-03-30")},
    ("spring", "fridays"): {"test": ("2023-03-17", "2023-04-07"), "issue_days": ["friday"]},
    ("autumn", "plain"): {"test": ("2023-10-26", "2023-10-31")},
    ("autumn", "validation"): {"validation": ("2023-10-20", "2023-10-22"), "test": ("2023-10-27", "2023-10-31")},
    ("autumn", "fridays"): {"test": ("2023-10-13", "2023-11-03"), "issue_days": ["friday"]},
}
# The rule over time switches inside the training rows of a "warn" cell (so they are judged late), long before at "raise".
SWITCH = {("spring", "warn"): "2023-03-10", ("autumn", "warn"): "2023-10-06"}


# --- comparing runs ---------------------------------------------------------------------------------------------------


def lines(path):
    return sorted(line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


def manifest(path):
    return {key: value for key, value in json.loads(Path(path).read_text()).items() if key not in PROVENANCE}


def files(save_dir):
    """Every model's records (sorted lines) and manifest (without provenance) in `save_dir`."""
    out = {}
    for path in sorted(Path(save_dir).iterdir()):
        if path.name.endswith(".jsonl"):
            out[path.name] = lines(path)
        elif path.name.endswith(".manifest.json"):
            out[path.name] = manifest(path)
    return out


def listings(report):
    """What each model's ref lists (the run's and each window's), and its counts."""
    out = {}
    for name, ref in report.refs.items():
        out[name] = (ref.availability_listing, ref.produced, ref.skipped)
        for window, part in (getattr(ref, "by_window", None) or {}).items():
            out[f"{name}/{window}"] = (part.availability_listing, part.produced, part.skipped)
    return out


def tables(report):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the models' counts differ (NaiveModel, the masks of "fit"), as they may
        out = dict(report.compute_all())
    out["forecast_counts"] = report.forecast_counts()
    return out


def run(cell, frames, save_dir, engine):
    """One cell's pipeline run into `save_dir`: (its report, its warnings as (category, text) with the directory left out)."""
    freq, switch, windows, level = cell
    pipeline = ModelPipeline(
        engine=engine, information_set=information_set(SWITCH.get((switch, level), "2022-06-01")), availability=level
    )
    for model in models(level):
        pipeline.add_model(model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = pipeline.run(
            frames[(freq, switch)], horizon=2, freq=freq, save_dir=str(save_dir), **WINDOWS[(switch, windows)]
        )
    said = [(w.category.__name__, str(w.message).replace(str(save_dir), "<dir>")) for w in caught]
    return report, said


@pytest.mark.parametrize("cell", CELLS, ids=["-".join(cell) for cell in CELLS])
@pytest.mark.parametrize("engine", ["off", "on"])
def test_shared_checks_move_no_bit(cell, engine, frames, tmp_path, monkeypatch):
    """Records, manifests (listings, vintages, masks and data hashes in them), each ref's listing and counts, the tables and
    every warning: with sharing what each model computing its own gives."""
    sharing(monkeypatch, False)
    own, own_said = run(cell, frames, tmp_path / "own", engine)
    sharing(monkeypatch, True)
    shared, shared_said = run(cell, frames, tmp_path / "shared", engine)
    mine, theirs = files(tmp_path / "own"), files(tmp_path / "shared")
    assert len(mine) == 16 and mine == theirs
    assert listings(own) == listings(shared)
    assert own_said == shared_said
    a, b = tables(own), tables(shared)
    assert list(a) == list(b)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name], check_exact=True)
    # What the cell is there for: listed uses at "warn" (the late load, the rule over time on the training rows, the column
    # without a rule), skipped keys (the missing row), and models whose masks read a column again.
    listed = [row for name, (rows, _, _) in listings(shared).items() if "/" not in name for row in rows or ()]
    if cell[3] == "warn":
        assert {"load_d+2", "wind_d+1", "aux", "settle_d-1"} <= {row["predictor"] for row in listed}
    assert any(skipped for _, _, skipped in listings(shared).values())


# --- what is computed once --------------------------------------------------------------------------------------------


def variants(n, **settings):
    names = ["load_d+1", "wind_d+1", "is_weekend_d+1", "is_weekday_d+1", "price_d-1", "week_load_d+1", "aux"]
    return [OLSModel([names[(m + i) % len(names)] for i in range(4)], 28, name=f"v{m}", **settings) for m in range(n)]


@pytest.mark.parametrize("engine", ["off", "on"])
def test_the_frame_contract_is_checked_once_per_read_days(frames, monkeypatch, engine):
    """A pipeline checks the frame contract once per (freq, read days, target, horizon): six models of one window once, a
    seventh of another window once more; without sharing once per model, as before. A frame that breaks it raises at the first
    model as before."""
    data = frames[("1h", "spring")]
    calls = []
    real = _contract.check_index
    monkeypatch.setattr(_contract, "check_index", lambda *a, **k: calls.append(a[2]) or real(*a, **k))
    for on, expected in ((True, 2), (False, 7)):
        sharing(monkeypatch, on)
        calls.clear()
        pipeline = ModelPipeline(engine=engine)
        for model in [*variants(6), OLSModel(["load_d+1", "price_d-1"], 56, name="w56")]:
            pipeline.add_model(model)
        pipeline.run(data, test=("2023-03-24", "2023-03-24"), horizon=1)
        assert len(calls) == expected
    doubled = pd.concat([data, data.iloc[[100]]])
    for on in (True, False):
        sharing(monkeypatch, on)
        pipeline = ModelPipeline(engine=engine)
        for model in variants(3):
            pipeline.add_model(model)
        with pytest.raises(ValueError, match="1 duplicate timestamp"):
            pipeline.run(doubled, test=("2023-03-24", "2023-03-24"), horizon=1)


def test_a_group_reads_each_column_s_mask_and_the_missing_rows_once(frames, monkeypatch, tmp_path):
    """In an engine group each column's binary-mask verdict is read once per (period, training rows), whatever lists hold it,
    and the missing rows of the test days once; the masks (their digests, and the forecasts they scale) are each model's own."""
    data = frames[("1h", "spring")]
    sizes, missing = [], []
    real_mask, real_missing = StandardScaler.get_scalable_mask, _contract.missing_test_rows
    monkeypatch.setattr(
        StandardScaler, "get_scalable_mask", staticmethod(lambda x: sizes.append(x.shape[1]) or real_mask(x))
    )
    monkeypatch.setattr(_contract, "missing_test_rows", lambda *a: missing.append(1) or real_missing(*a))
    out = {}
    for on in (True, False):
        sharing(monkeypatch, on)
        sizes.clear(), missing.clear()
        pipeline = ModelPipeline(engine="on")
        for model in variants(12):
            pipeline.add_model(model)
        pipeline.run(data, test=("2023-03-27", "2023-03-28"), horizon=2, save_dir=str(tmp_path / str(on)))
        out[on] = files(tmp_path / str(on))
        # 7 columns, 24 periods, 2 horizons: 336 column verdicts shared; 12 models x 4 columns x 48 without.
        assert (sum(sizes), len(missing)) == ((7 * 48, 1) if on else (12 * 4 * 48, 12))
    assert out[True] == out[False]


def test_mask_verdicts_are_kept_per_rows_and_frames(frames):
    """A column's verdict is kept per (period, rows) of one set of period frames: another window reads other rows, a model
    alone reads its own frames, and every mask equals one read of the whole sample."""
    data = frames[("1h", "autumn")]
    alone, grouped = OLSModel(["aux", "load_d+1", "week_load_d+1"], 28), OLSModel(["load_d+1", "aux"], 56)
    for model in (alone, grouped):
        model._set_freq("1h")
    periods = alone._load_periods(data, ["price", "aux", "load_d+1", "week_load_d+1"], 2, "price")
    grouped._use_periods(periods, data, ["price", "load_d+1", "aux"], 2, "price")
    assert grouped._memo is alone._memo
    first = (pd.Timestamp("2023-09-28").date() - data.index[0].date()).days
    for model in (alone, grouped):
        model._offset, model._last_day, model._newest = first, first + 5, None
        for hour in (0, 2, 23):
            for hz in (1, 2):
                names = model._expand_predictors(hz, hour=hour)
                sample = model._training_sample(hour, first, hz, names)
                mask, without = model._first_day_mask(hour, first, hz, names, model._members[0])
                assert mask.tobytes() == StandardScaler.get_scalable_mask(sample).tobytes()
                assert without.tobytes() == np.flatnonzero(np.isnan(sample).all(axis=0)).tobytes()
    assert not alone._first_day_mask(0, first, 1, ["aux"], 28)[0].any()  # 0/1: left unscaled
    assert len(alone._first_day_mask(0, first, 1, ["week_load_d+1"], 28)[1]) == 1  # no value before 2023-09-30
    spans = {key[1:4] for key in alone._memo if key[0] == "mask"}
    assert len(spans) == 3 * 4  # three periods, each with the rows of two windows at two horizons, kept apart
    other = OLSModel(["aux"], 28)
    other._set_freq("1h")
    other._load_periods(data, ["price", "aux"], 2, "price")
    assert other._memo is not alone._memo and other._memo == {}
    alone._release()
    assert alone._memo == {} and grouped._memo is periods.memo


# --- availability verdicts --------------------------------------------------------------------------------------------


def late_uses(info, data, expanded, **kwargs):
    """`late_uses` as text: a column without a rule has NaN times, which no two uses compare equal on."""
    return [repr(use) for use in info.late_uses(expanded, "price", "1h", data.index, model_name="m", **kwargs)]


def test_verdicts_are_kept_by_what_they_are_computed_from(frames, monkeypatch):
    """A verdict kept for one check answers another only where it was computed from the same: another issue time or zone,
    rule, accepted assumption, data zone, training rows or periods computes its own. Each listing equals the one computed
    without any verdict kept, and a repeated check computes nothing."""
    data = frames[("1h", "spring")].copy()
    data["generation_solar_forecast_d+1"] = data["load_d+1"]
    utc = data.tz_convert("UTC")
    full = {(p, h): ["load_d+1", "load_d+2", "wind_d+1", "settle_d-1", "aux"] for p in range(24) for h in (1, 2)}
    some = {(p, 1): ["load_d+1", "wind_d+1", "settle_d-1"] for p in range(6, 12)}
    solar = {(p, 1): ["generation_solar_forecast_d+1"] for p in range(24)}
    days = ["2023-03-24", "2023-03-25", "2023-03-26", "2023-03-27"]
    first = str(data.index[0].date())
    rows = TrainingRows(28, first_day=first)
    base = information_set("2023-03-10")

    def like(issue_time="11:45", zone="Europe/Brussels", **rules):
        return InformationSet(issue_time, zone, rules={**base.rules, **rules}, unknown="warn", use_defaults=False)

    entsoe = InformationSet(unknown="warn")
    checks = {
        "base": (base, data, full, dict(issue_days=days, training_rows=rows)),
        "fewer days": (base, data, full, dict(issue_days=days[:2], training_rows=rows)),
        "short window": (base, data, full, dict(issue_days=days, training_rows=TrainingRows(7, first_day=first))),
        "no target": (base, data, full, dict(issue_days=days, training_rows=rows, check_target=False)),
        "periods": (base, data, some, dict(issue_days=days, training_rows=rows)),
        "utc index": (base, utc, full, dict(issue_days=days, training_rows=rows)),
        "issue time": (like("09:30"), data, full, dict(issue_days=days, training_rows=rows)),
        "issue zone": (like(zone="Europe/London"), data, full, dict(issue_days=days, training_rows=rows)),
        "rule": (like(load=Availability.published("12:00", 1)), data, full, dict(issue_days=days, training_rows=rows)),
        "defaults": (entsoe, data, solar, dict(issue_days=days)),
        "assumption": (
            InformationSet(assumptions=["entsoe_a69_by_10"], unknown="warn"),
            data,
            solar,
            dict(issue_days=days),
        ),
    }

    def outcomes():
        out = {}
        for name, (info, frame, pairs, kwargs) in checks.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[name] = late_uses(info, frame, pairs, **kwargs)
        return out

    sharing(monkeypatch, False)
    own = outcomes()
    for one, other in [
        ("base", n) for n in ("short window", "periods", "utc index", "issue time", "issue zone", "rule")
    ]:
        assert own[one] != own[other], other  # so a verdict kept for the one would be wrong for the other
    assert own["defaults"] != own["assumption"]
    sharing(monkeypatch, True)
    computed = []
    real = InformationSet._late_of
    monkeypatch.setattr(InformationSet, "_late_of", lambda self, *a: computed.append(1) or real(self, *a))
    assert outcomes() == own
    fresh = len(computed)
    assert outcomes() == own and len(computed) == fresh  # every verdict kept
    monkeypatch.setattr(info_module, "_VERDICTS_KEPT", 3)
    info_module._VERDICTS.clear()
    assert outcomes() == own and len(info_module._VERDICTS) == 3  # the oldest let go


def test_a_check_at_raise_refuses_alike_with_verdicts_kept(frames, monkeypatch):
    """At "raise" a late use raises with the same message whether its verdict was computed or kept."""
    data = frames[("1h", "spring")]
    expanded = {(p, 2): ["load_d+2", "price_d-1"] for p in range(24)}
    messages = []
    for on in (False, True, True):  # computed, computed and kept, then read from what was kept
        if len(messages) < 2:
            sharing(monkeypatch, on)
        with pytest.raises(info_module.InformationSetError) as raised:
            information_set("2023-03-10").check(expanded, "price", "1h", data.index, "2023-03-24", "2023-03-26")
        messages.append(str(raised.value))
    assert messages[0] == messages[1] == messages[2] and "load_d+2" in messages[0]
