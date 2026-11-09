"""Fits from shared statistics for VST and cross-validated models in engine groups (step 5j).

A model with a VST (asinh, npit, mirror-log, polynomial, `_engine_worker.ENGINE_VSTS`) fits from a row set: the scaler's own
`fit_transform` once on the union of the columns that the fits of one span, ρ and spec read, and the exact sums of the
transformed columns over ρ (`_engine_worker._RowSet`). LassoCV and LEAR `"cv"` fit from the KFold blocks of ρ (`_cv_stats`).
`engine="check"` is "on" with a sample of the fits also run on the model path.

Here: the transformed columns and targets are the model path's to the bit (spans past 8,192 rows too); a fit with other rows
gets its own transform; engine against off within the per-class tolerances of the step 5 spec (OLS and LassoCV 1e-8, Ridge
and LEAR 1e-9) with the same kink and penalty, on a grid of the frames of `test_engine.py`, LEAR's defaults and `lago()` on
the Lago fixture, and on arrays; within the engine a model's records are the same bits alone, in any group, shuffled,
chunked, at 1 and 3 processes, 2 task threads, 1 and 8 BLAS threads, any τ, after crash and resume; nothing after D moves a
forecast, `verify` is ok with its control, a column one model reads moves no other model; hand-backs are the model path's;
`"check"`'s records, counts and warning. The tolerance sweep, the mutants and the measurement are in
`epftoolbox2-research/library_check_step5/engine_vst_cv/`.
"""

import datetime
import functools
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from threadpoolctl import threadpool_limits

from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import LassoCVModel, LEARModel, NaiveModel, OLSModel, RidgeModel
from epftoolbox2.models import _engine, _engine_worker
from epftoolbox2.models._worker import StatsFit, TaskInputs, finite_rows
from epftoolbox2.models.lear import LAGO_SCALER
from epftoolbox2.pipelines import ModelPipeline, Workflow
from epftoolbox2.scalers.registry import make_scaler
from tests.models import test_engine
from tests.models.test_engine import (
    INFO,
    POOL,
    TOLERANCE,
    build,
    compare,
    fits_of,
    flag_while_aux,
    keep_first_day,
    local_dates,
    moved,
    same_files,
    text,
)
from tests.models.test_engine_groups import Crashing, InProcessPool, by_weekday, lines, with_aux
from tests.synthetic import PREDICTORS, make_market_data

STANDARD = {"name": "standard", "params": {"scale_binary": True}}
PUBLISHED = {"price": Availability.published("13:00", 1)}
NAMES = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "aux"]


def frame_of(name):
    """`test_engine.py`'s frame fixture `name`, built here (a module-scoped fixture of this module)."""
    return getattr(test_engine, name).__wrapped__()


@pytest.fixture(scope="module", name="spring")
def spring_frame():
    return frame_of("spring")


@pytest.fixture(scope="module", name="autumn")
def autumn_frame():
    return frame_of("autumn")


@pytest.fixture(scope="module", name="quarter")
def quarter_frame():
    return frame_of("quarter")


@pytest.fixture(scope="module", name="daily")
def daily_frame():
    return frame_of("daily")


@pytest.fixture(scope="module", name="wide")
def wide_frame():
    return frame_of("wide")


@pytest.fixture(scope="module", name="market")
def market_frame():
    return frame_of("market")


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.delenv("ENGINE_MAX_MODELS", raising=False)
    monkeypatch.delenv("ENGINE_MAX_COLUMNS", raising=False)


@pytest.fixture
def in_process(monkeypatch):
    monkeypatch.setattr(_engine, "ProcessPoolExecutor", InProcessPool)
    with threadpool_limits(limits=1):  # what a pool's worker runs on
        yield


@pytest.fixture
def every_fit_checked(monkeypatch):
    monkeypatch.setattr(_engine_worker, "CHECK_EVERY", 1)
    # The manifests of a "check" run carry `engine_check`: provenance, like `engine`.
    monkeypatch.setattr(test_engine, "PROVENANCE", (*test_engine.PROVENANCE, "engine_check"))


def manifest(save_dir, name):
    return json.loads((Path(save_dir) / f"{name}.manifest.json").read_text())


# --- a row set's transform is the model path's ----------------------------------------------------------------------

VSTS = [
    "asinh",
    "npit",
    "mlog",
    "poly",
    {"name": "asinh", "params": {"standardise": False}},  # LEAR.lago()'s "Invariant"
    {"name": "mlog", "params": {"apply_to": ["c1", "c3*"], "c": 0.5}},
    {"name": "poly", "params": {"apply_to": "target", "lam": 0.3}},
]


def union_columns(rng, n):
    """Market-like columns: load, wind, a dummy, a spiky price, a column constant up to rounding, solar with zeros, a column of
    tiny spread far from 0, a 0/1 column constant on the first half; NaN cells in two of them."""
    x = np.column_stack(
        [
            rng.normal(1000, 200, n),
            rng.gamma(2.0, 50.0, n),
            (rng.random(n) < 2 / 7).astype(float),
            rng.standard_t(2, n) * 30 + 50,
            0.3 + 1e-16 * rng.normal(size=n),
            np.where(rng.random(n) < 0.4, 0.0, rng.gamma(1.0, 100.0, n)),
            7.0 + 1e-3 * rng.normal(size=n),
            np.r_[np.zeros(n // 2), (rng.random(n - n // 2) < 0.5).astype(float)],
        ]
    )
    x[(rng.random(n) < 0.02) | (np.arange(n) == 3), 1] = np.nan
    x[(rng.random(n) < 0.01) | (np.arange(n) == 7), 5] = np.nan
    y = 40 + 0.05 * x[:, 0] + rng.standard_t(2, n) * 8
    y[rng.random(n) < 0.01] = np.nan
    return x, y


@pytest.mark.parametrize("n", [60, 1449, 9000])
@pytest.mark.parametrize("spec", VSTS, ids=[json.dumps(s) for s in VSTS])
def test_a_row_sets_transform_is_the_model_paths_to_the_bit(n, spec):
    """Fits of one span read 1-6 of its 8 columns, each with its mask flag (one fit flips a flag): grouped by ρ, each row set's
    one `fit_transform` on the union gives every fit's transformed columns, target and issue row, and its inverse, bitwise as
    the scaler on that fit's own rows (`_fit_member`'s call), past numpy's 8,192-element buffer too."""
    rng = np.random.default_rng(n)
    x, y = union_columns(rng, n + 1)
    issue = x[-1].copy()
    issue[np.isnan(issue)] = 1.0
    x, y = x[:-1], y[:-1]
    names = [f"c{j}" for j in range(x.shape[1])]
    arrays = {"x": x, "y": y[:, None], "col_index": {name: j for j, name in enumerate(names)}}
    flags = make_scaler("standard").get_scalable_mask(x)
    fits = [sorted(rng.choice(8, size=rng.integers(1, 7), replace=False).tolist()) for _ in range(12)]
    fits += [[0, 2, 3, 4, 6, 7], [0, 3]]
    masks = [flags[cols].copy() for cols in fits]
    masks[-1][0] = False  # the same column unscaled in this fit
    spec_key = json.dumps(_engine_worker.scaler_spec(spec), sort_keys=True)
    sets = {}
    for cols, mask in zip(fits, masks):
        key = _engine_worker._rowset_key(0, 1, slice(0, n), finite_rows(x[:, cols], y), spec_key)
        sets.setdefault(key, set()).update(zip(cols, mask.tolist()))
    names_array = np.array(names, dtype=object)
    built = {key: _engine_worker._RowSet(arrays, key, sorted(virtual), names_array) for key, virtual in sets.items()}
    assert len(built) > 1  # NaN cells: fits of other rows have other row sets
    for cols, mask in zip(fits, masks):
        rho = finite_rows(x[:, cols], y)
        rowset = built[_engine_worker._rowset_key(0, 1, slice(0, n), rho, spec_key)]
        local = rowset.local(cols, mask)
        scaler = make_scaler(spec)
        z, zy, test = scaler.fit_transform(
            x[rho][:, cols].copy(),
            y[rho].copy(),
            issue[None, cols].copy(),
            scalable_mask=mask,
            columns=[names[c] for c in cols],
        )
        assert np.array_equal(z.view(np.int64), rowset.Z[:, local].view(np.int64)), (cols, mask)
        assert np.array_equal(zy.view(np.int64), rowset.zy.view(np.int64)), cols
        assert np.array_equal(test[0].view(np.int64), rowset.transform(local, issue[cols]).view(np.int64)), cols
        for value in (-2.5, 0.0, 0.7, 4.0):
            assert scaler.inverse(value) == rowset.scaler.inverse(value)


# --- engine against off ---------------------------------------------------------------------------------------------


def spring_models():
    return [
        LEARModel([*PREDICTORS, "price_d-1"], [28, 56], name="lear_asinh_list"),
        LEARModel([*PREDICTORS, "price_d-1", "aux"], 56, scaler="npit", refit="7D", name="lear_npit_7d"),
        LEARModel(with_aux, 28, scaler="mlog", binary_mask="fit", name="lear_mlog_function_fit"),
        LEARModel([*PREDICTORS, "price_d-1"], 56, criterion="cv", name="lear_cv_asinh"),
        LEARModel(
            [*PREDICTORS, "price_d-1"], [28, 56], criterion="cv", scaler="standard", refit="auto", name="lear_cv_auto"
        ),
        LassoCVModel([*PREDICTORS, "aux"], 28, cv=3, name="lasso_aux"),
        LassoCVModel([*PREDICTORS, "price_d-1"], 56, scaler="asinh", refit="7D", name="lasso_asinh_7d"),
        RidgeModel([*PREDICTORS, "price_d-1"], [14, 28], scaler="asinh", name="ridge_asinh_list"),
        OLSModel(
            [*PREDICTORS, "price_d-1"], 28, scaler={"name": "poly", "params": {"apply_to": ["price*"]}}, name="ols_poly"
        ),
        # A hand-back: a fold's training part of no more rows than columns (LassoCV declines).
        LassoCVModel(PREDICTORS, 5, cv=3, name="lasso_short"),
        # The model path in the same group: LEAR's coordinate descent, Naive.
        LEARModel([*PREDICTORS, "price_d-1"], 28, criterion="aic", solver="cd", name="lear_cd"),
        NaiveModel(["price_d-1"], name="naive"),
    ]


def autumn_models():
    return [
        LEARModel(
            [*NAMES, "price_d-1"], [28, 120], refit="auto", newest_target="published", name="lear_auto_published"
        ),
        LassoCVModel(by_weekday, 28, cv=3, refit="7D", scaler="npit", name="lasso_npit_function_7d"),
        LEARModel(
            [*NAMES, "price_d-1"],
            56,
            criterion="cv",
            newest_target="published",
            scaler=LAGO_SCALER,
            name="lear_cv_invariant",
        ),
        RidgeModel(NAMES, 28, scaler="mlog", binary_mask="fit", name="ridge_mlog_fit"),
        NaiveModel(["price_d-1"], name="naive"),
    ]


def quarter_models():
    return [
        LEARModel(PREDICTORS, [7, 14], refit="auto", name="lear_auto"),
        LassoCVModel([*PREDICTORS, "aux"], 14, cv=3, binary_mask="fit", scaler="asinh", name="lasso_asinh_fit"),
        OLSModel(with_aux, 14, scaler="npit", name="ols_npit_function"),
    ]


def daily_models():
    return [
        LEARModel([*NAMES, "price_d-1"], [42, 56], name="lear_list"),
        LassoCVModel(NAMES, 56, cv=3, fit_intercept=False, scaler=STANDARD, refit="7D", name="lasso_noint_7d"),
        LEARModel(by_weekday, 56, criterion="cv", refit="7D", scaler="poly", name="lear_cv_poly_7d"),
        RidgeModel(NAMES, [42, 90], refit="auto", scaler="npit", name="ridge_npit_auto"),
    ]


# Each cell: the frame, the pipeline, the run's arguments (`test_engine.CELLS`' frames and runs). Every class with a VST, LEAR
# and LassoCV with CV blocks, each with refit None, "7D" and "auto", a window list, a function, `binary_mask="fit"`, every
# VST and `standardise=False`, 1h, 15-min and 1D, validation, Fridays and published in some cell.
CELLS = {
    "1h-spring-plain-jsonl": ("spring", spring_models, None, dict(test=("2023-03-25", "2023-03-27"), horizon=2)),
    "1h-autumn-fridays-published-parquet": (
        "autumn",
        autumn_models,
        PUBLISHED,
        dict(test=("2023-10-27", "2023-11-03"), horizons=[1, 3], issue_days=["friday"], save_format="parquet"),
    ),
    "15min-autumn-validation-parquet": (
        "quarter",
        quarter_models,
        None,
        dict(
            validation=("2023-10-26", "2023-10-26"),
            test=("2023-10-28", "2023-10-29"),
            horizon=1,
            freq="15min",
            save_format="parquet",
        ),
    ),
    "1D-fridays-published-jsonl": (
        "daily",
        daily_models,
        PUBLISHED,
        dict(test=("2023-05-05", "2023-06-16"), horizons=[1, 3, 7], freq="1D", issue_days=["friday"]),
    ),
}


def run_off_and_check(build_, data, tmp_path, **arguments):
    reports = {}
    for engine in ("off", "check"):
        reports[engine] = build_(engine).run(data, save_dir=str(tmp_path / engine), **arguments)
    return reports["off"], reports["check"]


@pytest.mark.parametrize("cell", list(CELLS))
def test_engine_within_tolerance_of_off(cell, request, tmp_path, every_fit_checked):
    """Engine against off on every key (`test_engine.compare`: the same keys and fields, each forecast within its class's
    tolerance, LEAR's kink, the model path's models to the bit); run as `engine="check"` with every fit checked, whose counts
    say every penalty and kink was the model path's and every checked forecast within tolerance."""
    frame, models, rules, arguments = CELLS[cell]
    data = request.getfixturevalue(frame)
    off, on = run_off_and_check(build(models, rules), data, tmp_path, **arguments)
    seen = compare(tmp_path / "off", tmp_path / "check", models())
    pd.testing.assert_frame_equal(off.forecast_counts(), on.forecast_counts(), check_exact=True)
    stats = [m for m in models() if test_engine.expected_path(m) is not None]
    assert stats and all(_engine_worker.fit_path_of(m) for m in stats)
    for model in stats:
        stored = manifest(tmp_path / "check", model.name)
        engine, check = stored["engine"], stored["engine_check"]
        assert engine["fits"] + sum(engine["fallbacks"].values()) > 0, model.name
        assert check["fits"] <= engine["fits"] and check["other_choice"] == 0, (model.name, check)
        assert check["max_abs"] <= TOLERANCE[test_engine.cls_of(model)], (model.name, check)
        if engine["fits"]:
            assert check["fits"] > 0 and seen[model.name][2] < seen[model.name][0], (
                model.name
            )  # not the model path's bits
    if cell.startswith("1h-spring"):
        records, _, bitwise = seen["lasso_short"]
        assert bitwise == records
        engine = manifest(tmp_path / "check", "lasso_short")["engine"]
        assert engine["fits"] == 0 and set(engine["fallbacks"]) == {"declined"}, engine
        assert "fit_path" not in manifest(tmp_path / "check", "lear_cd")


def test_two_fits_on_other_rows_get_their_own_transform(spring, tmp_path, in_process, monkeypatch):
    """Asinh LEARs in one span, two reading a column with missing values there: their fits have other rows ρ, so other
    row sets (each built once per span, ρ and spec, holding the columns of every fit of its rows), and each model's records
    are those it has alone."""
    built = []
    real = _engine_worker._RowSet.__init__

    def spy(self, arrays, key, virtual, names):
        built.append((key, {names[c] for c, _ in virtual}))
        real(self, arrays, key, virtual, names)

    monkeypatch.setattr(_engine_worker._RowSet, "__init__", spy)

    def models():
        # A fit of other rows between two fits of the same rows: the first set is alive while the other is asked for.
        return [
            LEARModel([*PREDICTORS, "price_d-1"], 28, name="reads_load"),
            LEARModel(["wind_d+1", "is_weekend_d+1", "price_d-1"], 28, name="no_load"),
            LEARModel([*PREDICTORS, "price_d-1"], 28, criterion="aic", name="reads_load_aic"),
            LEARModel(["wind_d+1", "price_d-1"], 28, name="no_load_fewer"),
        ]

    def run(save_dir, which=None):
        pipeline = ModelPipeline(engine="on")
        for model in models():
            if which is None or model.name == which:
                pipeline.add_model(model)
        return pipeline.run(spring, "2023-03-28", "2023-03-29", horizon=1, save_dir=str(save_dir))

    run(tmp_path / "group")
    spans = Counter(key[:4] for key, _ in built)
    rows = {}
    for key, virtual in built:
        rows.setdefault(key[:4], set()).add(key[4])
    split = [span for span, sets in rows.items() if len(sets) > 1]
    assert split and all(spans[span] == len(rows[span]) for span in spans)  # one set per (span, ρ): built once each
    for key, held in built:
        if key[:4] in split and "load_d+1" not in held:  # the two "no_load" lists' columns: one transform
            assert held == {"wind_d+1", "is_weekend_d+1", "price_d-1"}, held
    for name in ("reads_load", "no_load", "reads_load_aic", "no_load_fewer"):
        run(tmp_path / name, name)
        assert same_files(tmp_path / name, tmp_path / "group", [name]) == 1


# --- LEAR's defaults and lago() on the Lago fixture -------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def lago_frame():
    from tests.fixtures.lago.frames import HERE, FixtureSource, lear_frame

    raw = pd.read_parquet(HERE / "data_DE.parquet")
    return lear_frame(FixtureSource("DE"), str(raw.index[0].date()), str(raw.index[-1].date()))


# The short members of LEAR's defaults ([56, 84, 728, "expanding"]) and of `lago()` ([49, 77, 1085, 1449]) with `solver="lars"`:
# the long ones and the whole lists run outside CI (`engine_vst_cv/lago_fixture.py`), a LEAR fit of 247 columns over 1,449 rows
# costing about a second with the numpy kernels.
LAGO_CASES = [("defaults", 56), ("defaults", 84), ("lago_lars", 49), ("lago_lars", 77)]


@pytest.mark.parametrize("kind, window", LAGO_CASES, ids=[f"{k}-{w}" for k, w in LAGO_CASES])
def test_lear_on_the_lago_fixture(kind, window, tmp_path, every_fit_checked):
    """Over Lago's 247 DE inputs on the fixture's issue day: LEAR's default settings (asinh, BIC, lars) and `lago()`'s
    (the Invariant scaler, AIC, published prices) with `solver="lars"`, one training window: every fit from statistics, each
    also run on the model path (`engine="check"`, every fit), with the model path's kink and within 1e-9 EUR/MWh. `lago()`
    itself (coordinate descent) stays on the model path."""
    from epftoolbox2.presets import lear_predictors
    from tests.fixtures.lago.frames import EXOGENOUS, FIXTURE_ISSUE_DAYS

    data = lago_frame()
    day = FIXTURE_ISSUE_DAYS["DE"]
    info = InformationSet(
        "11:45", "Europe/Brussels", rules={"price": Availability.published("13:00", 1, timezone="Etc/GMT-1")}
    )
    if kind == "defaults":
        model = LEARModel(lear_predictors(EXOGENOUS["DE"], "price", order="lago"), window, name=kind)
    else:
        model = LEARModel.lago(EXOGENOUS["DE"], name=kind)
        assert _engine_worker.fit_path_of(model) is None
        model.solver = "lars"
        model.training_window = window
    pipeline = ModelPipeline(information_set=info, availability="none", engine="check").add_model(model)
    pipeline.run(data, day, day, horizon=1, save_dir=str(tmp_path))
    stored = manifest(tmp_path, kind)
    engine, check = stored["engine"], stored["engine_check"]
    assert stored["fit_path"] == "engine-1" and engine == {
        "fits": 24,
        "fallbacks": {},
        "max_issue_z": engine["max_issue_z"],
    }
    assert check["fits"] == 24 and check["other_choice"] == 0 and check["max_abs"] <= 1e-9, check


# --- the same penalty on arrays ---------------------------------------------------------------------------------------


def engine_on_arrays(model, x, y, mask, tests):
    """`EngineFits` on one span of rows `x`, `y` (the issue rows `tests` after them, never training rows): (forecast per row
    or None, the counts, the engine's fit)."""
    n, p = x.shape
    arrays = {
        0: {
            "x": np.vstack([x, tests]),
            "y": np.r_[y, np.full(len(tests), np.nan)][:, None],
            "days": np.arange(n + len(tests)),
            "col_index": {f"c{j}": j for j in range(p)},
        }
    }
    spec = model.scaler
    scaler = make_scaler(spec)
    inputs = TaskInputs(arrays, {}, {"scaler": spec}, model)
    if type(scaler).__name__ == "StandardScaler":
        model_inputs = _engine_worker.ModelInputs("m", inputs, bool(scaler.scale_binary))
        key = (0, 1, 0, n)
    else:
        vst = json.dumps(_engine_worker.scaler_spec(spec), sort_keys=True)
        model_inputs = _engine_worker.ModelInputs("m", inputs, None, vst)
        key = _engine_worker._rowset_key(0, 1, slice(0, n), np.ones(n, dtype=bool), vst)
    if "blocks" in model._ENGINE_NEEDS:
        model_inputs = model_inputs._replace(folds=_engine_worker.cv_folds(model))
    counts = _engine_worker.EngineCounts()
    fits = _engine_worker.EngineFits(
        _engine_worker._Spans(arrays, {key: np.arange(p)}, Counter()), model_inputs, counts
    )
    made = []
    real = model._fit_stats
    model._fit_stats = lambda stats: made.append(real(stats)) or made[-1]
    try:
        out = fits(StatsFit(0, 1, slice(0, n), list(range(p)), np.ones(n, dtype=bool), x, y, mask, tests))
    finally:
        del model._fit_stats
    return out, counts, (made[-1] if made else None)


def model_path_on_arrays(model, x, y, mask, tests):
    scaler = make_scaler(model.scaler)
    tx, ty, tt = scaler.fit_transform(
        x.copy(), y.copy(), tests.copy(), scalable_mask=mask, columns=[f"c{j}" for j in range(x.shape[1])]
    )
    fitted = model._fit_model(tx, ty)
    return [scaler.inverse(float(model._predict_fitted(fitted, tt[i : i + 1])[0])) for i in range(len(tt))], fitted


def market_design(rng, n):
    load = rng.normal(1000, 200, n)
    wind = rng.gamma(2.0, 50.0, n)
    weekend = (rng.random(n) < 2 / 7).astype(float)
    noise = rng.normal(0, 1, (n, int(rng.integers(1, 6))))
    y = 20 + 0.08 * load - 0.15 * wind + 10 * weekend + rng.normal(0, 5, n)
    spikes = rng.random(n) < 0.03
    y[spikes] *= rng.uniform(3, 8, spikes.sum())
    lag = np.roll(y, 1) + rng.normal(0, 3, n)
    x = np.column_stack([load, wind, weekend, 1 - weekend, lag, noise])
    mask = np.ones(x.shape[1], dtype=bool)
    mask[2:4] = False
    return x, y, mask


ARRAY_MODELS = {
    "lasso": (LassoCVModel([], 28), 1e-8),
    "lasso_cv3_noint": (LassoCVModel([], 28, cv=3, fit_intercept=False, scaler=STANDARD), 1e-8),
    "lasso_asinh": (LassoCVModel([], 28, scaler="asinh"), 1e-8),
    "lear_cv": (LEARModel([], 28, criterion="cv", scaler="standard"), 1e-9),
    "lear_cv_asinh": (LEARModel([], 28, criterion="cv"), 1e-9),
    "lear_bic_npit": (LEARModel([], 28, scaler="npit"), 1e-9),
}


@pytest.mark.parametrize("name", list(ARRAY_MODELS))
def test_the_same_penalty_as_the_model_path_on_arrays(name):
    """40 market-like designs (a one-hot pair, a spiky target, n 40 to 400): the engine's fit chooses the model path's
    penalty (LassoCV: its α; LEAR "cv": its position in the grid; BIC: its kink), and each forecast of a training row and of
    rows 3 training sds out is within the class's tolerance, wherever the engine keeps it."""
    model, tolerance = ARRAY_MODELS[name]
    rng = np.random.default_rng(len(name))
    kept = 0
    for _ in range(40):
        n = int(rng.choice([40, 91, 200, 400]))
        x, y, mask = market_design(rng, n)
        sd = x.std(axis=0, ddof=1)
        tests = np.array([x[n // 2], x[n // 3] + 3 * sd * mask, x[n // 4] - 3 * sd * mask])
        off, own = model_path_on_arrays(model, x, y, mask, tests)
        out, counts, fitted = engine_on_arrays(model, x, y, mask, tests)
        if out is None:
            continue
        assert not _engine_worker._other_choice(fitted, own), name
        for made, theirs in zip(out, off):
            if made is not None:
                assert abs(made[0] - theirs) <= tolerance, (name, n, abs(made[0] - theirs))
                kept += 1
    assert kept >= 60, kept


def test_a_vst_forecast_where_the_inverse_is_steep_goes_back(monkeypatch):
    """OLS with asinh, an issue row whose load is 20 training sds up (within OLS's distance bound): its forecast lies where
    the inverse's slope is more than `VST_SLOPE_MAX` times its median at the training targets, so that row is forecast on
    the model path (`far_issue`), the fit's other row from statistics; without the bound the engine keeps it."""
    rng = np.random.default_rng(11)
    x, y, mask = market_design(rng, 364)
    sd = x.std(axis=0, ddof=1)
    tests = np.array([x[100], x[100] + 20 * sd * np.eye(x.shape[1])[0]])
    model = OLSModel([], 28, scaler="asinh")
    scaler = make_scaler("asinh")
    tx, zy, tt = scaler.fit_transform(x.copy(), y.copy(), tests.copy(), scalable_mask=mask)
    steep = _engine_worker.inverse_slopes(scaler, zy)
    forecast = float(model._predict_fitted(model._fit_model(tx, zy), tt[1:])[0])  # the model path's, scaled
    out, counts, _ = engine_on_arrays(model, x, y, mask, tests)
    assert _engine_worker.inverse_slope(scaler, [forecast]) > _engine_worker.VST_SLOPE_MAX * np.median(steep)
    assert out[0] is not None and out[1] is None and counts.fallbacks == {"far_issue": 1}
    monkeypatch.setattr(_engine_worker, "VST_SLOPE_MAX", np.inf)
    out, counts, _ = engine_on_arrays(model, x, y, mask, tests)
    assert out[1] is not None and not counts.fallbacks


def test_near_twin_columns_send_a_lassocv_or_lear_fit_back():
    """A column and a copy of it plus noise of 1e-5 of its sd (1 - |cos| about 5e-11): LassoCV and LEAR, "cv" and BIC, hand
    the fit back (`near_twin`); an exact copy (a twin, which LEAR drops) or the one-hot pair of every design sends nothing
    back."""
    rng = np.random.default_rng(12)
    x, y, mask = market_design(rng, 200)
    near = np.column_stack([x, x[:, 0] + 1e-5 * x[:, 0].std() * rng.normal(size=len(x))])
    exact = np.column_stack([x, x[:, 0]])
    mask = np.append(mask, True)
    tests = near[:1]
    for model, backs in (
        (LassoCVModel([], 28), True),
        (LEARModel([], 28, criterion="cv", scaler="standard"), True),
        (LEARModel([], 28, scaler="standard"), True),
    ):
        for design, twin in ((near, True), (exact, False)):
            out, counts, _ = engine_on_arrays(model, design, y, mask, tests)
            handed = counts.fallbacks.get("near_twin", 0) == 1
            assert handed == (backs and twin) and (out is None) == handed, (
                model.criterion if hasattr(model, "criterion") else "lasso",
                twin,
                counts.fallbacks,
            )
    assert _engine_worker.fit_path_of(LEARModel([], 28, criterion="cv")) and LEARModel([], 28)._ENGINE_NEAR_TWIN == 1e-5


def review_design(seed=101, trial=36):
    """The 5j review's near-twin design (`scratchpad/review5j/repro_trial.py`, replayed from its seed): Gaussian columns, a
    near twin of the first (noise 3e-3 of its sd, 1 - |cos| 4.5e-6) whose difference the target reads, n 728, p 4; and issue
    rows with that difference k of its sds out, by k."""
    rng = np.random.default_rng(seed)
    ks = [2, 3, 4, 5, 6, 7, 8, 9, 9.5, 10, 11]
    for _ in range(trial + 1):
        n = int(rng.choice([56, 91, 364, 728]))
        p = int(min(rng.choice([3, 5, 10, 15, 25]), n - 8))
        base = rng.normal(0, 1, (n + 1, p))
        for j in range(1, p, 3):
            base[:, j] = base[:, j - 1] + rng.choice([0.05, 0.3, 1.0]) * base[:, j]
        scale = 10.0 ** rng.uniform(-2, 4, p)
        mean = rng.normal(0, 1, p) * scale * rng.choice([0.0, 1.0, 50.0])
        x = base * scale + mean
        y = 50 + (base @ rng.normal(0, 1, p)) * 10 + rng.normal(0, 5, n + 1)
        ratio = float(rng.choice([1e-2, 3e-3, 1e-3, 3e-4]))
        sd = x[:n, 0].std(ddof=1)
        e = rng.normal(0, ratio * sd, n + 1)
        y = y + 3.0 * e / (ratio * sd)
        x = np.column_stack([x, x[:, 0] + e])
        issue = x[n].copy()
        x, y = x[:n], y[:n]
        d = x[:, -1] - x[:, 0]
        rows = {0: issue.copy()}
        for k in ks:
            for s in (-1.0, 1.0):
                r = issue.copy()
                r[-1] = r[0] + d.mean() + s * k * d.std(ddof=1)
                rows[s * k] = r
        for _ in ks:  # the review's single-column rows: drawn, not used, to keep its random stream
            rng.integers(0, x.shape[1])
            rng.choice([-1.0, 1.0])
    return x, y, rows


def test_the_reviews_near_twin_design_is_handed_back_or_within_1e_8():
    """The review's design, where every other bound passes at the row 8 of the pair's sds out (distance 9.66, slope 2.92 times
    the median, condition 6.7e5) and the engine's LEAR (asinh, BIC) forecast was 1.4e-8 from the model path's: now the fit
    goes back to the model path (its pair is a near twin), or every forecast it keeps is within LEAR's 1e-8."""
    x, y, rows = review_design()
    assert x.shape == (728, 4)
    tests = np.array([rows[0], rows[-8.0], rows[8.0], rows[-7.0]])
    mask = np.ones(x.shape[1], dtype=bool)
    model = LEARModel([], 28)
    off, _ = model_path_on_arrays(model, x, y, mask, tests)
    out, counts, _ = engine_on_arrays(model, x, y, mask, tests)
    if out is None:
        assert counts.fallbacks == {"near_twin": 1}
    else:
        for made, theirs in zip(out, off):
            assert made is None or abs(made[0] - theirs) <= 1e-8


def test_a_cv_fit_of_fewer_rows_than_folds_goes_back():
    """Four rows for LEAR's five folds, two for a LassoCV of three: the engine hands the fit back (`few_rows`), where the
    model path refuses it as scikit-learn and KFold do (a fit_error)."""
    rng = np.random.default_rng(3)
    for model, n in ((LEARModel([], 28, criterion="cv"), 4), (LassoCVModel([], 28, cv=3), 2)):
        x, y = rng.normal(size=(n, 3)), rng.normal(size=n) + 50
        out, counts, fitted = engine_on_arrays(model, x, y, np.ones(3, dtype=bool), x[:1])
        assert out is None and fitted is None and counts.fallbacks == {"few_rows": 1}
        with pytest.raises(ValueError):
            model_path_on_arrays(model, x, y, np.ones(3, dtype=bool), x[:1])


# --- engine="check" -----------------------------------------------------------------------------------------------------


def check_models():
    return [
        LassoCVModel([*PREDICTORS, "price_d-1"], 28, cv=3, name="lasso"),
        LEARModel([*PREDICTORS, "price_d-1"], [28, 56], name="lear"),
        OLSModel(PREDICTORS, 28, name="ols"),
        NaiveModel(["price_d-1"], name="naive"),
    ]


def run_check(data, save_dir, engine="check", models=check_models):
    pipeline = ModelPipeline(engine=engine)
    for model in models():
        pipeline.add_model(model)
    return pipeline.run(data, "2023-03-24", "2023-03-25", horizon=1, save_dir=str(save_dir))


def test_check_keeps_the_engines_records_and_counts_a_sample(spring, tmp_path, in_process, recwarn, monkeypatch):
    """`engine="check"`'s records and manifests (but `engine_check`) are "on"'s; the manifest of each model fitted from
    statistics records a sample of its fits run on the model path too, about one span in `CHECK_EVERY`, every one within
    tolerance and with the model path's choice; nothing warns. A model on the model path has none; a later "on" run that
    fits drops it."""
    monkeypatch.setattr(_engine_worker, "CHECK_EVERY", 10)
    run_check(spring, tmp_path / "on", engine="on")
    run_check(spring, tmp_path / "check")
    assert not [w for w in recwarn if "engine check" in str(w.message)]
    for model in check_models():
        assert lines(tmp_path / "on" / f"{model.name}.jsonl") == lines(tmp_path / "check" / f"{model.name}.jsonl")
        on, check = manifest(tmp_path / "on", model.name), manifest(tmp_path / "check", model.name)
        assert "engine_check" not in on
        sample = check.pop("engine_check", None)
        assert {k: v for k, v in on.items() if k not in test_engine.PROVENANCE} == {
            k: v for k, v in check.items() if k not in test_engine.PROVENANCE
        }
        if model.name == "naive":
            assert sample is None
            continue
        assert set(sample) == {"fits", "max_abs", "max_rel", "other_choice", "nonfinite", "model_path_errors"}
        assert sample["nonfinite"] == 0 and sample["model_path_errors"] == 0
        assert 0 < sample["fits"] < check["engine"]["fits"] / 3, (model.name, sample, check["engine"])
        assert sample["max_abs"] <= TOLERANCE[test_engine.cls_of(model)] and sample["other_choice"] == 0
        assert 0.0 <= sample["max_rel"] <= sample["max_abs"]
    keep_first_day(tmp_path / "check" / "lasso.jsonl")
    run_check(spring, tmp_path / "check", engine="on")
    assert "engine_check" not in manifest(tmp_path / "check", "lasso")
    assert "engine_check" in manifest(tmp_path / "check", "ols")  # nothing fitted: the last fitting run's counts stay


class Off(LassoCVModel):
    """A mutant: LassoCV whose fits from statistics forecast 1e-6 of a scaled unit off (its intercept moved)."""

    def _fit_predict(self, train_x, train_y, test_x):
        return super()._fit_predict(train_x, train_y, test_x)

    def _fit_model(self, train_x, train_y):
        return super()._fit_model(train_x, train_y)

    def _predict_fitted(self, fitted, test_x):
        return super()._predict_fitted(fitted, test_x)

    def _fit_stats(self, stats):
        fitted = super()._fit_stats(stats)
        if fitted is not None:
            fitted.intercept_ += 1e-6
        return fitted


def test_check_warns_about_a_model_over_its_tolerance(spring, tmp_path, in_process, every_fit_checked):
    """A mutant whose fits from statistics are 1e-6 off: `"check"` records the gap and warns, naming that model alone."""

    def models():
        return [Off([*PREDICTORS, "price_d-1"], 28, cv=3, name="mutant"), OLSModel(PREDICTORS, 28, name="ols")]

    with pytest.warns(UserWarning, match=r"^mutant: engine check: .* above the 1e-08 of its class") as caught:
        run_check(spring, tmp_path / "check", models=models)
    assert [str(w.message).split(":")[0] for w in caught if "engine check" in str(w.message)] == ["mutant"]
    sample = manifest(tmp_path / "check", "mutant")["engine_check"]
    assert sample["fits"] == manifest(tmp_path / "check", "mutant")["engine"]["fits"] and sample["max_abs"] > 1e-8
    assert manifest(tmp_path / "check", "ols")["engine_check"]["max_abs"] <= 1e-8


def test_a_check_gap_that_isnt_finite_is_counted_and_warned_about():
    """A NaN on one side is the largest gap of all: a later finite pair or a merge in either order can't hide it (`nonfinite`),
    and the check warns about the model, whatever its finite gaps."""
    import math

    one = _engine_worker.EngineCounts(check=True)
    one.checked([(50.0, math.nan), (50.0, 50.5)], other_choice=False)
    assert one.check_dict()["nonfinite"] == 1 and one.check_dict()["max_abs"] == 0.5
    a, b = _engine_worker.EngineCounts(check=True), _engine_worker.EngineCounts(check=True)
    a.checked([(50.0, math.nan)], False)
    b.checked([(50.0, 50.5)], False)
    for first, second in ((a, b), (b, a)):
        total = _engine_worker.EngineCounts()
        total.add(first)
        total.add(second)
        assert total.check_dict()["nonfinite"] == 1 and total.check_dict()["max_abs"] == 0.5
    small = _engine_worker.EngineCounts(check=True)
    small.checked([(50.0, math.inf), (50.0, 50.0)], False)
    with pytest.warns(UserWarning, match=r"^ols: engine check: .* and 1 not finite"):
        _engine._warn_over_tolerance(OLSModel(PREDICTORS, 28, name="ols"), small.check_dict())


class PathRaises(LassoCVModel):
    """LassoCV whose model-path fit raises (as scikit-learn does on a degenerate fold), while its fits from statistics run."""

    def _fit_predict(self, train_x, train_y, test_x):
        raise ValueError("planted: the model path refuses this fit")

    def _fit_model(self, train_x, train_y):
        raise ValueError("planted: the model path refuses this fit")

    def _predict_fitted(self, fitted, test_x):
        return super()._predict_fitted(fitted, test_x)

    def _fit_stats(self, stats):
        return super()._fit_stats(stats)


def test_check_counts_a_fit_the_model_path_refuses(spring, tmp_path, in_process, every_fit_checked):
    """A sampled fit the engine forecasts but the model path can't fit (with engine off it would be a fit_error skip, so the
    forecast counts differ): `engine_check.model_path_errors` counts it, and the check warns."""
    with pytest.warns(
        UserWarning, match=r"^refuses: engine check: \d+ of \d+ fits checked were forecast .* model path raises"
    ):
        run_check(spring, tmp_path / "check", models=lambda: [PathRaises(PREDICTORS, 28, cv=3, name="refuses")])
    stored = manifest(tmp_path / "check", "refuses")
    assert stored["engine_check"]["model_path_errors"] == stored["engine"]["fits"] > 0


def test_a_lear_subclass_sets_its_own_near_twin_bound():
    """`_ENGINE_NEAR_TWIN` set on a LEAR subclass's class wins over LEAR's, as on a LassoCV subclass, with any criterion; LEAR
    itself has 1e-5 with every criterion, also after the criterion changes."""

    class WiderLear(LEARModel):
        _ENGINE_NEAR_TWIN = 1e-4

    class WiderLasso(LassoCVModel):
        _ENGINE_NEAR_TWIN = 1e-4

    assert WiderLear([], 28, criterion="cv")._ENGINE_NEAR_TWIN == 1e-4 == WiderLasso([], 28)._ENGINE_NEAR_TWIN
    assert WiderLear([], 28)._ENGINE_NEAR_TWIN == 1e-4
    model = LEARModel([], 28)
    for criterion in ("bic", "cv", "aic"):
        model.criterion = criterion
        assert model._ENGINE_NEAR_TWIN == 1e-5 == LassoCVModel([], 28)._ENGINE_NEAR_TWIN


def test_check_in_yaml_and_a_workflow(tmp_path):
    """`engine: check` reads and writes back, in a pipeline file and in a workflow's `model_pipeline`."""
    pipeline = ModelPipeline(engine="check").add_model(OLSModel(PREDICTORS, 28, name="ols"))
    pipeline.save(tmp_path / "pipeline.yaml")
    assert yaml.safe_load((tmp_path / "pipeline.yaml").read_text())["engine"] == "check"
    assert ModelPipeline.load(tmp_path / "pipeline.yaml").engine == "check"
    (tmp_path / "experiment.yaml").write_text(
        "data_pipeline:\n  start: '2023-01-01'\n  end: '2023-02-01'\n"
        "model_pipeline:\n  path: pipeline.yaml\n  test: ['2023-01-20', '2023-01-21']\n  engine: check\n"
    )
    assert Workflow.load(tmp_path / "experiment.yaml")._build_model_pipeline().engine == "check"


# --- one model, the same bits anywhere --------------------------------------------------------------------------------


def many_models(n=48, seed=57):
    """LEAR (asinh, npit, "cv" standard and asinh), LassoCV (standard, mlog with masks of each fit), Ridge (poly) and OLS
    (asinh on some columns) over 2-6 of the 60 weather columns with gaps, one window or a list."""
    rng = random.Random(seed)
    models = []
    for i in range(n):
        names = [f"w{j:02d}" for j in sorted(rng.sample(range(POOL), rng.randint(2, 6)))]
        kind, name = i % 8, f"m{i:03d}"
        if kind == 0:
            models.append(LEARModel(["load_d+1", *names], rng.choice([21, 28]), name=name))
        elif kind == 1:
            models.append(LEARModel(["is_weekend_d+1", *names], [14, 28], scaler="npit", name=name))
        elif kind == 2:
            models.append(LassoCVModel(["load_d+1", *names], 28, cv=3, name=name))
        elif kind == 3:
            models.append(LassoCVModel(["load_d+1", *names], 35, cv=3, scaler="mlog", binary_mask="fit", name=name))
        elif kind == 4:
            models.append(LEARModel(["load_d+1", *names], 28, criterion="cv", scaler="standard", name=name))
        elif kind == 5:
            models.append(RidgeModel(["load_d+1", *names], 28, scaler="poly", name=name))
        elif kind == 6:
            models.append(
                OLSModel(["load_d+1", *names], 21, scaler={"name": "asinh", "params": {"apply_to": ["w0*"]}}, name=name)
            )
        else:
            models.append(LEARModel(["load_d+1", *names], 28, criterion="cv", name=name))
    return models


def run_models(models, data, save_dir, engine="on"):
    pipeline = ModelPipeline(engine=engine)
    for model in models:
        pipeline.add_model(model)
    return pipeline.run(data, "2023-04-10", "2023-04-14", horizon=1, freq="1D", save_dir=str(save_dir))


def test_a_model_is_the_same_alone_in_a_group_shuffled_and_chunked(wide, tmp_path, monkeypatch, in_process):
    models = many_models()
    names = [model.name for model in models]
    run_models(models, wide, tmp_path / "group")
    shuffled = many_models()
    random.Random(58).shuffle(shuffled)
    run_models(shuffled, wide, tmp_path / "shuffled")
    monkeypatch.setenv("ENGINE_MAX_MODELS", "7")
    run_models(many_models(), wide, tmp_path / "seven")
    monkeypatch.delenv("ENGINE_MAX_MODELS")
    monkeypatch.setenv("ENGINE_MAX_COLUMNS", "20")
    run_models(many_models(), wide, tmp_path / "columns")
    monkeypatch.delenv("ENGINE_MAX_COLUMNS")
    for other in ("shuffled", "seven", "columns"):
        assert same_files(tmp_path / "group", tmp_path / other) == len(models)
    for name in random.Random(59).sample(names, 8):
        run_models([m for m in many_models() if m.name == name], wide, tmp_path / "alone" / name)
        assert same_files(tmp_path / "alone" / name, tmp_path / "group", [name]) == 1
    counts = [manifest(tmp_path / "group", n)["engine"] for n in names]
    assert all(c["fits"] > 0 for c in counts)


@pytest.mark.parametrize("share", [0.0, 1.0])
def test_the_downdate_share_moves_no_bit(wide, tmp_path, monkeypatch, in_process, share):
    """τ moves no bit of a CV model's blocks and sums, nor of a VST model's (whose sums are over its row set's rows)."""
    run_models(many_models(16), wide, tmp_path / "half")
    monkeypatch.setattr(_engine_worker, "DOWNDATE_SHARE", share)
    run_models(many_models(16), wide, tmp_path / "other")
    assert same_files(tmp_path / "half", tmp_path / "other") == 16


def test_one_and_three_processes_and_two_task_threads(wide, tmp_path, monkeypatch):
    models = lambda: many_models(16)  # noqa: E731
    run_models(models(), wide, tmp_path / "one")
    monkeypatch.setenv("MAX_PROCESSES", "3")
    run_models(models(), wide, tmp_path / "three")
    monkeypatch.setenv("THREADS_PER_PROCESS", "2")
    run_models(models(), wide, tmp_path / "threads")
    assert same_files(tmp_path / "one", tmp_path / "three") == 16
    assert same_files(tmp_path / "one", tmp_path / "threads") == 16


def test_large_designs_at_one_and_eight_blas_threads(in_process):
    """A fit of 121 columns over 400 rows, asinh LEAR and Ridge and LassoCV: the same records at 1 and 8 BLAS threads."""
    base = make_market_data(start="2021-06-01", days=520, noise=5.0, seed=5, horizon=1).at_time("12:00").copy()
    rng = np.random.default_rng(6)
    extra = {f"c{i:03d}": rng.normal(0, 1, len(base)) + 0.002 * i * base["load"].to_numpy() for i in range(120)}
    data = pd.concat([base, pd.DataFrame(extra, index=base.index)], axis=1)
    names = ["load_d+1", *extra]

    def run(threads):
        with threadpool_limits(limits=threads):
            pipeline = ModelPipeline(engine="on")
            pipeline.add_model(LEARModel(names, 400, name="lear")).add_model(
                RidgeModel(names, 400, scaler="asinh", name="ridge")
            )
            pipeline.add_model(LassoCVModel(names, 400, cv=3, name="lasso"))
            report = pipeline.run(data, "2022-10-20", "2022-10-20", horizon=1, freq="1D")
        return {name: sorted(map(text, ref._results)) for name, ref in report.refs.items()}

    one, eight = run(1), run(8)
    assert one == eight and all(len(records) == 1 for records in one.values())


CRASH_DAY = datetime.date(2023, 3, 28)


class CrashingLear(LEARModel):
    """LEAR (asinh, engine-capable) whose fits a test makes raise."""


def test_a_crash_mid_group_then_resume_equals_fresh(spring, tmp_path, monkeypatch):
    """A VST fit of one model raises in a group from about the fifth day on: the group stops naming the model, and a resume
    equals the fresh run, VST and CV models alike."""
    period = ("2023-03-24", "2023-03-30")

    def models():
        return [
            LEARModel([*PREDICTORS, "price_d-1"], [14, 28], name="lear"),
            CrashingLear([*PREDICTORS, "price_d-1"], 28, name="crashing"),
            LassoCVModel(PREDICTORS, 28, cv=3, refit="7D", scaler="npit", name="lasso"),
            Crashing(PREDICTORS, 28, scaler="asinh", name="ols"),
        ]

    def run(save_dir):
        p = ModelPipeline(engine="on")
        for model in models():
            p.add_model(model)
        return p.run(spring, *period, horizon=1, save_dir=str(save_dir))

    run(tmp_path / "fresh")
    crash_from = (CRASH_DAY - spring.index[0].date()).days - 3
    real = _engine_worker.EngineFits.__call__

    def planted(self, fit):
        days = self.spans.hour_arrays[fit.period]["days"]
        if isinstance(self.model, CrashingLear) and fit.period == 3 and days[fit.rows.stop - 1] >= crash_from:
            raise RuntimeError("planted crash in a fit from statistics")
        return real(self, fit)

    monkeypatch.setattr(_engine_worker.EngineFits, "__call__", planted)
    with pytest.raises(RuntimeError, match="^crashing: forecasting failed for .*period 3: .*planted crash"):
        run(tmp_path / "crashed")
    monkeypatch.setattr(_engine_worker.EngineFits, "__call__", real)
    assert len(lines(tmp_path / "crashed" / "crashing.jsonl")) < len(lines(tmp_path / "fresh" / "crashing.jsonl"))
    run(tmp_path / "crashed")
    assert same_files(tmp_path / "fresh", tmp_path / "crashed") == 4


# --- nothing after D; verify; union isolation -------------------------------------------------------------------------

AFTER_D_MODELS = {
    "lear_asinh": lambda refit, fn, newest: LEARModel(fn, 21, refit=refit, newest_target=newest),
    "lear_cv_npit": lambda refit, fn, newest: LEARModel(
        fn, [14, 21], criterion="cv", scaler="npit", refit=refit, newest_target=newest
    ),
    "lasso": lambda refit, fn, newest: LassoCVModel(fn, 21, cv=3, refit=refit, newest_target=newest, binary_mask="fit"),
}
# Every level of each axis (model x list x issue days x horizon x refit) and every pair of levels, in half of the cells.
AFTER_D_CELLS = [
    ("lear_asinh", "fixed", None, 1, None),
    ("lear_asinh", "function", ["friday"], 3, "7D"),
    ("lear_asinh", "published", None, 3, "1MS"),
    ("lear_cv_npit", "fixed", ["friday"], 3, "1MS"),
    ("lear_cv_npit", "function", None, 1, None),
    ("lear_cv_npit", "published", ["friday"], 1, "7D"),
    ("lasso", "fixed", None, 3, "7D"),
    ("lasso", "function", ["friday"], 1, "1MS"),
    ("lasso", "published", None, 1, None),
]


@pytest.mark.parametrize(("kind", "lists", "issue_days", "horizon", "refit"), AFTER_D_CELLS)
def test_nothing_after_the_issue_day_moves_its_forecasts(market, kind, lists, issue_days, horizon, refit, in_process):
    """As `test_engine.py`'s, through the VST and CV paths: for D = Friday 31 March and Friday 7 April, every finite value
    dated after D moved and `aux` missing after D, or the test ending on D; D's predictions and coefficients stay bit for
    bit."""
    fn, newest = {
        "fixed": (PREDICTORS + ["flag"], "d-1"),
        "function": (flag_while_aux, "d-1"),
        "published": (flag_while_aux, "published"),
    }[lists]
    days = {} if issue_days is None else {"issue_days": issue_days}
    first, last = "2023-03-27", "2023-04-10"

    def run(frame, end=last):
        model = AFTER_D_MODELS[kind](refit, fn, newest)
        return fits_of(_engine.run_alone(model, frame, first, end, horizon=horizon, information_set=INFO, **days))

    base = run(market)
    dates = local_dates(market)
    columns = [c for c in market.columns if market[c].dtype.kind == "f"]
    for issue in ("2023-03-31", "2023-04-07"):
        on_d = {key: fit for key, fit in base.items() if key[0] == issue}
        assert len(on_d) == 24 * horizon
        after = dates > np.datetime64(issue)
        later = moved(market, after, columns)
        later.loc[after, "aux"] = np.nan
        for what, frame, end in (("later", later, last), ("end", market, issue)):
            other = run(frame, end)
            assert {key: fit for key, fit in other.items() if key[0] == issue} == on_d, (issue, what)
            later_keys = [key for key in base if key[0] > issue]
            if what == "later" and later_keys:  # the perturbation reaches the forecasts after D
                assert any(other.get(key) != base[key] for key in later_keys)


VERIFY_RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "load_actual": Availability.realised("1h"),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
}


def test_verify_through_the_engine_is_ok(spring):
    """`verify` through the VST and CV paths (a group of one): ok with its positive control for LEAR (asinh, published
    prices), a LEAR "cv" refit list and LassoCV; a column known only after the issue time is a leak for its model alone."""
    frame = spring.assign(load_actual=spring["load"])
    rules = {**VERIFY_RULES, "price": Availability.published("13:00", 1)}
    pipeline = ModelPipeline(information_set=InformationSet("11:45", "Europe/Brussels", rules=rules), engine="on")
    pipeline.add_model(LEARModel([*PREDICTORS, "price_d-1"], 28, newest_target="published", name="lear"))
    pipeline.add_model(LEARModel([*PREDICTORS, "price_d-1"], [14, 28], criterion="cv", refit="7D", name="lear_cv"))
    pipeline.add_model(LassoCVModel([*PREDICTORS, "price_d-1"], 28, cv=3, name="lasso"))
    report = pipeline.verify(frame, "2023-03-20", "2023-03-30", days=1)
    report.assert_ok()
    assert set(report.table()["status"]) == {"ok"}

    leaky = ModelPipeline(information_set=InformationSet(rules=VERIFY_RULES), engine="on")
    leaky.add_model(LassoCVModel([*PREDICTORS, "load_actual"], 28, cv=3, scaler="asinh", name="late"))
    leaky.add_model(LEARModel(PREDICTORS, 28, name="clean"))
    table = leaky.verify(frame, "2023-04-12", "2023-04-12", days=1, positive_control=False).table().set_index("model")
    assert table.loc["late", "status"] == "leak" and table.loc["late", "changed"] > 0
    assert table.loc["clean", "status"] == "ok"


def test_a_column_one_model_reads_moves_no_other_model(wide, tmp_path, in_process):
    """Through shared row sets and blocks: perturbing a column that one model alone reads leaves every other model's records
    bitwise, though its row sets' transforms and sums held that column."""
    models = many_models(32)
    reads = {}
    for model in models:
        for name in model.predictors:
            reads.setdefault(name, []).append(model.name)
    alone = next(name for name, readers in sorted(reads.items()) if len(readers) == 1 and name.startswith("w"))
    reader = reads[alone][0]
    run_models(many_models(32), wide, tmp_path / "base")
    perturbed = wide.copy()
    perturbed[alone] = perturbed[alone] * 3.0 + 7.0
    run_models(many_models(32), perturbed, tmp_path / "perturbed")
    for model in models:
        a = lines(tmp_path / "base" / f"{model.name}.jsonl")
        b = lines(tmp_path / "perturbed" / f"{model.name}.jsonl")
        assert (a == b) == (model.name != reader), model.name
