"""Fits from shared statistics in engine groups (step 5i): OLS, Ridge and LEAR with `scaler="standard"` fit from a span's exact
sums (`models/_engine_worker.py`). Within the engine a model's records are the same bits alone, in any group, order or chunking,
at any process, task thread or BLAS thread count, after a crash and resume, and at any downdate share τ; against engine off they
agree within the per-class tolerances of the step 5 spec (OLS 1e-8 EUR/MWh, Ridge and LEAR 1e-9, with the same penalty and the
same kink), every other field equal, and every fit the engine hands back is the model path's to the bit. Nothing after the issue
day and nothing before the training span moves a forecast; the manifest records `fit_path` (compared on resume) and counts.

The grid of `test_engine_within_tolerance_of_off` is step 5h's (`test_engine_groups.py`) restricted to StandardScaler models,
with the models the engine keeps on the model path (LassoCV, LEAR with asinh, Naive, a matching list, a fit without
`_fit_stats`) in the same pipelines: every level of every axis and every pair of levels across its cells (tests.md). The larger
grids, the mutants and the 1,000-variant measurement are in `epftoolbox2-research/library_check_step5/engine_stats/`.
"""

import datetime
import functools
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import LassoCVModel, LEARModel, NaiveModel, OLSModel, RidgeModel
from epftoolbox2.models import _engine, _engine_worker, _stats
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import ManifestMismatchError
from epftoolbox2.results.store import iter_file_records
from tests.models.test_engine_groups import (
    Crashing,
    InProcessPool,
    OwnFit,
    by_weekday,
    holes,
    lines,
    matching_list,
    with_aux,
    with_aux_column,
)
from tests.synthetic import PREDICTORS, make_market_data

PROVENANCE = ("package_version", "git_sha", "created_at", "updated_at", "record_schema", "kernels", "engine")
PUBLISHED = {"price": Availability.published("13:00", 1)}
STANDARD = {"name": "standard", "params": {"scale_binary": True}}
# Engine against off, per class, in EUR/MWh (the step 5 spec's tolerance table); coefficients (scaled units) alike.
TOLERANCE = {"OLSModel": 1e-8, "RidgeModel": 1e-9, "LEARModel": 1e-9}
NUMBERS = ("prediction", "coefficients", "member_predictions")


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


# --- frames -----------------------------------------------------------------------------------------------------------


def with_near_twin(data, seed):
    """`twin`: `load_d+1` plus noise of 1e-7 of it, so a fit reading both is far too ill-conditioned for the normal equations."""
    rng = np.random.default_rng(seed)
    return data.assign(twin=data["load_d+1"] * (1.0 + 1e-7 * rng.normal(0, 1, len(data))))


@pytest.fixture(scope="module")
def spring():
    data = make_market_data(start="2022-12-01", days=150, noise=5.0, seed=41, horizon=3)
    data = with_near_twin(with_aux_column(data, "2023-03-25", 42), 43)
    return holes(data, "2023-03-27")


@pytest.fixture(scope="module")
def autumn():
    data = make_market_data(start="2023-06-01", days=160, noise=5.0, seed=43, horizon=3)
    data = with_aux_column(data, "2023-10-28", 44)
    return holes(data, "2023-10-30", column="wind_d+2")


@pytest.fixture(scope="module")
def quarter():
    data = make_market_data(start="2023-09-01", days=62, noise=5.0, seed=45, horizon=1, freq="15min")
    data = with_aux_column(data, "2023-10-27", 46)
    issue = (data.index.date == pd.Timestamp("2023-10-30").date()) & (data.index.hour == 5)
    data.loc[issue & (data.index.minute == 15), "load_d+1"] = np.nan
    return data.drop(index=data.index[issue & (data.index.minute == 45)])


@pytest.fixture(scope="module")
def daily():
    data = make_market_data(start="2023-01-01", days=200, noise=5.0, seed=47, horizon=7).at_time("12:00")
    data = with_aux_column(data.copy(), "2023-05-20", 48)
    data.iloc[150, data.columns.get_loc("load_d+3")] = np.nan
    return data


POOL = 60  # weather-like columns of the many-model frame


@pytest.fixture(scope="module")
def wide():
    """A daily frame with 60 weather-like columns, a few cells missing, for groups of many models."""
    base = make_market_data(start="2023-01-01", days=120, noise=5.0, seed=51, horizon=1).at_time("12:00").copy()
    rng = np.random.default_rng(52)
    columns = {f"w{i:02d}": rng.normal(0, 1, len(base)) + 0.01 * base["load"].to_numpy() for i in range(POOL)}
    wide = pd.concat([base, pd.DataFrame(columns, index=base.index)], axis=1)
    for i in range(0, POOL, 7):
        wide.iloc[rng.integers(0, len(wide), 3), wide.columns.get_loc(f"w{i:02d}")] = np.nan
    return wide


# --- models -----------------------------------------------------------------------------------------------------------


class EveryOtherBack(OLSModel):
    """OLS whose every second fit from statistics is handed back (`_fit_stats` None), in the order the engine asks, and whose
    model-path fits note their coefficients (`model_path`), so a test can find their records."""

    asked = [0]
    model_path: list = []

    def _fit_predict(self, train_x, train_y, test_x):
        prediction, coefs = super()._fit_predict(train_x, train_y, test_x)
        EveryOtherBack.model_path.append(coefs)
        return prediction, coefs

    def _fit_model(self, train_x, train_y):
        return super()._fit_model(train_x, train_y)

    def _predict_fitted(self, fitted, test_x):
        prediction, coefs = super()._predict_fitted(fitted, test_x)
        if isinstance(fitted, LinearRegression):
            EveryOtherBack.model_path.append(coefs)
        return prediction, coefs

    def _fit_stats(self, stats):
        EveryOtherBack.asked[0] += 1
        return None if EveryOtherBack.asked[0] % 2 else super()._fit_stats(stats)


def lear(*args, **kwargs):
    return LEARModel(*args, scaler="standard", **kwargs)


def test_which_models_fit_from_statistics():
    """OLS, Ridge and LEAR with exactly `StandardScaler` (any `scale_binary`); not a VST, not CV blocks (5j), not a model that
    declares no needs."""
    path = _engine_worker.ENGINE_FIT_PATH
    assert path == "engine-1"
    for model in (
        OLSModel(PREDICTORS),
        OLSModel(PREDICTORS, scaler=STANDARD),
        RidgeModel(PREDICTORS),
        lear(PREDICTORS),
        lear(PREDICTORS, criterion="aic", solver="cd"),
        EveryOtherBack(PREDICTORS),
    ):
        assert _engine_worker.fit_path_of(model) == path, model
    for model in (
        LEARModel(PREDICTORS),  # asinh
        lear(PREDICTORS, criterion="cv"),
        LassoCVModel(PREDICTORS),
        OLSModel(PREDICTORS, scaler="npit"),
        RidgeModel(PREDICTORS, scaler="asinh"),
    ):
        assert _engine_worker.fit_path_of(model) is None, model

    class Undeclared(OLSModel):
        _ENGINE_NEEDS = None

    assert _engine_worker.fit_path_of(Undeclared(PREDICTORS)) is None


# --- comparing engine on and off --------------------------------------------------------------------------------------


def records_of(path):
    """A model's records by key (run date, period, horizon, window), every one as written (skip records too)."""
    out = {}
    for record in iter_file_records(Path(path)):
        key = (record["run_date"], record["hour"], record["horizon"], record.get("window"))
        assert key not in out, key
        out[key] = record
    return out


def results_path(save_dir, name):
    for suffix in (".jsonl", ".parquet"):
        if (Path(save_dir) / f"{name}{suffix}").exists():
            return Path(save_dir) / f"{name}{suffix}"
    raise FileNotFoundError(name)


def manifest_of(save_dir, name):
    content = json.loads((Path(save_dir) / f"{name}.manifest.json").read_text())
    return {key: value for key, value in content.items() if key not in PROVENANCE}


def text(record):
    """A record as JSON text: equal text is equal bits (floats as `repr`), NaN equal to NaN."""
    return json.dumps(record, sort_keys=True)


def support(coefficients, cut=1e-12):
    return tuple(abs(c) > cut for c in coefficients)


def compare(off_dir, on_dir, models):
    """Each model's records engine on against off: the same keys and every field but the numbers; a model on the model path
    (no `fit_path`) to the bit, one from statistics within its class's tolerance, with LEAR's coefficients on the same columns
    (the same kink); the manifests equal but for `fit_path`. Returns {name: (records, largest |Δ| of a forecast, bitwise)}."""
    out = {}
    for model in models:
        off = records_of(results_path(off_dir, model.name))
        on = records_of(results_path(on_dir, model.name))
        assert off.keys() == on.keys(), model.name
        path = expected_path(model)
        largest, bitwise = 0.0, 0
        for key, a in off.items():
            b = on[key]
            if path is None or text(a) == text(b):
                assert text(a) == text(b), (model.name, key)
                bitwise += 1
                continue
            assert text({k: v for k, v in a.items() if k not in NUMBERS}) == text(
                {k: v for k, v in b.items() if k not in NUMBERS}
            ), (model.name, key)
            assert list(a) == list(b)
            tolerance = TOLERANCE[cls_of(model)]
            delta = abs(a["prediction"] - b["prediction"])
            assert delta <= tolerance, (model.name, key, delta)
            for x, y in zip(a.get("member_predictions", []), b.get("member_predictions", [])):
                assert abs(x - y) <= tolerance, (model.name, key)
            assert len(a["coefficients"]) == len(b["coefficients"])
            if a["coefficients"]:
                assert np.max(np.abs(np.subtract(a["coefficients"], b["coefficients"]))) <= 1e3 * tolerance, (
                    model.name,
                    key,
                )
            if cls_of(model) == "LEARModel":
                assert support(a["coefficients"]) == support(b["coefficients"]), (model.name, key)  # the same kink
            largest = max(largest, delta)
        mine, theirs = manifest_of(on_dir, model.name), manifest_of(off_dir, model.name)
        assert mine.pop("fit_path", None) == path
        assert mine == theirs, model.name
        out[model.name] = (len(off), largest, bitwise)
    return out


def expected_path(model):
    """The fit path a model's manifest has with engine on: its class's, where the engine takes the model (not a matching
    list's plan, which this file's only matching model has, nor a model not engine-capable)."""
    if not model._engine_capable() or model._ctx_fn is matching_list:
        return None
    return _engine_worker.fit_path_of(model)


def cls_of(model):
    """The library class a model is (OLSModel, RidgeModel, LEARModel, ...), for its tolerance."""
    for klass in type(model).__mro__:
        if klass.__module__.startswith("epftoolbox2.models.") and klass.__name__ in (*TOLERANCE, "LassoCVModel"):
            return klass.__name__
    return type(model).__name__


def build(models, rules=None):
    def pipeline(engine):
        info = None if rules is None else InformationSet("11:45", "Europe/Brussels", rules=rules)
        out = ModelPipeline(information_set=info, availability="none" if info else None, engine=engine)
        for model in models():
            out.add_model(model)
        return out

    return pipeline


def run_both(build_, data, tmp_path, **arguments):
    reports = {}
    for engine in ("off", "on"):
        reports[engine] = build_(engine).run(data, save_dir=str(tmp_path / engine), **arguments)
    return reports["off"], reports["on"]


# --- the grid ---------------------------------------------------------------------------------------------------------


def spring_models():
    return [
        OLSModel(PREDICTORS, 28, name="ols"),
        OLSModel([*PREDICTORS, "aux"], 28, name="ols_aux"),  # the first day's mask: aux is 0/1 there
        OLSModel([*PREDICTORS, "aux"], 28, binary_mask="fit", name="ols_aux_fit"),  # each fit's: aux continuous later
        # The same pair as Ridge, whose penalty sees the scaling (OLS with a one-hot pair doesn't): one span, two masks.
        RidgeModel([*PREDICTORS, "aux"], 28, name="ridge_aux"),
        RidgeModel([*PREDICTORS, "aux"], 28, binary_mask="fit", name="ridge_aux_fit"),
        RidgeModel([*PREDICTORS, "price_d-1"], [14, 28], refit="7D", name="ridge_7d_list"),
        lear([*PREDICTORS, "price_d-1", "aux"], [28, 90], refit="auto", name="lear_auto"),
        OLSModel(by_weekday, 28, binary_mask="fit", scaler=STANDARD, name="ols_function_fit"),
        RidgeModel(with_aux, 28, refit="7D", name="ridge_function_7d"),
        # Hand-backs: an intercept with a one-hot pair (rank), a near twin (the guard), tiny penalties (Ridge's rounding
        # budget), two rows (fewer than 3), and every second fit of an OLS.
        OLSModel(PREDICTORS, 28, fit_intercept=True, name="ols_collinear"),
        OLSModel(["load_d+1", "twin", *PREDICTORS[1:]], 28, name="ols_twin"),
        RidgeModel(PREDICTORS, 28, alphas=[1e-6, 1e-5], name="ridge_tiny"),
        OLSModel(PREDICTORS, 2, name="ols_two_rows"),
        # The model path in the same groups and pipeline.
        LassoCVModel([*PREDICTORS, "aux"], 28, cv=3, refit="7D", binary_mask="fit", name="lasso_7d_fit"),
        LEARModel([*PREDICTORS, "price_d-1"], 28, name="lear_asinh"),
        NaiveModel(["price_d-1"], name="naive"),
        OLSModel(matching_list, 28, name="matching"),
        OwnFit(PREDICTORS, 28, name="own_fit"),
    ]


def autumn_models():
    names = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "aux"]
    return [
        OLSModel([*names, "price_d-1"], 28, newest_target="published", name="ols_published"),
        RidgeModel(names, [14, 28], binary_mask="fit", name="ridge_list_fit"),
        lear([*names, "price_d-1"], [28, 120], refit="auto", newest_target="published", name="lear_auto_published"),
        OLSModel(names, [14, 120], refit="auto", name="ols_auto"),
        LassoCVModel(by_weekday, 28, cv=3, refit="7D", name="lasso_function_7d"),
        NaiveModel(["price_d-1"], name="naive"),
    ]


def quarter_models():
    return [
        OLSModel([*PREDICTORS, "aux"], [7, 14], binary_mask="fit", name="ols_list_fit"),
        OLSModel(PREDICTORS, 14, refit="7D", name="ols_7d"),
        RidgeModel(PREDICTORS, [7, 14], refit="auto", name="ridge_auto"),
        lear(with_aux, 14, criterion="aic", name="lear_function"),
        LEARModel(with_aux, 14, name="lear_function_asinh"),
    ]


def daily_models():
    names = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "aux"]
    return [
        OLSModel([*names, "price_d-1"], 56, newest_target="published", binary_mask="fit", name="ols_published_fit"),
        RidgeModel(by_weekday, [42, 56], refit="7D", name="ridge_function_7d"),
        RidgeModel(names, [42, 90], refit="auto", scaler=STANDARD, name="ridge_auto"),
        lear([*names, "price_d-1"], [42, 56], solver="cd", name="lear_list"),
        lear([*names, "price_d-1"], 56, refit="7D", name="lear_7d"),
        LassoCVModel(names, [42, 56], cv=3, binary_mask="fit", name="lasso_list_fit"),
        NaiveModel(["price_d-1"], name="naive"),
    ]


def autumn_validation_models():
    names = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "aux"]
    return [
        RidgeModel(names, [14, 28], binary_mask="fit", name="ridge_list_fit"),
        OLSModel(names, [14, 120], refit="auto", name="ols_auto"),
        lear(by_weekday, [28, 90], refit="auto", name="lear_function_auto"),
        LassoCVModel(with_aux, 28, cv=3, refit="7D", binary_mask="fit", name="lasso_function_7d"),
        NaiveModel(["price_d-1"], name="naive"),
    ]


# Each cell: the frame, the pipeline, the run's arguments (step 5h's cells). Each class with refit None, "7D" and "auto", a
# window list, a function, `binary_mask="fit"` and `scale_binary=True` in some cell, BIC and AIC, both LEAR solvers.
CELLS = {
    "1h-spring-plain-jsonl": ("spring", spring_models, None, dict(test=("2023-03-24", "2023-03-27"), horizon=2)),
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
    "1h-autumn-validation-jsonl": (
        "autumn",
        autumn_validation_models,
        None,
        dict(validation=("2023-10-23", "2023-10-24"), test=("2023-10-28", "2023-10-30"), horizon=2),
    ),
}


@pytest.mark.parametrize("cell", list(CELLS))
def test_engine_within_tolerance_of_off(cell, request, tmp_path):
    frame, models, rules, arguments = CELLS[cell]
    data = request.getfixturevalue(frame)
    off, on = run_both(build(models, rules), data, tmp_path, **arguments)
    seen = compare(tmp_path / "off", tmp_path / "on", models())
    pd.testing.assert_frame_equal(off.forecast_counts(), on.forecast_counts(), check_exact=True)
    assert off.forecast_counts()["produced"].gt(0).all()
    stats = [m for m in models() if expected_path(m) is not None]
    assert stats and any(seen[m.name][2] < seen[m.name][0] for m in stats)  # the engine fitted something
    for model in stats:
        engine = json.loads((tmp_path / "on" / f"{model.name}.manifest.json").read_text())["engine"]
        assert engine["fits"] + sum(engine["fallbacks"].values()) > 0
    if cell.startswith("1h-spring"):
        # Every fit of these went back to the model path, for its own reason, so their records are the model path's bits.
        for name, reason in (
            ("ols_collinear", "declined"),
            ("ols_twin", "declined"),
            ("ridge_tiny", "declined"),
            ("ols_two_rows", "few_rows"),
        ):
            records, _, bitwise = seen[name]
            assert bitwise == records, name
            engine = json.loads((tmp_path / "on" / f"{name}.manifest.json").read_text())["engine"]
            assert engine["fits"] == 0 and set(engine["fallbacks"]) == {reason}, (name, engine)
        for name in ("ols", "ols_aux", "ols_aux_fit", "ridge_7d_list", "lear_auto"):
            assert seen[name][2] < seen[name][0], name  # from statistics, not bitwise


def test_a_fit_handed_back_is_the_model_paths(spring, tmp_path, in_process):
    """Every second fit of `EveryOtherBack` goes back: those records (found by the model path's coefficients) are engine
    off's to the bit, the others within tolerance and not bitwise, and the manifest counts both."""
    EveryOtherBack.asked[0] = 0
    EveryOtherBack.model_path.clear()

    def models():
        return [EveryOtherBack(PREDICTORS, 28, name="half"), EveryOtherBack(PREDICTORS, 28, refit="7D", name="half_7d")]

    run_both(build(models), spring, tmp_path, test=("2023-03-24", "2023-03-30"), horizon=2)
    compare(tmp_path / "off", tmp_path / "on", models())
    handed = {json.dumps(coefs) for coefs in EveryOtherBack.model_path}
    for name in ("half", "half_7d"):
        off = records_of(tmp_path / "off" / f"{name}.jsonl")
        on = records_of(tmp_path / "on" / f"{name}.jsonl")
        back = [key for key, record in on.items() if json.dumps(record["coefficients"]) in handed]
        assert back and len(back) < len(on)
        assert all(on[key] == off[key] for key in back)
        assert not any(on[key] == off[key] for key in on.keys() - set(back) if on[key]["prediction"] is not None)
        engine = json.loads((tmp_path / "on" / f"{name}.manifest.json").read_text())["engine"]
        assert engine["fits"] > 0 and engine["fallbacks"] == {"declined": engine["fallbacks"]["declined"]}
    unit_fits = json.loads((tmp_path / "on" / "half_7d.manifest.json").read_text())["engine"]
    assert unit_fits["fits"] + unit_fits["fallbacks"]["declined"] == 24 * 2 * 2  # 24 periods x 2 horizons x 2 blocks


def test_a_dummy_constant_on_its_rows_but_not_its_span(spring, tmp_path):
    """`event` is 1 on three days of the window whose `load_d+1` is missing (so ρ leaves them out) and on the issue day, 0
    elsewhere: constant on ρ, about 0.1 over the span. Its sums about the span's shift leave √eps noise, which as a standard
    deviation would make its issue value 1e8 standard deviations out; the candidate rule calls it constant on ρ, as the
    scaler does, with the dummy scaled (`scale_binary`) or not."""
    data = spring.copy()
    dates = np.asarray(data.index.date)
    event = np.zeros(len(data))
    for day in ("2023-03-10", "2023-03-14", "2023-03-17", "2023-03-29"):
        on_day = dates == datetime.date.fromisoformat(day)
        event[on_day] = 1.0
        if day != "2023-03-29":
            data.loc[on_day, "load_d+1"] = np.nan
    data["event"] = event

    # A one-hot pair would be collinear once the dummies are centred: one of them, and an intercept for the level.
    names = [*PREDICTORS[:3], "event"]

    def models():
        return [
            OLSModel(names, 28, scaler=STANDARD, fit_intercept=True, name="scaled"),
            OLSModel([*PREDICTORS, "event"], 28, name="unscaled"),
            RidgeModel(names, 28, scaler=STANDARD, name="ridge"),
            lear(names, 28, name="lear"),
        ]

    run_both(build(models), data, tmp_path, test=("2023-03-29", "2023-03-29"), horizon=1)
    seen = compare(tmp_path / "off", tmp_path / "on", models())
    for name in ("scaled", "unscaled", "ridge", "lear"):
        engine = json.loads((tmp_path / "on" / f"{name}.manifest.json").read_text())["engine"]
        assert engine["fits"] >= 20 and seen[name][2] <= 24 - engine["fits"], (name, engine)


# --- one model, the same bits anywhere --------------------------------------------------------------------------------


def many_models(n=200, seed=53):
    """OLS, Ridge and LEAR (standard) over 2-8 of the 60 weather columns with gaps, one window or a list, mask None or "fit"."""
    rng = random.Random(seed)
    models = []
    for i in range(n):
        names = [f"w{j:02d}" for j in sorted(rng.sample(range(POOL), rng.randint(2, 8)))]
        kind = i % 5
        if kind == 0:
            models.append(OLSModel(["load_d+1", *names], rng.choice([21, 28, 35]), name=f"m{i:03d}"))
        elif kind == 1:
            models.append(OLSModel(["load_d+1", *names], [14, 28], binary_mask="fit", name=f"m{i:03d}"))
        elif kind == 2:
            models.append(RidgeModel(["load_d+1", *names], 28, name=f"m{i:03d}"))
        elif kind == 3:
            models.append(lear(["is_weekend_d+1", *names], 28, name=f"m{i:03d}"))
        else:  # an intercept and a one-hot pair: every fit handed back (rank)
            models.append(
                OLSModel(["is_weekend_d+1", "is_weekday_d+1", *names], 21, fit_intercept=True, name=f"m{i:03d}")
            )
    return models


def run_models(models, data, save_dir, engine="on"):
    pipeline = ModelPipeline(engine=engine)
    for model in models:
        pipeline.add_model(model)
    return pipeline.run(data, "2023-04-10", "2023-04-14", horizon=1, freq="1D", save_dir=str(save_dir))


def same_files(one, other, names=None):
    """Two save_dirs hold the same records (sorted lines) and manifests without provenance; returns how many models."""
    names = names or sorted(p.name[: -len(".jsonl")] for p in Path(one).glob("*.jsonl"))
    for name in names:
        assert lines(Path(one) / f"{name}.jsonl") == lines(Path(other) / f"{name}.jsonl"), name
        assert manifest_of(one, name) == manifest_of(other, name), name
    return len(names)


def test_a_model_is_the_same_alone_in_a_group_shuffled_and_chunked(wide, tmp_path, monkeypatch, in_process):
    models = many_models()
    names = [model.name for model in models]
    run_models(models, wide, tmp_path / "group")
    shuffled = many_models()
    random.Random(54).shuffle(shuffled)
    run_models(shuffled, wide, tmp_path / "shuffled")
    monkeypatch.setenv("ENGINE_MAX_MODELS", "7")
    run_models(many_models(), wide, tmp_path / "seven")
    monkeypatch.delenv("ENGINE_MAX_MODELS")
    monkeypatch.setenv("ENGINE_MAX_COLUMNS", "50")
    run_models(many_models(), wide, tmp_path / "columns")
    monkeypatch.delenv("ENGINE_MAX_COLUMNS")
    for other in ("shuffled", "seven", "columns"):
        assert same_files(tmp_path / "group", tmp_path / other) == len(models)
    for name in random.Random(55).sample(names, 8):
        run_models([m for m in many_models() if m.name == name], wide, tmp_path / "alone" / name)
        assert same_files(tmp_path / "alone" / name, tmp_path / "group", [name]) == 1
    counts = [json.loads((tmp_path / "group" / f"{n}.manifest.json").read_text())["engine"] for n in names]
    assert sum(c["fits"] for c in counts) > 0 and sum(sum(c["fallbacks"].values()) for c in counts) > 0


@pytest.mark.parametrize("share", [0.0, 1.0])
def test_the_downdate_share_moves_no_bit(wide, tmp_path, monkeypatch, in_process, share):
    """τ = 0 (every fit summed over ρ directly), 0.5 (the default) and 1 (every fit as the span's totals minus the rows ρ
    leaves out): the same records."""
    run_models(many_models(40), wide, tmp_path / "half")
    monkeypatch.setattr(_engine_worker, "DOWNDATE_SHARE", share)
    asked = []
    real = _engine_worker.SpanSums.over

    def over(self, rows, cols, method="auto"):
        asked.append((method, int((~rows).sum())))
        return real(self, rows, cols, method=method)

    monkeypatch.setattr(_engine_worker.SpanSums, "over", over)
    run_models(many_models(40), wide, tmp_path / "other")
    if share == 0.0:  # a downdate only of nothing
        assert {left for method, left in asked if method == "downdate"} <= {0}
        assert any(method == "direct" for method, _ in asked)
    else:
        assert {method for method, _ in asked} == {"downdate"} and any(left > 0 for _, left in asked)
    assert same_files(tmp_path / "half", tmp_path / "other") == 40


def test_one_and_four_blas_threads(wide, tmp_path, monkeypatch, in_process):
    run_models(many_models(40), wide, tmp_path / "one")
    with threadpool_limits(limits=4):
        run_models(many_models(40), wide, tmp_path / "four")
    assert same_files(tmp_path / "one", tmp_path / "four") == 40


def test_large_designs_at_one_and_eight_blas_threads(in_process):
    """A fit of 121 columns over 400 rows: LAPACK's eigh and Cholesky and a threaded GEMM on the fit's rows change their
    last bits with the BLAS thread count at this size (Ridge's records moved at 8 threads), so a job with fits from
    statistics runs on one BLAS thread whatever the process's (`_engine_worker._one_blas_thread`)."""
    base = make_market_data(start="2021-06-01", days=520, noise=5.0, seed=5, horizon=1).at_time("12:00").copy()
    rng = np.random.default_rng(6)
    extra = {f"c{i:03d}": rng.normal(0, 1, len(base)) + 0.002 * i * base["load"].to_numpy() for i in range(120)}
    data = pd.concat([base, pd.DataFrame(extra, index=base.index)], axis=1)
    names = ["load_d+1", *extra]

    def run(threads):
        with threadpool_limits(limits=threads):
            pipeline = ModelPipeline(engine="on")
            pipeline.add_model(OLSModel(names, 400, name="ols")).add_model(RidgeModel(names, 400, name="ridge"))
            pipeline.add_model(lear(names, 400, name="lear"))
            report = pipeline.run(data, "2022-10-20", "2022-10-24", horizon=1, freq="1D")
        return {name: sorted(map(text, ref._results)) for name, ref in report.refs.items()}

    one, eight = run(1), run(8)
    assert one == eight and all(len(records) == 5 for records in one.values())


def test_one_and_three_processes_and_two_task_threads(wide, tmp_path, monkeypatch):
    """Real pools: 1 and 3 processes, and 2 task threads per process, give the same files; with threads, each thread has its
    own spans and the items reading a span run on one thread."""
    models = lambda: many_models(30)  # noqa: E731
    run_models(models(), wide, tmp_path / "one")
    monkeypatch.setenv("MAX_PROCESSES", "3")
    run_models(models(), wide, tmp_path / "three")
    monkeypatch.setenv("THREADS_PER_PROCESS", "2")
    run_models(models(), wide, tmp_path / "threads")
    assert same_files(tmp_path / "one", tmp_path / "three") == 30
    assert same_files(tmp_path / "one", tmp_path / "threads") == 30


def test_items_that_read_a_span_share_a_thread():
    keys = [[(0, 1, 0, 5)], [(0, 1, 0, 5)], [(0, 1, 3, 9)], [], [(0, 1, 0, 5), (0, 1, 3, 9)], [(0, 1, 3, 9)]]
    batches = _engine_worker._batches(keys, 2)
    assert sorted(i for batch in batches for i in batch) == list(range(6))
    where = {i: b for b, batch in enumerate(batches) for i in batch}
    assert where[0] == where[1] and where[2] == where[5]
    assert all(batch == sorted(batch) for batch in batches)
    assert _engine_worker._batches(keys, 2) == batches  # deterministic
    assert _engine_worker._batches(keys, 1) == [list(range(6))]


CRASH_DAY = datetime.date(2023, 3, 28)


@pytest.mark.parametrize("which", ["member", "unit"])
def test_a_crash_mid_group_then_resume_equals_fresh(spring, tmp_path, monkeypatch, which):
    """A fit from statistics of one model raises in a group (a member of a window list, or a refit unit, of period 3 from
    about the fifth day on): the group stops naming the model, and a resume equals the fresh run."""
    period = ("2023-03-24", "2023-03-30")

    def models():
        return [
            OLSModel(PREDICTORS, 28, name="ols"),
            Crashing(PREDICTORS, [14, 28], name="crashing"),
            RidgeModel(PREDICTORS, [14, 28], refit="7D", name="ridge"),
            Crashing(PREDICTORS, 28, refit="7D", name="crashing_refit"),
            lear([*PREDICTORS, "price_d-1"], 28, name="lear"),
        ]

    def run(save_dir):
        p = ModelPipeline(engine="on")
        for model in models():
            p.add_model(model)
        return p.run(spring, *period, horizon=2, save_dir=str(save_dir))

    run(tmp_path / "fresh")
    crash_from = (CRASH_DAY - spring.index[0].date()).days - 3
    real = _engine_worker.EngineFits.__call__
    refit = which == "unit"

    def planted(self, fit):
        days = self.spans.hour_arrays[fit.period]["days"]
        if isinstance(self.model, Crashing) and fit.period == 3 and days[fit.rows.stop - 1] >= crash_from:
            if (fit.test.shape[0] > 1) == refit:  # a member forecasts one row, a unit of these days several
                raise RuntimeError("planted crash in a fit from statistics")
        return real(self, fit)

    monkeypatch.setattr(_engine_worker.EngineFits, "__call__", planted)
    name = "crashing_refit" if refit else "crashing"
    with pytest.raises(RuntimeError, match=f"^{name}: forecasting failed for .*period 3: .*planted crash"):
        run(tmp_path / "crashed")
    monkeypatch.setattr(_engine_worker.EngineFits, "__call__", real)
    crashed = {p.name: len(lines(p)) for p in (tmp_path / "crashed").glob("*.jsonl")}
    fresh = {p.name: len(lines(p)) for p in (tmp_path / "fresh").glob("*.jsonl")}
    assert crashed[f"{name}.jsonl"] < fresh[f"{name}.jsonl"]
    run(tmp_path / "crashed")
    assert same_files(tmp_path / "fresh", tmp_path / "crashed") == 5


# --- nothing after D, nothing before the span -------------------------------------------------------------------------

RULES = {
    "load": Availability.published("10:00", 10),
    "wind": Availability.published("10:00", 10),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "flag": Availability.static(),
    "aux": Availability.static(),
    "price": Availability.published("13:00", 1),  # D's price is known at 11:45 on D, D+1's isn't
}
INFO = InformationSet("11:45", "Europe/Brussels", rules=RULES)


def flag_while_aux(ctx):
    return PREDICTORS + ["flag"] if ctx.published("aux") else PREDICTORS


@pytest.fixture(scope="module")
def market():
    """Over the spring switch of 26 March 2023, with `flag` turning from 0/1 to other values on 29 March."""
    data = make_market_data(start="2023-01-01", days=130, horizon=3, noise=5.0, seed=3)
    flag = (np.random.default_rng(5).random(len(data)) > 0.5).astype(float)
    after = np.asarray(pd.DatetimeIndex(data.index.tz_localize(None)).normalize()) >= np.datetime64("2023-03-29")
    flag[after] = flag[after] * 0.5 + 0.25
    return data.assign(flag=flag, aux=1.0)


def local_dates(data):
    return np.asarray(pd.DatetimeIndex(data.index.tz_localize(None)).normalize())


def fits_of(ref):
    return {(r["run_date"], r["hour"], r["horizon"]): (r["prediction"], r["coefficients"]) for r in ref._results}


def moved(data, rows, columns, seed=1):
    out = data.copy()
    values = out[columns].to_numpy(dtype=float, copy=True)
    where = rows[:, None] & np.isfinite(values)
    values[where] += np.random.default_rng(seed).normal(0, 3, size=int(where.sum()))
    out[columns] = values
    return out


AFTER_D_MODELS = {
    "ols": lambda refit, fn, newest: OLSModel(fn, 21, refit=refit, newest_target=newest, binary_mask="fit"),
    "ridge": lambda refit, fn, newest: RidgeModel(fn, [14, 21], refit=refit, newest_target=newest),
    "lear": lambda refit, fn, newest: lear(fn, 21, refit=refit, newest_target=newest),
}
# Every level of each axis (class x list x issue days x horizon x refit) and every pair of levels, in half of the cells.
AFTER_D_CELLS = [
    ("ols", "fixed", None, 1, None),
    ("ols", "function", ["friday"], 3, "7D"),
    ("ols", "published", None, 3, "1MS"),
    ("ridge", "fixed", ["friday"], 3, "1MS"),
    ("ridge", "function", None, 1, None),
    ("ridge", "published", ["friday"], 1, "7D"),
    ("lear", "fixed", None, 3, "7D"),
    ("lear", "function", ["friday"], 1, "1MS"),
    ("lear", "published", None, 1, None),
]


@pytest.mark.parametrize(("kind", "lists", "issue_days", "horizon", "refit"), AFTER_D_CELLS)
def test_nothing_after_the_issue_day_moves_its_forecasts(market, kind, lists, issue_days, horizon, refit, in_process):
    """Through the engine (a group of one): for D = Friday 31 March and Friday 7 April, every finite value dated after D moved,
    `aux` missing after D (which changes a function's list after D), and the test ending on D. D's predictions and coefficients
    stay bit for bit."""
    fn, newest = {
        "fixed": (PREDICTORS + ["flag"], "d-1"),
        "function": (flag_while_aux, "d-1"),
        "published": (flag_while_aux, "published"),
    }[lists]
    days = {} if issue_days is None else {"issue_days": issue_days}
    first, last = "2023-03-13", "2023-04-16"

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
        blank = market.copy()
        blank.loc[after, "aux"] = np.nan
        for what, frame, end in (
            ("values", moved(market, after, columns), last),
            ("aux", blank, last),
            ("end", market, issue),
        ):
            other = run(frame, end)
            assert {key: fit for key, fit in other.items() if key[0] == issue} == on_d, (issue, what)
            if what == "values":
                assert any(other.get(key) != fit for key, fit in base.items() if key[0] > issue)


@pytest.mark.parametrize("refit", [None, "7D"])
def test_a_frame_that_starts_a_year_earlier_moves_no_int_window_forecast(refit, tmp_path, in_process):
    """The same frame from 2022-01-01 and from 2023-01-01: every record of models with day-count windows (OLS, Ridge, LEAR;
    a list of two windows; masks of the first day and of each fit) equal to the bit through the engine."""
    long = make_market_data(start="2022-01-01", days=480, noise=5.0, seed=71, horizon=2)
    long = with_aux_column(long, "2023-03-25", 72)
    short = long[long.index >= pd.Timestamp("2023-01-01", tz=long.index.tz)]

    def models():
        return [
            OLSModel([*PREDICTORS, "aux"], 28, refit=refit, name="ols"),
            OLSModel([*PREDICTORS, "aux"], [14, 28], refit=refit, binary_mask="fit", name="ols_fit"),
            RidgeModel([*PREDICTORS, "price_d-1"], 35, refit=refit, name="ridge"),
            lear([*PREDICTORS, "price_d-1", "aux"], [28, 56], refit=refit, name="lear"),
        ]

    for name, frame in (("long", long), ("short", short)):
        run_models_on(models(), frame, tmp_path / name, ("2023-03-22", "2023-03-30"), horizon=2)
    for model in models():
        assert lines(tmp_path / "long" / f"{model.name}.jsonl") == lines(tmp_path / "short" / f"{model.name}.jsonl")


def run_models_on(models, data, save_dir, test, **arguments):
    pipeline = ModelPipeline(engine="on")
    for model in models:
        pipeline.add_model(model)
    return pipeline.run(data, *test, save_dir=str(save_dir), **arguments)


# --- leakage ----------------------------------------------------------------------------------------------------------

VERIFY_RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "load_actual": Availability.realised("1h"),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
}


def test_verify_through_the_engine_is_ok_and_finds_a_late_column_in_one_model(spring):
    """`verify` runs each model's probes through the engine's fits from statistics (a group of one): ok with the positive
    control for OLS, a Ridge refit list and LEAR (standard) with published prices; a column known only after the issue time,
    planted in one model, is a leak for that model only."""
    frame = spring.assign(load_actual=spring["load"])
    rules = {**VERIFY_RULES, "price": Availability.published("13:00", 1)}
    pipeline = ModelPipeline(information_set=InformationSet("11:45", "Europe/Brussels", rules=rules), engine="on")
    pipeline.add_model(OLSModel([*PREDICTORS, "price_d-1"], 28, name="ols"))
    pipeline.add_model(RidgeModel([*PREDICTORS, "price_d-1"], [14, 28], refit="7D", name="ridge"))
    pipeline.add_model(lear([*PREDICTORS, "price_d-1"], 28, newest_target="published", name="lear"))
    report = pipeline.verify(frame, "2023-03-20", "2023-03-30", days=1)
    report.assert_ok()
    assert set(report.table()["status"]) == {"ok"}

    leaky = ModelPipeline(information_set=InformationSet(rules=VERIFY_RULES), engine="on")
    leaky.add_model(OLSModel([*PREDICTORS, "load_actual"], 28, name="late"))
    leaky.add_model(OLSModel(PREDICTORS, 28, name="clean"))
    leaky.add_model(RidgeModel(PREDICTORS, 28, name="clean_ridge"))
    table = leaky.verify(frame, "2023-04-12", "2023-04-12", days=1, positive_control=False).table().set_index("model")
    assert table.loc["late", "status"] == "leak" and table.loc["late", "changed"] > 0
    assert table.loc["clean", "status"] == "ok" and table.loc["clean_ridge", "status"] == "ok"


def test_a_column_one_model_reads_moves_no_other_model(wide, tmp_path, in_process):
    """Through shared sums: perturbing a column that one model alone reads leaves every other model's records bitwise."""
    models = many_models(40)
    reads = {}
    for model in models:
        for name in model.predictors:
            reads.setdefault(name, []).append(model.name)
    alone = next(name for name, readers in sorted(reads.items()) if len(readers) == 1 and name.startswith("w"))
    reader = reads[alone][0]
    run_models(many_models(40), wide, tmp_path / "base")
    perturbed = wide.copy()
    perturbed[alone] = perturbed[alone] * 3.0 + 7.0
    run_models(many_models(40), perturbed, tmp_path / "perturbed")
    for model in models:
        a = lines(tmp_path / "base" / f"{model.name}.jsonl")
        b = lines(tmp_path / "perturbed" / f"{model.name}.jsonl")
        assert (a == b) == (model.name != reader), model.name


# --- the manifest -----------------------------------------------------------------------------------------------------


def keep_first_day(path):
    """Drop a results file's records of every day but its first, so that a resume has fits to make."""
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    first = min(record["run_date"] for record in records)
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in records if r["run_date"] == first))


def test_switching_the_engine_refuses_a_resume(spring, tmp_path):
    """`fit_path` is compared: engine on then off, or off then on, refuses for a model fitted from statistics, and
    `allow_manifest_mismatch=True` overrides (the file then mixes the two paths, as it says); a model the engine keeps on
    the model path (LassoCV) resumes either way."""
    days = ("2023-03-24", "2023-03-25")

    def pipeline(engine, model):
        return ModelPipeline(engine=engine).add_model(model)

    for first, then in (("on", "off"), ("off", "on")):
        save = tmp_path / first
        pipeline(first, OLSModel(PREDICTORS, 28, name="ols")).run(spring, *days, horizon=1, save_dir=str(save))
        stored = json.loads((save / "ols.manifest.json").read_text())
        assert stored.get("fit_path") == ("engine-1" if first == "on" else None)
        assert ("engine" in stored) == (first == "on")
        full = len(lines(save / "ols.jsonl"))
        keep_first_day(save / "ols.jsonl")
        with pytest.raises(ManifestMismatchError, match="fit_path"):
            pipeline(then, OLSModel(PREDICTORS, 28, name="ols")).run(spring, *days, horizon=1, save_dir=str(save))
        pipeline(then, OLSModel(PREDICTORS, 28, name="ols")).run(
            spring, *days, horizon=1, save_dir=str(save), allow_manifest_mismatch=True
        )
        assert json.loads((save / "ols.manifest.json").read_text()).get("fit_path") == (
            "engine-1" if then == "on" else None
        )
        assert len(lines(save / "ols.jsonl")) == full
        pipeline(first, LassoCVModel(PREDICTORS, 28, cv=3, name="lasso")).run(
            spring, *days, horizon=1, save_dir=str(save)
        )
        keep_first_day(save / "lasso.jsonl")
        pipeline(then, LassoCVModel(PREDICTORS, 28, cv=3, name="lasso")).run(
            spring, *days, horizon=1, save_dir=str(save)
        )
        assert len(lines(save / "lasso.jsonl")) == full


def test_the_manifest_counts_the_last_runs_fits(spring, tmp_path):
    """`engine` provenance: fits and hand-backs by reason, the largest |z| of an issue row and how many were far out; a resume
    with nothing to fit leaves it, one that fits replaces it. An issue row 10^4 training standard deviations out is counted."""
    data = spring.copy()
    issue = (np.asarray(data.index.date) == datetime.date(2023, 3, 21)) & (data.index.hour == 5)
    data.loc[issue, "wind_d+1"] = data["wind_d+1"].std() * 1e4
    save = tmp_path / "save"

    def run():
        return (
            ModelPipeline(engine="on")
            .add_model(OLSModel(PREDICTORS, 28, name="ols"))
            .run(data, "2023-03-20", "2023-03-21", horizon=1, save_dir=str(save))
        )

    run()
    engine = json.loads((save / "ols.manifest.json").read_text())["engine"]
    assert engine["fits"] == 48 and engine["fallbacks"] == {}
    assert engine["far_issue_rows"] == 1 and engine["max_issue_z"] > 1e3
    run()  # nothing to fit
    assert json.loads((save / "ols.manifest.json").read_text())["engine"] == engine
    keep_first_day(save / "ols.jsonl")
    run()
    again = json.loads((save / "ols.manifest.json").read_text())["engine"]
    assert again["fits"] == 24 and again["far_issue_rows"] == 1 and again["max_issue_z"] == engine["max_issue_z"]


# --- the goldens ------------------------------------------------------------------------------------------------------


def test_the_golden_pipelines_through_the_engine(monkeypatch):
    """`tests/test_golden_metrics.py`'s pipelines with engine on: every pinned number within its golden tolerance, the same
    forecast counts (OLS and Ridge from statistics; LassoCV and LEAR's asinh models on the model path)."""
    from tests import test_golden_metrics as golden

    monkeypatch.setattr(golden, "ModelPipeline", functools.partial(ModelPipeline, engine="on"))
    for compute, path, tolerance in (
        (golden.compute_metrics, golden.GOLDEN_FILE, golden.RTOL),
        (golden.compute_lear_ridge_metrics, golden.LEAR_RIDGE_GOLDEN_FILE, golden.LEAR_RIDGE_RTOL),
    ):
        actual, expected = compute(), json.loads(path.read_text())
        assert actual.keys() == expected.keys()
        for model, values in expected.items():
            assert actual[model]["forecasts"] == values["forecasts"], model
            for key, value in values.items():
                if key != "forecasts":
                    assert actual[model][key] == pytest.approx(value, rel=tolerance[model]), (model, key)


# --- the estimator check ----------------------------------------------------------------------------------------------


def test_the_estimator_check_is_kept_per_parameter_values():
    """`_stats.estimator_params` answers once per (class, parameter values): the same answer for an equal estimator, another
    for other values (a float `max_iter`, `fit_intercept=1`), and none kept for a value that isn't plain."""
    from sklearn.linear_model import RidgeCV

    _stats._ESTIMATOR_CHECKS.clear()
    calls = []
    real = _stats._estimator_params

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_stats, "_estimator_params", counted)
        for _ in range(3):
            assert _stats.estimator_params(LinearRegression(), LinearRegression, ("fit_intercept",)) is not None
        assert len(calls) == 1
        assert _stats.estimator_params(LinearRegression(fit_intercept=1), LinearRegression, ("fit_intercept",)) is None
        assert _stats.estimator_params(LinearRegression(fit_intercept=1), LinearRegression, ("fit_intercept",)) is None
        assert len(calls) == 2
        grid = np.array([0.1, 1.0])
        for _ in range(2):
            _stats.estimator_params(RidgeCV(alphas=grid, cv=None), RidgeCV, ("alphas", "fit_intercept"), {"cv": None})
        assert len(calls) == 4  # an array isn't plain: not kept
        assert _stats.estimator_params(
            RidgeCV(alphas=[0.1, 1.0], cv=None), RidgeCV, ("alphas", "fit_intercept"), {"cv": None}
        )
        assert (
            _stats.estimator_params(
                RidgeCV(alphas=[0.1, 1.0], cv=5), RidgeCV, ("alphas", "fit_intercept"), {"cv": None}
            )
            is None
        )
        assert len(calls) == 6
    _stats._ESTIMATOR_CHECKS.clear()
