"""`ResidualModel` and `HistGBModel` (step 6k): a base's out-of-sample forecasts corrected by a learner fitted on their residuals.

The base runs as a plain run of its own over a warm-up into `<name>.base.jsonl`; the learner trains on the rows of the issue
days in its span whose base forecast exists, target `y - base forecast`, and a forecast is the issue day's base forecast plus the
learner's prediction. Most tests use `PooledLS`, a least-squares stand-in for a pooled learner that can record what it trains on
(or predict 0 in its transformed target, so a forecast is the base's plus each period's median residual, computed by hand), and
an OLS base; `HistGBModel` is held to scikit-learn by hand and through the adapter suite.
"""

import dataclasses
import datetime
import json
import math
import os
import warnings
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import (
    HistGBModel,
    LassoCVModel,
    LEARModel,
    ModelCard,
    NaiveModel,
    OLSModel,
    PooledModel,
    ResidualModel,
    RidgeModel,
)
from epftoolbox2.models._calendar import refit_anchor
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.models.base import BaseModel
from epftoolbox2.models.hgb import INVARIANT_TARGET
from epftoolbox2.models.residual import SKIP_NO_BASE_FORECAST, parse_base_name
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.presets import lear_predictors, tree_predictors
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path
from epftoolbox2.scalers.target import invariant_params
from tests.models import adapter_suite as suite
from tests.synthetic import make_market_data

BASE_INPUTS = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "price_d-1"]
LEARNER_INPUTS = ["load_d+{horizon}", "wind_d+{horizon}", "price_d-1", "period", "base_fc_d+{horizon}"]
DAY_LEVEL = "base_fc_h08_d+{horizon}"  # the base's 08:00 forecast of the target day, on every row of the date
TEST = ("2023-03-13", "2023-03-26")  # two "7D" blocks, Monday to Sunday; the spring switch is on 26 March (Warsaw)
RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
}
INFO = InformationSet(rules=RULES)
GOLDEN_FILE = Path(__file__).parent.parent / "golden" / "residual_metrics.json"


class InProcessPool:
    """Stands in for the worker pool, as in tests/models/test_day_units.py: the fits run in this process, so a learner can
    record what it saw."""

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
    PooledLS.calls.clear()
    yield
    PooledLS.calls.clear()


class PooledLS(BaseModel):
    """Least squares with an intercept over a pooled unit's rows, exactly rounded per row (`math.fsum`); `record` keeps each
    fit's (inputs, targets), as the unit hands them over, in `calls`; `zero` predicts 0 in the transformed target, the median
    residual of each period once mapped back. Its inputs are as they are and its targets each period's Invariant."""

    _unit_kind = "pooled"
    calls: list = []

    def __init__(
        self,
        predictors=LEARNER_INPUTS,
        training_window="expanding",
        refit="7D",
        *,
        record=False,
        zero=False,
        name="ls",
        scaler=INVARIANT_TARGET,
    ):
        super().__init__(predictors, training_window, name, scaler=scaler, refit=refit)
        self.record = record
        self.zero = zero

    @property
    def _model_kwargs(self):
        return {"record": self.record, "zero": self.zero}

    def _fit_predict(self, train_x, train_y, test_x):
        raise NotImplementedError("a pooled unit fits through _fit_model and _predict_block")

    def _fit_model(self, train_x, train_y):
        if self.record:
            type(self).calls.append((train_x.copy(), train_y.copy()))
        if self.zero:
            return None
        design = np.column_stack([np.ones(len(train_x)), train_x])
        return np.linalg.lstsq(design, train_y, rcond=None)[0]

    def _predict_block(self, fitted, rows):
        if fitted is None:
            return np.zeros(len(rows)), [[] for _ in rows]
        predictions = [math.fsum(np.asarray(row, dtype=np.float64) * fitted[1:]) + fitted[0] for row in rows]
        return np.array(predictions), [[] for _ in rows]


def base(name="ols", **params):
    return OLSModel(BASE_INPUTS, training_window=params.pop("training_window", 21), name=name, **params)


def residual(name="residual", learner=None, base_model=None, **params):
    params.setdefault("training_window", 28)
    return ResidualModel(base_model or base(), learner or PooledLS(), name=name, **params)


def frame(start="2023-01-01", days=110, tz="Europe/Warsaw", freq="1h", horizon=2, seed=3):
    data = make_market_data(start=start, days=days, tz=tz, freq=freq, horizon=horizon, noise=5.0, seed=seed)
    data["day_number"] = np.asarray(
        pd.DatetimeIndex(data.index.tz_localize(None)).normalize().map(pd.Timestamp.toordinal)
    )
    return data.astype({"day_number": np.float64})


@pytest.fixture(scope="module")
def market():
    """1 January to 20 April 2023 in Warsaw, over the spring switch, with each row's date as a number (`day_number`)."""
    return frame()


def lines(path):
    return sorted(line for line in Path(path).read_text().splitlines() if line.strip())


def keyed(records):
    return {(r["run_date"], r["hour"], r["horizon"]): r for r in records}


def run(model, data, test=TEST, horizon=2, **kwargs):
    return ModelPipeline().add_model(model).run(data, *test, horizon=horizon, **kwargs)


def warm_start(model, data, test=TEST, horizon=2):
    """The base's first issue day of a run of `model`, as the run works it out."""
    span = tuple(datetime.date.fromisoformat(day) for day in test)
    model._set_freq("1h")
    return model._warmup_start(model._base_model(), data, "price", horizon, range(1, horizon + 1), "1h", None, span)


# ---------------------------------------------------------------------------------------------------------------------------
# The base stage
# ---------------------------------------------------------------------------------------------------------------------------


def test_the_base_s_records_are_a_plain_run_of_the_base_over_its_days(market, tmp_path):
    """`<name>.base.jsonl` is, line for line, a plain run of the base from the warm-up start to the last issue day, and each
    forecast's `base_prediction` is the base's own record of its key: `prediction - base_prediction` is the learner's part."""
    model = residual(learner=PooledLS(LEARNER_INPUTS + [DAY_LEVEL]))
    start = warm_start(model, market)
    assert start == datetime.date(2023, 3, 13) - datetime.timedelta(days=28 + 2)  # the anchor - W - the largest horizon
    report = run(model, market, save_dir=tmp_path / "residual")
    base().run(market, start.isoformat(), TEST[1], horizon=2, save_to=str(tmp_path / "plain" / "ols.jsonl"))
    assert lines(tmp_path / "residual" / "residual.base.jsonl") == lines(tmp_path / "plain" / "ols.jsonl")
    stored = keyed(json.loads(line) for line in lines(tmp_path / "plain" / "ols.jsonl"))
    assert report.forecast_counts().iloc[0]["missing"] == 0
    records = [json.loads(line) for line in lines(tmp_path / "residual" / "residual.jsonl")]
    assert records and all(r["base_prediction"] == stored[key]["prediction"] for key, r in keyed(records).items())
    assert all(list(r)[-1] == "base_prediction" for r in records)
    assert all(r["actual"] == stored[key]["actual"] for key, r in keyed(records).items())
    manifest = json.loads(manifest_path(tmp_path / "residual" / "residual.jsonl").read_text())
    assert set(manifest["residual"]) == {"base_manifest"}
    assert manifest["model_params"]["base"]["class"] == "OLSModel"


def test_a_forecast_is_the_base_s_plus_each_period_s_median_residual_for_a_learner_predicting_zero(market):
    """Hand-built: a learner that predicts 0 in its transformed target forecasts each period's median residual (the centre of
    its Invariant) over its training rows: the rows of its span, dated up to the anchor - 1 - h, whose base forecast exists. So a
    forecast is the base's forecast of its key plus the median of `price_d+h - base` over those rows of its period."""
    model = residual(learner=PooledLS(zero=True))
    start = warm_start(model, market)
    base_records = keyed(base().run(market, start.isoformat(), TEST[1], horizon=2)._results)
    records = run(model, market).refs["residual"]._results
    local = pd.DatetimeIndex(market.index.tz_localize(None))
    targets = {hz: _calendar_target(market, hz) for hz in (1, 2)}
    checked = 0
    for r in records:
        anchor = refit_anchor(r["run_date"], "7D")
        hz, hour = r["horizon"], r["hour"]
        inputs = market[[name.replace("{horizon}", str(hz)) for name in LEARNER_INPUTS[:3]]].to_numpy()
        residuals = []
        for day in pd.date_range(anchor - datetime.timedelta(days=28 + hz), anchor - datetime.timedelta(days=1 + hz)):
            key = (day.date().isoformat(), hour, hz)
            for row in np.flatnonzero((local.normalize() == day) & (local.hour == hour)):
                if key in base_records and np.isfinite(inputs[row]).all() and np.isfinite(targets[hz][row]):
                    residuals.append(targets[hz][row] - base_records[key]["prediction"])
        center, _ = invariant_params(np.array(residuals))
        base_value = base_records[(r["run_date"], hour, hz)]["prediction"]
        assert r["base_prediction"] == base_value
        assert r["prediction"] == base_value + center
        checked += 1
    assert checked == len(records) > 600


def test_actual_is_the_target_to_the_bit_however_far_the_base_is(market):
    """`actual` is the period's target itself, never `(y - b) + b`, which loses bits where the base is far from the target: a
    `NaiveModel` base forecasting the load (about 1000) for a price of about 60."""
    model = residual(base_model=NaiveModel(["load_d+{horizon}"], name="naive"))
    records = run(model, market).refs["residual"]._results
    local = pd.DatetimeIndex(market.index.tz_localize(None))
    targets = {}
    for hz in (1, 2):
        values = _calendar_target(market, hz)
        for row, (stamp, value) in enumerate(zip(local, values)):
            targets[(stamp.date().isoformat(), stamp.hour, hz)] = (
                value  # the date's last row of the hour: the issue row
            )
    assert records and all(r["base_prediction"] > 500 for r in records)
    assert all(np.array_equal(r["actual"], targets[key], equal_nan=True) for key, r in keyed(records).items())


def check_out_of_sample(model_cls, data, tmp_path):
    """The learner's rows against residuals built by hand from the base's stored out-of-sample records: every training row is
    dated at or before its unit's anchor - 1 - h (its target at or before the cut, the anchor - 1), its `base_fc_d+h` is the
    base's record of its period, horizon and date, and its target is `price_d+h` less that record, each period's Invariant;
    every row of the span with a base record and finite inputs trains."""
    PooledLS.calls.clear()
    learner = PooledLS(LEARNER_INPUTS + ["day_number"], record=True)
    model = model_cls(base(), learner, training_window=28, name="residual")
    run(model, data, ("2023-03-13", "2023-03-19"), horizon=1, save_dir=tmp_path)
    stored = keyed(json.loads(line) for line in lines(tmp_path / "residual.base.jsonl"))
    ((train_x, train_y),) = PooledLS.calls  # one unit: horizon 1, anchor 13 March
    names = [name.replace("{horizon}", "1") for name in LEARNER_INPUTS + ["day_number"]]
    target = _calendar_target(data, 1)
    frame_inputs = data[[name for name in names if name not in ("period", "base_fc_d+1")]].to_numpy()
    dates = [datetime.date.fromordinal(int(n)) for n in train_x[:, names.index("day_number")]]
    periods = train_x[:, names.index("period")].astype(int)
    cut = datetime.date(2023, 3, 13) - datetime.timedelta(days=1)
    assert max(dates) == cut - datetime.timedelta(days=1)  # the row of 11 March: its target is 12 March, the cut
    local = pd.DatetimeIndex(data.index.tz_localize(None))
    expected = {}
    for day in pd.date_range(cut - datetime.timedelta(days=28), cut - datetime.timedelta(days=1)):
        for hour in range(24):
            key = (day.date().isoformat(), hour, 1)
            rows = np.flatnonzero((local.normalize() == day) & (local.hour == hour))
            for row in rows:
                if key in stored and np.isfinite(target[row]) and np.isfinite(frame_inputs[row]).all():
                    expected.setdefault(hour, []).append((day.date(), stored[key]["prediction"], target[row]))
    assert sorted(zip(dates, periods)) == sorted((d, h) for h, rows in expected.items() for d, _, _ in rows)
    for hour, rows in expected.items():
        on = np.flatnonzero(periods == hour)
        assert [stored[(dates[i].isoformat(), hour, 1)]["prediction"] for i in on] == [b for _, b, _ in rows]
        assert train_x[on, names.index("base_fc_d+1")].tolist() == [b for _, b, _ in rows]
        residuals = np.array([y - b for _, b, y in rows])
        center, scale = invariant_params(residuals)
        np.testing.assert_array_equal(train_y[on], np.arcsinh((residuals - center) / scale))


class FittedValues(ResidualModel):
    """A mutant: the base's in-sample fitted values (an OLS fitted on every row of the base's days, the targets it forecasts
    included) in place of its out-of-sample forecasts."""

    def _run_base(self, model, data, **arguments):
        forecasts, field = ResidualModel._run_base(model, data, **arguments)
        names = [name.replace("{horizon}", "1") for name in BASE_INPUTS]
        x, y = data[names].to_numpy(), _calendar_target(data, 1)
        kept = np.isfinite(x).all(axis=1) & np.isfinite(y)
        design = np.column_stack([np.ones(int(kept.sum())), x[kept]])
        coefs = np.linalg.lstsq(design, y[kept], rcond=None)[0]
        local = pd.DatetimeIndex(data.index[kept].tz_localize(None))
        fitted = dict(zip(zip(local.hour, local.normalize().map(pd.Timestamp.toordinal)), design @ coefs))
        return {
            key: float(fitted.get((key[0], key[2]), value)) if key[1] == 1 else value
            for key, value in forecasts.items()
        }, field


def test_the_learner_trains_on_the_base_s_out_of_sample_forecasts_up_to_its_cut(market, tmp_path):
    """The hand-built check of `check_out_of_sample` holds for the model and catches a mutant that trains on fitted values."""
    check_out_of_sample(ResidualModel, market, tmp_path / "model")
    with pytest.raises(AssertionError):
        check_out_of_sample(FittedValues, market, tmp_path / "mutant")


def _calendar_target(data, h):
    """`price_d+h` as the library builds it: the price at the row's wall-clock time h local days later."""
    from tests.synthetic import calendar_shift

    return calendar_shift(data["price"], h).to_numpy()


def test_no_base_forecast_on_the_issue_day_is_no_forecast(market):
    """A key whose base forecast is missing on its issue day (the base's input is NaN there) is skipped as `no_base_forecast`
    before any fit; the others are made."""
    data = market.copy()
    on = pd.DatetimeIndex(data.index.tz_localize(None)).normalize() == pd.Timestamp("2023-03-15")
    data.loc[on & (data.index.hour == 9), "wind_d+1"] = np.nan
    report = run(residual(), data)
    counts = report.forecast_counts().iloc[0]
    assert counts["missing"] == 1 and counts["produced"] == counts["planned"] - 1
    assert report.refs["residual"].skipped == {SKIP_NO_BASE_FORECAST: 1}


# ---------------------------------------------------------------------------------------------------------------------------
# Warm-up and what is refused
# ---------------------------------------------------------------------------------------------------------------------------


def test_the_warm_up_start(market):
    """An int learner window: the first anchor less the window and the largest horizon. An expanding one: the base's first
    issue day with a full window (frame start + its window + its deepest lag + the largest horizon), on one of its refit days if
    it refits. `warmup=` days before the first issue day overrides. A frame too short for any of them raises."""
    assert warm_start(residual(training_window=28), market, ("2023-03-15", "2023-03-26")) == datetime.date(2023, 2, 11)
    expanding = residual(training_window="expanding")
    assert warm_start(expanding, market) == datetime.date(2023, 1, 1) + datetime.timedelta(days=21 + 1 + 2)
    weekly = residual(training_window="expanding", base_model=base(refit="7D"))
    assert warm_start(weekly, market) == datetime.date(2023, 1, 30)  # 25 January is a Wednesday: the next Monday
    assert warm_start(residual(warmup=10), market) == datetime.date(2023, 3, 3)
    with pytest.raises(ValueError, match="start the frame"):
        warm_start(residual(training_window=60), market)
    with pytest.raises(ValueError, match="before the frame's first date"):
        warm_start(residual(warmup=200), market)
    with pytest.raises(ValueError, match="no residual to train on"):
        warm_start(residual(training_window="expanding", base_model=base(training_window=90)), market)


def test_what_a_residual_model_refuses():
    with pytest.raises(ValueError, match="refit"):
        residual(refit=None)
    with pytest.raises(ValueError, match="list of training windows"):
        residual(training_window=[28, 56])
    with pytest.raises(TypeError, match="learner must be a model"):
        ResidualModel(base(), "hgb")
    with pytest.raises(ValueError, match="fits nothing"):
        ResidualModel(base(), NaiveModel(["load_d+1"]))
    with pytest.raises(ValueError, match="a list of names"):
        ResidualModel(base(), PooledLS(lambda horizon: ["load_d+1"]))
    with pytest.raises(ValueError, match="can't be a ResidualModel"):
        ResidualModel(residual(), PooledLS())
    with pytest.raises(ValueError, match="predictor function"):

        def by_ctx(ctx):
            return BASE_INPUTS

        ResidualModel(OLSModel(by_ctx), PooledLS())
    with pytest.raises(ValueError, match="pooled over zones"):
        PooledModel(residual())
    with pytest.raises(ValueError, match="the learner's"):
        residual().predictors = ["load_d+1"]
    with pytest.warns(UserWarning, match="the learner's training_window=56 is replaced"):
        ResidualModel(base(), PooledLS(training_window=56), training_window=28)


def test_base_forecast_names(market):
    """`base_fc_d+{horizon}` and `base_fc_hHH_d+{horizon}` at the list's horizon; another horizon, another form or a frame column
    of such a name is refused before any fit."""
    assert parse_base_name("base_fc_d+2", "1h") == (None, 2)
    assert parse_base_name("base_fc_h05_d+1", "1h") == (5, 1)
    assert parse_base_name("base_fc_s20_d+1", "15min") == (20, 1)
    for name, freq in (
        ("base_fc_h24_d+1", "1h"),
        ("base_fc_s20_d+1", "1h"),
        ("base_fc_h05_d+1", "1D"),
        ("base_fc", "1h"),
    ):
        with pytest.raises(ValueError):
            parse_base_name(name, freq)
    with pytest.raises(ValueError, match="another horizon"):
        run(residual(learner=PooledLS(LEARNER_INPUTS + ["base_fc_d+1"])), market)
    with pytest.raises(ValueError, match="rename the column"):
        run(residual(), market.assign(**{"base_fc_d+1": 0.0}), horizon=1)


def test_base_by_name_resolves_to_a_copy_of_an_earlier_model(market, tmp_path, registered):
    """`base="ols"` runs a copy of the pipeline's model `ols` added before it, which saves as the name and loads back to the
    same forecasts; a name added after it, or of no model, raises before any fit."""
    by_name = ModelPipeline().add_model(base()).add_model(ResidualModel("ols", PooledLS(), training_window=28))
    by_model = ModelPipeline().add_model(residual())
    a = by_name.run(market, *TEST, horizon=2).refs["residual"]._results
    b = by_model.run(market, *TEST, horizon=2).refs["residual"]._results
    assert keyed(a) == keyed(b)
    path = tmp_path / "pipeline.yaml"
    by_name.save(path)
    assert "base: ols" in path.read_text()
    loaded = ModelPipeline.load(path)
    assert keyed(loaded.run(market, *TEST, horizon=2).refs["residual"]._results) == keyed(a)
    sliced = ModelPipeline().add_model(base()).add_model(ResidualModel("ols", PooledLS(), training_window=28))
    sliced._resolve_bases()  # what a Workflow array task does before it takes its slice of the models, and marks it
    sliced.models = sliced.models[1:]
    sliced._bases_from_the_whole_list = True
    assert keyed(sliced.run(market, *TEST, horizon=2).refs["residual"]._results) == keyed(a)
    # Bound in one pipeline, the model runs in no other without that name: a stale binding would run a base nobody named.
    with pytest.raises(ValueError, match="names no model"):
        ModelPipeline().add_model(sliced.models[0]).run(market, *TEST, horizon=2)
    with pytest.raises(ValueError, match="is added after it"):
        ModelPipeline().add_model(ResidualModel("ols", PooledLS())).add_model(base()).run(market, *TEST, horizon=2)
    with pytest.raises(ValueError, match="names no model"):
        ModelPipeline().add_model(ResidualModel("lasso", PooledLS())).run(market, *TEST, horizon=2)
    with pytest.raises(ValueError, match="names a model of a pipeline"):
        ResidualModel("ols", PooledLS()).run(market, *TEST, horizon=2)
    with pytest.raises(ValueError, match="must be unique"):
        ModelPipeline().add_model(base("residual.base")).add_model(residual()).run(market, *TEST, horizon=2)


def test_parquet_results_are_refused_before_any_fit(market, tmp_path):
    with pytest.raises(ValueError, match="base_prediction"):
        run(residual(), market, save_dir=tmp_path, save_format="parquet")
    assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------------------------------------------------------------------
# Resume, loops, the engine, YAML, verify: the adapter suite
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def registered(monkeypatch):
    from epftoolbox2.pipelines.model_pipeline import COMPONENT_REGISTRY

    monkeypatch.setitem(COMPONENT_REGISTRY["models"], "PooledLS", __name__)


def test_another_learner_window_or_refit_refuses_before_the_base_runs(market, tmp_path):
    """The residual model's own window and refit rule are compared before the base runs, as its base and learner are: the
    refusal leaves the base's results and manifest as they were."""
    run(residual(base_model=base(refit="7D")), market, save_dir=tmp_path)
    kept = {name: (tmp_path / name).read_bytes() for name in ("residual.base.jsonl", "residual.base.manifest.json")}
    for changed, field in ((dict(training_window=35), "training_window"), (dict(refit="1D"), "training_cut.refit")):
        with pytest.raises(ManifestMismatchError, match=field):
            run(residual(base_model=base(refit="7D"), **changed), market, save_dir=tmp_path)
        assert {name: (tmp_path / name).read_bytes() for name in kept} == kept


def test_a_check_left_by_a_failed_pipeline_isn_t_run_for_a_changed_base(market):
    """A pipeline whose other model fails its check leaves the residual model's check of its base behind; replacing the base
    afterwards runs the new one, as a fresh model with it does."""
    model = residual(base_model=base(refit="7D"))
    bad = OLSModel(["no_such_column_d+{horizon}"], name="bad")
    with pytest.raises(ValueError, match="not in the data"):
        ModelPipeline().add_model(model).add_model(bad).run(market, *TEST, horizon=2)
    model.base = base(training_window=14, refit="7D")
    fresh = residual(base_model=base(training_window=14, refit="7D"))
    assert keyed(model.run(market, *TEST, horizon=2)._results) == keyed(fresh.run(market, *TEST, horizon=2)._results)


def test_a_resume_continues_the_base_from_its_file_and_a_changed_base_refuses(market, tmp_path):
    """Cut short (40 % of the records gone, and the base's file cut too) and resumed: the fresh run's records, base and model.
    A changed base refuses through the base's own manifest; with the base's files gone, through this model's (`residual`)."""
    fresh, resumed = tmp_path / "fresh", tmp_path / "resumed"
    run(residual(), market, save_dir=fresh)
    run(residual(), market, save_dir=resumed)
    for name in ("residual.jsonl", "residual.base.jsonl"):
        path = resumed / name
        kept = [line for i, line in enumerate(lines(path)) if i % 5 not in (1, 3)]
        path.write_text("".join(f"{line}\n" for line in kept))
    run(residual(), market, save_dir=resumed)
    for name in ("residual.jsonl", "residual.base.jsonl"):
        assert lines(resumed / name) == lines(fresh / name)
    changed = residual(base_model=base(training_window=14))
    with pytest.raises(ManifestMismatchError, match="residual.base"):
        run(changed, market, save_dir=resumed)
    for name in ("residual.base.jsonl", "residual.base.manifest.json"):
        (resumed / name).unlink()
    # Refused before the base runs: nothing is written, and the model's own configuration still resumes afterwards.
    with pytest.raises(ManifestMismatchError, match="Nothing was run or written") as refused:
        run(residual(base_model=base(training_window=14)), market, save_dir=resumed)
    assert "  - model_params:" in str(refused.value)
    assert not (resumed / "residual.base.jsonl").exists()
    run(residual(), market, save_dir=resumed)
    assert lines(resumed / "residual.base.jsonl") == lines(fresh / "residual.base.jsonl")
    # The base's configuration as its manifest recorded it (`residual.base_manifest`) is compared after its run.
    path = manifest_path(resumed / "residual.jsonl")
    manifest = json.loads(path.read_text())
    path.write_text(json.dumps({**manifest, "residual": {"base_manifest": "another base"}}))
    with pytest.raises(ManifestMismatchError, match="  - residual:"):
        run(residual(), market, save_dir=resumed)


def test_the_adapter_suite(market, tmp_path, monkeypatch, registered):
    """YAML round trip with the base and learner nested, a loop and its shards, a resume after deleting 40 % of the records,
    `engine="on"`, the pre-flight (no extra) and the provenance (none for this learner)."""

    def make(name, **params):
        return residual(name, **params)

    suite.check_yaml_round_trip(make, market, tmp_path, TEST, horizon=2)
    variants = ({}, {"training_window": 21}, {"refit": "1D"})
    suite.check_loop_and_shards(make, market, tmp_path / "loop", TEST, horizon=2, variants=variants)
    suite.check_resume(make, market, tmp_path / "resume", TEST, horizon=2)
    suite.check_engine(make, market, TEST, horizon=2)
    suite.check_preflight_without_extra(make, monkeypatch, market, TEST)


@pytest.mark.filterwarnings("ignore:overflow encountered in sinh")  # the control's noise, mapped back by sinh
def test_verify_is_ok_through_both_stages_and_finds_a_leak_in_either(market):
    """`verify` with its control, rerun, co-batch and cut-frame checks: ok on a block's first, middle and last (spring) day. A
    base reading a value unknown at the issue time is a leak through the residual model's forecasts, and so is a learner."""

    # Every probe runs both stages: a weekly base and the days the warm-ups and blocks of these issue days read keep it short.
    def make(name, **params):
        return residual(name, learner=PooledLS(LEARNER_INPUTS + [DAY_LEVEL]), base_model=base(refit="7D"), **params)

    data = market.loc["2023-01-15":"2023-03-31"]
    suite.check_verify(make, data, INFO, ["2023-03-20", "2023-03-22", "2023-03-26"], items=6)
    late = dict(RULES, wind=Availability.published("23:00", 1))  # wind_d+1 published on D at 23:00, after the issue
    without_wind = [name for name in BASE_INPUTS if not name.startswith("wind")]
    for leaky in (  # wind read by the base alone, then by the learner alone
        residual(
            learner=PooledLS([name for name in LEARNER_INPUTS if not name.startswith("wind")]),
            base_model=base(refit="7D"),
        ),
        residual(base_model=OLSModel(without_wind, training_window=21, refit="7D", name="ols")),
    ):
        report = ModelPipeline(information_set=InformationSet(rules=late)).add_model(leaky)
        found = report.verify(data, "2023-03-20", "2023-03-20", horizon=1, days=1, positive_control=False, rerun=False)
        assert found.table().iloc[0]["status"] == "leak", found.table().to_dict("records")


def test_the_adapter_suite_s_calendar_counts():
    """23 forecasts for the spring target day, 24 for the autumn one with the repeated hour's actual NaN."""
    both = pd.concat([frame(start="2023-01-15", days=75, horizon=1), frame(start="2023-08-25", days=70, horizon=1)])
    suite.check_dst_counts(lambda name, **p: residual(name, **p), both, "2023-03-26", "2023-10-29")


def test_with_validation_each_window_is_a_plain_run_over_it(market):
    validation, test = ("2023-03-13", "2023-03-19"), ("2023-03-20", "2023-03-26")
    report = ModelPipeline().add_model(residual()).run(market, validation=validation, test=test, horizon=2)
    records = report.refs["residual"]._results
    for name, (start, end) in (("validation", validation), ("test", test)):
        alone = keyed(run(residual(), market, (start, end)).refs["residual"]._results)
        windowed = {
            key: {k: v for k, v in r.items() if k != "window"}
            for key, r in keyed(records).items()
            if r["window"] == name
        }
        assert windowed == alone


# ---------------------------------------------------------------------------------------------------------------------------
# Warm-up forms, refusals, the base's masks and DST
# ---------------------------------------------------------------------------------------------------------------------------


def test_a_dated_warmup_makes_no_forecast_depend_on_the_test_start_and_an_int_one_refuses_another(market, tmp_path):
    """`warmup="2023-02-01"` starts the base on that date whatever the run's days: two runs from 13 and from 20 March agree on
    their common keys to the bit, an expanding learner and a base with first-day masks included. `warmup=14` starts it 14 days
    before the first issue day, so the manifest's `residual` field records that start and a resume from another refuses."""
    dated = dict(training_window="expanding", warmup="2023-02-01")
    early = keyed(run(residual(**dated), market, ("2023-03-13", "2023-03-26")).refs["residual"]._results)
    late = keyed(run(residual(**dated), market, ("2023-03-20", "2023-03-26")).refs["residual"]._results)
    same = [key for key in late if {**late[key], "day_in_test": 0} == {**early[key], "day_in_test": 0}]
    assert late and len(same) == len(late)  # `day_in_test` counts from each run's own start
    assert warm_start(residual(**dated), market) == datetime.date(2023, 2, 1)
    days = dict(training_window="expanding", warmup=14)
    run(residual(**days), market, ("2023-03-20", "2023-03-26"), save_dir=tmp_path)
    assert json.loads(manifest_path(tmp_path / "residual.jsonl").read_text())["residual"]["base_start"] == "2023-03-06"
    with pytest.raises(ManifestMismatchError, match="residual.base_start"):
        run(residual(**days), market, ("2023-03-13", "2023-03-26"), save_dir=tmp_path)
    with pytest.raises(ValueError, match="after the first issue day"):
        warm_start(residual(warmup="2023-03-20"), market)


def test_an_expanding_base_needs_a_warmup(market):
    """An expanding base's first fits train on a handful of rows, which every later learner fit would include: refused
    without `warmup=`, run with one."""
    expanding = base(training_window="expanding")
    with pytest.raises(ValueError, match="an expanding base"):
        run(residual(base_model=expanding), market)
    assert warm_start(residual(base_model=expanding, warmup="2023-02-01"), market) == datetime.date(2023, 2, 1)


def test_base_prediction_is_the_pipeline_s_own_base_records_where_no_mask_reads_a_first_day(market):
    """The base copy starts at the warm-up, so a base that reads its binary-column masks on its run's first day (refit=None,
    binary_mask=None) can give other forecasts than the same model run beside it: such a base warns. A refit base and one with
    `binary_mask="fit"` give `base_prediction` equal to the pipeline's own base records on every shared key."""
    for settings in ({"refit": "7D"}, {"binary_mask": "fit"}):
        pipeline = (
            ModelPipeline().add_model(base(**settings)).add_model(ResidualModel("ols", PooledLS(), training_window=28))
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)  # no warning about the base's masks
            report = pipeline.run(market, *TEST, horizon=2)
        own = keyed(report.refs["ols"]._results)
        corrected = keyed(report.refs["residual"]._results)
        assert corrected and all(r["base_prediction"] == own[key]["prediction"] for key, r in corrected.items())
    with pytest.warns(UserWarning, match="binary_mask='fit'"):
        run(residual(), market)


def test_base_forecasts_of_a_spring_switch_s_skipped_hour_take_two_issue_days_per_horizon(market):
    """`base_fc_h02_d+{horizon}` has no value where the base has no 02:00 forecast: issued the day before the switch's target
    day (02:00 isn't on its clock) and on the switch day itself (no 02:00 row to issue from). A list naming it forecasts
    nothing those days, so the residual model skips 2 issue days x 23 periods per horizon that its base forecasts."""
    learner = PooledLS(LEARNER_INPUTS + ["base_fc_h02_d+{horizon}"])
    pipeline = ModelPipeline().add_model(base()).add_model(ResidualModel("ols", learner, training_window=28))
    report = pipeline.run(market, "2023-03-20", "2023-03-26", horizon=2)
    counts = report.forecast_counts().set_index("model")
    assert counts.loc["ols", "missing"] == 0
    assert report.refs["residual"].skipped == {"missing_test_predictors": 2 * 23 * 2}
    skipped = set(keyed(report.refs["ols"]._results)) - set(keyed(report.refs["residual"]._results))
    assert sorted({(day, horizon) for day, _, horizon in skipped}) == [
        ("2023-03-24", 2),
        ("2023-03-25", 1),
        ("2023-03-26", 1),
        ("2023-03-26", 2),
    ]


def test_a_direct_run_into_parquet_is_refused(market, tmp_path):
    with pytest.raises(ValueError, match="base_prediction"):
        residual().run(market, *TEST, horizon=2, save_to=str(tmp_path / "residual.parquet"))
    assert not list(tmp_path.iterdir())


def test_results_convert_to_parquet_refuses_a_residual_model_s_before_converting_anything(market, tmp_path):
    from epftoolbox2.results.parquet_store import convert

    pipeline = ModelPipeline().add_model(base()).add_model(ResidualModel("ols", PooledLS(), training_window=28))
    pipeline.run(market, *TEST, horizon=2, save_dir=tmp_path)
    before = sorted(path.name for path in tmp_path.iterdir())
    for model in (None, "residual", "residual.base"):  # the base's results too: the residual model reads them as JSONL
        with pytest.raises(ValueError, match="Nothing was converted") as refused:
            convert(tmp_path, "parquet", model=model)
        assert "residual.base" in str(refused.value) or model == "residual"
    assert sorted(path.name for path in tmp_path.iterdir()) == before
    records = len(lines(tmp_path / "ols.jsonl"))
    assert convert(tmp_path, "parquet", model="ols") == [("ols", records)]  # a model of its own still converts


def _workflow(tmp_path, models, save_format="jsonl", index=None):
    from epftoolbox2.pipelines.workflow import Workflow

    config = {
        "environment": {} if index is None else {"model_index": index},
        "data_pipeline": {"start": "2023-01-01", "end": "2023-04-20", "sources": []},
        "model_pipeline": {
            "test_start": "2023-03-13",
            "test_end": "2023-03-14",
            "horizon": 1,
            "save_dir": str(tmp_path / "runs"),
            "save_format": save_format,
            "models": models,
        },
    }
    path = tmp_path / f"workflow_{save_format}_{index}.yaml"
    path.write_text(yaml.safe_dump(config))
    return Workflow.load(path)


WORKFLOW_MODELS = [
    {"class": "OLSModel", "params": {"predictors": BASE_INPUTS, "training_window": 21, "name": "ols"}},
    {
        "class": "ResidualModel",
        "params": {
            "base": "ols",
            "learner": {"class": "HistGBModel", "params": {"predictors": LEARNER_INPUTS, "max_iter": 5}},
            "training_window": 28,
        },
    },
]


def test_a_workflow_array_task_runs_a_residual_model_whose_base_is_another_task_s(market, tmp_path, monkeypatch):
    """Task 1 of an array of one model a task holds the ResidualModel alone; its base="ols" names task 0's model, which the
    Workflow resolves on the whole list before it takes the slice. A Parquet run, and a base named nowhere, are refused before
    the data pipeline runs, also in a task holding the base alone."""
    from epftoolbox2.pipelines.workflow import Workflow

    fetched = []
    monkeypatch.setattr(Workflow, "_run_data", lambda self, dp=None: fetched.append(1) or market)
    _workflow(tmp_path, WORKFLOW_MODELS, index=1).run()
    assert lines(tmp_path / "runs" / "residual.jsonl") and fetched == [1]
    fetched.clear()
    for index in (None, 0):
        with pytest.raises(ValueError, match="base_prediction"):
            _workflow(tmp_path, WORKFLOW_MODELS, "parquet", index).run()
    typo = [WORKFLOW_MODELS[0], {**WORKFLOW_MODELS[1], "params": {**WORKFLOW_MODELS[1]["params"], "base": "olss"}}]
    for index in (None, 0, 1):
        with pytest.raises(ValueError, match="names no model"):
            _workflow(tmp_path, typo, index=index).run()
    assert fetched == []


# ---------------------------------------------------------------------------------------------------------------------------
# Construction, YAML, provenance, explain
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "learner",
    [
        lambda: HistGBModel(LEARNER_INPUTS),
        lambda: OLSModel(["load_d+{horizon}", "price_d-1", "base_fc_d+{horizon}"]),
        lambda: RidgeModel(["load_d+{horizon}", "price_d-1", "base_fc_d+{horizon}"]),
        lambda: LassoCVModel(["load_d+{horizon}", "price_d-1", "base_fc_d+{horizon}"]),
        lambda: LEARModel(["load_d+{horizon}", "price_d-1", "base_fc_d+{horizon}"]),
    ],
)
def test_no_warning_for_a_learner_at_its_class_s_defaults_nor_on_load(learner, tmp_path, registered):
    """A learner left at its class's own `training_window` and `refit` (`OLSModel`'s refit is BaseModel's; LEAR's windows a
    list) warns of nothing, and its YAML leaves out the learner's window, refit and name, which this model replaces, so a load
    warns of nothing either."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model = ResidualModel(base(), learner(), training_window=28)
        path = tmp_path / "pipeline.yaml"
        ModelPipeline().add_model(model).save(path)
        loaded = ModelPipeline.load(path).models[0]
    saved = yaml.safe_load(path.read_text())["models"][0]["params"]["learner"]["params"]
    assert not {"training_window", "refit", "name"} & set(saved)
    assert loaded._model_kwargs == model._model_kwargs


def test_construction_errors_name_the_model_and_a_pooled_base_says_why():
    with pytest.raises(TypeError, match=r"ResidualModel\('mine'\): learner must be a model"):
        ResidualModel(base(), "hgb", name="mine")
    with pytest.raises(ValueError, match=r"ResidualModel\('mine'\): warmup"):
        ResidualModel(base(), PooledLS(), warmup=0, name="mine")
    with pytest.raises(ValueError, match="a PooledModel can't be a base"):
        ResidualModel(PooledModel(base(refit="7D")), PooledLS())
    with pytest.raises(ValueError, match="a PooledModel can't be a base"):
        ModelPipeline().add_model(PooledModel(base(refit="7D"), name="p")).add_model(
            ResidualModel("p", PooledLS())
        )._check_model_names()
    assert (
        repr(ResidualModel(base("o"), PooledLS()))
        == "ResidualModel(base=OLSModel('o'), learner=PooledLS('ls'), name='residual')"
    )
    assert repr(ResidualModel("o", PooledLS())) == "ResidualModel(base='o', learner=PooledLS('ls'), name='residual')"
    with pytest.raises(ValueError):
        parse_base_name("base_fc_h5_d+1", "1h")


def test_a_model_under_an_unlisted_parameter_isn_t_saved(tmp_path):
    """A model-valued attribute that its class doesn't list in `_NESTED_MODELS` raises at save time: a load couldn't rebuild
    it. A PooledModel saves its model as before."""
    model = base()
    model.helper = OLSModel(BASE_INPUTS, name="helper")
    with pytest.raises(TypeError, match="_NESTED_MODELS"):
        ModelPipeline().add_model(model).save(tmp_path / "bad.yaml")
    pooled = PooledModel(base(refit="7D"), over=("zone", "horizon"), name="p")
    ModelPipeline().add_model(pooled).save(tmp_path / "pooled.yaml")
    again = ModelPipeline.load(tmp_path / "pooled.yaml").models[0]
    assert (type(again.model), again.over, again.name) == (OLSModel, pooled.over, "p")


FAKE_CARD = ModelCard(
    family="fake-base",
    extra=None,
    licence="Fake-1.0",
    licence_url="https://example.invalid/fake",
    commercial_use=False,
    revenue_cap=None,
    attribution="Fake Labs",
    weights=None,
    runtime=("numpy",),
    corpus="undocumented",
    corpus_end=datetime.date(2023, 3, 1),
    released=datetime.date(2022, 1, 1),
)


class CardedOLS(OLSModel):
    """OLS with a card of its own: a carded base, whose card a residual model's provenance shows."""

    _card = FAKE_CARD
    _engine_ok = False


class MidFebruaryOLS(CardedOLS):
    """The same, its corpus ending on 15 February: inside a warm-up from 12 February, before one from 19 February."""

    _card = dataclasses.replace(FAKE_CARD, corpus_end=datetime.date(2023, 2, 15))


def test_a_resume_from_a_later_start_keeps_the_base_s_earliest_first_issue_day(market, tmp_path):
    """A run from 13 March (the base from 12 February) resumed from 20 March (from 19 February): the records of the first run
    stay in the file, their learners trained on base forecasts from 12 February, so the base's verdict stays judged from it,
    `possibly contaminated` by a corpus ending on 15 February, in the file and in the report from it."""

    def pipeline():
        learner = OLSModel(["load_d+{horizon}", "price_d-1", "base_fc_d+{horizon}"])
        base_model = MidFebruaryOLS(BASE_INPUTS, training_window=21, refit="7D", name="b")
        return ModelPipeline().add_model(ResidualModel(base_model, learner, training_window=28))

    pipeline().run(market, "2023-03-13", "2023-03-26", horizon=1, save_dir=tmp_path)
    resumed = pipeline().run(market, "2023-03-20", "2023-03-26", horizon=1, save_dir=tmp_path)
    stored = json.loads(manifest_path(tmp_path / "residual.jsonl").read_text())["residual_base"]
    assert stored["first_issue_day"] == "2023-02-12"
    report = pipeline().report(tmp_path, "2023-03-13", "2023-03-26", horizon=1)
    for table in (report.provenance(), resumed.provenance()):
        assert table.set_index("model").loc["residual.base", "test_verdict"] == "possibly contaminated", table


def test_the_provenance_shows_the_base_s_card_judged_from_its_first_issue_day(market, tmp_path):
    """A residual model over an OLS learner (no card) shows its base's card as a row of its own, `<name>.base`, in memory and
    from the files; its verdict is judged from the base's first issue day (11 February), the warm-up's: a corpus ending on 1
    March, before the test, reaches the base forecasts the learner trains on."""
    learner = OLSModel(["load_d+{horizon}", "price_d-1", "base_fc_d+{horizon}"])
    model = ResidualModel(CardedOLS(BASE_INPUTS, training_window=21, name="b"), learner, training_window=28)
    pipeline = ModelPipeline().add_model(model)
    reports = [pipeline.run(market, *TEST, horizon=2, save_dir=tmp_path)]
    reports.append(pipeline.report(tmp_path, *TEST, horizon=2))
    for report in reports:
        table = report.provenance()
        assert table["model"].tolist() == ["residual.base"], table
        row = table.iloc[0]
        assert (row["family"], row["licence"], row["test_verdict"]) == (
            "fake-base",
            "Fake-1.0",
            "possibly contaminated",
        )


def test_explain_shows_the_base_s_forecasts_as_such_and_the_base_s_inputs(market):
    pipeline = ModelPipeline().add_model(base()).add_model(ResidualModel("ols", PooledLS(), training_window=28))
    out = pipeline.explain(market, "residual", test=TEST, horizon=1, hour=8, echo=False)
    table = out["predictors"].set_index("predictor")
    assert table.loc["base_fc_d+1", "status"] == "base forecast"
    assert (
        "not in data" not in table.loc[[name for name in table.index if name.startswith("base_fc")], "status"].tolist()
    )
    assert out["base_predictors"]["predictor"].tolist() == [name.replace("{horizon}", "1") for name in BASE_INPUTS]


# ---------------------------------------------------------------------------------------------------------------------------
# HistGBModel
# ---------------------------------------------------------------------------------------------------------------------------

HGB = dict(max_iter=15, min_samples_leaf=5)
HGB_INPUTS = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "price_d-1", "period"]


def test_tree_predictors_are_lear_s_then_the_period_then_the_own_period_blocks():
    names = tree_predictors(["load_forecast"])
    assert len(names) == 183
    assert names[:175] == lear_predictors(["load_forecast"])
    assert names[175:] == [
        "period",
        "price",
        "price_d-1",
        "price_d-2",
        "price_d-6",
        "load_forecast_d+{horizon}",
        "load_forecast",
        "load_forecast_d-6",
    ]


def test_a_pooled_hgb_fit_is_scikit_learn_s_on_the_stacked_rows_by_hand(market):
    """One pooled unit by hand: the (date, period) rows of its span with finite inputs and target, by date then period, each
    period's target through its own Invariant, `HistGradientBoostingRegressor(loss="absolute_error", early_stopping=False)`
    fitted on them; a forecast is its row's prediction mapped back. To the bit."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits

    model = HistGBModel(HGB_INPUTS, training_window=28, **HGB)
    records = model.run(market, "2023-03-14", "2023-03-14", horizon=1)._results
    local = pd.DatetimeIndex(market.index.tz_localize(None))
    names = [name.replace("{horizon}", "1") for name in HGB_INPUTS]
    table = market.assign(period=local.hour, date=local.normalize(), target=_calendar_target(market, 1))
    span = (table["date"] >= pd.Timestamp("2023-02-12")) & (table["date"] <= pd.Timestamp("2023-03-11"))
    train = table[span].dropna(subset=[*names, "target"]).sort_values(["date", "period"], kind="stable")
    centers = {p: invariant_params(group["target"].to_numpy()) for p, group in train.groupby("period")}
    y = np.array([np.arcsinh((v - centers[p][0]) / centers[p][1]) for v, p in zip(train["target"], train["period"])])
    with threadpool_limits(limits=1, user_api="openmp"):
        fitted = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=15,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=5,
            l2_regularization=1.0,
            max_features=0.5,
            early_stopping=False,
            random_state=0,
        ).fit(train[names].to_numpy(), y)
        issue = table[table["date"] == pd.Timestamp("2023-03-14")].groupby("period").tail(1)
        z = fitted.predict(issue[names].to_numpy())
    expected = {int(p): centers[p][0] + centers[p][1] * np.sinh(v) for p, v in zip(issue["period"], z)}
    assert {r["hour"]: r["prediction"] for r in records} == expected


def test_hgb_each_period_is_one_fit_per_period_and_the_same_in_a_worker_and_here(market):
    """`periods="each"`: per-period units of the refit calendar with one block call each, the unit kind `"period"`; the forecasts
    are the same in a forked worker as in this process (one OpenMP thread for every fit)."""
    each = HistGBModel(HGB_INPUTS[:-1], training_window=28, periods="each", **HGB)
    assert each._unit_kind == "period" and each.block_predict and each._predicts_together
    here = keyed(each.run(market, *TEST, horizon=1)._results)
    with pytest.MonkeyPatch.context() as patch:
        from concurrent.futures import ProcessPoolExecutor

        patch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", ProcessPoolExecutor)
        forked = keyed(
            HistGBModel(HGB_INPUTS[:-1], training_window=28, periods="each", **HGB)
            .run(market, *TEST, horizon=1)
            ._results
        )
    # 14 days of 24 periods, less the spring target day's 02:00 (not planned) and the switch day's own 02:00 (no issue row)
    assert forked == here and len(here) == 14 * 24 - 2


def test_hgb_settings_and_card():
    with pytest.raises(ValueError, match="periods"):
        HistGBModel(HGB_INPUTS, periods="hourly")
    with pytest.raises(ValueError, match="loss"):
        HistGBModel(HGB_INPUTS, loss="huber")
    with pytest.raises(ValueError, match="needs refit"):
        HistGBModel(HGB_INPUTS, refit=None).run(make_market_data(days=60), "2023-02-20", "2023-02-21", horizon=1)
    model = HistGBModel(HGB_INPUTS)
    assert not model._engine_capable() and model.card.extra is None and model.card.runtime == ("scikit-learn>=1.4",)


@pytest.mark.filterwarnings("ignore:overflow encountered in sinh")
def test_the_adapter_suite_for_hgb(market, tmp_path, monkeypatch):
    def make(name, **params):
        # `period` is a feature of a pooled fit only: a per-period unit's rows are of one period
        inputs = HGB_INPUTS[:-1] if params.get("periods") == "each" else HGB_INPUTS
        return HistGBModel(inputs, **{"training_window": 28, **HGB, **params}, name=name)

    suite.check_yaml_round_trip(make, market, tmp_path, TEST, horizon=1)
    suite.check_loop_and_shards(make, market, tmp_path / "loop", TEST, horizon=1, variants=({}, {"periods": "each"}))
    suite.check_resume(make, market, tmp_path / "resume", TEST, horizon=1)
    suite.check_engine(make, market, TEST, horizon=1)
    suite.check_provenance(make, market, tmp_path / "provenance", TEST)
    # A tree of few iterations can leave a forecast in one leaf under the control's noise: 60 move every one.
    suite.check_verify(lambda name: make(name, max_iter=60), market, INFO, ["2023-03-20", "2023-03-26"], items=6)


def test_a_residual_hgb_s_card_and_runtime_are_its_learner_s(market, tmp_path):
    """The residual model shows its learner's card (scikit-learn's version recorded in the manifest's `runtime`, compared on
    resume) in `report.provenance()`."""
    learner = HistGBModel(LEARNER_INPUTS + [DAY_LEVEL], **HGB)
    suite.check_provenance(lambda name: residual(name, learner=learner), market, tmp_path, TEST)
    run(residual(learner=learner), market, save_dir=tmp_path / "run")
    manifest = json.loads(manifest_path(tmp_path / "run" / "residual.jsonl").read_text())
    assert "scikit-learn" in manifest["runtime"]["packages"]


# ---------------------------------------------------------------------------------------------------------------------------
# The golden numbers
# ---------------------------------------------------------------------------------------------------------------------------

# A ResidualModel with an OLS base and a small HGB learner. HGB is exact on one OpenMP thread, but its numbers move between
# scikit-learn releases (B1 of step 6k: 1.8.0 and 1.9.1 differ by up to 33 EUR/MWh on a year of DE forecasts), and the 3.15 and
# newest-dependencies CI cells resolve newer ones than the lock's: the residual model's numbers are compared on the release the
# file was written with (`scikit-learn` in it), its forecast count and the OLS base's numbers on every one.
GOLDEN_RTOL = 1e-9


def compute_residual_metrics() -> dict:
    """The golden test's data (`make_market_data(noise=5.0, seed=42)`, over the spring switch) and two weeks to 27 March:
    an OLS base at W 21 and its residual model, `HistGBModel` (40 iterations) over 28 days of rows, weekly."""
    df = make_market_data(noise=5.0, seed=42)
    learner = HistGBModel(LEARNER_INPUTS + [DAY_LEVEL], max_iter=40, min_samples_leaf=10)
    pipeline = (
        ModelPipeline()
        .add_model(OLSModel(BASE_INPUTS, training_window=21, name="OLS"))
        .add_model(ResidualModel("OLS", learner, training_window=28, name="Residual"))
        .add_evaluator(MAEEvaluator())
        .add_evaluator(RMSEEvaluator())
    )
    report = pipeline.run(df, test_start="2023-03-14", test_end="2023-03-27", target="price", horizon=2)
    tables = report.compute_all()
    metrics = {name: {"forecasts": len(details)} for name, details in report.iter_details()}
    for row in tables["summary"].to_dict("records"):
        metrics[row["model"]].update({k: float(row[k]) for k in ("MAE", "RMSE")})
    for row in tables["horizon"].to_dict("records"):
        metrics[row["model"]][f"MAE_h{row['horizon']}"] = float(row["MAE"])
    return metrics


def _sklearn_version() -> str:
    import sklearn

    return sklearn.__version__


def test_residual_metrics_match_golden(monkeypatch):
    actual = compute_residual_metrics()
    if os.environ.get("UPDATE_GOLDEN") == "1":
        written = {**actual, "scikit-learn": _sklearn_version()}
        GOLDEN_FILE.write_text(json.dumps(written, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"golden values written to {GOLDEN_FILE}")
    expected = json.loads(GOLDEN_FILE.read_text())
    release = expected.pop("scikit-learn")
    assert actual.keys() == expected.keys()
    for model, values in expected.items():
        assert actual[model]["forecasts"] == values["forecasts"], f"{model}: number of forecasts changed"
    for key, value in expected["OLS"].items():
        assert actual["OLS"][key] == pytest.approx(value, rel=GOLDEN_RTOL), f"OLS {key}"
    if _sklearn_version() != release:
        pytest.skip(
            f"the residual model's golden metrics are scikit-learn {release}'s and this is {_sklearn_version()}: HGB's "
            "numbers move between releases (the counts and the base's metrics were compared)"
        )
    for key, value in expected["Residual"].items():
        assert actual["Residual"][key] == pytest.approx(value, rel=GOLDEN_RTOL), f"Residual {key}"


def test_the_residual_golden_is_of_the_lock_s_scikit_learn():
    """The release the golden file was written with is the one `uv.lock` pins, so the locked CI cells compare it (the others
    skip its HGB numbers, visibly). Moving the lock's scikit-learn means writing the file again."""
    import re

    lock = (Path(__file__).parent.parent.parent / "uv.lock").read_text()
    pinned = re.search(r'\[\[package\]\]\nname = "scikit-learn"\nversion = "([^"]+)"', lock)
    assert pinned is not None, "no scikit-learn in uv.lock"
    assert json.loads(GOLDEN_FILE.read_text())["scikit-learn"] == pinned.group(1)
