"""`verify`'s rerun and co-batch checks (step 6c) and the shared adapter suite (`tests/models/adapter_suite.py`).

The co-batch check is the leakage probe run once more over runs widened to the other items of the call that predicts the issue
day's forecasts, as the model names them: the forecast must not depend on anything unknown at its issue time, whatever else is
in its call. The plain probe misses a call made of the run's own days, which holds the issue day alone in its runs. Known
values of other items (a known-ahead covariate, a block's earlier days) are no leak. The rerun check runs the issue day twice
first, from other seeds of the global generators: a model that isn't deterministic is `inconclusive`, never `ok`, and a move
under the probe is a leak on such a day only when it is far beyond the rerun's difference (`NOISE_MARGIN`).

No family of step 6 exists yet, so the models here are fakes on OLS: `FakeInContext` predicts the issue rows of its refit block
in one call, as an in-context model will (6d, 6e), `FakeBatch` the days of a run in batches, as a zero-shot model may (6i).
"""

import concurrent.futures
import datetime
import random
import sys
from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from epftoolbox2 import cli
from epftoolbox2._extras import EXTRAS
from epftoolbox2.information_set import Availability, ControlResult, InformationSet, ProbeResult, co_batch_probe
from epftoolbox2.models import LassoCVModel, NaiveModel, OLSModel, RidgeModel
from epftoolbox2.models._calendar import refit_anchor, refit_block_end
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.models._cards import preflight
from epftoolbox2.models.base import BaseModel
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.pipelines import model_pipeline
from epftoolbox2.pipelines.model_pipeline import COMPONENT_REGISTRY
from epftoolbox2.verification import (
    NOISE_MARGIN,
    CoBatchResult,
    IssueDayCheck,
    RerunResult,
    VerifyReport,
    checked_by_default,
    rerun_check,
)
from tests.models import adapter_suite as suite
from tests.models.carded_fake import run_data_card
from tests.synthetic import PREDICTORS, make_market_data

RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "load_actual": Availability.realised("1h"),
    "observed": Availability.realised("1h"),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
}
INFO = InformationSet(rules=RULES)
CLEAN = PREDICTORS + ["price_d-1"]
WINDOW = 40
# A "7D" block runs Monday to Sunday (the refit calendar counts from Monday 1970-01-05).
FIRST, MIDDLE, LAST = "2023-04-10", "2023-04-12", "2023-04-16"
SPRING, AUTUMN = "2023-03-26", "2023-10-29"  # Europe/Warsaw
NEW_COLUMNS = ["rerun", "co_batch"]
PRICES = [
    "price",
    "price_d-1",
]  # what the planted leaks read on the other items' rows: unknown there from D+1 or D+2 on
KNOWN_AHEAD = ["load_d+{horizon}", "is_weekend_d+{horizon}"]  # known at D's issue on every row of the batch


class InProcessPool:
    """Stands in for the worker pool, as in tests/pipelines/test_verify.py: every check here runs many one-day runs."""

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


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def real_pool(monkeypatch):
    """The real fork pool, one process: what a model drawing from the global generators inherits from."""
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", concurrent.futures.ProcessPoolExecutor)


@pytest.fixture(scope="module")
def data():
    frame = make_market_data(days=120, noise=5.0, seed=7, horizon=3)
    return frame.assign(observed=frame["load"])


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def block_days(day, refit):
    anchor, end = refit_anchor(day, refit), refit_block_end(day, refit)
    return [anchor + datetime.timedelta(days=i) for i in range((end - anchor).days + 1)]


def call_mean(frame, columns, record, days):
    """The mean of `columns` (templates with `{horizon}`) on the issue rows of the other items of a record's call: `days` other
    than its own, at its hour. 0 when the frame has none."""
    local = frame.index.tz_localize(None)
    others = [pd.Timestamp(str(day)) for day in days if str(day) != record["run_date"]]
    rows = np.asarray(local.normalize().isin(others) & (local.hour == record["hour"]))
    names = [name.format(horizon=record["horizon"]) for name in columns]
    values = frame.loc[rows, names].to_numpy(dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else 0.0


class FakeInContext(OLSModel):
    """Stands in for an in-context model (6e), which predicts every issue row of its refit block that the frame has in one call
    (6d), whatever days the run has: an OLS fit on the unit's rows, and with `leak` that times the mean of `reads` on the other
    rows of the call. A test double: a family never overrides `run`."""

    _predicts_together = True

    def __init__(self, predictors, training_window=WINDOW, *, leak=0.0, reads=None, **kwargs):
        super().__init__(predictors, training_window, **kwargs)
        self.leak = leak
        self.reads = list(PRICES if reads is None else reads)

    def _co_batch_items(self, run_day):
        return block_days(run_day, self.refit)

    def run(self, data, *args, **kwargs):
        ref = super().run(data, *args, **kwargs)
        if self.leak:
            for record in ref._results:
                days = block_days(record["run_date"], self.refit)
                record["prediction"] += self.leak * call_mean(data, self.reads, record, days)
        return ref


class FakeBatch(OLSModel):
    """Stands in for a zero-shot model that predicts the issue days of a run in batches of `batch` days: the call of a day holds
    the next `batch - 1` days, if the run has them. Its fit is OLS's, each day on its own; `leak` and `reads` as in
    `FakeInContext`, over the run's other days."""

    _predicts_together = True

    def __init__(self, predictors, training_window=WINDOW, *, leak=0.0, reads=None, batch=4, **kwargs):
        super().__init__(predictors, training_window, **kwargs)
        self.leak = leak
        self.reads = list(PRICES if reads is None else reads)
        self.batch = batch

    def _co_batch_items(self, run_day):
        first = datetime.date.fromisoformat(run_day)
        return [first + datetime.timedelta(days=i) for i in range(1, self.batch)]

    def run(self, data, *args, **kwargs):
        ref = super().run(data, *args, **kwargs)
        if self.leak:
            days = sorted({record["run_date"] for record in ref._results})
            for record in ref._results:
                record["prediction"] += self.leak * call_mean(data, self.reads, record, days)
        return ref


class FutureCovariate(FakeBatch):
    """A batch model reading the raw `load` on its target day's row, as a sequence model reads a future-known covariate: known
    at the issue time, so no leak (code review r5)."""

    def run(self, data, *args, **kwargs):
        ref = super().run(data, *args, **kwargs)
        local = data.index.tz_localize(None)
        for record in ref._results:
            target = pd.Timestamp(record["run_date"]) + pd.Timedelta(days=record["horizon"])
            rows = np.asarray((local.normalize() == target) & (local.hour == record["hour"]))
            record["prediction"] += 1e-3 * float(data.loc[rows, "load"].iloc[0])
        return ref


class AutoInContext(FakeInContext):
    """Windows on both sides of 84 days with `refit="auto"`: a "1D" member that trains on the days of the "7D" block before the
    issue day, which the hook names (code review r1)."""

    def _co_batch_items(self, run_day):
        return block_days(run_day, self._refit_block)


class Unseeded(OLSModel):
    """OLS plus unseeded noise of `scale` to twice `scale`, either sign (in standardised target units), on each forecast, as
    a model without a fixed `random_state`, or a GPU kernel whose sums run in another order, gives. Kept away from zero: a
    normal draw of under about 4e-4 at `scale=1e-12` rounds away in target units, so a run would now and then change 23
    forecasts of 24 (seen once on S02)."""

    def __init__(self, predictors, training_window=WINDOW, *, scale=1e-6, **kwargs):
        super().__init__(predictors, training_window, **kwargs)
        self.scale = scale

    @property
    def _model_kwargs(self):
        return {"scale": self.scale}

    def _fit_predict(self, train_x, train_y, test_x):
        prediction, coefs = super()._fit_predict(train_x, train_y, test_x)
        rng = np.random.default_rng()
        return prediction + self.scale * (1.0 + rng.random()) * rng.choice((-1.0, 1.0)), coefs


class GlobalRng(OLSModel):
    """OLS plus 1e-6 drawn from a process-global generator: numpy's legacy one (as scikit-learn's `random_state=None`) or
    `random`'s. A forked pool inherits its parent's state, so every run started from the same state draws the same number."""

    _card = run_data_card("fake")  # turns the rerun on by default

    def __init__(self, predictors, training_window=WINDOW, *, source="numpy", **kwargs):
        super().__init__(predictors, training_window, **kwargs)
        self.source = source

    @property
    def _model_kwargs(self):
        return {"source": self.source}

    def _fit_predict(self, train_x, train_y, test_x):
        prediction, coefs = super()._fit_predict(train_x, train_y, test_x)
        draw = np.random.standard_normal() if self.source == "numpy" else random.gauss(0.0, 1.0)
        return prediction + 1e-6 * draw, coefs


class OnDevice(OLSModel):
    """A carded model with a `device`: deterministic on "cpu", unseeded noise anywhere else (a GPU stand-in; nothing here
    touches a GPU)."""

    _card = run_data_card("fake-gpu")

    def __init__(self, predictors, training_window=WINDOW, *, device="cuda", **kwargs):
        super().__init__(predictors, training_window, **kwargs)
        self.device = device

    @property
    def _model_kwargs(self):
        return {"device": self.device}

    def _fit_predict(self, train_x, train_y, test_x):
        prediction, coefs = super()._fit_predict(train_x, train_y, test_x)
        noise = 0.0 if self.device == "cpu" else 1e-6 * np.random.default_rng().standard_normal()
        return prediction + noise, coefs


class TogetherUnnamed(OLSModel):
    """Says it predicts several items in one call but doesn't name them."""

    _predicts_together = True


class Undeclared(FakeBatch):
    """Names the other items of its calls but doesn't say it predicts together."""

    _predicts_together = False


def in_context(name="InContext", leak=0.0, refit="7D", **kwargs):
    return FakeInContext(CLEAN, leak=leak, refit=refit, name=name, **kwargs)


def pipeline(*models, **settings):
    out = ModelPipeline(information_set=settings.pop("information_set", INFO), **settings)
    for model in models:
        out.add_model(model)
    return out


def row(report, model=None):
    table = report.table()
    return table[table["model"] == (model or report.models[0])].iloc[0]


def on(data, day, *models, **kwargs):
    """`verify` on one issue day."""
    return pipeline(*models).verify(data, day, day, days=1, **kwargs)


# ---------------------------------------------------------------------------
# Existing models: the table as before, with two empty columns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(OLSModel(CLEAN, training_window=WINDOW, name="M"), id="ols"),
        pytest.param(LassoCVModel(CLEAN, training_window=30, cv=3, name="M"), id="lassocv"),
        pytest.param(RidgeModel(CLEAN, training_window=WINDOW, refit="7D", name="M"), id="ridge-refit"),
        pytest.param(OLSModel(CLEAN, training_window=WINDOW, newest_target="published", name="M"), id="published"),
        pytest.param(NaiveModel(["price_d-1"], name="M"), id="naive"),
    ],
)
def test_existing_models_run_neither_check_and_show_empty_columns(data, model):
    assert not checked_by_default(model)
    report = pipeline(model).verify(data, "2023-04-03", "2023-04-28", days=2)
    table = report.table()

    assert list(table.columns[-3:]) == [*NEW_COLUMNS, "status"]
    assert table[NEW_COLUMNS].isna().all().all()
    assert str(table["rerun"].dtype) == str(table["co_batch"].dtype) == "Int64"
    assert all(check.rerun is None and check.co_batch is None and check.device is None for check in report.checks)
    assert not [line for line in report.problems() if "nondeterministic" in line or "call" in line]
    forced = pipeline(model).verify(data, "2023-04-03", "2023-04-28", days=2, rerun=False, co_batch=False)
    pd.testing.assert_frame_equal(forced.table(), table)


def test_forcing_both_checks_on_an_existing_model(data):
    """Ridge is deterministic and predicts each forecast on its own (step 3 decision 16): the rerun finds no difference and the
    co-batch check has no other item to put in the call, so it runs nothing and counts 0."""
    report = on(
        data, MIDDLE, RidgeModel(CLEAN, training_window=WINDOW, refit="7D", name="Ridge"), rerun=True, co_batch=True
    )

    assert (row(report)["rerun"], row(report)["co_batch"], row(report)["status"]) == (0, 0, "ok")
    (check,) = report.checks
    assert check.rerun.forecasts == 24 and check.rerun.ok
    assert check.co_batch.items == [] and check.co_batch.probe is None


def test_the_defaults_follow_predicts_together_the_hook_and_the_card():
    assert checked_by_default(in_context())
    assert checked_by_default(Undeclared(CLEAN))  # the hook alone
    assert checked_by_default(OnDevice(CLEAN))
    assert not checked_by_default(Unseeded(CLEAN))
    assert not checked_by_default(OLSModel(CLEAN))


# ---------------------------------------------------------------------------
# Co-batch: the probe over the whole call
# ---------------------------------------------------------------------------


def with_weather(data, rule):
    """`data` with a weather forecast `wx_d+h` (h = 1 to 3) on every row under `rule`, and the information set with it."""
    rng = np.random.default_rng(1)
    columns = {f"wx_d+{h}": rng.normal(10.0, 3.0, len(data)) for h in (1, 2, 3)}
    return data.assign(**columns), InformationSet(rules={**RULES, "wx": rule})


WEATHER = {
    "cutoff": Availability.weather_cutoff("00", "6h"),
    "as_of": Availability.as_of("06:00"),
}


@pytest.mark.parametrize(
    "reads, weather",
    [
        pytest.param(["price"], None, id="their prices"),
        pytest.param(["price_d-1"], None, id="their price lags"),
        pytest.param(["wx_d+{horizon}"], "cutoff", id="their weather_cutoff forecasts"),
        pytest.param(["wx_d+{horizon}"], "as_of", id="their as_of forecasts"),
    ],
)
def test_a_batch_reading_what_its_other_items_can_t_know_is_a_leak_the_probe_alone_misses(data, reads, weather):
    """The call of D holds D+1 to D+3 when the run has them. Their prices (the issue day's own future targets), the lag of
    D+2 and D+3, and their weather forecasts, each from its own day's run (`weather_cutoff`, `as_of`), are unknown at D's issue
    (the same forecast read on D's row as `wx_d+1_d+1` is too). The plain probe runs D alone, a batch of one, and passes
    (methodology review `batch_label.py`, re-review `weather_rows.py`); the co-batch probe runs D to D+3 and the forecasts
    move by about 1e-3 x 1e6."""
    info = INFO
    if weather is not None:
        data, info = with_weather(data, WEATHER[weather])
    report = pipeline(FakeBatch(CLEAN, leak=1e-3, reads=reads, name="Batch"), information_set=info).verify(
        data, MIDDLE, MIDDLE, days=1
    )

    assert row(report)["changed"] == 0
    assert row(report)["co_batch"] == 24
    assert row(report)["status"] == "leak" and not report.ok
    (check,) = report.checks
    assert check.co_batch.items == ["2023-04-13", "2023-04-14", "2023-04-15"]
    assert check.co_batch.probe.max_abs_change > 100
    (line,) = report.problems()
    assert line.startswith(
        "Batch: leak on 2023-04-12: 24 of 24 forecasts depend on values not known at issue time (the "
    )
    assert "run over the other items of their call too, 3 issue day(s)" in line

    alone = pipeline(FakeBatch(CLEAN, leak=1e-3, reads=reads, name="Batch"), information_set=info)
    assert alone.verify(data, MIDDLE, MIDDLE, days=1, co_batch=False).ok
    if weather is not None:  # the same model without the leak, under the same rule
        assert pipeline(FakeBatch(CLEAN, name="Batch"), information_set=info).verify(data, MIDDLE, MIDDLE, days=1).ok


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            FakeBatch(CLEAN, leak=1e-3, reads=KNOWN_AHEAD, name="Known"), id="other items' known-ahead inputs"
        ),
        pytest.param(FutureCovariate(CLEAN, name="Known"), id="its target day's covariate"),
    ],
)
def test_a_batch_reading_only_what_is_known_ahead_passes(data, model):
    """A known-ahead covariate on another item's row is known at the issue time: no leak, however much the forecast reads it
    (code review `r5_future_covariate.py`, a move of 1013 under row noise)."""
    report = on(data, MIDDLE, model)

    assert (row(report)["co_batch"], row(report)["status"]) == (0, "ok"), report.problems()
    (check,) = report.checks
    assert check.co_batch.probe.forecasts == 24
    assert check.co_batch.control.moved  # the control over the widened runs reaches the forecasts


def test_an_in_context_model_reading_its_block_s_later_rows_is_a_leak(data):
    """On the block's first and middle days the other rows of the call include later days, whose prices aren't known at the
    issue time. On its last day they are the six days before it, all known: by the leak's definition no leak."""
    for day in (FIRST, MIDDLE):
        report = on(data, day, in_context("Leaky", leak=1e-3))
        assert row(report)["co_batch"] == 24, day
        assert row(report)["status"] == "leak", day
        assert len(report.checks[0].co_batch.items) == 6
    last = on(data, LAST, in_context("Leaky", leak=1e-3))
    assert (row(last)["changed"], row(last)["co_batch"], row(last)["status"]) == (0, 0, "ok")
    # A later day's weather forecast comes from that day's run (re-review `weather_incontext.py`).
    frame, info = with_weather(data, WEATHER["cutoff"])
    for day in (FIRST, MIDDLE):
        model = in_context("Weather", leak=1e-3, reads=["wx_d+{horizon}"])
        report = pipeline(model, information_set=info).verify(frame, day, day, days=1)
        assert (row(report)["co_batch"], row(report)["status"]) == (24, "leak"), day


def test_a_clean_in_context_model_passes_on_every_day_of_the_block(data):
    for day in (FIRST, MIDDLE, LAST):
        report = on(data, day, in_context())
        assert (row(report)["rerun"], row(report)["co_batch"], row(report)["status"]) == (0, 0, "ok"), day
        (check,) = report.checks
        assert len(check.co_batch.items) == 6 and check.co_batch.probe.forecasts == 24
        assert check.co_batch.probe.perturbed["price"] > 0


def test_a_member_that_trains_on_the_block_s_earlier_days_passes():
    """`refit="auto"` over windows of 30 and 100 days: the 30-day member refits daily ("1D") and trains on the days of the
    "7D" block before the issue day, which the hook names. Known at the issue time, so they stay (code review
    `r1_auto_members.py`: whole-row noise called this a leak on 07-13 and 07-16)."""
    frame = make_market_data(days=220, noise=5.0, seed=7, horizon=3)
    for day in ("2023-07-10", "2023-07-13", "2023-07-16"):
        model = AutoInContext(CLEAN, training_window=[30, 100], refit="auto", name="Auto")
        assert model._refit_rules == ("1D", "7D") and model._refit_block == "7D"
        report = on(frame, day, model)
        assert row(report)["status"] == "ok", (day, report.problems())
        assert row(report)["co_batch"] == 0


@pytest.mark.parametrize("predictors", [CLEAN, CLEAN + ["price"]], ids=["target only trains", "target is an input too"])
def test_what_a_model_trains_on_from_the_block_stays_when_known(data, predictors):
    """With `newest_target="published"` a unit trains on its anchor's price, which is on a row of the block and known at the
    issue time; a target that is also an input (`price` lag 0) trains through its copy. A clean model passes either way."""
    model = FakeInContext(predictors, refit="7D", newest_target="published", name="Published")
    report = on(data, MIDDLE, model)

    assert row(report)["status"] == "ok", report.problems()
    assert report.checks[0].co_batch.probe.forecasts == 24


def test_co_batch_probe_on_its_own(data):
    def leaky():
        return FakeBatch(CLEAN, leak=1e-3, reads=["price"])

    found = co_batch_probe(leaky, data, MIDDLE, INFO, check_target_boundary=True, control=True)
    assert not found.ok and len(found.probe.changed) == 24 and found.control.moved
    assert found.items == ["2023-04-13", "2023-04-14", "2023-04-15"]

    alone = co_batch_probe(leaky, data, MIDDLE, INFO, items=[MIDDLE, "2031-01-01"])
    assert (alone.items, alone.probe, alone.ok) == ([], None, True)  # the day itself and a day the frame lacks
    assert co_batch_probe(lambda: OLSModel(CLEAN), data, MIDDLE, INFO).items is None  # no hook: nothing ran


def test_a_model_that_predicts_together_but_names_nothing_is_inconclusive(data):
    report = on(data, MIDDLE, TogetherUnnamed(CLEAN, training_window=WINDOW, name="Unnamed"))

    assert row(report)["status"] == "inconclusive"
    assert pd.isna(row(report)["co_batch"]) and row(report)["rerun"] == 0
    assert any("doesn't name them (_co_batch_items)" in line for line in report.problems())


def test_a_model_with_the_hook_but_no_declaration_gets_the_check(data):
    """The hook is enough to turn the co-batch check on (and the rerun)."""
    report = on(data, MIDDLE, Undeclared(CLEAN, leak=1e-3, reads=["price"], name="Undeclared"))
    assert row(report)["co_batch"] == 24 and row(report)["status"] == "leak"


# ---------------------------------------------------------------------------
# Rerun
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e-6, 1e-12], ids=["above the probe's atol", "below it"])
def test_an_unseeded_model_is_inconclusive_never_ok(data, scale):
    """Above 1e-9 the probe alone would call the noise a leak; below it, a pass. The rerun, at atol 0, says neither."""
    report = on(data, MIDDLE, Unseeded(CLEAN, scale=scale, name="Unseeded"), rerun=True)

    assert row(report)["rerun"] == 24
    assert row(report)["status"] == "inconclusive" and not report.ok
    assert (row(report)["changed"] > 0) == (scale > 1e-9)
    problems = report.problems()
    assert problems[0].startswith("Unseeded: inconclusive: nondeterministic: two unperturbed runs differ on 2023-04-12")
    assert "device='cpu'" in problems[0]
    assert not [line for line in problems if ": leak" in line]

    without = on(data, MIDDLE, Unseeded(CLEAN, scale=scale, name="Unseeded"), rerun=False)
    assert pd.isna(row(without)["rerun"])
    assert row(without)["status"] == ("leak" if scale > 1e-9 else "ok")  # what the rerun is there for


@pytest.mark.parametrize("source", ["numpy", "random"])
def test_a_model_drawing_from_the_global_generators_is_inconclusive_under_the_real_pool(data, real_pool, source):
    """Both runs fork from the parent: without other seeds for each, both would draw the same numbers and pass (methodology
    review `rng_fork.py`)."""
    report = on(data, MIDDLE, GlobalRng(CLEAN, source=source, name="Global"), positive_control=False)

    assert row(report)["rerun"] == 24
    assert row(report)["status"] == "inconclusive"


def test_the_rerun_leaves_the_global_generators_as_it_found_them(data):
    """Its runs draw from other seeds, which are put back: the model's draws in the process (in-process pool here) don't
    advance the caller's state."""
    np.random.seed(11)
    random.seed(11)
    before = (np.random.get_state(), random.getstate())
    noisy = rerun_check(lambda: GlobalRng(CLEAN), data, MIDDLE, INFO)
    after = (np.random.get_state(), random.getstate())
    assert not noisy.ok
    assert all((a == b).all() if isinstance(a, np.ndarray) else a == b for a, b in zip(before[0], after[0]))
    assert before[1] == after[1]


def test_rerun_check_on_its_own(data):
    noisy = rerun_check(lambda: Unseeded(CLEAN), data, MIDDLE, INFO, horizon=2)
    assert (noisy.forecasts, len(noisy.differ), noisy.missing, noisy.bounded) == (48, 48, [], True)
    assert 0 < noisy.noise == noisy.max_abs_difference < 1e-3
    same = rerun_check(lambda: OLSModel(CLEAN, training_window=WINDOW), data, MIDDLE, INFO, horizon=2)
    assert same.ok and same.noise == 0.0


def test_a_late_read_stays_a_leak_in_a_nondeterministic_model(data):
    """A predictor function that reads a value known only after the issue time is a leak whatever the noise: no comparison of
    forecasts is needed to see it."""

    def reads_late(ctx):
        return CLEAN if ctx.value("observed") > 0 else PREDICTORS

    report = pipeline(Unseeded(reads_late, name="Late"), availability="warn").verify(
        data, MIDDLE, MIDDLE, days=1, rerun=True, positive_control=False
    )
    assert row(report)["status"] == "leak"
    assert any(line.startswith("Late: leak on 2023-04-12") for line in report.problems())


def test_a_leak_far_beyond_the_rerun_noise_stays_a_leak(data):
    """Noise of about 1e-11 doesn't hide a look-ahead that moves forecasts by tens of thousands (code review
    `r2_noise_masks_leak.py`), nor a co-batch leak of about 1e3 (methodology review `downgrade.py`)."""
    report = on(data, MIDDLE, Unseeded(CLEAN + ["observed"], scale=1e-12, name="Leaky"), rerun=True)
    assert row(report)["rerun"] == 24 and row(report)["status"] == "leak"
    assert any(line.startswith("Leaky: leak on 2023-04-12") for line in report.problems())
    assert report.problems()[0].startswith("Leaky: inconclusive: nondeterministic")

    class NoisyLeaky(FakeBatch):
        def _fit_predict(self, train_x, train_y, test_x):
            prediction, coefs = super()._fit_predict(train_x, train_y, test_x)
            return prediction + 1e-12 * np.random.default_rng().standard_normal(), coefs

    batch = on(data, MIDDLE, NoisyLeaky(CLEAN, leak=1e-3, name="NoisyLeaky"))
    assert row(batch)["rerun"] == 24 and row(batch)["co_batch"] == 24 and row(batch)["status"] == "leak"


def _probe(day, change):
    return ProbeResult(day, 24, [(0, 1)] if change else [], change, {"price": 1}, [])


def _check(day, change, noise=None, bounded=True, batch_change=None):
    rerun = None if noise is None else RerunResult(day, 24, [(0, 1)] if noise else [], noise, [], bounded)
    batch = None if batch_change is None else CoBatchResult(day, ["2023-04-13"], _probe(day, batch_change))
    return IssueDayCheck("M", day, 1, _probe(day, change), rerun=rerun, co_batch=batch)


def test_noise_is_judged_per_day_with_a_margin():
    """A move on a day whose runs differ by `noise` is a leak only above NOISE_MARGIN x noise; a day whose runs agree judges
    its moves as the probe does, whatever another day's noise."""
    assert NOISE_MARGIN == 100.0
    within = VerifyReport([MIDDLE], [_check(MIDDLE, 5e-5, noise=1e-6)])  # 50 times
    assert within.status("M") == "inconclusive"
    assert "within 100 times its rerun difference (1e-06)" in within.problems()[1]
    assert VerifyReport([MIDDLE], [_check(MIDDLE, 5e-5, noise=1e-7)]).status("M") == "leak"  # 500 times

    two_days = VerifyReport([MIDDLE, LAST], [_check(MIDDLE, 5e-5, noise=1e-6), _check(LAST, 5e-5, noise=0.0)])
    assert two_days.status("M") == "leak"
    assert [line.split(":")[1].strip() for line in two_days.problems()] == [
        "inconclusive",
        "inconclusive",
        "leak on 2023-04-16",
    ]

    unbounded = VerifyReport([MIDDLE], [_check(MIDDLE, 1e3, noise=1e-9, bounded=False)])  # a NaN or missing forecast
    assert unbounded.status("M") == "inconclusive"
    batch = VerifyReport([MIDDLE], [_check(MIDDLE, 0.0, noise=1e-6, batch_change=5e-5)])
    assert batch.status("M") == "inconclusive" and batch.table().iloc[0]["co_batch"] == 1
    assert VerifyReport([MIDDLE], [_check(MIDDLE, 0.0, noise=1e-6, batch_change=1.0)]).status("M") == "leak"


def test_a_widened_control_that_moves_nothing_is_a_blind_spot():
    """The co-batch probe is evidence only if its perturbation reaches the forecasts in the widened runs too."""
    keys = [(period, 1) for period in range(24)]
    moved = ControlResult(MIDDLE, 24, keys, 1.0, {"price_d-1": 24}, [])
    still = ControlResult(MIDDLE, 24, [], 0.0, {"price_d-1": 24}, [], keys)
    batch = CoBatchResult(MIDDLE, ["2023-04-13"], _probe(MIDDLE, 0.0), still)
    report = VerifyReport([MIDDLE], [IssueDayCheck("M", MIDDLE, 1, _probe(MIDDLE, 0.0), moved, co_batch=batch)])
    assert report.status("M") == "inconclusive"
    assert (
        "positive control over the runs widened to the other items of the call moved no forecast"
        in report.problems()[0]
    )
    reached = CoBatchResult(MIDDLE, ["2023-04-13"], _probe(MIDDLE, 0.0), moved)
    assert VerifyReport([MIDDLE], [IssueDayCheck("M", MIDDLE, 1, _probe(MIDDLE, 0.0), moved, co_batch=reached)]).ok


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------


def test_device_verifies_a_gpu_model_on_the_cpu_and_leaves_the_model_alone(data):
    model = OnDevice(CLEAN, name="GPU")
    noisy = on(data, MIDDLE, model)
    assert row(noisy)["status"] == "inconclusive" and noisy.devices == {}  # the card turns the rerun on

    report = on(data, MIDDLE, model, device="cpu")
    assert (row(report)["rerun"], row(report)["status"]) == (0, "ok")
    assert report.devices == {"GPU": "cpu"}
    assert model.device == "cuda"


def test_device_on_models_without_one_changes_nothing_and_says_so(data):
    with pytest.warns(UserWarning, match="no model verified has a `device`"):
        report = on(data, MIDDLE, OLSModel(CLEAN, training_window=WINDOW, name="OLS"), device="cpu")
    assert report.ok and report.devices == {}


def test_epf_verify_passes_device_rerun_and_co_batch_on(monkeypatch, capsys, tmp_path, data):
    from epftoolbox2._frame_csv import save_frame_csv

    data_path, yaml_path = tmp_path / "data.csv", tmp_path / "mp.yaml"
    save_frame_csv(data, data_path)
    pipeline(OLSModel(CLEAN, training_window=WINDOW, name="OLS")).save(yaml_path)
    seen = []
    real = ModelPipeline.verify

    def verify(self, *args, **kwargs):
        seen.append({key: kwargs[key] for key in ("device", "rerun", "co_batch")})
        return real(self, *args, **kwargs)

    monkeypatch.setattr(ModelPipeline, "verify", verify)
    argv = ["epf", "verify", str(yaml_path), "--data", str(data_path), "--test-start", MIDDLE, "--test-end", MIDDLE]
    for extra, expected in (
        (["--device", "cpu", "--rerun", "--no-co-batch"], {"device": "cpu", "rerun": True, "co_batch": False}),
        (["--no-rerun", "--co-batch"], {"device": None, "rerun": False, "co_batch": True}),
        ([], {"device": None, "rerun": None, "co_batch": None}),
    ):
        monkeypatch.setattr(sys, "argv", [*argv, "--days", "1", *extra])
        cli.main()  # exits only on a failure
        out = capsys.readouterr().out
        assert seen[-1] == expected
        assert "rerun" in out and "co_batch" in out and "Verification passed" in out


# ---------------------------------------------------------------------------
# The adapter suite, on the fake in-context family
# ---------------------------------------------------------------------------


def make(name, **params):
    return FakeInContext(CLEAN, **{"refit": "7D", "name": name, **params})


TEST = ("2023-04-05", "2023-04-18")  # two blocks and parts of two more


@pytest.fixture
def registered(monkeypatch):
    for cls in (FakeInContext, FakeBatch):
        monkeypatch.setitem(COMPONENT_REGISTRY["models"], cls.__name__, __name__)


def test_suite_yaml_round_trip(data, tmp_path, registered):
    suite.check_yaml_round_trip(make, data, tmp_path, TEST, horizon=2, information_set=INFO)


def test_suite_verify_passes_a_clean_adapter_and_fails_a_leaky_or_noisy_one(data):
    days = [FIRST, MIDDLE, LAST, "2023-03-25", "2023-03-26"]  # a block, and the days around the spring switch
    suite.check_verify(make, data, INFO, days, horizon=2)
    suite.check_verify(make, data, INFO, [MIDDLE], items=6)
    suite.check_verify(
        lambda name, **p: FakeBatch(CLEAN, leak=1e-3, reads=KNOWN_AHEAD, name=name), data, INFO, [MIDDLE]
    )

    with pytest.raises(AssertionError, match="run over the other items of their call too"):
        suite.check_verify(lambda name, **p: FakeBatch(CLEAN, leak=1e-3, name=name), data, INFO, [MIDDLE])
    with pytest.raises(AssertionError, match="nondeterministic"):
        suite.check_verify(lambda name, **p: Unseeded(CLEAN, name=name), data, INFO, [MIDDLE])
    with pytest.raises(AssertionError, match="doesn.t name them"):
        suite.check_verify(
            lambda name, **p: TogetherUnnamed(CLEAN, training_window=WINDOW, name=name), data, INFO, [LAST]
        )


def test_suite_verify_fails_a_model_drawing_from_the_global_generators(data, real_pool):
    """Methodology review `suite_rng.py`: under the real pool the suite passed it."""
    with pytest.raises(AssertionError, match="nondeterministic"):
        suite.check_verify(lambda name, **p: GlobalRng(CLEAN, name=name), data, INFO, [MIDDLE])


class Sticky(FakeInContext):
    """Carries state from one fit to the next in its process, as a global flag one model leaves set for the next would."""

    fits = 0

    def _fit_predict(self, train_x, train_y, test_x):
        return self._predict_fitted(self._fit_model(train_x, train_y), test_x)

    def _fit_model(self, train_x, train_y):
        type(self).fits += 1
        return super()._fit_model(train_x, train_y)

    def _predict_fitted(self, fitted, test_x):
        prediction, coefs = super()._predict_fitted(fitted, test_x)
        return prediction + 1e-3 * type(self).fits, coefs


class Ordinal(FakeInContext):
    """A forecast that depends on its place among the forecasts of its fit, as one of an unpadded batch may."""

    def _fit_predict(self, train_x, train_y, test_x):
        return self._predict_fitted(self._fit_model(train_x, train_y), test_x)

    def _fit_model(self, train_x, train_y):
        return super()._fit_model(train_x, train_y), [0]

    def _predict_fitted(self, fitted, test_x):
        model, place = fitted
        prediction, coefs = super()._predict_fitted(model, test_x)
        place[0] += 1
        return prediction + 1e-3 * place[0], coefs


def test_suite_loop_and_shards(data, tmp_path, monkeypatch):
    variants = ({}, {"training_window": 30}, {"refit": "1MS"})
    suite.check_loop_and_shards(make, data, tmp_path, TEST, horizon=2, variants=variants)

    monkeypatch.setattr(Sticky, "fits", 0)
    with pytest.raises(AssertionError, match="in a loop differs"):
        suite.check_loop_and_shards(
            lambda name, **p: Sticky(CLEAN, refit="7D", name=name, **p), data, tmp_path / "sticky", TEST
        )


def test_suite_resume(data, tmp_path):
    suite.check_resume(make, data, tmp_path, TEST, horizon=2)
    # A resumed block forecasts fewer days from its fit than a fresh one.
    with pytest.raises(AssertionError):
        suite.check_resume(lambda name, **p: Ordinal(CLEAN, refit="7D", name=name), data, tmp_path / "ordinal", TEST)


def test_suite_engine(data):
    suite.check_engine(make, data, TEST)
    with pytest.raises(AssertionError, match="the engine would take"):
        suite.check_engine(lambda name, **p: OLSModel(CLEAN, training_window=WINDOW, name=name), data, TEST)


def test_suite_dst_counts():
    spring = make_market_data(start="2023-02-01", days=70, noise=5.0, seed=4, horizon=1)
    autumn = make_market_data(start="2023-09-01", days=70, noise=5.0, seed=4, horizon=1)
    frame = pd.concat([spring, autumn])
    suite.check_dst_counts(make, frame, SPRING, AUTUMN)


def test_suite_quarter_hours():
    frame = make_market_data(start="2023-01-01", days=60, freq="15min", noise=5.0, seed=4, horizon=1)
    suite.check_quarter_hours(make, frame, ("2023-02-20", "2023-02-21"))


def test_suite_preflight_and_provenance_with_cards(data, tmp_path, monkeypatch):
    """No card: nothing to miss and nothing to show. A card with an extra: the pre-flight refuses before any run starts when
    the extra is hidden, and the card is in `report.provenance()`; a pre-flight that comes after a model's run started (here
    the pipeline's taken out and each model checked in its own `_prepare_run`) is caught."""
    suite.check_preflight_without_extra(make, monkeypatch, data, TEST)
    suite.check_provenance(make, data, tmp_path / "no-card", TEST)

    monkeypatch.setitem(EXTRAS, "fake-extra", ("json",))  # installed: the check hides it
    monkeypatch.setattr(FakeInContext, "_card", run_data_card("fake", extra="fake-extra"), raising=False)
    suite.check_preflight_without_extra(make, monkeypatch, data, TEST)
    suite.check_provenance(make, data, tmp_path / "card", TEST)

    monkeypatch.setattr(model_pipeline, "preflight", lambda models: None)
    real_prepare = BaseModel._prepare_run

    def late(self, *args, **kwargs):
        out = real_prepare(self, *args, **kwargs)
        preflight([self])
        return out

    monkeypatch.setattr(BaseModel, "_prepare_run", late)
    with pytest.raises(AssertionError, match="started before the pre-flight"):
        suite.check_preflight_without_extra(make, monkeypatch, data, TEST)
