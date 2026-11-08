"""`verify`'s rerun and co-batch checks (step 6c) and the shared adapter suite (`tests/models/adapter_suite.py`).

The co-batch check replaces every other item of the call that predicts the issue day's forecasts by noise: a forecast that
moves depends on what else was predicted with it, which the leakage probe misses where the other items hold only known values
(the days of a refit block before the issue day). The rerun check runs the issue day twice first: a model that isn't
deterministic is `inconclusive`, never `ok`, since a move under the probe can't then be told from its noise.

No family of step 6 exists yet, so the models here are fakes on OLS: `FakeInContext` predicts the issue rows of its refit block
in one call, as an in-context model will (6d, 6e), `FakeBatch` the days of a run in batches, as a zero-shot model may (6i).
"""

import datetime
import sys
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from epftoolbox2 import cli
from epftoolbox2._extras import EXTRAS, MissingExtraError
from epftoolbox2.information_set import Availability, InformationSet, co_batch_probe, perturb_items
from epftoolbox2.models import LassoCVModel, NaiveModel, OLSModel, RidgeModel
from epftoolbox2.models._calendar import refit_anchor, refit_block_end
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.models.base import BaseModel
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.pipelines.model_pipeline import COMPONENT_REGISTRY
from epftoolbox2.verification import checked_by_default, rerun_check
from tests.models import adapter_suite as suite
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


def call_mean(frame, predictors, record, days):
    """The mean of the predictors of the other items of a record's call: the issue rows of `days` other than its own, at its
    hour. 0 when the frame has none."""
    local = frame.index.tz_localize(None)
    others = [pd.Timestamp(str(day)) for day in days if str(day) != record["run_date"]]
    rows = np.asarray(local.normalize().isin(others) & (local.hour == record["hour"]))
    columns = [name.format(horizon=record["horizon"]) for name in predictors]
    values = frame.loc[rows, columns].to_numpy(dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else 0.0


class FakeInContext(OLSModel):
    """Stands in for an in-context model (6e), which predicts every issue row of its refit block that the frame has in one call
    (6d), whatever days the run has: an OLS fit on the unit's rows, and with `leak` that times the mean of the other rows of the
    call, the dependence the co-batch check is for. A test double: a family never overrides `run`."""

    _predicts_together = True

    def __init__(self, predictors, training_window=WINDOW, *, leak=0.0, **kwargs):
        super().__init__(predictors, training_window, **kwargs)
        self.leak = leak

    def _co_batch_items(self, run_day):
        return block_days(run_day, self.refit)

    def run(self, data, *args, **kwargs):
        ref = super().run(data, *args, **kwargs)
        if self.leak:
            for record in ref._results:
                days = block_days(record["run_date"], self.refit)
                record["prediction"] += self.leak * call_mean(data, self.predictors, record, days)
        return ref


class FakeBatch(OLSModel):
    """Stands in for a zero-shot model that predicts the issue days of a run in batches of `batch` days: the call of a day holds
    the next `batch - 1` days, if the run has them. Its fit is OLS's, each day on its own; `leak` as in `FakeInContext`."""

    _predicts_together = True

    def __init__(self, predictors, training_window=WINDOW, *, leak=0.0, batch=4, **kwargs):
        super().__init__(predictors, training_window, **kwargs)
        self.leak = leak
        self.batch = batch

    def _co_batch_items(self, run_day):
        first = datetime.date.fromisoformat(run_day)
        return [first + datetime.timedelta(days=i) for i in range(1, self.batch)]

    def run(self, data, *args, **kwargs):
        ref = super().run(data, *args, **kwargs)
        if self.leak:
            days = sorted({record["run_date"] for record in ref._results})
            for record in ref._results:
                record["prediction"] += self.leak * call_mean(data, self.predictors, record, days)
        return ref


class Unseeded(OLSModel):
    """OLS plus unseeded noise of `scale` (in standardised target units) on each forecast, as a model without a fixed
    `random_state`, or a GPU kernel whose sums run in another order, gives."""

    def __init__(self, predictors, training_window=WINDOW, *, scale=1e-6, **kwargs):
        super().__init__(predictors, training_window, **kwargs)
        self.scale = scale

    @property
    def _model_kwargs(self):
        return {"scale": self.scale}

    def _fit_predict(self, train_x, train_y, test_x):
        prediction, coefs = super()._fit_predict(train_x, train_y, test_x)
        return prediction + self.scale * np.random.default_rng().standard_normal(), coefs


class OnDevice(OLSModel):
    """A carded model with a `device`: deterministic on "cpu", unseeded noise anywhere else (a GPU stand-in; nothing here
    touches a GPU)."""

    _card = SimpleNamespace(family="fake-gpu", extra=None, licence="MIT")

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
    assert all(check.rerun is None and check.co_batch is None for check in report.checks)
    assert not [line for line in report.problems() if "nondeterministic" in line or "call" in line]
    forced = pipeline(model).verify(data, "2023-04-03", "2023-04-28", days=2, rerun=False, co_batch=False)
    pd.testing.assert_frame_equal(forced.table(), table)


def test_forcing_both_checks_on_an_existing_model(data):
    """OLS is deterministic and predicts each forecast on its own (step 3 decision 16): the rerun finds no difference and the
    co-batch check has no other item to replace, so it runs nothing and counts 0."""
    report = on(
        data, MIDDLE, RidgeModel(CLEAN, training_window=WINDOW, refit="7D", name="Ridge"), rerun=True, co_batch=True
    )

    assert (row(report)["rerun"], row(report)["co_batch"], row(report)["status"]) == (0, 0, "ok")
    (check,) = report.checks
    assert check.rerun.forecasts == 24 and check.rerun.ok
    assert check.co_batch.items == [] and check.co_batch.forecasts == 0


def test_the_defaults_follow_predicts_together_and_the_card():
    assert checked_by_default(in_context())
    assert checked_by_default(OnDevice(CLEAN))
    assert not checked_by_default(Unseeded(CLEAN))
    assert not checked_by_default(OLSModel(CLEAN))


# ---------------------------------------------------------------------------
# Co-batch
# ---------------------------------------------------------------------------


def test_a_model_that_reads_the_other_rows_of_its_call_is_a_leak_the_probe_alone_misses(data):
    """On the block's last day the other items are the six days before it, every value of them known at the issue time: the
    probe has nothing to perturb there and passes. Replacing them by noise moves every forecast by about 1e-3 x 1e6."""
    leaky = in_context("Leaky", leak=1e-3)
    report = on(data, LAST, leaky)

    assert row(report)["changed"] == 0
    assert row(report)["control moved"] == 24
    assert row(report)["co_batch"] == 24
    assert row(report)["status"] == "leak" and not report.ok
    (check,) = report.checks
    assert check.co_batch.items == [f"2023-04-{d}" for d in range(10, 16)]
    assert check.co_batch.max_abs_change > 100
    (line,) = report.problems()
    assert "Leaky: leak on 2023-04-16: 24 of 24 forecasts moved when the other items of their call" in line

    assert on(data, LAST, in_context("Leaky", leak=1e-3), co_batch=False).ok  # the probe and the control alone pass it


def test_a_leak_through_the_call_shows_on_every_day_of_the_block(data):
    for day in (FIRST, MIDDLE, LAST):
        report = on(data, day, in_context("Leaky", leak=1e-3))
        assert row(report)["co_batch"] == 24, day
        assert row(report)["status"] == "leak", day


def test_the_same_model_without_the_leak_passes_on_every_day_of_the_block(data):
    """The noise reaches neither the issue day's own rows nor what the unit trains on (rows before its anchor)."""
    for day in (FIRST, MIDDLE, LAST):
        report = on(data, day, in_context())
        assert (row(report)["rerun"], row(report)["co_batch"], row(report)["status"]) == (0, 0, "ok"), day
        (check,) = report.checks
        assert len(check.co_batch.items) == 6 and check.co_batch.forecasts == 24
        assert check.co_batch.perturbed["load_d+1"] == 6 * 24


def test_the_run_covers_a_batch_made_of_the_run_s_own_days(data):
    """A batch of the run's days holds the next three only if the run has them: the probe runs over every named day, though
    a model without refit runs a probe of one day."""
    report = on(data, MIDDLE, FakeBatch(CLEAN, leak=1e-3, name="Batch"))
    assert row(report)["changed"] == 0  # the probe's runs have the issue day alone: nothing else in the call
    assert row(report)["co_batch"] == 24 and row(report)["status"] == "leak"
    assert report.checks[0].co_batch.items == ["2023-04-13", "2023-04-14", "2023-04-15"]

    assert on(data, MIDDLE, FakeBatch(CLEAN, name="Batch")).ok


@pytest.mark.parametrize("predictors", [CLEAN, CLEAN + ["price"]], ids=["target only trains", "target is an input too"])
def test_what_a_model_trains_on_is_never_noised(data, predictors):
    """With `newest_target="published"` a unit trains on its anchor's price, which is on a row of the block: noising the target
    there would move a clean model. A target that is also an input (`price` lag 0) trains through its copy, which stays, while
    the input itself is noised on the other days."""
    model = FakeInContext(predictors, refit="7D", newest_target="published", name="Published")
    report = on(data, MIDDLE, model)

    assert row(report)["status"] == "ok", report.problems()
    batch = report.checks[0].co_batch
    assert "price__as_target" not in batch.perturbed
    assert ("price" in batch.perturbed) == ("price" in predictors)


def test_perturb_items_replaces_every_numeric_value_of_the_named_days_only(data):
    frame = data.assign(flag=data["is_weekend"] > 0)
    frame.iloc[5, frame.columns.get_loc("wind")] = np.nan
    frame.loc[frame.index[frame.index.tz_localize(None).normalize() == pd.Timestamp(MIDDLE)][0], "wind"] = np.nan
    out, counts = perturb_items(frame, [MIDDLE, datetime.date(2023, 4, 13), "2030-01-01"], keep=["price"], seed=3)

    named = np.asarray(frame.index.tz_localize(None).normalize().isin(pd.to_datetime([MIDDLE, "2023-04-13"])))
    numeric = [name for name in frame.columns if name not in ("price", "flag")]
    before, after = frame[numeric].to_numpy(dtype=float), out[numeric].to_numpy(dtype=float)
    changed = ~((before == after) | (np.isnan(before) & np.isnan(after)))
    assert (changed == (named[:, None] & np.isfinite(before))).all()
    assert out["price"].equals(frame["price"]) and out["flag"].equals(frame["flag"])
    assert counts["wind"] == 47 and counts["load"] == 48 and "price" not in counts
    assert perturb_items(frame, [MIDDLE], seed=3)[0].equals(perturb_items(frame, [MIDDLE], seed=3)[0])


def test_co_batch_probe_on_its_own(data):
    leaky = co_batch_probe(lambda: in_context(leak=1e-3), data, LAST, INFO)
    assert not leaky.ok and len(leaky.moved) == 24 and leaky.forecasts == 24

    alone = co_batch_probe(lambda: in_context(leak=1e-3), data, LAST, INFO, items=["2023-04-16", "2031-01-01"])
    assert (alone.items, alone.forecasts, alone.moved) == ([], 0, [])  # the day itself and a day the frame lacks
    assert co_batch_probe(lambda: OLSModel(CLEAN), data, LAST, INFO).items is None  # no hook: nothing ran


def test_a_model_that_predicts_together_but_names_nothing_is_inconclusive(data):
    report = on(data, MIDDLE, TogetherUnnamed(CLEAN, training_window=WINDOW, name="Unnamed"))

    assert row(report)["status"] == "inconclusive"
    assert pd.isna(row(report)["co_batch"]) and row(report)["rerun"] == 0
    assert any("doesn't name them (_co_batch_items)" in line for line in report.problems())


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


def test_rerun_check_on_its_own(data):
    noisy = rerun_check(lambda: Unseeded(CLEAN), data, MIDDLE, INFO, horizon=2)
    assert (noisy.forecasts, len(noisy.differ), noisy.missing) == (48, 48, [])
    assert 0 < noisy.max_abs_difference < 1e-3
    same = rerun_check(lambda: OLSModel(CLEAN, training_window=WINDOW), data, MIDDLE, INFO, horizon=2)
    assert same.ok and same.max_abs_difference == 0.0


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


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------


def test_device_verifies_a_gpu_model_on_the_cpu_and_leaves_the_model_alone(data):
    model = OnDevice(CLEAN, name="GPU")
    assert row(on(data, MIDDLE, model))["status"] == "inconclusive"  # the card turns the rerun on

    report = on(data, MIDDLE, model, device="cpu")
    assert (row(report)["rerun"], row(report)["status"]) == (0, "ok")
    assert model.device == "cuda"
    # A model without a device runs as it is.
    assert on(data, MIDDLE, OLSModel(CLEAN, training_window=WINDOW, name="OLS"), device="cpu").ok


def test_epf_verify_passes_device_on(monkeypatch, capsys, tmp_path, data):
    from epftoolbox2._frame_csv import save_frame_csv

    data_path, yaml_path = tmp_path / "data.csv", tmp_path / "mp.yaml"
    save_frame_csv(data, data_path)
    pipeline(OLSModel(CLEAN, training_window=WINDOW, name="OLS")).save(yaml_path)
    seen = {}
    real = ModelPipeline.verify

    def verify(self, *args, **kwargs):
        seen.update(kwargs)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(ModelPipeline, "verify", verify)
    argv = ["epf", "verify", str(yaml_path), "--data", str(data_path), "--test-start", MIDDLE, "--test-end", MIDDLE]
    monkeypatch.setattr(sys, "argv", [*argv, "--days", "1", "--device", "cpu"])
    cli.main()  # exits only on a failure
    out = capsys.readouterr().out
    assert seen["device"] == "cpu"
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

    with pytest.raises(AssertionError, match="replaced by noise"):
        suite.check_verify(lambda name, **p: make(name, leak=1e-3), data, INFO, [LAST])
    with pytest.raises(AssertionError, match="nondeterministic"):
        suite.check_verify(lambda name, **p: Unseeded(CLEAN, name=name), data, INFO, [MIDDLE])
    with pytest.raises(AssertionError, match="doesn.t name them"):
        suite.check_verify(
            lambda name, **p: TogetherUnnamed(CLEAN, training_window=WINDOW, name=name), data, INFO, [LAST]
        )


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


def test_suite_preflight_and_provenance_wait_for_cards(data, tmp_path, monkeypatch):
    """No card: nothing to miss and nothing to show. A card with an extra: the pre-flight check needs 6a's
    `BaseModel._preflight` and skips without it; with one that refuses before any run starts, it passes."""
    suite.check_preflight_without_extra(make, monkeypatch, data, TEST)
    suite.check_provenance(make, data, tmp_path, TEST)

    monkeypatch.setitem(EXTRAS, "fake-extra", ("no_such_module_for_6c",))
    carded = SimpleNamespace(family="fake", extra="fake-extra", licence="MIT")
    monkeypatch.setattr(FakeInContext, "_card", carded, raising=False)
    if not hasattr(BaseModel, "_preflight"):
        with pytest.raises(pytest.skip.Exception):
            suite.check_preflight_without_extra(make, monkeypatch, data, TEST)

    # 6a as the spec has it: every model's pre-flight before the first model runs.
    def preflight(self):
        card = getattr(self, "_card", None)
        if card is not None and card.extra is not None:
            from epftoolbox2._extras import has_extra

            if not has_extra(card.extra):
                raise MissingExtraError(type(self).__name__, card.extra)

    real_run = ModelPipeline.run

    def run(self, *args, **kwargs):
        for model in self.models:
            model._preflight()
        return real_run(self, *args, **kwargs)

    monkeypatch.setattr(BaseModel, "_preflight", preflight, raising=False)
    monkeypatch.setattr(ModelPipeline, "run", run)
    suite.check_preflight_without_extra(make, monkeypatch, data, TEST)
    # A pre-flight that comes too late (after a model's run started) is caught.
    monkeypatch.setattr(ModelPipeline, "run", real_run)
    real_prepare = BaseModel._prepare_run

    def late(self, *args, **kwargs):
        out = real_prepare(self, *args, **kwargs)
        self._preflight()
        return out

    monkeypatch.setattr(BaseModel, "_prepare_run", late)
    with pytest.raises(AssertionError, match="started before the pre-flight"):
        suite.check_preflight_without_extra(make, monkeypatch, data, TEST)
