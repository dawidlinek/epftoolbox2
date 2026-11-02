"""`ModelPipeline.verify`: a clean pipeline passes, a planted leak fails, and a blind check is not a pass."""

from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from epftoolbox2.data.transformers import DailyStatsTransformer, LagTransformer
from epftoolbox2.information_set import (
    Availability,
    InformationSet,
    InformationSetError,
    leakage_probe,
    perturb_recent,
    perturb_unknown,
    positive_control,
)
from epftoolbox2.models import LassoCVModel, OLSModel
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline, VerifyReport
from epftoolbox2.verification import sample_issue_days
from tests.synthetic import PREDICTORS, calendar_shift, make_market_data

# The synthetic drivers stand for forecasts published well before their delivery; price follows the A44 default rule.
RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "load_actual": Availability.realised("1h"),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
}
CALENDAR = ["is_weekend_d+{horizon}", "is_weekday_d+{horizon}"]
CLEAN = PREDICTORS + ["price_d-1"]
ISSUE_DAY = "2023-04-12"
TEST_PERIOD = ("2023-04-03", "2023-04-28")
COLUMNS = ["model", "days", "forecasts", "changed", "max change", "control moved", "status"]


class InProcessPool:
    """Stands in for the worker pool: starting one per one-day run is most of the time these tests would take, and
    nothing here depends on it. `epf verify` in tests/test_cli.py runs through the real pool."""

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
    df = make_market_data(days=120, noise=5.0, seed=7, horizon=3)
    df["price_d-2"] = calendar_shift(df["price"], -2)
    return df


def pipeline(*models, **settings):
    out = ModelPipeline(information_set=InformationSet(rules=settings.pop("rules", RULES), **settings))
    for model in models:
        out.add_model(model)
    return out


def ols(predictors=CLEAN, name="OLS", window=40):
    return OLSModel(predictors, training_window=window, name=name)


def row(report, model=None):
    table = report.table()
    return table[table["model"] == (model or report.models[0])].iloc[0]


def issue_row_cells(frame, run_day):
    """Whether each row is dated on the issue day."""
    return frame.index.tz_localize(None).normalize() == pd.Timestamp(run_day)


def changed_cells(perturbed, original):
    a, b = perturbed.to_numpy(dtype=float), original.to_numpy(dtype=float)
    return ~((a == b) | (np.isnan(a) & np.isnan(b)))


# ---------------------------------------------------------------------------
# A clean pipeline passes
# ---------------------------------------------------------------------------


def test_clean_models_pass_with_the_control_moving(data):
    lasso = LassoCVModel(CLEAN, training_window=30, cv=3, name="Lasso")
    report = pipeline(ols(), lasso).verify(data, *TEST_PERIOD, days=1)

    assert isinstance(report, VerifyReport)
    assert report.ok
    report.assert_ok()
    table = report.table()
    assert list(table.columns) == COLUMNS
    assert table["model"].tolist() == ["OLS", "Lasso"]
    assert table["days"].tolist() == [1, 1]
    assert table["forecasts"].tolist() == [24, 24]
    assert table["changed"].tolist() == [0, 0]
    assert table["max change"].tolist() == [0.0, 0.0]
    assert table["control moved"].tolist() == [24, 24]
    assert table["status"].tolist() == ["ok", "ok"]
    assert report.problems() == []
    for check in report.checks:
        assert check.probe.perturbed["price"] > 0  # the probe had something to do
        assert check.control.perturbed["price_d-1"] > 0
        assert check.control.max_abs_change > 1.0


def test_a_legal_input_built_from_the_target_is_not_flagged(data):
    """Today's price is published on D-1, so a forecast of D+1 may read it, as a column or through a transformer. The
    model's own boundary (training targets end at D-1) must not turn that into a leak."""
    report = pipeline(ols(PREDICTORS + ["price"])).verify(data, *TEST_PERIOD, days=1)
    assert row(report)["status"] == "ok"

    def rebuild(frame):
        return DailyStatsTransformer("price", stats=["max"]).transform(frame)

    from_source = pipeline(ols(PREDICTORS + ["price_day_max"])).verify(data, *TEST_PERIOD, days=1, rebuild=rebuild)
    assert row(from_source)["status"] == "ok"
    assert row(from_source)["control moved"] == 24


# ---------------------------------------------------------------------------
# Planted leaks fail
# ---------------------------------------------------------------------------


def lagged_the_wrong_way(raw):
    """Looks like a one-day lag of the price, but the shift has the wrong sign: `price_d-1` holds tomorrow's price."""
    out = raw.drop(columns=["price_d-1"], errors="ignore")
    out["price_d-1"] = calendar_shift(raw["price"], 1)
    return out


def lagged_properly(raw):
    return LagTransformer(columns=["price"], lags=[1], freq="day").transform(raw)


def test_a_leaking_transformer_fails_through_rebuild(data):
    """The column's name and rule are legal, so only the raw data can show it: the built frame passes."""
    source = data.drop(columns=["price_d-1"])
    built = lagged_the_wrong_way(source)
    report_on_built = pipeline(ols()).verify(built, *TEST_PERIOD, days=1, positive_control=False)
    assert report_on_built.ok  # what rebuild is for: by its name, `price_d-1` is known

    report = pipeline(ols()).verify(source, *TEST_PERIOD, days=1, positive_control=False, rebuild=lagged_the_wrong_way)
    assert not report.ok
    assert row(report)["status"] == "leak"
    assert row(report)["changed"] == 24
    assert row(report)["max change"] > 1.0
    with pytest.raises(AssertionError, match="OLS: leak on .*depend on values not known"):
        report.assert_ok()

    clean = pipeline(ols()).verify(source, *TEST_PERIOD, days=1, rebuild=lagged_properly)
    assert clean.ok
    assert row(clean)["control moved"] == 24


class ReadsIssueDayTarget(OLSModel):
    """Its `load_d+1` column is really the price of the row's own day: the name is legal, the frame is untouched."""

    def _preprocess(self, data, horizon, target):
        out = super()._preprocess(data, horizon, target)
        out["load_d+1"] = out[target]
        return out


class TrainsOnIssueDayTarget(OLSModel):
    """Its newest training row has the price of the issue day as its target, one day past the model's boundary."""

    def _preprocess(self, data, horizon, target):
        out = super()._preprocess(data, horizon, target)
        out[f"{target}_d+1"] = calendar_shift(out[target], 2)
        return out


@pytest.mark.parametrize("model_class", [ReadsIssueDayTarget, TrainsOnIssueDayTarget])
def test_a_model_using_the_target_of_the_issue_day_fails(data, model_class):
    model = model_class(CLEAN, training_window=40, name="Sneaky")
    information_set = InformationSet(rules=RULES)
    # Nothing about its predictors gives it away, and by its rule the price of D is known at the issue time.
    model._check_inputs(data, "price", 1, "1h", information_set, ISSUE_DAY, ISSUE_DAY)
    assert leakage_probe(lambda: model_class(CLEAN, training_window=40), data, ISSUE_DAY, information_set).ok

    report = pipeline(model).verify(data, *TEST_PERIOD, days=1, positive_control=False)

    assert not report.ok
    assert row(report)["status"] == "leak"
    assert row(report)["changed"] == 24


def test_a_predictor_known_too_late_fails_for_the_periods_it_is_late(data):
    frame = data.assign(load_actual=data["load"])
    report = pipeline(ols(PREDICTORS + ["load_actual"])).verify(frame, ISSUE_DAY, ISSUE_DAY, days=1)

    assert row(report)["status"] == "leak"
    assert sorted({period for period, _ in report.checks[0].probe.changed}) == list(range(10, 24))
    assert row(report)["changed"] == 14
    assert "leak on 2023-04-12: 14 of 24" in report.problems()[0]


def test_a_transformer_leak_that_only_exists_across_dst_fails():
    """An elapsed 24-hour shift named `_d+1`: on the row before the spring switch it reads two local days ahead, a
    forecast published only the next morning. A calendar-day shift passes on the same day."""
    frame = make_market_data(start="2024-02-01", days=70, noise=5.0, seed=3)
    source = frame[["price", "is_weekend_d+1", "is_weekday_d+1"]].assign(load_forecast=frame["load"])
    predictors = ["load_forecast_d+1", "is_weekend_d+1", "is_weekday_d+1"]

    def elapsed_shift(raw):
        return raw.assign(**{"load_forecast_d+1": raw["load_forecast"].shift(-24, freq="h").reindex(raw.index)})

    def calendar_day_shift(raw):
        return raw.assign(**{"load_forecast_d+1": calendar_shift(raw["load_forecast"], 1)})

    rules = {"is_weekend_d+1": Availability.static(), "is_weekday_d+1": Availability.static()}

    def verify(rebuild):
        return pipeline(ols(predictors), rules=rules).verify(source, "2024-03-30", "2024-03-30", rebuild=rebuild)

    leaky = verify(elapsed_shift)
    assert row(leaky)["status"] == "leak"
    assert leaky.checks[0].probe.changed == [(23, 1)]

    assert row(verify(calendar_day_shift))["status"] == "ok"


# ---------------------------------------------------------------------------
# The positive control can fail, and a failed control is not a pass
# ---------------------------------------------------------------------------


class IgnoresItsInputs(OLSModel):
    """Forecasts the mean of its training targets, whatever the forecast's own row holds."""

    def _fit_predict(self, train_x, train_y, test_x):
        return float(train_y.mean()), []


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(ols(CALENDAR), id="only calendar inputs"),
        pytest.param(ols(["price_d-2"] + CALENDAR), id="only inputs older than the control reaches"),
        pytest.param(IgnoresItsInputs(CLEAN, training_window=40), id="doesn't read its inputs"),
    ],
)
def test_a_model_the_control_cannot_move_is_inconclusive(data, model):
    report = pipeline(model).verify(data, *TEST_PERIOD, days=1)

    assert row(report)["changed"] == 0  # the probe alone would have called it clean
    assert row(report)["control moved"] == 0
    assert row(report)["status"] == "inconclusive"
    assert not report.ok
    with pytest.raises(AssertionError, match="inconclusive.*isn't reaching"):
        report.assert_ok()

    without_control = pipeline(model).verify(data, *TEST_PERIOD, days=1, positive_control=False)
    assert without_control.ok
    assert pd.isna(row(without_control)["control moved"])
    assert row(without_control)["status"] == "ok"


def test_a_rebuild_that_ignores_its_input_is_inconclusive(data):
    """The probe perturbs the raw data, but the features come from somewhere else: nothing reaches the model. Under
    the probe this is a clean pass whatever the features hold, here a leak: the very failure the control exists for."""
    source = data.drop(columns=["price_d-1"])
    stale = lagged_the_wrong_way(source)
    report = pipeline(ols()).verify(source, *TEST_PERIOD, days=1, rebuild=lambda frame: stale)

    assert row(report)["changed"] == 0
    assert row(report)["control moved"] == 0
    assert row(report)["status"] == "inconclusive"
    assert not report.ok


def test_the_target_gives_the_control_no_way_to_pass(data):
    """Every model trains on the target, so perturbing it would move every model and the control could never fail.
    With a rebuild the raw price has to change to reach `price_d-1`, and still a calendar model must not move."""
    source = data.drop(columns=["price_d-1"])
    report = pipeline(ols(CALENDAR)).verify(source, *TEST_PERIOD, days=1, rebuild=lagged_properly)
    assert report.checks[0].control.perturbed["price"] > 0
    assert row(report)["control moved"] == 0
    assert row(report)["status"] == "inconclusive"


def test_the_control_reaches_a_forecast_through_its_own_row_only(data):
    """Training rows stay as they are: an outlier there would move a fit whatever the forecast's inputs are."""
    source = data[["price", "load", "is_weekend_d+1", "is_weekday_d+1"]]
    info = InformationSet(
        rules={**RULES, "is_weekend_d+1": Availability.static(), "is_weekday_d+1": Availability.static()}
    )

    def rebuild(raw):
        return raw.assign(**{"load_d+1": calendar_shift(raw["load"], 1)})

    seen = []

    class Recording(OLSModel):
        def run(self, frame, **kwargs):
            seen.append(frame)
            return super().run(frame, **kwargs)

    model = Recording(["load_d+1", "is_weekend_d+1", "is_weekday_d+1"], training_window=40)
    control = positive_control(lambda: model, source, ISSUE_DAY, info, rebuild=rebuild)

    assert control.moved
    original, perturbed = seen
    differs = changed_cells(perturbed[["load_d+1"]], original[["load_d+1"]]).ravel()
    days = original.index.tz_localize(None).normalize()
    assert differs[days == pd.Timestamp(ISSUE_DAY)].all()
    assert not differs[days < pd.Timestamp(ISSUE_DAY) - pd.Timedelta(days=1)].any()
    pd.testing.assert_series_equal(perturbed["price"], original["price"])


def test_an_input_without_a_rule_is_inconclusive(data):
    """The probe can't perturb what it knows nothing about, so a leak through it would go unseen."""
    frame = data.assign(mystery=data["load"].shift(-24))
    model = ols(CLEAN + ["mystery"])

    with pytest.raises(InformationSetError, match="no availability rule"):
        pipeline(model).verify(frame, *TEST_PERIOD, days=1)

    report = pipeline(model, unknown="ignore").verify(frame, *TEST_PERIOD, days=1)
    assert row(report)["changed"] == 0
    assert row(report)["control moved"] == 24
    assert row(report)["status"] == "inconclusive"
    assert "no availability rule for ['mystery']" in report.problems()[0]


def test_a_column_the_model_does_not_use_needs_no_rule(data):
    frame = data.assign(unused=1.0)
    report = pipeline(ols()).verify(frame, *TEST_PERIOD, days=1)
    assert report.ok


# ---------------------------------------------------------------------------
# What the control perturbs
# ---------------------------------------------------------------------------


def recent_rows(column, info, run_day="2023-04-12", frame=None, freq="1h"):
    frame = frame if frame is not None else make_market_data(noise=1.0, seed=2, horizon=3)
    frame = frame.assign(**{column: 1.0})
    perturbed, _, _ = perturb_recent(frame, run_day, info, freq, columns=[column])
    return frame, np.asarray(perturbed[column] != frame[column])


def test_the_control_reaches_yesterdays_prices_though_they_were_published_two_days_ago():
    """Moving the issue time back one day would leave `price_d-1` of D alone: published at 13:00 on D-2, it was already
    known at 11:45 on D-1. What brings it in is its delivery on D-1."""
    info = InformationSet()
    frame, hit = recent_rows("price_d-1", info)
    on_issue_day = issue_row_cells(frame, "2023-04-12")
    assert hit[on_issue_day].all()
    assert on_issue_day.sum() == 24

    moved_back, _, _ = perturb_unknown(frame, "2023-04-11", info, columns=["price_d-1"])
    assert not np.asarray(moved_back["price_d-1"] != frame["price_d-1"])[on_issue_day].any()


@pytest.mark.parametrize(
    "column, reached",
    [
        ("price_d-1", True),
        ("price_d-2", False),  # delivered on D-2, published on D-3
        ("price", True),  # the price of D, published on D-1
        ("load_actual_d-1", True),
        ("load_forecast_d+1", True),  # today's forecast for tomorrow, published at 10:00 on D
        ("generation_wind_forecast_d-1", True),
        ("is_monday", False),  # calendar: never unknown
        ("price_h05_d-1", True),  # the price at 05:00 on D-1, on every row of D
        ("price_h05_d-2", False),
        ("price_day_max_d-1", True),  # a statistic of D-1
        ("price_day_max_d-2", False),
    ],
)
def test_which_values_of_the_issue_day_row_the_control_perturbs(column, reached):
    a44 = Availability.published("13:00", 1)
    frame, hit = recent_rows(column, InformationSet(rules={"price_h*": a44, "price_day_*": a44}))
    assert hit[issue_row_cells(frame, "2023-04-12")].all() == reached
    assert hit[issue_row_cells(frame, "2023-04-12")].any() == reached


def test_a_slow_series_is_reached_by_its_newest_value():
    """Known a day after its delivery, `x_d-2` was delivered on D-2 but first known on D-1. `x_d-3` is older than that."""
    info = InformationSet(rules={"x": Availability.realised("24h")})
    frame = make_market_data(noise=1.0, seed=2)
    for column, reached in (("x_d-2", True), ("x_d-3", False)):
        _, hit = recent_rows(column, info, frame=frame)
        assert hit[issue_row_cells(frame, "2023-04-12")].any() == reached


def test_a_value_known_too_late_is_left_to_the_probe():
    """Earlier than the 10:00 publication, today's forecast isn't known yet: it belongs to the probe, not the control."""
    frame = make_market_data(noise=1.0, seed=2)
    info = InformationSet("09:00", "Europe/Warsaw")
    _, hit = recent_rows("load_forecast_d+1", info, frame=frame)
    assert not hit[issue_row_cells(frame, "2023-04-12")].any()


def test_the_control_and_the_probe_perturb_different_cells():
    index = make_market_data(noise=1.0, seed=2).index
    columns = ["price", "price_d-1", "load_actual", "load_actual_d-1", "load_forecast_d+1", "load_forecast_d+2"]
    frame = pd.DataFrame(1.0, index=index, columns=columns)
    info = InformationSet()

    recent, recent_counts, _ = perturb_recent(frame, ISSUE_DAY, info, columns=columns)
    unknown, _, _ = perturb_unknown(frame, ISSUE_DAY, info, columns=columns)

    assert not (changed_cells(recent[columns], frame[columns]) & changed_cells(unknown[columns], frame[columns])).any()
    assert set(recent_counts) == set(columns)


def test_the_control_keeps_missing_values_missing_and_older_rows_as_they_are():
    frame = make_market_data(noise=1.0, seed=2)
    on_issue_day = issue_row_cells(frame, ISSUE_DAY)
    frame.loc[on_issue_day, "price_d-1"] = np.where(frame.index[on_issue_day].hour == 3, np.nan, 5.0)

    perturbed, _, _ = perturb_recent(frame, ISSUE_DAY, InformationSet(), columns=["price_d-1"])

    assert perturbed["price_d-1"].isna().sum() == frame["price_d-1"].isna().sum()
    older = frame.index.tz_localize(None).normalize() < pd.Timestamp(ISSUE_DAY) - pd.Timedelta(days=1)
    pd.testing.assert_series_equal(perturbed.loc[older, "price_d-1"], frame.loc[older, "price_d-1"])


def test_the_control_leaves_the_training_rows_of_a_forecast_alone():
    """On D-2's row, `load_d+1` is delivered on D-1, but that row is what a forecast of D+1 trains on."""
    frame = make_market_data(noise=1.0, seed=2)
    perturbed, _, _ = perturb_recent(frame, ISSUE_DAY, InformationSet(rules=RULES), columns=["load_d+1"])

    days = frame.index.tz_localize(None).normalize()
    two_days_ago = days == pd.Timestamp(ISSUE_DAY) - pd.Timedelta(days=2)
    assert perturbed.loc[two_days_ago, "load_d+1"].tolist() == frame.loc[two_days_ago, "load_d+1"].tolist()
    assert (perturbed.loc[days == pd.Timestamp(ISSUE_DAY), "load_d+1"] > 1e5).all()


def test_the_control_does_not_touch_the_frame_it_is_given():
    frame = make_market_data(noise=1.0, seed=2)
    before = frame.copy()
    perturb_recent(frame, ISSUE_DAY, InformationSet(rules=RULES))
    pd.testing.assert_frame_equal(frame, before)


# ---------------------------------------------------------------------------
# Reproducibility, and no effect on anything else
# ---------------------------------------------------------------------------


def test_same_seed_same_table_and_days(data):
    def verify(seed):
        return pipeline(ols()).verify(data, *TEST_PERIOD, days=1, seed=seed)

    first, second = verify(1), verify(1)
    assert first.issue_days == second.issue_days
    pd.testing.assert_frame_equal(first.table(), second.table())
    assert [c.control.max_abs_change for c in first.checks] == [c.control.max_abs_change for c in second.checks]
    assert first.issue_days != sample_issue_days(data.index, *TEST_PERIOD, days=1, seed=2)


def test_the_sampled_days_do_not_depend_on_the_model_order_or_selection(data):
    a, b = ols(name="A"), ols(CALENDAR, name="B")
    settings = dict(days=1, seed=4, positive_control=False)
    forward = pipeline(a, b).verify(data, *TEST_PERIOD, **settings)
    backward = pipeline(b, a).verify(data, *TEST_PERIOD, **settings)
    only_b = pipeline(a, b).verify(data, *TEST_PERIOD, models=["B"], **settings)

    assert forward.issue_days == backward.issue_days == only_b.issue_days
    assert only_b.models == ["B"]
    for name in ("A", "B"):
        pd.testing.assert_series_equal(row(forward, name), row(backward, name), check_names=False)
    pd.testing.assert_series_equal(row(forward, "B"), row(only_b, "B"), check_names=False)
    assert {c.run_day for c in forward.checks} == set(forward.issue_days)


def test_sample_issue_days():
    index = pd.date_range("2024-01-01", "2024-01-31 23:00", freq="h", tz="Europe/Warsaw")

    days = sample_issue_days(index, "2024-01-10", "2024-01-20", days=4, seed=0)
    assert days == sorted(days)
    assert len(set(days)) == 4
    assert all("2024-01-10" <= day <= "2024-01-20" for day in days)
    assert days == sample_issue_days(index, "2024-01-10", "2024-01-20", days=4, seed=0)

    assert sample_issue_days(index, "2024-01-30", "2024-03-01", days=5) == ["2024-01-30", "2024-01-31"]
    gap = index[(index < "2024-01-11") | (index >= "2024-01-12")]
    assert sample_issue_days(gap, "2024-01-10", "2024-01-12", days=5) == ["2024-01-10", "2024-01-12"]

    with pytest.raises(ValueError, match="no rows"):
        sample_issue_days(index, "2025-01-01", "2025-01-02")
    with pytest.raises(ValueError, match="at least 1"):
        sample_issue_days(index, "2024-01-10", "2024-01-20", days=0)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        sample_issue_days(pd.RangeIndex(10), "2024-01-10", "2024-01-20")


def test_verifying_has_no_effect_on_a_later_run(data):
    before = data.copy()
    verified = pipeline(ols())
    state = dict(verified.models[0].__dict__)
    assert verified.verify(data, *TEST_PERIOD, days=1).ok
    assert verified.models[0].__dict__ == state
    pd.testing.assert_frame_equal(data, before)

    run_args = dict(test_start="2023-04-10", test_end="2023-04-10", horizon=1, forecast_only=True)
    after = verified.run(data, **run_args).predictions()
    fresh = pipeline(ols()).run(data, **run_args).predictions()
    pd.testing.assert_frame_equal(after, fresh)


def test_a_rebuild_that_edits_its_argument_does_not_reach_the_callers_frame(data):
    source = data.drop(columns=["price_d-1"])
    before = source.copy()

    def rebuild(frame):
        frame["price_d-1"] = calendar_shift(frame["price"], -1)
        return frame

    assert pipeline(ols()).verify(source, *TEST_PERIOD, days=1, rebuild=rebuild).ok
    pd.testing.assert_frame_equal(source, before)


# ---------------------------------------------------------------------------
# Several horizons
# ---------------------------------------------------------------------------


def test_every_horizon_is_probed_and_controlled(data):
    report = pipeline(ols()).verify(data, *TEST_PERIOD, horizon=3, days=1)

    assert report.ok
    assert row(report)["forecasts"] == 24 * 3
    assert row(report)["control moved"] == 24 * 3
    assert {hz for check in report.checks for _, hz in check.control.moved} == {1, 2, 3}


def test_a_leak_in_one_horizon_is_found_there(data):
    def predictors(horizon):
        return [f"load_d+{horizon}", f"is_weekend_d+{horizon}", f"is_weekday_d+{horizon}", "price_d-1"] + (
            ["load_actual"] if horizon == 2 else []
        )

    frame = data.assign(load_actual=data["load"])
    report = pipeline(ols(predictors)).verify(frame, ISSUE_DAY, ISSUE_DAY, horizon=3)

    assert row(report)["status"] == "leak"
    assert report.checks[0].probe.changed == [(period, 2) for period in range(10, 24)]


def test_a_horizon_the_control_cannot_move_is_inconclusive(data):
    """The other horizon's movement says nothing about this one, whose inputs the perturbation never reaches."""

    def predictors(horizon):
        return CALENDAR if horizon == 2 else [f"load_d+{horizon}", "is_weekday_d+1", "is_weekend_d+1"]

    report = pipeline(ols(predictors)).verify(data, ISSUE_DAY, ISSUE_DAY, horizon=2)

    assert row(report)["forecasts"] == 48
    assert row(report)["control moved"] == 24
    assert report.checks[0].unreached_horizons == [2]
    assert row(report)["status"] == "inconclusive"
    assert "horizon(s) [2]" in report.problems()[0]


# ---------------------------------------------------------------------------
# DST switches
# ---------------------------------------------------------------------------


def test_a_clean_model_passes_on_the_spring_switch(data):
    """A 23-hour issue day: 23 forecasts of each horizon, as no forecast is issued from the hour the clock skips."""
    report = pipeline(ols()).verify(data, "2023-03-26", "2023-03-26", horizon=2, days=1)

    assert report.issue_days == ["2023-03-26"]
    assert report.ok
    assert row(report)["forecasts"] == 46
    assert row(report)["control moved"] == 46


def test_a_clean_model_passes_on_the_fall_back_switch():
    data = make_market_data(start="2023-08-01", days=120, noise=5.0, seed=1, horizon=2)
    report = pipeline(ols()).verify(data, "2023-10-29", "2023-10-29", horizon=2, days=1)

    assert report.ok
    assert row(report)["forecasts"] == 48
    assert row(report)["control moved"] == 48


@pytest.mark.parametrize(
    "run_day, hours",
    [("2023-03-26", 23), ("2023-03-25", 24), ("2023-10-29", 25)],
)
def test_the_control_reaches_every_row_of_a_switch_day(run_day, hours):
    """Local days: D-1 starts at the same wall-clock midnight whatever the length of D-1 or D."""
    frame = make_market_data(start="2023-01-01", days=320, noise=1.0, seed=2)
    frame_day = issue_row_cells(frame, run_day)
    assert frame_day.sum() == hours

    perturbed, _, _ = perturb_recent(frame, run_day, InformationSet(), columns=["price_d-1"])

    assert (perturbed.loc[frame_day, "price_d-1"].notna() == frame.loc[frame_day, "price_d-1"].notna()).all()
    finite = frame.loc[frame_day, "price_d-1"].notna().to_numpy()
    assert (perturbed.loc[frame_day, "price_d-1"].to_numpy()[finite] > 1e5).all()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_verify_needs_an_information_set(data):
    with pytest.raises(InformationSetError, match="issue_time or information_set"):
        ModelPipeline().add_model(ols()).verify(data, *TEST_PERIOD)


def test_verify_needs_a_model(data):
    with pytest.raises(ValueError, match="At least one model"):
        pipeline().verify(data, *TEST_PERIOD)


def test_verify_refuses_an_unknown_model_name(data):
    with pytest.raises(ValueError, match=r"Unknown model\(s\) \['Nope'\]"):
        pipeline(ols()).verify(data, *TEST_PERIOD, models=["Nope"])


def test_verify_checks_the_data_contract(data):
    with pytest.raises(ValueError, match="not in the data"):
        pipeline(ols(["nope"])).verify(data, *TEST_PERIOD)


def test_a_report_of_no_models_is_not_ok(data):
    report = pipeline(ols()).verify(data, *TEST_PERIOD, models=[])
    assert not report.ok
    with pytest.raises(AssertionError, match="Nothing was verified"):
        report.assert_ok()


# ---------------------------------------------------------------------------
# Other frames
# ---------------------------------------------------------------------------


def test_a_quarter_hourly_pipeline_is_verified_per_period():
    data = make_market_data(start="2023-03-01", days=45, freq="15min", noise=5.0, seed=5, horizon=1)
    report = pipeline(ols(window=20)).verify(data, "2023-04-05", "2023-04-10", freq="15min", days=1)

    assert report.ok
    assert row(report)["forecasts"] == 96
    assert row(report)["control moved"] == 96

    late = pipeline(ols(PREDICTORS + ["load_actual"], window=20)).verify(
        data.assign(load_actual=data["load"]), "2023-04-05", "2023-04-10", freq="15min", days=1, positive_control=False
    )
    assert row(late)["status"] == "leak"
    # Known 1 h after its 15-minute period ends: at 11:45 that is period 42 (10:30-10:45), and period 43 is not.
    assert sorted({period for period, _ in late.checks[0].probe.changed}) == list(range(43, 96))


def test_a_naive_wall_clock_index_is_verified():
    """DSTNormalizeTransformer's output: the rules are read in the zone of the issue time."""
    data = make_market_data(start="2023-04-01", days=75, noise=5.0, seed=5)
    data.index = data.index.tz_localize(None)
    report = pipeline(ols()).verify(data, "2023-06-01", "2023-06-10", days=1)

    assert report.ok
    assert row(report)["control moved"] == 24

    source = data.drop(columns=["price_d-1"])
    from_source = pipeline(ols()).verify(source, "2023-06-01", "2023-06-10", days=1, rebuild=lagged_properly)
    assert from_source.ok
    assert row(from_source)["control moved"] == 24


# ---------------------------------------------------------------------------
# Blind spots a passing control could hide
# ---------------------------------------------------------------------------


def test_a_boolean_input_is_probed_like_any_other(data):
    """`leakage_probe` skips boolean columns, and a model reads them as numbers, so a flag can carry a leak."""
    frame = data.assign(**{"spike_d+2": calendar_shift(data["price"], 2) > data["price"].median()})
    assert frame["spike_d+2"].dtype == bool
    late_by_its_rule = {**RULES, "spike": Availability.published("13:00", 1)}

    report = pipeline(ols(PREDICTORS + ["spike_d+2"]), rules=late_by_its_rule).verify(frame, ISSUE_DAY, ISSUE_DAY)
    assert row(report)["status"] == "leak"

    with pytest.raises(InformationSetError, match="no availability rule"):
        pipeline(ols(PREDICTORS + ["spike_d+2"])).verify(frame, ISSUE_DAY, ISSUE_DAY)

    unruled = pipeline(ols(PREDICTORS + ["spike_d+2"]), unknown="ignore").verify(frame, ISSUE_DAY, ISSUE_DAY)
    assert row(unruled)["status"] == "inconclusive"


def test_a_boolean_calendar_input_stays_legal(data):
    frame = data.assign(is_holiday=data["is_weekend"] > 0)
    report = pipeline(ols(CLEAN + ["is_holiday"])).verify(frame, ISSUE_DAY, ISSUE_DAY)
    assert row(report)["status"] == "ok"


@pytest.mark.parametrize("model_class", [ReadsIssueDayTarget, TrainsOnIssueDayTarget])
def test_listing_the_target_as_an_input_does_not_lift_the_boundary_on_what_a_model_trains_on(data, model_class):
    model = model_class(CLEAN + ["price"], training_window=40, name="Sneaky")
    report = pipeline(model).verify(data, ISSUE_DAY, ISSUE_DAY, positive_control=False)

    assert row(report)["status"] == "leak"
    assert row(report)["changed"] == 24


def test_listing_the_target_in_one_horizon_does_not_lift_the_boundary_in_another(data):
    def predictors(horizon):
        return [f"load_d+{horizon}", f"is_weekend_d+{horizon}", f"is_weekday_d+{horizon}"] + (
            ["price"] if horizon == 2 else []
        )

    report = pipeline(TrainsOnIssueDayTarget(predictors, training_window=40)).verify(
        data, ISSUE_DAY, ISSUE_DAY, horizon=2, positive_control=False
    )
    assert {horizon for _, horizon in report.checks[0].probe.changed} == {1}


@pytest.mark.parametrize("with_rebuild", [False, True])
def test_todays_price_as_the_only_fresh_input_moves_the_control(data, with_rebuild):
    """The price of D is a legal input of a forecast of D+1, so perturbing it must reach the forecast, not the labels."""
    model = ols(["price"] + CALENDAR)
    rebuild = (lambda frame: frame) if with_rebuild else None
    report = pipeline(model).verify(data, ISSUE_DAY, ISSUE_DAY, rebuild=rebuild)

    assert row(report)["changed"] == 0
    assert row(report)["control moved"] == 24
    assert row(report)["status"] == "ok"


@pytest.mark.parametrize(
    "predictors",
    [
        pytest.param(["load_actual_d-2"] + CALENDAR, id="one late slot of an old input"),
        pytest.param(
            lambda horizon, hour: (["price_d-1"] if hour == 0 else []) + CALENDAR, id="a fresh input in one period only"
        ),
    ],
)
def test_a_period_the_control_never_moves_is_inconclusive(data, predictors):
    """One forecast moving says as little about the other periods as about another horizon."""
    frame = data.assign(**{"load_actual_d-2": data["load"]})
    report = pipeline(ols(predictors)).verify(frame, ISSUE_DAY, ISSUE_DAY, days=1)

    assert row(report)["control moved"] == 1
    assert row(report)["status"] == "inconclusive"
    assert "never moved" in report.problems()[0]


def test_a_period_needs_to_move_on_one_sampled_day_only(data):
    """A sparse fit may drop its fresh inputs on a day. That day says nothing, the others vouch for that forecast."""
    report = pipeline(ols()).verify(data, *TEST_PERIOD, days=2)
    assert row(report)["control moved"] == 48

    # Every forecast of the first day moves on the second day instead, and the other way round.
    first, second = report.checks
    first.control.unmoved, first.control.moved = first.control.moved[:1], first.control.moved[1:]
    second.control.unmoved, second.control.moved = second.control.moved[1:], second.control.moved[:1]
    assert row(report)["control moved"] == 24
    assert row(report)["status"] == "ok"


def test_a_rebuild_that_reads_its_argument_for_some_inputs_only_is_inconclusive(data):
    """The control moves through `price_d-1`, but `load_d+1` comes from elsewhere, here with a leak in it."""
    source = data.drop(columns=["price_d-1"])
    stale = source["load_d+1"] + 3 * calendar_shift(source["price"], 1)

    def rebuild(raw):
        return lagged_properly(raw).assign(**{"load_d+1": stale})

    report = pipeline(ols()).verify(source, ISSUE_DAY, ISSUE_DAY, rebuild=rebuild)

    assert row(report)["changed"] == 0
    assert row(report)["control moved"] == 24
    assert row(report)["status"] == "inconclusive"
    assert "['load_d+1']" in report.problems()[0]


def test_an_input_built_from_the_index_only_needs_a_static_rule(data):
    source = data.drop(columns=["price_d-1"])

    def rebuild(raw):
        return lagged_properly(raw).assign(peak=(raw.index.hour >= 8).astype(float))

    def verify(rules):
        return pipeline(ols(CLEAN + ["peak"]), rules=rules).verify(source, ISSUE_DAY, ISSUE_DAY, rebuild=rebuild)

    assert row(verify(RULES))["status"] == "inconclusive"
    assert row(verify({**RULES, "peak": Availability.static()}))["status"] == "ok"


@pytest.mark.parametrize(
    "zone, predictors, status, moved",
    [
        ("Europe/Warsaw", ["price_d-1"] + CALENDAR, "ok", 24),
        ("America/New_York", ["price_d-1"] + CALENDAR, "ok", 24),
        ("America/New_York", ["price_d-2"] + CALENDAR, "inconclusive", 0),
    ],
)
def test_the_control_counts_market_days_on_a_utc_source(zone, predictors, status, moved):
    """Sources return UTC and `rebuild` puts the index in the market's zone, so "D-1" is a day of that zone."""
    index = pd.date_range("2023-01-01", periods=24 * 110, freq="h", tz=zone)
    wall = index.tz_localize(None)
    weekend = (wall.dayofweek >= 5).astype(float)
    rng = np.random.default_rng(1)
    source = pd.DataFrame({"price": 50 + 10 * weekend + rng.normal(0, 3, len(index))}, index=index).tz_convert("UTC")
    source["is_weekend"] = weekend
    source["is_weekday"] = 1.0 - weekend

    def rebuild(raw):
        local = raw.tz_convert(zone)
        lags = LagTransformer(columns=["price"], lags=[1, 2], freq="day").transform(local)
        return LagTransformer(columns=["is_weekend", "is_weekday"], lags=[-1], freq="day").transform(lags)

    report = pipeline(ols(predictors)).verify(source, "2023-04-12", "2023-04-12", rebuild=rebuild)

    assert row(report)["changed"] == 0
    assert row(report)["control moved"] == moved
    assert row(report)["status"] == status


def test_a_horizon_with_no_forecast_is_not_blamed_on_the_probe(data):
    """On the last days of a frame `load_d+2` has no value. That says nothing about the perturbation reaching a model."""
    frame = data.loc[:"2023-04-13"].copy()
    frame.loc["2023-04-12":, "load_d+2"] = np.nan
    report = pipeline(ols()).verify(frame, "2023-04-13", "2023-04-13", horizon=2)

    assert row(report)["control moved"] == row(report)["forecasts"]
    assert row(report)["status"] == "inconclusive"
    problems = "\n".join(report.problems())
    assert "no forecast of horizon(s) [2]" in problems
    assert "isn't reaching" not in problems


def test_status_of_a_model_that_was_not_verified_raises(data):
    report = pipeline(ols()).verify(data, ISSUE_DAY, ISSUE_DAY)
    with pytest.raises(KeyError, match="typo"):
        report.status("typo")
