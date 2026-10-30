"""`EvaluationReport.by_quarter()`, `by_day()` and `by_price_level()`.

They are tables like `by_year()`: one shared path cuts every model's frame into groups and runs every evaluator on each
group alone. These tests check the cut against hand-computed groups, the evaluators inside a group against the same
evaluator called by hand on the group's frames, and the price levels against the definition (evalproto's Q4).
"""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.evaluators import (
    DMEvaluator,
    Evaluator,
    MAEEvaluator,
    MCSEvaluator,
    RMSEEvaluator,
    rMAEEvaluator,
    stats,
)
from epftoolbox2.evaluators.base import FORECAST_KEYS
from epftoolbox2.results.report import EvaluationReport, metric_columns
from tests.forecast_records import daily_losses, forecast_records, shared_errors

pytestmark = [
    pytest.mark.filterwarnings("ignore:The comparison rests on"),
    pytest.mark.filterwarnings("ignore:The model confidence set needs"),
]

NEW_TABLES = {"by_quarter": ["year", "quarter"], "by_day": ["target_date"], "by_price_level": ["price_level"]}


def record(run_date, hour, error, actual=100.0, horizon=1):
    run = pd.Timestamp(run_date)
    return {
        "run_date": run.date().isoformat(),
        "target_date": (run + pd.Timedelta(days=horizon)).date().isoformat(),
        "run_weekday": int(run.dayofweek),
        "hour": hour,
        "horizon": horizon,
        "day_in_test": 0,
        "prediction": actual + error,
        "actual": actual,
    }


def small_study():
    """Five target days across the end of the first quarter of 2023, two hours each, with errors chosen by hand.

    base errors by target day (hour 0, hour 1): 03-30 (1, 3), 03-31 (2, 6), 04-01 (4, 4), 04-02 (1, 1), 04-03 (3, 5).
    m    errors:                                03-30 (2, 2), 03-31 (1, -), 04-01 (1, 1), 04-02 (2, 2), 04-03 (3, 3);
    it lacks the hour 1 of 03-31, where the base has a large error.
    """
    base_errors = {
        "2023-03-29": (1, 3),
        "2023-03-30": (2, 6),
        "2023-03-31": (4, 4),
        "2023-04-01": (1, 1),
        "2023-04-02": (3, 5),
    }
    m_errors = {
        "2023-03-29": (2, 2),
        "2023-03-30": (1, None),
        "2023-03-31": (1, 1),
        "2023-04-01": (2, 2),
        "2023-04-02": (3, 3),
    }
    base, m = [], []
    for results, errors in ((base, base_errors), (m, m_errors)):
        for run_date, pair in errors.items():
            for hour, error in enumerate(pair):
                if error is not None:
                    results.append(record(run_date, hour, error))
    return {"base": base, "m": m}


@pytest.fixture(scope="module")
def small():
    evaluators = [MAEEvaluator(), RMSEEvaluator(), rMAEEvaluator("base"), DMEvaluator("base", reps=50)]
    return EvaluationReport(small_study(), evaluators)


# ---------------------------------------------------------------------------------------------------------------------
# by_quarter and by_day against hand-computed groups
# ---------------------------------------------------------------------------------------------------------------------


def test_by_quarter_on_a_hand_computed_example(small):
    """Q1 holds the target days 03-30 and 03-31, Q2 the days 04-01 to 04-03.

    Q1 own MAEs: base (1 + 3 + 2 + 6) / 4 = 3, m (2 + 2 + 1) / 3. On the three keys they share the base has (1 + 3 + 2) / 3
    = 2, so rMAE is (5 / 3) / 2, not (5 / 3) / 3. Q2: base 18 / 6 = 3, m 12 / 6 = 2, rMAE 2 / 3.
    """
    table = small.by_quarter().set_index(["year", "quarter", "model"])

    assert list(table.index.get_level_values("quarter").unique()) == [1, 2]
    assert table.loc[(2023, 1, "base"), "MAE"] == 3.0
    assert table.loc[(2023, 1, "m"), "MAE"] == pytest.approx(5 / 3)
    assert table.loc[(2023, 1, "m"), "rMAE"] == pytest.approx((5 / 3) / 2)
    assert table.loc[(2023, 1, "base"), "rMAE"] == 1.0
    assert table.loc[(2023, 2, "base"), "MAE"] == 3.0
    assert table.loc[(2023, 2, "m"), "MAE"] == 2.0
    assert table.loc[(2023, 2, "m"), "rMAE"] == pytest.approx(2 / 3)
    assert table.loc[(2023, 1, "base"), "RMSE"] == pytest.approx(np.sqrt((1 + 9 + 4 + 36) / 4))
    assert table.loc[(2023, 2, "m"), "RMSE"] == pytest.approx(np.sqrt((1 + 1 + 4 + 4 + 9 + 9) / 6))


def test_by_quarter_counts_each_models_forecasts_and_the_shared_ones(small):
    table = small.by_quarter().set_index(["year", "quarter", "model"])

    assert table.loc[(2023, 1, "base"), "forecasts"] == 4
    assert table.loc[(2023, 1, "m"), "forecasts"] == 3
    assert table.loc[(2023, 2, "m"), "forecasts"] == 6
    # DM prints the days and forecasts the pair shares: the three keys of Q1 are on two run days.
    assert table.loc[(2023, 1, "m"), "DM n_forecasts"] == 3
    assert table.loc[(2023, 1, "m"), "DM n_days"] == 2
    assert table.loc[(2023, 2, "m"), "DM n_forecasts"] == 6
    assert table.loc[(2023, 2, "m"), "DM n_days"] == 3
    assert table.loc[(2023, 2, "m"), "DM ratio"] == pytest.approx(2 / 3)


def test_by_quarter_dm_is_the_kernel_on_the_quarters_daily_losses(small):
    """Q2 has three days: daily mean |error| of base (4, 1, 4) and of m (1, 2, 3)."""
    row = small.by_quarter().set_index(["year", "quarter", "model"]).loc[(2023, 2, "m")]

    expected = stats.dm_test(np.array([1.0, 2.0, 3.0]), np.array([4.0, 1.0, 4.0]))

    assert row["DM p"] == expected.p_value


def test_by_day_on_a_hand_computed_example(small):
    """Target day 03-31 has base errors (2, 6) and m only (1,): the pair shares the key of hour 0."""
    table = small.by_day()
    day = table[table["target_date"] == "2023-03-31"].set_index("model")

    assert list(table.columns[:3]) == ["model", "target_date", "forecasts"]
    assert table["target_date"].is_monotonic_increasing
    assert day.loc["base", "MAE"] == 4.0
    assert day.loc["m", "MAE"] == 1.0
    assert day.loc["m", "rMAE"] == pytest.approx(1 / 2)
    assert day.loc["m", "forecasts"] == 1
    assert day.loc["m", "DM n_forecasts"] == 1
    assert day.loc["m", "DM n_days"] == 1
    assert pd.isna(day.loc["m", "DM p"])  # one day is nothing to test


def test_by_day_has_one_group_per_target_date_and_model(small):
    table = small.by_day()

    assert len(table) == 5 + 5  # five target days for base, five for m
    assert table["target_date"].nunique() == 5
    assert table["forecasts"].sum() == sum(len(r) for r in small_study().values())
    assert table.set_index(["target_date", "model"])["forecasts"].loc["2023-03-31"].to_dict() == {"base": 2, "m": 1}


# ---------------------------------------------------------------------------------------------------------------------
# Against merges on the keys, and against the evaluators called by hand
# ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def year_and_a_bit():
    """Three models over 15 months (Q1 2023 to Q1 2024): one with holes, one with a different seed."""
    run_dates = pd.date_range("2023-01-01", "2024-03-31", freq="D")
    results = {
        "base": forecast_records(1, run_dates, noise=6.0),
        "whole": forecast_records(2, run_dates, noise=4.0),
        "holes": forecast_records(3, run_dates, noise=5.0, drop=lambda r: r["hour"] == 7 or r["day_in_test"] % 11 == 0),
    }
    evaluators = [MAEEvaluator(), rMAEEvaluator("base"), DMEvaluator("base", reps=50), MCSEvaluator(reps=50)]
    return results, EvaluationReport(results, evaluators)


def target_quarter(records):
    keep = {}
    for r in records:
        keep.setdefault(pd.Timestamp(r["target_date"]).to_period("Q"), []).append(r)
    return keep


@pytest.mark.parametrize("model", ["whole", "holes"])
def test_by_quarter_agrees_with_a_merge_on_the_keys_inside_each_quarter(year_and_a_bit, model):
    results, report = year_and_a_bit
    table = report.by_quarter().set_index(["year", "quarter", "model"])
    base, other = target_quarter(results["base"]), target_quarter(results[model])

    # The last run day, 2024-03-31, forecasts 2024-04-01: a second quarter of one day.
    assert sorted(base) == [pd.Period(q) for q in ("2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1", "2024Q2")]
    for period in base:
        errors = shared_errors(base[period], other[period])
        row = table.loc[(period.year, period.quarter, model)]
        assert row["DM n_forecasts"] == len(errors)
        assert row["DM n_days"] == errors["run_date"].nunique()
        ratio = errors["b"].mean() / errors["a"].mean()  # the model is `b`, the base `a`
        assert row["rMAE"] == pytest.approx(ratio, rel=1e-12)
        assert row["DM ratio"] == pytest.approx(ratio, rel=1e-12)
        losses = daily_losses(errors)
        if len(losses) < 2:  # a quarter of one day has nothing to test
            assert np.isnan(row["DM p"])
            continue
        expected = stats.dm_test(losses["b"].to_numpy(), losses["a"].to_numpy())
        assert row["DM p"] == pytest.approx(expected.p_value, rel=1e-12)


@pytest.mark.parametrize(
    "method, group",
    [("by_quarter", (2023, 3)), ("by_quarter", (2024, 1)), ("by_day", (pd.Timestamp("2023-08-15"),))],
)
def test_a_row_is_the_set_level_evaluators_run_on_that_group_alone(year_and_a_bit, method, group):
    """The table adds nothing of its own: a row equals each evaluator's `compute_group` on the group's frames."""
    _, report = year_and_a_bit
    keys = NEW_TABLES[method]
    table = getattr(report, method)().set_index([*keys, "model"])

    def in_group(frame):
        if method == "by_quarter":
            return (frame["target_date"].dt.year == group[0]) & (frame["target_date"].dt.quarter == group[1])
        return frame["target_date"] == group[0]

    frames = {name: frame[in_group(frame)] for name, frame in report._model_frames().items()}
    for evaluator in report.evaluators:
        for name, result in evaluator.compute_group(frames).items():
            row = table.loc[(*group, name)]
            for key, value in (result if isinstance(result, dict) else {"": result}).items():
                got = row[evaluator.column_name(key)]
                assert (pd.isna(got) and pd.isna(value)) or got == value, (name, evaluator.name, key)


@pytest.mark.parametrize("method", list(NEW_TABLES))
def test_every_comparing_table_prints_its_counts(year_and_a_bit, method):
    _, report = year_and_a_bit
    table = getattr(report, method)()

    for column in ("forecasts", "DM n_days", "DM n_forecasts", "MCS n_days", "MCS n_forecasts"):
        assert column in table.columns
    assert (table["forecasts"] > 0).all()
    assert (table["DM n_forecasts"] <= table["forecasts"]).all()


@pytest.mark.parametrize("method", list(NEW_TABLES))
def test_the_new_tables_leave_their_keys_out_of_the_metric_columns(year_and_a_bit, method):
    _, report = year_and_a_bit
    table = getattr(report, method)()

    assert metric_columns(table) == [c for c in table.columns if c not in ("model", "forecasts", *NEW_TABLES[method])]
    assert "MAE" in metric_columns(table)


class Recorder(Evaluator):
    """Keeps every frame the report hands to an evaluator."""

    name = "seen"

    def __init__(self):
        self.calls = []

    def compute(self, df, **kwargs):
        self.calls.append((df, kwargs["model_dfs"]))
        return float(len(df))


@pytest.mark.parametrize("method", list(NEW_TABLES))
def test_evaluators_get_the_same_keyed_frames_as_in_every_other_table(method):
    recorder = Recorder()
    run_dates = pd.date_range("2023-03-20", periods=20, freq="D")
    results = {"a": forecast_records(1, run_dates), "b": forecast_records(2, run_dates, drop=lambda r: r["hour"] == 3)}

    getattr(EvaluationReport(results, [recorder]), method)()

    assert recorder.calls
    for df, model_dfs in recorder.calls:
        for frame in (df, *model_dfs.values()):
            assert list(frame.columns) == [*FORECAST_KEYS, "prediction", "actual"]
            assert not frame.duplicated(list(FORECAST_KEYS)).any()


@pytest.mark.parametrize("method", list(NEW_TABLES))
def test_the_new_tables_do_not_depend_on_the_order_of_the_records(year_and_a_bit, method):
    results, report = year_and_a_bit
    shuffled = {
        name: [records[i] for i in np.random.default_rng(5).permutation(len(records))]
        for name, records in results.items()
    }

    other = EvaluationReport(shuffled, report.evaluators)

    pd.testing.assert_frame_equal(getattr(other, method)(), getattr(report, method)(), check_exact=True)


def test_the_new_tables_are_not_exporter_tables(year_and_a_bit):
    """`compute_all()` is what the exporters write: it keeps the tables it had."""
    _, report = year_and_a_bit

    assert list(report.compute_all()) == [
        "summary",
        "hour",
        "horizon",
        "hour_horizon",
        "year",
        "year_horizon",
        "run_weekday_horizon",
        "target_weekday_horizon",
    ]


# ---------------------------------------------------------------------------------------------------------------------
# by_price_level
# ---------------------------------------------------------------------------------------------------------------------


def priced_records(actual, errors, days=100):
    """One model's records over `days` target days of 24 hours, with the given actual prices and errors (both 2-D)."""
    run_dates = pd.date_range("2023-06-01", periods=days, freq="D")
    results = []
    for d, run_date in enumerate(run_dates):
        for hour in range(24):
            results.append(record(run_date, hour, float(errors[d, hour]), float(actual[d, hour])))
    return results


@pytest.fixture(scope="module")
def prices():
    """2,400 hours: 20 negative prices (-30 to -11), 30 high ones (200 to 229), the rest 40 to 63."""
    actual = np.tile(40.0 + np.arange(24), (100, 1))
    flat = actual.reshape(-1)
    flat[[i * 120 + 5 for i in range(20)]] = -30.0 + np.arange(20)
    flat[[1200 + i * 40 + 3 for i in range(30)]] = 200.0 + np.arange(30)
    return actual


def level_masks(actual, threshold):
    values = actual.reshape(-1)
    return values < 0, values >= threshold, (values >= 0) & (values < threshold)


def test_the_spike_threshold_is_the_99th_percentile_when_it_is_below_300(prices):
    """Sorted, the 30 high prices are the last 30 of 2,400; the 99th percentile sits at position 2,375.01: 205.01."""
    errors = np.ones_like(prices)
    report = EvaluationReport({"a": priced_records(prices, errors)}, [MAEEvaluator()])

    table = report.by_price_level()

    assert table.attrs["spike_threshold"] == pytest.approx(205.01)
    assert table.attrs["spike_quantile"] == pytest.approx(205.01)
    assert "205.01" in table.attrs["spike_rule"]
    counts = table.set_index("price_level")["forecasts"]
    assert counts.to_dict() == {"negative": 20, "normal": 2400 - 20 - 24, "spike": 24}


def test_the_level_comes_from_the_actual_price_never_from_the_prediction(prices):
    """`b` predicts 0 where the price is a spike and 1,000 where it is normal: nothing moves between groups."""
    values = prices.reshape(-1)
    threshold = 205.01
    errors = np.where(values >= threshold, -values, np.where(values >= 0, 1000.0 - values, 0.0)).reshape(prices.shape)
    flat = {"a": priced_records(prices, np.ones_like(prices)), "b": priced_records(prices, errors)}

    table = EvaluationReport(flat, [MAEEvaluator()]).by_price_level().set_index(["price_level", "model"])

    assert table.loc[("spike", "b"), "forecasts"] == 24
    assert table.loc[("spike", "b"), "MAE"] == pytest.approx(np.mean(np.arange(206, 230)))  # |0 - actual|
    assert table.loc[("normal", "b"), "forecasts"] == 2356
    assert table.loc[("normal", "a"), "MAE"] == 1.0
    assert table.loc[("negative", "b"), "forecasts"] == 20
    assert table.loc[("negative", "b"), "MAE"] == 0.0


def test_a_forecast_is_in_exactly_one_group(prices):
    errors = np.random.default_rng(0).normal(0, 3, prices.shape)
    records = priced_records(prices, errors)
    records[7]["actual"] = float("nan")  # no finite actual: no level
    records[9]["prediction"] = float("nan")  # no prediction: still a forecast, at its level
    report = EvaluationReport({"a": records}, [MAEEvaluator()])

    table = report.by_price_level().set_index("price_level")

    assert table["forecasts"].sum() == len(records) - 1


def test_the_threshold_is_one_number_for_all_models_from_all_their_actuals(prices):
    """`partial` forecasts only the first half of the days, where there are no spikes: its own 99th percentile would be
    far lower. The threshold comes from the actuals of both models, so `partial` is judged by the same line."""
    errors = np.ones_like(prices)
    full = priced_records(prices, errors)
    partial = [r for r in full if r["run_date"] < "2023-07-21"]
    report = EvaluationReport({"full": full, "partial": partial}, [MAEEvaluator(), rMAEEvaluator("full")])

    table = report.by_price_level()

    own = np.quantile([r["actual"] for r in partial], 0.99)
    assert own < 100
    assert table.attrs["spike_threshold"] == pytest.approx(205.01)
    assert not (table[(table["price_level"] == "spike")]["model"] == "partial").any()
    normal = table.set_index(["price_level", "model"]).loc["normal"]
    assert normal.loc["partial", "forecasts"] == len(partial) - sum(r["actual"] < 0 for r in partial)
    assert normal.loc["partial", "rMAE"] == 1.0


def test_the_threshold_counts_each_period_once_however_many_horizons_forecast_it():
    run_dates = pd.date_range("2023-01-01", periods=60, freq="D")
    results = {
        "a": forecast_records(1, run_dates, horizons=(1, 2, 3)),
        "b": forecast_records(2, run_dates, horizons=(1, 2)),
    }

    table = EvaluationReport(results, [MAEEvaluator()]).by_price_level()

    periods = pd.DataFrame([r for rs in results.values() for r in rs]).drop_duplicates(["target_date", "hour"])
    assert table.attrs["spike_quantile"] == np.quantile(periods["actual"], 0.99)
    assert table.attrs["spike_threshold"] == min(300.0, np.quantile(periods["actual"], 0.99))


def test_the_threshold_is_capped_at_300(prices):
    high = prices.copy()
    high.reshape(-1)[[1200 + i * 40 + 3 for i in range(30)]] = 700.0 + np.arange(30)
    report = EvaluationReport({"a": priced_records(high, np.ones_like(high))}, [MAEEvaluator()])

    table = report.by_price_level()

    assert table.attrs["spike_threshold"] == 300.0
    assert table.attrs["spike_quantile"] > 700
    assert table.set_index("price_level")["forecasts"]["spike"] == 30


def test_a_given_spike_threshold_replaces_the_rule(prices):
    report = EvaluationReport({"a": priced_records(prices, np.ones_like(prices))}, [MAEEvaluator()])

    table = report.by_price_level(spike=215.0)

    assert table.attrs["spike_threshold"] == 215.0
    assert table.attrs["spike_rule"] == "given: 215"
    assert np.isnan(table.attrs["spike_quantile"])
    assert table.set_index("price_level")["forecasts"].to_dict() == {
        "negative": 20,
        "normal": 2400 - 20 - 15,
        "spike": 15,
    }


@pytest.mark.parametrize("spike", [0, -5, float("nan"), float("inf")])
def test_a_spike_threshold_must_be_a_positive_price(prices, spike):
    report = EvaluationReport({"a": priced_records(prices, np.ones_like(prices))}, [MAEEvaluator()])

    with pytest.raises(ValueError, match="positive price"):
        report.by_price_level(spike=spike)


def test_negative_prices_are_never_spikes_and_levels_cover_the_range(prices):
    report = EvaluationReport({"a": priced_records(prices, np.ones_like(prices))}, [MAEEvaluator()])

    table = report.by_price_level(spike=1.0)  # everything non-negative at or above 1 is a spike

    counts = table.set_index("price_level")["forecasts"].to_dict()
    assert counts == {"negative": 20, "spike": 2380}


def test_price_levels_compare_models_on_their_common_keys_inside_each_group(prices):
    """`holes` lacks hours the base has in the spike group: its rMAE there is on the spikes both have."""
    rng = np.random.default_rng(3)
    base = priced_records(prices, rng.normal(0, 3, prices.shape))
    other = priced_records(prices, rng.normal(0, 2, prices.shape))
    spikes = sorted(r["actual"] for r in base if r["actual"] > 205.01)
    drop_actual = set(spikes[:10])  # the ten lowest spikes
    other = [r for r in other if r["actual"] not in drop_actual]
    report = EvaluationReport({"base": base, "other": other}, [rMAEEvaluator("base"), DMEvaluator("base", reps=50)])

    row = report.by_price_level().set_index(["price_level", "model"]).loc[("spike", "other")]

    merged = pd.DataFrame(base).merge(pd.DataFrame(other), on=list(FORECAST_KEYS), suffixes=("_b", "_o"))
    merged = merged[merged["actual_b"] > 205.01]
    errors_o = (merged["prediction_o"] - merged["actual_o"]).abs()
    expected = errors_o.mean() / (merged["prediction_b"] - merged["actual_b"]).abs().mean()
    assert len(merged) == 14
    assert row["rMAE"] == pytest.approx(expected, rel=1e-12)
    assert row["DM n_forecasts"] == 14
    assert row["forecasts"] == 14
