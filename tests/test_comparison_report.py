"""`EvaluationReport.tests()`, `mcs()`, `ranks()`, `compare()` and `leaderboard()`: what each comparison runs on.

The expected numbers come from merging the records on their keys with pandas (`tests/forecast_records.py`) and calling
the statistics kernels by hand, so a misalignment inside the report cannot go unnoticed.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.evaluators import DMEvaluator, MAEEvaluator, stats
from epftoolbox2.results import ModelResultRef
from epftoolbox2.results.report import EvaluationReport
from tests.forecast_records import KEYS, daily_losses, daily_sums, forecast_records, shared_errors

pytestmark = pytest.mark.filterwarnings("ignore:The comparison rests on")

YEAR = pd.date_range("2023-01-01", periods=365, freq="D")
SHORT = pd.date_range("2024-01-01", periods=30, freq="D")


def drop_days(*days):
    return lambda r: r["run_date"] in days


def study(run_dates=YEAR, **kwargs):
    return {
        "base": forecast_records(1, run_dates, noise=6.0, **kwargs),
        "better": forecast_records(2, run_dates, noise=4.0, **kwargs),
        "same": forecast_records(3, run_dates, noise=6.0, **kwargs),
    }


@pytest.fixture(scope="module")
def full_year():
    return EvaluationReport(study(), [])


@pytest.fixture(scope="module")
def planted():
    """`holes` lacks 40 days and one hour of the remaining days; `whole` has everything."""

    def drop(r):
        return r["day_in_test"] < 40 or r["hour"] == 7

    results = {
        "base": forecast_records(1, YEAR, noise=6.0),
        "whole": forecast_records(2, YEAR, noise=5.0),
        "holes": forecast_records(3, YEAR, noise=4.0, drop=drop),
    }
    return results, EvaluationReport(results, [])


# ---------------------------------------------------------------------------------------------------------------------
# Common keys
# ---------------------------------------------------------------------------------------------------------------------


def test_each_pair_uses_its_own_common_keys(planted):
    results, report = planted

    table = report.tests(benchmark="base", reps=200).set_index("model")

    assert table.loc["whole", "n_days"] == 365
    assert table.loc["whole", "n_forecasts"] == 365 * 24
    assert table.loc["holes", "n_days"] == 325
    assert table.loc["holes", "n_forecasts"] == 325 * 23


@pytest.mark.parametrize("model", ["holes", "whole"])
def test_a_pair_scores_exactly_what_a_merge_on_the_keys_gives(planted, model):
    results, report = planted
    errors = shared_errors(results[model], results["base"])
    daily, sums = daily_losses(errors), daily_sums(errors)

    row = report.tests(benchmark="base", reps=200).set_index("model").loc[model]

    expected = stats.dm_test(daily["a"], daily["b"])
    interval = stats.ratio_interval(sums["a"], sums["b"], reps=200)
    assert row["mae"] == pytest.approx(errors["a"].mean(), rel=1e-12)
    assert row["mae_benchmark"] == pytest.approx(errors["b"].mean(), rel=1e-12)
    assert row["ratio"] == pytest.approx(errors["a"].mean() / errors["b"].mean(), rel=1e-12)
    assert row["statistic"] == pytest.approx(expected.statistic, rel=1e-10)
    assert row["p_value"] == pytest.approx(expected.p_value, rel=1e-9)
    assert (row["ci_low"], row["ci_high"]) == pytest.approx((interval.low, interval.high), rel=1e-12)
    assert row["mde"] == pytest.approx(stats.minimum_detectable_effect(daily["a"], daily["b"]), rel=1e-10)


def test_a_pair_is_not_cut_to_a_global_intersection(planted):
    """The pair without holes gets the same numbers with or without the model that has them."""
    results, report = planted
    without = EvaluationReport({k: v for k, v in results.items() if k != "holes"}, [])

    full = report.tests(benchmark="base", reps=200).set_index("model").loc["whole"]
    alone = without.tests(benchmark="base", reps=200).set_index("model").loc["whole"]

    for column in ("n_days", "n_forecasts", "mae", "ratio", "ci_low", "ci_high", "statistic", "p_value", "mde"):
        assert full[column] == alone[column], column


def test_non_finite_forecasts_leave_the_pair_only():
    base = forecast_records(1, SHORT)
    model = forecast_records(2, SHORT)
    other = forecast_records(3, SHORT)
    for record in model[:5]:
        record["prediction"] = float("nan")
    model[5]["prediction"] = float("inf")
    base[6]["actual"] = None
    base[6]["prediction"] = None

    table = EvaluationReport({"base": base, "model": model, "other": other}, []).tests(reps=50, benchmark="base")
    table = table.set_index("model")

    assert table.loc["model", "n_forecasts"] == 30 * 24 - 7
    assert table.loc["other", "n_forecasts"] == 30 * 24 - 1


def test_record_order_and_rows_do_not_matter(planted):
    results, report = planted
    shuffled = {k: list(np.random.default_rng(5).permutation(np.array(v, dtype=object))) for k, v in results.items()}

    other = EvaluationReport(shuffled, [])

    exact = dict(check_exact=True)
    pd.testing.assert_frame_equal(
        report.tests(benchmark="base", reps=100), other.tests(benchmark="base", reps=100), **exact
    )
    pd.testing.assert_frame_equal(report.mcs(reps=100), other.mcs(reps=100), **exact)
    pd.testing.assert_frame_equal(report.ranks(reps=100), other.ranks(reps=100), **exact)
    pd.testing.assert_frame_equal(report.compare("whole", "holes"), other.compare("whole", "holes"), **exact)


def test_a_duplicated_key_raises_instead_of_counting_twice():
    base = forecast_records(1, SHORT)
    model = forecast_records(2, SHORT)
    model.append(dict(model[0]))
    report = EvaluationReport({"base": base, "model": model}, [])
    with pytest.raises(ValueError, match=r"'model' has duplicated keys"):
        report.tests(benchmark="base", reps=50)


def test_records_without_keys_raise():
    records = [{"prediction": 1.0, "actual": 2.0, "hour": 0, "horizon": 1}]
    report = EvaluationReport({"base": records, "model": records}, [])
    with pytest.raises(ValueError, match="run_date"):
        report.tests(benchmark="base", reps=50)


def test_models_scored_on_different_actuals_warn():
    base = forecast_records(1, SHORT)
    model = forecast_records(2, SHORT)
    for record in model[:3]:
        record["actual"] += 10.0
    report = EvaluationReport({"base": base, "model": model}, [])
    with pytest.warns(UserWarning, match="3 forecasts have different actual prices"):
        report.compare("model", "base")


def test_a_pair_without_shared_keys_is_nan_with_n_zero():
    base = forecast_records(1, SHORT[:15])
    model = forecast_records(2, SHORT[15:])
    row = EvaluationReport({"base": base, "model": model}, []).tests(benchmark="base", reps=50).iloc[0]
    assert (row["n_days"], row["n_forecasts"], row["m"]) == (0, 0, 0)
    assert np.isnan(row["ratio"]) and np.isnan(row["p_value"]) and row["verdict"] is None


# ---------------------------------------------------------------------------------------------------------------------
# The table of tests()
# ---------------------------------------------------------------------------------------------------------------------


def test_columns_and_sign(full_year):
    table = full_year.tests(benchmark="base", reps=200)

    assert list(table.columns) == [
        "model",
        "benchmark",
        "horizon",
        "n_days",
        "n_forecasts",
        "mae",
        "mae_benchmark",
        "ratio",
        "ci_low",
        "ci_high",
        "statistic",
        "p_value",
        "p_holm",
        "m",
        "mde",
        "verdict",
    ]
    row = table.set_index("model").loc["better"]
    assert row["ratio"] < 1 and row["ci_high"] < 1 and row["statistic"] > 0 and row["verdict"] == "better"
    worse = full_year.tests(benchmark="better", reps=200).set_index("model").loc["base"]
    assert worse["ratio"] > 1 and worse["statistic"] < 0 and worse["verdict"] == "worse"
    assert full_year.tests(benchmark="base", reps=200).set_index("model").loc["same", "verdict"] == "no difference"


def test_holm_runs_over_the_m_models_and_m_is_printed(full_year):
    table = full_year.tests(benchmark="base", reps=50)

    assert (table["m"] == 2).all()
    np.testing.assert_allclose(table["p_holm"], stats.holm(table["p_value"].to_numpy()), rtol=1e-12)
    assert table.attrs["correction"] == "holm" and table.attrs["benchmark"] == "base"


def test_without_correction_the_verdict_reads_the_raw_p_value(full_year):
    table = full_year.tests(benchmark="base", correction=None, reps=50)
    assert table["p_holm"].isna().all()
    assert (table["verdict"] == np.where(table["p_value"] < 0.05, "better", "no difference")).all()


def test_variance_plain_is_the_plain_test(full_year):
    table = full_year.tests(benchmark="base", variance="plain", interval=None).set_index("model")
    results = EvaluationReport(study(), [])._model_frames()
    a = (results["better"]["prediction"] - results["better"]["actual"]).abs().to_numpy().reshape(-1, 24).mean(axis=1)
    b = (results["base"]["prediction"] - results["base"]["actual"]).abs().to_numpy().reshape(-1, 24).mean(axis=1)
    assert table.loc["better", "p_value"] == pytest.approx(stats.dm_test(a, b, variance="plain").p_value, rel=1e-9)
    assert table["ci_low"].isna().all()


def test_the_benchmark_comes_from_the_dm_evaluator_or_it_is_an_error():
    results = study(SHORT)
    table = EvaluationReport(results, [DMEvaluator(benchmark="same")]).tests(reps=50)
    assert set(table["benchmark"]) == {"same"}

    with pytest.raises(ValueError, match="benchmark"):
        EvaluationReport(results, [MAEEvaluator()]).tests()
    with pytest.raises(ValueError, match="Unknown benchmark 'nope'"):
        EvaluationReport(results, []).tests(benchmark="nope")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"variance": "robust"}, "variance"),
        ({"correction": "bonferroni"}, "correction"),
        ({"by": "day"}, "by"),
        ({"variance": "plain", "lag": 3}, "lag"),
        ({"horizon": 9}, "horizon"),
        ({"horizon": "some"}, "horizon"),
    ],
)
def test_bad_arguments_raise(full_year, kwargs, message):
    with pytest.raises(ValueError, match=message):
        full_year.tests(benchmark="base", **kwargs)


def test_a_lag_longer_than_the_sample_gives_nan_not_a_crash():
    row = EvaluationReport(study(SHORT[:5]), []).tests(benchmark="base", lag=7, reps=50).iloc[0]
    assert row["n_days"] == 5 and np.isnan(row["p_value"]) and np.isfinite(row["ratio"])


# ---------------------------------------------------------------------------------------------------------------------
# Several horizons
# ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_horizons():
    drop = lambda r: r["horizon"] == 2 and r["day_in_test"] < 20  # noqa: E731
    results = {
        "base": forecast_records(1, YEAR, noise=6.0, horizons=(1, 2)),
        "model": forecast_records(2, YEAR, noise=5.0, horizons=(1, 2), drop=drop),
        "other": forecast_records(3, YEAR, noise=5.0, horizons=(1, 2)),
    }
    return results, EvaluationReport(results, [])


def test_one_set_of_rows_per_horizon(two_horizons):
    results, report = two_horizons

    table = report.tests(benchmark="base", reps=100)

    assert list(table["horizon"]) == [1, 1, 2, 2]
    assert list(table["n_days"]) == [365, 365, 345, 365]
    errors = shared_errors(results["model"], results["base"], horizon=2)
    expected = stats.dm_test(daily_losses(errors)["a"], daily_losses(errors)["b"])
    row = table[(table["horizon"] == 2) & (table["model"] == "model")].iloc[0]
    assert row["statistic"] == pytest.approx(expected.statistic, rel=1e-10)
    # Holm runs within a horizon: each of the two horizons has its own family of m models.
    for _, part in table.groupby("horizon"):
        np.testing.assert_allclose(part["p_holm"], stats.holm(part["p_value"].to_numpy()), rtol=1e-12)
        assert (part["m"] == 2).all()


def test_horizon_all_averages_the_daily_losses_per_run_date(two_horizons):
    results, report = two_horizons
    errors = shared_errors(results["model"], results["base"])
    daily, sums = daily_losses(errors), daily_sums(errors)

    row = report.tests(benchmark="base", horizon="all", reps=100).set_index("model").loc["model"]

    expected = stats.dm_test(daily["a"], daily["b"])
    interval = stats.ratio_interval(sums["a"], sums["b"], reps=100)
    assert row["horizon"] == "all"
    assert (row["n_days"], row["n_forecasts"]) == (365, 365 * 24 + 345 * 24)
    assert row["statistic"] == pytest.approx(expected.statistic, rel=1e-10)
    assert row["ratio"] == pytest.approx(errors["a"].mean() / errors["b"].mean(), rel=1e-12)
    assert (row["ci_low"], row["ci_high"]) == pytest.approx((interval.low, interval.high), rel=1e-12)


def test_one_named_horizon(two_horizons):
    _, report = two_horizons
    table = report.tests(benchmark="base", horizon=2, reps=50)
    assert set(table["horizon"]) == {2}
    pd.testing.assert_frame_equal(
        table.reset_index(drop=True),
        report.tests(benchmark="base", reps=50).query("horizon == 2").reset_index(drop=True),
    )


# ---------------------------------------------------------------------------------------------------------------------
# DST days and 15-minute forecasts
# ---------------------------------------------------------------------------------------------------------------------


def dst_periods(short: int, normal: int, long: int):
    def periods(day: pd.Timestamp) -> int:
        return {"2024-03-31": short, "2024-10-27": long}.get(day.date().isoformat(), normal)

    return periods


@pytest.mark.parametrize("lengths", [(23, 24, 25), (92, 96, 100)], ids=["hourly", "quarter-hourly"])
def test_dst_days_average_over_the_periods_that_exist(lengths):
    run_dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    periods = dst_periods(*lengths)
    results = {
        "base": forecast_records(1, run_dates, noise=6.0, periods=periods),
        "model": forecast_records(2, run_dates, noise=5.0, periods=periods),
    }
    errors = shared_errors(results["model"], results["base"])
    daily, sums = daily_losses(errors), daily_sums(errors)

    row = EvaluationReport(results, []).tests(benchmark="base", reps=200).iloc[0]

    assert row["n_days"] == 366
    assert row["n_forecasts"] == len(errors)
    assert errors.groupby("target_date").size().loc["2024-03-31"] == lengths[0]
    assert errors.groupby("target_date").size().loc["2024-10-27"] == lengths[2]
    # The DM takes the daily mean, the ratio the ratio of sums: forecast-weighted, also on the long and short days.
    assert row["statistic"] == pytest.approx(stats.dm_test(daily["a"], daily["b"]).statistic, rel=1e-10)
    assert row["ratio"] == pytest.approx(errors["a"].mean() / errors["b"].mean(), rel=1e-12)
    interval = stats.ratio_interval(sums["a"], sums["b"], reps=200)
    assert (row["ci_low"], row["ci_high"]) == pytest.approx((interval.low, interval.high), rel=1e-12)
    assert interval.ratio == pytest.approx(row["ratio"], rel=1e-12)


# ---------------------------------------------------------------------------------------------------------------------
# Per hour
# ---------------------------------------------------------------------------------------------------------------------


def test_per_hour_is_descriptive(full_year):
    table = full_year.tests(benchmark="base", by="hour", reps=50)

    assert "p_holm" not in table.columns and "verdict" not in table.columns and "m" not in table.columns
    assert "heatmap" in table.attrs["descriptive"]
    assert len(table) == 2 * 24
    assert "descriptive" not in full_year.tests(benchmark="base", reps=50).attrs


def test_per_hour_numbers_are_those_of_that_hour(full_year):
    frames = full_year._model_frames()
    a = (frames["better"]["prediction"] - frames["better"]["actual"]).abs().to_numpy().reshape(-1, 24)
    b = (frames["base"]["prediction"] - frames["base"]["actual"]).abs().to_numpy().reshape(-1, 24)

    table = full_year.tests(benchmark="base", by="hour", reps=50, interval=None).query("model == 'better'")

    for hour in (0, 13, 23):
        row = table[table["hour"] == hour].iloc[0]
        expected = stats.dm_test(a[:, hour], b[:, hour])
        assert row["statistic"] == pytest.approx(expected.statistic, rel=1e-10)
        assert (row["n_days"], row["n_forecasts"]) == (365, 365)


def test_the_switch_hour_missing_on_one_day_does_not_note_a_short_year():
    drop = lambda r: r["hour"] == 2 and r["day_in_test"] == 80  # noqa: E731
    results = {"base": forecast_records(1, YEAR, drop=drop), "model": forecast_records(2, YEAR, drop=drop)}
    table = EvaluationReport(results, []).tests(benchmark="base", by="hour", interval=None)
    assert table[table["hour"] == 2]["n_days"].iloc[0] == 364
    assert "note" not in table.attrs


# ---------------------------------------------------------------------------------------------------------------------
# Warnings and notes on the test length; seeds
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    [
        lambda r: r.tests(benchmark="base", reps=50),
        lambda r: r.mcs(reps=50),
        lambda r: r.ranks(reps=50),
        lambda r: r.compare("better", "base"),
        lambda r: r.leaderboard(benchmark="base"),
    ],
    ids=["tests", "mcs", "ranks", "compare", "leaderboard"],
)
def test_test_length_warnings(method):
    with pytest.warns(UserWarning, match="181 days"):
        short = method(EvaluationReport(study(YEAR[:181]), []))
    assert "181 days" in short.attrs["note"]

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        half = method(EvaluationReport(study(YEAR[:182]), []))
        whole = method(EvaluationReport(study(YEAR), []))
    assert "182 days" in half.attrs["note"]
    assert "note" not in whole.attrs


def test_same_seed_same_numbers_other_seed_other_interval(full_year):
    first = full_year.tests(benchmark="base", reps=200, seed=1)
    again = EvaluationReport(study(), []).tests(benchmark="base", reps=200, seed=1)
    other = full_year.tests(benchmark="base", reps=200, seed=2)
    pd.testing.assert_frame_equal(first, again, check_exact=True)
    assert (first["ci_low"] != other["ci_low"]).all()
    assert (first["statistic"] == other["statistic"]).all()
    pd.testing.assert_frame_equal(full_year.mcs(reps=200, seed=1), full_year.mcs(reps=200, seed=1), check_exact=True)
    pd.testing.assert_frame_equal(
        full_year.ranks(reps=200, seed=1), full_year.ranks(reps=200, seed=1), check_exact=True
    )


# ---------------------------------------------------------------------------------------------------------------------
# mcs()
# ---------------------------------------------------------------------------------------------------------------------


def set_losses(results: dict, models: list[str]) -> tuple:
    """Daily losses (days x models) on the keys all `models` hold, from a merge on the keys."""
    merged = None
    for name in models:
        frame = pd.DataFrame(results[name])
        frame[name] = (frame["prediction"] - frame["actual"]).abs()
        part = frame[[*KEYS, name]]
        merged = part if merged is None else merged.merge(part, on=KEYS)
    daily = merged.groupby(["run_date", "horizon"])[models].mean().groupby("run_date").mean()
    return daily.to_numpy(), merged


def test_mcs_on_the_keys_of_the_set(planted):
    results, report = planted
    losses, merged = set_losses(results, ["base", "whole", "holes"])

    table = report.mcs(reps=500, seed=4).set_index("model")

    expected = stats.model_confidence_set(losses, reps=500, seed=4)
    for j, name in enumerate(["base", "whole", "holes"]):
        assert table.loc[name, "p_value"] == pytest.approx(expected.p_values[j], rel=1e-12)
        assert table.loc[name, "in_90"] == (expected.p_values[j] > 0.10)
        assert table.loc[name, "in_75"] == (expected.p_values[j] > 0.25)
    assert (table["n_days"] == 325).all() and (table["n_forecasts"] == 325 * 23).all()
    maes = merged[["base", "whole", "holes"]].mean()
    assert table.loc["holes", "mae"] == pytest.approx(maes["holes"], rel=1e-12)
    assert table["ratio_to_best"].min() == 1.0
    assert table.loc["base", "ratio_to_best"] == pytest.approx(maes["base"] / maes.min(), rel=1e-12)


def test_mcs_attrs_name_the_model_that_limited_the_keys(planted):
    _, report = planted

    table = report.mcs(reps=100)

    assert table.attrs["n_days"] == 325
    assert table.attrs["n_forecasts"] == 325 * 23
    assert table.attrs["limited_by"] == {"holes": 365 * 24 - 325 * 23}


def test_mcs_on_a_subset_of_models_keeps_their_keys(planted):
    _, report = planted
    with pytest.warns(UserWarning, match="at least 3"):
        table = report.mcs(models=["base", "whole"], reps=100)
    assert table.attrs["n_days"] == 365 and table.attrs["limited_by"] == {}
    assert table["p_value"].isna().all() and table["in_90"].isna().all()


def test_mcs_of_a_full_set_has_no_limit_and_rows_by_mae(full_year):
    table = full_year.mcs(reps=100)
    assert table.attrs["limited_by"] == {}
    assert list(table["mae"]) == sorted(table["mae"])
    assert list(table.columns) == [
        "model",
        "n_days",
        "n_forecasts",
        "mae",
        "ratio_to_best",
        "p_value",
        "in_90",
        "in_75",
    ]


def test_mcs_with_another_alpha_adds_its_set(full_year):
    table = full_year.mcs(alpha=0.05, reps=100)
    assert (table["in_95"] == (table["p_value"] > 0.05)).all()


def test_mcs_of_identical_models_raises_naming_them():
    records = forecast_records(1, SHORT)
    results = {"a": records, "b": forecast_records(2, SHORT), "copy": records}
    with pytest.raises(ValueError, match="a, b, copy"):
        EvaluationReport(results, []).mcs(reps=100)


def test_mcs_needs_two_common_days():
    results = {
        "a": forecast_records(1, SHORT[:15]),
        "b": forecast_records(2, SHORT[15:]),
        "c": forecast_records(3, SHORT),
    }
    with pytest.warns(UserWarning, match="share only 0"):
        table = EvaluationReport(results, []).mcs(reps=100)
    assert table["p_value"].isna().all() and table.attrs["n_days"] == 0


def test_mcs_for_one_horizon(two_horizons):
    results, report = two_horizons
    table = report.mcs(reps=100, horizon=2)
    assert table.attrs["n_days"] == 345 and table.attrs["horizon"] == 2
    assert report.mcs(reps=100).attrs["horizon"] == "all"


def test_unknown_or_repeated_models_raise(full_year):
    with pytest.raises(ValueError, match="nope"):
        full_year.mcs(models=["base", "nope", "same"])
    with pytest.raises(ValueError, match="twice"):
        full_year.mcs(models=["base", "base", "same"])
    with pytest.raises(ValueError, match="nope"):
        full_year.compare("base", "nope")


# ---------------------------------------------------------------------------------------------------------------------
# ranks()
# ---------------------------------------------------------------------------------------------------------------------


def test_ranks_on_the_keys_of_the_set(planted):
    results, report = planted
    losses, _ = set_losses(results, ["base", "whole", "holes"])

    table = report.ranks(reps=300, seed=2).set_index("model")

    expected = stats.rank_bootstrap(losses, reps=300, seed=2)
    assert list(table.columns) == ["n_days", "n_forecasts", "mae", "rank", "p_best", "rank_low", "rank_high"]
    for j, name in enumerate(["base", "whole", "holes"]):
        assert table.loc[name, "rank"] == expected.rank[j]
        assert table.loc[name, "p_best"] == expected.p_best[j]
        assert table.loc[name, "rank_low"] == expected.rank_low[j]
        assert table.loc[name, "rank_high"] == expected.rank_high[j]
    assert table.attrs["limited_by"] == {"holes": 365 * 24 - 325 * 23}
    assert table.attrs["n_days"] == 325
    assert list(table["rank"]) == [1, 2, 3]


def test_ranks_without_the_limiting_model(planted):
    _, report = planted
    table = report.ranks(reps=100, models=["base", "whole"])
    assert table.attrs["n_days"] == 365 and table.attrs["limited_by"] == {}


# ---------------------------------------------------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------------------------------------------------


def test_compare_is_one_row_on_the_pairs_keys(planted):
    results, report = planted
    errors = shared_errors(results["holes"], results["whole"])
    daily = daily_losses(errors)

    table = report.compare("holes", "whole")

    assert len(table) == 1
    row = table.iloc[0]
    assert (row["a"], row["b"], row["horizon"]) == ("holes", "whole", 1)
    assert (row["n_days"], row["n_forecasts"]) == (325, 325 * 23)
    assert row["mae_a"] == pytest.approx(errors["a"].mean(), rel=1e-12)
    assert row["mae_b"] == pytest.approx(errors["b"].mean(), rel=1e-12)
    assert row["ratio"] == pytest.approx(errors["a"].mean() / errors["b"].mean(), rel=1e-12)
    assert row["p_plain"] == pytest.approx(stats.dm_test(daily["a"], daily["b"], variance="plain").p_value, rel=1e-9)
    assert row["p_hac"] == pytest.approx(stats.dm_test(daily["a"], daily["b"]).p_value, rel=1e-9)
    assert row["mde"] == pytest.approx(stats.minimum_detectable_effect(daily["a"], daily["b"]), rel=1e-10)
    assert "mae_a / mae_b" in table.attrs["ratio"]
    assert "descriptive" in table.attrs["descriptive"]


def test_compare_counts_the_hours_with_holm_over_hours(planted):
    results, report = planted
    errors = shared_errors(results["holes"], results["base"])

    row = report.compare("holes", "base").iloc[0]

    statistics, p_values = [], []
    for _, part in errors.groupby("hour"):
        found = stats.dm_test(part["a"], part["b"], variance="plain")
        statistics.append(found.statistic)
        p_values.append(found.p_value)
    rejected = stats.holm(p_values) < 0.05
    assert row["hours_tested"] == 23
    assert row["hours_a_better"] == (rejected & (np.array(statistics) > 0)).sum()
    assert row["hours_b_better"] == (rejected & (np.array(statistics) < 0)).sum()
    assert row["hours_a_better"] > 0


def test_compare_is_symmetric_in_its_sign(full_year):
    ab = full_year.compare("better", "base").iloc[0]
    ba = full_year.compare("base", "better").iloc[0]
    assert ab["statistic"] == pytest.approx(-ba["statistic"], rel=1e-12)
    assert ab["ratio"] == pytest.approx(1 / ba["ratio"], rel=1e-12)
    assert ab["p_hac"] == pytest.approx(ba["p_hac"], rel=1e-9)


# ---------------------------------------------------------------------------------------------------------------------
# leaderboard()
# ---------------------------------------------------------------------------------------------------------------------


def test_leaderboard_with_a_benchmark(planted):
    results, report = planted
    errors = shared_errors(results["holes"], results["base"])

    with pytest.warns(UserWarning, match="different forecast sets"):
        table = report.leaderboard(benchmark="base")

    assert list(table["model"]) == list(table.sort_values("MAE")["model"])
    row = table.set_index("model").loc["holes"]
    own = pd.DataFrame(results["holes"])
    assert row["n_forecasts"] == len(own)
    assert row["MAE"] == pytest.approx((own["prediction"] - own["actual"]).abs().mean(), rel=1e-12)
    assert (row["n_days"], row["n_common"]) == (325, len(errors))
    assert row["ratio"] == pytest.approx(errors["a"].mean() / errors["b"].mean(), rel=1e-12)
    assert row["ci_low"] < row["ratio"] < row["ci_high"]
    pair = report.tests(benchmark="base").set_index("model").loc["holes"]
    assert row["p_value"] == pair["p_value"] and row["p_holm"] == pair["p_holm"]
    reference = table.set_index("model").loc["base"]
    assert reference["ratio"] == 1.0 and reference["n_common"] == 365 * 24 and np.isnan(reference["p_value"])


def test_leaderboard_without_a_benchmark(full_year):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        table = full_year.leaderboard()
    assert list(table.columns) == ["model", "n_forecasts", "MAE", "n_days"]
    assert (table["n_days"] == 365).all() and list(table["MAE"]) == sorted(table["MAE"])


def test_leaderboard_takes_the_dm_evaluators_benchmark():
    report = EvaluationReport(study(), [DMEvaluator(benchmark="same", reps=50)])
    assert report.leaderboard().attrs["benchmark"] == "same"


# ---------------------------------------------------------------------------------------------------------------------
# Reading the results once
# ---------------------------------------------------------------------------------------------------------------------


def jsonl_report(tmp_path, results) -> EvaluationReport:
    refs = {}
    for name, records in results.items():
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        refs[name] = ModelResultRef(
            name=name,
            path=path,
            count=len(records),
            test_start=records[0]["run_date"],
            test_end="2100-01-01",
            horizon=1,
        )
    return EvaluationReport(refs, [])


def test_results_files_are_read_once_until_they_change(tmp_path, monkeypatch):
    results = study(SHORT)
    report = jsonl_report(tmp_path, results)
    reads = []
    original = EvaluationReport._model_frame
    monkeypatch.setattr(
        EvaluationReport, "_model_frame", lambda self, ref: (reads.append(ref.name), original(self, ref))[1]
    )

    first = report.tests(benchmark="base", reps=50)
    report.mcs(reps=50)
    report.summary()
    assert sorted(reads) == ["base", "better", "same"]

    extra = dict(results["base"][-1], run_date="2024-01-31", target_date="2024-02-01", day_in_test=30)
    with open(tmp_path / "base.jsonl", "a") as f:
        f.write(json.dumps(extra) + "\n")
    assert report.summary().set_index("model").loc["base", "forecasts"] == 30 * 24 + 1
    assert sorted(reads) == ["base", "base", "better", "same"]
    assert report.tests(benchmark="base", reps=50)["n_forecasts"].equals(first["n_forecasts"])


def test_a_table_cannot_change_what_the_next_table_reads(tmp_path):
    report = jsonl_report(tmp_path, study(SHORT))
    frames = report._model_frames()
    frames["base"]["extra"] = 1.0
    assert "extra" not in report._model_frames()["base"].columns
