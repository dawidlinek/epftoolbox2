"""`EvaluationReport.aggregate()`: forecasts of the maximum, minimum, mean or spread of a day, from hourly or 15-minute ones.

The daily values are checked against series with a known answer, including the days that clocks make 23 or 25 hours long
(92 and 100 quarter-hours). Which days count is the rule `EconomicEvaluator` uses, and a test pins that they agree.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.evaluators import (
    BiasEvaluator,
    DMEvaluator,
    EconomicEvaluator,
    MAEEvaluator,
    MCSEvaluator,
    RMSEEvaluator,
    rMAEEvaluator,
)
from epftoolbox2.results.report import EvaluationReport
from tests.forecast_records import forecast_records

pytestmark = [
    pytest.mark.filterwarnings("ignore:The comparison rests on"),
    pytest.mark.filterwarnings("ignore:The model confidence set needs"),
]

SKIPPED_HOUR = 2  # the hour the spring switch skips; the real run leaves its id out


def clock_ids(kind, per_hour=1):
    """The period ids a real run writes for a local day: by the clock, so a 23-hour day lacks the hour it skips."""
    if kind == "short":
        return [i for i in range(24 * per_hour) if not SKIPPED_HOUR * per_hour <= i < (SKIPPED_HOUR + 1) * per_hour]
    return list(range((25 if kind == "long" else 24) * per_hour))


# A Saturday, the spring switch (a Sunday: 23 hours), the Monday after, the autumn switch (a Sunday: 25 hours), a Monday.
DAYS = {
    "2023-03-25": "plain",
    "2023-03-26": "short",
    "2023-03-27": "plain",
    "2023-10-29": "long",
    "2023-10-30": "plain",
}


def day_records(target, ids, actual, prediction, horizon=1, freq=None):
    run = pd.Timestamp(target) - pd.Timedelta(days=horizon)
    records = []
    for period, a, p in zip(ids, actual, prediction):
        row = {
            "run_date": run.date().isoformat(),
            "target_date": pd.Timestamp(target).date().isoformat(),
            "run_weekday": int(run.dayofweek),
            "hour": int(period),
            "horizon": horizon,
            "day_in_test": 0,
            "prediction": float(p),
            "actual": float(a),
        }
        if freq:
            row["freq"] = freq
        records.append(row)
    return records


def hourly_price(day, period):
    """100 + 10 * day at the peak (period 10), falling 3 per period on either side."""
    return 100.0 + 10 * day - 3 * abs(period - 10)


def hourly_study(days=DAYS, skip=None):
    """One model's records. The prediction is 5 above the actual, except at the period of the actual's peak, where it is
    40 below: its maximum sits next to the peak, and its minimum is the dip at the peak."""
    records = []
    for k, (target, kind) in enumerate(days.items()):
        ids = clock_ids(kind)
        actual = [hourly_price(k, i) for i in ids]
        prediction = [a - 40 if i == 10 else a + 5 for i, a in zip(ids, actual)]
        records += day_records(target, ids, actual, prediction)
    return [r for r in records if not (skip and skip(r))]


def daily(report, model="m"):
    """`{target_date: (prediction, actual)}` of an aggregated report's model."""
    details = dict(report.iter_details())[model]
    return dict(zip(details["target_date"], zip(details["prediction"], details["actual"])))


def aggregate(records, agg="max"):
    return EvaluationReport({"m": records}, [MAEEvaluator()]).aggregate("1D", agg=agg)


# ---------------------------------------------------------------------------------------------------------------------
# The values
# ---------------------------------------------------------------------------------------------------------------------


def test_the_daily_max_of_a_known_series_including_23_and_25_hour_days():
    """Actual peak: 100 + 10 * k at period 10, for the k-th day. The prediction's own peak is the neighbours of period
    10, at 100 + 10 * k - 3 + 5."""
    found = daily(aggregate(hourly_study(), "max"))

    assert found == {target: (102.0 + 10 * k, 100.0 + 10 * k) for k, target in enumerate(DAYS)}


def test_the_daily_min_of_a_known_series():
    """The actual falls 3 per period away from period 10: its minimum is at the last period (23, or 24 on the 25-hour
    day). The prediction's minimum is the dip at period 10."""
    found = daily(aggregate(hourly_study(), "min"))

    last = {"plain": 23, "short": 23, "long": 24}
    assert found == {
        target: (60.0 + 10 * k, 100.0 + 10 * k - 3 * (last[kind] - 10)) for k, (target, kind) in enumerate(DAYS.items())
    }


@pytest.mark.parametrize("agg", ["max", "min", "mean", "spread"])
def test_each_aggregate_against_python_on_the_same_series(agg):
    records = hourly_study()
    reduce = {"max": max, "min": min, "mean": lambda v: sum(v) / len(v), "spread": lambda v: max(v) - min(v)}[agg]

    found = daily(aggregate(records, agg))

    for target in DAYS:
        day = [r for r in records if r["target_date"] == target]
        expected = (reduce([r["prediction"] for r in day]), reduce([r["actual"] for r in day]))
        assert found[target] == pytest.approx(expected, rel=1e-12)


def test_the_mean_of_a_23_and_a_25_hour_day_is_over_the_hours_that_exist():
    records = hourly_study()

    found = daily(aggregate(records, "mean"))

    short = [r["actual"] for r in records if r["target_date"] == "2023-03-26"]
    long = [r["actual"] for r in records if r["target_date"] == "2023-10-29"]
    assert (len(short), len(long)) == (23, 25)
    assert found["2023-03-26"][1] == pytest.approx(sum(short) / 23)
    assert found["2023-10-29"][1] == pytest.approx(sum(long) / 25)


def test_quarter_hour_days_of_92_96_and_100_periods():
    """Actual: 200 + 20 * k - |p - 40| at quarter-hour p. The prediction is the actual + 1 except at p = 40, where it is
    300 higher: its maximum is 300 + the peak."""
    records, expected = [], {}
    for k, (target, kind) in enumerate(DAYS.items()):
        ids = clock_ids(kind, per_hour=4)
        actual = [200.0 + 20 * k - abs(i - 40) for i in ids]
        prediction = [a + 300 if i == 40 else a + 1 for i, a in zip(ids, actual)]
        records += day_records(target, ids, actual, prediction, freq="15min")
        expected[target] = (500.0 + 20 * k, 200.0 + 20 * k)
        assert len(ids) == {"plain": 96, "short": 92, "long": 100}[kind]

    report = EvaluationReport({"m": records}, [MAEEvaluator()]).aggregate("1D", "max")

    assert daily(report) == expected
    assert report.forecast_counts()["missing"].tolist() == [0]


def test_the_new_report_is_daily_with_hour_0_and_the_horizon_kept():
    report = aggregate(hourly_study())

    details = dict(report.iter_details())["m"]
    assert set(details["hour"]) == {0}
    assert set(details["horizon"]) == {1}
    assert report._result_freqs() == {"m": "1D"}
    assert report.refs["m"].freq == "1D"
    assert (pd.to_datetime(details["target_date"]) - pd.to_datetime(details["run_date"])).dt.days.tolist() == [1] * 5
    first_run = pd.Timestamp("2023-03-24")
    assert details["day_in_test"].tolist() == [(pd.Timestamp(r) - first_run).days for r in details["run_date"]]
    assert details["day_in_test"].tolist()[:3] == [0, 1, 2]  # counted from the earliest run date


def test_each_horizon_of_a_target_day_is_its_own_forecast():
    """A target day forecast from two run dates has two aggregate forecasts, like the hourly ones had two."""
    records = []
    for horizon, shift in ((1, 0.0), (2, 10.0)):
        for k, (target, kind) in enumerate(list(DAYS.items())[:3]):
            ids = clock_ids(kind)
            actual = [hourly_price(k, i) for i in ids]
            records += day_records(target, ids, actual, [a + 1 + shift for a in actual], horizon=horizon)

    report = EvaluationReport({"m": records}, [MAEEvaluator()]).aggregate("1D", "max")

    details = dict(report.iter_details())["m"]
    assert len(details) == 6
    by_horizon = details.groupby("horizon")["prediction"].apply(list).to_dict()
    assert by_horizon == {1: [101.0, 111.0, 121.0], 2: [111.0, 121.0, 131.0]}
    assert report.by_horizon()["MAE"].tolist() == [1.0, 11.0]
    assert report.summary()["MAE"].tolist() == [6.0]


@pytest.mark.parametrize("agg", ["max", "min", "mean", "spread"])
def test_against_a_pandas_groupby_on_random_records(agg):
    run_dates = pd.date_range("2023-03-20", periods=12, freq="D")
    records = forecast_records(7, run_dates, noise=5.0, horizons=(1, 2))
    frame = pd.DataFrame(records)

    report = EvaluationReport({"m": records}, [MAEEvaluator()]).aggregate("1D", agg)

    grouped = frame.groupby(["run_date", "target_date", "horizon"])
    functions = {"max": "max", "min": "min", "mean": "mean", "spread": lambda s: s.max() - s.min()}
    expected = grouped[["prediction", "actual"]].agg(functions[agg])
    found = dict(report.iter_details())["m"]
    got = found.set_index(["run_date", "target_date", "horizon"])[["prediction", "actual"]]
    assert len(got) == len(expected) == 24
    pd.testing.assert_frame_equal(got.sort_index(), expected.sort_index(), rtol=1e-12, check_exact=False)


# ---------------------------------------------------------------------------------------------------------------------
# Days without all their periods
# ---------------------------------------------------------------------------------------------------------------------


def test_a_day_with_a_missing_period_is_dropped_counted_and_warned_about():
    records = hourly_study(skip=lambda r: r["target_date"] == "2023-03-27" and r["hour"] == 7)

    with pytest.warns(UserWarning, match=r"left out days.*\{'m': 1\}"):
        report = aggregate(records)

    assert "2023-03-27" not in daily(report)
    assert len(daily(report)) == 4
    counts = report.forecast_counts().iloc[0]
    assert (counts["planned"], counts["produced"], counts["missing"], counts["incomplete_day"]) == (5, 4, 1, 1)
    assert report.summary()["forecasts"].tolist() == [4]


def test_the_counts_say_zero_when_nothing_is_dropped():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = aggregate(hourly_study())

    counts = report.forecast_counts().iloc[0]
    assert (counts["planned"], counts["produced"], counts["missing"], counts["incomplete_day"]) == (5, 5, 0, 0)


def test_a_model_missing_a_period_does_not_lose_that_day_for_another_model():
    holes = hourly_study(skip=lambda r: r["target_date"] == "2023-03-27" and r["hour"] == 7)
    results = {"whole": hourly_study(), "holes": holes}

    with pytest.warns(UserWarning, match="left out days"):
        report = EvaluationReport(results, [MAEEvaluator(), rMAEEvaluator("whole")]).aggregate("1D", "max")

    assert "2023-03-27" in daily(report, "whole") and "2023-03-27" not in daily(report, "holes")
    assert report.summary()["forecasts"].tolist() == [5, 4]
    counts = report.forecast_counts().set_index("model")
    assert counts["missing"].to_dict() == {"whole": 0, "holes": 1}
    # A comparison uses the days both hold: holes' four, and says so.
    table = report.tests(benchmark="whole", reps=50).set_index("model")
    assert (table.loc["holes", "n_days"], table.loc["holes", "n_forecasts"]) == (4, 4)


def test_a_non_finite_prediction_takes_the_day_out():
    records = hourly_study()
    next(r for r in records if r["target_date"] == "2023-10-30" and r["hour"] == 4)["prediction"] = float("nan")

    with pytest.warns(UserWarning, match="left out days"):
        report = aggregate(records)

    assert sorted(daily(report)) == sorted(set(DAYS) - {"2023-10-30"})


def test_a_day_with_no_finite_forecast_at_all_is_counted_as_dropped():
    records = hourly_study()
    for r in records:
        if r["target_date"] == "2023-10-30":
            r["actual"] = float("nan")

    with pytest.warns(UserWarning, match="left out days"):
        report = aggregate(records)

    counts = report.forecast_counts().iloc[0]
    assert (counts["planned"], counts["produced"], counts["missing"]) == (5, 4, 1)


def test_a_day_that_lacks_a_period_outside_the_switch_hours_is_dropped_in_quarter_hours_too():
    records = []
    for k, (target, kind) in enumerate(list(DAYS.items())[:3]):
        ids = clock_ids(kind, per_hour=4)
        actual = [100.0 + i for i in ids]
        records += day_records(target, ids, actual, [a + 1 for a in actual], freq="15min")
    records = [r for r in records if not (r["target_date"] == "2023-03-27" and r["hour"] == 50)]

    with pytest.warns(UserWarning, match="left out days"):
        report = EvaluationReport({"m": records}, [MAEEvaluator()]).aggregate("1D", "max")

    assert sorted(daily(report)) == ["2023-03-25", "2023-03-26"]


ORDINARY = {"2023-01-09": "plain", "2023-01-10": "plain", "2023-01-11": "plain", "2023-01-12": "plain"}


def test_a_model_that_lacks_a_switch_hour_on_an_ordinary_day_is_not_aggregated_over_fewer_hours():
    """Without a clock change on 2023-01-10 the hour 02:00 exists. `holes` lacks it, so by its ids alone the day looks
    like a 23-hour one; but `whole` has all 24, and aggregating `holes` over 23 would score the two against different
    actuals on one key (a daily minimum can sit at 02:00)."""
    holes = hourly_study(ORDINARY, skip=lambda r: r["target_date"] == "2023-01-10" and r["hour"] == 2)
    results = {"whole": hourly_study(ORDINARY), "holes": holes}

    with pytest.warns(UserWarning, match=r"\{'holes': 1\}"):
        report = EvaluationReport({**results}, [MAEEvaluator()]).aggregate("1D", "min")

    assert sorted(daily(report, "holes")) == ["2023-01-09", "2023-01-11", "2023-01-12"]
    assert len(daily(report, "whole")) == 4
    assert report.forecast_counts().set_index("model")["incomplete_day"].to_dict() == {"whole": 0, "holes": 1}
    # Where both have the day they were scored against the same actual.
    for target, (_, actual) in daily(report, "holes").items():
        assert daily(report, "whole")[target][1] == actual


def test_the_day_after_the_spring_switch_is_not_aggregated_over_23_hours_when_another_horizon_has_all_24():
    """In a real run the horizon-1 forecasts of 2023-03-27 come from the switch Sunday, which has no 02:00 row, so they
    lack the id 2 of a day that has 24 hours. Horizon 2 forecasts all of them."""
    records = []
    for target in ("2023-03-27", "2023-03-28"):
        full = clock_ids("plain")
        actual = [hourly_price(0, i) for i in full]
        records += day_records(target, full, actual, [a + 1 for a in actual], horizon=2)
        ids = [i for i in full if not (i == 2 and target == "2023-03-27")]
        records += day_records(target, ids, [hourly_price(0, i) for i in ids], [hourly_price(0, i) + 1 for i in ids])

    with pytest.warns(UserWarning, match="left out days"):
        report = aggregate(records, "mean")

    details = dict(report.iter_details())["m"]
    assert sorted(zip(details["target_date"], details["horizon"])) == [
        ("2023-03-27", 2),
        ("2023-03-28", 1),
        ("2023-03-28", 2),
    ]
    # One actual per target date, whatever the horizon.
    assert details.groupby("target_date")["actual"].nunique().tolist() == [1, 1]
    counts = report.forecast_counts().iloc[0]
    assert (counts["planned"], counts["produced"], counts["missing"]) == (4, 3, 1)


def test_a_period_that_no_model_has_at_any_horizon_is_taken_at_its_word():
    """That is how a clock change looks in the ids, and the ids cannot tell it from the day after the spring switch in a
    run of horizon 1 alone. Documented: the day is aggregated over the periods it has."""
    plain = clock_ids("plain")
    ids = [i for i in plain if i != 2]
    records = day_records("2023-03-27", ids, [hourly_price(0, i) for i in ids], [hourly_price(0, i) for i in ids])

    report = aggregate(records, "mean")

    assert len(daily(report)) == 1
    assert report.forecast_counts().iloc[0]["missing"] == 0


def test_a_period_with_a_record_but_no_actual_takes_the_day_out_for_every_model():
    """The autumn switch day of a real run: the ambiguous 02:00 has a record and no actual. The day is not the aggregate of
    all its hours, whichever model it is."""
    results = {}
    for name in ("a", "b"):
        records = hourly_study(ORDINARY)
        for r in records:
            if r["target_date"] == "2023-01-11" and r["hour"] == 2:
                r["actual"] = float("nan")
        results[name] = records

    with pytest.warns(UserWarning, match=r"\{'a': 1, 'b': 1\}"):
        report = EvaluationReport(results, [MAEEvaluator()]).aggregate("1D", "max")

    assert "2023-01-11" not in daily(report, "a") and "2023-01-11" not in daily(report, "b")
    assert len(daily(report, "a")) == 3


def whole_day_cases():
    """(target date, ids, whether a schedule or a daily aggregate can use the day). Sundays carry the clock changes."""
    plain, short, long = clock_ids("plain"), clock_ids("short"), clock_ids("long")
    return [
        ("2024-01-02", plain, True),  # a Tuesday
        ("2024-03-31", short, True),  # the Sunday of the spring switch, ids by the clock
        ("2024-03-30", short, True),  # the same ids on a Saturday: the clock numbering is fine on any day
        ("2024-10-27", long, True),  # the Sunday of the autumn switch
        ("2024-01-03", plain[:-1], False),  # the last hour missing
        ("2024-01-04", [i for i in plain if i != 12], False),
        ("2024-01-05", list(range(23)), False),  # numbered through on a Friday: a 24-hour day without its last hour
        ("2024-01-07", list(range(23)), True),  # numbered through on a Sunday: a 23-hour day
        ("2024-01-08", list(range(25)), True),
        ("2024-01-09", plain[:22], False),
    ]


def test_aggregate_and_the_battery_judge_whole_days_the_same_way():
    """`profit days` counts the target days a schedule can be made for; a daily aggregate keeps exactly those."""
    records = []
    cases = whole_day_cases()
    for target, ids, _ in cases:
        records += day_records(target, ids, [50.0 + i for i in ids], [51.0 + i for i in ids])

    battery = EvaluationReport({"m": records}, [EconomicEvaluator()]).summary()["profit days"].iloc[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kept = daily(aggregate(records))

    assert battery == len(kept)
    assert sorted(kept) == sorted(target for target, _, whole in cases if whole)


# ---------------------------------------------------------------------------------------------------------------------
# What the new report does
# ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def year():
    run_dates = pd.date_range("2023-01-01", periods=120, freq="D")
    results = {
        "base": forecast_records(1, run_dates, noise=6.0),
        "better": forecast_records(2, run_dates, noise=4.0),
        "same": forecast_records(3, run_dates, noise=6.0),
    }
    evaluators = [MAEEvaluator(), RMSEEvaluator(), BiasEvaluator(), rMAEEvaluator("base"), DMEvaluator("base", reps=50)]
    return results, EvaluationReport(results, evaluators + [MCSEvaluator(reps=50)])


NO_PARTIAL_DAYS = pytest.mark.filterwarnings("error:Days with only part", "ignore:The comparison rests on")


@NO_PARTIAL_DAYS
def test_the_comparison_methods_work_on_a_daily_report(year):
    _, report = year
    daily_report = report.aggregate("1D", "max")

    summary = daily_report.summary()
    assert summary["forecasts"].tolist() == [120, 120, 120]
    assert summary["DM n_days"].tolist() == [120, 120, 120]
    tests = daily_report.tests(benchmark="base", reps=50)
    assert tests["n_days"].tolist() == [120, 120] and tests["n_forecasts"].tolist() == [120, 120]
    assert tests.attrs["partial_days"] == {}
    mcs = daily_report.mcs(reps=50)
    assert mcs.attrs["n_days"] == 120 and mcs.attrs["n_forecasts"] == 120
    assert daily_report.ranks(reps=50)["n_days"].tolist() == [120, 120, 120]
    assert daily_report.compare("better", "base")["n_days"].tolist() == [120]
    assert daily_report.leaderboard(benchmark="base")["n_days"].tolist() == [120, 120, 120]
    # Every breakdown that has something to cut by.
    assert daily_report.by_horizon()["MAE"].notna().all()
    assert daily_report.by_year()["MAE"].notna().all()
    assert len(daily_report.by_quarter()) == 3 * 2
    assert len(daily_report.by_day()) == 3 * 120
    assert set(daily_report.by_price_level()["price_level"]) <= {"negative", "normal", "spike"}
    assert daily_report.compute_all()["summary"]["MAE"].notna().all()


@NO_PARTIAL_DAYS
def test_a_daily_report_agrees_with_the_hourly_one_on_what_it_measures(year):
    """The MAE of daily maxima is computed from the daily maxima: here by hand, for the model `base`."""
    results, report = year
    frame = pd.DataFrame(results["base"])
    expected = frame.groupby(["run_date", "horizon"])[["prediction", "actual"]].max()

    row = report.aggregate("1D", "max").summary().set_index("model").loc["base"]

    assert row["MAE"] == pytest.approx((expected["prediction"] - expected["actual"]).abs().mean(), rel=1e-12)
    assert row["Bias"] == pytest.approx((expected["prediction"] - expected["actual"]).mean(), rel=1e-12)


def test_tests_by_hour_has_no_hours_to_work_with_on_a_daily_report(year):
    _, report = year
    daily_report = report.aggregate("1D", "max")

    with pytest.raises(ValueError, match="by='hour' needs periods within the day"):
        daily_report.tests(benchmark="base", by="hour")
    row = daily_report.compare("better", "base").iloc[0]
    assert np.isnan(row[["hours_a_better", "hours_b_better", "hours_tested"]].to_numpy(dtype=float)).all()
    assert not np.isnan(row["p_hac"])


def test_the_battery_is_left_out_of_a_daily_report_with_a_warning(year):
    results, _ = year
    report = EvaluationReport(results, [MAEEvaluator(), EconomicEvaluator()])

    with pytest.warns(UserWarning, match="leaves EconomicEvaluator out"):
        daily_report = report.aggregate("1D", "max")

    assert [type(e) for e in daily_report.evaluators] == [MAEEvaluator]
    assert "profit" not in daily_report.summary().columns
    assert len(report.evaluators) == 2  # the report it came from keeps its evaluators


def test_the_evaluators_carry_over_unchanged(year):
    _, report = year

    daily_report = report.aggregate("1D", "max")

    assert [type(e) for e in daily_report.evaluators] == [type(e) for e in report.evaluators]
    assert all(a is b for a, b in zip(daily_report.evaluators, report.evaluators))
    assert daily_report.source_data is None


def test_aggregating_leaves_the_report_it_came_from_alone(year):
    _, report = year
    before = report.summary()

    report.aggregate("1D", "max")

    pd.testing.assert_frame_equal(report.summary(), before, check_exact=True)


def test_the_same_results_give_the_same_daily_report(year):
    results, report = year
    shuffled = {
        name: [records[i] for i in np.random.default_rng(2).permutation(len(records))]
        for name, records in results.items()
    }

    one = report.aggregate("1D", "spread").summary()
    other = EvaluationReport(shuffled, report.evaluators).aggregate("1D", "spread").summary()

    pd.testing.assert_frame_equal(one, other, check_exact=True)


# ---------------------------------------------------------------------------------------------------------------------
# What it refuses
# ---------------------------------------------------------------------------------------------------------------------


def test_only_daily_series_are_built():
    report = EvaluationReport({"m": hourly_study()}, [MAEEvaluator()])

    with pytest.raises(ValueError, match="freq must be '1D'"):
        report.aggregate("1W")


def test_an_unknown_aggregate_is_refused():
    report = EvaluationReport({"m": hourly_study()}, [MAEEvaluator()])

    with pytest.raises(ValueError, match="agg must be one of"):
        report.aggregate("1D", agg="median")


def test_a_daily_report_cannot_be_aggregated_again():
    with pytest.raises(ValueError, match="daily already"):
        aggregate(hourly_study()).aggregate("1D")


def test_results_at_different_frequencies_are_refused():
    hourly = hourly_study()
    quarter = []
    for k, (target, kind) in enumerate(list(DAYS.items())[:2]):
        ids = clock_ids(kind, per_hour=4)
        quarter += day_records(target, ids, [50.0] * len(ids), [51.0] * len(ids), freq="15min")

    with pytest.raises(ValueError, match="Aggregating needs one frequency"):
        EvaluationReport({"hourly": hourly, "quarter": quarter}, [MAEEvaluator()]).aggregate("1D")


def test_a_duplicated_key_is_refused():
    records = hourly_study()
    records.append(dict(records[3]))

    with pytest.raises(ValueError, match="each forecast key once"):
        aggregate(records)


def test_records_without_the_keys_are_refused():
    records = [{k: v for k, v in r.items() if k != "horizon"} for r in hourly_study()]

    with pytest.raises(ValueError, match="needs the forecast keys"):
        aggregate(records)


def test_a_model_without_results_has_no_days():
    report = EvaluationReport({"m": hourly_study(), "empty": []}, [MAEEvaluator()])

    daily_report = report.aggregate("1D")

    counts = daily_report.forecast_counts().set_index("model")
    assert counts.loc["empty", "produced"] == 0 and counts.loc["empty", "missing"] == 0
    assert counts.loc["m", "produced"] == 5
