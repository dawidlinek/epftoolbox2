"""`EvaluationReport.join()`: models of separate runs put in one report.

The point of a joined report is that it is the report one run of all the models would have given: every table is computed
on forecast keys, so where a model's results came from cannot show.
"""

import json
import warnings

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
from epftoolbox2.results import ModelResultRef
from epftoolbox2.results.report import EvaluationReport
from tests.forecast_records import forecast_records

pytestmark = [
    pytest.mark.filterwarnings("ignore:The comparison rests on"),
    pytest.mark.filterwarnings("ignore:The model confidence set needs"),
    pytest.mark.filterwarnings("ignore:These models have different forecast sets"),
]

RUN_DATES = pd.date_range("2023-01-01", periods=60, freq="D")


def evaluators():
    return [
        MAEEvaluator(),
        RMSEEvaluator(),
        BiasEvaluator(),
        rMAEEvaluator("base"),
        DMEvaluator("base", reps=50),
        MCSEvaluator(reps=50),
        EconomicEvaluator(),
    ]


@pytest.fixture(scope="module")
def results():
    return {
        "base": forecast_records(1, RUN_DATES, noise=6.0, horizons=(1, 2)),
        "better": forecast_records(2, RUN_DATES, noise=4.0, horizons=(1, 2)),
        "holes": forecast_records(3, RUN_DATES, noise=5.0, horizons=(1, 2), drop=lambda r: r["hour"] == 7),
        "noisy": forecast_records(4, RUN_DATES, noise=9.0, horizons=(1, 2)),
    }


def pick(results, *names):
    return {name: results[name] for name in names}


def test_a_joined_report_gives_the_summary_of_one_report_over_all_the_models_bit_for_bit(results):
    one = EvaluationReport(results, evaluators())
    left = EvaluationReport(pick(results, "base", "better"), evaluators())
    right = EvaluationReport(pick(results, "holes", "noisy"), evaluators())

    joined = left.join(right)

    pd.testing.assert_frame_equal(joined.summary(), one.summary(), check_exact=True)
    assert list(joined.summary()["model"]) == ["base", "better", "holes", "noisy"]


def test_every_table_of_a_joined_report_is_the_one_of_a_single_report(results):
    joined = EvaluationReport(pick(results, "base", "holes"), evaluators()).join(
        EvaluationReport(pick(results, "better", "noisy"), evaluators())
    )
    # The joined report lists its models as `base, holes, better, noisy`.
    one = EvaluationReport(pick(results, "base", "holes", "better", "noisy"), evaluators())

    for name in ("by_hour", "by_horizon", "by_year", "by_quarter", "by_price_level"):
        pd.testing.assert_frame_equal(getattr(joined, name)(), getattr(one, name)(), check_exact=True)
    pd.testing.assert_frame_equal(joined.tests(benchmark="base", reps=50), one.tests(benchmark="base", reps=50))
    pd.testing.assert_frame_equal(joined.mcs(reps=50), one.mcs(reps=50), check_exact=True)
    pd.testing.assert_frame_equal(joined.ranks(reps=50), one.ranks(reps=50), check_exact=True)
    pd.testing.assert_frame_equal(joined.compare("better", "base"), one.compare("better", "base"), check_exact=True)
    pd.testing.assert_frame_equal(joined.leaderboard("base"), one.leaderboard("base"), check_exact=True)
    pd.testing.assert_frame_equal(joined.forecast_counts(), one.forecast_counts())
    expected = one.compute_all()  # once: computing every table again for each of them was most of this test's time
    found = joined.compute_all()
    assert set(found) == set(expected)
    for table, frame in found.items():
        pd.testing.assert_frame_equal(frame, expected[table], check_exact=True)


def test_results_read_from_separate_runs_files_join_into_the_same_report(results, tmp_path):
    """Two runs, each with its own folder of JSONL files, as a SLURM array writes them."""
    refs = {}
    for run, names in (("run1", ["base", "better"]), ("run2", ["holes", "noisy"])):
        folder = tmp_path / run
        folder.mkdir()
        for name in names:
            path = folder / f"{name}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in results[name]))
            refs[name] = ModelResultRef(
                name=name,
                path=path,
                count=len(results[name]),
                test_start=RUN_DATES[0].date().isoformat(),
                test_end=RUN_DATES[-1].date().isoformat(),
                horizon=2,
                produced=len(results[name]),
                skipped={"missing_test_predictors": 3} if name == "holes" else {},
            )
    left = EvaluationReport({n: refs[n] for n in ("base", "better")}, evaluators())
    right = EvaluationReport({n: refs[n] for n in ("holes", "noisy")}, evaluators())

    joined = left.join(right)
    one = EvaluationReport(refs, evaluators())

    pd.testing.assert_frame_equal(joined.summary(), one.summary(), check_exact=True)
    pd.testing.assert_frame_equal(joined.forecast_counts(), one.forecast_counts())
    assert joined.forecast_counts().set_index("model").loc["holes", "missing_test_predictors"] == 3


def test_joins_chain(results):
    parts = [EvaluationReport({name: results[name]}, evaluators()) for name in ("base", "better", "holes")]
    one = EvaluationReport(pick(results, "base", "better", "holes"), evaluators())

    pd.testing.assert_frame_equal(parts[0].join(parts[1]).join(parts[2]).summary(), one.summary(), check_exact=True)


def test_the_references_are_kept_as_they_are(results):
    left = EvaluationReport(pick(results, "base"), evaluators())
    right = EvaluationReport(pick(results, "better"), evaluators())

    joined = left.join(right)

    assert list(joined.refs) == ["base", "better"]
    assert joined.refs["base"] is left.refs["base"]
    assert joined.refs["better"] is right.refs["better"]
    assert list(left.refs) == ["base"] and list(right.refs) == ["better"]  # neither report changed


def test_the_evaluators_are_the_left_reports(results):
    left = EvaluationReport(pick(results, "base"), [MAEEvaluator(), rMAEEvaluator("base")])
    right = EvaluationReport(pick(results, "better"), [MAEEvaluator(), rMAEEvaluator("base")])

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the same evaluators by value, though not the same objects: nothing to say
        joined = left.join(right)

    assert all(a is b for a, b in zip(joined.evaluators, left.evaluators))
    joined.evaluators.append(RMSEEvaluator())
    assert len(left.evaluators) == 2  # the list is the joined report's own


def test_different_evaluators_warn_and_the_left_ones_win(results):
    left = EvaluationReport(pick(results, "base"), [MAEEvaluator()])
    right = EvaluationReport(pick(results, "better"), [MAEEvaluator(), RMSEEvaluator()])

    with pytest.warns(UserWarning, match="different evaluators.*left"):
        joined = left.join(right)

    assert "RMSE" not in joined.summary().columns
    assert list(joined.summary()["model"]) == ["base", "better"]


def test_evaluators_that_differ_only_in_a_setting_warn(results):
    left = EvaluationReport(pick(results, "base"), [rMAEEvaluator("base")])
    right = EvaluationReport(pick(results, "better"), [rMAEEvaluator("better")])

    with pytest.warns(UserWarning, match="different evaluators") as caught:
        left.join(right)

    assert "base_model='base'" in str(caught[0].message) and "base_model='better'" in str(caught[0].message)


def test_nested_settings_count_too(results):
    from epftoolbox2.evaluators import BatteryArbitrage

    left = EvaluationReport(pick(results, "base"), [EconomicEvaluator(BatteryArbitrage(power_mw=1))])
    same = EvaluationReport(pick(results, "better"), [EconomicEvaluator(BatteryArbitrage(power_mw=1))])
    other = EvaluationReport(pick(results, "holes"), [EconomicEvaluator(BatteryArbitrage(power_mw=2))])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        left.join(same)
    with pytest.warns(UserWarning, match="different evaluators"):
        left.join(other)


def test_a_name_in_both_reports_is_refused(results):
    left = EvaluationReport(pick(results, "base", "better"), evaluators())
    right = EvaluationReport(pick(results, "better", "noisy"), evaluators())

    with pytest.raises(ValueError, match=r"model named \['better'\]"):
        left.join(right)


def test_a_report_cannot_be_joined_to_itself(results):
    report = EvaluationReport(pick(results, "base"), evaluators())

    with pytest.raises(ValueError, match="rename one"):
        report.join(report)


def test_results_at_different_frequencies_are_refused(results):
    quarter = []
    for r in results["better"][:96]:
        quarter.append({**r, "hour": r["hour"] * 4, "freq": "15min"})
    left = EvaluationReport(pick(results, "base"), [MAEEvaluator()])
    right = EvaluationReport({"quarter": quarter}, [MAEEvaluator()])

    with pytest.raises(ValueError, match="Joining reports needs one frequency"):
        left.join(right)


def test_an_hourly_report_and_a_daily_one_are_refused(results):
    hourly = EvaluationReport(pick(results, "base"), [MAEEvaluator()])
    daily = EvaluationReport(pick(results, "better"), [MAEEvaluator()]).aggregate("1D")

    with pytest.raises(ValueError, match="Joining reports needs one frequency"):
        hourly.join(daily)
    with pytest.raises(ValueError, match="Joining reports needs one frequency"):
        daily.join(hourly)


def test_only_a_report_can_be_joined(results):
    with pytest.raises(TypeError, match="takes an EvaluationReport"):
        EvaluationReport(pick(results, "base"), []).join({"better": results["better"]})


def test_daily_reports_join_into_the_daily_report_of_all_the_models(results):
    """Aggregating each run and joining gives the aggregate of the joined report."""
    kept = [MAEEvaluator(), RMSEEvaluator(), rMAEEvaluator("base"), DMEvaluator("base", reps=50)]
    left = EvaluationReport(pick(results, "base", "better"), kept)
    right = EvaluationReport(pick(results, "holes", "noisy"), kept)

    with pytest.warns(UserWarning, match="left out days"):  # `holes` lacks an hour: no whole day
        separately = left.aggregate("1D", "max").join(right.aggregate("1D", "max"))
    with pytest.warns(UserWarning, match="left out days"):
        together = left.join(right).aggregate("1D", "max")

    pd.testing.assert_frame_equal(separately.summary(), together.summary(), check_exact=True)
    pd.testing.assert_frame_equal(separately.forecast_counts(), together.forecast_counts())


def test_the_source_data_is_the_left_reports_or_else_the_right_ones(results):
    data_left, data_right = pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [2]})
    left = EvaluationReport(pick(results, "base"), [], source_data=data_left)
    right = EvaluationReport(pick(results, "better"), [], source_data=data_right)
    bare = EvaluationReport(pick(results, "holes"), [])

    assert left.join(right).source_data is data_left
    assert bare.join(right).source_data is data_right
    assert bare.join(EvaluationReport(pick(results, "noisy"), [])).source_data is None
