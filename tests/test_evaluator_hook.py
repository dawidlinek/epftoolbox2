"""The hook that lets an evaluator see the whole set of models: one call per group, several columns from one result."""

import pandas as pd
import pytest

from epftoolbox2.evaluators import (
    DMEvaluator,
    Evaluator,
    MAEEvaluator,
    MCSEvaluator,
    rMAEEvaluator,
)
from epftoolbox2.evaluators import _compare
from epftoolbox2.results.report import EvaluationReport
from tests.forecast_records import forecast_records

pytestmark = pytest.mark.filterwarnings("ignore:The comparison rests on")

RUN_DATES = pd.date_range("2023-12-01", periods=70, freq="D")


def study(horizons=(1,), run_dates=RUN_DATES):
    return {
        "base": forecast_records(1, run_dates, noise=6.0, horizons=horizons),
        "better": forecast_records(2, run_dates, noise=4.0, horizons=horizons),
        "same": forecast_records(3, run_dates, noise=6.0, horizons=horizons),
    }


class TwoColumns(Evaluator):
    name = "two"

    def compute(self, df, **kwargs):
        return {"mean": float(df["prediction"].mean()), "n": len(df)}


class CountingGroup(Evaluator):
    name = "counted"

    def __init__(self):
        self.calls = 0

    def compute(self, df, **kwargs):
        raise AssertionError("compute_group is overridden, compute must not be called")

    def compute_group(self, model_dfs):
        self.calls += 1
        return dict.fromkeys(model_dfs, float(len(model_dfs)))


def test_the_default_compute_group_loops_compute():
    frames = {"a": pd.DataFrame({"prediction": [1.0, 2.0], "actual": [1.0, 4.0]})}
    assert MAEEvaluator().compute_group(frames) == {"a": 1.0}


def test_set_evaluators_refuse_mixed_frequencies_but_plain_metrics_do_not():
    hourly = forecast_records(1, RUN_DATES)
    quarters = [dict(r, freq="15min") for r in forecast_records(2, RUN_DATES, periods=lambda day: 96)]
    results = {"hourly": hourly, "quarters": quarters, "other": forecast_records(3, RUN_DATES)}

    assert len(EvaluationReport(results, [MAEEvaluator()]).summary()) == 3
    for report in (EvaluationReport(results, [DMEvaluator("hourly", reps=20)]), EvaluationReport(results, [])):
        with pytest.raises(ValueError, match="one frequency"):
            report.summary() if report.evaluators else report.tests(benchmark="hourly")


def test_a_dict_result_becomes_named_columns():
    table = EvaluationReport(study(), [MAEEvaluator(), TwoColumns()]).summary()
    assert list(table.columns) == ["model", "forecasts", "MAE", "two mean", "two n"]
    assert (table["two n"] == table["forecasts"]).all()


def test_each_group_calls_a_set_evaluator_once():
    evaluator = CountingGroup()
    report = EvaluationReport(study(horizons=(1, 2)), [evaluator])

    table = report.by_horizon()

    assert evaluator.calls == 2
    assert (table["counted"] == 3.0).all()
    evaluator.calls = 0
    report.summary()
    assert evaluator.calls == 1


def test_the_loss_panel_is_built_once_per_group(monkeypatch):
    built = []
    original = _compare.LossPanel.__init__

    def counting(self, model_dfs):
        built.append(tuple(model_dfs))
        original(self, model_dfs)

    monkeypatch.setattr(_compare.LossPanel, "__init__", counting)
    report = EvaluationReport(study(), [DMEvaluator("base", reps=20), MCSEvaluator(reps=20), rMAEEvaluator("base")])

    report.summary()
    assert len(built) == 1

    built.clear()
    report.by_horizon()
    assert len(built) == 1


def test_two_dict_evaluators_on_one_column_raise():
    report = EvaluationReport(study(), [DMEvaluator("base", reps=20), DMEvaluator("same", reps=20)])
    with pytest.raises(ValueError, match="'DM ratio'"):
        report.summary()


def test_a_one_value_evaluator_on_a_dict_column_raises_in_either_order():
    class Fake(Evaluator):
        name = "DM ratio"

        def compute(self, df, **kwargs):
            return -999.0

    for evaluators in ([DMEvaluator("base", reps=20), Fake()], [Fake(), DMEvaluator("base", reps=20)]):
        with pytest.raises(ValueError, match="'DM ratio'"):
            EvaluationReport(study(), evaluators).summary()


def test_two_one_value_evaluators_on_a_column_keep_the_last_and_warn():
    """That was the behaviour of two rMAEEvaluators, so such a report keeps its numbers, but no longer silently."""
    results = study()
    with pytest.warns(UserWarning, match="Two evaluators write the column 'rMAE'"):
        table = EvaluationReport(results, [rMAEEvaluator("base"), rMAEEvaluator("same")]).summary().set_index("model")
    assert table.loc["same", "rMAE"] == 1.0


def test_an_evaluator_cannot_write_a_column_of_the_table():
    class Hour(Evaluator):
        name = "hour"

        def compute(self, df, **kwargs):
            return 1.0

    with pytest.raises(ValueError, match="'hour'"):
        EvaluationReport(study(), [Hour()]).by_hour()


def test_compute_of_a_set_evaluator_reads_from_the_group():
    frames = EvaluationReport(study(), [])._model_frames()
    evaluator = DMEvaluator("base", reps=20)
    assert evaluator.compute(frames["better"], model_dfs=frames) == evaluator.compute_group(frames)["better"]
    with pytest.raises(ValueError, match="model_dfs"):
        evaluator.compute(frames["better"])
    with pytest.raises(ValueError, match="one of the frames"):
        evaluator.compute(frames["better"].copy(), model_dfs=frames)
