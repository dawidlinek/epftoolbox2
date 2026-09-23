"""EvaluationReport must score only the current run's test window, not every row a resumed JSONL holds."""

import pytest

from epftoolbox2.models import OLSModel
from epftoolbox2.pipelines import ModelPipeline
from tests.synthetic import make_market_data


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


def test_report_only_scores_forecasts_within_the_current_test_window(tmp_path):
    """Run 04-10..04-13 (4 days, 96 forecasts), then resume with the narrower 04-12..04-13 (2 days, 48
    forecasts) into the same save_dir: the second report must score 48, not the 96 rows still sitting in
    the JSONL from the wider first run."""
    df = make_market_data(days=120, noise=1.0, seed=1)

    wide = ModelPipeline().add_model(OLSModel(["price_d-1"], training_window=30, name="OLS"))
    wide_report = wide.run(df, test_start="2023-04-10", test_end="2023-04-13", horizon=1, save_dir=str(tmp_path))
    assert wide_report.forecast_counts().loc[0, "produced"] == 96

    narrow = ModelPipeline().add_model(OLSModel(["price_d-1"], training_window=30, name="OLS"))
    narrow_report = narrow.run(df, test_start="2023-04-12", test_end="2023-04-13", horizon=1, save_dir=str(tmp_path))

    assert narrow_report.forecast_counts().loc[0, "produced"] == 48
    assert narrow_report.summary().loc[0, "forecasts"] == 48
    assert len(narrow_report.predictions()) == 48
