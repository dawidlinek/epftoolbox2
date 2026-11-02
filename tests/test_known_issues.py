"""Known methodology issues, pinned with xfail(strict=True).

Each test asserts the *correct* behaviour and currently fails. When an issue is fixed its test starts
passing, strict mode turns that into a failure, and the marker must be removed (move the test next to
the code it covers). Details: .claude/rules/{models,data,evaluation}.md.
"""

import datetime

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.transformers import DailyProfileTransformer
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import OLSModel
from tests.synthetic import make_market_data


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="`_sNN` profile names don't say their slot length; the information set reads each as a quarter-hour",
)
def test_a_half_hour_profile_slot_is_judged_at_its_own_time():
    """DailyProfileTransformer names the slots of every length but one hour `_sNN`. With freq="30min" on hourly data,
    `load_actual_s30` is the load at 15:00 on D, realised at 16:30 with a 1 h rule, but the information set reads it
    as quarter-hour 30 (07:30-07:45, known 08:45) and accepts it at every period of an 11:45 issue: a look-ahead."""
    index = pd.date_range("2024-01-01", periods=24 * 30, freq="h", tz="Europe/Warsaw")
    raw = pd.DataFrame({"load_actual": np.arange(len(index), dtype=float)}, index=index)
    data = DailyProfileTransformer(columns="load_actual", freq="30min").transform(raw)
    day = data.loc["2024-01-10"]
    assert (day["load_actual_s30"] == raw.loc["2024-01-10 15:00", "load_actual"]).all()  # 15:00 on every row
    info = InformationSet("11:45", "Europe/Brussels", rules={"load_actual_s*": Availability.realised("1h")})
    expanded = {(period, 1): ["load_actual_s30"] for period in range(24)}

    found, _ = info.violations(expanded, "price", "1h", index, "2024-01-10", "2024-01-16")

    assert {v.period for v in found} == set(range(24))


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="whether a list is used on every issue day decides the re-read of a mask's column with no value, and the last day decides",
)
def test_a_later_end_of_the_test_moves_no_forecast_through_a_mask_column_with_no_value(monkeypatch):
    """`late` has no value before 1 March, the first issue day, so a mask read there can't tell whether it is binary. A list used
    on every issue day keeps it unscaled for good, as a fixed list does; a list not used on every issue day reads it again on the
    first of its days with a value (`BaseModel._plan_scalable_masks`). Whether it is used on every day depends on the last issue
    day, so the forecasts of 5 April move when the test goes on to 12 April, where another list starts: 24 of 24 by up to 0.25
    here, 22 of 24 by up to 0.14 with LassoCV in epftoolbox2-research/library_check_step3/issue_days/probes/p_residual.py, which
    finds the same through a value on 12 April that a function reads. No row after the issue day trains, but a later day decides
    how a column is scaled. Runs of every day have it as much as runs of some weekdays."""
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    data = make_market_data(start="2023-01-01", days=120, horizon=1, noise=5.0, seed=3)
    dates = np.asarray(pd.DatetimeIndex(data.index.tz_localize(None)).normalize())
    values = dates >= np.datetime64("2023-03-01")
    late = np.full(len(data), np.nan)
    late[values] = np.random.default_rng(4).normal(size=int(values.sum()))
    data = data.assign(late=late)
    drivers = ["load_d+1", "wind_d+1"]

    def before_10_april(ctx):
        return drivers + ["late"] if ctx.issue_day < datetime.date(2023, 4, 10) else drivers

    def forecasts_of_5_april(test_end):
        records = OLSModel(before_10_april, training_window=40).run(data, "2023-03-01", test_end, horizon=1)._results
        return {r["hour"]: r["prediction"] for r in records if r["run_date"] == "2023-04-05"}

    shorter = forecasts_of_5_april("2023-04-07")
    assert len(shorter) == 24
    assert forecasts_of_5_april("2023-04-12") == shorter


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="CsvExporter joins extra_columns of 1h results on (date, hour) without checking the source: rows repeat",
)
@pytest.mark.parametrize("source_freq", ["15min", "1h"])
def test_csv_extra_columns_write_each_hourly_forecast_once(tmp_path, monkeypatch, source_freq):
    """`_join_extra_columns` checks the source's resolution for 15min and 1D results only. For 1h results a 15-minute source has
    four rows per (date, hour), and even an hourly one two on the hour repeated on 2023-10-29, so the left join writes a forecast
    once per source row: 96 hourly records become 388 rows, 292 of them repeated keys, from a 15-minute source."""
    from epftoolbox2.exporters import CsvExporter
    from epftoolbox2.pipelines import ModelPipeline
    from epftoolbox2.results.report import EvaluationReport
    from tests.synthetic import PREDICTORS

    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    hourly = make_market_data(start="2023-09-01", days=70, horizon=2, noise=5.0, seed=3)
    report = (
        ModelPipeline()
        .add_model(OLSModel(PREDICTORS, training_window=30, name="m"))
        .run(hourly, test=("2023-10-27", "2023-10-30"), horizon=1)
    )
    source = (
        hourly
        if source_freq == "1h"
        else make_market_data(start="2023-09-01", days=70, horizon=2, freq="15min", seed=3)
    )
    path = tmp_path / "forecasts.csv"
    CsvExporter(str(path), extra_columns=["load"]).export(EvaluationReport(report.refs, [], source_data=source))

    table = pd.read_csv(path)
    records = sum(1 for _ in report.refs["m"].iter_records(cols=["hour"]))
    assert not table.duplicated(["target_date", "hour", "horizon"]).any()
    assert len(table) == records
