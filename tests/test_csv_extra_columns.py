"""`CsvExporter(extra_columns=...)` writes every forecast once, next to the source row of its own local date and period.

The source is joined onto the forecasts by (target date, period of the day). A source finer than the results, or the hour
that repeats in autumn, would write a forecast once per source row, and any count or Diebold-Mariano test run on the file
would be inflated, so the source is checked like a model's data and the repeated hour joins its second occurrence, the row
a run forecasts from. Where nothing was repeated, the files are those written before: `tests/fixtures/csv_extra_columns/`
holds what the exporter wrote for these cases before the check existed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.exporters import CsvExporter
from epftoolbox2.models import OLSModel
from epftoolbox2.models._contract import check_index
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.report import EvaluationReport
from tests.synthetic import PREDICTORS, make_market_data

ZONE = "Europe/Warsaw"
FIXTURES = Path(__file__).parent / "fixtures" / "csv_extra_columns"
STEP = {"1h": "h", "15min": "15min", "1D": "D"}
EXTRA = ["load", "is_holiday"]
KEYS = ["target_date", "hour", "horizon"]

SPRING_DAYS = ["2023-03-25", "2023-03-26", "2023-03-27"]  # clocks go forward on the 26th: 23 hours, 92 quarter-hours
AUTUMN_DAYS = ["2023-10-28", "2023-10-29", "2023-10-30"]  # clocks go back on the 29th: 25 hours, 100 quarter-hours


def _midnight(day) -> pd.Timestamp:
    return pd.Timestamp(day).tz_localize(ZONE)


def _periods(day, freq):
    """The period ids a model forecasts on a local date, and those of them that the clock has twice (no `actual`)."""
    if freq == "1D":
        return [0], set()
    clock = pd.date_range(
        _midnight(day), _midnight(pd.Timestamp(day) + pd.Timedelta(days=1)), freq=STEP[freq], inclusive="left"
    )
    ids = (clock.hour if freq == "1h" else clock.hour * 4 + clock.minute // 15).tolist()
    return sorted(set(ids)), {i for i in ids if ids.count(i) > 1}


def _records(days, freq, offset, horizons=(1,)):
    """One model's records, one per period of each target day and horizon, with exact binary fractions for values."""
    records = []
    for i, day in enumerate(days):
        periods, repeated = _periods(day, freq)
        for horizon in horizons:
            for period in periods:
                actual = 40.0 + 0.25 * period + horizon
                records.append(
                    {
                        "run_date": (pd.Timestamp(day) - pd.Timedelta(days=horizon)).date().isoformat(),
                        "target_date": day,
                        "hour": period,
                        "horizon": horizon,
                        "day_in_test": i,
                        # the repeated hour has a forecast and no actual
                        "actual": float("nan") if period in repeated else actual,
                        "prediction": actual + offset + 0.125 * (period % 4),
                        "freq": freq,
                    }
                )
    return records


def _source(start, end, freq):
    """A local-time frame from `start` up to `end` at `freq` whose `load` is the row's position, so no two instants agree."""
    if freq == "1D":  # one row a date, at noon, which the clock has once on every date
        index = pd.date_range(
            pd.Timestamp(start) + pd.Timedelta(hours=12), pd.Timestamp(end), freq="D", inclusive="left"
        )
        index = index.tz_localize(ZONE)
    else:
        index = pd.date_range(_midnight(start), _midnight(end), freq=STEP[freq], inclusive="left")
    position = np.arange(len(index))
    return pd.DataFrame({"load": position.astype(float), "is_holiday": position % 7}, index=index)


def _instant(day, period, freq):
    """The instant a (local date, period) names, the later one if the clock has it twice: pandas' own rule (DST flag off)."""
    minutes = 720 if freq == "1D" else period * (60 if freq == "1h" else 15)
    return (pd.Timestamp(day) + pd.Timedelta(minutes=minutes)).tz_localize(ZONE, ambiguous=False)


def _export(path, results, source, extra=EXTRA):
    CsvExporter(str(path), extra_columns=list(extra)).export(EvaluationReport(results, [], source_data=source))
    return pd.read_csv(path)


def _positions(table, source, freq):
    return [source.index.get_loc(_instant(d, h, freq)) for d, h in zip(table["target_date"], table["hour"])]


# What the exporter wrote for each of these before the source was checked: no autumn switch, so nothing was repeated.
_IDENTITY = {
    "1h": (
        "1h",
        ["2023-03-24", "2023-03-25", "2023-03-26", "2023-03-27", "2023-03-28"],
        (1, 2),
        "2023-03-23",
        "2023-03-27",
    ),
    "15min": ("15min", SPRING_DAYS, (1,), "2023-03-24", "2023-03-27"),
    "1D": (
        "1D",
        ["2023-03-24", "2023-03-25", "2023-03-26", "2023-03-27", "2023-03-28"],
        (1, 2),
        "2023-03-23",
        "2023-03-27",
    ),
}


def _identity_case(name):
    freq, days, horizons, start, end = _IDENTITY[name]
    ols = _records(days, freq, 1.5, horizons)
    results = {
        "OLS": ols,
        "Lasso": [r for r in _records(days, freq, -0.5, horizons) if r["hour"] % 5],
    }  # holes: NaN cells
    # The source ends before the last target day, so the rows after it have no extra columns.
    return results, _source(start, end, freq)


@pytest.mark.parametrize("name", list(_IDENTITY))
def test_a_file_without_a_repeated_hour_is_byte_for_byte_what_was_written_before(tmp_path, name):
    """Two models with holes in one, the spring switch day (23 hours, 92 quarter-hours) among the dates, a source that stops
    early (NaN extras), at 1h, 15min and 1D. The fixtures were written by the exporter before the source was checked."""
    results, source = _identity_case(name)
    path = tmp_path / "out.csv"
    CsvExporter(str(path), extra_columns=EXTRA + ["not_in_the_source"]).export(
        EvaluationReport(results, [], source_data=source)
    )
    assert path.read_bytes() == (FIXTURES / f"{name}.csv").read_bytes()


@pytest.mark.parametrize("horizons", [(1,), (1, 2, 3)])
def test_an_hourly_source_writes_each_forecast_once_across_the_autumn_switch(tmp_path, horizons):
    """The 2023-10-29 source has two rows at 02:00, one per UTC offset. The forecast for that hour gets the second
    occurrence's values (the row a run forecasts from), and nothing else in the file moves."""
    results = {"M": _records(AUTUMN_DAYS, "1h", 1.0, horizons)}
    source = _source("2023-10-26", "2023-11-02", "1h")
    assert (source.index.tz_localize(None).date == pd.Timestamp("2023-10-29").date()).sum() == 25
    plain = _export(tmp_path / "plain.csv", results, source, extra=[])
    table = _export(tmp_path / "out.csv", results, source)

    assert len(table) == len(results["M"]) == len(plain)
    assert not table.duplicated(KEYS).any()
    pd.testing.assert_frame_equal(table.drop(columns=EXTRA), plain)  # the forecasts are those of a file without extras
    assert _positions(table, source, "1h") == table["load"].astype(int).tolist()
    repeated = table[(table["target_date"] == "2023-10-29") & (table["hour"] == 2)]
    first = source.index.get_loc(pd.Timestamp("2023-10-29 02:00").tz_localize(ZONE, ambiguous=True))
    second = source.index.get_loc(pd.Timestamp("2023-10-29 02:00").tz_localize(ZONE, ambiguous=False))
    assert second == first + 1
    assert repeated["load"].tolist() == [float(second)] * len(horizons)
    assert repeated["is_holiday"].tolist() == [second % 7] * len(horizons)
    assert repeated["actual"].isna().all()  # no target for the repeated hour; its forecast and extras are still there


def test_the_second_occurrence_is_chosen_by_time_whatever_the_order_of_the_source(tmp_path):
    results = {"M": _records(AUTUMN_DAYS, "1h", 1.0)}
    source = _source("2023-10-26", "2023-11-02", "1h")
    ordered = _export(tmp_path / "ordered.csv", results, source)
    shuffled = _export(tmp_path / "shuffled.csv", results, source.sample(frac=1.0, random_state=4))
    pd.testing.assert_frame_equal(shuffled, ordered)
    assert (tmp_path / "shuffled.csv").read_bytes() == (tmp_path / "ordered.csv").read_bytes()


def test_a_quarter_hour_source_writes_each_forecast_once_across_both_switches(tmp_path):
    """The spring day has 92 rows on both sides, the autumn day 100 in the source and 96 periods in the results; the forecasts
    of 02:00-02:45 on the 29th get the second occurrence's values."""
    results = {"M": _records(SPRING_DAYS + AUTUMN_DAYS, "15min", 1.0, horizons=(1, 2))}
    source = _source("2023-03-23", "2023-11-02", "15min")
    plain = _export(tmp_path / "plain.csv", results, source, extra=[])
    table = _export(tmp_path / "out.csv", results, source)

    assert len(table) == len(results["M"]) == len(plain) == 2 * (96 + 92 + 96 + 3 * 96)
    assert not table.duplicated(KEYS).any()
    assert table.groupby("target_date").size().to_dict() == {
        "2023-03-25": 2 * 96,
        "2023-03-26": 2 * 92,
        "2023-03-27": 2 * 96,
        "2023-10-28": 2 * 96,
        "2023-10-29": 2 * 96,
        "2023-10-30": 2 * 96,
    }
    pd.testing.assert_frame_equal(table.drop(columns=EXTRA), plain)
    assert _positions(table, source, "15min") == table["load"].astype(int).tolist()
    repeated = table[(table["target_date"] == "2023-10-29") & table["hour"].between(8, 11)]
    assert len(repeated) == 2 * 4
    first_pass = source.index.get_loc(pd.Timestamp("2023-10-29 02:00").tz_localize(ZONE, ambiguous=True))
    second_pass = [first_pass + 4 + k for k in range(4)]  # the hour is gone through again, four quarter-hours later
    assert sorted(set(repeated["load"])) == [float(row) for row in second_pass]


def _run_hourly(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    hourly = make_market_data(start="2023-09-01", days=70, horizon=2, noise=5.0, seed=3)
    report = (
        ModelPipeline()
        .add_model(OLSModel(PREDICTORS, training_window=30, name="m"))
        .run(hourly, test=("2023-10-27", "2023-10-30"), horizon=1)
    )
    return hourly, report


def test_csv_extra_columns_write_each_hourly_forecast_once(tmp_path, monkeypatch):
    """The scenario once pinned as a known issue: an hourly source on 96 hourly records of a real run over the autumn switch
    wrote 292 repeated keys, the 02:00 of 2023-10-29 once per source row."""
    hourly, report = _run_hourly(monkeypatch)
    path = tmp_path / "forecasts.csv"
    CsvExporter(str(path), extra_columns=["load"]).export(EvaluationReport(report.refs, [], source_data=hourly))

    table = pd.read_csv(path)
    records = sum(1 for _ in report.refs["m"].iter_records(cols=["hour"]))
    assert not table.duplicated(["target_date", "hour", "horizon"]).any()
    assert len(table) == records
    row = table[(table["target_date"] == "2023-10-29") & (table["hour"] == 2)]
    second = hourly.loc[pd.Timestamp("2023-10-29 02:00").tz_localize(ZONE, ambiguous=False), "load"]
    assert row["load"].tolist() == [second]  # the second occurrence of the hour, from which a run forecasts


def test_a_quarter_hour_source_on_hourly_results_is_refused(tmp_path, monkeypatch):
    """It has four rows per (date, hour): a left join wrote 388 rows for these 96 hourly forecasts."""
    _, report = _run_hourly(monkeypatch)
    source = make_market_data(start="2023-09-01", days=70, horizon=2, freq="15min", seed=3)
    path = tmp_path / "forecasts.csv"
    with pytest.raises(ValueError, match=r"extra_columns need source data at the resolution of the 1h results.*15 min"):
        CsvExporter(str(path), extra_columns=["load"]).export(EvaluationReport(report.refs, [], source_data=source))
    assert not path.exists()


@pytest.mark.parametrize("source_freq", ["15min", "1D"])
def test_a_source_of_another_resolution_than_the_results_is_refused_with_what_to_do(tmp_path, source_freq):
    results = {"M": _records(AUTUMN_DAYS, "1h", 1.0)}
    source = _source("2023-10-20", "2023-11-02", source_freq)
    with pytest.raises(ValueError, match=r"1h results.*resample or aggregate the source to 1h"):
        _export(tmp_path / "out.csv", results, source)
    assert not (tmp_path / "out.csv").exists()


def test_a_source_that_holds_the_repeated_hour_twice_without_an_offset_is_refused(tmp_path):
    """A naive local index shows the autumn switch as two rows with one timestamp: which one is "the" 02:00 can't be told."""
    results = {"M": _records(AUTUMN_DAYS, "1h", 1.0)}
    source = _source("2023-10-26", "2023-11-02", "1h")
    source.index = source.index.tz_localize(None)
    with pytest.raises(ValueError, match=r"extra_columns need source data.*duplicate timestamp"):
        _export(tmp_path / "out.csv", results, source)


def test_rows_of_the_source_no_forecast_reads_are_not_checked(tmp_path):
    """Only the target dates of the file decide: a stretch of daily rows months before them, which a run over it would refuse,
    doesn't change the file."""
    results = {"M": _records(AUTUMN_DAYS, "1h", 1.0)}
    hourly = _source("2023-10-26", "2023-11-02", "1h")
    early = pd.concat([_source("2023-08-01", "2023-08-31", "1D"), hourly])
    with pytest.raises(ValueError, match="coarser resolution"):
        check_index(early, "1h")  # the premise: checked as a whole, this frame fails

    pd.testing.assert_frame_equal(
        _export(tmp_path / "early.csv", results, early), _export(tmp_path / "hourly.csv", results, hourly)
    )
    assert (tmp_path / "early.csv").read_bytes() == (tmp_path / "hourly.csv").read_bytes()
