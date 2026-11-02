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
from epftoolbox2.results.report import EvaluationReport

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
