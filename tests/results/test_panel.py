"""`PanelReport` (step 7g): one report per zone side by side, and the log gain of a multi-market study.

The log gain is held to a hand computation on records with holes: each cell is scored on the keys the model and the benchmark
share there, each cell weighs the same, and a cell lost is one with a negative gain.
"""

import math

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.results.panel import PanelReport
from epftoolbox2.results.report import EvaluationReport

DAYS = pd.date_range("2024-01-05", periods=12, freq="7D")


def records(seed: int, scale: float, bias: float = 0.0, holes=()) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for i, day in enumerate(DAYS):
        for horizon in (3, 4):
            if (i, horizon) in holes:
                continue
            actual = 1000.0 * scale + 50.0 * rng.normal()
            out.append(
                {
                    "run_date": day.date().isoformat(),
                    "target_date": (day + pd.Timedelta(days=horizon)).date().isoformat(),
                    "hour": 0,
                    "horizon": horizon,
                    "day_in_test": i * 7,
                    "actual": actual,
                    "prediction": actual + bias + 30.0 * scale * rng.normal(),
                    "freq": "1D",
                }
            )
    return out


def zone_report(seed: int, scale: float, holes=()) -> EvaluationReport:
    return EvaluationReport(
        {
            "TSO": records(seed, scale, bias=20.0 * scale),
            "model": records(seed + 100, scale, holes=holes),
        },
        [MAEEvaluator()],
    )


@pytest.fixture
def panel():
    return PanelReport(
        {"PL": zone_report(1, 1.0, holes={(0, 3), (5, 4)}), "FI": zone_report(2, 0.4), "PT": zone_report(3, 2.5)},
        target="max",
    )


def hand_cells(panel: PanelReport, by) -> dict:
    """(gain, forecasts) per cell: each zone's forecasts the model and the TSO share, then cut by `by` (across zones too)."""
    shared = []
    for (zone, target), report in panel.reports.items():
        frames = {name: pd.DataFrame(list(ref.iter_records())) for name, ref in report.refs.items()}
        keys = ["run_date", "target_date", "hour", "horizon"]
        both = frames["model"].merge(frames["TSO"], on=keys, suffixes=("", "_tso"))
        both["zone"], both["target"] = zone, target
        shared.append(both)
    gains = {}
    for cell, part in pd.concat(shared).groupby(list(by)):
        cell = cell if isinstance(cell, tuple) else (cell,)
        mae = (part["prediction"] - part["actual"]).abs().mean()
        mae_tso = (part["prediction_tso"] - part["actual_tso"]).abs().mean()
        gains[cell] = (100 * math.log(mae_tso / mae), len(part))
    return gains


@pytest.mark.parametrize("by", [("zone", "target", "horizon"), ("zone", "target"), ("zone",), ("horizon",)])
def test_the_log_gain_is_the_mean_over_cells_each_on_its_common_keys(panel, by):
    expected = hand_cells(panel, by)
    cells = panel.log_gain("TSO", by=by, cells=True)
    got = {tuple(row[key] for key in by): (row["log_gain"], row["forecasts"]) for _, row in cells.iterrows()}
    assert got.keys() == expected.keys()
    for cell, (gain, n) in expected.items():
        assert got[cell][0] == pytest.approx(gain, abs=1e-9, rel=0) and got[cell][1] == n
    table = panel.log_gain("TSO", by=by)
    row = table.set_index("model").loc["model"]
    gains = [gain for gain, _ in expected.values()]
    assert row["log_gain"] == pytest.approx(np.mean(gains), abs=1e-9, rel=0)
    assert row["cells"] == len(gains) and row["cells_lost"] == sum(gain < 0 for gain in gains)
    assert row["forecasts"] == sum(n for _, n in expected.values())
    assert row["min_cell_forecasts"] == min(n for _, n in expected.values())
    assert "TSO" not in set(table["model"])


def test_a_cell_lost_is_one_the_benchmark_wins(panel):
    by = ("zone", "horizon")
    lost = panel.cells_lost("TSO", by=by)
    expected = {cell for cell, (gain, _) in hand_cells(panel, by).items() if gain < 0}
    assert {tuple(row[key] for key in by) for _, row in lost.iterrows()} == expected
    assert (lost["log_gain"] < 0).all()
    assert panel.log_gain("TSO", by=by)["cells_lost"].iloc[0] == len(expected)


def test_tables_have_a_zone_and_a_target(panel):
    summary = panel.summary()
    assert list(summary.columns[:3]) == ["zone", "target", "model"]
    assert list(summary["zone"]) == ["PL", "PL", "FI", "FI", "PT", "PT"]
    pd.testing.assert_frame_equal(
        summary[summary["zone"] == "FI"].drop(columns=["zone", "target"]).reset_index(drop=True),
        panel.zones["FI"].summary(),
    )
    counts = panel.forecast_counts()
    assert counts.set_index(["zone", "model"]).loc[("PL", "model"), "produced"] == 22
    assert panel["PT"] is panel.zones["PT"] and list(panel) == ["PL", "FI", "PT"] and len(panel) == 3


def test_panels_of_several_targets_concatenate():
    low = PanelReport({"PL": zone_report(1, 1.0), "FI": zone_report(2, 0.4)}, target="min")
    high = PanelReport({"PL": zone_report(5, 1.2), "FI": zone_report(6, 0.5)}, target="max")
    both = PanelReport.concat([low, high])
    assert both.targets == ["min", "max"] and set(both.reports) == {("PL", "min"), ("FI", "min"), ("PL", "max"), ("FI", "max")}
    cells = both.log_gain("TSO", by=("zone", "target"), cells=True)
    assert len(cells) == 4
    one = low.log_gain("TSO", by=("zone", "target"), cells=True)
    pd.testing.assert_frame_equal(cells[cells["target"] == "min"].reset_index(drop=True), one)
    with pytest.raises(ValueError, match="several targets"):
        both.zones
    with pytest.raises(ValueError, match="more than one panel"):
        PanelReport.concat([low, low])


def test_what_the_log_gain_refuses(panel):
    with pytest.raises(ValueError, match="Unknown benchmark"):
        panel.log_gain("naive")
    with pytest.raises(ValueError, match="Unknown model"):
        panel.log_gain("TSO", models=["lasso"])
    with pytest.raises(ValueError, match="by must name"):
        panel.log_gain("TSO", by=("zone", "year"))
    with pytest.raises(ValueError, match="at least one zone"):
        PanelReport({})
