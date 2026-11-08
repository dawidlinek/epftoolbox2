"""`PanelReport` (step 7g): one report per zone side by side, and the log gain of a multi-market study.

The log gain is held to a hand computation on records with holes: each cell is scored on the keys the model and the benchmark
share there, each cell weighs the same, and a cell lost is one with a negative gain.
"""

import math
import warnings

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
    assert both.targets == ["min", "max"] and set(both.reports) == {
        ("PL", "min"),
        ("FI", "min"),
        ("PL", "max"),
        ("FI", "max"),
    }
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


def test_models_are_scored_on_the_same_cells_and_keys_by_default():
    # B has no forecast in PT and A misses two keys in PL: by default both are scored on the keys all three share.
    zones = {
        "PL": EvaluationReport(
            {"TSO": records(1, 1.0, 20.0), "A": records(2, 1.0, holes={(0, 3), (5, 4)}), "B": records(3, 1.0)},
            [MAEEvaluator()],
        ),
        "FI": EvaluationReport(
            {"TSO": records(4, 0.4, 8.0), "A": records(5, 0.4), "B": records(6, 0.4)}, [MAEEvaluator()]
        ),
        "PT": EvaluationReport({"TSO": records(7, 2.5, 50.0), "A": records(8, 2.5), "B": []}, [MAEEvaluator()]),
    }
    panel = PanelReport(zones, target="max")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing to warn about: every model on the same cells
        table = panel.log_gain("TSO", by=("zone",)).set_index("model")
    assert table.loc["A", "cells"] == table.loc["B", "cells"] == 2
    assert table.loc["A", "empty"] == table.loc["B", "empty"] == 1
    assert table.loc["A", "forecasts"] == table.loc["B", "forecasts"] == 22 + 24
    cells = panel.log_gain("TSO", by=("zone",), cells=True)
    pl = cells[cells["zone"] == "PL"].set_index("model")
    assert pl.loc["A", "forecasts"] == pl.loc["B", "forecasts"] == 22  # B on A's keys too
    keys = ["run_date", "target_date", "hour", "horizon"]
    frames = {name: pd.DataFrame(list(ref.iter_records())) for name, ref in zones["PL"].refs.items()}
    on_keys = frames["B"].merge(frames["A"][keys], on=keys)
    assert pl.loc["B", "mae"] == pytest.approx((on_keys["prediction"] - on_keys["actual"]).abs().mean(), abs=1e-12)

    with pytest.warns(UserWarning, match="different cells"):
        own = panel.log_gain("TSO", by=("zone",), common=False)
    assert own.attrs["unequal_cells"] == {"B": [("PT",)]}
    assert own.set_index("model").loc["A", "cells"] == 3 and own.set_index("model").loc["B", "cells"] == 2
    with pytest.warns(UserWarning, match="different cells"):
        lost = panel.cells_lost("TSO", by=("zone",), common=False)
        mine = panel.log_gain("TSO", by=("zone",), common=False, cells=True)
    assert len(lost) == ((mine["forecasts"] > 0) & (mine["log_gain"] < 0)).sum() == own["cells_lost"].sum()


def test_a_cell_with_a_zero_mae_is_a_win_a_loss_or_a_tie():
    perfect = [dict(r, prediction=r["actual"]) for r in records(2, 1.0)]
    zones = {
        "PL": EvaluationReport(
            {"TSO": records(1, 1.0, 20.0), "oracle": perfect, "A": records(3, 1.0)}, [MAEEvaluator()]
        )
    }
    panel = PanelReport(zones, target="max")
    with pytest.warns(UserWarning, match="infinite"):
        won = panel.log_gain("TSO", by=("zone",), models=["oracle"]).iloc[0]
    assert won["log_gain"] == math.inf and won["cells"] == 1 and won["cells_lost"] == 0 and won["empty"] == 0
    with pytest.warns(UserWarning, match="infinite"):
        lost = panel.log_gain("oracle", by=("zone",), models=["A"]).iloc[0]
    assert lost["log_gain"] == -math.inf and lost["cells"] == 1 and lost["cells_lost"] == 1
    with pytest.warns(UserWarning, match="infinite"):
        assert len(panel.cells_lost("oracle", by=("zone",), models=["A"])) == 1
    twins = PanelReport({"PL": EvaluationReport({"x": perfect, "y": perfect}, [MAEEvaluator()])})
    tie = twins.log_gain("x", by=("zone",)).iloc[0]
    assert tie["log_gain"] == 0.0 and tie["cells"] == 1 and tie["cells_lost"] == 0
