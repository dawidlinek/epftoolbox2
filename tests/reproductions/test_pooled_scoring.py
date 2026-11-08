"""Tier S of the pooled week-ahead study (step 7g, decision 13 of the step 7 spec), on 5 of its 20 markets.

`tests/fixtures/pooled/` holds the stored forecasts of the S02 study `final_final_tso_weekly_forecasting` (daily minimum and
maximum load of D+3 .. D+9 on 104 Fridays from 2024-01-05) for PL, DE and FR (CET), FI (EET) and PT (WET): the TSO's A31 forecast,
the Lasso on regional NWP, the pooled LightGBM and the pooled MLP, and the actual; `tables.json` has the study's gains per cell
computed from them with its formula, and over all 20 markets the report's +40.2 % (2 cells lost of 40) and +39.0 % (1 lost);
`SOURCES.json` the hashes and attribution. The builder is outside the repository
(`epftoolbox2-research/library_check_step7/pooled/build_fixture.py`), which also checks the 20-market cells.

The forecasts become library records in a folder per zone, `ModelPipeline.report()` of that `save_dir` is a `PanelReport`, and its
`log_gain` against the TSO, each cell on its own common keys, equals the study's per-cell gains within 1e-9, recomputed here with
numpy on the same arrays and as `tables.json` stores them, with 728 forecasts per (market, statistic) and 104 per horizon.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.models import NaiveModel
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.pipelines.model_pipeline import write_zone_index
from epftoolbox2.results.panel import PanelReport

HERE = Path(__file__).parents[1] / "fixtures" / "pooled"
MODELS = ("tso", "lasso", "pooled_lgbm", "pooled_mlp")
FRIDAYS = ("2024-01-05", "2025-12-26")
HORIZONS = range(3, 10)


@pytest.fixture(scope="module")
def stored() -> pd.DataFrame:
    return pd.read_parquet(HERE / "forecasts.parquet")


@pytest.fixture(scope="module")
def tables() -> dict:
    return json.loads((HERE / "tables.json").read_text())


def write_records(frame: pd.DataFrame, save_dir: Path) -> None:
    """The stored forecasts as each model's records in each zone's folder, with the zone index, as a run over zones writes
    them."""
    first = pd.Timestamp(FRIDAYS[0])
    write_zone_index(save_dir, list(dict.fromkeys(frame["zone"])))
    for zone, part in frame.groupby("zone", sort=False):
        folder = save_dir / zone
        folder.mkdir(parents=True)
        for model in MODELS:
            lines = [
                json.dumps(
                    {
                        "run_date": row.run_date,
                        "target_date": row.target_date,
                        "run_weekday": pd.Timestamp(row.run_date).weekday(),
                        "hour": 0,
                        "horizon": int(row.horizon),
                        "day_in_test": (pd.Timestamp(row.run_date) - first).days,
                        "prediction": float(getattr(row, model)),
                        "actual": float(row.actual),
                        "coefficients": [],
                        "freq": "1D",
                    }
                )
                for row in part.itertuples(index=False)
            ]
            (folder / f"{model}.jsonl").write_text("\n".join(lines) + "\n")


def panel_of(stored: pd.DataFrame, save_dir: Path) -> PanelReport:
    """One `PanelReport` per statistic from its zone folders, the two concatenated."""
    pipeline = ModelPipeline().add_evaluator(MAEEvaluator())
    for model in MODELS:
        pipeline.add_model(NaiveModel(predictors=["unused"], name=model))  # the names the reports read
    panels = []
    for statistic in ("min", "max"):
        write_records(stored[stored["target"] == statistic], save_dir / statistic)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # records with no manifest are warned about by name, as any are
            panel = pipeline.report(
                save_dir / statistic, test=FRIDAYS, horizons=HORIZONS, issue_days=["friday"], freq="1D"
            )
        assert isinstance(panel, PanelReport)
        panels.append(PanelReport({zone: panel.zones[zone] for zone in panel.zones}, target=statistic))
    return PanelReport.concat(panels)


@pytest.fixture(scope="module")
def panel(stored, tmp_path_factory) -> PanelReport:
    return panel_of(stored, tmp_path_factory.mktemp("pooled"))


def study_gains(stored: pd.DataFrame, model: str, by: list) -> dict:
    """The study's formula with numpy alone: 100 (ln MAE_TSO - ln MAE_model) per cell, on the cell's forecasts."""
    out = {}
    for key, part in stored.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        actual = part["actual"].to_numpy()
        gain = 100 * (np.log(np.mean(np.abs(actual - part["tso"]))) - np.log(np.mean(np.abs(actual - part[model]))))
        out[tuple(str(k) for k in key)] = (float(gain), len(part))
    return out


@pytest.mark.parametrize("by", [("zone", "target"), ("zone", "target", "horizon")])
def test_the_panel_s_log_gain_is_the_study_s_per_cell_gain(stored, tables, panel, by):
    stored_by = tables["by_zone_target" if len(by) == 2 else "by_zone_target_horizon"]
    for model in ("lasso", "pooled_lgbm", "pooled_mlp"):
        cells = panel.log_gain("tso", by=by, models=[model], cells=True)
        got = {tuple(str(row[key]) for key in by): (row["log_gain"], row["forecasts"]) for _, row in cells.iterrows()}
        expected = study_gains(stored, model, list(by))
        assert got.keys() == expected.keys() and len(got) == 5 * 2 * (1 if len(by) == 2 else len(HORIZONS))
        for cell, (gain, n) in expected.items():
            assert got[cell][0] == pytest.approx(gain, abs=1e-9, rel=0), (model, cell)
            assert got[cell][1] == n == (728 if len(by) == 2 else 104)
            assert got[cell][0] == pytest.approx(stored_by[model]["|".join(cell)]["log_gain"], abs=1e-9, rel=0)
        summary = panel.log_gain("tso", by=by, models=[model]).iloc[0]
        assert summary["log_gain"] == pytest.approx(np.mean([gain for gain, _ in expected.values()]), abs=1e-9, rel=0)
        assert summary["cells_lost"] == sum(gain < 0 for gain, _ in expected.values())
        assert summary["forecasts"] == 5 * 1456


def test_the_panel_counts_every_forecast_of_the_study(stored, panel):
    counts = panel.forecast_counts()
    assert set(counts["zone"]) == {"PL", "DE", "FR", "FI", "PT"} and set(counts["target"]) == {"min", "max"}
    assert (counts["planned"] == 728).all() and (counts["produced"] == 728).all() and (counts["missing"] == 0).all()
    lost = panel.cells_lost("tso", by=("zone", "target"), models=["pooled_lgbm", "pooled_mlp"])
    expected = {
        (model, zone, target)
        for model in ("pooled_lgbm", "pooled_mlp")
        for (zone, target), (gain, _) in study_gains(stored, model, ["zone", "target"]).items()
        if gain < 0
    }
    assert {(row.model, row.zone, row.target) for row in lost.itertuples()} == expected


def test_the_twenty_markets_are_the_report_s(tables):
    """What the builder checked on all 20 markets, kept here: the report's rounded cells."""
    for model, (gain, lost) in {"pooled_lgbm": (40.2, 2), "pooled_mlp": (39.0, 1)}.items():
        row = tables["all_20_markets"][model]
        assert row["cells"] == 40 and round(row["log_gain"], 1) == gain and row["cells_lost"] == lost
