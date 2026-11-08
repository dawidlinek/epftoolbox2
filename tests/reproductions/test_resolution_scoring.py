"""Tier S of the resolution study: its R1 table (honest DE_LU and AT), its DM list and the derived rows of R3, recomputed by the
library's report from the study's stored forecasts.

Three things are checked for every cell: the library's value is within 1e-9 of the study's formula (`evaluate.py`,
`run_daily.py`), written again here with numpy on the fixture's arrays; that formula gives the study's CSV cell to the precision
the CSV stores (`table_r1.csv` rounds to 4 decimals, `daily_r3_*.csv` holds full precision), which shows the fixture holds what
the study scored; and the library's value rounds to the cell `resolution/REPORT.md` prints. The fixture and its builder are in
`tests/fixtures/resolution/`.

The DM statistic is the one exception to "within 1e-9 of the study's formula": the study's Newey-West variance averages each
autocovariance over its n - k products (`evaluate.dm_test`), the library's (`stats.newey_west_variance`) over n, the usual
estimator. The library is checked within 1e-10 of a numpy DM with the usual estimator on the same daily losses, the study's own
formula against its CSV, and every verdict at 5 % against the study's.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sstats

from epftoolbox2.evaluators import BiasEvaluator, MAEEvaluator, RMSEEvaluator, ShapeMAEEvaluator
from epftoolbox2.results.report import EvaluationReport
from tests.reproductions.scoring import printed_equal, resolution, resolution_records, tables

ZONES = ("de_lu", "at")
LAG = 6
CSV_DECIMALS = 4


def arrays(zone):
    """`(actual, {variant: prediction})`, each (days, 96), from the fixture."""
    frame = resolution(zone).sort_values(["target_date", "slot"])
    variants = [c for c in frame.columns if c not in ("zone", "target_date", "slot", "actual")]
    shape = (-1, 96)
    return frame["actual"].to_numpy().reshape(shape), {v: frame[v].to_numpy().reshape(shape) for v in variants}


def study_metrics(y, p):
    """`evaluate.metrics` for one variant on days whose slots are all real (the fixture's)."""
    e = p - y
    eh = e.reshape(-1, 24, 4).mean(2)
    lvl = e.mean(1, keepdims=True)
    total = (e**2).mean()
    dev_p = p.reshape(-1, 24, 4) - p.reshape(-1, 24, 4).mean(2, keepdims=True)
    dev_y = y.reshape(-1, 24, 4) - y.reshape(-1, 24, 4).mean(2, keepdims=True)
    return {
        "days": len(y),
        "n15": y.size,
        "MAE15": np.abs(e).mean(),
        "RMSE15": np.sqrt((e**2).mean()),
        "MAEh": np.abs(eh).mean(),
        "RMSEh": np.sqrt((eh**2).mean()),
        "shapeMAE": np.abs(dev_p - dev_y).mean(),
        "mse_level": (lvl**2).mean() / total,
        "mse_hour": ((eh - lvl) ** 2).mean() / total,
        "mse_quarter": ((e.reshape(-1, 24, 4) - eh[:, :, None]) ** 2).mean() / total,
    }


def dm(y, pa, pb, divisor):
    """DM on the daily mean |e| differential A - B with a Newey-West variance of lag 6 (negative: A better). `divisor(n, k)`
    is what each autocovariance is averaged over: `n - k` in the study's `evaluate.dm_test`, `n` in the usual estimator."""
    d = np.abs(pa - y).mean(1) - np.abs(pb - y).mean(1)
    n = len(d)
    dc = d - d.mean()
    v = (dc**2).mean()
    for k in range(1, LAG + 1):
        v += 2 * (1 - k / (LAG + 1)) * (dc[k:] * dc[:-k]).sum() / divisor(n, k)
    stat = d.mean() / np.sqrt(v / n)
    return stat, 2 * (1 - sstats.norm.cdf(abs(stat)))


def csv_equal(value, cell):
    """The value rounds to the CSV's cell (4 decimals), allowing the last-digit tie of a rounding at exactly 5."""
    return abs(value - cell) <= 0.5 * 10**-CSV_DECIMALS + 1e-12


R3_AGG = {"base": "mean", "peak": "peak", "min": "min", "max": "max"}


@pytest.fixture(scope="module")
def computed():
    """Every library table the checks read, built once per zone (a report of plain lists reads its records per table)."""
    out = {}
    for zone in ZONES:
        records = resolution_records(zone)
        report = EvaluationReport(records, [MAEEvaluator(), RMSEEvaluator(), ShapeMAEEvaluator(), BiasEvaluator()])
        tables_of = {
            "summary": report.summary().set_index("model"),
            "hourly": report.aggregate("1h", agg="mean").summary().set_index("model"),
            "split": report.error_split().set_index("model"),
        }
        for stat, agg in R3_AGG.items():
            daily = report.aggregate("1D", agg=agg)
            tables_of[stat] = (daily.summary().set_index("model"), dict(daily.iter_details()))
        pairs = {}
        for row in tables("resolution")["table_r1_dm"]:
            if row["zone"] == zone:
                pair = EvaluationReport({row["A"]: records[row["A"]], row["B"]: records[row["B"]]}, [])
                pairs[(row["A"], row["B"])] = pair.tests(benchmark=row["B"], lag=LAG, interval=None).iloc[0]
        tables_of["dm"] = pairs
        out[zone] = tables_of
    return out


@pytest.mark.parametrize("zone", ZONES)
def test_r1_table_is_reproduced(zone, computed):
    summary, hourly, split = (computed[zone][k] for k in ("summary", "hourly", "split"))
    y, preds = arrays(zone)
    rows = [r for r in tables("resolution")["table_r1"] if r["zone"] == zone]
    assert sorted(r["model"] for r in rows) == sorted(preds)
    printed = tables("resolution")["report_r1"][zone]
    for row in rows:
        model = row["model"]
        oracle = study_metrics(y, preds[model])
        library = {
            "days": split.loc[model, "days"],
            "n15": summary.loc[model, "forecasts"],
            "MAE15": summary.loc[model, "MAE"],
            "RMSE15": summary.loc[model, "RMSE"],
            "MAEh": hourly.loc[model, "MAE"],
            "RMSEh": hourly.loc[model, "RMSE"],
            "shapeMAE": summary.loc[model, "ShapeMAE"],
            "mse_level": split.loc[model, "level"],
            "mse_hour": split.loc[model, "hour"],
            "mse_quarter": split.loc[model, "quarter"],
        }
        assert library["days"] == oracle["days"] == row["days"] == tables("resolution")["days"][zone]
        assert library["n15"] == oracle["n15"] == row["n15"]
        assert hourly.loc[model, "forecasts"] == row["days"] * 24
        for cell in ("MAE15", "RMSE15", "MAEh", "RMSEh", "shapeMAE", "mse_level", "mse_hour", "mse_quarter"):
            assert abs(library[cell] - oracle[cell]) <= 1e-9, (model, cell, library[cell], oracle[cell])
            assert csv_equal(oracle[cell], row[cell]), (model, cell, oracle[cell], row[cell])
            assert csv_equal(library[cell], row[cell]), (model, cell, library[cell], row[cell])
        names = {"level": "mse_level", "hour": "mse_hour", "quarter": "mse_quarter"}
        for cell, value in printed.get(model, {}).items():
            assert printed_equal(library[names.get(cell, cell)], value, 2, 3), (
                model,
                cell,
                library[names.get(cell, cell)],
            )


@pytest.mark.parametrize("zone", ZONES)
def test_r1_dm_list_is_reproduced(zone, computed):
    y, preds = arrays(zone)
    rows = [r for r in tables("resolution")["table_r1_dm"] if r["zone"] == zone]
    assert len(rows) == 7
    for row in rows:
        library = computed[zone]["dm"][(row["A"], row["B"])]
        assert library["model"] == row["A"] and library["benchmark"] == row["B"]
        assert library["n_days"] == len(y)
        usual, usual_p = dm(y, preds[row["A"]], preds[row["B"]], lambda n, k: n)
        study, study_p = dm(y, preds[row["A"]], preds[row["B"]], lambda n, k: n - k)
        # The library's statistic is positive when A is better: the study's sign the other way round.
        assert abs(-library["statistic"] - usual) <= 1e-10, (row, library["statistic"], usual)
        assert abs(library["p_value"] - usual_p) <= 1e-10
        assert csv_equal(study, row["DM_stat"]) and csv_equal(study_p, row["p"]), (row, study, study_p)
        assert (library["p_value"] < 0.05) == (row["p"] < 0.05), row
        assert np.sign(-library["statistic"]) == np.sign(row["DM_stat"])
    printed = tables("resolution")["report_r1_dm"][zone]
    for a, b, statistic, p_value in printed:
        study, study_p = dm(y, preds[a], preds[b], lambda n, k: n - k)
        if statistic is not None:
            digits = 1 if abs(statistic) >= 5 else 2
            assert abs(round(study, digits) - statistic) <= 1e-9, (a, b, study, statistic)
        if p_value is not None:
            assert abs(study_p - p_value) <= 0.005 + 1e-9 or (p_value == 0 and study_p < 1e-4), (a, b, study_p)


@pytest.mark.parametrize("zone", ZONES)
def test_r3_derived_rows_are_reproduced(zone, computed):
    y, preds = arrays(zone)
    rows = tables("resolution")["daily_r3"][zone]
    printed = tables("resolution")["report_r3"][zone]
    cells = {}
    for stat in R3_AGG:
        summary, details = computed[zone][stat]
        for model in preds:
            errors = (details[model]["prediction"] - details[model]["actual"]).to_numpy()
            cells[(model, stat)] = {
                "days": summary.loc[model, "forecasts"],
                "MAE": summary.loc[model, "MAE"],
                "bias": summary.loc[model, "Bias"],
                "medErr": float(np.median(errors)),
                "RMSE": summary.loc[model, "RMSE"],
            }
    blocks = {"base": slice(0, 96), "peak": slice(32, 80), "min": slice(0, 96), "max": slice(0, 96)}
    reduce = {"base": np.mean, "peak": np.mean, "min": np.min, "max": np.max}
    assert len(rows) == 4 * len(preds)
    for row in rows:
        model, stat = row["model"][len("derived_") :], row["stat"]
        p = reduce[stat](preds[model][:, blocks[stat]], axis=1)
        a = reduce[stat](y[:, blocks[stat]], axis=1)  # the study's raw daily statistics: these slots, all native
        e = p - a
        oracle = {"MAE": np.abs(e).mean(), "bias": e.mean(), "medErr": np.median(e), "RMSE": np.sqrt((e**2).mean())}
        library = cells[(model, stat)]
        assert library["days"] == row["days"] == len(y)
        for cell, value in oracle.items():
            stored = float(row[cell])
            assert abs(library[cell] - value) <= 1e-9, (model, stat, cell)
            assert abs(value - stored) <= 1e-9 * max(1.0, abs(stored)), (model, stat, cell, value, stored)
        mae, bias = printed.get(model, {}).get(stat, (None, None))
        if mae is not None:
            assert printed_equal(library["MAE"], mae, 1, 1), (model, stat, library["MAE"], mae)
        if bias is not None:
            assert printed_equal(library["bias"], bias, 1, 1), (model, stat, library["bias"], bias)


def test_the_fixture_holds_what_the_study_scored():
    for zone in ZONES:
        frame = resolution(zone)
        assert len(frame) == tables("resolution")["days"][zone] * 96
        assert frame.groupby("target_date")["slot"].apply(lambda s: list(s) == list(range(96))).all()
        assert np.isfinite(frame.drop(columns=["zone", "target_date", "slot"]).to_numpy()).all()
        dates = pd.to_datetime(frame["target_date"])
        assert dates.min() == pd.Timestamp("2026-04-02") and dates.max() == pd.Timestamp("2026-10-11")
