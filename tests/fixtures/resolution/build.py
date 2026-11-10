"""The resolution study's R1 forecasts (honest information set, DE_LU and AT) and its tables, for
`tests/reproductions/test_resolution_scoring.py`. Not run in CI: it reads the study's stored outputs, copied read-only from S02
`~/epf2_modeling_research/resolution/` into the research folder (`epftoolbox2-research/library_check_step7/s02_copy/resolution`).

    uv run python -I tests/fixtures/resolution/build.py SRC tests/fixtures/resolution

SRC holds the study's `out/` (`pred_r1_{zone}_ren0.npz`, `{zone}_grid.pkl`, `table_r1.csv`, `table_r1_dm.csv`,
`daily_r3_{zone}_P_ren0.csv`). The forecasts are those the study scored (`evaluate.py`, `load_pred` and `metrics`): the test
days (2026-04-02 to 2026-10-11, local Europe/Berlin and Europe/Vienna days) whose 96 quarter-hour slots are all real 15-minute
prices or DST-synthetic, with every variant finite. In this window none is synthetic (no clock change), which the build checks,
so every slot is scored. Writes to OUT_DIR:

- `r1_honest.parquet` (zstd; byte-stream-split for the models' predictions): `zone`, `target_date` (the local delivery day), `slot` (the wall-clock
  quarter-hour 0-95, the library's period id), `actual` (the day-ahead price, ENTSO-E A44) and the prediction of each of the
  nine variants, float64 as stored. The forecast of day T was issued on T-1 (horizon 1).
- `tables.json`: the study's cells for these forecasts: the honest rows of `table_r1.csv` (rounded to 4 decimals there) and
  `table_r1_dm.csv`, the derived rows of `daily_r3_{zone}_P_ren0.csv` (R3: daily base, peak 08-20, min and max of the R1
  forecasts; full precision), and the rounded cells of `resolution/REPORT.md` sections 3 and 5 (transcribed below).
- `SOURCES.json`: where the inputs come from, the sha256 of every file read and written, and the data's attribution.
"""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ZONES = {"de_lu": "Europe/Berlin", "at": "Europe/Vienna"}
VARIANTS = (
    "A96_QLOC",
    "A96_QLOC_ridge",
    "A96_FULL",
    "C96_HAGG",
    "B24_flat",
    "B24_static",
    "B24_dyn",
    "D24x4",
    "naive",
)

# resolution/REPORT.md section 3, honest rows: MAE15 / MAEh / shape MAE and the level/hour/quarter shares, where given.
REPORT_R1 = {
    "de_lu": {
        "A96_QLOC_ridge": {
            "MAE15": 26.81,
            "MAEh": 26.08,
            "shapeMAE": 6.25,
            "level": 0.36,
            "hour": 0.57,
            "quarter": 0.07,
        },
        "A96_FULL": {"MAE15": 27.33, "MAEh": 26.60, "shapeMAE": 6.47},
        "B24_dyn": {"MAE15": 27.41, "MAEh": 26.80, "shapeMAE": 5.93, "level": 0.38, "hour": 0.56, "quarter": 0.07},
        "A96_QLOC": {"MAE15": 27.43, "MAEh": 26.71, "shapeMAE": 6.43},
        "C96_HAGG": {"MAE15": 27.45, "MAEh": 26.72, "shapeMAE": 6.47},
        "B24_static": {"MAE15": 27.61, "MAEh": 26.79, "shapeMAE": 6.35},
        "D24x4": {"MAE15": 28.02, "MAEh": 27.40, "shapeMAE": 6.04},
        "B24_flat": {"MAE15": 28.65, "MAEh": 26.79, "shapeMAE": 9.06, "level": 0.35, "hour": 0.53, "quarter": 0.12},
        "naive": {"MAE15": 36.28, "MAEh": 35.90, "shapeMAE": 7.02},
    },
    "at": {
        "A96_QLOC_ridge": {"MAE15": 23.74, "MAEh": 22.30, "shapeMAE": 7.79},
        "A96_FULL": {"MAE15": 23.97},
        "B24_dyn": {"MAE15": 24.02, "MAEh": 22.79, "shapeMAE": 7.06},
        "A96_QLOC": {"MAE15": 24.11},
        "C96_HAGG": {"MAE15": 24.17},
        "B24_flat": {"MAE15": 26.00, "MAEh": 22.79, "shapeMAE": 10.60},
        "naive": {"MAE15": 28.98},
    },
}
# Section 3's DM list, honest DE and AT: (A, B, statistic with "negative = A better", two-sided p) at 2 decimals; "p<1e-4"
# is written as 0. "n.s." rows carry no number. The report states some pairs the other way round from `table_r1_dm.csv`.
REPORT_R1_DM = {
    "de_lu": [
        ("B24_dyn", "A96_QLOC", -0.32, 0.75),
        ("A96_QLOC", "C96_HAGG", -0.69, 0.49),
        ("B24_dyn", "B24_flat", -12.0, 0.0),
        ("B24_dyn", "B24_static", -7.3, None),
        ("D24x4", "A96_QLOC", 5.1, None),
        ("A96_QLOC_ridge", "A96_QLOC", None, 0.12),
    ],
    "at": [
        ("B24_dyn", "A96_QLOC", -1.07, 0.28),
        ("A96_QLOC", "C96_HAGG", -1.76, 0.08),
        ("B24_dyn", "B24_flat", -16.4, 0.0),
        ("B24_dyn", "B24_static", -7.4, None),
        ("D24x4", "A96_QLOC", 6.4, None),
        ("A96_QLOC_ridge", "A96_QLOC", None, 0.39),
    ],
}
# Section 5's table, derived rows (Apr-Oct 2026, honest): MAE and, where given, the mean bias, at 1 decimal.
REPORT_R3 = {
    "de_lu": {
        "A96_QLOC": {"base": (21.0, -3.7), "peak": (29.9, None), "min": (37.1, 0.2), "max": (56.1, -40.4)},
        "A96_QLOC_ridge": {"base": (19.7, -1.5), "peak": (28.4, None), "min": (37.8, None), "max": (53.3, -32.8)},
        "B24_dyn": {"base": (21.0, None), "peak": (29.9, None), "min": (36.8, 1.0), "max": (56.6, -42.9)},
        "B24_static": {"base": (20.9, None), "peak": (29.9, None), "min": (36.2, 0.4), "max": (55.5, -37.8)},
        "B24_flat": {"base": (20.9, None), "peak": (29.9, None), "min": (36.3, 6.0), "max": (61.0, -53.3)},
        "naive": {"base": (31.8, None), "peak": (38.4, None), "min": (46.1, None), "max": (76.5, None)},
    },
    "at": {
        "A96_QLOC": {"base": (17.2, None), "peak": (25.6, None), "min": (37.6, None), "max": (41.1, -31.7)},
        "A96_QLOC_ridge": {"base": (15.5, None), "peak": (24.0, None), "min": (39.9, None), "max": (40.5, -22.9)},
        "B24_dyn": {"base": (16.8, None), "peak": (25.0, None), "min": (37.7, 5.4), "max": (42.0, -31.2)},
        "B24_static": {"base": (16.8, None), "peak": (25.0, None), "min": (37.4, 4.7), "max": (39.9, -28.0)},
        "B24_flat": {"base": (16.8, None), "peak": (25.0, None), "min": (37.9, 11.1), "max": (48.4, -45.6)},
        "naive": {"base": (21.2, None), "peak": (29.0, None), "min": (42.6, None), "max": (55.6, None)},
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scored_days(grid: dict, t: np.ndarray, preds: dict) -> np.ndarray:
    """`evaluate.metrics`' `day_ok`: every slot real 15-minute or DST-synthetic, and every variant finite."""
    real = grid["P_nat"][t]
    day_ok = (real | grid["P_syn"][t]).all(1)
    for p in preds.values():
        day_ok &= np.isfinite(p).all(1)
    return day_ok


def build_zone(src: Path, zone: str, read: dict):
    grid_path, pred_path = src / f"out/{zone}_grid.pkl", src / f"out/pred_r1_{zone}_ren0.npz"
    with open(grid_path, "rb") as f:
        grid = pickle.load(f)
    npz = np.load(pred_path, allow_pickle=False)
    read[str(grid_path.relative_to(src))] = sha256(grid_path)
    read[str(pred_path.relative_to(src))] = sha256(pred_path)
    t = npz["t"]
    preds = {k[6:]: npz[k] for k in npz.files if k.startswith("pred__")}
    assert set(preds) == set(VARIANTS), sorted(preds)
    ok = scored_days(grid, t, preds)
    days = grid["days"][t][ok]
    actual = grid["P"][t][ok]
    assert grid["P_nat"][t][ok].all(), "a scored slot is DST-synthetic: the fixture would need its actual left out"
    # R3's days: a complete raw local day (its daily statistics finite) of native 15-minute slots, as `run_daily.py` reads them.
    raw = grid["P_daily"].reindex(days)[["base", "peak", "min", "max"]]
    assert np.isfinite(raw.to_numpy()).all()
    n, slots = len(days), np.tile(np.arange(96, dtype=np.int8), len(days))
    frame = pd.DataFrame(
        {
            "zone": zone,
            "target_date": np.repeat(days.date, 96),
            "slot": slots,
            "actual": actual.reshape(-1),
            **{v: preds[v][ok].reshape(-1) for v in VARIANTS},
        }
    )
    return frame, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    src, out = args.src.resolve(), args.out
    read: dict = {}
    frames, n_days = [], {}
    for zone in ZONES:
        frame, n_days[zone] = build_zone(src, zone, read)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["zone"] = data["zone"].astype("category")
    target = out / "r1_honest.parquet"
    # Prices with two decimals (the actual, and naive's copies of it) compress best with a dictionary; the models' full
    # float64 predictions with byte-stream-split and no dictionary (2.0 MB against 2.7 MB with the defaults).
    repeated = ["actual", "naive"]
    data.to_parquet(
        target,
        engine="pyarrow",
        index=False,
        compression="zstd",
        compression_level=19,
        use_dictionary=["zone", "target_date", "slot", *repeated],
        use_byte_stream_split=[v for v in VARIANTS if v not in repeated],
    )

    tables: dict = {"days": n_days, "zones": ZONES}
    for name in ("table_r1.csv", "table_r1_dm.csv"):
        path = src / "out" / name
        read[f"out/{name}"] = sha256(path)
        table = pd.read_csv(path)
        table = table[table["info"] == "honest"]
        tables[name[:-4]] = json.loads(table.to_json(orient="records", double_precision=15))
    tables["daily_r3"] = {}
    for zone in ZONES:
        path = src / f"out/daily_r3_{zone}_P_ren0.csv"
        read[str(path.relative_to(src))] = sha256(path)
        table = pd.read_csv(path)
        table = table[table["model"].str.startswith("derived_") & ~table["model"].str.endswith("+biascorr")]
        assert (table["days"] == n_days[zone]).all(), "R3 scored other days than R1"
        tables["daily_r3"][zone] = [
            {k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()} for row in table.to_dict("records")
        ]
    tables["report_r1"] = REPORT_R1
    tables["report_r1_dm"] = REPORT_R1_DM
    tables["report_r3"] = {
        z: {m: {s: list(c) for s, c in cells.items()} for m, cells in v.items()} for z, v in REPORT_R3.items()
    }
    (out / "tables.json").write_text(json.dumps(tables, indent=1, sort_keys=True) + "\n")

    sources = {
        "study": "resolution (epftoolbox2-research/resolution/REPORT.md), scripts evaluate.py, eval_all.py, run_daily.py",
        "s02_path": "~/epf2_modeling_research/resolution/",
        "copied_read_only": "2026-11-08, rsync into epftoolbox2-research/library_check_step7/s02_copy/resolution",
        "attribution": (
            "Day-ahead prices of DE_LU and AT: ENTSO-E Transparency Platform (A44, sequence 1), "
            "https://transparency.entsoe.eu; the predictions are the study's own forecasts of them."
        ),
        "inputs": dict(sorted(read.items())),
        "outputs": {p.name: sha256(p) for p in (target, out / "tables.json")},
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (out / "SOURCES.json").write_text(json.dumps(sources, indent=1, sort_keys=True) + "\n")
    print({p.name: p.stat().st_size for p in out.iterdir() if p.is_file()})


if __name__ == "__main__":
    main()
