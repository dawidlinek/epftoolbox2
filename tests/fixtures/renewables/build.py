"""The renewables study's stored forecasts and tables, for `tests/reproductions/test_renewables_scoring.py`: experiment 2's
ridge price forecasts (DE_LU and AT) and experiment 1's DE_LU solar and wind forecasts. Not run in CI: it reads the study's
outputs, copied read-only from S02 `~/epf2_modeling_research/renewables/` into the research folder
(`epftoolbox2-research/library_check_step7/s02_copy/renewables`).

    uv run python -I tests/fixtures/renewables/build.py SRC tests/fixtures/renewables

SRC holds the study's `data/` (`arrays.pkl`, `gen_de_lu_{SOL,WIND}_{ridge,hgb,hgb_anchor}.pkl`, `price_preds*.pkl`) and
`results/` (`gen_table.csv`, `gen_dm.csv`, `price_table.csv`, `price_dm.csv`). The forecasts are on the keys the study scored
them on (`evaluate.py`): local day x wall-clock hour (the study's `to_grid`, so the hour the spring switch skips holds the mean
of its neighbours and the repeated autumn hour the mean of its two), within the period, with the actual and every candidate
of the zone finite, LASSO variants included, so the CI subset is scored on the study's own sample. Writes to OUT_DIR:

- `prices.parquet`: `zone`, `target_date` (delivery day T, issued on T-1), `hour` (0-23), `actual` (the hourly mean of the
  day-ahead price) and the ridge forecasts `base`, `own`, `tso`, `both`, `own_rl`, `own_trainact`, and for AT `own_de`,
  `tso_de` (NaN for DE), on the test keys (2025-07-01 to 2026-10-10: 11,140 DE, 10,676 AT).
- `generation.parquet`: DE_LU `kind` (`SOL`, `WIND`), `period` (`val` 2025-01-01 to 06-30, `test` 2025-07-01 to 2026-10-10),
  `target_date`, `hour`, `actual` (Energy-Charts net generation, MW), `cap` (the study's capacity proxy of the day),
  `is_night` (1 where the study's sine of the sun's elevation at mid-hour is <= 0), and the forecasts `ridge`, `hgb`,
  `hgb_anchor`, `a69_d1` (the TSO's A69 forecast for T, post-gate), `a69_d` (A69 for T-1, the honest baseline) and
  `actual_dm1` (the actual of T-2 = D-1).
- `tables.json`: the study's cells for these forecasts (full precision CSV rows) and the rounded cells of
  `renewables/REPORT.md` sections 3 and 4 (transcribed below).
- `SOURCES.json`: where the inputs come from, the sha256 of every file read and written, and the data's attribution.
"""

import argparse
import glob
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TEST = ("2025-07-01", "2026-10-10")
PERIODS = {"val": ("2025-01-01", "2025-06-30"), "test": TEST}
PRICE_VARIANTS = ("base", "own", "tso", "both", "own_rl", "own_trainact", "own_de", "tso_de")
GEN_MODELS = ("ridge", "hgb", "hgb_anchor")

# renewables/REPORT.md section 3, DE rows of the test table: MAE, nMAE %, MAE/cap %, daylight MAE, bias (None: "–").
REPORT_GEN = {
    "SOL": {
        "ridge": (1105, 11.4, 2.10, 2103, 66),
        "hgb": (1027, 10.5, 1.94, 1955, 65),
        "hgb_anchor": (984, 10.1, 1.86, 1873, 49),
        "a69_d1": (760, 7.8, 1.45, 1448, 229),
        "a69_d": (2040, 21.0, 3.85, 3899, 254),
        "actual_dm1": (2492, 25.6, 4.70, 4771, 46),
    },
    "WIND": {
        "ridge": (2820, 18.3, 5.44, None, -1),
        "hgb": (2942, 19.1, 5.70, None, -1152),
        "hgb_anchor": (2917, 18.9, 5.65, None, -1054),
        "a69_d1": (1390, 9.0, 2.68, None, 100),
        "a69_d": (8109, 52.6, 15.66, None, None),
        "actual_dm1": (9801, 63.5, 18.95, None, None),
    },
}
REPORT_MEAN_ACTUAL = {"SOL": 9734, "WIND": 15426}
# Section 3's validation table (2025-H1 MAE, MW), DE.
REPORT_GEN_VAL = {
    "SOL": {"ridge": 1068, "hgb": 1041, "hgb_anchor": 992, "a69_d1": 587},
    "WIND": {"ridge": 2683, "hgb": 2811, "hgb_anchor": 2816, "a69_d1": 1412},
}
# Section 3's DM bullets for DE (test period; negative = first better), at 1 decimal.
REPORT_GEN_DM = {
    "SOL": [
        ("hgb_anchor", "a69_d1", 6.7),
        ("hgb", "a69_d", -10.7),
        ("hgb_anchor", "hgb", -4.2),
        ("hgb", "ridge", -2.7),
    ],
    "WIND": [("hgb_anchor", "a69_d1", 12.3), ("hgb", "a69_d", -16.7)],
}
# Section 4's table (test MAE, EUR/MWh), ridge columns.
REPORT_PRICE = {
    "de_lu": {"base": 20.43, "own": 17.21, "own_rl": 17.21, "own_trainact": 17.56, "tso": 16.03, "both": 16.03},
    "at": {
        "base": 18.44,
        "own": 17.52,
        "own_rl": 17.56,
        "own_trainact": 17.59,
        "tso": 17.72,
        "both": 17.46,
        "own_de": 16.82,
        "tso_de": 16.75,
    },
}
REPORT_PRICE_RMSE = {"de_lu": {"base": 33.10, "own": 28.48}}
# The recovered share of the A69 gain, %: (base - own) / (base - A69), and AT with DE inputs (own_de against tso_de).
REPORT_SHARES = {"de_lu": {"own": 73}, "at": {"own_de": 96}}
# Section 4's DM table, ridge: (A, B, statistic, p) at 2 decimals; None where the cell gives none.
REPORT_PRICE_DM = {
    "de_lu": [
        ("own", "base", -6.88, None),
        ("tso", "base", -8.38, None),
        ("own", "tso", 5.90, None),
        ("both", "tso", -0.16, 0.88),
        ("own_rl", "own", 0.13, 0.90),
        ("own_trainact", "own", 2.50, 0.012),
    ],
    "at": [
        ("own", "base", -4.38, None),
        ("tso", "base", -3.75, None),
        ("own", "tso", -1.35, 0.18),
        ("both", "tso", -2.34, 0.02),
        ("own_trainact", "own", 0.83, 0.41),
        ("own_de", "own", -3.15, 0.002),
        ("tso_de", "tso", -4.09, None),
        ("own_de", "tso_de", 0.58, 0.56),
    ],
}
# Section 4's sub-periods (ridge MAE): H2 2025 and 2026.
REPORT_PRICE_SUB = {
    "de_lu": {
        "test_H2_2025": {"base": 16.82, "own": 14.02, "tso": 13.50},
        "test_2026": {"base": 22.80, "own": 19.31, "tso": 17.70},
    },
    "at": {
        "test_H2_2025": {"base": 14.00, "own": 13.57, "tso": 13.63, "own_de": 12.86},
        "test_2026": {"base": 21.22, "own": 20.00, "tso": 20.29, "own_de": 19.30},
    },
}
# Cells the report prints that no rounding of the study's value gives: (zone, period, variant, printed, the value at 4 decimals).
REPORT_ERRATA = [
    ("at", "test_H2_2025", "tso", 13.63, 13.6357),  # price_table.csv 13.635713..., printed 13.636 by evaluate.py
]
GEN_NAMES = {
    "ridge": "ridge",
    "hgb": "hgb",
    "hgb_anchor": "hgb_anchor",
    "TSO_A69_T (post-gate)": "a69_d1",
    "naive: TSO_A69_D (honest)": "a69_d",
    "naive: actual D-1": "actual_dm1",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lag(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    out[k:] = a[:-k]
    return out


def load(path: Path, read: dict, src: Path):
    read[str(path.relative_to(src))] = sha256(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def frame_of(days, keys, columns):
    """The rows of `keys` (days x 24) as a long frame: `target_date`, `hour` and each column's values there."""
    d, h = np.nonzero(keys)
    return pd.DataFrame(
        {"target_date": days[d].date, "hour": h.astype(np.int8), **{k: v[d, h] for k, v in columns.items()}}
    )


def prices(src: Path, data: dict, read: dict) -> pd.DataFrame:
    preds = {}
    for path in sorted(glob.glob(str(src / "data/price_preds*.pkl"))):
        q = load(Path(path), read, src)
        q.pop("nsel")
        q.pop("own_choice", None)
        preds.update(q)
    frames = []
    for zone in ("de_lu", "at"):
        d = data[zone]
        days, y = d["days"], d["P"]
        candidates = {k: v for k, v in preds.items() if k[0] == zone}
        sel = (days >= TEST[0]) & (days <= TEST[1])
        keys = sel[:, None] & np.isfinite(y)
        for v in candidates.values():
            keys &= np.isfinite(v)
        columns = {"actual": y}
        for variant in PRICE_VARIANTS:
            columns[variant] = candidates.get((zone, variant, "ridge"), np.full_like(y, np.nan))
        frame = frame_of(days, keys, columns)
        frame.insert(0, "zone", zone)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["zone"] = out["zone"].astype("category")
    return out


def generation(src: Path, data: dict, read: dict) -> pd.DataFrame:
    frames = []
    d = data["de_lu"]
    days = d["days"]
    for kind in ("SOL", "WIND"):
        g = {}
        for model in GEN_MODELS:
            part = load(src / f"data/gen_de_lu_{kind}_{model}.pkl", read, src)
            for name in ("actual", "tso", "cap"):
                if name in g:  # every model's file holds the same actual, A69 and proxy
                    assert np.array_equal(g[name], part[("de_lu", kind, name)], equal_nan=True)
                g[name] = part[("de_lu", kind, name)]
            g[model] = part[("de_lu", kind, model)]
        y, tso, cap = g["actual"], g["tso"], g["cap"]
        candidates = {model: g[model] for model in GEN_MODELS}
        candidates.update(a69_d1=tso, a69_d=lag(tso, 1), actual_dm1=lag(y, 2))
        for period, (a, b) in PERIODS.items():
            sel = (days >= a) & (days <= b)
            keys = sel[:, None] & np.isfinite(y)
            for v in candidates.values():
                keys &= np.isfinite(v)
            columns = {
                "actual": y,
                "cap": np.repeat(cap[:, None], 24, axis=1),
                "is_night": (d["SINEL"] <= 0).astype(np.int8),
                **candidates,
            }
            frame = frame_of(days, keys, columns)
            frame.insert(0, "period", period)
            frame.insert(0, "kind", kind)
            frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["kind"] = out["kind"].astype("category")
    out["period"] = out["period"].astype("category")
    return out


def write(frame: pd.DataFrame, path: Path, repeated):
    # Values with few decimals (prices, actual MW, the proxy, the naive copies of them) compress best with a dictionary;
    # the models' full float64 predictions with byte-stream-split and no dictionary.
    floats = [c for c in frame.columns if frame[c].dtype == float]
    frame.to_parquet(
        path,
        engine="pyarrow",
        index=False,
        compression="zstd",
        compression_level=19,
        use_dictionary=[c for c in frame.columns if c not in floats or c in repeated],
        use_byte_stream_split=[c for c in floats if c not in repeated],
    )


def records_of(path: Path, keep) -> list:
    """The CSV's rows, floats as Python reads them (JSON writes their shortest repr, so they round-trip exactly)."""
    table = pd.read_csv(path)
    return [
        {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        for row in table[keep(table)].to_dict("records")
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    src, out = args.src.resolve(), args.out
    read: dict = {}
    data = load(src / "data/arrays.pkl", read, src)
    written = []
    for name, frame, repeated in (
        ("prices.parquet", prices(src, data, read), ["actual"]),
        ("generation.parquet", generation(src, data, read), ["actual", "cap", "a69_d1", "a69_d", "actual_dm1"]),
    ):
        write(frame, out / name, repeated)
        written.append(out / name)

    results = src / "results"
    for name in ("gen_table.csv", "gen_dm.csv", "price_table.csv", "price_dm.csv"):
        read[f"results/{name}"] = sha256(results / name)
    tables = {
        "gen_table": records_of(results / "gen_table.csv", lambda t: t["zone"] == "de_lu"),
        "gen_dm": records_of(results / "gen_dm.csv", lambda t: t["zone"] == "de_lu"),
        "price_table": records_of(results / "price_table.csv", lambda t: t["variant"].str.endswith("/ridge")),
        "price_dm": records_of(results / "price_dm.csv", lambda t: t["model"] == "ridge"),
        "gen_names": GEN_NAMES,
        "report_gen": REPORT_GEN,
        "report_mean_actual": REPORT_MEAN_ACTUAL,
        "report_gen_val": REPORT_GEN_VAL,
        "report_gen_dm": REPORT_GEN_DM,
        "report_price": REPORT_PRICE,
        "report_price_rmse": REPORT_PRICE_RMSE,
        "report_shares": REPORT_SHARES,
        "report_price_dm": REPORT_PRICE_DM,
        "report_price_sub": REPORT_PRICE_SUB,
        "report_errata": REPORT_ERRATA,
        "periods": {"test": TEST, "test_H2_2025": ("2025-07-01", "2025-12-31"), "test_2026": ("2026-01-01", TEST[1])},
    }
    (out / "tables.json").write_text(json.dumps(tables, indent=1, sort_keys=True) + "\n")
    written.append(out / "tables.json")

    sources = {
        "study": "renewables (epftoolbox2-research/renewables/REPORT.md), scripts evaluate.py, gen_models.py, price_models.py",
        "s02_path": "~/epf2_modeling_research/renewables/",
        "copied_read_only": "2026-11-08, rsync into epftoolbox2-research/library_check_step7/s02_copy/renewables",
        "attribution": (
            "Day-ahead prices (A44) and the TSOs' day-ahead wind and solar forecasts (A69) of DE_LU and AT: ENTSO-E "
            "Transparency Platform, https://transparency.entsoe.eu. Actual net generation of DE: Energy-Charts, "
            "Fraunhofer ISE, https://www.energy-charts.info, CC BY 4.0. The other forecasts are the study's own."
        ),
        "inputs": dict(sorted(read.items())),
        "outputs": {p.name: sha256(p) for p in written},
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (out / "SOURCES.json").write_text(json.dumps(sources, indent=1, sort_keys=True) + "\n")
    print({p.name: p.stat().st_size for p in out.iterdir() if p.is_file()})


if __name__ == "__main__":
    main()
