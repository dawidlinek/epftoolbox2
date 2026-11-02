"""Build the week-ahead fixtures from the local copy of the parent study (not run in CI: the source is outside the repo).

    uv run python tests/fixtures/week_ahead/build.py [SOURCE_DIR]

`SOURCE_DIR` defaults to `~/Desktop/min-max-hour-load-forecasting/data/parent`. For now this writes `PL_direct.csv` alone: the
stored forecasts of the published study for Poland, issued on the 104 Fridays from 2024-01-05 to 2025-12-26 at horizons 3 to 9,
with the columns the tests read (the TSO's forecast, the regional NWP Lasso and the actual daily minimum and maximum load). The
values are copied as text, so the file holds the source's numbers to the digit.
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = Path.home() / "Desktop" / "min-max-hour-load-forecasting" / "data" / "parent"
DIRECT_COLUMNS = ["run_date", "horizon", "target", "actual", "tso_prediction", "regional_nwp_prediction"]


def build_direct(source: Path, market: str = "PL") -> Path:
    """`{market}_direct.csv` with `DIRECT_COLUMNS`, rows in the source's order."""
    frame = pd.read_csv(source / "models_ecmwf" / f"{market}_direct.csv", dtype=str, keep_default_na=False)
    out = HERE / f"{market}_direct.csv"
    frame[DIRECT_COLUMNS].to_csv(out, index=False, lineterminator="\n")
    return out


if __name__ == "__main__":
    written = build_direct(Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCE)
    print(f"{written} ({written.stat().st_size} bytes)")
