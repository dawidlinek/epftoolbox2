"""The studies' stored forecasts as plain result records, for the tier-S reproductions (scoring with library calls only).

Each fixture holds the forecasts a study scored, on the keys it scored them on (its own masks, the same for every variant), so
that the library's tables are computed on exactly the study's sample. `records(frame, column)` turns one variant's column into
the records a run would write: `run_date` the issue day (the target day minus the horizon), `target_date`, `hour` the period of
the day, `horizon`, `prediction`, `actual` and `freq`. The fixtures and their builders are under `tests/fixtures/`.
"""

import functools
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@functools.lru_cache(maxsize=None)
def _parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / name)


@functools.lru_cache(maxsize=None)
def tables(study: str) -> dict:
    return json.loads((FIXTURES / study / "tables.json").read_text())


def records(frame: pd.DataFrame, column: str, period: str, freq: str, actual: str = "actual") -> List[Dict]:
    """One variant's forecasts as result records at horizon 1: issued on the day before the target day."""
    target = pd.to_datetime(frame["target_date"])
    run = target - pd.Timedelta(days=1)
    columns = zip(
        run.dt.strftime("%Y-%m-%d").tolist(),
        target.dt.strftime("%Y-%m-%d").tolist(),
        run.dt.dayofweek.tolist(),
        frame[period].astype(np.int64).tolist(),
        (run - run.min()).dt.days.tolist(),
        frame[column].astype(float).tolist(),
        frame[actual].astype(float).tolist(),
    )
    return [
        {
            "run_date": r,
            "target_date": t,
            "run_weekday": w,
            "hour": h,
            "horizon": 1,
            "day_in_test": i,
            "prediction": p,
            "actual": a,
            "freq": freq,
        }
        for r, t, w, h, i, p, a in columns
    ]


def resolution(zone: str) -> pd.DataFrame:
    """The R1 honest forecasts of one zone: `target_date`, `slot` (0-95), `actual` and one column per variant."""
    frame = _parquet("resolution/r1_honest.parquet")
    return frame[frame["zone"] == zone].reset_index(drop=True)


def resolution_records(zone: str) -> Dict[str, List[Dict]]:
    frame = resolution(zone)
    variants = [c for c in frame.columns if c not in ("zone", "target_date", "slot", "actual")]
    return {v: records(frame, v, "slot", "15min") for v in variants}


def renewables(name: str) -> pd.DataFrame:
    """One table of the renewables fixture (`prices`, `generation`)."""
    return _parquet(f"renewables/{name}.parquet")


def printed_equal(value: float, cell: float, digits: int, printout: int) -> bool:
    """Whether `value` is the cell a report prints with `digits` decimals: rounded once, or rounded half up from the
    `printout` decimals the study's script printed, which is how the reports were written (24.1648 printed as 24.165 by
    `round(3)`, then 24.17)."""
    if abs(value - cell) <= 0.5 * 10**-digits + 1e-9:
        return True
    printed = Decimal(f"{value:.{printout}f}").quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return float(printed) == cell
