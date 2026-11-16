"""What one live call of `ModelPipeline.forecast()` gives (step 8e): `ForecastResult`, its issue status and exit code, and the
export of the delivered forecasts on the market's products or on the forecast periods.

**Statuses** (decision 9 and 11 of the step 8 spec): an issue's status is the call's, never a record's.

| status | exit | meaning |
|---|---|---|
| `ok` | 0 | every target day of the delivered model from rung 0 (also an `ensure` no-op on such an issue, an unchanged rerun of it) |
| `fallback` | 10 | complete, some target day from a lower rung of a `FallbackEnsemble` |
| `skipped` | 20 | a planned key of the delivered model has no forecast |
| `late` | 30 | the deadline passed before a revision could be committed: nothing written but the call line |
| `busy` | 75 | another call holds `live.lock` (EX_TEMPFAIL): nothing written |
| `error` | 1 | an exception, raised to the caller after the call line says so |

**Export grids.** `"periods"` writes one row per forecast period as the records hold them (24 hours, 96 quarter-hours, the
repeated autumn hour one period). `"products"` writes one row per market product, the instants the clock gives the target day
(`results.checks.day_products`: 92, 96 or 100 quarter-hours, 23, 24 or 25 hours), with its UTC start, local date, slot (the
period) and occurrence (1, or 2 for the repeated hour's second occurrence), so the repeated hour's forecast is written twice and
nobody fills it in by hand.
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ..models._calendar import FREQ_STEPS
from .store import STATUS_OK

OK = "ok"
FALLBACK = "fallback"
SKIPPED = "skipped"
LATE = "late"
BUSY = "busy"
ERROR = "error"
EXIT_CODES = {OK: 0, FALLBACK: 10, SKIPPED: 20, LATE: 30, BUSY: 75, ERROR: 1}
STATUSES = tuple(EXIT_CODES)

FORMATS = ("csv", "json")
GRIDS = ("products", "periods")


def exit_code(status: str) -> int:
    """The exit code of an issue status (the table in the module docstring)."""
    if status not in EXIT_CODES:
        raise ValueError(f"unknown issue status {status!r}: one of {list(EXIT_CODES)}")
    return EXIT_CODES[status]


def _is_forecast(record: Dict[str, Any]) -> bool:
    return record.get("status", STATUS_OK) == STATUS_OK and record.get("prediction") is not None


def products(day: Any, freq: str, zone: Any) -> List[Tuple[pd.Timestamp, int, int]]:
    """The market products of local date `day` at `freq` in `zone`: (UTC start, slot, occurrence) in time order. The slot is the
    period id a record carries (the wall-clock hour or quarter-hour of the day), the occurrence 1, or 2 for the second
    occurrence of a repeated autumn slot. A spring day lacks the skipped slots. `"1D"` has one product, the day, starting at its
    local midnight; without a zone every period once, on the naive clock."""
    if freq not in FREQ_STEPS:
        raise ValueError(f"Unsupported freq '{freq}'. Supported: {list(FREQ_STEPS)}")
    date = day if isinstance(day, datetime.date) else datetime.date.fromisoformat(str(day)[:10])
    first, last = pd.Timestamp(date), pd.Timestamp(date + datetime.timedelta(days=1))
    if zone is not None:
        first = first.tz_localize(zone, ambiguous=True, nonexistent="shift_forward")
        last = last.tz_localize(zone, ambiguous=True, nonexistent="shift_forward")
    if freq == "1D":
        return [(first.tz_convert("UTC") if zone is not None else first, 0, 1)]
    step = FREQ_STEPS[freq]
    minutes = int(step / pd.Timedelta(minutes=1))
    instants = pd.date_range(first, last, freq=step, inclusive="left")
    seen: Dict[int, int] = {}
    out = []
    for instant in instants:
        slot = (instant.hour * 60 + instant.minute) // minutes
        seen[slot] = seen.get(slot, 0) + 1
        out.append((instant.tz_convert("UTC") if zone is not None else instant, slot, seen[slot]))
    return out


@dataclass
class ForecastResult:
    """One call of `ModelPipeline.forecast()`.

    Attributes:
        issue: The issue day D (ISO date) in the issue zone.
        status: `ok`, `fallback`, `skipped`, `late`, `busy` (see the module docstring); `exit_code` its exit code.
        revision: The revision committed (or, for `unchanged` and `noop`, the committed revision it stands for; for a dry run
            the number a commit would take). None when nothing is or would be committed.
        unchanged: The forecasts equal the latest committed revision's: no revision was written (decision 2).
        noop: `ensure=True` found a committed revision of D with status ok or fallback before the deadline: nothing was done.
        dry_run: Nothing was written anywhere (decision 6).
        delivered: The model or ensemble whose forecasts decide the status (`deliver=`, else the pipeline's last ensemble, else its
            last model) and which `frame()` and `export()` give.
        records: Each model's and ensemble's records of the issue as a revision writes them (live keys aside), forecasts and skip
            records, by name.
        models: Per model, how its forecasts were made: `source` ("stored fits": its fits stored or inline, `models._fits.serve`;
            "run": its own run of the issue day), `fits` (per anchor the fit used, its age and what it was made from), `notes`,
            `forecasts` and `skipped` (counts by reason).
        refresh: The data pipeline's `RefreshRecord` of the last refresh (None for a frame used as given).
        waits: Every poll of `wait_for`, in order.
        checks: `checks.json` of the issue (the checks before export over the members' records), None without checks.
        snapshot: The sha256 of the issue's input snapshot, which every record of the revision carries.
        planned: How many keys of D the delivered model plans; `missing` how many of them have no forecast; `rungs` the rung of
            each target day of a `FallbackEnsemble` delivered ({target_date: rung}).
        call: The line written to `issues.jsonl` (None when none was written).
    """

    issue: str
    status: str
    exit_code: int
    revision: Optional[int] = None
    unchanged: bool = False
    noop: bool = False
    dry_run: bool = False
    delivered: Optional[str] = None
    records: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict, repr=False)
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)
    refresh: Any = field(default=None, repr=False)
    waits: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    checks: Optional[Dict[str, Any]] = field(default=None, repr=False)
    snapshot: Optional[str] = None
    planned: Optional[int] = None
    missing: Optional[int] = None
    rungs: Dict[str, int] = field(default_factory=dict, repr=False)
    # (period, horizon) of every key the issue plans (`IssueCalendar`): the export writes each, with or without a record.
    planned_keys: Optional[List[Tuple[int, int]]] = field(default=None, repr=False)
    freq: str = "1h"
    zone: Any = None
    call: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def frame(self, model: Optional[str] = None) -> pd.DataFrame:
        """The records of `model` (the delivered one by default) as a DataFrame, one row per key: `run_date`, `target_date`,
        `hour` (the period), `horizon`, `prediction` (None for a skip record), `status`, and `rung`/`member` for a fallback's."""
        name = self.delivered if model is None else model
        if name not in self.records:
            raise KeyError(f"no records of {name!r} in this result: {sorted(self.records)}")
        rows = [
            {
                "run_date": record["run_date"],
                "target_date": record["target_date"],
                "hour": int(record["hour"]),
                "horizon": int(record["horizon"]),
                "prediction": record.get("prediction") if _is_forecast(record) else None,
                "status": record.get("status", STATUS_OK),
                **({key: record[key] for key in ("rung", "member") if key in record}),
            }
            for record in self.records[name]
        ]
        columns = ["run_date", "target_date", "hour", "horizon", "prediction", "status"]
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=columns)
        return frame.sort_values(["horizon", "target_date", "hour"], kind="stable").reset_index(drop=True)

    def export(
        self,
        path: Union[str, os.PathLike],
        format: str = "csv",
        grid: str = "products",
        model: Optional[str] = None,
    ) -> Path:
        """Write the delivered model's forecasts (or `model`'s) to `path` as `"csv"` or `"json"` (a list of objects), on the grid
        `"products"` (one row per market product: `target_date`, `slot`, `occurrence`, `start_utc`, `horizon`, `prediction`,
        `status`, so the repeated autumn hour's forecast is written for both of its occurrences: 100 quarter-hours that day) or
        `"periods"` (one row per period, as the records hold them). A key without a forecast is written with an empty
        prediction and its status, a planned key without any record with status "missing". Returns the path. Writing is the user's: nothing is delivered (decision 1)."""
        if format not in FORMATS:
            raise ValueError(f"format must be one of {list(FORMATS)}, got {format!r}")
        if grid not in GRIDS:
            raise ValueError(f"grid must be one of {list(GRIDS)}, got {grid!r}")
        table = self.table(grid, model)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if format == "csv":
            table.to_csv(path, index=False)
        else:
            rows = table.astype(object).where(table.notna(), None).to_dict(orient="records")
            path.write_text(json.dumps(rows, indent=1, default=str) + "\n", encoding="utf-8")
        return path

    def _target_date(self, horizon: int) -> str:
        return (datetime.date.fromisoformat(self.issue) + datetime.timedelta(days=int(horizon))).isoformat()

    def _planned_frame(self, periods: pd.DataFrame) -> pd.DataFrame:
        """`periods` with a row (no prediction, status "missing") for each planned key without a record."""
        if self.planned_keys is None:
            return periods
        held = {(int(row.hour), int(row.horizon)) for row in periods.itertuples(index=False)}
        lacking = [
            {
                "run_date": self.issue,
                "target_date": self._target_date(h),
                "hour": int(p),
                "horizon": int(h),
                "prediction": None,
                "status": "missing",
            }
            for p, h in self.planned_keys
            if (int(p), int(h)) not in held
        ]
        if not lacking:
            return periods
        out = pd.concat([periods, pd.DataFrame(lacking)], ignore_index=True) if len(periods) else pd.DataFrame(lacking)
        return out.sort_values(["horizon", "target_date", "hour"], kind="stable").reset_index(drop=True)

    def table(self, grid: str = "products", model: Optional[str] = None) -> pd.DataFrame:
        """What `export` writes, as a DataFrame."""
        periods = self._planned_frame(self.frame(model))
        if grid == "periods":
            return periods.rename(columns={"hour": "period"})
        if grid != "products":
            raise ValueError(f"grid must be one of {list(GRIDS)}, got {grid!r}")
        by_key = {
            (str(row.target_date), int(row.horizon), int(row.hour)): row for row in periods.itertuples(index=False)
        }
        days = sorted({(str(row.target_date), int(row.horizon)) for row in periods.itertuples(index=False)})
        if self.planned_keys is not None:  # every planned target day, whether or not a record of it was made
            days = sorted({*days, *((self._target_date(h), int(h)) for _, h in self.planned_keys)})
        extra = [column for column in ("rung", "member") if column in periods.columns]
        rows = []
        for target_date, horizon in days:
            for start, slot, occurrence in products(target_date, self.freq, self.zone):
                row = by_key.get((target_date, horizon, slot))
                rows.append(
                    {
                        "target_date": target_date,
                        "slot": slot,
                        "occurrence": occurrence,
                        "start_utc": start.isoformat(),
                        "horizon": horizon,
                        "prediction": None if row is None else row.prediction,
                        "status": "missing" if row is None else row.status,
                        **{column: (None if row is None else getattr(row, column)) for column in extra},
                    }
                )
        return pd.DataFrame(
            rows,
            columns=[
                "target_date",
                "slot",
                "occurrence",
                "start_utc",
                "horizon",
                "prediction",
                "status",
                *extra,
            ],
        )


__all__ = [
    "BUSY",
    "ERROR",
    "EXIT_CODES",
    "FALLBACK",
    "ForecastResult",
    "GRIDS",
    "LATE",
    "OK",
    "SKIPPED",
    "STATUSES",
    "exit_code",
    "products",
]
