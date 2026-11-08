"""`PanelReport`: the reports of a run over several zones (`ModelPipeline.run(data={zone: frame})`), one per zone.

Each zone's `EvaluationReport` is that zone's alone: its own forecasts, counts, availability listing and tables, exactly what a
run of that zone's frame alone gives a plain model. The panel puts them side by side (`summary()`, `forecast_counts()` with a
`zone` column) and scores models across zones the way a multi-market study does: `log_gain`, the mean over cells of
100 ln(MAE of the benchmark / MAE of the model), each cell on the keys the two share in it, with how many there are, and
`cells_lost`, the cells where the benchmark is better.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..evaluators.base import FORECAST_KEYS
from .report import EvaluationReport

# What a cell of `log_gain` can be cut by: the zone, the run's target, and any key of a forecast but its dates.
CELL_KEYS = ("zone", "target", "horizon", "hour")
DEFAULT_CELLS = ("zone", "target", "horizon")

# The columns of `log_gain(cells=True)`.
CELL_COLUMNS = ["model", "benchmark", "forecasts", "mae", "mae_benchmark", "log_gain"]


class PanelReport:
    """The reports of one run over several zones, or of several such runs (`concat`), one `EvaluationReport` per (zone, target).

    Args:
        zones: Zone name -> that zone's report, in the run's order.
        target: The target the run forecast (`run(target=...)`), which `log_gain` can cut by.
    """

    def __init__(self, zones: Mapping[str, EvaluationReport], target: Optional[str] = None):
        if not zones:
            raise ValueError("A PanelReport needs the report of at least one zone")
        self._parts: List[Tuple[str, Optional[str], EvaluationReport]] = [
            (str(zone), target, report) for zone, report in zones.items()
        ]

    @classmethod
    def concat(cls, panels: Iterable["PanelReport"]) -> "PanelReport":
        """One panel of several, such as the runs of a daily minimum and a daily maximum: its tables have every (zone, target)
        of each, and `log_gain` can cut by `target`. A (zone, target) in two of them raises."""
        parts = [part for panel in panels for part in panel._parts]
        seen = [(zone, target) for zone, target, _ in parts]
        repeated = sorted({key for key in seen if seen.count(key) > 1}, key=str)
        if repeated:
            raise ValueError(f"PanelReport.concat: these (zone, target) are in more than one panel: {repeated}")
        if not parts:
            raise ValueError("PanelReport.concat needs at least one panel")
        out = cls.__new__(cls)
        out._parts = parts
        return out

    @property
    def zones(self) -> Dict[str, EvaluationReport]:
        """Zone -> its report. A panel of several targets (`concat`) has more than one report per zone: use `reports`."""
        zones = [zone for zone, _, _ in self._parts]
        if len(set(zones)) != len(zones):
            raise ValueError("This panel holds several targets per zone: use .reports, keyed by (zone, target)")
        return {zone: report for zone, _, report in self._parts}

    @property
    def reports(self) -> Dict[Tuple[str, Optional[str]], EvaluationReport]:
        """(zone, target) -> its report, in the panel's order."""
        return {(zone, target): report for zone, target, report in self._parts}

    @property
    def targets(self) -> List[Optional[str]]:
        """The targets of the panel, each once, in order."""
        return list(dict.fromkeys(target for _, target, _ in self._parts))

    def __getitem__(self, zone: str) -> EvaluationReport:
        return self.zones[zone]

    def __iter__(self):
        return iter(self.zones)

    def __len__(self) -> int:
        return len(self._parts)

    def _with_keys(self, tables: Iterable[Tuple[str, Optional[str], pd.DataFrame]]) -> pd.DataFrame:
        frames = []
        several = len(self.targets) > 1 or self.targets != [None]
        for zone, target, table in tables:
            keyed = table.copy()
            keyed.insert(0, "zone", zone)
            if several:
                keyed.insert(1, "target", target)
            frames.append(keyed)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def summary(self) -> pd.DataFrame:
        """Each zone's `summary()`, one row per (zone, model): `zone` (and `target` with one), then its columns."""
        return self._with_keys((zone, target, report.summary()) for zone, target, report in self._parts)

    def forecast_counts(self) -> pd.DataFrame:
        """Each zone's `forecast_counts()`, one row per (zone, model), `zone` first (and `target`). A reason no model of a zone
        skipped for is 0 there. Compare models across zones only where they produced the same forecasts."""
        table = self._with_keys((zone, target, report.forecast_counts()) for zone, target, report in self._parts)
        reasons = [c for c in table.columns if c not in ("zone", "target", "model", "planned", "produced", "missing")]
        if reasons:
            table[reasons] = table[reasons].fillna(0).astype(int)
        return table

    def _errors(self, models: Sequence[str]) -> pd.DataFrame:
        """Every forecast of these models in every zone: `zone`, `target`, `model`, the forecast keys, `prediction`, `actual`."""
        frames = []
        for zone, target, report in self._parts:
            frames_of = report._model_frames()
            for name in models:
                frame = frames_of.get(name)
                if frame is None or not len(frame):
                    continue
                frame = frame.assign(zone=zone, target=target, model=name)
                frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=["zone", "target", "model", *FORECAST_KEYS, "prediction", "actual"])
        return pd.concat(frames, ignore_index=True)

    def log_gain(
        self,
        benchmark: str,
        by: Sequence[str] = DEFAULT_CELLS,
        models: Optional[Sequence[str]] = None,
        cells: bool = False,
    ) -> pd.DataFrame:
        """The log gain of each model over `benchmark`: the mean over cells of 100 ln(MAE of the benchmark / MAE of the model).

        A cell is one value of each key of `by` (`zone`, `target`, `horizon`, `hour`), such as a market's daily maximum at one
        horizon, and is scored on its own common keys: the forecasts (run date, target date, hour, horizon) that the model and
        the benchmark both hold with a finite prediction and actual there. So a model is never scored on forecasts the
        benchmark lacks, and each cell weighs the same however many forecasts it has. A cell where either has none is left
        out and counted as `empty`. `by=("zone", "target")` pools the horizons of a market and target, as the week-ahead studies
        report it.

        Returns one row per model (the benchmark left out): `model`, `benchmark`, `log_gain` (the mean), `cells` (scored),
        `cells_lost` (cells with a negative gain), `empty`, `forecasts` (summed over cells), and the smallest number of
        forecasts in a cell, `min_cell_forecasts`. With `cells=True`, one row per (model, cell) instead: the keys of `by`,
        `model`, `benchmark`, `forecasts`, `mae`, `mae_benchmark` and `log_gain`.
        """
        cell_keys = _check_cells(by)
        names = self._models(benchmark, models)
        table = self._cell_table(benchmark, names, cell_keys)
        if cells:
            return table
        rows = []
        for name in names:
            mine = table[table["model"] == name]
            scored = mine[np.isfinite(mine["log_gain"].to_numpy(dtype=float))]
            rows.append(
                {
                    "model": name,
                    "benchmark": benchmark,
                    "log_gain": float(scored["log_gain"].mean()) if len(scored) else math.nan,
                    "cells": int(len(scored)),
                    "cells_lost": int((scored["log_gain"] < 0).sum()),
                    "empty": int(len(mine) - len(scored)),
                    "forecasts": int(scored["forecasts"].sum()),
                    "min_cell_forecasts": int(scored["forecasts"].min()) if len(scored) else 0,
                }
            )
        out = pd.DataFrame(
            rows,
            columns=[
                "model",
                "benchmark",
                "log_gain",
                "cells",
                "cells_lost",
                "empty",
                "forecasts",
                "min_cell_forecasts",
            ],
        )
        out.attrs["by"] = cell_keys
        return out

    def cells_lost(
        self, benchmark: str, by: Sequence[str] = DEFAULT_CELLS, models: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        """The cells of `log_gain` where `benchmark` has the lower MAE: one row per (model, cell lost), the keys of `by`, `model`,
        `benchmark`, `forecasts`, `mae`, `mae_benchmark`, `log_gain`. `log_gain(...)["cells_lost"]` counts them per model."""
        cell_keys = _check_cells(by)
        table = self._cell_table(benchmark, self._models(benchmark, models), cell_keys)
        lost = table[table["log_gain"].to_numpy(dtype=float) < 0]
        return lost.reset_index(drop=True)

    def _models(self, benchmark: str, models: Optional[Sequence[str]]) -> List[str]:
        known = list(dict.fromkeys(name for _, _, report in self._parts for name in report.refs))
        if benchmark not in known:
            raise ValueError(f"Unknown benchmark {benchmark!r}; the panel's models are {known}")
        chosen = [name for name in known if name != benchmark] if models is None else list(models)
        unknown = [name for name in chosen if name not in known]
        if unknown:
            raise ValueError(f"Unknown model(s) {unknown}; the panel's models are {known}")
        return [name for name in chosen if name != benchmark]

    def _cell_table(self, benchmark: str, models: Sequence[str], cell_keys: Tuple[str, ...]) -> pd.DataFrame:
        """One row per (model, cell): the cell's keys, then `CELL_COLUMNS`, cells in sorted order. A cell is any value of the
        keys some model or the benchmark has a finite error in."""
        errors = self._errors([benchmark, *models])
        keys = ["zone", "target", *FORECAST_KEYS]
        values = errors[["prediction", "actual"]].to_numpy(dtype=float)
        errors = errors.assign(error=np.abs(values[:, 0] - values[:, 1]), target=errors["target"].fillna(""))
        errors = errors[np.isfinite(errors["error"].to_numpy(dtype=float))]
        bench = errors[errors["model"] == benchmark][[*keys, "error"]].rename(columns={"error": "benchmark_error"})
        cells_of = sorted(set(errors[list(cell_keys)].itertuples(index=False, name=None)), key=_sortable)
        place = {cell: i for i, cell in enumerate(cells_of)}
        rows = []
        for name in models:
            mine = errors[errors["model"] == name][[*keys, "error"]]
            joined = mine.merge(bench, on=keys, how="inner", validate="one_to_one")
            cell_of = np.array(
                [place[cell] for cell in joined[list(cell_keys)].itertuples(index=False, name=None)], dtype=np.int64
            )
            # Each cell's sums in key order, so a table is the same however the records were written.
            joined = joined.assign(_cell=cell_of).sort_values(["_cell", *keys], kind="stable")
            cell_of = joined["_cell"].to_numpy()
            mine_errors = joined["error"].to_numpy(dtype=float)
            bench_errors = joined["benchmark_error"].to_numpy(dtype=float)
            bounds = np.searchsorted(cell_of, np.arange(len(cells_of) + 1))
            for i, cell in enumerate(cells_of):
                lo, hi = int(bounds[i]), int(bounds[i + 1])
                n = hi - lo
                mae = float(np.mean(mine_errors[lo:hi])) if n else math.nan
                mae_b = float(np.mean(bench_errors[lo:hi])) if n else math.nan
                gain = 100.0 * math.log(mae_b / mae) if n and mae > 0 and mae_b > 0 else math.nan
                rows.append(
                    {
                        **dict(zip(cell_keys, cell)),
                        "model": name,
                        "benchmark": benchmark,
                        "forecasts": n,
                        "mae": mae,
                        "mae_benchmark": mae_b,
                        "log_gain": gain,
                    }
                )
        return pd.DataFrame(rows, columns=[*cell_keys, *CELL_COLUMNS])


def _check_cells(by: Sequence[str]) -> Tuple[str, ...]:
    keys = (by,) if isinstance(by, str) else tuple(by)
    unknown = [key for key in keys if key not in CELL_KEYS]
    if unknown or len(set(keys)) != len(keys):
        raise ValueError(f"by must name some of {list(CELL_KEYS)}, each once, got {by!r}")
    return keys


def _sortable(cell: Tuple) -> Tuple:
    return tuple(
        (value is None, str(value) if not isinstance(value, (int, float, np.integer)) else value) for value in cell
    )
