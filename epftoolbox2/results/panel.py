"""`PanelReport`: the reports of a run over several zones (`ModelPipeline.run(data={zone: frame})`), one per zone.

Each zone's `EvaluationReport` is that zone's alone: its own forecasts, counts, availability listing and tables, exactly what a
run of that zone's frame alone gives a plain model. The panel puts them side by side (`summary()`, `forecast_counts()` with a
`zone` column) and scores models across zones the way a multi-market study does: `log_gain`, the mean over cells of
100 ln(MAE of the benchmark / MAE of the model), each cell on the keys the two share in it, with how many there are, and
`cells_lost`, the cells where the benchmark is better.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..evaluators.base import FORECAST_KEYS
from .report import EvaluationReport

# What a cell of `log_gain` can be cut by: the zone, the run's target, and any key of a forecast but its dates.
CELL_KEYS = ("zone", "target", "horizon", "hour")
DEFAULT_CELLS = ("zone", "target", "horizon")

# The columns of `log_gain(cells=True)` after the cell's keys, and of `log_gain()`.
CELL_COLUMNS = ["model", "benchmark", "forecasts", "own_forecasts", "mae", "mae_benchmark", "log_gain"]
SUMMARY_COLUMNS = [
    "model",
    "benchmark",
    "log_gain",
    "cells",
    "cells_lost",
    "empty",
    "forecasts",
    "own_forecasts",
    "min_cell_forecasts",
]


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
        filled = [frame for frame in frames if len(frame)]
        if not filled:  # every zone's table empty: one empty table with the columns
            return frames[0].reset_index(drop=True) if frames else pd.DataFrame()
        return pd.concat(filled, ignore_index=True)

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

    def choices(self) -> pd.DataFrame:
        """Each zone's `choices()` (what its ensembles chose on validation), one row per (zone, ensemble, segment), `zone` first
        (and `target`); empty, with those columns, when nothing chose anything."""
        return self._with_keys((zone, target, report.choices()) for zone, target, report in self._parts)

    def availability(self) -> pd.DataFrame:
        """Each zone's `availability()` (the inputs known after the issue time, as the runs listed them), `zone` first (and
        `target`); a pooled model's training rows judged on an earlier zone's clock are among them, with a note."""
        return self._with_keys((zone, target, report.availability()) for zone, target, report in self._parts)

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
        common: bool = True,
    ) -> pd.DataFrame:
        """The log gain of each model over `benchmark`: the mean over cells of 100 ln(MAE of the benchmark / MAE of the model).

        A cell is one value of each key of `by` (`zone`, `target`, `horizon`, `hour`), such as a market's daily maximum at one
        horizon, and each cell weighs the same however many forecasts it has. `by=("zone", "target")` pools the horizons of a
        market and target, as the week-ahead studies report it.

        `common=True` (the default) scores every model on the same forecasts: the keys (zone, target, run date, target date,
        hour, horizon) at which the benchmark and every model listed (`models`, or all but the benchmark) hold a finite
        prediction and actual, so every model is averaged over the same cells, those with at least one such key. Leaving a model
        out of `models` can then add keys and cells to the others. `common=False` scores each model on the keys it shares with
        the benchmark alone, and its own cells; where the models' cells then differ, their means are over different cells, which
        warns and lists, per model, the cells it lacks in `attrs["unequal_cells"]`.

        A cell where the model has a zero MAE and the benchmark doesn't is a win, +inf; one where the benchmark's is zero and the
        model's isn't is lost, -inf; both zero is a tie, 0. Such cells count as scored, and the mean is then infinite (or NaN
        with both), which warns.

        Returns one row per model (the benchmark left out): `model`, `benchmark`, `log_gain` (the mean), `cells` (scored),
        `cells_lost` (cells with a negative gain), `empty` (cells some listed model or the benchmark has forecasts in where this
        model is not scored: with `common=True` the cells where any of them lacks every common key, the same for each model;
        with `common=False` those where it shares no forecast with the benchmark), `forecasts` (the forecasts it is scored on,
        summed over its scored cells), `own_forecasts` (those it holds with the benchmark at all, with a finite prediction and
        actual, over every cell: what `common=False` would score it on) and `min_cell_forecasts`, the fewest in a scored cell.
        Where `common=True` scores a model on fewer forecasts than its own, it warns, and `attrs["shared"]` holds (forecasts,
        own_forecasts) of each such model: the other models' gaps cut its sample, inside a cell too. With `cells=True`, one row
        per (model, cell) instead, every cell any of them has forecasts in: the keys of `by`, `model`, `benchmark`,
        `forecasts` (0 where not scored), `own_forecasts`, `mae`, `mae_benchmark` and `log_gain` (NaN where not scored).
        `attrs` holds `by` and `common`.
        """
        cell_keys = _check_cells(by)
        names = self._models(benchmark, models)
        table = self._cell_table(benchmark, names, cell_keys, common)
        if cells:
            return table
        rows = []
        for name in names:
            mine = table[table["model"] == name]
            scored = mine[mine["forecasts"] > 0]
            rows.append(
                {
                    "model": name,
                    "benchmark": benchmark,
                    "log_gain": float(scored["log_gain"].mean()) if len(scored) else math.nan,
                    "cells": int(len(scored)),
                    "cells_lost": int((scored["log_gain"] < 0).sum()),
                    "empty": int(len(mine) - len(scored)),
                    "forecasts": int(scored["forecasts"].sum()),
                    "own_forecasts": int(mine["own_forecasts"].sum()),
                    "min_cell_forecasts": int(scored["forecasts"].min()) if len(scored) else 0,
                }
            )
        out = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
        out.attrs.update(table.attrs)
        return out

    def cells_lost(
        self,
        benchmark: str,
        by: Sequence[str] = DEFAULT_CELLS,
        models: Optional[Sequence[str]] = None,
        common: bool = True,
    ) -> pd.DataFrame:
        """The cells of `log_gain` (scored as it scores them, `common` included) where `benchmark` has the lower MAE: one row per
        (model, cell lost), the keys of `by`, `model`, `benchmark`, `forecasts`, `mae`, `mae_benchmark`, `log_gain`.
        `log_gain(...)["cells_lost"]` counts them per model."""
        cell_keys = _check_cells(by)
        table = self._cell_table(benchmark, self._models(benchmark, models), cell_keys, common)
        lost = table[(table["forecasts"] > 0).to_numpy() & (table["log_gain"].to_numpy(dtype=float) < 0)]
        out = lost.reset_index(drop=True)
        out.attrs.update(table.attrs)
        return out

    def _models(self, benchmark: str, models: Optional[Sequence[str]]) -> List[str]:
        known = list(dict.fromkeys(name for _, _, report in self._parts for name in report.refs))
        if benchmark not in known:
            raise ValueError(f"Unknown benchmark {benchmark!r}; the panel's models are {known}")
        chosen = [name for name in known if name != benchmark] if models is None else list(models)
        unknown = [name for name in chosen if name not in known]
        if unknown:
            raise ValueError(f"Unknown model(s) {unknown}; the panel's models are {known}")
        return [name for name in chosen if name != benchmark]

    def _cell_table(
        self, benchmark: str, models: Sequence[str], cell_keys: Tuple[str, ...], common: bool
    ) -> pd.DataFrame:
        """One row per (model, cell), cells in sorted order: the cell's keys, then `CELL_COLUMNS`. A cell is any value of the keys
        some listed model or the benchmark has a finite error in. Each model is scored on the keys every listed model and the
        benchmark share (`common`), or on those it shares with the benchmark."""
        errors = self._errors([benchmark, *models])
        keys = ["zone", "target", *FORECAST_KEYS]
        values = errors[["prediction", "actual"]].to_numpy(dtype=float)
        errors = errors.assign(error=np.abs(values[:, 0] - values[:, 1]), target=errors["target"].fillna(""))
        errors = errors[np.isfinite(errors["error"].to_numpy(dtype=float))]
        try:
            wide = errors.set_index([*keys, "model"])["error"].unstack("model")
        except ValueError as exc:  # a model's frame holding one key twice
            raise ValueError(f"log_gain: a model has a forecast key twice in one zone: {exc}") from None
        wide = wide.reindex(columns=[benchmark, *models]).sort_index()
        cells_of = sorted(
            set(wide.index.to_frame(index=False)[list(cell_keys)].itertuples(index=False, name=None)), key=_sortable
        )
        place = {cell: i for i, cell in enumerate(cells_of)}
        frame = wide.index.to_frame(index=False)
        cell_of = np.array(
            [place[cell] for cell in frame[list(cell_keys)].itertuples(index=False, name=None)], dtype=np.int64
        )
        # Each cell's sums in key order (the index is sorted), so a table is the same however the records were written.
        order = np.argsort(cell_of, kind="stable")
        cell_of = cell_of[order]
        bounds = np.searchsorted(cell_of, np.arange(len(cells_of) + 1))
        found = np.isfinite(wide.to_numpy(dtype=float)[order])
        bench_errors = wide[benchmark].to_numpy(dtype=float)[order]
        shared = found.all(axis=1)
        rows, infinite = [], False
        scored_cells: Dict[str, set] = {}
        for j, name in enumerate(models, start=1):
            mine_errors = wide[name].to_numpy(dtype=float)[order]
            own = found[:, 0] & found[:, j]
            usable = shared if common else own
            for i, cell in enumerate(cells_of):
                lo, hi = int(bounds[i]), int(bounds[i + 1])
                take = np.nonzero(usable[lo:hi])[0] + lo
                n = len(take)
                n_own = int(own[lo:hi].sum())
                mae = float(np.mean(mine_errors[take])) if n else math.nan
                mae_b = float(np.mean(bench_errors[take])) if n else math.nan
                gain = _gain(mae_b, mae) if n else math.nan
                infinite = infinite or (n > 0 and not math.isfinite(gain))
                if n:
                    scored_cells.setdefault(name, set()).add(cell)
                rows.append(
                    {
                        **dict(zip(cell_keys, cell)),
                        "model": name,
                        "benchmark": benchmark,
                        "forecasts": n,
                        "own_forecasts": n_own,
                        "mae": mae,
                        "mae_benchmark": mae_b,
                        "log_gain": gain,
                    }
                )
        table = pd.DataFrame(rows, columns=[*cell_keys, *CELL_COLUMNS])
        table.attrs.update(by=cell_keys, common=common)
        if infinite:
            warnings.warn(
                "log_gain: a cell where the model's or the benchmark's MAE is zero has an infinite gain (a perfect forecast: "
                "+inf a win, -inf a loss), so its model's mean is infinite. log_gain(..., cells=True) shows which.",
                UserWarning,
                stacklevel=3,
            )
        totals = table.groupby("model", sort=False)[["forecasts", "own_forecasts"]].sum()
        shared_of = {
            name: (int(totals.loc[name, "forecasts"]), int(totals.loc[name, "own_forecasts"]))
            for name in models
            if name in totals.index and totals.loc[name, "forecasts"] < totals.loc[name, "own_forecasts"]
        }
        if shared_of and common:
            table.attrs["shared"] = shared_of
            shown = "; ".join(f"{name} on {used} of its {own}" for name, (used, own) in shared_of.items())
            warnings.warn(
                f"log_gain: common=True scores every model on the forecasts all of them and the benchmark share, so some are "
                f"scored on fewer than their own ({shown}): another model's gaps cut their sample. own_forecasts says how many "
                "each has; common=False scores each on its own (with cells that may then differ).",
                UserWarning,
                stacklevel=3,
            )
        union = set().union(*scored_cells.values()) if scored_cells else set()
        lacking = {
            name: sorted(union - scored_cells.get(name, set()), key=_sortable)
            for name in models
            if scored_cells.get(name, set()) != union
        }
        if lacking and not common:
            table.attrs["unequal_cells"] = lacking
            shown = "; ".join(f"{name}: {len(cells)} of {len(union)}" for name, cells in lacking.items())
            warnings.warn(
                f"log_gain(common=False): the models are averaged over different cells (cells lacking: {shown}), so their "
                "means aren't comparable; attrs['unequal_cells'] lists them. The default common=True scores every model on the "
                "same keys and cells.",
                UserWarning,
                stacklevel=3,
            )
        return table


def _gain(mae_benchmark: float, mae: float) -> float:
    """100 ln(MAE of the benchmark / MAE of the model), with a zero MAE defined: +inf (the model's alone), -inf (the
    benchmark's alone), 0 (both)."""
    if mae == 0 and mae_benchmark == 0:
        return 0.0
    if mae == 0:
        return math.inf
    if mae_benchmark == 0:
        return -math.inf
    return 100.0 * math.log(mae_benchmark / mae)


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
