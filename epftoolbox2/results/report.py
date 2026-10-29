from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd

from ..evaluators.base import FORECAST_KEYS, Evaluator
from .ref import ModelResultRef

# (name in compute_all(), grouping columns). `year` is the year of target_date; `run_weekday` and `target_weekday`
# are the weekday of run_date / target_date.
GROUP_CONFIGS: List[Tuple[str, List[str]]] = [
    ("hour", ["hour"]),
    ("horizon", ["horizon"]),
    ("hour_horizon", ["hour", "horizon"]),
    ("year", ["year"]),
    ("year_horizon", ["year", "horizon"]),
    ("run_weekday_horizon", ["run_weekday", "horizon"]),
    ("target_weekday_horizon", ["target_weekday", "horizon"]),
]


class EvaluationReport:
    def __init__(
        self,
        results_or_refs: Dict[str, Union[ModelResultRef, List[Dict]]],
        evaluators: List[Evaluator],
        source_data: Optional[pd.DataFrame] = None,
    ):
        self.evaluators = evaluators
        self.source_data = source_data
        self.refs: Dict[str, ModelResultRef] = {}

        for name, v in results_or_refs.items():
            if isinstance(v, ModelResultRef):
                self.refs[name] = v
            else:
                self.refs[name] = ModelResultRef(
                    name=name,
                    path=None,
                    count=len(v),
                    test_start="",
                    test_end="",
                    horizon=0,
                    _results=v,
                )

    def summary(self) -> pd.DataFrame:
        return self._summary_table(self._model_frames())

    def by_hour(self) -> pd.DataFrame:
        return self._compute_grouped(["hour"])

    def by_horizon(self) -> pd.DataFrame:
        return self._compute_grouped(["horizon"])

    def by_hour_horizon(self) -> pd.DataFrame:
        return self._compute_grouped(["hour", "horizon"])

    def by_year(self) -> pd.DataFrame:
        return self._compute_grouped(["year"])

    def by_year_horizon(self) -> pd.DataFrame:
        return self._compute_grouped(["year", "horizon"])

    def by_run_weekday_horizon(self) -> pd.DataFrame:
        return self._compute_grouped(["run_weekday", "horizon"])

    def by_target_weekday_horizon(self) -> pd.DataFrame:
        return self._compute_grouped(["target_weekday", "horizon"])

    def predictions(self) -> pd.DataFrame:
        """Return a tidy DataFrame of predictions only (no actuals required)."""
        cols = ["run_date", "target_date", "hour", "horizon", "prediction"]
        rows = []
        for model_name, ref in self.refs.items():
            for r in self._iter_ref(ref, cols=cols):
                row = {k: r[k] for k in cols if k in r}
                row["model"] = model_name
                rows.append(row)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values(["model", "horizon", "hour"]).reset_index(drop=True)

    def forecast_counts(self) -> pd.DataFrame:
        """Planned vs produced forecasts per model, with one column per skip reason.

        Models are only comparable on the forecasts they all produced; check this table before comparing.
        `planned` equals `produced` for results passed in as plain lists.
        """
        reasons = sorted({reason for ref in self.refs.values() for reason in ref.skipped})
        rows = []
        for name, ref in self.refs.items():
            produced = ref.produced
            if produced is None:
                produced = sum(1 for _ in self._iter_ref(ref, cols=["hour"]))
            planned = max(ref.count, produced)
            row = {"model": name, "planned": planned, "produced": produced, "missing": planned - produced}
            row.update({reason: ref.skipped.get(reason, 0) for reason in reasons})
            rows.append(row)
        return pd.DataFrame(rows, columns=["model", "planned", "produced", "missing", *reasons])

    def iter_details(self) -> Iterator[Tuple[str, pd.DataFrame]]:
        detail_cols = ["run_date", "target_date", "hour", "horizon", "day_in_test", "actual", "prediction"]
        for name, ref in self.refs.items():
            rows = [{k: r[k] for k in detail_cols if k in r} for r in self._iter_ref(ref, cols=detail_cols)]
            df = pd.DataFrame(rows)
            yield name, df
            del df, rows

    def _apply_evaluators(self, df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        return {ev.name: ev.compute(df, **kwargs) for ev in self.evaluators}

    def _iter_ref(self, ref: ModelResultRef, cols: List[str] = None) -> Iterator[Dict]:
        # A resumed JSONL can hold rows from an earlier run with a wider or different test window (its
        # rows are never deleted); filter to this ref's own window so a narrower resume isn't scored on
        # forecasts outside it. Refs built from plain result lists have no window (test_start/test_end
        # are "") and are read as given.
        filter_dates = bool(ref.test_start and ref.test_end)

        def in_window(r: Dict) -> bool:
            return not filter_dates or ref.test_start <= r["run_date"] <= ref.test_end

        if ref.path is not None:
            import json
            from pathlib import Path

            p = Path(ref.path)
            if p.exists():
                with open(p) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            r = json.loads(line)
                            if in_window(r):
                                yield {k: r[k] for k in cols if k in r} if cols else r
        else:
            for r in ref._results or []:
                if in_window(r):
                    yield {k: r[k] for k in cols if k in r} if cols else r

    def compute_all(self) -> Dict[str, pd.DataFrame]:
        model_dfs = self._model_frames()
        result: Dict[str, pd.DataFrame] = {"summary": self._summary_table(model_dfs)}
        for config_name, group_keys in GROUP_CONFIGS:
            result[config_name] = self._grouped_table(model_dfs, group_keys)
        return result

    def _compute_grouped(self, group_keys: List[str]) -> pd.DataFrame:
        return self._grouped_table(self._model_frames(), group_keys)

    def _model_frames(self) -> Dict[str, pd.DataFrame]:
        """One DataFrame per model, read once: `FORECAST_KEYS`, `prediction` and `actual`.

        This is what evaluators receive (whole, or cut into groups). `run_date` and `target_date` are
        datetime64 and `hour` / `horizon` integers, which keeps the frames compact. A key that no record of
        the model carries is left out of its frame; evaluators that need it (rMAE) say so.
        """
        return {name: self._model_frame(ref) for name, ref in self.refs.items()}

    def _model_frame(self, ref: ModelResultRef) -> pd.DataFrame:
        columns = [*FORECAST_KEYS, "prediction", "actual"]
        values: Dict[str, list] = {c: [] for c in columns}
        for r in self._iter_ref(ref, cols=columns):
            for c in columns:
                values[c].append(r.get(c))
        frame: Dict[str, Union[list, pd.Series]] = {}
        for c in columns:
            if c in ("prediction", "actual"):
                frame[c] = values[c]
            elif any(v is not None for v in values[c]):
                frame[c] = (
                    pd.to_datetime(pd.Series(values[c], dtype="object"), format="ISO8601")
                    if c.endswith("_date")
                    else values[c]
                )
        return pd.DataFrame(frame)

    def _summary_table(self, model_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        rows = []
        for name, df in model_dfs.items():
            rows.append({"model": name, "forecasts": len(df), **self._apply_evaluators(df, model_dfs=model_dfs)})
        return pd.DataFrame(rows)

    def _grouped_table(self, model_dfs: Dict[str, pd.DataFrame], group_keys: List[str]) -> pd.DataFrame:
        """One row per (group, model): each model's evaluators run on its own rows of the group, with every
        model's rows of that group as `model_dfs`. Rows are ordered by group, then by model."""
        groups = {name: dict(self._split(df, group_keys)) for name, df in model_dfs.items()}
        rows = []
        for key in sorted({k for g in groups.values() for k in g}):
            group_dfs = {name: g[key] for name, g in groups.items() if key in g}
            for name, df in group_dfs.items():
                row = {"model": name, **dict(zip(group_keys, key))}
                row.update(self._apply_evaluators(df, model_dfs=group_dfs))
                rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _split(df: pd.DataFrame, group_keys: List[str]) -> Iterator[Tuple[tuple, pd.DataFrame]]:
        """Cut one model's frame by `group_keys`. Rows without a value for a key take no part in the grouping."""
        columns = []
        for k in group_keys:
            if k == "year":
                column = df["target_date"].dt.year if "target_date" in df else None
            elif k == "run_weekday":
                column = df["run_date"].dt.dayofweek if "run_date" in df else None
            elif k == "target_weekday":
                column = df["target_date"].dt.dayofweek if "target_date" in df else None
            else:
                column = df[k] if k in df else None
            if column is None:
                return
            columns.append(column.rename(k))
        for key, part in df.groupby(columns if len(columns) > 1 else columns[0], sort=False):
            key = key if isinstance(key, tuple) else (key,)
            yield tuple(v.item() if hasattr(v, "item") else v for v in key), part
