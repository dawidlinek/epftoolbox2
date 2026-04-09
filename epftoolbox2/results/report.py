from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd

from ..evaluators.base import Evaluator
from .ref import ModelResultRef


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
        model_dfs: Dict[str, pd.DataFrame] = {}
        for name, ref in self.refs.items():
            preds, actuals = [], []
            for r in self._iter_ref(ref, cols=["prediction", "actual"]):
                preds.append(r["prediction"])
                actuals.append(r["actual"])
            model_dfs[name] = pd.DataFrame({"prediction": preds, "actual": actuals})
            del preds, actuals

        rows = []
        for name, df in model_dfs.items():
            rows.append({"model": name, **self._apply_evaluators(df, model_dfs=model_dfs)})
        return pd.DataFrame(rows)

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

    def iter_details(self) -> Iterator[Tuple[str, pd.DataFrame]]:
        detail_cols = ["run_date", "target_date", "hour", "horizon", "day_in_test", "actual", "prediction"]
        for name, ref in self.refs.items():
            rows = [
                {k: r[k] for k in detail_cols if k in r}
                for r in self._iter_ref(ref, cols=detail_cols)
            ]
            df = pd.DataFrame(rows)
            yield name, df
            del df, rows

    def _apply_evaluators(self, df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        return {ev.name: ev.compute(df, **kwargs) for ev in self.evaluators}

    def _iter_ref(self, ref: ModelResultRef, cols: List[str] = None) -> Iterator[Dict]:
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
                            yield {k: r[k] for k in cols if k in r} if cols else r
        else:
            for r in (ref._results or []):
                yield {k: r[k] for k in cols if k in r} if cols else r

    def compute_all(self) -> Dict[str, pd.DataFrame]:
        group_configs = [
            ("hour", ["hour"]),
            ("horizon", ["horizon"]),
            ("hour_horizon", ["hour", "horizon"]),
            ("year", ["year"]),
            ("year_horizon", ["year", "horizon"]),
            ("run_weekday_horizon", ["run_weekday", "horizon"]),
            ("target_weekday_horizon", ["target_weekday", "horizon"]),
        ]

        summary_dfs: Dict[str, pd.DataFrame] = {}
        # config_name -> model_name -> {group_key -> {prediction: [], actual: []}}
        all_buckets: Dict[str, Dict[str, Dict[tuple, Dict[str, list]]]] = {
            name: {} for name, _ in group_configs
        }

        for model_name, ref in self.refs.items():
            summary_preds: List[float] = []
            summary_actuals: List[float] = []
            for config_name, _ in group_configs:
                all_buckets[config_name][model_name] = {}

            cols_to_read = ["hour", "horizon", "run_date", "target_date", "prediction", "actual"]
            for r in self._iter_ref(ref, cols=cols_to_read):
                pred = r["prediction"]
                actual = r["actual"]
                year = int(r["target_date"][:4])
                r["run_weekday"] = pd.Timestamp(r["run_date"]).dayofweek
                r["target_weekday"] = pd.Timestamp(r["target_date"]).dayofweek

                summary_preds.append(pred)
                summary_actuals.append(actual)

                for config_name, group_keys in group_configs:
                    try:
                        key = tuple(
                            year if k == "year" else r[k]
                            for k in group_keys
                        )
                    except KeyError:
                        continue
                    bucket = all_buckets[config_name][model_name]
                    if key not in bucket:
                        bucket[key] = {"prediction": [], "actual": []}
                    bucket[key]["prediction"].append(pred)
                    bucket[key]["actual"].append(actual)

            summary_dfs[model_name] = pd.DataFrame(
                {"prediction": summary_preds, "actual": summary_actuals}
            )
            del summary_preds, summary_actuals

        rows_summary = []
        for model_name, df in summary_dfs.items():
            rows_summary.append(
                {"model": model_name, **self._apply_evaluators(df, model_dfs=summary_dfs)}
            )

        rows_by_config: Dict[str, List[Dict]] = {name: [] for name, _ in group_configs}
        for config_name, group_keys in group_configs:
            model_buckets = all_buckets[config_name]
            all_keys = sorted({k for b in model_buckets.values() for k in b})
            for key in all_keys:
                model_dfs = {
                    name: pd.DataFrame(model_buckets[name][key])
                    for name in model_buckets
                    if key in model_buckets[name]
                }
                for name, df in model_dfs.items():
                    row: Dict = {"model": name}
                    for k, v in zip(group_keys, key):
                        row[k] = v
                    row.update(self._apply_evaluators(df, model_dfs=model_dfs))
                    rows_by_config[config_name].append(row)

        result: Dict[str, pd.DataFrame] = {"summary": pd.DataFrame(rows_summary)}
        for config_name, _ in group_configs:
            result[config_name] = pd.DataFrame(rows_by_config[config_name])
        return result

    def _compute_grouped(self, group_keys: List[str]) -> pd.DataFrame:
        needs_year = "year" in group_keys
        needs_run_weekday = "run_weekday" in group_keys
        needs_target_weekday = "target_weekday" in group_keys
        read_keys = [k for k in group_keys if k not in ("year", "run_weekday", "target_weekday")]
        if needs_year or needs_target_weekday:
            read_keys.append("target_date")
        if needs_run_weekday:
            read_keys.append("run_date")
        cols_to_read = list(dict.fromkeys(read_keys + ["prediction", "actual"]))

        all_buckets: Dict[str, Dict[tuple, Dict[str, list]]] = {}
        for name, ref in self.refs.items():
            buckets: Dict[tuple, Dict[str, list]] = {}
            for r in self._iter_ref(ref, cols=cols_to_read):
                if needs_year:
                    r["year"] = int(r["target_date"][:4])
                if needs_run_weekday:
                    r["run_weekday"] = pd.Timestamp(r["run_date"]).dayofweek
                if needs_target_weekday:
                    r["target_weekday"] = pd.Timestamp(r["target_date"]).dayofweek
                try:
                    key = tuple(r[k] for k in group_keys)
                except KeyError:
                    continue
                if key not in buckets:
                    buckets[key] = {"prediction": [], "actual": []}
                buckets[key]["prediction"].append(r["prediction"])
                buckets[key]["actual"].append(r["actual"])
            all_buckets[name] = buckets

        rows = []
        all_keys = sorted({k for b in all_buckets.values() for k in b})
        for key in all_keys:
            model_dfs = {
                name: pd.DataFrame(all_buckets[name][key])
                for name in all_buckets
                if key in all_buckets[name]
            }
            for name, df in model_dfs.items():
                row = {"model": name}
                for k, v in zip(group_keys, key):
                    row[k] = v
                row.update(self._apply_evaluators(df, model_dfs=model_dfs))
                rows.append(row)

        return pd.DataFrame(rows)
