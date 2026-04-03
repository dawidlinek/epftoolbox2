from __future__ import annotations

from typing import Dict, Iterator, List, Tuple, Union

import pandas as pd

from ..evaluators.base import Evaluator
from .ref import ModelResultRef


class EvaluationReport:
    def __init__(
        self,
        results_or_refs: Dict[str, Union[ModelResultRef, List[Dict]]],
        evaluators: List[Evaluator],
    ):
        self.evaluators = evaluators
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
        rows = []
        for name, ref in self.refs.items():
            preds, actuals = [], []
            for r in self._iter_ref(ref, cols=["prediction", "actual"]):
                preds.append(r["prediction"])
                actuals.append(r["actual"])
            group_df = pd.DataFrame({"prediction": preds, "actual": actuals})
            rows.append({"model": name, **self._apply_evaluators(group_df)})
            del group_df, preds, actuals
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

    def _apply_evaluators(self, df: pd.DataFrame) -> Dict[str, float]:
        return {ev.name: ev.compute(df) for ev in self.evaluators}

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

        rows_summary: List[Dict] = []
        rows_by_config: Dict[str, List[Dict]] = {name: [] for name, _ in group_configs}

        for model_name, ref in self.refs.items():
            summary_preds: List[float] = []
            summary_actuals: List[float] = []
            model_buckets: Dict[str, Dict[tuple, Dict[str, list]]] = {
                name: {} for name, _ in group_configs
            }

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
                    bucket = model_buckets[config_name]
                    if key not in bucket:
                        bucket[key] = {"prediction": [], "actual": []}
                    bucket[key]["prediction"].append(pred)
                    bucket[key]["actual"].append(actual)

            group_df = pd.DataFrame({"prediction": summary_preds, "actual": summary_actuals})
            rows_summary.append({"model": model_name, **self._apply_evaluators(group_df)})
            del group_df, summary_preds, summary_actuals

            for config_name, group_keys in group_configs:
                for key, data in model_buckets[config_name].items():
                    group_df = pd.DataFrame(data)
                    row: Dict = {"model": model_name}
                    for k, v in zip(group_keys, key):
                        row[k] = v
                    row.update(self._apply_evaluators(group_df))
                    rows_by_config[config_name].append(row)
                    del group_df
                del model_buckets[config_name]

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

        rows = []
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

            for key, data in buckets.items():
                group_df = pd.DataFrame(data)
                row = {"model": name}
                for k, v in zip(group_keys, key):
                    row[k] = v
                row.update(self._apply_evaluators(group_df))
                rows.append(row)
                del group_df

            del buckets

        return pd.DataFrame(rows)
