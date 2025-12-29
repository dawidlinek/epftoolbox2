from typing import Dict, List
import pandas as pd

from ..evaluators.base import Evaluator


class EvaluationReport:
    def __init__(self, results: Dict[str, List[Dict]], evaluators: List[Evaluator]):
        self.evaluators = evaluators
        self.results: Dict[str, pd.DataFrame] = {}

        for name, records in results.items():
            df = pd.DataFrame(records)
            df["year"] = pd.to_datetime(df["target_date"]).dt.year
            self.results[name] = df

    def _apply_evaluators(self, df: pd.DataFrame) -> Dict[str, float]:
        return {ev.name: ev.compute(df) for ev in self.evaluators}

    def summary(self) -> pd.DataFrame:
        rows = []
        for model, df in self.results.items():
            row = {"model": model, **self._apply_evaluators(df)}
            rows.append(row)
        return pd.DataFrame(rows)

    def by_hour(self) -> pd.DataFrame:
        return self._group_by("hour")

    def by_horizon(self) -> pd.DataFrame:
        return self._group_by("horizon")

    def by_hour_horizon(self) -> pd.DataFrame:
        rows = []
        for model, df in self.results.items():
            for (hour, horizon), grp in df.groupby(["hour", "horizon"]):
                row = {"model": model, "hour": hour, "horizon": horizon, **self._apply_evaluators(grp)}
                rows.append(row)
        return pd.DataFrame(rows)

    def by_year(self) -> pd.DataFrame:
        return self._group_by("year")

    def by_year_horizon(self) -> pd.DataFrame:
        rows = []
        for model, df in self.results.items():
            for (year, horizon), grp in df.groupby(["year", "horizon"]):
                row = {"model": model, "year": year, "horizon": horizon, **self._apply_evaluators(grp)}
                rows.append(row)
        return pd.DataFrame(rows)

    def _group_by(self, col: str) -> pd.DataFrame:
        rows = []
        for model, df in self.results.items():
            for val, grp in df.groupby(col):
                row = {"model": model, col: val, **self._apply_evaluators(grp)}
                rows.append(row)
        return pd.DataFrame(rows)
