"""How many days the forecasts of `train_on="matching"` lists train on (`epftoolbox2.models.context.Use`).

Such a forecast trains on fewer rows than its window, which a list with more predictors than days can't afford: an OLS fit
then has no unique solution, though a regularised model is fine. Nothing is refused, so this says so: a warning naming the
model, how many forecasts and the smallest count, before any fit in `BaseModel.run` and again in `ModelPipeline.report`, and
`EvaluationReport.training_days()`, which the terminal exporter shows.
"""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd

from ..information_set import _ranges
from .ref import ModelResultRef

# The columns of `EvaluationReport.training_days()`.
COLUMNS = [
    "model",
    "list",
    "predictors",
    "forecasts",
    "periods",
    "horizons",
    "smallest_days",
    "median_days",
    "largest_days",
    "thin_forecasts",
]

Use = Tuple[str, int, int, int]  # (list hash, period, horizon, days it trains on)


def training_day_rows(model: str, uses: Iterable[Use], predictors: Mapping[str, int]) -> List[Dict]:
    """One row per list, in the order of `predictors`. `thin_forecasts` are those with a day or more, but fewer days than the
    list has predictors. A forecast with no day has no fit, and is counted in `forecast_counts()` instead."""
    by_list: Dict[str, List[Tuple[int, int, int]]] = {}
    for list_hash, period, horizon, days in uses:
        by_list.setdefault(list_hash, []).append((period, horizon, days))
    rows = []
    for list_hash, n_predictors in predictors.items():
        if list_hash not in by_list:
            continue
        periods, horizons, days = zip(*by_list[list_hash])
        counts = np.array(days)
        rows.append(
            {
                "model": model,
                "list": list_hash,
                "predictors": n_predictors,
                "forecasts": len(counts),
                "periods": _ranges(sorted(set(periods))),
                "horizons": _ranges(sorted(set(horizons))),
                "smallest_days": int(counts.min()),
                "median_days": float(np.median(counts)),
                "largest_days": int(counts.max()),
                "thin_forecasts": int(((counts > 0) & (counts < n_predictors)).sum()),
            }
        )
    return rows


def rows_of_refs(refs: Mapping[str, ModelResultRef]) -> List[Dict]:
    """The rows of `EvaluationReport.training_days()` from the records of these refs. Only a ref with matching lists is read."""
    rows: List[Dict] = []
    for name, ref in refs.items():
        if ref.matching_lists:
            records = ref.iter_records(cols=["hour", "horizon", "predictor_list", "training_days"])
            uses = (
                (r["predictor_list"], r["hour"], r["horizon"], r["training_days"])
                for r in records
                if "training_days" in r
            )
            rows.extend(training_day_rows(name, uses, ref.matching_lists))
    return rows


def training_day_frame(rows: Iterable[Mapping]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=COLUMNS)


def warn_about_thin(rows: Iterable[Mapping], stacklevel: int = 2) -> None:
    """One warning per model with forecasts on fewer days than predictors."""
    by_model: Dict[str, List[Mapping]] = {}
    for row in rows:
        if row["thin_forecasts"]:
            by_model.setdefault(row["model"], []).append(row)
    for model, thin in by_model.items():
        fewest = min(thin, key=lambda row: (row["smallest_days"], -row["predictors"]))
        warnings.warn(
            f"{model}: {sum(row['thin_forecasts'] for row in thin)} forecast(s) with train_on='matching' train on fewer days "
            f"than their list has predictors, as few as {fewest['smallest_days']} day(s) for {fewest['predictors']} "
            f"predictors (list {fewest['list']}). They are fitted all the same: a regularised model can do that, and OLS "
            "then has no unique solution. report.training_days() has the counts.",
            UserWarning,
            stacklevel=stacklevel,
        )


__all__ = ["COLUMNS", "rows_of_refs", "training_day_frame", "training_day_rows", "warn_about_thin"]
