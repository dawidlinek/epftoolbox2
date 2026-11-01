"""How many days the forecasts of `train_on="matching"` lists train on (`epftoolbox2.models.context.Use`).

Such a forecast trains on fewer rows than its window, which a list with as many predictors as days, or more, can't afford: an
OLS fit then passes through every row or has no unique solution. A regularised model may do better, and a fit can also fail,
since cross-validation needs at least `cv` rows. Nothing is refused, so this says so: a warning naming the model, how many
forecasts and the smallest count, before any fit in `BaseModel.run` and again in `ModelPipeline.report`, and
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
    """One row per list, by its first horizon and hash, so that a table doesn't depend on which list a plan met first.

    `thin_forecasts` are those with a day or more, and no more days than the list has predictors. A forecast with no day has
    no fit, and is counted in `forecast_counts()` instead."""
    by_list: Dict[str, List[Tuple[int, int, int]]] = {}
    for list_hash, period, horizon, days in uses:
        by_list.setdefault(list_hash, []).append((period, horizon, days))
    rows = []
    for list_hash in sorted(by_list, key=lambda name: (min(use[1] for use in by_list[name]), name)):
        periods, horizons, days = zip(*by_list[list_hash])
        counts = np.array(days)
        n_predictors = predictors.get(list_hash)  # none for a list a run went on over with allow_manifest_mismatch
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
                "thin_forecasts": 0 if n_predictors is None else int(((counts > 0) & (counts <= n_predictors)).sum()),
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
    """One warning per model with forecasts on no more days than predictors."""
    by_model: Dict[str, List[Mapping]] = {}
    for row in rows:
        if row["thin_forecasts"]:
            by_model.setdefault(row["model"], []).append(row)
    for model, thin in by_model.items():
        fewest = min(thin, key=lambda row: (row["smallest_days"], -row["predictors"]))
        warnings.warn(
            f"{model}: {sum(row['thin_forecasts'] for row in thin)} forecast(s) with train_on='matching' train on no more days "
            f"than their list has predictors, as few as {fewest['smallest_days']} day(s) for {fewest['predictors']} "
            f"predictors (list {fewest['list']}). They aren't refused, but an OLS fit then passes through every row or has no "
            "unique solution, and a fit can fail: cross-validation needs at least `cv` rows, and a smaller set is counted as "
            "fit_error in forecast_counts(). report.training_days() has the counts.",
            UserWarning,
            stacklevel=stacklevel,
        )


__all__ = ["COLUMNS", "rows_of_refs", "training_day_frame", "training_day_rows", "warn_about_thin"]
