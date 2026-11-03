"""Vintage classes: what the archive of each input a model read holds (step 3h).

A rule's class (`Availability(vintage=...)`) says what the frame holds of a column: `snapshot` (each value as published at a
stated time), `row_stamp` (a trustworthy stamp of each value's first publication), `last_write_stamp` (a stamp of its last
write: a leak detector, no vintage) or `latest_revision` (the newest value only). A run with an information set reads the class
of every input it uses, and of every column a predictor function reads through `ctx`, from its rule, and keeps them as rows:
on its ref (`ModelResultRef.vintages`, `EvaluationReport.vintages()`) and in its manifest (`vintages`), recorded and never
compared on resume. A run whose inputs have no class at all records nothing, so its manifest is what it always was.

Nothing here reads a number: a class moves no forecast. It says how far a backtest can be taken at its word. A latest revision
holds values written after the issue day, which no forecast saw then (in one measurement, Germany 2025, the TSO's own wind and
solar forecasts had an RMSE against actual generation 29-45 % lower in their latest revision than in their day-ahead vintage,
vintages matched by name only; the effect on a price model wasn't measured), so compared models that read one series with
different classes aren't compared like for like. They warn, and don't refuse, since all usable history of most series is a latest revision today
(decision 18 of the step 3 spec): `tests()`, `compare()`, `leaderboard()` and a table with a `DMEvaluator` column say so in a
warning and in `df.attrs["vintages"]`.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import pandas as pd

from ..information_set import _STEP_MINUTES, REVISED_VINTAGES, UNKNOWN_VINTAGE, InformationSet, vintage_of

# The columns of `EvaluationReport.vintages()`.
COLUMNS = ["model", "predictor", "series", "class", "ctx"]


def vintage_rows(
    information_set: Optional[InformationSet], freq: str, used: Iterable[str], queried: Iterable[str] = ()
) -> Optional[List[Dict[str, Any]]]:
    """One row per input of a model, `used` (its predictors) and `queried` (what a predictor function read through `ctx`):
    the name, the series its rule is for (`price` for `price_d-1`), its class and whether it was read through `ctx`. None
    without an information set, and when no input has a class: such a run records nothing."""
    if information_set is None:
        return None
    step = _STEP_MINUTES[freq]
    queried = set(queried)
    rows = []
    for name in sorted({*used, *queried}):
        col = information_set._resolve_for(name, step)
        rows.append({"predictor": name, "series": col.series, "class": vintage_of(col.rule), "ctx": name in queried})
    return rows if any(row["class"] != UNKNOWN_VINTAGE for row in rows) else None


def merge_vintages(earlier: Iterable[Mapping[str, Any]], later: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The rows of both, the later one's where both have a predictor: what a run into results kept from an earlier one adds."""
    rows = {row["predictor"]: dict(row) for row in earlier}
    rows.update({row["predictor"]: dict(row) for row in later})
    return [rows[name] for name in sorted(rows)]


def union_vintages(held: Iterable[Optional[Sequence[Mapping[str, Any]]]]) -> Optional[List[Dict[str, Any]]]:
    """Every row of these models' rows once: what an ensemble of them reads. None if none of them has any."""
    rows: Dict[tuple, Dict[str, Any]] = {}
    for one in held:
        for row in one or ():
            rows.setdefault(tuple(row[key] for key in COLUMNS[1:]), dict(row))
    return [rows[key] for key in sorted(rows, key=str)] or None


def read_vintages(path: Union[str, Path]) -> Optional[List[Dict[str, Any]]]:
    """The rows a manifest holds, or None (no manifest, or no input with a class)."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("vintages")


def vintage_frame(by_model: Mapping[str, Optional[Sequence[Mapping[str, Any]]]]) -> pd.DataFrame:
    """The rows of every model as `EvaluationReport.vintages()` shows them; no rows gives the same columns."""
    rows = [{"model": model, **row} for model, held in by_model.items() for row in held or ()]
    return pd.DataFrame(rows, columns=COLUMNS)


def is_revised_class(label: str) -> bool:
    """Whether a class as shown (one, or those of a column's rules over time joined by ", ") holds revised values."""
    return any(part in REVISED_VINTAGES for part in str(label).split(", "))


def mixed_classes(
    by_model: Mapping[str, Optional[Sequence[Mapping[str, Any]]]], groups: Iterable[Sequence[str]]
) -> Dict[str, Dict[str, str]]:
    """Series -> model -> class, for each series that two models compared in one of `groups` read with different classes.

    Only a declared class counts: an "unknown" one says nothing to compare. A model that reads a series through several
    names with different rules shows their classes joined by ", "."""
    mixed: Dict[str, Dict[str, str]] = {}
    for group in groups:
        by_series: Dict[str, Dict[str, set]] = {}
        for model in dict.fromkeys(group):
            for row in by_model.get(model) or ():
                classes = {part for part in str(row["class"]).split(", ") if part != UNKNOWN_VINTAGE}
                if classes:
                    by_series.setdefault(row["series"], {}).setdefault(model, set()).update(classes)
        for series, classes in by_series.items():
            if len(classes) > 1 and len({frozenset(held) for held in classes.values()}) > 1:
                shown = mixed.setdefault(series, {})
                shown.update({model: ", ".join(sorted(held)) for model, held in classes.items()})
    return {series: mixed[series] for series in sorted(mixed)}


def flag_mixed_classes(
    table: pd.DataFrame,
    by_model: Mapping[str, Optional[Sequence[Mapping[str, Any]]]],
    groups: Iterable[Sequence[str]],
    stacklevel: int = 2,
) -> None:
    """Warn, and put them in `table.attrs["vintages"]`, when compared models read one series with different classes. A table
    of models that don't is left as it is."""
    mixed = mixed_classes(by_model, groups)
    if not mixed:
        return
    table.attrs["vintages"] = mixed
    warnings.warn(
        f"Compared models read one series with different vintage classes: {mixed}. A latest revision (or a last-write stamp) "
        "holds values written after the issue day, which no forecast saw then, so these models are not compared like for "
        "like. report.vintages() lists every input's class; df.attrs['vintages'] holds these.",
        UserWarning,
        stacklevel=stacklevel + 1,
    )


__all__ = [
    "COLUMNS",
    "flag_mixed_classes",
    "is_revised_class",
    "merge_vintages",
    "mixed_classes",
    "read_vintages",
    "union_vintages",
    "vintage_frame",
    "vintage_rows",
]
