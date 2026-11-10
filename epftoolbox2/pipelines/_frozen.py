"""What `ModelPipeline.frozen()` checks before it freezes a choice: that each member it would freeze is the model whose stored
results made that choice.

A choice is made on the results in `save_dir`, and the frozen pipeline forecasts with its own models. Matched by name alone, a
member rebuilt with other predictors would be frozen over the results of the old one. So each member's manifest is compared with
what the model would record, in the fields a resume compares that the model alone decides: its class, its predictors (the layout
per period and horizon of a fixed list, the import path of a predictor function), its training window, its parameters
(`model_params`: the scaler, `_model_kwargs`, `block_predict`, a per-row target), and the kind and target transform of day and
pooled units. What depends on the data or the run's dates (data hashes, test period) is the study's, not the model's.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Mapping, Tuple

from ..models._row_target import manifest_params as row_target_params
from ..models._units import manifest_fields as unit_fields
from ..models.base import BaseModel
from ..models.context import function_path
from ..results.manifest import PREDICTOR_FUNCTION_LAYOUT, _block_params, hash_predictor_layout, horizons_of

# The text of an object without a stable one (a function's repr holds its address): no two processes write it alike.
_UNSTABLE = re.compile(r" at 0x[0-9a-fA-F]+")


def _plain(value: Any) -> Any:
    """`value` as a manifest file holds it: through JSON, objects as their text."""
    return json.loads(json.dumps(value, default=str))


def _model_params(model: BaseModel) -> Dict[str, Any]:
    """What `results.manifest.build_manifest` records as `model_params`."""
    return _plain(
        {
            "scaler": getattr(model, "scaler", "standard"),
            **dict(model._model_kwargs),
            **_block_params(model),
            **row_target_params(model),
        }
    )


def _layout_hash(model: BaseModel, manifest: Mapping[str, Any]) -> str:
    """The `predictor_layout_hash` a run of `model` at the manifest's frequency and horizons records, from a copy of it."""
    probe = copy.copy(model)
    probe._set_freq(manifest.get("freq") or "1h")
    return hash_predictor_layout(probe._precompute_expanded_predictors(horizons_of(manifest) or (1,)))


def model_differences(model: Any, manifest: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    """(the fields in which `model` differs from what its results' `manifest` records, the fields the manifest lacks)."""
    if not isinstance(model, BaseModel):
        return [], ["the model's configuration"]
    differences: List[str] = []
    lacking: List[str] = []

    def compare(field: str, mine: Any, theirs: Any) -> None:
        if theirs is None and field not in manifest:
            lacking.append(field)
        elif mine != theirs:
            differences.append(f"{field}: {theirs!r} in the results vs {mine!r} in this pipeline")

    compare("model_class", type(model).__name__, manifest.get("model_class"))
    function = getattr(model, "_ctx_fn", None)
    recorded = (manifest.get("predictor_function") or {}).get("path")
    fixed_on_file = manifest.get("predictor_layout_hash") not in (None, PREDICTOR_FUNCTION_LAYOUT)
    if function is not None:
        if recorded is not None:
            compare("predictor_function", function_path(function), recorded)
        elif fixed_on_file:
            differences.append(
                f"predictors: a fixed list in the results vs the function {function_path(function)} here"
            )
        else:
            lacking.append("predictor_function")
    elif recorded is not None or manifest.get("predictor_layout_hash") == PREDICTOR_FUNCTION_LAYOUT:
        differences.append(f"predictors: the function {recorded} in the results vs a fixed list in this pipeline")
    elif fixed_on_file:
        if _layout_hash(model, manifest) != manifest["predictor_layout_hash"]:
            differences.append(
                f"predictors: the results' list per period and horizon (of {manifest.get('predictors')}) isn't this "
                f"pipeline's ({model.predictors!r})"
            )
    else:
        lacking.append("predictor_layout_hash")
    compare("training_window", _plain(model.training_window), manifest.get("training_window"))
    if "model_params" in manifest:
        mine, theirs = _model_params(model), manifest["model_params"] or {}
        for key in sorted(set(mine) | set(theirs)):
            if any(_UNSTABLE.search(json.dumps(side.get(key))) for side in (mine, theirs)):
                continue  # an object's address: no run writes it alike, so it says nothing
            if mine.get(key) != theirs.get(key):
                differences.append(
                    f"model_params[{key!r}]: {theirs.get(key)!r} in the results vs {mine.get(key)!r} in this pipeline"
                )
    else:
        lacking.append("model_params")
    units = unit_fields(model)  # nothing for a period model, whose manifest has neither key
    for field in ("unit_kind", "target_transform"):
        if units.get(field) != manifest.get(field):
            differences.append(
                f"{field}: {manifest.get(field)!r} in the results vs {units.get(field)!r} in this pipeline"
            )
    return differences, lacking


__all__ = ["model_differences"]
