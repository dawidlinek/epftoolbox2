"""A fingerprint written next to each results JSONL (roadmap item 7).

`BaseModel.run(save_to=...)` writes a `Manifest` beside the JSONL file it resumes from. On the next run
into the same directory, the new run's manifest is compared to the stored one: if a field that defines
what a forecast *is* (predictors, their per-(period, horizon) order, target, horizon, freq, training
window, model class and hyperparameters, and the data actually used, including its time zone) has
changed, resuming would silently mix forecasts made under two different configurations. That raises
`ManifestMismatchError` unless the caller explicitly accepts it (`allow_manifest_mismatch=True`).

The information-set settings are recorded (for provenance) but never compared: they gate which predictors
are allowed before fitting, and don't themselves change what a fitted model predicts.

Provenance fields (package version, git SHA, timestamps) are recorded but never compared: a version bump
or a new commit does not by itself invalidate a results directory.

A directory with results but no manifest is not a mismatch. By default it is accepted with a warning and
a manifest is written for it. Pass `require_manifest=True` for the stricter behaviour of refusing those too.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ..logging import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 1

# Compared to decide whether an existing manifest is compatible with the current run. Order is the
# order differences are reported in.
_COMPARED_FIELDS = (
    "model_class",
    "predictor_layout_hash",
    "target",
    "horizon",
    "freq",
    "training_window",
    "model_params",
    "data_columns_hash",
    "data_timezone",
)


class ManifestMismatchError(ValueError):
    """An existing results directory was produced with a different configuration than this run."""


@dataclass
class Manifest:
    """Everything that defines a forecast, plus provenance. See module docstring for what is compared."""

    model_class: str
    predictors: List[str]
    predictor_layout_hash: str
    target: str
    horizon: int
    freq: str
    training_window: int
    model_params: Dict[str, Any]
    data_columns_hash: str
    data_timezone: str
    information_set: Optional[Dict[str, Any]]
    test_start: str
    test_end: str
    package_version: str
    git_sha: Optional[str]
    created_at: str
    updated_at: str
    schema_version: int = SCHEMA_VERSION

    def differences(self, existing: "Manifest") -> List[str]:
        """Human-readable list of compared fields that disagree with `existing`; empty if compatible."""
        out = []
        for name in _COMPARED_FIELDS:
            mine, theirs = getattr(self, name), getattr(existing, name)
            if mine != theirs:
                out.append(f"{name}: {theirs!r} (existing results) vs {mine!r} (this run)")
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Manifest":
        return cls(**data)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str, sort_keys=False) + "\n")


def hash_columns(data: pd.DataFrame, columns: Sequence[str]) -> str:
    """A content hash (values and index) of `columns`, stable across processes and machines.

    The hash is over each value's own representation (e.g. a tz-aware `DatetimeIndex`'s underlying epoch
    values), not over display metadata such as the index's time zone label - two indexes with the same
    instants but a different `.tz` hash the same. Compare `data_timezone` separately for that.
    """
    frame = data[sorted(set(columns))]
    row_hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def hash_predictor_layout(expanded: Dict[Tuple[int, int], Sequence[str]]) -> str:
    """A hash of the ordered predictor list per (period, horizon).

    Unlike a sorted union of names, this is sensitive to order (a reordering changes what stored
    coefficients mean) and to per-task differences (`["price_d-1", "price_d-2"]` and `["price_d-{horizon}"]`
    expanded per horizon use the same names overall but a different layout per task).
    """
    canonical = {f"{period}-{horizon}": list(names) for (period, horizon), names in sorted(expanded.items())}
    payload = json.dumps(canonical, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("epftoolbox2")
    except Exception:
        try:
            from .. import __version__

            return __version__
        except Exception:
            return "unknown"


def build_manifest(
    *,
    model: Any,
    data: pd.DataFrame,
    used_columns: Sequence[str],
    expanded: Dict[Tuple[int, int], Sequence[str]],
    target: str,
    horizon: int,
    freq: str,
    test_start: str,
    test_end: str,
    information_set: Optional[Any],
) -> Manifest:
    """The manifest for one model's run.

    `used_columns` is `[target, *predictors]` as actually resolved (after `{horizon}`/callable expansion),
    recorded for readability. `expanded` is the same resolution keyed by `(period, horizon)`
    (`BaseModel._precompute_expanded_predictors`); it is what `predictor_layout_hash` is compared on, since
    it (unlike a sorted union of names) is sensitive to predictor order and to per-task differences.

    `model_params` includes `scaler` alongside the model's own `_model_kwargs`: it changes the scale a
    model fits and predicts in just as much as a hyperparameter would, but `BaseModel` keeps it out of
    `_model_kwargs` (passed to workers separately, see `base.py`) so a subclass overriding `_model_kwargs`
    can't accidentally drop it.

    Only rows up to `test_end` are hashed: no forecast in `[test_start, test_end]` ever reads a later row,
    so data appended after `test_end` between runs - e.g. a source's own fetch window catching up a little
    between two runs on the same day - doesn't raise a mismatch. The lower bound is deliberately left open
    (not narrowed to the training window) so that resuming with a *different* test_start over the same,
    unchanged data - including a narrower one, as `ModelPipeline.report()`'s callers sometimes do - is not
    itself treated as a configuration change.
    """
    now = datetime.now(timezone.utc).isoformat()
    hashed_rows = data.sort_index().loc[:test_end]
    return Manifest(
        model_class=type(model).__name__,
        predictors=sorted(set(used_columns) - {target}),
        predictor_layout_hash=hash_predictor_layout(expanded),
        target=target,
        horizon=horizon,
        freq=freq,
        training_window=model.training_window,
        model_params={"scaler": getattr(model, "scaler", "standard"), **dict(model._model_kwargs)},
        data_columns_hash=hash_columns(hashed_rows, used_columns),
        data_timezone=str(data.index.tz),
        information_set=information_set.to_dict() if information_set is not None else None,
        test_start=test_start,
        test_end=test_end,
        package_version=_package_version(),
        git_sha=_git_sha(),
        created_at=now,
        updated_at=now,
    )


def manifest_path(results_path: Path) -> Path:
    """The manifest file for a results JSONL: written next to it, e.g. `ols.jsonl` -> `ols.manifest.json`."""
    results_path = Path(results_path)
    return results_path.with_name(results_path.stem + ".manifest.json")


def check_and_write(
    path: Path,
    manifest: Manifest,
    *,
    has_existing_results: bool,
    allow_mismatch: bool = False,
    require_manifest: bool = False,
    model_name: str = "Model",
) -> None:
    """Compare `manifest` to the one at `path` (if any), then write it.

    Raises `ManifestMismatchError` when an existing manifest disagrees on the run's configuration and
    `allow_mismatch` is not set, or when results exist with no manifest and `require_manifest` is set.
    A mismatch that is allowed, and a missing manifest, are both logged.
    """
    if path.exists():
        existing = Manifest.from_dict(json.loads(path.read_text()))
        diffs = manifest.differences(existing)
        if diffs and not allow_mismatch:
            raise ManifestMismatchError(
                f"{model_name}: the results in {path.parent} were produced with a different configuration "
                f"than this run ({path.name}):\n"
                + "\n".join(f"  - {d}" for d in diffs)
                + "\nResuming would silently mix forecasts made under two configurations. Use a different "
                "save_dir, remove the existing results, or pass allow_manifest_mismatch=True (`epf model "
                "--allow-manifest-mismatch`) to overwrite the manifest and resume anyway."
            )
        if diffs:
            logger.warning(f"{model_name}: manifest mismatch at {path} overridden: {'; '.join(diffs)}")
        manifest.created_at = existing.created_at  # keep the first run's provenance
    elif has_existing_results:
        message = f"{model_name}: {path.parent} has results but no manifest ({path.name} not found)."
        if require_manifest:
            raise ManifestMismatchError(
                f"{message} Pass require_manifest=False (the default) to accept them and start recording a "
                "manifest, or remove the results to start fresh."
            )
        logger.warning(f"{message} Continuing without checking they match this run's configuration.")
    manifest.write(path)


__all__ = [
    "Manifest",
    "ManifestMismatchError",
    "build_manifest",
    "check_and_write",
    "hash_columns",
    "hash_predictor_layout",
    "manifest_path",
]
