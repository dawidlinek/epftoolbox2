"""A fingerprint written next to each results JSONL (roadmap item 7).

`BaseModel.run(save_to=...)` writes a `Manifest` beside the JSONL file it resumes from. On the next run
into the same directory, the new run's manifest is compared to the stored one: if a field that defines
what a forecast *is* (predictors, target, horizon, freq, training window, model class and
hyperparameters, the data actually used, and the information-set settings) has changed, resuming would
silently mix forecasts made under two different configurations. That raises `ManifestMismatchError`
unless the caller explicitly accepts it (`allow_manifest_mismatch=True`).

Provenance fields (package version, git SHA, timestamps) are recorded but never compared: a version bump
or a new commit does not by itself invalidate a results directory.

A directory with results but no manifest (from before this feature existed) is not a mismatch — by
default it is accepted with a warning and a manifest is written from now on. Pass `require_manifest=True`
for the stricter behaviour of refusing those too.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ..logging import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 1

# Compared to decide whether an existing manifest is compatible with the current run. Order is the
# order differences are reported in.
_COMPARED_FIELDS = (
    "model_class",
    "predictors",
    "target",
    "horizon",
    "freq",
    "training_window",
    "model_params",
    "data_columns_hash",
    "information_set",
)


class ManifestMismatchError(ValueError):
    """An existing results directory was produced with a different configuration than this run."""


@dataclass
class Manifest:
    """Everything that defines a forecast, plus provenance. See module docstring for what is compared."""

    model_class: str
    predictors: List[str]
    target: str
    horizon: int
    freq: str
    training_window: int
    model_params: Dict[str, Any]
    data_columns_hash: str
    information_set: Optional[Dict[str, Any]]
    test_start: str
    test_end: str
    package_version: str
    git_sha: Optional[str]
    created_at: str
    updated_at: str
    schema_version: int = SCHEMA_VERSION
    extra: Dict[str, Any] = field(default_factory=dict)  # forward compatibility: unknown fields from newer schemas

    def differences(self, existing: "Manifest") -> List[str]:
        """Human-readable list of compared fields that disagree with `existing`; empty if compatible."""
        out = []
        for name in _COMPARED_FIELDS:
            mine, theirs = getattr(self, name), getattr(existing, name)
            if mine != theirs:
                out.append(f"{name}: {theirs!r} (existing results) vs {mine!r} (this run)")
        return out

    def to_dict(self) -> Dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "extra"}
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Manifest":
        known = {f.name for f in fields(cls)} - {"extra"}
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(**kwargs, extra=extra)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str, sort_keys=False) + "\n")


def hash_columns(data: pd.DataFrame, columns: Sequence[str]) -> str:
    """A content hash (values and index) of `columns`, stable across processes and machines."""
    frame = data[sorted(set(columns))]
    row_hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


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
    target: str,
    horizon: int,
    freq: str,
    test_start: str,
    test_end: str,
    information_set: Optional[Any],
) -> Manifest:
    """The manifest for one model's run. `used_columns` is `[target, *predictors]` as actually resolved
    (after `{horizon}`/callable expansion), so it reflects what was fit regardless of how predictors were
    specified."""
    now = datetime.now(timezone.utc).isoformat()
    return Manifest(
        model_class=type(model).__name__,
        predictors=sorted(set(used_columns) - {target}),
        target=target,
        horizon=horizon,
        freq=freq,
        training_window=model.training_window,
        model_params=dict(model._model_kwargs),
        data_columns_hash=hash_columns(data, used_columns),
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
        message = (
            f"{model_name}: {path.parent} has results but no manifest ({path.name} not found), most likely "
            "from before this feature existed."
        )
        if require_manifest:
            raise ManifestMismatchError(
                f"{message} Pass require_manifest=False (the default) to accept them and start recording a "
                "manifest, or remove the old results to start fresh."
            )
        logger.warning(f"{message} Continuing without checking they match this run's configuration.")
    manifest.write(path)


__all__ = [
    "Manifest",
    "ManifestMismatchError",
    "build_manifest",
    "check_and_write",
    "hash_columns",
    "manifest_path",
]
