"""A fingerprint written next to each results JSONL (roadmap item 7).

`BaseModel.run(save_to=...)` writes a `Manifest` beside the JSONL file it resumes from. On the next run
into the same directory, the new run's manifest is compared to the stored one: if a field that defines
what a forecast *is* (predictors, their per-(period, horizon) order, target, horizon, freq, training
window, model class and hyperparameters, and the data actually used, including its time zone) has
changed, resuming would silently mix forecasts made under two different configurations. That raises
`ManifestMismatchError` unless the caller explicitly accepts it (`allow_manifest_mismatch=True`).

The information-set settings and the `availability` level are recorded (for provenance) but never compared: they gate
which predictors are allowed before fitting, and don't themselves change what a fitted model predicts. Changing only the
level between two runs therefore resumes. At the levels "warn" and "raise" the manifest also holds the listing of inputs
known after the issue time (`epftoolbox2.results.availability`), which is how a report built later can show it; level
"none" adds nothing, and leaves in place what an earlier run wrote.

Provenance fields (package version, git SHA, timestamps) are recorded but never compared: a version bump
or a new commit does not by itself invalidate a results directory.

A predictor function of `ctx` has no list per (period, horizon) to hash: which list a forecast uses depends on its day.
Its manifest records the function (import path, fingerprint of its source and of the information set it answers
through) and every distinct list by hash, and each record carries its list's hash. Resuming compares what matters
directly (`check_stored_lists`): the list and binary-column mask a fresh run would give each stored forecast. So editing
a comment or an unrelated availability rule resumes, with a note, and a change the source can't show (a list defined next
to the function) doesn't.

A list with `train_on="matching"` trains on the past days for which the function returned it, and that set can change with no
change to the list a forecast uses. So its records also carry `matching_days`, a hash of that set, compared the same way.

A directory with results but no manifest is not a mismatch. By default it is accepted with a warning and
a manifest is written for it. Pass `require_manifest=True` for the stricter behaviour of refusing those too.

A run with validation (`epftoolbox2.results.windows`) adds `windows`, the digests of its fixed lists' binary-column masks
(`binary_masks`; a predictor function's stay with each list, a test mask under the key a plain run gives it) and `actuals`,
a hash of the target up to the last day a stored `actual` reads. A plain run's manifest has none of them, except the
`binary_masks` of fixed lists where a column with no value on the first day was read on a later day and that changed a mask:
results without them were scaled before that rule, and `unrecorded_masks` refuses to resume them. Such a run may
cover another span than the results it resumes, so that validation can come first and test later: what is compared then is
the data the stored forecasts read, up to the end of the span they cover, and its manifest grows to cover both. Resuming
also compares each stored key's window and mask to what a fresh run would give it (`check_stored_lists`). A window this run
doesn't name stays recorded, like the records it holds.

Every run also records `column_hashes`, a hash of each column it reads (its predictors, the target, and what a predictor
function reads through `ctx`) on its own, with numbers as float64, and `frame_hash`, a hash of the whole frame, both over the
rows `data_columns_hash` covers. They are never compared on resume, so a change in a column a run doesn't read still resumes.
They are for what can't be seen from one model's results: two models averaged by an ensemble must have read one thing, whatever
columns each of them took (`epftoolbox2.ensembles._compute`). Manifests written before them load, and simply don't have them.

All these hashes describe the frame of the last run, so after a resume they speak for the stored forecasts only because resuming
compared them. Where nothing compared them, on an override that accepted other data or for results with no manifest, the manifest
records `data_unverified: true`, and keeps it across later resumes. It is written on those paths alone.

A run that skipped forecasts records how many per window and reason (`skipped`, `record_skips`), which is how a report merged from
the files can tell `no_matching_days` or `fit_error` from a task that crashed. A run with no skips has no such key. Each run replaces
the counts of the windows it names, since a resume tries the skipped tasks again. They are written after the run, and never compared.

How a forecast's training rows are chosen is compared too (`training_cut`). Rows were once chosen by counting them, which left a
mislabelled record on a spring switch day and a look-ahead on the days after an autumn switch, and a resume would keep those
records in the file and read them. So a manifest without this field, written before rows were chosen by date, is refused. A model
with `newest_target="published"` adds that, the target's rule (without its note) and the issue time and zone: the information set
is not compared, but these decide which targets train, so changing one refuses a resume, and changing only the level doesn't.
Results that may still hold such records, accepted over that refusal with `allow_manifest_mismatch=True` or resumed with no
manifest at all, get `{"rows": "dates", "stored": "unverified"}`, which every later resume keeps and which compares equal to the
plain one: `ModelPipeline.report()` warns about them every time. Records of a slot the clock doesn't have are left out of every
table and count wherever the zone is known (`ModelResultRef.iter_records`).

A run of some horizons only (`horizons=[1, 3, 7]`) records them as `horizons`, compared, next to `horizon`, its largest. A run of
every horizon from 1 to `horizon` has no such key, so its manifest is what it was before the field existed, and a resume with
another list refuses: the file would hold forecasts of horizons no run plans, or lack ones it does.

A run that issues on some weekdays only (`issue_days=["friday"]`) records their names as `issue_days`, compared, and only then: a
run of every day has no such key. A resume with other weekdays refuses for the same reason as other horizons.

An ensemble's file has a manifest of its own shape (`"kind": "ensemble"`, written by `epftoolbox2.ensembles._compute`). It says
which results it was computed from and is never compared, as the file is recomputed from them every time. No model may
resume into it.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import warnings
from dataclasses import dataclass, fields
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._date_utils import zone_name
from ..models._calendar import issue_day_names, resolve_issue_days
from ..logging import get_logger
from .availability import LISTING_TIMES, merge_listings
from .windows import TEST, RunWindows, mask_key

logger = get_logger(__name__)

SCHEMA_VERSION = 1

# `predictor_layout_hash` of a predictor function of `ctx`: it never equals the hash of fixed lists.
PREDICTOR_FUNCTION_LAYOUT = "predictor function: lists are compared per record"

# The `kind` of an ensemble's manifest.
ENSEMBLE_KIND = "ensemble"

# Who wrote a manifest and when, which says nothing about the results it describes.
_PROVENANCE_FIELDS = ("package_version", "git_sha", "created_at", "updated_at")

# Compared to decide whether an existing manifest is compatible with the current run. Order is the
# order differences are reported in.
_COMPARED_FIELDS = (
    "model_class",
    "predictor_layout_hash",
    "target",
    "horizon",
    "horizons",
    "issue_days",
    "freq",
    "training_window",
    "model_params",
    "data_columns_hash",
    "data_timezone",
    "training_cut",
)

# How a run chooses a forecast's training rows: `Manifest.training_cut`.
TRAINING_CUT_BY_DATE = {"rows": "dates"}
# The same, for results that may hold records written another way: see the module docstring.
TRAINING_CUT_UNVERIFIED = {**TRAINING_CUT_BY_DATE, "stored": "unverified"}


def _cut_rule(cut: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """How a run chooses rows, without what is known of the records stored: what a resume compares."""
    return None if cut is None else {key: value for key, value in cut.items() if key != "stored"}


def _rows_by_date(cut: Optional[Dict[str, Any]]) -> bool:
    """Whether a manifest's `training_cut` vouches for every record in its file: rows chosen by date, and nothing stored that
    was accepted without it. A model with `newest_target="published"` adds what its newest target is read from."""
    return cut is not None and cut.get("rows") == TRAINING_CUT_BY_DATE["rows"] and "stored" not in cut


_OLD_RESULTS_NOTE = (
    ". These results were made before training rows were chosen by date, so they may hold a look-ahead on the days after an "
    "autumn switch and a mislabelled forecast on a spring one, which a resume would keep in the file"
)


class ManifestMismatchError(ValueError):
    """An existing results directory was produced with a different configuration than this run."""


# What `stored_actual` gives for an issue day after the frame's last day: nothing to compare a stored `actual` with.
PAST_THE_FRAME = object()


class PlannedForecast(NamedTuple):
    """What a fresh run would give a stored forecast's key (`check_stored_lists`)."""

    list_hash: Optional[str]  # None for fixed lists
    mask_digest: str
    mask_key: str  # where a manifest keeps the digest
    window: Optional[str] = None  # None in a plain run: the record says nothing about it
    matching_days: Optional[str] = None  # a matching list's: a hash of the days it trains on


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
    # A predictor function of `ctx`: {"path", "fingerprint"} and one entry per distinct list (`PredictorPlan.describe`).
    predictor_function: Optional[Dict[str, str]] = None
    predictor_lists: Optional[List[Dict[str, Any]]] = None
    # {"level": "warn" or "raise", "times": "instants", "listing": rows of `results.availability.listing_rows`}; None at
    # level "none". Without "times" the rows were listed on the wall clock (see `results.availability.LISTING_TIMES`).
    availability: Optional[Dict[str, Any]] = None
    # A run with validation: `RunWindows.to_dict()`, the masks of fixed lists by `mask_key`, and
    # {"through": day, "hash": ...}, the target up to the last day a stored `actual` reads.
    windows: Optional[Dict[str, Any]] = None
    binary_masks: Optional[Dict[str, str]] = None
    actuals: Optional[Dict[str, str]] = None
    # Through `test_end`: each column this run read on its own, and the whole frame. Recorded, never compared on resume.
    column_hashes: Optional[Dict[str, str]] = None
    frame_hash: Optional[str] = None
    # True if the stored forecasts may come from another frame than the hashes above describe: see the module docstring.
    data_unverified: Optional[bool] = None
    # Window -> skip reason -> how many forecasts the last run naming that window skipped, a plain run's as "test"; recorded, never
    # compared, and absent with nothing skipped.
    skipped: Optional[Dict[str, Dict[str, int]]] = None
    # How the training rows were chosen. None in a manifest older than the field: see the module docstring.
    training_cut: Optional[Dict[str, Any]] = None
    # The horizons planned, when they aren't every one from 1 to `horizon`; None (and no key) when they are.
    horizons: Optional[List[int]] = None
    # The weekdays issued on, by name, when they aren't all seven; None (and no key) for every day.
    issue_days: Optional[List[str]] = None

    def differences(self, existing: "Manifest", skip: Sequence[str] = ()) -> List[str]:
        """Human-readable list of compared fields that disagree with `existing`; empty if compatible."""
        out = []
        for name in _COMPARED_FIELDS:
            if name in skip:
                continue
            mine, theirs = getattr(self, name), getattr(existing, name)
            if name == "training_cut":
                mine, theirs = _cut_rule(mine), _cut_rule(theirs)
            if mine != theirs:
                note = _OLD_RESULTS_NOTE if name == "training_cut" and theirs is None else ""
                out.append(f"{name}: {theirs!r} (existing results) vs {mine!r} (this run){note}")
        return out

    def to_dict(self) -> Dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        if self.predictor_function is None:  # fixed lists: nothing to record
            del out["predictor_function"], out["predictor_lists"]
        if self.availability is None:  # level "none": nothing to record
            del out["availability"]
        for name in ("windows", "binary_masks", "actuals"):  # a plain run: nothing to record
            if out[name] is None:
                del out[name]
        for name in ("column_hashes", "frame_hash"):  # a manifest older than them, or a frame that can't be hashed
            if out[name] is None:
                del out[name]
        if out["data_unverified"] is None:  # the usual case: resuming compared what the stored forecasts read
            del out["data_unverified"]
        if out["skipped"] is None:  # nothing skipped
            del out["skipped"]
        if out["training_cut"] is None:  # a manifest older than it
            del out["training_cut"]
        if out["horizons"] is None:  # every horizon from 1 to `horizon`
            del out["horizons"]
        if out["issue_days"] is None:  # every day
            del out["issue_days"]
        return out

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


def hash_frame(data: pd.DataFrame) -> Optional[str]:
    """`hash_columns` over every column of `data`, or None if a column holds values that can't be hashed (lists, say): a
    hash nothing needs must not stop a run."""
    try:
        return hash_columns(data, list(data.columns))
    except TypeError:
        return None


def hash_column(data: pd.DataFrame, column: str) -> str:
    """`hash_columns` of one column, with numbers as float64 and one bit pattern for a NaN and for zero: the same values hash
    alike whatever dtype a frame holds them in. Only `column_hashes`, which compare frames, use it."""
    series = data[column]
    if not pd.api.types.is_numeric_dtype(series):
        return hash_columns(data, [column])
    values = series.to_numpy(dtype="float64", na_value=np.nan) + 0.0
    values[np.isnan(values)] = np.nan
    return hash_columns(pd.DataFrame({column: values}, index=data.index), [column])


class FrameHashes:
    """The hashes of one frame, each computed once, for one `ModelPipeline.run`: thousands of variants on one frame would
    otherwise hash its columns again for every model.

    It belongs to that call and must not outlive it, or a frame changed in place would get a stale hash. Rows are through a day,
    as `.loc[:day]` reads them.
    """

    def __init__(self, data: pd.DataFrame):
        self._ordered = data.sort_index()
        self._rows: Dict[str, pd.DataFrame] = {}
        self._sets: Dict[Tuple[str, Tuple[str, ...]], str] = {}
        self._columns: Dict[Tuple[str, str], str] = {}
        self._frames: Dict[str, Optional[str]] = {}

    def rows(self, through: str) -> pd.DataFrame:
        if through not in self._rows:
            self._rows[through] = self._ordered.loc[:through]
        return self._rows[through]

    def of_columns(self, through: str, columns: Sequence[str]) -> str:
        """`hash_columns` of these columns together."""
        key = (through, tuple(sorted(set(columns))))
        if key not in self._sets:
            self._sets[key] = hash_columns(self.rows(through), key[1])
        return self._sets[key]

    def of_column(self, through: str, column: str) -> str:
        """`hash_column` of this column."""
        key = (through, column)
        if key not in self._columns:
            self._columns[key] = hash_column(self.rows(through), column)
        return self._columns[key]

    def of_frame(self, through: str) -> Optional[str]:
        """`hash_frame` of every column."""
        if through not in self._frames:
            self._frames[through] = hash_frame(self.rows(through))
        return self._frames[through]


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
    predictor_function: Optional[Dict[str, str]] = None,
    predictor_lists: Optional[List[Dict[str, Any]]] = None,
    availability: Optional[Dict[str, Any]] = None,
    windows: Optional[RunWindows] = None,
    binary_masks: Optional[Dict[str, str]] = None,
    read_columns: Sequence[str] = (),
    frame_hashes: Optional[FrameHashes] = None,
    training_cut: Optional[Dict[str, Any]] = None,
    horizons: Optional[Sequence[int]] = None,
    issue_days: Optional[Sequence[int]] = None,
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

    `windows` is a run with validation, whose `test_start` and `test_end` are its first and last issue day.

    `read_columns` are what a predictor function read through `ctx` besides its lists: `column_hashes` covers them. `frame_hashes`
    is what a pipeline shares between the models of one run, so that it hashes each thing once.

    `training_cut` is how the model chooses its training rows (`BaseModel._training_cut`), rows by date if not given.

    `horizon` is the largest horizon and `horizons` the ones planned, recorded only when they aren't every one from 1 to it.
    `issue_days` are the weekdays issued on (`date.weekday()` numbers), recorded by name, and only when they aren't every day.
    """
    now = datetime.now(timezone.utc).isoformat()
    hashes = FrameHashes(data) if frame_hashes is None else frame_hashes
    actuals = None
    if windows is not None:
        through = (date.fromisoformat(test_end) + timedelta(days=horizon)).isoformat()
        actuals = {"through": through, "hash": hashes.of_columns(through, [target])}
    return Manifest(
        model_class=type(model).__name__,
        predictors=sorted(set(used_columns) - {target}),
        predictor_layout_hash=PREDICTOR_FUNCTION_LAYOUT if predictor_function else hash_predictor_layout(expanded),
        target=target,
        horizon=horizon,
        freq=freq,
        training_window=model.training_window,
        model_params={"scaler": getattr(model, "scaler", "standard"), **dict(model._model_kwargs)},
        data_columns_hash=hashes.of_columns(test_end, used_columns),
        data_timezone=zone_name(data.index.tz) or str(data.index.tz),
        information_set=information_set.to_dict() if information_set is not None else None,
        test_start=test_start,
        test_end=test_end,
        package_version=_package_version(),
        git_sha=_git_sha(),
        created_at=now,
        updated_at=now,
        predictor_function=predictor_function,
        predictor_lists=predictor_lists,
        availability=availability,
        windows=None if windows is None else windows.to_dict(),
        binary_masks=binary_masks,
        actuals=actuals,
        column_hashes={column: hashes.of_column(test_end, column) for column in {*used_columns, *read_columns}},
        frame_hash=hashes.of_frame(test_end),
        training_cut=dict(TRAINING_CUT_BY_DATE if training_cut is None else training_cut),
        horizons=None
        if horizons is None or list(horizons) == list(range(1, horizon + 1))
        else [int(h) for h in horizons],
        issue_days=issue_day_names(issue_days),
    )


def horizons_of(manifest: Mapping[str, Any]) -> Optional[Tuple[int, ...]]:
    """The horizons a manifest's run planned, as a dict read from its file: `horizons` if it has them, else every one from 1 to
    `horizon`; None if it records no horizon."""
    if manifest.get("horizons"):
        return tuple(int(h) for h in manifest["horizons"])
    if manifest.get("horizon") is None:
        return None
    return tuple(range(1, int(manifest["horizon"]) + 1))


def read_horizons(path: Union[str, Path]) -> Optional[Tuple[int, ...]]:
    """The horizons the run of a manifest file planned (`horizons_of`), or None without a manifest or a horizon in it."""
    path = Path(path)
    return horizons_of(json.loads(path.read_text())) if path.exists() else None


def issue_days_of(manifest: Mapping[str, Any]) -> Optional[Tuple[int, ...]]:
    """The weekdays a manifest's run issued on, as a dict read from its file, as `date.weekday()` numbers; None for every day."""
    return resolve_issue_days(manifest.get("issue_days"))


def read_issue_days(path: Union[str, Path]) -> Optional[Tuple[int, ...]]:
    """`issue_days_of` the manifest at `path`; None (every day) without one."""
    path = Path(path)
    return issue_days_of(json.loads(path.read_text())) if path.exists() else None


def read_freq(path: Union[str, Path]) -> Optional[str]:
    """The `freq` a manifest's run forecast at, or None without a manifest or a freq in it."""
    path = Path(path)
    return json.loads(path.read_text()).get("freq") if path.exists() else None


def manifest_path(results_path: Path) -> Path:
    """The manifest file for a results JSONL: written next to it, e.g. `ols.jsonl` -> `ols.manifest.json`."""
    results_path = Path(results_path)
    return results_path.with_name(results_path.stem + ".manifest.json")


def is_ensemble_manifest(path: Union[str, Path]) -> bool:
    """Whether the manifest at `path` exists and is an ensemble's."""
    path = Path(path)
    return path.exists() and json.loads(path.read_text()).get("kind") == ENSEMBLE_KIND


def manifest_digest(manifest: Dict[str, Any]) -> str:
    """A hash of a manifest's content without its provenance and skip counts, which say nothing about a stored forecast: a member
    re-run to the same results keeps it."""
    kept = {key: value for key, value in manifest.items() if key not in (*_PROVENANCE_FIELDS, "skipped")}
    return hashlib.sha256(json.dumps(kept, sort_keys=True).encode()).hexdigest()


def _read_manifest_of_a_model(path: Path, model_name: str) -> Manifest:
    data = json.loads(path.read_text())
    if data.get("kind") == ENSEMBLE_KIND:
        raise ManifestMismatchError(
            f"{model_name}: {path.name} is an ensemble's, so a model of this name can't resume there. Rename one of them, "
            "or use a different save_dir."
        )
    return Manifest.from_dict(data)


def read_matching_lists(path: Union[str, Path]) -> Optional[Dict[str, int]]:
    """The hash of each `train_on="matching"` list a model's manifest records -> how many predictors it has, or None if it
    has none or there is no manifest."""
    path = Path(path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text()).get("predictor_lists") or []
    return {e["hash"]: e["n_predictors"] for e in entries if e.get("train_on") == "matching"} or None


def record_skips(path: Path, manifest: Manifest, skipped: Mapping[str, Mapping[str, int]]) -> None:
    """Write what a finished run skipped into its manifest: `skipped` has every window it names, and each replaces what is on
    file for that window alone. Nothing to change writes nothing, so a run with no skips leaves its manifest as it is."""
    kept = dict(manifest.skipped or {})
    for window, counts in skipped.items():
        if counts:
            kept[window] = {reason: int(counts[reason]) for reason in sorted(counts)}
        else:
            kept.pop(window, None)
    kept = {window: kept[window] for window in sorted(kept)} or None
    if kept != manifest.skipped:
        manifest.skipped = kept
        manifest.write(path)


def read_skipped(
    path: Union[str, Path],
    window: str = TEST,
    period: Optional[Tuple[str, str]] = None,
    horizons: Optional[Sequence[int]] = None,
    issue_days: Optional[Sequence[int]] = None,
) -> Dict[str, int]:
    """Skip reason -> how many, of one window as a manifest records it, or nothing if it has none.

    `period` is what a plain report covers. The counts are per window and can't be split by day, so a period other than the one
    the manifest was run on gets none: such a report still has `missing`, and no reasons it can't vouch for. Nor can they be
    split by horizon, so a report told a list of `horizons` other than the run's gets none either, nor by weekday: a report on
    other `issue_days` (weekday numbers, None for every day) than the run's gets none.
    """
    path = Path(path)
    if not path.exists():
        return {}
    stored = json.loads(path.read_text())
    if period is not None and (stored.get("test_start"), stored.get("test_end")) != tuple(period):
        return {}
    if horizons is not None and horizons_of(stored) != tuple(horizons):
        return {}
    if issue_days_of(stored) != (None if issue_days is None else tuple(issue_days)):
        return {}
    return dict((stored.get("skipped") or {}).get(window, {}))


def rows_unverified(path: Union[str, Path]) -> bool:
    """Whether the results of the manifest at `path` may hold records written before training rows were chosen by date: no
    manifest, one older than `training_cut`, or one that says so (see the module docstring)."""
    path = Path(path)
    return not path.exists() or not _rows_by_date(json.loads(path.read_text()).get("training_cut"))


def read_zone(path: Union[str, Path]) -> Optional[str]:
    """The time zone a manifest's forecasts were planned in, for an `IssueCalendar` with no data at hand, or None if it says none.

    None for no manifest, a naive index and a zone that doesn't read back by its name (`BaseModel.run` warns about those). Those
    keep the rule without a zone.
    """
    path = Path(path)
    if not path.exists():
        return None
    zone = json.loads(path.read_text()).get("data_timezone")
    if zone in (None, "None"):
        return None
    try:
        pd.Timestamp("2000-01-01").tz_localize(zone)
    except Exception:
        return None
    return zone


def read_windows(path: Union[str, Path]) -> Optional[RunWindows]:
    """The windows and segments a manifest records, or None for a plain run or no manifest.

    `path` is a results JSONL or its manifest.
    """
    path = Path(path)
    path = path if path.name.endswith(".manifest.json") else manifest_path(path)
    if not path.exists():
        return None
    windows = json.loads(path.read_text()).get("windows")
    return None if windows is None else RunWindows.from_dict(windows)


def check_and_write(
    path: Path,
    manifest: Manifest,
    *,
    has_existing_results: bool,
    allow_mismatch: bool = False,
    require_manifest: bool = False,
    model_name: str = "Model",
    frame_hashes: Optional[FrameHashes] = None,
    check_stored: Optional[Callable[[], None]] = None,
) -> None:
    """Compare `manifest` to the one at `path` (if any), then write it.

    Raises `ManifestMismatchError` when an existing manifest disagrees on the run's configuration and
    `allow_mismatch` is not set, or when results exist with no manifest and `require_manifest` is set.
    A mismatch that is allowed, and a missing manifest, are both logged.

    `frame_hashes` hashes this run's frame up to a day, so that a run with validation can be compared on the span the stored
    forecasts cover. It also gives `check_stored`, which compares those forecasts themselves (`check_stored_lists`) once the
    configuration is known to agree.

    Stored forecasts that this frame can't be told to have made leave `data_unverified` on the manifest (see the module docstring).
    """
    if path.exists():
        existing = _read_manifest_of_a_model(path, model_name)
        data_diffs = _compare_data(existing, manifest, frame_hashes)
        skip = () if data_diffs is None else ("data_columns_hash",)
        diffs = [
            *(data_diffs or []),
            *manifest.differences(existing, skip=skip),
            *unrecorded_masks(existing, manifest, has_existing_results),
        ]
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
        if check_stored is not None:
            check_stored()
        manifest.created_at = existing.created_at  # keep the first run's provenance
        manifest.skipped = existing.skipped  # until this run finishes and says otherwise: it may not
        if existing.data_unverified or (
            has_existing_results and _accepted_other_data(existing, manifest, data_diffs, frame_hashes)
        ):
            manifest.data_unverified = True
        if has_existing_results and not _rows_by_date(existing.training_cut):
            _mark_rows_unverified(
                manifest, model_name, path, "its manifest was written before rows were chosen by date"
            )
        if data_diffs == []:
            _cover_both(existing, manifest, frame_hashes)
        _keep_windows(existing, manifest, allow_mismatch, model_name)
        if manifest.predictor_lists is not None and existing.predictor_lists:
            # Records outside this run's test period stay in the file: their hashes must still resolve to their lists, and
            # a wider run later must still find the masks they used.
            kept = {entry["hash"]: entry for entry in existing.predictor_lists}
            for entry in manifest.predictor_lists:
                earlier = kept.pop(entry["hash"], {}).get("binary_masks", {})
                entry["binary_masks"] = {**earlier, **entry.get("binary_masks", {})}
            manifest.predictor_lists += list(kept.values())
        if manifest.binary_masks is not None and existing.binary_masks:
            manifest.binary_masks = {**existing.binary_masks, **manifest.binary_masks}
        _keep_listing(existing, manifest)
    elif has_existing_results:
        message = f"{model_name}: {path.parent} has results but no manifest ({path.name} not found)."
        if require_manifest:
            raise ManifestMismatchError(
                f"{message} Pass require_manifest=False (the default) to accept them and start recording a "
                "manifest, or remove the results to start fresh."
            )
        logger.warning(f"{message} Continuing without checking they match this run's configuration.")
        if check_stored is not None:
            check_stored()
        manifest.data_unverified = True
        _mark_rows_unverified(manifest, model_name, path, "they have no manifest")
    manifest.write(path)


def unrecorded_masks(existing: Manifest, manifest: Manifest, has_existing_results: bool) -> List[str]:
    """Why a plain run of fixed lists can't resume `existing`, if the stored forecasts were scaled by another rule.

    A column with no value on the test's first day used to stay unscaled for good in a fixed list; it is now read on the first
    later day with one, which changes the masks of such a run (and only of such a run). Its manifest then records the digests
    (`BaseModel._fixed_mask_digests`), and one written before that has none, so stored forecasts and new ones would be scaled
    two ways in one file. A manifest of the same configuration made since has them too."""
    if (
        not has_existing_results
        or manifest.windows is not None
        or manifest.predictor_function is not None
        or not manifest.binary_masks
        or existing.binary_masks
    ):
        return []
    return [
        f"binary_masks: none (existing results) vs {len(manifest.binary_masks)} (this run): the scaling rule changed. A column "
        "with no value on the test's first day is now read on the first later day with one, which changes how this run scales "
        "it, and the existing results were scaled the old way; rerun into a new save_dir."
    ]


def _mark_rows_unverified(manifest: Manifest, model_name: str, path: Path, why: str) -> None:
    """Results accepted though they may hold records written before rows were chosen by date keep saying so, for good."""
    if manifest.training_cut is not None and manifest.training_cut.get("stored") is None:
        warnings.warn(
            f"{model_name}: the results in {path.parent} may hold records written before training rows were chosen by date "
            f"({why}). They stay in the file, and every report() of it will say so. Run again into a new save_dir to be rid "
            "of them.",
            UserWarning,
            stacklevel=4,
        )
    manifest.training_cut = {**(manifest.training_cut or TRAINING_CUT_BY_DATE), "stored": "unverified"}


def _same_float(a: Optional[float], b: float) -> bool:
    return a == b or (a is not None and math.isnan(a) and math.isnan(b))


def _accepted_other_data(
    existing: Manifest, manifest: Manifest, data_diffs: Optional[List[str]], frame_hashes: Optional[FrameHashes]
) -> bool:
    """Whether a run going on over a mismatch may have other data under the stored forecasts than its hashes will describe.

    A run with windows has compared them (`data_diffs`). A plain one hashes other rows if it ends on another day, so that alone
    says nothing: what counts is the columns on file, up to the day on file.
    """
    if existing.data_timezone != manifest.data_timezone:
        return True
    if data_diffs is not None:
        return bool(data_diffs)
    if existing.data_columns_hash == manifest.data_columns_hash:
        return False
    if frame_hashes is None:
        return True
    try:
        return (
            frame_hashes.of_columns(existing.test_end, [existing.target, *existing.predictors])
            != existing.data_columns_hash
        )
    except KeyError:
        return True


def _compare_data(existing: Manifest, manifest: Manifest, frame_hashes: Optional[FrameHashes]) -> Optional[List[str]]:
    """What differs in the data the stored forecasts read, for a run with windows; None to compare hashes as a plain run.

    Plain results that end on another day are left to that comparison, which refuses them, as one plain run into another.
    """
    if frame_hashes is None or manifest.windows is None:
        return None
    if existing.windows is None and existing.test_end != manifest.test_end:
        return None
    out = []
    try:
        stored = frame_hashes.of_columns(existing.test_end, [existing.target, *existing.predictors])
    except KeyError as exc:
        return [f"data_columns_hash: a column the existing results read is gone: {exc}"]
    if stored != existing.data_columns_hash:
        out.append(f"data_columns_hash: the data up to {existing.test_end} is not what the existing results read")
    if existing.actuals is not None and existing.target == manifest.target:
        if frame_hashes.of_columns(existing.actuals["through"], [existing.target]) != existing.actuals["hash"]:
            out.append(
                f"actuals: {existing.target} up to {existing.actuals['through']} is not what the existing results read as "
                "`actual` (if the data has since grown, its last days had no actual then, and a fresh run would have one)"
            )
    return out


def _cover_both(existing: Manifest, manifest: Manifest, frame_hashes: FrameHashes) -> None:
    """Make the manifest speak for the file: the stored forecasts stay in it, whatever span this run covers."""
    manifest.test_start = min(existing.test_start, manifest.test_start)
    manifest.test_end = max(existing.test_end, manifest.test_end)
    manifest.predictors = sorted({*existing.predictors, *manifest.predictors})
    manifest.data_columns_hash = frame_hashes.of_columns(manifest.test_end, [manifest.target, *manifest.predictors])
    read = {manifest.target, *manifest.predictors, *(manifest.column_hashes or ()), *(existing.column_hashes or ())}
    # What only an earlier function read may be gone from this frame, unlike what the stored forecasts read as inputs.
    manifest.column_hashes = {
        column: frame_hashes.of_column(manifest.test_end, column)
        for column in read
        if column in frame_hashes.rows(manifest.test_end).columns
    }
    manifest.frame_hash = frame_hashes.of_frame(manifest.test_end)
    if manifest.actuals is not None and existing.actuals is not None:
        through = max(existing.actuals["through"], manifest.actuals["through"])
        manifest.actuals = {"through": through, "hash": frame_hashes.of_columns(through, [manifest.target])}


def _keep_windows(existing: Manifest, manifest: Manifest, allow_mismatch: bool, model_name: str) -> None:
    """A window this run doesn't name stays on file, like its records. If it can't stay next to what this run names, the run
    must say what the windows are now: dropping it would take its records out of every report without a word."""
    if manifest.windows is None or existing.windows is None:
        return
    run, on_file = RunWindows.from_dict(manifest.windows), RunWindows.from_dict(existing.windows)
    dropped = run.would_drop(on_file)
    if dropped is not None and not allow_mismatch:
        raise ManifestMismatchError(
            f"{model_name}: this run on {run.periods} would drop the {dropped[0]} window {dropped[1]} on file, since the "
            "validation period must end before the test period starts. Pass both validation= and test= to say what the "
            "windows are now, use a different save_dir, or pass allow_manifest_mismatch=True to resume anyway."
        )
    manifest.windows = run.keeping(on_file).to_dict()


def _keep_listing(existing: Manifest, manifest: Manifest) -> None:
    """Keep what an earlier run listed: its records stay in the file, and with them the reason to mention it.

    A run over a narrower test period adds its rows to the earlier ones. A run at level "none" looks at nothing, so it can't
    take anything back, and results that were flagged must not read as clean because of how they were last resumed. Rules that
    changed leave nothing to keep: what the old ones said may no longer be true, and this run's rows are what the current ones
    say. Without an information set there are no rules left to say otherwise. A listing made when the check compared
    wall-clock times (no `"times"`) is kept no more than one made under other rules.
    """
    if not existing.availability or manifest.information_set not in (None, existing.information_set):
        return
    if existing.availability.get("times") != LISTING_TIMES:
        return  # listed on the wall clock: like rules that changed, it says nothing this run can vouch for
    earlier = existing.availability.get("windows")
    if (
        earlier is None
        and existing.windows is None
        and manifest.windows is not None
        and existing.availability.get("listing")
    ):
        earlier = {TEST: existing.availability["listing"]}  # a plain run's rows are about its test window
    if manifest.availability is None:
        manifest.availability = dict(existing.availability)
        if earlier is not None:
            manifest.availability["windows"] = earlier
        return
    manifest.availability["listing"] = merge_listings(
        existing.availability.get("listing", []), manifest.availability["listing"]
    )
    if earlier is not None:
        later = manifest.availability.get("windows") or {}
        manifest.availability["windows"] = {
            window: merge_listings(earlier.get(window, []), later.get(window, []))
            for window in sorted({*earlier, *later})
        }


def check_stored_lists(
    path: Path,
    manifest: Manifest,
    records: Iterable[Dict[str, Any]],
    planned: Callable[[int, int, str], Optional[PlannedForecast]],
    *,
    allow_mismatch: bool = False,
    model_name: str = "Model",
    stored_actual: Optional[Callable[[int, int, str], Any]] = None,
) -> None:
    """Raise `ManifestMismatchError` if a stored forecast isn't what a fresh run would give.

    `records` are the stored records, and `planned(period, horizon, run_date)` gives the `PlannedForecast` of this run for a
    key, or None for a key outside its windows. A stored forecast differs if it used another list of a predictor function,
    if the same list now gets another binary-column mask (it is read from the first day of the list in its window, which a
    changed function or start can move), if a matching list now has other matching days (`matching_days`: a data change can do
    that through a value the function only reads, which `data_columns_hash` doesn't cover, and a day whose row a fit would drop
    anyway counts too), or if its issue day now falls in another window (a record without a `window` is from a plain run, which is the test window). Pending forecasts may change
    freely: nothing stored depends on them.

    A run with windows keeps one mask per (window, list, period, horizon) in a file, whichever days its records are on: a
    digest says nothing about which records used it, so a run that overwrote one would let older records pass for what they
    are not. `stored_actual(period, horizon, run_date)` gives what a fresh run's `actual` would be, or `PAST_THE_FRAME`, for
    stored records that no `actuals` hash speaks for (plain results, or results without a manifest). This is the only time
    they are looked at, since the manifest this run writes has the hash from then on, so what can't be compared is refused.

    A record without a list hash comes from fixed lists, so nothing can be said about its list. `allow_mismatch` logs
    instead. A function that changed without changing anything stored resumes, with a note.
    """
    existing = _read_manifest_of_a_model(path, model_name) if path.exists() else None
    stored_masks = {
        entry["hash"]: entry.get("binary_masks", {}) for entry in (existing and existing.predictor_lists) or []
    }
    fixed_masks = (existing and existing.binary_masks) or {}
    windowed = manifest.windows is not None
    new_masks = {entry["hash"]: entry.get("binary_masks", {}) for entry in manifest.predictor_lists or []}
    new_fixed_masks = manifest.binary_masks or {}
    # A plain manifest of fixed lists holds no digests, but says on which day their masks were read.
    moved_plain = (
        existing is not None
        and existing.windows is None
        and windowed
        and manifest.windows[TEST] is not None
        and manifest.windows[TEST][0] != existing.test_start
    )
    compare_actuals = stored_actual is not None and (existing is None or existing.windows is None)
    other_list, other_mask, other_window, other_actual, past_frame, other_days = [], [], [], [], [], []
    for record in records:
        period, horizon, run_date = record["hour"], record["horizon"], record["run_date"]
        stored = record.get("predictor_list")
        tag = record.get("window", TEST)
        expected = planned(period, horizon, run_date)
        if compare_actuals:
            actual = stored_actual(period, horizon, run_date)
            if actual is PAST_THE_FRAME:
                past_frame.append((run_date, period, horizon))
            elif actual is not None and not _same_float(record.get("actual"), actual):
                other_actual.append((run_date, period, horizon))
        if expected is not None and expected.window is not None and tag != expected.window:
            other_window.append((run_date, period, horizon, tag, expected.window))
        if expected is not None and expected.list_hash is not None and stored != expected.list_hash:
            other_list.append((run_date, period, horizon))
            continue
        if expected is not None and expected.matching_days not in (None, record.get("matching_days")):
            other_days.append((run_date, period, horizon))
        if windowed:
            key = mask_key(tag, period, horizon)
            kept = (fixed_masks if stored is None else stored_masks.get(stored, {})).get(key)
            new = (new_fixed_masks if stored is None else new_masks.get(stored, {})).get(key)
            if (kept is not None and new is not None and kept != new) or (
                moved_plain and stored is None and tag == TEST
            ):
                other_mask.append((run_date, period, horizon))
        elif expected is not None:
            if stored_masks.get(stored, {}).get(expected.mask_key, expected.mask_digest) != expected.mask_digest:
                other_mask.append((run_date, period, horizon))

    problems = []
    if other_window:
        problems.append(
            f"{len(other_window)} stored forecast(s) would change window, e.g. (run date, period, horizon, stored as, now) "
            f"{sorted(other_window)[:3]}"
        )
    if other_list:
        problems.append(
            f"{len(other_list)} stored forecast(s) used another predictor list than the predictor function returns now, "
            f"e.g. (run date, period, horizon) {sorted(other_list)[:3]}"
        )
    if other_days:
        problems.append(
            f"{len(other_days)} stored forecast(s) of a train_on='matching' list have other matching days than a fresh run "
            "would give them (the function now returns another list on some of those days, or the data's rows moved), e.g. "
            f"(run date, period, horizon) {sorted(other_days)[:3]}"
        )
    if other_mask:
        problems.append(
            f"{len(other_mask)} stored forecast(s) used the same list with other binary columns than a fresh run would "
            f"(its first day in the window moved), e.g. {sorted(other_mask)[:3]}"
        )
    if other_actual:
        problems.append(
            f"{len(other_actual)} stored forecast(s) hold another `actual` than the data gives now (it has grown or was "
            f"revised since), e.g. {sorted(other_actual)[:3]}"
        )
    if past_frame:
        problems.append(
            f"{len(past_frame)} stored forecast(s) were issued on days past the end of this frame, so their `actual` can't be "
            f"compared with it, and later runs would no longer look at them, e.g. {sorted(past_frame)[:3]}"
        )
    if problems and not allow_mismatch:
        raise ManifestMismatchError(
            f"{model_name}: {'; '.join(problems)}. Resuming would mix forecasts of two plans in {path.parent}. Use a "
            "different save_dir, remove the existing results, or pass allow_manifest_mismatch=True to resume anyway."
        )
    if problems:
        logger.warning(f"{model_name}: resuming over a changed plan at {path}: {'; '.join(problems)}")
    elif existing is not None and existing.predictor_function != manifest.predictor_function:
        logger.warning(
            f"{model_name}: the predictor function or its information set changed since {path.name} was written "
            f"({existing.predictor_function} -> {manifest.predictor_function}), but every stored forecast keeps its list."
        )


__all__ = [
    "ENSEMBLE_KIND",
    "Manifest",
    "ManifestMismatchError",
    "PAST_THE_FRAME",
    "PREDICTOR_FUNCTION_LAYOUT",
    "PlannedForecast",
    "build_manifest",
    "check_and_write",
    "check_stored_lists",
    "FrameHashes",
    "hash_column",
    "hash_columns",
    "hash_frame",
    "hash_predictor_layout",
    "is_ensemble_manifest",
    "manifest_digest",
    "rows_unverified",
    "manifest_path",
    "read_matching_lists",
    "read_skipped",
    "read_windows",
    "read_zone",
    "record_skips",
]
