"""An ensemble's forecasts, from the stored forecasts of its members and nothing else.

The file is always recomputed and written atomically, never resumed: it costs nothing and can't go stale. Its records are in
key order whatever order the members' were written in, so the same member results give the same bytes, apart from the
manifest's timestamps. The manifest says what they were: each member manifest's hash without its provenance, a hash of the
numbers read from its records, and the choice per test period with the validation scores behind it.

What makes a choice is read from `RunWindows.segments` and `Segment.serves`, never from a record's `window`: the validation
days of a later segment are test days of the one before it, and stay tagged "test".
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple

import numpy as np

from ..logging import get_logger
from ..results.manifest import (
    ENSEMBLE_KIND,
    SCHEMA_VERSION,
    ManifestMismatchError,
    _git_sha,
    _package_version,
    is_ensemble_manifest,
    manifest_digest,
    manifest_path,
)
from ..results.ref import ModelResultRef, split_by_window
from ..results.windows import SKIP_TARGET_IN_TEST, TEST, VALIDATION, RunWindows
from .base import Choice, Ensemble

logger = get_logger(__name__)

# In `forecast_counts()`: a planned forecast that a member the ensemble averages has no finite prediction for.
SKIP_MEMBER_MISSING = "member_missing"

_COLUMNS = ["run_date", "target_date", "hour", "horizon", "day_in_test", "prediction", "actual", "freq", "window"]

# A forecast key as one integer, whose order is that of (run date, target date, horizon, period).
_RUN_SHIFT = 40
_TARGET_SHIFT = 24
_HORIZON_SHIFT = 8
_TARGET_ZERO = 1 << 15
_LOW_16 = (1 << 16) - 1


def _key_codes(run: np.ndarray, target: np.ndarray, horizon: np.ndarray, hour: np.ndarray) -> np.ndarray:
    return (run << _RUN_SHIFT) | ((target - run + _TARGET_ZERO) << _TARGET_SHIFT) | (horizon << _HORIZON_SHIFT) | hour


def _ordinal(day: str) -> int:
    return datetime.date.fromisoformat(day).toordinal()


def _iso(ordinal: int) -> str:
    return datetime.date.fromordinal(ordinal).isoformat()


class Computed(NamedTuple):
    """An ensemble's ref over everything it computed (with `by_window`, as a model's has), and what it chose."""

    ref: ModelResultRef
    choices: List[Dict[str, Any]]


@dataclass
class _Member:
    """A member's forecasts in key order."""

    codes: np.ndarray
    day_in_test: np.ndarray
    prediction: np.ndarray
    actual: np.ndarray
    freq: Optional[str]

    def digest(self) -> str:
        data = hashlib.sha256()
        for values in (self.codes, self.prediction, self.actual):
            data.update(values.tobytes())
        return data.hexdigest()


@dataclass
class _Grid:
    """All members on one sorted grid of keys. `predictions[i, j]` is NaN where member j has no forecast for key i."""

    codes: np.ndarray
    day_in_test: np.ndarray
    predictions: np.ndarray
    actual: np.ndarray

    def serving(self, serves: Callable[[str, str], bool]) -> np.ndarray:
        """Which keys `serves(run_date, target_date)` lets count."""
        pairs, inverse = np.unique(self.codes >> _TARGET_SHIFT, return_inverse=True)
        run = pairs >> (_RUN_SHIFT - _TARGET_SHIFT)
        target = run + (pairs & _LOW_16) - _TARGET_ZERO
        flags = [serves(_iso(r), _iso(t)) for r, t in zip(run.tolist(), target.tolist())]
        return np.asarray(flags, dtype=bool)[inverse]

    def issued_in(self, period: Tuple[str, str]) -> np.ndarray:
        run = self.codes >> _RUN_SHIFT
        return (run >= _ordinal(period[0])) & (run <= _ordinal(period[1]))


def compute(
    ensemble: Ensemble,
    refs: Mapping[str, ModelResultRef],
    stored: Optional[RunWindows],
    like: ModelResultRef,
    path: Optional[Path],
    test_only: bool,
) -> Computed:
    """Compute `ensemble` from `refs`, which hold its members, and write it to `path`, if any.

    `stored` is every window on file. `like` is a model's ref of this run or report, which says what that covers (`windows`,
    the part of `stored` it names, `test_start`, `test_end`, `count`, `horizon`, `freq`). `test_only` is for an ensemble that
    has forecasts in the test window alone.
    """
    members = {name: _read_member(ensemble, name, refs[name], stored) for name in ensemble.members}
    manifests = {name: _read_manifest(refs[name]) for name in ensemble.members}
    _check_members_agree(ensemble, members, manifests)
    grid = _on_one_grid(ensemble, members)

    choices: List[Dict[str, Any]] = []
    if ensemble.needs_validation:
        picked, prediction, choices = _chosen_forecasts(ensemble, grid, stored)
    else:
        picked = np.flatnonzero(np.isfinite(grid.predictions).all(axis=1))
        prediction = ensemble.combine(grid.predictions[picked])

    freq = next((member.freq for member in members.values() if member.freq is not None), like.freq)
    records = _records(grid, picked, prediction, freq, stored)
    if path is not None:
        _write_atomically(path, (json.dumps(record) + "\n" for record in records))
        _write_manifest(path, ensemble, stored, freq, members, manifests, choices)
    return Computed(_ref(ensemble, grid.codes[picked], records, like, stored, path, freq, test_only), choices)


def _read_member(ensemble: Ensemble, name: str, ref: ModelResultRef, stored: Optional[RunWindows]) -> _Member:
    """Every record on file, or of the windows on file with the tag they should have: a report of the member reads no other."""
    run, target, hour, horizon, day_in_test, prediction, actual = [], [], [], [], [], [], []
    ordinals: Dict[str, int] = {}
    window_of: Dict[str, Optional[str]] = {}
    freqs = set()
    whole = replace(ref, test_start="", test_end="", window=None, target_before=None, by_window=None)
    for record in whole.iter_records(cols=_COLUMNS):
        run_date, target_date = record["run_date"], record["target_date"]
        if stored is not None:
            if run_date not in window_of:
                window_of[run_date] = stored.window_of(run_date)
            if window_of[run_date] is None or record.get("window", TEST) != window_of[run_date]:
                continue
        for day in (run_date, target_date):
            if day not in ordinals:
                ordinals[day] = _ordinal(day)
        run.append(ordinals[run_date])
        target.append(ordinals[target_date])
        hour.append(record["hour"])
        horizon.append(record["horizon"])
        day_in_test.append(record["day_in_test"])
        prediction.append(record["prediction"])
        actual.append(record["actual"])
        freqs.add(record.get("freq", ref.freq))
    if len(freqs) > 1:
        raise ManifestMismatchError(
            f"{ensemble.name}: the results of {name!r} hold several frequencies: {sorted(freqs)}"
        )
    codes = _key_codes(*(np.asarray(values, dtype=np.int64) for values in (run, target, horizon, hour)))
    order = np.argsort(codes, kind="stable")
    codes = codes[order]
    if (np.diff(codes) == 0).any():
        raise ManifestMismatchError(
            f"{ensemble.name}: the results of {name!r} hold a forecast key twice, so its forecast there is not one number."
        )
    return _Member(
        codes,
        np.asarray(day_in_test, dtype=np.int64)[order],
        np.asarray(prediction, dtype=np.float64)[order],
        np.asarray(actual, dtype=np.float64)[order],
        next(iter(freqs), None),
    )


def _read_manifest(ref: ModelResultRef) -> Optional[Dict[str, Any]]:
    if ref.path is None:
        return None
    path = manifest_path(ref.path)
    return json.loads(path.read_text()) if path.exists() else None


def _check_members_agree(
    ensemble: Ensemble, members: Mapping[str, _Member], manifests: Mapping[str, Optional[Dict[str, Any]]]
) -> None:
    """`ManifestMismatchError` if the members aren't forecasts of one thing, as far as what they recorded can tell."""
    problems = []
    frequencies = {name: member.freq for name, member in members.items() if member.freq is not None}
    recorded = {name: manifest for name, manifest in manifests.items() if manifest is not None}
    frequencies.update({name: manifest["freq"] for name, manifest in recorded.items() if manifest.get("freq")})
    if len(set(frequencies.values())) > 1:
        problems.append(f"frequencies {frequencies}")
    for field in ("target", "horizon", "data_timezone"):
        values = {name: manifest[field] for name, manifest in recorded.items() if manifest.get(field) is not None}
        if len(set(values.values())) > 1:
            problems.append(f"{field} {values}")
    problems += _data_differences(recorded)
    if problems:
        raise ManifestMismatchError(
            f"{ensemble.name}: its members are not forecasts of one thing, so averaging or ranking them would mean nothing: "
            f"{'; '.join(problems)}. Run them on the same data with the same target, horizon and freq."
        )
    unrecorded = sorted(name for name, manifest in manifests.items() if manifest is None)
    if unrecorded and recorded:
        logger.warning(f"{ensemble.name}: no manifest for {unrecorded}, so only their `actual` values can be compared.")


def _data_differences(manifests: Mapping[str, Dict[str, Any]]) -> List[str]:
    """What the data hashes of the manifests can say.

    A manifest hashes the columns its model reads, so two data hashes are comparable only for the same columns up to the same
    day. The target's own hash holds in a run with validation, whatever the columns. `_on_one_grid` compares the `actual` of
    every key the members share as well. That leaves one thing this can't see: members on other columns of different frames,
    with the same target.
    """
    same_columns: Dict[Tuple, Dict[str, str]] = {}
    actuals: Dict[str, Dict[str, str]] = {}
    for name, manifest in manifests.items():
        if manifest.get("data_columns_hash") is not None:
            columns = (manifest.get("test_end"), manifest.get("target"), tuple(manifest.get("predictors", ())))
            same_columns.setdefault(columns, {})[name] = manifest["data_columns_hash"]
        if manifest.get("actuals"):
            actuals.setdefault(manifest["actuals"]["through"], {})[name] = manifest["actuals"]["hash"]
    problems = []
    for hashes in same_columns.values():
        if len(set(hashes.values())) > 1:
            problems.append(f"data hashes that differ on the same columns in {sorted(hashes)}")
    for through, hashes in actuals.items():
        if len(set(hashes.values())) > 1:
            problems.append(f"a different target up to {through} in {sorted(hashes)}")
    return problems


def _on_one_grid(ensemble: Ensemble, members: Mapping[str, _Member]) -> _Grid:
    codes = np.unique(np.concatenate([member.codes for member in members.values()]))
    day_in_test = np.zeros(len(codes), dtype=np.int64)
    predictions = np.full((len(codes), len(members)), np.nan)
    actual = np.full(len(codes), np.nan)
    seen = np.zeros(len(codes), dtype=bool)
    for j, (name, member) in enumerate(members.items()):
        rows = np.searchsorted(codes, member.codes)
        predictions[rows, j] = member.prediction
        known = seen[rows]
        held, theirs = actual[rows[known]], member.actual[known]
        differs = (held != theirs) & ~(np.isnan(held) & np.isnan(theirs))
        if differs.any():
            first = int(codes[rows[known][differs][0]])
            raise ManifestMismatchError(
                f"{ensemble.name}: {name!r} and an earlier member disagree on the `actual` of {int(differs.sum())} "
                f"forecast(s), the first issued on {_iso(first >> _RUN_SHIFT)}: they were not given the same task."
            )
        actual[rows[~known]] = member.actual[~known]
        day_in_test[rows[~known]] = member.day_in_test[~known]
        seen[rows] = True
    return _Grid(codes, day_in_test, predictions, actual)


def _chosen_forecasts(
    ensemble: Ensemble, grid: _Grid, stored: RunWindows
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """A choice per segment, and each test forecast by the choice of the segment whose test period holds its issue day."""
    if not stored.segments:  # no test period on file: every validation key counts, and there is nothing to forecast
        first, last = stored.validation
        choice = _choose(ensemble, grid, lambda run_date, _: first <= run_date <= last, 0, stored.validation, None)
        return np.empty(0, dtype=np.int64), np.empty(0), [choice]
    picked, prediction, choices = [], [], []
    for index, segment in enumerate(stored.segments):
        row = _choose(ensemble, grid, segment.serves, index, segment.validation, segment.test)
        choices.append(row)
        rows = np.flatnonzero(grid.issued_in(segment.test))
        chosen = grid.predictions[rows][:, [ensemble.members.index(name) for name in row["chosen"]]]
        finite = np.isfinite(chosen).all(axis=1)
        picked.append(rows[finite])
        prediction.append(ensemble.combine(chosen[finite]))
    return np.concatenate(picked), np.concatenate(prediction), choices


def _choose(
    ensemble: Ensemble,
    grid: _Grid,
    serves: Callable[[str, str], bool],
    index: int,
    validation: Tuple[str, str],
    test: Optional[Tuple[str, str]],
) -> Dict[str, Any]:
    """The row of a choice, made on the keys every member holds with a finite prediction and actual, of those that serve it."""
    keys = grid.serving(serves) & np.isfinite(grid.predictions).all(axis=1) & np.isfinite(grid.actual)
    if not keys.any():
        raise ValueError(
            f"{ensemble.name}: its members have no validation forecast in common with a finite prediction and actual, "
            f"among those issued in {validation} to serve {test}, and nothing is ever chosen on nothing."
        )
    choice: Choice = ensemble.choose(grid.predictions[keys], grid.actual[keys])
    return {
        "segment": index,
        "chosen": choice.chosen,
        "k": len(choice.chosen),
        "validation_mae": choice.mae,
        "validation_forecasts": int(keys.sum()),
        "validation": list(validation),
        "test": None if test is None else list(test),
        "member_mae": choice.member_mae,
        **({} if choice.k_mae is None else {"k_mae": {str(k): mae for k, mae in choice.k_mae.items()}}),
    }


def _records(
    grid: _Grid, picked: np.ndarray, prediction: np.ndarray, freq: str, stored: Optional[RunWindows]
) -> List[Dict[str, Any]]:
    """Records in key order, with the keys of a model's, in the same order."""
    records = []
    days: Dict[int, Tuple[str, int, Optional[str]]] = {}
    targets: Dict[int, str] = {}
    columns = (
        grid.codes[picked].tolist(),
        grid.day_in_test[picked].tolist(),
        prediction.tolist(),
        grid.actual[picked].tolist(),
    )
    for code, day_in_test, predicted, actual in zip(*columns):
        run = code >> _RUN_SHIFT
        if run not in days:
            run_date = _iso(run)
            days[run] = (run_date, datetime.date.fromordinal(run).weekday(), stored and stored.window_of(run_date))
        run_date, weekday, window = days[run]
        target = run + ((code >> _TARGET_SHIFT) & _LOW_16) - _TARGET_ZERO
        if target not in targets:
            targets[target] = _iso(target)
        record = {
            "run_date": run_date,
            "target_date": targets[target],
            "run_weekday": weekday,
            "hour": code & 0xFF,
            "horizon": (code >> _HORIZON_SHIFT) & _LOW_16,
            "day_in_test": day_in_test,
            "prediction": predicted,
            "actual": actual,
            "coefficients": [],
            "freq": freq,
        }
        if window is not None:
            record["window"] = window
        records.append(record)
    return records


def _write_atomically(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w") as f:
            f.writelines(lines)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_manifest(
    path: Path,
    ensemble: Ensemble,
    stored: Optional[RunWindows],
    freq: str,
    members: Mapping[str, _Member],
    manifests: Mapping[str, Optional[Dict[str, Any]]],
    choices: List[Dict[str, Any]],
) -> None:
    target = manifest_path(path)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    recorded = [manifest for manifest in manifests.values() if manifest is not None]

    def on_record(field: str) -> Any:
        return next((manifest[field] for manifest in recorded if manifest.get(field) is not None), None)

    created = json.loads(target.read_text()).get("created_at", now) if is_ensemble_manifest(target) else now
    params = {key: value for key, value in vars(ensemble).items() if not key.startswith("_") and key != "name"}
    manifest = {
        "kind": ENSEMBLE_KIND,
        "model_class": type(ensemble).__name__,
        **params,
        "target": on_record("target"),
        "horizon": on_record("horizon"),
        "freq": freq,
        "data_timezone": on_record("data_timezone"),
        "actuals": on_record("actuals"),
        "windows": None if stored is None else stored.to_dict(),
        "member_results": {
            name: {
                "manifest": None if manifests[name] is None else manifest_digest(manifests[name]),
                "results": member.digest(),
            }
            for name, member in members.items()
        },
        "choices": choices,
        "package_version": _package_version(),
        "git_sha": _git_sha(),
        "created_at": created,
        "updated_at": now,
        "schema_version": SCHEMA_VERSION,
    }
    _write_atomically(target, [json.dumps(manifest, indent=2, default=str, sort_keys=False) + "\n"])


def _ref(
    ensemble: Ensemble,
    codes: np.ndarray,
    records: List[Dict[str, Any]],
    like: ModelResultRef,
    stored: Optional[RunWindows],
    path: Optional[Path],
    freq: str,
    test_only: bool,
) -> ModelResultRef:
    """The ref of the whole, with a ref per window of what this run or report names, as a model's has."""
    ref = ModelResultRef(
        name=ensemble.name,
        path=path,
        count=like.count,
        test_start=like.test_start,
        test_end=like.test_end,
        horizon=like.horizon,
        freq=freq,
        _results=None if path is not None else records,
        windows=like.windows,
        test_only=test_only,
    )
    if like.windows is None:
        run = codes >> _RUN_SHIFT
        ref.produced = int(((run >= _ordinal(ref.test_start)) & (run <= _ordinal(ref.test_end))).sum())
        ref.skipped = _missing(ref.count - ref.produced)
        _say_what_is_missing(ref.name, {None: ref})
        return ref
    parts = split_by_window(ref, like.windows, 96 if freq == "15min" else 24, target_before=stored.target_before)
    if test_only:
        parts.pop(VALIDATION, None)
    for part in parts.values():
        part.skipped.update(_missing(part.count - part.produced - part.skipped.get(SKIP_TARGET_IN_TEST, 0)))
    ref.by_window = parts
    ref.count = sum(part.count for part in parts.values())
    ref.produced = sum(part.produced for part in parts.values())
    ref.skipped = _missing(sum(part.skipped.get(SKIP_MEMBER_MISSING, 0) for part in parts.values()))
    _say_what_is_missing(ref.name, parts)
    return ref


def _missing(count: int) -> Dict[str, int]:
    return {SKIP_MEMBER_MISSING: count} if count > 0 else {}


def _say_what_is_missing(name: str, parts: Mapping[Optional[str], ModelResultRef]) -> None:
    """As loud as a model's shortfall: a warning for no forecast at all in a window, a log line for some missing."""
    for window, part in parts.items():
        where = "" if window is None else f" in the {window} window"
        missing = part.skipped.get(SKIP_MEMBER_MISSING, 0)
        if part.produced == 0 and part.count > 0:
            warnings.warn(f"{name}: no forecasts{where}, for lack of a member's. See forecast_counts().", stacklevel=2)
        elif missing:
            logger.warning(
                f"{name}: produced {part.produced} of {part.count} planned forecasts{where}. "
                f"Skipped: {SKIP_MEMBER_MISSING} x{missing}"
            )


__all__ = ["Computed", "SKIP_MEMBER_MISSING", "compute"]
