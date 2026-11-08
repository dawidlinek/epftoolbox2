"""Model weights: where a run finds them, how they got there, and the sha256 that says they are the files a card names.

The library never downloads during `run()`, `verify()` or `report()` (step 6 spec, "Weights: where they come from"): a GPU node
can hang on the network, a SLURM array would fetch once per task, and a silent download would accept a licence on the user's
behalf. Weights reach a run in one of two ways:

- `epf weights fetch <Model>` (`fetch`), on a machine with network: it downloads the card's files through
  `huggingface_hub.snapshot_download` into the cache, verifies each file's sha256 against the card (a mismatch refuses unless
  the revision was given explicitly, then the new hashes are recorded) and writes the sidecar `epftoolbox2-weights.json`. A card
  with `commercial_use=False` or a `revenue_cap` needs `accept=<licence id>` (`--accept`): fetching is the act of accepting.
- `weights="/path/to/file-or-dir"` on the model: used as given, hashed, and recorded; hashes other than the card's are "not the
  reference weights", which the pre-flight warns about and the manifest records.

Cache layout: `weights_dir()` is `$EPFTOOLBOX2_WEIGHTS_DIR`, else `~/.cache/epftoolbox2/weights`; a card's files live in
`<root>/<repo>/<revision>/` under the names the repository gives them, next to the sidecar `SIDECAR` (repo, revision, family,
each file's sha256 and bytes, the total, licence and its URL, the licence id accepted, whether the hashes are the card's, the fetch
time). `resolve` reads only that: a directory without a sidecar is not fetched, and a file whose hash is not the sidecar's has
changed since (`epf weights verify` re-hashes everything).

A file is hashed once per process: `sha256_of` keeps each hash under its path, device, inode, size, mtime and ctime, so a file
rewritten in place or replaced by another (even with its size and mtime) hashes again. Inside an adapter's load, `offline()`
holds `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` (and huggingface_hub's own constant, which it reads at import) and puts
them back afterwards, so a package that would fetch on its own fails instead.

Nothing here imports an extra: `snapshot_download` imports huggingface_hub when it is called.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

from .._extras import MissingExtraError

if TYPE_CHECKING:  # pragma: no cover
    from ._cards import ModelCard, WeightsRef

# Where the cache lives: this variable, else DEFAULT_WEIGHTS_DIR.
WEIGHTS_DIR_ENV = "EPFTOOLBOX2_WEIGHTS_DIR"
DEFAULT_WEIGHTS_DIR = "~/.cache/epftoolbox2/weights"
# The file `fetch` writes next to the weights it fetched, and the only thing `resolve` trusts.
SIDECAR = "epftoolbox2-weights.json"
# What `offline()` sets to "1" while an adapter loads.
OFFLINE_VARIABLES = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")

# (resolved path, device, inode, size, mtime and ctime in ns) -> sha256: see `sha256_of`.
_HASHES: Dict[Tuple[str, int, int, int, int, int], str] = {}


class MissingWeightsError(ImportError):
    """A model's weights are not where a run looks for them, or not the files that were fetched. The message names the command
    that fetches them (`epf weights fetch ...`). `problems` holds each problem's line when the pre-flight found several."""

    def __init__(self, message: str, problems: Sequence[str] = ()):
        self.message = message
        self.problems = tuple(problems)
        super().__init__(message)

    def __reduce__(self):
        return type(self), (self.message, self.problems)


class WeightsMismatchError(ValueError):
    """`fetch` downloaded files whose sha256 is not the card's: upstream replaced them. Nothing was recorded."""


class LicenceNotAcceptedError(ValueError):
    """`fetch` of weights with a non-commercial or revenue-capped licence without `accept` naming that licence."""


def weights_dir() -> Path:
    """The root of the weights cache: `$EPFTOOLBOX2_WEIGHTS_DIR`, else `~/.cache/epftoolbox2/weights`."""
    raw = os.environ.get(WEIGHTS_DIR_ENV)
    return Path(raw if raw else DEFAULT_WEIGHTS_DIR).expanduser()


def cache_dir(ref: "WeightsRef", root: Optional[Union[str, Path]] = None, revision: Optional[str] = None) -> Path:
    """Where the files of `ref` (at `revision`, the card's by default) live: `<root>/<repo>/<revision>/`."""
    return (Path(root).expanduser() if root is not None else weights_dir()) / ref.repo / (revision or ref.revision)


def sha256_of(path: Union[str, Path], cached: bool = True) -> str:
    """The sha256 of a file, read once per process for each (path, device, inode, size, mtime and ctime in ns). A file rewritten
    in place changes its size or its times, and one replaced (`cp -p`, `rsync -a` and tar restore the old mtime, and a file
    system may keep it to the second) has another inode and a new ctime, which no copy can set: either is hashed again.
    `cached=False` reads it again whatever is kept (`epf weights verify`)."""
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if cached:
        found = _HASHES.get(key)
        if found is not None:
            return found
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    _HASHES[key] = digest
    return digest


class ResolvedWeights(NamedTuple):
    """Weights a run found: `ref` is `repo@revision` for the cache and the path as given for `weights=`, `root` the directory
    holding the files (a single file's own directory), `files` each file's name under `root` -> sha256, and `reference`
    whether those are the card's hashes (True), not all of them (False), or the card records none to compare with (None)."""

    ref: str
    root: Path
    files: Dict[str, str]
    reference: Optional[bool]

    def path(self, name: Optional[str] = None) -> Path:
        """The path of the file `name`, or of the only file when there is one."""
        if name is None:
            if len(self.files) != 1:
                raise ValueError(f"{self.ref} has {len(self.files)} files: name one of {sorted(self.files)}")
            name = next(iter(self.files))
        return self.root / name

    def record(self) -> Dict[str, Any]:
        """What a manifest records: the ref, each file's sha256 (compared on resume), and `reference` (provenance)."""
        return {"ref": self.ref, "files": dict(sorted(self.files.items())), "reference": self.reference}


def _reference(ref: Optional["WeightsRef"], files: Mapping[str, str], by_name: bool = True) -> Optional[bool]:
    """Whether `files` are the card's: every file the card hashes has that hash (by name, or for a single file given by path
    the card's only file), None when the card hashes none."""
    if ref is None:
        return None
    known = {name: digest for name, digest in ref.files.items() if digest}
    if not known:
        return None
    if not by_name and len(files) == 1 and len(ref.files) == 1:
        return next(iter(files.values())) == next(iter(known.values()))
    return all(files.get(name) == digest for name, digest in known.items())


def _listed_files(directory: Path) -> List[str]:
    """Every regular file under `directory`, by its path relative to it, without the sidecar and hidden directories (the
    `.cache` huggingface_hub writes into a download directory)."""
    names = []
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if path.is_file() and relative.name != SIDECAR and not any(part.startswith(".") for part in relative.parts):
            names.append(relative.as_posix())
    return names


def resolve(
    card: Optional["ModelCard"],
    weights: Optional[Union[str, Path]] = None,
    *,
    root: Optional[Union[str, Path]] = None,
    what: str = "This model",
    fetch_command: str = "epf weights fetch",
) -> Optional[ResolvedWeights]:
    """The weights a model runs with, hashed, or None for a card without weights (trained on the run's data) and no `weights`.

    `weights` (the model's parameter) is a file or a directory used as given: a directory must hold the card's files, or every
    file in it for a card that names none. Without it the card's files come from the cache (`cache_dir`), as `fetch` left them:
    the sidecar must be there and each file must still have the sha256 it was fetched with. Anything else raises
    `MissingWeightsError` naming `fetch_command`. Never downloads anything."""
    ref = None if card is None else card.weights
    hint = f"run `{fetch_command}` on a machine with network, or pass weights='/path/to/file-or-dir'"
    if weights is not None:
        given = Path(weights).expanduser()
        if given.is_file():
            files = {given.name: sha256_of(given)}
            return ResolvedWeights(str(weights), given.parent, files, _reference(ref, files, by_name=False))
        if not given.is_dir():
            raise MissingWeightsError(f"{what}: weights={str(weights)!r} is neither a file nor a directory")
        names = list(ref.files) if ref is not None and ref.files else _listed_files(given)
        missing = [name for name in names if not (given / name).is_file()]
        if missing or not names:
            lacks = ", ".join(missing) if missing else "any file"
            raise MissingWeightsError(f"{what}: weights={str(weights)!r} lacks {lacks}")
        files = {name: sha256_of(given / name) for name in names}
        return ResolvedWeights(str(weights), given, files, _reference(ref, files))
    if ref is None:
        return None
    directory = cache_dir(ref, root)
    sidecar = directory / SIDECAR
    if not sidecar.is_file():
        raise MissingWeightsError(
            f"{what}: no weights for {ref.repo}@{ref.revision} in {directory} (no {SIDECAR} there): {hint}"
        )
    stored = json.loads(sidecar.read_text())
    on_file = stored.get("files", {})
    problems = [f"{name} was never fetched" for name in ref.files if name not in on_file]
    files: Dict[str, str] = {}
    for name, entry in on_file.items():
        path = directory / name
        if not path.is_file():
            problems.append(f"{name} is missing")
            continue
        digest = sha256_of(path)
        if digest != entry.get("sha256"):
            problems.append(
                f"{name} changed since it was fetched (sha256 {digest[:12]}, fetched {entry.get('sha256')})"
            )
        files[name] = digest
    if problems:
        raise MissingWeightsError(
            f"{what}: the weights in {directory} are not the files fetched: {'; '.join(problems)}. Check them with `epf "
            f"weights verify`, or {hint}"
        )
    return ResolvedWeights(f"{ref.repo}@{ref.revision}", directory, files, _reference(ref, files))


_OFFLINE_LOCK = threading.Lock()
_OFFLINE_DEPTH = 0
_OFFLINE_SAVED: Dict[str, Any] = {}


@contextlib.contextmanager
def offline() -> Iterator[None]:
    """Hold `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` (and huggingface_hub's `constants.HF_HUB_OFFLINE`, which it reads
    when it is imported, if it is) while an adapter loads, and put back what was there afterwards, also after an exception. Nested
    and overlapping scopes of several threads restore once the last one ends."""
    global _OFFLINE_DEPTH
    with _OFFLINE_LOCK:
        if _OFFLINE_DEPTH == 0:
            _OFFLINE_SAVED.clear()
            _OFFLINE_SAVED.update({name: os.environ.get(name) for name in OFFLINE_VARIABLES})
            for name in OFFLINE_VARIABLES:
                os.environ[name] = "1"
            constants = sys.modules.get("huggingface_hub.constants")
            if constants is not None and hasattr(constants, "HF_HUB_OFFLINE"):
                _OFFLINE_SAVED["constants"] = (constants, constants.HF_HUB_OFFLINE)
                constants.HF_HUB_OFFLINE = True
        _OFFLINE_DEPTH += 1
    try:
        yield
    finally:
        with _OFFLINE_LOCK:
            _OFFLINE_DEPTH -= 1
            if _OFFLINE_DEPTH == 0:
                saved = dict(_OFFLINE_SAVED)
                _OFFLINE_SAVED.clear()
                held = saved.pop("constants", None)
                if held is not None:
                    held[0].HF_HUB_OFFLINE = held[1]
                for name, value in saved.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value


def licence_flagged(card: "ModelCard") -> bool:
    """Whether fetching the card's weights needs `accept=<licence id>`: a non-commercial licence or a revenue cap."""
    return not card.commercial_use or card.revenue_cap is not None


def snapshot_download(**kwargs: Any) -> str:
    """`huggingface_hub.snapshot_download(**kwargs)`, imported when called (the one place the library downloads). Tests replace
    this function."""
    from huggingface_hub import snapshot_download as download

    return download(**kwargs)


class FetchPlan(NamedTuple):
    """What `fetch` will download, shown before it does: `what` names the model class (and size)."""

    what: str
    repo: str
    revision: str
    explicit_revision: bool
    files: Tuple[str, ...]
    bytes: Optional[int]
    destination: Path
    licence: str
    licence_url: str
    flagged: bool
    revenue_cap: Optional[str]
    attribution: Optional[str]


def fetch_plan(
    card: "ModelCard", what: str, revision: Optional[str] = None, root: Optional[Union[str, Path]] = None
) -> FetchPlan:
    """The plan of `fetch(card, ...)`. A card without weights raises ValueError: there is nothing to fetch."""
    ref = card.weights
    if ref is None:
        raise ValueError(f"{what} has no weights to fetch: it trains on the run's data")
    return FetchPlan(
        what=what,
        repo=ref.repo,
        revision=revision or ref.revision,
        explicit_revision=revision is not None,
        files=tuple(ref.files),
        bytes=ref.bytes,
        destination=cache_dir(ref, root, revision),
        licence=card.licence,
        licence_url=card.licence_url,
        flagged=licence_flagged(card),
        revenue_cap=card.revenue_cap,
        attribution=card.attribution,
    )


def fetch(
    card: "ModelCard",
    what: str,
    *,
    revision: Optional[str] = None,
    accept: Optional[str] = None,
    root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Download the card's weights into the cache and record them; returns the sidecar written.

    A licence that `licence_flagged` names needs `accept` equal to the card's `licence` (`LicenceNotAcceptedError` before any
    download). Each file's sha256 must be the card's (`WeightsMismatchError`, nothing recorded) unless `revision` was given, which
    records the new hashes. A card that records no hash for a file gets the one downloaded. The library never accepts a gated
    repository's terms: that needs the user's own Hugging Face login and click-through."""
    plan = fetch_plan(card, what, revision, root)
    if plan.flagged and (accept or "").strip() != card.licence:
        reason = "non-commercial" if not card.commercial_use else f"revenue cap: {card.revenue_cap}"
        raise LicenceNotAcceptedError(
            f"{what}'s weights are under {card.licence!r} ({reason}; {card.licence_url}). Fetching them accepts that "
            f"licence, which only you can do: pass --accept {card.licence!r} (accept={card.licence!r})."
        )
    ref = card.weights
    plan.destination.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = snapshot_download(
            repo_id=plan.repo,
            revision=plan.revision,
            allow_patterns=list(plan.files) or None,
            local_dir=str(plan.destination),
        )
    except ModuleNotFoundError as error:
        if (error.name or "").split(".")[0] != "huggingface_hub":
            raise
        install = f"pip install 'epftoolbox2[{card.extra}]'" if card.extra else "pip install huggingface_hub"
        raise MissingExtraError(
            f"epf weights fetch {what}", card.extra or "", f"epf weights fetch {what} needs huggingface_hub: {install}"
        ) from error
    directory = Path(downloaded) if downloaded else plan.destination
    names = list(plan.files) or _listed_files(directory)
    missing = [name for name in names if not (directory / name).is_file()]
    if missing or not names:
        raise WeightsMismatchError(
            f"{what}: the download of {plan.repo}@{plan.revision} has no {', '.join(missing) or 'file'}; nothing recorded"
        )
    files: Dict[str, Dict[str, Any]] = {}
    differ = []
    for name in names:
        path = directory / name
        digest = sha256_of(path, cached=False)
        files[name] = {"sha256": digest, "bytes": path.stat().st_size}
        pinned = ref.files.get(name)
        if pinned and pinned != digest:
            differ.append(f"{name}: sha256 {digest} downloaded, {pinned} on the card")
    if differ and not plan.explicit_revision:
        raise WeightsMismatchError(
            f"{what}: {plan.repo}@{plan.revision} is not the files the card pins, so upstream replaced them:\n  - "
            + "\n  - ".join(differ)
            + f"\nNothing was recorded, and no run uses the files left in {directory}. To use them anyway, fetch again with "
            f"--revision {plan.revision} given explicitly, which records these hashes (a run then warns: not the reference "
            "weights)."
        )
    from .. import __version__

    sidecar = {
        "repo": plan.repo,
        "revision": plan.revision,
        "family": card.family,
        "files": files,
        "bytes": sum(entry["bytes"] for entry in files.values()),
        "licence": card.licence,
        "licence_url": card.licence_url,
        "accepted": card.licence if plan.flagged else None,
        "reference": not differ if any(ref.files.values()) else None,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "epftoolbox2": __version__,
    }
    target = directory / SIDECAR
    partial = target.with_name(target.name + ".partial")
    partial.write_text(json.dumps(sidecar, indent=2) + "\n")
    partial.replace(target)
    return sidecar


def cached(root: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Every fetch recorded in the cache: each sidecar's content with `directory` (where it is), by repo and revision."""
    base = Path(root).expanduser() if root is not None else weights_dir()
    if not base.is_dir():
        return []
    found = []
    for sidecar in base.rglob(SIDECAR):
        try:
            content = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            content = {"repo": str(sidecar.parent.relative_to(base)), "revision": "?", "files": {}, "unreadable": True}
        found.append({**content, "directory": sidecar.parent})
    return sorted(found, key=lambda entry: (str(entry.get("repo")), str(entry.get("revision"))))


def size_text(n: Optional[int]) -> str:
    """Bytes as `837.3 MB`, `?` when unknown."""
    if n is None:
        return "?"
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} B"


def cache_rows(root: Optional[Union[str, Path]] = None) -> List[Tuple[str, str, str, str, str, str]]:
    """(weights, family, files, size, licence, fetched) of each fetch in the cache (`cached`), what `epf weights list` and
    `epftoolbox2.verify()` print. The licence shows the id accepted when fetching needed one."""
    rows = []
    for entry in cached(root):
        licence = str(entry.get("licence") or "?")
        if entry.get("accepted"):
            licence += " (accepted)"
        if entry.get("reference") is False:
            licence += "; not the card's hashes"
        rows.append(
            (
                f"{entry.get('repo')}@{entry.get('revision')}",
                str(entry.get("family") or "?"),
                str(len(entry.get("files") or {})),
                size_text(entry.get("bytes")),
                licence,
                str(entry.get("fetched_at") or "?"),
            )
        )
    return rows


class Finding(NamedTuple):
    """One file of `verify_cache`: `status` "ok", "changed" or "missing"."""

    directory: Path
    name: str
    status: str
    detail: str = ""


def verify_cache(root: Optional[Union[str, Path]] = None) -> List[Finding]:
    """Re-hash every file the cache's sidecars record (whatever `sha256_of` kept) and compare with the hash fetched."""
    findings = []
    for entry in cached(root):
        directory = entry["directory"]
        if entry.get("unreadable"):
            findings.append(Finding(directory, SIDECAR, "changed", "the sidecar can't be read"))
            continue
        for name, stored in entry.get("files", {}).items():
            path = directory / name
            if not path.is_file():
                findings.append(Finding(directory, name, "missing"))
                continue
            digest = sha256_of(path, cached=False)
            if digest == stored.get("sha256"):
                findings.append(Finding(directory, name, "ok"))
            else:
                findings.append(Finding(directory, name, "changed", f"sha256 {digest}, fetched {stored.get('sha256')}"))
    return findings
