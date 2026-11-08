"""Model cards: what a model family is under (licence, corpus, runtime packages, weights), checked before a run and recorded.

Every class whose fit needs an extra or weights declares one `ModelCard` as its `_card` (plan section 13: one place per
component; core models have none, and nothing of theirs changes). From it:

- **The pre-flight** (`preflight`, through `BaseModel._preflight`): `ModelPipeline.run`, each `Workflow` array task and `verify`
  check every model before the first one runs: the extra's modules found (`has_extra`), the card's runtime distributions installed
  within their pins (`importlib.metadata`, a commit pin read from PEP 610's `direct_url.json`), and the weights resolved
  (`_weights.resolve`: the cache or the model's `weights=` path, sha256). Every problem of every model goes into one error,
  `MissingWeightsError` when weights are among them, else `MissingExtraError`, so a long run never dies halfway.
- **The manifest** (`manifest_fields`, read by `results.manifest.build_manifest`): `card` (provenance, never compared), `runtime`
  (the version, and VCS commit, of each runtime distribution, plus the model's `_runtime_fields()`: device type, dtype, ...)
  and `weights` (each file's sha256), both compared on resume. A core model has none of the three keys.
- **The contamination verdict** of each window (`contamination_verdict`): `clean` when the corpus ends before the window's first
  issue day or the model trains on the run's data only, `clean (released before)` when the corpus end is undocumented and the
  model was released before that day, `possibly contaminated` otherwise; a card's `stated_verdict` wins.
- **The report** (`provenance_frame`, `EvaluationReport.provenance()`): one row per model with a card, which the terminal, CSV
  and Excel exporters print.

Nothing here imports an extra.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import warnings
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

from .._extras import EXTRAS, MissingExtraError, has_extra
from ._weights import MissingWeightsError, ResolvedWeights, resolve

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

# `ModelCard.corpus` of a family trained on the run's own data (networks, trees): its verdict is always `clean`.
RUN_DATA = "the run's data"
# The verdicts `contamination_verdict` gives (a card's `stated_verdict` is its own text).
CLEAN = "clean"
CLEAN_RELEASED_BEFORE = "clean (released before)"
POSSIBLY_CONTAMINATED = "possibly contaminated"

# The columns of `EvaluationReport.provenance()`.
PROVENANCE_COLUMNS = [
    "model",
    "family",
    "licence",
    "licence_url",
    "commercial_use",
    "revenue_cap",
    "attribution",
    "weights",
    "sha256",
    "runtime",
    "corpus",
    "corpus_end",
    "released",
    "validation_verdict",
    "test_verdict",
]

_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")
_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _as_date(value: Union[date, str, None], what: str) -> Optional[date]:
    if value is None or (isinstance(value, date) and not hasattr(value, "hour")):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    if hasattr(value, "date"):
        return value.date()
    raise TypeError(f"{what} must be a date or an ISO date string, got {value!r}")


@dataclass(frozen=True)
class WeightsRef:
    """Where a family's weights come from: a Hugging Face `repo` at `revision`, the `files` a run needs, each with its sha256
    (None where the research recorded none: the first fetch records it), and their total `bytes` (None: unknown)."""

    repo: str
    revision: str
    files: Mapping[str, Optional[str]]
    bytes: Optional[int] = None

    def __post_init__(self):
        if not _REPO.match(self.repo or ""):
            raise ValueError(f"WeightsRef.repo must be 'owner/name', got {self.repo!r}")
        if not _REVISION.match(self.revision or ""):
            raise ValueError(f"WeightsRef.revision must be a branch, tag or commit, got {self.revision!r}")
        files = dict(self.files)
        for name, digest in files.items():
            parts = name.split("/")
            if not name or name.startswith("/") or ".." in parts or "" in parts:
                raise ValueError(f"WeightsRef.files names a file by its path in the repository, got {name!r}")
            if digest is not None and not _SHA256.match(digest):
                raise ValueError(f"WeightsRef.files[{name!r}] must be a sha256 (64 hex digits) or None, got {digest!r}")
        object.__setattr__(self, "files", files)

    def record(self) -> Dict[str, Any]:
        return {"repo": self.repo, "revision": self.revision, "files": dict(self.files), "bytes": self.bytes}


@dataclass(frozen=True)
class ModelCard:
    """A family's licence, corpus, runtime and weights (step 6 spec, "Cards"). `extra` is the extra its fit needs (None for a
    family of the core, such as a scikit-learn tree), `runtime` the distributions whose versions are recorded and compared, each a
    name with an optional pin (`"tfc-t0==0.5.0"`, `"tabpfn>=6,<10"`, `"timestral@a534bdb..."` for a VCS commit, `"torch"`),
    `corpus` what the weights were trained on (`RUN_DATA` for a family trained on the run's data), `corpus_end` its last day and
    `released` the weights' release, which decide the verdict, and `stated_verdict` a verdict that overrides that rule, with its
    reason (`"clean (synthetic): trained on synthetic priors only"`)."""

    family: str
    extra: Optional[str]
    licence: str
    licence_url: str
    commercial_use: bool
    revenue_cap: Optional[str]
    attribution: Optional[str]
    weights: Optional[WeightsRef]
    runtime: Tuple[str, ...]
    corpus: str
    corpus_end: Optional[date]
    released: Optional[date]
    stated_verdict: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "runtime", tuple(self.runtime))
        object.__setattr__(self, "corpus_end", _as_date(self.corpus_end, "ModelCard.corpus_end"))
        object.__setattr__(self, "released", _as_date(self.released, "ModelCard.released"))
        for pin in self.runtime:
            parse_pin(pin)  # raises on a pin it can't read
        # `extra` is checked against `EXTRAS` by the pre-flight (`has_extra` raises on an unknown one), not here: a card is
        # made when its module is imported.

    def record(self) -> Dict[str, Any]:
        """The card as a manifest's provenance holds it (JSON values)."""
        out: Dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, date):
                value = value.isoformat()
            elif isinstance(value, WeightsRef):
                value = value.record()
            elif isinstance(value, tuple):
                value = list(value)
            out[item.name] = value
        return out


def contamination_verdict(
    card: Union[ModelCard, Mapping[str, Any]], first_issue_day: Union[date, str, None]
) -> Optional[str]:
    """The verdict of a window whose first issue day is `first_issue_day`, from a card or its record: the card's
    `stated_verdict` if it has one; `clean` for a family trained on the run's data, or a corpus that ends before that day;
    `clean (released before)` for a corpus with no documented end and a release before that day; else `possibly contaminated`.
    None without a day."""

    def get(name: str) -> Any:
        return card.get(name) if isinstance(card, Mapping) else getattr(card, name)

    if get("stated_verdict"):
        return get("stated_verdict")
    if get("corpus") == RUN_DATA:
        return CLEAN
    day = _as_date(first_issue_day, "first_issue_day") if first_issue_day not in (None, "") else None
    if day is None:
        return None
    corpus_end = _as_date(get("corpus_end"), "corpus_end")
    if corpus_end is not None:
        return CLEAN if corpus_end < day else POSSIBLY_CONTAMINATED
    released = _as_date(get("released"), "released")
    if released is not None and released < day:
        return CLEAN_RELEASED_BEFORE
    return POSSIBLY_CONTAMINATED


# --- runtime pins -----------------------------------------------------------------------------------------------------

_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(.*?)\s*$")
_SPECIFIER = re.compile(r"^\s*(===|==|!=|~=|<=|>=|<|>)\s*([A-Za-z0-9.*+!_-]+)\s*$")
_VERSION = re.compile(
    r"^v?(?:(\d+)!)?(\d+(?:\.\d+)*)"
    r"(?:[-_.]?(a|alpha|b|beta|rc|c|pre|preview)[-_.]?(\d*))?"
    r"(?:-(\d+)|[-_.]?(?:post|rev|r)[-_.]?(\d*))?"
    r"(?:[-_.]?dev[-_.]?(\d*))?"
    r"(?:\+[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*)?$",
    re.IGNORECASE,
)
_PRE_RANK = {"a": 0, "alpha": 0, "b": 1, "beta": 1, "rc": 2, "c": 2, "pre": 2, "preview": 2}


def parse_pin(text: str) -> Tuple[str, Tuple[Tuple[str, str], ...], Optional[str]]:
    """(distribution name, ((operator, version), ...), commit) of a runtime entry: `"torch"`, `"tabpfn>=6,<10"` (PEP 440
    specifiers, no markers or extras) or `"timestral@a534bdb..."` (a VCS commit or its prefix). ValueError on anything else."""
    match = _NAME.match(text or "")
    if not match:
        raise ValueError(f"Can't read the runtime pin {text!r}: expected a distribution name and optional specifiers")
    name, rest = match.groups()
    if rest.startswith("@"):
        commit = rest[1:].strip()
        if not re.fullmatch(r"[0-9a-f]{7,40}", commit):
            raise ValueError(f"Can't read the runtime pin {text!r}: a commit pin is 7 to 40 hex digits after '@'")
        return name, (), commit
    specifiers = []
    for part in [p for p in rest.split(",") if p.strip()]:
        spec = _SPECIFIER.match(part)
        if not spec or (spec.group(2).endswith(".*") and spec.group(1) not in ("==", "!=")):
            raise ValueError(f"Can't read the runtime pin {text!r} (specifier {part.strip()!r})")
        if _version_key(spec.group(2).removesuffix(".*")) is None:
            raise ValueError(f"Can't read the runtime pin {text!r} (version {spec.group(2)!r})")
        specifiers.append((spec.group(1), spec.group(2)))
    return name, tuple(specifiers), None


def _version_key(text: str) -> Optional[Tuple]:
    """A sort key of a PEP 440 version (epoch, release without trailing zeros, pre, post, dev; the local part ignored), or
    None for a version it can't read."""
    match = _VERSION.match(text.strip())
    if not match:
        return None
    # An optional part that matched has its number group set ("" without digits); one that didn't is None.
    epoch, release, pre_letter, pre_number, post_dash, post_number, dev = match.groups()
    numbers = [int(part) for part in release.split(".")]
    while len(numbers) > 1 and numbers[-1] == 0:
        numbers.pop()
    post = None if post_dash is None and post_number is None else int(post_dash or post_number or 0)
    if pre_letter is not None:
        pre_key: Tuple = (1, _PRE_RANK[pre_letter.lower()], int(pre_number or 0))
    elif post is None and dev is not None:
        pre_key = (0,)  # 1.0.dev1 comes before 1.0a1
    else:
        pre_key = (2,)
    post_key = (0,) if post is None else (1, post)
    dev_key = (0, int(dev or 0)) if dev is not None else (1,)
    return (int(epoch or 0), tuple(numbers), pre_key, post_key, dev_key)


def _release(text: str) -> List[int]:
    match = _VERSION.match(text.strip())
    return [int(part) for part in match.group(2).split(".")] if match else []


def version_matches(version: str, specifiers: Iterable[Tuple[str, str]]) -> bool:
    """Whether `version` meets every (operator, version) of a pin: PEP 440's comparisons, `==X.*`/`!=X.*` prefixes and
    `~=` (pre-releases are compared as versions, never excluded)."""
    key = _version_key(version)
    if key is None:
        return False
    for operator, wanted in specifiers:
        if operator == "===":
            ok = version.strip() == wanted
        elif wanted.endswith(".*"):
            prefix = _release(wanted[:-2])
            mine = _release(version) + [0] * len(prefix)
            ok = mine[: len(prefix)] == prefix
            ok = ok if operator == "==" else not ok
        else:
            other = _version_key(wanted)
            if operator == "~=":
                prefix = _release(wanted)[:-1]
                ok = key >= other and (_release(version) + [0] * len(prefix))[: len(prefix)] == prefix
            else:
                ok = {
                    "==": key == other,
                    "!=": key != other,
                    "<=": key <= other,
                    ">=": key >= other,
                    "<": key < other,
                    ">": key > other,
                }[operator]
        if not ok:
            return False
    return True


def installed_distribution(name: str, seen: Optional[Dict] = None) -> Optional[Tuple[str, Optional[str]]]:
    """(version, VCS commit or None) of an installed distribution, or None when it isn't installed. The commit is PEP 610's
    `vcs_info.commit_id` (`direct_url.json`) of a package installed from a repository. `seen` keeps answers for one pre-flight."""
    key = ("distribution", name)
    if seen is not None and key in seen:
        return seen[key]
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        found = None
    else:
        commit = None
        try:
            direct = distribution.read_text("direct_url.json")
            commit = json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None
        except (OSError, ValueError, AttributeError):
            commit = None
        found = (distribution.version, commit)
    if seen is not None:
        seen[key] = found
    return found


def runtime_versions(card: ModelCard, seen: Optional[Dict] = None) -> Dict[str, Optional[str]]:
    """Each runtime distribution of the card -> its version, `version@commit` for one installed from a repository, None when it
    isn't installed: what a manifest records and a resume compares."""
    out: Dict[str, Optional[str]] = {}
    for pin in card.runtime:
        name = parse_pin(pin)[0]
        found = installed_distribution(name, seen)
        out[name] = None if found is None else (found[0] if found[1] is None else f"{found[0]}@{found[1]}")
    return out


# --- the pre-flight -----------------------------------------------------------------------------------------------------


class Problem(NamedTuple):
    """One thing a model lacks before a run: `kind` is "extra" (with the extra to install), "runtime" or "weights"."""

    model: str
    kind: str
    message: str
    extra: Optional[str] = None


def describe(model: Any) -> str:
    """`ClassName 'name'`, how a problem names a model."""
    return f"{type(model).__name__} {getattr(model, 'name', '?')!r}"


def class_path(cls: type) -> str:
    """How `epf weights` names a class: its name in `COMPONENT_REGISTRY["models"]` if it is registered there under its module,
    else `module:Class`."""
    from ..pipelines.model_pipeline import COMPONENT_REGISTRY

    if COMPONENT_REGISTRY["models"].get(cls.__name__) == cls.__module__:
        return cls.__name__
    return f"{cls.__module__}:{cls.__qualname__}"


def card_variant(model_or_cls: Any) -> Optional[str]:
    """The size (key of the class's `_card_variants`) whose card the model has, or None for a class with one card."""
    cls = model_or_cls if isinstance(model_or_cls, type) else type(model_or_cls)
    variants = getattr(cls, "_card_variants", None) or {}
    card = getattr(model_or_cls, "_card", None)
    return next((size for size, candidate in variants.items() if candidate is card), None)


def fetch_command(model: Any) -> str:
    """The `epf weights fetch` command that fetches this model's weights."""
    size = card_variant(model)
    return f"epf weights fetch {class_path(type(model))}" + (f" --size {size}" if size else "")


def resolve_model_weights(model: Any, seen: Optional[Dict] = None) -> Optional[ResolvedWeights]:
    """`_weights.resolve` for a model: its card and its `weights` parameter, if it has one. `seen` keeps one answer (or the
    error) per card weights and path for one pre-flight, so a loop of a thousand variants hashes once."""
    card = model._card
    weights = getattr(model, "weights", None)
    ref = None if card is None else card.weights
    command = fetch_command(model)
    key = (
        "weights",
        None if ref is None else (ref.repo, ref.revision, tuple(ref.files.items())),
        str(weights),
        command,
    )
    if seen is not None and key in seen:
        found = seen[key]
    else:
        try:
            found = resolve(card, weights, what="{model}", fetch_command=command)
        except MissingWeightsError as error:
            found = error
        if seen is not None:
            seen[key] = found
    if isinstance(found, MissingWeightsError):
        raise MissingWeightsError(found.message.replace("{model}", describe(model)))
    return found


def card_problems(model: Any, seen: Optional[Dict] = None) -> List[Problem]:
    """What `model` lacks to run, from its card: the extra's modules, the runtime pins (checked once the extra is there) and
    its weights. Empty for a model without a card. Warns when the weights are not the card's (`not the reference weights`)."""
    card = model._card
    if card is None:
        return []
    seen = {} if seen is None else seen
    who = describe(model)
    problems = []
    extra_ok = True
    if card.extra is not None:
        key = ("extra", card.extra)
        if key not in seen:
            seen[key] = has_extra(card.extra)
        extra_ok = seen[key]
        if not extra_ok:
            modules = ", ".join(EXTRAS[card.extra])
            problems.append(
                Problem(
                    who,
                    "extra",
                    f"needs the [{card.extra}] extra ({modules} not found): pip install 'epftoolbox2[{card.extra}]'",
                    card.extra,
                )
            )
    if extra_ok:
        for pin in card.runtime:
            name, specifiers, commit = parse_pin(pin)
            found = installed_distribution(name, seen)
            install = f"pip install 'epftoolbox2[{card.extra}]'" if card.extra else f"pip install '{pin}'"
            if found is None:
                problems.append(Problem(who, "runtime", f"needs {pin}, which is not installed: {install}", card.extra))
            elif commit is not None and not (found[1] or "").startswith(commit):
                where = f"commit {found[1]}" if found[1] else "not from a repository"
                problems.append(
                    Problem(
                        who, "runtime", f"needs {name} at commit {commit}, installed: {found[0]} ({where})", card.extra
                    )
                )
            elif specifiers and not version_matches(found[0], specifiers):
                problems.append(
                    Problem(who, "runtime", f"needs {pin}, installed: {name} {found[0]}: {install}", card.extra)
                )
    try:
        resolved = resolve_model_weights(model, seen)
    except MissingWeightsError as error:
        problems.append(Problem(who, "weights", error.message.removeprefix(f"{who}: ")))
    else:
        if resolved is not None and resolved.reference is False:
            warnings.warn(
                f"{who}: the weights {resolved.ref} are not the reference weights (their sha256 differs from the card's "
                f"{card.weights.repo}@{card.weights.revision}); the manifest records the hashes used.",
                UserWarning,
                stacklevel=5,  # card_problems, _preflight, preflight, the pipeline's run, its caller
            )
    return problems


def preflight(models: Sequence[Any]) -> None:
    """Check every model before any of them runs (`BaseModel._preflight`) and raise one error naming every problem:
    `MissingWeightsError` when any weights are missing, else `MissingExtraError` with every extra to install. Models without a
    card cost a dict lookup each."""
    seen: Dict = {}
    problems: List[Problem] = []
    for model in models:
        problems.extend(model._preflight(seen))
    if not problems:
        return
    grouped: Dict[Tuple[str, str], List[str]] = {}
    for problem in problems:
        grouped.setdefault((problem.kind, problem.message), []).append(problem.model)
    lines = []
    for (kind, message), names in grouped.items():
        shown = ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")
        lines.append(f"{shown}: {message}")
    header = (
        f"{len(lines)} problem{'s' if len(lines) > 1 else ''} found before any model ran (nothing was fitted; the pre-flight "
        "checks every model's extra, runtime packages and weights first):"
    )
    message = header + "\n" + "\n".join(f"  - {line}" for line in lines)
    if any(problem.kind == "weights" for problem in problems):
        raise MissingWeightsError(message, lines)
    extras = sorted({problem.extra for problem in problems if problem.extra})
    who = ", ".join(dict.fromkeys(problem.model for problem in problems))
    raise MissingExtraError(who, ",".join(extras), message)


# --- the manifest and the report ----------------------------------------------------------------------------------------


def window_starts(windows: Any, test_start: Optional[str]) -> Dict[str, Optional[str]]:
    """The first issue day of each window of a run: `windows.periods` of a run with validation, else the test start."""
    if windows is not None:
        return {name: start for name, (start, _) in windows.periods}
    return {"test": test_start}


def manifest_fields(model: Any, starts: Mapping[str, Optional[str]]) -> Dict[str, Any]:
    """`card` (with the verdict of each window in `starts`), `runtime` and `weights` of a model's manifest; nothing for a model
    without a card, whose manifest keeps its keys. Weights that can't be resolved raise `MissingWeightsError`."""
    card = getattr(model, "_card", None)
    if card is None:
        return {}
    record = card.record()
    record["verdicts"] = {name: contamination_verdict(card, start) for name, start in starts.items()}
    resolved = resolve_model_weights(model)
    return {
        "card": record,
        "runtime": {"packages": runtime_versions(card), **dict(model._runtime_fields())},
        "weights": None if resolved is None else resolved.record(),
    }


def run_provenance(model: Any, manifest: Any, windows: Any, test_start: Optional[str]) -> Optional[Dict[str, Any]]:
    """What a run's ref holds for `EvaluationReport.provenance()`: the manifest's `card`, `runtime` and `weights` when the run
    wrote one, else the model's own; None for a model without a card."""
    if getattr(model, "_card", None) is None:
        return None
    if manifest is not None and getattr(manifest, "card", None) is not None:
        return {"card": manifest.card, "runtime": manifest.runtime, "weights": manifest.weights}
    return manifest_fields(model, window_starts(windows, test_start))


def read_provenance(stored: Union[str, "os.PathLike", Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """`run_provenance` of a manifest file (its path, or its content as a dict), or None without a manifest or a card in it."""
    if not isinstance(stored, Mapping):
        path = Path(stored)
        if not path.is_file():
            return None
        stored = json.loads(path.read_text())
    if not stored.get("card"):
        return None
    return {"card": stored["card"], "runtime": stored.get("runtime"), "weights": stored.get("weights")}


def _short_hashes(weights: Optional[Mapping[str, Any]]) -> str:
    files = (weights or {}).get("files") or {}
    if len(files) == 1:
        return next(iter(files.values()))[:12]
    return "; ".join(f"{name} {digest[:12]}" for name, digest in files.items())


def _runtime_text(runtime: Optional[Mapping[str, Any]]) -> str:
    runtime = dict(runtime or {})
    packages = runtime.pop("packages", {}) or {}
    shown = [f"{name} {version if version is not None else 'not installed'}" for name, version in packages.items()]
    settings = [f"{name}={value}" for name, value in runtime.items()]
    return "; ".join(part for part in (", ".join(shown), ", ".join(settings)) if part)


def provenance_frame(
    records: Mapping[str, Optional[Mapping[str, Any]]], starts: Mapping[str, Mapping[str, Any]]
) -> "pd.DataFrame":
    """`EvaluationReport.provenance()`: one row per model whose record (`run_provenance`) has a card, in `records` order, with
    the verdict of each window in `starts[model]` (window -> first issue day). Empty, with the same columns, when none has."""
    import pandas as pd

    rows = []
    for name, record in records.items():
        if not record or not record.get("card"):
            continue
        card = record["card"]
        weights = record.get("weights")
        verdicts = {window: contamination_verdict(card, start) for window, start in starts.get(name, {}).items()}
        rows.append(
            {
                "model": name,
                "family": card.get("family"),
                "licence": card.get("licence"),
                "licence_url": card.get("licence_url"),
                "commercial_use": card.get("commercial_use"),
                "revenue_cap": card.get("revenue_cap"),
                "attribution": card.get("attribution"),
                "weights": (weights or {}).get("ref"),
                "sha256": _short_hashes(weights) or None,
                "runtime": _runtime_text(record.get("runtime")) or None,
                "corpus": card.get("corpus"),
                "corpus_end": card.get("corpus_end"),
                "released": card.get("released"),
                "validation_verdict": verdicts.get("validation"),
                "test_verdict": verdicts.get("test"),
            }
        )
    return pd.DataFrame(rows, columns=PROVENANCE_COLUMNS)
