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

import dataclasses
import importlib.metadata
import itertools
import json
import os
import re
import warnings
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

from .._extras import EXTRAS, MissingExtraError, has_extra
from ._weights import REPO_PATTERN, REVISION_PATTERN, MissingWeightsError, ResolvedWeights, resolve

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
        if not REPO_PATTERN.match(self.repo or "") or ".." in self.repo:
            raise ValueError(f"WeightsRef.repo must be 'owner/name', got {self.repo!r}")
        if not REVISION_PATTERN.match(self.revision or "") or ".." in self.revision:
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


# --- runtime pins: PEP 440 versions and specifiers (`packaging` is no core dependency, so its rules are here) -------------

_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(.*?)\s*$")
_SPECIFIER = re.compile(r"^\s*(?P<op>~=|===|==|!=|<=|>=|<|>)\s*(?P<version>[^\s,;]*)\s*$")
_VERSION_PARTS = r"""
    v?
    (?:(?P<epoch>[0-9]+)!)?
    (?P<release>[0-9]+(?:\.[0-9]+)*)
    (?P<pre>[-_.]?(?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)[-_.]?(?P<pre_n>[0-9]+)?)?
    (?P<post>(?:-(?P<post_n1>[0-9]+))|(?:[-_.]?(?P<post_l>post|rev|r)[-_.]?(?P<post_n2>[0-9]+)?))?
    (?P<dev>[-_.]?(?P<dev_l>dev)[-_.]?(?P<dev_n>[0-9]+)?)?
    (?:\+(?P<local>[a-z0-9]+(?:[-_.][a-z0-9]+)*))?
"""
_VERSION = re.compile(r"^\s*" + _VERSION_PARTS + r"\s*$", re.VERBOSE | re.IGNORECASE)
# `==`/`!=` with a wildcard: an epoch and a release only, then `.*`.
_WILDCARD = re.compile(r"^v?(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*\.\*$", re.IGNORECASE)
_PRE_LETTER = {"alpha": "a", "beta": "b", "c": "rc", "pre": "rc", "preview": "rc"}
_PRE_RANK = {"a": 0, "b": 1, "rc": 2}
_SPLIT_PRE = re.compile(r"^([0-9]+)((?:a|b|c|rc)[0-9]+)$")


@dataclass(frozen=True)
class _Version:
    """A PEP 440 version, normalised: compare by `key`, print canonically with `str`."""

    epoch: int
    release: Tuple[int, ...]
    pre: Optional[Tuple[str, int]]
    post: Optional[int]
    dev: Optional[int]
    local: Optional[Tuple[Union[int, str], ...]]

    @property
    def is_prerelease(self) -> bool:
        return self.pre is not None or self.dev is not None

    @property
    def public(self) -> "_Version":
        return dataclasses.replace(self, local=None)

    @property
    def base(self) -> "_Version":
        return _Version(self.epoch, self.release, None, None, None, None)

    @property
    def key(self) -> Tuple:
        """PEP 440's order: trailing zeros of the release don't count; a dev-only release before its pre-releases, those
        before the release, then post-releases, each dev release before its own; a local version after the public one."""
        release = list(self.release)
        while release and release[-1] == 0:
            release.pop()
        if self.pre is None and self.post is None and self.dev is not None:
            pre: Tuple = (0,)
        elif self.pre is None:
            pre = (2,)
        else:
            pre = (1, _PRE_RANK[self.pre[0]], self.pre[1])
        post = (0,) if self.post is None else (1, self.post)
        dev = (1,) if self.dev is None else (0, self.dev)
        local: Tuple = (0,)
        if self.local is not None:
            local = (1, tuple((1, part, "") if isinstance(part, int) else (0, 0, part) for part in self.local))
        return (self.epoch, tuple(release), pre, post, dev, local)

    def __str__(self) -> str:
        text = (f"{self.epoch}!" if self.epoch else "") + ".".join(str(part) for part in self.release)
        if self.pre is not None:
            text += f"{self.pre[0]}{self.pre[1]}"
        if self.post is not None:
            text += f".post{self.post}"
        if self.dev is not None:
            text += f".dev{self.dev}"
        if self.local is not None:
            text += "+" + ".".join(str(part) for part in self.local)
        return text


def parse_version(text: str) -> Optional[_Version]:
    """`text` as a PEP 440 version, or None for one that isn't."""
    match = _VERSION.match(text or "")
    if not match:
        return None
    pre = None
    if match["pre_l"]:
        letter = match["pre_l"].lower()
        pre = (_PRE_LETTER.get(letter, letter), int(match["pre_n"] or 0))
    post = None
    if match["post_n1"] is not None:
        post = int(match["post_n1"])
    elif match["post_l"]:
        post = int(match["post_n2"] or 0)
    local = None
    if match["local"]:
        local = tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"[-_.]", match["local"]))
    return _Version(
        int(match["epoch"] or 0),
        tuple(int(part) for part in match["release"].split(".")),
        pre,
        post,
        int(match["dev_n"] or 0) if match["dev_l"] else None,
        local,
    )


def _check_specifier(operator: str, version: str) -> bool:
    """Whether `operator version` is a PEP 440 specifier: `===` takes any text; `==` and `!=` a version (a local one too) or
    a release with `.*`; `~=` a version with at least two release parts and no local part; the others a version without a
    local part or wildcard."""
    if operator == "===":
        return bool(version)
    if operator in ("==", "!=") and _WILDCARD.match(version):
        return True
    parsed = parse_version(version)
    if parsed is None:
        return False
    if operator in ("==", "!="):
        return True
    return parsed.local is None and (operator != "~=" or len(parsed.release) >= 2)


def parse_pin(text: str) -> Tuple[str, Tuple[Tuple[str, str], ...], Optional[str]]:
    """(distribution name, ((operator, version), ...), commit) of a runtime entry: `"torch"`, `"tabpfn>=6,<10"` (PEP 440
    specifiers, no markers or extras) or `"timestral@a534bdb..."` (a VCS commit or its prefix). ValueError on anything else,
    such as a specifier PEP 440 doesn't allow (`~=1`, `<=2.3+cu121`, `>=2.*`)."""
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
    for part in rest.split(",") if rest else ():
        spec = _SPECIFIER.match(part)
        if not spec or not _check_specifier(spec["op"], spec["version"]):
            raise ValueError(f"Can't read the runtime pin {text!r}: {part.strip()!r} is no PEP 440 specifier")
        specifiers.append((spec["op"], spec["version"]))
    return name, tuple(specifiers), None


def _version_split(text: str) -> List[str]:
    """PEP 440's components of a canonical version for prefix matching: the epoch, then each dot-separated part, a release
    part with a pre-release split in two (`0rc1` -> `0`, `rc1`)."""
    epoch, _, rest = text.rpartition("!")
    out = [epoch or "0"]
    for item in rest.split("."):
        found = _SPLIT_PRE.match(item)
        out.extend(found.groups() if found else [item])
    return out


def _prefix_matches(candidate: _Version, prefix: List[str]) -> bool:
    """Whether the candidate's public version starts with `prefix` (components of `_version_split`), its release padded with
    zeros to the prefix's length: `==1.0.*` holds for 1, 1.0.3 and 1.0rc1, not 1.1."""
    mine = _version_split(str(candidate.public))
    release = list(itertools.takewhile(str.isdigit, mine))
    wanted = list(itertools.takewhile(str.isdigit, prefix))
    padded = release + ["0"] * max(0, len(wanted) - len(release)) + mine[len(release) :]
    return padded[: len(prefix)] == prefix


def _matches(candidate: _Version, operator: str, text: str) -> bool:
    """One PEP 440 specifier, with pre-releases allowed (an installed pre-release is what it is): `<V` excludes V's own
    pre-releases unless V is one, `>V` V's post-releases unless V is one and V's local versions; `==V` compares the public
    version unless V has a local part; `<=`, `>=` and `~=` compare the public version."""
    if operator == "===":
        return str(candidate).lower() == text.lower()
    if text.endswith(".*"):
        prefix = _version_split(str(parse_version(text[:-2])))
        return _prefix_matches(candidate, prefix) == (operator == "==")
    wanted = parse_version(text)
    if operator in ("==", "!="):
        mine = candidate if wanted.local is not None else candidate.public
        return (mine.key == wanted.key) == (operator == "==")
    if operator == "~=":
        prefix = [str(wanted.epoch), *(str(part) for part in wanted.release[:-1])]
        return candidate.public.key >= wanted.key and _prefix_matches(candidate, prefix)
    if operator == "<=":
        return candidate.public.key <= wanted.key
    if operator == ">=":
        return candidate.public.key >= wanted.key
    same_base = candidate.base.key == wanted.base.key
    if operator == "<":
        return candidate.key < wanted.key and not (not wanted.is_prerelease and candidate.is_prerelease and same_base)
    return (
        candidate.key > wanted.key
        and not (wanted.post is None and candidate.post is not None and same_base)
        and not (candidate.local is not None and same_base)
    )


def version_matches(version: str, specifiers: Iterable[Tuple[str, str]]) -> bool:
    """Whether the installed `version` meets every (operator, version) of a pin (`parse_pin`), by PEP 440 with pre-releases
    allowed. False for a version PEP 440 can't read."""
    candidate = parse_version(version)
    return candidate is not None and all(_matches(candidate, operator, text) for operator, text in specifiers)


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
    error) per card, path and fetch command for one pre-flight, so a loop of a thousand variants hashes once (and warns once,
    naming the first of them)."""
    card = model._card
    weights = getattr(model, "weights", None)
    command = fetch_command(model)
    who = describe(model)
    key = ("weights", id(card), str(weights), command)
    if seen is not None and key in seen:
        first, found = seen[key]
    else:
        first = who
        try:
            found = resolve(card, weights, what=who, fetch_command=command)
        except MissingWeightsError as error:
            found = error
        if seen is not None:
            seen[key] = (first, found)
    if isinstance(found, MissingWeightsError):
        raise MissingWeightsError(found.message.replace(first, who, 1))
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
        warned = (
            "not the reference",
            None if resolved is None else (resolved.ref, tuple(sorted(resolved.files.items()))),
        )
        if resolved is not None and resolved.reference is False and warned not in seen:
            seen[warned] = True  # once per weights in a pre-flight, not once per variant of a loop
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
        out = {"card": manifest.card, "runtime": manifest.runtime, "weights": manifest.weights}
        if manifest.weights_history:
            out["weights_history"] = manifest.weights_history
        return out
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
    out = {"card": stored["card"], "runtime": stored.get("runtime"), "weights": stored.get("weights")}
    if stored.get("weights_history"):
        out["weights_history"] = stored["weights_history"]
    return out


def _short_hashes(weights: Optional[Mapping[str, Any]]) -> str:
    files = (weights or {}).get("files") or {}
    if len(files) == 1:
        return next(iter(files.values()))[:12]
    return "; ".join(f"{name} {digest[:12]}" for name, digest in files.items())


def _weights_text(weights: Optional[Mapping[str, Any]], history: Optional[Sequence[Any]]) -> Optional[str]:
    ref = (weights or {}).get("ref")
    if ref is not None and history and len(history) > 1:
        return f"{ref} (the results file mixes {len(history)} sets of weights: weights_history)"
    return ref


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
                "weights": _weights_text(weights, record.get("weights_history")),
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
