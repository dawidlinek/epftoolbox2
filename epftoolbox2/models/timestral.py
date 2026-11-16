"""`TimestralModel`: Mistral AI's Timestral (209M, 123M or 28M), zero-shot with known-ahead and past covariates (step 6j), on
`FoundationModel` (`models/foundation.py`), its backend in `models/_fm/timestral.py`.

**Flagged.** The weights are under a modified MIT licence: no rights at all for a user whose company, or employer, had more
than USD 20 million of global consolidated revenue in the preceding month, and the attribution "2026 - Mistral AI" goes with
every copy (`epf weights fetch TimestralModel --accept Mistral-Modified-MIT`). The training corpus is undocumented (synthetic
pretraining, real and synthetic post-training) and the weights were released on 2026-10-15, so every window before that is
`possibly contaminated`.

**Runtime.** The `timestral` package is not on PyPI: the `[timestral]` extra installs its PyPI dependencies, and the package
comes from the commit the research ran (`_extras.TIMESTRAL_COMMIT`, installed with `_extras.install_line("timestral")`). The
pre-flight refuses another installed commit, or a build not installed from the repository (PEP 610's `direct_url.json`), unless
the model says so: `runtime_commit="<commit>"` accepts that commit instead, and `weights=` (weights of the user's own) accepts
any build with a warning. The manifest records the commit installed and a resume compares it.

The research's Timestral-209M at 364 days of context with the load forecast known ahead was the best single model of the zoo:
18.48 / 15.90 EUR/MWh on the DE_LU / AT test year (`zoo_timestral/REPORT.md`).
"""

from __future__ import annotations

import re
import warnings
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .._extras import TIMESTRAL_COMMIT
from . import _cards
from ._cards import ModelCard, WeightsRef, installed_distribution, parse_pin
from ._weights import MissingWeightsError
from .foundation import FoundationModel, _Internal

__all__ = ["TimestralModel", "TIMESTRAL_CARDS", "TIMESTRAL_LICENCE", "TIMESTRAL_SIZES"]

TIMESTRAL_LICENCE = "Mistral-Modified-MIT"
_COMMIT = re.compile(r"[0-9a-f]{7,40}")


def _card(size: str, revision: str, config: str, digest: str, size_bytes: int) -> ModelCard:
    return ModelCard(
        family=f"timestral-{size}",
        extra="timestral",
        licence=TIMESTRAL_LICENCE,
        licence_url=f"https://huggingface.co/mistralai/Timestral-{size.upper()}/blob/main/LICENSE",
        commercial_use=True,
        revenue_cap=(
            "no rights at all if the global consolidated monthly revenue of the user's company, or of their employer, "
            "exceeded USD 20 million (or its equivalent) in the preceding month"
        ),
        attribution="2026 - Mistral AI",
        # The revisions the research pinned on 2026-10-20 and the repositories' LFS hashes (checked again 2026-11-10), and
        # config.json's sha256 at those revisions (209M's as the first `epf weights fetch` recorded it, 2026-11-10).
        weights=WeightsRef(
            f"mistralai/Timestral-{size.upper()}",
            revision,
            {"config.json": config, "model.safetensors": digest},
            bytes=size_bytes,
        ),
        runtime=(f"timestral@{TIMESTRAL_COMMIT}", "torch"),
        # Not on PyPI: a build from a local directory or a wheel is recorded by its files, not its version alone.
        local_builds=("timestral",),
        corpus=(
            "synthetic pretraining (8,192-step contexts), then post-training on real and synthetic series (32,768 steps); "
            "the real datasets and their dates are undocumented"
        ),
        corpus_end=None,
        released=date(2026, 10, 15),
    )


# The sizes the research ran (5M dropped: worse than LEAR in AT), each with its own weights and card.
TIMESTRAL_CARDS: Dict[str, ModelCard] = {
    "209m": _card(
        "209m",
        "9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
        "1764139bb6d37ee5ff543f470495c086cfa6c373f7134fc61c7e3d844538bfd8",
        "1456effb9927e3daa8c7234f10497e3424233869245a9c72b8ecebaf7ad2b7fa",
        837_271_648 + 1_130,
    ),
    "123m": _card(
        "123m",
        "5a147287addda85f6acd8da3fb8858fe9af7e8f7",
        "7bb4c042fb2d2ed6469b6758ba380dfeda3ab32df47690a87aec1527bdae79bf",
        "599fafa73cadbbc800b5a346ca900fa741ee3e5fd22e3b4a4430e151e74f4bc2",
        490_445_904 + 1_074,
    ),
    "28m": _card(
        "28m",
        "49b26409ac87b5935539a5d8ffe7f397b720db47",
        "0cdf44b8ea981191de53078e0a5ecbbd437046bd939a546d5dbe17868f71281f",
        "dfc83c6220937ce5fc6687b35a4df94cd25e28a2cfe6fb92a23dad6301b2015d",
        111_296_464 + 961,
    ),
}
TIMESTRAL_SIZES = tuple(TIMESTRAL_CARDS)


class TimestralModel(FoundationModel):
    """Timestral (Mistral AI; 209M, 123M or 28M), zero-shot: known-ahead covariates as its `known_covariates` (over the context
    and the steps ahead), past ones as its `past_covariates` (over the context); the forecast is the level nearest 0.5 of its
    21. Its licence caps the user's company's (or employer's) revenue, and its corpus is undocumented (see the module). Needs
    the `[timestral]` extra, the package from its commit and `epf weights fetch TimestralModel --size <size> --accept
    Mistral-Modified-MIT`.

    Args:
        size: "209m" (the research's best), "123m" or "28m" (the CPU option, about 3x cheaper than 209M).
        context: Local days of the target's history a forecast reads (Timestral reads up to 32,768 steps).
        known_ahead: Base column names read over the context and the steps ahead (its `known_covariates`).
        past: Base column names read over the context, NaN after their newest known step (its `past_covariates`).
        symmetric_averaging: Average the forecast of the context with the reflected forecast of its negation (the package's
            default and the research's).
        clip_nonnegative: Clamp the forecast at 0 where the context has no negative value. False by default: prices can be
            negative (the research ran it on, where it changed no forecast at 182 days of context or more).
        batch_size: Items per backend call, every call padded to it. Compared on resume.
        device: "auto", "cpu", "cuda" or "cuda:N".
        weights: A directory of the size's two files used as given (then any installed timestral build runs, with a warning),
            or `"<repo>@<revision>"` of the cache; None for the size's card.
        runtime_commit: Another commit of the timestral package to run on instead of the pinned one (7 to 40 hex digits); the
            pre-flight then requires that one, and the manifest records it.
        newest_target: `"d-1"` (the context ends at D-1) or `"published"` (where the target's rule says at the issue time).
        name: The model's name.
    """

    _card = TIMESTRAL_CARDS["209m"]
    _card_variants = TIMESTRAL_CARDS
    _default_name = "timestral"
    # Timestral's `context_length`: a longer context is cut to its newest 32,768 steps by the package.
    _max_context_steps = 32768
    # Zero-shot only here: the base's fine-tune settings are no parameters of this class.
    fine_tune_every = _Internal()
    fine_tune = _Internal()
    seeds = _Internal()
    random_state = _Internal()

    def __init__(
        self,
        size: str = "209m",
        context: int = 364,
        known_ahead: Sequence[str] = (),
        past: Sequence[str] = (),
        symmetric_averaging: bool = True,
        clip_nonnegative: bool = False,
        batch_size: int = 64,
        device: str = "auto",
        weights: Optional[str] = None,
        runtime_commit: Optional[str] = None,
        newest_target: str = "d-1",
        name: str = "timestral",
    ):
        self.size = size
        self.symmetric_averaging = symmetric_averaging
        self.clip_nonnegative = clip_nonnegative
        self.runtime_commit = runtime_commit
        super().__init__(
            context=context,
            known_ahead=known_ahead,
            past=past,
            batch_size=batch_size,
            device=device,
            weights=weights,
            newest_target=newest_target,
            name=name,
        )

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "size":
            if not isinstance(value, str) or value.lower() not in TIMESTRAL_CARDS:
                raise ValueError(f"TimestralModel: size is one of {list(TIMESTRAL_SIZES)}, got {value!r}")
            value = value.lower()
            self.__dict__["_card"] = TIMESTRAL_CARDS[value]
        elif key in ("symmetric_averaging", "clip_nonnegative"):
            if not isinstance(value, bool):
                raise ValueError(f"TimestralModel: {key} is True or False, got {value!r}")
        elif key == "runtime_commit":
            if value is not None and (not isinstance(value, str) or not _COMMIT.fullmatch(value)):
                raise ValueError(
                    f"TimestralModel: runtime_commit is None or a commit of the timestral package (7 to 40 lower-case hex "
                    f"digits), got {value!r}"
                )
        super().__setattr__(key, value)

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "context": self.context,
            "known_ahead": list(self.known_ahead),
            "past": list(self.past),
            "symmetric_averaging": self.symmetric_averaging,
            "clip_nonnegative": self.clip_nonnegative,
            "batch_size": self.batch_size,
        }

    def _predict_options(self) -> Dict[str, Any]:
        return {"symmetric_averaging": self.symmetric_averaging, "clip_nonnegative": self.clip_nonnegative}

    def _own_weights(self, seen: Optional[Dict] = None) -> bool:
        """Whether `weights=` names weights other than the card's (the user's own, `ResolvedWeights.reference` False): only
        then may another timestral build run. The card's files under another path, or a path that doesn't resolve (the
        pre-flight names that), keep the pinned commit required. `seen` is the pre-flight's, which resolves the weights
        once for every question (and warns once of files nobody listed)."""
        if self.weights is None:
            return False
        try:
            resolved = _cards.resolve_model_weights(self, seen)
        except MissingWeightsError:
            return False
        return resolved is not None and resolved.reference is False

    def _runtime_pins(self, seen: Optional[Dict] = None) -> List[str]:
        """The card's pins, the timestral commit replaced by `runtime_commit` when given, or by any build with weights of the
        user's own (`_own_weights`)."""
        pins = []
        for pin in self._card.runtime:
            name, _, commit = parse_pin(pin)
            if name == "timestral" and commit is not None:
                if self.runtime_commit is not None:
                    pin = f"timestral@{self.runtime_commit}"
                elif self._own_weights(seen):
                    pin = "timestral"
            pins.append(pin)
        return pins

    def _preflight(self, seen: Optional[Dict] = None) -> list:
        """The card's checks (`_runtime_pins` for the commit), the problem of another commit naming the way to run on it,
        and with `weights=` a warning when the installed build isn't the pinned commit."""
        seen = {} if seen is None else seen
        problems = []
        for problem in super()._preflight(seen):
            if problem.kind == "runtime" and problem.message.startswith("needs timestral at commit"):
                installed = installed_distribution("timestral", seen)
                if installed is not None and installed[1]:
                    problem = problem._replace(
                        message=f"{problem.message}, or run on the installed one: runtime_commit={installed[1][:12]!r}"
                    )
            problems.append(problem)
        installed = installed_distribution("timestral", seen)
        if self.runtime_commit is None and installed is not None and self._own_weights(seen):
            if not (installed[1] or "").startswith(TIMESTRAL_COMMIT) and ("timestral build", installed) not in seen:
                seen[("timestral build", installed)] = True
                where = f"commit {installed[1]}" if installed[1] else "not from a repository"
                warnings.warn(
                    f"TimestralModel {self.name!r}: the timestral package installed ({installed[0]}, {where}) isn't the "
                    f"pinned commit {TIMESTRAL_COMMIT[:12]}; it runs because weights= is given (the manifest records it)",
                    UserWarning,
                    stacklevel=4,
                )
        return problems

    def _load_backend(self, resolved: Any, device: Any, options: Mapping[str, Any]) -> Any:
        from ._fm.timestral import TimestralBackend

        return TimestralBackend(resolved, device)
