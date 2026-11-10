"""`PatchTSTFMModel`: IBM's PatchTST-FM-r2 (385M, Apache-2.0 or OpenMDW-1.0), zero-shot and univariate (step 6j), on
`FoundationModel` (`models/foundation.py`), its backend in `models/_fm/patchtst_fm.py`.

It reads the target's history alone, so it refuses covariates: it has no `known_ahead` or `past` parameter, and setting either
attribute raises. Of the context it reads the newest `8,192 - max(S, 128)` steps (8,064 for a day of hours ahead), and the mean
of the whole context fills the rest of its window: the research's choice, 341 days of hours (8,184 steps), gave 19.85 / 16.51
EUR/MWh on the DE_LU / AT test year. Needs the `[patchtst-fm]` extra (granite-tsfm, Python < 3.14) and `epf weights fetch
PatchTSTFMModel`.
"""

from __future__ import annotations

import sys
import warnings
from datetime import date
from typing import Any, Dict, List, Mapping, Optional

from ._calendar import periods_per_day
from ._cards import ModelCard, WeightsRef
from .foundation import FoundationModel, _Internal

__all__ = ["PatchTSTFMModel", "PATCHTST_FM_CARD", "PATCHTST_FM_WINDOW", "PATCHTST_FM_FORECAST_STEPS"]

# The package's window (its config's `context_length`), the forecast's steps inside it included, and the fewest steps it
# forecasts (`d_patch` 16 x twice `pretrain_mask_cont` 8): the newest 8,192 - max(S, 128) context steps go in.
PATCHTST_FM_WINDOW = 8192
PATCHTST_FM_FORECAST_STEPS = 128

PATCHTST_FM_CARD = ModelCard(
    family="patchtst-fm-r2",
    extra="patchtst-fm",
    licence="Apache-2.0 OR OpenMDW-1.0",
    licence_url="https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r2/blob/main/LICENSE",
    commercial_use=True,
    revenue_cap=None,
    attribution=None,
    # The revision the research ran (the repository's last change, 2026-09-09, before its runs of 2026-10), pinned 2026-11-10: both
    # hashes as the first `epf weights fetch` recorded them, the safetensors one the repository's LFS hash.
    weights=WeightsRef(
        "ibm-granite/granite-timeseries-patchtst-fm-r2",
        "b125275f9204d37cbb81fe47b9cf5e08a521e829",
        {
            "config.json": "e66341b9bfa9f61eb38c435ec7d7aec353fad68857ae93182698c387213d662d",
            "model.safetensors": "32f7730abd711fecf5431ff0296e4d5939e876c9e8992168b5e539aa409cae6f",
        },
        bytes=1_538_472_531,
    ),
    runtime=("granite-tsfm>=0.3.10,<0.4", "torch", "transformers"),
    # The model card lists the corpus but no dates: a subset of the GIFT-Eval pretraining split (which t0-alpha's card dates to
    # 2021), a TSMixup of public datasets outside GIFT-Eval's test set, and synthetic KernelSynth and CauKer series. With no end
    # date the verdict goes by the release (decision 10), as Chronos-2's: a window from 2026-08-08 is clean (released before).
    corpus=(
        "a subset of the GIFT-Eval pretraining split, TSMixup of public datasets outside GIFT-Eval's test set, and synthetic "
        "KernelSynth and CauKer series (documented; no end date given)"
    ),
    corpus_end=None,
    released=date(2026, 8, 7),
)


class PatchTSTFMModel(FoundationModel):
    """PatchTST-FM-r2 (IBM, 385M, Apache-2.0 or OpenMDW-1.0), zero-shot from the target's history alone: it takes no
    covariates. The forecast is the middle of the research's 19 levels of its 99, sorted per step (the 0.5 level). On a CUDA
    device the package runs its backbone in bfloat16 autocast (the research's H100 runs did too; the manifest records it), on the
    CPU in float32. Needs the `[patchtst-fm]` extra and `epf weights fetch PatchTSTFMModel`.

    Args:
        context: Local days of the target's history a forecast reads (the package's window holds the newest
            8,192 - max(S, 128) steps; the whole context's mean fills the rest).
        batch_size: Items per backend call, every call padded to it. Compared on resume.
        device: "auto", "cpu", "cuda" or "cuda:N".
        weights: A directory of the card's two files used as given, or `"<repo>@<revision>"` of the cache; None for the card's.
        newest_target: `"d-1"` (the context ends at D-1) or `"published"` (where the target's rule says at the issue time).
        name: The model's name.
    """

    _card = PATCHTST_FM_CARD
    _default_name = "patchtst_fm"
    _takes_known = False
    _takes_past = False
    # Univariate and zero-shot: the base's covariate and fine-tune settings are no parameters of this class, so a pipeline
    # doesn't save them and the GPU host rebuilds the model from its own.
    known_ahead = _Internal()
    past = _Internal()
    fine_tune_every = _Internal()
    fine_tune = _Internal()
    seeds = _Internal()
    random_state = _Internal()

    def __init__(
        self,
        context: int = 364,
        batch_size: int = 64,
        device: str = "auto",
        weights: Optional[str] = None,
        newest_target: str = "d-1",
        name: str = "patchtst_fm",
    ):
        super().__init__(
            context=context,
            batch_size=batch_size,
            device=device,
            weights=weights,
            newest_target=newest_target,
            name=name,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        return {"context": self.context, "batch_size": self.batch_size}

    def _check_inputs(self, data, target, horizon, freq, *args, stacklevel=4, **kwargs):
        """FoundationModel's checks, then a warning when the context is longer than the package's window takes in."""
        expanded = super()._check_inputs(data, target, horizon, freq, *args, stacklevel=stacklevel + 1, **kwargs)
        steps = self.context * periods_per_day(freq)
        taken = PATCHTST_FM_WINDOW - PATCHTST_FM_FORECAST_STEPS
        if steps > taken:
            warnings.warn(
                f"{self.name}: context={self.context} days of {freq} periods is {steps} steps, and PatchTST-FM-r2 takes in at "
                f"most the newest {PATCHTST_FM_WINDOW} - max(S, {PATCHTST_FM_FORECAST_STEPS}) of them (S the steps ahead; "
                f"{taken} for {PATCHTST_FM_FORECAST_STEPS} or fewer): the older steps enter only through the context's mean, "
                "which fills its window's other steps",
                UserWarning,
                stacklevel=stacklevel,
            )
        return expanded

    def _preflight(self, seen: Optional[Dict] = None) -> List[Any]:
        """The card's checks; where the extra is missing on Python 3.14 or later, the problem says that granite-tsfm needs 3.13
        or older (the extra installs nothing there) instead of naming the extra alone."""
        problems = super()._preflight(seen)
        if sys.version_info >= (3, 14):
            here = f"{sys.version_info[0]}.{sys.version_info[1]}"
            problems = [
                problem._replace(
                    message=(
                        f"needs Python 3.13 or older: granite-tsfm, which PatchTST-FM runs on, installs only there, so "
                        f"[patchtst-fm] installs nothing on Python {here}: use a Python 3.11-3.13 environment and pip install "
                        "'epftoolbox2[patchtst-fm]' there"
                    )
                )
                if problem.kind == "extra"
                else problem
                for problem in problems
            ]
        return problems

    def _runtime_fields(self) -> Dict[str, Any]:
        """FoundationModel's, and on a CUDA device the package's bfloat16 autocast (compared on resume)."""
        fields = super()._runtime_fields()
        if fields.get("device") == "cuda":
            fields["autocast"] = "bfloat16"
        return fields

    def _load_backend(self, resolved: Any, device: Any, options: Mapping[str, Any]) -> Any:
        from ._fm.patchtst_fm import PatchTSTFMBackend

        return PatchTSTFMBackend(resolved, device)
