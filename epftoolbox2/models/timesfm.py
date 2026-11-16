"""`TimesFMModel`: Google's TimesFM-3.0 (331M), zero-shot with known-ahead and past covariates (step 6j), on `FoundationModel`
(`models/foundation.py`), its backend in `models/_fm/timesfm.py`.

**Flagged.** The weights are under the TimesFM Non-Commercial License v1.0: testing, evaluation and research not tied to
commercial gain, production or revenue only (`epf weights fetch TimesFMModel --accept TimesFM-Non-Commercial-1.0`). The corpus
is documented and ends in November 2023 (GIFT-Eval pretraining, Wikipedia pageviews to 2023-11, Google Trends to 2022, synthetic
data), so a window from December 2023 on is `clean`.

The research's TimesFM-3 at 364 days of context with the load forecast known ahead gave 19.79 EUR/MWh on the DE_LU test year,
the best clean zero-shot model with covariates there (`zoo_tsfm/REPORT.md`).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Mapping, Optional, Sequence

from ._cards import ModelCard, WeightsRef
from .foundation import FoundationModel, _Internal

__all__ = ["TimesFMModel", "TIMESFM_CARD", "TIMESFM_LICENCE"]

TIMESFM_LICENCE = "TimesFM-Non-Commercial-1.0"

TIMESFM_CARD = ModelCard(
    family="timesfm-3.0",
    extra="timesfm",
    licence=TIMESFM_LICENCE,
    licence_url="https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE",
    commercial_use=False,
    revenue_cap=None,
    attribution=None,
    # The revision the research ran (the repository's last change, 2026-09-02, before its runs of 2026-10), pinned 2026-11-10: both
    # hashes as the first `epf weights fetch` recorded them, the safetensors one the repository's LFS hash.
    weights=WeightsRef(
        "google/timesfm-3.0-pytorch",
        "43046b85ec22d584a13f8098c2ed39c889e129c2",
        {
            "config.json": "ff17bbc07b792c5a904cca265b8468579d736a4fe84981da25eb871b0a125bc6",
            "model.safetensors": "a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da",
        },
        bytes=1_322_900_097,
    ),
    runtime=("timesfm==3.0.2", "torch"),
    corpus=(
        "GIFT-Eval pretraining split (less the datasets in fev-bench), Wikipedia pageviews to November 2023, Google Trends to "
        "2022, synthetic and augmented series (the model card)"
    ),
    corpus_end=date(2023, 11, 30),
    released=date(2026, 8, 24),
)


class TimesFMModel(FoundationModel):
    """TimesFM-3.0 (Google, 331M), zero-shot: known-ahead covariates as its `past_future_covariates` (over the context and the
    steps ahead), past ones as its `past_only_covariates` (over the context); the forecast is its median, decile index 4. Its
    licence is non-commercial (see the module). It reads at most 15,360 steps (640 days of hours): a longer context is cut to
    its newest steps, which a run warns about. Needs the `[timesfm]` extra and `epf weights fetch TimesFMModel --accept
    TimesFM-Non-Commercial-1.0`.

    Args:
        context: Local days of the target's history a forecast reads.
        known_ahead: Base column names read over the context and the steps ahead.
        past: Base column names read over the context, NaN after their newest known step (the package interpolates gaps and
            holds a series' newest value after it).
        batch_size: Items per backend call, every call padded to it. Compared on resume.
        device: "auto", "cpu", "cuda" or "cuda:N".
        weights: A directory of the card's two files used as given, or `"<repo>@<revision>"` of the cache; None for the card's.
        newest_target: `"d-1"` (the context ends at D-1) or `"published"` (where the target's rule says at the issue time).
        name: The model's name.
    """

    _card = TIMESFM_CARD
    _default_name = "timesfm3"
    # The package's `_MAX_CONTEXT_LENGTH`: a longer context is cut to its newest 15,360 steps.
    _max_context_steps = 15360
    # Zero-shot only here: the base's fine-tune settings are no parameters of this class.
    fine_tune_every = _Internal()
    fine_tune = _Internal()
    seeds = _Internal()
    random_state = _Internal()

    def __init__(
        self,
        context: int = 364,
        known_ahead: Sequence[str] = (),
        past: Sequence[str] = (),
        batch_size: int = 64,
        device: str = "auto",
        weights: Optional[str] = None,
        newest_target: str = "d-1",
        name: str = "timesfm3",
    ):
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

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        return {
            "context": self.context,
            "known_ahead": list(self.known_ahead),
            "past": list(self.past),
            "batch_size": self.batch_size,
        }

    def _load_backend(self, resolved: Any, device: Any, options: Mapping[str, Any]) -> Any:
        from ._fm.timesfm import TimesFMBackend

        return TimesFMBackend(resolved, device)
