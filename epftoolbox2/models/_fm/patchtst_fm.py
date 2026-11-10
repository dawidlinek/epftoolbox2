"""PatchTST-FM-r2's backend (`PatchTSTFMModel`): granite-tsfm's `PatchTSTFMForPrediction`, loaded from the card's two files,
called as the research's adapter called it (`zoo_tsfm/code/zoo_extra.py` `PatchTSTFM`).

- **Univariate**: `past_values` is the target alone, `(B, T, 1)` float32 (the model class refuses covariates when it is built).
- **What it reads**: the package forecasts `max(S, 128)` steps (`d_patch` 16 x twice its `pretrain_mask_cont` 8) inside its
  8,192-step window, so the newest `8,192 - max(S, 128)` context steps go in (8,064 for a day of hours ahead); the mean of the
  whole context fills the forecast's placeholders, the window's left padding and any gap (NaN steps are masked).
- **Levels**: of its 99 native levels (0.01 to 0.99) the research's 19 (0.05 to 0.95), sorted per step as the research sorted
  them; the forecast is the middle one, the 0.5 level after sorting.
- **Precision**: on a CUDA device the package runs its backbone under bfloat16 autocast (`get_autocast_context`), as the
  research's H100 runs did; on the CPU float32. The model records it (`autocast` in the manifest's runtime).

Needs the `[patchtst-fm]` extra (granite-tsfm, Python < 3.14); importing this module imports torch and tsfm_public.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..._extras import MissingExtraError
from . import LEVELS_19, Backend, staged

try:
    import torch
    from tsfm_public.models.patchtst_fm.modeling_patchtst_fm import PatchTSTFMForPrediction
except ImportError:  # pragma: no cover - the core-only CI job has no granite-tsfm
    raise MissingExtraError("PatchTSTFMModel", "patchtst-fm") from None

FILES = ("config.json", "model.safetensors")


def level_indices(native: Any) -> list:
    """Where each of the 19 levels is among the model's native ones; ValueError when one of them isn't there."""
    native = np.asarray(native, dtype=float)
    index = [int(np.argmin(np.abs(native - level))) for level in LEVELS_19]
    if not np.allclose(native[index], LEVELS_19, atol=1e-6):
        raise ValueError(f"PatchTST-FM's quantile levels {native.tolist()} don't hold all of {list(LEVELS_19)}")
    return index


class PatchTSTFMBackend(Backend):
    """`PatchTSTFMForPrediction` on `device`, from `resolved` (`config.json` and `model.safetensors`, each by name)."""

    levels = LEVELS_19

    def __init__(self, resolved: Any, device: Any):
        with staged(resolved, FILES) as root:
            model = PatchTSTFMForPrediction.from_pretrained(str(root), local_files_only=True)
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.index = level_indices(self.model.config.quantile_levels)

    def predict(self, call: Any, options: Mapping[str, Any]) -> np.ndarray:
        steps = int(call.steps_ahead)
        values = torch.as_tensor(call.target.astype(np.float32)[:, :, None], device=self.device)
        with torch.no_grad():
            out = self.model(past_values=values, prediction_length=steps)
        quantiles = out.quantile_outputs
        quantiles = torch.stack(list(quantiles)) if isinstance(quantiles, (list, tuple)) else quantiles
        quantiles = quantiles.float().cpu().numpy()  # (B, Q, S[, 1])
        if quantiles.ndim == 4:
            quantiles = quantiles[..., 0]
        return np.sort(np.transpose(quantiles[:, self.index, :steps], (0, 2, 1)), axis=-1)
