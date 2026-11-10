"""Chronos-2's backend (`ChronosModel`): `chronos-forecasting`'s pipeline, loaded from the card's two files, called as the
research's adapter called it (`zoo_tsfm/code/zoo.py` `Chronos`, `recent`'s runs).

Each item is a dict: `target` (T,), with covariates `past_covariates` (each known-ahead series over the context and each past
series) and `future_covariates` (each known-ahead series over the steps ahead), float32, no cross-learning (each item its own
group, so no item attends to another of its call). `predict_quantiles` at the research's 19 levels, all native levels of
Chronos-2's 21, so the median is its own 0.5 level. Chronos-2 reads at most `context_length` (8,192) steps: a longer context is
cut to its newest steps by the package, which each model's run warns about (`ChronosModel._max_context_steps`).

Needs the `[chronos]` extra; importing this module imports torch and chronos.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..._extras import MissingExtraError
from . import LEVELS_19, Backend, staged

try:
    import torch
    from chronos import BaseChronosPipeline
except ImportError:  # pragma: no cover - the core-only CI job has no chronos
    raise MissingExtraError("ChronosModel", "chronos") from None

FILES = ("config.json", "model.safetensors")


class ChronosBackend(Backend):
    """Chronos-2's pipeline on `device`, from `resolved` (`config.json` and `model.safetensors`, each by name)."""

    levels = LEVELS_19

    def __init__(self, resolved: Any, device: Any):
        with staged(resolved, FILES) as root:
            self.pipeline = BaseChronosPipeline.from_pretrained(str(root), device_map=str(device))
        self.pipeline.model.eval()

    @property
    def max_context(self) -> int:
        return int(self.pipeline.model_context_length)

    def predict(self, call: Any, options: Mapping[str, Any]) -> np.ndarray:
        steps = int(call.steps_ahead)
        width = call.target.shape[1]
        inputs = []
        for b in range(call.target.shape[0]):
            item = {"target": call.target[b].astype(np.float32)}
            if call.known or call.past:
                item["past_covariates"] = {
                    **{name: values[b, :width].astype(np.float32) for name, values in call.known.items()},
                    **{name: values[b].astype(np.float32) for name, values in call.past.items()},
                }
                item["future_covariates"] = {
                    name: values[b, width:].astype(np.float32) for name, values in call.known.items()
                }
            inputs.append(item)
        with torch.no_grad():
            quantiles, _ = self.pipeline.predict_quantiles(
                inputs, prediction_length=steps, quantile_levels=list(self.levels), batch_size=len(inputs)
            )
        return np.stack([np.asarray(q.float().cpu())[0, :steps, :] for q in quantiles])
