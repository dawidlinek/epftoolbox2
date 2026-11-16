"""Timestral's backend (`TimestralModel`): the `timestral` package at its pinned commit (not on PyPI; the model's pre-flight
checks the commit installed), its `TimestralForecaster` around the checkpoint loaded from the card's two files, called as the
research's adapter called it (`zoo_timestral/code/zoo.py` `Timestral`).

- **Inputs**: the context `(B, 1, T)` float32, NaN where a step has no value (the package masks it); each known-ahead series a
  row of `known_covariates` `(B, F, T + S)`, each past series a row of `past_covariates` `(B, P, T)`, in the model's order.
- **Options per call** (`TimestralModel._predict_options`): `symmetric_averaging` (the forecast of the context and of its
  negation averaged, the package's default) and `clip_nonnegative` (forecasts clamped at 0 for a series whose context has no
  negative value: off for prices, which can be negative; the research ran it on, where it changed no forecast at 182 days of
  context or more).
- **Levels**: its 21 native levels (0.01, 0.05 to 0.95, 0.99), sorted by the package; the forecast is the level nearest 0.5.

Needs the `[timestral]` extra and the package from its commit (`_extras.install_line("timestral")`); importing this module
imports torch and timestral.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..._extras import MissingExtraError
from . import Backend, nearest_level, staged

try:
    import torch  # noqa: F401 - the package's own torch, imported here so the extra is checked before it loads
    from timestral import TimestralForecaster
    from timestral.loading import load_model
except ImportError:  # pragma: no cover - the core-only CI job has no timestral
    raise MissingExtraError("TimestralModel", "timestral") from None

FILES = ("config.json", "model.safetensors")


class TimestralBackend(Backend):
    """`TimestralForecaster` on `device`, from `resolved` (`config.json` and `model.safetensors`, each by name)."""

    def __init__(self, resolved: Any, device: Any):
        with staged(resolved, FILES) as root:
            model = load_model(str(root), device=str(device))
        self.forecaster = TimestralForecaster(model)
        self.levels = tuple(float(level) for level in self.forecaster.quantile_levels)
        self.point_level = nearest_level(self.levels, 0.5)

    def predict(self, call: Any, options: Mapping[str, Any]) -> np.ndarray:
        context = call.target.astype(np.float32)[:, None, :]
        past = np.stack(list(call.past.values()), axis=1).astype(np.float32) if call.past else None
        known = np.stack(list(call.known.values()), axis=1).astype(np.float32) if call.known else None
        forecast = self.forecaster.forecast(
            context,
            prediction_length=int(call.steps_ahead),
            past_covariates=past,
            known_covariates=known,
            symmetric_averaging=bool(options["symmetric_averaging"]),
            clip_nonnegative=bool(options["clip_nonnegative"]),
        )
        return forecast.quantiles[:, 0].float().cpu().numpy()  # (B, V, S, Q): the target is variate 0
