"""`TabICLModel`: TabICLv2's regressor (`jingang/TabICL`, `tabicl-regressor-v2-20260212.ckpt`, BSD-3-Clause) as an
in-context model (step 6e; `models/_in_context.py` holds what it shares with `TabPFNModel`).

The checkpoint is read once per worker process and handed to each fit's regressor, which would otherwise read it again on
every fit (the research's `_Cached` regressor, `zoo_tabfm/scripts/tabfm_run.py`); `allow_auto_download=False`, so the package
never fetches one. The research's settings: 728 days of context, a weekly refit, the target as the Invariant, raw predictors,
8 estimators in one batch (`batch_size=n_estimators`), seed 0 for every fit, the median. TabICLv2 was pretrained on 2 to 100
columns; the research's 175 and 247 worked, and whether it degrades there is untested. On a GPU the package autocasts to half
precision for contexts of 1,024 rows or more, or 60 columns or more (`use_amp="auto"`), on a CPU never: so GPU and CPU
forecasts differ by more than rounding, and the device type is compared on resume.
"""

from datetime import date
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ._cards import ModelCard, WeightsRef
from ._in_context import INVARIANT_TARGET, InContextModel, package, torch_fit_scope

REPO = "jingang/TabICL"
REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"  # main at 2026-11-10; the checkpoint was uploaded on 2026-02-12
CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"

CARD = ModelCard(
    family="tabicl-v2",
    extra="tabicl",
    licence="BSD-3-Clause",
    licence_url="https://github.com/soda-inria/tabicl/blob/main/LICENSE",
    commercial_use=True,
    revenue_cap=None,
    attribution=None,
    weights=WeightsRef(REPO, REVISION, {CHECKPOINT: None}, bytes=114_324_594),
    runtime=("tabicl>=2.2,<3", "torch"),
    corpus="synthetic priors",
    corpus_end=None,
    released=date(2026, 2, 12),
    stated_verdict="clean (synthetic)",
)


class _Backend:
    """TabICLv2 on one device: the checkpoint's model read once (the package's own loader), handed to a regressor per fit."""

    def __init__(self, model: "TabICLModel", path: str, device: str):
        package("tabicl", model)
        from tabicl import TabICLRegressor

        self.regressor = TabICLRegressor
        self.path = path
        self.device = device
        self.n_estimators = model.n_estimators
        self.random_state = model.random_state
        loader = TabICLRegressor(model_path=path, allow_auto_download=False, device=device)
        loader._load_model()
        self.loaded = (loader.model_, loader.model_config_, loader.model_path_)

    def scope(self):
        return torch_fit_scope(self.device)

    def fit(self, x: np.ndarray, y: np.ndarray) -> Any:
        regressor = self.regressor(
            n_estimators=self.n_estimators,
            batch_size=self.n_estimators,
            model_path=self.path,
            allow_auto_download=False,
            device=self.device,
            random_state=self.random_state,
        )
        loaded = self.loaded

        def load() -> None:  # the model this process read, never the file again
            regressor.model_, regressor.model_config_, regressor.model_path_ = loaded

        regressor._load_model = load
        return regressor.fit(x, y)

    def predict(self, regressor: Any, rows: np.ndarray) -> np.ndarray:
        return regressor.predict(rows, output_type="median")


class TabICLModel(InContextModel):
    """TabICLv2 (`jingang/TabICL`) on the refit calendar: one model per period, horizon and refit day, whose context is the
    unit's rows and whose forecast is the median of its predictive distribution (`[tabicl]` extra; fetch the weights first
    with `epf weights fetch TabICLModel`).

    Args:
        predictors: Full column names (`{horizon}` allowed), saved verbatim, as raw inputs. TabICLv2 was pretrained on 2 to
            100 columns (175 and 247 worked in the research; degradation untested).
        training_window: Days of context (728 by default).
        refit: How often the context is refreshed, on the fixed calendar (`"7D"`, Mondays, by default); every issue day of a
            block is predicted in one call (`block_predict`).
        n_estimators: Ensemble members of one call (preprocessed, shuffled views of the context), 8 by default, run in one
            batch.
        random_state: The seed of every fit's views.
        device: "auto" (CUDA if a matmul runs on it, else the CPU), "cpu", "cuda" or "cuda:N". The device type is recorded
            and compared on resume; on a GPU the package computes in half precision for wide contexts.
        weights: None (the cache `epf weights fetch` filled), a path to the checkpoint, or `"jingang/TabICL@<rev>"`.
        block_predict: Predict a block's issue days in one call (True), or one row at a time.
        name: Display name.
        scaler: The target as the Invariant and the inputs raw, by default.
        newest_target, binary_mask, normalise_by, bounds, zero_when: As for every model (see `BaseModel`).
    """

    _card = CARD

    def __init__(
        self,
        predictors: List[str],
        training_window: Union[int, str] = 728,
        refit: Optional[str] = "7D",
        n_estimators: int = 8,
        random_state: int = 0,
        device: str = "auto",
        weights: Optional[str] = None,
        block_predict: bool = True,
        name: str = "TabICL",
        *,
        scaler: Any = INVARIANT_TARGET,
        newest_target: str = "d-1",
        binary_mask: Optional[str] = None,
        normalise_by: Optional[str] = None,
        bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
        zero_when: Optional[str] = None,
    ):
        super().__init__(
            predictors,
            training_window,
            name,
            scaler=scaler,
            newest_target=newest_target,
            refit=None,
            binary_mask=binary_mask,
            normalise_by=normalise_by,
            bounds=bounds,
            zero_when=zero_when,
        )
        self.n_estimators = n_estimators
        self._setup(refit, random_state, device, weights, block_predict)

    def _family_kwargs(self) -> Dict[str, Any]:
        return {"n_estimators": self.n_estimators}

    def _load_backend(self, resolved, device: str) -> _Backend:
        return _Backend(self, str(resolved.path(CHECKPOINT if CHECKPOINT in resolved.files else None)), device)


__all__ = ["TabICLModel", "CARD"]
