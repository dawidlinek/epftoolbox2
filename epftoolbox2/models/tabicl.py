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
# TabICL's inference managers size their batches on a GPU from its free memory and halve them on an out-of-memory error, which
# another process on the card changes. A smallest batch of tabicl's own cap (50,000) makes every batch that size: one call
# for the research's contexts (8 views of 735 rows), and an out-of-memory error stops the run instead of halving.
MIN_BATCH_SIZE = 50_000
# Where a GPU run keeps the managers' outputs: on the GPU, not moved by free memory (tabicl's "auto").
GPU_OFFLOAD = "gpu"
SHA256 = "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a"

CARD = ModelCard(
    family="tabicl-v2",
    extra="tabicl",
    licence="BSD-3-Clause",
    licence_url="https://github.com/soda-inria/tabicl/blob/main/LICENSE",
    commercial_use=True,
    revenue_cap=None,
    attribution=None,
    # sha256 recorded by the first `epf weights fetch` (S02, 2026-11-10), the repository's LFS hash of the file too.
    weights=WeightsRef(REPO, REVISION, {CHECKPOINT: SHA256}, bytes=114_324_594),
    # scikit-learn, scipy and numpy preprocess every context (its power transform among them), so their versions are compared.
    runtime=("tabicl>=2.2,<3", "torch", "scikit-learn", "scipy", "numpy"),
    corpus="synthetic priors",
    corpus_end=None,
    released=date(2026, 2, 12),
    stated_verdict="clean (synthetic)",
)


class _Backend:
    """TabICLv2 on one device: the checkpoint's model read once (the package's own loader), handed to a regressor per fit. On
    a CPU half precision is off (`use_amp=False`, as TabICL's "auto" has it there), on a GPU it is TabICL's "auto", float16
    autocast for contexts of 1,024 rows or 60 columns; a fit whose regressor resolves otherwise stops the run."""

    def __init__(self, model: "TabICLModel", path: str, device: str):
        package("tabicl", model)
        import torch
        from tabicl import TabICLRegressor

        self.on_cpu = torch.device(device).type == "cpu"
        self.use_amp = False if self.on_cpu else "auto"
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
            use_amp=self.use_amp,
            offload_mode="auto" if self.on_cpu else GPU_OFFLOAD,
            inference_config={
                name: {"min_batch_size": MIN_BATCH_SIZE} for name in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")
            },
        )
        loaded = self.loaded

        def load() -> None:  # the model this process read, never the file again
            regressor.model_, regressor.model_config_, regressor.model_path_ = loaded

        regressor._load_model = load
        regressor.fit(x, y)
        self._check_precision(regressor, x.shape)
        return regressor

    def _check_precision(self, regressor: Any, shape: tuple) -> None:
        """The precision the manifest records (`TabICLModel._PRECISION`): no autocast on a CPU; on a GPU autocast exactly for
        a context of 1,024 rows or more, or 60 columns or more."""
        resolve = getattr(regressor, "_resolve_amp_fa3", None)
        if resolve is None:
            raise RuntimeError("TabICLModel: this tabicl can't say whether it autocasts (no _resolve_amp_fa3)")
        expected = False if self.on_cpu else not (shape[0] < 1024 and shape[1] < 60)
        if bool(resolve()[0]) != expected:
            raise RuntimeError(
                f"TabICLModel: tabicl autocasts={not expected} on {self.device} for a context of {shape[0]} rows and "
                f"{shape[1]} columns, not what the manifest records ({TabICLModel._PRECISION['cpu' if self.on_cpu else 'cuda']})"
            )

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
        device: "auto" (CUDA if GPU 0's probe passes, then every GPU of GPU 0's model, worker k on GPU k modulo their
            number, GPUs of another model left out with a warning; else the CPU), "cpu", "cuda" or "cuda:N", resolved once
            for a run. On a CPU the fits compute in float32; on a GPU TabICL autocasts to float16 for contexts of 1,024 rows
            or 60 columns, and the two differ by up to 0.54 EUR/MWh (mean 0.044) over 1,200 forecasts of the research's
            rows. TabICL's batch sizes (`min_batch_size`) and on a GPU its output placement (`offload`) are pinned, so
            neither follows a GPU's free memory. The device type, the precision and those settings are recorded and
            compared on resume.
        weights: None (the cache `epf weights fetch` filled), a path to the checkpoint, or `"jingang/TabICL@<rev>"`.
        block_predict: Predict a block's issue days in one call (True), or one row at a time.
        name: Display name.
        scaler: The target as the Invariant and the inputs raw, by default.
        newest_target, binary_mask, normalise_by, bounds, zero_when: As for every model (see `BaseModel`).
    """

    _card = CARD
    _PRECISION = {"cpu": "float32", "cuda": "float16 autocast from 1,024 rows or 60 columns"}

    def _memory_settings(self, kind: str) -> Dict[str, Any]:
        """TabICL's batch sizing and output placement, pinned where tabicl would follow a GPU's free memory (on a CPU it
        neither batches nor offloads)."""
        return {"min_batch_size": MIN_BATCH_SIZE, **({"offload": GPU_OFFLOAD} if kind == "cuda" else {})}

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
