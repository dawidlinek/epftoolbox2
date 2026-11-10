"""`TabPFNModel`: Prior Labs' TabPFN v2 regressor as an in-context model (step 6e; `models/_in_context.py` holds what it shares
with `TabICLModel`).

Only the v2 regression weights (`Prior-Labs/TabPFN-v2-reg`, `tabpfn-v2-regressor.ckpt`, Prior Labs License 1.1: Apache-2.0
with an attribution clause) are ever used: the regressor is built with `ModelVersion.V2` and the checkpoint the run resolved,
and a weights file whose name the package reads as another version is refused, since the package's newer default weights
(2.5 and later) are under non-commercial licences. The research's settings (`zoo_tabfm/scripts/tabfm_run.py`,
`mlgpu/code/tabpfn_run.py`): 728 days of context, a weekly refit, the target as the Invariant, raw predictors, 4 estimators
(8 gave 19.56 against 19.58 EUR/MWh MAE on DE, not significant), seed 0 for every fit, one preprocessing job, the median.
"""

from datetime import date
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ._cards import ModelCard, WeightsRef
from ._in_context import INVARIANT_TARGET, InContextModel, package, torch_fit_scope

REPO = "Prior-Labs/TabPFN-v2-reg"
# The repository's main at 2026-11-10; the checkpoint itself last changed on 2025-01-05 (commit 230d61748bae).
REVISION = "02a44705ff7790b50cded404bc59adb94da456a1"
CHECKPOINT = "tabpfn-v2-regressor.ckpt"
SHA256 = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"

CARD = ModelCard(
    family="tabpfn-v2",
    extra="tabpfn",
    licence="Prior-Labs-1.1",
    licence_url="https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/main/LICENSE.txt",
    commercial_use=True,
    revenue_cap=None,
    attribution=(
        "Built with PriorLabs-TabPFN (Prior Labs License 1.1, section 10: shown wherever a product or its documentation built "
        "on the weights or their outputs is made available; internal benchmarking needs none)"
    ),
    # sha256 recorded by the first `epf weights fetch` (S02, 2026-11-10), the repository's LFS hash of the file too.
    weights=WeightsRef(REPO, REVISION, {CHECKPOINT: SHA256}, bytes=44_390_977),
    # tabpfn 9.1 is the first with `ModelSpecs` (the network handed to each fit); scikit-learn, scipy and numpy preprocess
    # every context (Yeo-Johnson moved by 2.9e-15 between two of their releases), so their versions are compared too.
    runtime=("tabpfn>=9.1,<10", "torch", "scikit-learn", "scipy", "numpy"),
    corpus="synthetic priors",
    corpus_end=None,
    released=date(2025, 1, 5),
    stated_verdict="clean (synthetic)",
)


class _Backend:
    """TabPFN v2 on one device. The network is built once, here, from the resolved checkpoint (tabpfn's own loader with
    `download_if_not_exists=False`, inside the caller's `offline()`), and handed to every fit's regressor as `ModelSpecs`, so no
    fit reaches tabpfn's loader or its download paths. Each fit is a regressor of the research's settings, its context shown by
    `fit`, the median asked of `predict`. On a CPU the precision is pinned to float32 (tabpfn's "auto" autocasts to bfloat16
    on CPUs with AVX512-BF16 or AMX); on a GPU it is tabpfn's "auto", float16 autocast. A fit whose regressor computes in
    another precision stops the run."""

    def __init__(self, model: "TabPFNModel", path: str, device: str):
        package("tabpfn", model)
        import torch
        from tabpfn import TabPFNRegressor
        from tabpfn.base import ModelSpecs
        from tabpfn.constants import ModelVersion
        from tabpfn.model_loading import load_model_criterion_config, resolve_model_version

        if resolve_model_version(path) != ModelVersion.V2:
            raise RuntimeError(
                f"TabPFNModel: tabpfn reads {path} as {resolve_model_version(path).value} weights, not v2: only the v2 "
                f"regressor ({REPO}, {CHECKPOINT}) is used, whose licence allows commercial use"
            )
        self.torch = torch
        self.on_cpu = torch.device(device).type == "cpu"
        forced = torch.float32 if self.on_cpu else None
        models, criterion, configs, inference_config = load_model_criterion_config(
            model_path=path,
            check_bar_distribution_criterion=True,
            estimator_type="regressor",
            version=ModelVersion.V2.value,
            download_if_not_exists=False,
            devices=[torch.device(device)],
            force_inference_dtype=forced,
        )
        self.specs = ModelSpecs(
            model=models[0], architecture_config=configs[0], inference_config=inference_config, norm_criterion=criterion
        )
        self.regressor = TabPFNRegressor
        self.version = ModelVersion.V2
        self.precision = forced if self.on_cpu else "auto"
        self.path = path
        self.device = device
        self.n_estimators = model.n_estimators
        self.random_state = model.random_state

    def scope(self):
        return torch_fit_scope(self.device)

    def fit(self, x: np.ndarray, y: np.ndarray) -> Any:
        regressor = self.regressor.create_default_for_version(
            self.version,
            model_path=self.specs,
            device=self.device,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_preprocessing_jobs=1,
            inference_precision=self.precision,
        )
        regressor.fit(x, y)
        self._check_precision(regressor)
        return regressor

    def _check_precision(self, regressor: Any) -> None:
        """The precision the manifest records (`TabPFNModel._PRECISION`): float32 and no autocast on a CPU, autocast on a GPU."""
        autocast = getattr(regressor, "use_autocast_", None)
        forced = getattr(regressor, "forced_inference_dtype_", None)
        expected = (False, self.torch.float32) if self.on_cpu else (True, None)
        if (autocast, forced) != expected:
            raise RuntimeError(
                f"TabPFNModel: tabpfn computes with autocast={autocast}, dtype={forced} on {self.device}, not the "
                f"{TabPFNModel._PRECISION['cpu' if self.on_cpu else 'cuda']} the manifest records"
            )

    def predict(self, regressor: Any, rows: np.ndarray) -> np.ndarray:
        return regressor.predict(rows, output_type="median")


class TabPFNModel(InContextModel):
    """TabPFN v2 (`Prior-Labs/TabPFN-v2-reg`) on the refit calendar: one model per period, horizon and refit day, whose
    context is the unit's rows and whose forecast is the median of its predictive distribution (`[tabpfn]` extra; fetch the
    weights first with `epf weights fetch TabPFNModel`).

    Args:
        predictors: Full column names (`{horizon}` allowed), saved verbatim, as raw inputs: `presets.lear_predictors(...)` gives
            the research's rows. TabPFN v2 was pretrained on up to 500 columns and 10,000 rows; on a CPU the package refuses
            contexts of more than 1,000 rows unless `TABPFN_ALLOW_CPU_LARGE_DATASET=1` is set.
        training_window: Days of context (728 by default).
        refit: How often the context is refreshed, on the fixed calendar (`"7D"`, Mondays, by default); every issue day of a
            block is predicted in one call (`block_predict`).
        n_estimators: Ensemble members of one call (preprocessed views of the context), 4 by default.
        random_state: The seed of every fit's preprocessing.
        device: "auto" (every GPU a matmul runs on, worker k on GPU k modulo their number, else the CPU), "cpu", "cuda" or
            "cuda:N", resolved once for a run. On a CPU the fits compute in float32, on a GPU with tabpfn's float16 autocast,
            and the two differ by more than rounding: up to 4.3 EUR/MWh (mean 0.28) over 1,200 forecasts of the research's
            rows. The device type and the precision are recorded and compared on resume.
        weights: None (the cache `epf weights fetch` filled), a path to the checkpoint, or `"Prior-Labs/TabPFN-v2-reg@<rev>"`.
        block_predict: Predict a block's issue days in one call (True), or one row at a time.
        name: Display name.
        scaler: The target as the Invariant and the inputs raw, by default.
        newest_target, binary_mask, normalise_by, bounds, zero_when: As for every model (see `BaseModel`).
    """

    _card = CARD
    _PRECISION = {"cpu": "float32", "cuda": "float16 autocast"}

    def __init__(
        self,
        predictors: List[str],
        training_window: Union[int, str] = 728,
        refit: Optional[str] = "7D",
        n_estimators: int = 4,
        random_state: int = 0,
        device: str = "auto",
        weights: Optional[str] = None,
        block_predict: bool = True,
        name: str = "TabPFN",
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


__all__ = ["TabPFNModel", "CARD"]
