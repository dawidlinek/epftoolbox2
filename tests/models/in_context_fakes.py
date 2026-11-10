"""Stand-ins for the in-context families' heavy parts (step 6e): weights files, a numpy backend, and fakes of the real packages'
`fit`/`predict`, so that every library-side line runs without a model or, in the extras job, with the real package around a
fake core.

- `install_fake_weights(monkeypatch, root, cls)`: a small file in place of the family's checkpoint, fetched into the cache
  `root` (`EPFTOOLBOX2_WEIGHTS_DIR`) as `epf weights fetch` leaves it, under the class's card with that file's sha256 (so the
  pre-flight passes and nothing warns). Without `real_extra` the card's extra is a fake one whose module is there (`json`)
  and its runtime pins numpy alone, so the pre-flight passes without the family's package.
- `KernelBackend`: what `_load_backend` returns in the core job: the context is the fit's rows, a row's median is the
  kernel-weighted mean of the context's targets (each column scaled by the context's spread), each row computed on its own,
  so one at a time and in a block give the same bits. `leak` adds that times the mean of the call's first column (a model
  that sees its call: the co-batch check's planted leak); `shift` predicts each row from the next row of its call (a forecast
  reading the next day's inputs).
- `use_fake_backend(monkeypatch, cls, **settings)`: `cls._load_backend` returns a `KernelBackend` and every load is counted.
- `patch_package(monkeypatch, family)`: the real package's regressor `fit` keeps the context, `predict` answers the kernel
  mean, and each call's arguments are recorded (the extras job).
"""

import contextlib
import dataclasses
import hashlib
import os
from typing import Any, Dict, List

import numpy as np

from epftoolbox2._extras import EXTRAS
from epftoolbox2.models import TabICLModel, TabPFNModel, _in_context, _weights
from epftoolbox2.models._cards import WeightsRef
from epftoolbox2.models._weights import WEIGHTS_DIR_ENV
from tests.models.carded_fake import fake_download

FAKE_EXTRA = "fakeicl"
FAMILIES = {"tabpfn": TabPFNModel, "tabicl": TabICLModel}


def weights_content(cls) -> bytes:
    return (f"fake {cls._card.family} weights\n" * 16).encode()


def install_fake_weights(monkeypatch, root, cls, *, real_extra: bool = False) -> bytes:
    """A fake checkpoint for `cls` in the cache `root`, under a card with its hash (and, without `real_extra`, a fake extra
    that is installed and no runtime pin but numpy). Returns the file's content."""
    card = cls._card
    (name,) = card.weights.files
    content = weights_content(cls)
    changes: Dict[str, Any] = {
        "weights": WeightsRef(
            card.weights.repo, card.weights.revision, {name: hashlib.sha256(content).hexdigest()}, bytes=len(content)
        )
    }
    if not real_extra:
        changes.update(extra=FAKE_EXTRA, runtime=("numpy>=2",))
        monkeypatch.setitem(EXTRAS, FAKE_EXTRA, ("json",))
    fake_card = dataclasses.replace(card, **changes)
    monkeypatch.setattr(cls, "_card", fake_card)
    monkeypatch.setenv(WEIGHTS_DIR_ENV, str(root))
    monkeypatch.setattr(_weights, "snapshot_download", fake_download({name: content}, []))
    _weights.fetch(fake_card, cls.__name__)
    return content


def kernel_mean(context_x: np.ndarray, context_y: np.ndarray, row: np.ndarray, bandwidth: float) -> float:
    """The mean of `context_y` weighted by a Gaussian kernel of the scaled distance of each context row to `row`."""
    spread = context_x.std(axis=0)
    spread = np.where(spread > 0, spread, 1.0)
    distance = (((context_x - row) / spread) ** 2).sum(axis=1)
    weights = np.exp(-(distance - distance.min()) / (2 * bandwidth**2 * context_x.shape[1]))
    return float(weights @ context_y / weights.sum())


class KernelBackend:
    """A numpy in-context regressor (see the module docstring); `fits` and `calls` count what it was asked."""

    def __init__(self, path: str, device: str, bandwidth: float = 0.5, leak: float = 0.0, shift: bool = False):
        self.path = path
        self.device = device
        self.bandwidth = bandwidth
        self.leak = leak
        self.shift = shift
        self.fits: List[tuple] = []
        self.calls: List[np.ndarray] = []

    def scope(self):
        return contextlib.nullcontext()

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.fits.append((x.copy(), y.copy()))
        return x.copy(), y.copy()

    def predict(self, context, rows: np.ndarray) -> np.ndarray:
        self.calls.append(rows.copy())
        context_x, context_y = context
        source = np.roll(rows, -1, axis=0) if self.shift else rows
        out = np.array([kernel_mean(context_x, context_y, row, self.bandwidth) for row in source])
        if self.leak:
            out = out + self.leak * float(rows[:, 0].mean())
        return out


def use_fake_backend(monkeypatch, cls, **settings) -> List[KernelBackend]:
    """`cls._load_backend` returns a fresh `KernelBackend(**settings)`; returns the list of every backend loaded."""
    loaded: List[KernelBackend] = []

    def load(self, resolved, device):
        backend = KernelBackend(str(resolved.path()), device, **settings)
        loaded.append(backend)
        return backend

    monkeypatch.setattr(cls, "_load_backend", load)
    monkeypatch.setattr(_in_context, "_BACKENDS", {})
    return loaded


def patch_package(monkeypatch, family: str, bandwidth: float = 0.5) -> Dict[str, list]:
    """The real package's regressor with its model calls replaced: TabPFN's loader (`load_model_criterion_config`) and
    TabICL's `_load_model` build stand-ins and record each load (its path, `HF_HUB_OFFLINE` at the time and, for TabPFN, its
    keyword arguments); `fit` keeps the context and resolves the precision as the package does (TabPFN's
    `determine_precision`, TabICL's `_resolve_amp_fa3` after its size attributes), `predict` returns the kernel mean.
    Records each regressor's settings, each `fit`, each `predict` (its `output_type`) and each load."""
    seen: Dict[str, list] = {"made": [], "fits": [], "predicts": [], "loads": [], "fit_offline": []}
    if family == "tabpfn":
        import tabpfn.model_loading as loading
        from tabpfn import TabPFNRegressor as regressor
        from tabpfn.base import determine_precision
        from tabpfn.utils import infer_devices

        def load(model_path, **kwargs):
            seen["loads"].append({"path": str(model_path), "offline": os.environ.get("HF_HUB_OFFLINE"), **kwargs})
            return [object()], object(), [object()], object()

        monkeypatch.setattr(loading, "load_model_criterion_config", load)
    else:
        from tabicl import TabICLRegressor as regressor

    def fit(self, X, y):
        seen["made"].append(self.get_params(deep=False))
        seen["fits"].append((np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)))
        seen["fit_offline"].append(os.environ.get("HF_HUB_OFFLINE"))
        if family == "tabicl":
            self._load_model()
            self.n_samples_in_, self.n_features_in_ = np.shape(X)
            self._resolve_device()
        else:
            self.devices_ = infer_devices(self.device)
            self.use_autocast_, self.forced_inference_dtype_, _ = determine_precision(
                self.inference_precision, self.devices_
            )
        self._epf_context = (np.array(X, dtype=np.float64), np.array(y, dtype=np.float64))
        return self

    def predict(self, X, output_type="mean", **kwargs):
        seen["predicts"].append(output_type)
        context_x, context_y = self._epf_context
        return np.array([kernel_mean(context_x, context_y, row, bandwidth) for row in np.asarray(X, dtype=np.float64)])

    monkeypatch.setattr(regressor, "fit", fit)
    monkeypatch.setattr(regressor, "predict", predict)
    if family == "tabicl":

        class Network:
            def to(self, device):
                return self

        def load_model(self):
            seen["loads"].append({"path": str(self.model_path), "offline": os.environ.get("HF_HUB_OFFLINE")})
            self.model_, self.model_config_, self.model_path_ = Network(), {}, self.model_path

        monkeypatch.setattr(regressor, "_load_model", load_model)
    monkeypatch.setattr(_in_context, "_BACKENDS", {})
    return seen
