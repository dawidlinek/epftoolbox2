"""A model family with a card, for the tests of step 6a (`models/_cards.py`, `models/_weights.py`) and the families after it.

`CardedModel` is OLS whose fits add the number its weights file holds (0.0 in the reference file, so its forecasts are OLS's to
the bit), read once per worker under `_weights.offline()` from the file its card or `weights=` names, as an adapter loads its
backend: the weights reach the fit. Its card is non-commercial with a revenue cap and an attribution, and its corpus ends after
the tests' test start (`possibly contaminated`). Its extra is `fakefam`, which a test registers in `EXTRAS` (`fake_extra`) with
a module that is there (`json`) or one that isn't. `fake_download` stands in for `huggingface_hub.snapshot_download`.

Import it as `tests.models.carded_fake`, the name `COMPONENT_REGISTRY` and `epf weights` use for it.
"""

import dataclasses
import hashlib
from datetime import date
from pathlib import Path
from typing import Dict, List, Mapping

from epftoolbox2._extras import EXTRAS
from epftoolbox2.models import OLSModel
from epftoolbox2.models._cards import RUN_DATA, ModelCard, WeightsRef, resolve_model_weights
from epftoolbox2.models._weights import offline

FAKE_EXTRA = "fakefam"
WEIGHTS_FILE = "fake-fm.bin"
REPO = "fake-org/fake-fm"
REVISION = "v1"
LICENCE = "Fake-NC-1.0"
CLASS_PATH = "tests.models.carded_fake:CardedModel"


def weights_bytes(offset: float = 0.0, note: str = "") -> bytes:
    """The content of a weights file whose fits add `offset` (in the standardised target); `note` changes the bytes alone."""
    return (f"fake weights{note}; offset {offset!r}\n" * 64).encode()


REFERENCE = weights_bytes(0.0)
REFERENCE_SHA = hashlib.sha256(REFERENCE).hexdigest()

CARD = ModelCard(
    family="fake-fm",
    extra=FAKE_EXTRA,
    licence=LICENCE,
    licence_url="https://example.invalid/fake-nc-1.0",
    commercial_use=False,
    revenue_cap="no rights above USD 20M monthly consolidated revenue",
    attribution="2026 - Fake Labs",
    weights=WeightsRef(REPO, REVISION, {WEIGHTS_FILE: REFERENCE_SHA}, bytes=len(REFERENCE)),
    runtime=("numpy>=2", "pandas"),
    corpus="undocumented web text",
    corpus_end=date(2023, 6, 30),
    released=date(2023, 1, 15),
)


class CardedModel(OLSModel):
    """OLS plus its weights file's offset; `device` and `threads_per_fit` are its runtime settings (`_runtime_fields`)."""

    _card = CARD
    _engine_ok = False
    _WRITTEN_WHEN_SET = {**OLSModel._WRITTEN_WHEN_SET, "weights": None, "device": "cpu"}
    threads_per_fit = 1  # a class attribute a test can change: compared on resume

    def __init__(self, predictors, training_window=365, *, weights=None, device="cpu", **kwargs):
        super().__init__(predictors, training_window, **kwargs)
        self.weights = weights
        self.device = device
        self._offset = None

    @property
    def _model_kwargs(self):
        out = dict(super()._model_kwargs)
        if self.weights is not None:
            out["weights"] = self.weights
        if self.device != "cpu":
            out["device"] = self.device
        return out

    def _runtime_fields(self):
        return {"device": self.device, "threads_per_fit": type(self).threads_per_fit}

    def _loaded_offset(self) -> float:
        if self._offset is None:
            with offline():
                path = resolve_model_weights(self).path()
                self._offset = float(path.read_text().split("offset ")[1].split()[0])
        return self._offset

    def _fit_predict(self, train_x, train_y, test_x):
        prediction, coefficients = super()._fit_predict(train_x, train_y, test_x)
        return prediction + self._loaded_offset(), coefficients

    def _fit_model(self, train_x, train_y):
        return super()._fit_model(train_x, train_y)

    def _predict_fitted(self, fitted, test_x):
        prediction, coefficients = super()._predict_fitted(fitted, test_x)
        return prediction + self._loaded_offset(), coefficients


class OtherCardedModel(CardedModel):
    """A second family, its weights in another repository with no hash on its card (the first fetch records it)."""

    _card = dataclasses.replace(CARD, weights=WeightsRef("fake-org/other-fm", "v1", {WEIGHTS_FILE: None}))


def run_data_card(family: str, extra=None) -> ModelCard:
    """The card of a family trained on the run's data only (always `clean`), permissive, with no weights and no runtime pin:
    what a test needs to make a model carded (the checks `verify` turns on for a card, a provenance row) and nothing more."""
    return ModelCard(
        family=family,
        extra=extra,
        licence="MIT",
        licence_url="https://opensource.org/license/mit",
        commercial_use=True,
        revenue_cap=None,
        attribution=None,
        weights=None,
        runtime=(),
        corpus=RUN_DATA,
        corpus_end=None,
        released=None,
    )


def fake_extra(monkeypatch, present: bool = True) -> None:
    """Register `fakefam` in `EXTRAS` for one test: installed (the module `json`) or not (a module that doesn't exist)."""
    monkeypatch.setitem(EXTRAS, FAKE_EXTRA, ("json",) if present else ("epftoolbox2_no_such_backend",))


def fake_download(files: Mapping[str, bytes], calls: List[Dict]) -> callable:
    """A `snapshot_download` that writes `files` into `local_dir` (the ones `allow_patterns` names) and records each call."""

    def download(repo_id, revision, allow_patterns=None, local_dir=None, **kwargs):
        calls.append(
            {"repo_id": repo_id, "revision": revision, "allow_patterns": allow_patterns, "local_dir": local_dir}
        )
        for name, content in files.items():
            if allow_patterns is None or name in allow_patterns:
                path = Path(local_dir) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        return local_dir

    return download
