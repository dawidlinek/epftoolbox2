"""The scikit-learn release of the core models that fit with it (OLS, Ridge, LassoCV), in their manifests' provenance.

`uv.lock` holds two scikit-learn releases since `[patchtst-fm]` resolves apart (1.7.2 there, 1.8.0 for every other install), so
one results folder can be resumed under another release. The manifest records the release (`scikit_learn`), as provenance: never
compared, since every manifest written before has none and would then refuse every resume. A resume under another release
warns, every time the file mixes them, and the field lists each release in the order first used, as `kernels` does.
"""

import importlib.metadata
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.models import LassoCVModel, NaiveModel, OLSModel, RidgeModel
from epftoolbox2.pipelines import ModelPipeline

INSTALLED = importlib.metadata.version("scikit-learn")


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")


@pytest.fixture(scope="module")
def frame():
    index = pd.date_range("2023-01-01", "2023-04-01", freq="h", tz="Europe/Warsaw", inclusive="left")
    rng = np.random.default_rng(0)
    load = 1e4 + 500 * np.cos(np.arange(len(index)) / 24 * 2 * np.pi) + rng.normal(0, 50, len(index))
    price = 50 + 0.004 * load + rng.normal(0, 3, len(index))
    out = pd.DataFrame({"price": price, "load": load}, index=index)
    out["price_d-1"] = out["price"].shift(24)
    out["load_d+1"] = out["load"].shift(-24)
    return out


def run(model, frame, save_dir, test=("2023-03-20", "2023-03-23")):
    return ModelPipeline().add_model(model).run(frame, *test, horizon=1, save_dir=save_dir)


def manifest(save_dir, name):
    return json.loads((save_dir / f"{name}.manifest.json").read_text())


@pytest.mark.parametrize(
    "make",
    [
        lambda: OLSModel(["price_d-1", "load_d+1"], 28, name="m"),
        lambda: RidgeModel(["price_d-1", "load_d+1"], training_window=28, name="m"),
        lambda: LassoCVModel(["price_d-1", "load_d+1"], 28, name="m"),
    ],
    ids=["OLS", "Ridge", "LassoCV"],
)
def test_a_core_model_that_fits_with_scikit_learn_records_its_release(make, frame, tmp_path):
    run(make(), frame, tmp_path)
    assert manifest(tmp_path, "m")["scikit_learn"] == INSTALLED


def test_a_model_that_fits_without_it_records_none(frame, tmp_path):
    run(NaiveModel(["price_d-1"], name="n"), frame, tmp_path)
    assert "scikit_learn" not in manifest(tmp_path, "n")


def test_a_resume_under_another_release_warns_and_resumes_and_an_old_manifest_resumes_quietly(frame, tmp_path):
    """Results of another release (a manifest that says so) resume with a warning naming both, and the field lists both in
    the order first used; every later resume warns while the file mixes them. A manifest without the field (written before
    it) resumes without a word and takes this run's release."""
    model = lambda: OLSModel(["price_d-1", "load_d+1"], 28, name="m")  # noqa: E731
    run(model(), frame, tmp_path)
    results = tmp_path / "m.jsonl"
    lines = results.read_text().splitlines()
    results.write_text("\n".join(lines[: len(lines) // 2]) + "\n")  # cut short: the next run resumes it
    path = tmp_path / "m.manifest.json"
    stored = json.loads(path.read_text())
    path.write_text(json.dumps({**stored, "scikit_learn": "0.24.2"}))
    with pytest.warns(UserWarning, match=rf"scikit-learn 0.24.2 and this run fits with scikit-learn {INSTALLED}"):
        run(model(), frame, tmp_path)
    assert manifest(tmp_path, "m")["scikit_learn"] == ["0.24.2", INSTALLED]
    with pytest.warns(UserWarning, match="mixes them"):
        run(model(), frame, tmp_path)
    old = tmp_path / "old"
    run(model(), frame, old)
    stored = json.loads((old / "m.manifest.json").read_text())
    del stored["scikit_learn"]
    (old / "m.manifest.json").write_text(json.dumps(stored))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        run(model(), frame, old)
    assert manifest(old, "m")["scikit_learn"] == INSTALLED
