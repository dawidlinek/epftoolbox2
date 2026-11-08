"""Step 6a: model cards, weights and the pre-flight (`models/_cards.py`, `models/_weights.py`).

A carded family (`tests/models/carded_fake.py`: OLS plus the offset its weights file holds) is checked before any model of a run
fits, every problem named in one error; its weights come from `epf weights fetch` (here a stand-in for `snapshot_download`),
are hashed once per process and compared on resume with its runtime; its card is in the manifest's provenance, in
`report.provenance()` and in the exporters, with the contamination verdict of each window. No test reaches a network
(`no_network`), and core models' manifests and outputs keep their keys.
"""

import dataclasses
import hashlib
import json
import os
import pickle
import subprocess
import sys
import types
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from openpyxl import load_workbook
from rich.console import Console

from epftoolbox2 import cli
from epftoolbox2._extras import EXTRAS, MissingExtraError
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.exporters import CsvExporter, ExcelExporter, TerminalExporter
from epftoolbox2.models import OLSModel
from epftoolbox2.models import _cards, _weights
from epftoolbox2.models._cards import (
    CLEAN,
    CLEAN_RELEASED_BEFORE,
    POSSIBLY_CONTAMINATED,
    PROVENANCE_COLUMNS,
    RUN_DATA,
    ModelCard,
    WeightsRef,
    contamination_verdict,
    parse_pin,
    preflight,
    version_matches,
)
from epftoolbox2.models._weights import (
    SIDECAR,
    WEIGHTS_DIR_ENV,
    LicenceNotAcceptedError,
    MissingWeightsError,
    WeightsMismatchError,
    offline,
    resolve,
    sha256_of,
)
from epftoolbox2.models.base import BaseModel
from epftoolbox2.pipelines import ModelPipeline, Workflow
from epftoolbox2.pipelines.model_pipeline import COMPONENT_REGISTRY
from epftoolbox2.results.manifest import ManifestMismatchError
from tests.models.carded_fake import (
    CARD,
    CLASS_PATH,
    FAKE_EXTRA,
    LICENCE,
    REFERENCE,
    REFERENCE_SHA,
    REPO,
    REVISION,
    WEIGHTS_FILE,
    CardedModel,
    fake_download,
    fake_extra,
    weights_bytes,
)
from tests.synthetic import PREDICTORS, make_market_data

TEST = ("2023-03-06", "2023-03-08")
WINDOW = 30


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


@pytest.fixture(scope="module")
def data():
    return make_market_data(start="2023-01-01", days=75, noise=2.0, seed=3)


@pytest.fixture
def weights_root(tmp_path, monkeypatch):
    """An empty weights cache of the test's own (`EPFTOOLBOX2_WEIGHTS_DIR`), and the fake extra installed."""
    root = tmp_path / "weights"
    monkeypatch.setenv(WEIGHTS_DIR_ENV, str(root))
    fake_extra(monkeypatch, present=True)
    return root


def fetch_into_cache(monkeypatch, content: bytes = REFERENCE, **kwargs):
    """`_weights.fetch` of the card's weights through the stand-in download; returns (sidecar, download calls)."""
    calls = []
    monkeypatch.setattr(_weights, "snapshot_download", fake_download({WEIGHTS_FILE: content}, calls))
    return _weights.fetch(CARD, "CardedModel", accept=LICENCE, **kwargs), calls


@pytest.fixture
def fetched(weights_root, monkeypatch):
    """The reference weights in the cache, as `epf weights fetch` leaves them."""
    fetch_into_cache(monkeypatch)
    return weights_root


def carded(name="carded", **params):
    return CardedModel(PREDICTORS, WINDOW, name=name, **params)


def run(data, *models, save_dir=None, **kwargs):
    pipeline = ModelPipeline()
    for model in models:
        pipeline.add_model(model)
    return pipeline.run(
        data,
        test_start=TEST[0],
        test_end=TEST[1],
        horizon=1,
        save_dir=None if save_dir is None else str(save_dir),
        **kwargs,
    )


def forecasts(report, name):
    return sorted((r["run_date"], r["hour"], r["horizon"], r["prediction"]) for r in report.refs[name].iter_records())


def manifest(save_dir, name):
    return json.loads((Path(save_dir) / f"{name}.manifest.json").read_text())


# --- cards and verdicts -------------------------------------------------------------------------------------------------


def test_a_card_holds_its_fields_and_records_them_as_json():
    record = CARD.record()
    assert [f.name for f in dataclasses.fields(ModelCard)] == [
        "family",
        "extra",
        "licence",
        "licence_url",
        "commercial_use",
        "revenue_cap",
        "attribution",
        "weights",
        "runtime",
        "corpus",
        "corpus_end",
        "released",
        "stated_verdict",
    ]
    assert json.loads(json.dumps(record)) == record
    assert record["weights"] == {
        "repo": REPO,
        "revision": REVISION,
        "files": {WEIGHTS_FILE: REFERENCE_SHA},
        "bytes": len(REFERENCE),
    }
    assert record["corpus_end"] == "2023-06-30" and record["runtime"] == ["numpy>=2", "pandas"]
    assert dataclasses.replace(CARD, corpus_end="2024-01-31").corpus_end == date(2024, 1, 31)
    with pytest.raises(dataclasses.FrozenInstanceError):
        CARD.licence = "MIT"
    for bad in ({"repo": "../etc"}, {"revision": "a/b"}, {"files": {"../x": None}}, {"files": {"x": "not-a-hash"}}):
        with pytest.raises(ValueError):
            WeightsRef(**{"repo": REPO, "revision": REVISION, "files": {}, **bad})
    with pytest.raises(ValueError, match="runtime pin"):
        dataclasses.replace(CARD, runtime=("torch>=two",))


def test_the_contamination_verdict_follows_the_window_s_first_issue_day():
    # 'possibly contaminated' as the corpus ends on or after the first issue day, `clean` before it.
    assert contamination_verdict(CARD, "2023-07-01") == CLEAN
    assert contamination_verdict(CARD, "2023-06-30") == POSSIBLY_CONTAMINATED
    assert contamination_verdict(CARD, date(2023, 3, 6)) == POSSIBLY_CONTAMINATED
    undocumented = dataclasses.replace(CARD, corpus_end=None)
    assert contamination_verdict(undocumented, "2023-01-16") == CLEAN_RELEASED_BEFORE
    assert contamination_verdict(undocumented, "2023-01-15") == POSSIBLY_CONTAMINATED
    assert (
        contamination_verdict(dataclasses.replace(undocumented, released=None), "2030-01-01") == POSSIBLY_CONTAMINATED
    )
    synthetic = dataclasses.replace(CARD, stated_verdict="clean (synthetic): synthetic priors only")
    assert contamination_verdict(synthetic, "2000-01-01") == "clean (synthetic): synthetic priors only"
    assert contamination_verdict(dataclasses.replace(CARD, corpus=RUN_DATA, weights=None), "2000-01-01") == CLEAN
    assert contamination_verdict(CARD.record(), "2023-07-01") == CLEAN  # from a manifest's record too
    assert contamination_verdict(CARD, None) is None
    # A corpus that ends inside a window taints it, whatever its last day: a mutant comparing with the end would call it clean.
    inside = dataclasses.replace(CARD, corpus_end=date(2023, 3, 7))
    assert contamination_verdict(inside, TEST[0]) == POSSIBLY_CONTAMINATED


@pytest.mark.parametrize(
    "version, pin, expected",
    [
        ("9.1.0", "tabpfn>=6,<10", True),
        ("10.0", "tabpfn>=6,<10", False),
        ("0.5.0", "tfc-t0==0.5.0", True),
        ("0.6.0", "tfc-t0==0.5.0", False),
        ("2.9.0+cpu", "torch==2.9.*", True),
        ("2.10.1", "torch~=2.9", True),
        ("3.0", "torch~=2.9", False),
        ("2.9.0rc1", "torch>=2.9", False),
        ("2.9.0.post1", "torch>2.9", True),
        ("1.0.dev1", "x<1.0a1", True),
        ("2.3.2", "chronos-forecasting>=2.3.2,<3", True),
        ("1.6.3", "autogluon.tabular!=1.6.3", False),
    ],
)
def test_runtime_pins_follow_pep_440(version, pin, expected):
    assert version_matches(version, parse_pin(pin)[1]) is expected


def test_a_commit_pin_and_bad_pins():
    assert parse_pin("timestral@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff") == (
        "timestral",
        (),
        "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff",
    )
    assert parse_pin("torch") == ("torch", (), None)
    for bad in ("torch>=", "torch@xyz", "torch>=2.*", "torch; python_version<'3.14'"):
        with pytest.raises(ValueError):
            parse_pin(bad)


# --- the pre-flight -------------------------------------------------------------------------------------------------------


class PinnedModel(CardedModel):
    """A card whose runtime pin no installed numpy meets."""

    _card = dataclasses.replace(CARD, runtime=("numpy<1",))


def record_starts(monkeypatch):
    """The names of the models whose run started (`_prepare_run`) and that were planned (`_plan_predictors`)."""
    started = []
    real_prepare, real_plan = BaseModel._prepare_run, BaseModel._plan_predictors

    def prepare(self, *args, **kwargs):
        started.append(("run", self.name))
        return real_prepare(self, *args, **kwargs)

    def plan(self, *args, **kwargs):
        started.append(("plan", self.name))
        return real_plan(self, *args, **kwargs)

    monkeypatch.setattr(BaseModel, "_prepare_run", prepare)
    monkeypatch.setattr(BaseModel, "_plan_predictors", plan)
    return started


def test_the_preflight_names_every_missing_extra_and_weight_before_any_model_runs(data, weights_root, monkeypatch):
    fake_extra(monkeypatch, present=False)
    started = record_starts(monkeypatch)
    # A core model first: a pre-flight after the first model (the mutant) would start its run.
    models = [
        OLSModel(PREDICTORS, WINDOW, name="ols"),
        carded("a"),
        carded("b"),
        PinnedModel(PREDICTORS, WINDOW, name="c"),
    ]
    with pytest.raises(MissingWeightsError) as raised:
        run(data, *models)
    assert started == []
    message = str(raised.value)
    assert "nothing was fitted" in message
    assert "CardedModel 'a', CardedModel 'b', PinnedModel 'c': needs the [fakefam] extra" in message
    assert "pip install 'epftoolbox2[fakefam]'" in message
    assert f"no weights for {REPO}@{REVISION} in {weights_root / REPO / REVISION}" in message
    assert f"epf weights fetch {CLASS_PATH}" in message
    assert f"epf weights fetch {_cards.class_path(PinnedModel)}" in message
    assert "'ols'" not in message
    # One line per problem, the models with the same one together: the extra of all three, the weights of each class.
    assert len(raised.value.problems) == 3
    assert raised.value.problems[1].startswith("CardedModel 'a', CardedModel 'b': no weights")
    copy = pickle.loads(pickle.dumps(raised.value))
    assert (str(copy), copy.problems) == (message, raised.value.problems)


def test_the_preflight_raises_missing_extra_error_when_only_extras_or_pins_are_missing(data, fetched, monkeypatch):
    started = record_starts(monkeypatch)
    with pytest.raises(MissingExtraError) as raised:
        run(data, carded("a"), PinnedModel(PREDICTORS, WINDOW, name="c"))
    assert started == []
    message = str(raised.value)
    assert "PinnedModel 'c': needs numpy<1, installed: numpy " in message and "'a'" not in message
    assert raised.value.extra == FAKE_EXTRA
    fake_extra(monkeypatch, present=False)
    with pytest.raises(MissingExtraError) as raised:
        run(data, carded("a"))
    assert raised.value.extra == FAKE_EXTRA and "pip install 'epftoolbox2[fakefam]'" in str(raised.value)
    copy = pickle.loads(pickle.dumps(raised.value))
    assert (str(copy), copy.extra, copy.what) == (str(raised.value), FAKE_EXTRA, "CardedModel 'a'")
    assert started == []


def test_a_missing_runtime_package_and_a_commit_pin_are_problems(fetched, monkeypatch):
    class Gone(CardedModel):
        _card = dataclasses.replace(CARD, runtime=("epftoolbox2-no-such-distribution>=1",))

    class AtCommit(CardedModel):
        _card = dataclasses.replace(CARD, runtime=("numpy@0123456789abcdef",))

    with pytest.raises(MissingExtraError) as raised:
        preflight([Gone(PREDICTORS, name="g"), AtCommit(PREDICTORS, name="h")])
    message = str(raised.value)
    assert "needs epftoolbox2-no-such-distribution>=1, which is not installed" in message
    assert "needs numpy at commit 0123456789abcdef, installed: " in message and "(not from a repository)" in message


def test_a_core_model_has_no_preflight_and_an_unknown_extra_is_a_card_bug(monkeypatch):
    assert OLSModel(PREDICTORS)._preflight() == [] and OLSModel(PREDICTORS).card is None
    preflight([OLSModel(PREDICTORS)])

    class Typo(CardedModel):
        _card = dataclasses.replace(CARD, extra="fakefan")

    with pytest.raises(ValueError, match="Unknown extra 'fakefan'"):
        preflight([Typo(PREDICTORS)])


def test_a_workflow_array_task_runs_the_preflight_of_its_models_before_the_data(tmp_path, weights_root, monkeypatch):
    monkeypatch.setitem(COMPONENT_REGISTRY["models"], "CardedModel", "tests.models.carded_fake")
    index = pd.date_range("2023-01-01", "2023-03-16", freq="1h", tz="UTC", inclusive="left")
    price = (
        100 + 20 * np.sin(2 * np.pi * np.arange(len(index)) / 24) + np.random.default_rng(0).normal(0, 5, len(index))
    )
    csv = tmp_path / "prices.csv"
    pd.DataFrame({"datetime": index.strftime("%Y-%m-%d %H:%M:%S"), "price": price}).to_csv(csv, index=False)
    params = {"predictors": ["price_d-1", "price_d-7"], "training_window": WINDOW}
    config = {
        "data_pipeline": {
            "start": "2023-01-01",
            "end": "2023-03-15",
            "sources": [{"class": "CsvSource", "params": {"file_path": str(csv)}}],
            "transformers": [
                {"class": "TimezoneTransformer", "params": {"target_tz": "Europe/Warsaw"}},
                {"class": "LagTransformer", "params": {"columns": ["price"], "lags": [1, 7], "freq": "day"}},
            ],
        },
        "model_pipeline": {
            "test_start": TEST[0],
            "test_end": TEST[1],
            "horizon": 1,
            "save_dir": "results",
            "models": [
                {"class": "OLSModel", "params": {**params, "name": "ols"}},
                {"class": "CardedModel", "params": {**params, "name": "carded"}},
            ],
        },
    }
    fetched_data = []
    real_run_data = Workflow._run_data
    monkeypatch.setattr(Workflow, "_run_data", lambda self, dp=None: fetched_data.append(1) or real_run_data(self, dp))

    def workflow(**environment):
        path = tmp_path / "experiment.yaml"
        path.write_text(yaml.safe_dump({**config, "environment": environment}))
        return Workflow.load(path)

    with pytest.raises(MissingWeightsError, match="CardedModel 'carded'"):
        workflow(model_index=1).run()
    assert fetched_data == []  # refused before the data pipeline ran
    with pytest.raises(MissingWeightsError):
        workflow().run()
    assert fetched_data == []
    # The task of the core model alone has nothing missing, whatever another task lacks.
    report = workflow(model_index=0).run()
    assert list(report.refs) == ["ols"] and fetched_data == [1]


# --- weights: the hash, the cache and the offline scope ------------------------------------------------------------------


def test_a_file_is_hashed_once_per_path_size_and_mtime(tmp_path, monkeypatch):
    path = tmp_path / "w.bin"
    path.write_bytes(b"a" * 1000)
    reads = []
    real = hashlib.file_digest
    monkeypatch.setattr(hashlib, "file_digest", lambda handle, name: reads.append(1) or real(handle, name))
    first = sha256_of(path)
    assert first == hashlib.sha256(b"a" * 1000).hexdigest()
    assert sha256_of(path) == first and len(reads) == 1  # kept
    # Rewritten in place with the same size: the mtime differs, so it is read again (a cache keyed by path alone wouldn't).
    stat = path.stat()
    path.write_bytes(b"b" * 1000)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert sha256_of(path) == hashlib.sha256(b"b" * 1000).hexdigest() and len(reads) == 2
    path.write_bytes(b"c" * 999)  # another size
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert sha256_of(path) == hashlib.sha256(b"c" * 999).hexdigest() and len(reads) == 3
    assert sha256_of(path, cached=False) == hashlib.sha256(b"c" * 999).hexdigest() and len(reads) == 4


def test_offline_holds_the_variables_and_puts_back_what_was_there(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    constants = types.SimpleNamespace(HF_HUB_OFFLINE=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    with offline():
        assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert constants.HF_HUB_OFFLINE is True
        with offline():
            pass
        assert os.environ["HF_HUB_OFFLINE"] == "1" and constants.HF_HUB_OFFLINE is True  # still inside the outer one
    assert "HF_HUB_OFFLINE" not in os.environ and os.environ["TRANSFORMERS_OFFLINE"] == "0"
    assert constants.HF_HUB_OFFLINE is False
    with pytest.raises(RuntimeError), offline():
        raise RuntimeError("an adapter's load failed")
    assert "HF_HUB_OFFLINE" not in os.environ and os.environ["TRANSFORMERS_OFFLINE"] == "0"
    assert constants.HF_HUB_OFFLINE is False


def test_resolve_reads_the_cache_and_paths_and_refuses_a_changed_file(fetched, tmp_path):
    resolved = resolve(CARD)
    assert resolved.ref == f"{REPO}@{REVISION}" and resolved.files == {WEIGHTS_FILE: REFERENCE_SHA}
    assert resolved.reference is True and resolved.path() == fetched / REPO / REVISION / WEIGHTS_FILE
    assert resolved.record() == {"ref": f"{REPO}@{REVISION}", "files": {WEIGHTS_FILE: REFERENCE_SHA}, "reference": True}
    # A file given by path is used as given; another one than the card's is no reference.
    other = tmp_path / "mine.bin"
    other.write_bytes(weights_bytes(0.25))
    given = resolve(CARD, str(other))
    assert given.ref == str(other) and given.reference is False and list(given.files) == ["mine.bin"]
    directory = tmp_path / "dir"
    directory.mkdir()
    (directory / WEIGHTS_FILE).write_bytes(REFERENCE)
    assert resolve(CARD, directory).reference is True
    with pytest.raises(MissingWeightsError, match="lacks fake-fm.bin"):
        resolve(CARD, tmp_path / "weights")
    with pytest.raises(MissingWeightsError, match="neither a file nor a directory"):
        resolve(CARD, tmp_path / "nowhere")
    # Changed after the fetch: refused, and `epf weights verify` named.
    cached_file = fetched / REPO / REVISION / WEIGHTS_FILE
    cached_file.write_bytes(weights_bytes(0.5))
    with pytest.raises(MissingWeightsError, match="changed since it was fetched.*epf weights verify"):
        resolve(CARD)
    cached_file.unlink()
    with pytest.raises(MissingWeightsError, match="fake-fm.bin is missing"):
        resolve(CARD)
    assert resolve(dataclasses.replace(CARD, weights=None, corpus=RUN_DATA)) is None


# --- fetch ------------------------------------------------------------------------------------------------------------------


def test_fetch_needs_the_licence_accepted_verifies_and_writes_the_sidecar(weights_root, monkeypatch, no_network):
    calls = []
    monkeypatch.setattr(_weights, "snapshot_download", fake_download({WEIGHTS_FILE: REFERENCE}, calls))
    with pytest.raises(LicenceNotAcceptedError, match=r"--accept 'Fake-NC-1.0'"):
        _weights.fetch(CARD, "CardedModel")
    with pytest.raises(LicenceNotAcceptedError):
        _weights.fetch(CARD, "CardedModel", accept="MIT")
    assert calls == [] and not weights_root.exists()  # refused before any download
    sidecar = _weights.fetch(CARD, "CardedModel", accept=LICENCE)
    assert calls == [
        {
            "repo_id": REPO,
            "revision": REVISION,
            "allow_patterns": [WEIGHTS_FILE],
            "local_dir": str(weights_root / REPO / REVISION),
        }
    ]
    on_disk = json.loads((weights_root / REPO / REVISION / SIDECAR).read_text())
    assert on_disk == sidecar
    assert sidecar["files"] == {WEIGHTS_FILE: {"sha256": REFERENCE_SHA, "bytes": len(REFERENCE)}}
    assert (sidecar["licence"], sidecar["accepted"], sidecar["reference"]) == (LICENCE, LICENCE, True)
    assert sidecar["family"] == "fake-fm" and sidecar["bytes"] == len(REFERENCE)
    # A permissive card needs nothing accepted.
    permissive = dataclasses.replace(CARD, commercial_use=True, revenue_cap=None)
    assert _weights.fetch(permissive, "CardedModel")["accepted"] is None


def test_fetch_refuses_files_upstream_replaced_unless_the_revision_is_given(weights_root, monkeypatch, no_network):
    replaced = weights_bytes(0.0, note=" (squashed upstream)")
    with pytest.raises(
        WeightsMismatchError, match=f"{WEIGHTS_FILE}: sha256 .* downloaded, {REFERENCE_SHA} on the card"
    ):
        fetch_into_cache(monkeypatch, replaced)
    assert not (weights_root / REPO / REVISION / SIDECAR).exists()  # nothing recorded
    with pytest.raises(MissingWeightsError, match="no weights"):
        resolve(CARD)
    sidecar, _ = fetch_into_cache(monkeypatch, replaced, revision=REVISION)
    assert sidecar["files"][WEIGHTS_FILE]["sha256"] == hashlib.sha256(replaced).hexdigest()
    assert sidecar["reference"] is False
    resolved = resolve(CARD)
    assert resolved.reference is False and resolved.files[WEIGHTS_FILE] == hashlib.sha256(replaced).hexdigest()
    # A card that records no hash takes the one fetched.
    unhashed = dataclasses.replace(CARD, weights=WeightsRef("fake-org/other", "main", {WEIGHTS_FILE: None}))
    monkeypatch.setattr(_weights, "snapshot_download", fake_download({WEIGHTS_FILE: REFERENCE}, []))
    assert _weights.fetch(unhashed, "Other", accept=LICENCE)["reference"] is None
    assert resolve(unhashed).reference is None


def epf(monkeypatch, *argv):
    monkeypatch.setattr(cli, "console", Console(width=300))  # a table's cells unwrapped
    monkeypatch.setattr(sys, "argv", ["epf", *argv])
    try:
        cli.main()
    except SystemExit as stop:
        return stop.code
    return 0


def said(capsys) -> str:
    return " ".join(capsys.readouterr().out.split())


def test_epf_weights_fetch_list_and_verify(weights_root, monkeypatch, capsys, no_network):
    calls = []
    monkeypatch.setattr(_weights, "snapshot_download", fake_download({WEIGHTS_FILE: REFERENCE}, calls))
    assert epf(monkeypatch, "weights", "list") == 0
    assert "No model weights in" in said(capsys)
    assert epf(monkeypatch, "weights", "fetch", CLASS_PATH) == 1
    out = said(capsys)
    assert "--accept 'Fake-NC-1.0'" in out and "revenue cap: no rights above USD 20M" in out and calls == []
    assert epf(monkeypatch, "weights", "fetch", CLASS_PATH, "--accept", LICENCE) == 0
    out = said(capsys)
    assert f"{REPO}@{REVISION} (1 file(s), 1.6 kB)" in out and "attribution: 2026 - Fake Labs" in out
    assert "commercial use: not allowed" in out and f"destination: {weights_root / REPO / REVISION}" in out
    assert "fetched 1 file(s), 1.6 kB, sha256 verified" in out and len(calls) == 1
    assert epf(monkeypatch, "weights", "list") == 0
    out = said(capsys)
    assert f"{REPO}@{REVISION}" in out and "Fake-NC-1.0 (accepted)" in out and "fake-fm" in out
    assert epf(monkeypatch, "weights", "verify") == 0
    assert "1 file(s) verified" in said(capsys)
    (weights_root / REPO / REVISION / WEIGHTS_FILE).write_bytes(weights_bytes(1.0))
    assert epf(monkeypatch, "weights", "verify") == 1
    assert "changed" in said(capsys)
    assert epf(monkeypatch, "weights", "fetch", "NoSuchModel") == 1
    assert "Unknown model class 'NoSuchModel'" in said(capsys)
    assert epf(monkeypatch, "weights", "fetch", CLASS_PATH, "--size", "209m") == 1
    assert "has no size '209m'" in said(capsys)
    assert epf(monkeypatch, "weights", "fetch", "all") == 0
    assert "No registered model has weights to fetch" in said(capsys)
    monkeypatch.setitem(COMPONENT_REGISTRY["models"], "CardedModel", "tests.models.carded_fake")
    assert epf(monkeypatch, "weights", "fetch", "all", "--accept", LICENCE) == 0
    assert "CardedModel: fake-org/fake-fm@v1" in said(capsys) and len(calls) == 2


def test_a_class_with_sizes_fetches_the_size_named(weights_root, monkeypatch, capsys, no_network):
    small = dataclasses.replace(CARD, weights=WeightsRef("fake-org/fake-fm-small", "v1", {WEIGHTS_FILE: REFERENCE_SHA}))
    monkeypatch.setattr(CardedModel, "_card_variants", {"28m": small, "209m": CARD}, raising=False)
    monkeypatch.setattr(_weights, "snapshot_download", fake_download({WEIGHTS_FILE: REFERENCE}, []))
    assert epf(monkeypatch, "weights", "fetch", CLASS_PATH, "--size", "28m", "--accept", LICENCE) == 0
    assert f"{CLASS_PATH} --size 28m: fake-org/fake-fm-small@v1" in said(capsys)
    assert (weights_root / "fake-org/fake-fm-small/v1" / SIDECAR).is_file()
    model = carded()
    model._card = small
    assert _cards.fetch_command(model) == f"epf weights fetch {CLASS_PATH} --size 28m"


# --- runs: no network, the manifest, the report, the exporters --------------------------------------------------------------


def test_a_run_reads_its_weights_offline_and_records_its_card(data, fetched, tmp_path, monkeypatch, no_network):
    save_dir = tmp_path / "out"
    report = run(data, OLSModel(PREDICTORS, WINDOW, name="ols"), carded(), save_dir=save_dir)
    assert forecasts(report, "carded") == forecasts(report, "ols")  # the reference weights add 0.0
    assert "HF_HUB_OFFLINE" not in os.environ and "TRANSFORMERS_OFFLINE" not in os.environ
    stored = manifest(save_dir, "carded")
    assert stored["card"]["family"] == "fake-fm" and stored["card"]["verdicts"] == {"test": POSSIBLY_CONTAMINATED}
    assert stored["runtime"]["device"] == "cpu" and stored["runtime"]["threads_per_fit"] == 1
    assert set(stored["runtime"]["packages"]) == {"numpy", "pandas"}
    assert stored["runtime"]["packages"]["pandas"] == pd.__version__
    assert stored["weights"] == {"ref": f"{REPO}@{REVISION}", "files": {WEIGHTS_FILE: REFERENCE_SHA}, "reference": True}
    assert not {"card", "runtime", "weights"} & set(manifest(save_dir, "ols"))  # a core model's manifest is as before
    for shown in (
        report,
        ModelPipeline()
        .add_model(OLSModel(PREDICTORS, WINDOW, name="ols"))
        .add_model(carded())
        .report(save_dir, test_start=TEST[0], test_end=TEST[1], horizon=1),
    ):
        table = shown.provenance()
        assert list(table.columns) == PROVENANCE_COLUMNS and list(table["model"]) == ["carded"]
        row = table.iloc[0]
        assert (row["family"], row["licence"], row["commercial_use"]) == ("fake-fm", LICENCE, False)
        assert row["revenue_cap"] == CARD.revenue_cap and row["attribution"] == "2026 - Fake Labs"
        assert row["weights"] == f"{REPO}@{REVISION}" and row["sha256"] == REFERENCE_SHA[:12]
        assert row["runtime"].startswith("numpy ") and "device=cpu, threads_per_fit=1" in row["runtime"]
        assert (row["corpus_end"], row["released"]) == ("2023-06-30", "2023-01-15")
        assert row["validation_verdict"] is None and row["test_verdict"] == POSSIBLY_CONTAMINATED
    # Weights of the user's own reach the fit in the worker, and are no reference.
    mine = tmp_path / "mine.bin"
    mine.write_bytes(weights_bytes(0.25))
    with pytest.warns(UserWarning, match="not the reference weights"):
        moved = run(data, carded(weights=str(mine)))
    assert all(a[3] != b[3] for a, b in zip(forecasts(moved, "carded"), forecasts(report, "ols")))
    assert moved.provenance().iloc[0]["weights"] == str(mine)
    # A loop of variants on the same weights warns once.
    with pytest.warns(UserWarning, match="not the reference weights") as caught:
        preflight([carded(f"v{i}", weights=str(mine)) for i in range(5)])
    assert sum("not the reference weights" in str(w.message) for w in caught) == 1


def test_verify_and_report_reach_no_network(data, fetched, tmp_path, no_network):
    from epftoolbox2.information_set import Availability, InformationSet

    rules = {
        "load": Availability.published("10:00", 7),
        "wind": Availability.published("10:00", 7),
        "is_weekend": Availability.static(),
        "is_weekday": Availability.static(),
    }
    model = CardedModel([*PREDICTORS, "price_d-1"], WINDOW, name="carded")
    pipeline = ModelPipeline(information_set=InformationSet(rules=rules)).add_model(model)
    checked = pipeline.verify(data, TEST[0], TEST[1], days=1, horizon=1)
    assert checked.table()["status"].tolist() == ["ok"]
    pipeline.run(data, test_start=TEST[0], test_end=TEST[1], horizon=1, save_dir=str(tmp_path / "out"))
    assert pipeline.report(tmp_path / "out", test_start=TEST[0], test_end=TEST[1], horizon=1).provenance().shape[0] == 1


def test_the_verdict_of_each_window_is_the_card_s(data, fetched, tmp_path, no_network):
    class Ended(CardedModel):
        _card = dataclasses.replace(CARD, corpus_end=date(2023, 2, 22))

    model = Ended(PREDICTORS, WINDOW, name="ended")
    report = (
        ModelPipeline()
        .add_model(model)
        .run(data, validation=("2023-02-20", "2023-02-23"), test=TEST, horizon=1, save_dir=str(tmp_path / "w"))
    )
    for shown in (report, report.validation):
        row = shown.provenance().iloc[0]
        assert (row["validation_verdict"], row["test_verdict"]) == (POSSIBLY_CONTAMINATED, CLEAN)
    assert manifest(tmp_path / "w", "ended")["card"]["verdicts"] == {"validation": POSSIBLY_CONTAMINATED, "test": CLEAN}

    # A plain run whose test window holds the corpus end: tainted from its first day on, in memory and from the files.
    class Inside(CardedModel):
        _card = dataclasses.replace(CARD, corpus_end=date(2023, 3, 7))

    plain = run(data, Inside(PREDICTORS, WINDOW, name="inside"), save_dir=tmp_path / "p")
    stored = (
        ModelPipeline()
        .add_model(Inside(PREDICTORS, WINDOW, name="inside"))
        .report(tmp_path / "p", test_start=TEST[0], test_end=TEST[1], horizon=1)
    )
    for shown in (plain, stored):
        assert shown.provenance().iloc[0]["test_verdict"] == POSSIBLY_CONTAMINATED
    assert manifest(tmp_path / "p", "inside")["card"]["verdicts"] == {"test": POSSIBLY_CONTAMINATED}


def test_the_exporters_print_provenance_only_for_a_carded_model(
    data, fetched, tmp_path, capsys, monkeypatch, no_network
):
    monkeypatch.setenv("COLUMNS", "300")  # the terminal exporter's table unwrapped

    def export(*models, where):
        pipeline = ModelPipeline()
        for model in models:
            pipeline.add_model(model)
        pipeline.add_exporter(TerminalExporter(show=["summary"]))
        pipeline.add_exporter(CsvExporter(str(where / "forecasts.csv")))
        pipeline.add_exporter(ExcelExporter(str(where / "report.xlsx"), sheets=["summary"]))
        pipeline.add_evaluator(MAEEvaluator())
        pipeline.run(data, test_start=TEST[0], test_end=TEST[1], horizon=1)
        return " ".join(capsys.readouterr().out.split())

    core = tmp_path / "core"
    out = export(OLSModel(PREDICTORS, WINDOW, name="ols"), where=core)
    assert "Model provenance" not in out
    assert not (core / "provenance.csv").exists()
    assert load_workbook(core / "report.xlsx").sheetnames == ["Summary"]

    both = tmp_path / "both"
    out = export(OLSModel(PREDICTORS, WINDOW, name="ols"), carded(), where=both)
    assert "Model provenance" in out
    for text in ("fake-fm", LICENCE, "2026 - Fake Labs", "possibly contaminated", REFERENCE_SHA[:12]):
        assert text in out
    table = pd.read_csv(both / "provenance.csv")
    assert list(table.columns) == PROVENANCE_COLUMNS and list(table["model"]) == ["carded"]
    assert table.loc[0, "licence"] == LICENCE
    book = load_workbook(both / "report.xlsx")
    assert book.sheetnames == ["Summary", "Provenance"]
    sheet = book["Provenance"]
    assert [cell.value for cell in sheet[1]] == PROVENANCE_COLUMNS and sheet["A2"].value == "carded"


# --- resume -----------------------------------------------------------------------------------------------------------------


def test_a_changed_runtime_or_weight_hash_refuses_a_resume(data, fetched, tmp_path, monkeypatch, no_network):
    save_dir = tmp_path / "out"
    run(data, carded(), save_dir=save_dir)
    lines = sorted((save_dir / "carded.jsonl").read_text().splitlines())
    run(data, carded(), save_dir=save_dir)  # the same runtime and weights resume

    real = _cards.installed_distribution

    def installed(version=None, commit=None):
        def found(name, seen=None):
            hit = real(name, seen)
            return (version or hit[0], commit) if name == "numpy" and hit else hit

        return found

    cases = {
        "a version": lambda patch: patch.setattr(_cards, "installed_distribution", installed("9.9.9")),
        "a commit": lambda patch: patch.setattr(_cards, "installed_distribution", installed(commit="0123abc")),
        "threads per fit": lambda patch: patch.setattr(CardedModel, "threads_per_fit", 2),
    }
    expected = {
        "a version": "runtime.packages.numpy: ",
        "a commit": "@0123abc' (this run)",
        "threads per fit": "runtime.threads_per_fit: 1 (existing results) vs 2 (this run)",
    }
    for case, change in cases.items():
        with monkeypatch.context() as patch:
            change(patch)
            with pytest.raises(ManifestMismatchError, match="different configuration") as raised:
                run(data, carded(), save_dir=save_dir)
        assert expected[case] in str(raised.value), (case, str(raised.value))
        assert "model_params" not in str(raised.value)
    with pytest.raises(ManifestMismatchError, match=r"runtime\.device: 'cpu' \(existing results\) vs 'cuda'"):
        run(data, carded(device="cuda"), save_dir=save_dir)

    # Weights replaced upstream and fetched again with the revision given: other hashes, refused, then overridden.
    fetch_into_cache(monkeypatch, weights_bytes(0.0, note=" (squashed)"), revision=REVISION)
    with pytest.warns(UserWarning, match="not the reference weights"):
        with pytest.raises(
            ManifestMismatchError, match=rf"weights\.files\.{WEIGHTS_FILE}: '{REFERENCE_SHA}'"
        ) as raised:
            run(data, carded(), save_dir=save_dir)
    assert f"(weights '{REPO}@{REVISION}' vs '{REPO}@{REVISION}')" in str(raised.value)
    with pytest.warns(UserWarning, match="not the reference weights"):
        run(data, carded(), save_dir=save_dir, allow_manifest_mismatch=True)
    assert manifest(save_dir, "carded")["weights"]["reference"] is False
    assert sorted((save_dir / "carded.jsonl").read_text().splitlines()) == lines  # resumed: nothing was left to run


def test_the_weights_compare_their_hashes_not_where_they_were_found(data, fetched, tmp_path, no_network):
    from epftoolbox2.results.manifest import Manifest

    save_dir = tmp_path / "out"
    run(data, carded(), save_dir=save_dir)
    stored = Manifest.from_dict(manifest(save_dir, "carded"))
    elsewhere = dataclasses.replace(stored, weights={**stored.weights, "ref": "/scratch/pd/weights", "reference": None})
    assert elsewhere.differences(stored) == []  # the same files on another machine
    other = dataclasses.replace(stored, weights={**stored.weights, "files": {WEIGHTS_FILE: "0" * 64}})
    assert other.differences(stored) == [
        f"weights.files.{WEIGHTS_FILE}: '{REFERENCE_SHA}' (existing results) vs '{'0' * 64}' (this run) "
        f"(weights '{REPO}@{REVISION}' vs '{REPO}@{REVISION}')"
    ]
    assert dataclasses.replace(stored, weights=None).differences(stored)[0].startswith("weights: {'files': ")
    # The class's own `weights` parameter is in its model_params, compared as every parameter is.
    copy = tmp_path / "elsewhere"
    copy.mkdir()
    (copy / WEIGHTS_FILE).write_bytes(REFERENCE)
    with pytest.raises(ManifestMismatchError) as raised:
        run(data, carded(weights=str(copy)), save_dir=save_dir)
    assert "model_params" in str(raised.value) and "weights.files" not in str(raised.value)


def test_old_manifests_resume(data, fetched, tmp_path, no_network):
    save_dir = tmp_path / "out"
    run(data, OLSModel(PREDICTORS, WINDOW, name="ols"), carded(), save_dir=save_dir)
    ols_keys = set(manifest(save_dir, "ols"))
    # A carded manifest written before cards were provenance (no `card`) resumes, and gets one.
    path = save_dir / "carded.manifest.json"
    stored = json.loads(path.read_text())
    del stored["card"]
    path.write_text(json.dumps(stored))
    run(data, OLSModel(PREDICTORS, WINDOW, name="ols"), carded(), save_dir=save_dir)
    assert set(manifest(save_dir, "ols")) == ols_keys and manifest(save_dir, "carded")["card"]["family"] == "fake-fm"


def test_a_core_model_s_manifest_fields_are_unchanged(data, tmp_path):
    from epftoolbox2.results.manifest import Manifest

    run(data, OLSModel(PREDICTORS, WINDOW, name="ols"), save_dir=tmp_path)
    stored = manifest(tmp_path, "ols")
    assert not {"card", "runtime", "weights"} & set(stored)
    assert Manifest.from_dict(stored).differences(Manifest.from_dict(stored)) == []


# --- the engine, imports ----------------------------------------------------------------------------------------------------


class AllFour(OLSModel):
    """Defines all four engine methods itself, so it is engine-capable unless `_engine_ok` says otherwise."""

    def _fit_predict(self, train_x, train_y, test_x):
        return super()._fit_predict(train_x, train_y, test_x)

    def _fit_model(self, train_x, train_y):
        return super()._fit_model(train_x, train_y)

    def _predict_fitted(self, fitted, test_x):
        return super()._predict_fitted(fitted, test_x)

    def _fit_stats(self, stats):
        return super()._fit_stats(stats)


class NotForTheEngine(AllFour):
    _engine_ok = False


def test_engine_ok_false_keeps_a_class_out_of_the_engine_whatever_it_defines(data, fetched, no_network):
    assert AllFour(PREDICTORS)._engine_capable() is True
    assert NotForTheEngine(PREDICTORS)._engine_capable() is False
    assert carded()._engine_capable() is False
    on = ModelPipeline(engine="on").add_model(carded()).run(data, test_start=TEST[0], test_end=TEST[1], horizon=1)
    off = run(data, carded())
    assert forecasts(on, "carded") == forecasts(off, "carded")


def test_importing_the_package_imports_no_extra():
    code = (
        "import sys\n"
        "import epftoolbox2, epftoolbox2.models, epftoolbox2.cli\n"
        "from epftoolbox2.models import _cards, _weights\n"
        "epftoolbox2.verify()\n"
        "heavy = ('numba', 'torch', 'huggingface_hub', 'transformers', 'safetensors', 'tabpfn', 'tabicl', 'autogluon')\n"
        "found = sorted(m for m in sys.modules if m.split('.')[0] in heavy)\n"
        "assert not found, found\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Extras" in result.stdout and "fast" in result.stdout


def test_the_extras_table_lists_every_extra(monkeypatch):
    from epftoolbox2._extras import extras_rows

    fake_extra(monkeypatch, present=True)
    rows = {name: (installed, modules) for name, installed, modules in extras_rows()}
    assert set(rows) == set(EXTRAS)
    assert rows[FAKE_EXTRA] == (True, "json found")
    fake_extra(monkeypatch, present=False)
    assert dict((n, (i, m)) for n, i, m in extras_rows())[FAKE_EXTRA] == (False, "epftoolbox2_no_such_backend -")


def test_the_socket_guard_refuses_and_records(no_network):
    import socket

    with pytest.raises(OSError, match="network access"):
        socket.create_connection(("example.invalid", 80))
    with pytest.raises(OSError, match="network access"):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("192.0.2.1", 80))
    assert [what for what, _ in no_network] == ["create_connection", "connect"]
    no_network.clear()  # caught here on purpose


def test_the_carded_family_passes_the_adapter_suite(data, fetched, tmp_path, monkeypatch, no_network):
    """The checks of `tests/models/adapter_suite.py` that cards turn on (the pre-flight with the extra hidden, the provenance
    row), and the others a carded family runs through, on the fake family with its weights from the cache."""
    from epftoolbox2.information_set import Availability, InformationSet
    from tests.models import adapter_suite as suite

    rules = {
        "load": Availability.published("10:00", 7),
        "wind": Availability.published("10:00", 7),
        "is_weekend": Availability.static(),
        "is_weekday": Availability.static(),
    }
    information_set = InformationSet(rules=rules)
    monkeypatch.setitem(COMPONENT_REGISTRY["models"], "CardedModel", "tests.models.carded_fake")

    def make(name, **params):
        return CardedModel([*PREDICTORS, "price_d-1"], WINDOW, name=name, **params)

    suite.check_preflight_without_extra(make, monkeypatch, data, TEST)
    suite.check_provenance(make, data, tmp_path / "provenance", TEST, information_set=information_set)
    (tmp_path / "yaml").mkdir()
    suite.check_yaml_round_trip(make, data, tmp_path / "yaml", TEST, information_set=information_set)
    suite.check_engine(make, data, TEST, information_set=information_set)
    suite.check_resume(make, data, tmp_path / "resume", TEST, information_set=information_set)
    suite.check_verify(make, data, information_set, [TEST[0]])


def test_verify_runs_the_preflight_before_any_probe(data, weights_root, monkeypatch, no_network):
    from epftoolbox2.information_set import Availability, InformationSet
    from epftoolbox2.verification import verify_model

    rules = {"load": Availability.published("10:00", 7), "wind": Availability.published("10:00", 7)}
    information_set = InformationSet(
        rules={**rules, "is_weekend": Availability.static(), "is_weekday": Availability.static()}
    )
    started = record_starts(monkeypatch)
    pipeline = ModelPipeline(information_set=information_set)
    pipeline.add_model(OLSModel(PREDICTORS, WINDOW, name="ols")).add_model(carded())
    with pytest.raises(MissingWeightsError, match="CardedModel 'carded': no weights"):
        pipeline.verify(data, TEST[0], TEST[1], days=1, horizon=1)
    assert started == []  # not even the core model's probe ran
    pipeline.verify(data, TEST[0], TEST[1], days=1, horizon=1, models=["ols"])  # a selection checks its own models
    started.clear()
    with pytest.raises(MissingWeightsError):
        verify_model(carded(), data, [TEST[0]], information_set)
    assert started == []
    fake_extra(monkeypatch, present=False)
    fetch_into_cache(monkeypatch)
    with pytest.raises(MissingExtraError, match=r"\[fakefam\]"):
        pipeline.verify(data, TEST[0], TEST[1], days=1, horizon=1)


# --- review fixes -------------------------------------------------------------------------------------------------------------


def test_each_model_is_judged_by_its_own_windows_in_a_joined_report(data, fetched, no_network):
    """A plain run's carded model in one report with another run's windowed model: its verdict is its own test window's."""
    from epftoolbox2.results.report import EvaluationReport

    class Inside(CardedModel):
        _card = dataclasses.replace(CARD, corpus_end=date(2023, 3, 7))

    plain = run(data, Inside(PREDICTORS, WINDOW, name="inside"))
    windowed = (
        ModelPipeline()
        .add_model(OLSModel(PREDICTORS, WINDOW, name="ols"))
        .run(data, validation=("2023-03-01", "2023-03-03"), test=("2023-03-09", "2023-03-10"), horizon=1)
    )
    for refs in ({**windowed.refs, **plain.refs}, {**plain.refs, **windowed.refs}):
        row = EvaluationReport(refs, [MAEEvaluator()]).provenance().iloc[0]
        assert (row["model"], row["validation_verdict"], row["test_verdict"]) == ("inside", None, POSSIBLY_CONTAMINATED)
