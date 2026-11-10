"""6j's foundation families (`PatchTSTFMModel`, `TimestralModel`, `TimesFMModel`) without their packages: parameters and
refusals, the cards (licences, attribution, verdicts), the flagged licences at fetch, the Timestral pre-flight's commit check, the
extras and install lines, the lean import, and the adapter suite with a numpy backend per family.

Each fake (`FakePatchTST`, `FakeTimestral`, `FakeTimesFM`) is the family's class with its card's package and weights taken off
and a numpy backend (`FlaggedBackend`, 6i's persistence fake) at the family's levels and point level, which asserts the call has
the arrays the real backend reads: univariate for PatchTST-FM, known-ahead series over the context and the steps ahead, past
series over the context. The real packages' adapters, with tiny random weights, are in `test_patchtst_fm.py`,
`test_timestral.py` and `test_timesfm.py` (the extras jobs).
"""

import dataclasses
import json
import tomllib
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from epftoolbox2._extras import (
    EXTRAS,
    TIMESTRAL_COMMIT,
    MissingExtraError,
    install_line,
)
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.exporters import CsvExporter, TerminalExporter
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import PatchTSTFMModel, TimesFMModel, TimestralModel
from epftoolbox2.models import _cards, _weights
from epftoolbox2.models._cards import (
    CLEAN,
    CLEAN_RELEASED_BEFORE,
    POSSIBLY_CONTAMINATED,
    card_variant,
    contamination_verdict,
    fetch_command,
)
from epftoolbox2.models._fm import LEVELS_19, Backend, nearest_level
from epftoolbox2.models.patchtst_fm import PATCHTST_FM_CARD
from epftoolbox2.models.timesfm import TIMESFM_CARD, TIMESFM_LICENCE
from epftoolbox2.models.timestral import TIMESTRAL_CARDS, TIMESTRAL_LICENCE
from epftoolbox2.pipelines import ModelPipeline
from tests.import_stubs import FAMILY_PACKAGES, run_without
from tests.models import adapter_suite as suite
from tests.models.carded_fake import fake_download
from tests.models.test_foundation import CONTEXT, INFO, INFO_AHEAD, market, pipeline

ROOT = Path(__file__).resolve().parents[2]
TIMESTRAL_LEVELS = (0.01, *LEVELS_19, 0.99)
DECILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


@pytest.fixture(scope="module")
def hourly():
    return market("1h", start="2023-01-20", end="2023-11-10")


@pytest.fixture(scope="module")
def quarter():
    return market("15min", start="2023-01-20", end="2023-03-10")


# ---------------------------------------------------------------------------------------------------------------------------
# Fakes: the family's class, no package and no weights, a numpy backend at its levels
# ---------------------------------------------------------------------------------------------------------------------------


def without_package(card):
    """The family's card with no extra, weights or runtime: a fake's pre-flight has nothing to check, its provenance is the
    family's (licence, attribution, corpus, dates)."""
    return dataclasses.replace(card, extra=None, weights=None, runtime=())


class FlaggedBackend(Backend):
    """Persistence of the context's last day, plus 0.1 x each known-ahead series ahead, minus 0.01 x each past series' mean
    over the context (its known steps), plus 0.5 when the call's `symmetric_averaging` is off and clamped at 0 with
    `clip_nonnegative` where the context has no negative value (a family's options reach its calls); the levels spread around
    it with the point level's column the forecast. Asserts the shapes of the arrays the real backends read."""

    def __init__(self, model, levels, point_level):
        self.model = model
        self.levels = tuple(levels)
        self.point_level = point_level
        self.calls = []

    def predict(self, call, options):
        target, steps = call.target, call.steps_ahead
        width = target.shape[1]
        size = target.shape[0]
        assert size == self.model.batch_size
        for values in call.known.values():
            assert values.shape == (size, width + steps)
        for values in call.past.values():
            assert values.shape == (size, width)
        assert list(call.known) == list(self.model.known_ahead) and list(call.past) == list(self.model.past)
        self.calls.append((dict(options), sorted(call.known), sorted(call.past)))
        periods = self.model._periods_per_day
        out = target[:, width - periods + np.arange(steps) % periods]
        for values in call.known.values():
            out = out + 0.1 * values[:, width:]
        for values in call.past.values():
            out = out - 0.01 * np.nanmean(values, axis=1)[:, None]
        if options.get("symmetric_averaging") is False:
            out = out + 0.5
        if options.get("clip_nonnegative"):
            eligible = (np.nan_to_num(target, nan=0.0) >= 0).all(axis=1)[:, None]
            out = np.where(eligible, np.maximum(out, 0.0), out)
        spread = np.asarray(self.levels) - self.point_level
        return out[..., None] + 10.0 * spread


class _Fake:
    """One numpy backend per model (the fake reads the model's own settings), the family's levels."""

    _uses_torch = False
    _fake_levels = LEVELS_19

    def _backend(self, device):
        if self.__dict__.get("_fake") is None:
            self.__dict__["_fake"] = FlaggedBackend(self, self._fake_levels, nearest_level(self._fake_levels))
        return self._fake


class FakePatchTST(_Fake, PatchTSTFMModel):
    _card = without_package(PATCHTST_FM_CARD)


FAKE_TIMESTRAL_CARDS = {size: without_package(card) for size, card in TIMESTRAL_CARDS.items()}


class FakeTimestral(_Fake, TimestralModel):
    _card = FAKE_TIMESTRAL_CARDS["209m"]
    _card_variants = FAKE_TIMESTRAL_CARDS
    _fake_levels = TIMESTRAL_LEVELS

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == "size":
            self.__dict__["_card"] = FAKE_TIMESTRAL_CARDS[self.size]


class FakeTimesFM(_Fake, TimesFMModel):
    _card = without_package(TIMESFM_CARD)
    _fake_levels = DECILES


FAKES = {"FakePatchTST": FakePatchTST, "FakeTimestral": FakeTimestral, "FakeTimesFM": FakeTimesFM}


@pytest.fixture
def registered(monkeypatch):
    from epftoolbox2.pipelines.model_pipeline import COMPONENT_REGISTRY

    for name in FAKES:
        monkeypatch.setitem(COMPONENT_REGISTRY["models"], name, __name__)


def factory(cls, **fixed):
    """`make(name, **params)` for the adapter suite: a fake of 7 days of context, batches of 8, on the CPU."""

    def make(name, **params):
        settings = {"context": CONTEXT, "batch_size": 8, "device": "cpu", **fixed, **params}
        return cls(name=name, **settings)

    return make


MAKERS = {
    "patchtst": factory(FakePatchTST),
    "timestral": factory(FakeTimestral, known_ahead=["load_forecast"], past=["load_actual"]),
    "timesfm": factory(FakeTimesFM, known_ahead=["load_forecast"], past=["load_actual"]),
}


def by_key(report, name):
    return {(r["run_date"], r["hour"], r["horizon"]): r["prediction"] for r in report.refs[name].iter_records()}


# ---------------------------------------------------------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------------------------------------------------------


def test_the_families_build_with_the_spec_s_defaults_and_save_only_their_parameters(tmp_path):
    patchtst, timestral, timesfm = PatchTSTFMModel(), TimestralModel(), TimesFMModel()
    assert (patchtst.context, patchtst.batch_size, patchtst.device, patchtst.name) == (364, 64, "auto", "patchtst_fm")
    assert (timestral.size, timestral.context, timestral.symmetric_averaging, timestral.clip_nonnegative) == (
        "209m",
        364,
        True,
        False,
    )
    assert (timestral.name, timestral.runtime_commit, timesfm.name) == ("timestral", None, "timesfm3")
    for model in (patchtst, timestral, timesfm):
        assert (model.refit, model.fine_tune_every, model.newest_target) == ("1D", None, "d-1")
    models = [
        PatchTSTFMModel(context=341, batch_size=16, device="cpu", newest_target="published", name="P"),
        TimestralModel(
            size="28m",
            context=182,
            known_ahead=["load_forecast"],
            past=["ren_forecast"],
            symmetric_averaging=False,
            clip_nonnegative=True,
            weights="/somewhere/timestral",
            runtime_commit="0123abc",
            name="T",
        ),
        TimesFMModel(context=91, known_ahead=["load_forecast"], past=["ren_forecast"], device="cuda:1", name="F"),
    ]
    path = tmp_path / "models.yaml"
    pipeline(*models).save(path)
    loaded = ModelPipeline.load(path)
    assert [type(m) for m in loaded.models] == [PatchTSTFMModel, TimestralModel, TimesFMModel]
    for again, model in zip(loaded.models, models):
        assert again._model_kwargs == model._model_kwargs
        assert {k: v for k, v in vars(again).items() if not k.startswith("_")} == {
            k: v for k, v in vars(model).items() if not k.startswith("_")
        }
    saved = path.read_text()
    for internal in ("fine_tune", "seeds", "random_state", "refit", "training_window"):
        assert internal not in saved
    patchtst_part = saved.split("TimestralModel")[0]
    assert "known_ahead" not in patchtst_part and "past" not in patchtst_part


def test_patchtst_fm_takes_no_covariates():
    """Univariate: the constructor has no `known_ahead` or `past`, and setting either raises; the base's fine-tune settings
    are no parameter either."""
    for name in ("known_ahead", "past"):
        with pytest.raises(TypeError, match=name):
            PatchTSTFMModel(**{name: ["load_forecast"]})
        model = PatchTSTFMModel()
        with pytest.raises(ValueError, match="univariate"):
            setattr(model, name, ["load_forecast"])
        assert getattr(model, name) == []
    model = PatchTSTFMModel()
    assert not {"known_ahead", "past", "fine_tune_every", "seeds"} & {k for k in vars(model) if not k.startswith("_")}
    with pytest.raises(ValueError, match="doesn't fine-tune"):
        model.fine_tune_every = "12MS"


def test_patchtst_fm_records_the_package_s_autocast_on_a_gpu(monkeypatch):
    """On a CUDA device granite-tsfm runs PatchTST-FM's backbone under bfloat16 autocast: the runtime records it (compared on
    resume), on the CPU it doesn't."""
    from epftoolbox2.models.foundation import FoundationModel

    for device, expected in (("cuda", {"autocast": "bfloat16"}), ("cpu", {})):
        monkeypatch.setattr(
            FoundationModel, "_runtime_fields", lambda self, d=device: {"device": d, "dtype": "float32"}
        )
        assert PatchTSTFMModel()._runtime_fields() == {"device": device, "dtype": "float32", **expected}
        assert "autocast" not in TimesFMModel()._runtime_fields()


def test_timestral_s_size_options_and_commit_are_checked():
    for size, card in TIMESTRAL_CARDS.items():
        model = TimestralModel(size=size.upper())
        assert model.size == size and model._card is card and card_variant(model) == size
        assert fetch_command(model) == f"epf weights fetch TimestralModel --size {size}"
    model = TimestralModel(size="28m")
    model.size = "123m"
    assert model._card is TIMESTRAL_CARDS["123m"] and model._model_kwargs["size"] == "123m"
    assert TimestralModel._card is TIMESTRAL_CARDS["209m"]  # what `epf weights fetch TimestralModel` fetches
    for bad in (
        dict(size="5m"),
        dict(size=209),
        dict(symmetric_averaging=1),
        dict(clip_nonnegative="no"),
        dict(runtime_commit="abc"),
        dict(runtime_commit="ABCDEF01"),
        dict(runtime_commit="a534bdb-dirty"),
    ):
        with pytest.raises(ValueError):
            TimestralModel(**bad)
    assert TimestralModel(runtime_commit="deadbeef")._runtime_pins() == ["timestral@deadbeef", "torch"]
    assert TimestralModel()._runtime_pins() == [f"timestral@{TIMESTRAL_COMMIT}", "torch"]
    assert TimestralModel(weights="/mine")._runtime_pins() == ["timestral", "torch"]


def test_nearest_level_is_the_point_level_of_a_family_without_an_exact_half():
    assert nearest_level(TIMESTRAL_LEVELS) == 0.5
    assert nearest_level((0.1, 0.45, 0.6)) == 0.45
    assert nearest_level((0.4, 0.6)) == 0.4  # the lower of two as near, as the research's argmin
    assert nearest_level(DECILES) == 0.5 and DECILES.index(nearest_level(DECILES)) == 4


# ---------------------------------------------------------------------------------------------------------------------------
# Cards, licences, the fetch, the pre-flight
# ---------------------------------------------------------------------------------------------------------------------------


def test_the_cards_say_what_the_spec_says():
    patchtst, timesfm = PatchTSTFMModel._card, TimesFMModel._card
    assert (patchtst.family, patchtst.extra, patchtst.commercial_use, patchtst.revenue_cap) == (
        "patchtst-fm-r2",
        "patchtst-fm",
        True,
        None,
    )
    assert "granite-tsfm>=0.3.10,<0.4" in patchtst.runtime and patchtst.weights.repo.startswith("ibm-granite/")
    # No corpus end date (the GIFT-Eval split and TSMixup carry none on its card): the verdict goes by the release, 2026-08-07.
    assert patchtst.stated_verdict is None and patchtst.corpus_end is None
    for day in ("2017-01-02", "2021-06-01", "2024-10-01", "2025-10-01", "2026-08-07"):
        assert contamination_verdict(patchtst, day) == POSSIBLY_CONTAMINATED, day
    assert contamination_verdict(patchtst, "2026-08-08") == CLEAN_RELEASED_BEFORE
    assert (timesfm.family, timesfm.extra, timesfm.licence, timesfm.commercial_use) == (
        "timesfm-3.0",
        "timesfm",
        TIMESFM_LICENCE,
        False,
    )
    assert "Non-Commercial" in timesfm.licence and "timesfm==3.0.2" in timesfm.runtime
    assert timesfm.corpus_end == date(2023, 11, 30) and timesfm.released == date(2026, 8, 24)
    assert contamination_verdict(timesfm, "2024-10-01") == CLEAN
    assert contamination_verdict(timesfm, "2023-11-01") == POSSIBLY_CONTAMINATED
    for size, card in TIMESTRAL_CARDS.items():
        assert (card.family, card.extra, card.licence, card.attribution) == (
            f"timestral-{size}",
            "timestral",
            TIMESTRAL_LICENCE,
            "2026 - Mistral AI",
        )
        assert card.commercial_use and "USD 20 million" in card.revenue_cap and "employer" in card.revenue_cap
        assert card.runtime == (f"timestral@{TIMESTRAL_COMMIT}", "torch")
        assert card.weights.repo == f"mistralai/Timestral-{size.upper()}" and card.corpus_end is None
        # Released 2026-10-15 with an undocumented corpus: every window before is possibly contaminated.
        for day in ("2024-10-01", "2025-10-01", "2026-10-14", "2026-10-15"):
            assert contamination_verdict(card, day) == POSSIBLY_CONTAMINATED, day
        assert contamination_verdict(card, "2026-10-16") == CLEAN_RELEASED_BEFORE
    assert TIMESTRAL_CARDS["209m"].weights.revision == "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"
    assert TIMESTRAL_CARDS["209m"].weights.files["model.safetensors"] == (
        "1456effb9927e3daa8c7234f10497e3424233869245a9c72b8ecebaf7ad2b7fa"
    )


def test_patchtst_fm_is_judged_no_cleaner_than_t0_on_the_corpus_they_share():
    """PatchTST-FM-r2 trains on (a subset of) the GIFT-Eval pretraining split, which t0-alpha's card dates to 2021: a window
    t0 is possibly contaminated for can't read clean for PatchTST-FM, whose card gives no end date at all."""
    from epftoolbox2.models import T0Model

    for day in ("2017-01-02", "2021-06-01"):
        assert contamination_verdict(PatchTSTFMModel._card, day) == contamination_verdict(T0Model._card, day), day
        assert contamination_verdict(PatchTSTFMModel._card, day) == POSSIBLY_CONTAMINATED


@pytest.mark.parametrize(
    "card, licence",
    [(TIMESFM_CARD, TIMESFM_LICENCE), *((card, TIMESTRAL_LICENCE) for card in TIMESTRAL_CARDS.values())],
    ids=["timesfm", *TIMESTRAL_CARDS],
)
def test_a_flagged_licence_is_fetched_only_with_its_accept(card, licence, tmp_path, monkeypatch, no_network):
    """TimesFM-3 (non-commercial) and every Timestral size (a revenue cap) refuse a fetch without `--accept <licence>`, before
    any download; with it the download starts (here a stand-in whose bytes aren't the card's, so the hash check refuses)."""
    monkeypatch.setenv(_weights.WEIGHTS_DIR_ENV, str(tmp_path / "weights"))
    calls = []
    monkeypatch.setattr(_weights, "snapshot_download", fake_download(dict.fromkeys(card.weights.files, b"x"), calls))
    for accept in (None, "MIT", "Apache-2.0"):
        with pytest.raises(_weights.LicenceNotAcceptedError, match=licence):
            _weights.fetch(card, card.family, accept=accept)
    assert calls == []
    with pytest.raises(_weights.WeightsMismatchError):
        _weights.fetch(card, card.family, accept=licence)
    assert (
        len(calls) == 1 and calls[0]["repo_id"] == card.weights.repo and calls[0]["revision"] == card.weights.revision
    )


def test_patchtst_fm_is_fetched_without_an_accept(tmp_path, monkeypatch, no_network):
    monkeypatch.setenv(_weights.WEIGHTS_DIR_ENV, str(tmp_path / "weights"))
    calls = []
    files = dict.fromkeys(PATCHTST_FM_CARD.weights.files, b"x")
    monkeypatch.setattr(_weights, "snapshot_download", fake_download(files, calls))
    with pytest.raises(_weights.WeightsMismatchError):  # the stand-in's bytes: past the licence, refused by the hash
        _weights.fetch(PATCHTST_FM_CARD, "PatchTSTFMModel")
    assert len(calls) == 1


def installed_timestral(monkeypatch, version="1.0.0", commit=TIMESTRAL_COMMIT, present=True):
    """The timestral package installed (`installed_distribution`), from a repository at `commit` (None: not from one), or not
    installed at all (`present=False`), its extra found, and every card's weights resolved: what the pre-flight then says is
    the runtime's alone."""
    real = _cards.installed_distribution

    def installed(name, seen=None):
        if name == "timestral":
            return (version, commit) if present else None
        return real(name, seen)

    providers = _cards.runtime_providers
    monkeypatch.setattr(_cards, "installed_distribution", installed)
    monkeypatch.setattr(
        _cards,
        "runtime_providers",
        lambda n, s=None: (["timestral"] if present else []) if n == "timestral" else providers(n, s),
    )
    monkeypatch.setattr(_cards, "has_extra", lambda name: True)
    monkeypatch.setattr(_cards, "resolve_model_weights", lambda model, seen=None: None)
    import epftoolbox2.models.timestral as module

    monkeypatch.setattr(module, "installed_distribution", installed)


def runtime_problems(model):
    return [problem.message for problem in model._preflight({}) if problem.kind == "runtime"]


def timestral_problems(model):
    """The pre-flight's problems with the timestral package (not torch's, whose install line names timestral too)."""
    return [message for message in runtime_problems(model) if message.startswith("needs timestral")]


def test_the_timestral_pre_flight_refuses_another_commit_unless_the_model_says_so(monkeypatch):
    """PEP 610's commit of the installed package against the pin: the pinned commit passes; another commit, or a build not
    from the repository, is refused (naming the install line and, for a commit, `runtime_commit=`); `runtime_commit=` makes
    that commit the one required; `weights=` (the user's own) lets any build run, with a warning."""
    torch_installed = _cards.installed_distribution("torch") is not None
    other = "f" * 40
    installed_timestral(monkeypatch)
    assert timestral_problems(TimestralModel()) == []
    installed_timestral(monkeypatch, commit=other)
    (refused,) = timestral_problems(TimestralModel())
    assert f"needs timestral at commit {TIMESTRAL_COMMIT}, installed: 1.0.0 (commit {other})" in refused
    assert f"git+https://github.com/mistralai/timestral@{TIMESTRAL_COMMIT}" in refused
    assert f"runtime_commit={other[:12]!r}" in refused
    assert timestral_problems(TimestralModel(runtime_commit=other[:12])) == []
    installed_timestral(monkeypatch, commit=None)
    (refused,) = timestral_problems(TimestralModel())
    assert "(not from a repository)" in refused and "runtime_commit=" not in refused
    (refused,) = timestral_problems(TimestralModel(runtime_commit=other[:12]))
    assert f"needs timestral at commit {other[:12]}" in refused
    with pytest.warns(UserWarning, match="isn't the pinned commit a534bdbd244d; it runs because weights= is given"):
        assert timestral_problems(TimestralModel(weights="/mine")) == []
    if not torch_installed:  # the core job: torch is refused as not installed, by its own pin
        assert any("needs torch" in m for m in runtime_problems(TimestralModel()))


def test_with_runtime_commit_a_refusal_names_the_commit_the_run_needs(monkeypatch):
    """`runtime_commit=` makes another commit the one a run needs: the install line of its refusal names that commit, whether
    another commit is installed or none, never the pinned one."""
    wanted, other = "0123abcd", "f" * 40
    installed_timestral(monkeypatch, commit=other)
    (refused,) = timestral_problems(TimestralModel(runtime_commit=wanted))
    assert f"timestral@{wanted}'" in refused and TIMESTRAL_COMMIT not in refused, refused
    installed_timestral(monkeypatch, present=False)
    (missing,) = timestral_problems(TimestralModel(runtime_commit=wanted))
    assert "which is not installed" in missing and f"timestral@{wanted}'" in missing, missing
    assert TIMESTRAL_COMMIT not in missing
    (pinned,) = timestral_problems(TimestralModel())
    assert f"timestral@{TIMESTRAL_COMMIT}'" in pinned, pinned


def test_the_manifest_records_the_installed_commit(monkeypatch):
    """`runtime.packages.timestral` is `version@commit`, which a resume compares: another commit refuses it."""
    installed_timestral(monkeypatch, commit="f" * 40)
    assert _cards.runtime_versions(TIMESTRAL_CARDS["28m"])["timestral"] == "1.0.0@" + "f" * 40


def test_the_extras_and_their_install_lines():
    assert EXTRAS["patchtst-fm"][0] == "tsfm_public" and EXTRAS["timesfm"][0] == "timesfm3"
    assert EXTRAS["timestral"][0] == "timestral"
    line = install_line("timestral")
    assert line == (
        "pip install 'epftoolbox2[timestral]' "
        f"'timestral @ git+https://github.com/mistralai/timestral@{TIMESTRAL_COMMIT}'"
    )
    assert install_line("timesfm") == "pip install 'epftoolbox2[timesfm]'"
    assert install_line("t0,timestral").startswith("pip install 'epftoolbox2[t0,timestral]' 'timestral @ git+")
    assert str(MissingExtraError("TimestralModel", "timestral")) == f"TimestralModel needs: {line}"
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]
    assert extras["patchtst-fm"] == [
        "granite-tsfm>=0.3.10,<0.4; python_version < '3.14'",
        "scikit-learn>=1.8; python_version >= '3.14'",
    ]
    assert extras["timesfm"] == ["timesfm[torch]==3.0.2"]
    assert not any("timestral" in requirement.split(";")[0] for requirement in extras["timestral"])  # from git, by hand
    # Flagged licences are in no extra of everything; granite-tsfm's pins are resolved apart (uv conflicts).
    for name, requirements in extras.items():
        if name not in ("timesfm", "timestral"):
            assert not any("[timesfm" in r or "timestral]" in r or "timesfm[" in r for r in requirements), name
    assert [{"extra": "patchtst-fm"}, {"extra": "torch"}] in pyproject["tool"]["uv"]["conflicts"]
    assert "all" not in extras or "patchtst-fm" not in str(extras["all"])  # 6q's [all] leaves it out


def locked_versions(lock, extra, python):
    """{direct dependency: the versions `uv.lock` installs for it} for an install of the project with `extra` (None: no extra)
    on CPython `python` on Linux, from the project's dependency entries and their markers (uv writes a version on an entry
    only where the lock holds several)."""
    from packaging.markers import Marker, default_environment

    project = next(package for package in lock["package"] if package["name"] == "epftoolbox2")
    versions = {}
    for package in lock["package"]:
        versions.setdefault(package["name"], []).append(package["version"])
    environment = {
        **default_environment(),
        "python_full_version": python,
        "python_version": ".".join(python.split(".")[:2]),
        "sys_platform": "linux",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "extra": f"extra-11-epftoolbox2-{extra}" if extra else "",
    }
    entries = list(project.get("dependencies", []))
    if extra:
        entries += project.get("optional-dependencies", {}).get(extra, [])
    out = {}
    for entry in entries:
        if "marker" in entry and not Marker(entry["marker"]).evaluate(environment):
            continue
        out.setdefault(entry["name"], set()).add(entry.get("version") or versions[entry["name"]][0])
    return out


def test_the_patchtst_fm_extra_is_resolved_apart_and_moves_nothing_on_python_3_14_and_later():
    """granite-tsfm (Python < 3.14) pins torch < 2.12 and scikit-learn < 1.8. The lock resolves `[patchtst-fm]` apart from every
    extra that holds torch (declared conflicting, so a PatchTST-FM environment is its own and its torch 2.11 never runs another
    family), and on 3.14 and later, where it installs nothing, an install with it gets the core's versions (scikit-learn 1.7.2
    has no 3.14t wheel; torch 2.11.0+cpu none for 3.15)."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]
    with_torch = {
        name
        for name, requirements in extras.items()
        if any(r.split(";")[0].strip().startswith("torch") or "[torch]" in r for r in requirements)
    }
    assert {"torch", "t0", "chronos", "tabpfn", "tabicl", "timesfm", "timestral"} <= with_torch
    declared = {frozenset(item["extra"] for item in pair) for pair in pyproject["tool"]["uv"]["conflicts"]}
    locked = {frozenset(item["extra"] for item in pair) for pair in lock.get("conflicts", [])}
    for name in sorted(with_torch - {"patchtst-fm"}):
        assert frozenset({"patchtst-fm", name}) in declared, f"[patchtst-fm] isn't declared conflicting with [{name}]"
        assert frozenset({"patchtst-fm", name}) in locked, f"uv.lock doesn't hold the conflict with [{name}]: uv lock"
    for python in ("3.14.0", "3.15.0"):
        core, flagged = locked_versions(lock, None, python), locked_versions(lock, "patchtst-fm", python)
        assert flagged == core, (python, {k: (core.get(k), v) for k, v in flagged.items() if core.get(k) != v})
    below = locked_versions(lock, "patchtst-fm", "3.12.0")
    assert "granite-tsfm" in below and below["scikit-learn"] != locked_versions(lock, None, "3.12.0")["scikit-learn"]


def test_importing_or_building_the_families_imports_no_model_package(tmp_path):
    """Dawid's lean condition for 6j: `import epftoolbox2`, a calendar data pipeline and building the three families (their
    cards, a pipeline's YAML, `epftoolbox2.verify()`) try to import none of torch, transformers, safetensors, huggingface_hub
    or any family's package (`tests.import_stubs.FAMILY_PACKAGES`); each is a stub that logs an attempt and raises."""
    code = (
        "import epftoolbox2\n"
        "from epftoolbox2.data.sources import CalendarSource\n"
        "from epftoolbox2.pipelines import DataPipeline, ModelPipeline\n"
        "from epftoolbox2.models import PatchTSTFMModel, TimestralModel, TimesFMModel\n"
        "frame = DataPipeline(sources=[CalendarSource(country='PL')]).run('2024-01-01', '2024-01-08')\n"
        "assert len(frame), 'no calendar data'\n"
        "models = [PatchTSTFMModel(), TimestralModel(size='28m', known_ahead=['load_forecast']), TimesFMModel(past=['x'])]\n"
        "pipeline = ModelPipeline()\n"
        "[pipeline.add_model(model) for model in models]\n"
        "pipeline.to_dict()\n"
        "[model.card for model in models]\n"
        "epftoolbox2.verify()\n"
    )
    result, attempted = run_without(code, FAMILY_PACKAGES, tmp_path)
    assert result.returncode == 0, result.stderr
    assert attempted == [], f"tried to import {attempted}"
    for package in ("tsfm_public", "timesfm3", "timestral"):  # the guard sees what it guards
        probe = f"try:\n    import {package}\nexcept ImportError:\n    pass\n"
        result, attempted = run_without(probe, FAMILY_PACKAGES, tmp_path / f"probe-{package}")
        assert result.returncode == 0 and attempted == [package], (result.stderr, attempted)


@pytest.mark.parametrize("version", [(3, 14), (3, 15), (3, 13), (3, 11)])
def test_patchtst_fm_s_pre_flight_says_it_needs_python_below_3_14(version, monkeypatch):
    """granite-tsfm installs on Python < 3.14 only, so on 3.14 and later `[patchtst-fm]` installs nothing: the pre-flight says
    which Python it needs instead of naming the extra; below 3.14 it names the extra."""
    import epftoolbox2.models.patchtst_fm as module

    monkeypatch.setattr(module.sys, "version_info", version)
    monkeypatch.setattr(_cards, "has_extra", lambda name: name != "patchtst-fm")
    (problem,) = [p for p in PatchTSTFMModel()._preflight({}) if p.kind == "extra"]
    if version >= (3, 14):
        assert "needs Python 3.13 or older" in problem.message and "granite-tsfm" in problem.message, problem.message
        assert "pip install 'epftoolbox2[patchtst-fm]'" in problem.message  # in a Python 3.11-3.13 environment
        assert "3.11-3.13" in problem.message
    else:
        assert problem.message.endswith("pip install 'epftoolbox2[patchtst-fm]'"), problem.message
    assert problem.extra == "patchtst-fm"


@pytest.mark.filterwarnings("ignore:.*not the reference weights")
@pytest.mark.parametrize("family", [PatchTSTFMModel, TimestralModel, TimesFMModel], ids=lambda cls: cls.__name__)
def test_the_pre_flight_names_the_family_s_extra(hourly, monkeypatch, tmp_path, family):
    """With the extra's modules hidden, a run refuses before any model's run starts, naming the family's extra (weights given
    as a directory of the card's file names, so the extra is the one problem)."""
    for name in family._card.weights.files:
        (tmp_path / name).write_bytes(b"not weights")
    params = {} if family is PatchTSTFMModel else {"known_ahead": ["load_forecast"]}
    suite.check_preflight_without_extra(
        lambda name: family(name=name, context=CONTEXT, weights=str(tmp_path), **params),
        monkeypatch,
        hourly,
        ("2023-06-12", "2023-06-13"),
    )


# ---------------------------------------------------------------------------------------------------------------------------
# The adapter suite with the numpy fakes
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("family", list(MAKERS))
def test_verify_is_ok_for_each_family(hourly, family):
    days = ["2023-03-25", "2023-03-26", "2023-06-14", "2023-10-28", "2023-10-29"]
    suite.check_verify(MAKERS[family], hourly, INFO, days, items=7)
    suite.check_verify(lambda name: MAKERS[family](name, newest_target="published"), hourly, INFO, days[2:3], items=7)


@pytest.mark.parametrize("family", list(MAKERS))
def test_the_adapter_suite(family, hourly, quarter, tmp_path, registered):
    make = MAKERS[family]
    test = ("2023-03-15", "2023-03-28")
    variants = ({}, {"context": 5}, {"batch_size": 3}, {"newest_target": "published"})
    if family == "timestral":
        variants = (*variants, {"size": "28m", "symmetric_averaging": False}, {"clip_nonnegative": True})
    suite.check_yaml_round_trip(make, hourly, tmp_path, test, horizon=2, information_set=INFO_AHEAD)
    suite.check_loop_and_shards(make, hourly, tmp_path / "loop", test, 2, variants, information_set=INFO_AHEAD)
    suite.check_resume(make, hourly, tmp_path / "resume", test, horizon=2, information_set=INFO_AHEAD)
    suite.check_engine(make, hourly, test, horizon=2, information_set=INFO_AHEAD)
    suite.check_dst_counts(make, hourly, "2023-03-26", "2023-10-29", information_set=INFO)
    suite.check_quarter_hours(make, quarter, ("2023-02-20", "2023-02-21"), information_set=INFO)
    suite.check_provenance(make, hourly, tmp_path / "provenance", test, information_set=INFO_AHEAD)


def test_each_family_s_calls_carry_its_inputs_and_options(hourly):
    """The univariate fake's calls hold the target alone; Timestral's and TimesFM-3's the known-ahead and past series in the
    model's order; Timestral's options reach every call, and a forecast is the point level's column of the backend's."""
    test = ("2023-06-12", "2023-06-13")
    patchtst = MAKERS["patchtst"]("P")
    pipeline(patchtst).run(hourly, *test, horizon=1, forecast_only=True)
    assert patchtst._fake.calls and all(c == ({}, [], []) for c in patchtst._fake.calls)
    base = MAKERS["timestral"]("T")
    plain = by_key(pipeline(base).run(hourly, *test, horizon=1, forecast_only=True), "T")
    assert all(
        c == ({"symmetric_averaging": True, "clip_nonnegative": False}, ["load_forecast"], ["load_actual"])
        for c in base._fake.calls
    )
    unaveraged = MAKERS["timestral"]("T", symmetric_averaging=False)
    moved = by_key(pipeline(unaveraged).run(hourly, *test, horizon=1, forecast_only=True), "T")
    assert moved.keys() == plain.keys() and all(moved[k] == pytest.approx(v + 0.5) for k, v in plain.items())
    # Two known-ahead series, in the model's order (not sorted): a second forecast column known as the load forecast is.
    frame = hourly.copy()
    frame["wind_forecast"] = 0.1 * frame["load_forecast"]
    ahead = Availability.published("10:00", 2, vintage="latest_revision")
    info = InformationSet(rules={"load_forecast": ahead, "wind_forecast": ahead})
    timesfm = MAKERS["timesfm"]("F", known_ahead=["wind_forecast", "load_forecast"], past=[])
    pipeline(timesfm, information_set=info).run(frame, *test, horizon=1, forecast_only=True)
    assert timesfm._fake.calls and all(c[1:] == (["load_forecast", "wind_forecast"], []) for c in timesfm._fake.calls)


def test_clip_nonnegative_clamps_only_where_asked(hourly):
    """The adapter's default is off (prices can be negative); on, a forecast below 0 of a context with no negative value is
    clamped at 0, and one of a context with a negative value isn't (the research ran the package's clip on)."""
    frame = hourly.copy()
    frame["price"] = frame["price"].abs() + 1.0
    frame["load_forecast"] = -frame["load_forecast"]  # the fake's forecast: persistence + 0.1 x a large negative series
    test = ("2023-06-12", "2023-06-12")
    off = by_key(pipeline(MAKERS["timestral"]("T")).run(frame, *test, horizon=1, forecast_only=True), "T")
    on = by_key(
        pipeline(MAKERS["timestral"]("T", clip_nonnegative=True)).run(frame, *test, horizon=1, forecast_only=True), "T"
    )
    assert off.keys() == on.keys() and max(off.values()) < 0 and set(on.values()) == {0.0}
    frame.loc["2023-06-05 12:00":"2023-06-05 12:00", "price"] = -5.0  # one negative price in every item's context
    kept = by_key(
        pipeline(MAKERS["timestral"]("T", clip_nonnegative=True)).run(frame, *test, horizon=1, forecast_only=True), "T"
    )
    assert max(kept.values()) < 0


def test_the_attribution_and_the_verdicts_show_in_every_report(hourly, tmp_path, capsys, monkeypatch):
    """Timestral's attribution "2026 - Mistral AI" and its verdict, TimesFM-3's non-commercial licence, PatchTST-FM's verdict by
    its release: in `report.provenance()`, the terminal exporter and the CSV exporter's provenance file."""
    test = ("2023-06-12", "2023-06-13")
    models = [MAKERS["patchtst"]("P"), MAKERS["timestral"]("T"), MAKERS["timesfm"]("F")]
    monkeypatch.setenv("COLUMNS", "400")  # the terminal exporter's table unwrapped
    run = pipeline(*models, information_set=INFO_AHEAD).add_evaluator(MAEEvaluator())
    run.add_exporter(TerminalExporter(show=["summary"])).add_exporter(CsvExporter(str(tmp_path / "forecasts.csv")))
    report = run.run(hourly, *test, horizon=1, save_dir=tmp_path / "out")
    table = report.provenance().set_index("model")
    assert table.loc["T", "attribution"] == "2026 - Mistral AI"
    assert table.loc["T", "test_verdict"] == POSSIBLY_CONTAMINATED
    assert "USD 20 million" in table.loc["T", "revenue_cap"]
    assert table.loc["F", "licence"] == TIMESFM_LICENCE and not table.loc["F", "commercial_use"]
    assert table.loc["F", "test_verdict"] == POSSIBLY_CONTAMINATED  # 2023-06: before its corpus end, 2023-11-30
    assert table.loc["P", "test_verdict"] == POSSIBLY_CONTAMINATED  # released 2026-08-07, no corpus end
    shown = " ".join(capsys.readouterr().out.split())
    assert "2026 - Mistral AI" in shown and TIMESFM_LICENCE in shown
    written = (tmp_path / "forecasts.provenance.csv").read_text()
    assert "2026 - Mistral AI" in written and TIMESFM_LICENCE in written
    manifest = json.loads((tmp_path / "out" / "t.manifest.json").read_text())
    assert manifest["card"]["attribution"] == "2026 - Mistral AI"
    assert manifest["card"]["verdicts"] == {"test": POSSIBLY_CONTAMINATED}


@pytest.mark.parametrize(
    "family, change",
    [
        ("timestral", {"clip_nonnegative": True}),
        ("timestral", {"symmetric_averaging": False}),
        ("timestral", {"size": "28m"}),
        ("timestral", {"past": []}),
        ("timesfm", {"known_ahead": []}),
        ("patchtst", {"context": 6}),
        ("patchtst", {"batch_size": 4}),
    ],
)
def test_a_changed_setting_refuses_the_resume(hourly, tmp_path, family, change):
    """Each setting that changes a family's forecasts is in its manifest's model parameters: resuming results made with
    another value refuses."""
    from epftoolbox2.results.manifest import ManifestMismatchError

    test = ("2023-06-12", "2023-06-13")
    pipeline(MAKERS[family]("M")).run(hourly, *test, horizon=1, forecast_only=True, save_dir=tmp_path)
    with pytest.raises(ManifestMismatchError):
        pipeline(MAKERS[family]("M", **change)).run(hourly, *test, horizon=1, forecast_only=True, save_dir=tmp_path)
