"""Foundation models (step 6i): `FoundationModel`'s run with a deterministic numpy backend, the fine-tune scheduler and its
checkpoint store, and what `T0Model` and `ChronosModel` are before any package of theirs is imported.

`FakeFM` is a `FoundationModel` whose backend is numpy (`PersistenceBackend`): each step ahead is the target at the same step of
the context's last day plus 0.1 x the known-ahead series there (6h's fake), plus a bias, at the levels (0.25, 0.5, 0.75) with
the median in the middle; its fine-tune learns the bias, the mean residual of that persistence over the training windows
and a thousandth of the holdout's (which early stopping reads), plus a small amount per seed, so seeds differ, and asserts the trainer's cut as `_torch.finetune` does. It runs without
torch, in the core CI job. The real packages' adapters with tiny random weights are in `test_t0.py` and `test_chronos.py`.
"""

import contextlib
import dataclasses
import datetime
import json
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import ChronosModel, FoundationModel, T0Model
from epftoolbox2.models._cards import ModelCard
from epftoolbox2.models._fm import Backend, FineTuneOutcome
from epftoolbox2.models._history import HistorySpec, history_batch
from epftoolbox2.models.foundation import FINE_TUNE_DEFAULTS, check_fine_tune
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import ManifestMismatchError
from tests.import_stubs import run_without
from tests.models import adapter_suite as suite

ROOT = Path(__file__).resolve().parents[2]
ZONE = "Europe/Warsaw"
CONTEXT = 7
INFO = InformationSet()  # 11:45 Brussels, the built-in rules: prices out 13:00 on D-1, the load forecast 10:00 on D-1
INFO_AHEAD = InformationSet(rules={"load_forecast": Availability.published("10:00", 2, vintage="latest_revision")})
QUICK = {"holdout_days": 14, "max_steps": 3, "eval_every": 1}


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


def market(freq="1h", start="2023-01-20", end="2023-12-01", seed=0):
    """A local frame: each series a seasonal shape plus noise, so persistence and the fine-tune's bias have work to do."""
    index = pd.date_range(pd.Timestamp(start, tz=ZONE), pd.Timestamp(end, tz=ZONE), freq=freq, inclusive="left")
    rng = np.random.default_rng(seed)
    hours = np.asarray(index.hour + index.minute / 60, dtype=float)
    position = np.arange(len(index), dtype=float)
    price = 80 + 20 * np.sin(hours / 24 * 2 * np.pi) + 0.01 * position + rng.normal(0, 3, len(index))
    load = 1e4 + 500 * np.cos(hours / 24 * 2 * np.pi) + rng.normal(0, 50, len(index))
    actual = load + rng.normal(0, 30, len(index))
    return pd.DataFrame({"price": price, "load_forecast": load, "load_actual": actual}, index=index)


@pytest.fixture(scope="module")
def hourly():
    return market("1h")


@pytest.fixture(scope="module")
def quarter():
    return market("15min", start="2023-01-20", end="2023-03-10")


FAKE_CARD = ModelCard(
    family="fake-foundation",
    extra=None,
    licence="MIT",
    licence_url="https://example.invalid/mit",
    commercial_use=True,
    revenue_cap=None,
    attribution=None,
    weights=None,
    runtime=(),
    corpus="synthetic test series",
    corpus_end=date(2022, 12, 31),
    released=date(2023, 1, 1),
)


class PersistenceBackend(Backend):
    """Persistence of the context's last day plus 0.1 x the known-ahead series plus `bias`, at three levels; `by_batch` adds
    that much times the mean of the call's last context values (a backend that reads the other items of its call)."""

    levels = (0.25, 0.5, 0.75)

    def __init__(self, model):
        self.model = model
        self.bias = 0.0
        self.fine_tunes = []
        self.starts = []  # the bias each fine-tune started from: 0.0, the base weights

    def point(self, target, known, steps):
        periods = self.model._periods_per_day
        width = target.shape[1]
        out = target[:, width - periods + np.arange(steps) % periods]
        for values in known.values():
            out = out + 0.1 * values[:, width:]
        return out

    def predict(self, call, options):
        out = self.point(call.target, call.known, call.steps_ahead) + self.bias + self.model.offset
        if self.model.by_batch:
            out = out + self.model.by_batch * float(call.target[:, -1].mean())
        return np.stack([out - 1.0, out, out + 1.0], axis=-1)

    def fine_tune(self, data, settings, seed):
        if len(data.dates) and data.dates.max() > np.datetime64(data.cut, "D"):
            raise AssertionError("a window after the cut")  # as the trainer asserts
        hold = data.dates > data.dates.max() - np.timedelta64(settings["holdout_days"], "D")
        residuals = {False: [], True: []}
        for item, y, held in zip(data.items, data.targets, hold):
            known = {name: item.known[name][None] for name in data.known_ahead}
            fitted = self.point(item.target[None], known, item.steps_ahead)[0]
            finite = np.isfinite(y)
            residuals[bool(held)].extend((y[finite] - fitted[finite]).tolist())
        # From the state it is in, as a fine-tune of weights starts from them: the training windows' mean residual, and a
        # little of the holdout's, as early stopping reads it to choose a step.
        self.starts.append(self.bias)
        self.bias += float(np.nanmean(residuals[False])) + 1e-3 * float(np.nanmean(residuals[True]))
        self.bias += (int(seed) % 101) * 1e-3
        self.fine_tunes.append((data.cut, len(data), int(seed)))
        return FineTuneOutcome(settings["max_steps"], data.dates.max().astype(date), {"windows": len(data)})

    def save_state(self, path):
        Path(path).write_text(json.dumps({"bias": self.bias}))

    def load_state(self, path):
        self.bias = json.loads(Path(path).read_text())["bias"]

    def reset(self):
        self.bias = 0.0


class FakeFM(FoundationModel):
    """A foundation model of the numpy backend: no torch, no weights, a card of its own."""

    _card = FAKE_CARD
    _can_fine_tune = True
    _uses_torch = False
    _default_name = "FakeFM"

    def __init__(
        self,
        context=CONTEXT,
        known_ahead=("load_forecast",),
        past=(),
        batch_size=8,
        device="cpu",
        weights=None,
        fine_tune_every=None,
        fine_tune=None,
        seeds=1,
        random_state=0,
        newest_target="d-1",
        name="FakeFM",
        by_batch=0.0,
        offset=0.0,
    ):
        self.by_batch = by_batch
        self.offset = offset
        self._fake = None
        super().__init__(
            context=context,
            known_ahead=known_ahead,
            past=past,
            batch_size=batch_size,
            device=device,
            weights=weights,
            fine_tune_every=fine_tune_every,
            fine_tune=fine_tune,
            seeds=seeds,
            random_state=random_state,
            newest_target=newest_target,
            name=name,
        )

    @property
    def _model_kwargs(self):
        return {**super()._model_kwargs, "by_batch": self.by_batch, "offset": self.offset}

    def _load_backend(self, resolved, device, options):
        return PersistenceBackend(self)

    def _backend(self, device):  # one backend per model (the fake reads the model's own settings), not one per process
        if self._fake is None:
            self._fake = self._load_backend(None, device, self._backend_options())
        return self._fake


class Univariate(FakeFM):
    _takes_known = False
    _takes_past = False


@pytest.fixture
def registered(monkeypatch):
    from epftoolbox2.pipelines.model_pipeline import COMPONENT_REGISTRY

    monkeypatch.setitem(COMPONENT_REGISTRY["models"], "FakeFM", __name__)


def make(name, **params):
    return FakeFM(name=name, **params)


def tuned(name, **params):
    return FakeFM(name=name, **{"fine_tune_every": "7D", "fine_tune": QUICK, **params})


def pipeline(*models, information_set=INFO, **settings):
    out = ModelPipeline(information_set=information_set, **settings)
    for model in models:
        out.add_model(model)
    return out


def run(model, frame, test, horizon=1, information_set=INFO, **kwargs):
    return pipeline(model, information_set=information_set).run(
        frame, *test, horizon=horizon, forecast_only=True, **kwargs
    )


def by_key(report, name="FakeFM"):
    return {(r["run_date"], r["hour"], r["horizon"]): r["prediction"] for r in report.refs[name].iter_records()}


# ---------------------------------------------------------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------------------------------------------------------


def test_parameters_are_checked_and_the_derived_ones_are_no_saved_parameter():
    model = FakeFM()
    assert (model.refit, model.training_window) == ("1D", CONTEXT)
    model.fine_tune_every = "1MS"
    assert (model.refit, model.training_window) == ("1MS", "expanding")
    model.fine_tune = {"window": 90}
    assert model.training_window == CONTEXT + 90 and model.fine_tune["lr"] == FINE_TUNE_DEFAULTS["lr"]
    hidden = {"predictors", "training_window", "scaler", "refit", "binary_mask", "normalise_by", "bounds", "zero_when"}
    assert not hidden & {key for key in vars(model) if not key.startswith("_")}
    for bad in (
        dict(context=0),
        dict(known_ahead="load_forecast"),
        dict(batch_size=0),
        dict(device="gpu"),
        dict(fine_tune_every="auto"),
        dict(fine_tune_every="weekly"),
        dict(fine_tune={"steps": 3}),
        dict(fine_tune={"grad_clip": 0}),
        dict(fine_tune={"window": 0}),
        dict(seeds=0),
    ):
        with pytest.raises(ValueError):
            FakeFM(**bad)
    with pytest.raises(ValueError, match="univariate"):
        Univariate(known_ahead=["load_forecast"])
    Univariate(known_ahead=())
    with pytest.raises(ValueError, match="doesn't fine-tune"):
        ChronosModel(known_ahead=["load_forecast"]).fine_tune_every = "12MS"
    assert check_fine_tune({"max_steps": 2}) == {**FINE_TUNE_DEFAULTS, "max_steps": 2}
    assert check_fine_tune(None) is None


def test_t0_and_chronos_build_with_the_spec_s_defaults_and_save_only_their_parameters(tmp_path):
    t0 = T0Model(known_ahead=["load_forecast"])
    assert (t0.context, t0.fine_tune_every, t0.seeds, t0.fine_tune, t0.name) == (
        364,
        "12MS",
        3,
        FINE_TUNE_DEFAULTS,
        "t0",
    )
    chronos = ChronosModel(context=182, known_ahead=["load_forecast"])
    assert chronos.fine_tune_every is None and chronos.name == "chronos2" and chronos.refit == "1D"
    path = tmp_path / "models.yaml"
    pipeline(t0, chronos).save(path)
    loaded = ModelPipeline.load(path)
    assert [type(m) for m in loaded.models] == [T0Model, ChronosModel]
    assert [m._model_kwargs for m in loaded.models] == [t0._model_kwargs, chronos._model_kwargs]
    saved = path.read_text()
    assert "fine_tune_every" not in saved.split("ChronosModel")[1] and "refit" not in saved


def test_importing_the_package_or_building_these_models_imports_no_model_package(tmp_path):
    """Dawid's lean condition: `import epftoolbox2`, a data pipeline of calendar data (fetched) and building T0Model and
    ChronosModel try to import none of torch, t0, chronos, huggingface_hub, transformers or safetensors: each is a stub on the
    path that logs an attempt and raises, so a guarded `try: import t0` counts too (`tests.import_stubs`)."""
    code = (
        "import epftoolbox2\n"
        "from epftoolbox2.data.sources import CalendarSource\n"
        "from epftoolbox2.pipelines import DataPipeline, ModelPipeline\n"
        "from epftoolbox2.models import T0Model, ChronosModel, FoundationModel\n"
        "frame = DataPipeline(sources=[CalendarSource(country='PL')]).run('2024-01-01', '2024-01-08')\n"
        "assert len(frame), 'no calendar data'\n"
        "models = [T0Model(known_ahead=['load_forecast']), ChronosModel(context=182)]\n"
        "ModelPipeline().add_model(models[0]).add_model(models[1]).to_dict()\n"
        "[model.card for model in models]\n"
    )
    heavy = ("torch", "t0", "chronos", "huggingface_hub", "transformers", "safetensors")
    result, attempted = run_without(code, heavy, tmp_path)
    assert result.returncode == 0, result.stderr
    assert attempted == [], f"tried to import {attempted}"
    # The guard sees what it guards: the same code with an import of t0 caught by an except is logged.
    result, attempted = run_without("try:\n    import t0\nexcept ImportError:\n    pass\n", heavy, tmp_path / "probe")
    assert result.returncode == 0 and attempted == ["t0"], (result.stderr, attempted)


# ---------------------------------------------------------------------------------------------------------------------------
# Zero-shot
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("newest", ["d-1", "published"])
def test_zero_shot_forecasts_are_the_backend_s_median_of_the_accessor_s_items(hourly, newest):
    """Each forecast is the 0.5 level of the backend's output for its item, which the history accessor reads for its issue day
    alone; its record's `refit_anchor` is its run date (no fit), its coefficients empty."""
    model = make("FakeFM", newest_target=newest)
    days = ("2023-03-24", "2023-03-29")
    report = run(model, hourly, days, horizon=2, information_set=INFO_AHEAD)
    records = report.refs["FakeFM"]._results
    counts = report.forecast_counts().iloc[0]
    assert counts["produced"] == counts["planned"] > 0
    spec = HistorySpec(CONTEXT, ("load_forecast",), (), "price", True, newest)
    backend = PersistenceBackend(model)
    for record in records:
        assert record["refit_anchor"] == record["run_date"] and record["coefficients"] == []
        (item,) = history_batch(spec, hourly, [record["run_date"]], INFO_AHEAD, "1h", record["horizon"]).items
        expected = backend.point(
            item.target[None], {"load_forecast": item.known["load_forecast"][None]}, item.steps_ahead
        )
        step = list(item.slot_periods).index(record["hour"])
        assert record["prediction"] == expected[0, step]


def test_a_zero_shot_call_holds_its_calendar_block_whatever_the_run_has(hourly):
    """A call holds the issue days of a fixed block of `batch_size` days that the frame has: the same forecast alone, in a
    longer run and in a run that starts elsewhere; the co-batch hook names the other days of the block."""
    model = make("FakeFM", batch_size=8)
    assert len(model._co_batch_items("2023-06-14")) == 7
    long = by_key(run(make("FakeFM"), hourly, ("2023-06-01", "2023-06-30")))
    for day in ("2023-06-03", "2023-06-14", "2023-06-29"):
        alone = by_key(run(make("FakeFM"), hourly, (day, day)))
        assert alone and all(long[key] == value for key, value in alone.items())


def test_a_call_s_items_are_its_calendar_block_s_whatever_the_run_covers(hourly):
    """A backend that reads its call's other items (`by_batch`) gives the same forecast alone and in a run of a month: the
    call holds the same days either way (the block of `batch_size` days that the frame has), not the run's days."""
    long = by_key(run(make("FakeFM", by_batch=1e-3), hourly, ("2023-06-01", "2023-06-30")))
    for day in ("2023-06-03", "2023-06-14"):
        alone = by_key(run(make("FakeFM", by_batch=1e-3), hourly, (day, day)))
        assert alone and all(long[key] == value for key, value in alone.items())


def test_a_backend_that_reads_its_call_is_caught_by_verify(hourly):
    with pytest.raises(AssertionError):
        suite.check_verify(lambda name: make(name, by_batch=1e-3), hourly, INFO, ["2023-06-14"])


@pytest.mark.filterwarnings("ignore:overflow")
def test_verify_is_ok_for_zero_shot(hourly):
    days = ["2023-03-25", "2023-03-26", "2023-06-14", "2023-10-28", "2023-10-29"]
    suite.check_verify(make, hourly, INFO, days, items=7)
    suite.check_verify(lambda name: make(name, newest_target="published"), hourly, INFO, days[2:3], items=7)


def test_the_adapter_suite_zero_shot(hourly, quarter, tmp_path, monkeypatch, registered):
    test = ("2023-03-15", "2023-03-28")
    suite.check_yaml_round_trip(make, hourly, tmp_path, test, horizon=2, information_set=INFO_AHEAD)
    variants = ({}, {"context": 5}, {"batch_size": 3, "offset": 1.0}, {"newest_target": "published"})
    suite.check_loop_and_shards(make, hourly, tmp_path / "loop", test, 2, variants, information_set=INFO_AHEAD)
    suite.check_resume(make, hourly, tmp_path / "resume", test, horizon=2, information_set=INFO_AHEAD)
    suite.check_engine(make, hourly, test, horizon=2, information_set=INFO_AHEAD)
    suite.check_dst_counts(make, hourly, "2023-03-26", "2023-10-29", information_set=INFO)
    suite.check_quarter_hours(make, quarter, ("2023-02-20", "2023-02-21"), information_set=INFO)
    suite.check_provenance(make, hourly, tmp_path / "provenance", test, information_set=INFO_AHEAD)


@pytest.mark.filterwarnings("ignore:.*not the reference weights")
@pytest.mark.parametrize("family", [T0Model, ChronosModel])
def test_the_pre_flight_names_the_family_s_extra(hourly, monkeypatch, tmp_path, family):
    """With the extra's modules hidden, a run refuses before any model's run starts, naming `[t0]` or `[chronos]` (weights
    given as a directory of the card's file names, so the extra is the one problem)."""
    for name in family._card.weights.files:
        (tmp_path / name).write_bytes(b"not weights")
    suite.check_preflight_without_extra(
        lambda name: family(name=name, context=CONTEXT, known_ahead=["load_forecast"], weights=str(tmp_path)),
        monkeypatch,
        hourly,
        ("2023-06-12", "2023-06-13"),
    )


def test_the_cards_say_what_the_spec_says():
    t0, chronos = T0Model._card, ChronosModel._card
    assert (t0.family, t0.extra, t0.licence, t0.commercial_use, t0.corpus_end) == (
        "t0-alpha",
        "t0",
        "Apache-2.0",
        True,
        date(2021, 12, 31),
    )
    assert "tfc-t0==0.5.0" in t0.runtime and t0.weights.repo == "theforecastingcompany/t0-alpha"
    assert set(t0.weights.files) == {"config.json", "model.safetensors"}
    assert (chronos.family, chronos.extra, chronos.corpus_end, chronos.released) == (
        "chronos-2",
        "chronos",
        None,
        date(2025, 10, 30),
    )
    from epftoolbox2.models._cards import contamination_verdict

    assert contamination_verdict(t0, "2025-10-01") == "clean"
    assert contamination_verdict(chronos, "2025-10-01") == "possibly contaminated"  # released after the test start
    assert contamination_verdict(chronos, "2026-01-01") == "clean (released before)"


# ---------------------------------------------------------------------------------------------------------------------------
# Fine-tunes
# ---------------------------------------------------------------------------------------------------------------------------


def test_a_fine_tune_learns_on_windows_up_to_the_anchor_s_cut(hourly):
    """At each anchor of "7D" (Mondays) the fake fine-tunes on the target days up to D-1 ("d-1") or D ("published"), each read
    as a forecast issued a horizon before it; the forecasts of the block hold the learned bias, and `refit_anchor` is the
    anchor."""
    for newest, last in (("d-1", date(2023, 6, 11)), ("published", date(2023, 6, 12))):
        model = tuned("FakeFM", newest_target=newest)
        report = run(model, hourly, ("2023-06-12", "2023-06-14"))
        (cut, windows, _), *_ = model._fake.fine_tunes
        assert cut == last and windows > 100 and model._fine_tunes == 1
        anchors = {r["refit_anchor"] for r in report.refs["FakeFM"]._results}
        assert anchors == {"2023-06-12"}
        zero = by_key(run(make("FakeFM", newest_target=newest), hourly, ("2023-06-12", "2023-06-14")))
        bias = model._fake.bias
        assert bias == 0.0  # reset after the block
        moved = {key: value - zero[key] for key, value in by_key(report).items()}
        assert len(set(np.round(list(moved.values()), 9))) == 1 and abs(next(iter(moved.values()))) > 0


def test_seeds_are_averaged_and_each_seed_is_its_own(hourly):
    one = by_key(run(tuned("FakeFM", seeds=1), hourly, ("2023-06-12", "2023-06-12")))
    model = tuned("FakeFM", seeds=3)
    three = by_key(run(model, hourly, ("2023-06-12", "2023-06-12")))
    seeds = [seed for _, _, seed in model._fake.fine_tunes]
    assert len(set(seeds)) == 3 and model._fine_tunes == 3
    assert model._fake.starts == [0.0, 0.0, 0.0]  # each from the base weights, never the previous seed's
    assert one.keys() == three.keys() and one != three


def test_the_model_asserts_no_window_after_the_cut(hourly, monkeypatch):
    """With the anchor's cut taken a day later than its instant allows, the windows would read a target after the cut: the
    model raises before the trainer runs."""
    real = FoundationModel._anchor_cut

    def later(self, anchor, frame, spec):
        cut, cut_date = real(self, anchor, frame, spec)
        return cut, cut_date + datetime.timedelta(days=1)

    monkeypatch.setattr(FoundationModel, "_anchor_cut", later)
    with pytest.raises(AssertionError, match="after its cut"):
        run(tuned("FakeFM"), hourly, ("2023-06-12", "2023-06-12"))


@pytest.mark.filterwarnings("ignore:overflow")
def test_verify_with_fine_tuning_on_is_ok_and_catches_a_cut_one_day_late(hourly, monkeypatch):
    """`verify` of the fine-tuned fake on an anchor day and inside its block: ok with the control, the rerun and the co-batch
    check. With the cut moved one day later (instant and date), the anchor's fine-tune reads the target of D, which the probe
    perturbs: a leak."""
    days = ["2023-06-12", "2023-06-14", "2023-06-18"]
    suite.check_verify(tuned, hourly, INFO, days, items=6)
    suite.check_verify(lambda name: tuned(name, newest_target="published"), hourly, INFO, days[:1], items=6)
    real = FoundationModel._anchor_cut

    def late(self, anchor, frame, spec):
        cut, cut_date = real(self, anchor, frame, spec)
        return cut + pd.Timedelta(days=1), cut_date + datetime.timedelta(days=1)

    monkeypatch.setattr(FoundationModel, "_anchor_cut", late)
    with pytest.raises(AssertionError):
        suite.check_verify(tuned, hourly, INFO, days[:1])


def test_stored_fine_tunes_are_reused_and_a_deleted_one_refits(hourly, tmp_path):
    """A run into a save_dir stores each fine-tune and its sidecar in `<name>.fits/`; a resume after 40 % of the records were
    deleted loads them (no fine-tune runs) and writes the fresh run's records; a deleted checkpoint is fine-tuned again (with a
    warning); a checkpoint whose bytes changed raises."""
    test = ("2023-06-05", "2023-06-25")
    fresh, resumed = tmp_path / "fresh", tmp_path / "resumed"
    run(tuned("FakeFM", seeds=2), hourly, test, save_dir=fresh)
    first = tuned("FakeFM", seeds=2)
    run(first, hourly, test, save_dir=resumed)
    fits = sorted((resumed / "fakefm.fits").glob("*.ckpt"))
    assert first._fine_tunes == 6 and len(fits) == 6  # three Mondays, two seeds
    sidecar = json.loads(sorted((resumed / "fakefm.fits").glob("*.json"))[0].read_text())
    assert {
        "anchor",
        "horizon",
        "seed",
        "file",
        "sha256",
        "data_end",
        "best_step",
        "windows",
        "data_digest",
        "key",
    } <= set(sidecar)
    assert sidecar["file"] == fits[0].name and fits[0].name.endswith(f".{sidecar['sha256'][:16]}.ckpt")
    path = suite.results(resumed, "FakeFM")
    kept = [line for i, line in enumerate(suite.lines(path)) if i % 5 not in (1, 3)]
    path.write_text("".join(f"{line}\n" for line in kept))
    again = tuned("FakeFM", seeds=2)
    run(again, hourly, test, save_dir=resumed)
    assert again._fine_tunes == 0 and again._checkpoints_loaded == 6
    assert suite.lines(path) == suite.lines(suite.results(fresh, "FakeFM"))
    fits[0].unlink()
    path.write_text("".join(f"{line}\n" for line in kept))
    refit = tuned("FakeFM", seeds=2)
    with pytest.warns(UserWarning, match="is gone"):
        run(refit, hourly, test, save_dir=resumed)
    assert refit._fine_tunes == 1 and fits[0].is_file()
    assert suite.lines(path) == suite.lines(suite.results(fresh, "FakeFM"))  # on the CPU the refit is the same bits
    fits[1].write_text(json.dumps({"bias": 99.0}))
    path.write_text("".join(f"{line}\n" for line in kept))
    other = tuned("FakeFM", seeds=2)
    with pytest.warns(UserWarning, match="changed after it was stored"):
        run(other, hourly, test, save_dir=resumed)
    assert other._fine_tunes == 1 and json.loads(fits[1].read_text())["bias"] != 99.0  # fine-tuned again, not raised
    assert suite.lines(path) == suite.lines(suite.results(fresh, "FakeFM"))


def test_a_changed_fine_tune_refuses_the_resume(hourly, tmp_path):
    test = ("2023-06-05", "2023-06-11")
    run(tuned("FakeFM"), hourly, test, save_dir=tmp_path)
    path = suite.results(tmp_path, "FakeFM")
    lines = suite.lines(path)
    path.write_text("".join(f"{line}\n" for line in lines[::2]))
    with pytest.raises(ManifestMismatchError, match="fine_tune"):
        run(tuned("FakeFM", fine_tune={**QUICK, "lr": 1e-4}), hourly, test, save_dir=tmp_path)
    with pytest.raises(ManifestMismatchError, match="seeds"):
        run(tuned("FakeFM", seeds=2), hourly, test, save_dir=tmp_path)
    # Let in anyway, another fine-tune's checkpoint is not this one's: it fine-tunes again, into a file of its own.
    other = tuned("FakeFM", fine_tune={**QUICK, "lr": 1e-4})
    with pytest.warns(UserWarning, match="is stored under another key"):
        pipeline(other).run(
            hourly, *test, horizon=1, save_dir=tmp_path, forecast_only=True, allow_manifest_mismatch=True
        )
    assert other._fine_tunes == 1 and len(list((tmp_path / "fakefm.fits").glob("*.ckpt"))) == 2


def test_too_few_windows_skip_the_unit(hourly):
    """An anchor with no window before its holdout skips its forecasts as too few rows, counted."""
    short = hourly.loc[:"2023-02-15"]
    with pytest.raises(RuntimeError, match="too_few_rows"):
        run(tuned("FakeFM", fine_tune={**QUICK, "holdout_days": 61}), short, ("2023-02-06", "2023-02-07"))


def test_the_adapter_suite_fine_tuned(hourly, tmp_path, registered):
    test = ("2023-06-08", "2023-06-20")
    suite.check_yaml_round_trip(tuned, hourly, tmp_path, test, information_set=INFO)
    variants = ({}, {"seeds": 2}, {"fine_tune": {**QUICK, "window": 30}})
    suite.check_loop_and_shards(tuned, hourly, tmp_path / "loop", test, 1, variants, information_set=INFO)
    suite.check_resume(tuned, hourly, tmp_path / "resume", test, information_set=INFO)
    suite.check_engine(tuned, hourly, test, information_set=INFO)
    suite.check_dst_counts(tuned, hourly, "2023-03-26", "2023-10-29", information_set=INFO)


def test_a_window_of_days_trains_on_the_newest_days_only(hourly):
    model = tuned("FakeFM", fine_tune={**QUICK, "window": 30})
    run(model, hourly, ("2023-06-12", "2023-06-12"))
    (cut, windows, _), *_ = model._fake.fine_tunes
    assert cut == date(2023, 6, 11) and windows == 30


def test_the_fine_tune_windows_are_read_as_forecasts_issued_a_horizon_before(hourly):
    """At horizon 2 a window of target day t is the item issued on t - 2: its context and known-ahead values as the accessor
    reads them on that day, and its targets the actual prices of t on the target day's steps (NaN on the steps of t - 1)."""
    model = tuned("FakeFM")
    pipeline(model, information_set=INFO_AHEAD).run(hourly, "2023-06-12", "2023-06-12", horizon=2, forecast_only=True)
    spec = model._history_spec("price")
    unit = dataclasses.make_dataclass("Unit", ["horizon", "anchor"])(2, date(2023, 6, 12))
    data, digest, cut, counts = model._fine_tune_data(unit, hourly, spec)
    assert counts["kept"] == len(data) and set(counts["dropped"]) == {"a context starting before the frame"}
    assert data.cut == date(2023, 6, 11) and len(data) > 100 and len(digest) == 64
    # Every window's series are whole: the first is the first target day whose context lies in the frame.
    assert all(np.isfinite(i.target).all() and np.isfinite(i.known["load_forecast"]).all() for i in data.items)
    assert hourly.index[0] <= data.items[0].first_step < hourly.index[0] + pd.Timedelta(days=1)
    item, y = data.items[-1], data.targets[-1]
    assert item.issue_day == date(2023, 6, 9) and item.target_day == date(2023, 6, 11)
    (again,) = history_batch(spec, hourly, [item.issue_day], INFO_AHEAD, "1h", 2).items
    assert np.array_equal(again.target, item.target) and np.array_equal(
        again.known["load_forecast"], item.known["load_forecast"]
    )
    on_target = np.asarray(item.slot_periods) >= 0
    assert np.isnan(y[~on_target]).all()
    actual = hourly.loc["2023-06-11", "price"].to_numpy()
    assert np.array_equal(y[on_target], actual)


# ---------------------------------------------------------------------------------------------------------------------------
# The review: seeds and keys, missing values, CUDA and pools, the store, options, the context a family reads, the TUI
# ---------------------------------------------------------------------------------------------------------------------------


def test_another_random_state_fine_tunes_again_into_a_file_of_its_own(hourly, tmp_path):
    """`random_state` decides every seed, so it is in the key: a run with another one into the same save_dir (results and
    manifest deleted) fine-tunes again and keeps the other's checkpoint; `find` also compares the seed a sidecar records."""
    test = ("2023-06-05", "2023-06-11")
    run(tuned("FakeFM", random_state=0), hourly, test, save_dir=tmp_path)
    for name in ("fakefm.jsonl", "fakefm.manifest.json"):
        (tmp_path / name).unlink()
    other = tuned("FakeFM", random_state=1)
    with pytest.warns(UserWarning, match=r"is stored under another key \([0-9a-f]{16}"):
        run(other, hourly, test, save_dir=tmp_path)
    assert other._fine_tunes == 1 and other._checkpoints_loaded == 0
    assert len(list((tmp_path / "fakefm.fits").glob("*.ckpt"))) == 2
    from epftoolbox2.models.foundation import _Checkpoints

    store = _Checkpoints(tmp_path / "fakefm.fits")
    record = json.loads(next((tmp_path / "fakefm.fits").glob("*.json")).read_text())
    unit = dataclasses.make_dataclass("Unit", ["horizon", "anchor"])(1, date.fromisoformat(record["anchor"]))
    with pytest.warns(UserWarning, match="another key or seed"):
        assert store.find(unit, 0, record["key"], record["data_digest"], record["seed"] + 1) is None
    assert store.find(unit, 0, record["key"], record["data_digest"], record["seed"]) is not None


def test_a_missing_value_drops_only_the_windows_the_backend_cant_take_and_says_so(hourly, tmp_path):
    """One missing hour of the load forecast: a window whose steps ahead need it is dropped (2 of them, the item a forecast
    couldn't be made from), one whose context holds it is kept (the backend masks it); the run warns with the counts, and the
    manifest's `fine_tunes` records them per anchor, with the key its checkpoints are stored under."""
    holed = hourly.copy()
    holed.loc[holed.index[holed.index == pd.Timestamp("2023-05-20 03:00", tz=ZONE)], "load_forecast"] = np.nan
    whole_model, holed_model = tuned("FakeFM"), tuned("FakeFM")
    run(whole_model, hourly, ("2023-06-12", "2023-06-12"))
    with pytest.warns(
        UserWarning,
        match=r"keeps (\d+) of (\d+) windows, (\d+) of them with gaps the backend masks: dropped 2 for missing",
    ):
        run(holed_model, holed, ("2023-06-12", "2023-06-12"), save_dir=tmp_path)
    (_, whole, _), (_, kept, _) = whole_model._fake.fine_tunes[0], holed_model._fake.fine_tunes[0]
    assert kept == whole - 2
    entry = json.loads((tmp_path / "fakefm.manifest.json").read_text())["fine_tunes"]
    counts = entry["windows"]["2023-06-12_h1"]
    assert counts["kept"] == kept and counts["dropped"]["missing values of load_forecast"] == 2
    assert counts["with_gaps"] >= CONTEXT and entry["fine_tuned"] == 1 and entry["loaded"] == 0
    assert entry["key"] == json.loads(next((tmp_path / "fakefm.fits").glob("*.json")).read_text())["key"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a whole frame drops only windows that start before it: no warning
        run(tuned("FakeFM"), hourly, ("2023-06-12", "2023-06-12"))


class CapturePool:
    """A stand-in for the worker pool that runs its jobs here and records the start method it was given (`contexts`)."""

    contexts: list = []

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        from epftoolbox2.models import _worker

        CapturePool.contexts.append(mp_context.get_start_method())
        _worker._worker_init(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, task, *args):
        from concurrent.futures import Future

        future = Future()
        future.set_result(task(*args))
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def an_ols_run(frame, save_dir=None):
    from epftoolbox2.models import OLSModel

    return (
        ModelPipeline()
        .add_model(OLSModel(["load_forecast"], 28, name="OLS"))
        .run(frame, "2023-06-12", "2023-06-12", horizon=1, forecast_only=True, save_dir=save_dir)
    )


def test_after_cuda_starts_in_this_process_every_pool_spawns(hourly, monkeypatch, tmp_path):
    """The safety net: once `torch.cuda.is_initialized()` in this process (a stand-in here: GPU work that ran outside the
    library's host process) every pool of the library spawns its workers, as on Windows, warns once per process naming CUDA,
    and the run's manifest records `pool_start`; before, on Linux, they fork and the manifest says nothing."""
    import types
    from pathlib import Path as _Path

    from epftoolbox2.models import _worker

    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", CapturePool)
    monkeypatch.setattr(CapturePool, "contexts", [])
    monkeypatch.setattr(_worker, "_CUDA_WARNED", False)
    an_ols_run(hourly, tmp_path / "before")
    assert "pool_start" not in json.loads((tmp_path / "before" / "ols.manifest.json").read_text())
    initialised = types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", initialised)
    with pytest.warns(UserWarning, match="CUDA is initialised in this process"):
        an_ols_run(hourly, tmp_path / "after")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # once per process
        an_ols_run(hourly)
    assert CapturePool.contexts == (["spawn"] * 3 if _worker._POOLS_SPAWN else ["fork", "spawn", "spawn"])
    manifest = json.loads((tmp_path / "after" / "ols.manifest.json").read_text())
    assert manifest["pool_start"] == _worker.POOL_SPAWN_FOR_CUDA
    package = _Path(_worker.__file__).parents[1]
    forked = [p for p in package.rglob("*.py") if 'get_context("fork"' in p.read_text()]
    assert forked == [], f"a pool that forks whatever CUDA did: {forked}"


def test_a_crash_between_a_checkpoint_and_its_sidecar_leaves_a_whole_pair(hourly, tmp_path, monkeypatch):
    """A rerun on changed data stopped after its new checkpoint is written and before its sidecar: the old pair is whole, a
    rerun fine-tunes once more (the old windows aren't these) with a warning, keeps one checkpoint per stem and no leftover,
    and its records are a fresh run's on that data."""
    from epftoolbox2.models import foundation

    test = ("2023-06-12", "2023-06-12")
    run(tuned("FakeFM"), hourly, test, save_dir=tmp_path)
    changed = hourly.copy()
    changed.loc[changed.index[(changed.index >= "2023-05-10") & (changed.index < "2023-05-11")], "load_forecast"] += (
        100.0
    )

    def clear():
        for path in tmp_path.iterdir():
            if path.is_file():
                path.unlink()

    clear()

    def killed():
        raise KeyboardInterrupt("the time limit")

    created_at = foundation._created_at
    monkeypatch.setattr(foundation, "_created_at", killed)
    with pytest.raises(KeyboardInterrupt), pytest.warns(UserWarning, match="other windows"):
        run(tuned("FakeFM"), changed, test, save_dir=tmp_path)
    monkeypatch.setattr(foundation, "_created_at", created_at)
    fits = tmp_path / "fakefm.fits"
    (sidecar,) = fits.glob("*.json")
    record = json.loads(sidecar.read_text())
    assert (fits / record["file"]).is_file() and len(list(fits.glob("*.ckpt"))) == 2  # the old pair, the new file
    clear()
    again = tuned("FakeFM")
    with pytest.warns(UserWarning, match="other windows"):
        run(again, changed, test, save_dir=tmp_path)
    assert again._fine_tunes == 1 and len(list(fits.glob("*.ckpt"))) == 1 and not list(fits.glob("*.partial"))
    fresh = tmp_path / "fresh"
    run(tuned("FakeFM"), changed, test, save_dir=fresh)
    assert suite.lines(suite.results(tmp_path, "FakeFM")) == suite.lines(suite.results(fresh, "FakeFM"))
    (fits / "leftover.ckpt.partial").write_text("half")  # a crash's leftover: gone when a run next opens the store
    clear()
    run(tuned("FakeFM"), changed, test, save_dir=tmp_path)
    assert not list(fits.glob("*.partial"))


def test_prune_keeps_the_last_run_s_checkpoints_and_removes_the_rest(hourly, tmp_path):
    """Two runs of other fine-tune settings into one save_dir (the second let in by `allow_manifest_mismatch`): `prune_fine_tunes`
    keeps the pairs of the key the manifest records (the last run's) and removes the other key's, orphans and leftovers;
    without a recorded key it refuses."""
    from epftoolbox2.models.foundation import prune_fine_tunes

    test = ("2023-06-05", "2023-06-11")
    with pytest.raises(ValueError, match="records no key"):
        prune_fine_tunes(tmp_path, "FakeFM")
    run(tuned("FakeFM"), hourly, test, save_dir=tmp_path)
    path = suite.results(tmp_path, "FakeFM")
    path.write_text("".join(f"{line}\n" for line in suite.lines(path)[::2]))  # half gone: the second run has work
    second = tuned("FakeFM", fine_tune={**QUICK, "lr": 1e-4})
    with pytest.warns(UserWarning, match="is stored under another key"):
        pipeline(second).run(
            hourly, *test, horizon=1, save_dir=tmp_path, forecast_only=True, allow_manifest_mismatch=True
        )
    fits = tmp_path / "fakefm.fits"
    assert len(list(fits.glob("*.ckpt"))) == 2
    (fits / "orphan.0123456789abcdef.ckpt").write_text("{}")
    removed = prune_fine_tunes(tmp_path, "FakeFM")
    assert len(removed) == 3 and len(list(fits.glob("*.ckpt"))) == 1 == len(list(fits.glob("*.json")))
    key = json.loads((tmp_path / "fakefm.manifest.json").read_text())["fine_tunes"]["key"]
    assert json.loads(next(fits.glob("*.json")).read_text())["key"] == key
    assert prune_fine_tunes(tmp_path, "FakeFM") == []


class WithOptions(FakeFM):
    """A family whose backend is built with a setting (`scale`, a load option) and called with another (`shift`): two models
    share a loaded backend only when their load options are equal, and each call gets the model's own call options."""

    def __init__(self, scale=1.0, shift=0.0, **kwargs):
        self.scale, self.shift = scale, shift
        super().__init__(**kwargs)

    def _backend(self, device):  # the base's process-wide cache, unlike the fake's one per model
        return FoundationModel._backend(self, device)

    def _load_backend(self, resolved, device, options):
        backend = Scaled(self)
        backend.scale = options["scale"]
        return backend

    def _backend_options(self):
        return {"scale": self.scale}

    def _predict_options(self):
        return {"shift": self.shift}


class Scaled(PersistenceBackend):
    def predict(self, call, options):
        return super().predict(call, options) * self.scale + options["shift"]


def test_backend_options_are_part_of_the_key_and_calls_get_the_model_s_options(hourly):
    from epftoolbox2.models import foundation

    test = ("2023-06-12", "2023-06-12")
    one, two, shifted = WithOptions(scale=1.0), WithOptions(scale=2.0), WithOptions(scale=1.0, shift=5.0)
    base = by_key(run(one, hourly, test))
    double = by_key(run(two, hourly, test))
    moved = by_key(run(shifted, hourly, test))
    assert one._backend("cpu") is shifted._backend("cpu") is not two._backend("cpu")
    assert all(double[k] == pytest.approx(2 * v) for k, v in base.items())
    assert all(moved[k] == pytest.approx(v + 5.0) for k, v in base.items())
    assert len({key for key in foundation._LOADED if key[1] == "WithOptions"}) == 2


class Capped(FakeFM):
    _max_context_steps = 5 * 24


def test_a_context_beyond_what_a_family_reads_warns_for_each_model_and_is_recorded(hourly, tmp_path):
    """A family that reads at most 120 steps: models of 7 and 6 days of hours each warn in their own run, and the manifest's
    runtime records the steps read (120 both); 5 days doesn't warn."""
    for context in (7, 6):
        with pytest.warns(UserWarning, match=f"context={context} days of 1h periods is {context * 24} steps"):
            run(Capped(context=context, name=f"C{context}"), hourly, ("2023-06-12", "2023-06-12"), save_dir=tmp_path)
        runtime = json.loads((tmp_path / f"c{context}.manifest.json").read_text())["runtime"]
        assert runtime["context_steps"] == 120
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run(Capped(context=5, name="C5"), hourly, ("2023-06-12", "2023-06-12"))


def test_the_tui_builds_a_zero_shot_t0(monkeypatch):
    from epftoolbox2 import tui

    field = next(f for f in tui.PARAM_SCHEMAS["T0Model"] if f["name"] == "fine_tune_every")
    for answer, expected in (("", None), ("none", None), ("None", None), ("1MS", "1MS"), ("12MS", "12MS")):
        monkeypatch.setattr(
            tui.questionary,
            "text",
            lambda label, default="", value=answer: type("A", (), {"ask": lambda self: value})(),
        )
        assert tui._ask_field(field) == expected, answer


# ---------------------------------------------------------------------------------------------------------------------------
# The second review: a context without a change, GPU work in a child, another key, prune under the lock, windows with gaps
# ---------------------------------------------------------------------------------------------------------------------------


def test_a_window_whose_target_context_has_no_change_is_dropped_and_counted(hourly, tmp_path):
    """The price starts at the last hour of a day (the load forecast earlier): the window whose context holds that one value
    has no step-to-step change to scale t0's loss by, so it is dropped, counted and warned about, not given a NaN scale."""
    late = hourly.copy()
    late.loc[late.index < pd.Timestamp("2023-03-01 23:00", tz=ZONE), "price"] = np.nan
    model = FakeFM(context=CONTEXT, fine_tune_every="7D", fine_tune=QUICK)
    with pytest.warns(UserWarning, match="1 for a target context without two neighbouring values"):
        run(model, late, ("2023-06-12", "2023-06-12"), save_dir=tmp_path)
    counts = json.loads((tmp_path / "fakefm.manifest.json").read_text())["fine_tunes"]["windows"]["2023-06-12_h1"]
    assert counts["dropped"]["a target context without two neighbouring values"] == 1


def test_fine_tune_raises_on_a_loss_that_isnt_finite():
    """A window whose loss is NaN, in a training batch or in the holdout: the trainer raises instead of keeping the base weights
    as its best state with no word."""
    torch = pytest.importorskip("torch")
    from epftoolbox2.models._torch.finetune import FineTuneWindows, fine_tune

    dates = np.array([np.datetime64("2023-01-01") + np.timedelta64(i, "D") for i in range(40)])
    for bad in ("training", "holdout"):
        model = torch.nn.Linear(1, 1)
        bad_rows = set(range(0, 10)) if bad == "training" else set(range(35, 40))

        def loss(module, indices, bad_rows=bad_rows):
            out = module(torch.ones(len(indices), 1)).sum(dim=1) ** 2
            mask = torch.tensor([float("nan") if int(i) in bad_rows else 0.0 for i in indices])
            return out + mask

        windows = FineTuneWindows(dates=dates, cut=dates.max(), loss=loss)
        with pytest.raises(FloatingPointError, match=f"the {bad} loss"):
            fine_tune(model, windows, max_steps=5, batch_size=30, holdout_days=5, eval_every=1, seed=0)


class OnGpu(FakeFM):
    """The fake on a "GPU": its backend is hosted in the spawned child (`_device_worker`), where building it stands in for
    initialising CUDA (a torch whose `cuda.is_initialized()` is True there)."""

    def _backend(self, device):  # the base's process-wide cache and hosting, unlike the fake's one per model
        return FoundationModel._backend(self, device)

    @contextlib.contextmanager
    def _scope(self):
        yield "cuda:0"

    def _load_backend(self, resolved, device, options):
        import types

        sys.modules["torch"] = types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda: True))
        return PersistenceBackend(self)


def test_gpu_work_runs_in_a_spawned_child_and_this_process_keeps_forking(hourly, monkeypatch, tmp_path):
    """A model on a GPU hosts its backend in a spawned child: zero-shot forecasts and a stored fine-tune are the in-process
    fake's, the backend's process is another, this process never initialises CUDA (the stand-in is the child's), and the pool of
    a model after it still forks, with nothing in its manifest."""
    import os

    from epftoolbox2.models import _worker, foundation

    test = ("2023-06-12", "2023-06-13")
    here = by_key(run(make("FakeFM"), hourly, test))
    there = OnGpu(name="FakeFM")
    assert by_key(run(there, hourly, test)) == here
    remote = there._backend("cuda:0")
    assert isinstance(remote, foundation._RemoteBackend) and remote.pid != os.getpid()
    tuned_here = by_key(run(tuned("FakeFM"), hourly, test))
    tuned_there = OnGpu(name="FakeFM", fine_tune_every="7D", fine_tune=QUICK)
    assert by_key(run(tuned_there, hourly, test, save_dir=tmp_path)) == tuned_here
    assert tuned_there._fine_tunes == 1 and len(list((tmp_path / "fakefm.fits").glob("*.ckpt"))) == 1
    assert not _worker.cuda_initialised()
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", CapturePool)
    monkeypatch.setattr(CapturePool, "contexts", [])
    an_ols_run(hourly, tmp_path / "after")
    assert CapturePool.contexts == (["spawn"] if _worker._POOLS_SPAWN else ["fork"])
    assert "pool_start" not in json.loads((tmp_path / "after" / "ols.manifest.json").read_text())


def test_the_fine_tunes_of_an_earlier_version_s_folder_are_named(hourly, tmp_path):
    """A model named with capitals stored its fine-tunes in `<name>.fits/` before this version (`<stem>.fits/` now): the run
    warns, naming both folders."""
    (tmp_path / "FakeFM.fits").mkdir()
    with pytest.warns(UserWarning, match="holds fine-tunes of an earlier version"):
        run(tuned("FakeFM"), hourly, ("2023-06-12", "2023-06-12"), save_dir=tmp_path)


def test_prune_refuses_while_a_run_holds_the_results_lock(hourly, tmp_path):
    from epftoolbox2.models.foundation import prune_fine_tunes
    from epftoolbox2.results.store import ResultsLockedError, _hold_lock

    run(tuned("FakeFM"), hourly, ("2023-06-12", "2023-06-12"), save_dir=tmp_path)
    release = _hold_lock(tmp_path / "fakefm.lock")
    try:
        with pytest.raises(ResultsLockedError, match="is writing"):
            prune_fine_tunes(tmp_path, "FakeFM")
    finally:
        release()
    assert prune_fine_tunes(tmp_path, "FakeFM") == []


def test_windows_kept_with_gaps_warn_though_none_is_dropped(hourly):
    """One missing price hour: the windows whose contexts hold it are kept with the gap masked and none is dropped for data;
    the run says how many."""
    holed = hourly.copy()
    holed.loc[holed.index == pd.Timestamp("2023-05-20 03:00", tz=ZONE), "price"] = np.nan
    with pytest.warns(UserWarning, match=r"(\d+) of them with gaps the backend masks: dropped none for missing data"):
        run(tuned("FakeFM"), holed, ("2023-06-12", "2023-06-12"))
