"""Foundation models (step 6i): `FoundationModel`'s run with a deterministic numpy backend, the fine-tune scheduler and its
checkpoint store, and what `T0Model` and `ChronosModel` are before any package of theirs is imported.

`FakeFM` is a `FoundationModel` whose backend is numpy (`PersistenceBackend`): each step ahead is the target at the same step of
the context's last day plus 0.1 x the known-ahead series there (6h's fake), plus a bias, at the levels (0.25, 0.5, 0.75) with
the median in the middle; its fine-tune learns the bias, the mean residual of that persistence over the training windows
and a thousandth of the holdout's (which early stopping reads), plus a small amount per seed, so seeds differ, and asserts the trainer's cut as `_torch.finetune` does. It runs without
torch, in the core CI job. The real packages' adapters with tiny random weights are in `test_t0.py` and `test_chronos.py`.
"""

import dataclasses
import datetime
import json
import subprocess
import sys
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

    def point(self, target, known, steps):
        periods = self.model._periods_per_day
        width = target.shape[1]
        out = target[:, width - periods + np.arange(steps) % periods]
        for values in known.values():
            out = out + 0.1 * values[:, width:]
        return out

    def predict(self, call):
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
        # The training windows' mean residual, and a little of the holdout's, as early stopping reads it to choose a step.
        self.bias = float(np.mean(residuals[False])) + 1e-3 * float(np.mean(residuals[True]))
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

    def _load_backend(self, resolved, device):
        return PersistenceBackend(self)

    def _backend(self, device):  # one backend per model (the fake reads the model's own settings), not one per process
        if self._fake is None:
            self._fake = self._load_backend(None, device)
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
    return {(r["run_date"], r["hour"], r["horizon"]): r["prediction"] for r in report.refs[name]._results}


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
    ChronosModel import none of torch, t0, chronos, huggingface_hub, transformers or safetensors."""
    code = (
        "import sys\n"
        "import epftoolbox2\n"
        "from epftoolbox2.data.sources import CalendarSource\n"
        "from epftoolbox2.pipelines import DataPipeline, ModelPipeline\n"
        "from epftoolbox2.models import T0Model, ChronosModel, FoundationModel\n"
        "frame = DataPipeline(sources=[CalendarSource(country='PL')]).run('2024-01-01', '2024-01-08')\n"
        "assert len(frame), 'no calendar data'\n"
        "models = [T0Model(known_ahead=['load_forecast']), ChronosModel(context=182)]\n"
        "ModelPipeline().add_model(models[0]).add_model(models[1]).to_dict()\n"
        "[model.card for model in models]\n"
        "heavy = ('torch', 't0', 'chronos', 'huggingface_hub', 'transformers', 'safetensors')\n"
        "found = sorted(m for m in sys.modules if m.split('.')[0] in heavy)\n"
        "assert not found, found\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr


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
    fits = sorted((resumed / "FakeFM.fits").glob("*.ckpt"))
    assert first._fine_tunes == 6 and len(fits) == 6  # three Mondays, two seeds
    sidecar = json.loads(fits[0].with_suffix(".json").read_text())
    assert {"anchor", "horizon", "seed", "sha256", "data_end", "best_step", "windows", "data_digest", "key"} <= set(
        sidecar
    )
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
    with pytest.raises(ValueError, match="changed after it was stored"):
        run(tuned("FakeFM", seeds=2), hourly, test, save_dir=resumed)


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
    data, digest, cut = model._fine_tune_data(unit, hourly, spec)
    assert data.cut == date(2023, 6, 11) and len(data) > 100 and len(digest) == 64
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
