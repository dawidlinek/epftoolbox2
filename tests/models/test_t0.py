"""`T0Model` with the real package (`tfc-t0==0.5.0`, the `[t0]` extra) and tiny random weights made here: the adapter's wiring,
the adapter suite, `verify`, the fine-tune through the library's trainer and its checkpoints, and the fine-tune loss against
the research's (`zoo_tsfm/code/t0_ft.py` `forward_q` and `loss_fn`, written out below as the research wrote them). Skipped
without the extra (the families-extra CI job has it; no job has the real weights, which `-m weights` tests read from
`EPFTOOLBOX2_WEIGHTS_DIR`)."""

import json
import os
import warnings

import numpy as np
import pytest

pytest.importorskip("t0")
torch = pytest.importorskip("torch")

from epftoolbox2.models import T0Model  # noqa: E402
from epftoolbox2.models._fm import LEVELS_19  # noqa: E402
from epftoolbox2.models._history import history_batch  # noqa: E402
from tests.models import adapter_suite as suite  # noqa: E402
from tests.models.test_foundation import CONTEXT, INFO, INFO_AHEAD, market, pipeline  # noqa: E402

QUICK = {"holdout_days": 14, "max_steps": 2, "eval_every": 1, "batch_size": 4}

pytestmark = pytest.mark.filterwarnings("ignore:.*not the reference weights")


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A T0Forecaster of 16 dimensions and two layers, seeded, saved as the card's two files."""
    from t0 import T0Forecaster

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = T0Forecaster(
            embed_dim=16,
            num_layers=2,
            num_heads=2,
            mlp_hidden_dim=32,
            patch_size=8,
            group_every_n=2,
            dropout=0.1,
            quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9),
        )
    root = tmp_path_factory.mktemp("t0-tiny")
    model.save_pretrained(str(root))
    (root / "README.md").unlink(missing_ok=True)
    return str(root)


@pytest.fixture(scope="module")
def hourly():
    return market("1h", start="2023-01-20", end="2023-11-05")


def make(name, weights, **params):
    return T0Model(
        name=name,
        context=CONTEXT,
        known_ahead=params.pop("known_ahead", ["load_forecast"]),
        fine_tune_every=params.pop("fine_tune_every", None),
        seeds=params.pop("seeds", 1),
        batch_size=params.pop("batch_size", 8),
        device="cpu",
        weights=weights,
        **params,
    )


def test_a_forecast_is_the_package_s_median_of_the_accessor_s_item(hourly, tiny):
    """The forecast of each period is `T0Forecaster.predict`'s 0.5 level (of the 19) of the item the accessor reads, the
    target (and past series) as variates of the context and the known-ahead series as future covariates, float32."""
    from t0 import T0Forecaster

    model = make("t0", tiny)
    report = pipeline(model).run(hourly, "2023-06-14", "2023-06-14", horizon=1, forecast_only=True)
    records = report.refs["t0"]._results
    assert len(records) == 24
    (item,) = history_batch(model._history_spec("price"), hourly, ["2023-06-14"], INFO, "1h", 1).items
    reference = T0Forecaster.from_pretrained(tiny).eval()
    context = torch.as_tensor(np.stack([item.target])[None].astype(np.float32))
    future = torch.as_tensor(item.known["load_forecast"][None, None].astype(np.float32))
    with torch.no_grad():
        out = reference.predict(
            context, horizon=item.steps_ahead, quantile_levels=list(LEVELS_19), future_covariates=future
        )
    median = out.quantiles[0, 0, :, LEVELS_19.index(0.5)].numpy()
    for record in records:
        step = list(item.slot_periods).index(record["hour"])
        assert abs(record["prediction"] - float(median[step])) < 1e-4  # a padded call of 8 against a call of 1


def test_zero_shot_verify_and_the_adapter_suite(hourly, tiny, tmp_path, monkeypatch):
    def factory(name, **params):
        return make(name, tiny, **params)

    suite.check_verify(factory, hourly, INFO, ["2023-06-14", "2023-03-25"], items=7)
    test = ("2023-06-08", "2023-06-20")
    suite.check_loop_and_shards(
        factory, hourly, tmp_path / "loop", test, 1, ({}, {"known_ahead": []}), information_set=INFO
    )
    suite.check_resume(factory, hourly, tmp_path / "resume", test, information_set=INFO)
    suite.check_yaml_round_trip(factory, hourly, tmp_path, test, information_set=INFO)
    suite.check_dst_counts(factory, hourly, "2023-03-26", "2023-10-29", information_set=INFO)
    suite.check_provenance(factory, hourly, tmp_path / "provenance", test, information_set=INFO)


def test_a_past_series_and_published_cut(hourly, tiny):
    model = make("t0", tiny, past=["load_actual"], newest_target="published")
    report = pipeline(model, information_set=INFO_AHEAD).run(
        hourly, "2023-06-14", "2023-06-15", horizon=2, forecast_only=True
    )
    counts = report.forecast_counts().iloc[0]
    assert counts["produced"] == counts["planned"] == 96
    # The past series is a row of the context the model reads: verify's control of it moves the forecasts.
    suite.check_verify(lambda name: make(name, tiny, past=["load_actual"]), hourly, INFO, ["2023-06-14"], items=7)


def test_the_fine_tune_runs_through_the_trainer_and_is_stored_and_reused(hourly, tiny, tmp_path):
    """Two steps of the research's recipe at a Monday anchor ("7D"), two seeds: stored with sidecars, a resume loads them (no
    fine-tune runs) and gives the fresh run's records; the fine-tuned forecasts differ from the zero-shot ones; `verify` with
    fine-tuning on is ok."""

    def tuned(name, **params):
        return make(name, tiny, fine_tune_every="7D", fine_tune=QUICK, seeds=2, **params)

    test = ("2023-06-12", "2023-06-14")
    before = pipeline(make("t0", tiny)).run(hourly, *test, horizon=1, forecast_only=True)
    fresh = tmp_path / "fresh"
    first = tuned("t0")
    pipeline(first).run(hourly, *test, horizon=1, save_dir=fresh, forecast_only=True)
    assert first._fine_tunes == 2
    stored = sorted((fresh / "t0.fits").glob("*.ckpt"))
    assert len(stored) == 2
    sidecar = json.loads(stored[0].with_suffix(".json").read_text())
    assert sidecar["data_end"] == "2023-06-11" and sidecar["windows"] > 100 and sidecar["best_step"] in (0, 1, 2)
    path = suite.results(fresh, "t0")
    lines = suite.lines(path)
    path.write_text("".join(f"{line}\n" for line in lines[::3]))
    again = tuned("t0")
    pipeline(again).run(hourly, *test, horizon=1, save_dir=fresh, forecast_only=True)
    assert again._fine_tunes == 0 and again._checkpoints_loaded == 2 and suite.lines(path) == lines
    zero = pipeline(make("t0", tiny)).run(hourly, *test, horizon=1, forecast_only=True)
    assert {r["prediction"] for r in zero.refs["t0"]._results} != {json.loads(line)["prediction"] for line in lines}
    # The backend this process loaded once is back on the base weights for a zero-shot model after the fine-tunes.
    assert suite.forecasts(zero, "t0") == suite.forecasts(before, "t0")
    suite.check_verify(lambda name: tuned(name), hourly, INFO, ["2023-06-12"], items=6)


def research_loss(model, ctx, fc, ys, steps=24):
    """t0_ft.py's `forward_q` and `loss_fn`, as the research wrote them (one window's loss is its `loss_fn` alone)."""
    from t0.data import VariateType, time_series_from_array
    from t0.model.rollout import RolloutManager

    dev = "cpu"
    native = torch.tensor(model.config.quantile_levels, dtype=torch.float32, device=dev)
    ctx_t = torch.as_tensor(ctx, device=dev)
    fc_t = None if fc is None else torch.as_tensor(fc, device=dev)
    ts, _ = time_series_from_array(ctx_t, steps, torch.device(dev), fc_t, None, None)
    T = ctx_t.shape[-1]
    rm = RolloutManager(model)
    buf = rm.prepare_rollout_buffer(ts, steps, T)
    ps = model.patch_size
    width = -(-T // ps) * ps + ps
    pred = rm.predict_step(buf.time_slice(0, width), ps)
    rows = buf.variate_type[:, -1] == VariateType.TARGET
    pred = pred[rows]
    V = ctx.shape[1]
    pred = pred.reshape(ctx.shape[0], V, *pred.shape[1:])[:, 0][:, :steps, :]
    sc = torch.as_tensor(
        np.array([np.mean(np.abs(np.diff(tg))) + 1e-3 for tg in ctx[:, 0]]), dtype=torch.float32, device=dev
    )
    y = torch.as_tensor(ys, dtype=torch.float32, device=dev)
    e = y[..., None] - pred
    pin = torch.maximum(native * e, (native - 1) * e).mean(dim=(1, 2))
    return pin / sc


def test_the_window_loss_is_the_research_s(tmp_path):
    """On random windows of 24 steps ahead of a model with t0-alpha's patch of 32 (the research's one patch ahead holds its 24
    steps), with and without a covariate, the backend's loss of each window equals t0_ft.py's to float32 rounding; a masked
    target step leaves the other windows' losses as they were."""
    from t0 import T0Forecaster

    from epftoolbox2.models._fm.t0 import T0Backend
    from epftoolbox2.models._weights import resolve

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        model = T0Forecaster(
            embed_dim=16,
            num_layers=2,
            num_heads=2,
            mlp_hidden_dim=32,
            patch_size=32,
            group_every_n=2,
            dropout=0.0,
            quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9),
        )
    model.save_pretrained(str(tmp_path))
    (tmp_path / "README.md").unlink(missing_ok=True)
    backend = T0Backend(resolve(T0Model._card, str(tmp_path)), torch.device("cpu"))
    rng = np.random.default_rng(1)
    ctx = (50 + 10 * rng.normal(size=(5, 1, 96))).astype(np.float32)
    fc = (1e4 + 100 * rng.normal(size=(5, 1, 120))).astype(np.float32)
    ys = (50 + 10 * rng.normal(size=(5, 24))).astype(np.float32)
    with torch.no_grad():
        for future in (None, fc):
            ours = backend.window_losses(ctx, future, ys, 24).numpy()
            theirs = research_loss(backend.model, ctx, future, ys).numpy()
            np.testing.assert_allclose(ours, theirs, rtol=1e-6)
        masked = ys.copy()
        masked[0, :5] = np.nan
        part = backend.window_losses(ctx, None, masked, 24).numpy()
        whole = backend.window_losses(ctx, None, ys, 24).numpy()
        assert np.isfinite(part).all() and np.array_equal(part[1:], whole[1:]) and part[0] != whole[0]


def weights_dir():
    root = os.environ.get("EPFTOOLBOX2_WEIGHTS_DIR")
    ref = T0Model._card.weights
    if not root or not os.path.isfile(os.path.join(root, ref.repo, ref.revision, "epftoolbox2-weights.json")):
        pytest.skip("t0-alpha's weights are not in EPFTOOLBOX2_WEIGHTS_DIR (epf weights fetch T0Model)")


@pytest.mark.weights
@pytest.mark.integration
def test_verify_with_the_real_weights_on_the_cpu():
    """`verify` of t0-alpha fine-tuned (two steps) on five issue days of a synthetic frame, on the CPU: ok with the control.
    Needs the fetched weights (`EPFTOOLBOX2_WEIGHTS_DIR`); outside CI."""
    weights_dir()
    frame = market("1h", start="2022-09-01", end="2023-08-01")
    days = ["2023-06-05", "2023-06-06", "2023-06-08", "2023-06-11", "2023-03-25"]

    def factory(name, **params):
        return T0Model(
            name=name,
            context=28,
            known_ahead=["load_forecast"],
            fine_tune_every="1MS",
            fine_tune={"max_steps": 2, "holdout_days": 14, "window": 120},
            seeds=1,
            batch_size=8,
            device="cpu",
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suite.check_verify(factory, frame, INFO, days)
