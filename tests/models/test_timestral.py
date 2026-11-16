"""`TimestralModel` with the real package (`timestral` at its pinned commit, installed from git next to the `[timestral]`
extra) and a tiny random Timestral made here: the adapter's wiring against the research's call (`zoo_timestral/code/zoo.py`
`Timestral`, written out below), the covariates' mapping and the options, the pre-flight on the installed commit, the adapter
suite and `verify`. Skipped without the package; no CI job has the real weights, which the `-m weights` test reads from
`EPFTOOLBOX2_WEIGHTS_DIR`."""

import os

import numpy as np
import pytest

pytest.importorskip("timestral")
torch = pytest.importorskip("torch")

from epftoolbox2._extras import TIMESTRAL_COMMIT  # noqa: E402
from epftoolbox2.models import TimestralModel  # noqa: E402
from epftoolbox2.models._cards import installed_distribution  # noqa: E402
from epftoolbox2.models._history import history_batch  # noqa: E402
from tests.models import adapter_suite as suite  # noqa: E402
from tests.models.test_foundation import CONTEXT, INFO, INFO_AHEAD, market, pipeline  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*not the reference weights", "ignore:.*isn't the pinned commit")

LEVELS_21 = (
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
)


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


def tiny_timestral(root):
    """A Timestral of 32 dimensions, a time and a time-variate layer, its 21 levels, seeded, saved as the card's two files."""
    import safetensors.torch
    from timestral import TimestralConfig
    from timestral import TimestralModel as Network

    config = TimestralConfig(
        patch_size=32,
        d_model=32,
        num_heads=2,
        mlp_hidden_dim=64,
        block_schedule=("time", "time_variate"),
        quantile_knots=LEVELS_21,
        block_size=4,
        context_length=4096,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        network = Network(config)
        for parameter in network.parameters():  # no zero-initialised query: every input reaches the forecast
            torch.nn.init.normal_(parameter, std=0.05)
    config.save(root / "config.json")
    state = {key: value.contiguous() for key, value in network.state_dict().items()}
    safetensors.torch.save_file(state, str(root / "model.safetensors"))
    return str(root)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return tiny_timestral(tmp_path_factory.mktemp("timestral-tiny"))


@pytest.fixture(scope="module")
def hourly():
    return market("1h", start="2023-01-20", end="2023-11-05")


def make(name, weights, **params):
    return TimestralModel(
        name=name,
        context=params.pop("context", CONTEXT),
        known_ahead=params.pop("known_ahead", ["load_forecast"]),
        batch_size=params.pop("batch_size", 8),
        device="cpu",
        weights=weights,
        **params,
    )


def research_median(weights, item, clip):
    """`zoo.py`'s Timestral.predict on one item: target (1, 1, T) and the known-ahead series (1, 1, T + S) as
    `known_covariates`, float32, symmetric averaging on; the level nearest 0.5 of the 21."""
    from timestral import TimestralForecaster

    m = TimestralForecaster.from_pretrained(weights, device="cpu")
    levels = np.round(np.asarray(m.quantile_levels, dtype=float), 3)
    mid = int(np.argmin(np.abs(levels - 0.5)))
    context = np.stack([item.target.astype(np.float32)])[:, None, :]
    known = np.stack([np.stack([item.known["load_forecast"].astype(np.float32)])])
    f = m.forecast(context, prediction_length=item.steps_ahead, known_covariates=known, clip_nonnegative=clip)
    return f.quantiles[:, 0].float().cpu().numpy()[0, :, mid]


@pytest.mark.parametrize("clip", [False, True])
def test_a_forecast_is_the_research_s_call_of_the_accessor_s_item(hourly, tiny, clip):
    """Each period's forecast is the research's call on the item the accessor reads: the known-ahead series as known
    covariates over the context and the steps ahead, the level nearest 0.5; with the clip as asked (the research's was on)."""
    model = make("t", tiny, clip_nonnegative=clip)
    report = pipeline(model).run(hourly, "2023-06-14", "2023-06-14", horizon=1, forecast_only=True)
    records = report.refs["t"]._results
    (item,) = history_batch(model._history_spec("price"), hourly, ["2023-06-14"], INFO, "1h", 1).items
    median = research_median(tiny, item, clip)
    assert len(records) == 24
    for record in records:
        step = list(item.slot_periods).index(record["hour"])
        assert abs(record["prediction"] - float(median[step])) < 1e-4, record


def test_known_ahead_and_past_series_reach_their_own_arguments(hourly, tiny, monkeypatch):
    """`known_ahead` -> `known_covariates` (B, F, T + S), `past` -> `past_covariates` (B, P, T), in the model's order, the
    context (B, 1, T), and the model's options; the backend's point level is the one nearest 0.5 of the package's 21."""
    from timestral import TimestralForecaster

    seen = []
    real = TimestralForecaster.forecast

    def forecast(self, context, prediction_length, **kwargs):
        seen.append((np.asarray(context).shape, prediction_length, kwargs, np.asarray(context)))
        return real(self, context, prediction_length, **kwargs)

    monkeypatch.setattr(TimestralForecaster, "forecast", forecast)
    model = make("t", tiny, past=["load_actual"], symmetric_averaging=False, newest_target="published")
    pipeline(model, information_set=INFO_AHEAD).run(hourly, "2023-06-14", "2023-06-14", horizon=2, forecast_only=True)
    (item,) = history_batch(model._history_spec("price"), hourly, ["2023-06-14"], INFO_AHEAD, "1h", 2).items
    assert seen
    for shape, steps, kwargs, _ in seen:
        width = shape[-1]
        assert shape == (8, 1, width) and width == len(item.target)
        assert kwargs["known_covariates"].shape == (8, 1, width + steps)
        assert kwargs["past_covariates"].shape == (8, 1, width)
        assert kwargs["symmetric_averaging"] is False and kwargs["clip_nonnegative"] is False
    # The issue day's row of its call at horizon 2: its own known-ahead and past series, each in its own argument.
    target = item.target.astype(np.float32)
    (row,) = [
        (kwargs, r)
        for _, steps, kwargs, context in seen
        if steps == item.steps_ahead
        for r in range(len(context))
        if np.array_equal(context[r, 0], target, equal_nan=True)
    ][:1]
    kwargs, r = row
    np.testing.assert_array_equal(kwargs["known_covariates"][r, 0], item.known["load_forecast"].astype(np.float32))
    np.testing.assert_array_equal(kwargs["past_covariates"][r, 0], item.past["load_actual"].astype(np.float32))
    backend = model._backend("cpu")
    assert backend.levels == LEVELS_21 and backend.point_level == 0.5 and backend.point_index == 10


def test_symmetric_averaging_changes_the_forecast(hourly, tiny):
    test = ("2023-06-14", "2023-06-14")
    on = pipeline(make("t", tiny)).run(hourly, *test, horizon=1, forecast_only=True).refs["t"]._results
    off = pipeline(make("t", tiny, symmetric_averaging=False)).run(hourly, *test, horizon=1, forecast_only=True)
    assert [r["prediction"] for r in on] != [r["prediction"] for r in off.refs["t"]._results]


def test_the_pre_flight_passes_on_the_pinned_commit_it_is_installed_at():
    """The installed package is the pinned commit (PEP 610), so the card's pin passes and the manifest records it; asking
    for another commit refuses."""
    version, commit = installed_distribution("timestral")
    assert commit == TIMESTRAL_COMMIT, "the timestral package here isn't installed from its pinned commit"
    model = TimestralModel(size="28m")
    assert not [p for p in model._preflight({}) if p.kind == "runtime"]
    other = [p.message for p in TimestralModel(runtime_commit="0" * 12)._preflight({}) if p.kind == "runtime"]
    assert other and other[0].startswith("needs timestral at commit 000000000000")


def test_verify_and_the_adapter_suite(hourly, tiny, tmp_path):
    def factory(name, **params):
        return make(name, tiny, **params)

    suite.check_verify(factory, hourly, INFO, ["2023-06-14", "2023-10-28"], items=7)
    suite.check_verify(lambda name: make(name, tiny, newest_target="published"), hourly, INFO, ["2023-06-14"], items=7)
    test = ("2023-06-08", "2023-06-20")
    variants = ({}, {"known_ahead": []}, {"clip_nonnegative": True})
    suite.check_loop_and_shards(factory, hourly, tmp_path / "loop", test, 1, variants, information_set=INFO)
    suite.check_resume(factory, hourly, tmp_path / "resume", test, information_set=INFO)
    suite.check_yaml_round_trip(factory, hourly, tmp_path, test, information_set=INFO)
    suite.check_dst_counts(factory, hourly, "2023-03-26", "2023-10-29", information_set=INFO)
    suite.check_provenance(factory, hourly, tmp_path / "provenance", test, information_set=INFO)


@pytest.mark.weights
@pytest.mark.integration
def test_verify_with_the_real_weights_on_the_cpu():
    """`verify` of Timestral-209M on five issue days of a synthetic frame, on the CPU: ok with the control. Needs the fetched
    weights (`EPFTOOLBOX2_WEIGHTS_DIR`); outside CI."""
    ref = TimestralModel._card.weights
    root = os.environ.get("EPFTOOLBOX2_WEIGHTS_DIR")
    if not root or not os.path.isfile(os.path.join(root, ref.repo, ref.revision, "epftoolbox2-weights.json")):
        pytest.skip("Timestral-209M's weights are not in EPFTOOLBOX2_WEIGHTS_DIR (epf weights fetch TimestralModel)")
    frame = market("1h", start="2023-01-20", end="2023-08-01")

    def factory(name, **params):
        return TimestralModel(name=name, context=28, known_ahead=["load_forecast"], batch_size=8, device="cpu")

    suite.check_verify(factory, frame, INFO, ["2023-06-05", "2023-06-06", "2023-06-08", "2023-06-11", "2023-03-25"])
