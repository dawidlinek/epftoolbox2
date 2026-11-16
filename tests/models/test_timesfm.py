"""`TimesFMModel` with the real package (`timesfm[torch]==3.0.2`, the `[timesfm]` extra) and a tiny random TimesFM-3 made
here: the adapter's wiring against the research's call (`zoo_tsfm/code/zoo_extra.py` `TimesFM3`, written out below), the
covariates' mapping, the adapter suite and `verify`. Skipped without the extra; no CI job has the real weights, which the `-m
weights` test reads from `EPFTOOLBOX2_WEIGHTS_DIR`."""

import os

import numpy as np
import pytest

pytest.importorskip("timesfm3")
torch = pytest.importorskip("torch")

from epftoolbox2.models import TimesFMModel  # noqa: E402
from epftoolbox2.models._history import history_batch  # noqa: E402
from tests.models import adapter_suite as suite  # noqa: E402
from tests.models.test_foundation import CONTEXT, INFO, INFO_AHEAD, market, pipeline  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*not the reference weights")

DECILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


def tiny_timesfm(root):
    """A TimesFM-3 of 32 dimensions and two layers, patches of 32 in and 64 out, its deciles, seeded, saved as the card's
    two files by its hub mixin (`config.json` its own `to_dict()`)."""
    from timesfm3.torch import ResidualBlockConfig, StackedTransformersConfig, TimesFM3Torch, TransformerConfig

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        network = TimesFM3Torch(
            input_patch_len=32,
            output_patch_len=64,
            quantiles=list(DECILES),
            residual_block_config=ResidualBlockConfig(
                hidden_dims=32, output_dims=32, use_bias=False, activation="relu"
            ),
            transformer_config=StackedTransformersConfig(
                num_layers=2,
                transformer=TransformerConfig(
                    model_dims=32,
                    hidden_dims=32,
                    num_heads=2,
                    attention_norm="rms",
                    feedforward_norm="rms",
                    qk_norm="rms",
                    use_bias=False,
                    use_rope_seq=True,
                    use_rope_var=False,
                    ff_activation="relu",
                    deterministic=True,
                ),
            ),
        )
    network.save_pretrained(str(root))
    for extra in root.iterdir():
        if extra.name not in ("config.json", "model.safetensors"):
            extra.unlink()
    return str(root)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return tiny_timesfm(tmp_path_factory.mktemp("timesfm-tiny"))


@pytest.fixture(scope="module")
def hourly():
    return market("1h", start="2023-01-20", end="2023-11-05")


def make(name, weights, **params):
    return TimesFMModel(
        name=name,
        context=params.pop("context", CONTEXT),
        known_ahead=params.pop("known_ahead", ["load_forecast"]),
        batch_size=params.pop("batch_size", 8),
        device="cpu",
        weights=weights,
        **params,
    )


def research_median(weights, item):
    """`zoo_extra.py`'s TimesFM3.predict on one item: the context, the known-ahead series over context and steps ahead as
    `past_future_covariates` (F, T + S), the deciles sorted, index 4."""
    from timesfm3.torch.timesfm3_forecaster import TimesFM3Forecaster

    m = TimesFM3Forecaster.from_pretrained(weights, device="cpu", per_core_batch_size=64)
    fut = {"x1": item.known["load_forecast"].astype(np.float32)}
    pf = [np.stack([fut[k] for k in sorted(fut)])]
    (o,) = list(
        m.predict_batch(
            contexts=[item.target.astype(np.float32)],
            horizon=item.steps_ahead,
            past_only_covariates=None,
            past_future_covariates=pf,
            return_quantiles=True,
        )
    )
    q = np.squeeze(np.asarray(o.quantiles, dtype=np.float32))
    return np.sort(q[: item.steps_ahead, -9:], axis=-1)[:, 4]


def test_a_forecast_is_the_research_s_call_of_the_accessor_s_item(hourly, tiny):
    """Each period's forecast is the research's call on the item the accessor reads: the known-ahead series as past-future
    covariates, the median of the sorted deciles."""
    model = make("f", tiny)
    report = pipeline(model).run(hourly, "2023-06-14", "2023-06-14", horizon=1, forecast_only=True)
    records = report.refs["f"]._results
    (item,) = history_batch(model._history_spec("price"), hourly, ["2023-06-14"], INFO, "1h", 1).items
    median = research_median(tiny, item)
    assert len(records) == 24
    for record in records:
        step = list(item.slot_periods).index(record["hour"])
        assert abs(record["prediction"] - float(median[step])) < 1e-4, record


def test_known_ahead_and_past_series_reach_their_own_arguments(hourly, tiny, monkeypatch):
    """`known_ahead` -> `past_future_covariates` (F, T + S) per item, `past` -> `past_only_covariates` (P, T), the contexts
    (T,), one batch of the call's size; the backend's levels are the deciles and the forecast index 4."""
    from timesfm3.torch.timesfm3_forecaster import TimesFM3Forecaster

    seen = []
    real = TimesFM3Forecaster.predict_batch

    def predict_batch(self, contexts, horizon, **kwargs):
        record = {**kwargs, "contexts": contexts}
        seen.append((len(contexts), [c.shape for c in contexts], horizon, record, self.config.per_core_batch_size))
        return real(self, contexts, horizon, **kwargs)

    monkeypatch.setattr(TimesFM3Forecaster, "predict_batch", predict_batch)
    model = make("f", tiny, past=["load_actual"], newest_target="published")
    pipeline(model, information_set=INFO_AHEAD).run(hourly, "2023-06-14", "2023-06-14", horizon=2, forecast_only=True)
    (item,) = history_batch(model._history_spec("price"), hourly, ["2023-06-14"], INFO_AHEAD, "1h", 2).items
    assert seen
    for count, shapes, steps, kwargs, batch in seen:
        width = len(item.target)
        assert count == batch == 8 and shapes == [(width,)] * 8
        assert [k.shape for k in kwargs["past_future_covariates"]] == [(1, width + steps)] * 8
        assert [k.shape for k in kwargs["past_only_covariates"]] == [(1, width)] * 8
    # The issue day's item of its call at horizon 2: its own known-ahead and past series, each in its own argument.
    target = item.target.astype(np.float32)
    (row,) = [
        (kwargs, r)
        for _, _, steps, kwargs, _ in seen
        if steps == item.steps_ahead
        for r in range(len(kwargs["contexts"]))
        if np.array_equal(kwargs["contexts"][r], target, equal_nan=True)
    ][:1]
    kwargs, r = row
    np.testing.assert_array_equal(
        kwargs["past_future_covariates"][r][0], item.known["load_forecast"].astype(np.float32)
    )
    np.testing.assert_array_equal(kwargs["past_only_covariates"][r][0], item.past["load_actual"].astype(np.float32))
    backend = model._backend("cpu")
    assert backend.levels == DECILES and backend.point_index == 4


def test_verify_and_the_adapter_suite(hourly, tiny, tmp_path):
    def factory(name, **params):
        return make(name, tiny, **params)

    suite.check_verify(factory, hourly, INFO, ["2023-06-14", "2023-10-28"], items=7)
    suite.check_verify(lambda name: make(name, tiny, newest_target="published"), hourly, INFO, ["2023-06-14"], items=7)
    test = ("2023-06-08", "2023-06-20")
    suite.check_loop_and_shards(
        factory, hourly, tmp_path / "loop", test, 1, ({}, {"known_ahead": []}), information_set=INFO
    )
    suite.check_resume(factory, hourly, tmp_path / "resume", test, information_set=INFO)
    suite.check_yaml_round_trip(factory, hourly, tmp_path, test, information_set=INFO)
    suite.check_dst_counts(factory, hourly, "2023-03-26", "2023-10-29", information_set=INFO)
    suite.check_provenance(factory, hourly, tmp_path / "provenance", test, information_set=INFO)


@pytest.mark.weights
@pytest.mark.integration
def test_verify_with_the_real_weights_on_the_cpu():
    """`verify` of TimesFM-3 on five issue days of a synthetic frame, on the CPU: ok with the control. Needs the fetched
    weights (`EPFTOOLBOX2_WEIGHTS_DIR`); outside CI."""
    ref = TimesFMModel._card.weights
    root = os.environ.get("EPFTOOLBOX2_WEIGHTS_DIR")
    if not root or not os.path.isfile(os.path.join(root, ref.repo, ref.revision, "epftoolbox2-weights.json")):
        pytest.skip("TimesFM-3's weights are not in EPFTOOLBOX2_WEIGHTS_DIR (epf weights fetch TimesFMModel)")
    frame = market("1h", start="2023-01-20", end="2023-08-01")

    def factory(name, **params):
        return TimesFMModel(name=name, context=28, known_ahead=["load_forecast"], batch_size=8, device="cpu")

    suite.check_verify(factory, frame, INFO, ["2023-06-05", "2023-06-06", "2023-06-08", "2023-06-11", "2023-03-25"])
