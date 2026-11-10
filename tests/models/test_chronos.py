"""`ChronosModel` with the real package (`chronos-forecasting`, the `[chronos]` extra) and a tiny random Chronos-2 made here: the
adapter's wiring against the research's call (`zoo_tsfm/code/zoo.py` `Chronos`), the adapter suite and `verify`. Skipped without
the extra; the real weights are for `-m weights` tests and the B1 checks outside CI."""

import os

import numpy as np
import pytest

pytest.importorskip("chronos")
torch = pytest.importorskip("torch")

from epftoolbox2.models import ChronosModel  # noqa: E402
from epftoolbox2.models._fm import LEVELS_19  # noqa: E402
from epftoolbox2.models._history import history_batch  # noqa: E402
from tests.models import adapter_suite as suite  # noqa: E402
from tests.models.test_foundation import CONTEXT, INFO, INFO_AHEAD, market, pipeline  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*not the reference weights")

QUANTILES = [0.01, *LEVELS_19, 0.99]


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


def tiny_chronos(root, context_length=2048):
    from chronos.chronos2 import Chronos2Model
    from chronos.chronos2.config import Chronos2CoreConfig

    config = Chronos2CoreConfig(
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=2,
        num_heads=4,
        chronos_config=dict(
            context_length=context_length,
            input_patch_size=16,
            input_patch_stride=16,
            max_output_patches=8,
            output_patch_size=16,
            quantiles=QUANTILES,
            time_encoding_scale=context_length,
            use_arcsinh=True,
            use_reg_token=True,
        ),
        chronos_pipeline_class="Chronos2Pipeline",
        architectures=["Chronos2Model"],
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        Chronos2Model(config).save_pretrained(str(root))
    return str(root)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return tiny_chronos(tmp_path_factory.mktemp("chronos-tiny"))


@pytest.fixture(scope="module")
def hourly():
    return market("1h", start="2023-01-20", end="2023-11-05")


def make(name, weights, **params):
    return ChronosModel(
        name=name,
        context=params.pop("context", CONTEXT),
        known_ahead=params.pop("known_ahead", ["load_forecast"]),
        batch_size=params.pop("batch_size", 8),
        device="cpu",
        weights=weights,
        **params,
    )


def test_a_forecast_is_the_pipeline_s_median_as_the_research_called_it(hourly, tiny):
    """The research's item for the same day (target, past covariates = the known-ahead series over the context, future
    covariates = over the steps ahead, float32), through `predict_quantiles` at its 19 levels: the forecast is the 0.5 one."""
    from chronos import BaseChronosPipeline

    model = make("c2", tiny)
    report = pipeline(model).run(hourly, "2023-06-14", "2023-06-14", horizon=1, forecast_only=True)
    records = report.refs["c2"]._results
    (item,) = history_batch(model._history_spec("price"), hourly, ["2023-06-14"], INFO, "1h", 1).items
    pipe = BaseChronosPipeline.from_pretrained(tiny, device_map="cpu")
    width = len(item.target)
    known = item.known["load_forecast"].astype(np.float32)
    research = {
        "target": item.target.astype(np.float32),
        "past_covariates": {"x1": known[:width]},
        "future_covariates": {"x1": known[width:]},
    }
    quantiles, _ = pipe.predict_quantiles(
        [research], prediction_length=item.steps_ahead, quantile_levels=list(LEVELS_19)
    )
    median = np.asarray(quantiles[0])[0, :, LEVELS_19.index(0.5)]
    assert len(records) == 24
    for record in records:
        step = list(item.slot_periods).index(record["hour"])
        assert abs(record["prediction"] - float(median[step])) < 1e-4


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


def test_a_past_series_at_horizon_two(hourly, tiny):
    model = make("c2", tiny, past=["load_actual"], newest_target="published")
    report = pipeline(model, information_set=INFO_AHEAD).run(
        hourly, "2023-06-14", "2023-06-15", horizon=2, forecast_only=True
    )
    counts = report.forecast_counts().iloc[0]
    assert counts["produced"] == counts["planned"] == 96


def test_a_context_longer_than_the_model_reads_warns(hourly, tmp_path):
    short = tiny_chronos(tmp_path / "short", context_length=96)
    with pytest.warns(UserWarning, match="longer than Chronos-2's 96"):
        pipeline(make("c2", short)).run(hourly, "2023-06-14", "2023-06-14", horizon=1, forecast_only=True)


@pytest.mark.weights
@pytest.mark.integration
def test_verify_with_the_real_weights_on_the_cpu():
    """`verify` of Chronos-2 on five issue days of a synthetic frame, on the CPU: ok with the control. Needs the fetched
    weights (`EPFTOOLBOX2_WEIGHTS_DIR`); outside CI."""
    ref = ChronosModel._card.weights
    root = os.environ.get("EPFTOOLBOX2_WEIGHTS_DIR")
    if not root or not os.path.isfile(os.path.join(root, ref.repo, ref.revision, "epftoolbox2-weights.json")):
        pytest.skip("Chronos-2's weights are not in EPFTOOLBOX2_WEIGHTS_DIR (epf weights fetch ChronosModel)")
    frame = market("1h", start="2023-01-20", end="2023-08-01")

    def factory(name, **params):
        return ChronosModel(name=name, context=28, known_ahead=["load_forecast"], batch_size=8, device="cpu")

    suite.check_verify(factory, frame, INFO, ["2023-06-05", "2023-06-06", "2023-06-08", "2023-06-11", "2023-03-25"])
