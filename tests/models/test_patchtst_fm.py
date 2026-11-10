"""`PatchTSTFMModel` with the real package (granite-tsfm, the `[patchtst-fm]` extra, Python < 3.14) and a tiny random
PatchTST-FM made here: the adapter's wiring against the research's call (`zoo_tsfm/code/zoo_extra.py` `PatchTSTFM`, written out
below), the adapter suite and `verify`. Skipped without the extra; no CI job has the real weights, which the `-m weights` test
reads from `EPFTOOLBOX2_WEIGHTS_DIR`."""

import os

import numpy as np
import pytest

pytest.importorskip("tsfm_public")
torch = pytest.importorskip("torch")

from epftoolbox2.models import PatchTSTFMModel  # noqa: E402
from epftoolbox2.models._fm import LEVELS_19  # noqa: E402
from epftoolbox2.models._history import history_batch  # noqa: E402
from tests.models import adapter_suite as suite  # noqa: E402
from tests.models.test_foundation import CONTEXT, INFO, market, pipeline  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*not the reference weights")


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


def tiny_patchtst(root):
    """A PatchTST-FM of 32 dimensions and two conformer blocks over a 512-step window, its 99 levels, seeded, saved as the
    card's two files."""
    from tsfm_public.models.patchtst_fm.configuration_patchtst_fm import PatchTSTFMConfig
    from tsfm_public.models.patchtst_fm.modeling_patchtst_fm import PatchTSTFMForPrediction

    config = PatchTSTFMConfig(
        context_length=512,
        prediction_length=64,
        d_patch=16,
        patch_stride=8,
        patch_loss_windowing="hamming",
        d_model=32,
        n_head=2,
        n_layer=2,
        num_quantile=99,
        block_type="conformer",
        conv_kernel_size=[3, 5],
        prehead_norm=True,
        qkv_bias=False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        PatchTSTFMForPrediction(config).save_pretrained(str(root))
    for extra in root.iterdir():
        if extra.name not in ("config.json", "model.safetensors"):
            extra.unlink()
    return str(root)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return tiny_patchtst(tmp_path_factory.mktemp("patchtst-tiny"))


@pytest.fixture(scope="module")
def hourly():
    return market("1h", start="2023-01-20", end="2023-11-05")


def make(name, weights, **params):
    return PatchTSTFMModel(
        name=name,
        context=params.pop("context", CONTEXT),
        batch_size=params.pop("batch_size", 8),
        device="cpu",
        weights=weights,
        **params,
    )


def research_median(weights, target, steps):
    """`zoo_extra.py`'s PatchTSTFM.predict on one context: (1, T, 1) float32, the 19 levels of the 99, sorted, index 9."""
    from tsfm_public.models.patchtst_fm.modeling_patchtst_fm import PatchTSTFMForPrediction

    m = PatchTSTFMForPrediction.from_pretrained(weights).eval()
    ql = np.asarray(m.config.quantile_levels, dtype=float)
    idx = [int(np.argmin(np.abs(ql - lv))) for lv in LEVELS_19]
    x = torch.as_tensor(np.stack([target.astype(np.float32)])[:, :, None], dtype=torch.float32)
    with torch.no_grad():
        o = m(past_values=x, prediction_length=steps)
    q = o.quantile_outputs
    q = torch.stack(list(q)) if isinstance(q, (list, tuple)) else q
    q = q.float().cpu().numpy()
    if q.ndim == 4:
        q = q[..., 0]
    Q = np.sort(np.transpose(q[:, idx, :steps], (0, 2, 1)), axis=-1)
    return Q[0, :, 9]


@pytest.mark.parametrize("newest", ["d-1", "published"])
def test_a_forecast_is_the_research_s_median_of_the_accessor_s_item(hourly, tiny, newest):
    """The forecast of each period is the research's call on the item the accessor reads (the target alone, float32): of the
    99 levels the 19, sorted per step, the middle one."""
    model = make("p", tiny, newest_target=newest)
    report = pipeline(model).run(hourly, "2023-06-14", "2023-06-14", horizon=1, forecast_only=True)
    records = report.refs["p"]._results
    (item,) = history_batch(model._history_spec("price"), hourly, ["2023-06-14"], INFO, "1h", 1).items
    assert item.known == {} and item.past == {}
    median = research_median(tiny, item.target, item.steps_ahead)
    assert len(records) == 24
    for record in records:
        step = list(item.slot_periods).index(record["hour"])
        assert abs(record["prediction"] - float(median[step])) < 1e-4, record


def test_the_backend_reads_only_the_target_and_keeps_the_nineteen_levels(hourly, tiny):
    from epftoolbox2.models._fm.patchtst_fm import PatchTSTFMBackend, level_indices
    from epftoolbox2.models._weights import resolve

    backend = PatchTSTFMBackend(resolve(PatchTSTFMModel._card, tiny), "cpu")
    assert backend.levels == LEVELS_19 and backend.point_index == 9
    assert level_indices([k / 100 for k in range(1, 100)]) == [5 * k - 1 for k in range(1, 20)]
    with pytest.raises(ValueError, match="don't hold"):
        level_indices([0.1, 0.5, 0.9])
    assert "autocast" not in make("p", tiny)._runtime_fields()  # float32 on the CPU


def test_the_backend_sorts_the_levels_per_step_as_the_research_did(hourly, tiny, monkeypatch):
    """The package's head can't cross its quantiles (a cumulative softplus), but the research sorted the 19 levels per step
    before taking the middle one, and so does the backend: quantiles handed back in reverse order come out sorted."""
    from tsfm_public.models.patchtst_fm.modeling_patchtst_fm import PatchTSTFMForPrediction

    from epftoolbox2.models._fm.patchtst_fm import PatchTSTFMBackend
    from epftoolbox2.models._weights import resolve

    backend = PatchTSTFMBackend(resolve(PatchTSTFMModel._card, tiny), "cpu")
    model = make("p", tiny)
    batch = history_batch(model._history_spec("price"), hourly, ["2023-06-14"], INFO, "1h", 1)
    (call,) = batch.batches(1)
    plain = backend.predict(call, {})
    real = PatchTSTFMForPrediction.forward

    def reversed_levels(self, *args, **kwargs):
        out = real(self, *args, **kwargs)
        out.quantile_outputs = torch.flip(out.quantile_outputs, dims=[1])
        return out

    monkeypatch.setattr(PatchTSTFMForPrediction, "forward", reversed_levels)
    crossed = backend.predict(call, {})
    assert (np.diff(crossed, axis=-1) >= 0).all()
    np.testing.assert_array_equal(crossed, plain)


def test_verify_and_the_adapter_suite(hourly, tiny, tmp_path):
    def factory(name, **params):
        return make(name, tiny, **params)

    suite.check_verify(factory, hourly, INFO, ["2023-06-14", "2023-10-28"], items=7)
    suite.check_verify(lambda name: make(name, tiny, newest_target="published"), hourly, INFO, ["2023-06-14"], items=7)
    test = ("2023-06-08", "2023-06-20")
    suite.check_loop_and_shards(factory, hourly, tmp_path / "loop", test, 1, ({}, {"context": 5}), information_set=INFO)
    suite.check_resume(factory, hourly, tmp_path / "resume", test, information_set=INFO)
    suite.check_yaml_round_trip(factory, hourly, tmp_path, test, information_set=INFO)
    suite.check_dst_counts(factory, hourly, "2023-03-26", "2023-10-29", information_set=INFO)
    suite.check_provenance(factory, hourly, tmp_path / "provenance", test, information_set=INFO)


def test_a_context_beyond_the_window_warns(hourly, tiny):
    """The package's window takes in the newest 8,192 - max(S, 128) steps: 341 days of hours (8,184) warns, 336 (8,064)
    doesn't."""
    frame = market("1h", start="2022-01-01", end="2023-07-01")
    with pytest.warns(UserWarning, match="context=341 days of 1h periods is 8184 steps"):
        pipeline(make("p", tiny, context=341)).run(frame, "2023-06-14", "2023-06-14", horizon=1, forecast_only=True)


@pytest.mark.weights
@pytest.mark.integration
def test_verify_with_the_real_weights_on_the_cpu():
    """`verify` of PatchTST-FM-r2 on five issue days of a synthetic frame, on the CPU: ok with the control. Needs the fetched
    weights (`EPFTOOLBOX2_WEIGHTS_DIR`); outside CI."""
    ref = PatchTSTFMModel._card.weights
    root = os.environ.get("EPFTOOLBOX2_WEIGHTS_DIR")
    if not root or not os.path.isfile(os.path.join(root, ref.repo, ref.revision, "epftoolbox2-weights.json")):
        pytest.skip("PatchTST-FM's weights are not in EPFTOOLBOX2_WEIGHTS_DIR (epf weights fetch PatchTSTFMModel)")
    frame = market("1h", start="2023-01-20", end="2023-08-01")

    def factory(name, **params):
        return PatchTSTFMModel(name=name, context=28, batch_size=8, device="cpu")

    suite.check_verify(factory, frame, INFO, ["2023-06-05", "2023-06-06", "2023-06-08", "2023-06-11", "2023-03-25"])
