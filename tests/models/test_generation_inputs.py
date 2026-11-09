"""A solar forecast built from step 7e's inputs alone, under the built-in rules: `SolarGeometryTransformer`'s night flag as
`zero_when`, `RollingStatsTransformer`'s capacity proxy as `normalise_by`, run at `availability="raise"`; `verify` passes
with its positive control, also with the inputs rebuilt from the source columns, and a proxy without its lag is refused by
the check and found by the probe."""

import warnings

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.transformers import LagTransformer, RollingStatsTransformer, SolarGeometryTransformer
from epftoolbox2.information_set import Availability, InformationSet, InformationSetError
from epftoolbox2.models import OLSModel
from epftoolbox2.pipelines import ModelPipeline

ZONE = "Europe/Berlin"
SITE = (51.0, 10.0)
PROXY = "generation_solar_roll14max_d-1"
# `cloud` stands in for a weather forecast: the only input that needs a rule of the test's own.
INFO = InformationSet("11:45", "Europe/Brussels", rules={"cloud": Availability.published("10:00", 2)})
PREDICTORS = ["clear_sky_ghi_d+{horizon}", "cloud_d+{horizon}"]
TEST = ("2023-03-27", "2023-04-02")  # the week after the spring switch


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


def source(seed=0):
    """`generation_solar` (MW) of a capacity growing by 2 MW a day, the clear sky and a cloud cover, hourly in Berlin from
    15 January 2023, over the spring switch."""
    index = pd.date_range("2023-01-15", "2023-04-25", freq="1h", tz=ZONE, inclusive="left")
    rng = np.random.default_rng(seed)
    days = (index.tz_localize(None).normalize() - pd.Timestamp("2023-01-15")).days.to_numpy()
    sky = SolarGeometryTransformer([SITE]).transform(pd.DataFrame(index=index))["clear_sky_ghi"].to_numpy()
    cloud = rng.uniform(0, 1, len(index))
    generation = (1000.0 + 2.0 * days) * sky / 1000.0 * (1.0 - 0.7 * cloud) * rng.uniform(0.95, 1.05, len(index))
    return pd.DataFrame({"generation_solar": generation, "cloud": cloud}, index=index)


def build(frame, lag_days=1):
    """The model's inputs from the source columns, as a DataPipeline's transformers would make them."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = SolarGeometryTransformer([SITE]).transform(frame)
        out = RollingStatsTransformer("generation_solar", 14, "max", lag_days).transform(out)
        return LagTransformer(columns=["clear_sky_ghi", "is_night", "cloud"], lags=[-1, -2], freq="day").transform(out)


@pytest.fixture(scope="module")
def data():
    return build(source())


def model(scale=PROXY, **kwargs):
    return OLSModel(
        PREDICTORS,
        training_window=28,
        normalise_by=scale,
        bounds=(0, 1.05),
        zero_when="is_night_d+{horizon}",
        name="solar",
        **kwargs,
    )


def test_a_run_at_raise_on_the_built_in_rules_zeroes_the_night_and_keeps_the_day_under_the_proxy(data):
    run = model().run(
        data,
        *TEST,
        target="generation_solar",
        horizon=2,
        information_set=INFO,
        availability="raise",
    )
    records = run._results
    assert records and all(r["status"] == "ok" for r in records)
    issue = data.index.tz_localize(None)
    for record in records:
        row = (issue.normalize() == pd.Timestamp(record["run_date"])) & (issue.hour == record["hour"])
        night = data.loc[row, f"is_night_d+{record['horizon']}"].iloc[-1]
        cap = data.loc[row, PROXY].iloc[-1]
        if night == 1:
            assert record["prediction"] == 0.0 and record["coefficients"] == [] and record["detail"] == "zero_when"
        else:
            assert 0.0 <= record["prediction"] <= 1.05 * cap + 1e-9
    nights = sum(r.get("detail") == "zero_when" for r in records)
    assert 0 < nights < len(records)


def test_a_proxy_without_its_lag_is_refused_at_raise_and_listed_at_warn(data):
    """`generation_solar_roll14max` holds the maximum of D, whose last hour is out at 01:00 on D+1."""
    late = build(source(), lag_days=0)
    with pytest.raises(InformationSetError, match="generation_solar_roll14max"):
        model("generation_solar_roll14max").run(
            late, *TEST, target="generation_solar", horizon=1, information_set=INFO, availability="raise"
        )
    pipeline = ModelPipeline(information_set=INFO, availability="warn").add_model(model("generation_solar_roll14max"))
    with pytest.warns(UserWarning):
        report = pipeline.run(late, test=TEST, target="generation_solar", horizon=1)
    listing = report.availability()
    assert "generation_solar_roll14max" in set(listing["predictor"])


def verified(frame, scale=PROXY, **kwargs):
    pipeline = ModelPipeline(information_set=INFO).add_model(model(scale))
    return pipeline.verify(frame, *TEST, target="generation_solar", horizon=1, days=2, **kwargs)


def test_verify_is_ok_and_its_control_moves_the_day_through_the_proxy(data):
    report = verified(data)
    report.assert_ok()
    row = report.table().iloc[0]
    assert row["changed"] == 0 and row["control moved"] > 0


def rebuild(lag_days):
    def run(frame):
        return build(frame[["generation_solar", "cloud"]], lag_days)

    return run


@pytest.mark.parametrize(("lag_days", "status"), [(1, "ok"), (0, "leak")], ids=["the window ends D-1", "it ends D"])
def test_verify_with_the_inputs_rebuilt_from_the_source_finds_a_window_that_reads_d(lag_days, status):
    """The probe perturbs the generation of D after the issue and rebuilds the inputs: a window ending on D moves with it. (At
    `"warn"` the late proxy is listed, so the run goes on to the probe.)"""
    scale = "generation_solar_roll14max_d-1" if lag_days else "generation_solar_roll14max"
    pipeline = ModelPipeline(information_set=INFO, availability="warn").add_model(model(scale))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        report = pipeline.verify(
            source(),
            *TEST,
            target="generation_solar",
            horizon=1,
            days=2,
            rebuild=rebuild(lag_days),
            positive_control=status == "ok",
            rerun=False,
        )
    row = report.table().iloc[0]
    assert row["status"] == status, report.problems()
    if status == "ok":
        assert row["control moved"] > 0
