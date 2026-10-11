"""Literature predictor presets (epftoolbox2.presets) and the data pipeline they need."""

import numpy as np
import pandas as pd
import pytest
import yaml

from epftoolbox2.data.transformers import (
    DailyProfileTransformer,
    DailyStatsTransformer,
    DSTNormalizeTransformer,
    LagTransformer,
)
from epftoolbox2.models import LassoCVModel, OLSModel
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.presets import WEEKDAYS, expert_predictors, lear_predictors, profile_columns
from tests.synthetic import make_market_data

EXOGENOUS = ["load", "wind"]


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


def expand(predictors, horizon):
    return [p.replace("{horizon}", str(horizon)) for p in predictors]


def test_profile_columns_match_the_transformer():
    assert profile_columns("price") == [f"price_h{h:02d}" for h in range(24)]
    assert profile_columns("price", freq="15min")[:2] == ["price_s00", "price_s01"]
    assert len(profile_columns("price", freq="15min")) == 96


def test_lear_has_247_inputs_for_two_exogenous_series():
    predictors = lear_predictors(exogenous=EXOGENOUS)
    assert len(predictors) == 247
    assert len(set(predictors)) == 247
    assert "price_h05" in predictors  # day-ahead prices of the issue day = LEAR's d-1
    assert "price_h05_d-6" in predictors  # LEAR's d-7
    assert "load_h05_d+{horizon}" in predictors and "load_h05" in predictors and "load_h05_d-6" in predictors
    assert "is_monday_d+{horizon}" in predictors


def test_expert_inputs():
    predictors = expert_predictors(exogenous=["load"])
    assert predictors[:6] == ["price", "price_d-1", "price_d-6", "price_day_min", "price_day_max", "price_day_last"]
    assert "load_d+{horizon}" in predictors
    assert [p for p in predictors if p.startswith("is_")] == [f"is_{d}_d+{{horizon}}" for d in WEEKDAYS]
    original = expert_predictors(exogenous=["load"], weekdays=["monday", "saturday", "sunday"])
    assert [p for p in original if p.startswith("is_")] == [
        "is_monday_d+{horizon}",
        "is_saturday_d+{horizon}",
        "is_sunday_d+{horizon}",
    ]


def test_presets_are_plain_strings_that_survive_yaml():
    for predictors in (lear_predictors(exogenous=EXOGENOUS), expert_predictors()):
        assert all(isinstance(p, str) for p in predictors)
        assert yaml.safe_load(yaml.safe_dump(predictors)) == predictors


def test_unknown_weekday_raises():
    with pytest.raises(ValueError, match="funday"):
        expert_predictors(weekdays=["funday"])


# --------------------------------------------------------------------------- the documented pipeline


def weekday_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({f"is_{d}": (index.dayofweek == i).astype(float) for i, d in enumerate(WEEKDAYS)}, index=index)


def build_features(raw: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """The transformer chain from docs/model-pipeline/presets.mdx, on synthetic data (hourly)."""
    base = raw[["price", *EXOGENOUS]]
    df = DSTNormalizeTransformer().transform(base)
    df = pd.concat([df, weekday_dummies(df.index)], axis=1)
    df = DailyProfileTransformer(columns=["price", *EXOGENOUS]).transform(df)
    df = DailyStatsTransformer(columns="price", stats=["min", "max", "last"]).transform(df)
    leads = list(range(-1, -horizon - 1, -1))
    df = LagTransformer(columns="price", lags=[1, 6], freq="day").transform(df)
    df = LagTransformer(columns=profile_columns("price"), lags=[1, 2, 6], freq="day").transform(df)
    exog_profiles = [c for x in EXOGENOUS for c in profile_columns(x)]
    df = LagTransformer(columns=exog_profiles, lags=[*leads, 6], freq="day").transform(df)
    df = LagTransformer(columns=EXOGENOUS, lags=leads, freq="day").transform(df)
    return LagTransformer(columns=[f"is_{d}" for d in WEEKDAYS], lags=leads, freq="day").transform(df)


@pytest.mark.parametrize("horizon", [1, 2])
def test_documented_pipeline_produces_every_preset_column(horizon):
    df = build_features(make_market_data(days=40, seed=2), horizon=horizon)
    for predictors in (lear_predictors(exogenous=EXOGENOUS), expert_predictors(exogenous=EXOGENOUS)):
        for h in range(1, horizon + 1):
            missing = [p for p in expand(predictors, h) if p not in df.columns]
            assert missing == []


@pytest.mark.parametrize(
    "model_cls, predictors",
    [
        (OLSModel, expert_predictors(exogenous=EXOGENOUS)),
        (LassoCVModel, lear_predictors(exogenous=EXOGENOUS)),
    ],
)
def test_forecasts_ignore_prices_after_the_issue_day(model_cls, predictors):
    raw = make_market_data(start="2023-02-01", days=90, noise=5.0, seed=4)
    run_day = "2023-03-26"  # spring DST switch
    scrambled = raw.copy()
    later = raw.index.date > pd.Timestamp(run_day).date()
    scrambled.loc[later, "price"] = 1e4 + np.arange(later.sum())

    def predictions(data):
        kwargs = {"cv": 3} if model_cls is LassoCVModel else {}
        model = model_cls(predictors, training_window=30, name="m", **kwargs)
        report = (
            ModelPipeline().add_model(model).run(build_features(data), test_start=run_day, test_end=run_day, horizon=1)
        )
        (_, details), *_ = report.iter_details()
        return details.set_index("hour")["prediction"]

    original, perturbed = predictions(raw), predictions(scrambled)
    assert len(original) == 24
    pd.testing.assert_series_equal(original, perturbed)
