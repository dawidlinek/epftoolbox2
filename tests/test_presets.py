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
from epftoolbox2.data.sources import CalendarSource
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.presets import (
    WEEKDAYS,
    exaa_profile,
    expert_predictors,
    lear_predictors,
    live_defaults,
    profile_columns,
    week_ahead_predictors,
)
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


def test_lago_s_order_is_the_original_s_slot_major_order_of_the_same_names():
    lago = lear_predictors(exogenous=EXOGENOUS, order="lago")
    assert sorted(lago) == sorted(lear_predictors(exogenous=EXOGENOUS)) and len(set(lago)) == 247
    # `LEAR._build_and_split_XYs`: per hour the prices of d-1, d-2, d-3, d-7; then per hour each exogenous series for d-1,
    # each for d-7, each for d; then the dummies
    expected = [f"price_h{h:02d}{lag}" for h in range(24) for lag in ("", "_d-1", "_d-2", "_d-6")]
    for h in range(24):
        expected += [f"{x}_h{h:02d}{lag}" for lag in ("", "_d-6") for x in EXOGENOUS]
        expected += [f"{x}_h{h:02d}_d+{{horizon}}" for x in EXOGENOUS]
    expected += [f"is_{d}_d+{{horizon}}" for d in WEEKDAYS]
    assert lago == expected
    assert lago[96:102] == [
        "load_h00",
        "wind_h00",
        "load_h00_d-6",
        "wind_h00_d-6",
        "load_h00_d+{horizon}",
        "wind_h00_d+{horizon}",
    ]
    # the exogenous series keep the order they are given in, which the original's file order decides
    swapped = lear_predictors(exogenous=["wind", "load"], order="lago")
    assert swapped[96:98] == ["wind_h00", "load_h00"]


def test_the_default_order_is_the_library_s_as_before():
    library = lear_predictors(exogenous=EXOGENOUS)
    assert library == lear_predictors(exogenous=EXOGENOUS, order="library")
    assert library[:3] == ["price_h00", "price_h01", "price_h02"] and library[96:98] == [
        "load_h00_d+{horizon}",
        "load_h01_d+{horizon}",
    ]
    with pytest.raises(ValueError, match="order"):
        lear_predictors(exogenous=EXOGENOUS, order="hour-major")


def test_lago_s_order_has_every_slot_of_a_15_minute_day():
    lago = lear_predictors(exogenous=["load"], freq="15min", order="lago", calendar_prefix="cal_")
    assert len(lago) == 4 * 96 + 3 * 96 + 7
    assert lago[:5] == ["price_s00", "price_s00_d-1", "price_s00_d-2", "price_s00_d-6", "price_s01"]
    assert (
        lago[384:387] == ["load_s00", "load_s00_d-6", "load_s00_d+{horizon}"]
        and lago[-1] == "cal_is_sunday_d+{horizon}"
    )


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


# --------------------------------------------------------------------------- EXAA profile


def test_exaa_profile_names():
    assert exaa_profile() == [f"price_exaa_h{h:02d}_d+{{horizon}}" for h in range(24)]
    assert exaa_profile(prefix="de")[0] == "de_price_exaa_h00_d+{horizon}"
    assert exaa_profile(prefix="de_") == exaa_profile(prefix="de")
    quarter = exaa_profile("15min")
    assert len(quarter) == 96 and quarter[:2] == ["price_exaa_s00_d+{horizon}", "price_exaa_s01_d+{horizon}"]


def test_exaa_profile_is_what_the_transformers_produce():
    index = pd.date_range("2024-01-01", periods=24 * 12, freq="1h", tz="Europe/Warsaw")
    df = pd.DataFrame({"de_price_exaa": np.arange(len(index), dtype=float)}, index=index)
    df = DailyProfileTransformer(columns="de_price_exaa").transform(df)
    df = LagTransformer(columns=profile_columns("de_price_exaa"), lags=[-1], freq="day").transform(df)
    assert [c for c in exaa_profile(prefix="de") if c.replace("{horizon}", "1") not in df.columns] == []


# --------------------------------------------------------------------------- week-ahead lists

# Section 3.5 of FORECASTING_PLAN.md and docs/.../week-ahead.mdx, written out once more here on purpose: a change to the
# preset must change this copy too.
PLAN_CALENDAR_X = (
    ["is_holiday"]
    + [f"is_holiday_d-{k}" for k in range(1, 8)]
    + [f"is_holiday_d+{k}" for k in range(1, 10)]
    + [f"is_{d}_d+{{horizon}}" for d in WEEKDAYS]
    + ["daylight_hours_d+{horizon}"]
)
PLAN_REGIONAL_NWP = (
    PLAN_CALENDAR_X
    + [f"load_actual_h-{h}" for h in range(1, 169)]
    + [f"load_forecast_h+{h}" for h in range(1, 36)]
    + ["load_forecast"]
    + [f"actual_daily_{s}_d-{k}" for s in ("min", "max") for k in range(1, 8)]
    + [f"load_forecast_daily_{s}_d+{k}" for s in ("min", "max") for k in range(3, 10)]
    + [f"all_avg_{v}_dmean_d+{{horizon}}" for v in ("2t", "wind_speed", "rh", "tcc", "sp", "ssr_inc", "tp")]
    + ["all_avg_hdd_d+{horizon}", "all_avg_cdd_d+{horizon}"]
)


def plan_expert(stat):
    return [
        "is_monday_d+{horizon}",
        "is_saturday_d+{horizon}",
        "is_sunday_or_holiday_d+{horizon}",
        f"load_forecast_daily_{stat}_d+0",
        f"load_forecast_daily_{stat}_d+{{horizon}}",
        f"actual_daily_{stat}_d-1",
        f"actual_daily_{stat}_d-6",
        "capital_2t_dmean_d+{horizon}",
    ]


def test_week_ahead_lists_equal_the_plan_s():
    assert week_ahead_predictors("calendar_x") == PLAN_CALENDAR_X
    assert week_ahead_predictors("regional_nwp", study=True) == PLAN_REGIONAL_NWP
    # the default is the study's list without the one late name, in the same order
    assert week_ahead_predictors() == [n for n in PLAN_REGIONAL_NWP if n != "load_actual_h-1"]
    assert week_ahead_predictors("regional_nwp") == week_ahead_predictors()
    for stat in ("min", "max"):
        assert week_ahead_predictors("expert", stat=stat) == plan_expert(stat)


def test_week_ahead_sizes_are_the_documented_ones():
    assert len(week_ahead_predictors("calendar_x")) == 25
    nwp = week_ahead_predictors("regional_nwp")
    assert len(nwp) == len(set(nwp)) == 265 and "load_actual_h-1" not in nwp
    study = week_ahead_predictors("regional_nwp", study=True)
    assert len(study) == len(set(study)) == 266 and "load_actual_h-1" in study
    assert len(week_ahead_predictors("expert", stat="min")) == 8


def test_week_ahead_lists_are_plain_strings_that_survive_yaml():
    for lst in (week_ahead_predictors("regional_nwp"), week_ahead_predictors("expert", stat="max")):
        assert all(isinstance(p, str) for p in lst)
        assert yaml.safe_load(yaml.safe_dump(lst)) == lst


def test_week_ahead_arguments_are_checked():
    with pytest.raises(ValueError, match="which"):
        week_ahead_predictors("lasso")
    with pytest.raises(ValueError, match="stat"):
        week_ahead_predictors("expert")
    with pytest.raises(ValueError, match="stat"):
        week_ahead_predictors("expert", stat="mean")
    with pytest.raises(ValueError, match="stat"):
        week_ahead_predictors("regional_nwp", stat="min")
    with pytest.raises(ValueError, match="study"):
        week_ahead_predictors("calendar_x", study=True)


# Names no library component builds from its source alone: weather averaged over a market's regions and read at its capital
# (the study's own frame), the Sunday-or-holiday flag (a calendar column the user derives), and the unshifted TSO forecast
# of the issue row's own date, which `LagTransformer` skips (lag 0). The week-ahead page says where they come from.
STUDY_FRAME_ONLY = {
    *(f"all_avg_{v}_dmean_d+{{horizon}}" for v in ("2t", "wind_speed", "rh", "tcc", "sp", "ssr_inc", "tp")),
    "all_avg_hdd_d+{horizon}",
    "all_avg_cdd_d+{horizon}",
    "capital_2t_dmean_d+{horizon}",
    "is_sunday_or_holiday_d+{horizon}",
    "load_forecast_daily_min_d+0",
    "load_forecast_daily_max_d+0",
}


def week_ahead_frame() -> pd.DataFrame:
    """The page's daily-frame chain on a short hourly frame: CalendarSource columns, hourly lags and day lags."""
    index = pd.date_range("2024-01-01", periods=24 * 30, freq="1h", tz="Europe/Warsaw")
    calendar = CalendarSource("PL", holidays="binary", weekday="onehot", daylight=True).fetch(
        index[0].tz_convert("UTC"), index[-1].tz_convert("UTC")
    )
    df = calendar.loc[:, ["is_holiday", *[f"is_{d}" for d in WEEKDAYS], "daylight_hours"]].tz_convert("Europe/Warsaw")
    df = df.reindex(index)
    rng = np.random.default_rng(0)
    for column in ("load_actual", "load_forecast"):
        df[column] = rng.normal(size=len(index))
    for stat in ("min", "max"):
        df[f"actual_daily_{stat}"] = rng.normal(size=len(index))
        df[f"load_forecast_daily_{stat}"] = rng.normal(size=len(index))
    df = LagTransformer(columns=["load_actual"], lags=list(range(1, 169)), freq="hour").transform(df)
    df = LagTransformer(columns=["load_forecast"], lags=list(range(-1, -36, -1)), freq="hour").transform(df)
    df = LagTransformer(columns=["is_holiday"], lags=[*range(1, 8), *range(-1, -10, -1)], freq="day").transform(df)
    day_columns = [*[f"is_{d}" for d in WEEKDAYS], "daylight_hours"]
    df = LagTransformer(columns=day_columns, lags=list(range(-1, -10, -1)), freq="day").transform(df)
    df = LagTransformer(columns=["actual_daily_min", "actual_daily_max"], lags=list(range(1, 8)), freq="day").transform(
        df
    )
    forecasts = ["load_forecast_daily_min", "load_forecast_daily_max"]
    return LagTransformer(columns=forecasts, lags=list(range(-1, -10, -1)), freq="day").transform(df)


@pytest.mark.parametrize("horizon", [3, 9])
def test_week_ahead_names_are_what_the_library_builds(horizon):
    df = week_ahead_frame()
    lists = [week_ahead_predictors("calendar_x"), week_ahead_predictors("regional_nwp", study=True)]
    lists += [week_ahead_predictors("expert", stat=s) for s in ("min", "max")]
    for predictors in lists:
        missing = {p for p in predictors if p.replace("{horizon}", str(horizon)) not in df.columns}
        assert missing <= STUDY_FRAME_ONLY, sorted(missing - STUDY_FRAME_ONLY)[:10]
    # and the exception set is real: all of it appears in the lists
    everything = {p for lst in lists for p in lst}
    assert STUDY_FRAME_ONLY <= everything


def test_the_default_list_has_nothing_late_at_a_noon_issue_and_the_study_s_has_one_name():
    """By the rules of the docs page: a Friday 12:00 Warsaw issue refuses the study's list on exactly `load_actual_h-1`."""
    index = pd.date_range("2024-10-01 12:00", periods=40, freq="1D", tz="Europe/Warsaw")
    info = InformationSet(
        "12:00",
        "Europe/Warsaw",
        rules={
            "actual_daily_*": Availability.realised("1h"),
            "all_avg_*": Availability.weather_cutoff("00", "8h"),
            "capital_*": Availability.weather_cutoff("00", "8h"),
            "is_sunday_or_holiday": Availability.static(),
        },
    )

    def late(names):
        expanded = {(0, 3): [n.replace("{horizon}", "3") for n in names]}
        found, _ = info.violations(
            expanded,
            "actual_daily_max",
            "1D",
            index,
            "2024-10-25",
            "2024-10-25",
            check_target=False,
        )
        return sorted({v.predictor for v in found})

    assert late(week_ahead_predictors("regional_nwp")) == []
    assert late(week_ahead_predictors("regional_nwp", study=True)) == ["load_actual_h-1"]
    for stat in ("min", "max"):
        assert late(week_ahead_predictors("expert", stat=stat)) == []


# --------------------------------------------------------------------------- live defaults


def test_live_defaults_follow_the_docs_issue_times():
    assert live_defaults("DE_LU") == {
        "issue_time": "11:45",
        "issue_timezone": "Europe/Berlin",
        "deadline": "11:55",
        "wait_for": {"load_forecast": "10:45"},
    }
    assert live_defaults("PL")["issue_timezone"] == "Europe/Warsaw"
    assert live_defaults("PL")["wait_for"] == {"pse_load_fcst": "10:45"}
    assert live_defaults("BE")["issue_timezone"] == "Europe/Brussels"
    assert live_defaults("AT", exaa=True)["wait_for"] == {"load_forecast": "10:45", "price_exaa": "11:40"}


def test_live_defaults_are_plain_data_and_a_fresh_dict_each_time():
    d = live_defaults("DE_LU")
    assert yaml.safe_load(yaml.safe_dump(d)) == d
    d["wait_for"]["x"] = "1:00"
    assert "x" not in live_defaults("DE_LU")["wait_for"]


def test_live_defaults_unknown_zone_and_exaa_outside_at_de():
    with pytest.raises(ValueError, match="XX"):
        live_defaults("XX")
    with pytest.raises(ValueError, match="exaa"):
        live_defaults("PL", exaa=True)
