"""Information availability rules, the pre-fit check and the leakage probe (roadmap item 1)."""

import numpy as np
import pandas as pd
import pytest
import yaml

from epftoolbox2.information_set import (
    Availability,
    InformationSet,
    InformationSetError,
    Shift,
    leakage_probe,
    parse_predictor,
    perturb_unknown,
)
from epftoolbox2.models import OLSModel
from epftoolbox2.pipelines import ModelPipeline, Workflow
from tests.synthetic import PREDICTORS, calendar_shift, make_market_data

WARSAW = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="Europe/Warsaw")
# Synthetic drivers are known in advance (like calendar data or perfect forecasts); price follows the A44 default.
SYNTHETIC_RULES = {name: Availability.static() for name in ("load", "wind", "is_weekend", "is_weekday")}


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


def expand(predictors, periods=24, horizon=7):
    return {
        (p, hz): [x.replace("{horizon}", str(hz)) for x in predictors]
        for p in range(periods)
        for hz in range(1, horizon + 1)
    }


def violations(predictors, info=None, target="price", start="2024-01-10", end="2024-01-23", horizon=7, index=WARSAW):
    info = info or InformationSet("11:30", "Europe/Brussels")
    found, unknown = info.violations(expand(predictors, horizon=horizon), target, "1h", index, start, end)
    return {(v.predictor, v.period, v.horizon) for v in found}, unknown


# ---------------------------------------------------------------------------
# Names and rules
# ---------------------------------------------------------------------------


def test_parse_predictor_strips_suffixes_innermost_first():
    assert parse_predictor("price_d-1") == [("price_d-1", ()), ("price", (Shift("d", -1),))]
    chain = parse_predictor("warsaw_temperature_2m_d+1_h-24")
    assert chain[-1] == ("warsaw_temperature_2m", (Shift("d", 1), Shift("h", -24)))
    assert parse_predictor("is_holiday") == [("is_holiday", ())]


@pytest.mark.parametrize(
    "name, base, kind",
    [
        ("price_d-7", "price", "published"),
        ("load_actual_d-1", "load_actual", "realised"),
        ("generation_wind_onshore_forecast_d+1", "generation_wind_onshore_forecast", "published"),
        ("generation_wind_onshore", "generation_wind_onshore", "realised"),
        ("load_forecast_daily_min_d+3", "load_forecast_daily_min", "weekly"),
        ("is_sunday_d+2", "is_sunday", "static"),
        ("berlin_temperature_2m_d+2", "berlin_temperature_2m", "weather_run"),
    ],
)
def test_default_rules_cover_the_builtin_sources(name, base, kind):
    col = InformationSet().resolve(name)
    assert (col.base, col.rule.kind) == (base, kind)


def test_user_rules_take_precedence_and_exact_names_beat_suffix_parsing():
    info = InformationSet(
        rules={"price": Availability.realised("0h"), "wind_d+1": Availability.published("09:00", days_before=0)}
    )
    assert info.resolve("price_d-1").rule.kind == "realised"
    col = info.resolve("wind_d+1")
    assert (col.base, col.shifts, col.rule.time) == ("wind_d+1", (), "09:00")
    assert info.resolve("wind").rule is None


def test_rules_round_trip_through_yaml(tmp_path):
    info = InformationSet(
        "11:00",
        "Europe/Warsaw",
        rules={"fuel_*": Availability.daily_close("18:00"), "load": Availability.realised("2h")},
        assumptions=["entsoe_a69_by_10"],
        unknown="warn",
    )
    pipeline = ModelPipeline(information_set=info).add_model(OLSModel(["price_d-1"], name="OLS"))
    pipeline.save(tmp_path / "mp.yaml")
    loaded = ModelPipeline.load(tmp_path / "mp.yaml").information_set
    assert loaded.to_dict() == info.to_dict()
    assert loaded.resolve("fuel_ttf_d-1").rule == Availability.daily_close("18:00")


def test_invalid_settings_raise():
    with pytest.raises(ValueError, match="HH:MM"):
        InformationSet("1130")
    with pytest.raises(ValueError, match="time zone"):
        InformationSet("11:30", "Mars/Olympus")
    with pytest.raises(ValueError, match="assumption"):
        InformationSet(assumptions=["everything_is_known"])
    with pytest.raises(ValueError, match="kind"):
        Availability("sometimes")


# ---------------------------------------------------------------------------
# The check, rule by rule (issue 11:30 Brussels, Warsaw data: same wall clock)
# ---------------------------------------------------------------------------


def test_same_day_actual_load_is_unknown_after_the_morning():
    found, _ = violations(["load_actual"], horizon=1)
    # Realised 1 h after the end of the hour: hour h is known at h+2, so up to hour 9 (known 11:00).
    assert {p for _, p, _ in found} == set(range(10, 24))
    assert violations(["load_actual_d-1"])[0] == set()


def test_day_ahead_values_are_known_for_the_next_day_only():
    assert violations(["price", "load_forecast_d+1"])[0] == set()
    found, _ = violations(["price_d+1", "load_forecast_d+{horizon}"])
    assert {name for name, _, _ in found} == {"price_d+1"} | {f"load_forecast_d+{h}" for h in range(2, 8)}


def test_day_ahead_values_are_known_for_the_next_day_only_with_utc_data():
    """Same day-ahead rules (published 13:00/10:00 Brussels D-1), but UTC-indexed data: winter CET is
    UTC+1, so the last UTC hour of the day (23:00-00:00 UTC) delivers in the *next* CET calendar day. The
    delivery day must be read in the rule's own (market) zone, not floored in the data's zone, or this
    hour's value is wrongly treated as already published a full day early."""
    utc = WARSAW.tz_convert("UTC")
    found, _ = violations(["price", "load_forecast_d+1"], index=utc, horizon=1)
    assert {(name, p) for name, p, _ in found} == {("price", 23), ("load_forecast_d+1", 23)}


def test_week_ahead_load_depends_on_the_issue_weekday():
    thursday, friday = "2024-01-11", "2024-01-12"
    found, _ = violations(["load_forecast_daily_max_d+{horizon}"], start=thursday, end=thursday)
    assert {hz for _, _, hz in found} == {4, 5, 6, 7}  # next Monday (D+4) is published on Friday
    found, _ = violations(["load_forecast_daily_max_d+{horizon}"], start=friday, end=friday)
    assert found == set()  # Friday 10:00 publishes Mon-Sun of next week: D+3..D+9


def test_a69_needs_the_explicit_assumption():
    predictors = ["generation_solar_forecast", "generation_solar_forecast_d+1"]
    found, _ = violations(predictors, horizon=1)
    assert {name for name, _, _ in found} == {"generation_solar_forecast_d+1"}

    with pytest.raises(InformationSetError, match="entsoe_a69_by_10"):
        InformationSet().check(expand(predictors, horizon=1), "price", "1h", WARSAW, "2024-01-10", "2024-01-10")
    accepted = InformationSet(assumptions=["entsoe_a69_by_10"])
    assert violations(predictors, info=accepted, horizon=1)[0] == set()


def test_weather_forecasts_depend_on_the_row_time():
    found, _ = violations(["warsaw_temperature_2m_d+{horizon}"])
    # Run started about at the row time and published 6 h later: rows up to 05:00 are known by 11:30.
    assert {p for _, p, _ in found} == set(range(6, 24))
    assert {hz for _, _, hz in found} == set(range(1, 8))


def test_hourly_lags_and_static_columns():
    found, _ = violations(["load_actual_h-12", "is_holiday_d+{horizon}", "daylight_hours_d+7"], horizon=1)
    # load_actual 12 hours earlier ends 11h before the row (12h gap - 1h period) and is known 1h after that,
    # i.e. 10h before the row: that is after the 11:30 issue only for the last two hours of the day.
    assert {(name, p) for name, p, _ in found} == {("load_actual_h-12", p) for p in (22, 23)}


def test_training_target_known_too_late_is_reported():
    info = InformationSet(rules={"monthly_settlement": Availability.published("12:00", days_before=-3)})
    found, _ = violations([], info=info, target="monthly_settlement", horizon=1)
    assert found and {name for name, _, _ in found} == {"monthly_settlement"}


def test_issue_time_is_converted_to_the_data_zone():
    utc = WARSAW.tz_convert("UTC")
    # 11:30 Brussels is 10:30 UTC in winter: load_actual rows up to 08:00 UTC are known in time.
    found, _ = violations(["load_actual"], horizon=1, index=utc, start="2024-01-10", end="2024-01-10")
    assert min(p for _, p, _ in found) == 9
    # In summer it is 09:30 UTC.
    summer = pd.date_range("2024-06-01", periods=24 * 30, freq="h", tz="UTC")
    found, _ = violations(["load_actual"], horizon=1, index=summer, start="2024-06-10", end="2024-06-10")
    assert min(p for _, p, _ in found) == 8


def test_unknown_columns_follow_the_policy():
    expanded = expand(["load_d+{horizon}"], horizon=1)
    with pytest.raises(InformationSetError, match="no availability rule.*'load'"):
        InformationSet().check(expanded, "price", "1h", WARSAW, "2024-01-10", "2024-01-10")
    with pytest.warns(UserWarning, match="load"):
        InformationSet(unknown="warn").check(expanded, "price", "1h", WARSAW, "2024-01-10", "2024-01-10")
    InformationSet(unknown="ignore").check(expanded, "price", "1h", WARSAW, "2024-01-10", "2024-01-10")


def test_shipped_example_predictors_are_flagged():
    """The predictor list of model_pipeline.yaml / examples 02-05 (as of this change) uses look-ahead inputs."""
    shipped = [
        "load_actual",
        *[f"is_{d}_d+{{horizon}}" for d in ("monday", "tuesday", "wednesday", "thursday", "friday")],
        "is_saturday_d+{horizon}",
        "is_sunday_d+{horizon}",
        "is_holiday_d+{horizon}",
        "daylight_hours_d+{horizon}",
        "load_actual_d-1",
        "load_actual_d-2",
        "price_d-1",
        "price_d-2",
        "warsaw_temperature_2m_d+{horizon}",
    ]
    found, unknown = violations(shipped)
    assert unknown == set()
    assert {name for name, _, _ in found} == {"load_actual"} | {f"warsaw_temperature_2m_d+{h}" for h in range(1, 8)}


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def test_model_pipeline_checks_before_fitting(tmp_path):
    df = make_market_data(days=120).rename(columns={"load": "load_actual"})
    pipeline = ModelPipeline(issue_time="11:30").add_model(
        OLSModel(["load_actual", "price_d-1"], training_window=30, name="OLS")
    )
    with pytest.raises(InformationSetError, match="load_actual"):
        pipeline.run(df, test_start="2023-04-12", test_end="2023-04-12", horizon=1, save_dir=str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_clean_predictors_pass_and_forecasts_are_unchanged():
    df = make_market_data(days=120, noise=5.0, seed=3)
    info = InformationSet(rules=SYNTHETIC_RULES)
    model = OLSModel(PREDICTORS + ["price_d-1"], training_window=45)
    checked = model.run(df, test_start="2023-04-12", test_end="2023-04-12", horizon=2, information_set=info)
    plain = OLSModel(PREDICTORS + ["price_d-1"], training_window=45).run(
        df, test_start="2023-04-12", test_end="2023-04-12", horizon=2
    )
    assert checked._results == plain._results


def test_issue_time_shortcut_and_conflicts():
    assert ModelPipeline(issue_time="11:00").information_set.issue_time == "11:00"
    assert ModelPipeline().information_set is None
    with pytest.raises(ValueError, match="not both"):
        ModelPipeline(information_set=InformationSet(), issue_time="11:00")


def test_workflow_inline_information_set_knows_prefixed_calendar_columns(tmp_path):
    index = pd.date_range("2023-01-01", "2023-04-01", freq="1h", tz="UTC", inclusive="left")
    rng = np.random.default_rng(0)
    pd.DataFrame({"datetime": index.strftime("%Y-%m-%d %H:%M:%S"), "price": rng.normal(100, 10, len(index))}).to_csv(
        tmp_path / "prices.csv", index=False
    )
    config = {
        "data_pipeline": {
            "start": "2023-01-01",
            "end": "2023-03-31",
            "sources": [
                {"class": "CsvSource", "params": {"file_path": str(tmp_path / "prices.csv")}},
                {"class": "CalendarSource", "params": {"country": "PL", "weekday": "onehot", "prefix": "pl_"}},
            ],
            "transformers": [
                {"class": "TimezoneTransformer", "params": {"target_tz": "Europe/Warsaw"}},
                {"class": "LagTransformer", "params": {"columns": ["price"], "lags": [1, -2], "freq": "day"}},
                {"class": "LagTransformer", "params": {"columns": ["pl_is_monday"], "lags": [-1], "freq": "day"}},
            ],
        },
        "model_pipeline": {
            "test_start": "2023-03-01",
            "test_end": "2023-03-01",
            "horizon": 1,
            "information_set": {"issue_time": "11:30", "issue_timezone": "Europe/Brussels"},
            "models": [
                {
                    "class": "OLSModel",
                    "params": {"predictors": ["price_d-1", "pl_is_monday_d+1"], "training_window": 30, "name": "OLS"},
                }
            ],
        },
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))

    report = Workflow.load(path).run()
    assert report.forecast_counts().loc[0, "produced"] == 24

    # price_d+2 is not the model's own target (that's price_d+1 at horizon=1), so it reaches the information-set
    # check rather than the "predictor names the target" contract check.
    config["model_pipeline"]["models"][0]["params"]["predictors"] = ["price", "price_d+2"]
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(InformationSetError, match="price_d\\+2"):
        Workflow.load(path).run()


# ---------------------------------------------------------------------------
# Leakage probe
# ---------------------------------------------------------------------------

# The model's own boundary: training targets end at D-1, i.e. D's price is treated as unknown on D.
PRICE_KNOWN_AFTER_DELIVERY = {**SYNTHETIC_RULES, "price": Availability.published("00:00", days_before=-1)}


def _changed(new: pd.Series, old: pd.Series) -> pd.Series:
    """Element-wise "differs", but two NaNs (e.g. the D-1 lag on the series' first day) count as unchanged."""
    return ~(new.isna() & old.isna()) & (new != old)


def test_perturbation_follows_each_columns_delivery_time():
    df = make_market_data(days=40)
    info = InformationSet(rules=SYNTHETIC_RULES)
    perturbed, counts, unchecked = perturb_unknown(df, "2023-01-20", info)
    changed = _changed(perturbed["price"], df["price"])
    # A44 default: prices for 2023-01-21 are published 2023-01-20 13:00, after the 11:30 issue.
    assert df.index[changed].min() == pd.Timestamp("2023-01-21 00:00", tz="Europe/Warsaw")
    lag_changed = _changed(perturbed["price_d-1"], df["price_d-1"])
    assert df.index[lag_changed].min() == pd.Timestamp("2023-01-22 00:00", tz="Europe/Warsaw")
    assert "load" not in counts and unchecked == []


def test_probe_passes_for_the_models_information_set():
    df = make_market_data(noise=5.0, seed=7)
    info = InformationSet(rules=PRICE_KNOWN_AFTER_DELIVERY)
    for run_day in ("2023-04-12", "2023-05-02"):
        result = leakage_probe(
            lambda: OLSModel(PREDICTORS + ["price_d-1"], training_window=45), df, run_day, info, horizon=2
        )
        assert result.perturbed["price"] > 0
        result.assert_ok()


def test_probe_detects_a_predictor_known_after_issue_time():
    df = make_market_data(noise=5.0, seed=7)
    df["load_same_day"] = df["load"]
    info = InformationSet(rules={**PRICE_KNOWN_AFTER_DELIVERY, "load_same_day": Availability.realised("1h")})
    predictors = ["load_same_day", "wind_d+{horizon}", "is_weekend_d+{horizon}", "is_weekday_d+{horizon}"]

    result = leakage_probe(lambda: OLSModel(predictors, training_window=45), df, "2023-04-12", info, horizon=1)
    assert not result.ok
    assert {period for period, _ in result.changed} == set(range(10, 24))
    with pytest.raises(AssertionError, match="depend on values not known"):
        result.assert_ok()


def test_probe_matches_the_hand_built_scramble_of_the_known_answer_test():
    """The probe generalises tests/test_known_answer.py: same perturbed prices, same verdict."""
    df = make_market_data(noise=5.0, seed=7)
    run_day = "2023-04-12"
    perturbed, _, _ = perturb_unknown(df, run_day, InformationSet(rules=PRICE_KNOWN_AFTER_DELIVERY))
    future = df.index.date >= pd.Timestamp(run_day).date()
    assert (_changed(perturbed["price"], df["price"]) == future).all()
    recomputed = calendar_shift(perturbed["price"], -1)
    assert (_changed(perturbed["price_d-1"], df["price_d-1"]) == _changed(recomputed, df["price_d-1"])).all()
