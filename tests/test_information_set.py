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
    source_rules,
)
from epftoolbox2.data.sources.open_meteo import OpenMeteoSource
from epftoolbox2.models import OLSModel
from epftoolbox2.pipelines import ModelPipeline, Workflow
from tests.synthetic import PREDICTORS, calendar_shift, make_market_data

WARSAW = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="Europe/Warsaw")
# Synthetic drivers are known in advance (like calendar data or perfect forecasts); price follows the A44 default.
SYNTHETIC_RULES = {name: Availability.static() for name in ("load", "wind", "is_weekend", "is_weekday")}


def _warsaw_weather_source(**overrides) -> OpenMeteoSource:
    source = OpenMeteoSource(latitude=52.23, longitude=21.01, prefix="warsaw", columns=["temperature_2m"])
    for key, value in overrides.items():
        setattr(source, key, value)
    return source


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
    ],
)
def test_default_rules_cover_the_builtin_sources(name, base, kind):
    col = InformationSet().resolve(name)
    assert (col.base, col.rule.kind) == (base, kind)


def test_default_rules_have_no_weather_glob():
    """A `*_temperature_2m`-style suffix alone doesn't say whether a column is a forecast or an observed
    value, so weather timing is assigned only through source_rules (see the tests below), never by the
    unprefixed defaults."""
    col = InformationSet().resolve("berlin_temperature_2m_d+2")
    assert col.rule is None


def test_as_of_rule_ignores_a_leading_day_shift():
    """`as_of` is known at a fixed time on the issue day D; a leading `_d+k` must not change that."""
    info = InformationSet(rules={"x": Availability.as_of("09:00", "Europe/Brussels")})
    assert info.resolve("x").rule.kind == "as_of"
    early, _ = violations(["x", "x_d+1", "x_d+3"], info=info, horizon=3, start="2024-01-10", end="2024-01-10")
    assert early == set()  # 09:00 Brussels is before the 11:30 issue, for every shift alike

    late_info = InformationSet(rules={"x": Availability.as_of("12:00", "Europe/Brussels")})
    late, _ = violations(["x", "x_d+1", "x_d+3"], info=late_info, horizon=3, start="2024-01-10", end="2024-01-10")
    assert {name for name, _, _ in late} == {"x", "x_d+1", "x_d+3"}  # all fail identically (12:00 > 11:30 issue)


def test_source_rules_as_of_for_a_weather_source_with_issue_time_vintage():
    """`vintage`/`issue_time`/`timezone`/`publication_delay_h` are added to `OpenMeteoSource` on another
    branch; source_rules reads them with getattr so this works with a stub that has them."""
    source = _warsaw_weather_source(vintage="issue_time", issue_time="09:00", timezone="Europe/Brussels")
    rule = source_rules([source])["warsaw_temperature_2m"]
    assert (rule.kind, rule.time, rule.timezone) == ("as_of", "09:00", "Europe/Brussels")


def test_source_rules_weather_run_uses_the_sources_publication_delay():
    source = _warsaw_weather_source(publication_delay_h=3)  # no vintage: falls back to weather_run
    rule = source_rules([source])["warsaw_temperature_2m"]
    assert (rule.kind, rule.delay) == ("weather_run", "3h")


def test_source_rules_weather_run_defaults_without_a_publication_delay():
    rule = source_rules([_warsaw_weather_source()])["warsaw_temperature_2m"]
    assert (rule.kind, rule.delay) == ("weather_run", "6h")


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
    info = InformationSet("11:30", "Europe/Brussels").with_source_rules([_warsaw_weather_source()])
    found, _ = violations(["warsaw_temperature_2m_d+{horizon}"], info=info)
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
    info = InformationSet("11:30", "Europe/Brussels").with_source_rules([_warsaw_weather_source()])
    found, unknown = violations(shipped, info=info)
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


def test_target_from_issue_extends_perturbation_to_the_issue_day_itself():
    """The default day-ahead rule for 'price' already perturbs D+1 onward (see the delivery-time test
    above); `target_from_issue` additionally perturbs day D's own rows, matching the model's stricter
    boundary (training targets end at D-1) even though a real day-ahead market would have published D's
    price by D-1."""
    df = make_market_data(days=40)
    info = InformationSet(rules=SYNTHETIC_RULES)  # unmodified default ("published") rule for "price"
    run_day = "2023-01-20"

    _, counts, _ = perturb_unknown(df, run_day, info)
    _, counts_strict, _ = perturb_unknown(df, run_day, info, target_from_issue="price")

    assert counts_strict["price"] == counts["price"] + 24  # the 24 hours of the issue day itself


def test_check_target_boundary_lets_the_probe_use_the_unmodified_default_rule():
    """With `check_target_boundary=True`, `leakage_probe` enforces the model's own D-1 boundary on the
    target regardless of its Availability rule, so a compliant model passes without the hand-built
    PRICE_KNOWN_AFTER_DELIVERY override used elsewhere in this file."""
    df = make_market_data(noise=5.0, seed=7)
    info = InformationSet(rules=SYNTHETIC_RULES)  # unmodified default rule for "price"
    result = leakage_probe(
        lambda: OLSModel(PREDICTORS + ["price_d-1"], training_window=45),
        df,
        "2023-04-12",
        info,
        horizon=2,
        check_target_boundary=True,
    )
    result.assert_ok()


def test_source_rebuild_catches_a_transformer_leak_that_the_final_column_probe_misses():
    """A transformer that shifts by 24 *elapsed* hours instead of one *calendar* day reads the wrong raw
    row into `load_forecast_d+1` across a DST spring-forward (Europe/Warsaw, 2024-03-31): the issue row
    2024-03-30 23:00 ends up holding the raw source's 2024-04-01 00:00 value instead of 2024-03-31 23:00's.
    By its parsed name, `load_forecast_d+1` is a day-ahead forecast published 10:00 on the issue day D,
    known well before the 11:30 issue, so perturbing the already-built column never touches that cell.
    Perturbing the raw source and rebuilding does, because the raw 2024-04-01 00:00 cell it actually reads
    from is correctly identified as not yet known.
    """
    tz = "Europe/Warsaw"
    index = pd.date_range("2024-03-15", "2024-04-05", freq="1h", tz=tz, inclusive="left")
    load_forecast = pd.Series(1000 + 50 * np.sin(2 * np.pi * np.arange(len(index)) / 24), index=index)
    rng = np.random.default_rng(0)
    source = pd.DataFrame({"load_forecast": load_forecast})
    source["price"] = 20 + 0.05 * calendar_shift(load_forecast, 1) + rng.normal(0, 0.1, len(index))

    def rebuild_with_elapsed_shift(raw: pd.DataFrame) -> pd.DataFrame:
        out = raw.copy()
        out["load_forecast_d+1"] = raw["load_forecast"].shift(-24, freq="h").reindex(raw.index)
        return out

    info = InformationSet(rules={"price": Availability.static()})  # isolate the load_forecast leak
    run_day = "2024-03-30"

    def make_model():
        return OLSModel(["load_forecast_d+1"], training_window=10)

    built = rebuild_with_elapsed_shift(source)
    final_only = leakage_probe(make_model, built, run_day, info, horizon=1)
    assert final_only.ok

    from_source = leakage_probe(make_model, source, run_day, info, horizon=1, rebuild=rebuild_with_elapsed_shift)
    assert not from_source.ok
    assert (23, 1) in from_source.changed
