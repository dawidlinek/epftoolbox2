"""Information availability rules, the pre-fit check and the leakage probe (roadmap item 1)."""

import warnings

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
from epftoolbox2.data.sources.entsoe import EntsoeSource
from epftoolbox2.data.transformers import DailyProfileTransformer, DailyStatsTransformer, LagTransformer
from epftoolbox2.data.sources.open_meteo import OpenMeteoSource
from epftoolbox2.models import OLSModel
from epftoolbox2.pipelines import DataPipeline, ModelPipeline, Workflow
from epftoolbox2.presets import profile_columns
from tests.synthetic import PREDICTORS, calendar_shift, make_market_data

WARSAW = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="Europe/Warsaw")
# Spans 2024-03-31 (spring-forward, 23 rows that day) and 2024-10-27 (fall-back, 25 rows that day).
DST_SPRING_WARSAW = pd.date_range("2024-03-20", periods=24 * 20, freq="h", tz="Europe/Warsaw")
DST_FALL_WARSAW = pd.date_range("2024-10-15", periods=24 * 20, freq="h", tz="Europe/Warsaw")
# Synthetic drivers are known in advance (like calendar data or perfect forecasts); price follows the A44 default.
SYNTHETIC_RULES = {name: Availability.static() for name in ("load", "wind", "is_weekend", "is_weekday")}


def _warsaw_weather_source(**overrides) -> OpenMeteoSource:
    params = {
        "latitude": 52.23,
        "longitude": 21.01,
        "prefix": "warsaw",
        "columns": ["temperature_2m"],
        "timezone": "Europe/Warsaw",
    }
    params.update(overrides)
    return OpenMeteoSource(**params)


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
    info = info or InformationSet("11:45", "Europe/Brussels")
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
    assert early == set()  # 09:00 Brussels is before the 11:45 issue, for every shift alike

    late_info = InformationSet(rules={"x": Availability.as_of("12:00", "Europe/Brussels")})
    late, _ = violations(["x", "x_d+1", "x_d+3"], info=late_info, horizon=3, start="2024-01-10", end="2024-01-10")
    assert {name for name, _, _ in late} == {"x", "x_d+1", "x_d+3"}  # all fail identically (12:00 > 11:45 issue)


def test_source_rules_weather_cutoff_for_a_weather_source():
    """`run`/`model` are read with `getattr`, so this works with a stub that has them."""
    source = _warsaw_weather_source(run="09", model="ecmwf_ifs")
    rule = source_rules([source])["warsaw_temperature_2m"]
    assert (rule.kind, rule.time, rule.delay, rule.timezone) == ("weather_cutoff", "09:00", "9.0h", "UTC")


def test_source_rules_weather_cutoff_uses_the_models_publication_delay():
    # ICON 4.5h, unknown models the conservative 10h default (see publication_delay_hours).
    icon = source_rules([_warsaw_weather_source(model="icon_seamless")])["warsaw_temperature_2m"]
    assert icon.delay == "4.5h"
    unknown = source_rules([_warsaw_weather_source(model="some_unpublished_model")])["warsaw_temperature_2m"]
    assert unknown.delay == "10.0h"


def test_default_run_passes_the_cutoff_check_at_1145_but_not_1000_brussels():
    """JMA's default publication delay (9.6h, see publication_delay_hours) makes the default run="00" cutoff
    (00:00 UTC + 9.6h = 09:36 UTC = 10:36 Brussels in winter) known in time for an 11:45 Brussels issue but not
    for a 10:00 one. The error must then suggest an earlier run or a later issue time."""
    source = _warsaw_weather_source()  # model="jma_seamless", run="00" (both defaults)

    late_issue = InformationSet("11:45", "Europe/Brussels").with_source_rules([source])
    found, _ = violations(["warsaw_temperature_2m_d+{horizon}"], info=late_issue)
    assert found == set()

    early_issue = InformationSet("10:00", "Europe/Brussels").with_source_rules([source])
    found, _ = violations(["warsaw_temperature_2m_d+{horizon}"], info=early_issue)
    assert found != set()
    with pytest.raises(InformationSetError, match="earlier run or a later issue time"):
        early_issue.check(
            expand(["warsaw_temperature_2m_d+{horizon}"]), "price", "1h", WARSAW, "2024-01-10", "2024-01-10"
        )


SUMMER_WARSAW = pd.date_range("2024-07-01", periods=24 * 40, freq="h", tz="Europe/Warsaw")


@pytest.mark.parametrize(
    "index, start, end",
    [(WARSAW, "2024-01-10", "2024-01-23"), (SUMMER_WARSAW, "2024-07-10", "2024-07-23")],
    ids=["winter", "summer"],
)
def test_default_run_passes_the_cutoff_check_at_1145_brussels_all_year(index, start, end):
    """JMA's 00 UTC run is available 9.6h later, 09:36 UTC. An 11:45 Brussels issue is 09:45 UTC in summer and
    10:45 UTC in winter, so the default model passes in both seasons."""
    info = InformationSet("11:45", "Europe/Brussels").with_source_rules([_warsaw_weather_source()])
    found, _ = violations(["warsaw_temperature_2m_d+{horizon}"], info=info, start=start, end=end, index=index)
    assert found == set()


def test_issue_time_defaults_to_1145_brussels():
    info = InformationSet()
    assert (info.issue_time, info.issue_timezone) == ("11:45", "Europe/Brussels")


@pytest.mark.parametrize(
    "index, start, end",
    [(WARSAW, "2024-01-10", "2024-01-23"), (SUMMER_WARSAW, "2024-07-10", "2024-07-23")],
    ids=["winter", "summer"],
)
def test_default_information_set_accepts_the_default_weather_model_all_year(index, start, end):
    """`InformationSet()` with no arguments, plus the default JMA `run="00"` source, flags nothing in either season."""
    info = InformationSet().with_source_rules([_warsaw_weather_source()])
    found, _ = violations(["warsaw_temperature_2m_d+{horizon}"], info=info, start=start, end=end, index=index)
    assert found == set()


def test_jma_run_00_is_not_known_by_1130_in_summer():
    """JMA's 9.6h delay puts the 00 UTC run at 09:36 UTC, i.e. 11:36 CEST: an 11:30 summer issue is too early."""
    info = InformationSet("11:30", "Europe/Brussels").with_source_rules([_warsaw_weather_source()])
    found, _ = violations(
        ["warsaw_temperature_2m_d+{horizon}"], info=info, start="2024-07-10", end="2024-07-23", index=SUMMER_WARSAW
    )
    assert found != set()


def test_a_day_lag_of_a_weather_column_is_read_from_the_next_days_run():
    """`LagTransformer(lags=[-1])` on `warsaw_temperature_2m_d+1` names its output `..._d+1_d+1`, and its value on
    day D is the column's value on D+1, i.e. from D+1's run. The rule is fixed to the run of the day a value is read
    from, so a shift after the source's own `_d+k` moves the issue day, in either direction."""
    frame = pd.DataFrame({"warsaw_temperature_2m_d+1": np.arange(len(WARSAW), dtype=float)}, index=WARSAW)
    lagged = LagTransformer(columns=["warsaw_temperature_2m_d+1"], lags=[-1, 1], freq="day").transform(frame)
    forward, backward = "warsaw_temperature_2m_d+1_d+1", "warsaw_temperature_2m_d+1_d-1"
    assert {forward, backward} <= set(lagged.columns)

    info = InformationSet("11:45", "Europe/Brussels").with_source_rules([_warsaw_weather_source()])
    names = [
        "warsaw_temperature_2m_d+1",
        "warsaw_temperature_2m_d+3",
        forward,
        "warsaw_temperature_2m_d+2_d+1",
        backward,
    ]
    found, _ = violations(names, info=info, horizon=1)
    assert {name for name, _, _ in found} == {forward, "warsaw_temperature_2m_d+2_d+1"}
    assert {p for name, p, _ in found if name == forward} == set(range(24))
    # D-1's run is known long before D's issue time, even for a run published late in the evening.
    late_run = InformationSet("11:45", "Europe/Brussels").with_source_rules([_warsaw_weather_source(run="18")])
    found, _ = violations(["warsaw_temperature_2m_d+1", backward], info=late_run, horizon=1)
    assert {name for name, _, _ in found} == {"warsaw_temperature_2m_d+1"}


def test_a_forward_hour_lag_of_a_weather_column_is_read_from_the_next_days_run_after_midnight():
    """`x_d+1_h+3` at 22:00 is the column's value at 01:00 the next day, which belongs to the next day's run."""
    info = InformationSet("11:45", "Europe/Brussels").with_source_rules([_warsaw_weather_source()])
    found, _ = violations(
        ["warsaw_temperature_2m_d+1_h+3", "warsaw_temperature_2m_d+1_h-3", "warsaw_temperature_2m_d+1_h-24"],
        info=info,
        horizon=1,
    )
    assert {(name, p) for name, p, _ in found} == {("warsaw_temperature_2m_d+1_h+3", p) for p in (21, 22, 23)}


def test_a_day_lag_of_an_as_of_column_is_read_from_the_next_day():
    info = InformationSet(rules={"x": Availability.as_of("09:00", "Europe/Brussels")})
    found, _ = violations(["x", "x_d+1_d+1", "x_d+1_d-1", "x_d-1"], info=info, horizon=1)
    assert {name for name, _, _ in found} == {"x_d+1_d+1"}


UTC_SUMMER = pd.date_range("2024-06-20", periods=24 * 30, freq="h", tz="UTC")


def test_a_utc_index_with_weather_columns_is_refused():
    """A pipeline without `TimezoneTransformer` leaves the index in UTC. The row at 2024-07-03 22:00Z belongs to
    Warsaw's 07-04 and so to its 07-04 00Z run, published about 09:36Z the next day, but on the UTC index it is a
    row of 07-03. The check would read the wrong day's run and pass, so it refuses such an index."""
    info = InformationSet().with_source_rules([_warsaw_weather_source()])
    names = ["warsaw_temperature_2m_d+{horizon}"]
    with pytest.raises(InformationSetError, match="TimezoneTransformer") as raised:
        violations(names, info=info, index=UTC_SUMMER, start="2024-07-01", end="2024-07-05")
    assert "'Europe/Warsaw'" in str(raised.value) and "'UTC'" in str(raised.value)
    assert "DataPipeline(timezone='Europe/Warsaw')" in str(raised.value)
    with pytest.raises(InformationSetError, match="TimezoneTransformer"):
        info.check(expand(names), "price", "1h", UTC_SUMMER, "2024-07-01", "2024-07-05")


def test_a_utc_index_with_weather_columns_is_refused_by_the_probe_too():
    info = InformationSet().with_source_rules([_warsaw_weather_source()])
    df = pd.DataFrame({"warsaw_temperature_2m_d+1": np.arange(len(UTC_SUMMER), dtype=float)}, index=UTC_SUMMER)
    with pytest.raises(InformationSetError, match="TimezoneTransformer"):
        perturb_unknown(df, "2024-07-03", info)


def test_a_utc_index_with_weather_columns_is_refused_before_a_model_runs(tmp_path):
    # No day-lag columns, so the model's own UTC checks have nothing to object to: only the weather column does.
    df = make_market_data(days=120, noise=5.0, seed=3, start="2024-05-01", tz="UTC").drop(columns=["price_d-1"])
    df["warsaw_temperature_2m_d+1"] = np.random.default_rng(1).normal(15, 5, len(df))
    info = InformationSet(rules=SYNTHETIC_RULES).with_source_rules([_warsaw_weather_source()])
    pipeline = ModelPipeline(information_set=info).add_model(
        OLSModel(["warsaw_temperature_2m_d+1", "wind_d+1"], training_window=30, name="OLS")
    )
    with pytest.raises(InformationSetError, match="TimezoneTransformer"):
        pipeline.run(df, test_start="2024-07-03", test_end="2024-07-03", horizon=1, save_dir=str(tmp_path))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("zone", ["Europe/Warsaw", "Europe/Berlin"])
def test_an_index_in_the_market_zone_or_one_with_its_wall_clock_is_accepted(zone):
    index = pd.date_range("2024-06-20", periods=24 * 30, freq="h", tz=zone)
    info = InformationSet().with_source_rules([_warsaw_weather_source()])
    found, _ = violations(
        ["warsaw_temperature_2m_d+{horizon}"], info=info, index=index, start="2024-07-01", end="2024-07-05"
    )
    assert found == set()


def test_a_zone_with_another_wall_clock_is_refused_and_a_user_rule_or_other_columns_are_not_checked():
    info = InformationSet().with_source_rules([_warsaw_weather_source()])
    london = pd.date_range("2024-06-20", periods=24 * 30, freq="h", tz="Europe/London")
    with pytest.raises(InformationSetError, match="Europe/London"):
        violations(["warsaw_temperature_2m_d+{horizon}"], info=info, index=london, start="2024-07-01", end="2024-07-05")
    # A rule the user wrote for the column replaces the source's, and the other columns have rules of their own zone.
    own = InformationSet(rules={"warsaw_temperature_2m": Availability.static()}).with_source_rules(
        [_warsaw_weather_source()]
    )
    found, _ = violations(
        ["warsaw_temperature_2m_d+{horizon}"], info=own, index=UTC_SUMMER, start="2024-07-01", end="2024-07-05"
    )
    assert found == set()
    found, _ = violations(
        ["load_forecast_d+{horizon}"], info=info, index=UTC_SUMMER, start="2024-07-01", end="2024-07-05", horizon=1
    )
    assert {period for _, period, _ in found} <= {22, 23}  # its own zone conversion handles the UTC index


def test_source_rules_agg_mean_locations_cover_the_aggregated_columns():
    # agg="mean" emits `{prefix}_{variable}_mean_d+i`, suffixed just like a list containing "mean".
    source = _warsaw_weather_source(
        latitude=None,
        longitude=None,
        locations={"warsaw": {"latitude": 52.23, "longitude": 21.01, "weight": 1.0}},
        agg="mean",
    )
    rules = source_rules([source])
    assert "warsaw_temperature_2m_mean" in rules
    assert "warsaw_warsaw_temperature_2m_mean" not in rules  # no per-location columns without keep_locations


def test_source_rules_agg_with_keep_locations_cover_both_shapes():
    source = _warsaw_weather_source(
        latitude=None,
        longitude=None,
        locations={
            "warsaw": {"latitude": 52.23, "longitude": 21.01, "weight": 1.0},
            "krakow": {"latitude": 50.06, "longitude": 19.94, "weight": 1.0},
        },
        agg="mean",
        keep_locations=True,
    )
    rules = source_rules([source])
    assert "warsaw_temperature_2m_mean" in rules
    assert "warsaw_warsaw_temperature_2m" in rules
    assert "warsaw_krakow_temperature_2m" in rules


def test_source_rules_per_location_only_locations_cover_the_per_location_columns():
    # Without `agg` only the per-location columns exist; the base name must not get a rule for a column that
    # is never actually produced.
    source = _warsaw_weather_source(
        latitude=None,
        longitude=None,
        locations={
            "warsaw": {"latitude": 52.23, "longitude": 21.01},
            "krakow": {"latitude": 50.06, "longitude": 19.94},
        },
    )
    rules = source_rules([source])
    assert "warsaw_warsaw_temperature_2m" in rules
    assert "warsaw_krakow_temperature_2m" in rules
    assert "warsaw_temperature_2m" not in rules


def test_source_rules_agg_list_covers_the_suffixed_columns_and_skips_what_isnt_emitted():
    # A list of aggregations names each output column `{prefix}_{variable}_{kind}` (see
    # OpenMeteoSource.agg / aggregated_variable_names); weather_code is aggregated like any other numeric
    # column, and wind_direction_10m only gets a rule for "mean" (the other aggregations skip circular columns).
    source = _warsaw_weather_source(
        latitude=None,
        longitude=None,
        columns=["temperature_2m", "wind_direction_10m", "weather_code"],
        locations={
            "warsaw": {"latitude": 52.23, "longitude": 21.01},
            "krakow": {"latitude": 50.06, "longitude": 19.94},
        },
        agg=["mean", "max"],
        keep_locations=True,
    )
    rules = source_rules([source])
    assert "warsaw_temperature_2m_mean" in rules
    assert "warsaw_temperature_2m_max" in rules
    assert "warsaw_wind_direction_10m_mean" in rules
    assert "warsaw_wind_direction_10m_max" not in rules  # circular: no meaningful max
    assert "warsaw_weather_code_mean" in rules
    assert "warsaw_weather_code_max" in rules
    # keep_locations=True: the per-location columns are still there too, for every column including the ones
    # skipped above.
    for name in ("warsaw", "krakow"):
        assert f"warsaw_{name}_temperature_2m" in rules
        assert f"warsaw_{name}_wind_direction_10m" in rules
        assert f"warsaw_{name}_weather_code" in rules


def test_source_rules_week_ahead_uses_the_markets_own_zone_for_a_helsinki_index():
    """entsoe/information-set: `default_rules()`'s weekly rule for load_forecast_daily_min/max is evaluated in
    Europe/Brussels regardless of the actual market. For a Helsinki index issued Thursday 2024-01-11,
    `load_forecast_daily_max_d+4` delivers on Monday 2024-01-15, whose controlling Friday (2024-01-12) hasn't
    published yet -- every period must be flagged alike. Period 0 is the one at risk because Monday 00:00 EET
    is still Sunday 23:00 in Brussels (UTC+2 vs UTC+1 in January), which would land the value in the wrong ISO week."""
    source = EntsoeSource("FI", api_key="test-key", type=["load"])
    DataPipeline(sources=[source], timezone="Europe/Helsinki")
    info = InformationSet("11:45", "Europe/Helsinki").with_source_rules([source])
    index = pd.date_range("2024-01-01", periods=24 * 30, freq="h", tz="Europe/Helsinki")

    found, unknown = violations(
        ["load_forecast_daily_max_d+{horizon}"],
        info=info,
        horizon=4,
        start="2024-01-11",
        end="2024-01-11",
        index=index,
    )

    assert unknown == set()
    flagged_periods = {period for _, period, hz in found if hz == 4}
    assert flagged_periods == set(range(24))  # period 0 alongside periods 1-23, not missing


def test_source_rules_week_ahead_uses_the_markets_own_zone_for_a_warsaw_control():
    """Control: Warsaw (CET) and Brussels (CET) share the same UTC offset in January, so this scenario was
    never broken there -- binding the market's own zone must not change an already-correct result."""
    source = EntsoeSource("PL", api_key="test-key", type=["load"])
    DataPipeline(sources=[source], timezone="Europe/Warsaw")
    info = InformationSet("11:45", "Europe/Warsaw").with_source_rules([source])
    index = pd.date_range("2024-01-01", periods=24 * 30, freq="h", tz="Europe/Warsaw")

    found, unknown = violations(
        ["load_forecast_daily_max_d+{horizon}"],
        info=info,
        horizon=4,
        start="2024-01-11",
        end="2024-01-11",
        index=index,
    )

    assert unknown == set()
    flagged_periods = {period for _, period, hz in found if hz == 4}
    assert flagged_periods == set(range(24))


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
        InformationSet("11:45", "Mars/Olympus")
    with pytest.raises(ValueError, match="assumption"):
        InformationSet(assumptions=["everything_is_known"])
    with pytest.raises(ValueError, match="kind"):
        Availability("sometimes")


# ---------------------------------------------------------------------------
# The check, rule by rule (issue 11:45 Brussels, Warsaw data: same wall clock)
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


def test_model_contract_allows_the_bare_target_column_as_a_predictor():
    """Whether the same-day (lag 0) target column is available at issue time is the information set's call,
    not the model data contract's: the contract only reserves '{target}_d+<h>' names (the model's own future
    targets, checked by _contract.check_predictors), never the bare target name itself."""
    df = make_market_data(days=120)
    model = OLSModel(["price", "price_d-1"], training_window=30)
    ref = model.run(df, test_start="2023-04-12", test_end="2023-04-12", target="price", horizon=1)
    assert ref.produced == ref.count == 24


def test_price_d_plus_1_is_rejected_by_the_model_contract_when_target_is_price():
    """For target 'price', 'price_d+1' names the model's own horizon-1 target: the data contract rejects it
    as look-ahead purely by name, before the raw data or the information set are even consulted (it would
    also fail the information-set check on its own merits, see test_price_availability_follows_issue_time
    below, but this path never gets there)."""
    df = make_market_data(days=120)
    assert "price_d+1" not in df.columns
    model = OLSModel(["price_d-1", "price_d+1"], training_window=30)
    with pytest.raises(ValueError, match="name the model's own targets"):
        model.run(df, test_start="2023-04-12", test_end="2023-04-12", target="price", horizon=1)


@pytest.mark.parametrize(
    "index, start, end",
    [
        (WARSAW, "2024-01-10", "2024-01-20"),
        (DST_SPRING_WARSAW, "2024-03-28", "2024-04-02"),  # spans the 2024-03-31 spring-forward day
        (DST_FALL_WARSAW, "2024-10-24", "2024-10-30"),  # spans the 2024-10-27 fall-back day
    ],
)
def test_price_availability_follows_issue_time(index, start, end):
    """The convention: a price predictor's availability is decided by publication time (~13:00 Brussels on
    D-1) versus issue_time, not by a fixed lag cutoff. Target 'price', issued 11:00 Brussels: day D's own
    price (lag 0) and older prices were already published; tomorrow's price (the value being forecast) was
    not. Holds on a Europe/Warsaw index across both DST transitions."""
    info = InformationSet("11:00", "Europe/Brussels")
    passing, unknown = violations(
        ["price", "price_d-1", "price_d-7"], info=info, target="price", horizon=1, start=start, end=end, index=index
    )
    assert passing == set()
    assert unknown == set()

    rejected, _ = violations(["price_d+1"], info=info, target="price", horizon=1, start=start, end=end, index=index)
    assert {name for name, _, _ in rejected} == {"price_d+1"}


@pytest.mark.parametrize(
    "index, start, end, crossing_hours",
    [
        (WARSAW.tz_convert("UTC"), "2024-01-10", "2024-01-20", {23}),
        (DST_SPRING_WARSAW.tz_convert("UTC"), "2024-03-28", "2024-04-02", {22, 23}),
        (DST_FALL_WARSAW.tz_convert("UTC"), "2024-10-24", "2024-10-30", {22, 23}),
    ],
)
def test_price_availability_follows_issue_time_with_utc_data(index, start, end, crossing_hours):
    """Same convention, UTC-indexed data. In CET (UTC+1) the last UTC hour(s) of the day already deliver in
    the *next* local calendar day (two hours in CEST, UTC+2), so bare 'price' at those rows genuinely holds
    tomorrow's price and is correctly rejected, matching
    test_day_ahead_values_are_known_for_the_next_day_only_with_utc_data above; every other hour, and every
    price_d-1/price_d-7 lag, still passes."""
    info = InformationSet("11:00", "Europe/Brussels")
    found, unknown = violations(
        ["price", "price_d-1", "price_d-7"], info=info, target="price", horizon=1, start=start, end=end, index=index
    )
    assert {p for name, p, _ in found if name == "price"} == crossing_hours
    assert {name for name, _, _ in found if name != "price"} == set()
    assert unknown == set()


@pytest.mark.parametrize(
    "index, start, end",
    [
        (WARSAW, "2024-01-10", "2024-01-20"),
        (DST_SPRING_WARSAW, "2024-03-28", "2024-04-02"),
        (DST_FALL_WARSAW, "2024-10-24", "2024-10-30"),
    ],
)
def test_non_price_target_may_use_tomorrows_price_only_after_it_is_published(index, start, end):
    """A non-price target (e.g. load_actual) has no target-boundary exemption for 'price': it may use
    tomorrow's day-ahead price once, and only once, it is actually published. Issued at 11:00 Brussels
    (before the ~13:00 publication), price_d+1 is rejected; issued at 14:00 (after), it passes.
    load_forecast_d+1 (published 10:00 D-1) is available at both times either way."""
    early, unknown_early = violations(
        ["price_d+1"],
        info=InformationSet("11:00", "Europe/Brussels"),
        target="load_actual",
        horizon=1,
        start=start,
        end=end,
        index=index,
    )
    assert {name for name, _, _ in early} == {"price_d+1"}
    assert unknown_early == set()

    late, unknown_late = violations(
        ["price_d+1", "load_forecast_d+1"],
        info=InformationSet("14:00", "Europe/Brussels"),
        target="load_actual",
        horizon=1,
        start=start,
        end=end,
        index=index,
    )
    assert late == set()
    assert unknown_late == set()

    early_load_forecast, _ = violations(
        ["load_forecast_d+1"],
        info=InformationSet("11:00", "Europe/Brussels"),
        target="load_actual",
        horizon=1,
        start=start,
        end=end,
        index=index,
    )
    assert early_load_forecast == set()


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


def test_weather_run_availability_depends_on_the_row_time():
    """`Availability.weather_run` is still available for a user-defined rule (e.g. a custom rolling-forecast
    source), even though no built-in source produces it through `source_rules` any more — OpenMeteoSource's own
    rule is always `weather_cutoff`, see the tests above."""
    info = InformationSet("11:45", "Europe/Brussels", rules={"warsaw_temperature_2m": Availability.weather_run("10h")})
    found, _ = violations(["warsaw_temperature_2m_d+{horizon}"], info=info)
    # Run started about at the row time and published 10 h later: rows up to 01:00 are known by 11:45.
    assert {p for _, p, _ in found} == set(range(2, 24))
    assert {hz for _, _, hz in found} == set(range(1, 8))


def test_hourly_lags_and_static_columns():
    found, _ = violations(["load_actual_h-12", "is_holiday_d+{horizon}", "daylight_hours_d+7"], horizon=1)
    # load_actual 12 hours earlier ends 11h before the row (12h gap - 1h period) and is known 1h after that,
    # i.e. 10h before the row: that is after the 11:45 issue only for the last two hours of the day.
    assert {(name, p) for name, p, _ in found} == {("load_actual_h-12", p) for p in (22, 23)}


def test_training_target_known_too_late_is_reported():
    info = InformationSet(rules={"monthly_settlement": Availability.published("12:00", days_before=-3)})
    found, _ = violations([], info=info, target="monthly_settlement", horizon=1)
    assert found and {name for name, _, _ in found} == {"monthly_settlement"}


def test_issue_time_is_converted_to_the_data_zone():
    utc = WARSAW.tz_convert("UTC")
    # 11:45 Brussels is 10:45 UTC in winter: load_actual rows up to 08:00 UTC are known in time.
    found, _ = violations(["load_actual"], horizon=1, index=utc, start="2024-01-10", end="2024-01-10")
    assert min(p for _, p, _ in found) == 9
    # In summer it is 09:45 UTC.
    summer = pd.date_range("2024-06-01", periods=24 * 30, freq="h", tz="UTC")
    found, _ = violations(["load_actual"], horizon=1, index=summer, start="2024-06-10", end="2024-06-10")
    assert min(p for _, p, _ in found) == 8


@pytest.mark.parametrize("data_zone", ["UTC", "Etc/GMT-1", "Asia/Tokyo"])
@pytest.mark.parametrize("issue_time", ["11:45", "12:30", "13:30"])
def test_violations_use_the_delivery_and_publication_instants_own_offsets(data_zone, issue_time):
    """The pre-fit check's counterpart of test_perturbation_uses_the_delivery_and_publication_instants_own_offsets.
    On a frame whose zone has no DST (UTC, a benchmark source's fixed offset) checked against Brussels-anchored rules,
    each conversion must use the offset at the instant it converts, on every issue day. Converting every delivery and
    publication time with the offsets at noon of the issue day D, and letting an earlier day with the same weekday
    and noon offsets stand for a day just before a switch, misjudged predictors near every switch, both ways. Issued
    on 2024-03-30, `load_forecast_d+1` at 22:00Z (delivered 2024-04-01 00:00 CEST, published 10:00 on 2024-03-31)
    passed: a leak. Issued on Thursday 2024-10-24, `load_forecast_daily_max_d+3` at 22:00Z (delivered Sunday
    2024-10-27 23:00 CET, published the Friday before that week) was rejected. The expected verdict is computed
    independently, from real instants, for test ranges ending two days before to one day after each switch."""
    columns = {  # name: (published time Brussels, days_before or "weekly", day shift)
        "price": ("13:00", 1, 0),
        "price_d-1": ("13:00", 1, -1),
        "price_d+1": ("13:00", 1, 1),
        "load_forecast_d+1": ("10:00", 1, 1),
        "load_forecast_d+2": ("10:00", 1, 2),
        "load_forecast_daily_max_d+2": ("10:00", "weekly", 2),
        "load_forecast_daily_max_d+3": ("10:00", "weekly", 3),
    }
    info = InformationSet(issue_time, "Europe/Brussels")
    expanded = {(period, 1): list(columns) for period in range(24)}

    wrong = {}
    for switch in ["2024-03-31", "2024-10-27", "2025-10-26"]:
        for end in pd.date_range(pd.Timestamp(switch) - pd.Timedelta(days=2), periods=4, freq="D"):
            days = pd.date_range(end - pd.Timedelta(days=9), end, freq="D")
            index = pd.date_range(days[0] - pd.Timedelta(days=7), end + pd.Timedelta(days=7), freq="h", tz=data_zone)
            found, _ = info.violations(expanded, "price", "1h", index, str(days[0].date()), str(end.date()))
            got = {(v.predictor, v.period) for v in found if v.role == "predictor"}

            expected = set()
            for day in days:
                rows = pd.date_range(day, periods=24, freq="h").tz_localize(data_zone)
                issue = pd.Timestamp(f"{day.date()} {issue_time}", tz="Europe/Brussels")
                for name, (time, days_before, shift) in columns.items():
                    delivery = rows + pd.Timedelta(days=shift)  # no DST in the data zone: a calendar day is 24 h
                    delivery_day = delivery.tz_convert("Europe/Brussels").normalize().tz_localize(None)
                    if days_before == "weekly":  # Friday of the week before the delivery's Monday-Sunday week
                        published_day = delivery_day - pd.to_timedelta(delivery_day.weekday + 3, unit="D")
                    else:
                        published_day = delivery_day - pd.Timedelta(days=days_before)
                    published = (published_day + pd.Timedelta(f"{time}:00")).tz_localize("Europe/Brussels")
                    expected |= {(name, int(period)) for period in np.flatnonzero(published > issue)}
            if got != expected:
                wrong[str(end.date())] = {"missed": sorted(expected - got), "extra": sorted(got - expected)}
    assert wrong == {}


def test_violations_convert_the_issue_time_at_its_own_instant():
    """Issued 00:30 Brussels on the autumn switch day 2024-10-27, the issue is still CEST (22:30Z on 2024-10-26), not
    CET like noon of that day. The newest training target, load_actual on 2024-10-26 at 21:00Z, is known at 23:00Z:
    after the issue."""
    utc = pd.date_range("2024-10-20", "2024-11-03", freq="h", tz="UTC", inclusive="left")
    info = InformationSet("00:30", "Europe/Brussels")
    found, _ = violations(
        [], info=info, target="load_actual", horizon=1, index=utc, start="2024-10-27", end="2024-10-27"
    )
    assert {p for _, p, _ in found} == {21, 22, 23}


@pytest.mark.parametrize("issue_timezone", ["Europe/Brussels", "Europe/Warsaw"])
@pytest.mark.parametrize("data_zone", ["UTC", "Europe/Warsaw"])
def test_an_issue_time_skipped_by_the_spring_switch_is_read_at_its_earliest(issue_timezone, data_zone):
    """02:30 doesn't exist in Brussels or Warsaw on 2024-03-31. Read with the offset after the switch it is 00:30Z,
    its earliest reading, so the issue is never later than asked: a value known at 00:45Z is not known yet, for the
    check and the perturbation alike, also when the issue zone is the data zone. Moved forward to the switch (01:00Z),
    or just before it (00:59:59Z), the issue would come after that value."""
    index = pd.date_range("2024-03-25", "2024-04-05", freq="h", tz=data_zone)
    info = InformationSet("02:30", issue_timezone, rules={"x": Availability.as_of("00:45", "UTC")})
    found, _ = violations(["x"], info=info, horizon=1, index=index, start="2024-03-31", end="2024-03-31")
    assert {p for _, p, _ in found} == set(range(24))

    perturbed, _, _ = perturb_unknown(pd.DataFrame({"x": 1.0}, index=index), "2024-03-31", info)
    assert (perturbed["x"] != 1.0).all()


@pytest.mark.parametrize("rule", [Availability.as_of("09:00", "UTC"), Availability.weather_cutoff("00", "9h")])
def test_as_of_and_cutoff_rules_are_anchored_to_each_issue_day(rule):
    """Known 09:00Z on each issue day: before the 10:30 Brussels issue in CET (09:30Z), after it in CEST (08:30Z).
    A test range from winter into summer must be checked on its summer days too, not with the first day's cutoff."""
    info = InformationSet("10:30", "Europe/Brussels", rules={"x": rule})
    utc = pd.date_range("2024-03-20", "2024-04-10", freq="h", tz="UTC")
    winter, _ = violations(["x"], info=info, horizon=1, index=utc, start="2024-03-25", end="2024-03-30")
    spanning, _ = violations(["x"], info=info, horizon=1, index=utc, start="2024-03-25", end="2024-04-02")
    assert winter == set()
    assert {p for _, p, _ in spanning} == set(range(24))


def test_unknown_columns_follow_the_policy():
    expanded = expand(["load_d+{horizon}"], horizon=1)
    with pytest.raises(InformationSetError, match="no availability rule.*'load'"):
        InformationSet().check(expanded, "price", "1h", WARSAW, "2024-01-10", "2024-01-10")
    with pytest.warns(UserWarning, match="load"):
        InformationSet(unknown="warn").check(expanded, "price", "1h", WARSAW, "2024-01-10", "2024-01-10")
    InformationSet(unknown="ignore").check(expanded, "price", "1h", WARSAW, "2024-01-10", "2024-01-10")


def test_shipped_example_predictors_are_flagged():
    """The predictor list of model_pipeline.yaml / examples 02-05 (as of this change) uses look-ahead inputs.

    warsaw_temperature_2m_d+{horizon} is not flagged: OpenMeteoSource's default cutoff (run="00" plus JMA's
    conservative 9.6h publication delay) is known before the model's own issue time, see
    test_default_run_passes_the_cutoff_check_at_1145_brussels_all_year above.
    """
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
    info = InformationSet("11:45", "Europe/Brussels").with_source_rules([_warsaw_weather_source()])
    found, unknown = violations(shipped, info=info)
    assert unknown == set()
    assert {name for name, _, _ in found} == {"load_actual"}


# ---------------------------------------------------------------------------
# Profile columns (DailyProfileTransformer): judged at the slot they name, not at the row's period
# ---------------------------------------------------------------------------

LOAD_ACTUAL_PROFILE = {"load_actual_h*": Availability.realised("1h")}


def _refused_periods(found) -> dict:
    out = {}
    for name, period, _ in found:
        out.setdefault(name, set()).add(period)
    return out


def test_profile_slot_names():
    info = InformationSet(rules={"x*": Availability.static()})
    assert info.resolve("x_h05").slot == (300, 60)
    assert info.resolve("x_h05_d-1").slot == (300, 60)  # the shift comes after the slot
    assert info.resolve("x_d+1_h05").slot == (300, 60)  # a profile of the lag x_d+1
    assert info.resolve("x_s95").slot == (1425, 15)  # DailyProfileTransformer(freq="15min")
    assert info.resolve("x_h24").slot is None  # not a slot of a day
    assert info.resolve("x_h-12").slot is None  # an hour lag of x
    assert info.resolve("x").slot is None


def test_profile_slot_published_before_the_issue_is_accepted_at_every_period():
    """False refusal. On the DE benchmark index (Etc/GMT-1, UTC+1 all year) with Brussels rules, `price_h00` on the
    rows of D is the price at 00:00 on D (23:00Z on D-1: delivered on D in Brussels, published 13:00 on D-1), known
    when the forecast is issued at 11:45 Berlin on D. Judged at the row's own period, the row at 23:00 (22:00Z, which
    is 00:00 CEST on D+1 in summer) reads it as delivered on D+1 and refuses it ("known D 12:00, issued D 10:45")."""
    index = pd.date_range("2015-05-20", "2015-06-20", freq="h", tz="Etc/GMT-1")
    info = InformationSet("11:45", "Europe/Berlin", rules={"price_h*": Availability.published("13:00", 1)})
    expanded = {(period, 1): ["price_h00"] for period in range(24)}

    found, unknown = info.violations(expanded, "price", "1h", index, "2015-06-01", "2015-06-07")

    assert unknown == set()
    assert [(v.predictor, v.period) for v in found] == []


def test_profile_slot_realised_after_the_issue_is_refused_at_every_period():
    """Look-ahead. With a user rule `load_actual_h*` realised 1 h after its period, `load_actual_h23` on the rows of D
    is the load at 23:00 on D, known at 01:00 on D+1: not known at the 11:45 issue on D, whatever the row's period.
    Judged at the row's own period, it is accepted for periods 0-9 (each realised by 11:00)."""
    index = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="Europe/Warsaw")
    info = InformationSet("11:45", "Europe/Brussels", rules={"load_actual_h*": Availability.realised("1h")})
    expanded = {(period, 1): ["load_actual_h23"] for period in range(24)}

    found, unknown = info.violations(expanded, "price", "1h", index, "2024-01-10", "2024-01-23")

    assert unknown == set()
    assert {v.period for v in found} == set(range(24))


def test_today_s_realised_slots_are_usable_and_the_rest_are_not_at_every_period():
    """Both directions: realised 1 h after its hour, slot hNN of D is known at NN+2:00 on D, so h00-h09 are known at
    the 11:45 issue from every period of D and h10-h23 from none, and the message names the slot's own time."""
    info = InformationSet("11:45", "Europe/Brussels", rules=LOAD_ACTUAL_PROFILE)

    found, unknown = violations(profile_columns("load_actual"), info=info, horizon=1)

    assert unknown == set()
    assert _refused_periods(found) == {f"load_actual_h{h:02d}": set(range(24)) for h in range(10, 24)}
    with pytest.raises(InformationSetError, match=r"Periods 0-23.*period 0 is known D\+1 01:00, issued D 11:45"):
        info.check(expand(["load_actual_h23"], horizon=1), "price", "1h", WARSAW, "2024-01-10", "2024-01-10")


def test_a_profile_column_is_read_from_the_day_its_shifts_reach():
    """`load_actual_h23_d-1` is 23:00 on D-1 (known 01:00 on D) and `price_h05_d+1` the price of 05:00 on D+1
    (published 13:00 on D). `load_actual_h23_h-12` reads the profile of the row 12 hours earlier: D-1's for periods
    0-11, D's (23:00 on D, known on D+1) from period 12 on."""
    rules = {**LOAD_ACTUAL_PROFILE, "price_h*": Availability.published("13:00", 1)}
    info = InformationSet("11:45", "Europe/Brussels", rules=rules)
    names = ["load_actual_h23_d-1", "price_h05_d-1", "price_h05", "price_h05_d+1", "load_actual_h23_h-12"]

    found, unknown = violations(names, info=info, horizon=1)

    assert unknown == set()
    assert _refused_periods(found) == {"price_h05_d+1": set(range(24)), "load_actual_h23_h-12": set(range(12, 24))}


def test_quarter_hour_and_hour_slots_on_15_minute_data():
    """`_sNN` is a quarter-hour (DailyProfileTransformer(freq="15min")) and `_hNN` an hour, whatever the model's freq.
    Realised 1 h after it ends, quarter-hour 42 (10:30-10:45) is known at the 11:45 issue and quarter-hour 43 is
    not, nor is hour 10 (known 12:00), from every one of the 96 periods."""
    index = pd.date_range("2024-01-01", periods=96 * 30, freq="15min", tz="Europe/Warsaw")
    rules = {"load_actual_s*": Availability.realised("1h"), **LOAD_ACTUAL_PROFILE}
    info = InformationSet("11:45", "Europe/Brussels", rules=rules)
    names = ["load_actual_s42", "load_actual_s43", "load_actual_h09", "load_actual_h10"]
    expanded = {(period, 1): names for period in range(96)}

    found, unknown = info.violations(expanded, "price", "15min", index, "2024-01-10", "2024-01-16")

    assert unknown == set()
    assert _refused_periods({(v.predictor, v.period, v.horizon) for v in found}) == {
        "load_actual_s43": set(range(96)),
        "load_actual_h10": set(range(96)),
    }


def test_a_profile_of_a_shifted_column_holds_the_shifted_value():
    """LagTransformer before DailyProfileTransformer: `generation_solar_d+1_h05` on row D is `generation_solar_d+1` at
    05:00 on D, i.e. generation at 05:00 on D+1 (realised, by the built-in `generation_*` rule, at 07:00 on D+1), and
    `generation_solar_d-1_h23` is 23:00 on D-1 (known 01:00 on D). Reading the slot on D without the shift before it
    would accept the first at every period."""
    names = ["generation_solar_d+1_h05", "generation_solar_d-1_h23"]
    found, unknown = violations(names, horizon=1)
    assert unknown == set()
    assert _refused_periods(found) == {"generation_solar_d+1_h05": set(range(24))}

    index = pd.date_range("2024-01-08", "2024-01-13", freq="h", tz="Europe/Warsaw", inclusive="left")
    perturbed, counts, _ = perturb_unknown(
        pd.DataFrame(1.0, index=index, columns=names), "2024-01-10", InformationSet()
    )
    run_day = np.asarray(index.tz_localize(None).normalize() == pd.Timestamp("2024-01-10"))
    assert (perturbed.loc[run_day, "generation_solar_d+1_h05"] != 1.0).all()
    assert (perturbed.loc[run_day, "generation_solar_d-1_h23"] == 1.0).all()


@pytest.mark.parametrize(
    "rule",
    [Availability.as_of("09:00"), Availability.weather_cutoff("00", "6h"), Availability.weather_run("6h")],
    ids=["as_of", "weather_cutoff", "weather_run"],
)
def test_a_day_lead_of_a_profile_column_is_a_shift_of_the_row(rule):
    """Look-ahead. `x_h05_d+1` (LagTransformer(lags=[-1]) on the profile column `x_h05`) holds x at 05:00 on D+1, known
    only on D+1 under these rules. They drop a leading `_d+k` as the source's own lead, but a `_d+k` after a slot
    suffix always shifts the row, so dropping it judges the value of D instead: accepted at the 11:45 issue on D."""
    index = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="Europe/Warsaw")
    info = InformationSet("11:45", "Europe/Brussels", rules={"x_h*": rule})
    expanded = {(period, 1): ["x_h05_d+1"] for period in range(24)}

    found, unknown = info.violations(expanded, "price", "1h", index, "2024-01-10", "2024-01-23")

    assert unknown == set()
    assert {v.period for v in found} == set(range(24))
    # Without the lead, the same slot of D is known (09:00 snapshot, 06:00Z cutoff, 05:00 run + 6 h) at 11:45.
    assert violations(["x_h05"], info=info, horizon=1)[0] == set()


def test_probe_perturbs_a_profile_slot_realised_after_the_issue_on_every_row():
    """The leakage probe's side of the look-ahead above: every row of the run day carries the same `load_actual_h23`
    (23:00 on that day, known 01:00 the next day), so every row must be perturbed, not only the rows from 10:00 on."""
    index = pd.date_range("2024-01-08", "2024-01-13", freq="h", tz="Europe/Warsaw", inclusive="left")
    data = pd.DataFrame({"load_actual_h23": 1.0}, index=index)
    info = InformationSet("11:45", "Europe/Brussels", rules={"load_actual_h*": Availability.realised("1h")})

    perturbed, _counts, unchecked = perturb_unknown(data, "2024-01-10", info)

    assert unchecked == []
    run_day = np.asarray(index.tz_localize(None).normalize() == pd.Timestamp("2024-01-10"))
    assert (perturbed.loc[run_day, "load_actual_h23"] != 1.0).all()


@pytest.mark.parametrize(
    "run_day, index, rows", [("2024-03-31", DST_SPRING_WARSAW, 23), ("2024-10-27", DST_FALL_WARSAW, 25)]
)
def test_profile_slots_have_one_verdict_on_every_row_of_a_dst_day(run_day, index, rows):
    """The switch days have 23 and 25 rows, each carrying the day's whole profile. Realised 1 h after its hour, slot 23
    (known 01:00 the next day) is perturbed on every one of them and refused at every period, including period 2,
    which has no row or two rows that day; slot 05, and the skipped or repeated slot 02 itself, on none."""
    names = ["load_actual_h02", "load_actual_h05", "load_actual_h23"]
    data = pd.DataFrame(1.0, index=index, columns=names)
    info = InformationSet("11:45", "Europe/Brussels", rules=LOAD_ACTUAL_PROFILE)

    perturbed, _, _ = perturb_unknown(data, run_day, info)
    found, unknown = violations(names, info=info, horizon=1, start=run_day, end=run_day, index=index)

    day = np.asarray(index.tz_localize(None).normalize() == pd.Timestamp(run_day))
    assert day.sum() == rows
    changed = perturbed.loc[day] != 1.0
    assert changed["load_actual_h23"].all()
    assert not changed[["load_actual_h02", "load_actual_h05"]].to_numpy().any()
    assert unknown == set()
    assert _refused_periods(found) == {"load_actual_h23": set(range(24))}


@pytest.mark.parametrize("switch", ["2024-03-31", "2024-10-27"])
@pytest.mark.parametrize("rule_zone", ["Europe/Brussels", "Atlantic/Cape_Verde"])
@pytest.mark.parametrize("data_zone", ["Europe/Warsaw", "Etc/GMT-1"])
def test_profile_verdicts_match_the_delivery_each_cell_holds(data_zone, rule_zone, switch):
    """Known answer from the data itself. `px` holds its own UTC instant, so every cell of DailyProfileTransformer's
    profiles (of `px` and of its day lags) and of their day lags says which delivery it holds: on the switch days' 23
    or 25 rows too, and the repeated autumn slot is its first occurrence, which the transformer copies. A cell is published 13:00 in the rule's zone on
    the day before its delivery day there, so it is unknown at the 11:45 Brussels issue exactly when that comes later;
    the probe must perturb exactly those cells and the check refuse exactly those (predictor, issue day) pairs, at every
    period. Cape Verde (UTC-1) puts Warsaw's two 02:00s of 2024-10-27 on different delivery days, and Etc/GMT-1 (the
    benchmark index) moves against Brussels at each switch. Cells the transformers leave NaN (the skipped spring slot,
    a day lag onto a skipped hour) hold no delivery and are not compared."""
    start = pd.Timestamp(switch) - pd.Timedelta(days=4)
    index = pd.date_range(start, start + pd.Timedelta(days=9), freq="h", tz=data_zone, inclusive="left")
    raw = pd.DataFrame({"px": (index.asi8 // 60_000_000_000).astype(float)}, index=index)  # minutes since the epoch
    columns = ["px", "px_d-1", "px_d+1"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the 23- and 25-hour days of a DST index
        data = LagTransformer(columns="px", lags=[1, -1], freq="day").transform(raw)
        data = DailyProfileTransformer(columns=columns).transform(data)
        profiles = [name for column in columns for name in profile_columns(column)]
        data = LagTransformer(columns=profiles, lags=[1, -1], freq="day").transform(data)
    # The slots around midnight, the DST hour (02:00) and one from the middle of the day.
    data = data[[name for name in data.columns if any(f"_h{h}" in name for h in ("00", "01", "02", "03", "12", "23"))]]

    assert _verdicts_against_held_deliveries(data, rule_zone, switch) == {}


def _verdicts_against_held_deliveries(data: pd.DataFrame, rule_zone: str, switch: str) -> dict:
    """Where the probe and the check disagree with the delivery each cell of `data` holds (minutes since the epoch), on
    the four issue days around `switch`. Under the rule `px*`, a cell is published 13:00 in `rule_zone` on the day
    before its delivery day there, so it is unknown at the 11:45 Brussels issue exactly when that comes later; the probe
    must perturb exactly those cells and the check refuse exactly those (predictor, issue day) pairs, at every period.
    NaN cells hold no delivery and are not compared."""
    index = data.index
    info = InformationSet("11:45", "Europe/Brussels", rules={"px*": Availability.published("13:00", 1, rule_zone)})
    # When each cell is published: 13:00 in the rule's zone on the day before the delivery day there (NaT for NaN).
    held = data.notna().to_numpy()
    delivery = pd.to_datetime(data.to_numpy()[held] * 60, unit="s", utc=True)
    delivery_day = delivery.tz_convert(rule_zone).normalize().tz_localize(None)
    published = (delivery_day - pd.Timedelta(days=1) + pd.Timedelta(hours=13)).tz_localize(rule_zone)
    on_day = np.repeat(index.tz_localize(None).normalize().to_numpy()[:, None], data.shape[1], axis=1)[held]
    names = np.repeat(np.asarray(data.columns)[None, :], len(index), axis=0)[held]
    rows = np.repeat(np.asarray(index)[:, None], data.shape[1], axis=1)[held]
    expanded = {(period, 1): list(data.columns) for period in range(24)}

    wrong = {}
    for run_day in pd.date_range(pd.Timestamp(switch) - pd.Timedelta(days=2), periods=4, freq="D"):
        unknown = np.asarray(published > pd.Timestamp(f"{run_day.date()} 11:45", tz="Europe/Brussels"))
        perturbed, _, _ = perturb_unknown(data, str(run_day.date()), info)
        mismatch = (perturbed.to_numpy() != data.to_numpy())[held] != unknown
        for name, row in zip(names[mismatch][:5], rows[mismatch][:5]):
            wrong.setdefault((str(run_day.date()), "probe"), []).append((name, str(row)))

        found, _ = info.violations(expanded, "price", "1h", index, str(run_day.date()), str(run_day.date()))
        refused = _refused_periods({(v.predictor, v.period, v.horizon) for v in found if v.role == "predictor"})
        today = on_day == run_day.to_datetime64()
        late_today = set(names[today & unknown])
        assert not late_today & set(names[today & ~unknown]), "every cell of a column on one day holds one delivery"
        refused = {name: periods for name, periods in refused.items() if name in set(names[today])}
        if refused != {name: set(range(24)) for name in late_today}:
            wrong[(str(run_day.date()), "check")] = sorted(set(refused) ^ late_today)[:5]
    return wrong


# ---------------------------------------------------------------------------
# Day aggregates (DailyStatsTransformer): known when the last slot of the day they aggregate is
# ---------------------------------------------------------------------------

LOAD_ACTUAL_ALL = {"load_actual*": Availability.realised("1h")}


def test_day_aggregate_names():
    info = InformationSet(rules={"x*": Availability.static()})
    for name in [
        "x_day_min",
        "x_day_max",
        "x_day_mean",
        "x_day_last",
        "x_day_max_d-1",
        "x_d-1_day_max",
        "x_h05_day_max",
    ]:
        assert info.resolve(name).day_aggregate, name
    for name in ["x", "x_day", "x_day_median", "x_daily_max", "x_day_max_h05", "x_h-12"]:
        assert not info.resolve(name).day_aggregate, name
    # ENTSO-E's week-ahead daily forecasts are columns of their own.
    assert not InformationSet().resolve("load_forecast_daily_min").day_aggregate


@pytest.mark.parametrize("stat", ["min", "max", "mean", "last"])
def test_day_aggregate_realised_after_the_issue_is_refused_at_every_period(stat):
    """Look-ahead. With a user rule `load_actual*` realised 1 h after its period, `load_actual_day_max` on the rows of D
    is the maximum over all 24 hours of D, known only when the last one is (01:00 on D+1): not at the 11:45 issue on D,
    whatever the row's period. Judged at the row's own period, it was accepted for periods 0-9."""
    index = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="Europe/Warsaw")
    info = InformationSet("11:45", "Europe/Brussels", rules={"load_actual*": Availability.realised("1h")})
    expanded = {(period, 1): [f"load_actual_day_{stat}"] for period in range(24)}

    found, unknown = info.violations(expanded, "price", "1h", index, "2024-01-10", "2024-01-23")

    assert unknown == set()
    assert {v.period for v in found} == set(range(24))


def test_day_aggregate_of_a_lagged_column_is_accepted_at_every_period():
    """False refusal. LagTransformer before DailyStatsTransformer: `load_actual_d-1_day_max` on the rows of D is the
    maximum of `load_actual_d-1` over D, i.e. of the load on D-1, known at 01:00 on D. Judged at the row's own period
    as unshifted `load_actual`, it was refused for periods 10-23."""
    index = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="Europe/Warsaw")
    info = InformationSet("11:45", "Europe/Brussels", rules={"load_actual*": Availability.realised("1h")})
    expanded = {(period, 1): ["load_actual_d-1_day_max"] for period in range(24)}

    found, unknown = info.violations(expanded, "price", "1h", index, "2024-01-10", "2024-01-23")

    assert unknown == set()
    assert [(v.predictor, v.period) for v in found] == []


def test_a_day_aggregate_is_read_from_the_day_its_shifts_reach():
    """Both directions. Realised 1 h after its hour, a statistic of D is known at 01:00 on D+1 and one of D-1 at 01:00
    on D, whether the day shift comes before the aggregate (`load_actual_d-1_day_max`) or after it; a lead either way
    reads D+1. `load_actual_day_max_h-12` reads the aggregate of the row 12 hours earlier: D-1's for periods 0-11, D's
    from 12 on. The day-ahead prices of D are published 13:00 on D-1, so `price_day_*` of D (the expert model's
    inputs) is known at every period, and of D+1 at none. The message names the last slot's own time."""
    rules = {**LOAD_ACTUAL_ALL, "price_day_*": Availability.published("13:00", 1)}
    info = InformationSet("11:45", "Europe/Brussels", rules=rules)
    late = ["load_actual_day_max", "load_actual_d+1_day_max", "load_actual_day_max_d+1", "price_day_max_d+1"]
    known = ["load_actual_d-1_day_max", "load_actual_day_max_d-1", "price_day_min", "price_day_max", "price_day_last"]

    found, unknown = violations(late + known + ["load_actual_day_max_h-12"], info=info, horizon=1)

    assert unknown == set()
    assert _refused_periods(found) == {
        **{name: set(range(24)) for name in late},
        "load_actual_day_max_h-12": set(range(12, 24)),
    }
    with pytest.raises(InformationSetError, match=r"Periods 0-23.*period 0 is known D\+1 01:00, issued D 11:45"):
        info.check(expand(["load_actual_day_max"], horizon=1), "price", "1h", WARSAW, "2024-01-10", "2024-01-10")


def test_the_built_in_generation_rule_covers_its_day_aggregates():
    """`generation_*` (actual generation, realised 1 h after its period) also matches `generation_solar_day_max`, so
    with the built-in rules alone the maximum of D is refused at every period and that of D-1 at none."""
    found, unknown = violations(["generation_solar_day_max", "generation_solar_day_max_d-1"], horizon=1)

    assert unknown == set()
    assert _refused_periods(found) == {"generation_solar_day_max": set(range(24))}


def test_day_aggregates_of_15_minute_data():
    """On 15-minute data (DailyStatsTransformer(freq="15min")) the day's last slot is 23:45-24:00, realised 1 h after
    it ends: `load_actual_day_max` of D is known at 01:00 on D+1, refused from every one of the 96 periods, and that of
    D-1 from none."""
    index = pd.date_range("2024-01-01", periods=96 * 30, freq="15min", tz="Europe/Warsaw")
    info = InformationSet("11:45", "Europe/Brussels", rules=LOAD_ACTUAL_ALL)
    expanded = {(period, 1): ["load_actual_day_max", "load_actual_day_max_d-1"] for period in range(96)}

    found, unknown = info.violations(expanded, "price", "15min", index, "2024-01-10", "2024-01-16")

    assert unknown == set()
    assert _refused_periods({(v.predictor, v.period, v.horizon) for v in found}) == {
        "load_actual_day_max": set(range(96))
    }


@pytest.mark.parametrize(
    "rule",
    [Availability.as_of("09:00"), Availability.weather_cutoff("00", "6h"), Availability.weather_run("6h")],
    ids=["as_of", "weather_cutoff", "weather_run"],
)
def test_a_day_lead_of_a_day_aggregate_is_a_shift_of_the_row(rule):
    """Look-ahead. `x_day_max_d+1` (LagTransformer(lags=[-1]) on the day aggregate `x_day_max`) is the maximum of x over
    D+1, known only on D+1 under these rules. They drop a leading `_d+k` as the source's own lead, but a `_d+k` after
    an aggregate always shifts the row, so dropping it judged the value of D instead: accepted at the 11:45 issue on D
    (all periods for as_of and weather_cutoff, periods 0-5 for weather_run)."""
    index = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="Europe/Warsaw")
    info = InformationSet("11:45", "Europe/Brussels", rules={"x*": rule})
    expanded = {(period, 1): ["x_day_max_d+1"] for period in range(24)}

    found, unknown = info.violations(expanded, "price", "1h", index, "2024-01-10", "2024-01-23")

    assert unknown == set()
    assert {v.period for v in found} == set(range(24))
    # A source's lead before the aggregate stays one: `x_d+1_day_max` holds D's snapshot, D's cutoff.
    lead_found, _ = violations(["x_d+1_day_max"], info=info, horizon=1)
    expected = set(range(24)) if rule.kind == "weather_run" else set()  # weather_run: the last row (23:00) + 6 h
    assert {period for _, period, _ in lead_found} == expected


def test_probe_perturbs_a_day_aggregate_realised_after_the_issue_on_every_row():
    """The leakage probe's side of the look-ahead above: every row of the run day carries the same
    `load_actual_day_max` (known 01:00 the next day), so every row must be perturbed, not only the rows from 10:00 on.
    The aggregate of a lag, `load_actual_d-1_day_max` (the load of the day before), is known on every row of it."""
    index = pd.date_range("2024-01-08", "2024-01-13", freq="h", tz="Europe/Warsaw", inclusive="left")
    raw = pd.DataFrame({"load_actual": np.arange(len(index), dtype=float)}, index=index)
    raw = LagTransformer(columns="load_actual", lags=[1], freq="day").transform(raw)
    data = DailyStatsTransformer(columns=["load_actual", "load_actual_d-1"], stats=["max"]).transform(raw)
    data = data[["load_actual_day_max", "load_actual_d-1_day_max"]]
    info = InformationSet("11:45", "Europe/Brussels", rules={"load_actual*": Availability.realised("1h")})

    perturbed, _counts, unchecked = perturb_unknown(data, "2024-01-10", info)

    assert unchecked == []
    run_day = np.asarray(index.tz_localize(None).normalize() == pd.Timestamp("2024-01-10"))
    changed = perturbed.loc[run_day] != data.loc[run_day]
    assert changed["load_actual_day_max"].all()
    assert not changed["load_actual_d-1_day_max"].any()


@pytest.mark.parametrize("freq", ["1h", "15min"])
@pytest.mark.parametrize("run_day, hours", [("2024-03-31", 23), ("2024-10-27", 25)])
def test_day_aggregates_have_one_verdict_on_every_row_of_a_dst_day(run_day, hours, freq):
    """The switch days have 23 and 25 hours (92 and 100 quarter-hours), all of them in each row's aggregate. Realised
    1 h after its slot, the aggregate of the switch day is known at 01:00 on the next day: perturbed on every one of its
    rows and refused at every period, including the switch hour's, which has no row or two rows that day. The
    aggregate of the day before, `_d-1` on the switch day, on none."""
    per_hour = 4 if freq == "15min" else 1
    start = pd.Timestamp(run_day) - pd.Timedelta(days=5)
    index = pd.date_range(start, start + pd.Timedelta(days=10), freq=freq, tz="Europe/Warsaw", inclusive="left")
    raw = pd.DataFrame({"load_actual": np.arange(len(index), dtype=float)}, index=index)
    data = DailyStatsTransformer(columns="load_actual", stats=["max", "mean"], freq=freq).transform(raw)
    data = LagTransformer(columns="load_actual_day_max", lags=[1], freq="day").transform(data)
    late, known = ["load_actual_day_max", "load_actual_day_mean"], ["load_actual_day_max_d-1"]
    data = data[late + known]
    info = InformationSet("11:45", "Europe/Brussels", rules=LOAD_ACTUAL_ALL)
    expanded = {(period, 1): late + known for period in range(24 * per_hour)}

    perturbed, _, _ = perturb_unknown(data, run_day, info, freq=freq)
    found, unknown = info.violations(expanded, "price", freq, index, run_day, run_day)

    day = np.asarray(index.tz_localize(None).normalize() == pd.Timestamp(run_day))
    assert day.sum() == hours * per_hour
    assert data.loc[day].notna().all().all()  # a complete day: every slot it has holds a value
    changed = perturbed.loc[day] != data.loc[day]
    assert changed[late].to_numpy().all()
    assert not changed[known].to_numpy().any()
    assert unknown == set()
    found = {(v.predictor, v.period, v.horizon) for v in found}
    assert _refused_periods(found) == {name: set(range(24 * per_hour)) for name in late}


@pytest.mark.parametrize("switch", ["2024-03-31", "2024-10-27"])
@pytest.mark.parametrize("rule_zone", ["Europe/Brussels", "Atlantic/Cape_Verde"])
@pytest.mark.parametrize("data_zone", ["Europe/Warsaw", "Etc/GMT-1"])
def test_day_aggregate_verdicts_match_the_latest_delivery_each_cell_holds(data_zone, rule_zone, switch):
    """Known answer from the data itself. `px` holds its own UTC instant, so the maximum and the last value of a day
    (DailyStatsTransformer, of `px` and of its day lags, and the day lags of those) hold the latest delivery the cell
    aggregates, on the switch days' 23 or 25 rows too. A cell is known when that delivery's price is (published 13:00 in
    the rule's zone on the day before its delivery day there), so the probe must perturb exactly the cells unknown at
    the 11:45 Brussels issue, and the check refuse exactly those (predictor, issue day) pairs, at every period. On the
    benchmark index (Etc/GMT-1) the day's last hour, 22:00Z, is delivered on D+1 in Brussels in summer and on D in
    winter. Days the transformer leaves NaN (a day lag onto a skipped hour makes a day incomplete) are not compared."""
    start = pd.Timestamp(switch) - pd.Timedelta(days=4)
    index = pd.date_range(start, start + pd.Timedelta(days=9), freq="h", tz=data_zone, inclusive="left")
    raw = pd.DataFrame({"px": (index.asi8 // 60_000_000_000).astype(float)}, index=index)  # minutes since the epoch
    columns = ["px", "px_d-1", "px_d+1"]
    data = LagTransformer(columns="px", lags=[1, -1], freq="day").transform(raw)
    data = DailyStatsTransformer(columns=columns, stats=["max", "last"]).transform(data)
    aggregates = [f"{column}_day_{stat}" for column in columns for stat in ("max", "last")]
    data = LagTransformer(columns=aggregates, lags=[1, -1], freq="day").transform(data).drop(columns=columns)

    assert _verdicts_against_held_deliveries(data, rule_zone, switch) == {}


@pytest.mark.parametrize("rules", [["warsaw_is_day_mean", "warsaw_is_day_max"], ["warsaw_*"]], ids=["exact", "glob"])
def test_a_user_rule_for_an_is_day_column_reads_it_as_a_column(rules):
    """`warsaw_is_day_mean` is Open-Meteo's `is_day` averaged over locations, whoever wrote its rule: under a user's rule
    too its `_d+1` is the source's lead, known with D's run, and a day aggregate or a profile of it keeps that lead."""
    cutoff = Availability.weather_cutoff("00", "6h")
    info = InformationSet("11:45", "Europe/Brussels", rules={name: cutoff for name in rules})
    names = ["warsaw_is_day_mean_d+1", "warsaw_is_day_max_d+1"]
    if "warsaw_*" in rules:  # the exact rules don't name these
        names += ["warsaw_is_day_mean_d+1_day_max", "warsaw_is_day_mean_d+1_h05"]

    assert not info.resolve("warsaw_is_day_mean_d+1").day_aggregate
    assert violations(names, info=info, horizon=1) == (set(), set())


def test_only_is_day_is_excluded_from_the_day_aggregates():
    """The exclusion is by name, for any layer: `is_day_*` and `{prefix}_is_day_*` are Open-Meteo's. `this_day_max` and
    `basis_day_max` only end in the same letters, and an exact user rule on a real aggregate doesn't make it a plain
    column: `load_actual_day_max` judged at the row's own period is the look-ahead these rules exist to stop."""
    info = InformationSet(rules={"*": Availability.static()})
    for name in ["is_day_max", "is_day_min", "is_day_mean", "warsaw_is_day_max", "warsaw_is_day_mean_d+1"]:
        assert not info.resolve(name).day_aggregate, name
    for name in ["this_day_max", "basis_day_max", "x_day_max", "warsaw_is_day_day_max", "warsaw_is_day_mean_day_max"]:
        assert info.resolve(name).day_aggregate, name

    exact = InformationSet("11:45", "Europe/Brussels", rules={"load_actual_day_max": Availability.realised("1h")})
    found, unknown = violations(["load_actual_day_max"], info=exact, horizon=1)
    assert unknown == set()
    assert _refused_periods(found) == {"load_actual_day_max": set(range(24))}


def test_a_source_column_named_like_a_day_aggregate_keeps_its_own_rule():
    """An OpenMeteoSource's `is_day` averaged over locations is `warsaw_is_day_mean`: a column of its own, which the
    source's rule describes as it is (its `_d+k` is the source's lead), not a day aggregate of `warsaw_is`. So
    `warsaw_is_day_mean_d+1` comes from D's run, known at the 11:45 issue; read as an aggregate, the `_d+1` would shift
    the row to D+1's run and be refused at every period."""
    source = _warsaw_weather_source(
        latitude=None,
        longitude=None,
        columns=["is_day"],
        locations={"warsaw": {"latitude": 52.23, "longitude": 21.01, "weight": 1.0}},
        agg=["mean", "max"],
    )
    info = InformationSet("11:45", "Europe/Brussels").with_source_rules([source])

    assert not info.resolve("warsaw_is_day_mean_d+1").day_aggregate
    assert violations(["warsaw_is_day_mean_d+{horizon}", "warsaw_is_day_max_d+{horizon}"], info=info) == (set(), set())


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def test_model_pipeline_checks_before_fitting(tmp_path):
    df = make_market_data(days=120).rename(columns={"load": "load_actual"})
    pipeline = ModelPipeline(issue_time="11:45").add_model(
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
            "information_set": {"issue_time": "11:45", "issue_timezone": "Europe/Brussels"},
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
    # A44 default: prices for 2023-01-21 are published 2023-01-20 13:00, after the 11:45 issue.
    assert df.index[changed].min() == pd.Timestamp("2023-01-21 00:00", tz="Europe/Warsaw")
    lag_changed = _changed(perturbed["price_d-1"], df["price_d-1"])
    assert df.index[lag_changed].min() == pd.Timestamp("2023-01-22 00:00", tz="Europe/Warsaw")
    assert "load" not in counts and unchecked == []


@pytest.mark.parametrize(
    "run_day, first_changed_utc",
    [
        ("2024-03-30", "2024-03-31 22:00"),  # spring-forward the next day (2024-03-31)
        ("2024-10-26", "2024-10-27 23:00"),  # fall-back the next day (2024-10-27)
    ],
)
def test_perturbation_uses_each_rows_own_offset_near_a_dst_switch(run_day, first_changed_utc):
    """DST/timezone review finding 3: `perturb_unknown` applies each row's own UTC offset, not the issue
    day's own noon offset. On a UTC-indexed frame, `load_forecast` (published 10:00 Brussels on
    D-1) is delivered on the local calendar day, in Brussels, of the row itself; once that day has crossed a
    DST switch relative to run_day, the issue day's offset would put some rows on the wrong side of the
    publication boundary. With it, run day 2024-03-30 would leave `load_forecast` at 2024-03-31 22:00Z
    unperturbed (it must not be, since 2024-03-31 is delivered in CEST, one hour further from UTC than
    run_day's own CET offset); the mirror case near the autumn switch is off by an hour the other way."""
    idx = pd.date_range("2024-03-20", "2024-11-05", freq="1h", tz="UTC", inclusive="left")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"load_forecast": rng.normal(1000, 10, len(idx))}, index=idx)
    info = InformationSet("11:45", "Europe/Brussels")

    perturbed, counts, _ = perturb_unknown(df, run_day, info)
    changed = _changed(perturbed["load_forecast"], df["load_forecast"])

    row = pd.Timestamp(first_changed_utc, tz="UTC")
    assert changed.loc[row]
    assert not changed.loc[row - pd.Timedelta(hours=1)]
    assert df.index[changed].min() == row
    assert counts["load_forecast"] > 0


def test_as_of_perturbation_ignores_a_rows_own_dst_offset():
    """Companion to the test above: `as_of` is fixed to a time on the issue day D itself (DST/timezone review
    finding 3's fix must not make it vary by the row's own date). Every row gets the same verdict even though
    the index spans a DST switch, because run_day's own offset — not each row's — decides it."""
    idx = pd.date_range("2024-03-25", "2024-04-05", freq="1h", tz="UTC", inclusive="left")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"snap": rng.normal(0, 1, len(idx))}, index=idx)
    info = InformationSet("11:45", "Europe/Brussels", rules={"snap": Availability.as_of("14:00", "Europe/Brussels")})

    perturbed, counts, _ = perturb_unknown(df, "2024-03-30", info)
    changed = _changed(perturbed["snap"], df["snap"])

    assert changed.all()  # 14:00 Brussels on run_day is after the 11:45 issue: unknown for every row alike
    assert counts["snap"] == len(idx)


@pytest.mark.parametrize("data_zone", ["UTC", "Etc/GMT-1", "Asia/Tokyo"])
@pytest.mark.parametrize("issue_time", ["11:45", "12:30", "13:30"])
@pytest.mark.parametrize(
    "run_day", ["2024-03-29", "2024-03-30", "2024-03-31", "2024-10-26", "2024-10-27", "2025-10-25", "2025-10-26"]
)
def test_perturbation_uses_the_delivery_and_publication_instants_own_offsets(data_zone, issue_time, run_day):
    """Follow-up to test_perturbation_uses_each_rows_own_offset_near_a_dst_switch. On a frame whose zone has no
    DST (UTC, a benchmark source's fixed offset) checked against Brussels-anchored rules, the Brussels offset moves
    at the switch but the data zone's doesn't, so each conversion must use the offset at the instant it converts:
    the shifted delivery instant (for its Brussels calendar day) and the publication instant (back to the data
    zone). Taking both from the row's own, unshifted date misjudged cells near every switch, both ways. Run day
    2024-03-30 perturbed `price_d-1` at 2024-03-31 22:00Z (delivered 2024-03-30 23:00 CET, published the day
    before) and kept `load_forecast_d+1` at 2024-03-30 22:00Z (delivered 2024-04-01 00:00 CEST, published 10:00 on
    2024-03-31); issued 12:30, it kept every 2024-03-31 `price` (published 13:00 CET = 12:00Z, after the 11:30Z
    issue). The expected verdict is computed independently, from real instants."""
    idx = pd.date_range(pd.Timestamp(run_day) - pd.Timedelta(days=6), periods=24 * 12, freq="1h", tz=data_zone)
    columns = {  # name: (published time Brussels, days_before or "weekly", day shift)
        "price": ("13:00", 1, 0),
        "price_d-1": ("13:00", 1, -1),
        "price_d+1": ("13:00", 1, 1),
        "load_forecast_d+1": ("10:00", 1, 1),
        "load_forecast_d+2": ("10:00", 1, 2),
        "load_forecast_daily_max_d+2": ("10:00", "weekly", 2),
        "load_forecast_daily_max_d+3": ("10:00", "weekly", 3),
    }
    df = pd.DataFrame(0.0, index=idx, columns=list(columns))
    perturbed, _, _ = perturb_unknown(df, run_day, InformationSet(issue_time, "Europe/Brussels"))

    issue = pd.Timestamp(f"{run_day} {issue_time}", tz="Europe/Brussels")
    for name, (time, days_before, shift) in columns.items():
        delivery = idx + pd.Timedelta(days=shift)  # no DST in the data zone: a calendar day is 24 h
        delivery_day = delivery.tz_convert("Europe/Brussels").normalize().tz_localize(None)
        if days_before == "weekly":  # Friday of the week before the delivery's Monday-Sunday week
            published_day = delivery_day - pd.to_timedelta(delivery_day.weekday + 3, unit="D")
        else:
            published_day = delivery_day - pd.Timedelta(days=days_before)
        published = (published_day + pd.Timedelta(f"{time}:00")).tz_localize("Europe/Brussels")
        wrong = idx[(perturbed[name] != 0).to_numpy() != (published > issue)]
        assert list(wrong) == [], f"{name}: wrong verdict at {list(wrong)}"


def test_perturbation_converts_the_issue_time_at_its_own_instant():
    """Issued 00:30 Brussels on the autumn switch day 2024-10-27, the issue is still CEST (22:30Z on 2024-10-26),
    not CET like the rest of the day: a value known 01:30 Brussels (23:30Z) is not known yet."""
    idx = pd.date_range("2024-10-26", "2024-10-29", freq="1h", tz="UTC", inclusive="left")
    df = pd.DataFrame({"snap": 1.0}, index=idx)
    info = InformationSet("00:30", "Europe/Brussels", rules={"snap": Availability.as_of("01:30", "Europe/Brussels")})

    _, counts, _ = perturb_unknown(df, "2024-10-27", info)

    assert counts["snap"] == len(idx)


def test_perturbation_keeps_a_repeated_hours_own_occurrence():
    """New York's repeated hour 01:00-02:00 on 2024-11-03 straddles midnight in Mexico City (UTC-6 all year):
    01:00 EDT (05:00Z) delivers on 2024-11-02 there, 01:00 EST (06:00Z) on 2024-11-03, published 13:00 Mexico
    City the day before (19:00Z on 2024-11-02, after the 11:45 Brussels = 10:45Z issue)."""
    idx = pd.date_range("2024-10-30", "2024-11-06", freq="1h", tz="America/New_York", inclusive="left")
    df = pd.DataFrame({"px": 1.0}, index=idx)
    info = InformationSet(
        "11:45", "Europe/Brussels", rules={"px": Availability.published("13:00", 1, "America/Mexico_City")}
    )

    perturbed, _, _ = perturb_unknown(df, "2024-11-02", info)

    assert perturbed["px"].loc[pd.Timestamp("2024-11-03 05:00", tz="UTC")] == 1.0
    assert perturbed["px"].loc[pd.Timestamp("2024-11-03 06:00", tz="UTC")] != 1.0


def test_perturbation_covers_a_day_lag_of_a_weather_column():
    """The same rule drives the probe: the issue day's value of `x_d+1_d+1` comes from the next day's run, which is
    unknown at issue time, while `x_d+1` itself (D's run, published by 10:36 CET) is known."""
    index = pd.date_range("2024-01-01", "2024-02-01", freq="h", tz="Europe/Warsaw", inclusive="left")
    df = pd.DataFrame(
        {
            name: np.linspace(0, 1, len(index))
            for name in ("warsaw_temperature_2m_d+1", "warsaw_temperature_2m_d+1_d+1")
        },
        index=index,
    )
    info = InformationSet("11:45", "Europe/Brussels").with_source_rules([_warsaw_weather_source()])
    perturbed, counts, _ = perturb_unknown(df, "2024-01-16", info)
    assert "warsaw_temperature_2m_d+1" not in counts
    assert (perturbed["warsaw_temperature_2m_d+1"] == df["warsaw_temperature_2m_d+1"]).all()
    issue_day = df.index.normalize() == pd.Timestamp("2024-01-16", tz="Europe/Warsaw")
    assert (perturbed.loc[issue_day, "warsaw_temperature_2m_d+1_d+1"] > 1e5).all()


def test_probe_detects_a_model_reading_a_weather_column_through_a_forward_day_lag():
    df = make_market_data(noise=5.0, seed=7)
    df["temperature_d+1"] = np.random.default_rng(3).normal(10, 5, len(df))
    df["temperature_d+1_d+1"] = df["temperature_d+1"].shift(-24)
    info = InformationSet(
        rules={
            **PRICE_KNOWN_AFTER_DELIVERY,
            "temperature": Availability.weather_cutoff("00", "9.6h"),
        }
    )
    predictors = ["load_d+{horizon}", "is_weekend_d+{horizon}", "is_weekday_d+{horizon}"]

    clean = leakage_probe(
        lambda: OLSModel(predictors + ["temperature_d+1"], training_window=45), df, "2023-04-12", info, horizon=1
    )
    clean.assert_ok()
    assert "temperature_d+1" not in clean.perturbed

    leaky = leakage_probe(
        lambda: OLSModel(predictors + ["temperature_d+1_d+1"], training_window=45), df, "2023-04-12", info, horizon=1
    )
    assert leaky.perturbed["temperature_d+1_d+1"] > 0
    assert not leaky.ok


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
    known well before the 11:45 issue, so perturbing the already-built column never touches that cell.
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
