"""EconomicEvaluator and BatteryArbitrage. The battery's reproduction of the evaluation-protocol study on stored forecasts
is in `tests/test_report_evalproto.py`, through the report."""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.evaluators import BatteryArbitrage, EconomicEvaluator
from epftoolbox2.results.report import EvaluationReport
from tests.forecast_records import forecast_records


def study():
    run_dates = pd.date_range("2023-12-01", periods=70, freq="D")
    return {"base": forecast_records(1, run_dates, noise=6.0), "better": forecast_records(2, run_dates, noise=4.0)}


def one_day(prices):
    return np.asarray([prices], dtype=float)


def test_battery_on_a_hand_computed_day():
    """Hours 3 and 4 are cheapest, 18 and 19 dearest, in that order, so the split lies between them.

    Plan: charge at the forecast's 3 and 4 and discharge at its 18 and 19. Forecast profit: 0.9 * (100 + 90) - (10 + 20)
    = 141, so it trades. Settled at the actual prices (which differ from the forecast only in those four hours):
    0.9 * (80 + 110) - (15 + 5) = 151.
    """
    forecast = [50.0] * 24
    forecast[3], forecast[4], forecast[18], forecast[19] = 10.0, 20.0, 100.0, 90.0
    actual = [50.0] * 24
    actual[3], actual[4], actual[18], actual[19] = 15.0, 5.0, 80.0, 110.0

    profit = BatteryArbitrage().profit(one_day(forecast), one_day(actual))

    assert profit == pytest.approx([151.0])


def test_battery_charges_before_it_discharges():
    """Falling prices: the cheapest hours are the last, the dearest the first, so the unconstrained pick is impossible.

    Best split: charge in 22-23 cannot come before discharge. With prices 24, 23, ... 1, the plan needs charge before
    discharge: the largest forecast profit over all splits k is charge at the two cheapest before k, discharge at the two
    dearest from k on. For k=2: charge 24+23, discharge 0.9*(22+21); value 0.9*43 - 47 = -8.3. For k=22: charge 4+3,
    discharge 0.9*(2+1); value 2.7 - 7 = -4.3, the best. It is negative: no trade.
    """
    prices = np.arange(24, 0, -1, dtype=float)
    assert BatteryArbitrage().profit(one_day(prices), one_day(prices)) == pytest.approx([0.0])


def test_battery_does_not_trade_when_the_spread_does_not_cover_the_losses():
    """A flat forecast with 5 % spread: 0.9 * 105 - 100 < 0, even though the actual day would have earned."""
    forecast = [100.0] * 12 + [105.0] * 12
    actual = [0.0] * 12 + [500.0] * 12
    assert BatteryArbitrage().profit(one_day(forecast), one_day(actual)) == pytest.approx([0.0])


def test_battery_loses_money_when_the_forecast_is_wrong():
    forecast = [10.0] * 12 + [100.0] * 12
    actual = [100.0] * 12 + [10.0] * 12
    # The first hours and hours from 12 on: charge in 0, 1 and discharge in 12, 13.
    assert BatteryArbitrage().profit(one_day(forecast), one_day(actual)) == pytest.approx([0.9 * 20 - 200])


def test_battery_on_quarter_hours_moves_the_same_energy():
    """96 periods: 8 quarter-hours of charge and 8 of discharge, 0.25 MWh each."""
    forecast = np.full(96, 50.0)
    forecast[8:16] = 10.0  # 02:00-04:00
    forecast[72:80] = 100.0  # 18:00-20:00
    actual = forecast + 1.0
    expected = 0.25 * (0.9 * 8 * 101.0 - 8 * 11.0)
    assert BatteryArbitrage().profit(one_day(forecast), one_day(actual), 4) == pytest.approx([expected])


def test_battery_with_hourly_prices_on_quarter_hours_is_the_hourly_rule():
    """Hourly prices repeated four times: every hourly plan is also a quarter-hour plan, so on the true prices the
    quarter-hour battery earns at least as much, and the same unless a split inside an hour pays."""
    with np.load(Path(__file__).parent / "fixtures" / "evalproto" / "de_honest_placebo.npz") as z:
        prices = z["Y"]
    battery = BatteryArbitrage()

    hourly = battery.profit(prices, prices)
    quarters = battery.profit(np.repeat(prices, 4, axis=1), np.repeat(prices, 4, axis=1), 4)

    assert (quarters >= hourly - 1e-9).all()
    assert np.isclose(quarters, hourly).mean() > 0.9


def test_battery_is_the_best_of_all_schedules_on_true_prices():
    """Brute force: any charge periods before any discharge periods. The rule's splits leave none of them out."""
    prices = np.random.default_rng(3).normal(50, 30, (100, 10))
    best = []
    for day in prices:
        values = [0.0]
        for charge in itertools.combinations(range(10), 2):
            for discharge in itertools.combinations(range(max(charge) + 1, 10), 2):
                values.append(0.9 * day[list(discharge)].sum() - day[list(charge)].sum())
        best.append(max(values))
    np.testing.assert_allclose(BatteryArbitrage().profit(prices, prices), best)


def test_battery_scales_with_power_and_efficiency():
    forecast = [10.0, 20.0] + [50.0] * 20 + [100.0, 90.0]
    actual = forecast
    # 2 MW for one hour: one period each way, at the cheapest and the dearest hour.
    assert BatteryArbitrage(power_mw=2, energy_mwh=2, efficiency=0.8).profit(one_day(forecast), one_day(actual)) == (
        pytest.approx([2 * (0.8 * 100 - 10)])
    )
    # 1 MW for four hours: 4 periods each way.
    expected = 0.8 * (100 + 90 + 50 + 50) - (10 + 20 + 50 + 50)
    assert BatteryArbitrage(power_mw=1, energy_mwh=4, efficiency=0.8).profit(one_day(forecast), one_day(actual)) == (
        pytest.approx([expected])
    )


def test_battery_on_23_and_25_period_days():
    for periods in (23, 25):
        prices = np.full(periods, 50.0)
        prices[1:3] = 10.0
        prices[-3:-1] = 100.0
        assert BatteryArbitrage().profit(one_day(prices), one_day(prices)) == pytest.approx([0.9 * 200 - 20])


def test_battery_breaks_ties_the_same_way_every_time():
    forecast = np.array([[10.0, 10.0, 10.0] + [50.0] * 18 + [90.0, 90.0, 90.0]])
    actual = np.array([[1.0, 2.0, 3.0] + [50.0] * 18 + [70.0, 80.0, 90.0]])
    assert BatteryArbitrage().profit(forecast, actual) == pytest.approx([0.9 * (70 + 80) - (1 + 2)])


@pytest.mark.parametrize(
    "kwargs",
    [{"power_mw": 0}, {"energy_mwh": -1}, {"efficiency": 0}, {"efficiency": 1.2}],
)
def test_battery_refuses_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        BatteryArbitrage(**kwargs)


def test_battery_refuses_a_cycle_that_is_not_whole_periods():
    day = one_day(np.arange(24.0))
    with pytest.raises(ValueError, match="whole number of periods"):
        BatteryArbitrage(power_mw=1, energy_mwh=1.5).profit(day, day)
    assert BatteryArbitrage(power_mw=1, energy_mwh=1.5).profit(np.repeat(day, 4, axis=1), np.repeat(day, 4, axis=1), 4)


def test_battery_refuses_a_cycle_longer_than_the_day():
    day = one_day(np.arange(24.0))
    with pytest.raises(ValueError, match="cycle takes"):
        BatteryArbitrage(energy_mwh=13).profit(day, day)


# ---------------------------------------------------------------------------------------------------------------------
# EconomicEvaluator
# ---------------------------------------------------------------------------------------------------------------------


def day_records(run_date, prices, forecast=None, horizon=1, ids=None):
    run = pd.Timestamp(run_date)
    target = run + pd.Timedelta(days=horizon)
    forecast = prices if forecast is None else forecast
    ids = range(len(prices)) if ids is None else ids
    return [
        {
            "run_date": run.date().isoformat(),
            "target_date": target.date().isoformat(),
            "hour": int(i),
            "horizon": horizon,
            "prediction": float(f),
            "actual": float(a),
        }
        for i, f, a in zip(ids, forecast, prices)
    ]


def evaluate(records):
    return EvaluationReport({"m": records}, [EconomicEvaluator()]).summary().iloc[0]


PRICES = np.array([50.0] * 24)
PRICES[3:5] = [10.0, 20.0]
PRICES[18:20] = [100.0, 90.0]
PERFECT = 0.9 * 190 - 30


def test_profit_columns_of_a_hand_computed_day():
    forecast = PRICES.copy()
    forecast[3], forecast[5] = 60.0, 5.0  # now charges in 4 and 5: pays 20 + 50 in the actual prices
    row = evaluate(day_records("2024-01-01", PRICES, forecast))
    expected = 0.9 * 190 - 70
    assert row["profit"] == pytest.approx(expected)
    assert row["profit % of perfect"] == pytest.approx(100 * expected / PERFECT)
    assert row["profit days"] == 1


def test_perfect_foresight_is_100_percent():
    records = day_records("2024-01-01", PRICES) + day_records("2024-01-02", PRICES)
    row = evaluate(records)
    assert row["profit"] == pytest.approx(2 * PERFECT)
    assert row["profit % of perfect"] == pytest.approx(100.0)
    assert row["profit days"] == 2


def test_a_target_day_counts_once_over_horizons():
    # The same target day from two horizons: the first plans well, the second badly. The day earns their mean.
    bad = PRICES[::-1]
    records = day_records("2024-01-01", PRICES, horizon=2) + day_records("2024-01-02", PRICES, bad, horizon=1)
    row = evaluate(records)
    one = EvaluationReport({"m": records[24:]}, [EconomicEvaluator()]).summary().iloc[0]
    assert row["profit days"] == 1
    assert row["profit"] == pytest.approx((PERFECT + one["profit"]) / 2)
    assert row["profit % of perfect"] == pytest.approx(100 * (PERFECT + one["profit"]) / 2 / PERFECT)

    by_horizon = EvaluationReport({"m": records}, [EconomicEvaluator()]).by_horizon().set_index("horizon")
    assert by_horizon.loc[2, "profit"] == pytest.approx(PERFECT)
    assert by_horizon.loc[1, "profit"] == pytest.approx(one["profit"])


def test_a_day_with_missing_periods_is_not_traded():
    whole = day_records("2024-01-01", PRICES)
    partial = day_records("2024-01-02", PRICES)
    del partial[10]
    del partial[9]
    assert evaluate(whole + partial)["profit days"] == 1
    assert evaluate(whole + partial)["profit"] == pytest.approx(PERFECT)


@pytest.mark.parametrize("hole", [0, 6, 18, 23])
def test_one_hole_outside_the_switch_hours_takes_the_day_out(hole):
    """23 finite forecasts are not a 23-hour day when the hour they lack is not one the clocks can skip."""
    partial = day_records("2024-01-02", PRICES)
    del partial[hole]
    row = evaluate(day_records("2024-01-01", PRICES) + partial)
    assert row["profit days"] == 1 and row["profit"] == pytest.approx(PERFECT)


@pytest.mark.parametrize("switch_hour", [1, 2, 3])
def test_a_day_without_a_switch_hour_is_a_whole_23_hour_day(switch_hour):
    """That is what a real run writes around a clock change: ids by the clock, without the hour that has no target."""
    records = day_records("2024-01-01", PRICES)
    del records[switch_hour]
    row = evaluate(records)
    assert row["profit days"] == 1
    assert np.isfinite(row["profit"])


def test_a_model_that_never_forecasts_one_hour_has_no_whole_day():
    records = day_records("2024-01-01", PRICES) + day_records("2024-01-02", PRICES)
    row = evaluate([r for r in records if r["hour"] != 18])
    assert row["profit days"] == 0 and np.isnan(row["profit"])


def test_one_hole_in_a_quarter_hour_day_takes_it_out():
    prices = np.full(96, 50.0)
    prices[8:16] = 10.0
    prices[72:80] = 100.0
    whole = day_records("2024-01-01", prices)
    assert evaluate(whole)["profit days"] == 1
    for holes in ([40], [40, 41, 42, 43], [20, 21, 22, 23]):
        partial = [r for r in day_records("2024-01-02", prices) if r["hour"] not in holes]
        assert evaluate(whole + partial)["profit days"] == 1, holes


def test_a_quarter_hour_day_without_a_switch_hour_is_whole():
    prices = np.full(96, 50.0)
    prices[16:24] = 10.0
    prices[72:80] = 100.0
    records = [r for r in day_records("2024-01-01", prices) if r["hour"] not in (8, 9, 10, 11)]
    row = evaluate(records)
    assert row["profit days"] == 1
    assert row["profit"] == pytest.approx(0.25 * (0.9 * 8 * 100 - 8 * 10))


def test_twenty_four_quarter_hours_are_not_an_hourly_day():
    """Read as hourly, they would earn four times what they can."""
    prices = np.full(96, 50.0)
    prices[8:16] = 10.0
    prices[72:80] = 100.0
    whole = day_records("2024-01-01", prices)
    kept = day_records("2024-01-02", prices)[:24]
    row = evaluate(whole + kept)
    assert row["profit days"] == 1
    assert row["profit"] == pytest.approx(0.25 * (0.9 * 8 * 100 - 8 * 10))


def test_a_non_finite_forecast_takes_the_day_out():
    broken = day_records("2024-01-02", PRICES)
    broken[0]["prediction"] = float("nan")
    broken[1]["prediction"] = float("nan")
    row = evaluate(day_records("2024-01-01", PRICES) + broken)
    assert row["profit days"] == 1


# Run dates whose target day is a Sunday (the switch days of 2024), and the Saturday-to-Sunday day before them.
SHORT_DAY, LONG_DAY, PLAIN_DAY = "2024-03-30", "2024-10-26", "2024-01-03"


def test_days_of_all_lengths_are_traded():
    records = []
    for run_date, periods in ((SHORT_DAY, 23), (PLAIN_DAY, 24), (LONG_DAY, 25)):
        prices = np.full(periods, 50.0)
        prices[1:3] = 10.0
        prices[-3:-1] = 100.0
        records += day_records(run_date, prices)
    row = evaluate(records)
    assert row["profit days"] == 3
    assert row["profit"] == pytest.approx(3 * (0.9 * 200 - 20))


def test_quarter_hour_days_of_all_lengths_are_traded():
    records = []
    for run_date, periods in ((SHORT_DAY, 92), (PLAIN_DAY, 96), (LONG_DAY, 100)):
        prices = np.full(periods, 50.0)
        prices[8:16] = 10.0
        prices[-16:-8] = 100.0
        records += day_records(run_date, prices)
    row = evaluate(records)
    assert row["profit days"] == 3
    assert row["profit"] == pytest.approx(3 * 0.25 * (0.9 * 8 * 100 - 8 * 10))


def test_a_day_numbered_through_counts_as_short_only_on_a_sunday():
    """Ids 0-22 on a Thursday are a 24-hour day without its last hour, and a whole day has that hour."""
    prices = np.full(23, 50.0)
    prices[1:3] = 10.0
    prices[-3:-1] = 100.0
    assert evaluate(day_records(SHORT_DAY, prices))["profit days"] == 1
    assert evaluate(day_records("2024-01-03", prices))["profit days"] == 0


def test_periods_are_taken_in_the_order_of_their_id():
    shuffled = day_records("2024-01-01", PRICES)
    np.random.default_rng(0).shuffle(shuffled)
    assert evaluate(shuffled)["profit"] == pytest.approx(PERFECT)


def test_a_table_cut_by_hour_has_no_profit():
    report = EvaluationReport(study(), [EconomicEvaluator()])
    table = report.by_hour()
    assert table["profit"].isna().all()
    assert (table["profit days"] == 0).all()


def test_each_model_is_scored_on_its_own_days_and_says_how_many():
    results = {"whole": day_records("2024-01-01", PRICES) + day_records("2024-01-02", PRICES)}
    results["short"] = day_records("2024-01-01", PRICES)
    table = EvaluationReport(results, [EconomicEvaluator()]).summary().set_index("model")
    assert table.loc["whole", "profit days"] == 2 and table.loc["short", "profit days"] == 1


def test_a_model_without_forecasts_has_no_profit():
    table = EvaluationReport({"a": day_records("2024-01-01", PRICES), "b": []}, [EconomicEvaluator()]).summary()
    assert np.isnan(table.set_index("model").loc["b", "profit"])


def test_a_duplicated_key_raises():
    records = day_records("2024-01-01", PRICES)
    records.append(dict(records[0]))
    with pytest.raises(ValueError, match="duplicated"):
        evaluate(records)


def test_records_without_keys_raise():
    with pytest.raises(ValueError, match="forecast keys"):
        EconomicEvaluator().compute(pd.DataFrame({"prediction": [1.0], "actual": [1.0]}))


def test_the_default_strategy_is_the_documented_battery():
    battery = EconomicEvaluator().strategy
    assert (battery.power_mw, battery.energy_mwh, battery.efficiency) == (1, 2, 0.9)
