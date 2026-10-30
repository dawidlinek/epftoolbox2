from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import FORECAST_KEYS, Evaluator

# The local hours a clock change can fall in: the only periods a whole day may lack.
_SWITCH_HOURS = (1, 2, 3)


def _whole_days(per_hour: int) -> Dict[int, List[Tuple[np.ndarray, bool]]]:
    """`{periods in the day: [(the period ids such a whole day may hold, whether it must be a Sunday)]}`.

    A real run numbers periods by the clock, so a day of 23 hours lacks the hour it skips, also on the day after the spring
    switch and on the autumn switch day, when that hour has no target. Other results may number a short day through, and
    that is also what a day looks like that only lacks its last hour, so it counts only on a Sunday, when clocks change.
    """
    clock = np.arange(24 * per_hour)
    return {
        24 * per_hour: [(clock, False)],
        25 * per_hour: [(np.arange(25 * per_hour), False)],
        23 * per_hour: [
            (np.arange(23 * per_hour), True),
            *((np.delete(clock, np.arange(hour * per_hour, (hour + 1) * per_hour)), False) for hour in _SWITCH_HOURS),
        ],
    }


class BatteryArbitrage:
    """One charge and one discharge a day, scheduled on the forecast and settled at the actual prices.

    For every split period k: charge in the forecast-cheapest periods before k and discharge in the forecast-dearest
    periods from k on, as many of each as it takes to move `energy_mwh` at `power_mw`. The split with the largest forecast
    profit (`efficiency * discharge value - charge value`) is the plan, and the battery trades only if that profit is
    positive. Exact ties go to the earlier period. Perfect foresight is the same rule on the actual prices.

    Args:
        power_mw: Charge and discharge power.
        energy_mwh: Energy bought per cycle. `energy_mwh / power_mw` must be a whole number of periods.
        efficiency: Round-trip efficiency: the share of the energy bought that is sold.
    """

    def __init__(self, power_mw: float = 1, energy_mwh: float = 2, efficiency: float = 0.9):
        if not power_mw > 0 or not energy_mwh > 0:
            raise ValueError(f"power_mw and energy_mwh must be positive, got {power_mw} and {energy_mwh}")
        if not 0 < efficiency <= 1:
            raise ValueError(f"efficiency must be in (0, 1], got {efficiency}")
        self.power_mw = power_mw
        self.energy_mwh = energy_mwh
        self.efficiency = efficiency

    def __repr__(self) -> str:
        return f"BatteryArbitrage(power_mw={self.power_mw}, energy_mwh={self.energy_mwh}, efficiency={self.efficiency})"

    def profit(self, forecast: np.ndarray, actual: np.ndarray, periods_per_hour: int = 1) -> np.ndarray:
        """Profit of each day (EUR when prices are EUR/MWh). `forecast` and `actual` are days x periods."""
        forecast, actual = np.asarray(forecast, dtype=float), np.asarray(actual, dtype=float)
        if forecast.ndim != 2 or forecast.shape != actual.shape:
            raise ValueError(f"forecast and actual must be days x periods, got {forecast.shape} and {actual.shape}")
        days, periods = forecast.shape
        exact = self.energy_mwh / self.power_mw * periods_per_hour
        cycle = int(round(exact))
        if cycle < 1 or abs(exact - cycle) > 1e-9:
            raise ValueError(
                f"energy_mwh / power_mw = {self.energy_mwh / self.power_mw} h is not a whole number of periods at "
                f"{periods_per_hour} period(s) per hour."
            )
        if periods < 2 * cycle:
            raise ValueError(f"A cycle takes {2 * cycle} periods, but the days have {periods}.")

        best = np.full(days, -np.inf)
        charge = np.zeros((days, cycle), dtype=np.int64)
        discharge = np.zeros((days, cycle), dtype=np.int64)
        # `kind="stable"`: the default sort depends on the CPU's vector instructions when prices tie.
        for split in range(cycle, periods - cycle + 1):
            cheapest = np.argsort(forecast[:, :split], axis=1, kind="stable")[:, :cycle]
            dearest = split + np.argsort(-forecast[:, split:], axis=1, kind="stable")[:, :cycle]
            value = self.efficiency * np.take_along_axis(forecast, dearest, 1).sum(1)
            value -= np.take_along_axis(forecast, cheapest, 1).sum(1)
            better = value > best
            best[better] = value[better]
            charge[better], discharge[better] = cheapest[better], dearest[better]
        settled = self.efficiency * np.take_along_axis(actual, discharge, 1).sum(1)
        settled -= np.take_along_axis(actual, charge, 1).sum(1)
        return np.where(best > 0, settled * self.power_mw / periods_per_hour, 0.0)


class EconomicEvaluator(Evaluator):
    """What a model's forecasts earn when they schedule a trading strategy. A value column, not a ranking criterion.

    Columns: `profit` (EUR over the forecasts' target days), `profit % of perfect` (against the same strategy with
    perfect foresight on the same days) and `profit days` (the target days traded). A target day counts once: with several
    horizons it earns the mean of what they earn on it. A day counts only when its periods are all there, since a schedule
    needs the whole day: finite forecasts for all of 24 hours (96 quarter-hours), for 25 hours (100) numbered through, or
    for 23 hours (92) lacking one hour between 01:00 and 04:00, where clocks change, or numbered through on a Sunday. In
    a table cut by hour no day is whole, so the columns are NaN. Each model is scored on its own days: check `profit days`
    before reading two models side by side.

    Args:
        strategy: The strategy to run, by default `BatteryArbitrage()` (1 MW, 2 MWh, round-trip efficiency 0.9). Any object
            with a method `profit(forecast, actual, periods_per_hour)` that takes days x periods arrays and gives the
            profit per day; it must be registered in `COMPONENT_REGISTRY["strategies"]` to load from YAML.
    """

    name = "profit"

    def __init__(self, strategy: Optional[BatteryArbitrage] = None):
        self.strategy = BatteryArbitrage() if strategy is None else strategy

    def compute(self, df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        nothing = {"": float("nan"), "% of perfect": float("nan"), "days": 0}
        if df.empty:
            return nothing
        missing = [k for k in FORECAST_KEYS if k not in df.columns]
        if missing:
            raise ValueError(
                f"EconomicEvaluator needs the forecast keys {list(FORECAST_KEYS)}; the frame lacks {missing}."
            )
        prediction = np.asarray(df["prediction"], dtype=float)
        actual = np.asarray(df["actual"], dtype=float)
        finite = np.isfinite(prediction) & np.isfinite(actual)
        target, target_dates = pd.factorize(df["target_date"].to_numpy()[finite], sort=True)
        horizon, horizons = pd.factorize(df["horizon"].to_numpy()[finite], sort=True)
        if (target < 0).any() or (horizon < 0).any():
            raise ValueError("EconomicEvaluator: a frame has rows with a missing forecast key.")
        hour = df["hour"].to_numpy()[finite]
        prediction, actual = prediction[finite], actual[finite]

        day = target * max(len(horizons), 1) + horizon
        order = np.lexsort((hour, day))
        day, hour, prediction, actual = day[order], hour[order], prediction[order], actual[order]
        if ((day[1:] == day[:-1]) & (hour[1:] == hour[:-1])).any():
            raise ValueError("EconomicEvaluator needs each forecast key once; the frame has duplicated keys.")

        # The frame says what a period is: the number of finite forecasts of one day cannot, since it may have holes.
        per_hour = 4 if df["hour"].max() > 24 else 1
        length = np.bincount(day) if len(day) else np.empty(0, dtype=np.int64)
        target_of_day = np.arange(len(length)) // max(len(horizons), 1)
        earned = np.full(len(length), np.nan)
        perfect = np.full(len(length), np.nan)
        sunday = pd.DatetimeIndex(target_dates).dayofweek == 6
        for periods, allowed in _whole_days(per_hour).items():
            days = np.flatnonzero(length == periods)
            if not len(days):
                continue
            rows = np.isin(day, days)
            ids = hour[rows].reshape(-1, periods)
            on_sunday = sunday[target_of_day[days]]
            whole = np.any(
                [(ids == template).all(axis=1) & (on_sunday | ~sunday_only) for template, sunday_only in allowed],
                axis=0,
            )
            if not whole.any():
                continue
            forecast = prediction[rows].reshape(-1, periods)[whole]
            settled = actual[rows].reshape(-1, periods)[whole]
            earned[days[whole]] = self.strategy.profit(forecast, settled, per_hour)
            perfect[days[whole]] = self.strategy.profit(settled, settled, per_hour)

        traded = np.isfinite(earned)
        if not traded.any():
            return nothing
        # One value per target day: the mean over the horizons that traded it.
        spots = target_of_day[traded]
        horizons_per_target = np.bincount(spots)
        kept = horizons_per_target > 0
        profit = (np.bincount(spots, weights=earned[traded])[kept] / horizons_per_target[kept]).sum()
        best = (np.bincount(spots, weights=perfect[traded])[kept] / horizons_per_target[kept]).sum()
        share = 100 * profit / best if best > 0 else float("nan")
        return {"": float(profit), "% of perfect": float(share), "days": int(kept.sum())}
