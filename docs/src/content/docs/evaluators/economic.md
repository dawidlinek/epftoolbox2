---
title: EconomicEvaluator
description: What a model's forecasts earn when they schedule a battery
---

Adds what a model's forecasts earn to the report's tables, when they schedule a trading strategy and are settled at the actual prices. It is a value column, not a ranking criterion: in the study behind the library's defaults, ranking models by profit agreed with ranking them by MAE only so far (rank correlation about 0.7), and two seeds of one model differed by about 300 EUR a year.

## Output columns

| Column | Meaning |
|--------|---------|
| `profit` | EUR (with prices in EUR/MWh) over the forecasts' target days |
| `profit % of perfect` | Against the same strategy with perfect foresight on the same days |
| `profit days` | The target days traded |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | BatteryArbitrage | `None` | The strategy to run; `None` is `BatteryArbitrage()` |

### BatteryArbitrage

One charge and one discharge a day.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `power_mw` | float | `1` | Charge and discharge power |
| `energy_mwh` | float | `2` | Energy bought per cycle; `energy_mwh / power_mw` must be a whole number of periods |
| `efficiency` | float | `0.9` | Round-trip efficiency: the share of the energy bought that is sold |

**The rule.** Per target day, for every split period k:

1. Charge in the forecast-cheapest periods before k and discharge in the forecast-dearest periods from k on, as many of each as it takes to move `energy_mwh` at `power_mw`.
2. Take the split with the largest forecast profit, `efficiency × Σ discharge prices − Σ charge prices`.
3. Trade only if that forecast profit is positive, then settle at the actual prices.

Perfect foresight is the same rule on the actual prices. On hourly data 1 MW / 2 MWh is 2 + 2 hours at 1 MWh each; on 15-minute data 8 + 8 quarter-hours at 0.25 MWh each. Exact ties in the forecast go to the earlier period. This is the rule of the evaluation-protocol study behind the library's defaults, and it reproduces that study's profits exactly on stored forecasts.

## Basic Usage

```python
from epftoolbox2.evaluators import BatteryArbitrage, EconomicEvaluator

evaluator = EconomicEvaluator(BatteryArbitrage(power_mw=1, energy_mwh=2, efficiency=0.9))
```

## In Pipeline

```python
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.models import OLSModel
from epftoolbox2.evaluators import MAEEvaluator, EconomicEvaluator

pipeline = (
    ModelPipeline()
    .add_model(OLSModel(predictors=predictors, name="OLS"))
    .add_evaluator(MAEEvaluator(), EconomicEvaluator())
)

report = pipeline.run(...)
print(report.summary())
#   model  forecasts      MAE    profit  profit % of perfect  profit days
# 0   OLS       8760  26.0199  78653.04                93.90          365
```

## Notes

- **Whole days only.** A schedule needs the whole day, so a day counts only when all its periods have finite forecasts: 24 hours (96 quarter-hours), 25 hours (100) numbered through, or 23 hours (92) lacking one hour between 01:00 and 04:00, where clocks change. A real run numbers periods by the clock and writes that form around a switch (the repeated autumn hour is one period, and the hour that has no target is absent), and a short day numbered through (0-22) counts on a Sunday. Other days are left out, even for one missing hour, and `profit days` shows how many were used. In a table cut by hour no day is whole, so the columns are NaN there.
- **Each model on its own days.** The columns of two models with different forecasts cover different days. Compare `profit days` before you read them side by side.
- **A target day counts once.** With several horizons in a group (`summary()`), a target day earns the mean of what its horizons earn, so the sum is what one battery could have earned over those days. `by_horizon()` shows each horizon on its own.
- A `strategy` can be any object with a method `profit(forecast, actual, periods_per_hour)` that takes days × periods arrays and returns the profit per day. To load from YAML its class must be in `COMPONENT_REGISTRY["strategies"]`.
- Serializable to YAML: the strategy is saved inside the evaluator as `{class: BatteryArbitrage, params: {...}}`.
