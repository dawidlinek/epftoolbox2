---
title: Working with Results
description: Using EvaluationReport to analyze model performance
---

# Working with Results

The `ModelPipeline.run()` method returns an `EvaluationReport` object with various analysis methods.

## EvaluationReport Methods

```python
report = pipeline.run(...)

# Overall summary per model
summary = report.summary()

# Breakdown by hour of day
by_hour = report.by_hour()

# Breakdown by forecast horizon
by_horizon = report.by_horizon()

# Combined hour × horizon
by_hour_horizon = report.by_hour_horizon()

# Breakdown by year
by_year = report.by_year()

# Combined year × horizon
by_year_horizon = report.by_year_horizon()
```

## Example Output

```python
print(report.summary())
#    model       MAE
# 0    OLS  26.0199
# 1  Lasso  28.1098

print(report.by_horizon())
#    model  horizon      MAE
# 0    OLS        1  18.5432
# 1    OLS        2  22.1234
# 2    OLS        7  35.4567
```

## Accessing Raw Predictions

```python
# Get all predictions as DataFrame
predictions = report.predictions

print(predictions.head())
#                  model  horizon  prediction  actual
# 2024-02-01 00:00  OLS        1       45.23   48.50
```
