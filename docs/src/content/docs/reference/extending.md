---
title: Extending the Library
description: Creating custom components for epftoolbox2
---

epftoolbox2 is designed to be extensible. Create custom sources, transformers, validators, models, evaluators, and exporters.

## Custom DataSource

```python
from epftoolbox2.data.sources.base import DataSource
import pandas as pd
from typing import Optional

class MyApiSource(DataSource):
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        self._validate_config()
    
    def _validate_config(self) -> bool:
        if not self.api_key:
            raise ValueError("API key is required")
        return True
    
    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        # Convert timestamps to UTC
        start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
        end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")
        
        # Fetch data from API
        # ... your implementation ...
        
        return df  # UTC DatetimeIndex covering at least [start, end)
    
    def get_cache_config(self) -> Optional[dict]:
        # Return dict for caching, or None to disable
        return {
            "source_type": "my_api",
            "api_url": self.api_url,
            "schema_version": 1,  # bump when parsing changes, so cached frames are rebuilt
        }
```

`get_cache_config()` must contain every parameter that changes the output (not the API key). The pipeline drops
rows outside `[start, end)`, and a fetch for a sub-range must give the same values as a fetch for a larger range.
Otherwise a run assembled from cached pieces differs from a fresh one.

If some of the rows you return are placeholders or provisional for longer than the cache's two unsettled days (for
example NaN for a period the provider has not published yet), say so on the frame:

```python
df.attrs["settled_until"] = pd.Timestamp("2026-09-01", tz="UTC")  # rows at or after this are not settled
```

The cache then does not record those rows as covered and fetches them again next time, while the settled rows
before the timestamp come from disk. Leave it out when every row is settled. A naive timestamp is taken to be UTC.
The value may also be an ISO 8601 string (`timestamp.isoformat()`), which you need if the frame is saved with
`to_parquet`, because pandas writes `attrs` as JSON. See [Data Caching](/epftoolbox2/data-pipeline/caching/).

---

## Custom Transformer

```python
from epftoolbox2.data.transformers.base import Transformer
import pandas as pd

class RollingMeanTransformer(Transformer):
    def __init__(self, columns: list, window: int = 24):
        self.columns = columns
        self.window = window
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for col in self.columns:
            result[f"{col}_rolling_mean_{self.window}"] = df[col].rolling(self.window).mean()
        return result
```

---

## Custom Validator

```python
from epftoolbox2.data.validators.base import Validator
from epftoolbox2.data.validators.result import ValidationResult
import pandas as pd

class OutlierValidator(Validator):
    def __init__(self, columns: list, threshold: float = 3.0):
        self.columns = columns
        self.threshold = threshold
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        result = ValidationResult()
        
        for col in self.columns:
            if col not in df.columns:
                result.errors.append(f"Column '{col}' not found")
                result.is_valid = False
                continue
            
            z_scores = (df[col] - df[col].mean()) / df[col].std()
            outliers = (z_scores.abs() > self.threshold).sum()
            
            if outliers > 0:
                result.warnings.append(f"Column '{col}' has {outliers} outliers")
                result.info[f"{col}_outliers"] = int(outliers)
        
        return result
```

---

## Custom Model

```python
from epftoolbox2.models.base import BaseModel
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple
import numpy as np

class RandomForestModel(BaseModel):
    def __init__(self, predictors, training_window=365, n_estimators=100, **kwargs):
        super().__init__(predictors, training_window, **kwargs)
        self.n_estimators = n_estimators
    
    def _fit_predict(self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> Tuple[float, list]:
        model = RandomForestRegressor(n_estimators=self.n_estimators, n_jobs=1)
        model.fit(train_x, train_y)
        prediction = model.predict(test_x)[0]
        importances = model.feature_importances_.tolist()
        return prediction, importances
```

Under [`train_on="matching"`](/epftoolbox2/model-pipeline/predictor-functions/#training-only-on-the-days-with-the-same-list) a model receives only the rows of the days that matched, so `train_x` and `train_y` aren't consecutive days. A custom model that weights rows by their position must not assume they are.

A model with only `_fit_predict` fits once per forecast and raises `ValueError` if given `refit`. To take
[`refit`](/epftoolbox2/model-pipeline/time-and-calendars/#refitting-on-a-calendar), also implement `_fit_model(train_x, train_y)`, which
returns the fit, and `_predict_fitted(fitted, test_x)`, which returns `(prediction, coefficients)` for the one row `test_x` and is called once
per forecast. `_predict_fitted(_fit_model(x, y), test_x)` must equal `_fit_predict(x, y, test_x)` to the bit. Define all three in the
same class: a subclass of `OLSModel`, `LassoCVModel`, `RidgeModel` or `LEARModel` that overrides one or two of them raises on `refit`, since `refit=None` and
`refit` would fit different models; one that overrides none keeps its parent's fit and refits.

### Predicting a refit block in one call

A model whose call costs more than its rows, such as an in-context model that reads its whole context on every call, can also
define `_predict_block(fitted, rows)`: from the fit of `_fit_model`, a prediction (in the scaled target units `_predict_fitted`
returns) and a coefficient list for each standardised row of `rows`, in their order. With `refit`, each unit is then fitted once and
predicted in one call holding the issue rows of every day of its block that the frame has with finite inputs, not only the days the
run forecasts, in date order, padded to the rule's nominal length (7 rows for `"7D"`, 31 for `"1MS"`, 31 k for `"kMS"`) with copies
of the first row; each forecast is taken from its own row. A call so has one shape and the same rows whether a day runs alone or in
its block, from any `test_start`, on some weekdays only or after a resume, and its forecasts are the same to the bit in each. The
block's other days enter as issue rows only, never their targets and never a day after the block, so a row's prediction must not
depend on the other rows of its call: [`verify`](/epftoolbox2/model-pipeline/verification/#rerun-and-co-batch) checks that with its co-batch check,
which such a class gets on by default (it gets `_predicts_together` and `_co_batch_items`, the other days of the block).

```python
import numpy as np
from epftoolbox2.models.base import BaseModel


class KernelModel(BaseModel):
    """The kernel-weighted mean of the training targets: the fit is the training rows, each forecast row on its own."""

    def __init__(self, predictors, training_window=728, bandwidth=2.0, block_predict=True, **kwargs):
        super().__init__(predictors, training_window, **kwargs)
        self.bandwidth = bandwidth
        self.block_predict = block_predict

    @property
    def _model_kwargs(self):
        return {"bandwidth": self.bandwidth, "block_predict": self.block_predict}

    def _fit_predict(self, train_x, train_y, test_x):
        return self._predict_fitted(self._fit_model(train_x, train_y), test_x)

    def _fit_model(self, train_x, train_y):
        return train_x.copy(), train_y.copy()

    def _predict_fitted(self, fitted, test_x):
        predictions, coefficients = self._predict_block(fitted, test_x)
        return predictions[0], coefficients[0]

    def _predict_block(self, fitted, rows):
        context_x, context_y = fitted
        predictions = []
        for row in rows:
            distance = ((context_x - row) ** 2).sum(axis=1)
            weights = np.exp(-(distance - distance.min()) / (2 * self.bandwidth**2))
            predictions.append(weights @ context_y / weights.sum())
        return np.array(predictions), [[] for _ in rows]


model = KernelModel(["load_d+{horizon}", "wind_d+{horizon}"], refit="7D")
```

Such a class has `block_predict = True`; `block_predict=False` predicts one row at a time through `_predict_fitted`, as every other
model does. The model you run decides: the workers get its answer with their configuration, whatever `_model_kwargs` holds, and the
manifest records it, so a resume with the other setting is refused. `_predict_block` is called only when it is defined in the class
that defines `_predict_fitted`: a subclass that changes the estimator (`_fit_predict`, `_fit_model`, `_predict_fitted`) but inherits
`_predict_block`, or one that overrides `_predict_block` alone, predicts one row at a time, so its forecasts are always those of its
own estimator. With `refit=None` every forecast is fitted on its own, as for any model. Each row's coefficients are a flat list of
numbers (or a 1-D numeric array, written as its list); anything else raises a `TypeError` naming the model.

`ModelPipeline(engine="on")` runs a model in an [engine group](/epftoolbox2/model-pipeline/engine/) only
if one class defines all three and `_fit_stats` (the fit from shared statistics), as the four built-in fitting models do, and keeps
`BaseModel`'s `run`. A custom model like the one above runs on its own there, with the numbers it has with `engine="off"`; so does a
subclass of a built-in model that overrides its fit without `_fit_stats`. A subclass of `OLSModel`, `RidgeModel` or `LEARModel` that
keeps all four is fitted there from shared statistics as its parent is (with `scaler="standard"`), through its parent's estimator:
`_fit_stats` hands a fit back to the model path when `_estimator()` isn't exactly the estimator the statistics reproduce. Each instance of a subclass that builds its period
frames itself (`_preprocess`) is a group of its own, which reads its own frames, so a setting of the instance that its
`_preprocess` reads (a lag, say) is that instance's; so is a model with such a method set on the instance itself, and one with
its own `run`, `_run_tasks` or `_build_hour_arrays` (on the class or the instance) runs on its own.

### Day and pooled units

Every model above fits one period of the day at a time: a unit is the rows of one period and one target. A class can set
`_unit_kind` to fit every period of a day at once instead, on the [refit calendar](/epftoolbox2/model-pipeline/time-and-calendars/#refitting-on-a-calendar)
(so it needs `refit`; `"1D"` fits every issue day):

- **`_unit_kind = "day"`**: the rows are local dates and the outputs the P periods of the target day (24, 96 at 15 minutes). Define
  `_fit_days(x, y, y_mask)`, which gets the scaled inputs of the training dates (dates, inputs), the targets (dates, P) and which of
  them train, and `_predict_days(fitted, rows)`, which returns (rows, P). Every predictor must hold one value per local date, such
  as a daily profile (`price_h05_d-1` from [`DailyProfileTransformer`](/epftoolbox2/transformers/daily-profile/)) or a calendar
  flag (`is_monday_d+{horizon}`): a column that changes with the period, such as `price_d-1`, raises before any fit, naming its
  profile names (`price_h00_d-1 … price_h23_d-1`). A date with a value on some rows and NaN on others is fine. The check reads
  the dates the run's fits train on, not the issue dates (the probes of `verify` perturb an issue date's cells one by one); a
  date's input is the value on its last row, in period order, that has one, so a plain run reads that row's value of a column
  that changes with the period only on dates no fit trains on. `verify` checks every date of the frame first. A slot with no
  target is masked and drops out of its own period's fit alone, never shifted to another period: the hour a spring switch skips,
  the repeated autumn hour, a missing value. With `newest_target="published"` each period trains on the W dates its own period
  unit would, up to its own newest target known by the anchor's issue time, and its slots outside them are masked. Masked slots of
  `y` are NaN.
- **`_unit_kind = "pooled"`**: the rows are the (date, period) rows of every period, by date then period, with one output: each
  predictor is read on its period's own row (`price_d-1` is that hour's price, `price_h05_d-1` the date's 05:00 price), and the
  name `period` in a list adds the row's period (0 to P-1) as a feature. Define `_fit_model(train_x, train_y)` and
  `_predict_block(fitted, rows)` as above, and a `target_transform` attribute: `"per_period"` (each period's own Invariant) or
  `"pooled"` (one over every row). A period with no training row is skipped (`no_finite_training_rows`) with either, so the
  forecasts a run makes don't depend on the transform.

Both units scale the inputs with the model's `scaler` (its input part; binary 0/1 columns are kept) and transform the targets with
each period's Invariant, asinh of the median and MAD of its training targets (`epftoolbox2.scalers.target.MultiTargetAsinh`,
`AsinhScaler(apply_to="target", standardise=False)` per column); a forecast of period p is mapped back with p's. Each unit predicts
every issue day of its block that the frame has in one call of the block's nominal length (7 days for `"7D"`, times P rows for a
pooled unit), padded with copies of the first, so a forecast is the same alone, in its block, after a resume and whatever the run's
`test_start` or the data's end; a row's forecasts must not depend on the call's other rows, which `verify`'s co-batch check runs on
by default. Records are those of every planned (period, horizon, issue day), keyed as any model's; a day unit's have no
coefficients. A list may change with the horizon and, for a predictor function of `ctx`, with the issue day, never with the period of
one issue day. The manifest records `unit_kind` and `target_transform`, and a resume with another refuses. Not in this release: the
engine, `PooledModel`, `train_on="matching"` and `period` in a predictor function's list.

```python
import numpy as np
from epftoolbox2.models.base import BaseModel


class DailyRidge(BaseModel):
    """A ridge per period of the day, all fitted from one call on the dates of a unit."""

    _unit_kind = "day"

    def __init__(self, predictors, training_window=728, alpha=1.0, refit="7D", **kwargs):
        super().__init__(predictors, training_window, refit=refit, **kwargs)
        self.alpha = alpha

    @property
    def _model_kwargs(self):
        return {"alpha": self.alpha}

    def _fit_predict(self, train_x, train_y, test_x):
        raise NotImplementedError("fits day units")

    def _fit_days(self, x, y, y_mask):
        coefs, intercepts = np.zeros((x.shape[1], y.shape[1])), np.zeros(y.shape[1])
        for period in range(y.shape[1]):
            rows = y_mask[:, period]
            if rows.any():
                xc, yc = x[rows] - x[rows].mean(axis=0), y[rows, period] - y[rows, period].mean()
                coefs[:, period] = np.linalg.solve(xc.T @ xc + self.alpha * np.eye(x.shape[1]), xc.T @ yc)
                intercepts[period] = y[rows, period].mean() - x[rows].mean(axis=0) @ coefs[:, period]
        return coefs, intercepts

    def _predict_days(self, fitted, rows):
        coefs, intercepts = fitted
        return np.array([row @ coefs + intercepts for row in rows])


profiles = [f"price_h{hour:02d}_d-1" for hour in range(24)] + ["is_weekend_d+{horizon}"]
model = DailyRidge(profiles, training_window=364, name="daily_ridge")
```

### Sequence models: series instead of rows (internal)

A foundation model reads series, not the predictor columns of a row. `epftoolbox2.models._history` (internal, for the foundation
models to come) gives it what one forecast may read, cut by the information set. `HistorySpec(context, known_ahead, past,
target, accepts_missing=False, newest_target="d-1")` names the series as base columns (`"load_forecast"`, never
`"load_forecast_d+1"`), and `history_batch(spec, frame, issue_days, information_set, freq, horizon)` returns, for each issue day D
(`history_batches(..., horizons)` reads every horizon at once, and `arrays=False` the cuts alone):

- `target` (T,): the target's last T = context × periods-per-day steps in physical time, ending at its cut: under
  `newest_target="published"` its newest step known at the issue time by its rule, at the latest the last step of the day before
  the target day (so at 14:00, when D+1's prices are out, the forecast at h = 1 is still of all of D+1), under `"d-1"` (every
  model's default) the earlier of that and the last step of D-1, the newest target such a model trains on. T is the same on
  every day; a context across a DST switch covers an hour more or less of local time.
- `past` {name: (T,)}: on the same steps, NaN after the series' own newest known step where the backend takes missing values
  (`accepts_missing`), else the series is late.
- `known` {name: (T + S,)}: the same steps and the S steps ahead, to the end of the target day D + h; one not known that far is late.
- `slots` (S,): the (date, period) each step ahead forecasts, None before the target day. The repeated autumn hour is forecast by its
  second step, and the hour a spring switch skips has none. For a price issued at 11:45 under `"published"`, S is 24 (23 or 25 when
  the target day is a DST day); for a load whose actuals are out to 10:00, 38; under `"d-1"`, 48 for either at horizon 1.
- `gaps` and `missing`: the data's missing values of each series in the context and in the steps ahead, and the series the backend
  can't take as they are (a context with no value at all, a known-ahead value missing ahead, and without `accepts_missing` any
  missing value).

Which values are known comes from `information_set.StepsReader`, the same per-cell reading `verify`'s probe perturbs by. A value the
probe perturbs is in a context only where a series is late, and a late series is listed at `availability="warn"` and refused at
`"raise"`, as a late predictor is; `report.availability()` and `explain()` show one row per series (how far it is read and its
rule's vintage class), not one per lagged step. `HistoryBatch.batches(batch_size)` groups the items by S and pads each call to
`batch_size` with copies of its first item, so a call's shape doesn't depend on what else a run holds. A model reads history when
its class defines `_history_spec(target)`, which must give the model's own `newest_target`. `BaseModel`'s checks then take its
series as inputs it reads (the manifest hashes them, `report.vintages()` lists them, `verify` perturbs them) and refuse a late one at
`"raise"` or list it at `"warn"`; its manifest records every series' rule, the issue time and zone (`training_cut`, compared on
resume whatever its `newest_target`) and hashes what its last issue day reads after `test_end` (`history_ahead`: its known-ahead
series through `test_end` plus the largest horizon, and its target where a late issue time reads it there). `verify` gives its runs
the information set whatever its `newest_target` (and still perturbs the target from D on under `"d-1"`); its positive control
perturbs each series of its spec on its own, the target too (a series whose perturbation moves nothing, such as one read from a
copy of the model's own, leaves it `inconclusive`); and its co-batch check runs each forecast on the frame a
run on D has for it (`target_day_cut`): every row up to D, and after D only what that forecast reads (its known-ahead series to the
end of D + h), so no later day of its call keeps what a run on D couldn't have. A forecast that moves there depends on where the
frame ends. Such a model can't be a `ResidualModel`'s base in this release.

---

## Custom Evaluator

Create custom metrics by extending the `Evaluator` base class:

```python
from epftoolbox2.evaluators.base import Evaluator
import pandas as pd
import numpy as np

class MAPEEvaluator(Evaluator):
    name = "MAPE"
    
    def compute(self, df: pd.DataFrame, **kwargs) -> float:
        return ((df["prediction"] - df["actual"]).abs() / df["actual"].abs()).mean() * 100


class sMAPEEvaluator(Evaluator):
    name = "sMAPE"
    
    def compute(self, df: pd.DataFrame, **kwargs) -> float:
        numerator = (df["prediction"] - df["actual"]).abs()
        denominator = (df["prediction"].abs() + df["actual"].abs()) / 2
        return (numerator / denominator).mean() * 100
```

### Evaluator Interface

Every evaluator must implement:

- **`name`**: class attribute used as the column header in reports
- **`compute(self, df, **kwargs) -> float`**: aggregate metric over a DataFrame with `prediction` and `actual` columns

The frame also carries the forecast keys `run_date`, `target_date` (datetimes), `hour` and `horizon`, which identify each forecast. Metrics that only look at `prediction` and `actual` can ignore them.

The `**kwargs` may include `model_dfs`: a dict of all model DataFrames (same columns) for the current data slice, used by cross-model evaluators like `rMAEEvaluator`. A cross-model metric must join the frames on the forecast keys, because models can produce different forecast sets.

`compute` may return a dict instead of a float. Each entry becomes a column of the tables, named `"{name} {key}"` (the key `""` gives the name alone, as `EconomicEvaluator` does for `profit`):

```python
class ErrorQuantiles(Evaluator):
    name = "AE"

    def compute(self, df: pd.DataFrame, **kwargs) -> dict:
        errors = (df["prediction"] - df["actual"]).abs()
        return {"p50": errors.quantile(0.5), "p95": errors.quantile(0.95)}  # columns "AE p50", "AE p95"
```

### Evaluators over a set of models

A metric whose value for a model depends on the others (a test, a confidence set) overrides `compute_group(self, model_dfs) -> dict`, which the report calls once per table group with every model's frame of that group. It returns `{model: float or dict}`. By default it loops `compute`, so most evaluators never need it; this is what `DMEvaluator` and `MCSEvaluator` use. Subclass `SetEvaluator` from `epftoolbox2.evaluators.base` so that you don't have to write `compute` too:

```python
from epftoolbox2.evaluators.base import SetEvaluator

class BestShare(SetEvaluator):
    """The share of the models' total MAE that each model holds."""

    name = "MAE share"

    def compute_group(self, model_dfs) -> dict:
        maes = {name: (df["prediction"] - df["actual"]).abs().mean() for name, df in model_dfs.items()}
        total = sum(maes.values())
        return {name: mae / total for name, mae in maes.items()}
```

A set-level evaluator that compares models must align them on the forecast keys first and print how many it used: the kernels in `epftoolbox2.evaluators.stats` pair by position and drop any index.

### Using Custom Evaluators

```python
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.models import OLSModel
from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator
from epftoolbox2.exporters import TerminalExporter

pipeline = (
    ModelPipeline()
    .add_model(OLSModel(predictors=predictors, name="OLS"))
    .add_evaluator(MAEEvaluator())
    .add_evaluator(RMSEEvaluator())
    .add_evaluator(MAPEEvaluator())   # Custom
    .add_exporter(TerminalExporter())
)

report = pipeline.run(...)

print(report.summary())
#    model       MAE      RMSE     MAPE
# 0    OLS  26.0199   32.4512   15.23
```

---

## Custom Exporter

```python
from epftoolbox2.exporters.base import Exporter
from epftoolbox2.results.report import EvaluationReport
import json

class JsonExporter(Exporter):
    def __init__(self, path: str):
        self.path = path
    
    def export(self, report: EvaluationReport) -> None:
        data = {
            "summary": report.summary().to_dict(orient="records"),
            "by_horizon": report.by_horizon().to_dict(orient="records"),
            "by_hour": report.by_hour().to_dict(orient="records"),
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)
```

---

## Using Custom Components

```python
from epftoolbox2.pipelines import DataPipeline, ModelPipeline

# Data Pipeline with custom components
data_pipeline = (
    DataPipeline()
    .add_source(MyApiSource(api_url="...", api_key="..."))
    .add_transformer(RollingMeanTransformer(columns=["price"]))
    .add_validator(OutlierValidator(columns=["price"]))
)

# Model Pipeline with custom components
model_pipeline = (
    ModelPipeline()
    .add_model(RandomForestModel(predictors=predictors, name="RF"))
    .add_evaluator(RMSEEvaluator())
    .add_evaluator(MAPEEvaluator())
    .add_exporter(JsonExporter("results.json"))
)
```
