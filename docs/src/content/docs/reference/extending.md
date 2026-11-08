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

`ModelPipeline(engine="on")` runs a model in an [engine group](/epftoolbox2/model-pipeline/forecasting/#many-models-the-engine) only
if one class defines all three and `_fit_stats` (the fit from shared statistics), as the four built-in fitting models do, and keeps
`BaseModel`'s `run`. A custom model like the one above runs on its own there, with the numbers it has with `engine="off"`; so does a
subclass of a built-in model that overrides its fit without `_fit_stats`. A subclass of `OLSModel`, `RidgeModel` or `LEARModel` that
keeps all four is fitted there from shared statistics as its parent is (with `scaler="standard"`), through its parent's estimator:
`_fit_stats` hands a fit back to the model path when `_estimator()` isn't exactly the estimator the statistics reproduce. Each instance of a subclass that builds its period
frames itself (`_preprocess`) is a group of its own, which reads its own frames, so a setting of the instance that its
`_preprocess` reads (a lag, say) is that instance's; so is a model with such a method set on the instance itself, and one with
its own `run`, `_run_tasks` or `_build_hour_arrays` (on the class or the instance) runs on its own.

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
