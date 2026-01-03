---
title: Extending the Library
description: Creating custom components for epftoolbox2
---

# Extending the Library

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
        
        return df
    
    def get_cache_config(self) -> Optional[dict]:
        # Return dict for caching, or None to disable
        return {
            "source_type": "my_api",
            "api_url": self.api_url,
        }
```

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

---

## Custom Evaluator

Create custom metrics by extending the `Evaluator` base class:

```python
from epftoolbox2.evaluators.base import Evaluator
import pandas as pd
import numpy as np

class RMSEEvaluator(Evaluator):
    name = "RMSE"
    
    def compute(self, df: pd.DataFrame) -> float:
        return np.sqrt(((df["prediction"] - df["actual"]) ** 2).mean())


class MAPEEvaluator(Evaluator):
    name = "MAPE"
    
    def compute(self, df: pd.DataFrame) -> float:
        return ((df["prediction"] - df["actual"]).abs() / df["actual"].abs()).mean() * 100


class sMAPEEvaluator(Evaluator):
    name = "sMAPE"
    
    def compute(self, df: pd.DataFrame) -> float:
        numerator = (df["prediction"] - df["actual"]).abs()
        denominator = (df["prediction"].abs() + df["actual"].abs()) / 2
        return (numerator / denominator).mean() * 100
```

### Using Custom Evaluators

```python
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.models import OLSModel
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.exporters import TerminalExporter

pipeline = (
    ModelPipeline()
    .add_model(OLSModel(predictors=predictors, name="OLS"))
    .add_evaluator(MAEEvaluator())
    .add_evaluator(RMSEEvaluator())   # Custom
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
