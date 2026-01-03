---
title: API Reference
description: Complete API reference for epftoolbox2
---

# API Reference

## DataPipeline

```python
class DataPipeline:
    def __init__(
        self,
        sources: Optional[List[DataSource]] = None,
        transformers: Optional[List[Transformer]] = None,
        validators: Optional[List[Validator]] = None,
    ): ...
    
    def add_source(self, source: DataSource) -> "DataPipeline": ...
    def add_transformer(self, transformer: Transformer) -> "DataPipeline": ...
    def add_validator(self, validator: Validator) -> "DataPipeline": ...
    
    def run(
        self,
        start: Union[str, pd.Timestamp],  # Start date
        end: Union[str, pd.Timestamp],    # End date
        cache: Union[bool, str] = False,  # Caching option
    ) -> pd.DataFrame: ...
    
    def save(self, path: Union[str, Path]) -> None: ...
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "DataPipeline": ...
```

---

## ModelPipeline

```python
class ModelPipeline:
    def __init__(self): ...
    
    def add_model(self, model: BaseModel) -> "ModelPipeline": ...
    def add_evaluator(self, evaluator: Evaluator) -> "ModelPipeline": ...
    def add_exporter(self, exporter: Exporter) -> "ModelPipeline": ...
    
    def run(
        self,
        data: pd.DataFrame,              # Input DataFrame
        test_start: str,                  # Test period start
        test_end: str,                    # Test period end
        target: str = "price",            # Target column
        horizon: int = 7,                 # Max horizon
        save_dir: Optional[str] = None,   # Result cache directory
    ) -> EvaluationReport: ...
```

---

## EvaluationReport

```python
class EvaluationReport:
    def summary(self) -> pd.DataFrame: ...
    def by_hour(self) -> pd.DataFrame: ...
    def by_horizon(self) -> pd.DataFrame: ...
    def by_hour_horizon(self) -> pd.DataFrame: ...
    def by_year(self) -> pd.DataFrame: ...
    def by_year_horizon(self) -> pd.DataFrame: ...
```

---

## Data Sources

```python
class EntsoeSource(DataSource):
    def __init__(self, country_code: str, api_key: str, type: List[str]): ...
    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...

class OpenMeteoSource(DataSource):
    def __init__(self, latitude: float, longitude: float, horizon: int = 7,
                 model: str = "jma_seamless", columns: List[str] = None,
                 prefix: str = ""): ...
    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...

class CalendarSource(DataSource):
    def __init__(self, country: str, timezone: str = None,
                 holidays: Union[str, bool] = "binary",
                 weekday: Union[str, bool] = "number",
                 hour: Union[str, bool] = False,
                 month: Union[str, bool] = False,
                 daylight: bool = False, prefix: str = ""): ...
    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...

class CsvSource(DataSource):
    def __init__(self, file_path: str, datetime_column: str = "datetime",
                 columns: List[str] = None, prefix: str = "",
                 datetime_format: str = None, separator: str = ","): ...
    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...
```

---

## Transformers

```python
class ResampleTransformer(Transformer):
    def __init__(self, freq: str = "1h", method: str = "linear"): ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...

class LagTransformer(Transformer):
    def __init__(self, columns: List[str], lags: List[int], freq: str): ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...

class TimezoneTransformer(Transformer):
    def __init__(self, target_tz: str): ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

---

## Validators

```python
class NullCheckValidator(Validator):
    def __init__(self, columns: List[str] = None, allow_nulls: bool = False): ...
    def validate(self, df: pd.DataFrame) -> ValidationResult: ...

class ContinuityValidator(Validator):
    def __init__(self, freq: str = "1h"): ...
    def validate(self, df: pd.DataFrame) -> ValidationResult: ...

class EdaValidator(Validator):
    def __init__(self, columns: List[str] = None): ...
    def validate(self, df: pd.DataFrame) -> ValidationResult: ...
```

---

## Models

```python
class OLSModel(BaseModel):
    def __init__(self, predictors: List, training_window: int = 365,
                 name: str = "Model"): ...

class LassoCVModel(BaseModel):
    def __init__(self, predictors: List, training_window: int = 365,
                 cv: int = 5, max_iter: int = 10000, name: str = "Model"): ...
```

---

## Evaluators

```python
class MAEEvaluator(Evaluator):
    name = "MAE"
    def compute(self, df: pd.DataFrame) -> float: ...
```

---

## Exporters

```python
class TerminalExporter(Exporter):
    def __init__(self, show: List[str] = None): ...
    def export(self, report: EvaluationReport) -> None: ...

class ExcelExporter(Exporter):
    def __init__(self, path: str, sheets: List[str] = None): ...
    def export(self, report: EvaluationReport) -> None: ...
```
