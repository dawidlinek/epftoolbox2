# epftoolbox2

A modern Python library for electricity price forecasting with modular data pipelines and model evaluation. Built as a complete rewrite of the original [epftoolbox](https://github.com/jeslago/epftoolbox), this library provides a flexible, extensible framework for downloading energy market data, building forecasting models, and evaluating their performance.

📖 **[Documentation](https://dawidlinek.github.io/epftoolbox2)** 

## Installation

```bash
pip install epftoolbox2
```

or with uv 

```bash
uv add epftoolbox2
```


## Key Features

- **Data Sources**: ENTSOE (load, generation, prices), Open-Meteo (weather forecasts), Calendar (holidays, weekday)
- **Transformers**: Resample, Timezone conversion, Lag features
- **Validators**: Null checks, Continuity checks, EDA statistics
- **Models**: OLS, LassoCV
- **Evaluators**: MAE
- **Exporters**: Excel with conditional formatting, Terminal output
- **Caching**: Built-in data caching to avoid redundant API calls
- **Pipelines**: Data and model pipelines that can be saved and loaded with .yaml files
- **Multithreading**: Multithreading with free GIL support in python 3.13t and above
- **Extensibility**: Extensible base classes for data sources, transformers, validators, models, evaluators, and exporters

## Quick Start

### Standalone source use

```python
from epftoolbox2.data.sources import EntsoeSource

source = EntsoeSource("PL", api_key="YOUR_KEY", type=["load", "price"])
df = source.run("2024-01-01", "2024-06-01", cache=True)
```

### Standalone transformer use

```python
from epftoolbox2.data.transformers import ResampleTransformer

transformer = ResampleTransformer(freq="1h")
df = transformer.run(df)
```

### Standalone validator use

```python
from epftoolbox2.data.validators import NullCheckValidator

validator = NullCheckValidator(columns=["load_actual", "price"])
df = validator.run(df)
```

### Standalone model use

```python
from epftoolbox2.models import OLSModel

model = OLSModel(predictors=["load_actual", "weekday"], name="OLS")
report = model.run(df, test_start="2024-04-01", test_end="2024-06-01", target="price")
```

### Standalone evaluator use

```python
from epftoolbox2.evaluators import MAEEvaluator

evaluator = MAEEvaluator()
report = evaluator.run(df, test_start="2024-04-01", test_end="2024-06-01", target="price")
```

### Standalone exporter use

```python
from epftoolbox2.exporters import ExcelExporter

exporter = ExcelExporter("results.xlsx")
report = exporter.run(df, test_start="2024-04-01", test_end="2024-06-01", target="price")
```

### Data Pipeline

Download and process electricity market data from ENTSOE, weather forecasts from Open-Meteo, and calendar features.

```python
from epftoolbox2.pipelines import DataPipeline
from epftoolbox2.data.sources import EntsoeSource, OpenMeteoSource, CalendarSource
from epftoolbox2.data.transformers import ResampleTransformer
from epftoolbox2.data.validators import NullCheckValidator

pipeline = (
    DataPipeline()
    .add_source(EntsoeSource("PL", api_key="YOUR_KEY", type=["load", "price"]))
    .add_source(OpenMeteoSource(latitude=52.23, longitude=21.01))
    .add_source(CalendarSource("PL", holidays="binary", weekday="number"))
    .add_transformer(ResampleTransformer(freq="1h"))
    .add_validator(NullCheckValidator(columns=["load_actual", "price"]))
)

df = pipeline.run("2024-01-01", "2024-06-01", cache=True)
```

### Model Pipeline

Train and evaluate forecasting models with built-in metrics and export capabilities.

```python
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.models import OLSModel, LassoCVModel
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.exporters import ExcelExporter

pipeline = (
    ModelPipeline()
    .add_model(OLSModel(predictors=["load_actual", "weekday"], name="OLS"))
    .add_model(LassoCVModel(predictors=["load_actual", "weekday"], name="Lasso"))
    .add_evaluator(MAEEvaluator())
    .add_exporter(ExcelExporter("results.xlsx"))
)

report = pipeline.run(df, test_start="2024-04-01", test_end="2024-06-01", target="price")
```



## Examples

See the `examples/` folder for complete working examples.