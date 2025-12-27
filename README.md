# epftoolbox2
Second version of open-access benchmark and toolbox for electricity price forecasting

epftoolbox/
│
├── __init__.py
│
├── pipelines/                     # 3 + 1 pipelines
│   ├── data_pipeline.py
│   ├── model_pipeline.py
│   ├── evaluation_pipeline.py
│   ├── epf_pipeline.py
│   └── base.py
│
├── data/
│   ├── sources/                   # ENTSOE, Open-Meteo, Calendar, CSV
│   │   ├── base.py
│   │   ├── entsoe.py
│   │   ├── open_meteo.py
│   │   ├── calendar.py
│   │   └── csv.py
│   └── cache/
│       └── file_cache.py
│
├── preprocessing/
│   ├── transformations/           # Lags, timezone, rolling, datetime
│   ├── validators/                # Continuity, range, schema
│   └── scalers/                   # Standard, minmax
│
├── models/
│   ├── base.py
│   ├── deterministic/             # Naive, OLS, LASSO, Ridge, NARX
│   └── probabilistic/             # Quantile, Conformal
│
├── evaluation/
│   ├── metrics/                   # MAE, RMSE, MAPE, Pinball
│   ├── reporting.py               # SIMPLIFIED: Aggregation + Excel export + basic plots
│
├── features/                     
│   ├── predictor.py              # Parse "load_d-{7-horizon}"
│   └── calendar.py               # Calendar feature helpers
│
└── utils/                        # Datetime, I/O, validation