---
paths:
  - "epftoolbox2/data/**"
  - "tests/data/**"
---

# Data sources, transformers, validators

## Source contract
- `fetch(start, end)` returns a DataFrame with a **UTC** `DatetimeIndex` covering at least `[start, end)`. Local time only appears after `TimezoneTransformer`. `DataPipeline` clips every source to `[start, end)` (cached or not), and a cached run is assembled from pieces fetched separately, so a fetch of a sub-range must give the same values as a fetch of a larger range.
- `get_cache_config()` must include **every** parameter that changes the output; a missing parameter means stale data is served silently. The cache key (`CacheManager`, Parquet files under `.cache/sources/<key>/`) hashes `CACHE_FORMAT_VERSION`, the source class and that dict. When parsing logic changes, add or bump `"schema_version"` in `get_cache_config()`; when the cache layout changes, bump `CACHE_FORMAT_VERSION` in `data/cache_manager.py`.
- The cache records only what a source returned, never the last `UNSETTLED_PERIOD` (2 days) before now; each file is read only inside its recorded range; writes are atomic and each key is locked (`filelock`) around fetch + write.
- Unit tests never hit the network: mock `session.get` (patterns in `tests/data/sources/test_entsoe.py`, `test_open_meteo.py`). Real-API tests get `@pytest.mark.integration` and the `entsoe_api_key` fixture, which skips without `ENTSOE_API_KEY`.

## Serialization contract (YAML)
`DataPipeline._serialize_component` dumps every public attribute (skipping `session`, `console`, `logger`, `lat`, `lon`) and reloads with `Class(**params)`. So constructor parameter names must equal attribute names, and derived/internal state must start with `_`. New components must be registered in `COMPONENT_REGISTRY` in `pipelines/data_pipeline.py` (see the `add-component` skill).

## Column naming (relative to the row's local date D)
- `x_d-k`: value of `x` on D-k (LagTransformer, positive lag). `x_d+h`: value of `x` on D+h as known on D (negative lag, Open-Meteo forecasts, model targets). Hourly/minute lags use `h`/`min` units, e.g. `price_h-24`.
- Open-Meteo uses the **previous-runs** API: `{prefix}_{col}_d+i` at row t is the forecast for t+i days issued about i days earlier, and rows later than now+13h are dropped. Never switch a backtest to the plain forecast/archive API — it returns latest or observed values, i.e. look-ahead.

## ENTSO-E specifics
Document types: A65/A16 actual load, A65/A01 day-ahead load forecast, A65/A31 week-ahead min/max (period starts anchored at UTC+2, hence `TimezoneTransformer.daily_columns` snapping), A75 generation, A69 wind/solar forecast, A44 prices. `prefer_15min` fetches 15-minute day-ahead prices where published. The API key travels as `securityToken`; never log it.

## Known issues (don't fix as a side effect of other work)
- `LagTransformer` shifts by elapsed time (`index + Timedelta * lag`), not calendar days. On a local-time index, `price_d-1` on the DST switch day comes from the wrong hour, and `d-7` lags are off by one hour for a week after each switch. The model's own targets use calendar-day shifts, so features and targets disagree around DST. Pinned by an `xfail(strict=True)` test in `tests/test_known_issues.py`.
- `api_key` is serialized into pipeline YAML in plain text by `DataPipeline.save()`.
