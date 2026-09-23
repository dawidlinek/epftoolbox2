---
paths:
  - "epftoolbox2/data/**"
  - "tests/data/**"
---

# Data sources, transformers, validators

## Source contract
- `fetch(start, end)` returns a DataFrame with a **UTC** `DatetimeIndex`. Local time only appears after `TimezoneTransformer`.
- `get_cache_config()` must include **every** parameter that changes the output. The cache key is a hash of that dict (`CacheManager`, files under `.cache/sources/<key>/`); a missing parameter means stale data is served silently. Changing parsing logic does not invalidate existing caches — tell the user to clear `.cache/sources/` when it matters.
- Unit tests never hit the network: mock `session.get` (patterns in `tests/data/sources/test_entsoe.py`, `test_open_meteo.py`). Real-API tests get `@pytest.mark.integration` and the `entsoe_api_key` fixture, which skips without `ENTSOE_API_KEY`.

## Serialization contract (YAML)
`DataPipeline._serialize_component` dumps every public attribute (skipping `session`, `console`, `logger`, `lat`, `lon`) and reloads with `Class(**params)`. So constructor parameter names must equal attribute names, and derived/internal state must start with `_`. New components must be registered in `COMPONENT_REGISTRY` in `pipelines/data_pipeline.py` (see the `add-component` skill).

## Column naming (relative to the row's local date D)
- `x_d-k`: value of `x` on D-k (LagTransformer, positive lag). `x_d+h`: value of `x` on D+h as known on D (negative lag, Open-Meteo forecasts, model targets). Hourly/minute lags use `h`/`min` units, e.g. `price_h-24`.
- Day lags shift by **local calendar days** (`shift_calendar_days` in `epftoolbox2/_date_utils.py`, shared with the model's target shift): same wall-clock time on D-k; the hour skipped in spring is NaN; the repeated autumn hour maps occurrence to occurrence, else to the first occurrence (the model's targets keep NaN there instead, see `BaseModel._shift_index_by_calendar_days`). `h`/`min`/`s` lags, and any frequency that isn't whole days (`"24h"`, `"15min"`), are elapsed time and never named `d`. Day lags need a local index: run `LagTransformer` after `TimezoneTransformer` (it warns on a UTC index). The skipped spring hour gives NaN in `x_d-k` at 02:00 on the day k days later, so the model skips those forecasts (hourly with `d-1` and `d-2`: 14 a year).
- Open-Meteo uses the **previous-runs** API: `{prefix}_{col}_d+i` at row t is the forecast for t+i days issued about i days earlier, and rows later than now+13h are dropped. Never switch a backtest to the plain forecast/archive API — it returns latest or observed values, i.e. look-ahead.

## ENTSO-E specifics
Document types: A65/A16 actual load, A65/A01 day-ahead load forecast, A65/A31 week-ahead min/max (period starts anchored at UTC+2, hence `TimezoneTransformer.daily_columns` snapping), A75 generation, A69 wind/solar forecast, A44 prices. `prefer_15min` fetches 15-minute day-ahead prices where published. The API key travels as `securityToken`; never log it.

## Known issues (don't fix as a side effect of other work)
- Open-Meteo `{prefix}_{col}_d+i` columns are shifted by `24h * i` in UTC (`open_meteo.py`, `_parse_weather_data`), not by local calendar days, so around each DST switch they point one hour off the model's targets (23 rows per switch per i). `shift_calendar_days` is the fix.
- `api_key` is serialized into pipeline YAML in plain text by `DataPipeline.save()`.
