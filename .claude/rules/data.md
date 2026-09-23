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
- Open-Meteo uses the **previous-runs** API: `{prefix}_{col}_d+i` at row t is the forecast for t+i days issued about i days earlier, and rows later than now+13h are dropped. Never switch a backtest to the plain forecast/archive API — it returns latest or observed values, i.e. look-ahead.

## ENTSO-E specifics
- Document types: A65/A16 actual load, A65/A01 day-ahead load forecast, A65/A31 week-ahead daily min/max, A75 actual generation, A69 wind/solar day-ahead forecast, A44 day-ahead prices. Parsing is `xml.etree` with namespaces stripped; responses with a DTD are refused.
- `fetch(start, end)` returns [start, end). Daily columns (`_PERIOD_COLUMNS`) whose day covers `start` are re-stamped at `start`.
- curveType A03 omits a point that repeats the previous one: every Period is expanded to positions 1..N from its `timeInterval` and forward-filled (also trailing positions). A01 gaps stay missing.
- A31 comes as one Period per request starting at local midnight of the first requested day (verified for PL and DE_LU in winter, summer and across both DST switches in 2024; the old "UTC+2 anchor" claim was wrong). P1D positions are mapped onto local calendar days of the area (`AREA_TIMEZONES`, default Europe/Brussels), so daily values sit at local midnight and a week with the autumn switch (169 h) still has 7 positions. With fixed 24 h steps they drifted to 23:00/01:00 local after a switch; that drift is what `TimezoneTransformer.daily_columns` snapping repaired, and it is now a no-op for ENTSO-E data.
- Prices: only classificationSequence 1 or absent (SDAC). DE_LU/AT also publish sequence 2, a separate 15-minute auction (EXAA), on the same timestamps; never use it as SDAC. From 2025-10-01 PL and DE_LU publish only 15-minute SDAC prices, before it only hourly. Output is one grid: hourly with `prefer_15min=False` (complete quarter-hours averaged), 15-minute with `True` (hourly periods step-filled).
- A75: series in `outBiddingZone_Domain` are consumption (pumping; for AT every type, mostly zeros) and go to `generation_<type>_consumption`, never into the generation column.
- The API key travels in the `SECURITY_TOKEN` header, not the URL; error messages are redacted and carry the Acknowledgement `<Reason><text>`. 429/5xx/network errors are retried (`MAX_RETRIES`, honouring Retry-After); anything else raises `EntsoeApiError`, so a failed chunk never becomes a silent gap.
- `get_cache_config()` has `prefer_15min` and `schema_version` (`SCHEMA_VERSION`); bump it whenever parsing output changes.
- Publication deadlines (Reg. 543/2013) vs 12:00 CET gate closure, i.e. what a forecast issued on D may use, are tabulated on the docs page `docs/src/content/docs/data-sources/entsoe.mdx`. Short version: `load_actual` and `generation_*` actuals only `_d-1` and older; `load_forecast` up to `_d+1`; `load_forecast_daily_*` up to `_d+2` (`_d+3` after 10:00); `generation_*_forecast` only `_d+0` and older.

## Known issues (don't fix as a side effect of other work)
- `LagTransformer` shifts by elapsed time (`index + Timedelta * lag`), not calendar days. On a local-time index, `price_d-1` on the DST switch day comes from the wrong hour, and `d-7` lags are off by one hour for a week after each switch. The model's own targets use calendar-day shifts, so features and targets disagree around DST. Pinned by an `xfail(strict=True)` test in `tests/test_known_issues.py`.
- `api_key` is serialized into pipeline YAML in plain text by `DataPipeline.save()`.
