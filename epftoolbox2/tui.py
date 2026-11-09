"""Interactive wizard for the epf CLI.

Launched when `epf` is run with no subcommand. Guides the user through
building and running pipelines without writing YAML by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

# ---------------------------------------------------------------------------
# Param schemas
# ---------------------------------------------------------------------------
# Each entry is a list of field dicts:
#   name, description, required, default, kind
# kind values:
#   "text"         – free text input
#   "int"          – integer input (validated)
#   "float"        – float input (validated)
#   "confirm"      – yes/no
#   "select"       – single choice from choices list
#   "multiselect"  – one or more choices (checkbox)
#   "comma_list"   – comma-separated text → list[str]
#   "column_select"– checkbox from available_columns (falls back to comma_list)
#   "lag_list"     – comma-separated text → list[int] (e.g. "1,2,7" or "-7,-1,0")
#   "window"       – a training window: days, "expanding", or a comma-separated list of them (e.g. "56,84,728,expanding")
#   "json"         – free text parsed as JSON → dict/list (e.g. {"price": [-500, 4000]})
#   "bounds"       – "low,high", either side empty for none (e.g. "0,1.05" or ",100") → (low, high), empty for None


# The training settings every network shares (TorchModel and its presets), after each class's own fields.
def _torch_fields(seeds: int) -> list[dict]:
    return [
        {
            "name": "training_window",
            "description": "Training window in days, 'expanding', or a list averaged inside the model (e.g. 364,728,expanding)",
            "required": False,
            "default": "728",
            "kind": "window",
        },
        {
            "name": "refit",
            "description": "Refit calendar: 7D once a week (Mondays), 1MS once a month, 1D every issue day",
            "required": False,
            "default": "7D",
            "kind": "text",
        },
        {
            "name": "seeds",
            "description": "Networks per fit, averaged",
            "required": False,
            "default": seeds,
            "kind": "int",
        },
        {"name": "lr", "description": "Adam's learning rate", "required": False, "default": 3e-4, "kind": "float"},
        {
            "name": "max_epochs",
            "description": "At most this many epochs",
            "required": False,
            "default": 300,
            "kind": "int",
        },
        {
            "name": "patience",
            "description": "Stop after this many epochs without a better early-stopping loss",
            "required": False,
            "default": 20,
            "kind": "int",
        },
        {
            "name": "early_stopping",
            "description": "Share of each fit's training days held out for early stopping (0: none)",
            "required": False,
            "default": 0.2,
            "kind": "float",
        },
        {
            "name": "loss",
            "description": "Loss: l1, mse, or an elementwise function as package.module:function",
            "required": False,
            "default": "l1",
            "kind": "text",
        },
        {
            "name": "device",
            "description": "auto (CUDA if it runs, else the CPU), cpu or cuda",
            "required": False,
            "default": "auto",
            "kind": "select",
            "choices": ["auto", "cpu", "cuda"],
        },
        {
            "name": "newest_target",
            "description": "Newest training target: d-1, or every target published by the issue time (needs an issue time)",
            "required": False,
            "default": "d-1",
            "kind": "select",
            "choices": ["d-1", "published"],
        },
        {"name": "random_state", "description": "Base seed", "required": False, "default": 0, "kind": "int"},
        {
            "name": "batch_size",
            "description": "Rows per training step",
            "required": False,
            "default": 64,
            "kind": "int",
        },
        {
            "name": "weight_decay",
            "description": "Adam's L2 weight decay",
            "required": False,
            "default": 0.0,
            "kind": "float",
        },
        {
            "name": "stack",
            "description": "auto (one network after another on a CPU, a padded stack on a GPU), loop or vmap",
            "required": False,
            "default": "auto",
            "kind": "select",
            "choices": ["auto", "loop", "vmap"],
        },
        {
            "name": "tf32",
            "description": "TF32 products on a GPU",
            "required": False,
            "default": False,
            "kind": "confirm",
        },
    ]


# A per-row target (`models._row_target`): the fields every model that takes one shares, after its own; `bounds_only` for a
# model that takes `bounds` alone (`NaiveModel`).
def _row_target_fields(bounds_only: bool = False) -> list[dict]:
    bounds = {
        "name": "bounds",
        "description": (
            "Clip the forecast: low,high (either side empty for none), in multiples of the scale with normalise_by, else "
            "target units; empty for none"
        ),
        "required": False,
        "default": None,
        "kind": "bounds",
    }
    if bounds_only:
        return [bounds]
    return [
        {
            "name": "normalise_by",
            "description": (
                "Divide each training row's target by this column, multiply the forecast by the issue row's value (e.g. "
                "installed_capacity_solar_d+{horizon}); empty for none"
            ),
            "required": False,
            "default": None,
            "kind": "text",
        },
        bounds,
        {
            "name": "zero_when",
            "description": "Forecast 0, with no fit, where this 0/1 column is 1 on the issue row (e.g. is_night_d+{horizon})",
            "required": False,
            "default": None,
            "kind": "text",
        },
    ]


_ACTIVATION_FIELD = {
    "name": "activation",
    "description": "Activation of the hidden layers",
    "required": False,
    "default": "relu",
    "kind": "select",
    "choices": ["relu", "gelu", "tanh", "sigmoid", "softplus", "elu", "selu", "silu", "leaky_relu"],
}


PARAM_SCHEMAS: dict[str, list[dict]] = {
    "EntsoeSource": [
        {
            "name": "country_code",
            "description": "Area code (e.g. PL, DE_LU); prices need a bidding zone",
            "required": True,
            "default": None,
            "kind": "text",
        },
        {"name": "api_key", "description": "ENTSOE API key", "required": True, "default": None, "kind": "text"},
        {
            "name": "type",
            "description": "Data types to fetch",
            "required": True,
            "default": None,
            "kind": "multiselect",
            "choices": ["load", "generation", "price", "transfer_capacity", "installed_capacity"],
        },
        {
            "name": "prefer_15min",
            "description": "15-minute price grid (hourly periods step-filled)? No = hourly grid (quarter-hours averaged)",
            "required": False,
            "default": False,
            "kind": "confirm",
        },
        {
            "name": "neighbours",
            "description": "Neighbouring bidding zones for transfer_capacity, comma-separated (e.g. DE_LU,CZ)",
            "required": False,
            "default": None,
            "kind": "text",
        },
        {
            "name": "prefix",
            "description": "Column name prefix (optional)",
            "required": False,
            "default": "",
            "kind": "text",
        },
        {
            "name": "price_exaa",
            "description": "Add the EXAA auction as price_exaa (AT, DE_LU and DE_AT_LU only; needs type price)?",
            "required": False,
            "default": False,
            "kind": "confirm",
        },
    ],
    "OpenMeteoSource": [
        {
            "name": "latitude",
            "description": "Latitude (-90 to 90); leave blank together with longitude if using locations",
            "required": False,
            "default": None,
            "kind": "float",
        },
        {
            "name": "longitude",
            "description": "Longitude (-180 to 180); leave blank together with latitude if using locations",
            "required": False,
            "default": None,
            "kind": "float",
        },
        {
            "name": "locations",
            "description": 'Several locations instead of latitude/longitude, as "name:lat:lon:weight, '
            'name:lat:lon:weight, ..." (weight optional, but given for all locations or none, and only used by '
            'the "mean" aggregation below; population or installed capacity, fixed for the whole sample)',
            "required": False,
            "default": None,
            "kind": "text",
        },
        {
            "name": "agg",
            "description": "Aggregate locations with these (blank = per-location columns only; otherwise "
            '"{variable}_{agg}_d+i" for each pick; weight above needs "mean" among them)',
            "required": False,
            "default": None,
            "kind": "multiselect",
            "choices": ["mean", "median", "min", "max", "sum"],
        },
        {
            "name": "keep_locations",
            "description": "With agg set, also return each location's own columns?",
            "required": False,
            "default": False,
            "kind": "confirm",
        },
        {
            "name": "horizon",
            "description": "Forecast horizon in days (blank = the largest without look-ahead for the model and run "
            "under previous_runs, 5 under single_runs; 1-14 under single_runs)",
            "required": False,
            "default": None,
            "kind": "int",
        },
        {
            "name": "model",
            "description": "Weather model (e.g. jma_seamless, icon_seamless, ecmwf_ifs025)",
            "required": False,
            "default": "jma_seamless",
            "kind": "text",
        },
        {
            "name": "columns",
            "description": "Weather variables to fetch, comma-separated (blank = the default 14)",
            "required": False,
            "default": None,
            "kind": "comma_list",
        },
        {
            "name": "temporal_resolution",
            "description": "One of hourly_3, hourly_6, native (blank = hourly, no extra API parameter)",
            "required": False,
            "default": None,
            "kind": "text",
        },
        {
            "name": "api",
            "description": "previous_runs (long archive, lead-time buckets) or single_runs (one exact run per day)",
            "required": False,
            "default": "previous_runs",
            "kind": "select",
            "choices": ["previous_runs", "single_runs"],
        },
        {
            "name": "run",
            "description": 'UTC hour "HH" of the model run the leak-free cutoff is anchored to (D at run:00 UTC)',
            "required": False,
            "default": "00",
            "kind": "text",
        },
        {
            "name": "timezone",
            "description": "Market time zone (defines local days and the cutoff), e.g. Europe/Warsaw",
            "required": True,
            "default": None,
            "kind": "text",
        },
        {
            "name": "strict",
            "description": "Fail when a weather column has missing values?",
            "required": False,
            "default": False,
            "kind": "confirm",
        },
        {
            "name": "prefix",
            "description": "Column name prefix (optional)",
            "required": False,
            "default": "",
            "kind": "text",
        },
        {
            "name": "proxy_url",
            "description": "HTTP(S) proxy URL (blank = none)",
            "required": False,
            "default": None,
            "kind": "text",
        },
    ],
    "CsvSource": [
        {"name": "file_path", "description": "Path to CSV file", "required": True, "default": None, "kind": "text"},
        {
            "name": "datetime_column",
            "description": "Datetime column name",
            "required": False,
            "default": "datetime",
            "kind": "text",
        },
        {"name": "separator", "description": "CSV separator", "required": False, "default": ",", "kind": "text"},
        {
            "name": "prefix",
            "description": "Column name prefix (optional)",
            "required": False,
            "default": "",
            "kind": "text",
        },
        {
            "name": "columns",
            "description": "Columns to include (comma-separated, empty = all)",
            "required": False,
            "default": None,
            "kind": "column_select",
        },
        {
            "name": "datetime_format",
            "description": "strftime format of the datetime column (empty = auto-detect)",
            "required": False,
            "default": None,
            "kind": "text",
        },
        {
            "name": "timezone",
            "description": "Time zone of naive timestamps (e.g. Europe/Warsaw)",
            "required": False,
            "default": "UTC",
            "kind": "text",
        },
    ],
    "CalendarSource": [
        {
            "name": "country",
            "description": "Country code",
            "required": True,
            "default": None,
            "kind": "select",
            "choices": [
                "PL",
                "DE",
                "FR",
                "ES",
                "IT",
                "NL",
                "BE",
                "AT",
                "CH",
                "CZ",
                "SK",
                "HU",
                "RO",
                "BG",
                "GR",
                "PT",
                "DK",
                "SE",
                "NO",
                "FI",
                "GB",
                "IE",
            ],
        },
        {
            "name": "holidays",
            "description": "Holiday encoding",
            "required": False,
            "default": "binary",
            "kind": "select",
            "choices": ["binary", "onehot", "name", "false"],
        },
        {
            "name": "weekday",
            "description": "Weekday encoding",
            "required": False,
            "default": "number",
            "kind": "select",
            "choices": ["number", "onehot", "name", "false"],
        },
        {
            "name": "hour",
            "description": "Hour encoding",
            "required": False,
            "default": "false",
            "kind": "select",
            "choices": ["false", "number", "onehot"],
        },
        {
            "name": "month",
            "description": "Month encoding",
            "required": False,
            "default": "false",
            "kind": "select",
            "choices": ["false", "number", "onehot", "name"],
        },
        {
            "name": "daylight",
            "description": "Include sunrise/sunset/daylight hours",
            "required": False,
            "default": False,
            "kind": "confirm",
        },
        {
            "name": "lat",
            "description": "Latitude for daylight (empty = the country's capital)",
            "required": False,
            "default": None,
            "kind": "float",
        },
        {
            "name": "lon",
            "description": "Longitude for daylight (empty = the country's capital)",
            "required": False,
            "default": None,
            "kind": "float",
        },
        {
            "name": "region",
            "description": "Subdivision for regional holidays, e.g. BY, ENG (empty = national only)",
            "required": False,
            "default": None,
            "kind": "text",
        },
        {
            "name": "categories",
            "description": "Holiday categories (none selected = public holidays)",
            "required": False,
            "default": None,
            "kind": "multiselect",
            "choices": ["public", "bank", "optional", "catholic", "half_day", "school", "unofficial", "workday"],
        },
        {
            "name": "timezone",
            "description": "Time zone override (empty = the country's)",
            "required": False,
            "default": None,
            "kind": "text",
        },
        {
            "name": "freq",
            "description": "Row spacing (e.g. 1h, 15min)",
            "required": False,
            "default": "1h",
            "kind": "text",
        },
        {
            "name": "bridge_days",
            "description": "Flag bridge days (working day between a holiday and a weekend)",
            "required": False,
            "default": False,
            "kind": "confirm",
        },
        {
            "name": "christmas_period",
            "description": "Flag 24 December to 1 January",
            "required": False,
            "default": False,
            "kind": "confirm",
        },
        {
            "name": "holiday_distance",
            "description": "Days to/since the nearest holiday, capped at N (0 = off)",
            "required": False,
            "default": 0,
            "kind": "int",
        },
        {
            "name": "dst_switch",
            "description": "Flag DST switch days (23 or 25 hours)",
            "required": False,
            "default": False,
            "kind": "confirm",
        },
        {
            "name": "regimes",
            "description": (
                "Market-regime dummies: true (default catalog for the country), false, a JSON list of "
                'catalog names (e.g. ["core_fb", "sdac_15min_mtu"]), or a JSON dict of custom periods'
            ),
            "required": False,
            "default": False,
            "kind": "json",
        },
        {
            "name": "prefix",
            "description": "Column name prefix (optional)",
            "required": False,
            "default": "",
            "kind": "text",
        },
        {
            "name": "school_holidays",
            "description": "Download school holiday flags from the OpenHolidays API",
            "required": False,
            "default": "false",
            "kind": "select",
            "choices": ["false", "national", "all"],
        },
    ],
    "EpftoolboxBenchmarkSource": [
        {
            "name": "dataset",
            "description": "Benchmark dataset of Lago et al. (2021)",
            "required": True,
            "default": None,
            "kind": "select",
            "choices": ["NP", "PJM", "FR", "BE", "DE"],
        },
        {
            "name": "cache_dir",
            "description": "Download directory (blank = ~/.cache/epftoolbox2/epftoolbox-benchmark)",
            "required": False,
            "default": None,
            "kind": "text",
        },
        {
            "name": "prefix",
            "description": "Column name prefix (optional)",
            "required": False,
            "default": "",
            "kind": "text",
        },
    ],
    "PseSource": [
        {
            "name": "datasets",
            "description": "PSE datasets to fetch (no API key)",
            "required": True,
            "default": None,
            "kind": "multiselect",
            "choices": ["kse-load", "his-wlk-cal", "csdac-pln", "rce-pln", "pdgsz", "przeplywy-mocy"],
        },
        {
            "name": "columns",
            "description": "Columns to keep, comma-separated (blank = the datasets' default columns)",
            "required": False,
            "default": None,
            "kind": "comma_list",
        },
        {
            "name": "prefix",
            "description": "Column name prefix (keep one: PSE load_actual shares ENTSO-E's name)",
            "required": False,
            "default": "pse",
            "kind": "text",
        },
        {
            "name": "pdgsz_cutoff",
            "description": "pdgsz: keep the version stamped by this time (HH:MM Warsaw) on the day before",
            "required": False,
            "default": "19:00",
            "kind": "text",
        },
    ],
    "ForecastSource": [
        {
            "name": "path",
            "description": "The upstream run's save_dir (its results and manifests)",
            "required": True,
            "default": None,
            "kind": "text",
        },
        {
            "name": "model",
            "description": "The upstream model's (or ensemble's) name",
            "required": True,
            "default": None,
            "kind": "text",
        },
        {
            "name": "column",
            "description": "Name of the column it emits (lag it like any other: <column>_d+1)",
            "required": True,
            "default": None,
            "kind": "text",
        },
        {
            "name": "horizon",
            "description": "Which upstream horizon to read (days between issue day and target day)",
            "required": False,
            "default": 1,
            "kind": "int",
        },
        {
            "name": "available",
            "description": "A later time its forecasts are known at (HH:MM, upstream issue zone); blank = its issue time",
            "required": False,
            "default": None,
            "kind": "text",
        },
    ],
    "EnergyChartsSource": [
        {
            "name": "country",
            "description": "Country code of the forecast, power and cross-border data (de, at, fr, ...)",
            "required": False,
            "default": "de",
            "kind": "text",
        },
        {
            "name": "type",
            "description": "Data types to fetch (blank = forecast, power, price)",
            "required": False,
            "default": None,
            "kind": "multiselect",
            "choices": ["forecast", "power", "price", "cbet", "cbpf"],
        },
        {
            "name": "bzn",
            "description": "Bidding zone of the price (e.g. DE-LU, CH, DK1); blank = the country's single zone",
            "required": False,
            "default": None,
            "kind": "text",
        },
        {
            "name": "forecast_types",
            "description": "Forecast vintages (blank = day-ahead; current is a latest revision, refused by the check)",
            "required": False,
            "default": None,
            "kind": "multiselect",
            "choices": ["day-ahead", "intraday", "current"],
        },
        {
            "name": "production_types",
            "description": "Forecast quantities (blank = all four)",
            "required": False,
            "default": None,
            "kind": "multiselect",
            "choices": ["solar", "wind_onshore", "wind_offshore", "load"],
        },
        {
            "name": "power",
            "description": "Actual series to keep, comma-separated (e.g. load,solar,fossil_gas); blank = all",
            "required": False,
            "default": None,
            "kind": "comma_list",
        },
        {
            "name": "allow_restricted_prices",
            "description": "Accept a zone whose prices are for private use only (outside the CC BY 4.0 list)?",
            "required": False,
            "default": False,
            "kind": "confirm",
        },
        {
            "name": "prefix",
            "description": "Column name prefix (default ec, so no column takes another source's availability rule)",
            "required": False,
            "default": "ec",
            "kind": "text",
        },
        {
            "name": "cache_dir",
            "description": "Directory for the downloaded answers (blank = ~/.cache/epftoolbox2/energy-charts)",
            "required": False,
            "default": None,
            "kind": "text",
        },
        {
            "name": "use_cache",
            "description": "Keep and reuse the answers on disk?",
            "required": False,
            "default": True,
            "kind": "confirm",
        },
    ],
    "TimezoneTransformer": [
        {
            "name": "target_tz",
            "description": "IANA timezone (e.g. Europe/Warsaw)",
            "required": True,
            "default": None,
            "kind": "text",
        },
        {
            "name": "daily_columns",
            "description": "Columns with one value per day, moved to local midnight (none = ENTSO-E daily min/max)",
            "required": False,
            "default": None,
            "kind": "column_select",
        },
        {
            "name": "daily_anchor",
            "description": "Zone whose calendar date a daily value belongs to (empty = target_tz, the market zone)",
            "required": False,
            "default": None,
            "kind": "text",
        },
    ],
    "ResampleTransformer": [
        {
            "name": "freq",
            "description": "Target frequency (e.g. 1h, 15min)",
            "required": False,
            "default": "1h",
            "kind": "text",
        },
        {
            "name": "agg",
            "description": "Aggregation when downsampling",
            "required": False,
            "default": "mean",
            "kind": "select",
            "choices": ["mean", "sum", "first", "last", "min", "max", "median"],
        },
        {
            "name": "fill",
            "description": "Upsampled slots: repeat the value within its own period, or leave empty",
            "required": False,
            "default": "ffill",
            "kind": "select",
            "choices": ["ffill", "none"],
        },
    ],
    "LagTransformer": [
        {
            "name": "columns",
            "description": "Columns to lag (blank = all)",
            "required": False,
            "default": None,
            "kind": "column_select",
        },
        {
            "name": "lags",
            "description": "Lag values, comma-separated integers (e.g. 1,2,24 or -7,-1)",
            "required": False,
            "default": "1",
            "kind": "lag_list",
        },
        {"name": "freq", "description": "Frequency (e.g. 1h, day)", "required": False, "default": "1h", "kind": "text"},
    ],
    "DailyProfileTransformer": [
        {
            "name": "columns",
            "description": "Columns to spread",
            "required": True,
            "default": None,
            "kind": "column_select",
        },
        {
            "name": "freq",
            "description": "Slot length of the data (1h = 24 columns each, 15min = 96)",
            "required": False,
            "default": "1h",
            "kind": "text",
        },
    ],
    "DailyStatsTransformer": [
        {
            "name": "columns",
            "description": "Columns to summarise",
            "required": True,
            "default": None,
            "kind": "column_select",
        },
        {
            "name": "stats",
            "description": "Statistics per local day",
            "required": False,
            "default": None,
            "kind": "multiselect",
            "choices": ["min", "max", "mean", "last"],
        },
        {"name": "freq", "description": "Slot length of the data", "required": False, "default": "1h", "kind": "text"},
    ],
    # `func` (a callable df -> df) can only be set in code, not through the wizard or YAML.
    "LambdaTransformer": [
        {
            "name": "columns",
            "description": 'New/existing column(s), as JSON, e.g. {"residual_load": "load_forecast - wind_forecast"}',
            "required": True,
            "default": None,
            "kind": "json",
        },
        {
            "name": "func",
            "description": "Full-control callable df -> df (code only; leave blank here)",
            "required": False,
            "default": None,
            "kind": "text",
        },
    ],
    "DSTNormalizeTransformer": [
        {
            "name": "freq",
            "description": "Slot length of the data (1h = 24 slots a day, 15min = 96)",
            "required": False,
            "default": "1h",
            "kind": "text",
        },
        {
            "name": "tz",
            "description": "Convert to this IANA timezone first (blank = index is already local)",
            "required": False,
            "default": None,
            "kind": "text",
        },
    ],
    "ImputeTransformer": [
        {
            "name": "columns",
            "description": "Columns to impute",
            "required": True,
            "default": None,
            "kind": "column_select",
        },
        {
            "name": "method",
            "description": "Fill method (leak-safe only)",
            "required": False,
            "default": "ffill",
            "kind": "select",
            "choices": ["ffill", "previous_day", "previous_week"],
        },
        {
            "name": "limit",
            "description": "Max gap to fill: integer rows, a duration (e.g. 6h, 2D), or blank for no limit",
            "required": False,
            "default": None,
            "kind": "text",
        },
        {
            "name": "flag",
            "description": "Add a {column}_imputed flag column?",
            "required": False,
            "default": True,
            "kind": "confirm",
        },
        {
            "name": "suffix",
            "description": "Write filled values to {column}{suffix} instead of overwriting (optional)",
            "required": False,
            "default": "",
            "kind": "text",
        },
    ],
    "EdaValidator": [
        {
            "name": "columns",
            "description": "Columns to analyse (comma-separated, blank = auto)",
            "required": False,
            "default": "",
            "kind": "column_select",
        },
        {
            "name": "verbose",
            "description": "Print the statistics table?",
            "required": False,
            "default": True,
            "kind": "confirm",
        },
    ],
    "DSTStructureValidator": [
        {
            "name": "tz",
            "description": "Time zone of the local days (e.g. Europe/Warsaw; blank = the index's own)",
            "required": False,
            "default": None,
            "kind": "text",
        },
        {
            "name": "freq",
            "description": "Period length (e.g. 15min, 1h; blank = infer from the data)",
            "required": False,
            "default": None,
            "kind": "text",
        },
        {
            "name": "allow_partial_edges",
            "description": "Only warn about a partial first/last day?",
            "required": False,
            "default": True,
            "kind": "confirm",
        },
    ],
    "CoverageValidator": [
        {
            "name": "columns",
            "description": "Columns to check (blank = all)",
            "required": False,
            "default": "",
            "kind": "column_select",
        },
        {
            "name": "start",
            "description": "First expected timestamp (e.g. 2024-01-01; blank = each column's first value)",
            "required": False,
            "default": None,
            "kind": "text",
        },
        {
            "name": "end",
            "description": "End of the range, exclusive (blank = each column's last value)",
            "required": False,
            "default": None,
            "kind": "text",
        },
        {
            "name": "freq",
            "description": "Grid step (e.g. 15min, 1h; blank = infer from the data)",
            "required": False,
            "default": None,
            "kind": "text",
        },
        {
            "name": "min_coverage",
            "description": "Smallest share of timestamps with a value (0-1)",
            "required": False,
            "default": 1.0,
            "kind": "float",
        },
        {
            "name": "max_gap",
            "description": "Longest missing run allowed (e.g. 6h; blank = no limit)",
            "required": False,
            "default": None,
            "kind": "text",
        },
    ],
    # `ranges` (a {pattern: [min, max]} mapping) is set in YAML; the wizard uses the defaults.
    "PlausibilityValidator": [
        {
            "name": "ranges",
            "description": (
                'Column pattern -> [min, max] overrides, as JSON, e.g. {"price": [-500, 4000]} '
                "(blank = price* within the SDAC limits, load* >= 1 MW)"
            ),
            "required": False,
            "default": None,
            "kind": "json",
        },
        {
            "name": "max_constant_run",
            "description": "Longest run of identical values before a warning (rows)",
            "required": False,
            "default": 24,
            "kind": "int",
        },
        {
            "name": "constant_columns",
            "description": "Columns checked for constant runs (blank = price* and load_actual*)",
            "required": False,
            "default": "",
            "kind": "column_select",
        },
    ],
    "OLSModel": [
        {"name": "name", "description": "Model display name", "required": False, "default": "OLS", "kind": "text"},
        {
            "name": "predictors",
            "description": "Predictor column names",
            "required": True,
            "default": None,
            "kind": "column_select",
        },
        {
            "name": "training_window",
            "description": "Training window in days, 'expanding', or a list averaged inside the model (e.g. 56,84,728,expanding)",
            "required": False,
            "default": "365",
            "kind": "window",
        },
        {
            "name": "scaler",
            "description": "Scaling: standard, or a variance-stabilising transformation (asinh, mlog, poly, npit)",
            "required": False,
            "default": "standard",
            "kind": "select",
            "choices": ["standard", "asinh", "mlog", "poly", "npit"],
        },
        {
            "name": "newest_target",
            "description": "Newest training target: d-1, or every target published by the issue time (needs an issue time)",
            "required": False,
            "default": "d-1",
            "kind": "select",
            "choices": ["d-1", "published"],
        },
        {
            "name": "fit_intercept",
            "description": "Fit an intercept (off by default: the target is standardised and one-hot dummies carry the level)",
            "required": False,
            "default": False,
            "kind": "confirm",
        },
        {
            "name": "refit",
            "description": (
                "Refit on a fixed calendar: empty fits every forecast, 7D once a week (Mondays), 1MS once a month, auto "
                "a window of up to 84 days daily and a longer one weekly"
            ),
            "required": False,
            "default": None,
            "kind": "text",
        },
        {
            "name": "binary_mask",
            "description": (
                "Which rows decide the binary 0/1 columns left unscaled: empty for the default (the test's first day, a "
                "refit's own rows), fit for each fit's own rows (no forecast then depends on the test start)"
            ),
            "required": False,
            "default": None,
            "kind": "text",
        },
        *_row_target_fields(),
    ],
    "LassoCVModel": [
        {"name": "name", "description": "Model display name", "required": False, "default": "LassoCV", "kind": "text"},
        {
            "name": "predictors",
            "description": "Predictor column names",
            "required": True,
            "default": None,
            "kind": "column_select",
        },
        {
            "name": "training_window",
            "description": "Training window in days, 'expanding', or a list averaged inside the model (e.g. 56,84,728,expanding)",
            "required": False,
            "default": "365",
            "kind": "window",
        },
        {"name": "cv", "description": "Cross-validation folds", "required": False, "default": "5", "kind": "int"},
        {
            "name": "max_iter",
            "description": "Maximum iterations per fit",
            "required": False,
            "default": "10000",
            "kind": "int",
        },
        {
            "name": "scaler",
            "description": "Scaling: standard, or a variance-stabilising transformation (asinh, mlog, poly, npit)",
            "required": False,
            "default": "standard",
            "kind": "select",
            "choices": ["standard", "asinh", "mlog", "poly", "npit"],
        },
        {
            "name": "newest_target",
            "description": "Newest training target: d-1, or every target published by the issue time (needs an issue time)",
            "required": False,
            "default": "d-1",
            "kind": "select",
            "choices": ["d-1", "published"],
        },
        {
            "name": "fit_intercept",
            "description": "Fit an intercept",
            "required": False,
            "default": True,
            "kind": "confirm",
        },
        {
            "name": "refit",
            "description": (
                "Refit on a fixed calendar: empty fits every forecast, 7D once a week (Mondays), 1MS once a month, auto "
                "a window of up to 84 days daily and a longer one weekly"
            ),
            "required": False,
            "default": None,
            "kind": "text",
        },
        {
            "name": "binary_mask",
            "description": (
                "Which rows decide the binary 0/1 columns left unscaled: empty for the default (the test's first day, a "
                "refit's own rows), fit for each fit's own rows (no forecast then depends on the test start)"
            ),
            "required": False,
            "default": None,
            "kind": "text",
        },
        *_row_target_fields(),
    ],
    "RidgeModel": [
        {"name": "name", "description": "Model display name", "required": False, "default": "Ridge", "kind": "text"},
        {
            "name": "predictors",
            "description": "Predictor column names",
            "required": True,
            "default": None,
            "kind": "column_select",
        },
        {
            "name": "training_window",
            "description": "Training window in days, 'expanding', or a list averaged inside the model (e.g. 56,84,728,expanding)",
            "required": False,
            "default": "365",
            "kind": "window",
        },
        {
            "name": "alphas",
            "description": "Penalties to choose from by leave-one-out, as a JSON list such as [0.1, 1, 10]; empty: 36 values from 1e-3 to 1e4",
            "required": False,
            "default": None,
            "kind": "json",
        },
        {
            "name": "scaler",
            "description": "Scaling: standard, or a variance-stabilising transformation (asinh, mlog, poly, npit)",
            "required": False,
            "default": "standard",
            "kind": "select",
            "choices": ["standard", "asinh", "mlog", "poly", "npit"],
        },
        {
            "name": "newest_target",
            "description": "Newest training target: d-1, or every target published by the issue time (needs an issue time)",
            "required": False,
            "default": "d-1",
            "kind": "select",
            "choices": ["d-1", "published"],
        },
        {
            "name": "fit_intercept",
            "description": "Fit an intercept",
            "required": False,
            "default": True,
            "kind": "confirm",
        },
        {
            "name": "refit",
            "description": (
                "Refit on a fixed calendar: empty fits every forecast, 7D once a week (Mondays), 1MS once a month, auto "
                "a window of up to 84 days daily and a longer one weekly"
            ),
            "required": False,
            "default": None,
            "kind": "text",
        },
        {
            "name": "binary_mask",
            "description": (
                "Which rows decide the binary 0/1 columns left unscaled: empty for the default (the test's first day, a "
                "refit's own rows), fit for each fit's own rows (no forecast then depends on the test start)"
            ),
            "required": False,
            "default": None,
            "kind": "text",
        },
        *_row_target_fields(),
    ],
    "LEARModel": [
        {"name": "name", "description": "Model display name", "required": False, "default": "LEAR", "kind": "text"},
        {
            "name": "predictors",
            "description": "Predictor column names (presets.lear_predictors gives LEAR's 247)",
            "required": True,
            "default": None,
            "kind": "column_select",
        },
        {
            "name": "training_window",
            "description": "Training window in days, 'expanding', or a list averaged inside the model (e.g. 56,84,728,expanding)",
            "required": False,
            "default": "56,84,728,expanding",
            "kind": "window",
        },
        {
            "name": "criterion",
            "description": "How the penalty is chosen: bic or aic (Lago et al.'s) on the LARS path, or cv (5 blocked folds)",
            "required": False,
            "default": "bic",
            "kind": "select",
            "choices": ["bic", "aic", "cv"],
        },
        {
            "name": "solver",
            "description": "The Lasso at that penalty: lars (exact; cv takes it) or cd (Lago's coordinate descent, 2500 sweeps)",
            "required": False,
            "default": "lars",
            "kind": "select",
            "choices": ["lars", "cd"],
        },
        {
            "name": "scaler",
            "description": "Scaling: asinh (median/MAD asinh, standardised), standard, mlog, poly or npit",
            "required": False,
            "default": "asinh",
            "kind": "select",
            "choices": ["asinh", "standard", "mlog", "poly", "npit"],
        },
        {
            "name": "newest_target",
            "description": "Newest training target: d-1, or every target published by the issue time (needs an issue time)",
            "required": False,
            "default": "d-1",
            "kind": "select",
            "choices": ["d-1", "published"],
        },
        {
            "name": "refit",
            "description": (
                "Refit on a fixed calendar: empty fits every forecast, 7D once a week (Mondays), 1MS once a month, auto "
                "a window of up to 84 days daily and a longer one weekly"
            ),
            "required": False,
            "default": None,
            "kind": "text",
        },
        {
            "name": "binary_mask",
            "description": (
                "Which rows decide the binary 0/1 columns left unscaled: empty for the default (the test's first day, a "
                "refit's own rows), fit for each fit's own rows (no forecast then depends on the test start)"
            ),
            "required": False,
            "default": None,
            "kind": "text",
        },
        *_row_target_fields(),
    ],
    "NaiveModel": [
        {"name": "name", "description": "Model display name", "required": False, "default": "Naive", "kind": "text"},
        {
            "name": "predictors",
            "description": "The one column whose value on the issue row is the forecast, e.g. a TSO's published forecast",
            "required": True,
            "default": None,
            "kind": "column_select",
        },
        {
            "name": "known_only",
            "description": "With an issue time, no forecast where the column isn't published by then (not_published)",
            "required": False,
            "default": True,
            "kind": "confirm",
        },
        *_row_target_fields(bounds_only=True),
    ],
    "MLPDailyModel": [
        {
            "name": "name",
            "description": "Model display name",
            "required": False,
            "default": "mlp_daily",
            "kind": "text",
        },
        {
            "name": "predictors",
            "description": "Day-level predictor columns (one value per local day: daily profiles such as price_h05_d-1)",
            "required": True,
            "default": None,
            "kind": "column_select",
        },
        {
            "name": "hidden",
            "description": "Hidden layer widths as a JSON list",
            "required": False,
            "default": (256, 128),
            "kind": "json",
        },
        {
            "name": "dropout",
            "description": "Dropout rate of each hidden layer",
            "required": False,
            "default": 0.1,
            "kind": "float",
        },
        _ACTIVATION_FIELD,
        *_torch_fields(4),
    ],
    "MLPHourlyModel": [
        {
            "name": "name",
            "description": "Model display name",
            "required": False,
            "default": "mlp_hourly",
            "kind": "text",
        },
        {
            "name": "predictors",
            "description": "Predictor column names (one network per period of the day)",
            "required": True,
            "default": None,
            "kind": "column_select",
        },
        {
            "name": "hidden",
            "description": "Hidden layer widths as a JSON list",
            "required": False,
            "default": (64,),
            "kind": "json",
        },
        {
            "name": "dropout",
            "description": "Dropout rate of each hidden layer",
            "required": False,
            "default": 0.0,
            "kind": "float",
        },
        _ACTIVATION_FIELD,
        *_torch_fields(3),
        *_row_target_fields(),
    ],
    "TorchModel": [
        {"name": "name", "description": "Model display name", "required": False, "default": "torch", "kind": "text"},
        {
            "name": "module",
            "description": "Your torch.nn.Module class as package.module:Class (importable where the pipeline runs)",
            "required": True,
            "default": None,
            "kind": "text",
        },
        {
            "name": "predictors",
            "description": "Predictor column names (output 'day': day-level columns only)",
            "required": True,
            "default": None,
            "kind": "column_select",
        },
        {
            "name": "module_params",
            "description": 'Keyword arguments of the module\'s __init__, as JSON, e.g. {"width": 256}',
            "required": False,
            "default": None,
            "kind": "json",
        },
        {
            "name": "output",
            "description": "day: one network per horizon with an output per period; period: one per period",
            "required": False,
            "default": "day",
            "kind": "select",
            "choices": ["day", "period"],
        },
        *_torch_fields(4),
        *_row_target_fields(),  # output "period" only: a day model refuses them before any fit
    ],
    "HistGBModel": [
        {"name": "name", "description": "Model display name", "required": False, "default": "hgb", "kind": "text"},
        {
            "name": "predictors",
            "description": "Predictor column names, each read on the row's own period (presets.tree_predictors gives 183)",
            "required": True,
            "default": None,
            "kind": "column_select",
        },
        {
            "name": "training_window",
            "description": "Training window in days, or 'expanding' (every row from the frame's first date)",
            "required": False,
            "default": "expanding",
            "kind": "window",
        },
        {
            "name": "refit",
            "description": "Refit on a fixed calendar: 7D once a week (Mondays), 1D every day, 1MS once a month",
            "required": False,
            "default": "7D",
            "kind": "text",
        },
        {
            "name": "periods",
            "description": "pooled: one fit over every period's rows (the period a feature where listed); each: one per period",
            "required": False,
            "default": "pooled",
            "kind": "select",
            "choices": ["pooled", "each"],
        },
        {
            "name": "loss",
            "description": "l1 (absolute error) or l2 (squared error)",
            "required": False,
            "default": "l1",
            "kind": "select",
            "choices": ["l1", "l2"],
        },
        {"name": "max_iter", "description": "Boosting iterations", "required": False, "default": 500, "kind": "int"},
        {"name": "learning_rate", "description": "Learning rate", "required": False, "default": 0.05, "kind": "float"},
        {"name": "max_leaf_nodes", "description": "Leaves per tree", "required": False, "default": 31, "kind": "int"},
        {
            "name": "min_samples_leaf",
            "description": "Rows per leaf at least",
            "required": False,
            "default": 20,
            "kind": "int",
        },
        {
            "name": "l2_regularization",
            "description": "L2 penalty of the leaf values",
            "required": False,
            "default": 1.0,
            "kind": "float",
        },
        {
            "name": "max_features",
            "description": "Share of the predictors each split chooses from",
            "required": False,
            "default": 0.5,
            "kind": "float",
        },
        {
            "name": "random_state",
            "description": "Seed of the feature subsets",
            "required": False,
            "default": 0,
            "kind": "int",
        },
        {
            "name": "newest_target",
            "description": "Newest training target: d-1, or every target published by the issue time (needs an issue time)",
            "required": False,
            "default": "d-1",
            "kind": "select",
            "choices": ["d-1", "published"],
        },
        {
            "name": "binary_mask",
            "description": "Which rows decide the binary 0/1 columns: empty for each refit unit's own rows",
            "required": False,
            "default": None,
            "kind": "text",
        },
        *_row_target_fields(),
    ],
    "LightGBMModel": [
        {"name": "name", "description": "Model display name", "required": False, "default": "lgbm", "kind": "text"},
        {
            "name": "predictors",
            "description": "Predictor column names, each read on the row's own period (presets.tree_predictors gives 183)",
            "required": True,
            "default": None,
            "kind": "column_select",
        },
        {
            "name": "training_window",
            "description": "Training window in days, or 'expanding' (every row from the frame's first date)",
            "required": False,
            "default": "expanding",
            "kind": "window",
        },
        {
            "name": "refit",
            "description": "Refit on a fixed calendar: 7D once a week (Mondays), 1D every day, 1MS once a month",
            "required": False,
            "default": "7D",
            "kind": "text",
        },
        {
            "name": "periods",
            "description": "pooled: one fit over every period's rows (the period a feature where listed); each: one per period",
            "required": False,
            "default": "pooled",
            "kind": "select",
            "choices": ["pooled", "each"],
        },
        {
            "name": "objective",
            "description": "l1 (absolute error) or l2 (squared error)",
            "required": False,
            "default": "l1",
            "kind": "select",
            "choices": ["l1", "l2"],
        },
        {"name": "n_estimators", "description": "Boosting rounds", "required": False, "default": 500, "kind": "int"},
        {"name": "learning_rate", "description": "Learning rate", "required": False, "default": 0.05, "kind": "float"},
        {"name": "num_leaves", "description": "Leaves per tree", "required": False, "default": 31, "kind": "int"},
        {
            "name": "min_child_samples",
            "description": "Rows per leaf at least",
            "required": False,
            "default": 20,
            "kind": "int",
        },
        {
            "name": "subsample",
            "description": "Share of the rows each tree samples",
            "required": False,
            "default": 0.8,
            "kind": "float",
        },
        {
            "name": "subsample_freq",
            "description": "Sample the rows every k trees (0: never)",
            "required": False,
            "default": 1,
            "kind": "int",
        },
        {
            "name": "colsample_bytree",
            "description": "Share of the predictors each tree samples",
            "required": False,
            "default": 0.5,
            "kind": "float",
        },
        {
            "name": "reg_lambda",
            "description": "L2 penalty of the leaf values",
            "required": False,
            "default": 1.0,
            "kind": "float",
        },
        {
            "name": "random_state",
            "description": "Seed of the row and column samples, the same in every fit",
            "required": False,
            "default": 0,
            "kind": "int",
        },
        {
            "name": "threads",
            "description": "OpenMP threads of one fit: 1 where workers fork (Linux, macOS); recorded and compared on resume",
            "required": False,
            "default": 1,
            "kind": "int",
        },
        {
            "name": "newest_target",
            "description": "Newest training target: d-1, or every target published by the issue time (needs an issue time)",
            "required": False,
            "default": "d-1",
            "kind": "select",
            "choices": ["d-1", "published"],
        },
        {
            "name": "binary_mask",
            "description": "Which rows decide the binary 0/1 columns: empty for each refit unit's own rows",
            "required": False,
            "default": None,
            "kind": "text",
        },
        *_row_target_fields(),
    ],
    "XGBoostModel": [
        {"name": "name", "description": "Model display name", "required": False, "default": "xgb", "kind": "text"},
        {
            "name": "predictors",
            "description": "Predictor column names, each read on the row's own period (presets.tree_predictors gives 183)",
            "required": True,
            "default": None,
            "kind": "column_select",
        },
        {
            "name": "training_window",
            "description": "Training window in days, or 'expanding' (every row from the frame's first date)",
            "required": False,
            "default": "expanding",
            "kind": "window",
        },
        {
            "name": "refit",
            "description": "Refit on a fixed calendar: 7D once a week (Mondays), 1D every day, 1MS once a month",
            "required": False,
            "default": "7D",
            "kind": "text",
        },
        {
            "name": "periods",
            "description": "pooled: one fit over every period's rows (the period a feature where listed); each: one per period",
            "required": False,
            "default": "pooled",
            "kind": "select",
            "choices": ["pooled", "each"],
        },
        {
            "name": "objective",
            "description": "l1 (absolute error) or l2 (squared error)",
            "required": False,
            "default": "l1",
            "kind": "select",
            "choices": ["l1", "l2"],
        },
        {"name": "n_estimators", "description": "Boosting rounds", "required": False, "default": 500, "kind": "int"},
        {"name": "learning_rate", "description": "Learning rate", "required": False, "default": 0.05, "kind": "float"},
        {"name": "max_depth", "description": "Depth of a tree at most", "required": False, "default": 6, "kind": "int"},
        {
            "name": "subsample",
            "description": "Share of the rows each tree samples",
            "required": False,
            "default": 0.8,
            "kind": "float",
        },
        {
            "name": "colsample_bytree",
            "description": "Share of the predictors each tree samples",
            "required": False,
            "default": 0.5,
            "kind": "float",
        },
        {
            "name": "min_child_weight",
            "description": "Hessian weight per leaf at least",
            "required": False,
            "default": 5.0,
            "kind": "float",
        },
        {
            "name": "reg_lambda",
            "description": "L2 penalty of the leaf values",
            "required": False,
            "default": 1.0,
            "kind": "float",
        },
        {
            "name": "random_state",
            "description": "Seed of the row and column samples, the same in every fit",
            "required": False,
            "default": 0,
            "kind": "int",
        },
        {
            "name": "threads",
            "description": "OpenMP threads of one fit: 1 where workers fork (Linux, macOS); recorded and compared on resume",
            "required": False,
            "default": 1,
            "kind": "int",
        },
        {
            "name": "device",
            "description": "cpu, or a CUDA GPU (cuda, cuda:1): forecasts differ between the two",
            "required": False,
            "default": "cpu",
            "kind": "text",
        },
        {
            "name": "newest_target",
            "description": "Newest training target: d-1, or every target published by the issue time (needs an issue time)",
            "required": False,
            "default": "d-1",
            "kind": "select",
            "choices": ["d-1", "published"],
        },
        {
            "name": "binary_mask",
            "description": "Which rows decide the binary 0/1 columns: empty for each refit unit's own rows",
            "required": False,
            "default": None,
            "kind": "text",
        },
        *_row_target_fields(),
    ],
    "CatBoostModel": [
        {"name": "name", "description": "Model display name", "required": False, "default": "catboost", "kind": "text"},
        {
            "name": "predictors",
            "description": "Predictor column names, each read on the row's own period (presets.tree_predictors gives 183)",
            "required": True,
            "default": None,
            "kind": "column_select",
        },
        {
            "name": "training_window",
            "description": "Training window in days, or 'expanding' (every row from the frame's first date)",
            "required": False,
            "default": "expanding",
            "kind": "window",
        },
        {
            "name": "refit",
            "description": "Refit on a fixed calendar: 7D once a week (Mondays), 1D every day, 1MS once a month",
            "required": False,
            "default": "7D",
            "kind": "text",
        },
        {
            "name": "periods",
            "description": "pooled: one fit over every period's rows (the period a feature where listed); each: one per period",
            "required": False,
            "default": "pooled",
            "kind": "select",
            "choices": ["pooled", "each"],
        },
        {
            "name": "objective",
            "description": "l1 (absolute error) or l2 (squared error)",
            "required": False,
            "default": "l1",
            "kind": "select",
            "choices": ["l1", "l2"],
        },
        {"name": "iterations", "description": "Boosting rounds", "required": False, "default": 1000, "kind": "int"},
        {"name": "learning_rate", "description": "Learning rate", "required": False, "default": 0.05, "kind": "float"},
        {"name": "depth", "description": "Depth of a (symmetric) tree", "required": False, "default": 8, "kind": "int"},
        {
            "name": "l2_leaf_reg",
            "description": "L2 penalty of the leaf values",
            "required": False,
            "default": 5.0,
            "kind": "float",
        },
        {
            "name": "random_state",
            "description": "Seed of the row and column samples, the same in every fit",
            "required": False,
            "default": 0,
            "kind": "int",
        },
        {
            "name": "threads",
            "description": "OpenMP threads of one fit: 1 where workers fork (Linux, macOS); recorded and compared on resume",
            "required": False,
            "default": 1,
            "kind": "int",
        },
        {
            "name": "newest_target",
            "description": "Newest training target: d-1, or every target published by the issue time (needs an issue time)",
            "required": False,
            "default": "d-1",
            "kind": "select",
            "choices": ["d-1", "published"],
        },
        {
            "name": "binary_mask",
            "description": "Which rows decide the binary 0/1 columns: empty for each refit unit's own rows",
            "required": False,
            "default": None,
            "kind": "text",
        },
        *_row_target_fields(),
    ],
    "MAEEvaluator": [],
    "RMSEEvaluator": [],
    "rMAEEvaluator": [
        {
            "name": "base_model",
            "description": "Name of the benchmark model (rMAE = MAE / benchmark MAE)",
            "required": True,
            "default": None,
            "kind": "text",
        },
    ],
    "BiasEvaluator": [],
    "ShapeMAEEvaluator": [
        {
            "name": "period",
            "description": "Period the within-hour shape is measured in (15-minute results)",
            "required": False,
            "default": "1h",
            "kind": "select",
            "choices": ["1h"],
        },
    ],
    "NMAEEvaluator": [],
    "ScaledMAEEvaluator": [
        {
            "name": "column",
            "description": "Source-data column to divide each error by (e.g. installed capacity)",
            "required": True,
            "default": None,
            "kind": "text",
        },
    ],
    "DMEvaluator": [
        {
            "name": "benchmark",
            "description": "Name of the model the others are compared with",
            "required": True,
            "default": None,
            "kind": "text",
        },
        {
            "name": "variance",
            "description": "Variance of the test: hac (Newey-West) or plain (epftoolbox)",
            "required": False,
            "default": "hac",
            "kind": "select",
            "choices": ["hac", "plain"],
        },
        {
            "name": "lag",
            "description": "Newey-West lag (empty: chosen from the number of days)",
            "required": False,
            "default": None,
            "kind": "int",
        },
        {
            "name": "correction",
            "description": "Correction for testing several models",
            "required": False,
            "default": "holm",
            "kind": "select",
            "choices": ["holm", "none"],
        },
        {
            "name": "alpha",
            "description": "Size behind the verdict",
            "required": False,
            "default": "0.05",
            "kind": "float",
        },
        {
            "name": "interval",
            "description": "Coverage of the bootstrap interval of the MAE ratio",
            "required": False,
            "default": "0.95",
            "kind": "float",
        },
        {
            "name": "block",
            "description": "Bootstrap block length in days",
            "required": False,
            "default": "7",
            "kind": "int",
        },
        {"name": "reps", "description": "Bootstrap repetitions", "required": False, "default": "4000", "kind": "int"},
        {"name": "seed", "description": "Bootstrap seed", "required": False, "default": "0", "kind": "int"},
    ],
    "MCSEvaluator": [
        {
            "name": "alpha",
            "description": "Size of the set (0.10: the 90 % set)",
            "required": False,
            "default": "0.1",
            "kind": "float",
        },
        {
            "name": "statistic",
            "description": "Test statistic: R (recommended) or max",
            "required": False,
            "default": "R",
            "kind": "select",
            "choices": ["R", "max"],
        },
        {
            "name": "block",
            "description": "Mean bootstrap block length in days",
            "required": False,
            "default": "7",
            "kind": "int",
        },
        {"name": "reps", "description": "Bootstrap repetitions", "required": False, "default": "1000", "kind": "int"},
        {"name": "seed", "description": "Bootstrap seed", "required": False, "default": "0", "kind": "int"},
    ],
    "EconomicEvaluator": [
        {
            "name": "strategy",
            "description": "Battery: power MW, energy MWh, round-trip efficiency",
            "required": False,
            "default": None,
            "kind": "battery",
        },
    ],
    "CsvExporter": [
        {"name": "path", "description": "Output CSV file path", "required": True, "default": None, "kind": "text"},
        {
            "name": "extra_columns",
            "description": "Source columns to add to each row (comma-separated, optional)",
            "required": False,
            "default": None,
            "kind": "comma_list",
        },
        {
            "name": "summary_path",
            "description": "Also write the metric columns to this CSV file (optional)",
            "required": False,
            "default": None,
            "kind": "text",
        },
    ],
    "ExcelExporter": [
        {"name": "path", "description": "Output Excel file path", "required": True, "default": None, "kind": "text"},
        {
            "name": "sheets",
            "description": "Sheets to include",
            "required": False,
            "default": None,
            "kind": "multiselect",
            "choices": [
                "summary",
                "hour",
                "horizon",
                "hour_horizon",
                "year",
                "year_horizon",
                "run_weekday_horizon",
                "target_weekday_horizon",
                "details",
            ],
        },
    ],
    "TerminalExporter": [
        {
            "name": "show",
            "description": "Sections to display",
            "required": False,
            "default": None,
            "kind": "multiselect",
            "choices": ["summary", "hour", "horizon", "hour_horizon", "year", "year_horizon"],
        },
    ],
}


# ---------------------------------------------------------------------------
# Low-level prompt helpers
# ---------------------------------------------------------------------------


def _ask_field(field: dict, available_columns: Optional[list[str]] = None) -> Any:
    """Prompt the user for a single parameter field. Returns the value or sentinel SKIP."""
    desc = field["description"]
    kind = field["kind"]
    default = field["default"]
    required = field["required"]
    choices = field.get("choices", [])

    label = f"  {desc}" + ("" if required else f" (default: {default!r})")

    if kind == "confirm":
        val = questionary.confirm(label, default=bool(default)).ask()
        if val is None:
            _abort()
        return val

    if kind == "select":
        val = questionary.select(label, choices=choices, default=default).ask()
        if val is None:
            _abort()
        # "false" string → Python False
        return False if val == "false" else val

    if kind == "multiselect":
        val = questionary.checkbox(label, choices=choices).ask()
        if val is None:
            _abort()
        return val if val else None

    if kind == "json":
        raw = questionary.text(label, default=json.dumps(default) if default is not None else "").ask()
        if raw is None:
            _abort()
        raw = raw.strip()
        if not raw:
            if required:
                console.print("[red]This field is required.[/red]")
                return _ask_field(field, available_columns)
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            console.print('[red]Please enter valid JSON, e.g. {"price": [-500, 4000]}[/red]')
            return _ask_field(field, available_columns)

    if kind == "lag_list":
        while True:
            raw = questionary.text(label, default=str(default) if default is not None else "1").ask()
            if raw is None:
                _abort()
            raw = raw.strip()
            if not raw:
                raw = str(default)
            try:
                result = [int(x.strip()) for x in raw.split(",") if x.strip()]
                if not result:
                    raise ValueError
                return result
            except ValueError:
                console.print("[red]Please enter comma-separated integers, e.g. 1,2,7 or -7,-1,0[/red]")

    if kind == "window":
        from .models._calendar import resolve_training_window

        raw = questionary.text(label, default=str(default) if default is not None else "").ask()
        if raw is None:
            _abort()
        raw = raw.strip() or str(default)
        try:
            parts = [part.strip() for part in raw.split(",") if part.strip()]
            return resolve_training_window([part if part == "expanding" else int(part) for part in parts])[0]
        except ValueError as exc:
            console.print(
                f"[red]Please enter days, 'expanding', or a list of them, e.g. 365 or 56,84,728,expanding ({exc})[/red]"
            )
            return _ask_field(field, available_columns)

    if kind == "bounds":
        from .models._row_target import check_bounds

        raw = questionary.text(label, default="").ask()
        if raw is None:
            _abort()
        raw = raw.strip()
        if not raw:
            return default
        try:
            low, high = (part.strip() for part in raw.split(","))
            return check_bounds((float(low) if low else None, float(high) if high else None))
        except ValueError as exc:
            console.print(
                f"[red]Please enter low,high with either side empty for none, e.g. 0,1.05 or ,100 ({exc})[/red]"
            )
            return _ask_field(field, available_columns)

    if kind == "battery":
        from .evaluators import BatteryArbitrage

        raw = questionary.text(label, default="1, 2, 0.9").ask()
        if raw is None:
            _abort()
        try:
            power, energy, efficiency = (float(x) for x in raw.split(","))
            return BatteryArbitrage(power_mw=power, energy_mwh=energy, efficiency=efficiency)
        except ValueError as exc:
            console.print(f"[red]Please enter three numbers, e.g. 1, 2, 0.9 ({exc})[/red]")
            return _ask_field(field, available_columns)

    if kind == "column_select":
        if available_columns:
            # Required fields (predictors/columns) must pick at least one
            val = questionary.checkbox(label, choices=available_columns).ask()
            if val is None:
                _abort()
            if not val and required:
                console.print("[red]At least one column is required.[/red]")
                return _ask_field(field, available_columns)
            return val if val else None
        else:
            # Fall back to comma-separated text
            val = questionary.text(label, default=default or "").ask()
            if val is None:
                _abort()
            val = val.strip()
            if not val:
                if required:
                    console.print("[red]At least one column is required.[/red]")
                    return _ask_field(field, available_columns)
                return None
            return [c.strip() for c in val.split(",") if c.strip()]

    # text / int / float
    raw = questionary.text(label, default=str(default) if default is not None else "").ask()
    if raw is None:
        _abort()
    raw = raw.strip()

    if not raw:
        if required:
            console.print("[red]This field is required.[/red]")
            return _ask_field(field, available_columns)
        return default

    if kind == "int":
        try:
            return int(raw)
        except ValueError:
            console.print("[red]Please enter a valid integer.[/red]")
            return _ask_field(field, available_columns)

    if kind == "float":
        try:
            return float(raw)
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")
            return _ask_field(field, available_columns)

    return raw


def _abort() -> None:
    console.print("\n[yellow]Cancelled.[/yellow]")
    sys.exit(0)


def _ask_date(label: str, default: str = "") -> str:
    import re

    while True:
        val = questionary.text(label, default=default).ask()
        if val is None:
            _abort()
        val = val.strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", val):
            return val
        console.print("[red]Expected format: YYYY-MM-DD[/red]")


def _load_data_csv(path: Path):  # noqa: ANN202
    """Read a data CSV with its local time zone; ask for the zone when the file doesn't name it."""
    from ._frame_csv import load_frame_csv

    tz = None
    while True:
        try:
            return load_frame_csv(path, tz=tz)
        except ValueError as exc:
            console.print(f"  [yellow]{exc}[/yellow]")
            tz = questionary.text("  Time zone of the data (e.g. Europe/Warsaw)").ask()
            if tz is None:
                _abort()
            tz = tz.strip() or None


def _ask_freq() -> str:
    freq = questionary.select(
        "  Data resolution (one model per period)", choices=["1h", "15min", "1D"], default="1h"
    ).ask()
    if freq is None:
        _abort()
    return freq


# ---------------------------------------------------------------------------
# Component configuration
# ---------------------------------------------------------------------------


def configure_component(class_name: str, available_columns: Optional[list[str]] = None) -> dict:
    """Ask the user to fill in all params for a component. Returns a params dict."""
    schema = PARAM_SCHEMAS.get(class_name, [])
    params: dict[str, Any] = {}
    for field in schema:
        val = _ask_field(field, available_columns=available_columns)
        if val is not None and val != "":
            params[field["name"]] = val
    return params


def _instantiate(class_name: str, params: dict) -> Any:
    """Import and instantiate a component by class name using the pipeline registries."""
    from .pipelines.data_pipeline import COMPONENT_REGISTRY as DR
    from .pipelines.model_pipeline import COMPONENT_REGISTRY as MR

    all_registries = {
        **DR.get("sources", {}),
        **DR.get("transformers", {}),
        **DR.get("validators", {}),
        **MR.get("models", {}),
        **MR.get("evaluators", {}),
        **MR.get("exporters", {}),
    }
    if class_name not in all_registries:
        raise ValueError(f"Unknown component: {class_name}")
    import importlib

    module = importlib.import_module(all_registries[class_name])
    return getattr(module, class_name)(**params)


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

_SOURCES = [
    "EntsoeSource",
    "OpenMeteoSource",
    "CsvSource",
    "CalendarSource",
    "EpftoolboxBenchmarkSource",
    "PseSource",
    "EnergyChartsSource",
    "ForecastSource",
]
_TRANSFORMERS = [
    "TimezoneTransformer",
    "ResampleTransformer",
    "LagTransformer",
    "DSTNormalizeTransformer",
    "DailyProfileTransformer",
    "DailyStatsTransformer",
    "LambdaTransformer",
    "ImputeTransformer",
]
_VALIDATORS = [
    "EdaValidator",
    "DSTStructureValidator",
    "CoverageValidator",
    "PlausibilityValidator",
]
_MODELS = [
    "OLSModel",
    "LassoCVModel",
    "RidgeModel",
    "LEARModel",
    "NaiveModel",
    "HistGBModel",
    "LightGBMModel",
    "XGBoostModel",
    "CatBoostModel",
    "MLPDailyModel",
    "MLPHourlyModel",
    "TorchModel",
]
# Not offered: `PooledModel` (registered in `COMPONENT_REGISTRY`, saved and loaded from YAML) fits over the frames of
# `ModelPipeline.run(data={zone: frame})`, and the wizard builds and runs a pipeline on one CSV. `ResidualModel` holds two
# models (a base and a learner), which the wizard's one-model-at-a-time questions don't build; it loads from YAML.
_EVALUATORS = [
    "MAEEvaluator",
    "RMSEEvaluator",
    "rMAEEvaluator",
    "BiasEvaluator",
    "ShapeMAEEvaluator",
    "NMAEEvaluator",
    "ScaledMAEEvaluator",
    "DMEvaluator",
    "MCSEvaluator",
    "EconomicEvaluator",
]
_EXPORTERS = ["TerminalExporter", "ExcelExporter", "CsvExporter"]


def _build_component_list(choices: list[str], label: str, available_columns: Optional[list[str]] = None) -> list[Any]:
    """Generic loop: select component type → configure → add another?"""
    components = []
    while True:
        class_name = questionary.select(f"  Select {label} type", choices=choices).ask()
        if class_name is None:
            _abort()
        console.print(f"  [bold]Configure {class_name}[/bold]")
        params = configure_component(class_name, available_columns=available_columns)
        components.append(_instantiate(class_name, params))
        again = questionary.confirm(f"  Add another {label}?", default=False).ask()
        if again is None:
            _abort()
        if not again:
            break
    return components


def build_data_pipeline_interactive():  # noqa: ANN201
    from .pipelines.data_pipeline import DataPipeline

    console.print("\n[bold cyan]── Data Pipeline ──────────────────────────[/bold cyan]")

    # Sources (at least one required)
    console.print("\n[bold]Step 1/6: Sources[/bold]")
    sources = _build_component_list(_SOURCES, "source")

    # Transformers (optional)
    console.print("\n[bold]Step 2/6: Transformers[/bold]")
    transformers = []
    if questionary.confirm("  Add transformers?", default=False).ask():
        transformers = _build_component_list(_TRANSFORMERS, "transformer")

    # Validators (optional)
    console.print("\n[bold]Step 3/6: Validators[/bold]")
    validators = []
    if questionary.confirm("  Add validators?", default=False).ask():
        validators = _build_component_list(_VALIDATORS, "validator")

    # Run config
    console.print("\n[bold]Step 4/6: Date range[/bold]")
    start = _ask_date("  Start date (YYYY-MM-DD)")
    end = _ask_date("  End date   (YYYY-MM-DD)")

    cache_enabled = questionary.confirm("  Enable source caching?", default=False).ask()
    if cache_enabled is None:
        _abort()
    cache: Any = False
    if cache_enabled:
        cache_path = questionary.text("  Cache path (leave blank for automatic)").ask()
        if cache_path is None:
            _abort()
        cache = cache_path.strip() if cache_path.strip() else True

    # Output
    console.print("\n[bold]Step 5/6: Output[/bold]")
    while True:
        output = questionary.text("  Output CSV path").ask()
        if output is None:
            _abort()
        output = output.strip()
        if output:
            break
        console.print("  [red]Output path is required.[/red]")

    dp = DataPipeline(sources=sources, transformers=transformers, validators=validators)

    # Save YAML?
    console.print("\n[bold]Step 6/6: Save & run[/bold]")
    if questionary.confirm("  Save pipeline YAML?", default=True).ask():
        yaml_path = questionary.text("  YAML path", default="data_pipeline.yaml").ask()
        if yaml_path is None:
            _abort()
        dp.save(yaml_path)
        console.print(f"  [green]Saved →[/green] {yaml_path}")

    if questionary.confirm("  Run now?", default=True).ask():
        from ._frame_csv import save_frame_csv

        df = dp.run(start=start, end=end, cache=cache)
        if df.empty:
            console.print("[yellow]Warning: pipeline returned an empty DataFrame.[/yellow]")
        save_frame_csv(df, output)
        console.print(f"[green]Saved {len(df)} rows →[/green] {output}")


def build_model_pipeline_interactive() -> None:
    from .information_set import DEFAULT_ISSUE_TIME
    from .pipelines.model_pipeline import ModelPipeline

    console.print("\n[bold cyan]── Model Pipeline ─────────────────────────[/bold cyan]")

    # Data CSV
    console.print("\n[bold]Step 1/7: Input data[/bold]")
    while True:
        data_path = questionary.text("  Input CSV path (from epf data)").ask()
        if data_path is None:
            _abort()
        p = Path(data_path.strip())
        if p.exists():
            break
        console.print(f"  [red]File not found: {p}[/red]")

    df = _load_data_csv(p)
    available_columns = list(df.columns)
    console.print(f"  Loaded {len(df)} rows, columns: {', '.join(available_columns)}")

    # Models
    console.print("\n[bold]Step 2/7: Models[/bold]")
    models = _build_component_list(_MODELS, "model", available_columns=available_columns)

    # Evaluators
    console.print("\n[bold]Step 3/7: Evaluators[/bold]")
    evaluators = []
    if questionary.confirm("  Add evaluators?", default=True).ask():
        evaluators = _build_component_list(_EVALUATORS, "evaluator")

    # Exporters
    console.print("\n[bold]Step 4/7: Exporters[/bold]")
    exporters = []
    if questionary.confirm("  Add exporters?", default=True).ask():
        exporters = _build_component_list(_EXPORTERS, "exporter")

    # Run config
    console.print("\n[bold]Step 5/7: Test period[/bold]")
    test_start = _ask_date("  Test start (YYYY-MM-DD)")
    test_end = _ask_date("  Test end   (YYYY-MM-DD)")

    target = questionary.select("  Target column", choices=available_columns).ask()
    if target is None:
        _abort()

    horizon_raw = questionary.text("  Forecast horizon (days)", default="7").ask()
    if horizon_raw is None:
        _abort()
    horizon = int(horizon_raw.strip())
    freq = _ask_freq()

    forecast_only = questionary.confirm("  Forecast-only mode (skip evaluation)?", default=False).ask()
    if forecast_only is None:
        _abort()

    save_dir_raw = questionary.text("  Save predictions to directory (blank to skip)").ask()
    if save_dir_raw is None:
        _abort()
    save_dir = save_dir_raw.strip() or None

    issue_time = questionary.text(
        "  Issue time to check predictors against, Brussels time (HH:MM, blank to skip)", default=DEFAULT_ISSUE_TIME
    ).ask()
    if issue_time is None:
        _abort()

    mp = ModelPipeline(issue_time=issue_time.strip() or None)
    for m in models:
        mp.add_model(m)
    for e in evaluators:
        mp.add_evaluator(e)
    for ex in exporters:
        mp.add_exporter(ex)

    # Save YAML?
    console.print("\n[bold]Step 6/7: Save YAML[/bold]")
    if questionary.confirm("  Save pipeline YAML?", default=True).ask():
        yaml_path = questionary.text("  YAML path", default="model_pipeline.yaml").ask()
        if yaml_path is None:
            _abort()
        mp.save(yaml_path)
        console.print(f"  [green]Saved →[/green] {yaml_path}")

    # Run
    console.print("\n[bold]Step 7/7: Run[/bold]")
    if questionary.confirm("  Run now?", default=True).ask():
        mp.run(
            data=df,
            test_start=test_start,
            test_end=test_end,
            target=target,
            horizon=horizon,
            freq=freq,
            save_dir=save_dir,
            forecast_only=forecast_only,
        )


# ---------------------------------------------------------------------------
# YAML-based runners (thin wrappers used from main menu)
# ---------------------------------------------------------------------------


def _run_experiment_from_yaml() -> None:
    path = questionary.text("  Workflow YAML path").ask()
    if path is None:
        _abort()
    from .pipelines.workflow import Workflow

    Workflow.load(path.strip()).run()


def _run_data_from_yaml() -> None:
    from .pipelines.data_pipeline import DataPipeline

    path = questionary.text("  DataPipeline YAML path").ask()
    if path is None:
        _abort()
    start = _ask_date("  Start date (YYYY-MM-DD)")
    end = _ask_date("  End date   (YYYY-MM-DD)")
    output = questionary.text("  Output CSV path").ask()
    if output is None:
        _abort()

    from ._frame_csv import save_frame_csv

    dp = DataPipeline.load(path.strip())
    df = dp.run(start=start, end=end)
    save_frame_csv(df, output.strip())
    console.print(f"[green]Saved {len(df)} rows →[/green] {output.strip()}")


def _run_model_from_yaml() -> None:
    from .pipelines.model_pipeline import ModelPipeline

    path = questionary.text("  ModelPipeline YAML path").ask()
    if path is None:
        _abort()

    while True:
        data_path = questionary.text("  Input CSV path").ask()
        if data_path is None:
            _abort()
        p = Path(data_path.strip())
        if p.exists():
            break
        console.print(f"  [red]File not found: {p}[/red]")

    df = _load_data_csv(p)

    test_start = _ask_date("  Test start (YYYY-MM-DD)")
    test_end = _ask_date("  Test end   (YYYY-MM-DD)")
    target = questionary.text("  Target column", default="price").ask()
    if target is None:
        _abort()
    horizon_raw = questionary.text("  Forecast horizon (days)", default="7").ask()
    if horizon_raw is None:
        _abort()
    freq = _ask_freq()

    mp = ModelPipeline.load(path.strip())
    mp.run(
        data=df,
        test_start=test_start,
        test_end=test_end,
        target=target.strip(),
        horizon=int(horizon_raw.strip()),
        freq=freq,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_MENU_CHOICES = [
    questionary.Choice("Run experiment from YAML", value="run_experiment"),
    questionary.Choice("Run data pipeline from YAML → CSV", value="run_data"),
    questionary.Choice("Run model pipeline from YAML", value="run_model"),
    questionary.Separator(),
    questionary.Choice("Build & run data pipeline interactively", value="build_data"),
    questionary.Choice("Build & run model pipeline interactively", value="build_model"),
    questionary.Separator(),
    questionary.Choice("Exit", value="exit"),
]

_DISPATCH = {
    "run_experiment": _run_experiment_from_yaml,
    "run_data": _run_data_from_yaml,
    "run_model": _run_model_from_yaml,
    "build_data": build_data_pipeline_interactive,
    "build_model": build_model_pipeline_interactive,
}


def run_interactive() -> None:
    banner = Text("EPF Toolbox 2", style="bold cyan", justify="center")
    console.print(Panel(banner, subtitle="electricity price forecasting", padding=(0, 4)))

    while True:
        action = questionary.select("What do you want to do?", choices=_MENU_CHOICES).ask()
        if action is None or action == "exit":
            console.print("[dim]Bye.[/dim]")
            break
        try:
            _DISPATCH[action]()
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            break
        except SystemExit:
            raise
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")
