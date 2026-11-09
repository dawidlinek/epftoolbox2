"""
Example: a solar generation forecast chained into a day-ahead price forecast

The renewables study (docs page reproductions/renewables) in small, for DE-LU, issued at 11:45 Berlin on D for D+1:

1. Solar: a ridge per hour on issue-time weather (Open-Meteo's ICON from the run of 00 UTC on D) and the sun's position,
   fitted on the capacity factor of a capacity proxy (the largest hour of the year to D-1), clipped to 0..105 % of it and 0
   where the sun is down. Its out-of-sample forecasts are stored in OUTPUT_DIR/runs/solar, with the TSO's forecast of D (out
   at 18:00 on D-1) as an honest benchmark.
2. Price: `ForecastSource` reads the stored forecasts back as a column of the price model's data, so that on the row of D
   `solar_fc_d+1` is the forecast for D+1 issued on D, known from its issue time. A ridge per hour forecasts the hourly
   day-ahead price with and without it, on identical keys, with a Diebold-Mariano test.

The solar run covers the price model's training rows as well as its test, so the price model trains on the forecasts it is
served. Training it on actual generation and serving it forecasts is a train/serve mismatch: in the study it cost 0.35 EUR/MWh
in DE (significant at 5 %).

No API key is needed: EnergyChartsSource (Energy-Charts.info, Fraunhofer ISE, CC BY 4.0; the DE-LU prices from
Bundesnetzagentur | SMARD.de, CC BY 4.0, attribution required where you publish a result) and OpenMeteoSource (ICON
archive from 2024-01-18, so the solar forecasts start mid-2024). The solar run is checked at availability="raise". The price
run is at "warn": where the solar run has no forecast for an hour (a missing weather hour, say), the price model's training
rows there have no value of `solar_fc_d+1` and drop out of training, which "warn" reports and "raise" refuses; the run
then prints the late inputs it found, of which there must be none.
"""

import os

os.environ["MAX_PROCESSES"] = "4"

from pathlib import Path

from epftoolbox2.data.sources import CalendarSource, EnergyChartsSource, ForecastSource, OpenMeteoSource
from epftoolbox2.data.transformers import (
    LagTransformer,
    ResampleTransformer,
    RollingStatsTransformer,
    SolarGeometryTransformer,
    TimezoneTransformer,
)
from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator
from epftoolbox2.exporters import CsvExporter, TerminalExporter
from epftoolbox2.information_set import InformationSet
from epftoolbox2.models import NaiveModel, RidgeModel
from epftoolbox2.pipelines import DataPipeline, ModelPipeline

ZONE = "Europe/Berlin"
SITE = (51.0, 10.0)  # the middle of Germany; the study averaged eight cities
WEEKDAYS = ["is_tuesday", "is_wednesday", "is_thursday", "is_friday", "is_saturday", "is_sunday"]


def solar_data(data_start, data_end):
    """Actual solar generation, the TSO's day-ahead solar forecast, ICON radiation and cloud cover for D+1, the sun's
    position, and the capacity proxy `ec_actual_solar_roll365max_d-1`."""
    pipeline = (
        DataPipeline(timezone=ZONE)
        .add_source(
            EnergyChartsSource(
                "de",
                type=["power", "forecast"],
                power=["solar"],
                production_types=["solar"],
                forecast_types=["day-ahead"],
            )
        )
        .add_source(
            OpenMeteoSource(
                latitude=SITE[0],
                longitude=SITE[1],
                columns=["shortwave_radiation", "cloud_cover"],
                model="icon_seamless",
                horizon=1,
                prefix="de",
            )
        )
        .add_transformer(TimezoneTransformer())
        .add_transformer(ResampleTransformer(freq="1h"))  # Energy-Charts' 15-minute grid as hourly means
        .add_transformer(SolarGeometryTransformer(locations=[SITE]))
        # The largest hour of the 365 days to D-1, where 300 of them are complete (a missing hour leaves a day incomplete)
        .add_transformer(
            RollingStatsTransformer("ec_actual_solar", window_days=365, stat="max", lag_days=1, min_days=300)
        )
        .add_transformer(LagTransformer(columns=["clear_sky_ghi", "is_night"], lags=[-1], freq="day"))
    )
    return pipeline, pipeline.run(data_start, data_end, cache=True)


def price_data(data_start, data_end, solar_dir):
    """The hourly day-ahead price and the TSO's load forecast of D, weekday dummies, and the stored solar forecasts with their
    daily mean, led by a day (`solar_fc_d+1`: the forecast for D+1 issued on D)."""
    pipeline = (
        DataPipeline(timezone=ZONE)
        .add_source(EnergyChartsSource("de", type=["price", "forecast"], production_types=["load"]))
        .add_source(CalendarSource(country="DE", holidays=False, weekday="onehot"))
        .add_source(ForecastSource(solar_dir, model="solar_ridge", column="solar_fc"))
        .add_source(ForecastSource(solar_dir, model="solar_ridge", column="solar_fc", agg="mean"))
        .add_transformer(TimezoneTransformer())
        .add_transformer(ResampleTransformer(freq="1h"))
        .add_transformer(LagTransformer(columns=["ec_price_day_ahead"], lags=[1, 6], freq="day"))
        .add_transformer(LagTransformer(columns=["solar_fc", "solar_fc_day_mean", *WEEKDAYS], lags=[-1], freq="day"))
    )
    return pipeline, pipeline.run(data_start, data_end, cache=True)


def main(
    data_start="2023-06-01",
    data_end="2026-10-31",
    solar_test=("2024-06-30", "2026-10-30"),
    price_test=("2025-06-30", "2026-10-30"),
    output_dir=".",
):
    """Stage 1 over `solar_test` (issue days; it must start a price training window, 364 days, before `price_test`), then
    stage 2 over `price_test`. Returns the two reports."""
    output = Path(output_dir)
    solar_dir = output / "runs" / "solar"

    # 1. Solar generation
    sources, df = solar_data(data_start, data_end)
    solar = ModelPipeline(InformationSet("11:45", ZONE).with_source_rules(sources.sources), availability="raise")
    solar.add_model(
        RidgeModel(
            ["de_shortwave_radiation_d+{horizon}", "de_cloud_cover_d+{horizon}", "clear_sky_ghi_d+{horizon}"],
            training_window=365,
            refit="7D",
            normalise_by="ec_actual_solar_roll365max_d-1",
            bounds=(0, 1.05),
            zero_when="is_night_d+{horizon}",
            name="solar_ridge",
        )
    )
    solar.add_model(NaiveModel(["ec_solar_forecast_day_ahead"], name="tso_of_d"))  # the TSO's forecast of D, out on D-1
    solar.add_evaluator(MAEEvaluator())
    solar.add_exporter(TerminalExporter())
    solar_report = solar.run(df, *solar_test, target="ec_actual_solar", horizon=1, save_dir=solar_dir)

    # 2. Price, with and without the chained solar forecast
    sources, df = price_data(data_start, data_end, solar_dir)
    base = ["ec_price_day_ahead", "ec_price_day_ahead_d-1", "ec_price_day_ahead_d-6", "ec_load_forecast_day_ahead"]
    base += [f"{day}_d+{{horizon}}" for day in WEEKDAYS]
    with_solar = base + ["solar_fc_d+{horizon}", "solar_fc_day_mean_d+{horizon}"]
    # The target and the price inputs through asinh, every other input standardised
    vst = {"name": "asinh", "params": {"apply_to": ["ec_price_day_ahead*"]}}
    price = ModelPipeline(InformationSet("11:45", ZONE).with_source_rules(sources.sources), availability="warn")
    price.add_model(RidgeModel(base, training_window=364, scaler=vst, name="price_base"))
    price.add_model(RidgeModel(with_solar, training_window=364, scaler=vst, name="price_solar"))
    price.add_evaluator(MAEEvaluator())
    price.add_evaluator(RMSEEvaluator())
    price.add_exporter(TerminalExporter())
    price.add_exporter(CsvExporter(output / "chained_solar_price.csv"))
    price_report = price.run(
        df, *price_test, target="ec_price_day_ahead", horizon=1, save_dir=output / "runs" / "price"
    )
    print(price_report.availability())  # inputs read after their issue time or without a rule: none
    print(price_report.tests(benchmark="price_base", lag=6))  # Newey-West, lag 6, as in the study
    return solar_report, price_report


if __name__ == "__main__":
    main()
