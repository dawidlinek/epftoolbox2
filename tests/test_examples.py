"""End-to-end runs of every script in examples/ against the live APIs, on a short window.

Each example has a `main(...)` that takes its date ranges and an output directory, so a test can run the real
script on a few weeks of data. These tests need ENTSOE_API_KEY and the network, which is why they are marked
`integration` (CI's fast suite skips them) and `examples` (run only these with `uv run pytest -m examples`).

Every test runs in its own `live_dir` (see conftest.py): it is the working directory, so the source cache
(`.cache/sources`) and the output files land there, and the example is loaded from its file like a user would run
it. A test asserts real outcomes: a non-empty data frame with the expected columns, forecasts for every model on
the same keys, no unexpected missing forecasts, a passing information-set check and the output files.
"""

import datetime as dt
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from epftoolbox2.models._calendar import IssueCalendar, clock_periods
from tests.live import (
    DATA_END,
    DATA_START,
    HORIZON,
    LOAD_15MIN_COLUMNS,
    PRICE_COLUMNS,
    RUN_DAYS,
    TEST_END,
    TEST_START,
    assert_csv,
    assert_data_frame,
    assert_excel,
    assert_forecasts,
    assert_information_set_checked,
    assert_metrics,
    switch_days,
)

pytestmark = [pytest.mark.integration, pytest.mark.examples]

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load_example(filename: str, monkeypatch):
    """Import examples/<filename> from its path, then set one process and one thread again.

    The examples set MAX_PROCESSES while they are imported.
    """
    spec = importlib.util.spec_from_file_location(f"example_{Path(filename).stem}", EXAMPLES / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    return module


def test_01_data_pipeline_only(live_dir, monkeypatch):
    example = load_example("01_data_pipeline_only.py", monkeypatch)

    df = example.main(data_start=DATA_START, data_end=DATA_END, output_dir=live_dir)

    assert_data_frame(
        df,
        PRICE_COLUMNS,
        "Europe/Warsaw",
        dt.date.fromisoformat(DATA_START),
        dt.date.fromisoformat(DATA_END),
    )
    written = live_dir / "data_output.csv"
    assert written.is_file()
    reread = pd.read_csv(written, index_col=0)
    assert reread.shape == df.shape
    assert list(reread.columns) == list(df.columns)


def test_02_model_pipeline_only(live_dir, monkeypatch, information_set_checks):
    # Example 2 reads the CSV that example 1 writes
    data_example = load_example("01_data_pipeline_only.py", monkeypatch)
    data_example.main(data_start=DATA_START, data_end=DATA_END, output_dir=live_dir)
    example = load_example("02_model_pipeline_only.py", monkeypatch)
    models = ["OLS Baseline", "Lasso CV"]
    planned = RUN_DAYS * HORIZON * 24

    report = example.main(
        data_path=live_dir / "data_output.csv", test_start=TEST_START, test_end=TEST_END, output_dir=live_dir
    )

    assert_data_frame(
        report.source_data,
        PRICE_COLUMNS,
        "Europe/Warsaw",
        dt.date.fromisoformat(DATA_START),
        dt.date.fromisoformat(DATA_END),
    )
    assert_forecasts(report, models, planned)
    assert_metrics(report, models, planned, columns=("MAE", "RMSE", "rMAE"))
    assert_information_set_checked(information_set_checks, models)
    assert_excel(live_dir / "model_results.xlsx", models, planned)
    assert_csv(live_dir / "model_results.csv", models, planned)
    assert sorted(p.name for p in (live_dir / "results").glob("*.jsonl")) == ["lasso_cv.jsonl", "ols_baseline.jsonl"]


def test_03_full_workflow(live_dir, monkeypatch, information_set_checks):
    example = load_example("03_full_workflow.py", monkeypatch)
    models = ["OLS", "LassoCV"]
    planned = RUN_DAYS * HORIZON * 24

    report = example.main(
        data_start=DATA_START, data_end=DATA_END, test_start=TEST_START, test_end=TEST_END, output_dir=live_dir
    )

    assert_data_frame(
        report.source_data,
        PRICE_COLUMNS,
        "Europe/Warsaw",
        dt.date.fromisoformat(DATA_START),
        dt.date.fromisoformat(DATA_END),
    )
    assert_forecasts(report, models, planned)
    assert_metrics(report, models, planned)
    assert_information_set_checked(information_set_checks, models)
    assert_excel(live_dir / "full_workflow_results.xlsx", models, planned)
    assert_csv(live_dir / "full_workflow_results.csv", models, planned)
    assert sorted(p.name for p in (live_dir / "results").glob("*.jsonl")) == ["lassocv.jsonl", "ols.jsonl"]
    assert (live_dir / ".cache" / "sources").is_dir()


def test_04_yaml_workflow(live_dir, monkeypatch, information_set_checks, entsoe_api_key):
    example = load_example("04_yaml_workflow.py", monkeypatch)
    models = ["OLS", "LassoCV"]
    planned = RUN_DAYS * HORIZON * 24

    report = example.main(
        data_start=DATA_START,
        data_end=DATA_END,
        test_start=TEST_START,
        test_end=TEST_END,
        output_dir=live_dir,
        run=True,
    )

    yamls = {
        name: (live_dir / name).read_text() for name in ("data_pipeline.yaml", "model_pipeline.yaml", "experiment.yaml")
    }
    assert "${ENTSOE_API_KEY}" in yamls["data_pipeline.yaml"]
    assert all(entsoe_api_key not in text for text in yamls.values()), "the API key was written to a YAML file"
    assert_data_frame(
        report.source_data,
        PRICE_COLUMNS,
        "Europe/Warsaw",
        dt.date.fromisoformat(DATA_START),
        dt.date.fromisoformat(DATA_END),
    )
    assert_forecasts(report, models, planned)
    assert_metrics(report, models, planned)
    assert_information_set_checked(information_set_checks, models)
    assert_excel(live_dir / "workflow_results.xlsx", models, planned)


def test_05_forecast_only(live_dir, monkeypatch, information_set_checks):
    example = load_example("05_forecast_only.py", monkeypatch)
    models = ["OLS", "LassoCV"]
    horizon = 5

    # The example's dates are relative to today, so the window is too: 60 days of history through today + 5.
    before = dt.date.today()
    report = example.main(data_start="now_d-60")
    after = dt.date.today()

    predictions = report.predictions()
    run_dates = {dt.date.fromisoformat(d) for d in predictions["run_date"]}
    assert len(run_dates) == 1 and before <= run_dates.pop() <= after
    today = before
    # Weather and calendar rows run through the whole 5-day prediction window
    assert_data_frame(
        report.source_data,
        PRICE_COLUMNS,
        "Europe/Warsaw",
        today - dt.timedelta(days=60),
        today + dt.timedelta(days=5),
    )
    # The hour a spring switch skips, on today or a target day, is no planned forecast; a lag or lead that lands on it may cost one
    # forecast per model and switch day
    planned = IssueCalendar("1h", "Europe/Warsaw", range(1, horizon + 1)).count(today, today)
    switching = switch_days(today + dt.timedelta(days=1), today + dt.timedelta(days=horizon), "Europe/Warsaw")
    assert_forecasts(report, models, planned, max_missing=len(switching))
    assert len(predictions) == report.forecast_counts()["produced"].sum()
    assert set(predictions["horizon"]) == set(range(1, horizon + 1))
    assert set(predictions["hour"]) == set(clock_periods(today, "1h", "Europe/Warsaw"))
    assert_information_set_checked(information_set_checks, models)
    # forecast_only skips the evaluators and the exporters
    assert report.evaluators == []
    assert not list(live_dir.glob("*.xlsx")) and not list(live_dir.glob("*.csv"))


def test_06_15min_forecasting(live_dir, monkeypatch, information_set_checks):
    example = load_example("06_15min_forecasting.py", monkeypatch)
    models = ["OLS", "LassoCV"]
    planned = RUN_DAYS * HORIZON * 96

    report = example.main(
        data_start=DATA_START, data_end=DATA_END, test_start=TEST_START, test_end=TEST_END, output_dir=live_dir
    )

    assert_data_frame(
        report.source_data,
        LOAD_15MIN_COLUMNS,
        "Europe/Berlin",
        dt.date.fromisoformat(DATA_START),
        dt.date.fromisoformat(DATA_END),
    )
    assert report.source_data.index.to_series().diff().dropna().eq(pd.Timedelta("15min")).all()
    assert_forecasts(report, models, planned)
    assert_metrics(report, models, planned, columns=("MAE", "RMSE", "rMAE"))
    assert_information_set_checked(information_set_checks, models, zone="Europe/Berlin")
    assert pd.read_csv(live_dir / "15min_forecasting_data.csv", index_col=0).shape == report.source_data.shape
    assert_excel(live_dir / "15min_workflow_results.xlsx", models, planned)
    assert_csv(live_dir / "15min_workflow_results.csv", models, planned)
    assert sorted(p.name for p in (live_dir / "results_15min").glob("*.jsonl")) == ["lassocv.jsonl", "ols.jsonl"]


def test_lear_lago(keyless_live_dir, monkeypatch, information_set_checks):
    """The benchmark needs no key (Zenodo), so this one runs without ENTSOE_API_KEY; it downloads DE.csv unless the md5-checked
    copy is cached already (EpftoolboxBenchmarkSource's cache, not the working directory's)."""
    example = load_example("lear_lago.py", monkeypatch)
    members = ["LEAR_asinh", "LEAR_mlog", "LEAR_poly", "LEAR_npit"]
    models = ["LEAR_lago", *members, "LEAR"]
    planned = 2 * 24  # two issue days, h = 1

    report = example.main(market="DE", test_start="2016-01-03", test_end="2016-01-04", output_dir=keyless_live_dir)

    assert_data_frame(
        report.source_data,
        ["price", "price_h00", "price_h23_d-6", "load_forecast_amprion_h12_d+1", "wind_solar_forecast_h12_d-6"],
        "None",  # DSTNormalizeTransformer's 24 slots a day on the naive wall clock
        dt.date(2012, 1, 9),
        dt.date(2016, 1, 5),
    )
    assert_forecasts(report, models, planned)
    assert_metrics(report, models, planned)
    assert_information_set_checked(information_set_checks, ["LEAR_lago", *members], zone="Europe/Brussels")
    predictions = report.predictions().pivot_table(index=["run_date", "hour"], columns="model", values="prediction")
    assert (predictions["LEAR"] - predictions[members].mean(axis=1)).abs().max() < 1e-9  # the ensemble is their mean
    assert_csv(keyless_live_dir / "lear_DE.csv", models, planned)
    assert sorted(p.name for p in (keyless_live_dir / "results_lear_DE").glob("*.jsonl")) == sorted(
        f"{name.lower()}.jsonl" for name in models
    )


def test_mlp_search(keyless_live_dir, monkeypatch, information_set_checks):
    """Two configurations of one seed on four validation and two test issue days of the DE benchmark (no key: Zenodo), the
    ensemble of the better one on validation; needs the [torch] extra."""
    pytest.importorskip("torch")
    example = load_example("mlp_search.py", monkeypatch)
    models = ["mlp_00", "mlp_01", "mlp_top4"]
    planned = 2 * 24  # two test issue days, h = 1

    report = example.main(
        n_configs=2,
        seeds=1,
        validation=("2015-12-27", "2015-12-30"),
        test=("2016-01-03", "2016-01-04"),
        output_dir=keyless_live_dir,
    )

    assert_forecasts(report, models, planned)
    assert_metrics(report, models, planned)
    assert_information_set_checked(information_set_checks, models[:2], zone="Europe/Brussels")
    chosen = report.choices()
    assert len(chosen) == 1 and list(chosen.iloc[0]["chosen"]) == models[:2]  # top=4 of two: both
    assert sorted(p.name for p in (keyless_live_dir / "results_mlp_search").glob("*.jsonl")) == sorted(
        f"{name}.jsonl" for name in models
    )


# ---------------------------------------------------------------------------------------------------------------------
# examples/chained_solar_price.py, offline: its two keyless sources answer from a synthetic market
# ---------------------------------------------------------------------------------------------------------------------

CHAIN_DATA = ("2021-12-01", "2024-03-05")
# The price test, and the solar test from a price training window (364 days and the issue day) before it to its last day.
CHAIN_PRICE = ("2024-02-26", "2024-03-03")
CHAIN_SOLAR = ("2023-02-26", "2024-03-03")


def synthetic_market() -> pd.DataFrame:
    """Hourly UTC series of a sunny market, seeded: the sine of the sun's elevation at (51, 10), daily cloud cover, solar
    generation growing with its capacity, the TSO's forecast of it, ICON-like radiation and cloud cover for the same hour a
    day later (as `OpenMeteoSource(horizon=1)` emits it), a load forecast and a price that falls with solar generation."""
    import numpy as np

    from epftoolbox2.data.transformers import SolarGeometryTransformer

    rng = np.random.default_rng(7)
    index = pd.date_range(CHAIN_DATA[0], pd.Timestamp(CHAIN_DATA[1]) + pd.Timedelta(days=2), freq="1h", tz="UTC")
    sun = (
        SolarGeometryTransformer([(51.0, 10.0)]).transform(pd.DataFrame(index=index))["sun_sin_elevation"].clip(lower=0)
    )
    days = (index - index[0]).days.to_numpy()
    daily = np.zeros(days.max() + 1)
    for i in range(1, len(daily)):
        daily[i] = 0.7 * daily[i - 1] + rng.normal(0, 0.25)
    cloud = np.clip(0.5 + daily[days] + rng.normal(0, 0.05, len(index)), 0, 1)
    capacity = 40_000 + 15_000 * days / days.max()
    solar = capacity * 0.75 * sun.to_numpy() * (1 - 0.6 * cloud) * (1 + rng.normal(0, 0.05, len(index)))
    solar = np.clip(solar, 0, None)
    hour = index.tz_convert("Europe/Berlin").hour.to_numpy()
    weekend = index.tz_convert("Europe/Berlin").dayofweek.to_numpy() >= 5
    load = 55_000 + 8_000 * np.sin(2 * np.pi * (hour - 8) / 24) - 6_000 * weekend + rng.normal(0, 800, len(index))
    noise = np.zeros(len(index))
    for i in range(1, len(index)):
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 6)
    frame = pd.DataFrame(
        {
            "ec_actual_solar": solar,
            "ec_solar_forecast_day_ahead": solar * (1 + rng.normal(0, 0.1, len(index))),
            "ec_load_forecast_day_ahead": load,
            "ec_price_day_ahead": 90 + 0.002 * (load - 55_000) - 0.0012 * solar + noise,
            "de_shortwave_radiation_d+1": 1000 * sun.to_numpy() * (1 - 0.7 * cloud + rng.normal(0, 0.08, len(index))),
            "de_cloud_cover_d+1": 100 * np.clip(cloud + rng.normal(0, 0.1, len(index)), 0, 1),
        },
        index=index,
    )
    # Row t holds the forecast for the same hour a day later.
    frame[["de_shortwave_radiation_d+1", "de_cloud_cover_d+1"]] = frame[
        ["de_shortwave_radiation_d+1", "de_cloud_cover_d+1"]
    ].shift(-24)
    return frame


@pytest.fixture
def synthetic_sources(monkeypatch, no_network):
    """EnergyChartsSource and OpenMeteoSource answer from `synthetic_market()` with the columns their settings ask for, in
    [start, end) and UTC, as their `fetch` does; anything that reaches for the network fails the test (`no_network`)."""
    from epftoolbox2.data.sources import EnergyChartsSource, OpenMeteoSource

    market = synthetic_market()

    def energy_charts(self, start, end):
        names = []
        if "power" in self.type:
            names += [f"actual_{series}" for series in self.power]
        if "forecast" in self.type:
            names += [
                "load_forecast_day_ahead" if p == "load" else f"{p}_forecast_day_ahead" for p in self.production_types
            ]
        if "price" in self.type:
            names.append("price_day_ahead")
        return market.loc[(market.index >= start) & (market.index < end), [f"{self.prefix}_{n}" for n in names]]

    def open_meteo(self, start, end):
        names = [f"{self.prefix}_{column}_d+{self.horizon}" for column in self.columns]
        return market.loc[(market.index >= start) & (market.index < end), names]

    monkeypatch.setattr(EnergyChartsSource, "fetch", energy_charts)
    monkeypatch.setattr(OpenMeteoSource, "fetch", open_meteo)


def test_chained_solar_price(keyless_live_dir, monkeypatch, information_set_checks, synthetic_sources):
    """Both stages end to end on a synthetic market, without a key or a network: the solar run covers the price model's
    training rows, the price model reads the solar run's stored forecasts back at their target instants, both price models
    forecast the same keys, and neither stage reads anything after its issue time."""
    from epftoolbox2.models._calendar import IssueCalendar

    example = load_example("chained_solar_price.py", monkeypatch)

    solar, price = example.main(
        data_start=CHAIN_DATA[0],
        data_end=CHAIN_DATA[1],
        solar_test=CHAIN_SOLAR,
        price_test=CHAIN_PRICE,
        output_dir=keyless_live_dir,
    )

    solar_planned = IssueCalendar("1h", "Europe/Berlin", [1]).count(
        dt.date.fromisoformat(CHAIN_SOLAR[0]), dt.date.fromisoformat(CHAIN_SOLAR[1])
    )
    assert_forecasts(solar, ["solar_ridge", "tso_of_d"], solar_planned)
    predictions = solar.predictions()
    ridge = predictions[predictions["model"] == "solar_ridge"]
    assert (ridge["prediction"] >= 0).all() and (ridge["prediction"] > 0).any()
    assert (ridge["prediction"] == 0).mean() > 0.3  # the hours the sun is down: 0 with no fit (zero_when)

    price_planned = 7 * 24  # a week of February, no switch
    assert_forecasts(price, ["price_base", "price_solar"], price_planned)
    assert_metrics(price, ["price_base", "price_solar"], price_planned)
    assert price.availability().empty
    tested = price.tests(benchmark="price_base", lag=6)
    assert list(tested["model"]) == ["price_solar"] and tested["p_value"].between(0, 1).all()
    # The fitted models' checks (the NaiveModel benchmark skips a value not out by the issue time instead)
    assert_information_set_checked(
        information_set_checks, ["solar_ridge", "price_base", "price_solar"], zone="Europe/Berlin"
    )

    # The chained column on the price frame is the solar run's forecast for that hour, issued the day before.
    frame = price.source_data
    assert_data_frame(
        frame,
        [
            "ec_price_day_ahead",
            "ec_price_day_ahead_d-6",
            "solar_fc",
            "solar_fc_d+1",
            "solar_fc_day_mean_d+1",
            "is_sunday_d+1",
        ],
        "Europe/Berlin",
        dt.date.fromisoformat(CHAIN_DATA[0]),
        dt.date.fromisoformat(CHAIN_DATA[1]),
    )
    stored = ridge.set_index(["target_date", "hour"])["prediction"]
    hours = frame.loc["2024-03-01", "solar_fc"]
    assert len(hours) == 24
    for stamp, value in hours.items():
        assert value == stored[(stamp.date().isoformat(), stamp.hour)]
    assert (frame.loc["2024-02-29", "solar_fc_d+1"].to_numpy() == hours.to_numpy()).all()

    assert_csv(keyless_live_dir / "chained_solar_price.csv", ["price_base", "price_solar"], price_planned)
    assert sorted(p.name for p in (keyless_live_dir / "runs" / "solar").glob("*.jsonl")) == [
        "solar_ridge.jsonl",
        "tso_of_d.jsonl",
    ]
    assert sorted(p.name for p in (keyless_live_dir / "runs" / "price").glob("*.jsonl")) == [
        "price_base.jsonl",
        "price_solar.jsonl",
    ]
    manifest = json.loads((keyless_live_dir / "runs" / "price" / "price_solar.manifest.json").read_text())
    assert set(manifest["chained_inputs"]) == {"solar_fc", "solar_fc_day_mean"}
