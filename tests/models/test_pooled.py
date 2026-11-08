"""Runs over several zones and `PooledModel` (step 7g).

A run of `data={zone: frame}` runs a plain model on each zone's frame exactly as a run of that frame alone does: records,
manifests and tables, bit for bit. A pooled model over one zone is the model itself, to the bit, for OLS, Ridge and LEAR, with
and without `refit`, hourly and on daily Friday rows. Over three zones its forecasts are a hand fit: each zone's rows scaled by
their own scaler, stacked with one-hot zones, one OLS, each zone's issue row scaled and inverted by its own. One fit serves zones
on three clocks, so every zone's training rows are judged as of the earliest issue instant of the day. A resume compares what a
pooled forecast depends on, including the other zones' data, and a resume cut short equals a fresh run.
"""

import json
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.ensembles import AverageEnsemble
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.exporters import TerminalExporter
from epftoolbox2.information_set import Availability, InformationSet, InformationSetError
from epftoolbox2.models import LEARModel, NaiveModel, OLSModel, RidgeModel, Use
from epftoolbox2.models._pooled import _pooled_init, _pooled_worker_init
from epftoolbox2.models._worker import _pool_init, _worker_init
from epftoolbox2.models.pooled import PooledModel
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import ManifestMismatchError
from epftoolbox2.results.panel import PanelReport
from tests.synthetic import calendar_shift, make_market_data, with_daily_columns

ZONES = {"PL": "Europe/Warsaw", "FI": "Europe/Helsinki", "PT": "Europe/Lisbon"}
PREDICTORS = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}"]
RULES = {
    "load": Availability.published("10:00", 7),  # its `_d+h` columns stand for a load forecast
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "price": Availability.published("13:00", 1),
    # Published at 13:00 Brussels on its own day: late at noon in Warsaw and Helsinki, known at noon in Lisbon (13:00 there).
    "price_next": Availability.published("13:00", 0),
}
HOURLY_TEST = ("2023-03-24", "2023-03-28")  # over the spring switch of 26 March
FRIDAYS = ("2023-03-03", "2023-04-28")
STAMPS = ("created_at", "updated_at")
_IN_PROCESS = {_pool_init: _worker_init, _pooled_init: _pooled_worker_init}


class InProcessPool:
    """Stands in for both pools, the model's and the pooled one, running their jobs here."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        _IN_PROCESS[initializer](*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, task, *args):
        future = Future()
        try:
            future.set_result(task(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass


@pytest.fixture(autouse=True)
def in_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)
    monkeypatch.setattr("epftoolbox2.models._pooled.ProcessPoolExecutor", InProcessPool)


def hourly(zone: str, start: str = "2023-01-01", seed: int = 0, scale: float = 1.0) -> pd.DataFrame:
    frame = make_market_data(start=start, days=125, tz=ZONES[zone], horizon=3, noise=2.0, seed=seed)
    frame["price_next"] = calendar_shift(frame["price"], 1)
    for column in ("load", "price", *[f"load_d+{h}" for h in (1, 2, 3)]):
        frame[column] = frame[column] * scale
    return frame


def daily(zone: str, start: str = "2023-01-01", seed: int = 0, scale: float = 1.0) -> pd.DataFrame:
    return with_daily_columns(hourly(zone, start, seed, scale)).at_time("12:00")


def lines(path) -> list:
    return sorted(Path(path).read_text().splitlines())


def manifest(path) -> dict:
    found = json.loads(Path(path).read_text())
    return {key: value for key, value in found.items() if key not in STAMPS}


def model_of(kind: str, refit=None, window: int = 30, name: str = "m"):
    if kind == "ols":
        return OLSModel(predictors=PREDICTORS, training_window=window, refit=refit, name=name)
    if kind == "ridge":
        return RidgeModel(predictors=PREDICTORS, training_window=window, refit=refit, name=name)
    return LEARModel(predictors=PREDICTORS, training_window=window, refit=refit, name=name)


# -- a pooled model over one zone is the model itself ------------------------------------------------------------------------

# Every model kind with each refit rule and frequency: 3 x 2 x 2 cells, each a few hundred fits.
ONE_ZONE_CELLS = [
    (kind, refit, freq) for kind in ("ols", "ridge", "lear") for refit in (None, "7D") for freq in ("1h", "1D")
]


@pytest.mark.parametrize("kind, refit, freq", ONE_ZONE_CELLS)
def test_a_pooled_model_over_one_zone_is_the_model_itself(tmp_path, kind, refit, freq):
    frame = hourly("PL") if freq == "1h" else daily("PL")
    run = (
        dict(test=HOURLY_TEST, horizon=2, freq="1h", target="price")
        if freq == "1h"
        else dict(test=FRIDAYS, horizons=[1, 3], freq="1D", issue_days=["friday"], target="price")
    )
    plain = ModelPipeline().add_model(model_of(kind, refit)).run(frame, save_dir=str(tmp_path / "plain"), **run)
    pooled = ModelPipeline().add_model(PooledModel(model_of(kind, refit)))
    panel = pooled.run({"PL": frame}, save_dir=str(tmp_path / "pooled"), **run)

    assert isinstance(panel, PanelReport) and list(panel.zones) == ["PL"]
    assert lines(tmp_path / "plain" / "m.jsonl") == lines(tmp_path / "pooled" / "PL" / "m.jsonl")
    assert len(lines(tmp_path / "plain" / "m.jsonl")) == plain.refs["m"].count
    alone, zoned = (
        manifest(tmp_path / "plain" / "m.manifest.json"),
        manifest(tmp_path / "pooled" / "PL" / "m.manifest.json"),
    )
    assert zoned.pop("pooled") == {
        "zones": ["PL"],
        "over": ["zone"],
        "zone_feature": "onehot",
        "missing": "raise",
        "data": {},
    }
    assert alone == zoned


def test_a_pooled_run_in_processes_is_the_run_in_this_process(tmp_path, monkeypatch):
    frames = {"PL": daily("PL"), "FI": daily("FI", seed=1, scale=3.0)}
    run = dict(test=FRIDAYS, horizons=[1, 2], freq="1D", issue_days=["friday"], target="price")
    model = PooledModel(OLSModel(predictors=PREDICTORS, training_window=40, name="m"))
    ModelPipeline().add_model(model).run(frames, save_dir=str(tmp_path / "here"), **run)
    monkeypatch.setattr("epftoolbox2.models._pooled.ProcessPoolExecutor", ProcessPoolExecutor)
    monkeypatch.setenv("MAX_PROCESSES", "2")
    ModelPipeline().add_model(model).run(frames, save_dir=str(tmp_path / "pool"), **run)
    for zone in frames:
        assert lines(tmp_path / "here" / zone / "m.jsonl") == lines(tmp_path / "pool" / zone / "m.jsonl")


# -- a plain model in a zone run is its run on the zone alone ----------------------------------------------------------------


def pipeline(issue_timezone: str) -> ModelPipeline:
    return (
        ModelPipeline(information_set=InformationSet("12:00", issue_timezone, rules=RULES), availability="warn")
        .add_model(OLSModel(predictors=[*PREDICTORS, "price_d-1", "price_next"], training_window=30, name="ols"))
        .add_model(RidgeModel(predictors=PREDICTORS, training_window=40, refit="7D", name="ridge"))
        .add_ensemble(AverageEnsemble(members=["ols", "ridge"], name="avg"))
        .add_evaluator(MAEEvaluator())
    )


def test_a_plain_model_in_a_zone_run_is_its_run_on_the_zone_alone(tmp_path):
    frames = {zone: hourly(zone, seed=i) for i, zone in enumerate(ZONES)}
    with pytest.warns(UserWarning, match="price_next"):
        panel = pipeline("data").run(frames, test=HOURLY_TEST, horizon=2, save_dir=str(tmp_path / "zones"))
    # Late on another set of periods in each zone: its values are delivered on Brussels days.
    periods = {zone: list(panel.zones[zone].availability()["periods"]) for zone in ZONES}
    assert len({str(found) for found in periods.values()}) == len(ZONES)
    for zone, frame in frames.items():
        alone = pipeline(ZONES[zone]).run(frame, test=HOURLY_TEST, horizon=2, save_dir=str(tmp_path / zone))
        for name in ("ols", "ridge", "avg"):
            assert lines(tmp_path / "zones" / zone / f"{name}.jsonl") == lines(tmp_path / zone / f"{name}.jsonl")
            assert manifest(tmp_path / "zones" / zone / f"{name}.manifest.json") == manifest(
                tmp_path / zone / f"{name}.manifest.json"
            )
        pd.testing.assert_frame_equal(panel.zones[zone].summary(), alone.summary())
        pd.testing.assert_frame_equal(panel.zones[zone].forecast_counts(), alone.forecast_counts())
        pd.testing.assert_frame_equal(panel.zones[zone].availability(), alone.availability())
    # The data's zone issues each market on its own clock: what the manifest records is that zone.
    recorded = manifest(tmp_path / "zones" / "PT" / "ols.manifest.json")["information_set"]
    assert recorded["issue_timezone"] == "Europe/Lisbon"

    summary = panel.summary()
    assert list(summary.columns[:3]) == ["zone", "target", "model"] and list(summary["zone"].unique()) == list(ZONES)
    report = pipeline("data").report(str(tmp_path / "zones"), test=HOURLY_TEST, horizon=2)
    assert isinstance(report, PanelReport) and sorted(report.zones) == sorted(ZONES)
    pd.testing.assert_frame_equal(report.zones["FI"].summary(), panel.zones["FI"].summary())


def test_a_zone_run_with_the_engine_is_the_run_without_it(tmp_path):
    frames = {zone: hourly(zone, seed=i) for i, zone in enumerate(ZONES)}
    for engine in ("off", "on"):
        pipeline = (
            ModelPipeline(engine=engine)
            .add_model(OLSModel(predictors=PREDICTORS, training_window=30, name="ols"))
            .add_model(RidgeModel(predictors=PREDICTORS, training_window=40, refit="7D", name="ridge"))
            .add_model(PooledModel(OLSModel(predictors=PREDICTORS, training_window=30), name="pooled"))
        )
        pipeline.run(frames, test=HOURLY_TEST, horizon=2, save_dir=str(tmp_path / engine))
    for zone in ZONES:
        for name in ("ols", "ridge", "pooled"):
            assert lines(tmp_path / "off" / zone / f"{name}.jsonl") == lines(tmp_path / "on" / zone / f"{name}.jsonl")


def test_issue_timezone_data_on_one_frame_is_its_zone_written_out(tmp_path):
    frame = hourly("FI")
    by_data = pipeline("data").run(frame, test=HOURLY_TEST, horizon=2, save_dir=str(tmp_path / "data"))
    named = pipeline("Europe/Helsinki").run(frame, test=HOURLY_TEST, horizon=2, save_dir=str(tmp_path / "named"))
    for name in ("ols", "ridge", "avg"):
        assert lines(tmp_path / "data" / f"{name}.jsonl") == lines(tmp_path / "named" / f"{name}.jsonl")
        assert manifest(tmp_path / "data" / f"{name}.manifest.json") == manifest(
            tmp_path / "named" / f"{name}.manifest.json"
        )
    pd.testing.assert_frame_equal(by_data.availability(), named.availability())
    checked = [
        pipeline(zone).verify(frame, *HOURLY_TEST, horizon=2, days=2, models=["ols"]).table()
        for zone in ("data", "Europe/Helsinki")
    ]
    pd.testing.assert_frame_equal(*checked)
    shown = ModelPipeline(issue_time="12:00", issue_timezone="data").add_model(
        PooledModel(OLSModel(predictors=PREDICTORS, training_window=30), name="pooled")
    )
    assert list(shown.explain(frame, model="pooled", horizon=2, echo=False)["predictors"]["predictor"]) == [
        name.replace("{horizon}", "2") for name in PREDICTORS
    ]
    with pytest.raises(ValueError, match="has none"):
        InformationSet("12:00", "data").in_data_zone(frame.index.tz_localize(None))


# -- the hand fit ------------------------------------------------------------------------------------------------------------


def hand_member(frame: pd.DataFrame, day: pd.Timestamp, window: int, horizon: int, columns: list):
    """One zone's part of a pooled fit by hand: rows dated D - W - h .. D - 1 - h by local date, the target of each row's date
    + h, the non-binary columns and the target standardised on those rows (sample standard deviation), the issue row of D."""
    dates = frame.index.tz_localize(None).normalize()
    target = pd.Series(frame["price"].to_numpy(), index=dates)
    rows = (dates >= day - pd.Timedelta(days=window + horizon)) & (dates <= day - pd.Timedelta(days=1 + horizon))
    x = frame.loc[rows, columns].to_numpy(dtype=float)
    y = target.reindex(dates[rows] + pd.Timedelta(days=horizon)).to_numpy(dtype=float)
    issue = frame.loc[dates == day, columns].to_numpy(dtype=float)[-1:]
    keep = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x, y = x[keep], y[keep]
    binary = np.array([set(np.unique(x[:, j])) <= {0.0, 1.0} for j in range(x.shape[1])])
    mean, std = x.mean(axis=0), x.std(axis=0, ddof=1)
    mean[binary], std[binary] = 0.0, 1.0
    return (x - mean) / std, (y - y.mean()) / y.std(ddof=1), (issue - mean) / std, (y.mean(), y.std(ddof=1))


@pytest.mark.parametrize(
    "over, zone_feature",
    [(("zone",), "onehot"), (("zone", "horizon"), "onehot"), (("zone",), "code"), (("zone",), None)],
)
def test_pooled_ols_over_three_zones_is_the_hand_fit(tmp_path, over, zone_feature):
    # Other first dates, so a day id is another date in each frame, and other scales, so each zone's scaler differs.
    frames = {
        "PL": daily("PL"),
        "FI": daily("FI", start="2023-01-09", seed=1, scale=4.0),
        "PT": daily("PT", seed=2, scale=0.3),
    }
    window, horizons = 40, [1, 3]
    model = PooledModel(
        OLSModel(predictors=PREDICTORS, training_window=window, name="m"), over=over, zone_feature=zone_feature
    )
    panel = (
        ModelPipeline()
        .add_model(model)
        .run(frames, test=FRIDAYS, horizons=horizons, freq="1D", issue_days=["friday"], save_dir=str(tmp_path))
    )
    zones = list(frames)
    checked = 0
    for day in pd.date_range(*FRIDAYS, freq="W-FRI"):
        units = [[hz] for hz in horizons] if over == ("zone",) else [horizons]
        for unit in units:
            parts = {}
            for z, zone in enumerate(zones):
                for hz in unit:
                    columns = [name.replace("{horizon}", str(hz)) for name in PREDICTORS]
                    x, y, issue, params = hand_member(frames[zone], day, window, hz, columns)
                    extra = {"onehot": np.eye(len(zones))[z], "code": np.array([float(z)]), None: np.empty(0)}[
                        zone_feature
                    ]
                    if len(unit) > 1:
                        extra = np.concatenate([extra, np.eye(len(unit))[unit.index(hz)]])
                    parts[(zone, hz)] = (
                        np.hstack([x, np.tile(extra, (len(x), 1))]),
                        y,
                        np.hstack([issue, extra[None]]),
                        params,
                    )
            design = np.vstack([part[0] for part in parts.values()])
            response = np.concatenate([part[1] for part in parts.values()])
            beta = np.linalg.lstsq(design, response, rcond=None)[0]
            for (zone, hz), (_, _, issue, (mean, std)) in parts.items():
                expected = (issue @ beta).item() * std + mean
                got = [
                    r["prediction"]
                    for r in panel.zones[zone].refs["m"].iter_records()
                    if r["run_date"] == day.date().isoformat() and r["horizon"] == hz
                ]
                assert got == pytest.approx([expected], abs=1e-9, rel=0)
                checked += 1
    assert checked == len(zones) * len(horizons) * len(pd.date_range(*FRIDAYS, freq="W-FRI"))


# -- three clocks ------------------------------------------------------------------------------------------------------------


def three_clocks(delay: str, availability: str, refit=None) -> ModelPipeline:
    rules = {**RULES, "actual_daily_max": Availability.realised(delay)}
    return ModelPipeline(
        information_set=InformationSet("12:00", "data", rules=rules), availability=availability
    ).add_model(PooledModel(OLSModel(predictors=PREDICTORS, training_window=40, refit=refit, name="m")))


def run_three_clocks(pipeline: ModelPipeline, tmp_path, name: str = "run"):
    frames = {zone: daily(zone, seed=i) for i, zone in enumerate(ZONES)}
    return pipeline.run(
        frames,
        test=FRIDAYS,
        horizons=[1, 2],
        freq="1D",
        issue_days=["friday"],
        target="actual_daily_max",
        save_dir=str(tmp_path / name),
    )


def test_a_shared_fit_judges_every_zone_s_training_rows_at_the_earliest_issue(tmp_path):
    # D-1's maximum is known 12h after D's midnight: at noon of D on each zone's own clock, so each zone alone may train on it.
    # One fit serves Warsaw, Helsinki and Lisbon, and it is made by Helsinki's noon, 11:00 in Warsaw and 10:00 in Lisbon.
    with pytest.raises(InformationSetError, match="zone 'PL' with zone 'FI'"):
        run_three_clocks(three_clocks("12h", "raise"), tmp_path, "raise")
    with pytest.warns(UserWarning, match="actual_daily_max"):
        panel = run_three_clocks(three_clocks("12h", "warn"), tmp_path, "warn")
    for zone, late in (("PL", True), ("FI", False), ("PT", True)):
        listed = panel.zones[zone].availability()
        rows = listed[listed["predictor"] == "actual_daily_max"]
        assert (len(rows) == 1) == late, zone
        if late:
            assert rows["role"].iloc[0] == "training target" and "pooled with zone 'FI'" in rows["note"].iloc[0]
            manifest_rows = manifest(tmp_path / "warn" / zone / "m.manifest.json")["availability"]["listing"]
            assert [row["predictor"] for row in manifest_rows] == ["actual_daily_max"]
    # Known 11:00 local: noon in Helsinki for Warsaw, 13:00 there for Lisbon, which alone is late.
    with pytest.raises(InformationSetError, match="zone 'PT' with zone 'FI'"):
        run_three_clocks(three_clocks("11h", "raise"), tmp_path, "eleven")
    run_three_clocks(three_clocks("10h", "raise"), tmp_path, "ten")
    # With refit a unit's rows are cut as of the issue time on its anchor: judged there, on the earlier clock too.
    with pytest.raises(InformationSetError, match="zone 'PL' with zone 'FI'"):
        run_three_clocks(three_clocks("12h", "raise", refit="7D"), tmp_path, "refit")
    run_three_clocks(three_clocks("10h", "raise", refit="7D"), tmp_path, "refit_ten")
    # Each zone alone issues on its own clock and may train on D-1's maximum at noon.
    for zone in ZONES:
        ModelPipeline(
            information_set=InformationSet(
                "12:00", "data", rules={**RULES, "actual_daily_max": Availability.realised("12h")}
            ),
            availability="raise",
        ).add_model(OLSModel(predictors=PREDICTORS, training_window=40, name="m")).run(
            daily(zone), test=FRIDAYS, horizons=[1, 2], freq="1D", issue_days=["friday"], target="actual_daily_max"
        )


def test_zones_on_one_issue_clock_judge_nothing_more(tmp_path):
    pipeline = ModelPipeline(
        information_set=InformationSet(
            "12:00", "Europe/Brussels", rules={**RULES, "actual_daily_max": Availability.realised("12h")}
        ),
        availability="warn",
    ).add_model(PooledModel(OLSModel(predictors=PREDICTORS, training_window=40, name="m")))
    panel = run_three_clocks(pipeline, tmp_path)
    for zone in ZONES:
        assert "pooled with" not in " ".join(panel.zones[zone].availability()["note"])


# -- resume ------------------------------------------------------------------------------------------------------------------


def run_pooled(tmp_path, frames, **settings):
    model = PooledModel(OLSModel(predictors=PREDICTORS, training_window=40, name="m"), **settings)
    return (
        ModelPipeline()
        .add_model(model)
        .run(frames, test=FRIDAYS, horizons=[1, 2], freq="1D", issue_days=["friday"], save_dir=str(tmp_path))
    )


def test_a_resume_refuses_what_moves_a_pooled_forecast(tmp_path):
    frames = {zone: daily(zone, seed=i) for i, zone in enumerate(ZONES)}
    two = {zone: frames[zone] for zone in ("PL", "FI")}
    run_pooled(tmp_path, two)
    run_pooled(tmp_path, two)  # the same run resumes
    with pytest.raises(ManifestMismatchError, match="pooled"):
        run_pooled(tmp_path, frames)  # another zone set
    with pytest.raises(ManifestMismatchError, match="pooled"):
        run_pooled(tmp_path, {"FI": two["FI"], "PL": two["PL"]})  # another order, another stack
    with pytest.raises(ManifestMismatchError, match="pooled"):
        run_pooled(tmp_path, two, over=("zone", "horizon"))
    with pytest.raises(ManifestMismatchError, match="pooled"):
        run_pooled(tmp_path, two, zone_feature="code")
    moved = dict(two, FI=two["FI"].copy())
    moved["FI"].iloc[20, moved["FI"].columns.get_loc("load_d+1")] += 1.0  # a row before the test, in the other zone
    with pytest.raises(ManifestMismatchError, match="pooled"):
        run_pooled(tmp_path, moved)


def test_a_pooled_resume_cut_short_is_a_fresh_run(tmp_path):
    frames = {zone: daily(zone, seed=i) for i, zone in enumerate(ZONES)}
    run_pooled(tmp_path / "fresh", frames)
    run_pooled(tmp_path / "cut", frames)
    path = tmp_path / "cut" / "FI" / "m.jsonl"
    kept = path.read_text().splitlines()
    path.write_text("\n".join(kept[::3]) + "\n")  # FI lost two forecasts in three; PL and PT have all of theirs
    run_pooled(tmp_path / "cut", frames)
    for zone in frames:
        assert lines(tmp_path / "cut" / zone / "m.jsonl") == lines(tmp_path / "fresh" / zone / "m.jsonl")


# -- missing columns, refusals, YAML -----------------------------------------------------------------------------------------


def test_a_predictor_a_zone_lacks_raises_or_is_zero(tmp_path):
    frames = {"PL": daily("PL"), "FI": daily("FI", seed=1)}
    lacking = dict(frames, FI=frames["FI"].drop(columns=["wind_d+1", "wind_d+2"]))
    with pytest.raises(ValueError, match="zone 'FI' has no column"):
        run_pooled(tmp_path / "raise", lacking)
    run_pooled(tmp_path / "zero", lacking, missing="zero")
    zeros = dict(frames, FI=frames["FI"].assign(**{"wind_d+1": 0.0, "wind_d+2": 0.0}))
    run_pooled(tmp_path / "zeros", zeros, missing="zero")
    for zone in frames:
        assert lines(tmp_path / "zero" / zone / "m.jsonl") == lines(tmp_path / "zeros" / zone / "m.jsonl")


def test_what_a_pooled_model_refuses(tmp_path):
    ols = OLSModel(predictors=PREDICTORS, training_window=40, name="m")
    with pytest.raises(ValueError, match="fits nothing"):
        PooledModel(NaiveModel(predictors=["load_d+{horizon}"]))
    with pytest.raises(ValueError, match="newest_target"):
        PooledModel(OLSModel(predictors=PREDICTORS, newest_target="published"))
    with pytest.raises(ValueError, match="list of training windows"):
        PooledModel(OLSModel(predictors=PREDICTORS, training_window=[20, 40]))
    with pytest.raises(ValueError, match="predictor function"):
        PooledModel(OLSModel(predictors=lambda ctx: Use(PREDICTORS)))
    with pytest.raises(TypeError):
        PooledModel("ols")
    for over in (("horizon",), ("zone", "hour"), ()):
        with pytest.raises(ValueError, match="over"):
            PooledModel(ols, over=over)
    with pytest.raises(ValueError, match="zone_feature"):
        PooledModel(ols, zone_feature="embedding")
    with pytest.raises(ValueError, match="missing"):
        PooledModel(ols, missing="mean")

    frames = {"PL": daily("PL"), "FI": daily("FI", seed=1)}
    pooled = ModelPipeline().add_model(PooledModel(ols))
    run = dict(test=FRIDAYS, horizons=[1, 2], freq="1D", issue_days=["friday"])
    with pytest.raises(ValueError, match="pool zones"):
        pooled.run(frames["PL"], **run)
    with pytest.raises(ValueError, match="test window alone"):
        pooled.run(frames, validation=("2023-02-03", "2023-02-24"), **run)
    with pytest.raises(ValueError, match="exporter"):
        ModelPipeline().add_model(ols).add_exporter(TerminalExporter()).run(frames, **run)
    with pytest.raises(ValueError, match="letters, digits"):
        ModelPipeline().add_model(ols).run({"P L": frames["PL"]}, **run)
    with pytest.raises(ValueError, match="whatever their case"):
        ModelPipeline().add_model(ols).run({"PL": frames["PL"], "pl": frames["FI"]}, **run)
    uneven = OLSModel(predictors=lambda horizon: PREDICTORS[: 1 + horizon], training_window=40, name="m")
    with pytest.raises(ValueError, match="same length"):
        ModelPipeline().add_model(PooledModel(uneven, over=("zone", "horizon"))).run(frames, **run)
    ModelPipeline().add_model(ols).run(frames["PL"], save_dir=str(tmp_path), **run)  # a run of one frame there first
    with pytest.raises(ValueError, match="one frame"):
        ModelPipeline().add_model(ols).run(frames, save_dir=str(tmp_path), **run)


def test_a_pooled_model_saves_and_loads(tmp_path):
    pipeline = ModelPipeline(issue_time="12:00", issue_timezone="data").add_model(
        PooledModel(
            RidgeModel(predictors=PREDICTORS, training_window=40, refit="7D", name="ridge"),
            over=("zone", "horizon"),
            zone_feature="code",
            missing="zero",
            name="pooled_ridge",
        )
    )
    path = tmp_path / "pipeline.yaml"
    pipeline.save(path)
    loaded = ModelPipeline.load(path)
    assert loaded.to_dict() == pipeline.to_dict()
    model = loaded.models[0]
    assert isinstance(model, PooledModel) and isinstance(model.model, RidgeModel)
    assert (model.over, model.zone_feature, model.missing, model.name) == (
        ("zone", "horizon"),
        "code",
        "zero",
        "pooled_ridge",
    )
    assert model.model.refit == "7D" and loaded.information_set.issue_timezone == "data"


# -- verify ------------------------------------------------------------------------------------------------------------------


def verify_pooled_model(predictors, rules, frames, missing="raise"):
    model = PooledModel(OLSModel(predictors=predictors, training_window=40, name="m"), missing=missing)
    pipeline = ModelPipeline(information_set=InformationSet("12:00", "data", rules=rules)).add_model(model)
    return pipeline.verify(frames, *FRIDAYS, target="actual_daily_max", horizon=2, freq="1D", days=3, seed=1)


def test_verify_of_a_pooled_model_is_ok_with_its_control():
    frames = {zone: daily(zone, seed=i) for i, zone in enumerate(ZONES)}
    report = verify_pooled_model(PREDICTORS, {**RULES, "actual_daily_max": Availability.realised("10h")}, frames)
    assert report.models == [f"m ({zone})" for zone in ZONES]
    assert report.ok, report.problems()
    # Each zone's own control, which moves its own forecasts.
    assert all(check.control.moved for check in report.checks)
    by_day = {}
    for check in report.checks:
        by_day.setdefault(check.run_day, []).append(check.control)
    assert all(len({id(control) for control in controls}) == len(ZONES) for controls in by_day.values())


def test_a_zone_whose_own_control_moves_nothing_is_inconclusive():
    # Finland has none of the predictors (zeros by missing="zero"): its forecasts read nothing of its own frame, as alone.
    frames = {zone: daily(zone, seed=i) for i, zone in enumerate(("PL", "FI"))}
    frames["FI"] = frames["FI"].drop(columns=[c for c in frames["FI"].columns if c.startswith(("load_d+", "wind_d+"))])
    rules = {**RULES, "actual_daily_max": Availability.realised("10h")}
    pooled = PooledModel(OLSModel(predictors=PREDICTORS[:2], training_window=40, name="m"), missing="zero")
    pipeline = ModelPipeline(information_set=InformationSet("12:00", "data", rules=rules, unknown="ignore")).add_model(
        pooled
    )
    report = pipeline.verify(frames, *FRIDAYS, target="actual_daily_max", horizon=2, freq="1D", days=2, seed=1)
    assert {model: report.status(model) for model in report.models} == {"m (PL)": "ok", "m (FI)": "inconclusive"}
    alone = pipeline.verify(
        {"FI": frames["FI"]}, *FRIDAYS, target="actual_daily_max", horizon=2, freq="1D", days=2, seed=1
    )
    assert alone.status("m (FI)") == "inconclusive"


def test_a_leak_in_one_zone_s_frame_is_a_leak_for_every_zone():
    # Known 60 hours after its day ends: a training row of D-2 holds a value no forecast issued on D has. Only Finland's frame has
    # it (the others a column of zeros, which the probe leaves alone), and its rows move the fit every zone shares.
    frames = {zone: daily(zone, seed=i) for i, zone in enumerate(ZONES)}
    frames["FI"] = frames["FI"].assign(late=frames["FI"]["actual_daily_max_d-1"].shift(-1))
    rules = {**RULES, "actual_daily_max": Availability.realised("10h"), "late": Availability.realised("60h")}
    report = verify_pooled_model([*PREDICTORS, "late"], rules, {**frames, "PL": frames["PL"]}, missing="zero")
    assert {model: report.status(model) for model in report.models} == {f"m ({zone})": "leak" for zone in ZONES}


def test_verify_judges_the_shared_fit_at_the_earliest_issue():
    # As the check before a run: D-1's maximum known at noon in each zone is unknown at Helsinki's noon in Warsaw and Lisbon.
    frames = {zone: daily(zone, seed=i) for i, zone in enumerate(ZONES)}
    late = verify_pooled_model(PREDICTORS, {**RULES, "actual_daily_max": Availability.realised("12h")}, frames)
    assert {model: late.status(model) for model in late.models} == {f"m ({zone})": "leak" for zone in ZONES}
    for zone in ZONES:  # each zone alone is ok: its own clock
        alone = verify_pooled_model(
            PREDICTORS, {**RULES, "actual_daily_max": Availability.realised("12h")}, {zone: frames[zone]}
        )
        assert alone.ok, alone.problems()


def test_pooled_horizons_need_one_length_per_period_before_any_fit(tmp_path):
    def names(horizon, hour):  # one more input per horizon at hour 0, all three at every other hour
        full = [name.replace("{horizon}", str(horizon)) for name in PREDICTORS]
        return full[: 1 + horizon] if hour == 0 else full

    model = PooledModel(OLSModel(predictors=names, training_window=20, name="m"), over=("zone", "horizon"))
    frames = {"PL": hourly("PL"), "FI": hourly("FI", seed=1)}
    with pytest.raises(ValueError, match="period 0"):
        ModelPipeline().add_model(model).run(
            frames, test=("2023-03-24", "2023-03-25"), horizon=2, save_dir=str(tmp_path)
        )
    assert not any(tmp_path.rglob("*.jsonl"))  # refused before any record


def test_every_zone_is_checked_before_any_zone_s_files_are_written(tmp_path):
    def pipeline():
        return (
            ModelPipeline()
            .add_model(NaiveModel(predictors=["load_d+{horizon}"], name="tso", known_only=False))
            .add_model(PooledModel(OLSModel(predictors=PREDICTORS, training_window=40, name="m")))
        )

    run = dict(test=FRIDAYS, horizons=[1, 2], freq="1D", issue_days=["friday"], save_dir=str(tmp_path))
    short = {"PL": daily("PL"), "FI": with_daily_columns(hourly("FI", seed=1).iloc[: 100 * 24]).at_time("12:00")}
    with pytest.raises(ValueError, match="zone 'FI': 'test_end' date '2023-04-28' is not in the data"):
        pipeline().run(short, **run)
    assert not any(tmp_path.rglob("*.json*"))  # no manifest and no record, PL's plain model's included
    pipeline().run({"PL": daily("PL"), "FI": daily("FI", seed=1)}, **run)  # FI's data fetched through the test: it runs

    mixed = {"PL": daily("PL"), "FI": hourly("FI", seed=1)}
    with pytest.raises(ValueError, match="zone 'FI': Data rows are most commonly 1 h apart"):
        pipeline().run(mixed, test=FRIDAYS, horizons=[1, 2], freq="1D", issue_days=["friday"])


def test_a_refused_resume_writes_no_manifest_in_a_new_zone(tmp_path):
    def pipeline():
        return (
            ModelPipeline()
            .add_model(NaiveModel(predictors=["load_d+{horizon}"], name="tso", known_only=False))
            .add_model(PooledModel(OLSModel(predictors=PREDICTORS, training_window=40, name="m")))
        )

    run = dict(test=FRIDAYS, horizons=[1, 2], freq="1D", issue_days=["friday"])
    pipeline().run({"PL": daily("PL")}, save_dir=str(tmp_path), **run)
    index = (tmp_path / "zone_run.json").read_text()
    on_file = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    with pytest.raises(ManifestMismatchError, match="zone 'PL'"):  # FI first in the stack, PL on file
        pipeline().run({"FI": daily("FI", seed=1), "PL": daily("PL")}, save_dir=str(tmp_path), **run)
    assert (tmp_path / "zone_run.json").read_text() == index  # the index unchanged
    assert not (tmp_path / "FI").exists()  # no folder for the new zone, its plain model's included
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.name != "tso.lock") == [
        path for path in on_file if path.name != "tso.lock"
    ]
    assert list(pipeline().report(tmp_path, **run).zones) == ["PL"]  # the run on file still reports


def test_a_zone_differing_only_in_case_from_one_on_file_is_refused(tmp_path):
    run_pooled(tmp_path, {"FI": daily("FI")})
    with pytest.raises(ValueError, match=r"'fi' \(the index has 'FI'\)"):
        run_pooled(tmp_path, {"fi": daily("FI"), "PL": daily("PL")})
    assert sorted(path.name for path in tmp_path.iterdir()) == ["FI", "zone_run.json"]


def test_folders_of_runs_of_one_frame_are_never_zones(tmp_path):
    def pipeline():
        return ModelPipeline().add_model(OLSModel(predictors=PREDICTORS, training_window=40, name="m"))

    run = dict(test=FRIDAYS, horizons=[1, 2], freq="1D", issue_days=["friday"])
    pipeline().run(daily("PL"), save_dir=str(tmp_path / "runs"), **run)
    pipeline().run(daily("PL", seed=3), save_dir=str(tmp_path / "runs" / "old"), **run)  # an older run kept inside
    report = pipeline().report(tmp_path / "runs", **run)
    assert not isinstance(report, PanelReport)
    assert report.forecast_counts()["produced"].tolist() == [len(lines(tmp_path / "runs" / "m.jsonl"))]
    pipeline().run(daily("PL"), save_dir=str(tmp_path / "exp" / "v2"), **run)
    with pytest.raises(FileNotFoundError, match="No results"):  # a parent of runs is no run, as before
        pipeline().report(tmp_path / "exp", **run)


def test_a_report_refuses_a_zone_of_the_run_whose_folder_is_gone(tmp_path):
    import shutil

    frames = {zone: daily(zone, seed=i) for i, zone in enumerate(ZONES)}
    pipeline = (
        ModelPipeline()
        .add_model(NaiveModel(predictors=["load_d+{horizon}"], name="tso", known_only=False))
        .add_model(PooledModel(OLSModel(predictors=PREDICTORS, training_window=40, name="m")))
    )
    run = dict(test=FRIDAYS, horizons=[1, 2], freq="1D", issue_days=["friday"])
    pipeline.run(frames, save_dir=str(tmp_path), **run)
    assert json.loads((tmp_path / "zone_run.json").read_text())["zones"] == list(ZONES)
    assert list(pipeline.report(tmp_path, **run).zones) == list(ZONES)  # in the run's order
    shutil.rmtree(tmp_path / "PT")
    with pytest.raises(FileNotFoundError, match=r"\['PT'\] hold none"):
        pipeline.report(tmp_path, **run)
    with pytest.warns(UserWarning, match="PT"):
        assert list(pipeline.report(tmp_path, allow_missing=True, **run).zones) == ["PL", "FI"]
    one_frame = ModelPipeline().add_model(OLSModel(predictors=PREDICTORS, training_window=40, name="x"))
    with pytest.raises(
        ValueError, match="holds a run over zones"
    ):  # a run of one frame there is refused before any fit
        one_frame.run(frames["PL"], save_dir=str(tmp_path), **run)


def test_epf_report_of_a_run_over_zones(tmp_path, capsys):
    import yaml

    from epftoolbox2 import cli

    frames = {zone: daily(zone, seed=i) for i, zone in enumerate(("PL", "FI"))}
    pipeline = (
        ModelPipeline()
        .add_model(NaiveModel(predictors=["load_d+{horizon}"], name="tso", known_only=False))
        .add_model(PooledModel(OLSModel(predictors=PREDICTORS, training_window=40, name="m")))
        .add_evaluator(MAEEvaluator())
    )
    pipeline.run(
        frames, test=FRIDAYS, horizons=[1, 2], freq="1D", issue_days=["friday"], save_dir=str(tmp_path / "results")
    )
    pipeline.save(tmp_path / "model_pipeline.yaml")
    (tmp_path / "prices.csv").write_text("datetime,price\n2023-01-01 00:00:00,1.0\n")
    config = {
        "data_pipeline": {  # never run: the report reads the files
            "start": "2023-01-01",
            "end": "2023-05-01",
            "sources": [{"class": "CsvSource", "params": {"file_path": str(tmp_path / "prices.csv")}}],
        },
        "model_pipeline": {
            "path": "model_pipeline.yaml",
            "test_start": FRIDAYS[0],
            "test_end": FRIDAYS[1],
            "horizons": [1, 2],
            "issue_days": ["friday"],
            "freq": "1D",
            "save_dir": "results",
        },
    }
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(config))
    capsys.readouterr()
    cli.cmd_report(cli.build_parser().parse_args(["report", str(tmp_path / "experiment.yaml")]))
    shown = capsys.readouterr().out
    assert "zone" in shown and all(f"{zone} " in shown for zone in frames)


def test_a_zone_holds_its_targets_alone_once_its_arrays_are_built(tmp_path, monkeypatch):
    """During a pooled pool each zone's period frames hold the target columns its skip records read, not the predictors the
    arrays already hold: the records are those of the whole frames (the one-zone equality above), with half the memory."""
    held = []
    collect = OLSModel._collect

    def watched(self, state, results, skipped, sink):
        held.append(sorted({column for frame in self._hour_data.values() for column in frame.columns}))
        return collect(self, state, results, skipped, sink)

    monkeypatch.setattr(OLSModel, "_collect", watched)
    frames = {"PL": hourly("PL"), "FI": hourly("FI", seed=1)}
    issue_day = frames["FI"].index.tz_localize(None).normalize() == pd.Timestamp("2023-03-27")
    frames["FI"].loc[issue_day, "load_d+1"] = np.nan  # skipped in the worker: its skip record is built in the parent
    ModelPipeline().add_model(PooledModel(OLSModel(predictors=PREDICTORS, training_window=30, name="m"))).run(
        frames, test=HOURLY_TEST, horizon=2, save_dir=str(tmp_path)
    )
    assert held and all(columns == ["price_d+1", "price_d+2"] for columns in held)
    skips = [r for r in map(json.loads, lines(tmp_path / "FI" / "m.jsonl")) if r["status"] != "ok"]
    assert len(skips) == 24 and all(r["status"] == "missing_test_predictors" for r in skips)
    assert sorted(r["actual"] for r in skips) == sorted(calendar_shift(frames["FI"]["price"], 1)[issue_day].tolist())


def test_a_zone_that_makes_no_forecast_is_named(tmp_path):
    frames = {"PL": daily("PL"), "FI": daily("FI", seed=1).assign(price=np.nan)}  # FI has no target to train on
    pooled = ModelPipeline().add_model(PooledModel(OLSModel(predictors=PREDICTORS, training_window=40, name="m")))
    run = dict(test=FRIDAYS, horizons=[1, 2], freq="1D", issue_days=["friday"])
    with pytest.raises(RuntimeError, match="zone 'FI': m: no forecasts produced"):
        pooled.run(frames, **run)
    plain = ModelPipeline().add_model(OLSModel(predictors=PREDICTORS, training_window=40, name="m"))
    with pytest.raises(RuntimeError, match="zone 'FI': m: no forecasts produced"):
        plain.run(frames, **run)
