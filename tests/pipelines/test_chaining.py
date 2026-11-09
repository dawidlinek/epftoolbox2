"""Two pipelines chained through the data: an upstream's stored forecasts read back by `ForecastSource` as a downstream input.

The upstream forecasts `load` with OLS at 09:30 Warsaw and stores its records; the downstream forecasts `price` at 11:45 with
`load_fc_d+1`, the upstream's forecast for D+1 issued on D. What is pinned: every training row of a whole downstream run holds
exactly the stored record of its date (instrumented in `_fit_member`), the issue-time check and `verify` see a chained column
as known exactly from the upstream's issue time, the coverage of the rows the fits read is counted and warned about at every
level, an upstream that read something late taints its column downstream, and a resume refuses another upstream identity.
"""

import json
import shutil
import warnings
from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.sources import ForecastSource
from epftoolbox2.data.transformers import LagTransformer
from epftoolbox2.ensembles import AverageEnsemble
from epftoolbox2.information_set import Availability, InformationSet, InformationSetError
from epftoolbox2.models import OLSModel, _worker
from epftoolbox2.models._worker import _worker_init, member_rows
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import ManifestMismatchError
from tests.synthetic import PREDICTORS, make_market_data

ZONE = "Europe/Warsaw"
RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "price": Availability.published("13:00", 1),
}
INFO = InformationSet("11:45", ZONE, rules=RULES)
TEST = ("2023-04-05", "2023-04-14")
UPSTREAM = ["load", "is_weekend_d+{horizon}", "is_weekday_d+{horizon}"]
UP_TEST = ("2023-02-10", "2023-04-20")  # covers every row a 45-day downstream window reads for TEST
WIDE = (pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))
LATE_WIND = {"wind": Availability.published("10:00", 0)}  # an upstream input out at 10:00 on its own day


class InProcessPool:
    """Stands in for the worker pool: in this process, so `_fit_member` can be watched, and fast."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        _worker_init(*initargs)

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
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)


@pytest.fixture(scope="module")
def df():
    return make_market_data(noise=5.0, seed=7)


def run_upstream(
    df,
    directory,
    test=UP_TEST,
    issue_time="09:30",
    zone=ZONE,
    availability="warn",
    predictors=UPSTREAM,
    window=30,
    name="load_up",
    horizon=2,
    rules=None,
    **kwargs,
):
    info = InformationSet(issue_time, zone, rules={**RULES, **(rules or {})})
    pipeline = ModelPipeline(information_set=info, availability=availability)
    pipeline.add_model(OLSModel(predictors, training_window=window, name=name))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # a late input's listing, where a test plants one
        frame = df.drop(columns=[name for name in df.columns if name.startswith("load_d+")])  # its targets
        pipeline.run(frame, *test, target="load", horizon=horizon, save_dir=str(directory), **kwargs)
    return pipeline


def chained(df, *sources):
    """`df` with each source's column at its rows and lagged by local days: `x_d+1`, `x_d+2`, `x_d-1`."""
    data = df.copy()
    for source in sources:
        column = source.fetch(*WIDE)[source.column]
        data[source.column] = column.reindex(df.index.tz_convert("UTC")).to_numpy()
        lagged = LagTransformer([source.column], [-1, -2, 1], "day").transform(data[[source.column]])
        for name in lagged.columns.drop(source.column):
            data[name] = lagged[name]
    return data


def stored(directory, name="load_up", horizon=1):
    """(run_date, hour) -> prediction of the upstream's records at `horizon`."""
    out = {}
    for line in (directory / f"{name}.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["horizon"] == horizon and record.get("status", "ok") == "ok":
            out[(record["run_date"], record["hour"])] = record["prediction"]
    return out


def downstream(predictors, info, availability="raise", name="price_m", window=30, **kwargs):
    pipeline = ModelPipeline(information_set=info, availability=availability)
    return pipeline.add_model(OLSModel(predictors, training_window=window, name=name, **kwargs))


def test_the_prototype_s_chain_runs_with_the_upstream_issue_time(df, tmp_path):
    run_upstream(df, tmp_path / "up")
    source = ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc")
    data = chained(df, source)
    # The forecast for 2023-04-11 12:00 stored by the upstream run is on the row of D = 2023-04-10 as load_fc_d+1.
    assert data.loc["2023-04-10 12:00", "load_fc_d+1"] == stored(tmp_path / "up")[("2023-04-10", 12)]

    info = INFO.with_source_rules([source])
    assert info.resolve("load_fc_d+1").rule.time == "09:30"
    ok = OLSModel(PREDICTORS + ["load_fc_d+1"], training_window=30).run(data, *TEST, horizon=1, information_set=info)
    assert ok.produced == 240
    with pytest.raises(InformationSetError, match="load_fc_d\\+2"):
        OLSModel(PREDICTORS + ["load_fc_d+2"], training_window=30).run(data, *TEST, horizon=1, information_set=info)


def test_every_training_row_holds_the_upstream_record_issued_on_its_date(tmp_path, monkeypatch):
    """A frame over the autumn switch: each training row dated d of every fit holds, in load_fc_d+1, the record issued on d at
    horizon 1 for its period, or nothing where the upstream has no such record."""
    frame = make_market_data(start="2023-09-01", days=80, noise=5.0, seed=3)
    run_upstream(frame, tmp_path / "up", test=("2023-09-20", "2023-11-15"))
    source = ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc")
    data = chained(frame, source)
    records = stored(tmp_path / "up")
    seen = []
    original = _worker._fit_member

    def spy(arrays, cfg, hour, hz, day, member, newest, test, list_id, predictors, *rest, **kwargs):
        if "load_fc_d+1" in predictors:
            rows = member_rows(arrays["days"], day, member, hz, newest)
            at = pd.DatetimeIndex(arrays["timestamps"][rows]).tz_localize("UTC").tz_convert(ZONE)
            seen.append((at, arrays["x"][rows, arrays["col_index"]["load_fc_d+1"]]))
        return original(arrays, cfg, hour, hz, day, member, newest, test, list_id, predictors, *rest, **kwargs)

    monkeypatch.setattr(_worker, "_fit_member", spy)
    model = OLSModel(PREDICTORS + ["load_fc_d+1"], training_window=21)
    ref = model.run(data, "2023-10-20", "2023-11-03", horizon=2, information_set=INFO.with_source_rules([source]))
    assert ref.produced > 0 and seen
    checked = 0
    for at, values in seen:
        wall = at.tz_localize(None)
        for when, value in zip(wall, values):
            key = (when.date().isoformat(), when.hour)
            if np.isnan(value):
                assert key not in records, key
            else:
                assert value == records[key], key
                checked += 1
    assert checked > 10_000
    assert any(w.date().isoformat() == "2023-10-29" for at, _ in seen for w in at.tz_localize(None))  # the autumn day


# -- coverage --------------------------------------------------------------------------------------------------------------


def independent_count(data, name, first_row, last_row):
    """Rows dated first_row..last_row (local dates) where `name` (a `_d+1`) is missing though it could hold a value: its row
    plus one day is on the clock, and isn't 02:00 the day after a spring switch, which no forecast issued on the switch day
    is planned for. What the fits read and can't use, and how many they read."""
    wall = data.index.tz_localize(None)
    dates = wall.normalize()
    inside = (dates >= pd.Timestamp(first_row)) & (dates <= pd.Timestamp(last_row))
    read = wall + pd.Timedelta(days=1)
    on_clock = ~read.tz_localize(ZONE, ambiguous="NaT", nonexistent="NaT").isna() | read.isin(wall)
    switch = pd.Timestamp("2023-03-26")
    unplanned = (read.normalize() == switch + pd.Timedelta(days=1)) & (read.hour == 2)
    missing = inside & data[name].isna().to_numpy() & on_clock & ~unplanned
    return int(missing.sum()), int(inside.sum())


@pytest.mark.parametrize("availability", ["none", "warn"])
def test_an_upstream_starting_mid_window_warns_with_its_counts_and_the_day_to_start_from(df, tmp_path, availability):
    run_upstream(df, tmp_path / "up", test=("2023-03-20", "2023-04-20"))
    source = ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc")
    data = chained(df, source)
    info = INFO.with_source_rules([source])
    pipeline = downstream(PREDICTORS + ["load_fc_d+1"], info, availability=availability)
    with pytest.warns(UserWarning, match="load_fc_d\\+1 has no value on") as caught:
        pipeline.run(data, *TEST, horizon=1, save_dir=str(tmp_path / "down"))
    # A 30-day window and horizon 1 read rows from D-31 to D-2: 2023-03-05 to 2023-04-12. The first forecast is issued on
    # 2023-03-20, for 03-21: the row of 03-20 is the first covered, so the upstream must start on 03-05.
    expected, read = independent_count(data, "load_fc_d+1", "2023-03-05", "2023-04-12")
    message = next(str(w.message) for w in caught if "has no value" in str(w.message))
    assert f"has no value on {expected} of the {read} rows" in message
    assert "from 2023-03-05 to 2023-03-19" in message and "run load_up from 2023-03-05" in message
    manifest = json.loads((tmp_path / "down" / "price_m.manifest.json").read_text())
    coverage = manifest["chained_inputs"]["load_fc"]["coverage"]["load_fc_d+1"]
    assert coverage == {
        "read": read,
        "uncovered": expected,
        "first": "2023-03-05",
        "last": "2023-03-19",
        "run_from": "2023-03-05",
    }
    assert expected == 15 * 24  # 15 whole days before the upstream's first forecast


def test_with_refit_the_rows_read_are_those_of_the_refit_days(df, tmp_path):
    """refit="7D": the forecasts of 04-05 to 04-14 use the fits of Mondays 04-03 and 04-10, whose rows run from 03-03 (04-03 - 31)
    to 04-08: two more days before the upstream's first forecast than without refit."""
    run_upstream(df, tmp_path / "up", test=("2023-03-20", "2023-04-20"))
    source = ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc")
    data = chained(df, source)
    pipeline = downstream(
        PREDICTORS + ["load_fc_d+1"], INFO.with_source_rules([source]), availability="warn", refit="7D"
    )
    with pytest.warns(UserWarning, match="has no value on"):
        pipeline.run(data, *TEST, horizon=1, save_dir=str(tmp_path / "down"))
    found = json.loads((tmp_path / "down" / "price_m.manifest.json").read_text())["chained_inputs"]["load_fc"][
        "coverage"
    ]
    expected, read = independent_count(data, "load_fc_d+1", "2023-03-03", "2023-04-08")
    assert found["load_fc_d+1"] == {
        "read": read,
        "uncovered": expected,
        "first": "2023-03-03",
        "last": "2023-03-19",
        "run_from": "2023-03-03",
    }
    assert expected == 17 * 24


def test_an_upstream_starting_mid_window_is_refused_at_raise(df, tmp_path):
    run_upstream(df, tmp_path / "up", test=("2023-03-20", "2023-04-20"))
    source = ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc")
    data = chained(df, source)
    pipeline = downstream(PREDICTORS + ["load_fc_d+1"], INFO.with_source_rules([source]))
    with pytest.raises(InformationSetError, match="run load_up from 2023-03-05"):
        pipeline.run(data, *TEST, horizon=1)


def test_a_complete_upstream_is_silent_over_a_spring_switch_and_its_manifest_records_it(df, tmp_path):
    run_upstream(df, tmp_path / "up")
    source = ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc")
    data = chained(df, source)
    pipeline = downstream(PREDICTORS + ["load_fc_d+1", "load_fc_d-1"], INFO.with_source_rules([source]), window=45)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*has no value on")
        pipeline.run(data, *TEST, horizon=2, save_dir=str(tmp_path / "down"))
    manifest = json.loads((tmp_path / "down" / "price_m.manifest.json").read_text())
    entry = manifest["chained_inputs"]["load_fc"]
    assert {key: entry[key] for key in ("model", "horizon", "issue_time", "issue_timezone", "data_timezone")} == {
        "model": "load_up",
        "horizon": 1,
        "issue_time": "09:30",
        "issue_timezone": ZONE,
        "data_timezone": ZONE,
    }
    assert entry["available"] is None and len(entry["digest"]) == 64
    assert {name: found["uncovered"] for name, found in entry["coverage"].items()} == {
        "load_fc_d+1": 0,
        "load_fc_d-1": 0,
    }
    # load_fc_d+1 on the row of 2023-03-25 at 02:00 reads a slot the switch day doesn't have: no hole of the upstream.
    assert np.isnan(data.loc["2023-03-25 02:00", "load_fc_d+1"])


def test_a_model_without_a_chained_column_has_no_key_in_its_manifest(df, tmp_path):
    run_upstream(df, tmp_path / "up")
    source = ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc")
    data = chained(df, source)
    downstream(PREDICTORS, INFO.with_source_rules([source])).run(data, *TEST, horizon=1, save_dir=str(tmp_path / "d"))
    assert "chained_inputs" not in json.loads((tmp_path / "d" / "price_m.manifest.json").read_text())


# -- late upstream ---------------------------------------------------------------------------------------------------------


def test_an_upstream_listing_a_late_input_taints_its_column_downstream(df, tmp_path):
    # The upstream reads `wind` of D, published 10:00 on D by its rules: after its 09:30 issue.
    run_upstream(df, tmp_path / "up", predictors=[*UPSTREAM, "wind"], rules=LATE_WIND)
    source = ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc")
    assert source.chained_input().late == ("wind",)
    data = chained(df, source)
    info = INFO.with_source_rules([source])
    with pytest.warns(UserWarning, match="whose upstream is late"):
        report = downstream(PREDICTORS + ["load_fc_d+1"], info, availability="warn").run(data, *TEST, horizon=1)
    rows = report.availability()
    late = rows[rows["status"] == "late upstream"]
    assert late["predictor"].tolist() == ["load_fc_d+1"] and "wind" in late["note"].iloc[0]
    assert late["periods"].iloc[0] == "0-23" and late["horizons"].iloc[0] == "1"
    with pytest.raises(InformationSetError, match="load_fc_d\\+1 is a forecast of load_up"):
        downstream(PREDICTORS + ["load_fc_d+1"], info).run(data, *TEST, horizon=1)


def test_a_late_upstream_is_listed_in_each_window_of_a_run_with_validation(df, tmp_path):
    run_upstream(df, tmp_path / "up", predictors=[*UPSTREAM, "wind"], rules=LATE_WIND)
    source = ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc")
    data = chained(df, source)
    pipeline = downstream(PREDICTORS + ["load_fc_d+1"], INFO.with_source_rules([source]), availability="warn")
    with pytest.warns(UserWarning, match="whose upstream is late"):
        report = pipeline.run(
            data, validation=("2023-03-28", "2023-04-03"), test=TEST, horizon=1, save_dir=str(tmp_path / "down")
        )
    for listing in (report.availability(), report.validation.availability()):
        assert listing[listing["status"] == "late upstream"]["predictor"].tolist() == ["load_fc_d+1"]
    reread = pipeline.report(str(tmp_path / "down"), test=TEST, horizon=1).availability()
    assert reread[reread["status"] == "late upstream"]["predictor"].tolist() == ["load_fc_d+1"]


def test_an_upstream_run_at_none_taints_its_column_and_a_clean_one_passes(df, tmp_path):
    run_upstream(df, tmp_path / "none", availability="none")
    run_upstream(df, tmp_path / "clean", availability="raise")
    for name, refused in (("none", True), ("clean", False)):
        source = ForecastSource(str(tmp_path / name), model="load_up", column="load_fc")
        data = chained(df, source)
        pipeline = downstream(PREDICTORS + ["load_fc_d+1"], INFO.with_source_rules([source]))
        if refused:
            with pytest.raises(InformationSetError, match="ran at availability='none'"):
                pipeline.run(data, *TEST, horizon=1)
        else:
            assert pipeline.run(data, *TEST, horizon=1).forecast_counts()["produced"].iloc[0] == 240


def test_an_ensemble_upstream_is_tainted_by_any_of_its_members(df, tmp_path):
    info = InformationSet("09:30", ZONE, rules={**RULES, **LATE_WIND})
    pipeline = ModelPipeline(information_set=info, availability="warn")
    pipeline.add_model(OLSModel(UPSTREAM, training_window=30, name="clean"))
    pipeline.add_model(OLSModel([*UPSTREAM, "wind"], training_window=30, name="late"))
    pipeline.add_ensemble(AverageEnsemble(["clean", "late"], name="avg"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        frame = df.drop(columns=[name for name in df.columns if name.startswith("load_d+")])
        pipeline.run(frame, *UP_TEST, target="load", horizon=1, save_dir=str(tmp_path / "up"))
    source = ForecastSource(str(tmp_path / "up"), model="avg", column="load_fc")
    assert source.availability().time == "09:30"
    assert source.chained_input().late == ("late: wind",)
    data = chained(df, source)
    with pytest.raises(InformationSetError, match="late: wind"):
        downstream(PREDICTORS + ["load_fc_d+1"], INFO.with_source_rules([source])).run(data, *TEST, horizon=1)


# -- the issue time and verify ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("horizon", [1, 2])
def test_verify_is_ok_with_its_control_on_a_chained_column(df, tmp_path, horizon):
    run_upstream(df, tmp_path / "up")
    source = ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc", horizon=horizon)
    data = chained(df, source)
    pipeline = downstream(PREDICTORS + ["load_fc_d+{horizon}"], INFO.with_source_rules([source]))
    report = pipeline.verify(data, *TEST, horizon=horizon, days=2)
    assert report.table().set_index("model").loc["price_m", "status"] == "ok"


def test_a_chained_input_read_before_its_upstream_issue_time_is_a_leak_and_refused_at_raise(df, tmp_path):
    """The upstream issues at 12:30, the downstream at 11:45: its forecast for D+1, issued on D, is not out yet."""
    run_upstream(df, tmp_path / "up", issue_time="12:30")
    source = ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc")
    data = chained(df, source)
    info = INFO.with_source_rules([source])
    with pytest.raises(InformationSetError, match="load_fc_d\\+1"):
        downstream(PREDICTORS + ["load_fc_d+1"], info).run(data, *TEST, horizon=1)
    pipeline = downstream(PREDICTORS + ["load_fc_d+1"], info, availability="warn")
    report = pipeline.verify(data, *TEST, horizon=1, days=2)
    assert report.table().set_index("model").loc["price_m", "status"] == "leak"
    # The forecast issued the day before (load_fc_d+1 of D-1, i.e. load_fc on D) is out: no leak there.
    known = downstream(PREDICTORS + ["load_fc"], info, availability="warn").verify(data, *TEST, horizon=1, days=2)
    assert known.table().set_index("model").loc["price_m", "status"] == "ok"


def test_explain_shows_the_upstream_in_the_rule_column(df, tmp_path):
    run_upstream(df, tmp_path / "up")
    source = ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc")
    data = chained(df, source)
    pipeline = downstream(PREDICTORS + ["load_fc_d+1"], INFO.with_source_rules([source]))
    table = pipeline.explain(data, model="price_m", test=TEST, horizon=1, hour=12, echo=False)
    rule = table["predictors"].set_index("predictor").loc["load_fc_d+1", "rule"]
    assert "published 09:30 Europe/Warsaw on D-1" in rule and f"forecasts of load_up (horizon 1) in {tmp_path}" in rule


# -- resume ----------------------------------------------------------------------------------------------------------------


def _resume(df, tmp_path, source, availability="raise"):
    data = chained(df, source)
    pipeline = downstream(PREDICTORS + ["load_fc_d+1"], INFO.with_source_rules([source]), availability=availability)
    return pipeline.run(data, *TEST, horizon=1, save_dir=str(tmp_path / "down"))


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"issue_time": "10:00"}, "issue_time"),
        ({"zone": "Europe/Berlin"}, "issue_timezone"),
    ],
)
def test_a_resume_over_another_upstream_issue_time_or_zone_is_refused(df, tmp_path, change, field):
    run_upstream(df, tmp_path / "up")
    _resume(df, tmp_path, ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc"))
    run_upstream(df, tmp_path / "other", **change)  # the same forecasts, issued at another time
    other = ForecastSource(str(tmp_path / "other"), model="load_up", column="load_fc")
    assert chained(df, other)["load_fc"].equals(
        chained(df, ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc"))["load_fc"]
    )
    with pytest.raises(ManifestMismatchError, match=f"chained_inputs.load_fc.{field}"):
        _resume(df, tmp_path, other)


def test_a_resume_with_another_horizon_or_available_is_refused(df, tmp_path):
    run_upstream(df, tmp_path / "up")
    _resume(df, tmp_path, ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc"))
    with pytest.raises(ManifestMismatchError, match="chained_inputs.load_fc.available"):
        _resume(
            df, tmp_path, ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc", available="10:00")
        )
    with pytest.raises(ManifestMismatchError, match="chained_inputs.load_fc.horizon"):
        _resume(df, tmp_path, ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc", horizon=2))


def test_a_changed_upstream_digest_warns_and_other_numbers_are_refused_by_the_data_hash(df, tmp_path):
    run_upstream(df, tmp_path / "up")
    source = ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc")
    _resume(df, tmp_path, source)
    # Run again at another level: its manifest records it, its forecasts don't move.
    run_upstream(df, tmp_path / "up", availability="raise")
    with pytest.warns(UserWarning, match="the upstream of load_fc .* changed"):
        _resume(df, tmp_path, source)
    # Rerun with another window: other numbers in the column.
    shutil.rmtree(tmp_path / "up")
    run_upstream(df, tmp_path / "up", window=40)
    with pytest.raises(ManifestMismatchError, match="data_columns_hash"):
        _resume(df, tmp_path, source)


def test_a_resume_where_one_run_knew_the_chain_and_the_other_didn_t_is_refused(df, tmp_path):
    run_upstream(df, tmp_path / "up")
    source = ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc")
    _resume(df, tmp_path, source)
    data = chained(df, source)
    rules = INFO.copy()
    rules.rules["load_fc"] = Availability.published("09:30", 1, timezone=ZONE)  # the same rule, written by hand
    pipeline = downstream(PREDICTORS + ["load_fc_d+1"], rules)
    with pytest.raises(ManifestMismatchError, match="chained_inputs.load_fc: 'load_up'"):
        pipeline.run(data, *TEST, horizon=1, save_dir=str(tmp_path / "down"))


# -- a daily upstream ------------------------------------------------------------------------------------------------------


def test_a_daily_upstream_records_its_stamp_and_its_forecasts_land_there(tmp_path):
    hourly = make_market_data(start="2023-02-01", days=60, noise=5.0, seed=11)
    daily = hourly.drop(columns=[name for name in hourly.columns if name.startswith("load_d+")]).at_time("12:00")
    info = InformationSet("09:30", ZONE, rules=RULES)
    pipeline = ModelPipeline(information_set=info).add_model(OLSModel(UPSTREAM, training_window=20, name="daily_up"))
    pipeline.run(daily, "2023-03-10", "2023-03-31", target="load", horizon=2, freq="1D", save_dir=str(tmp_path / "up"))
    manifest = json.loads((tmp_path / "up" / "daily_up.manifest.json").read_text())
    assert manifest["daily_stamp"] == "12:00"
    hourly_run = ModelPipeline(information_set=info).add_model(OLSModel(UPSTREAM, training_window=20, name="h"))
    hourly_run.run(
        hourly.drop(columns=[name for name in hourly.columns if name.startswith("load_d+")]),
        "2023-03-10",
        "2023-03-12",
        target="load",
        horizon=1,
        save_dir=str(tmp_path / "h"),
    )
    assert "daily_stamp" not in json.loads((tmp_path / "h" / "h.manifest.json").read_text())
    for horizon in (1, 2):
        frame = ForecastSource(str(tmp_path / "up"), "daily_up", "load_fc", horizon=horizon).fetch(*WIDE)
        local = frame.index.tz_convert(ZONE)
        assert set(local.strftime("%H:%M")) == {"12:00"}
        records = {
            r["target_date"]: r["prediction"]
            for r in map(json.loads, (tmp_path / "up" / "daily_up.jsonl").read_text().splitlines())
            if r["horizon"] == horizon and r["status"] == "ok"
        }
        assert dict(zip(local.strftime("%Y-%m-%d"), frame["load_fc"])) == records
