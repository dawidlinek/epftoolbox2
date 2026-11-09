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


@pytest.fixture(scope="module")
def shared(df, tmp_path_factory):
    """The upstreams most tests read, run once per module: complete over every row a 45-day window reads ("up"), and starting
    on 2023-03-20, mid-window ("mid"). A test that writes into one works on a copy (`copied`)."""
    root = tmp_path_factory.mktemp("upstreams")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MAX_PROCESSES", "1")
        patch.setenv("THREADS_PER_PROCESS", "1")
        patch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)
        run_upstream(df, root / "up")
        run_upstream(df, root / "mid", test=("2023-03-20", "2023-04-20"))
    up = ForecastSource(str(root / "up"), model="load_up", column="load_fc")
    mid = ForecastSource(str(root / "mid"), model="load_up", column="load_fc")
    return {"root": root, "up": up, "mid": mid, "up_data": chained(df, up), "mid_data": chained(df, mid)}


def copied(shared, tmp_path, which="up"):
    """A copy of a shared upstream in `tmp_path`, for a test that runs into it again."""
    shutil.copytree(shared["root"] / which, tmp_path / which)
    return tmp_path / which


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
        if "validation" in kwargs:  # a run with windows takes both by keyword
            pipeline.run(frame, test=test, target="load", horizon=horizon, save_dir=str(directory), **kwargs)
        else:
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


def test_the_prototype_s_chain_runs_with_the_upstream_issue_time(shared):
    source, data = shared["up"], shared["up_data"].copy()
    # The forecast for 2023-04-11 12:00 stored by the upstream run is on the row of D = 2023-04-10 as load_fc_d+1.
    assert data.loc["2023-04-10 12:00", "load_fc_d+1"] == stored(shared["root"] / "up")[("2023-04-10", 12)]

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


def independent_let_off(data, name, first_row, last_row):
    """The rows `independent_count` leaves out: missing, but reading a wall time off the clock or a slot no forecast is planned
    for."""
    missing_all = int(
        (
            (data.index.tz_localize(None).normalize() >= pd.Timestamp(first_row))
            & (data.index.tz_localize(None).normalize() <= pd.Timestamp(last_row))
            & data[name].isna().to_numpy()
        ).sum()
    )
    return missing_all - independent_count(data, name, first_row, last_row)[0]


@pytest.mark.parametrize("availability", ["none", "warn"])
def test_an_upstream_starting_mid_window_warns_with_its_counts_and_the_days_to_cover(shared, tmp_path, availability):
    source, data = shared["mid"], shared["mid_data"]
    info = INFO.with_source_rules([source])
    pipeline = downstream(PREDICTORS + ["load_fc_d+1", "load_fc_d-1"], info, availability=availability)
    with pytest.warns(UserWarning, match="rows its fits read have no value of load_fc") as caught:
        pipeline.run(data, *TEST, horizon=1, save_dir=str(tmp_path / "down"))
    # A 30-day window and horizon 1 read rows from D-31 to D-2: 2023-03-05 to 2023-04-12. The first forecast is issued on
    # 2023-03-20, for 03-21: the row of 03-20 is the first covered by load_fc_d+1, 03-22 by load_fc_d-1 (issued 03-20 for 03-21).
    expected, read = independent_count(data, "load_fc_d+1", "2023-03-05", "2023-04-12")
    found = [w for w in caught if "have no value" in str(w.message)]
    assert len(found) == 1 and found[0].filename == __file__  # one per chained column, attributed to the caller
    message = str(found[0].message)
    assert f"load_fc_d+1 on {expected} of {read} rows, from 2023-03-05 to 2023-03-19" in message
    assert "load_fc_d-1 on" in message and "covers the issue days 2023-03-03 to 2023-03-19" in message
    manifest = json.loads((tmp_path / "down" / "price_m.manifest.json").read_text())
    coverage = manifest["chained_inputs"]["load_fc"]["coverage"]
    assert coverage["load_fc_d+1"] == {
        "read": read,
        "uncovered": expected,
        "let_off": independent_let_off(data, "load_fc_d+1", "2023-03-05", "2023-04-12"),
        "first": "2023-03-05",
        "last": "2023-03-19",
        "run_from": "2023-03-05",
        "run_to": "2023-03-19",
    }
    assert expected == 15 * 24  # 15 whole days before the upstream's first forecast
    assert coverage["load_fc_d+1"]["let_off"] == 1  # 03-25 02:00 reads 03-26 02:00, which the switch skips
    assert (coverage["load_fc_d-1"]["first"], coverage["load_fc_d-1"]["last"]) == ("2023-03-05", "2023-03-21")
    assert (coverage["load_fc_d-1"]["run_from"], coverage["load_fc_d-1"]["run_to"]) == ("2023-03-03", "2023-03-19")


def test_with_refit_the_rows_read_are_those_of_the_refit_days(shared, tmp_path):
    """refit="7D": the forecasts of 04-05 to 04-14 use the fits of Mondays 04-03 and 04-10, whose rows run from 03-03 (04-03 - 31)
    to 04-08: two more days before the upstream's first forecast than without refit."""
    source, data = shared["mid"], shared["mid_data"]
    pipeline = downstream(
        PREDICTORS + ["load_fc_d+1"], INFO.with_source_rules([source]), availability="warn", refit="7D"
    )
    with pytest.warns(UserWarning, match="have no value of"):
        pipeline.run(data, *TEST, horizon=1, save_dir=str(tmp_path / "down"))
    found = json.loads((tmp_path / "down" / "price_m.manifest.json").read_text())["chained_inputs"]["load_fc"][
        "coverage"
    ]
    expected, read = independent_count(data, "load_fc_d+1", "2023-03-03", "2023-04-08")
    assert found["load_fc_d+1"] == {
        "read": read,
        "uncovered": expected,
        "let_off": independent_let_off(data, "load_fc_d+1", "2023-03-03", "2023-04-08"),
        "first": "2023-03-03",
        "last": "2023-03-19",
        "run_from": "2023-03-03",
        "run_to": "2023-03-19",
    }
    assert expected == 17 * 24


def test_an_upstream_starting_mid_window_is_refused_at_raise(shared):
    pipeline = downstream(PREDICTORS + ["load_fc_d+1"], INFO.with_source_rules([shared["mid"]]))
    with pytest.raises(InformationSetError, match="covers the issue days 2023-03-05 to 2023-03-19"):
        pipeline.run(shared["mid_data"], *TEST, horizon=1)


def test_an_upstream_ending_early_names_the_days_at_the_end(df, tmp_path):
    """An upstream that stops before the last rows the fits read: the days to cover are at the end, not a start."""
    run_upstream(df, tmp_path / "up", test=("2023-02-10", "2023-04-09"))
    source = ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc")
    pipeline = downstream(PREDICTORS + ["load_fc_d+1"], INFO.with_source_rules([source]), availability="warn")
    with pytest.warns(UserWarning, match="covers the issue days 2023-04-10 to 2023-04-12"):
        report = pipeline.run(chained(df, source), *TEST, horizon=1)
    counts = report.forecast_counts().iloc[0]
    assert counts["produced"] < counts["planned"]  # the issue rows after the upstream's end have no value either


def test_coverage_counts_the_periods_whose_lists_read_the_column(shared, tmp_path):
    """An upstream without night forecasts read only in the hours that have them: nothing is missing from what the fits read.
    Read at every hour, the night rows are counted, and only they."""
    data = shared["up_data"].copy()
    night = (data.index.hour < 6) | (data.index.hour > 20)
    data.loc[night, "load_fc_d+1"] = np.nan
    info = INFO.with_source_rules([shared["up"]])

    def by_day(horizon, hour):
        return PREDICTORS + (["load_fc_d+1"] if 6 <= hour <= 20 else [])

    for level in ("warn", "raise"):
        pipeline = ModelPipeline(information_set=info, availability=level)
        pipeline.add_model(OLSModel(by_day, training_window=30, name="by_day"))
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*have no value of")
            report = pipeline.run(data, *TEST, horizon=1, save_dir=str(tmp_path / level))
        assert report.forecast_counts()["produced"].iloc[0] == 240
        coverage = json.loads((tmp_path / level / "by_day.manifest.json").read_text())["chained_inputs"]["load_fc"]
        assert coverage["coverage"]["load_fc_d+1"]["uncovered"] == 0
    every_hour = downstream(PREDICTORS + ["load_fc_d+1"], info, availability="warn")
    with pytest.warns(UserWarning, match="have no value of load_fc"):
        every_hour.run(data, *TEST, horizon=1, save_dir=str(tmp_path / "every"))
    found = json.loads((tmp_path / "every" / "price_m.manifest.json").read_text())["chained_inputs"]["load_fc"]
    expected, _ = independent_count(
        data, "load_fc_d+1", "2023-03-05", "2023-04-12"
    )  # the night rows, but 02:00 of 03-25
    assert found["coverage"]["load_fc_d+1"]["uncovered"] == expected == 39 * 9 - 2  # and no 02:00 on 03-26


def test_a_complete_upstream_is_silent_over_a_spring_switch_and_its_manifest_records_it(shared, tmp_path):
    source, data = shared["up"], shared["up_data"]
    pipeline = downstream(PREDICTORS + ["load_fc_d+1", "load_fc_d-1"], INFO.with_source_rules([source]), window=45)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*have no value of")
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


def test_a_model_without_a_chained_column_has_no_key_in_its_manifest(shared, tmp_path):
    info = INFO.with_source_rules([shared["up"]])
    downstream(PREDICTORS, info).run(shared["up_data"], *TEST, horizon=1, save_dir=str(tmp_path / "d"))
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
def test_verify_is_ok_with_its_control_on_a_chained_column(df, shared, horizon):
    source = ForecastSource(str(shared["root"] / "up"), model="load_up", column="load_fc", horizon=horizon)
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


def test_explain_shows_the_upstream_in_the_rule_column(shared):
    source, data = shared["up"], shared["up_data"]
    pipeline = downstream(PREDICTORS + ["load_fc_d+1"], INFO.with_source_rules([source]))
    table = pipeline.explain(data, model="price_m", test=TEST, horizon=1, hour=12, echo=False)
    rule = table["predictors"].set_index("predictor").loc["load_fc_d+1", "rule"]
    assert (
        "published 09:30 Europe/Warsaw on D-1" in rule
        and f"forecasts of load_up (horizon 1) in {shared['root']}" in rule
    )


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
def test_a_resume_over_another_upstream_issue_time_or_zone_is_refused(df, shared, tmp_path, change, field):
    _resume(df, tmp_path, shared["up"])
    run_upstream(df, tmp_path / "other", **change)  # the same forecasts, issued at another time
    other = ForecastSource(str(tmp_path / "other"), model="load_up", column="load_fc")
    assert chained(df, other)["load_fc"].equals(shared["up_data"]["load_fc"])
    with pytest.raises(ManifestMismatchError, match=f"chained_inputs.load_fc.{field}"):
        _resume(df, tmp_path, other)


def test_a_resume_with_another_horizon_or_available_is_refused(df, shared, tmp_path):
    _resume(df, tmp_path, shared["up"])
    up = str(shared["root"] / "up")
    with pytest.raises(ManifestMismatchError, match="chained_inputs.load_fc.available"):
        _resume(df, tmp_path, ForecastSource(up, model="load_up", column="load_fc", available="10:00"))
    with pytest.raises(ManifestMismatchError, match="chained_inputs.load_fc.horizon"):
        _resume(df, tmp_path, ForecastSource(up, model="load_up", column="load_fc", horizon=2))


def test_a_changed_upstream_digest_warns_and_other_numbers_are_refused_by_the_data_hash(df, shared, tmp_path):
    source = ForecastSource(str(copied(shared, tmp_path)), model="load_up", column="load_fc")
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


def test_a_resume_where_one_run_knew_the_chain_and_the_other_didn_t_is_refused(df, shared, tmp_path):
    _resume(df, tmp_path, shared["up"])
    data = shared["up_data"]
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


# -- what vouches for an upstream (the 7a review) --------------------------------------------------------------------------


def test_a_rule_for_a_chained_column_written_after_the_sources_is_refused_at_every_level(shared):
    info = INFO.with_source_rules([shared["up"]])
    info.rules["load_fc"] = Availability.published(
        "08:00", 1, timezone=ZONE
    )  # would let a 12:30 upstream pass at 11:45
    for level in ("none", "warn", "raise"):
        pipeline = downstream(PREDICTORS + ["load_fc_d+1"], info, availability=level)
        with pytest.raises(InformationSetError, match="nothing replaces it"):
            pipeline.run(shared["up_data"], *TEST, horizon=1)


def test_an_upstream_resumed_at_another_issue_time_is_not_vouched_for(df, tmp_path):
    """Run at 09:30 with `wind` of D (out at 10:00 on D until 2023-03-15, 09:00 from 03-16) listed late on its first days, then
    resumed at 09:45 over later days only: the manifest says 09:45 and lists nothing, but the file still holds the forecasts the
    first run made with a late input. They aren't vouched for, by the mark of the other rules and by their run dates."""
    wind = [
        Availability.published("10:00", 0).during(valid_until="2023-03-15"),
        Availability.published("09:00", 0).during(valid_from="2023-03-16"),
    ]
    run_upstream(df, tmp_path / "up", predictors=[*UPSTREAM, "wind"], rules={"wind": wind}, horizon=1)
    run_upstream(
        df,
        tmp_path / "up",
        predictors=[*UPSTREAM, "wind"],
        rules={"wind": wind},
        horizon=1,
        issue_time="09:45",
        availability="raise",
        test=("2023-03-20", "2023-04-20"),
    )
    manifest = json.loads((tmp_path / "up" / "load_up.manifest.json").read_text())
    assert manifest["information_set"]["issue_time"] == "09:45" and manifest["availability"]["listing"] == []
    assert [row["predictor"] for row in manifest["availability"]["stored_under_other_rules"]["listing"]] == ["wind"]
    source = ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc")
    chained_input = source.chained_input()
    assert chained_input.late_upstream and chained_input.late == ()
    assert any("listed under other rules" in reason for reason in chained_input.unchecked)
    assert any("outside the test period 2023-03-20 to 2023-04-20" in reason for reason in chained_input.unchecked)
    with pytest.raises(InformationSetError, match="listed under other rules"):
        downstream(PREDICTORS + ["load_fc_d+1"], INFO.with_source_rules([source])).run(
            chained(df, source), *TEST, horizon=1
        )
    # Every later resume keeps the mark, at the same rules too.
    run_upstream(
        df, tmp_path / "up", predictors=[*UPSTREAM, "wind"], rules={"wind": wind}, horizon=1, issue_time="09:45"
    )
    again = json.loads((tmp_path / "up" / "load_up.manifest.json").read_text())["availability"]
    assert "stored_under_other_rules" in again


WIND_OVER_TIME = {
    "wind": [
        Availability.published("10:00", 0).during(valid_until="2023-03-15"),
        Availability.published("09:00", 0).during(valid_from="2023-03-16"),
    ]
}
VALIDATION_AND_TEST = {"validation": ("2023-02-10", "2023-03-15"), "test": ("2023-03-16", "2023-04-20")}


def _up_with_wind(df, directory, issue_time, availability, save_format, test, validation=None):
    run_upstream(
        df,
        directory,
        test=test,
        predictors=[*UPSTREAM, "wind"],
        rules=WIND_OVER_TIME,
        horizon=1,
        issue_time=issue_time,
        availability=availability,
        save_format=save_format,
        **({} if validation is None else {"validation": validation}),
    )
    return ForecastSource(str(directory), model="load_up", column="load_fc")


@pytest.mark.parametrize("save_format", ["jsonl", "parquet"])
def test_records_of_a_run_at_none_are_vouched_for_only_where_a_later_run_checked_their_days(df, tmp_path, save_format):
    """A run at "none" over validation and test, resumed at "raise" over the test window alone: the validation records (late
    `wind` until 03-15) are no later run's, so the upstream is unchecked. A resume over every day of the earlier run judges them
    all, and stays clean."""
    source = _up_with_wind(df, tmp_path / "part", "09:30", "none", save_format, **VALIDATION_AND_TEST)
    _up_with_wind(df, tmp_path / "part", "09:30", "raise", save_format, test=VALIDATION_AND_TEST["test"])
    availability = json.loads((tmp_path / "part" / "load_up.manifest.json").read_text())["availability"]
    assert availability["stored_under_other_rules"] == {"listing": [], "level": "none"}
    unchecked = source.chained_input().unchecked
    assert source.chained_input().late_upstream and "availability='none', on days no later run checked" in unchecked[0]
    clean = _up_with_wind(df, tmp_path / "all", "09:30", "none", save_format, test=VALIDATION_AND_TEST["test"])
    _up_with_wind(df, tmp_path / "all", "09:30", "raise", save_format, test=VALIDATION_AND_TEST["test"])
    assert (
        "stored_under_other_rules"
        not in json.loads((tmp_path / "all" / "load_up.manifest.json").read_text())["availability"]
    )
    assert not clean.chained_input().late_upstream


@pytest.mark.parametrize("save_format", ["jsonl", "parquet"])
def test_a_clean_upstream_issued_later_stays_clean_and_earlier_does_not(df, tmp_path, save_format):
    """Nothing late at 09:30: issued at 09:45 with the same rules, everything known at 09:30 still is. Issued at 09:00, the
    records made at 09:30 aren't judged by it, and the mark says so."""
    for issue_time, marked in (("09:45", False), ("09:00", True)):
        directory = tmp_path / issue_time.replace(":", "")
        run_upstream(df, directory, horizon=1, availability="raise", save_format=save_format)
        run_upstream(df, directory, horizon=1, availability="raise", save_format=save_format, issue_time=issue_time)
        availability = json.loads((directory / "load_up.manifest.json").read_text())["availability"]
        assert ("stored_under_other_rules" in availability) is marked, issue_time
        source = ForecastSource(str(directory), model="load_up", column="load_fc")
        assert source.chained_input().late_upstream is marked, issue_time


def test_an_upstream_at_raise_that_let_a_column_without_a_rule_through_is_not_vouched_for(df, tmp_path):
    """`unknown="warn"` at `availability="raise"` runs a column no rule describes (here the next day's load) and lists nothing."""
    frame = df.drop(columns=[name for name in df.columns if name.startswith("load_d+")])
    frame["load_next_day"] = frame["load"].shift(-24)
    info = InformationSet("09:30", ZONE, rules=RULES, unknown="warn")
    pipeline = ModelPipeline(information_set=info, availability="raise")
    pipeline.add_model(OLSModel([*UPSTREAM, "load_next_day"], training_window=30, name="load_up"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pipeline.run(frame, *UP_TEST, target="load", horizon=1, save_dir=str(tmp_path / "up"))
    assert json.loads((tmp_path / "up" / "load_up.manifest.json").read_text())["availability"]["listing"] == []
    source = ForecastSource(str(tmp_path / "up"), model="load_up", column="load_fc")
    assert source.chained_input().late_upstream
    data = chained(df, source)
    info = INFO.with_source_rules([source])
    with pytest.raises(InformationSetError, match="unknown='warn'"):
        downstream(PREDICTORS + ["load_fc_d+1"], info).run(data, *TEST, horizon=1)
    with pytest.warns(UserWarning, match="whose upstream is late"):
        report = downstream(PREDICTORS + ["load_fc_d+1"], info, availability="warn").run(data, *TEST, horizon=1)
    assert (report.availability()["status"] == "late upstream").sum() == 1


# -- a Workflow of two stages ----------------------------------------------------------------------------------------------


def test_a_workflow_runs_the_upstream_then_reads_it_back_with_its_rule(df, tmp_path, monkeypatch):
    """Two Workflow files: the upstream's, then the downstream's with a ForecastSource whose path is relative to its YAML. The
    Workflow gives the downstream's information set the source's rule, so the chained input passes at raise, and the manifest
    records its upstream by an absolute path."""
    import yaml

    from epftoolbox2.pipelines import Workflow

    upstream_frame = df.drop(columns=[name for name in df.columns if name.startswith("load_d+")])
    upstream_frame.tz_convert("UTC").to_csv(tmp_path / "upstream.csv", index_label="datetime")
    df.tz_convert("UTC").to_csv(tmp_path / "downstream.csv", index_label="datetime")
    rules = {name: rule.to_dict() for name, rule in RULES.items()}

    def stage(name, sources, transformers, models, target, issue):
        (tmp_path / name).mkdir()
        config = {
            "data_pipeline": {
                "start": "2023-01-01",
                "end": "2023-05-31",
                "sources": sources,
                "transformers": transformers,
            },
            "model_pipeline": {
                "target": target,
                "horizon": 1,
                "test_start": "2023-04-05" if name == "down" else UP_TEST[0],
                "test_end": "2023-04-14" if name == "down" else UP_TEST[1],
                "save_dir": "runs",
                "information_set": {"issue_time": issue, "issue_timezone": ZONE, "rules": rules},
                "availability": "raise",
                "models": models,
            },
        }
        path = tmp_path / name / "workflow.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        return path

    csv = {"class": "CsvSource", "params": {"datetime_column": "datetime"}}
    up = stage(
        "up",
        [{**csv, "params": {**csv["params"], "file_path": "../upstream.csv"}}],
        [{"class": "TimezoneTransformer", "params": {"target_tz": ZONE}}],
        [{"class": "OLSModel", "params": {"predictors": UPSTREAM, "training_window": 30, "name": "load_up"}}],
        "load",
        "09:30",
    )
    down = stage(
        "down",
        [
            {**csv, "params": {**csv["params"], "file_path": "../downstream.csv"}},
            {"class": "ForecastSource", "params": {"path": "../up/runs", "model": "load_up", "column": "load_fc"}},
        ],
        [
            {"class": "TimezoneTransformer", "params": {"target_tz": ZONE}},
            {"class": "LagTransformer", "params": {"columns": ["load_fc"], "lags": [-1], "freq": "day"}},
        ],
        [
            {
                "class": "OLSModel",
                "params": {"predictors": PREDICTORS + ["load_fc_d+1"], "training_window": 30, "name": "m"},
            }
        ],
        "price",
        "11:45",
    )
    monkeypatch.chdir(tmp_path / "up")  # neither stage's directory: the paths are their YAML's
    Workflow.load(up).run()
    report = Workflow.load(down).run()
    assert report.forecast_counts()["produced"].tolist() == [240]
    entry = json.loads((tmp_path / "down" / "runs" / "m.manifest.json").read_text())["chained_inputs"]["load_fc"]
    assert entry["path"] == str((tmp_path / "up" / "runs").resolve()) and entry["issue_time"] == "09:30"
    assert entry["coverage"]["load_fc_d+1"]["uncovered"] == 0


# -- a day statistic of the forecasts (ForecastSource(agg=), 7d) ------------------------------------------------------------


def with_day_statistic(df, source):
    """`df` with the source's day statistic at its rows (`{column}_day_{agg}`) and its day lags `_d+1`, `_d-1`."""
    data = df.copy()
    name = source.output_column
    data[name] = source.fetch(*WIDE)[name].reindex(df.index.tz_convert("UTC")).to_numpy()
    lagged = LagTransformer([name], [-1, 1], "day").transform(data[[name]])
    for column in lagged.columns.drop(name):
        data[column] = lagged[column]
    return data


def test_a_day_statistic_misses_the_day_after_a_spring_switch_without_a_coverage_warning(df, shared, tmp_path):
    """No forecast of 02:00 on 2023-03-27 is planned at horizon 1 (the switch day, its issue day, has no 02:00), so that date is
    never whole: its maximum is NaN on every row, which no rerun of the upstream fills. The downstream's fits read it (the rows
    of the switch day hold `load_fc_day_max_d+1` of 03-27); they are let off, and a run at raise goes through."""
    source = ForecastSource(str(shared["root"] / "up"), model="load_up", column="load_fc", agg="max")
    data = with_day_statistic(df, source)
    dates = data.index.tz_localize(None).normalize()
    assert data.loc[dates == pd.Timestamp("2023-03-27"), "load_fc_day_max"].isna().all()
    assert (
        data.loc[(dates >= pd.Timestamp("2023-02-11")) & (dates != pd.Timestamp("2023-03-27")), "load_fc_day_max"]
        .iloc[: 24 * 60]
        .notna()
        .all()
    )
    pipeline = downstream(PREDICTORS + ["load_fc_day_max_d+1"], INFO.with_source_rules([source]), window=45)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*have no value of")
        pipeline.run(data, *TEST, horizon=1, save_dir=str(tmp_path / "down"))
    entry = json.loads((tmp_path / "down" / "price_m.manifest.json").read_text())["chained_inputs"]["load_fc_day_max"]
    found = entry["coverage"]["load_fc_day_max_d+1"]
    assert (entry["agg"], entry["horizon"], found["uncovered"]) == ("max", 1, 0)
    # The trace of the exemption: every row of the switch day, which reads 03-27, and 03-25 02:00, which reads the hour the
    # switch skips.
    assert found["let_off"] == 23 + 1


def test_a_day_statistic_of_an_upstream_starting_mid_window_counts_its_missing_days(df, shared, tmp_path):
    """The `mid` upstream's first forecast is issued on 2023-03-20: the rows of 03-05 to 03-19 read dates it has no forecast
    for, 15 days of 24 rows; the switch day's 23 rows read 03-27, which no upstream can complete, and are not counted."""
    source = ForecastSource(str(shared["root"] / "mid"), model="load_up", column="load_fc", agg="max")
    data = with_day_statistic(df, source)
    pipeline = downstream(PREDICTORS + ["load_fc_day_max_d+1"], INFO.with_source_rules([source]), availability="warn")
    with pytest.warns(UserWarning, match="rows its fits read have no value of load_fc_day_max"):
        pipeline.run(data, *TEST, horizon=1, save_dir=str(tmp_path / "down"))
    found = json.loads((tmp_path / "down" / "price_m.manifest.json").read_text())["chained_inputs"]["load_fc_day_max"]
    assert found["coverage"]["load_fc_day_max_d+1"]["uncovered"] == 15 * 24
    assert (found["coverage"]["load_fc_day_max_d+1"]["first"], found["coverage"]["load_fc_day_max_d+1"]["last"]) == (
        "2023-03-05",
        "2023-03-19",
    )
    with pytest.raises(InformationSetError, match="have no value of load_fc_day_max"):
        downstream(PREDICTORS + ["load_fc_day_max_d+1"], INFO.with_source_rules([source])).run(data, *TEST, horizon=1)


def test_verify_is_ok_on_a_day_statistic_and_finds_the_leak_of_one_known_after_the_issue(df, shared):
    """`verify` with its positive control on the chained day maximum; stated known at 12:30 (`available`), after the 11:45
    issue, it is a leak and refused at raise."""
    path = str(shared["root"] / "up")
    source = ForecastSource(path, model="load_up", column="load_fc", agg="max")
    data = with_day_statistic(df, source)
    pipeline = downstream(PREDICTORS + ["load_fc_day_max_d+1"], INFO.with_source_rules([source]))
    assert pipeline.verify(data, *TEST, horizon=1, days=2).table().set_index("model").loc["price_m", "status"] == "ok"
    late = ForecastSource(path, model="load_up", column="load_fc", agg="max", available="12:30")
    info = INFO.with_source_rules([late])
    with pytest.raises(InformationSetError, match="load_fc_day_max_d\\+1"):
        downstream(PREDICTORS + ["load_fc_day_max_d+1"], info).run(data, *TEST, horizon=1)
    report = downstream(PREDICTORS + ["load_fc_day_max_d+1"], info, availability="warn").verify(
        data, *TEST, horizon=1, days=2
    )
    assert report.table().set_index("model").loc["price_m", "status"] == "leak"


# -- hour aggregates of a chained column (PeriodStatsTransformer, 7d) -------------------------------------------------------


@pytest.mark.parametrize("shift", ["", "_d-1", "_d+1"])
def test_an_hour_aggregate_of_a_chained_column_has_no_hole_where_the_upstream_plans_no_forecast(shift):
    """A 15-minute upstream at horizon 1 plans no forecast of 02:00 to 02:45 on 2024-04-01 (its issue day, the spring switch,
    has no 02:00), so the hour mean of that hour is NaN on its four rows, and on the rows its lags read it from. The coverage
    check reads the column it was built from at the same row through the shifts after it (`x_d-1` for `x_hour_mean_d-1`) and
    lets those rows off; a profile and a day aggregate are judged as they stand."""
    from epftoolbox2.data.transformers import PeriodStatsTransformer
    from epftoolbox2.information_set import ChainedInput
    from epftoolbox2.models._chained import _off_the_clock

    chained_x = ChainedInput("x", "/p", "up", 1, "11:45", "Europe/Brussels", ZONE, None, "d", freq="15min")
    index = pd.date_range("2024-03-30", "2024-04-03", freq="15min", tz=ZONE, inclusive="left")
    wall = index.tz_localize(None)
    x = pd.Series(np.arange(len(index), dtype=float), index=index)
    x[(wall.normalize() == pd.Timestamp("2024-04-01")) & (wall.hour == 2)] = np.nan  # what the source places
    frame = PeriodStatsTransformer("x").transform(pd.DataFrame({"x": x}))
    frame = LagTransformer(["x", "x_hour_mean", "x_hour_dev"], [1, -1], "day").transform(frame)
    days = {"": 0, "_d-1": -1, "_d+1": 1}[shift]
    read = wall.normalize() + pd.Timedelta(days=days)
    inside = np.asarray(
        (read >= pd.Timestamp("2024-03-30")) & (read <= pd.Timestamp("2024-04-02"))
    )  # not the frame's edges
    for name in (f"x{shift}", f"x_hour_mean{shift}", f"x_hour_dev{shift}"):
        missing = frame[name].isna().to_numpy() & inside
        off = _off_the_clock(frame.index, name, chained_x)
        assert missing.any() and not (missing & ~off).any(), name
    assert not _off_the_clock(frame.index, "x_h02_d-1", chained_x).any()
    assert not _off_the_clock(frame.index, "x_day_max_d-1", chained_x).any()


def test_a_model_reading_an_hour_mean_of_a_chained_column_runs_at_raise(df, shared, tmp_path):
    """End to end over the spring switch: `load_fc_hour_mean_d-1` of the hourly upstream reads the hour mean of 03-26 02:00
    (no row) and of 03-27 02:00 (no forecast planned) and no other hole, so the run goes through at raise, with those rows
    let off in its manifest."""
    from epftoolbox2.data.transformers import PeriodStatsTransformer

    source = shared["up"]
    data = PeriodStatsTransformer("load_fc", freq="1h").transform(shared["up_data"].copy())
    data = LagTransformer(["load_fc_hour_mean"], [1], "day").transform(data)
    pipeline = downstream(PREDICTORS + ["load_fc_hour_mean_d-1"], INFO.with_source_rules([source]), window=45)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*have no value of")
        pipeline.run(data, *TEST, horizon=1, save_dir=str(tmp_path / "down"))
    entry = json.loads((tmp_path / "down" / "price_m.manifest.json").read_text())["chained_inputs"]["load_fc"]
    found = entry["coverage"]["load_fc_hour_mean_d-1"]
    # 03-27 02:00 reads the hour the switch skips on 03-26, 03-28 02:00 the hour no forecast is planned for on 03-27.
    assert found["uncovered"] == 0 and found["let_off"] == 2
