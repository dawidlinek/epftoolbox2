"""Stored fits (step 8d): `ModelPipeline.refit` fits every unit of an issue day's anchors on the backtest's rows and stores it as
arrays (`fits/<model>/<anchor>/units.npz` and `fit.json`), and a forecast served from it is the backtest's record to the bit.

The identity is checked over a grid that holds every level of every axis and every pair of levels (`.claude/rules/tests.md`),
not every cell: model (OLS, Ridge, LassoCV, LEAR with "lars" and "cd", window lists, "auto" with two rules), refit (None,
"7D", "1MS", "auto"), freq (1h, 15-min, 1D), the switch the test days cross (spring, autumn) and the engine (off, on), each path
against itself. Every record but `day_in_test` (the day's place in its run) is compared as JSON text, skips included, and every
key of the backtest must be served. Then: `if_due` (a second call is a no-op, another configuration refits), the publication
check of decision 5, the digests and the sha256 on load, the newest older fit of decision 15, `fit_missing`, the families
that fit at serve time, the live directory, and the models' `_fit_state` round trips.
"""

import datetime
import json
import os
from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from epftoolbox2.clock import FakeClock, use_clock
from epftoolbox2.information_set import Availability, InformationSet, InformationSetError
from epftoolbox2.models import LassoCVModel, LEARModel, NaiveModel, OLSModel, RidgeModel
from epftoolbox2.models import _engine_worker, _fits
from epftoolbox2.models._fits import FIT_FILE, UNITS_FILE, FitTooEarlyError, StoredFitError, fit_dir, stores_fits
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.live import LiveBusyError, LiveStore
from epftoolbox2.results.store import LiveDirectoryError
from tests.synthetic import PREDICTORS, make_market_data

pytestmark = pytest.mark.usefixtures("no_network")

RULES = {
    "load": Availability.published("10:00", 10),
    "wind": Availability.published("10:00", 10),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "price": Availability.published("13:00", 1),
}
INFO = InformationSet("11:45", "Europe/Warsaw", rules=RULES)
SPRING = ("2023-03-24", "2023-03-29")  # the clock goes forward on 26 March 2023 in Warsaw
AUTUMN = ("2023-10-27", "2023-11-01")  # and back on 29 October


class InProcessPool:
    """Stands in for the worker pools (a backtest's, an engine group's and refit's), whose start is most of these tests' time;
    `test_a_real_pool_stores_what_an_in_process_one_does` runs the real ones."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        if initializer is _engine_worker._engine_init:
            _engine_worker._engine_load(*initargs)
        else:
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


POOLED = ("epftoolbox2.models.base", "epftoolbox2.models._engine", "epftoolbox2.models._fits")


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    for module in POOLED:
        monkeypatch.setattr(f"{module}.ProcessPoolExecutor", InProcessPool)
    with threadpool_limits(limits=1):  # what a pool's worker runs on
        yield


def frame(switch, freq, start=None, days=None):
    """Synthetic prices around a switch, with noise so that no fit is exact; a daily frame is the hourly one at noon."""
    start = start or ("2023-01-20" if switch == SPRING else "2023-08-20")
    data = make_market_data(start=start, days=days or 80, freq="1h" if freq == "1D" else freq, horizon=2, noise=2.0)
    return data.at_time("12:00") if freq == "1D" else data


def days_of(window):
    first, last = (datetime.date.fromisoformat(day) for day in window)
    return [first + datetime.timedelta(days=i) for i in range((last - first).days + 1)]


def keyed(records):
    """(model, run_date, hour, horizon) -> the record as JSON text without `day_in_test`, its place in its own run."""
    return {
        (r["run_date"], r["hour"], r["horizon"]): json.dumps(
            {k: v for k, v in r.items() if k != "day_in_test"}, sort_keys=True
        )
        for r in records
    }


def backtest(pipeline, data, window, freq, horizon, save_dir):
    pipeline.run(data, *window, horizon=horizon, freq=freq, save_dir=str(save_dir))
    out = {}
    for model in pipeline.models:
        lines = (save_dir / f"{model.name.lower()}.jsonl").read_text().splitlines()
        out[model.name] = keyed(json.loads(line) for line in lines if line.strip())
    return out


def served(pipeline, data, live, days, freq, horizon, **kwargs):
    """Each model's records of every day served from its fits, refitted first for that day (`refit` then `_serve_stored`)."""
    out = {model.name: {} for model in pipeline.models}
    for day in days:
        pipeline.refit(data, live, issue=day, horizon=horizon, freq=freq)
        for name, result in pipeline._serve_stored(data, live, issue=day, horizon=horizon, freq=freq, **kwargs).items():
            assert all(entry["source"] == "stored" for entry in result.fits.values()), result.fits
            out[name].update(keyed(result.records))
    return out


def assert_served_is_backtest(models, switch, freq, engine, tmp_path, horizon=1, data=None, info=INFO, **frame_kwargs):
    data = frame(switch, freq, **frame_kwargs) if data is None else data
    pipeline = ModelPipeline(information_set=info, engine=engine)
    for model in models:
        pipeline.add_model(model)
    expected = backtest(pipeline, data, switch, freq, horizon, tmp_path / "backtest")
    got = served(pipeline, data, tmp_path / "live", days_of(switch), freq, horizon)
    for name, records in expected.items():
        assert set(got[name]) == set(records), name
        differ = [key for key in records if got[name][key] != records[key]]
        assert not differ, f"{name}: {len(differ)} of {len(records)} records differ, e.g. {differ[:3]}"
        assert any('"status": "ok"' in text for text in records.values()), name


def lear(window, solver="lars", **kwargs):
    return LEARModel(PREDICTORS, training_window=window, solver=solver, criterion="bic", **kwargs)


# Every level of model, refit, freq, switch and engine, and every pair of levels (see the module docstring).
CELLS = [
    ("ols", lambda: OLSModel(PREDICTORS, training_window=21, name="m"), None, "1h", SPRING, "off"),
    ("ols", lambda: OLSModel(PREDICTORS, 21, scaler="mlog", refit="7D", name="m"), "7D", "15min", AUTUMN, "on"),
    ("ols", lambda: OLSModel(PREDICTORS, 21, fit_intercept=True, refit="1MS", name="m"), "1MS", "1D", SPRING, "on"),
    ("ridge", lambda: RidgeModel(PREDICTORS, 21, refit="7D", name="m"), "7D", "1h", AUTUMN, "off"),
    ("ridge", lambda: RidgeModel(PREDICTORS, 21, name="m"), None, "15min", SPRING, "on"),
    ("ridge", lambda: RidgeModel(PREDICTORS, 21, scaler="npit", refit="1MS", name="m"), "1MS", "1D", AUTUMN, "off"),
    ("lasso", lambda: LassoCVModel(PREDICTORS, 21, cv=3, refit="1MS", name="m"), "1MS", "1h", SPRING, "on"),
    ("lasso", lambda: LassoCVModel(PREDICTORS, 21, cv=3, name="m"), None, "1D", AUTUMN, "on"),
    ("lasso", lambda: LassoCVModel(PREDICTORS, 21, cv=3, refit="7D", name="m"), "7D", "15min", SPRING, "off"),
    ("lear", lambda: lear([14, 21], refit="7D", name="m"), "7D", "1D", AUTUMN, "on"),
    ("lear-cd", lambda: lear([14, 21], solver="cd", name="m"), None, "1h", SPRING, "on"),
    ("lear", lambda: lear(21, refit="1MS", name="m"), "1MS", "15min", AUTUMN, "off"),
    ("lear-cd", lambda: lear(21, solver="cd", refit="7D", name="m"), "7D", "1h", AUTUMN, "off"),
    ("lear", lambda: lear(21, name="m"), None, "1D", SPRING, "off"),
]


@pytest.mark.parametrize(
    "kind, make, refit, freq, switch, engine",
    CELLS,
    ids=[f"{c[0]}-{c[2]}-{c[3]}-{'spring' if c[4] == SPRING else 'autumn'}-{c[5]}" for c in CELLS],
)
def test_a_forecast_served_from_a_stored_fit_is_the_backtest_s_record(
    kind, make, refit, freq, switch, engine, tmp_path
):
    assert_served_is_backtest([make()], switch, freq, engine, tmp_path)


@pytest.mark.parametrize("refit", [None, "7D"])
def test_the_mask_is_the_span_s_before_any_filter(refit, tmp_path):
    """`aux` is 0/1 on the rows a fit keeps and 0.5 on rows the finite filter drops (their load is missing): binary on the kept
    rows, scaled by the span's mask. A stored scaler with a mask of its own kept rows would scale another column set."""
    data = frame(SPRING, "1h")
    rng = np.random.default_rng(4)
    aux = (rng.random(len(data)) > 0.5).astype(float)
    dropped = rng.random(len(data)) < 0.05
    aux[dropped] = 0.5
    data = data.assign(aux=aux)
    data.loc[dropped, "load_d+1"] = np.nan
    info = InformationSet("11:45", "Europe/Warsaw", rules={**RULES, "aux": Availability.static()})
    model = OLSModel([*PREDICTORS, "aux"], 21, refit=refit, binary_mask="fit" if refit is None else None, name="m")
    assert_served_is_backtest([model], SPRING, "1h", "off", tmp_path, data=data, info=info)


@pytest.mark.parametrize("engine", ["off", "on"])
def test_auto_with_two_rules_and_two_horizons_is_the_backtest_s_record(engine, tmp_path):
    """`refit="auto"` over windows on both sides of 84 days: the short member refits daily, the long one weekly, so an issue day
    reads two anchors' fits (the long one stored once in its week), at two horizons, over the autumn switch."""
    model = lear([14, 90], refit="auto", name="m")
    assert_served_is_backtest([model], AUTUMN, "1h", engine, tmp_path, horizon=2, start="2023-07-10", days=125)


def test_a_real_pool_stores_what_an_in_process_one_does(tmp_path, monkeypatch):
    """The fits made in pool workers (one BLAS thread each, `_pool_init`) serve the backtest's records, made in its own pool."""
    from concurrent.futures import ProcessPoolExecutor

    for module in POOLED:
        monkeypatch.setattr(f"{module}.ProcessPoolExecutor", ProcessPoolExecutor)
    model = lear([14, 21], refit="7D", name="m")
    assert_served_is_backtest([model], SPRING, "1h", "on", tmp_path)


# --- if_due ------------------------------------------------------------------------------------------------------------


def pipeline_of(*models, engine="off", info=INFO):
    """A pipeline of `models`; `engine=None` is the default ("on")."""
    pipeline = ModelPipeline(information_set=info, engine=engine)
    for model in models:
        pipeline.add_model(model)
    return pipeline


def files_of(live):
    return {path: path.stat().st_mtime_ns for path in sorted((live / "fits").rglob("*")) if path.is_file()}


def test_a_second_call_is_a_no_op_and_another_configuration_refits(tmp_path):
    data = frame(SPRING, "1h")
    live = tmp_path / "live"
    pipeline = pipeline_of(OLSModel(PREDICTORS, 21, refit="7D", name="weekly"), OLSModel(PREDICTORS, 21, name="daily"))
    first = pipeline.refit(data, live, issue="2023-03-22", horizon=2)
    assert first.fitted == {"weekly": ["2023-03-20"], "daily": ["2023-03-22"]} and not first.noop
    assert first.units == {"weekly": 48, "daily": 48}
    files = files_of(live)
    again = pipeline.refit(data, live, issue="2023-03-22", horizon=2)
    assert again.noop and again.not_due == {"weekly": ["2023-03-20"], "daily": ["2023-03-22"]}
    assert files_of(live) == files
    # Later in the week the weekly model's anchor is stored: only the daily one is due.
    later = pipeline.refit(data, live, issue="2023-03-23", horizon=2)
    assert later.fitted == {"weekly": [], "daily": ["2023-03-23"]}
    # Another training window is another configuration: refitted, and the old fit is never served for it.
    pipeline.models[0].training_window = 28
    changed = pipeline.refit(data, live, issue="2023-03-22", horizon=2)
    assert changed.fitted["weekly"] == ["2023-03-20"] and changed.not_due == {"daily": ["2023-03-22"]}
    stored = json.loads((fit_dir(live, "weekly", datetime.date(2023, 3, 20)) / FIT_FILE).read_text())
    assert {unit["first_row"] for unit in stored["units"] if unit["key"][1] == 1} == {"2023-02-19"}  # 28 days back
    forced = pipeline.refit(data, live, issue="2023-03-22", horizon=2, if_due=False)
    assert forced.fitted == {"weekly": ["2023-03-20"], "daily": ["2023-03-22"]}


def test_a_fit_asked_for_before_its_newest_target_is_out_is_refused(tmp_path):
    """Decision 5. With `newest_target="published"` a forecast issued on 22 March trains on that day's prices (out at 13:00 on
    the 21st, Warsaw being CET then); a fit made at noon on the 21st would train on a day fewer than the backtest's."""
    data = frame(SPRING, "1h")
    live = tmp_path / "live"
    pipeline = pipeline_of(OLSModel(PREDICTORS, 21, newest_target="published", name="m"))
    with use_clock(FakeClock(pd.Timestamp("2023-03-21 11:59", tz="UTC"))):
        with pytest.raises(
            FitTooEarlyError, match=r"publishes at 2023-03-21 13:00 Europe/Warsaw \(2023-03-21 12:00 UTC\)"
        ):
            pipeline.refit(data, live, issue="2023-03-22", horizon=1)
        assert not (live / "fits").exists()  # nothing written
    with use_clock(FakeClock(pd.Timestamp("2023-03-21 12:00", tz="UTC"))):
        assert pipeline.refit(data, live, issue="2023-03-22", horizon=1).fitted == {"m": ["2023-03-22"]}
    # "d-1" trains on 21 March's prices, out at 13:00 on the 20th: a fit for the 22nd can be made on the evening of the 20th.
    daily = pipeline_of(OLSModel(PREDICTORS, 21, name="d1"))
    with use_clock(FakeClock(pd.Timestamp("2023-03-20 11:00", tz="UTC"))):
        with pytest.raises(FitTooEarlyError, match="'price' of 2023-03-21"):
            daily.refit(data, live, issue="2023-03-22", horizon=1)
    with use_clock(FakeClock(pd.Timestamp("2023-03-20 20:00", tz="UTC"))):
        assert daily.refit(data, live, issue="2023-03-22", horizon=1).fitted == {"d1": ["2023-03-22"]}


def test_a_fit_needs_the_target_s_rule(tmp_path):
    data = frame(SPRING, "1h")
    with pytest.raises(FitTooEarlyError, match="no information set"):
        pipeline_of(OLSModel(PREDICTORS, 21, name="m"), info=None).refit(
            data, tmp_path / "a", issue="2023-03-22", horizon=1
        )
    bare = InformationSet(
        "11:45", "Europe/Warsaw", rules={k: v for k, v in RULES.items() if k != "price"}, use_defaults=False
    )
    with pytest.raises(InformationSetError, match="'price' has none"):
        pipeline_of(OLSModel(PREDICTORS, 21, name="m"), info=bare).refit(
            data, tmp_path / "b", issue="2023-03-22", horizon=1
        )


# --- loading: digests and hashes ---------------------------------------------------------------------------------------


def serve_one(pipeline, data, live, day, horizon=2, **kwargs):
    return pipeline._serve_stored(data, live, issue=day, horizon=horizon, **kwargs)["m"]


def test_a_live_directory_from_before_the_engine_flip_refits_once_and_never_mixes_the_two_paths(tmp_path):
    """Fits stored with the engine off, as every live directory up to 2.3 holds them (`fit_path` None), are of another
    configuration digest than the default engine's: the default serves none of them, at their own anchor or as an older one,
    refits each anchor a day needs once, and from then on serves its own fits, the bits of a live directory made under the
    default from scratch. Every unit of a served fit is of one path; the engine-off pipeline never serves the new fits either."""
    data = frame(SPRING, "1h")
    live, fresh = tmp_path / "live", tmp_path / "fresh"

    def weekly(engine=None):
        model = OLSModel(PREDICTORS, 21, refit="7D", name="m")
        return pipeline_of(model, engine=engine)

    before = weekly("off")
    before.refit(data, live, issue="2023-03-17", horizon=2)  # anchor Monday 13 March
    before.refit(data, live, issue="2023-03-22", horizon=2)  # anchor Monday 20 March
    meta = json.loads((fit_dir(live, "m", datetime.date(2023, 3, 20)) / FIT_FILE).read_text())
    assert meta["fit_path"] is None and not any(unit["engine"] for unit in meta["units"])

    after = weekly()
    assert after.engine == "on"
    # Neither 20 March's own fit nor 13 March's as an older one: both are of the engine-off digest.
    missing = serve_one(after, data, live, "2023-03-22")
    assert missing.fits["2023-03-20"]["source"] == "missing"
    assert {r["status"] for r in missing.records} == {"no_stored_fit"}

    assert after.refit(data, live, issue="2023-03-22", horizon=2).fitted == {"m": ["2023-03-20"]}
    assert after.refit(data, live, issue="2023-03-22", horizon=2).noop  # once
    meta = json.loads((fit_dir(live, "m", datetime.date(2023, 3, 20)) / FIT_FILE).read_text())
    assert meta["fit_path"] == "engine-2" and all(unit["engine"] for unit in meta["units"])
    served = serve_one(after, data, live, "2023-03-22")
    assert served.fits["2023-03-20"]["source"] == "stored"
    weekly().refit(data, fresh, issue="2023-03-22", horizon=2)
    assert keyed(served.records) == keyed(serve_one(weekly(), data, fresh, "2023-03-22").records)
    # The old pipeline finds no fit of its own at 20 March now, and doesn't take the engine's.
    old = serve_one(before, data, live, "2023-03-22", fit_missing="skip")
    assert old.fits["2023-03-20"]["source"] == "missing"


@pytest.mark.parametrize("engine", ["off", None])
def test_a_live_directory_from_before_the_mask_flip_refits_once_and_never_serves_the_old_fits(tmp_path, engine):
    """`refit=None` fits stored under the first-day rule (`binary_mask="first_day"`, the default up to 2.3, record for record
    and manifest for manifest) on the model path: LEAR's coordinate descent, which the engine fits on the model path whatever
    the engine, so only the mask rule differs (`training_cut["binary_mask"]` in the configuration digest). The default serves
    none of them, at their own issue day or as an older one (it fits the day inline, or skips it), refits the day once, and
    then serves its own, the bits of a live directory made under the default from scratch."""
    data = frame(SPRING, "1h")
    live, fresh = tmp_path / "live", tmp_path / "fresh"

    def daily(**mask):
        return pipeline_of(lear(21, solver="cd", name="m", **mask), engine=engine)

    before = daily(binary_mask="first_day")
    for day in ("2023-03-21", "2023-03-22"):
        before.refit(data, live, issue=day, horizon=2)
    meta = json.loads((fit_dir(live, "m", datetime.date(2023, 3, 22)) / FIT_FILE).read_text())
    assert "binary_mask" not in meta["training_cut"] and meta["fit_path"] is None

    after = daily()
    # Neither the 22nd's own fit nor the 21st's as an older one: a refit=None day is fitted inline instead, or skipped.
    inline = serve_one(after, data, live, "2023-03-22")
    assert inline.fits["2023-03-22"]["source"] == "inline"
    skipped = serve_one(after, data, live, "2023-03-22", fit_missing="skip")
    assert skipped.fits["2023-03-22"]["source"] == "missing"
    assert {r["status"] for r in skipped.records} == {"no_stored_fit"}
    assert after.refit(data, live, issue="2023-03-22", horizon=2).fitted == {"m": ["2023-03-22"]}
    assert after.refit(data, live, issue="2023-03-22", horizon=2).noop  # once
    meta = json.loads((fit_dir(live, "m", datetime.date(2023, 3, 22)) / FIT_FILE).read_text())
    assert meta["training_cut"]["binary_mask"] == "fit" and meta["fit_path"] is None
    served = serve_one(after, data, live, "2023-03-22")
    assert served.fits["2023-03-22"]["source"] == "stored"
    daily().refit(data, fresh, issue="2023-03-22", horizon=2)
    assert (
        keyed(served.records) == keyed(serve_one(daily(), data, fresh, "2023-03-22").records) == keyed(inline.records)
    )
    assert serve_one(before, data, live, "2023-03-22", fit_missing="skip").fits["2023-03-22"]["source"] == "missing"


def test_a_tampered_units_file_raises_and_another_digest_is_never_served(tmp_path):
    data = frame(SPRING, "1h")
    live = tmp_path / "live"
    pipeline = pipeline_of(OLSModel(PREDICTORS, 21, refit="7D", name="m"))
    pipeline.refit(data, live, issue="2023-03-22", horizon=2)
    path = fit_dir(live, "m", datetime.date(2023, 3, 20)) / UNITS_FILE
    payload = bytearray(path.read_bytes())
    original = bytes(payload)
    payload[len(payload) // 2] ^= 0x01
    path.write_bytes(bytes(payload))
    with pytest.raises(StoredFitError, match="isn't the file .* names"):
        serve_one(pipeline, data, live, "2023-03-22")
    path.write_bytes(original)
    assert serve_one(pipeline, data, live, "2023-03-22").fits["2023-03-20"]["source"] == "stored"
    # fit.json says which configuration the fit is of: under another one it isn't served, nor taken as an older fit.
    pipeline.models[0].training_window = 28
    other = serve_one(pipeline, data, live, "2023-03-22", fit_missing="previous")
    assert other.fits["2023-03-20"] == {"source": "missing", "used": None, "fit_age_days": None}
    assert {r["status"] for r in other.records} == {"no_stored_fit"}
    # An unreadable fit.json is no fit, with a warning: under "skip" its forecasts are skips, and refit refits the anchor.
    pipeline.models[0].training_window = 21
    (fit_dir(live, "m", datetime.date(2023, 3, 20)) / FIT_FILE).write_text("{")
    with pytest.warns(UserWarning, match="can't be read"):
        broken = serve_one(pipeline, data, live, "2023-03-22", fit_missing="skip")
    assert broken.fits["2023-03-20"]["source"] == "missing"
    with pytest.warns(UserWarning, match="refitted whole"):
        assert pipeline.refit(data, live, issue="2023-03-22", horizon=2).fitted == {"m": ["2023-03-20"]}
    assert serve_one(pipeline, data, live, "2023-03-22").fits["2023-03-20"]["source"] == "stored"


def test_a_missing_fit_serves_the_newest_older_one_with_its_age(tmp_path):
    """Decision 15: no refit on the Monday of a weekly model leaves the last stored fit in use, its age recorded, and its
    forecasts are those of the older anchor's fit (its own `refit_anchor`), the same bits a backtest of that block gives."""
    data = frame(SPRING, "1h")
    live = tmp_path / "live"
    pipeline = pipeline_of(OLSModel(PREDICTORS, 21, refit="7D", name="m"))
    pipeline.refit(data, live, issue="2023-03-24", horizon=2)  # anchor Monday 20 March
    with pytest.warns(UserWarning, match="served from the fit of 2023-03-20, 7 day"):
        result = serve_one(pipeline, data, live, "2023-03-28")  # anchor Monday 27 March: not stored
    # The age is the issue day's: the fit served is of 20 March, the forecast of the 28th.
    assert result.fits == {"2023-03-27": {"source": "previous", "used": "2023-03-20", "fit_age_days": 8}}
    forecasts = [r for r in result.records if r["status"] == "ok"]
    assert forecasts and {r["refit_anchor"] for r in forecasts} == {"2023-03-20"}
    # The same as serving the 28th from the 20th's fit: what the 20th's fit forecasts from the 28th's rows.
    inline = serve_one(pipeline, data, live, "2023-03-28", fit_missing="fit")
    assert inline.fits == {"2023-03-27": {"source": "inline", "used": "2023-03-27", "fit_age_days": 1}}
    assert {r["refit_anchor"] for r in inline.records if r["status"] == "ok"} == {"2023-03-27"}
    assert [r["prediction"] for r in inline.records] != [r["prediction"] for r in forecasts]
    skipped = serve_one(pipeline, data, live, "2023-03-28", fit_missing="skip")
    assert {r["status"] for r in skipped.records} == {"no_stored_fit"}
    assert len(skipped.records) == len(result.records)
    # No older fit at all: nothing to serve.
    empty = serve_one(pipeline, data, tmp_path / "empty", "2023-03-28")
    assert empty.fits["2023-03-27"]["source"] == "missing"


def test_an_inline_fit_is_the_stored_one(tmp_path):
    """`fit_missing="fit"`, and a refit=None model without a stored fit, fit the anchor inline: the backtest's record too."""
    data = frame(AUTUMN, "1h")
    for model in (OLSModel(PREDICTORS, 21, refit="7D", name="m"), lear([14, 21], name="m")):
        pipeline = pipeline_of(model, engine="on")
        expected = backtest(pipeline, data, AUTUMN, "1h", 2, tmp_path / f"bt-{model.refit}")["m"]
        got = {}
        for day in days_of(AUTUMN):
            result = serve_one(pipeline, data, None, day, fit_missing="fit" if model.refit else "previous")
            assert {entry["source"] for entry in result.fits.values()} == {"inline"}
            got.update(keyed(result.records))
        assert got == expected


def test_no_warm_start_the_order_of_refits_moves_no_bit(tmp_path):
    """A stored fit depends on its anchor and the configuration alone: fitted on its own, after other anchors or in reverse."""
    data = frame(SPRING, "1h")
    days = days_of(SPRING)[1:4]
    stored = {}
    for name, order in (("forward", days), ("backward", days[::-1])):
        live = tmp_path / name
        pipeline = pipeline_of(lear([14, 21], name="m"), engine="on")
        for day in order:
            pipeline.refit(data, live, issue=day, horizon=1)
        stored[name] = {day: np.load(fit_dir(live, "m", day) / UNITS_FILE, allow_pickle=False) for day in days}
    for day in days:
        forward, backward = stored["forward"][day], stored["backward"][day]
        assert sorted(forward.files) == sorted(backward.files)
        for key in forward.files:
            assert np.array_equal(forward[key], backward[key]), (day, key)


# --- what isn't stored, and the live directory ------------------------------------------------------------------------


def test_families_without_a_format_fit_at_serve_time(tmp_path):
    from epftoolbox2.models import HistGBModel

    data = frame(SPRING, "1h")
    naive = NaiveModel(["price_d-1"], name="naive")
    trees = HistGBModel(PREDICTORS, training_window=21, name="trees")
    assert "fits nothing" in stores_fits(naive) and "fits at serve time" in stores_fits(trees)
    pipeline = pipeline_of(naive, trees, OLSModel(PREDICTORS, 21, name="ols"))
    result = pipeline.refit(data, tmp_path / "live", issue="2023-03-22", horizon=1)
    assert set(result.not_stored) == {"naive", "trees"} and result.fitted == {"ols": ["2023-03-22"]}
    served = pipeline._serve_stored(data, tmp_path / "live", issue="2023-03-22", horizon=1)
    assert {name: result.status for name, result in served.items()} == {
        "naive": "not_stored",
        "trees": "not_stored",
        "ols": "served",
    }
    assert served["trees"].records == [] and "serve time" in served["trees"].reason


def test_a_subclass_with_another_estimator_stores_no_fit():
    class Other(OLSModel):
        def _fit_model(self, train_x, train_y):
            return super()._fit_model(train_x, train_y)

        def _predict_fitted(self, fitted, test_x):
            return super()._predict_fitted(fitted, test_x)

        def _fit_predict(self, train_x, train_y, test_x):
            return self._predict_fitted(self._fit_model(train_x, train_y), test_x)

    assert "no stored format" in stores_fits(Other(PREDICTORS))


def test_refit_writes_into_a_live_directory_only_and_holds_its_lock(tmp_path):
    data = frame(SPRING, "1h")
    pipeline = pipeline_of(OLSModel(PREDICTORS, 21, name="m"))
    backtest_dir = tmp_path / "backtest"
    pipeline.run(data, "2023-03-22", "2023-03-22", horizon=1, save_dir=str(backtest_dir))
    with pytest.raises(LiveDirectoryError, match="holds a backtest's results"):
        pipeline.refit(data, backtest_dir, issue="2023-03-22", horizon=1)
    live = tmp_path / "live"
    pipeline.refit(data, live, issue="2023-03-22", horizon=1)
    assert LiveStore(live).identity() == {"mode": "live", "issue_timezone": "Europe/Warsaw"}
    with LiveStore(live).hold():
        with pytest.raises(LiveBusyError):
            pipeline.refit(data, live, issue="2023-03-22", horizon=1)
    with pytest.raises(LiveDirectoryError):
        pipeline.run(data, "2023-03-22", "2023-03-22", horizon=1, save_dir=str(live))


def test_fit_json_says_what_the_fit_is_of_and_what_it_read(tmp_path):
    data = frame(SPRING, "1h")
    live = tmp_path / "live"
    with use_clock(FakeClock(pd.Timestamp("2023-03-21 20:00", tz="UTC"))):
        pipeline_of(lear([14, 21], refit="7D", name="m"), engine="on").refit(data, live, issue="2023-03-22", horizon=2)
    directory = fit_dir(live, "m", datetime.date(2023, 3, 20))
    meta = json.loads((directory / FIT_FILE).read_text())
    assert meta["anchor"] == "2023-03-20" and meta["issue"] == "2023-03-22" and meta["fit_path"] == "engine-2"
    assert meta["created_at"] == "2023-03-21T20:00:00+00:00" and meta["refit"] == "7D"
    assert meta["training_cut"] == {"rows": "dates", "refit": "7D", "refit_epoch": "1970-01-05"}
    assert meta["rows"] == {"from": "2023-02-25", "through": "2023-03-19", "columns": meta["rows"]["columns"]}
    assert all(len(unit["rows_hash"]) == 64 for unit in meta["units"])
    assert (
        meta["artifacts"][UNITS_FILE] == __import__("hashlib").sha256((directory / UNITS_FILE).read_bytes()).hexdigest()
    )
    units = meta["units"]
    assert len(units) == 24 * 2 * 2  # periods x horizons x windows
    first = next(u for u in units if u["key"][:2] == [0, 1] and u["key"][3] == 0)
    assert first["window"] == 14 and first["newest_target_day"] == "2023-03-19" and first["engine"] is True
    assert (first["first_row"], first["last_row"], first["rows"]) == ("2023-03-05", "2023-03-18", 14)
    with np.load(directory / UNITS_FILE, allow_pickle=False) as npz:
        assert all(npz[name].dtype != object for name in npz.files)


# --- the models' states -------------------------------------------------------------------------------------------------


def bits(values):
    return np.asarray(values, dtype=np.float64).view(np.int64)


def round_trip(state):
    import io

    buffer = io.BytesIO()
    np.savez(buffer, **state)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


@pytest.mark.parametrize(
    "model",
    [
        OLSModel(PREDICTORS),
        OLSModel(PREDICTORS, fit_intercept=True),
        RidgeModel(PREDICTORS),
        RidgeModel(PREDICTORS, alphas=[0.5, 3.0], fit_intercept=False),
        LassoCVModel(PREDICTORS, cv=3),
        LassoCVModel(PREDICTORS, fit_intercept=False),
        LEARModel(PREDICTORS),
        LEARModel(PREDICTORS, solver="cd", criterion="aic"),
        LEARModel(PREDICTORS, criterion="cv"),
    ],
    ids=lambda m: f"{type(m).__name__}-{m._model_kwargs}",
)
def test_a_model_s_fit_state_forecasts_to_the_bit(model):
    rng = np.random.default_rng(7)
    for _ in range(10):
        n, p = int(rng.integers(12, 60)), int(rng.integers(1, 8))
        x = rng.normal(size=(n, p))
        y = x @ rng.normal(size=p) + rng.normal(0, 0.5, n)
        fitted = model._fit_model(x, y)
        restored = model._from_state(round_trip(model._fit_state(fitted)))
        for row in rng.normal(0, 3, (6, p)):
            mine, theirs = model._predict_fitted(restored, row[None, :]), model._predict_fitted(fitted, row[None, :])
            assert bits(mine[0]) == bits(theirs[0]) and np.array_equal(bits(mine[1]), bits(theirs[1]))


def test_an_engine_fit_s_state_forecasts_to_the_bit(tmp_path):
    """The engine's fits from shared statistics as stored (`engine_fit_state`) forecast each row as the fit made in the job,
    rows far enough out to go to the model path included: OLS's solve error, Ridge's rounding, LEAR's metric, a VST's row set."""
    from epftoolbox2.models._fits import _npz_units, _section, engine_fit_from_state, read_fit

    data = frame(SPRING, "1h")
    rng = np.random.default_rng(0)
    for model in (
        OLSModel(PREDICTORS, 21, refit="7D", name="m"),
        RidgeModel(PREDICTORS, 21, refit="7D", scaler="asinh", name="m"),
        lear(21, refit="7D", name="m"),
    ):
        live = tmp_path / type(model).__name__
        pipeline_of(model, engine="on").refit(data, live, issue="2023-03-22", horizon=1)
        directory = fit_dir(live, "m", datetime.date(2023, 3, 20))
        stored = read_fit(directory, model)
        meta = json.loads((directory / FIT_FILE).read_text())
        groups = _npz_units(directory / UNITS_FILE, len(meta["units"]))
        made = 0
        for place, entry in enumerate(meta["units"]):
            unit = stored.units[tuple(entry["key"])]
            assert entry["engine"] == (unit.engine is not None)
            if unit.engine is None:
                continue
            made += 1
            again = engine_fit_from_state(_section(groups[place], "engine"), model)
            for scale in (1.0, 50.0, 1e4):
                row = rng.normal(0, scale, (1, 4)) + np.array([[1000.0, 100.0, 0.0, 1.0]])
                assert unit.engine.forecast(model, row) == again.forecast(model, row)
        assert made >= 12, type(model).__name__


# --- review round 1: what the frame holds, formats, kernels, rules over time, notes, writes --------------------------------


def backtest_of_day(pipeline, data, day, tmp_path, name="m", horizon=1):
    pipeline.run(data, day, day, horizon=horizon, save_dir=str(tmp_path))
    lines = (tmp_path / f"{name}.jsonl").read_text().splitlines()
    return keyed(json.loads(line) for line in lines if line.strip())


def test_a_fit_from_a_frame_missing_its_newest_target_is_marked_and_refitted_once_it_is_there(tmp_path):
    """The rule says 21 March's prices are out, the frame doesn't hold them yet: the fit is stored with 20 rows, marked
    incomplete with a warning; the next refit on the complete frame finds the rows it read moved (`stale`), refits the anchor
    whole, and serves the backtest's records. A revised price inside the rows read refits too; one before them doesn't."""
    full = frame(SPRING, "1h")
    partial = full.copy()
    partial.loc["2023-03-21", "price"] = np.nan
    live = tmp_path / "live"
    pipeline = pipeline_of(OLSModel(PREDICTORS, 21, name="m"))
    with use_clock(FakeClock(pd.Timestamp("2023-03-21 20:00", tz="UTC"))):
        with pytest.warns(UserWarning, match="doesn't hold every 'price' value of 2023-03-21"):
            first = pipeline.refit(partial, live, issue="2023-03-22", horizon=1)
        assert first.incomplete == {"m": ["2023-03-22"]}
        meta = json.loads((fit_dir(live, "m", datetime.date(2023, 3, 22)) / FIT_FILE).read_text())
        assert meta["incomplete"] == ["2023-03-21"] and meta["units"][0]["rows"] == 20
        again = pipeline.refit(full, live, issue="2023-03-22", horizon=1)
        assert again.stale == {"m": ["2023-03-22"]} and again.fitted == {"m": ["2023-03-22"]} and not again.incomplete
        assert pipeline.refit(full, live, issue="2023-03-22", horizon=1).noop
        served = keyed(serve_one_day(pipeline, full, live, "2023-03-22"))
    assert served == backtest_of_day(
        pipeline_of(OLSModel(PREDICTORS, 21, name="m")), full, "2023-03-22", tmp_path / "bt"
    )
    revised = full.copy()
    revised.loc["2023-03-10 05:00", "price"] += 1.0
    with use_clock(FakeClock(pd.Timestamp("2023-03-21 20:00", tz="UTC"))):
        assert pipeline.refit(revised, live, issue="2023-03-22", horizon=1).stale == {"m": ["2023-03-22"]}
        earlier = revised.copy()
        earlier.loc["2023-02-01 05:00", "price"] += 1.0  # before the rows the fit reads
        assert pipeline.refit(earlier, live, issue="2023-03-22", horizon=1).noop


def serve_one_day(pipeline, data, live, day, horizon=1):
    return pipeline._serve_stored(data, live, issue=day, horizon=horizon)["m"].records


def test_an_input_of_the_newest_training_row_not_yet_out_refuses_the_fit(tmp_path):
    """Issued at 14:00, a day's load forecast is out at 13:30 the day before, after its price (13:00): at 13:15 on 20 March the
    newest target of a fit for the 22nd is out, the newest training row's `load_d+1` isn't."""
    rules = {
        **RULES,
        "load": Availability.published("13:30", 1, timezone="Europe/Warsaw"),
        "price": Availability.published("13:00", 1, timezone="Europe/Warsaw"),
    }
    pipeline = pipeline_of(
        OLSModel(PREDICTORS, 21, name="m"), info=InformationSet("14:00", "Europe/Warsaw", rules=rules)
    )
    data = frame(SPRING, "1h")
    with use_clock(FakeClock(pd.Timestamp("2023-03-20 12:15", tz="UTC"))):
        with pytest.raises(FitTooEarlyError, match=r"reads 'load_d\+1' on its newest training row, 2023-03-20 .*13:30"):
            pipeline.refit(data, tmp_path / "live", issue="2023-03-22", horizon=1)
    assert not (tmp_path / "live" / "live.json").exists()  # refused before anything of the live directory is written
    with use_clock(FakeClock(pd.Timestamp("2023-03-20 12:30", tz="UTC"))):
        assert pipeline.refit(data, tmp_path / "live", issue="2023-03-22", horizon=1).fitted == {"m": ["2023-03-22"]}


def test_a_rule_over_time_is_judged_over_the_whole_span(tmp_path):
    """Before 6 February a day's price was out 45 days later: a fit for 22 March over 50 days reads the prices of 4 and 5
    February, out on 21 and 22 March, so at 20:00 UTC on the 20th it is refused, though its newest target is out."""
    chain = [
        Availability.published("10:00", -45, timezone="Europe/Warsaw").during(valid_until="2023-02-05"),
        Availability.published("13:00", 1, timezone="Europe/Warsaw").during(valid_from="2023-02-06"),
    ]
    pipeline = pipeline_of(
        OLSModel(PREDICTORS, 50, name="m"),
        info=InformationSet("11:45", "Europe/Warsaw", rules={**RULES, "price": chain}),
    )
    data = make_market_data(start="2023-01-01", days=100, freq="1h", horizon=2, noise=2.0)
    with use_clock(FakeClock(pd.Timestamp("2023-03-20 20:00", tz="UTC"))):
        with pytest.raises(FitTooEarlyError, match="'price' of 2023-02-05 .* publishes at 2023-03-22 10:00"):
            pipeline.refit(data, tmp_path / "live", issue="2023-03-22", horizon=1)
    with use_clock(FakeClock(pd.Timestamp("2023-03-22 09:00", tz="UTC"))):
        assert pipeline.refit(data, tmp_path / "live", issue="2023-03-22", horizon=1).fitted == {"m": ["2023-03-22"]}


def rewrite_meta(directory, **changes):
    path = directory / FIT_FILE
    meta = json.loads(path.read_text())
    meta.update(changes)
    path.write_text(json.dumps(meta))


def test_another_format_or_kernel_backend_is_due_and_another_version_warns(tmp_path):
    data = frame(SPRING, "1h")
    live = tmp_path / "live"
    pipeline = pipeline_of(lear(21, refit="7D", name="m"))
    pipeline.refit(data, live, issue="2023-03-22", horizon=1)
    directory = fit_dir(live, "m", datetime.date(2023, 3, 20))
    rewrite_meta(directory, format=_fits.FORMAT - 1)
    assert pipeline.refit(data, live, issue="2023-03-22", horizon=1).fitted == {"m": ["2023-03-20"]}
    other = "numpy" if pipeline.models[0]._kernel_backend == "numba" else "numba"
    rewrite_meta(directory, kernels=other)  # fit.json alone: units.npz is still the file it names
    assert (
        serve_one(pipeline, data, live, "2023-03-22", horizon=1, fit_missing="skip").fits["2023-03-20"]["source"]
        == "missing"
    )
    assert pipeline.refit(data, live, issue="2023-03-22", horizon=1).fitted == {"m": ["2023-03-20"]}
    rewrite_meta(directory, package_version="0.0.1")
    with pytest.warns(UserWarning, match="made by epftoolbox2 0.0.1"):
        assert serve_one(pipeline, data, live, "2023-03-22", horizon=1).fits["2023-03-20"]["source"] == "stored"


def test_an_unreadable_older_fit_is_passed_over_and_an_old_fit_has_a_maximum_age(tmp_path):
    data = frame(SPRING, "1h", days=110)
    live = tmp_path / "live"
    pipeline = pipeline_of(OLSModel(PREDICTORS, 21, refit="7D", name="m"))
    pipeline.refit(data, live, issue="2023-03-14", horizon=1)  # anchor 13 March
    pipeline.refit(data, live, issue="2023-03-21", horizon=1)  # anchor 20 March
    (fit_dir(live, "m", datetime.date(2023, 3, 20)) / FIT_FILE).write_text("not json")
    with pytest.warns(UserWarning, match="passed over"):
        result = serve_one(pipeline, data, live, "2023-03-28", horizon=1)
    assert result.fits["2023-03-27"] == {"source": "previous", "used": "2023-03-13", "fit_age_days": 15}
    # 13 March stands in for no anchor more than PREVIOUS_MAX_DAYS later.
    with pytest.warns(UserWarning, match="more than the 31 an older fit is served for"):
        late = serve_one(pipeline, data, live, "2023-04-18", horizon=1)
    assert _fits.previous_cap(pipeline.models[0]) == 31
    assert late.fits["2023-04-17"] == {
        "source": "missing (older fit beyond cap)",
        "used": None,
        "fit_age_days": None,
        "beyond_cap": "2023-03-13",
        "cap_days": 31,
    }


def test_a_refit_none_model_says_its_mask_is_the_issue_day_s(tmp_path):
    """Only the `binary_mask="first_day"` rule (the default up to 2.3) says so; the default reads each fit's own rows."""
    data = frame(SPRING, "1h")
    live = tmp_path / "live"
    plain = pipeline_of(OLSModel(PREDICTORS, 21, binary_mask="first_day", name="m"))
    plain.refit(data, live, issue="2023-03-22", horizon=1)
    assert json.loads((fit_dir(live, "m", datetime.date(2023, 3, 22)) / FIT_FILE).read_text())["notes"] == [
        _fits.MASK_ISSUE_DAY_NOTE
    ]
    assert plain._serve_stored(data, live, issue="2023-03-22", horizon=1)["m"].notes == [_fits.MASK_ISSUE_DAY_NOTE]
    for own in (OLSModel(PREDICTORS, 21, binary_mask="fit", name="m"), OLSModel(PREDICTORS, 21, name="m")):
        assert pipeline_of(own)._serve_stored(data, None, issue="2023-03-22", horizon=1)["m"].notes == []
    plan = type("Plan", (), {"model": OLSModel(PREDICTORS, 21, refit="7D")})()
    units = [{"engine": True, "fit_error": "fit_error: x"}, {"engine": True, "fit_error": None}]
    assert _fits.anchor_notes(plan, units) == [_fits.ENGINE_FIT_ERROR_NOTE.format(n=1)]


def test_a_write_stopped_between_its_renames_keeps_the_old_fit(tmp_path):
    """The old fit is renamed aside before the new one moves in: a crash between leaves it, which serving reads and the next
    refit puts back; staging debris is swept."""
    data = frame(SPRING, "1h")
    live = tmp_path / "live"
    pipeline = pipeline_of(OLSModel(PREDICTORS, 21, refit="7D", name="m"))
    pipeline.refit(data, live, issue="2023-03-22", horizon=1)
    directory = fit_dir(live, "m", datetime.date(2023, 3, 20))
    expected = keyed(serve_one_day(pipeline, data, live, "2023-03-22"))
    aside = directory.parent / ".2023-03-20.old-abc"
    os.replace(directory, aside)
    (directory.parent / ".2023-03-20.new-def").mkdir()
    assert keyed(serve_one_day(pipeline, data, live, "2023-03-22")) == expected
    assert pipeline.refit(data, live, issue="2023-03-22", horizon=1).noop  # swept: put back, found, not due
    assert sorted(p.name for p in directory.parent.iterdir()) == ["2023-03-20"]


def test_units_a_stored_fit_lacks_are_fitted_inline(tmp_path):
    """A stored anchor without some of the units the day needs (here the stored fit of horizon 1 alone, written by hand) has the
    others fitted inline for a refit=None model, as the backtest does, and served to its bit."""
    data = frame(SPRING, "1h")
    live = tmp_path / "live"
    pipeline = pipeline_of(OLSModel(PREDICTORS, 21, name="m"))
    pipeline.refit(data, live, issue="2023-03-22", horizon=2)
    directory = fit_dir(live, "m", datetime.date(2023, 3, 22))
    plan = _fits.plan_issue(
        pipeline.models[0],
        data,
        datetime.date(2023, 3, 22),
        target="price",
        horizon=2,
        horizons=[1, 2],
        freq="1h",
        information_set=INFO,
        availability="warn",
        issue_days=None,
        engine=False,
    )
    keys = _fits._sorted_keys(k for k in plan.items[datetime.date(2023, 3, 22)] if k[1] == 1)
    entries, _ = _fits.fitted_units(plan, datetime.date(2023, 3, 22), keys)
    meta = _fits.anchor_meta(plan, datetime.date(2023, 3, 22), [m for m, _ in entries], data)
    _fits.write_fit(directory, meta, entries)
    result = pipeline._serve_stored(data, live, issue="2023-03-22", horizon=2)["m"]
    assert result.fits["2023-03-22"]["source"] == "stored" and result.fits["2023-03-22"]["inline_units"] == 24
    expected = backtest_of_day(
        pipeline_of(OLSModel(PREDICTORS, 21, name="m")), data, "2023-03-22", tmp_path / "bt", horizon=2
    )
    assert keyed(result.records) == expected


def exaa_when_out(ctx):
    """A list that switches on `wind` (standing in for EXAA) where it is published: the refit's plan and a later call's differ."""
    return PREDICTORS if ctx.published("wind_d+{horizon}") else [p for p in PREDICTORS if not p.startswith("wind")]


@pytest.mark.parametrize("fit_missing", ["previous", "skip"])
def test_a_list_the_stored_anchor_lacks_is_fitted_inline_at_the_anchor_under_previous(tmp_path, fit_missing):
    """A refit model's anchor stored by a refit whose plan gave the list without `wind` (out after the refit ran): a call that
    needs the list with it gets those units fitted inline at the anchor, the backtest's record to the bit, under the default
    "previous" (not `no_stored_fit` skips, which took the first rung out of a fallback ladder on every last revision of a
    day whose function switches on a late input); "skip" still skips them."""
    data = frame(SPRING, "1h")
    early = data.copy()
    early.loc["2023-03-20", ["wind_d+1", "wind_d+2"]] = np.nan  # the anchor's issue day, before `wind` was out
    live = tmp_path / "live"

    def make():
        return pipeline_of(OLSModel(exaa_when_out, 21, name="m", refit="7D"))

    make().refit(early, live, issue="2023-03-20", horizon=2)
    result = serve_one(make(), data, live, "2023-03-22", fit_missing=fit_missing)
    entry = result.fits["2023-03-20"]
    assert entry["source"] == "stored"
    if fit_missing == "skip":
        assert "inline_units" not in entry and {r["status"] for r in result.records} == {"no_stored_fit"}
        return
    assert entry["inline_units"] == 48
    expected = backtest_of_day(make(), data, "2023-03-22", tmp_path / "bt", horizon=2)
    assert keyed(result.records) == expected


# --- review round 2 -----------------------------------------------------------------------------------------------------


def test_an_incomplete_fit_says_so_when_served_and_fit_refits_it_from_a_frame_that_holds_the_rows(tmp_path):
    """A separate serving process finds a fit made from a frame one day short: the anchor's entry, a note and a warning say it
    was incomplete and that the frame now holds what it lacked; `fit_missing="fit"` fits it inline from the frame, the backtest's
    records to the bit."""
    full = frame(SPRING, "1h")
    partial = full.copy()
    partial.loc["2023-03-21", "price"] = np.nan
    live = tmp_path / "live"
    pipeline = pipeline_of(OLSModel(PREDICTORS, 21, name="m"))
    with use_clock(FakeClock(pd.Timestamp("2023-03-21 20:00", tz="UTC"))):
        with pytest.warns(UserWarning, match="doesn't hold every"):
            pipeline.refit(partial, live, issue="2023-03-22", horizon=1)
        with pytest.warns(UserWarning) as caught:
            served = serve_one(pipeline, full, live, "2023-03-22", horizon=1)
        assert any("without every 'price' value of 2023-03-21" in str(w.message) for w in caught)
        assert any("served as it is" in str(w.message) for w in caught)
        entry = served.fits["2023-03-22"]
        assert entry["source"] == "stored" and entry["incomplete"] == ["2023-03-21"] and entry["stale"] is True
        assert any(note.startswith("incomplete:") for note in served.notes)
        assert any(note.startswith("stale:") for note in served.notes)
        with pytest.warns(UserWarning, match="fitted inline from the frame"):
            refitted = serve_one(pipeline, full, live, "2023-03-22", horizon=1, fit_missing="fit")
    assert refitted.fits["2023-03-22"]["source"] == "inline" and refitted.fits["2023-03-22"]["replaced"] == "stale"
    expected = backtest_of_day(pipeline_of(OLSModel(PREDICTORS, 21, name="m")), full, "2023-03-22", tmp_path / "bt")
    assert keyed(refitted.records) == expected
    assert keyed(served.records) != expected


def test_the_rows_hash_covers_the_cells_a_fit_reads_alone(tmp_path):
    """A weekly fit of 20 March reads the rows up to 18 March (newest target 19 March at h = 1). A revised day-ahead forecast on
    the row of 19 March, which no unit reads, leaves it as it is; one on a training row refits it."""
    data = frame(SPRING, "1h")
    live = tmp_path / "live"
    pipeline = pipeline_of(OLSModel(PREDICTORS, 21, refit="7D", name="m"))
    pipeline.refit(data, live, issue="2023-03-22", horizon=1)
    unread = data.copy()
    unread.loc["2023-03-19", "load_d+1"] += 50.0
    assert pipeline.refit(unread, live, issue="2023-03-22", horizon=1).noop
    read = unread.copy()
    read.loc["2023-03-18 07:00", "load_d+1"] += 50.0
    assert pipeline.refit(read, live, issue="2023-03-22", horizon=1).stale == {"m": ["2023-03-20"]}


def test_the_age_cap_is_at_least_two_refit_periods(tmp_path):
    """A model refitted every two months still has its last fit served after a missed refit, and one past the cap says so."""
    data = make_market_data(start="2022-11-01", days=260, freq="1h", horizon=2, noise=2.0)
    live = tmp_path / "live"
    pipeline = pipeline_of(OLSModel(PREDICTORS, 21, refit="2MS", name="m"))
    assert _fits.previous_cap(pipeline.models[0]) == 124
    pipeline.refit(data, live, issue="2023-01-10", horizon=1)  # anchor 1 January
    with pytest.warns(UserWarning, match="served from the fit of 2023-01-01, 59 day"):
        served = serve_one(pipeline, data, live, "2023-03-10", horizon=1)  # anchor 1 March: not stored
    assert served.fits["2023-03-01"] == {"source": "previous", "used": "2023-01-01", "fit_age_days": 68}
    with pytest.warns(UserWarning, match="181 days older, more than the 124"):
        beyond = serve_one(pipeline, data, live, "2023-07-10", horizon=1)
    assert beyond.fits["2023-07-01"]["source"] == "missing (older fit beyond cap)"


# --- WCSS: a strided issue row ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("scaler", ["standard", "mlog", "npit"])
def test_every_issue_row_reaches_the_model_as_a_c_contiguous_row(scaler, tmp_path, monkeypatch):
    """A refit unit's issue rows are taken from the period's arrays by fancy indexing, which gives a Fortran-ordered block; a
    VST's `fit_transform` kept that order, so with several days in a block the model got a strided row. numpy multiplies a
    strided row by another kernel than a contiguous one, and on an AVX-512 node (WCSS's Xeon 8268) the two differ in the last
    bit: a forecast then depended on how many days its block call held, and a served one (a call of one day) differed from the
    backtest's (`ols-7D-15min-autumn-on`, 1 of 576 records). Every row must reach `_predict_fitted` C-contiguous, in a backtest
    of a block and when served."""
    seen = []
    real = OLSModel._predict_fitted

    def probe(self, fitted, test_x):
        seen.append(bool(np.asarray(test_x).flags.c_contiguous))
        return real(self, fitted, test_x)

    monkeypatch.setattr(OLSModel, "_predict_fitted", probe)
    data = frame(AUTUMN, "1h")
    pipeline = pipeline_of(OLSModel(PREDICTORS, 21, scaler=scaler, refit="7D", name="m"))
    backtest(pipeline, data, ("2023-10-23", "2023-10-26"), "1h", 1, tmp_path / "bt")
    assert seen and all(seen), f"{seen.count(False)} of {len(seen)} backtest rows strided"
    seen.clear()
    pipeline.refit(data, tmp_path / "live", issue="2023-10-25", horizon=1)
    pipeline._serve_stored(data, tmp_path / "live", issue="2023-10-25", horizon=1)
    assert seen and all(seen), f"{seen.count(False)} of {len(seen)} served rows strided"
