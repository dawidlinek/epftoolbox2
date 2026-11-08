"""`binary_mask=`: which training rows decide the columns a fit leaves unscaled because they are binary 0/1.

None, the default, is today's rule: without `refit` a list's mask is read on the first calendar day of its window
(`BaseModel._scalable_mask`, `tests/models/test_masks.py`), with `refit` from each unit's rows (`_worker._fit_unit`).
`"first_day"` names the first and raises with `refit`. `"fit"` reads each fit's mask from the rows of its own span, before the
matching and finite filters (`_worker._fit_member`), member by member, so that no forecast depends on where its test starts.

These tests hold `"fit"` to `refit="1D"` record for record (a unit of one day is a fit with its own mask), to runs starting 1, 3
and 17 days later on every common key, to everything after the issue day, to a hand OLS, and count by hand the keys where it differs
from the default on a frame with night-time solar. That the default depends on `test_start` is pinned as a strict xfail in
`tests/test_known_issues.py`. Records compare as JSON text, so a NaN `actual` equals itself.
"""

import datetime
import json
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import LassoCVModel, LEARModel, NaiveModel, OLSModel, RidgeModel, Use
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path
from tests.synthetic import make_market_data

WINDOW = 21
# No column of these is constant on the rows, nor any sum of them, so centring a column moves a forecast: a mask that differs
# shows in the numbers, not only in the coefficients' scale.
NO_LEVEL = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}"]
MASKED = NO_LEVEL + ["turning"]
RULES = {
    "load": Availability.published("10:00", 10),
    "wind": Availability.published("10:00", 10),
    "is_weekend": Availability.static(),
    "turning": Availability.static(),
    "aux": Availability.static(),
    "price": Availability.published("13:00", 1),  # D's price is known at 11:45 on D: what "published" trains on
}
INFO = InformationSet("11:45", "Europe/Brussels", rules=RULES)
PROVENANCE = ("package_version", "git_sha", "created_at", "updated_at")


class InProcessPool:
    """Stands in for the worker pool, whose start is most of the time these runs would take."""

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


def day(text):
    return datetime.date.fromisoformat(text)


def shift(text, days):
    return (day(text) + datetime.timedelta(days=days)).isoformat()


def local_dates(data):
    return np.asarray(pd.DatetimeIndex(data.index.tz_localize(None)).normalize())


def keyed(records, *drop):
    """Key -> the record as JSON text without `drop`."""
    return {
        (r["run_date"], r["hour"], r["horizon"]): json.dumps(
            {k: v for k, v in r.items() if k not in drop}, sort_keys=True
        )
        for r in records
    }


def fits(records):
    """Key -> (prediction, coefficients): what a fit made of a forecast. `actual` is read after D."""
    return {(r["run_date"], r["hour"], r["horizon"]): (r["prediction"], r["coefficients"]) for r in records}


def lines(path, *drop):
    """A result file's records as sorted JSON text without `drop`: the order a pool finishes them in isn't fixed."""
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            out.append(json.dumps({k: v for k, v in json.loads(line).items() if k not in drop}, sort_keys=True))
    return sorted(out)


def manifest_of(path):
    manifest = json.loads(manifest_path(path).read_text())
    for name in PROVENANCE:
        manifest.pop(name)
    return manifest


def with_turning(data, turns, seed=4, reverse=False):
    """`turning`: random 0/1 on the rows dated before `turns` and continuous from then on, or the other way round with `reverse`.
    A mask read on rows before `turns` leaves it unscaled, one read on rows from `turns` on scales it."""
    rng = np.random.default_rng(seed)
    values = (rng.random(len(data)) > 0.5).astype(float)
    later = local_dates(data) >= np.datetime64(turns)
    continuous = ~later if reverse else later
    values[continuous] = rng.normal(0.0, 1.0, int(continuous.sum()))
    return data.assign(turning=values)


def market(freq, start, days, turns, horizon=2, reverse=False):
    """`make_market_data` with `turning`, at 1 h, 15 min or (its noon rows) daily."""
    hourly = make_market_data(
        start=start, days=days, freq="15min" if freq == "15min" else "1h", horizon=horizon, noise=5.0, seed=3
    )
    return with_turning(hourly.at_time("12:00") if freq == "1D" else hourly, turns, reverse=reverse)


def ols(predictors=MASKED, window=WINDOW, **kwargs):
    return OLSModel(predictors, training_window=window, **kwargs)


def weekday_lists(ctx):
    """Two lists, one on weekdays and one at weekends: each has its own mask day under the default."""
    return MASKED if ctx.issue_day.weekday() < 5 else NO_LEVEL + ["turning", "price_d-1"]


def weekday_matching(ctx):
    return Use(MASKED, train_on="matching") if ctx.issue_day.weekday() < 5 else NO_LEVEL


def turning_while_aux(ctx):
    return MASKED if ctx.published("aux") else NO_LEVEL


def matching_while_aux(ctx):
    return Use(MASKED, train_on="matching") if ctx.published("aux") else NO_LEVEL


# ---------------------------------------------------------------------------
# The parameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["Fit", "first-day", "", "fits", 1, True, ("fit",)])
def test_other_values_raise(bad):
    with pytest.raises(ValueError, match="binary_mask must be None, 'first_day' or 'fit'"):
        ols(binary_mask=bad)
    model = ols()
    with pytest.raises(ValueError, match="binary_mask must be"):
        model.binary_mask = bad


def test_first_day_with_refit_raises_whichever_is_set_second():
    with pytest.raises(ValueError, match="binary_mask='first_day' reads a mask on the window's first day"):
        ols(binary_mask="first_day", refit="7D")
    model = ols(binary_mask="first_day")
    with pytest.raises(ValueError, match="refit='1MS'"):
        model.refit = "1MS"
    model = ols(refit="auto")
    with pytest.raises(ValueError, match="refit='auto'"):
        model.binary_mask = "first_day"
    assert (ols(refit="7D", binary_mask="fit").binary_mask, ols(refit="7D").binary_mask) == ("fit", None)


def test_every_fitting_model_takes_it_and_naive_model_has_none():
    for model in (
        LassoCVModel(MASKED, binary_mask="fit"),
        RidgeModel(MASKED, binary_mask="first_day"),
        LEARModel(MASKED, binary_mask="fit"),
        LEARModel.lago(["load"]),
    ):
        assert model.binary_mask in ("fit", "first_day", None)
    assert LEARModel.lago(["load"]).binary_mask is None
    naive = NaiveModel(["load_d+1"])
    assert naive.binary_mask is None
    for rule in ("fit", "first_day"):
        with pytest.raises(ValueError, match="NaiveModel fits nothing, so it scales no column"):
            naive.binary_mask = rule


def test_yaml_writes_it_only_when_set_and_loads_it_back(tmp_path):
    pipeline = ModelPipeline()
    for name, rule in (("plain", None), ("first", "first_day"), ("fit", "fit")):
        pipeline.add_model(ols(name=name, binary_mask=rule))
    pipeline.add_model(LEARModel(MASKED, name="lear", binary_mask="fit"))
    pipeline.add_model(NaiveModel(["load_d+1"]))
    saved = pipeline.to_dict()["models"]
    assert "binary_mask" not in saved[0]["params"] and "binary_mask" not in saved[4]["params"]
    assert [saved[i]["params"]["binary_mask"] for i in (1, 2, 3)] == ["first_day", "fit", "fit"]
    assert saved[2]["params"] == {**saved[0]["params"], "name": "fit", "binary_mask": "fit"}
    path = tmp_path / "pipeline.yaml"
    pipeline.save(path)
    assert yaml.safe_load(path.read_text())["models"][2]["params"]["binary_mask"] == "fit"
    loaded = ModelPipeline.load(path)
    assert [model.binary_mask for model in loaded.models] == [None, "first_day", "fit", "fit", None]
    assert loaded.to_dict() == pipeline.to_dict()


# ---------------------------------------------------------------------------
# The default stays what it was, and so does refit
# ---------------------------------------------------------------------------

S, E = "2023-03-20", "2023-04-05"  # over the spring switch of 26 March 2023 in Europe/Warsaw


@pytest.fixture(scope="module")
def spring():
    """`turning` is 0/1 in the rows of the first day and continuous from 24 March: the default leaves it unscaled for the whole
    run, `"fit"` scales it in every fit whose rows reach 24 March."""
    return market("1h", "2023-01-20", 85, "2023-03-24")


def test_first_day_is_the_default_byte_for_byte_and_fit_with_refit_is_refit(spring, tmp_path):
    def stored(name, predictors=MASKED, **kwargs):
        path = tmp_path / name / "m.jsonl"
        ols(predictors, **kwargs).run(spring, S, E, horizon=2, save_to=str(path))
        return lines(path), manifest_of(path)

    for predictors, kind in ((MASKED, "fixed"), (weekday_lists, "function")):
        plain = stored(f"plain-{kind}", predictors)
        assert stored(f"first-{kind}", predictors, binary_mask="first_day") == plain
        assert plain[1]["training_cut"] == {"rows": "dates"}
        weekly = stored(f"weekly-{kind}", predictors, refit="7D")
        assert stored(f"weekly-fit-{kind}", predictors, refit="7D", binary_mask="fit") == weekly
        assert "binary_mask" not in weekly[1]["training_cut"]
        own = stored(f"fit-{kind}", predictors, binary_mask="fit")
        assert own[1]["training_cut"] == {"rows": "dates", "binary_mask": "fit"}
        assert own[1]["model_params"] == plain[1]["model_params"]  # the worker's model is rebuilt without it
        assert own[0] != plain[0]
        # A fit's mask is a function of its rows, which the data hash covers: no digest anywhere.
        assert own[1].get("binary_masks") is None
        assert all(not entry.get("binary_masks") for entry in own[1].get("predictor_lists") or [])
        if kind == "function":
            assert any(entry["binary_masks"] for entry in plain[1]["predictor_lists"])


# ---------------------------------------------------------------------------
# What "fit" is: refit="1D" record for record, a hand OLS, and the keys it moves
# ---------------------------------------------------------------------------

# (model, predictors, newest target, issue days, freq, training window, horizon): every kind of list, both newest targets, Fridays,
# a list of windows and each frequency, with each model that refits.
ONE_DAY_CELLS = {
    "ols-fixed-1h": ("ols", MASKED, "d-1", None, "1h", WINDOW, 2),
    "ols-function-fridays-window-list": ("ols", weekday_lists, "d-1", ["friday"], "1h", [14, 28], 2),
    "ols-matching-15min": ("ols", weekday_matching, "d-1", None, "15min", 14, 1),
    "ols-published-1D": ("ols", turning_while_aux, "published", None, "1D", WINDOW, 2),
    "ridge-published-1h": ("ridge", MASKED, "published", None, "1h", WINDOW, 1),
    "lear-window-list-1h": ("lear", MASKED, "d-1", None, "1h", [14, 28], 1),
}


def model_of(kind, predictors, window, **kwargs):
    if kind == "ridge":
        return RidgeModel(predictors, training_window=window, **kwargs)
    if kind == "lear":
        return LEARModel(predictors, training_window=window, **kwargs)
    return OLSModel(predictors, training_window=window, **kwargs)


@pytest.mark.parametrize("cell", list(ONE_DAY_CELLS))
def test_fit_is_refit_1d_record_for_record(cell):
    """A `"1D"` unit is the forecast of its issue day, fitted on its rows with the mask read from them before any filter: what
    `"fit"` gives a forecast without refit. So every record is the same, `refit_anchor` (the run date either way) included, and the
    default differs where its first-day mask does."""
    kind, predictors, newest, issue_days, freq, window, horizon = ONE_DAY_CELLS[cell]
    data = market(freq, "2023-01-20", 85, "2023-03-24").assign(aux=1.0)
    start, end = ("2023-03-24", "2023-03-28") if freq == "15min" else (S, E)
    days = {} if issue_days is None else {"issue_days": issue_days}

    def run(**kwargs):
        model = model_of(kind, predictors, window, newest_target=newest, **kwargs)
        return keyed(model.run(data, start, end, horizon=horizon, freq=freq, information_set=INFO, **days)._results)

    own = run(binary_mask="fit")
    assert own and own == run(refit="1D")
    plain = run()
    assert plain.keys() == own.keys() and plain != own


def with_solar(data, seed=6):
    """`solar`: 0 before sunrise and after sunset, a value from 30 to 100 in between. Sunrise is 07:40 on 1 January and moves earlier
    every day until June, so at 06:00 (and at 18:00, as sunset is 24 h minus sunrise) the column is 0 all winter and has values
    from 12 March on (the first date with sunrise at or before 06:00)."""
    wall = data.index.tz_localize(None)
    hour = np.asarray(wall.hour + wall.minute / 60)
    sunrise = 5.6 + 2.1 * np.cos(2 * np.pi * (np.asarray(wall.dayofyear) + 10) / 365)
    light = (hour >= sunrise) & (hour <= 24 - sunrise)
    values = np.random.default_rng(seed).uniform(30.0, 100.0, len(data))
    return data.assign(solar=np.where(light, values, 0.0))


SOLAR = ("2023-03-01", "2023-03-31")  # a winter start: the first day's rows at 06:00 are all 0
SOLAR_PREDICTORS = NO_LEVEL + ["solar"]


def solar_frame():
    return with_solar(make_market_data(start="2022-12-15", days=120, horizon=2, noise=5.0, seed=3))


@pytest.fixture(scope="module")
def solar():
    return solar_frame()


def rows_of(data, period, first, last):
    """The rows of `period` dated from `first` to `last`, by local date, as a boolean mask."""
    wall = data.index.tz_localize(None)
    dates = wall.normalize()
    return (
        np.asarray(wall.hour == period)
        & np.asarray(dates >= pd.Timestamp(first))
        & np.asarray(dates <= pd.Timestamp(last))
    )


def binary_by_hand(data, column, period, first, last):
    values = data[column].to_numpy()[rows_of(data, period, first, last)]
    return set(np.unique(values[np.isfinite(values)]).tolist()) <= {0.0, 1.0}


def span(issue, horizon, window=WINDOW):
    """The first and last date of a forecast's training rows: D - W - h to D - 1 - h."""
    return day(issue) - datetime.timedelta(days=window + horizon), day(issue) - datetime.timedelta(days=1 + horizon)


def test_fit_moves_exactly_the_keys_whose_own_rows_disagree_with_the_first_day_s(solar):
    """The default reads every mask on 1 March, where `solar` is 0 at 06:00 and 18:00 in the rows (binary: unscaled for the whole
    run). A fit's own rows have values at those hours once they reach 12 March, from D = 12 March + 1 + h: there `"fit"` scales it
    and the forecasts differ, everywhere else they are the same to the bit. Counted by hand: 18 issue days at h=1 and 17 at h=2,
    at each of the two hours, 70 keys."""
    default = fits(ols(SOLAR_PREDICTORS).run(solar, *SOLAR, horizon=2)._results)
    own = fits(ols(SOLAR_PREDICTORS, binary_mask="fit").run(solar, *SOLAR, horizon=2)._results)
    assert own.keys() == default.keys()
    moved = {key for key in own if own[key] != default[key]}

    expected = set()
    for key in own:
        issue, period, horizon = key
        first_day = binary_by_hand(solar, "solar", period, *span(SOLAR[0], horizon))
        if binary_by_hand(solar, "solar", period, *span(issue, horizon)) != first_day:
            expected.add(key)
    assert moved == expected
    first_values = {
        period: solar.index[(solar["solar"] > 0).to_numpy() & (solar.index.hour == period)][0].date()
        for period in (6, 18)
    }
    assert first_values == {6: datetime.date(2023, 3, 12), 18: datetime.date(2023, 3, 12)}
    by_hand = {
        (issue.isoformat(), period, horizon)
        for period in (6, 18)
        for horizon in (1, 2)
        for issue in pd.date_range(first_values[period] + datetime.timedelta(days=1 + horizon), SOLAR[1]).date
    }
    assert moved == by_hand and len(moved) == 70


def hand_fit(data, names, period, issue, horizon, window=WINDOW):
    """OLS without intercept on the standardised target, from the rows of `period` dated D - W - h to D - 1 - h that have every
    value. A column 0/1 on those rows, read before dropping any, stays as it is; the others are standardised on the rows used.
    Forecast from the last row of `issue`. Returns the forecast, the coefficients and which columns stayed as they are."""
    wall = data.index.tz_localize(None)
    rows = np.flatnonzero(rows_of(data, period, *span(issue, horizon, window)))
    prices = pd.Series(data["price"].to_numpy(), index=wall)
    target = prices.reindex(wall[rows] + pd.Timedelta(days=horizon)).to_numpy()
    x = data[names].to_numpy(dtype=float)
    binary = np.array([set(np.unique(column[np.isfinite(column)])) <= {0.0, 1.0} for column in x[rows].T])
    usable = np.isfinite(target) & np.isfinite(x[rows]).all(axis=1)
    y, train = target[usable], x[rows][usable]
    means = np.where(binary, 0.0, train.mean(axis=0))
    deviations = np.where(binary, 1.0, train.std(axis=0, ddof=1))
    coefficients, *_ = np.linalg.lstsq((train - means) / deviations, (y - y.mean()) / y.std(ddof=1), rcond=None)
    test_row = np.flatnonzero(rows_of(data, period, issue, issue))[-1]
    forecast = ((x[test_row] - means) / deviations) @ coefficients * y.std(ddof=1) + y.mean()
    return float(forecast), coefficients, binary


@pytest.mark.parametrize(
    ("issue", "period", "horizon", "scaled"),
    [("2023-03-25", 6, 1, True), ("2023-03-05", 6, 2, False), ("2023-03-31", 18, 2, True), ("2023-03-02", 12, 1, True)],
)
def test_a_forecast_is_a_hand_ols_with_the_mask_of_its_own_rows(solar, issue, period, horizon, scaled):
    records = ols(SOLAR_PREDICTORS, binary_mask="fit").run(solar, issue, issue, horizon=2)._results
    record = next(r for r in records if (r["hour"], r["horizon"]) == (period, horizon))
    names = [name.replace("{horizon}", str(horizon)) for name in SOLAR_PREDICTORS]
    forecast, coefficients, binary = hand_fit(solar, names, period, issue, horizon)
    assert binary.tolist() == [False, False, True, not scaled]
    assert record["prediction"] == pytest.approx(forecast, abs=1e-9)
    assert np.allclose(record["coefficients"], coefficients, atol=1e-9)


def test_the_mask_is_read_from_the_span_before_the_filters(spring):
    """`odd` is 0/1 on every row a fit keeps and 0.5 on the rows it drops: those with no wind forecast (the finite filter) and,
    for the list of `weekday_matching`, the weekend rows (the matching filter). Read on the span, it is no binary column, so `"fit"`
    scales it; read on the rows kept, it would be. Set those 0.5 to 0 and nothing a fit keeps changes but the mask, so every
    forecast whose span holds such a row moves, and each is the hand OLS with the span's mask."""
    dates = local_dates(spring)
    rng = np.random.default_rng(8)
    odd = (rng.random(len(spring)) > 0.5).astype(float)
    dropped = rng.random(len(spring)) < 0.05
    weekend = pd.DatetimeIndex(dates).dayofweek.to_numpy() >= 5
    data = spring.copy()
    for column in ("wind_d+1", "wind_d+2"):
        data.loc[dropped, column] = np.nan
    names = NO_LEVEL + ["odd"]

    def matching(ctx):
        return Use(names, train_on="matching") if ctx.issue_day.weekday() < 5 else NO_LEVEL

    for predictors, drops in ((names, dropped), (matching, dropped | weekend)):
        with_half = data.assign(odd=np.where(drops, 0.5, odd))
        with_zero = data.assign(odd=np.where(drops, 0.0, odd))

        def run(frame):
            return fits(ols(predictors, binary_mask="fit").run(frame, S, E, horizon=2)._results)

        half, zero = run(with_half), run(with_zero)
        assert half.keys() == zero.keys()
        # A weekend forecast of `matching` has no `odd`; any other moves when its span holds a row the filters drop.
        expected = {
            key
            for key in half
            if (predictors is names or day(key[0]).weekday() < 5)
            and (drops & rows_of(data, key[1], *span(key[0], key[2]))).any()
        }
        assert {key for key in half if half[key] != zero[key]} == expected
        assert len(expected) > len(half) // 3
    for issue, period, horizon in (("2023-03-23", 8, 1), ("2023-04-03", 2, 2)):
        frame = data.assign(odd=np.where(dropped, 0.5, odd))
        record = next(
            r
            for r in ols(names, binary_mask="fit").run(frame, issue, issue, horizon=2)._results
            if (r["hour"], r["horizon"]) == (period, horizon)
        )
        expanded = [name.replace("{horizon}", str(horizon)) for name in names]
        forecast, coefficients, binary = hand_fit(frame, expanded, period, issue, horizon)
        assert not binary[-1]
        assert record["prediction"] == pytest.approx(forecast, abs=1e-9)
        assert np.allclose(record["coefficients"], coefficients, atol=1e-9)


def test_each_member_of_a_window_list_reads_its_own_mask():
    """`turning` is continuous before 10 March and 0/1 from then on, so the 14-day member's rows are binary where the 42-day
    member's aren't, from D = 10 March + 14 + h on. Each member's forecast is its single-window model's with `"fit"` to the bit."""
    data = market("1h", "2023-01-01", 110, "2023-03-10", reverse=True)
    start, end = "2023-03-25", "2023-04-08"
    listed = ols(window=[14, 42], binary_mask="fit").run(data, start, end, horizon=2)._results
    singles = [fits(ols(window=w, binary_mask="fit").run(data, start, end, horizon=2)._results) for w in (14, 42)]
    disagree = 0
    for record in listed:
        key = (record["run_date"], record["hour"], record["horizon"])
        assert record["member_predictions"] == [single[key][0] for single in singles], key
        disagree += binary_by_hand(data, "turning", key[1], *span(key[0], key[2], 14)) != binary_by_hand(
            data, "turning", key[1], *span(key[0], key[2], 42)
        )
    assert disagree > 500  # the members' masks differ on most keys


# ---------------------------------------------------------------------------
# Independent of test_start
# ---------------------------------------------------------------------------

# (freq, first issue day S, k, predictors, newest target, issue days, training window, refit, horizon). The second run starts on
# S + k and ends two days after the first, and `turning` turns on S + k - 2: the default reads a binary column on S and a
# continuous one on S + k. Both switches, each frequency, Fridays, "published", a list of windows, a function, and a start inside a
# refit block.
START_CELLS = {
    "1h-spring-k1": ("1h", "2023-03-24", 1, MASKED, "d-1", None, WINDOW, None, 2),
    "1h-spring-k3-function": ("1h", "2023-03-22", 3, weekday_lists, "d-1", None, WINDOW, None, 2),
    "1h-autumn-k17": ("1h", "2023-10-16", 17, MASKED, "d-1", None, WINDOW, None, 2),
    "15min-spring-k1": ("15min", "2023-03-25", 1, MASKED, "d-1", None, 14, None, 1),
    "15min-autumn-k3-function": ("15min", "2023-10-26", 3, weekday_lists, "d-1", None, 14, None, 1),
    "1D-autumn-k17": ("1D", "2023-10-12", 17, MASKED, "d-1", None, WINDOW, None, 2),
    "1D-spring-k3-window-list": ("1D", "2023-03-20", 3, MASKED, "d-1", None, [14, 28], None, 2),
    "1h-fridays-k17-function": ("1h", "2023-03-17", 17, weekday_lists, "d-1", ["friday"], WINDOW, None, 2),
    "1h-autumn-k3-published": ("1h", "2023-10-25", 3, turning_while_aux, "published", None, WINDOW, None, 2),
    "1h-spring-k1-window-list-function": ("1h", "2023-03-23", 1, weekday_lists, "d-1", None, [14, 28], None, 1),
    "1h-spring-k3-inside-a-refit-block": ("1h", "2023-03-21", 3, MASKED, "d-1", None, WINDOW, "7D", 2),
}


@pytest.mark.parametrize("cell", list(START_CELLS))
def test_a_later_start_moves_no_forecast(cell):
    """Runs over [S, E] and [S + k, E + 2]: with `"fit"` every forecast they share is the same, bit for bit, `actual` included. The
    default differs on some of them (the case is exercised), except with `refit`, whose units always read their own rows."""
    freq, first, k, predictors, newest, issue_days, window, refit, horizon = START_CELLS[cell]
    later = shift(first, k)
    end = shift(later, 6 if issue_days else 3)
    data = market(freq, shift(first, -70), 100, shift(later, -2), horizon=horizon).assign(aux=1.0)
    days = {} if issue_days is None else {"issue_days": issue_days}

    def run(start, stop, rule):
        model = ols(predictors, window, newest_target=newest, refit=refit, binary_mask=rule)
        result = model.run(data, start, stop, horizon=horizon, freq=freq, information_set=INFO, **days)
        return keyed(result._results, "day_in_test")

    for rule in ("fit", None):
        whole, part = run(first, end, rule), run(later, shift(end, 2), rule)
        shared = whole.keys() & part.keys()
        assert {key[0] for key in shared} == {key[0] for key in part if key[0] <= end}
        same = all(whole[key] == part[key] for key in shared)
        assert same == (rule == "fit" or refit is not None), rule


def test_validation_and_test_are_one_plain_run_over_both():
    """With `"fit"` a run with validation gives each window's forecasts what one plain run from the validation's first day to the
    test's last gives them, and the test's what a plain run of the test gives. The default reads each window's masks on its own
    first day, which here differ from the validation's."""
    validation, test = ("2023-03-20", "2023-03-26"), ("2023-03-29", "2023-04-06")
    data = market("1h", "2023-01-20", 85, shift(test[0], -2))

    def windowed(rule):
        pipeline = ModelPipeline()
        pipeline.add_model(ols(name="m", binary_mask=rule))
        records = pipeline.run(data, validation=validation, test=test, horizon=2).refs["m"]._results
        return {
            name: keyed([r for r in records if r["window"] == name], "window", "day_in_test")
            for name in ("validation", "test")
        }

    def plain(start, stop, rule):
        return keyed(ols(binary_mask=rule).run(data, start, stop, horizon=2)._results, "day_in_test")

    for rule in ("fit", None):
        windows, whole = windowed(rule), plain(validation[0], test[1], rule)
        assert windows["test"] == plain(*test, rule)
        for name, (start, stop) in (("validation", validation), ("test", test)):
            same = windows[name] == {key: record for key, record in whole.items() if start <= key[0] <= stop}
            assert same == (rule == "fit" or name == "validation"), (rule, name)


# ---------------------------------------------------------------------------
# Nothing after D moves D's forecasts
# ---------------------------------------------------------------------------

AFTER_D_LISTS = {
    "fixed": (MASKED, "d-1"),
    "function": (turning_while_aux, "d-1"),
    "matching": (matching_while_aux, "d-1"),
    "published": (turning_while_aux, "published"),
}
# Every level of each axis (kind of list x issue days x horizon) and every pair of the last two, as in test_refit.py's grid.
AFTER_D_CELLS = [
    ("fixed", None, 1),
    ("function", ["friday"], 3),
    ("matching", None, 3),
    ("published", ["friday"], 1),
]


@pytest.mark.parametrize(("lists", "issue_days", "horizon"), AFTER_D_CELLS)
def test_nothing_after_the_issue_day_moves_its_forecasts(lists, issue_days, horizon):
    """For D = Friday 31 March and Friday 7 April: every finite value dated after D changed, `aux` missing after D (which changes
    a function's list after D), and the test ending on D. D's predictions and coefficients stay bit for bit. `turning` turns on 29
    March, so a mask read from a later fit's rows would show."""
    first, last = "2023-03-27", "2023-04-10"
    data = market("1h", "2023-01-20", 85, "2023-03-29", horizon=3).assign(aux=1.0)
    predictors, newest = AFTER_D_LISTS[lists]
    days = {} if issue_days is None else {"issue_days": issue_days}

    def run(frame, end=last):
        model = ols(predictors, newest_target=newest, binary_mask="fit")
        return fits(model.run(frame, first, end, horizon=horizon, information_set=INFO, **days)._results)

    base = run(data)
    dates = local_dates(data)
    columns = [c for c in data.columns if data[c].dtype.kind == "f"]
    for issue in ("2023-03-31", "2023-04-07"):
        on_d = {key: fit for key, fit in base.items() if key[0] == issue}
        assert len(on_d) == 24 * horizon
        after = dates > np.datetime64(issue)
        moved = data.copy()
        values = moved[columns].to_numpy(dtype=float, copy=True)
        where = after[:, None] & np.isfinite(values)
        values[where] += np.random.default_rng(1).normal(0, 3, size=int(where.sum()))
        moved[columns] = values
        blank = data.copy()
        blank.loc[after, "aux"] = np.nan
        for what, frame, end in (("values", moved, last), ("aux", blank, last), ("end", data, issue)):
            other = run(frame, end)
            assert {key: fit for key, fit in other.items() if key[0] == issue} == on_d, (issue, what)
            after_d = [key for key in base if key[0] > issue]
            if what == "values" and after_d:  # the change is seen after D
                assert any(other.get(key) != base[key] for key in after_d)


def test_verify_is_ok_with_its_positive_control(spring):
    pipeline = ModelPipeline(information_set=INFO)
    pipeline.add_model(ols(name="fixed", binary_mask="fit"))
    pipeline.add_model(ols(turning_while_aux, name="published", newest_target="published", binary_mask="fit"))
    pipeline.add_model(ols(window=[14, 28], name="windows", binary_mask="fit"))
    report = pipeline.verify(spring.assign(aux=1.0), "2023-03-22", "2023-04-04", horizon=2, days=3)
    report.assert_ok()
    assert report.models == ["fixed", "published", "windows"]


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_another_rule_refuses_a_resume_and_the_same_rule_by_another_name_doesn_t(spring, tmp_path):
    for first, then in ((None, "fit"), ("fit", None), ("fit", "first_day"), ("first_day", "fit")):
        path = tmp_path / f"{first}-{then}" / "m.jsonl"
        ols(binary_mask=first).run(spring, "2023-03-27", "2023-04-02", horizon=1, save_to=str(path))
        with pytest.raises(ManifestMismatchError, match="training_cut"):
            ols(binary_mask=then).run(spring, "2023-03-27", "2023-04-02", horizon=1, save_to=str(path))
    for first, then, refit in (
        (None, "first_day", None),
        ("first_day", None, None),
        (None, "fit", "7D"),
        ("fit", None, "7D"),
    ):
        path = tmp_path / f"same-{first}-{then}-{refit}" / "m.jsonl"
        ols(binary_mask=first, refit=refit).run(spring, "2023-03-27", "2023-04-02", horizon=1, save_to=str(path))
        stored = lines(path)
        ols(binary_mask=then, refit=refit).run(spring, "2023-03-27", "2023-04-02", horizon=1, save_to=str(path))
        assert lines(path) == stored


def test_a_fit_run_resumes_over_another_test_start_to_a_fresh_one(spring, tmp_path):
    """Stored forecasts of a function's lists are compared with what a fresh run would give them, mask digests included. With
    `"fit"` there are none, and none is needed: the forecasts of [S + 5, E] are those of a run over [S, E], so the resume runs the
    first five days and the file is a fresh run's. The default refuses, since the first day moved the lists' masks."""
    later = shift(S, 5)
    data = with_turning(spring, shift(later, -2))
    fresh = tmp_path / "fresh" / "m.jsonl"
    ols(weekday_lists, binary_mask="fit").run(data, S, E, horizon=2, save_to=str(fresh))
    for rule in ("fit", None):
        path = tmp_path / str(rule) / "m.jsonl"
        ols(weekday_lists, binary_mask=rule).run(data, later, E, horizon=2, save_to=str(path))
        if rule is None:
            with pytest.raises(ManifestMismatchError, match="other binary columns"):
                ols(weekday_lists, binary_mask=rule).run(data, S, E, horizon=2, save_to=str(path))
            continue
        ols(weekday_lists, binary_mask=rule).run(data, S, E, horizon=2, save_to=str(path))
        assert lines(path, "day_in_test") == lines(fresh, "day_in_test")


def test_a_run_with_validation_resumes_plain_fit_results_whose_test_started_earlier(spring, tmp_path):
    """A plain manifest of fixed lists holds no mask digests, only its first day, so a run with windows whose test starts on another
    day can't tell its masks from the stored ones and the default refuses. A `"fit"` mask is a function of its fit's rows: the stored
    forecasts are what the run with windows gives them, so it resumes, and its test forecasts are a fresh run's with the windows."""
    validation, test = ("2023-03-15", "2023-03-20"), ("2023-03-29", "2023-04-05")
    data = with_turning(spring, "2023-03-25")

    def run(rule, path, **days):
        pipeline = ModelPipeline()
        pipeline.add_model(ols(name="m", binary_mask=rule))
        pipeline.run(data, horizon=1, save_dir=str(path), **days)

    run("fit", tmp_path / "fresh", validation=validation, test=test)
    for rule in ("fit", None):
        path = tmp_path / str(rule)
        run(rule, path, test_start="2023-03-27", test_end=test[1])
        if rule is None:
            with pytest.raises(ManifestMismatchError, match="other binary columns"):
                run(rule, path, validation=validation, test=test)
            continue
        run(rule, path, validation=validation, test=test)
        stored = lines(path / "m.jsonl", "day_in_test", "window")
        fresh = lines(tmp_path / "fresh" / "m.jsonl", "day_in_test", "window")
        assert set(fresh) <= set(stored)  # with the plain run's 27 and 28 March, as a refit model's would stay
        assert len(set(stored) - set(fresh)) == 2 * 24


def friday_list(ctx):
    return MASKED if ctx.issue_day.weekday() == 4 else NO_LEVEL


def test_a_friday_run_of_a_function_is_the_run_of_every_day_s_fridays():
    """The default reads the Friday list's mask on the first Friday in a run of every day and on the window's first day, a Monday,
    in a run of Fridays, so the two differ (`tests/models/test_issue_days.py`). With `"fit"` there is no mask day: every Friday's
    record is the run of every day's."""
    data = market("1h", "2023-01-20", 85, "2023-03-22")

    def run(rule, **days):
        records = ols(friday_list, binary_mask=rule).run(data, S, E, horizon=2, **days)._results
        return keyed([r for r in records if r["run_weekday"] == 4])

    for rule in ("fit", None):
        assert (run(rule, issue_days=["friday"]) == run(rule)) == (rule == "fit"), rule


def every_day_masked(ctx):
    return MASKED


def test_a_fit_resume_still_compares_each_stored_forecast_s_list(spring, tmp_path):
    """No mask digest to compare is no reason to compare nothing: a stored forecast whose list a fresh run wouldn't give it is
    refused, as for the default."""
    for rule in ("fit", None):
        path = tmp_path / str(rule) / "m.jsonl"
        ols(weekday_lists, binary_mask=rule).run(spring, "2023-03-27", "2023-04-02", horizon=1, save_to=str(path))
        with pytest.raises(ManifestMismatchError, match="another predictor list"):
            ols(every_day_masked, binary_mask=rule).run(
                spring, "2023-03-27", "2023-04-02", horizon=1, save_to=str(path)
            )


def test_a_column_that_starts_inside_the_test_needs_no_re_read(spring, tmp_path):
    """`late` has no value before 23 March. The default reads it again on the first day whose rows have one (`test_masks.py`) and
    records the digests that changed; under `"fit"` a fit whose rows have no value of it has no finite row and is skipped, and every
    other fit reads its own rows. Both scale the continuous column from then on, so every forecast and every skip is the same, and
    only the default's manifest has digests."""
    late = np.where(
        local_dates(spring) >= np.datetime64("2023-03-23"), np.random.default_rng(9).normal(size=len(spring)), np.nan
    )
    data = spring.assign(late=late)
    stored = {}
    for rule in (None, "fit"):
        path = tmp_path / str(rule) / "m.jsonl"
        ols(NO_LEVEL + ["late"], binary_mask=rule).run(data, S, E, horizon=2, save_to=str(path))
        stored[rule] = (lines(path), manifest_of(path))
    assert stored["fit"][0] == stored[None][0]
    assert any('"no_finite_training_rows"' in line for line in stored["fit"][0])
    assert stored[None][1]["binary_masks"] and "binary_masks" not in stored["fit"][1]


def test_scale_binary_ignores_the_rule():
    """`StandardScaler(scale_binary=True)` scales every column whatever the mask, so `"fit"` is the default to the bit."""
    data = market("1h", "2023-01-20", 85, "2023-03-24")
    scaler = {"name": "standard", "params": {"scale_binary": True}}
    own, plain = (
        keyed(ols(scaler=scaler, binary_mask=rule).run(data, S, E, horizon=2)._results) for rule in ("fit", None)
    )
    assert own == plain


def resumed_and_fresh(binary_mask, tmp_path):
    """Records over [S, E] of a run over [S+5, E] resumed over [S, E] into the same file, and of a fresh [S, E] run.
    `turning` is 0/1 on S's rows and continuous from S+3, so a mask read on S+5's first day differs from one read on S's."""
    later = shift(S, 5)
    data = market("1h", "2023-01-20", 85, shift(later, -2))
    fresh, resumed = tmp_path / "fresh" / "m.jsonl", tmp_path / "resumed" / "m.jsonl"
    ols(binary_mask=binary_mask).run(data, S, E, horizon=2, save_to=str(fresh))
    ols(binary_mask=binary_mask).run(data, later, E, horizon=2, save_to=str(resumed))
    ols(binary_mask=binary_mask).run(data, S, E, horizon=2, save_to=str(resumed))
    return lines(resumed, "day_in_test"), lines(fresh, "day_in_test")


def test_fit_resumed_over_an_earlier_start_equals_the_fresh_run(tmp_path):
    """The specific half of the xfail in `tests/test_known_issues.py`: with `binary_mask="fit"` no mask depends on the start."""
    resumed, fresh = resumed_and_fresh("fit", tmp_path)
    assert len(fresh) > 0
    assert resumed == fresh
