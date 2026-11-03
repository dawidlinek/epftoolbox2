"""`refit=`: a model fitted on a fixed calendar, once per unit (period, horizon, anchor, list), each forecast predicted from it.

A forecast issued on D uses the fit of its anchor a, the refit day on or before D counted from Monday 1970-01-05: rows cut as for a
forecast issued on a (`training_span(a, W, h, k(a))`), its mask read from those rows, and its own row predicted on its own. So it
depends on its issue day and the configuration alone: not on where a run starts or ends, which other forecasts share the run, how
many processes run it, or whether it was resumed. Nothing dated after a - 1 - h trains, which the perturbations below hold bit for
bit, with positive controls that the perturbation reaches the fit. Numbers compare as JSON text, so a NaN `actual` equals itself.
"""

import datetime
import json
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import Ridge

from epftoolbox2.ensembles import AverageEnsemble
from epftoolbox2.information_set import (
    Availability,
    InformationSet,
    InformationSetError,
    leakage_probe,
    positive_control,
)
from epftoolbox2.models import BaseModel, LassoCVModel, OLSModel, Use
from epftoolbox2.models._calendar import FitUnit, check_refit, refit_anchor, refit_anchors
from epftoolbox2.models import _worker as worker
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path
from tests.synthetic import PREDICTORS, make_market_data

WINDOW = 21
RULES = {
    "load": Availability.published("10:00", 10),
    "wind": Availability.published("10:00", 10),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "flag": Availability.static(),
    "aux": Availability.static(),
    "price": Availability.published("13:00", 1),  # D's price is known at 11:45 on D, D+1's isn't
}
INFO = InformationSet("11:45", "Europe/Brussels", rules=RULES)
PROVENANCE = ("package_version", "git_sha", "created_at", "updated_at")


class InProcessPool:
    """Stands in for the worker pool, whose start is most of the time these runs would take. The tests about the pool use the
    real one."""

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


def local_dates(data):
    return np.asarray(pd.DatetimeIndex(data.index.tz_localize(None)).normalize())


def day(text):
    return datetime.date.fromisoformat(text)


def shift(text, days):
    return (day(text) + datetime.timedelta(days=days)).isoformat()


def keyed(records, *drop):
    """Key -> the record as JSON text without `drop`."""
    return {
        (r["run_date"], r["hour"], r["horizon"]): json.dumps(
            {k: v for k, v in r.items() if k not in drop}, sort_keys=True
        )
        for r in records
    }


def fits(records):
    """Key -> what the fit made of it: prediction, coefficients and anchor. `actual` is read after D, `day_in_test` from the start."""
    return {
        (r["run_date"], r["hour"], r["horizon"]): (r["prediction"], r["coefficients"], r.get("refit_anchor"))
        for r in records
    }


def lines(path):
    """A result file's records as sorted text: the order a pool finishes them in isn't fixed."""
    return sorted(line for line in Path(path).read_text().splitlines() if line.strip())


def with_flag(data, turns):
    """`flag`: 0/1 before `turns`, other values from then on, so a mask read from rows before it differs from one after."""
    flag = (np.random.default_rng(5).random(len(data)) > 0.5).astype(float)
    after = local_dates(data) >= np.datetime64(turns)
    flag[after] = flag[after] * 0.5 + 0.25
    return data.assign(flag=flag)


def ols(refit, predictors=PREDICTORS, window=WINDOW, **kwargs):
    return OLSModel(predictors, training_window=window, refit=refit, **kwargs)


def move(data, rows, columns, seed=1):
    """A copy of `data` with every finite value of `columns` on `rows` moved by noise."""
    out = data.copy()
    values = out[columns].to_numpy(dtype=float, copy=True)
    where = rows[:, None] & np.isfinite(values)
    values[where] += np.random.default_rng(seed).normal(0, 3, size=int(where.sum()))
    out[columns] = values
    return out


# ---------------------------------------------------------------------------
# The calendar and the parameter
# ---------------------------------------------------------------------------


def test_anchors_count_from_monday_5_january_1970():
    assert refit_anchor("2024-01-05", "7D") == day("2024-01-01")
    assert refit_anchor("2024-01-01", "7D") == day("2024-01-01")
    assert refit_anchor("2024-01-07", "7D") == day("2024-01-01")
    assert refit_anchor("2024-02-29", "1D") == day("2024-02-29")
    assert refit_anchor("2024-02-29", "1MS") == day("2024-02-01")
    assert refit_anchor("2024-03-01", "1MS") == day("2024-03-01")
    assert [refit_anchor(f"2024-{m:02d}-15", "3MS").month for m in range(1, 13)] == [1] * 3 + [4] * 3 + [7] * 3 + [
        10
    ] * 3
    assert refit_anchor("2024-12-31", "12MS") == day("2024-01-01")
    assert refit_anchor("1969-12-31", "7D") == day("1969-12-29")  # before the epoch too
    assert refit_anchor("1969-12-31", "5MS") == day("1969-08-01")
    epoch = day("1970-01-05")
    for k in (1, 2, 5, 7, 10):
        for offset in range(-40, 400, 3):
            issue = day("2023-01-01") + datetime.timedelta(days=offset)
            anchor = refit_anchor(issue, f"{k}D")
            assert 0 <= (issue - anchor).days < k and (anchor - epoch).days % k == 0
            assert refit_anchor(anchor, f"{k}D") == anchor
    assert refit_anchors(["2024-01-05", "2024-01-02", "2024-01-09"], "7D") == [day("2024-01-01"), day("2024-01-08")]
    assert FitUnit(3, 1, day("2024-01-01"), None).anchor == day("2024-01-01")


@pytest.mark.parametrize("bad", ["auto", "0D", "7d", "07D", "D", "1M", "1W", "1 D", "", 7, 7.0, True])
def test_other_refit_values_raise(bad):
    with pytest.raises(ValueError, match="refit must be"):
        check_refit(bad)
    with pytest.raises(ValueError, match="refit must be"):
        OLSModel(PREDICTORS, refit=bad)
    model = OLSModel(PREDICTORS)
    with pytest.raises(ValueError, match="refit must be"):
        model.refit = bad
    assert model.refit is None


def test_auto_is_named_as_not_there_yet():
    with pytest.raises(ValueError, match="'auto' isn't one yet"):
        OLSModel(PREDICTORS, refit="auto")


def test_a_model_with_only_fit_predict_raises_on_refit():
    class OneStep(BaseModel):
        def _fit_predict(self, train_x, train_y, test_x):
            return 0.0, []

    OneStep(PREDICTORS)  # refit=None runs as before
    with pytest.raises(ValueError, match="can't refit"):
        OneStep(PREDICTORS, refit="7D")
    assert OLSModel(PREDICTORS, refit="7D").refit == "7D"
    assert LassoCVModel(PREDICTORS, refit="1MS").refit == "1MS"


class RidgeOLS(OLSModel):
    """Another estimator under OLS's name, the way `tests/pipelines/test_verify.py` builds its models: refit would fit OLS."""

    def _fit_predict(self, train_x, train_y, test_x):
        model = Ridge(alpha=50.0, fit_intercept=False).fit(train_x, train_y)
        return model.predict(test_x)[0], model.coef_.tolist()


class RidgeBoth(RidgeOLS):
    """Ridge in all three methods, defined in one class."""

    def _fit_predict(self, train_x, train_y, test_x):
        return self._predict_fitted(self._fit_model(train_x, train_y), test_x)

    def _fit_model(self, train_x, train_y):
        return Ridge(alpha=50.0, fit_intercept=False).fit(train_x, train_y)

    def _predict_fitted(self, fitted, test_x):
        return fitted.predict(test_x)[0], fitted.coef_.tolist()


class RidgeCV(LassoCVModel):
    def _fit_predict(self, train_x, train_y, test_x):
        return RidgeOLS._fit_predict(self, train_x, train_y, test_x)


class RidgeRefit(OLSModel):
    """The other way round: Ridge in the seam, OLS's `_fit_predict`. refit=None would fit OLS and refit Ridge."""

    def _fit_model(self, train_x, train_y):
        return Ridge(alpha=50.0, fit_intercept=False).fit(train_x, train_y)

    def _predict_fitted(self, fitted, test_x):
        return fitted.predict(test_x)[0], fitted.coef_.tolist()


class Seam(OLSModel):
    """Overrides `_predict_fitted` alone: it may forecast otherwise than `_fit_predict` does, so it doesn't refit either."""

    def _predict_fitted(self, fitted, test_x):
        return fitted.predict(test_x)[0], fitted.coef_.tolist()


class Renamed(LassoCVModel):
    """Overrides none of the three: LassoCV's fit, whatever else it adds."""

    def __init__(self, predictors, **kwargs):
        super().__init__(predictors, cv=3, **kwargs)


def test_refit_needs_one_class_to_define_all_three_fit_methods(market):
    """Inheriting `_fit_model` and `_predict_fitted` from OLSModel while fitting Ridge in `_fit_predict` would have refitted it
    with OLS, silently, up to 47 off its own refit=None run, and the other way round refit would fit Ridge where refit=None fits
    OLS. So one class defines all three, or refit raises naming their owners. OLS, LassoCV, a subclass overriding all three and one
    overriding none refit; overriding all three refits the subclass's own estimator."""
    for cls, parent in ((RidgeOLS, "OLSModel"), (RidgeCV, "LassoCVModel"), (RidgeRefit, "RidgeRefit"), (Seam, "Seam")):
        cls(PREDICTORS)  # refit=None runs as before
        with pytest.raises(ValueError, match=f"can't refit.*{parent}'s"):
            cls(PREDICTORS, refit="7D")
    for cls in (OLSModel, LassoCVModel, RidgeBoth, Renamed):
        assert cls(PREDICTORS, refit="7D").refit == "7D"
    both = RidgeBoth(PREDICTORS, training_window=WINDOW, refit="1D").run(market, "2023-03-20", "2023-03-22", horizon=1)
    plain = RidgeBoth(PREDICTORS, training_window=WINDOW).run(market, "2023-03-20", "2023-03-22", horizon=1)
    assert keyed(both._results, "refit_anchor") == keyed(plain._results)


def test_yaml_names_refit_only_when_set(tmp_path):
    pipeline = ModelPipeline()
    pipeline.add_model(OLSModel(PREDICTORS, name="plain"))
    pipeline.add_model(OLSModel(PREDICTORS, name="none", refit=None))
    pipeline.add_model(LassoCVModel(PREDICTORS, name="weekly", refit="7D", cv=3))
    saved = pipeline.to_dict()["models"]
    assert "refit" not in saved[0]["params"] and "refit" not in saved[1]["params"]
    assert saved[1]["params"] == {**saved[0]["params"], "name": "none"}
    assert saved[2]["params"]["refit"] == "7D"
    path = tmp_path / "pipeline.yaml"
    pipeline.save(path)
    assert yaml.safe_load(path.read_text())["models"][2]["params"]["refit"] == "7D"
    loaded = ModelPipeline.load(path)
    assert [m.refit for m in loaded.models] == [None, None, "7D"]


# ---------------------------------------------------------------------------
# What a forecast is
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def market():
    """Over the spring switch of 26 March 2023, with `flag` turning from 0/1 to other values on 29 March."""
    return with_flag(make_market_data(start="2023-01-01", days=130, horizon=3, noise=5.0, seed=3), "2023-03-29")


def test_refit_1d_is_refit_none_where_the_masks_agree(market):
    """With `"1D"` every forecast is its own unit, on the rows `refit=None` gives it, so it is the same forecast wherever the
    mask agrees: `refit=None` reads a list's mask on the window's first day, a unit on its own rows. `flag` is 0/1 in the rows of
    the first day and in those of a forecast until its rows reach 29 March; from there `"1D"` scales it and `None` doesn't."""
    start, end = "2023-03-20", "2023-04-12"
    predictors = PREDICTORS + ["flag"]
    plain = keyed(ols(None, predictors).run(market, start, end, horizon=2)._results)
    daily = ols("1D", predictors).run(market, start, end, horizon=2)._results
    assert {r["refit_anchor"] for r in daily} == {r["run_date"] for r in daily}
    daily = keyed(daily, "refit_anchor")
    assert daily.keys() == plain.keys()
    for key, record in daily.items():
        run_date, _, horizon = key
        mask_agrees = shift(run_date, -1 - horizon) < "2023-03-29"  # the newest row has no other value of `flag` yet
        assert (record == plain[key]) == mask_agrees, key
    without_flag = keyed(ols("1D").run(market, start, end, horizon=2)._results, "refit_anchor")
    assert without_flag == keyed(ols(None).run(market, start, end, horizon=2)._results)


def hand_fit(data, period, anchor, issue, horizon, names, window=WINDOW):
    """OLS without intercept on the standardised target, from the rows of `period` dated anchor-W-h to anchor-1-h with every
    value; columns that are 0/1 on those rows stay as they are, others are standardised on the rows used. Forecast from the row of
    `issue`. Returns the forecast and the coefficients."""
    wall = data.index.tz_localize(None)
    dates = wall.normalize()
    here = np.asarray(wall.hour == period)
    a = pd.Timestamp(anchor)
    rows = np.flatnonzero(
        here
        & np.asarray(dates >= a - pd.Timedelta(days=window + horizon))
        & np.asarray(dates <= a - pd.Timedelta(days=1 + horizon))
    )
    prices = pd.Series(data["price"].to_numpy(), index=wall)
    target = prices.reindex(wall[rows] + pd.Timedelta(days=horizon)).to_numpy()
    x = data[names].to_numpy(dtype=float)
    binary = np.array([set(np.unique(column[np.isfinite(column)])) <= {0.0, 1.0} for column in x[rows].T])
    usable = np.isfinite(target) & np.isfinite(x[rows]).all(axis=1)
    y, train = target[usable], x[rows][usable]
    means = np.where(binary, 0.0, train.mean(axis=0))
    deviations = np.where(binary, 1.0, train.std(axis=0, ddof=1))
    coefficients, *_ = np.linalg.lstsq((train - means) / deviations, (y - y.mean()) / y.std(ddof=1), rcond=None)
    test_row = np.flatnonzero(here & np.asarray(dates == pd.Timestamp(issue)))[-1]
    return float(((x[test_row] - means) / deviations) @ coefficients * y.std(ddof=1) + y.mean()), coefficients


@pytest.mark.parametrize(
    ("refit", "issues", "anchor"),
    [
        ("7D", ["2024-02-28", "2024-02-29", "2024-03-01", "2024-03-03"], "2024-02-26"),
        ("1MS", ["2024-02-05", "2024-02-28", "2024-02-29"], "2024-02-01"),
        ("1MS", ["2024-03-02", "2024-03-04"], "2024-03-01"),
    ],
    ids=["7D-over-the-leap-day", "1MS-to-the-leap-day", "1MS-after-it"],
)
def test_a_forecast_is_a_hand_fit_at_its_anchor(refit, issues, anchor):
    """The forecasts of a block are one OLS fit on the anchor's rows, scaler included, each predicted from its own row. The runs
    start after the anchor, so nothing but the anchor can put the cut there."""
    data = with_flag(make_market_data(start="2023-12-01", days=105, horizon=2, noise=5.0, seed=11), "2024-02-10")
    names = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "is_weekday_d+{horizon}", "flag"]
    records = ols(refit, names).run(data, issues[0], issues[-1], horizon=2)._results
    found = {(r["run_date"], r["hour"], r["horizon"]): r for r in records}
    for issue in issues:
        for period in (0, 7, 18):
            for horizon in (1, 2):
                record = found[(issue, period, horizon)]
                assert record["refit_anchor"] == anchor
                expanded = [name.replace("{horizon}", str(horizon)) for name in names]
                forecast, coefficients = hand_fit(data, period, anchor, issue, horizon, expanded)
                assert record["prediction"] == pytest.approx(forecast, abs=1e-9)
                assert np.allclose(record["coefficients"], coefficients, atol=1e-9)


def test_records_carry_the_anchor_and_the_manifest_the_rule(market, tmp_path):
    path = tmp_path / "weekly" / "m.jsonl"
    ols("7D").run(market, "2023-03-29", "2023-04-04", horizon=1, save_to=str(path))
    records = [json.loads(line) for line in lines(path)]
    assert {(r["run_date"], r["refit_anchor"]) for r in records} >= {
        ("2023-03-29", "2023-03-27"),
        ("2023-04-04", "2023-04-03"),
    }
    assert all(day(r["refit_anchor"]).weekday() == 0 for r in records)
    manifest = json.loads(manifest_path(path).read_text())
    assert manifest["training_cut"] == {"rows": "dates", "refit": "7D", "refit_epoch": "1970-01-05"}
    assert "binary_masks" not in manifest

    plain = tmp_path / "plain" / "m.jsonl"
    explicit = tmp_path / "explicit" / "m.jsonl"
    ols(None).run(market, "2023-03-29", "2023-04-04", horizon=1, save_to=str(explicit))
    OLSModel(PREDICTORS, training_window=WINDOW).run(market, "2023-03-29", "2023-04-04", horizon=1, save_to=str(plain))
    assert lines(plain) == lines(explicit)
    assert all("refit_anchor" not in json.loads(line) for line in lines(plain))
    manifests = [json.loads(manifest_path(p).read_text()) for p in (plain, explicit)]
    for manifest in manifests:
        for name in PROVENANCE:
            manifest.pop(name)
    assert manifests[0] == manifests[1]
    assert manifests[0]["training_cut"] == {"rows": "dates"}


def test_another_refit_refuses_a_resume(market, tmp_path):
    path = tmp_path / "m.jsonl"
    ols("7D").run(market, "2023-03-29", "2023-04-04", horizon=1, save_to=str(path))
    for other in ("1MS", None):
        with pytest.raises(ManifestMismatchError, match="training_cut"):
            ols(other).run(market, "2023-03-29", "2023-04-04", horizon=1, save_to=str(path))


def starting_on(data, first):
    """`data` from the local date `first` on."""
    return data[local_dates(data) >= np.datetime64(first)]


@pytest.mark.parametrize(
    ("refit", "window", "test", "needed", "extra"),
    [
        # The reviewer's cases: a frame sized for refit=None, test_start - W - h.
        ("1MS", WINDOW, ("2023-02-15", "2023-02-20"), "2023-01-10", {}),
        ("12MS", 364, ("2023-07-03", "2023-07-09"), "2022-01-01", {}),
        # The first issue day of a run of Fridays is 17 February: its anchor, not the test start's, decides.
        ("1D", WINDOW, ("2023-02-13", "2023-02-20"), "2023-01-26", {"issue_days": ["friday"]}),
        # D-1's price is out at 13:00 on D, so the newest known target is D-2 (reach 1).
        ("7D", WINDOW, ("2023-02-15", "2023-02-20"), "2023-01-21", {"newest": True}),
    ],
    ids=["1MS", "12MS-a-year-back", "Fridays", "published-reach"],
)
def test_a_frame_that_doesn_t_reach_back_from_the_first_anchor_raises(refit, window, test, needed, extra):
    """The first block is fitted on its anchor's rows, from a - W - h(max) - reach: a frame starting later would train it on fewer
    days, silently (up to 5.0 off with "1MS", 1.3 with "12MS"). A frame starting on that day runs; one a day later raises, naming
    it. refit=None is unchanged: it runs on the shorter frame as before."""
    data = make_market_data(start="2021-12-01", days=600, horizon=1, noise=2.0, seed=3)
    days = {} if "issue_days" not in extra else {"issue_days": extra["issue_days"]}
    late = InformationSet("11:45", "Europe/Brussels", rules={**RULES, "price": Availability.published("13:00", -1)})
    cut = {"newest_target": "published", "information_set": late} if extra.get("newest") else {}

    def run(frame, refit=refit):
        model = OLSModel(PREDICTORS, training_window=window, refit=refit, newest_target=cut.get("newest_target", "d-1"))
        return model.run(frame, *test, horizon=1, information_set=cut.get("information_set"), **days)

    run(starting_on(data, needed))
    with pytest.raises(ValueError, match=f"start the frame on {needed} or earlier".replace("start", "Start")):
        run(starting_on(data, shift(needed, 1)))
    run(starting_on(data, shift(needed, 1)), refit=None)


@pytest.mark.parametrize(
    ("newest", "issue_time", "needed"),
    [("d-1", "11:45", "2023-02-18"), ("published", "11:45", "2023-02-19"), ("published", "14:00", "2023-02-20")],
    ids=["d-1", "published-k0", "published-k1"],
)
def test_the_frame_check_asks_for_exactly_the_earliest_row(market, monkeypatch, newest, issue_time, needed):
    """The re-check's trimming probe (`recheck_probes/probe_frame_check.py`): the day asked for is the earliest row any unit
    trains on, `training_span(a, W, h, k(a))` over periods and horizons. At 14:00 the price of D+1 is out (k = 1, capped at H-1),
    so with H = 2 the newest row is a - 1 at both horizons and the frame needs a - W, not a - W - 2: a frame from that day gives the
    long frame's records, and one from the day after is refused, and would have changed them."""
    info = InformationSet(issue_time, "Europe/Brussels", rules=RULES)

    def run(frame):
        model = ols("7D", newest_target=newest)
        return keyed(model.run(frame, "2023-03-15", "2023-03-20", horizon=2, information_set=info)._results)

    long = run(market)
    assert run(starting_on(market, needed)) == long
    with pytest.raises(ValueError, match=f"earliest row they train on is dated {needed}"):
        run(starting_on(market, shift(needed, 1)))
    monkeypatch.setattr(BaseModel, "_check_reaches_back", lambda self, *args: None)
    assert run(starting_on(market, shift(needed, 1))) != long


def test_validation_starts_the_frame_check(market):
    """With validation and test windows the earliest anchor is the validation window's."""
    pipeline = ModelPipeline()
    pipeline.add_model(ols("1MS", name="m"))
    validation, test = ("2023-03-06", "2023-03-10"), ("2023-04-10", "2023-04-14")
    needed = "2023-02-07"  # 1 March - 21 - 1
    pipeline.run(starting_on(market, needed), validation=validation, test=test, horizon=1)
    with pytest.raises(ValueError, match=f"earliest row they train on is dated {needed}"):
        pipeline.run(starting_on(market, shift(needed, 1)), validation=validation, test=test, horizon=1)


# ---------------------------------------------------------------------------
# A forecast depends on its issue day and the configuration alone
# ---------------------------------------------------------------------------

FULL = ("2023-03-13", "2023-04-16")


def common(a, b):
    shared = a.keys() & b.keys()
    assert shared
    return {key: a[key] for key in shared}, {key: b[key] for key in shared}


@pytest.mark.parametrize("refit", ["7D", "1MS"])
def test_a_forecast_is_the_same_whatever_the_run_around_it(market, refit):
    """Runs starting or ending inside a block, a day alone (as `leakage_probe` runs it), and a run ending on D: every forecast they
    share with the full run is the same, bit for bit, `actual` included. `flag` makes a mask read anywhere but the unit's own rows
    differ: it is 0/1 in the rows of the first days and not in those after 29 March."""
    predictors = PREDICTORS + ["flag"]

    def run(start, end):
        return keyed(ols(refit, predictors).run(market, start, end, horizon=2)._results, "day_in_test")

    full = run(*FULL)
    for start, end in [
        ("2023-03-22", FULL[1]),
        ("2023-04-05", "2023-04-12"),
        ("2023-04-02", "2023-04-02"),
        ("2023-03-26", "2023-03-26"),
        ("2023-04-05", "2023-04-05"),
        (FULL[0], "2023-04-05"),
    ]:
        part = run(start, end)
        assert {key[0] for key in part} == {d for d in {key[0] for key in full} if start <= d <= end}
        mine, theirs = common(part, full)
        assert mine == theirs, (start, end)


def test_one_two_or_three_processes_and_any_order_give_the_same_records(market, monkeypatch):
    predictors = PREDICTORS + ["flag"]

    def run():
        return keyed(ols("7D", predictors).run(market, *FULL, horizon=2)._results)

    expected = run()
    original = BaseModel._run_tasks

    def reversed_tasks(self, tasks, *args, **kwargs):
        return original(self, list(reversed(tasks)), *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(BaseModel, "_run_tasks", reversed_tasks)
        assert run() == expected
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", ProcessPoolExecutor)
    for processes in ("1", "2", "3"):
        monkeypatch.setenv("MAX_PROCESSES", processes)
        assert run() == expected, processes
    monkeypatch.setenv("MAX_PROCESSES", "2")
    monkeypatch.setenv("THREADS_PER_PROCESS", "3")
    assert run() == expected


def test_a_run_cut_short_resumes_to_a_fresh_one(market, tmp_path):
    """A crash leaves some forecasts of a unit stored and others not. The resume fits each unit with pending forecasts again and
    writes those alone: the file is then a fresh run's, line for line."""
    predictors = PREDICTORS + ["flag"]
    fresh, cut = tmp_path / "fresh" / "m.jsonl", tmp_path / "cut" / "m.jsonl"
    ols("7D", predictors).run(market, *FULL, horizon=2, save_to=str(fresh))
    ols("7D", predictors).run(market, *FULL, horizon=2, save_to=str(cut))
    stored = lines(cut)
    kept = [line for line, keep in zip(stored, np.random.default_rng(2).random(len(stored)) < 0.6) if keep]
    cut.write_text("\n".join(kept) + "\n")
    units = {}
    for line in stored:
        record = json.loads(line)
        units.setdefault((record["hour"], record["horizon"], record["refit_anchor"]), []).append(line in kept)
    assert sum(any(v) and not all(v) for v in units.values()) > 100  # many units are stored in part

    calls = []
    original = BaseModel._run_tasks

    def counted(self, tasks, *args, **kwargs):
        calls.append(len(tasks))
        return original(self, tasks, *args, **kwargs)

    BaseModel._run_tasks = counted
    try:
        ols("7D", predictors).run(market, *FULL, horizon=2, save_to=str(cut))
    finally:
        BaseModel._run_tasks = original
    assert calls == [len(stored) - len(kept)]
    assert lines(cut) == lines(fresh)


def test_results_fitted_on_other_anchors_are_refused(market, tmp_path):
    """A stored forecast whose anchor a fresh run wouldn't give it came from other code or another calendar: it is refused, for
    fixed lists and functions alike, though the manifest agrees."""

    def same(ctx):
        return PREDICTORS

    for name, predictors in (("fixed", PREDICTORS), ("function", same)):
        path = tmp_path / name / "m.jsonl"
        ols("7D", predictors).run(market, "2023-03-27", "2023-04-09", horizon=1, save_to=str(path))
        stored = lines(path)
        records = [json.loads(line) for line in stored]
        records[5]["refit_anchor"] = "2023-03-20"
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        with pytest.raises(ManifestMismatchError, match="another refit anchor"):
            ols("7D", predictors).run(market, "2023-03-27", "2023-04-09", horizon=1, save_to=str(path))
        path.write_text("\n".join(stored) + "\n")
        ols("7D", predictors).run(market, "2023-03-27", "2023-04-09", horizon=1, save_to=str(path))
        assert lines(path) == stored


def flag_while_aux(ctx):
    return PREDICTORS + ["flag"] if ctx.published("aux") else PREDICTORS


def matching_while_aux(ctx):
    return Use(PREDICTORS + ["flag"], train_on="matching") if ctx.published("aux") else PREDICTORS


AFTER_D_LISTS = {
    "fixed": (PREDICTORS + ["flag"], "d-1"),
    "function": (flag_while_aux, "d-1"),
    "matching": (matching_while_aux, "d-1"),
    "published": (flag_while_aux, "published"),
}
# Every level of every axis and every pair of levels (kind of list x issue days x horizon x refit), in half of the cells, as
# test_masks.py's grid for refit=None.
AFTER_D_CELLS = [
    ("fixed", None, 1, "7D"),
    ("fixed", ["friday"], 3, "1MS"),
    ("function", None, 3, "7D"),
    ("function", ["friday"], 1, "1MS"),
    ("matching", None, 1, "1MS"),
    ("matching", ["friday"], 3, "7D"),
    ("published", None, 3, "1MS"),
    ("published", ["friday"], 1, "7D"),
]


@pytest.mark.parametrize(("lists", "issue_days", "horizon", "refit"), AFTER_D_CELLS)
def test_nothing_after_the_issue_day_moves_its_forecasts(market, lists, issue_days, horizon, refit):
    """For D = Friday 31 March and Friday 7 April: every finite value dated after D changed (missing ones kept), `aux` missing
    after D (which changes a function's list after D), and the test ending on D. D's predictions, coefficients and anchors stay bit
    for bit, as `test_masks.py` holds for refit=None. `flag` turns from 0/1 to other values on 29 March, so a mask read on another
    unit's rows, or on the first test day's, would show."""
    data = market.assign(aux=1.0)
    fn, newest = AFTER_D_LISTS[lists]
    days = {} if issue_days is None else {"issue_days": issue_days}

    def run(frame, end=FULL[1]):
        model = ols(refit, fn, newest_target=newest)
        return fits(model.run(frame, FULL[0], end, horizon=horizon, information_set=INFO, **days)._results)

    base = run(data)
    dates = local_dates(data)
    columns = [c for c in data.columns if data[c].dtype.kind == "f"]
    for issue in ("2023-03-31", "2023-04-07"):
        on_d = {key: fit for key, fit in base.items() if key[0] == issue}
        assert len(on_d) == 24 * horizon
        after = dates > np.datetime64(issue)
        blank = data.copy()
        blank.loc[after, "aux"] = np.nan
        for what, frame, end in (
            ("values", move(data, after, columns), FULL[1]),
            ("aux", blank, FULL[1]),
            ("end", data, issue),
        ):
            other = run(frame, end)
            assert {key: fit for key, fit in other.items() if key[0] == issue} == on_d, (issue, what)
            if what == "values":  # the change is seen after D
                assert any(other.get(key) != fit for key, fit in base.items() if key[0] > issue)


# ---------------------------------------------------------------------------
# Leakage: nothing after a - 1 - h trains
# ---------------------------------------------------------------------------

# (freq, refit, anchor, block end, frame start): blocks over each switch, at 1 h and 15 min, an anchor on a switch day, and a
# daily frame. 26 March 2023 is an anchor of "2D" and "7D" blocks start on Mondays; 29 October is an anchor of "5D".
LEAK_CASES = {
    "1h-spring": ("1h", "7D", "2023-03-20", "2023-03-26", "2023-02-01"),
    "1h-autumn": ("1h", "7D", "2023-10-23", "2023-10-29", "2023-09-01"),
    "15min-spring": ("15min", "7D", "2023-03-20", "2023-03-26", "2023-02-20"),
    "15min-autumn": ("15min", "7D", "2023-10-23", "2023-10-29", "2023-09-25"),
    "1h-anchor-on-autumn-switch": ("1h", "5D", "2023-10-29", "2023-11-02", "2023-09-01"),
    "15min-anchor-on-spring-switch": ("15min", "2D", "2023-03-26", "2023-03-27", "2023-02-20"),
    "1h-anchor-on-spring-switch": ("1h", "2D", "2023-03-26", "2023-03-27", "2023-02-01"),
    "1D-autumn": ("1D", "7D", "2023-10-23", "2023-10-29", "2023-09-01"),
}
LEAK_WINDOW = 14
INPUTS = [c for c in make_market_data(days=1, horizon=2).columns if c != "price"]


def leak_frame(freq, start, end):
    days = (day(end) - day(start)).days + 4
    hourly = make_market_data(
        start=start, days=days, freq="15min" if freq == "15min" else "1h", horizon=2, noise=5.0, seed=7
    )
    return hourly.at_time("12:00") if freq == "1D" else hourly


@pytest.mark.parametrize("case", list(LEAK_CASES))
def test_nothing_after_a_minus_1_minus_h_trains(case):
    """For issue days D of a block with anchor a: every input of the rows dated after a - 1 - h but D's own, and every target of
    those rows (the price from a on), moved; D's forecasts at h keep their coefficients and value bit for bit. Positive controls:
    the target at a - 1 moves every forecast of the block, and D's inputs move D's forecasts alone. The runs start the day after
    the anchor, so that no forecast of the run is issued on it."""
    freq, refit, anchor_day, last, frame_start = LEAK_CASES[case]
    data = leak_frame(freq, frame_start, last)
    dates = local_dates(data)
    first = shift(anchor_day, 1)

    def run(frame):
        return fits(ols(refit, window=LEAK_WINDOW).run(frame, first, last, horizon=2, freq=freq)._results)

    base = run(data)
    anchor = refit_anchor(first, refit)
    assert anchor == day(anchor_day) and refit_anchor(last, refit) == anchor
    assert {fit[2] for fit in base.values()} == {anchor.isoformat()}
    issues = sorted({key[0] for key in base})
    a = np.datetime64(anchor)
    for issue in [issues[0], issues[len(issues) // 2], issues[-1]]:
        not_d = dates != np.datetime64(issue)
        for horizon in (1, 2):
            frame = move(data, (dates > a - np.timedelta64(1 + horizon, "D")) & not_d, INPUTS)
            frame = move(frame, (dates >= a) & not_d, ["price"], seed=2)
            moved = run(frame)
            on_d = {key: fit for key, fit in base.items() if key[0] == issue and key[2] == horizon}
            assert on_d and {key: moved[key] for key in on_d} == on_d, (issue, horizon)

    target_before = run(move(data, dates == a - np.timedelta64(1, "D"), ["price"]))
    assert all(target_before[key][0] != fit[0] for key, fit in base.items()), "the target at a - 1 trains every unit"
    issue = issues[len(issues) // 2]
    inputs_of_d = run(move(data, dates == np.datetime64(issue), INPUTS))
    for key, fit in base.items():
        assert (inputs_of_d[key][0] != fit[0]) == (key[0] == issue), key
        assert inputs_of_d[key][1] == fit[1]


def test_verify_is_ok_for_a_refit_model(market):
    pipeline = ModelPipeline(information_set=INFO)
    pipeline.add_model(ols("7D", name="weekly"))
    pipeline.add_model(ols("1MS", PREDICTORS + ["flag"], name="monthly"))
    pipeline.verify(market, "2023-03-20", "2023-04-16", horizon=2, days=3).assert_ok()
    # The last day of a "7D" block: a fit anchored on the target day's Monday would train on the target of D, which the probe moves.
    pipeline.verify(market, "2023-04-09", "2023-04-09", horizon=2, days=1).assert_ok()


def same_list(ctx):
    return PREDICTORS


def cut_on_the_last_issue_day(fit_unit):
    """The review's mutant: a unit's rows cut on the last issue day of its forecasts in the run, not on its anchor. A run of one
    day has one forecast per unit, which hides it."""

    def mutant(unit, tasks):
        start = day(worker._W_CONFIG["test_start"])
        last = max(start + datetime.timedelta(days=task[2]) for task in tasks)
        return fit_unit(unit._replace(anchor=last), tasks)

    return mutant


def test_the_probes_run_a_refit_model_over_the_rest_of_its_block(market, monkeypatch):
    """A leak that needs several days of a block in one run: the probes run a refit model from the issue day to the end of its
    block (or of the frame) and compare that day's forecasts, so `leakage_probe` and `verify` see it, and the positive control
    still moves them. With a model that doesn't leak both are clean."""
    monday = "2023-04-03"

    def model():
        return ols("7D")

    pipeline = ModelPipeline(information_set=INFO)
    pipeline.add_model(ols("7D", name="weekly"))
    assert not leakage_probe(model, market, monday, INFO, horizon=2, check_target_boundary=True).changed
    pipeline.verify(market, monday, monday, horizon=2, days=1).assert_ok()
    runs = []
    original = BaseModel.run

    def spy(self, data, test_start=None, test_end=None, *args, **kwargs):
        runs.append((test_start, test_end))
        return original(self, data, test_start, test_end, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(BaseModel, "run", spy)
        control = positive_control(model, market, monday, INFO, horizon=2)
    assert control.ok and len(control.moved) == 48
    assert set(runs) == {(monday, "2023-04-09")}
    end_of_frame = market[local_dates(market) <= np.datetime64("2023-04-05")]
    assert not leakage_probe(model, end_of_frame, monday, INFO, horizon=1, check_target_boundary=True).changed

    monkeypatch.setattr(worker, "_fit_unit", cut_on_the_last_issue_day(worker._fit_unit))
    probe = leakage_probe(model, market, monday, INFO, horizon=2, check_target_boundary=True)
    assert len(probe.changed) == 48
    assert pipeline.verify(market, monday, monday, horizon=2, days=1).status("weekly") == "leak"


@pytest.mark.parametrize("predictors", [PREDICTORS, same_list], ids=["fixed", "function"])
def test_the_training_target_is_checked_on_the_anchors(market, monkeypatch, predictors):
    """A unit's rows are cut as of the issue time on its anchor, so that is where its newest training target, a - 1, must be
    known: the check reads the anchors for the target, and the issue days for the inputs. A target known too late still
    refuses."""
    calls = []
    original = InformationSet.violations

    def spy(
        self, expanded, target, freq, index, test_start=None, test_end=None, issue_days=None, check_target=True, **more
    ):
        names = {name for names in expanded.values() for name in names}
        calls.append((tuple(issue_days or ()), check_target, bool(names)))
        return original(
            self, expanded, target, freq, index, test_start, test_end, issue_days, check_target=check_target, **more
        )

    monkeypatch.setattr(InformationSet, "violations", spy)
    ols("7D", predictors).run(market, "2023-03-01", "2023-03-10", horizon=1, information_set=INFO)
    anchors = ("2023-02-27", "2023-03-06")
    assert (anchors, True, False) in calls
    assert all(days == anchors for days, check_target, _ in calls if check_target)
    monkeypatch.setattr(InformationSet, "violations", original)

    late = InformationSet("11:45", "Europe/Brussels", rules={**RULES, "price": Availability.realised("12h")})
    with pytest.raises(InformationSetError, match="training target"):
        ols("7D", predictors).run(market, "2023-03-01", "2023-03-10", horizon=1, information_set=late)


# ---------------------------------------------------------------------------
# newest_target="published": the cut as of the anchor's issue time
# ---------------------------------------------------------------------------


def test_published_cuts_on_the_anchor_not_on_the_issue_day(market):
    """D's price is known at 11:45 on D (published 13:00 the day before), so the newest target of a unit is its anchor's price:
    the prices of a + 1 to D, known at D but not at a, move nothing on D. The price of a does."""

    def run(frame):
        model = ols("7D", newest_target="published")
        return model.run(frame, "2023-04-03", "2023-04-09", horizon=2, information_set=INFO)._results

    records = run(market)
    assert {r["newest_target_day"] for r in records if r["horizon"] == 1} == {"2023-04-03"}
    assert {r["newest_target_day"] for r in records if r["horizon"] == 2} == {"2023-04-03"}
    base = fits(records)
    dates = local_dates(market)
    a = np.datetime64("2023-04-03")
    for issue in ("2023-04-04", "2023-04-06", "2023-04-09"):
        known_at_d_only = (dates > a) & (dates <= np.datetime64(issue))
        moved = fits(run(move(market, known_at_d_only, ["price"])))
        assert {k: moved[k] for k in base if k[0] == issue} == {k: v for k, v in base.items() if k[0] == issue}
    moved = fits(run(move(market, dates == a, ["price"])))
    assert all(moved[key][0] != fit[0] for key, fit in base.items())


def test_published_reads_its_newest_target_on_the_anchor_when_it_differs_by_weekday():
    """A weekly target, published on Tuesdays at 10:00 for the following Monday-Sunday week. At 11:45 on Monday 3 April the
    newest known target is Sunday 9 April (k = 6); on Tuesday 4 April it is Sunday 16 April, capped at k = 9 by horizon 10. A unit
    anchored on 3 April trains on targets up to 9 April at horizon 10, so the prices of 10 to 12 April, which the Tuesday's issue
    time knows, don't reach the Tuesday's forecasts; the price of 9 April does."""
    data = make_market_data(start="2023-01-01", days=120, horizon=10, noise=5.0, seed=3)
    rules = {**RULES, "price": Availability.weekly("tuesday", "10:00")}
    info = InformationSet("11:45", "Europe/Brussels", rules=rules)

    def run(frame):
        model = ols("7D", newest_target="published")
        return model.run(frame, "2023-04-03", "2023-04-05", horizons=[10], information_set=info)._results

    records = run(data)
    assert {r["newest_target_day"] for r in records} == {"2023-04-09"}
    base = fits(records)
    dates = local_dates(data)
    tuesday = {key: fit for key, fit in base.items() if key[0] == "2023-04-04"}
    next_week = (dates >= np.datetime64("2023-04-10")) & (dates <= np.datetime64("2023-04-12"))
    moved = fits(run(move(data, next_week, ["price"])))
    assert {key: moved[key] for key in tuesday} == tuesday
    moved = fits(run(move(data, dates == np.datetime64("2023-04-09"), ["price"])))
    assert all(moved[key][0] != fit[0] for key, fit in tuesday.items())


# ---------------------------------------------------------------------------
# With the rest of a run's options
# ---------------------------------------------------------------------------


def weekday_matching(ctx):
    """Weekday forecasts train on weekday rows alone; weekend ones on every row, with `flag`."""
    if ctx.issue_day.weekday() < 5:
        return Use(PREDICTORS, train_on="matching")
    return PREDICTORS + ["flag"]


def test_train_on_matching_with_refit_filters_the_anchor_s_rows_and_resumes_bitwise(market, tmp_path):
    fresh, cut = tmp_path / "fresh" / "m.jsonl", tmp_path / "cut" / "m.jsonl"
    start, end = "2023-03-22", "2023-04-12"
    ols("7D", weekday_matching).run(market, start, end, horizon=2, information_set=INFO, save_to=str(fresh))
    records = [json.loads(line) for line in lines(fresh)]
    weekday = [r for r in records if day(r["run_date"]).weekday() < 5]
    assert weekday
    for r in weekday:
        if r["hour"] == 2:  # the spring switch skips 26 March 02:00, so a row's target there is missing and drops
            continue
        anchor = day(r["refit_anchor"])
        span = [anchor - datetime.timedelta(days=d) for d in range(1 + r["horizon"], WINDOW + r["horizon"] + 1)]
        assert r["training_days"] == sum(d.weekday() < 5 for d in span), r
    assert len({r["matching_days"] for r in weekday if r["refit_anchor"] == "2023-03-27" and r["horizon"] == 1}) == 1

    ols("7D", weekday_matching).run(market, start, end, horizon=2, information_set=INFO, save_to=str(cut))
    stored = lines(cut)
    kept = [line for line, keep in zip(stored, np.random.default_rng(3).random(len(stored)) < 0.5) if keep]
    cut.write_text("\n".join(kept) + "\n")
    ols("7D", weekday_matching).run(market, start, end, horizon=2, information_set=INFO, save_to=str(cut))
    assert lines(cut) == lines(fresh)


def two_lists(ctx):
    return PREDICTORS if ctx.issue_day.weekday() in (0, 2, 4) else PREDICTORS + ["flag"]


def test_each_list_of_a_block_has_its_own_fit(market):
    """A function returning two lists inside one block fits a unit per list: each forecast is that of the fixed list it got."""
    start, end = "2023-03-27", "2023-04-09"
    mixed = fits(ols("7D", two_lists).run(market, start, end, horizon=2, information_set=INFO)._results)
    plain = fits(ols("7D").run(market, start, end, horizon=2)._results)
    flagged = fits(ols("7D", PREDICTORS + ["flag"]).run(market, start, end, horizon=2)._results)
    for key, fit in mixed.items():
        expected = plain if day(key[0]).weekday() in (0, 2, 4) else flagged
        assert fit == expected[key], key


def test_issue_days_horizons_and_a_daily_frame_keep_each_forecast(market):
    start, end = "2023-03-20", "2023-04-16"
    every = keyed(ols("7D").run(market, start, end, horizon=3)._results)
    fridays = keyed(ols("7D").run(market, start, end, horizon=3, issue_days=["friday"])._results)
    assert fridays and fridays == {key: r for key, r in every.items() if day(key[0]).weekday() == 4}
    some = keyed(ols("7D").run(market, start, end, horizons=[1, 3])._results)
    assert some and some == {key: r for key, r in every.items() if key[2] in (1, 3)}

    daily = ols("7D").run(market.at_time("12:00"), start, end, horizon=3, freq="1D")._results
    noon = {key: r for key, r in keyed(ols("7D").run(market, start, end, horizon=3)._results, "hour", "freq").items()}
    assert {(k[0], 12, k[2]): r for k, r in keyed(daily, "hour", "freq").items()} == {
        key: r for key, r in noon.items() if key[1] == 12
    }


def test_validation_and_test_windows_give_each_window_s_plain_run(market):
    validation, test = ("2023-03-22", "2023-03-30"), ("2023-04-02", "2023-04-12")
    pipeline = ModelPipeline()
    pipeline.add_model(ols("7D", PREDICTORS + ["flag"], name="m"))
    records = pipeline.run(market, validation=validation, test=test, horizon=2).refs["m"]._results
    for window, (start, end) in (("validation", validation), ("test", test)):
        mine = keyed([r for r in records if r["window"] == window], "window")
        plain = keyed(ols("7D", PREDICTORS + ["flag"]).run(market, start, end, horizon=2)._results)
        assert mine == plain, window


def test_an_ensemble_averages_a_refit_member_on_common_keys(market, tmp_path):
    pipeline = ModelPipeline()
    pipeline.add_model(ols("7D", name="weekly"))
    pipeline.add_model(ols(None, name="daily"))
    pipeline.add_ensemble(AverageEnsemble(["weekly", "daily"], name="avg"))
    report = pipeline.run(market, test=("2023-04-03", "2023-04-09"), horizon=1, save_dir=str(tmp_path))
    counts = report.forecast_counts().set_index("model")
    assert counts.loc[["weekly", "daily", "avg"], "produced"].tolist() == [7 * 24] * 3
    weekly = {json.loads(line)["refit_anchor"] for line in lines(tmp_path / "weekly.jsonl")}
    assert weekly == {"2023-04-03"}
