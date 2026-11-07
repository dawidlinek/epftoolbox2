"""`horizons=`: a run of the horizons it lists, each forecast what a run of every horizon up to the largest gives it.

The equivalences are held key by key against runs with `horizon=H`, which share no code path with a list of horizons where
it matters: `range(1, H + 1)` must be `horizon=H` in every byte, and `range(3, 10)` the h >= 3 records of `horizon=9`. Counts
are written down from the clock by hand, and the probes are held against models that leak on purpose and a control that can't
reach a horizon.
"""

import datetime
import json
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from epftoolbox2 import cli
from epftoolbox2.ensembles import AverageEnsemble
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.information_set import Availability, InformationSet, leakage_probe, positive_control
from epftoolbox2.models import OLSModel, Use
from epftoolbox2.models._calendar import resolve_horizons
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline, Workflow
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path
from tests.synthetic import PREDICTORS, make_market_data

SWITCH_DAYS = {"spring": "2023-03-26", "autumn": "2023-10-29"}
WINDOW = 40
# Driver forecasts out 10 days ahead, so every horizon up to 9 may read them; the A44 rule for the price.
RULES = {
    "load": Availability.published("10:00", 10),
    "wind": Availability.published("10:00", 10),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
}
CALENDAR = ["is_weekend_d+{horizon}", "is_weekday_d+{horizon}"]
TIMESTAMPS = ("created_at", "updated_at")


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


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)


def day_of(day, offset=0):
    return (pd.Timestamp(day) + pd.Timedelta(days=offset)).date().isoformat()


def market(switch, freq="1h", horizon=9, seed=3):
    """`WINDOW` days and more before the days around a switch, with drivers out to `horizon` days."""
    return make_market_data(
        start=day_of(SWITCH_DAYS[switch], -70), days=95, freq=freq, horizon=horizon, noise=5.0, seed=seed
    )


def by_key(records):
    """Records by forecast key, each as its JSON text, so that a NaN `actual` equals itself."""
    return {(r["run_date"], r["hour"], r["horizon"]): json.dumps(r, sort_keys=True) for r in records}


def weekday_lists(ctx):
    """Another list on weekdays, so that a list's mask and its first test day matter."""
    return PREDICTORS + (["price_d-1"] if ctx.issue_day.weekday() < 5 else [])


def weekend_matching(ctx):
    """Forecasts of a weekend train on the days whose forecast was of a weekend too: the training days are planned."""
    return Use(PREDICTORS, train_on="matching") if ctx.target_day.weekday() >= 5 else PREDICTORS


def short_on_far_days(ctx):
    """Reaches horizon 3 through the price of D-1 and horizon 9 through calendar columns alone, which the control can't move."""
    return CALENDAR + (["price_d-1"] if ctx.horizon == 3 else [])


LISTS = {"fixed": PREDICTORS + ["price_d-1"], "function": weekday_lists, "matching": weekend_matching}


# ---------------------------------------------------------------------------
# What horizons= takes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("horizon", "horizons", "expected"),
    [
        (None, None, (7, tuple(range(1, 8)))),
        (3, None, (3, (1, 2, 3))),
        (None, range(1, 4), (3, (1, 2, 3))),
        (None, [7, 1, 3], (7, (1, 3, 7))),
        (None, range(3, 10), (9, tuple(range(3, 10)))),
        (None, (5,), (5, (5,))),
    ],
)
def test_horizons_are_normalised_to_a_sorted_tuple(horizon, horizons, expected):
    assert resolve_horizons(horizon, horizons, 7) == expected


@pytest.mark.parametrize(
    ("horizon", "horizons", "message"),
    [
        (3, [1, 2, 3], "not both"),
        (None, [], "empty"),
        (None, [1, 3, 3], "more than once"),
        (None, [0, 1], "from 1"),
        (None, [1.0, 2], "from 1"),
        (None, [True], "from 1"),
        (None, "1,3", "sequence"),
        (None, 3, "sequence"),
        (0, None, "from 1"),
    ],
)
def test_what_horizons_refuses(horizon, horizons, message):
    with pytest.raises(ValueError, match=message):
        resolve_horizons(horizon, horizons, 7)


def test_both_raise_at_every_entry_point():
    data = market("autumn", horizon=3)
    pipeline = ModelPipeline(information_set=InformationSet(rules=RULES)).add_model(
        OLSModel(PREDICTORS, training_window=WINDOW, name="m")
    )
    day = day_of(SWITCH_DAYS["autumn"])
    for call in (
        lambda: pipeline.run(data, test=(day, day), horizon=3, horizons=[1, 3]),
        lambda: pipeline.report("nowhere", test=(day, day), horizon=3, horizons=[1, 3]),
        lambda: pipeline.verify(data, day, day, horizon=3, horizons=[1, 3], days=1),
        lambda: pipeline.explain(data, "m", horizon=3, horizons=[1, 3], echo=False),
        lambda: pipeline.models[0].run(data, test_start=day, test_end=day, horizon=3, horizons=[1, 3]),
        lambda: leakage_probe(pipeline.models[0], data, day, pipeline.information_set, horizon=3, horizons=[3]),
        lambda: positive_control(pipeline.models[0], data, day, pipeline.information_set, horizon=3, horizons=[3]),
    ):
        with pytest.raises(ValueError, match="not both"):
            call()


@pytest.mark.parametrize("lists", [PREDICTORS + ["price_d+2"], lambda ctx: PREDICTORS + ["price_d+2"]])
def test_a_target_below_the_largest_horizon_is_refused_as_an_input(lists):
    """Targets are built for every horizon up to the largest, so `price_d+2` is the future price at horizons=[3] too."""
    data = market("autumn", horizon=3)
    day = day_of(SWITCH_DAYS["autumn"])
    with pytest.raises(ValueError, match=r"\['price_d\+2'\] name the model's own targets"):
        OLSModel(lists, training_window=WINDOW).run(data, test_start=day, test_end=day, horizons=[3])
    # Above the largest it is no target, and only missing from the frame.
    with pytest.raises(ValueError, match=r"not in the data: \['price_d\+2'\]"):
        OLSModel(lists, training_window=WINDOW).run(data, test_start=day, test_end=day, horizons=[1])


def test_a_plan_for_other_horizons_is_refused():
    data = market("autumn", horizon=3)
    day = day_of(SWITCH_DAYS["autumn"])
    model = OLSModel(weekday_lists, training_window=WINDOW)
    plan = model._plan_predictors(data, "price", 3, "1h", None, day, day, horizons=(1, 3))
    with pytest.raises(ValueError, match="another horizons"):
        model.run(data, test_start=day, test_end=day, horizon=3, plan=plan)
    assert set(plan.assignment) == {(period, h) for period in range(24) for h in (1, 3)}
    model.run(data, test_start=day, test_end=day, horizons=[1, 3], plan=plan)


# ---------------------------------------------------------------------------
# range(1, H + 1) is horizon=H, byte for byte
# ---------------------------------------------------------------------------


def snapshot(directory):
    """Every file of a results directory: records by their sorted lines, manifests without their timestamps."""
    out = {}
    for path in sorted(Path(directory).iterdir()):
        if path.name.endswith(".manifest.json"):
            manifest = json.loads(path.read_text())
            out[path.name] = {key: value for key, value in manifest.items() if key not in TIMESTAMPS}
        else:
            out[path.name] = sorted(line for line in path.read_text().splitlines() if line)
    return out


def small_pipeline(info=None):
    pipeline = ModelPipeline(information_set=info or InformationSet(rules=RULES))
    pipeline.add_model(OLSModel(LISTS["fixed"], training_window=WINDOW, name="fixed"))
    pipeline.add_model(OLSModel(weekday_lists, training_window=WINDOW, name="function"))
    pipeline.add_ensemble(AverageEnsemble(["fixed", "function"], name="average"))
    return pipeline.add_evaluator(MAEEvaluator())


def test_every_horizon_up_to_h_is_horizon_h_in_every_byte(tmp_path):
    data = market("spring", horizon=3)
    windows = {"validation": ("2023-03-20", "2023-03-24"), "test": ("2023-03-25", "2023-03-28")}
    told = small_pipeline().run(data, horizon=3, save_dir=tmp_path / "told", **windows)
    listed = small_pipeline().run(data, horizons=range(1, 4), save_dir=tmp_path / "listed", **windows)

    assert snapshot(tmp_path / "told") == snapshot(tmp_path / "listed")
    assert "horizons" not in json.loads(manifest_path(tmp_path / "listed" / "fixed.jsonl").read_text())
    pd.testing.assert_frame_equal(told.summary(), listed.summary())
    pd.testing.assert_frame_equal(told.validation.forecast_counts(), listed.validation.forecast_counts())
    for name, ref in told.refs.items():
        assert replace(ref, path=None) == replace(listed.refs[name], path=None)
        assert listed.refs[name].horizons is None

    merged = small_pipeline().report(tmp_path / "listed", horizons=[1, 2, 3], **windows)
    pd.testing.assert_frame_equal(merged.summary(), told.summary())
    pd.testing.assert_frame_equal(merged.forecast_counts(), told.forecast_counts())

    in_memory = OLSModel(weekday_lists, training_window=WINDOW).run(data, *windows["test"], horizons=(3, 2, 1))
    plain = OLSModel(weekday_lists, training_window=WINDOW).run(data, *windows["test"], horizon=3)
    assert by_key(in_memory._results) == by_key(plain._results)  # in the order days finish, which isn't fixed
    assert replace(in_memory, _results=None) == replace(plain, _results=None)

    def workflow(**horizon):
        return Workflow(data_start="2023-01-01", data_end="2023-04-01", model_test=windows["test"], **horizon)

    assert workflow(model_horizons=range(1, 4)).to_dict() == workflow(model_horizon=3).to_dict()


# ---------------------------------------------------------------------------
# range(3, 10) is the h >= 3 part of horizon=9
# ---------------------------------------------------------------------------

# Every level of each axis, and every pair of levels of two axes, as tests/models rules ask: switch x freq x kind of list.
CELLS = [
    ("spring", "1h", "fixed"),
    ("spring", "15min", "function"),
    ("autumn", "1h", "function"),
    ("autumn", "15min", "fixed"),
    ("autumn", "1h", "matching"),
]


@pytest.mark.parametrize(("switch", "freq", "lists"), CELLS)
def test_a_gap_in_the_list_changes_nothing_for_the_others(switch, freq, lists):
    """Issue days from S-9 to S+2: every horizon from 3 to 9 has a key whose target day is the switch, one issued on it and some
    whose window holds it."""
    data = market(switch, freq=freq)
    S = SWITCH_DAYS[switch]
    first, last = day_of(S, -9), day_of(S, 2)

    def run(**horizon):
        model = OLSModel(LISTS[lists], training_window=WINDOW)
        return by_key(model.run(data, test_start=first, test_end=last, freq=freq, **horizon)._results)

    every = run(horizon=9)
    listed = run(horizons=range(3, 10))
    expected = {key: record for key, record in every.items() if key[2] >= 3}
    assert listed == expected
    on_the_switch = {key for key in listed if key[0] == S or day_of(key[0], key[2]) == S}
    assert {key[2] for key in on_the_switch} == set(range(3, 10))


def test_a_gap_changes_nothing_under_newest_target_published():
    """Price published 13:00 on D-1 and issued 11:45: the newest target is D's, at every period, so h=3 trains up to D-3."""
    data = market("autumn")
    S = SWITCH_DAYS["autumn"]
    info = InformationSet("11:45", "Europe/Brussels", rules={**RULES, "price": Availability.published("13:00", 1)})

    def run(**horizon):
        model = OLSModel(PREDICTORS, training_window=WINDOW, newest_target="published")
        return by_key(model.run(data, day_of(S, -9), day_of(S, 2), information_set=info, **horizon)._results)

    every = run(horizon=9)
    listed = run(horizons=[3, 5, 9])
    assert listed == {key: record for key, record in every.items() if key[2] in (3, 5, 9)}
    newest = {json.loads(r)["newest_target_day"] for key, r in listed.items() if key[0] == S and key[2] == 3}
    assert newest == {day_of(S, 0)}


# ---------------------------------------------------------------------------
# [1, 3, 7] end to end
# ---------------------------------------------------------------------------

THREE = [1, 3, 7]
SPRING_TEST = ("2023-03-20", "2023-03-31")


def crash(path, kept):
    """What a run that stopped after `kept` forecasts leaves: the first of them in the file."""
    path = Path(path)
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:kept]))


def planned_by_hand(first, last, horizons, switch=SWITCH_DAYS["spring"], periods=24, skipped=(2,)):
    """Keys of the test days whose issue day and target day both have the period on the clock (Warsaw skips 02:00 on S)."""
    count = 0
    day = datetime.date.fromisoformat(first)
    while day <= datetime.date.fromisoformat(last):
        for h in horizons:
            on_switch = switch in (day.isoformat(), (day + datetime.timedelta(days=h)).isoformat())
            count += periods - (len(skipped) if on_switch else 0)
        day += datetime.timedelta(days=1)
    return count


@pytest.fixture(scope="module")
def spring_data():
    return make_market_data(start="2023-01-01", days=110, horizon=7, noise=5.0, seed=5)


def test_a_list_of_horizons_runs_resumes_and_reports(tmp_path, spring_data):
    small_pipeline().run(spring_data, test=SPRING_TEST, horizons=THREE, save_dir=tmp_path / "a")
    for name, kept in (("fixed", 300), ("function", 451)):
        crash(tmp_path / "a" / f"{name}.jsonl", kept)
    small_pipeline().run(spring_data, test=SPRING_TEST, horizons=THREE, save_dir=tmp_path / "a")
    small_pipeline().run(spring_data, test=SPRING_TEST, horizons=THREE, save_dir=tmp_path / "fresh")
    small_pipeline().run(spring_data, test=SPRING_TEST, horizon=7, save_dir=tmp_path / "every")

    resumed = snapshot(tmp_path / "a")
    assert resumed == snapshot(tmp_path / "fresh")
    every = snapshot(tmp_path / "every")
    for name in ("fixed.jsonl", "function.jsonl", "average.jsonl"):
        kept = [line for line in every[name] if json.loads(line)["horizon"] in THREE]
        assert resumed[name] == kept
    manifest = resumed["fixed.manifest.json"]
    assert (manifest["horizon"], manifest["horizons"]) == (7, THREE)

    # The day after the switch has no price of D-1 at 02:00, which the clock skipped: one key per horizon goes without.
    with pytest.warns(UserWarning, match=r"fixed \(856 of 859, 3 missing, missing_test_predictors 3\)"):
        report = small_pipeline().report(tmp_path / "a", test=SPRING_TEST, horizons=THREE)
    counts = report.forecast_counts().set_index("model")
    planned = planned_by_hand(*SPRING_TEST, THREE)
    assert planned == 12 * 3 * 24 - 5  # issued on S at three horizons, aimed at S from S-1 and S-3
    assert counts["planned"].tolist() == [planned] * 3
    assert counts["missing"].tolist() == [3, 3, 3]
    assert counts["missing_test_predictors"].tolist() == [3, 3, 0]

    assert sorted(report.by_horizon()["horizon"].unique().tolist()) == THREE
    three = report.tests(benchmark="fixed", horizon=3, reps=50)
    assert three["horizon"].unique().tolist() == [3] and int(three["n_forecasts"].iloc[0]) == 12 * 24 - 3
    pooled = report.tests(benchmark="fixed", horizon="all", reps=50)
    assert pooled["horizon"].unique().tolist() == ["all"]
    with pytest.raises(ValueError, match="horizon must be"):
        report.tests(benchmark="fixed", horizon=2, reps=50)

    # A report of one horizon of the three reads that one alone, the ensemble's records too.
    only_three = small_pipeline().report(tmp_path / "a", test=SPRING_TEST, horizons=[3])
    assert only_three.by_horizon().groupby("model")["horizon"].unique().map(list).to_dict() == {
        name: [3] for name in ("fixed", "function", "average")
    }
    counts = only_three.forecast_counts().set_index("model")
    assert counts["planned"].tolist() == [planned_by_hand(*SPRING_TEST, [3])] * 3 == [12 * 24 - 2] * 3
    assert counts["produced"].tolist() == [12 * 24 - 3] * 3
    # The manifest's skip reasons are of all three horizons and can't be split: a report of one has `missing` alone.
    assert "missing_test_predictors" not in counts.columns
    assert counts["missing"].tolist() == [1, 1, 1]

    # Told every horizon to 7, the models and the ensemble plan seven and lack four.
    with pytest.warns(UserWarning, match="missing forecasts"):
        told_seven = small_pipeline().report(tmp_path / "a", test=SPRING_TEST, horizon=7)
    counts = told_seven.forecast_counts().set_index("model")
    assert counts["planned"].tolist() == [planned_by_hand(*SPRING_TEST, range(1, 8))] * 3
    assert counts["produced"].tolist() == [planned_by_hand(*SPRING_TEST, THREE) - 3] * 3

    # Told horizon 3, only horizons 1 and 3 are read: 7 is no forecast of it, and 2 is missing.
    with pytest.warns(UserWarning, match="missing forecasts"):
        told_three = small_pipeline().report(tmp_path / "a", test=SPRING_TEST, horizon=3)
    assert sorted(told_three.by_horizon()["horizon"].unique().tolist()) == [1, 3]
    counts = told_three.forecast_counts().set_index("model")
    assert counts["planned"].tolist() == [planned_by_hand(*SPRING_TEST, [1, 2, 3])] * 3
    assert counts["produced"].tolist() == [planned_by_hand(*SPRING_TEST, [1, 3]) - 2] * 3
    assert counts["missing"].tolist() == [planned_by_hand(*SPRING_TEST, [2]) + 2] * 3
    assert "missing_test_predictors" not in counts.columns


@pytest.mark.parametrize("other", [{"horizons": [1, 3]}, {"horizon": 7}, {"horizons": [1, 3, 6]}])
def test_a_resume_with_other_horizons_is_refused(tmp_path, spring_data, other):
    pipeline = small_pipeline()
    pipeline.run(spring_data, test=SPRING_TEST, horizons=THREE, save_dir=tmp_path)
    with pytest.raises(ManifestMismatchError, match="horizon"):
        small_pipeline().run(spring_data, test=SPRING_TEST, save_dir=tmp_path, **other)


def test_an_ensemble_of_members_run_with_other_horizons_is_refused(tmp_path, spring_data):
    """One largest horizon, two lists of them: the members plan different keys."""
    one = ModelPipeline(information_set=InformationSet(rules=RULES))
    one.add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="a")).run(
        spring_data, test=SPRING_TEST, horizons=THREE, save_dir=tmp_path
    )
    other = ModelPipeline(information_set=InformationSet(rules=RULES))
    other.add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="b")).run(
        spring_data, test=SPRING_TEST, horizon=7, save_dir=tmp_path
    )
    both = ModelPipeline(information_set=InformationSet(rules=RULES))
    both.add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="a"))
    both.add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="b"))
    both.add_ensemble(AverageEnsemble(["a", "b"], name="e"))
    with pytest.raises(ManifestMismatchError, match=r"horizons \{'a': \[1, 3, 7\], 'b': \[1, 2, 3, 4, 5, 6, 7\]\}"):
        both.report(tmp_path, test=SPRING_TEST, horizons=THREE)


def test_validation_forecasts_past_the_first_test_issue_count_per_horizon(tmp_path, spring_data):
    """The last validation day is the day before the test: at horizon h its last h issue days aim at the test."""
    windows = {"validation": ("2023-02-01", "2023-02-10"), "test": ("2023-02-11", "2023-02-15")}
    report = small_pipeline().run(spring_data, horizons=THREE, save_dir=tmp_path, **windows)
    excluded = (1 + 3 + 7) * 24
    counts = report.validation.forecast_counts().set_index("model")
    assert counts["target_in_test"].tolist() == [excluded] * 3
    assert counts["planned"].tolist() == [10 * 3 * 24] * 3
    assert counts["produced"].tolist() == [10 * 3 * 24 - excluded] * 3

    merged = small_pipeline().report(tmp_path, horizons=THREE, **windows)
    pd.testing.assert_frame_equal(merged.validation.forecast_counts(), report.validation.forecast_counts())
    one = ModelPipeline(information_set=InformationSet(rules=RULES))
    every = one.add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="m")).run(spring_data, horizon=7, **windows)
    assert every.validation.forecast_counts()["target_in_test"].tolist() == [(1 + 2 + 3 + 4 + 5 + 6 + 7) * 24]


# ---------------------------------------------------------------------------
# verify, the probe and the control
# ---------------------------------------------------------------------------

VERIFY_DAY = "2023-04-12"


@pytest.fixture(scope="module")
def april():
    return make_market_data(start="2023-01-01", days=130, horizon=9, noise=5.0, seed=6)


def verified(model, data, rules=RULES, **horizon):
    pipeline = ModelPipeline(information_set=InformationSet(rules=rules)).add_model(model)
    return pipeline.verify(data, "2023-04-10", "2023-04-14", days=2, **horizon)


def test_verify_checks_the_listed_horizons_and_no_others(april):
    report = verified(OLSModel(PREDICTORS + ["price_d-1"], training_window=WINDOW, name="m"), april, horizons=[3, 9])
    report.assert_ok()
    assert {key[1] for check in report.checks for key in check.control.moved} == {3, 9}
    assert all(check.probe.forecasts == 2 * 24 for check in report.checks)


def test_verify_says_no_forecast_only_of_a_listed_horizon(april):
    """No driver value at horizon 9: nothing forecast there. Horizons 1, 2 and 4 to 8 aren't asked for, so not missing."""
    data = april.assign(**{name: float("nan") for name in ("load_d+9", "wind_d+9")})
    report = verified(OLSModel(PREDICTORS + ["price_d-1"], training_window=WINDOW, name="m"), data, horizons=[3, 9])
    assert report.status("m") == "inconclusive"
    assert [line for line in report.problems() if "no forecast of horizon" in line] == [
        "m: inconclusive: no forecast of horizon(s) [9] on any sampled issue day, so nothing checked them."
    ]


def test_the_probe_finds_a_leak_at_a_listed_horizon(april):
    """Driver forecasts out 7 days only: at horizon 9 the drivers of D + 9 aren't out on D."""
    rules = {**RULES, "load": Availability.published("10:00", 7), "wind": Availability.published("10:00", 7)}
    model = OLSModel(PREDICTORS + ["price_d-1"], training_window=WINDOW, name="m")
    info = InformationSet(rules=rules)
    probe = leakage_probe(model, april, VERIFY_DAY, info, horizons=[3, 9], check_target_boundary=True)
    assert not probe.ok
    assert {h for _, h in probe.changed} == {9}
    assert probe.forecasts == 2 * 24
    assert leakage_probe(model, april, VERIFY_DAY, info, horizons=[3], check_target_boundary=True).ok
    assert verified(model, april, rules=rules, horizons=[3, 9]).status("m") == "leak"


def test_the_control_can_fail_at_a_listed_horizon(april):
    """A control that can't fail is no control: horizon 9 reads calendar columns alone, which no recent value moves."""
    model = OLSModel(short_on_far_days, training_window=WINDOW, name="m")
    info = InformationSet(rules=RULES)
    control = positive_control(model, april, VERIFY_DAY, info, horizons=[3, 9])
    assert {h for _, h in control.moved} == {3}
    assert {h for _, h in control.unmoved} == {9}
    report = verified(model, april, horizons=[3, 9])
    assert report.status("m") == "inconclusive"
    assert any("moved no forecast of horizon(s) [9]" in line for line in report.problems())

    calendar = OLSModel(CALENDAR, training_window=WINDOW, name="m")
    assert not positive_control(calendar, april, VERIFY_DAY, info, horizons=[3, 9]).ok


def test_explain_lists_the_horizons_of_the_run(april):
    pipeline = ModelPipeline(information_set=InformationSet(rules=RULES))
    pipeline.add_model(OLSModel(short_on_far_days, training_window=WINDOW, name="m"))
    out = pipeline.explain(april, "m", test=("2023-04-10", "2023-04-12"), horizons=[3, 9], echo=False)
    assert sorted(out["lists"]["horizon"].unique().tolist()) == [3, 9]
    assert out["predictors"]["predictor"].tolist() == ["is_weekend_d+9", "is_weekday_d+9"]


# ---------------------------------------------------------------------------
# YAML and the command line
# ---------------------------------------------------------------------------


def test_a_workflow_keeps_its_horizons_through_yaml(tmp_path):
    workflow = Workflow(
        data_start="2023-01-01", data_end="2023-04-01", model_test=SPRING_TEST, model_horizons=[7, 1, 3]
    )
    workflow.save(tmp_path / "experiment.yaml")
    written = yaml.safe_load((tmp_path / "experiment.yaml").read_text())["model_pipeline"]
    assert written["horizons"] == THREE and "horizon" not in written
    loaded = Workflow.load(tmp_path / "experiment.yaml")
    assert (loaded.model_horizon, list(loaded.model_horizons)) == (None, THREE)
    assert loaded.to_dict() == workflow.to_dict()

    default = Workflow(data_start="2023-01-01", data_end="2023-04-01", model_test=SPRING_TEST)
    assert default.to_dict()["model_pipeline"]["horizon"] == 7

    written["horizon"] = 7
    (tmp_path / "both.yaml").write_text(
        yaml.safe_dump({"data_pipeline": {"start": "x", "end": "y"}, "model_pipeline": written})
    )
    with pytest.raises(ValueError, match="not both"):
        Workflow.load(tmp_path / "both.yaml")


def test_a_workflow_runs_and_reports_its_horizons(tmp_path):
    index = pd.date_range("2023-01-01", "2023-04-01", freq="1h", tz="UTC", inclusive="left")
    prices = 100 + 20 * pd.Series(range(len(index))).mod(24).to_numpy() / 24.0
    pd.DataFrame({"datetime": index.strftime("%Y-%m-%d %H:%M:%S"), "price": prices}).to_csv(
        tmp_path / "prices.csv", index=False
    )
    config = {
        "data_pipeline": {
            "start": "2023-01-01",
            "end": "2023-03-31",
            "sources": [{"class": "CsvSource", "params": {"file_path": str(tmp_path / "prices.csv")}}],
            "transformers": [
                {"class": "TimezoneTransformer", "params": {"target_tz": "Europe/Warsaw"}},
                {"class": "LagTransformer", "params": {"columns": ["price"], "lags": [1, 7], "freq": "day"}},
            ],
        },
        "model_pipeline": {
            "test": ["2023-03-01", "2023-03-02"],
            "horizons": [2, 4],
            "save_dir": "results",
            "models": [
                {"class": "OLSModel", "params": {"predictors": ["price_d-1", "price_d-7"], "training_window": 30}}
            ],
        },
    }
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(config))
    Workflow.load(tmp_path / "experiment.yaml").run()
    lines = (tmp_path / "results" / "model.jsonl").read_text().splitlines()
    assert sorted({json.loads(line)["horizon"] for line in lines}) == [2, 4]
    counts = Workflow.load(tmp_path / "experiment.yaml").report().forecast_counts()
    assert counts[["planned", "produced"]].values.tolist() == [[2 * 2 * 24, 2 * 2 * 24]]


@pytest.mark.parametrize(
    ("text", "expected"), [("3-9", tuple(range(3, 10))), ("1,3,7", (1, 3, 7)), ("1-3,7", (1, 2, 3, 7)), ("7,1", (1, 7))]
)
def test_the_command_line_reads_ranges_and_lists(text, expected):
    args = cli.build_parser().parse_args(["model", "mp.yaml", "--data", "d.csv", "--horizons", text])
    assert (args.horizon, args.horizons) == (None, expected)


@pytest.mark.parametrize(
    "argv", [["--horizons", "1,1"], ["--horizons", "9-3"], ["--horizons", "a-b"], ["--horizon", "3", "--horizons", "3"]]
)
@pytest.mark.parametrize("command", ["model", "verify"])
def test_the_command_line_refuses_what_isn_t_a_list(command, argv):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([command, "mp.yaml", "--data", "d.csv", *argv])


def test_epf_model_with_horizons_gives_the_in_process_forecasts(tmp_path, spring_data):
    from epftoolbox2._frame_csv import save_frame_csv

    predictors = PREDICTORS + ["price_d-1"]
    ModelPipeline().add_model(OLSModel(predictors, training_window=WINDOW, name="OLS")).save(tmp_path / "mp.yaml")
    data = spring_data[["price", "load", "wind", "is_weekend", "is_weekday", "price_d-1"]].join(
        spring_data[[f"{name}_d+{h}" for h in (1, 3) for name in ("load", "wind", "is_weekend", "is_weekday")]]
    )
    save_frame_csv(data, tmp_path / "data.csv")
    argv = ["model", str(tmp_path / "mp.yaml"), "--data", str(tmp_path / "data.csv"), "--test-start", "2023-03-24"]
    argv += ["--test-end", "2023-03-27", "--horizons", "1,3", "--save-dir", str(tmp_path / "out")]
    cli.cmd_model(cli.build_parser().parse_args(argv))

    stored = by_key(
        r
        for r in map(json.loads, (tmp_path / "out" / "ols.jsonl").read_text().splitlines())
        if r["status"] == "ok"  # not the skip records
    )
    in_process = by_key(
        OLSModel(predictors, training_window=WINDOW, name="OLS")
        .run(data, test_start="2023-03-24", test_end="2023-03-27", horizons=[1, 3])
        ._results
    )
    assert {key: json.loads(r)["prediction"] for key, r in stored.items()} == {
        key: json.loads(r)["prediction"] for key, r in in_process.items()
    }
    assert {key[2] for key in stored} == {1, 3}


def test_a_report_told_fewer_horizons_than_a_run_of_every_one_reads_those_alone(tmp_path, spring_data):
    """`report(horizon=3)` of a horizon-7 run: horizons 4 to 7 are no forecasts of it, whatever the file holds."""
    small_pipeline().run(spring_data, test=SPRING_TEST, horizon=7, save_dir=tmp_path)
    report = small_pipeline().report(tmp_path, test=SPRING_TEST, horizon=3)
    assert sorted(report.by_horizon()["horizon"].unique().tolist()) == [1, 2, 3]
    counts = report.forecast_counts().set_index("model")
    planned = planned_by_hand(*SPRING_TEST, [1, 2, 3])
    assert counts["planned"].tolist() == [planned] * 3
    assert counts["produced"].tolist() == [planned - 3] * 3  # S+1 has no price of D-1 at 02:00, one key per horizon


def test_a_report_told_no_horizon_plans_what_the_runs_had(tmp_path, spring_data):
    small_pipeline().run(spring_data, test=SPRING_TEST, horizons=THREE, save_dir=tmp_path / "a")
    with pytest.warns(UserWarning, match="missing_test_predictors 3"):
        found = small_pipeline().report(tmp_path / "a", test=SPRING_TEST)
    with pytest.warns(UserWarning, match="missing_test_predictors 3"):
        told = small_pipeline().report(tmp_path / "a", test=SPRING_TEST, horizons=THREE)
    pd.testing.assert_frame_equal(found.forecast_counts(), told.forecast_counts())
    counts = found.forecast_counts().set_index("model")
    assert counts["planned"].tolist() == [planned_by_hand(*SPRING_TEST, THREE)] * 3
    assert counts["missing"].tolist() == [3, 3, 3]  # every one explained: S+1 has no price of D-1 at 02:00
    assert sorted(found.by_horizon()["horizon"].unique().tolist()) == THREE

    # Without a manifest there is nothing to read the list from: 1 to 7, as before, and no zone, so every period of every day.
    (tmp_path / "bare").mkdir()
    (tmp_path / "bare" / "fixed.jsonl").write_text((tmp_path / "a" / "fixed.jsonl").read_text())
    bare = ModelPipeline().add_model(OLSModel(PREDICTORS, name="fixed"))
    with pytest.warns(UserWarning, match="missing forecasts"):
        counts = bare.report(tmp_path / "bare", test=SPRING_TEST).forecast_counts()
    assert counts["planned"].tolist() == [12 * 7 * 24]


def test_models_run_with_other_horizons_need_the_report_told_which(tmp_path, spring_data):
    one = ModelPipeline(information_set=InformationSet(rules=RULES))
    one.add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="a"))
    one.run(spring_data, test=SPRING_TEST, horizons=THREE, save_dir=tmp_path)
    other = ModelPipeline(information_set=InformationSet(rules=RULES))
    other.add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="b"))
    other.run(spring_data, test=SPRING_TEST, horizon=7, save_dir=tmp_path)
    both = ModelPipeline(information_set=InformationSet(rules=RULES))
    both.add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="a"))
    both.add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="b"))
    with pytest.raises(
        ValueError, match=r"different horizons \(a: \[1, 3, 7\], b: \[1, 2, 3, 4, 5, 6, 7\]\).*horizons=\[1, 3, 7\]"
    ):
        both.report(tmp_path, test=SPRING_TEST)
    counts = both.report(tmp_path, test=SPRING_TEST, horizons=THREE).forecast_counts().set_index("model")
    assert counts["planned"].tolist() == [planned_by_hand(*SPRING_TEST, THREE)] * 2


def by_horizon_quality(horizon):
    """Worse at horizons 1, 3 and 7 than at the others: a choice pooled over another set of horizons can come out otherwise."""
    return PREDICTORS if horizon in (2, 4, 5, 6) else ["load_d+{horizon}", "is_weekday_d+{horizon}"]


def test_a_top_choice_pools_the_horizons_the_run_made(tmp_path, spring_data):
    """Not a look-ahead, and not the run of every horizon: a `top` ensemble chooses on the validation forecasts of the horizons
    on file, by hand here, also in a report told fewer of them."""
    windows = {"validation": ("2023-02-10", "2023-03-05"), "test": ("2023-03-06", "2023-03-12")}

    def pipeline():
        built = ModelPipeline(information_set=InformationSet(rules=RULES))
        built.add_model(OLSModel(by_horizon_quality, training_window=30, name="m1"))
        built.add_model(OLSModel(["load_d+{horizon}", *CALENDAR], training_window=30, name="m2"))
        return built.add_ensemble(AverageEnsemble(["m1", "m2"], top=1, name="top1"))

    listed = pipeline().run(spring_data, horizons=THREE, save_dir=tmp_path / "list", **windows).choices()
    every = pipeline().run(spring_data, horizon=7, save_dir=tmp_path / "all", **windows).choices()

    def serving(name):
        records = (json.loads(line) for line in (tmp_path / "list" / f"{name}.jsonl").read_text().splitlines())
        return {
            (r["run_date"], r["target_date"], r["hour"], r["horizon"]): (r["prediction"], r["actual"])
            for r in records
            if windows["validation"][0] <= r["run_date"] <= windows["validation"][1] and r["target_date"] < "2023-03-06"
        }

    m1, m2 = serving("m1"), serving("m2")
    keys = sorted(key for key in m1.keys() & m2.keys() if all(pd.notna(v) for v in (*m1[key], *m2[key])))
    assert {key[3] for key in keys} == set(THREE)
    mae = {
        name: sum(abs(p - a) for p, a in (held[key] for key in keys)) / len(keys)
        for name, held in (("m1", m1), ("m2", m2))
    }
    chosen = min(mae, key=mae.get)
    row = listed.iloc[0]
    assert list(row["chosen"]) == [chosen] and int(row["validation_forecasts"]) == len(keys)
    assert row["validation_mae"] == pytest.approx(mae[chosen], rel=1e-12)
    assert list(every.iloc[0]["chosen"]) != [chosen]  # pooled over 1 to 7, the other member wins

    again = pipeline().report(tmp_path / "list", horizons=[3], **windows).choices()
    pd.testing.assert_frame_equal(again, listed)
