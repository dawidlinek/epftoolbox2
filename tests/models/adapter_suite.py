"""The checks every model family's adapter passes (step 6c of the step 6 spec), as functions its test module calls.

A family's test module gives a factory, `make(name, **params) -> model`, which returns a fresh model named `name` each time
(`params` are the family's own, such as a training window, for the variants of a loop), and the frames to run on, then calls
the `check_*` functions it needs. The module sets one process and one thread for every run (`MAX_PROCESSES=1`,
`THREADS_PER_PROCESS=1`, `.claude/rules/tests.md`), and registers its fake classes in `COMPONENT_REGISTRY` for the YAML check.

- `check_yaml_round_trip`: saved and loaded, the same parameters and the same forecasts.
- `check_preflight_without_extra`: with the family's extra hidden (`importlib.util.find_spec`), `run` raises
  `MissingExtraError` before any model's run starts.
- `check_verify`: `verify` is `ok` on each day given, with the positive control, the rerun and the co-batch checks, on the CPU;
  for a family that predicts its refit blocks in one call, the issue day's forecasts on the frame cut after its date are the
  whole frame's, to the bit.
- `check_loop_and_shards`: each model of a loop equals the same model run alone, as a `models_per_task=1` array task runs it
  (bitwise, records as sorted lines).
- `check_resume`: a run resumed after 40 % of its records were deleted equals a fresh run (bitwise).
- `check_engine`: `engine="on"` gives what `"off"` gives, and the engine takes no such model (`_engine_capable()` is False).
- `check_dst_counts`: the calendar's counts over both switches: 23 forecasts for the spring target day, 24 for the autumn
  one, the repeated hour's actual NaN, nothing missing.
- `check_quarter_hours`: on a 15-minute frame, 96 forecasts a day and nothing missing (for a family that supports it).
- `check_provenance`: the card is in `report.provenance()`.

Pieces of step 6a are read through the attributes the spec names, with a default, so that 6a slots in without a change here:
`model._card` (None: a family with no card), `BaseModel._preflight` (absent: `check_preflight_without_extra` skips), and
`EvaluationReport.provenance` (absent: `check_provenance` skips). A card's `extra`, `family` and `licence` are read as
`getattr(card, name, None)`.
"""

import datetime
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

from epftoolbox2._extras import EXTRAS, MissingExtraError
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.information_set import InformationSet, cuts_at_issue_day
from epftoolbox2.models.base import BaseModel
from epftoolbox2.pipelines import ModelPipeline

Factory = Callable[..., BaseModel]
NAME = "Adapter"


def lines(path: Path) -> List[str]:
    """A results file's records as sorted text: the order a pool finishes them in isn't fixed."""
    return sorted(line for line in Path(path).read_text().splitlines() if line.strip())


def results(save_dir: Path, name: str = NAME) -> Path:
    """The JSONL results file of the model `name` in `save_dir` (`ModelPipeline.results_path`)."""
    return ModelPipeline.results_path(save_dir, SimpleNamespace(name=name), "jsonl")


def forecasts(report: Any, name: str = NAME) -> Dict[Tuple[str, int, int], Tuple[float, Any]]:
    """(run_date, hour, horizon) -> (prediction, coefficients) of one model of a run's report, in memory."""
    return {
        (r["run_date"], r["hour"], r["horizon"]): (r["prediction"], r.get("coefficients"))
        for r in report.refs[name]._results
    }


def _pipeline(
    models: Iterable[BaseModel], information_set: Optional[InformationSet] = None, **settings
) -> ModelPipeline:
    pipeline = ModelPipeline(information_set=information_set, **settings)
    for model in models:
        pipeline.add_model(model)
    return pipeline


def _run(pipeline: ModelPipeline, frame: pd.DataFrame, test: Tuple[str, str], horizon: int, **kwargs) -> Any:
    return pipeline.run(frame, *test, horizon=horizon, forecast_only=True, **kwargs)


def check_yaml_round_trip(
    make: Factory,
    frame: pd.DataFrame,
    tmp_path: Path,
    test: Tuple[str, str],
    horizon: int = 1,
    information_set: Optional[InformationSet] = None,
) -> None:
    """Saved to YAML and loaded back, the model has the same class, parameters and forecasts, and saves to the same text."""
    model = make(NAME)
    first = tmp_path / "pipeline.yaml"
    _pipeline([model], information_set).save(first)
    loaded = ModelPipeline.load(first)
    (again,) = loaded.models
    assert type(again) is type(model)
    assert again._model_kwargs == model._model_kwargs
    second = tmp_path / "again.yaml"
    loaded.save(second)
    assert second.read_text() == first.read_text()
    original = forecasts(_run(_pipeline([make(NAME)], information_set), frame, test, horizon))
    assert original, "the model made no forecast"
    assert forecasts(_run(loaded, frame, test, horizon)) == original


def check_preflight_without_extra(
    make: Factory, monkeypatch: Any, frame: pd.DataFrame, test: Tuple[str, str], horizon: int = 1
) -> None:
    """With the modules of the family's extra hidden, `ModelPipeline.run` raises `MissingExtraError` naming the extra before
    any model's run starts (no `_prepare_run`), as 6a's pre-flight does. A family with no card, or a card without an extra,
    has nothing to miss; before 6a (no `BaseModel._preflight`) the check skips."""
    model = make(NAME)
    extra = getattr(getattr(model, "_card", None), "extra", None)
    if extra is None:
        return
    if not hasattr(BaseModel, "_preflight"):
        pytest.skip("the pre-flight arrives with step 6a (`BaseModel._preflight`)")
    hidden = {module.split(".")[0] for module in EXTRAS[extra]}
    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        return None if name.split(".")[0] in hidden else real_find_spec(name, *args, **kwargs)

    started = []
    real_prepare = BaseModel._prepare_run

    def prepare(self, *args, **kwargs):
        started.append(self.name)
        return real_prepare(self, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(BaseModel, "_prepare_run", prepare)
    with pytest.raises(MissingExtraError, match=extra):
        _run(_pipeline([model]), frame, test, horizon)
    assert started == [], f"a model's run started before the pre-flight refused it: {started}"


def check_verify(
    make: Factory,
    frame: pd.DataFrame,
    information_set: InformationSet,
    days: Sequence[str],
    horizon: int = 1,
    items: Optional[int] = None,
    **verify_kwargs: Any,
) -> Any:
    """`verify` is `ok` on each of `days` (each verified on its own), with the positive control, the rerun and the co-batch
    checks on, on the CPU for a model with a `device`: no forecast moves under the probe, every one moves under the control,
    two runs are the same and no forecast moves when the other items of its call are replaced by noise. A family that predicts
    its refit blocks in one call (`_predict_block`) gives the same bits on the frame cut after the issue day's date, the rows a
    run on that day has, as on the whole frame (`CoBatchResult.cut`, run on every day that names other items). Give `days`
    that include the first, a middle and the last day of a block, and the days around a DST switch.

    A model that predicts together (`_predicts_together`) must name the other items of its calls on some day, so that the
    co-batch check isn't vacuous; with `items`, it names that many on every day (a block's length less one, where the frame has
    every day of it). Returns the last day's report."""
    model = make(NAME)
    device = "cpu" if hasattr(model, "device") else None
    report = None
    named = []
    for day in days:
        report = _pipeline([make(NAME)], information_set).verify(
            frame, day, day, horizon=horizon, days=1, rerun=True, co_batch=True, device=device, **verify_kwargs
        )
        report.assert_ok()
        row = report.table().iloc[0]
        assert (row["changed"], row["rerun"], row["co_batch"]) == (0, 0, 0), row.to_dict()
        assert row["control moved"] > 0
        (check,) = report.checks
        assert check.co_batch.items is not None, "it predicts together but names no items"
        named.append(len(check.co_batch.items))
        if items is not None:
            assert len(check.co_batch.items) == items, (day, check.co_batch.items)
        if cuts_at_issue_day(model) and check.co_batch.items:
            cut = check.co_batch.cut
            assert cut is not None and cut.forecasts and cut.ok, (day, cut)
    if getattr(model, "_predicts_together", False):
        assert any(named), f"no other item of a call was named on {list(days)}: the co-batch check saw nothing"
    return report


def check_loop_and_shards(
    make: Factory,
    frame: pd.DataFrame,
    tmp_path: Path,
    test: Tuple[str, str],
    horizon: int = 1,
    variants: Sequence[Dict[str, Any]] = ({}, {}, {}),
    information_set: Optional[InformationSet] = None,
) -> None:
    """Each model of a loop of `variants` (at least two, each the family's own parameters) writes the records it writes run
    alone, as a Workflow array task with `models_per_task=1` runs it into the shared save_dir, and `report()` of the merged
    shards equals the loop's: the same predictions, forecast counts and MAE. A forecast doesn't depend on which other models
    share its pipeline or process."""
    assert len(variants) >= 2, "a loop needs at least two models"
    names = [f"{NAME}{i}" for i in range(len(variants))]
    loop, shards = tmp_path / "loop", tmp_path / "shards"

    def every_model():
        pipeline = _pipeline([make(n, **v) for n, v in zip(names, variants)], information_set)
        return pipeline.add_evaluator(MAEEvaluator())

    _run(every_model(), frame, test, horizon, save_dir=loop)
    for name, params in zip(names, variants):  # one array task per model, into the shared save_dir
        _run(_pipeline([make(name, **params)], information_set), frame, test, horizon, save_dir=shards)
    for name in names:
        alone = lines(results(shards, name))
        assert alone, f"{name} wrote no record"
        assert lines(results(loop, name)) == alone, f"{name} in a loop differs from {name} alone"
    merged, looped = (every_model().report(path, *test, horizon=horizon) for path in (shards, loop))
    pd.testing.assert_frame_equal(_sorted(merged.predictions()), _sorted(looped.predictions()), check_exact=True)
    pd.testing.assert_frame_equal(merged.forecast_counts(), looped.forecast_counts(), check_exact=True)
    pd.testing.assert_frame_equal(merged.summary(), looped.summary(), check_exact=True)


def _sorted(predictions: pd.DataFrame) -> pd.DataFrame:
    """A predictions table in one order, whatever order the files were read in."""
    keys = [name for name in ("model", "run_date", "horizon", "hour") if name in predictions.columns]
    return predictions.sort_values(keys).reset_index(drop=True)


def check_resume(
    make: Factory,
    frame: pd.DataFrame,
    tmp_path: Path,
    test: Tuple[str, str],
    horizon: int = 1,
    information_set: Optional[InformationSet] = None,
) -> None:
    """A run cut short (40 % of its records deleted, every second and fourth of each five in sorted order, so inside refit
    blocks too) and resumed into the same save_dir equals a fresh run, record for record."""
    fresh, resumed = tmp_path / "fresh", tmp_path / "resumed"
    _run(_pipeline([make(NAME)], information_set), frame, test, horizon, save_dir=fresh)
    _run(_pipeline([make(NAME)], information_set), frame, test, horizon, save_dir=resumed)
    path = results(resumed)
    kept = [line for i, line in enumerate(lines(path)) if i % 5 not in (1, 3)]
    assert len(kept) < len(lines(path)), "nothing to delete"
    path.write_text("".join(f"{line}\n" for line in kept))
    _run(_pipeline([make(NAME)], information_set), frame, test, horizon, save_dir=resumed)
    assert lines(path) == lines(results(fresh))


def check_engine(
    make: Factory,
    frame: pd.DataFrame,
    test: Tuple[str, str],
    horizon: int = 1,
    information_set: Optional[InformationSet] = None,
) -> None:
    """The engine takes no model of the family (`_engine_capable()` False, which 6a makes explicit with `_engine_ok =
    False`), so `engine="on"` runs it on its own path and gives the forecasts of `engine="off"`, bitwise."""
    assert not make(NAME)._engine_capable(), "the engine would take this model"
    off = forecasts(_run(_pipeline([make(NAME)], information_set), frame, test, horizon))
    on = forecasts(_run(_pipeline([make(NAME)], information_set, engine="on"), frame, test, horizon))
    assert off and on == off


def _repeated_hour(frame: pd.DataFrame, day: datetime.date) -> int:
    local = frame.index.tz_localize(None)
    hours = pd.Series(local[local.normalize() == pd.Timestamp(day)].hour)
    repeated = hours[hours.duplicated()].unique()
    assert len(repeated) == 1, f"{day} is no autumn switch day of the frame"
    return int(repeated[0])


def check_dst_counts(
    make: Factory,
    frame: pd.DataFrame,
    spring: str,
    autumn: str,
    information_set: Optional[InformationSet] = None,
) -> None:
    """At horizon 1, issued the day before each switch day of an hourly frame: 23 forecasts planned and made for the spring
    target day, 24 for the autumn one (local periods 0-23), whose repeated hour has a NaN actual, as the calendar plans them
    (`IssueCalendar`)."""
    for target_day, expected in ((spring, 23), (autumn, 24)):
        issue = (datetime.date.fromisoformat(target_day) - datetime.timedelta(days=1)).isoformat()
        report = _run(_pipeline([make(NAME)], information_set), frame, (issue, issue), 1)
        counts = report.forecast_counts().iloc[0]
        assert (counts["planned"], counts["produced"]) == (expected, expected), (target_day, counts.to_dict())
        records = report.refs[NAME]._results
        assert sorted(r["hour"] for r in records) == sorted(set(r["hour"] for r in records))
        if expected == 24:
            hour = _repeated_hour(frame, datetime.date.fromisoformat(target_day))
            assert [np.isnan(r["actual"]) for r in records if r["hour"] == hour] == [True]


def check_quarter_hours(
    make: Factory,
    frame: pd.DataFrame,
    test: Tuple[str, str],
    horizon: int = 1,
    information_set: Optional[InformationSet] = None,
) -> None:
    """On a 15-minute frame the model runs with `freq="15min"`: every planned forecast is made, 96 periods on a day without a
    switch. For a family that supports quarter-hours."""
    report = _run(_pipeline([make(NAME)], information_set), frame, test, horizon, freq="15min")
    counts = report.forecast_counts().iloc[0]
    assert counts["missing"] == 0 and counts["produced"] > 0, counts.to_dict()
    periods = {r["hour"] for r in report.refs[NAME]._results}
    assert periods == set(range(96)), sorted(periods)


def check_provenance(
    make: Factory,
    frame: pd.DataFrame,
    tmp_path: Path,
    test: Tuple[str, str],
    horizon: int = 1,
    information_set: Optional[InformationSet] = None,
) -> None:
    """A model with a card has a row in `report.provenance()` naming its family and licence, also in a report read back from
    the save_dir. A model with no card has nothing to show; before 6a (no `report.provenance`) the check skips."""
    card = getattr(make(NAME), "_card", None)
    if card is None:
        return
    save_dir = tmp_path / "provenance"
    pipeline = _pipeline([make(NAME)], information_set)
    reports = [_run(pipeline, frame, test, horizon, save_dir=save_dir)]
    if not hasattr(reports[0], "provenance"):
        pytest.skip("report.provenance() arrives with step 6a")
    reports.append(pipeline.report(save_dir))
    for report in reports:
        table = report.provenance()
        row = table[table["model"] == NAME]
        assert len(row) == 1, table
        for field in ("family", "licence"):
            if getattr(card, field, None) is not None:
                assert row.iloc[0][field] == getattr(card, field), (field, row.iloc[0].to_dict())
