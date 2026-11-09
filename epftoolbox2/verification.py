"""The harness behind `ModelPipeline.verify`: does a pipeline's forecast use anything it couldn't have known?

For each model and a sample of issue days it runs two experiments on top of `epftoolbox2.information_set`:

- `leakage_probe`: perturb every value not known at the issue time, and what the model trains on from the issue day on
  (for `newest_target="published"`, what the target's rule calls unknown). Forecasts must not move. If they do, the model
  used information from after the issue time.
- `positive_control`: perturb the freshest values that are known. Forecasts must move. If they don't, the probe's
  silence is no evidence: its perturbation isn't reaching the model.

Two more, on by default for a model that predicts several items in one call (`_predicts_together = True`, or naming them
with `_co_batch_items`) or has a card (`_card`, a family with an extra or weights), and on request for any other:

- the rerun check: the unperturbed issue day is run twice, each run from other seeds of the process's global generators.
  Forecasts that differ at all (atol 0) make the model `inconclusive: nondeterministic`. A move under the probe or the
  co-batch check on such a day is a `leak` only when it is more than `NOISE_MARGIN` times that day's largest rerun difference;
  a smaller one can't be told from noise and is `inconclusive`.
- `co_batch_probe`: the leakage probe over runs widened to the other items of the issue day's call, as the model names them
  (`_co_batch_items(run_day)`), so that the call holds them: the issue day's forecasts must not depend on anything unknown at
  its issue time, whatever else is predicted with them. The plain probe misses a call made of the run's own days, which holds
  the issue day alone in its runs. For a model that predicts its refit blocks in one call (`_predict_block`), whose call holds
  the later days the frame has, the issue day's forecasts on the frame cut after its date (what a run on that day has) must
  also be those of the whole frame, to the bit (`issue_day_cut`), within the rerun's noise margin as any other move. For a model
  that reads history (`models._history`), whose series read the target day's known-ahead values on later rows, the frame is cut
  after what each of its forecasts reads instead (`target_day_cut`), whatever else it declares (`frame_cut`).

A model's status is `leak`, `inconclusive` (nothing moved, but that can't be vouched for) or `ok`, and only `ok` passes.

One blind spot no perturbation can reach: a predictor function with a `train_on="matching"` list that reads, on a day it trains
on, a column whose archive holds revised values (a rule of class `latest_revision` or `last_write_stamp`). The frame holds the
value as revised later, not what was out on that day, so the set of days a forecast trains on can be one no live run would have
chosen, and a probe sees a known value either way. Such a model is `inconclusive`, naming the column (decision 19 of the step 3
spec). A class not declared doesn't count, and nor does a read on an issue day, which is the forecast's own.
"""

import contextlib
import copy
import random
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._date_utils import resolve_date
from .information_set import (
    _STEP_MINUTES,
    ROLE_TARGET,
    Availability,
    CoBatchResult,
    ControlResult,
    InformationSet,
    ProbeResult,
    _issue_day_forecasts,
    co_batch_items,
    frame_cut,
    is_revised,
    leakage_probe,
    positive_control,
    vintage_of,
)
from .models._calendar import every_horizon, horizon_arguments, resolve_horizons, resolve_issue_days
from .models._cards import preflight
from .models._engine import ENGINE_OFF, EngineAlone, check_engine
from .models._contract import wall_clock
from .results.availability import LATE

LEAK = "leak"
INCONCLUSIVE = "inconclusive"
OK = "ok"

_EXAMPLES = 10
# A move under the probe (or the co-batch probe) on a day whose two unperturbed runs differ is a leak only when it is more than
# this many times that day's largest rerun difference (and above the probe's atol): the rerun measures the run-to-run noise
# over the same forecasts, and two orders of magnitude leave room for a heavier tail than its largest difference shows.
NOISE_MARGIN = 100.0
_PROBE_ATOL = 1e-9  # leakage_probe's default, which verify uses


@dataclass
class RerunResult:
    """Two runs of one issue day on the same, unperturbed frame, compared bit for bit."""

    run_day: str
    forecasts: int
    differ: List[Tuple[int, int]]  # (period, horizon) of the forecasts that are not the same in both runs
    max_abs_difference: float  # the largest |difference| between two finite forecasts
    missing: List[Tuple[int, int]] = field(default_factory=list)  # forecasts present in only one of the runs
    # False when a forecast is missing from one run, or finite in one and not in the other: no size bounds the noise then.
    bounded: bool = True

    @property
    def ok(self) -> bool:
        return not self.differ and not self.missing

    @property
    def noise(self) -> float:
        """The run-to-run noise the rerun saw: 0 when both runs are the same, inf when it can't be bounded."""
        if self.ok:
            return 0.0
        return self.max_abs_difference if self.bounded else float("inf")


@contextlib.contextmanager
def seeded_global_generators(seed: int) -> Iterator[None]:
    """Seed the process's global generators, numpy's legacy one (`np.random.*`, scikit-learn's `random_state=None`), `random`
    and torch's default ones if torch is already imported (it is never imported here), and put their states back afterwards.

    A pool forks from this process, so without it every run of a model drawing from them would start from the same state and
    draw the same numbers: two runs alike, though a production run's draws depend on the session and on which worker a job
    lands on."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    torch = sys.modules.get("torch")
    torch_state = cuda_state = None
    if torch is not None:
        torch_state = torch.random.get_rng_state()
        if torch.cuda.is_initialized():
            cuda_state = torch.cuda.get_rng_state_all()
    np.random.seed(seed % 2**32)
    random.seed(seed)
    if torch is not None:
        torch.default_generator.manual_seed(seed)
        if cuda_state is not None:
            torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        if torch is not None:
            torch.random.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)


def rerun_check(
    model: Any,
    data: pd.DataFrame,
    run_day: str,
    information_set: InformationSet,
    target: str = "price",
    horizon: Optional[int] = None,
    freq: str = "1h",
    availability: str = "raise",
    horizons: Optional[Sequence[int]] = None,
    seed: int = 0,
) -> RerunResult:
    """Run `model` twice on `data` for `run_day`, as the probe runs it (to the end of a refit block), and compare the forecasts
    at atol 0: equal, or both missing. Each run starts from other seeds of the global generators (`seeded_global_generators`,
    2 * seed + 1 and 2 * seed + 2), so a model drawing from them differs too. `model` is a model instance or a zero-argument
    factory, a fresh copy for each run."""
    run_day = str(pd.Timestamp(run_day).date())
    common = (run_day, target, horizon_arguments(*resolve_horizons(horizon, horizons, 1)), freq, information_set)
    with seeded_global_generators(2 * seed + 1):
        first, *_ = _issue_day_forecasts(model, data, *common, availability)
    with seeded_global_generators(2 * seed + 2):
        second, *_ = _issue_day_forecasts(model, data, *common, availability)
    keys = sorted(first.keys() & second.keys())
    differ = [key for key in keys if not (first[key] == second[key] or (pd.isna(first[key]) and pd.isna(second[key])))]
    finite = [abs(first[key] - second[key]) for key in differ if np.isfinite(first[key]) and np.isfinite(second[key])]
    missing = sorted(first.keys() ^ second.keys())
    bounded = not missing and len(finite) == len(differ)
    return RerunResult(run_day, len(first), differ, max(finite, default=0.0), missing, bounded)


def checked_by_default(model: Any) -> bool:
    """Whether `verify` runs the rerun and co-batch checks on `model` unless told otherwise: it predicts several items in one
    call (`_predicts_together = True`, or it names them with `_co_batch_items`), or it has a card (`_card`, a family with an
    extra or weights)."""
    return (
        bool(getattr(model, "_predicts_together", False))
        or getattr(model, "_co_batch_items", None) is not None
        or getattr(model, "_card", None) is not None
    )


def _beyond_noise(change: float, noise: float) -> bool:
    """Whether a move of `change` is no run-to-run noise of size `noise` (the day's largest rerun difference)."""
    return noise == 0.0 or (change > NOISE_MARGIN * noise and change > _PROBE_ATOL)


@dataclass
class IssueDayCheck:
    """One model's probe and control on one issue day."""

    model: str
    run_day: str
    horizon: int
    probe: ProbeResult
    control: Optional[ControlResult] = None  # None with positive_control=False
    not_from_frame: List[str] = field(default_factory=list)  # inputs a `rebuild` doesn't build from its argument
    not_sampled: List[str] = field(default_factory=list)  # inputs of a predictor function on no sampled issue day
    late_in_period: List[str] = field(
        default_factory=list
    )  # what a predictor function uses late in the whole test period
    horizons: Optional[Tuple[int, ...]] = None  # the horizons verified; None is every one from 1 to `horizon`
    # Columns of a revised class (with it) a predictor function reads on a training day of a `train_on="matching"` list.
    revised_reads: Dict[str, str] = field(default_factory=dict)
    rerun: Optional[RerunResult] = None  # None when the rerun check didn't run
    # None when the co-batch check didn't run. Its `items` are None for a model that predicts several items in one call
    # (`_predicts_together`) but names none (`_co_batch_items`), and empty for one that predicts each issue day on its own.
    co_batch: Optional[CoBatchResult] = None
    device: Optional[str] = None  # what `verify(device=)` set the copies' `device` to; None when it set nothing

    @property
    def planned_horizons(self) -> Tuple[int, ...]:
        """The horizons verified: `horizons`, or every one from 1 to `horizon`."""
        return tuple(range(1, self.horizon + 1)) if self.horizons is None else tuple(self.horizons)

    @property
    def unreached_horizons(self) -> List[int]:
        """Horizons with forecasts on this day, none of which moved under the control."""
        if self.control is None:
            return []
        moved = {horizon for _, horizon in self.control.moved}
        return sorted({horizon for _, horizon in self.control.unmoved} - moved)


@dataclass
class VerifyReport:
    """What `ModelPipeline.verify` found: one `IssueDayCheck` per model and issue day."""

    issue_days: List[str]
    checks: List[IssueDayCheck] = field(default_factory=list)

    @property
    def models(self) -> List[str]:
        return list(dict.fromkeys(check.model for check in self.checks))

    @property
    def devices(self) -> Dict[str, str]:
        """Model -> the device `verify(device=)` ran its copies on, for the models that have a `device`; the others ran as
        they are."""
        return {check.model: check.device for check in self.checks if check.device is not None}

    def status(self, model: str) -> str:
        """`leak` if a forecast moved on any issue day; else `inconclusive` if the check has a blind spot; else `ok`."""
        statuses = {status for status, _ in self._findings(model)}
        if LEAK in statuses:
            return LEAK
        return INCONCLUSIVE if statuses else OK

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(self.status(model) == OK for model in self.models)

    def table(self) -> pd.DataFrame:
        """One row per model, summed over its issue days.

        `forecasts` counts the forecasts compared, `changed` those that moved under the probe (or exist in only one of
        its runs), `max change` is the largest move under the probe, and `control moved` counts the forecasts that moved
        under the control (missing if it did not run). `rerun` counts the forecasts that differ between two unperturbed runs,
        and `co_batch` those that moved under the probe run over the other items of their call too, or, for a model that
        predicts its refit blocks in one call, on the frame cut after the issue day (each missing if its check did not run).
        """
        rows = []
        for model in self.models:
            checks = self._of(model)
            controls = [check.control for check in checks if check.control is not None]
            reruns = [check.rerun for check in checks if check.rerun is not None]
            # A model that names no items (`items` None) had nothing replaced: no count, and `_blind_spots` says so.
            batches = [
                check.co_batch for check in checks if check.co_batch is not None and check.co_batch.items is not None
            ]
            rows.append(
                {
                    "model": model,
                    "days": len(checks),
                    "forecasts": sum(check.probe.forecasts for check in checks),
                    "changed": sum(len(check.probe.changed) + len(check.probe.missing) for check in checks),
                    "max change": max(check.probe.max_abs_change for check in checks),
                    "control moved": sum(len(control.moved) for control in controls) if controls else pd.NA,
                    "rerun": sum(len(rerun.differ) + len(rerun.missing) for rerun in reruns) if reruns else pd.NA,
                    "co_batch": sum(len(batch.moved) for batch in batches) if batches else pd.NA,
                    "status": self.status(model),
                }
            )
        columns = [
            "model",
            "days",
            "forecasts",
            "changed",
            "max change",
            "control moved",
            "rerun",
            "co_batch",
            "status",
        ]
        table = pd.DataFrame(rows, columns=columns)
        for name in ("control moved", "rerun", "co_batch"):
            table[name] = table[name].astype("Int64")
        return table

    def problems(self) -> List[str]:
        """One line for each thing that keeps a model from `ok`, with the issue days and forecasts concerned."""
        return [line for model in self.models for _, line in self._findings(model)]

    def assert_ok(self) -> None:
        """Raise AssertionError unless every model is `ok`, naming what went wrong."""
        if not self.ok:
            raise AssertionError("\n".join(self.problems()) or "Nothing was verified.")

    def _of(self, model: str) -> List[IssueDayCheck]:
        checks = [check for check in self.checks if check.model == model]
        if not checks:
            raise KeyError(f"No model {model!r} was verified. Verified: {self.models}")
        return checks

    def _findings(self, model: str) -> List[Tuple[str, str]]:
        """(status, line) for each thing that keeps a model from `ok`."""
        checks = self._of(model)
        found = []
        noisy = [check.rerun for check in checks if check.rerun is not None and not check.rerun.ok]
        if noisy:
            differ = sorted({key for rerun in noisy for key in [*rerun.differ, *rerun.missing]})
            found.append(
                (
                    INCONCLUSIVE,
                    f"{model}: inconclusive: nondeterministic: two unperturbed runs differ on "
                    f"{', '.join(rerun.run_day for rerun in noisy)} in {sum(len(r.differ) for r in noisy)} of "
                    f"{sum(r.forecasts for r in noisy)} forecasts (max |difference| "
                    f"{max(r.noise for r in noisy):.6g}), e.g. (period, horizon) {differ[:_EXAMPLES]}, so on those days a "
                    f"forecast that moves under the probe or the co-batch check by no more than {NOISE_MARGIN:g} times "
                    "the day's largest difference can't be told from noise. Verify it on the CPU (verify(..., "
                    "device='cpu')) or with the model's deterministic settings.",
                )
            )
        for check in checks:
            noise = 0.0 if check.rerun is None else check.rerun.noise
            batch = check.co_batch
            wide = None if batch is None else batch.probe
            for probe, what in ((check.probe, ""), (wide, " run over the other items of their call too")):
                if probe is None or probe.ok:
                    continue
                line = (
                    f"{len(probe.changed)} of {probe.forecasts} forecasts depend on values not known at issue time"
                    + (f" (the probe{what}, {len(batch.items)} issue day(s) {batch.items[:_EXAMPLES]})" if what else "")
                    + f" (max |change| {probe.max_abs_change:.6g}), e.g. (period, horizon) {probe.changed[:_EXAMPLES]}; "
                    f"in only one run: {probe.missing[:_EXAMPLES]}"
                    + (f"; read through ctx and moved: {probe.moved_reads}" if probe.moved_reads else "")
                    + (f"; read through ctx though known later: {probe.late_reads}" if probe.late_reads else "")
                )
                # A cell a predictor function read differently, or read late, is no forecast moving: noise doesn't explain it.
                if probe.moved_reads or probe.late_reads or _beyond_noise(probe.max_abs_change, noise):
                    found.append((LEAK, f"{model}: leak on {check.run_day}: {line}"))
                else:
                    found.append(
                        (
                            INCONCLUSIVE,
                            f"{model}: inconclusive: on {check.run_day}, within {NOISE_MARGIN:g} times its rerun "
                            f"difference ({noise:.6g}), {line}",
                        )
                    )
            cut = None if batch is None else batch.cut
            if cut is not None and not cut.ok:
                line = (
                    f"{len(cut.changed)} of {cut.forecasts} forecasts differ when the frame ends "
                    f"{cut.ends or 'on ' + check.run_day}, as a run on that day has it (max |change| "
                    f"{cut.max_abs_change:.6g}), e.g. "
                    "(period, horizon) "
                    f"{cut.changed[:_EXAMPLES]}; in only one run: {cut.missing[:_EXAMPLES]}: they depend on which later days "
                    "their call holds, so a backtest and a run on the day differ"
                )
                if _beyond_noise(cut.max_abs_change, noise):
                    found.append((LEAK, f"{model}: leak on {check.run_day}: {line}"))
                else:
                    found.append(
                        (
                            INCONCLUSIVE,
                            f"{model}: inconclusive: on {check.run_day}, within {NOISE_MARGIN:g} times its rerun "
                            f"difference ({noise:.6g}), {line}",
                        )
                    )
        late = sorted({name for check in checks for name in check.late_in_period})
        if late:
            found.append(
                (
                    LEAK,
                    f"{model}: leak: over the whole test period its predictor function uses {late} though its rule puts "
                    "them after the issue time, on sampled issue days or not. No perturbation was needed to see it: the "
                    "plan says so. report.availability() of a run lists where.",
                )
            )
        found += [(INCONCLUSIVE, f"{model}: inconclusive: {line}") for line in _blind_spots(checks)]
        return found


def _blind_spots(checks: List[IssueDayCheck]) -> List[str]:
    """What the control and the rules can't vouch for among one model's checks."""
    lines = []
    controls = [check.control for check in checks if check.control is not None]
    if controls:
        produced = {key for control in controls for key in [*control.moved, *control.unmoved]}
        moved = {key for control in controls for key in control.moved}
        absent = [h for h in checks[0].planned_horizons if h not in {horizon for _, horizon in produced}]
        if absent:
            lines.append(f"no forecast of horizon(s) {absent} on any sampled issue day, so nothing checked them.")
        unreached = [check for check in checks if check.unreached_horizons]
        horizons = sorted({horizon for check in unreached for horizon in check.unreached_horizons})
        if unreached:
            days = ", ".join(check.run_day for check in unreached)
            lines.append(
                f"on {days} the positive control moved no forecast of horizon(s) {horizons}, so the probe isn't "
                "reaching them. A forecast needs an input known at issue time and delivered or published on the day "
                "before the issue day or later, read from the frame it is given."
            )
        never = sorted(key for key in produced - moved if key[1] not in horizons)
        if never:
            lines.append(
                f"{len(never)} forecast(s) never moved under the positive control on any sampled issue day, e.g. "
                f"(period, horizon) {never[:_EXAMPLES]}, so the probe isn't reaching them. Each needs an input "
                "known at issue time and delivered or published on the day before the issue day or later; a sparse "
                "model may need more days."
            )
    unnamed = [check.run_day for check in checks if check.co_batch is not None and check.co_batch.items is None]
    if unnamed:
        lines.append(
            "it predicts several items in one call (_predicts_together) but doesn't name them (_co_batch_items), so the "
            "co-batch check can't put them in its runs, and a forecast that reads what it can't know through another "
            "item of its call can't be seen."
        )
    blind = [
        check.run_day
        for check in checks
        if check.co_batch is not None
        and check.co_batch.control is not None
        and check.co_batch.control.forecasts
        and not check.co_batch.control.moved
    ]
    if blind:
        lines.append(
            f"on {', '.join(blind)} the positive control over the runs widened to the other items of the call moved no "
            "forecast, so the co-batch probe isn't reaching them."
        )
    unchecked = sorted({name for check in checks for name in check.probe.unchecked})
    if unchecked:
        lines.append(
            f"no availability rule for {unchecked}, so a leak through them can't be seen. Declare their rules in the "
            "information set."
        )
    not_sampled = sorted({name for check in checks for name in check.not_sampled})
    if not_sampled:
        lines.append(
            f"its predictor function uses {not_sampled} only on issue days outside the sample, so a leak through them "
            "can't be seen. Sample more days (`days=`), up to the whole period."
        )
    revised = {name: vintage for check in checks for name, vintage in check.revised_reads.items()}
    if revised:
        lines.append(
            f"its predictor function reads {dict(sorted(revised.items()))} on days a train_on='matching' list trains on, "
            "and their archive holds revised values: the frame has each value as revised later, not what was out on that "
            "day, so a forecast's set of days can be one no live run would have chosen, and no probe can see it. Snapshots "
            "of these columns would settle it."
        )
    not_from_frame = sorted({name for check in checks for name in check.not_from_frame})
    if not_from_frame:
        lines.append(
            f"{not_from_frame} stay the same whatever the source data holds, so `rebuild` doesn't build them from its "
            "argument and a leak in them can't be seen. If they really are known in advance, declare a static rule."
        )
    return lines


def sample_issue_days(
    index: pd.DatetimeIndex,
    test_start: Any,
    test_end: Any,
    days: int = 5,
    seed: int = 0,
    issue_days: Optional[Sequence[int]] = None,
) -> List[str]:
    """`days` local days of [test_start, test_end] that have rows, sampled with `seed` and sorted (all if fewer). With
    `issue_days` (weekday numbers, `resolve_issue_days`) only days of those weekdays, the issue days of such a run."""
    if days < 1:
        raise ValueError(f"days must be at least 1, got {days}")
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"Data must have a DatetimeIndex, got {type(index).__name__}")
    start, end = pd.Timestamp(resolve_date(test_start)), pd.Timestamp(resolve_date(test_end))
    dates = wall_clock(index).normalize().unique().sort_values()
    candidates = dates[(dates >= start) & (dates <= end)]
    if issue_days is not None:
        candidates = candidates[np.isin(candidates.day_of_week, list(issue_days))]
    if not len(candidates):
        raise ValueError(f"The data has no rows from {start.date()} to {end.date()}, so there is nothing to verify")
    chosen = np.random.default_rng(seed).choice(len(candidates), size=min(days, len(candidates)), replace=False)
    return [str(candidates[i].date()) for i in sorted(chosen.tolist())]


def verify_model(
    model: Any,
    data: pd.DataFrame,
    run_days: Sequence[str],
    information_set: InformationSet,
    target: str = "price",
    horizon: Optional[int] = None,
    freq: str = "1h",
    seed: int = 0,
    control: bool = True,
    rebuild: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    test_period: Optional[Tuple[Any, Any]] = None,
    availability: str = "raise",
    horizons: Optional[Sequence[int]] = None,
    issue_days: Optional[Sequence[str]] = None,
    engine: str = "off",
    device: Optional[str] = None,
    co_batch: Optional[bool] = None,
    rerun: Optional[bool] = None,
) -> List[IssueDayCheck]:
    """Probe (and control) one model on each of `run_days`. The model itself is never run, only copies of it.

    `test_period` is the period `run_days` were sampled from. A fixed list reads the same columns on every day, but a
    predictor function may use a column on a few days only, and an input that no sampled day reads is a blind spot.

    `availability` is the level a predictor function is planned at (see `leakage_probe`). The boundary is the same at every
    level: a late input is a leak whether the pipeline would list it or refuse it. The sample is no shelter for it either: a
    predictor function is planned on the whole of `test_period` anyway, which says what it uses late on any issue day.

    What a model trains on is checked against its own boundary. With `newest_target="d-1"` that is D-1, stricter than any rule,
    so the probe also perturbs the target from the issue day on (`check_target_boundary`). With `"published"` it is the target's
    rule, and the probe perturbs what the rule calls unknown, as for any column.

    `horizon=H` verifies every horizon from 1 to H (1 if neither is given), `horizons` the ones it lists (`run(horizons=...)`).
    `issue_days` are the weekdays of a run that issues on some only (`run(issue_days=["friday"])`): `test_period` is planned on
    those days alone, as the run plans it.

    `engine="on"` (`ModelPipeline(engine="on")`) runs each probe of a model the engine takes through it, as a group of one
    (`models._engine.EngineAlone`): those are its forecasts in any group, so the probe and its control cover that path.

    `rerun` runs the unperturbed issue day twice first (`rerun_check`), and `co_batch` runs the probe, and the control if it
    runs, once more over runs widened to the other items of the issue day's call (`co_batch_probe`), and for a model that predicts
    its refit blocks in one call compares the issue day's forecasts on the frame cut after its date (`issue_day_cut`), and for a
    model that reads history on the frame cut after what each forecast reads (`target_day_cut`); None runs
    each for a model
    `checked_by_default` says, and True or False for any. A model that names no items (no `_co_batch_items`) predicts each issue
    day on its own, unless it declares `_predicts_together`, which makes it `inconclusive`. `device` is set on every copy of a
    model that has an attribute named `device`, for every run here, and recorded in each check; a model without one (every core
    model) runs as it is.

    A model with a card is checked first as `run` checks it (`models._cards.preflight`): a missing extra, a runtime package outside
    the card's pins or missing weights raise `MissingExtraError` or `MissingWeightsError` before any probe runs, and nothing is
    downloaded.
    """
    horizon, horizons = resolve_horizons(horizon, horizons, 1)
    weekdays = resolve_issue_days(issue_days)
    check_engine(engine)
    # Its extra, runtime packages and weights before any probe (`ModelPipeline.verify` checked every model verified already).
    preflight([model])
    run_rerun = checked_by_default(model) if rerun is None else bool(rerun)
    run_co_batch = checked_by_default(model) if co_batch is None else bool(co_batch)

    on_device = device if device is not None and hasattr(model, "device") else None

    def make_model():
        copied = copy.deepcopy(model)
        if on_device is not None:
            copied.device = on_device
        return copied

    def probed():
        copied = make_model()
        return copied if engine == ENGINE_OFF else EngineAlone(copied)

    # A rebuild that edits its argument must not reach the caller's frame.
    features = None if rebuild is None else (lambda frame: rebuild(frame.copy()))
    built = data if features is None else features(data)
    checked = make_model()
    # A predictor function's inputs are the lists it returns on these days, and whatever it reads to choose them.
    plan = checked._plan_predictors(
        built,
        target,
        horizon,
        freq,
        information_set,
        issue_days=run_days,
        warn_unpublished=False,
        availability=availability,
        horizons=horizons,
    )
    revised_reads = _revised_training_reads(plan, information_set, freq)  # before the check lets go of what it read
    checked._check_inputs(built, target, horizon, freq, plan=plan, cut_rules=information_set, horizons=horizons)
    used = sorted({*checked._used_predictors, *plan.queried}) if plan is not None else list(checked._used_predictors)
    not_sampled: List[str] = []
    late_in_period: List[str] = []
    if plan is not None and test_period is not None:
        start, end = resolve_date(test_period[0]), resolve_date(test_period[1])
        whole = checked._plan_predictors(
            built,
            target,
            horizon,
            freq,
            information_set,
            start,
            end,
            warn_unpublished=False,
            availability=availability,
            horizons=horizons,
            weekdays=weekdays,
        )
        not_sampled = sorted(({name for names in whole.lists for name in names} | whole.queried) - set(used))
        revised_reads = {**revised_reads, **_revised_training_reads(whole, information_set, freq)}
        late_in_period = sorted(
            {row["predictor"] for row in whole.listing(target) if row["status"] == LATE and row["role"] != ROLE_TARGET}
        )

    not_from_frame = []
    columns = None
    if features is None:
        data = built = _booleans_as_numbers(data, [target, *used])
        columns = _probe_columns(data, information_set, target, used)
    else:
        not_from_frame = _inputs_not_from_frame(data, built, features, used, information_set, seed)

    # A forecast of D+1 may read today's published price, so a model can list its target as an input. What it reads as an
    # input is then up to the target's rule, and only what it trains on is up to the boundary (training targets end at
    # D-1): these runs train on a copy of the target under another name, leaving the input alone. Under
    # newest_target="published" the boundary is the target's rule, so the copy has it: it decides which of its values train.
    published = getattr(model, "newest_target", "d-1") == "published"
    label = target
    label_rules = information_set
    if target in used:
        label = f"{target}__as_target"
        label_rules = information_set.copy()
        label_rules.rules[label] = (
            information_set._resolve_for(target, _STEP_MINUTES[freq]).rule if published else Availability.static()
        )

    def with_label(frame):
        return frame if label == target else frame.assign(**{label: frame[target]})

    labelled = None if features is None else (lambda frame: with_label(features(frame)))
    runs = dict(**horizon_arguments(horizon, horizons), freq=freq, availability=availability)
    common = dict(**runs, seed=seed)
    # The rerun runs on the frame the model runs on, built, with the label in place of the target where the target is also an
    # input, as the boundary probe and the control do.
    unperturbed = with_label(built)

    def probe_and_control(run_day, span=None, columns=columns):
        """The probe (with its boundary part) and, if asked, the control on `run_day`; with `span`, over runs widened to it."""
        if features is None and label == target:
            probe = leakage_probe(
                probed,
                data,
                run_day,
                information_set,
                target,
                columns=columns,
                check_target_boundary=not published,
                span=span,
                **common,
            )
        else:
            probe = leakage_probe(
                probed, data, run_day, information_set, target, columns=columns, rebuild=features, span=span, **common
            )
            # On the built frame, not the source: perturbing the raw target from D on would also move what `rebuild`
            # legally builds from D's published price, e.g. `price_day_max` for a forecast of D+1.
            boundary = leakage_probe(
                probed,
                with_label(built),
                run_day,
                label_rules,
                label,
                columns=[label],
                check_target_boundary=not published,
                span=span,
                **common,
            )
            probe = _merge(probe, boundary)
        result = None
        if control:
            frame = with_label(data) if features is None else data
            result = positive_control(
                probed, frame, run_day, label_rules, label, columns=columns, rebuild=labelled, span=span, **common
            )
        return probe, result

    checks = []
    for run_day in run_days:
        reran = rerun_check(probed, unperturbed, run_day, label_rules, label, seed=seed, **runs) if run_rerun else None
        probe, result = probe_and_control(run_day)
        batch = None
        if run_co_batch:
            # The other items of the call, on the days the frame the model runs on has; the probe again, over all of them.
            others = co_batch_items(probed, run_day, built)
            if others is None:
                # No hook: each issue day on its own, unless the model says it predicts together, which `_blind_spots` flags.
                batch = CoBatchResult(run_day, None if getattr(model, "_predicts_together", False) else [])
            elif not others:
                batch = CoBatchResult(run_day, [])
            else:
                batch = CoBatchResult(run_day, others, *probe_and_control(run_day, (others[0], others[-1]), columns))
                # Its call holds the later days the frame has: D's forecasts on the frame a run on D has (cut after D, or for a
                # sequence model after what each forecast reads) must be the same.
                batch.cut = frame_cut(
                    probed,
                    unperturbed,
                    run_day,
                    label_rules,
                    label,
                    runs_in=lambda: seeded_global_generators(2 * seed + 1),
                    **runs,
                )
        checks.append(
            IssueDayCheck(
                model.name,
                probe.run_day,
                horizon,
                probe,
                result,
                not_from_frame,
                not_sampled,
                late_in_period,
                None if every_horizon(horizon, horizons) else horizons,
                revised_reads,
                reran,
                batch,
                on_device,
            )
        )
    return checks


def _revised_training_reads(plan: Any, information_set: InformationSet, freq: str) -> Dict[str, str]:
    """Column -> class, for each column of a revised class (`information_set.REVISED_VINTAGES`) a plan's function read through
    `ctx` on a training day of a `train_on="matching"` list (`PredictorPlan._training_cells`): with `value`, or with `published`
    where the rule calls it known, so that only whether the value is there decides. The noise of a probe needn't flip a list, so
    a read counts, whatever the function did with it. A plan with no matching list reads no training day."""
    cells = None if plan is None else plan._training_cells
    if cells is None:
        return {}
    step = _STEP_MINUTES[freq]
    found = {}
    for (name, _), (read, _) in cells.read_cells().items():
        rule = information_set._resolve_for(name, step).rule
        if read.any() and is_revised(rule):
            found[name] = vintage_of(rule)
    return dict(sorted(found.items()))


def _booleans_as_numbers(data: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """`perturb_unknown` skips boolean columns, but a model reads them as numbers, so a flag can carry a leak too."""
    flags = {name: "float64" for name in columns if pd.api.types.is_bool_dtype(data[name])}
    return data.astype(flags) if flags else data


def _probe_columns(data: pd.DataFrame, information_set: InformationSet, target: str, used: Sequence[str]) -> List[str]:
    """Every column of a built frame with a rule, in case a model reads more than it declares, and those it does
    declare: a column the model doesn't use needs no rule, like in `InformationSet.check`."""
    needed = {target, *used}
    return [
        name
        for name in data.columns
        if name in needed or (isinstance(name, str) and information_set.resolve(name).rule is not None)
    ]


def _inputs_not_from_frame(
    data: pd.DataFrame,
    built: pd.DataFrame,
    features: Callable[[pd.DataFrame], pd.DataFrame],
    used: Sequence[str],
    information_set: InformationSet,
    seed: int,
) -> List[str]:
    """The inputs that stay the same when every raw value changes, other than those with a static rule.

    The control moving says a `rebuild` reads its argument for some input, not for every one, and an input it takes from
    elsewhere is out of the probe's reach, whatever it holds.
    """
    rng = np.random.default_rng(seed)
    noise = data.copy()
    for name in noise.columns:
        if pd.api.types.is_numeric_dtype(noise[name]) and not pd.api.types.is_bool_dtype(noise[name]):
            noise[name] = 1e6 + rng.normal(0.0, 1e4, len(noise))
    rebuilt = features(noise)

    def known_in_advance(name):
        rule = information_set.resolve(name).rule
        return rule is not None and rule.kind == "static"

    return [name for name in used if not known_in_advance(name) and rebuilt[name].equals(built[name])]


def _merge(probe: ProbeResult, other: ProbeResult) -> ProbeResult:
    perturbed = {name: max(n, other.perturbed.get(name, 0)) for name, n in probe.perturbed.items()}
    return replace(
        probe,
        changed=sorted({*probe.changed, *other.changed}),
        max_abs_change=max(probe.max_abs_change, other.max_abs_change),
        perturbed={**other.perturbed, **perturbed},
        unchecked=sorted({*probe.unchecked, *other.unchecked}),
        missing=sorted({*probe.missing, *other.missing}),
        moved_reads={**other.moved_reads, **probe.moved_reads},
        late_reads={**other.late_reads, **probe.late_reads},
    )
