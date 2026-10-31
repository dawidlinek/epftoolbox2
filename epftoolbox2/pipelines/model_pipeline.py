from dataclasses import replace
from datetime import date
from typing import Callable, List, Optional, Sequence, Tuple, Union, Dict, Any
from pathlib import Path
import copy
import importlib
import warnings
import pandas as pd
import yaml

from .._date_utils import resolve_date
from ..information_set import InformationSet, InformationSetError, check_level
from ..models.base import BaseModel
from ..models.context import PredictorPlan, importable_path, load_function
from ..models.explain import explain_model
from ..ensembles._compute import compute
from ..ensembles.base import Ensemble
from ..evaluators.base import Evaluator
from ..evaluators.dm import DMEvaluator
from ..exporters.base import Exporter
from ..results.availability import read_listing, warn_about
from ..results.manifest import (
    FrameHashes,
    ManifestMismatchError,
    is_ensemble_manifest,
    manifest_path,
    read_matching_lists,
    read_windows,
)
from ..results.matching import rows_of_refs, warn_about_thin
from ..results.report import EvaluationReport
from ..results.ref import ModelResultRef, split_by_window
from ..results.windows import SKIP_TARGET_IN_TEST, TEST, RunWindows, to_period
from ..verification import VerifyReport, sample_issue_days, verify_model
from ..logging import get_logger
from ._yaml import dump_yaml, to_yaml_value

logger = get_logger(__name__)


COMPONENT_REGISTRY = {
    "models": {
        "OLSModel": "epftoolbox2.models.ols",
        "LassoCVModel": "epftoolbox2.models.lasso",
    },
    "ensembles": {
        "AverageEnsemble": "epftoolbox2.ensembles.average",
    },
    "evaluators": {
        "MAEEvaluator": "epftoolbox2.evaluators.mae",
        "RMSEEvaluator": "epftoolbox2.evaluators.rmse",
        "rMAEEvaluator": "epftoolbox2.evaluators.rmae",
        "BiasEvaluator": "epftoolbox2.evaluators.bias",
        "DMEvaluator": "epftoolbox2.evaluators.dm",
        "MCSEvaluator": "epftoolbox2.evaluators.mcs",
        "EconomicEvaluator": "epftoolbox2.evaluators.economic",
    },
    # What `EconomicEvaluator(strategy=...)` runs; saved inside the evaluator as `{class: ..., params: ...}`.
    "strategies": {
        "BatteryArbitrage": "epftoolbox2.evaluators.economic",
    },
    "exporters": {
        "CsvExporter": "epftoolbox2.exporters.csv",
        "ExcelExporter": "epftoolbox2.exporters.excel",
        "TerminalExporter": "epftoolbox2.exporters.terminal",
    },
}


class ModelPipeline:
    """Runs models over a test period, computes ensembles of them, then runs evaluators and exporters.

    Args:
        information_set: When each input becomes known and when forecasts are issued
            (see epftoolbox2.information_set). None means no issue time and no checks.
        issue_time: Shortcut for `information_set=InformationSet(issue_time, issue_timezone)` with the
            built-in availability rules, e.g. "11:45".
        issue_timezone: Zone of `issue_time`.
        availability: What a run does with a predictor (or the newest training target) that its rule puts after the
            issue time. "warn", the default with an issue time, goes on and lists it, and any column without a rule, in
            `report.availability()`, a warning, the terminal exporter and the manifest. "raise" refuses it before any
            fit, and `unknown` of the information set decides about columns without a rule. "none", the default
            without an issue time, checks and lists nothing; a predictor function still gets its answers from the
            information set. "warn" and "raise" need an issue time. The level doesn't change a forecast, unless a
            predictor function catches the error a late `ctx.value()` raises at "raise" and chooses another list, which it
            must not do. Changing the level alone resumes a run.
    """

    def __init__(
        self,
        information_set: Optional[InformationSet] = None,
        issue_time: Optional[str] = None,
        issue_timezone: str = "Europe/Brussels",
        availability: Optional[str] = None,
    ):
        if issue_time is not None:
            if information_set is not None:
                raise ValueError("Pass either issue_time/issue_timezone or information_set, not both")
            information_set = InformationSet(issue_time, issue_timezone)
        self.models: List[BaseModel] = []
        self.ensembles: List[Ensemble] = []
        self.evaluators: List[Evaluator] = []
        self.exporters: List[Exporter] = []
        self.information_set: Optional[InformationSet] = information_set
        self.availability = availability

    @property
    def availability(self) -> str:
        """The level in force: the one asked for, else "warn" with an information set and "none" without one."""
        if self._availability is not None:
            return self._availability
        return "warn" if self.information_set is not None else "none"

    @availability.setter
    def availability(self, level: Optional[str]) -> None:
        self._availability = None if level is None else check_level(level)
        self._check_availability()

    def _check_availability(self) -> None:
        """`information_set` can be set after the level, as `Workflow` and `epf model` do, so `run` asks again."""
        if self.availability != "none" and self.information_set is None:
            raise ValueError(
                f"availability={self.availability!r} checks predictors against the time a forecast is issued: pass "
                "issue_time or information_set too, or leave availability out."
            )

    def add_model(self, model: BaseModel) -> "ModelPipeline":
        self.models.append(model)
        return self

    def add_ensemble(self, ensemble: Ensemble) -> "ModelPipeline":
        """Add an ensemble of models or other ensembles of this pipeline, named by `ensemble.members`.

        The members are resolved when the pipeline runs, so the order of `add_model` and `add_ensemble` doesn't matter. See
        `epftoolbox2.ensembles.AverageEnsemble` and the docs page on ensembles.
        """
        if not isinstance(ensemble, Ensemble):
            raise TypeError(f"add_ensemble takes an ensemble such as AverageEnsemble, got {type(ensemble).__name__}")
        ensemble._check()
        self.ensembles.append(ensemble)
        return self

    def add_evaluator(self, *evaluators: Evaluator) -> "ModelPipeline":
        self.evaluators.extend(evaluators)
        return self

    def add_exporter(self, exporter: Exporter) -> "ModelPipeline":
        self.exporters.append(exporter)
        return self

    def run(
        self,
        data: pd.DataFrame,
        test_start: str = None,
        test_end: str = None,
        target: str = "price",
        horizon: int = 7,
        freq: str = "1h",
        save_dir: Optional[str] = None,
        forecast_only: bool = False,
        allow_manifest_mismatch: bool = False,
        require_manifest: bool = False,
        validation: Union[Tuple[Any, Any], str, None] = None,
        test: Optional[Tuple[Any, Any]] = None,
    ) -> EvaluationReport:
        """Run every model, compute the ensembles from what they stored, then (unless `forecast_only`) run the evaluators
        and exporters.

        With `save_dir`, each model's results are resumable JSONL with a manifest recording the
        configuration that produced them (epftoolbox2.results.manifest). Resuming into results made with
        a different configuration raises `ManifestMismatchError`; `allow_manifest_mismatch` overwrites the
        manifest and resumes anyway, and `require_manifest` refuses results that predate the manifest
        feature instead of accepting them with a warning.

        `test=(start, end)` is the test period, the same as `test_start` and `test_end` (passing both raises). With
        `validation`, each model forecasts the days of both windows in one run, into one results file, and nothing between
        them. Test and validation records are what a plain run over that window alone gives, apart from a `window` tag: each
        window reads its binary-column masks from its own first day. `validation=(start, end)` must end before the test
        starts. `validation="previous_year"` cuts the test period into 12-month segments, each served by the 12 months
        before it (`report.windows.segments`), and adds one year at the start, tagged "validation". `validation` alone
        forecasts only those days, and nothing of the test period is computed.

        The report is on the test window, and `report.validation` on the validation window, without forecasts for the first
        test issue day or later (see `EvaluationReport.validation`). Without a test window the validation report itself
        is returned. See the docs page on validation and test.

        Ensembles are computed after every model has run, from what the models stored. An ensemble that chooses (`top`) needs a
        validation window in this run, and a problem with one that shows before any fit raises before any fit. See the docs
        page on ensembles.
        """
        if not self.models:
            raise ValueError("At least one model is required")
        self._check_model_names()
        self._check_availability()
        if not forecast_only:
            self._check_benchmarks()
        windows, test_start, test_end = self._resolve_windows(validation, test, test_start, test_end, forecast_only)
        self._check_ensembles(save_dir)
        self._refuse_top_without_validation(self.ensembles, windows)
        if windows is not None:
            check_start, check_end = windows.first_day, windows.last_day
        elif forecast_only:
            check_start, check_end = resolve_date(test_start or "today"), resolve_date(test_end or "today")
        else:
            check_start, check_end = test_start, test_end
        level = self.availability
        # Check every model's inputs before running any, so a bad predictor in the last model fails in seconds. A
        # predictor function is planned here, once, and the run uses that plan.
        info = self.information_set
        plans: Dict[str, PredictorPlan] = {}
        listings: Dict[str, List[Dict]] = {}
        for model in self.models:
            plan = model._plan_predictors(
                data, target, horizon, freq, info, check_start, check_end, availability=level, windows=windows
            )
            model._check_inputs(
                data,
                target,
                horizon,
                freq,
                info,
                check_start,
                check_end,
                plan=plan,
                availability=level,
                windows=windows,
            )
            listings[model.name] = model._availability_listing
            if plan is not None:
                plans[model.name] = plan
        warn_about(listings, stacklevel=3)  # once, for every model, and before the hours a run can take

        refs: dict[str, ModelResultRef] = {}
        # Of this call alone: a frame changed before the next one must be hashed again.
        frame_hashes = FrameHashes(data) if save_dir else None
        for model in self.models:
            save_to = str(self.results_path(save_dir, model)) if save_dir else None
            planned = {"plan": plans[model.name]} if model.name in plans else {}
            ref = model.run(
                data=data,
                test_start=test_start,
                test_end=test_end,
                target=target,
                horizon=horizon,
                freq=freq,
                save_to=save_to,
                forecast_only=forecast_only,
                information_set=self.information_set,
                allow_manifest_mismatch=allow_manifest_mismatch,
                require_manifest=require_manifest,
                availability=level,
                warn_listing=False,
                windows=windows,
                frame_hashes=frame_hashes,
                **planned,
            )
            refs[model.name] = ref

        choices: Dict[str, List[Dict]] = {}
        if self.ensembles:
            paths = {model.name: self.results_path(save_dir, model) for model in self.models} if save_dir else {}
            on_file = self._stored_windows(paths)
            computed, choices = self._compute_ensembles(self._ordered_ensembles(), refs, on_file or windows, save_dir)
            refs.update(computed)

        evaluators = [] if forecast_only else self.evaluators
        report = EvaluationReport(refs, evaluators, source_data=data, choices=choices)

        if not forecast_only:
            for exporter in self.exporters:
                exporter.export(report)

        return report

    @staticmethod
    def _resolve_windows(validation, test, test_start, test_end, forecast_only: bool, required: bool = True):
        """(the windows of a run with validation or None, first test day, last test day), from `run`'s arguments.

        A `ValueError` if they are given twice, make no sense, or (with `required`) name no window at all.
        """
        if test is not None and (test_start is not None or test_end is not None):
            raise ValueError("Pass the test period once: test=(start, end) or test_start and test_end, not both")
        if forecast_only and validation is not None:
            raise ValueError("validation= is for backtests: forecast_only has no actuals to validate on")
        if test is None and forecast_only:
            return None, test_start, test_end  # today, unless they say otherwise
        if test is None and (test_start is not None or test_end is not None):
            if test_start is None or test_end is None:
                raise ValueError("test_start and test_end are required unless forecast_only=True")
            test = (test_start, test_end)
        if validation is not None:
            windows = RunWindows.resolve(validation, test)
            return windows, *(windows.test or (None, None))
        if test is None:
            if not required:
                return None, None, None
            raise ValueError(
                "No window to forecast: pass test=(start, end), validation=(start, end) or both (test_start and test_end "
                "are required unless forecast_only=True)"
            )
        test_start, test_end = to_period(test, TEST)
        return None, test_start, test_end

    def _stored_windows(self, paths: Dict[str, Path]) -> Optional[RunWindows]:
        """The windows every shard's manifest records, or None for plain runs; shards that disagree raise."""
        found = {name: read_windows(path) for name, path in paths.items()}
        distinct = {windows for windows in found.values()}
        if len(distinct) > 1:
            shown = "; ".join(f"{name}: {windows}" for name, windows in found.items())
            raise ValueError(
                f"These results were made with different windows, so their models can't be reported together ({shown}). "
                "Run every model with the same validation= and test=."
            )
        return next(iter(distinct), None)

    def report(
        self,
        save_dir: Union[str, Path],
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
        horizon: int = 7,
        freq: str = "1h",
        forecast_only: bool = False,
        data: Optional[pd.DataFrame] = None,
        allow_missing: bool = False,
        validation: Union[Tuple[Any, Any], str, None] = None,
        test: Optional[Tuple[Any, Any]] = None,
    ) -> EvaluationReport:
        """Build one report from the per-model JSONL files in `save_dir`, then run evaluators and exporters.

        This is the merge step after runs that each wrote one model's results (e.g. a SLURM array with
        `Workflow(model_index=...)`). The test period, horizon and freq must be those of the runs; they give
        each model's planned forecast count. Raises FileNotFoundError if a model has no results file, unless
        `allow_missing` is set. `data` is only needed by exporters that join source columns.

        Ensembles are computed here from the stored results of their members, every time, so they can't go stale, and written
        next to them. `allow_missing` leaves out an ensemble with a missing member, with a warning: it is never averaged over
        the members that happen to be there.

        What the runs listed about inputs known after the issue time comes from the manifests, whatever this pipeline's
        `availability` is: `report.availability()`, one warning and the terminal exporter show it.

        Forecasts of a `train_on="matching"` list on fewer days than the list has predictors are warned about again, from the
        stored records: an array task's warning went to that task's log.

        After runs with validation, `validation` and `test` say what `run` was told, and each must be what the manifests
        record. Without any window the report is on every window they record. Results whose manifests record different
        windows raise `ValueError`, as they would be compared on different forecasts. The report is what `run` returns: on
        the test window, with `report.validation`, or the validation report if there is no test window.
        """
        if not self.models:
            raise ValueError("At least one model is required")
        self._check_model_names()
        if not forecast_only:
            self._check_benchmarks()
        self._check_ensembles(save_dir)
        paths = {model.name: self.results_path(save_dir, model) for model in self.models}
        for name, path in paths.items():
            if is_ensemble_manifest(manifest_path(path)):
                raise ManifestMismatchError(
                    f"Model {name!r}: {path} is an ensemble's results, which go stale as soon as their members move unless "
                    "they are computed again. Add it to this pipeline with add_ensemble, or rename the model."
                )
        stored = self._stored_windows({name: path for name, path in paths.items() if path.exists()})
        windows, test_start, test_end = self._resolve_windows(
            validation, test, test_start, test_end, forecast_only, required=stored is None or forecast_only
        )
        if stored is not None and not forecast_only:
            windows = self._windows_to_report(windows, test_start, test_end, stored)
        elif windows is not None:
            raise ValueError(
                f"The results in {save_dir} were made without validation: their manifests record no windows, so there is "
                "no validation window to report. Run with validation=... first."
            )
        if windows is not None:
            test_start, test_end = windows.first_day, windows.last_day
            planned = 0
        else:
            if forecast_only:
                test_start, test_end = resolve_date(test_start or "today"), resolve_date(test_end or "today")
            test_start, test_end = resolve_date(test_start), resolve_date(test_end)
            n_days = (date.fromisoformat(test_end) - date.fromisoformat(test_start)).days + 1
            planned = n_days * horizon * (96 if freq == "15min" else 24)

        refs: Dict[str, ModelResultRef] = {}
        missing = []
        for model in self.models:
            path = self.results_path(save_dir, model)
            if not path.exists():
                missing.append(f"{model.name} ({path})")
                continue
            with open(path) as f:
                produced = sum(1 for line in f if line.strip())
            refs[model.name] = ModelResultRef(
                name=model.name,
                path=path,
                count=planned,
                test_start=test_start,
                test_end=test_end,
                horizon=horizon,
                freq=freq,
                produced=produced,
                availability_listing=read_listing(manifest_path(path)),
                windows=windows,
                matching_lists=read_matching_lists(manifest_path(path)),
            )
        if missing and not allow_missing:
            raise FileNotFoundError(f"No results for model(s): {', '.join(missing)}. Run them first.")
        for name in missing:
            logger.warning(f"Report: no results for {name}; it is left out")

        parts = dict(refs)
        if windows is not None:
            parts = {}
            for name, ref in refs.items():
                ref.by_window = split_by_window(
                    ref,
                    windows,
                    96 if freq == "15min" else 24,
                    listings={w: read_listing(manifest_path(ref.path), w) for w, _ in windows.periods},
                    target_before=stored.target_before,
                )
                parts.update({f"{name} in the {window} window": part for window, part in ref.by_window.items()})

        choices: Dict[str, List[Dict]] = {}
        reported = self._ensembles_with_their_members(set(refs))
        if reported:
            self._refuse_top_without_validation(reported, stored)
            # A forecast-only report covers some days of what is on file: the file on disk holds all of it.
            computed, choices = self._compute_ensembles(reported, refs, stored, None if forecast_only else save_dir)
            refs.update(computed)
            for name, ref in computed.items():
                if ref.by_window is None:
                    parts[name] = ref
                else:
                    parts.update({f"{name} in the {window} window": part for window, part in ref.by_window.items()})

        shortfall = {
            name: ref
            for name, ref in parts.items()
            if ref.produced is not None and ref.produced + ref.skipped.get(SKIP_TARGET_IN_TEST, 0) < ref.count
        }
        if shortfall:
            details = "; ".join(
                f"{name} ({ref.produced} of {ref.count}, {ref.count - ref.produced} missing)"
                for name, ref in shortfall.items()
            )
            warnings.warn(
                f"Report: model(s) with missing forecasts: {details}. Metrics are then computed on different "
                "forecast sets for these models; compare with care.",
                stacklevel=2,
            )
        warn_about({name: ref.availability_listing for name, ref in refs.items()}, stacklevel=3)

        report = EvaluationReport(refs, [] if forecast_only else self.evaluators, source_data=data, choices=choices)
        warn_about_thin(rows_of_refs(refs), stacklevel=2)  # on the whole file: a report holds one window
        if not forecast_only:
            for exporter in self.exporters:
                exporter.export(report)
        return report

    def _ordered_ensembles(self) -> List[Ensemble]:
        """The ensembles, each after the ensembles it is made of; `ValueError` for an unknown member or a cycle."""
        by_name = {ensemble.name: ensemble for ensemble in self.ensembles}
        models = {model.name for model in self.models}
        ordered: Dict[str, Ensemble] = {}

        def visit(ensemble: Ensemble, path: Tuple[str, ...]) -> None:
            if ensemble.name in ordered:
                return
            if ensemble.name in path:
                raise ValueError(f"These ensembles are members of each other: {' -> '.join([*path, ensemble.name])}")
            for member in ensemble.members:
                if member in by_name:
                    visit(by_name[member], (*path, ensemble.name))
                elif member not in models:
                    raise ValueError(
                        f"Ensemble {ensemble.name!r}: unknown member {member!r}. The pipeline has the models "
                        f"{sorted(models)} and the ensembles {sorted(by_name)}."
                    )
            ordered[ensemble.name] = ensemble

        for ensemble in self.ensembles:
            visit(ensemble, ())
        return list(ordered.values())

    def _check_before_any_fit(
        self,
        save_dir: Union[str, Path, None],
        validation: Union[Tuple[Any, Any], str, None],
        test: Optional[Tuple[Any, Any]],
        test_start: Optional[str],
        test_end: Optional[str],
        forecast_only: bool,
    ) -> None:
        """What `run` would refuse about the ensembles, for an array task, which runs one model and leaves them to the report."""
        self._check_model_names()
        windows, _, _ = self._resolve_windows(validation, test, test_start, test_end, forecast_only)
        self._check_ensembles(save_dir)
        self._refuse_top_without_validation(self.ensembles, windows)

    def _check_ensembles(self, save_dir: Union[str, Path, None]) -> None:
        """What can be known about the ensembles before any fit, whatever the windows."""
        names = {ensemble.name for ensemble in self.ensembles}
        for ensemble in self._ordered_ensembles():
            ensemble._check()
            nested = [member for member in ensemble.members if member in names]
            if ensemble.needs_validation and nested:
                raise ValueError(
                    f"Ensemble {ensemble.name!r} chooses on validation, so it can't have the ensembles {nested} as members: "
                    "their forecasts are already chosen or averaged on those days. A plain average of ensembles is fine."
                )
            if save_dir:
                path = self.results_path(save_dir, ensemble)
                if (path.exists() or manifest_path(path).exists()) and not is_ensemble_manifest(manifest_path(path)):
                    raise ManifestMismatchError(
                        f"Ensemble {ensemble.name!r} would overwrite {path}, which is not an ensemble's. Rename it, or use a "
                        "different save_dir."
                    )

    @staticmethod
    def _refuse_top_without_validation(ensembles: Sequence[Ensemble], windows: Optional[RunWindows]) -> None:
        if windows is not None and windows.validation is not None:
            return
        for ensemble in ensembles:
            if ensemble.needs_validation:
                raise ValueError(
                    f"Ensemble {ensemble.name!r} chooses its members on validation, and there is no validation window to "
                    "choose on: run with validation=(start, end) or validation='previous_year'."
                )

    def _ensembles_with_their_members(self, available: set) -> List[Ensemble]:
        """The ensembles whose members all have results, in their order; the others are left out with a warning."""
        reported = []
        known = set(available)
        for ensemble in self._ordered_ensembles():
            lacking = [member for member in ensemble.members if member not in known]
            if lacking:
                warnings.warn(
                    f"Report: ensemble {ensemble.name!r} is left out: no results for its member(s) {lacking}. It is never "
                    "averaged over the members that happen to be there.",
                    stacklevel=3,
                )
            else:
                reported.append(ensemble)
                known.add(ensemble.name)
        return reported

    def _compute_ensembles(
        self,
        ensembles: Sequence[Ensemble],
        refs: Dict[str, ModelResultRef],
        stored: Optional[RunWindows],
        save_dir: Union[str, Path, None],
    ) -> Tuple[Dict[str, ModelResultRef], Dict[str, List[Dict]]]:
        """`({name: ref} of what a report shows, {name: rows of report.choices()})`, in the order they were added."""
        like = next(iter(refs.values()))
        held = dict(refs)
        leaves: Dict[str, List[str]] = {}
        test_only_names: set = set()
        computed: Dict[str, ModelResultRef] = {}
        choices: Dict[str, List[Dict]] = {}
        for ensemble in ensembles:
            test_only = ensemble.needs_validation or not test_only_names.isdisjoint(ensemble.members)
            if test_only:
                test_only_names.add(ensemble.name)
            path = self.results_path(save_dir, ensemble) if save_dir else None
            leaves[ensemble.name] = list(
                dict.fromkeys(leaf for member in ensemble.members for leaf in leaves.get(member, [member]))
            )
            result = compute(ensemble, held, stored, like, path, test_only, leaves[ensemble.name])
            held[ensemble.name] = result.ref
            # A ref with no window to show, as a `top` ensemble has in a validation report, is not a row of its tables.
            if result.ref.by_window is None or result.ref.by_window:
                computed[ensemble.name] = result.ref
            if result.choices:
                choices[ensemble.name] = result.choices
        order = [ensemble.name for ensemble in self.ensembles]
        if choices and stored is not None:
            # What the rows of `choices()` count in, whichever part of what is on file this run or report names.
            for ref in [*refs.values(), *computed.values()]:
                for part in [ref, *(ref.by_window or {}).values()]:
                    if part.windows is not None:
                        part.windows = replace(part.windows, mode=stored.mode, segments=stored.segments)
        return {name: computed[name] for name in order if name in computed}, {
            name: choices[name] for name in order if name in choices
        }

    @staticmethod
    def _windows_to_report(
        asked: Optional[RunWindows], test_start: Optional[str], test_end: Optional[str], stored: RunWindows
    ) -> RunWindows:
        """What a report covers after runs with windows: what `report` was told, which must be what the manifests record,
        or all of it if it was told nothing."""
        if asked is None:
            if test_start is None:
                return stored
            asked = RunWindows.fixed(None, (test_start, test_end))
        different = [
            f"{name} {period} (the manifests record {getattr(stored, name)})"
            for name, period in asked.periods
            if period != getattr(stored, name)
        ]
        if different:
            raise ValueError(
                f"These windows are not what the results were made with: {'; '.join(different)}. Pass the windows of the "
                "run, or none to report what the manifests record."
            )
        return stored if asked.validation is not None and asked.test is not None else asked

    def verify(
        self,
        data: pd.DataFrame,
        test_start: str,
        test_end: str,
        target: str = "price",
        horizon: int = 1,
        freq: str = "1h",
        days: int = 5,
        seed: int = 0,
        models: Optional[Sequence[str]] = None,
        positive_control: bool = True,
        rebuild: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ) -> VerifyReport:
        """Check on a sample of issue days that no forecast depends on anything unknown when it is issued.

        For each model, `days` issue days are sampled with `seed` from [test_start, test_end] (the same days for every
        model, whichever ones and in whatever order). On each of them:

        - the leakage probe perturbs every value not known at the issue time, and what the model trains on from the
          issue day on. Forecasts must not move; if they do, the status is `leak`.
        - the positive control perturbs the freshest known values: those delivered or first known on the day before the
          issue day or later (see `epftoolbox2.information_set.perturb_recent`). Every forecast must move on some
          sampled day, and on every day some forecast of each horizon; if not, the probe isn't reaching the model and
          the status is `inconclusive`, which doesn't pass either. So is a model with an input that has no availability
          rule, which the probe can't perturb, and a predictor function with an input that no sampled day reads.

        A predictor function of `ctx` is planned on each frame the probe tests, so a list chosen by a value it can't know
        moves with it. A cell it read that holds another value in the perturbed frame is a `leak` too, whether or not the
        noise changed its list.

        `rebuild` treats `data` as source data, before feature engineering: the perturbation goes into the raw columns
        and `rebuild(frame) -> features` reproduces the transformers. Without it a transformer that writes a future
        value into a legally named column can't be seen, since the check only has the column's name to go by. An input
        with a non-static rule that `rebuild` doesn't build from its argument is `inconclusive` too.

        Only copies of the models run, in memory, so a later `run()` is unaffected. Each issue day costs four one-day
        runs per model, six with `rebuild` or a target among the predictors. A model that isn't deterministic shows up
        as a leak.

        `verify` probes the period you give it, not the windows of a run: to check a validation period as well as a test
        period, call it on each.

        The boundary is the same at every `availability` level: a late predictor is a `leak` also where `run` only lists
        it. With "warn" or "none" a predictor function that reads a late `ctx.value()` is a `leak` too; with "raise" that
        read raises `InformationSetError`. A predictor function is planned on the whole test period, so what it uses late
        on any issue day is a `leak`, sampled or not.

        Args:
            data: The frame the models run on, or the source data with `rebuild`.
            test_start: First issue day to sample from.
            test_end: Last issue day to sample from.
            target: Target column.
            horizon: Max horizon. Leave no horizon you run out: a leak in another one can't be seen (`run` defaults to 7).
            freq: "1h" or "15min"; must match the data.
            days: Issue days per model (all of the period if it has fewer).
            seed: Picks the issue days and the noise.
            models: Names of the models to verify. None verifies them all.
            positive_control: False skips the control, so `ok` only says that no forecast moved.
            rebuild: Builds the model's input from `data` (see above).

        Raises:
            InformationSetError: The pipeline has no information set, or a checked column has no rule and the
                information set's `unknown` is "raise".
        """
        if self.information_set is None:
            raise InformationSetError(
                "verify() needs to know when forecasts are issued: pass issue_time or information_set to ModelPipeline"
            )
        if not self.models:
            raise ValueError("At least one model is required")
        self._check_model_names()
        run_days = sample_issue_days(data.index, test_start, test_end, days, seed)
        report = VerifyReport(issue_days=run_days)
        for model in self._select_models(models):
            report.checks.extend(
                verify_model(
                    model,
                    data,
                    run_days,
                    self.information_set,
                    target=target,
                    horizon=horizon,
                    freq=freq,
                    seed=seed,
                    control=positive_control,
                    rebuild=rebuild,
                    test_period=(test_start, test_end),
                    availability=self.availability,
                )
            )
        return report

    def explain(
        self,
        data: pd.DataFrame,
        model: str,
        test: Optional[Tuple[Any, Any]] = None,
        horizon: int = 1,
        hour: int = 12,
        target: str = "price",
        freq: str = "1h",
        echo: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Show what a model's predictors mean for one forecast: an optional view, with no effect on a later run.

        The result has `predictors`, one row per predictor of the forecast of period `hour` at `horizon`: `predictor`,
        `target-relative` (when its value was delivered, relative to the target day T = D + horizon, so at horizon 2
        `price_d-1` at hour 12 is `T-3 12:00`), `rule`, `known` (relative to the issue day D, on the issue day it is
        latest), and `status` (`ok`, `late`, `late on 3 of 10 days`, `no rule`, `no issue time`, `not in data`). `grouped`
        is the same with runs of slots folded into one row (`price_h00 … price_h23_d-1 (24)`, `T-2 00:00 … 23:00`).

        For a predictor function of `ctx`, `test` is required and `lists` has one row per horizon up to `horizon` and list
        that horizon uses: how many forecasts and issue days use it, its first and last issue day, its periods, and what it
        `adds` to and `drops` from the most used list of that horizon. `predictors` then shows the list used most at
        (`horizon`, `hour`), judged on the days it is used. A late `ctx.value()` doesn't stop the view; see
        `report.availability()` for what a function read.

        With `train_on="matching"` lists `lists` also has `smallest days`, `median days` and `largest days`: how many days the
        forecasts of such a list would train on, as a run counts them, without fitting anything; a dash for any other list.

        Args:
            data: The frame the models run on.
            model: The model's name.
            test: (first, last) issue day. None takes every day of `data`, for fixed lists only.
            horizon: Days ahead of the issue day of the forecast shown.
            hour: Its period of the target day: 0-23, or 0-95 for 15-minute data.
            target: Target column.
            freq: "1h" or "15min"; must match the data.
            echo: Print the tables.
        """
        chosen = copy.deepcopy(self._select_models([model])[0])  # planning leaves state on a model; this is only a look
        if test is not None:
            test = (resolve_date(test[0]), resolve_date(test[1]))
        return explain_model(chosen, data, self.information_set, test, horizon, hour, target, freq, echo)

    def _select_models(self, names: Optional[Sequence[str]]) -> List[BaseModel]:
        if names is None:
            return self.models
        names = [names] if isinstance(names, str) else list(names)
        by_name = {model.name: model for model in self.models}
        unknown = [name for name in names if name not in by_name]
        if unknown:
            raise ValueError(f"Unknown model(s) {unknown}. The pipeline has: {list(by_name)}")
        return [by_name[name] for name in dict.fromkeys(names)]

    @staticmethod
    def results_path(save_dir: Union[str, Path], model: BaseModel) -> Path:
        """The JSONL file a model's results are written to inside `save_dir`."""
        return Path(save_dir) / f"{model.name.lower().replace(' ', '_')}.jsonl"

    def _check_model_names(self) -> None:
        named = [*self.models, *self.ensembles]
        names = [m.name for m in named]
        stems = [self.results_path("", m).name for m in named]
        if len(set(names)) != len(names) or len(set(stems)) != len(stems):
            what = "Model and ensemble names" if self.ensembles else "Model names"
            raise ValueError(
                f"{what} must be unique (also after lower-casing and replacing spaces, since they name the "
                f"result files): {names}"
            )

    def _check_benchmarks(self) -> None:
        """A mistyped benchmark would only show as empty columns, after every model has run."""
        names = [m.name for m in [*self.models, *self.ensembles]]
        for evaluator in self.evaluators:
            if isinstance(evaluator, DMEvaluator) and evaluator.benchmark not in names:
                raise ValueError(f"DMEvaluator: unknown benchmark {evaluator.benchmark!r}; models: {names}")

    @staticmethod
    def _check_predictors_can_be_saved(component: Any) -> None:
        """A model built with callable predictors keeps ``predictors == []`` and the function privately, so a plain
        dump would write an empty list: the reloaded pipeline would be a different study."""
        predictors = getattr(component, "predictors", None)
        if getattr(component, "_predictors_fn", None) is not None or callable(predictors):
            what = "its predictors are a function"
        elif isinstance(predictors, (list, tuple)) and any(callable(p) for p in predictors):
            what = "its predictors contain a function"
        else:
            return
        raise TypeError(
            f"Cannot save model {getattr(component, 'name', type(component).__name__)!r} "
            f"({type(component).__name__}): {what}, and callable predictors can't be saved to YAML yet. "
            "List the predictors explicitly instead: strings, with '{horizon}' where a column depends on the horizon. "
            "A function of `ctx` at the top level of a module can be saved, by its import path."
        )

    def _serialize_component(self, component: Any) -> Dict[str, Any]:
        class_name = type(component).__name__
        self._check_predictors_can_be_saved(component)
        label = f"{class_name}({component.name!r})" if isinstance(component, BaseModel) else class_name
        params = {
            key: self._serialize_component(value)
            if type(value).__name__ in COMPONENT_REGISTRY["strategies"]
            else to_yaml_value(value, f"{label}.{key}")
            for key, value in getattr(component, "__dict__", {}).items()
            if not key.startswith("_")
        }
        if getattr(component, "_ctx_fn", None) is not None:
            params["predictors"] = {"function": importable_path(component._ctx_fn, label)}
        return {"class": class_name, "params": params}

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "models": [self._serialize_component(m) for m in self.models],
            **({"ensembles": [self._serialize_component(e) for e in self.ensembles]} if self.ensembles else {}),
            "evaluators": [self._serialize_component(e) for e in self.evaluators],
            "exporters": [self._serialize_component(e) for e in self.exporters],
        }
        if self.information_set is not None:
            out["information_set"] = self.information_set.to_dict()
        if self._availability is not None:  # what was asked for; left out, a load takes the default again
            out["availability"] = self._availability
        return out

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        # The text is built before the file is opened: a pipeline that can't be saved must not truncate it.
        path.write_text(dump_yaml(self.to_dict()))
        logger.info(f"Pipeline: Saved configuration to {path}")

    @classmethod
    def _load_component(cls, component_type: str, config: Dict[str, Any]) -> Any:
        class_name = config["class"]
        params = dict(config.get("params") or {})
        if class_name not in COMPONENT_REGISTRY[component_type]:
            kind = "strategy" if component_type == "strategies" else component_type[:-1]
            raise ValueError(f"Unknown {kind}: {class_name}")
        if component_type == "evaluators" and isinstance(params.get("strategy"), dict):
            params["strategy"] = cls._load_component("strategies", params["strategy"])
        if component_type == "models" and isinstance(params.get("predictors"), dict):
            params["predictors"] = cls._load_predictor_function(class_name, params["predictors"])
        module = importlib.import_module(COMPONENT_REGISTRY[component_type][class_name])
        return getattr(module, class_name)(**params)

    @staticmethod
    def _load_predictor_function(class_name: str, saved: Dict[str, Any]) -> Callable:
        if set(saved) != {"function"}:
            raise ValueError(f"{class_name}: predictors must be a list or {{function: 'module:name'}}, got {saved}")
        return load_function(saved["function"])

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelPipeline":
        path = Path(path)
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        info = config.get("information_set")
        pipeline = cls(
            information_set=InformationSet.from_dict(info) if info else None, availability=config.get("availability")
        )
        for c in config.get("models", []):
            pipeline.add_model(cls._load_component("models", c))
        for c in config.get("ensembles", []):
            pipeline.add_ensemble(cls._load_component("ensembles", c))
        for c in config.get("evaluators", []):
            pipeline.add_evaluator(cls._load_component("evaluators", c))
        for c in config.get("exporters", []):
            pipeline.add_exporter(cls._load_component("exporters", c))
        logger.info(f"Pipeline: Loaded configuration from {path}")
        return pipeline
