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
from ..evaluators.base import Evaluator
from ..evaluators.dm import DMEvaluator
from ..exporters.base import Exporter
from ..results.availability import read_listing, warn_about
from ..results.manifest import manifest_path
from ..results.report import EvaluationReport
from ..results.ref import ModelResultRef
from ..verification import VerifyReport, sample_issue_days, verify_model
from ..logging import get_logger
from ._yaml import dump_yaml, to_yaml_value

logger = get_logger(__name__)


COMPONENT_REGISTRY = {
    "models": {
        "OLSModel": "epftoolbox2.models.ols",
        "LassoCVModel": "epftoolbox2.models.lasso",
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
    """Runs models over a test period, then evaluators and exporters.

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
            information set. "warn" and "raise" need an issue time. The level never changes a forecast, and changing it
            alone resumes a run.
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
    ) -> EvaluationReport:
        """Run every model, then (unless `forecast_only`) the evaluators and exporters.

        With `save_dir`, each model's results are resumable JSONL with a manifest recording the
        configuration that produced them (epftoolbox2.results.manifest). Resuming into results made with
        a different configuration raises `ManifestMismatchError`; `allow_manifest_mismatch` overwrites the
        manifest and resumes anyway, and `require_manifest` refuses results that predate the manifest
        feature instead of accepting them with a warning.
        """
        if not self.models:
            raise ValueError("At least one model is required")
        self._check_model_names()
        self._check_availability()
        if not forecast_only:
            self._check_benchmarks()
        if forecast_only:
            check_start, check_end = resolve_date(test_start or "today"), resolve_date(test_end or "today")
        else:
            check_start, check_end = resolve_date(test_start), resolve_date(test_end)
        level = self.availability
        # Check every model's inputs before running any, so a bad predictor in the last model fails in seconds. A
        # predictor function is planned here, once, and the run uses that plan.
        info = self.information_set
        plans: Dict[str, PredictorPlan] = {}
        listings: Dict[str, List[Dict]] = {}
        for model in self.models:
            plan = model._plan_predictors(data, target, horizon, freq, info, check_start, check_end, availability=level)
            model._check_inputs(
                data, target, horizon, freq, info, check_start, check_end, plan=plan, availability=level
            )
            listings[model.name] = model._availability_listing
            if plan is not None:
                plans[model.name] = plan
        warn_about(listings, stacklevel=3)  # once, for every model, and before the hours a run can take

        refs: dict[str, ModelResultRef] = {}
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
                **planned,
            )
            refs[model.name] = ref

        evaluators = [] if forecast_only else self.evaluators
        report = EvaluationReport(refs, evaluators, source_data=data)

        if not forecast_only:
            for exporter in self.exporters:
                exporter.export(report)

        return report

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
    ) -> EvaluationReport:
        """Build one report from the per-model JSONL files in `save_dir`, then run evaluators and exporters.

        This is the merge step after runs that each wrote one model's results (e.g. a SLURM array with
        `Workflow(model_index=...)`). The test period, horizon and freq must be those of the runs; they give
        each model's planned forecast count. Raises FileNotFoundError if a model has no results file, unless
        `allow_missing` is set. `data` is only needed by exporters that join source columns.

        What the runs listed about inputs known after the issue time comes from the manifests, whatever this pipeline's
        `availability` is: `report.availability()`, one warning and the terminal exporter show it.
        """
        if not self.models:
            raise ValueError("At least one model is required")
        self._check_model_names()
        if not forecast_only:
            self._check_benchmarks()
        if forecast_only:
            test_start, test_end = resolve_date(test_start or "today"), resolve_date(test_end or "today")
        elif test_start is None or test_end is None:
            raise ValueError("test_start and test_end are required unless forecast_only=True")
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
            )
        if missing and not allow_missing:
            raise FileNotFoundError(f"No results for model(s): {', '.join(missing)}. Run them first.")
        for name in missing:
            logger.warning(f"Report: no results for {name}; it is left out")

        shortfall = {name: ref for name, ref in refs.items() if ref.produced is not None and ref.produced < ref.count}
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

        report = EvaluationReport(refs, [] if forecast_only else self.evaluators, source_data=data)
        if not forecast_only:
            for exporter in self.exporters:
                exporter.export(report)
        return report

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
        names = [m.name for m in self.models]
        stems = [self.results_path("", m).name for m in self.models]
        if len(set(names)) != len(names) or len(set(stems)) != len(stems):
            raise ValueError(
                f"Model names must be unique (also after lower-casing and replacing spaces, since they name the "
                f"result files): {names}"
            )

    def _check_benchmarks(self) -> None:
        """A mistyped benchmark would only show as empty columns, after every model has run."""
        names = [m.name for m in self.models]
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
        for c in config.get("evaluators", []):
            pipeline.add_evaluator(cls._load_component("evaluators", c))
        for c in config.get("exporters", []):
            pipeline.add_exporter(cls._load_component("exporters", c))
        logger.info(f"Pipeline: Loaded configuration from {path}")
        return pipeline
