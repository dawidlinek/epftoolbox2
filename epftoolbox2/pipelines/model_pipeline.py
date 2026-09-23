from datetime import date
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import importlib
import pandas as pd
import yaml

from .._date_utils import resolve_date
from ..models.base import BaseModel
from ..evaluators.base import Evaluator
from ..exporters.base import Exporter
from ..results.report import EvaluationReport
from ..results.ref import ModelResultRef
from ..logging import get_logger

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
    },
    "exporters": {
        "CsvExporter": "epftoolbox2.exporters.csv",
        "ExcelExporter": "epftoolbox2.exporters.excel",
        "TerminalExporter": "epftoolbox2.exporters.terminal",
    },
}


class ModelPipeline:
    def __init__(self):
        self.models: List[BaseModel] = []
        self.evaluators: List[Evaluator] = []
        self.exporters: List[Exporter] = []

    def add_model(self, model: BaseModel) -> "ModelPipeline":
        self.models.append(model)
        return self

    def add_evaluator(self, evaluator: Evaluator) -> "ModelPipeline":
        self.evaluators.append(evaluator)
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
    ) -> EvaluationReport:
        if not self.models:
            raise ValueError("At least one model is required")
        self._check_model_names()
        # Check every model's inputs before running any, so a bad predictor in the last model fails in seconds.
        for model in self.models:
            model._check_inputs(data, target, horizon, freq)

        refs: dict[str, ModelResultRef] = {}
        for model in self.models:
            save_to = str(self.results_path(save_dir, model)) if save_dir else None
            ref = model.run(
                data=data,
                test_start=test_start,
                test_end=test_end,
                target=target,
                horizon=horizon,
                freq=freq,
                save_to=save_to,
                forecast_only=forecast_only,
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
        """
        if not self.models:
            raise ValueError("At least one model is required")
        self._check_model_names()
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
            )
        if missing and not allow_missing:
            raise FileNotFoundError(f"No results for model(s): {', '.join(missing)}. Run them first.")
        for name in missing:
            logger.warning(f"Report: no results for {name}; it is left out")

        report = EvaluationReport(refs, [] if forecast_only else self.evaluators, source_data=data)
        if not forecast_only:
            for exporter in self.exporters:
                exporter.export(report)
        return report

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

    def _serialize_component(self, component: Any) -> Dict[str, Any]:
        class_name = type(component).__name__
        params = {}
        if hasattr(component, "__dict__"):
            for key, value in component.__dict__.items():
                if key.startswith("_"):
                    continue
                if key == "predictors" and isinstance(value, list) and any(callable(p) for p in value):
                    raise ValueError(
                        f"Cannot serialize {class_name}: 'predictors' contains callable(s). "
                        "Replace callables with string column names before saving."
                    )
                if isinstance(value, Path):
                    params[key] = str(value)
                elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    params[key] = value
        return {"class": class_name, "params": params}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "models": [self._serialize_component(m) for m in self.models],
            "evaluators": [self._serialize_component(e) for e in self.evaluators],
            "exporters": [self._serialize_component(e) for e in self.exporters],
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info(f"Pipeline: Saved configuration to {path}")

    @classmethod
    def _load_component(cls, component_type: str, config: Dict[str, Any]) -> Any:
        class_name = config["class"]
        params = config.get("params", {})
        if class_name not in COMPONENT_REGISTRY[component_type]:
            raise ValueError(f"Unknown {component_type[:-1]}: {class_name}")
        module = importlib.import_module(COMPONENT_REGISTRY[component_type][class_name])
        return getattr(module, class_name)(**params)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelPipeline":
        path = Path(path)
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        pipeline = cls()
        for c in config.get("models", []):
            pipeline.add_model(cls._load_component("models", c))
        for c in config.get("evaluators", []):
            pipeline.add_evaluator(cls._load_component("evaluators", c))
        for c in config.get("exporters", []):
            pipeline.add_exporter(cls._load_component("exporters", c))
        logger.info(f"Pipeline: Loaded configuration from {path}")
        return pipeline
