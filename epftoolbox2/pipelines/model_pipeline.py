from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import importlib
import pandas as pd
import yaml

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
    },
    "exporters": {
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
        save_dir: Optional[str] = None,
        forecast_only: bool = False,
    ) -> EvaluationReport:
        if not self.models:
            raise ValueError("At least one model is required")

        refs: dict[str, ModelResultRef] = {}
        for model in self.models:
            save_to = (
                f"{save_dir}/{model.name.lower().replace(' ', '_')}.jsonl"
                if save_dir
                else None
            )
            ref = model.run(
                data=data,
                test_start=test_start,
                test_end=test_end,
                target=target,
                horizon=horizon,
                save_to=save_to,
                forecast_only=forecast_only,
            )
            refs[model.name] = ref

        evaluators = [] if forecast_only else self.evaluators
        report = EvaluationReport(refs, evaluators)

        if not forecast_only:
            for exporter in self.exporters:
                exporter.export(report)

        return report

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
