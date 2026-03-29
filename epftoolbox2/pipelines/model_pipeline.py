from typing import List, Optional
import pandas as pd

from ..models.base import BaseModel
from ..evaluators.base import Evaluator
from ..exporters.base import Exporter
from ..results.report import EvaluationReport
from ..results.ref import ModelResultRef


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
        test_start: str,
        test_end: str,
        target: str = "price",
        horizon: int = 7,
        save_dir: Optional[str] = None,
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
            )
            refs[model.name] = ref
        report = EvaluationReport(refs, self.evaluators)

        for exporter in self.exporters:
            exporter.export(report)

        return report
