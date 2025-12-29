from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..results.report import EvaluationReport


class Exporter(ABC):
    @abstractmethod
    def export(self, report: "EvaluationReport") -> None:
        """Export the evaluation report."""
        pass
