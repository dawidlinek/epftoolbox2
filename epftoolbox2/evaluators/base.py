from abc import ABC, abstractmethod
import pandas as pd


class Evaluator(ABC):
    name: str

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> float:
        """Compute metric from DataFrame with 'prediction' and 'actual' columns."""
        pass
