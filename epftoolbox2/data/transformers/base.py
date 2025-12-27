from abc import ABC, abstractmethod
import pandas as pd


class Transformer(ABC):
    """Abstract base class for all data transformers.

    Transformers take a DataFrame and return a transformed DataFrame.
    """

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
