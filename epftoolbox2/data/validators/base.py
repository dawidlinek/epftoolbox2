from abc import ABC, abstractmethod
import pandas as pd
from .result import ValidationResult


class Validator(ABC):
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        pass
