from abc import ABC, abstractmethod
import pandas as pd
from .result import ValidationResult


class Validator(ABC):
    """Base class for validators.

    `validate(df)` never raises on bad data: it returns a `ValidationResult` with errors for data that can't be used
    and warnings for data that is suspicious. Invalid configuration raises `ValueError` in the constructor.
    """

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        pass

    def _result(self) -> ValidationResult:
        return ValidationResult(validator=type(self).__name__)
