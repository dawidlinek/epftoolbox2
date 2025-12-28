import pandas as pd
from .base import Validator
from .result import ValidationResult


class ContinuityValidator(Validator):
    def __init__(self, freq: str = "1h"):
        self.freq = freq

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        result = ValidationResult()

        if df.empty:
            result.warnings.append("DataFrame is empty")
            return result

        if not isinstance(df.index, pd.DatetimeIndex):
            result.errors.append("Index is not a DatetimeIndex")
            result.is_valid = False
            return result

        expected_delta = pd.Timedelta(self.freq)
        actual_deltas = df.index.to_series().diff().dropna()
        gaps = actual_deltas[actual_deltas > expected_delta]

        if len(gaps) > 0:
            result.is_valid = False
            gap_info = []
            for gap_end, delta in gaps.items():
                gap_start = gap_end - delta
                gap_info.append({"start": gap_start, "end": gap_end, "duration": delta})
                result.errors.append(f"Gap detected (expected {self.freq} frequency): {gap_start} to {gap_end} ({delta})")
            result.info["gaps"] = gap_info
            result.info["gap_count"] = len(gaps)
            result.info["expected_freq"] = self.freq

        return result
