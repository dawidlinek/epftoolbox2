import pandas as pd
from epftoolbox2.data.validators import (
    ContinuityValidator,
    NullCheckValidator,
    EdaValidator,
    ValidationResult,
)


class TestContinuityValidator:
    def test_valid_continuous_data(self):
        index = pd.date_range("2024-01-01", periods=24, freq="1h", tz="UTC")
        df = pd.DataFrame({"value": range(24)}, index=index)

        validator = ContinuityValidator(expected_freq="1h")
        result = validator.validate(df)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_gap_detected(self):
        index = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 05:00"]).tz_localize("UTC")
        df = pd.DataFrame({"value": [1, 2, 3]}, index=index)

        validator = ContinuityValidator(expected_freq="1h")
        result = validator.validate(df)

        assert not result.is_valid
        assert len(result.errors) == 1
        assert "gap_count" in result.info
        assert result.info["gap_count"] == 1

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        validator = ContinuityValidator()
        result = validator.validate(df)

        assert result.is_valid
        assert len(result.warnings) == 1

    def test_non_datetime_index(self):
        df = pd.DataFrame({"value": [1, 2, 3]})
        validator = ContinuityValidator()
        result = validator.validate(df)

        assert not result.is_valid
        assert "DatetimeIndex" in result.errors[0]


class TestNullCheckValidator:
    def test_valid_no_nulls(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        validator = NullCheckValidator(required_columns=["a", "b"])
        result = validator.validate(df)

        assert result.is_valid

    def test_missing_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3]})

        validator = NullCheckValidator(required_columns=["a", "b", "c"])
        result = validator.validate(df)

        assert not result.is_valid
        assert "missing_columns" in result.info
        assert set(result.info["missing_columns"]) == {"b", "c"}

    def test_null_values_detected(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})

        validator = NullCheckValidator()
        result = validator.validate(df)

        assert not result.is_valid
        assert "null_counts" in result.info
        assert result.info["null_counts"]["a"] == 1
        assert result.info["null_counts"]["b"] == 1

    def test_allow_nulls(self):
        df = pd.DataFrame({"a": [1, None, 3]})

        validator = NullCheckValidator(allow_nulls=True)
        result = validator.validate(df)

        assert result.is_valid


class TestEdaValidator:
    def test_basic_stats(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

        validator = EdaValidator()
        result = validator.validate(df)

        assert result.is_valid
        assert not result.stats.empty
        assert len(result.stats) == 2
        assert "min" in result.stats.columns
        assert "max" in result.stats.columns
        assert "mean" in result.stats.columns
        assert "std" in result.stats.columns

    def test_stats_values(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        validator = EdaValidator(columns=["a"])
        result = validator.validate(df)

        stats = result.stats
        assert stats.loc[0, "min"] == 1
        assert stats.loc[0, "max"] == 5
        assert stats.loc[0, "mean"] == 3
        assert stats.loc[0, "count"] == 5

    def test_null_percentage(self):
        df = pd.DataFrame({"a": [1, 2, None, None, 5]})

        validator = EdaValidator()
        result = validator.validate(df)

        stats = result.stats
        assert stats.loc[0, "null_count"] == 2
        assert stats.loc[0, "null_pct"] == 40.0

    def test_specific_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

        validator = EdaValidator(columns=["a", "b"])
        result = validator.validate(df)

        assert len(result.stats) == 2
        assert set(result.stats["column"]) == {"a", "b"}

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        validator = EdaValidator()
        result = validator.validate(df)

        assert result.is_valid
        assert len(result.warnings) == 1


class TestValidationResult:
    def test_bool_valid(self):
        result = ValidationResult(is_valid=True)
        assert bool(result) is True

    def test_bool_invalid(self):
        result = ValidationResult(is_valid=False)
        assert bool(result) is False

    def test_str_representation(self):
        result = ValidationResult(is_valid=False, errors=["Error 1"])
        str_repr = str(result)
        assert "INVALID" in str_repr
        assert "Error 1" in str_repr
