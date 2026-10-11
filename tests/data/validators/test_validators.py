import pandas as pd
from epftoolbox2.data.validators import (
    EdaValidator,
    ValidationResult,
)


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


class TestEdaValidatorPrinting:
    def test_prints_table_by_default(self, capsys):
        EdaValidator().validate(pd.DataFrame({"a": [1.0, 2.0]}))
        assert "EDA Statistics" in capsys.readouterr().out

    def test_verbose_false_is_silent_but_keeps_stats(self, capsys):
        result = EdaValidator(verbose=False).validate(pd.DataFrame({"a": [1.0, 2.0]}))
        assert capsys.readouterr().out == ""
        assert len(result.stats) == 1


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
