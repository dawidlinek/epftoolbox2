import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.validators import NullCheckValidator
from tests.synthetic import make_market_data


@pytest.fixture
def hourly():
    index = pd.date_range("2024-01-01", periods=24 * 10, freq="1h", tz="UTC")
    return pd.DataFrame({"price": 50.0 + np.arange(len(index)), "load_actual": 1000.0}, index=index)


def test_columns_limit_the_check_to_those_columns():
    # Regression: lag columns are missing at the start of the frame by construction and used to fail the check.
    df = make_market_data(days=30)
    assert df["price_d-1"].isna().sum() > 0

    result = NullCheckValidator(columns=["price", "load"]).validate(df)

    assert result.is_valid, result.errors
    assert result.errors == []
    assert list(result.stats["column"]) == ["price", "load"]


def test_null_in_listed_column_fails(hourly):
    hourly.loc[hourly.index[48:120], "price"] = np.nan
    hourly.loc[hourly.index[:5], "load_actual"] = np.nan  # not listed

    result = NullCheckValidator(columns=["price"]).validate(hourly)

    assert not result.is_valid
    assert len(result.errors) == 1
    assert "'price' has 72 null values" in result.errors[0]
    assert result.info["null_counts"] == {"price": 72}
    assert "load_actual" not in result.info["null_counts"]


def test_reports_longest_missing_run(hourly):
    hourly.loc[hourly.index[10:12], "price"] = np.nan
    hourly.loc[hourly.index[48:120], "price"] = np.nan

    result = NullCheckValidator(columns=["price"]).validate(hourly)

    row = result.stats.set_index("column").loc["price"]
    assert row["null_count"] == 74
    assert row["longest_null_run"] == 72
    assert row["longest_null_run_start"] == hourly.index[48]
    assert row["longest_null_run_end"] == hourly.index[119]
    assert result.info["longest_null_run"] == {"price": 72}
    assert "longest missing run 72 rows" in result.errors[0]


def test_max_null_frac_is_per_column(hourly):
    hourly.loc[hourly.index[:12], "price"] = np.nan  # 5 %
    hourly.loc[hourly.index[:48], "load_actual"] = np.nan  # 20 %

    result = NullCheckValidator(max_null_frac=0.1).validate(hourly)

    assert not result.is_valid
    assert len(result.errors) == 1 and "'load_actual'" in result.errors[0]
    assert len(result.warnings) == 1 and "'price'" in result.warnings[0]
    assert result.info["null_fraction"] == {"price": pytest.approx(0.05), "load_actual": pytest.approx(0.2)}


def test_allow_nulls_turns_errors_into_warnings(hourly):
    hourly.loc[hourly.index[:3], "price"] = np.nan

    result = NullCheckValidator(columns=["price"], allow_nulls=True).validate(hourly)

    assert result.is_valid
    assert len(result.warnings) == 1


def test_no_columns_checks_every_column(hourly):
    hourly.loc[hourly.index[0], "load_actual"] = np.nan

    result = NullCheckValidator().validate(hourly)

    assert not result.is_valid
    assert set(result.stats["column"]) == {"price", "load_actual"}


def test_single_column_name_is_accepted(hourly):
    assert NullCheckValidator(columns="price").columns == ["price"]


@pytest.mark.parametrize("frac", [-0.1, 1.5])
def test_invalid_max_null_frac_raises(frac):
    with pytest.raises(ValueError, match="max_null_frac"):
        NullCheckValidator(max_null_frac=frac)
