import pytest
import pandas as pd

from epftoolbox2.data.transformers import Transformer, TimezoneTransformer, LagTransformer


class TestTransformerAbstract:
    """Test Transformer abstract base class"""

    def test_transformer_is_abstract(self):
        """Test that Transformer cannot be instantiated directly"""
        with pytest.raises(TypeError):
            Transformer()

    def test_timezone_transformer_is_transformer(self):
        """Test that TimezoneTransformer is a Transformer subclass"""
        transformer = TimezoneTransformer(target_tz="UTC")
        assert isinstance(transformer, Transformer)


class TestLagTransformerInit:
    """Test LagTransformer initialization"""

    def test_init_with_required_columns(self):
        """Test initialization with required columns"""
        transformer = LagTransformer(columns=["price"])
        assert transformer.columns == ["price"]
        assert transformer.lags == [1]
        assert transformer.freq == "1h"

    def test_init_single_column_string(self):
        """Test initialization with single column as string"""
        transformer = LagTransformer(columns="price")
        assert transformer.columns == ["price"]

    def test_init_multiple_columns(self):
        """Test initialization with multiple columns"""
        transformer = LagTransformer(columns=["price", "load"])
        assert transformer.columns == ["price", "load"]

    def test_init_single_lag(self):
        """Test initialization with single lag as int"""
        transformer = LagTransformer(columns=["price"], lags=7)
        assert transformer.lags == [7]

    def test_init_multiple_lags(self):
        """Test initialization with multiple lags"""
        transformer = LagTransformer(columns=["price"], lags=[1, 7, 14])
        assert transformer.lags == [1, 7, 14]

    def test_init_custom_freq(self):
        """Test initialization with custom frequency"""
        transformer = LagTransformer(columns=["price"], freq="1d")
        assert transformer.freq == "1d"

    def test_init_empty_lags(self):
        """Test initialization with empty lags raises error"""
        with pytest.raises(ValueError, match="At least one lag"):
            LagTransformer(columns=["price"], lags=[])

    def test_init_no_columns(self):
        """Test initialization with no columns raises error"""
        with pytest.raises(ValueError, match="At least one column"):
            LagTransformer(columns=None)


class TestLagTransformerTransform:
    """Test LagTransformer transform method"""

    @pytest.fixture
    def sample_hourly_dataframe(self):
        """Create a sample DataFrame with 48 hours of hourly data"""
        dates = pd.date_range("2024-01-01", periods=48, freq="h")
        return pd.DataFrame(
            {"price": range(48), "load": range(100, 148)},
            index=dates,
        )

    def test_transform_day_lag(self, sample_hourly_dataframe):
        """Test day-based lag transformation"""
        transformer = LagTransformer(columns=["price"], lags=[1], freq="1d")
        result = transformer.transform(sample_hourly_dataframe)

        assert "price_d-1" in result.columns
        # First 24 values should be NaN (1 day lag)
        assert result["price_d-1"].isna().sum() == 24
        # Value at index 24 should equal original value at index 0
        assert result["price_d-1"].iloc[24] == 0

    def test_transform_hour_lag(self, sample_hourly_dataframe):
        """Test hour-based lag transformation"""
        transformer = LagTransformer(columns=["price"], lags=[1], freq="1h")
        result = transformer.transform(sample_hourly_dataframe)

        assert "price_h-1" in result.columns
        # First value should be NaN (1 hour lag)
        assert pd.isna(result["price_h-1"].iloc[0])
        # Value at index 1 should equal original value at index 0
        assert result["price_h-1"].iloc[1] == 0

    def test_transform_multiple_lags(self, sample_hourly_dataframe):
        """Test transformation with multiple lags"""
        transformer = LagTransformer(columns=["price"], lags=[1, 2], freq="1d")
        result = transformer.transform(sample_hourly_dataframe)

        assert "price_d-1" in result.columns
        assert "price_d-2" in result.columns
        # 2 day lag should have 48 NaN values (all data)
        assert result["price_d-2"].isna().sum() == 48

    def test_transform_multiple_columns(self, sample_hourly_dataframe):
        """Test transformation with multiple columns"""
        transformer = LagTransformer(columns=["price", "load"], lags=[1], freq="1d")
        result = transformer.transform(sample_hourly_dataframe)

        assert "price_d-1" in result.columns
        assert "load_d-1" in result.columns

    def test_transform_negative_lag(self, sample_hourly_dataframe):
        """Test negative lag (forward shift)"""
        transformer = LagTransformer(columns=["price"], lags=[-1], freq="1h")
        result = transformer.transform(sample_hourly_dataframe)

        assert "price_h+1" in result.columns
        # Last value should be NaN (forward shift)
        assert pd.isna(result["price_h+1"].iloc[-1])
        # Value at index 0 should equal original value at index 1
        assert result["price_h+1"].iloc[0] == 1

    def test_transform_preserves_original_columns(self, sample_hourly_dataframe):
        """Test that transform preserves original columns"""
        transformer = LagTransformer(columns=["price"], lags=[1], freq="1d")
        result = transformer.transform(sample_hourly_dataframe)

        assert "price" in result.columns
        assert "load" in result.columns
        assert list(result["price"]) == list(sample_hourly_dataframe["price"])

    def test_transform_does_not_modify_original(self, sample_hourly_dataframe):
        """Test that transform returns a copy, not modifying original"""
        original_columns = list(sample_hourly_dataframe.columns)
        transformer = LagTransformer(columns=["price"], lags=[1])
        transformer.transform(sample_hourly_dataframe)

        assert list(sample_hourly_dataframe.columns) == original_columns


class TestLagTransformerIsTransformer:
    """Test LagTransformer inheritance"""

    def test_lag_transformer_is_transformer(self):
        """Test that LagTransformer is a Transformer subclass"""
        transformer = LagTransformer(columns=["test"])
        assert isinstance(transformer, Transformer)
