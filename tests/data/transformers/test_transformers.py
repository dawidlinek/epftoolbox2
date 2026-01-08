import pytest
import pandas as pd

from epftoolbox2.data.transformers import Transformer, TimezoneTransformer, ResampleTransformer, LagTransformer


class TestTimezoneTransformerInit:
    """Test TimezoneTransformer initialization"""

    def test_init_valid_timezone(self):
        """Test initialization with valid timezone"""
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")
        assert transformer.target_tz == "Europe/Warsaw"

    def test_init_utc(self):
        """Test initialization with UTC"""
        transformer = TimezoneTransformer(target_tz="UTC")
        assert transformer.target_tz == "UTC"

    def test_init_invalid_timezone(self):
        """Test initialization with invalid timezone raises error"""
        with pytest.raises(ValueError, match="Invalid timezone"):
            TimezoneTransformer(target_tz="Invalid/Timezone")


class TestTimezoneTransformerTransform:
    """Test TimezoneTransformer transform method"""

    @pytest.fixture
    def sample_utc_dataframe(self):
        """Create a sample DataFrame with UTC index"""
        dates = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        return pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=dates)

    @pytest.fixture
    def sample_naive_dataframe(self):
        """Create a sample DataFrame with timezone-naive index"""
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        return pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=dates)

    def test_transform_utc_to_warsaw(self, sample_utc_dataframe):
        """Test converting UTC to Europe/Warsaw"""
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")
        result = transformer.transform(sample_utc_dataframe)

        assert str(result.index.tz) == "Europe/Warsaw"
        # Warsaw is UTC+1 in winter
        assert result.index[0].hour == 1  # 00:00 UTC -> 01:00 Warsaw

    def test_transform_utc_to_new_york(self, sample_utc_dataframe):
        """Test converting UTC to America/New_York"""
        transformer = TimezoneTransformer(target_tz="America/New_York")
        result = transformer.transform(sample_utc_dataframe)

        assert str(result.index.tz) == "America/New_York"
        # New York is UTC-5 in winter
        assert result.index[0].hour == 19  # 00:00 UTC on Jan 1 -> 19:00 Dec 31 NY

    def test_transform_naive_to_warsaw(self, sample_naive_dataframe):
        """Test converting naive timestamps (assumed UTC) to Europe/Warsaw"""
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")
        result = transformer.transform(sample_naive_dataframe)

        assert str(result.index.tz) == "Europe/Warsaw"
        assert result.index[0].hour == 1  # 00:00 UTC -> 01:00 Warsaw

    def test_transform_preserves_data(self, sample_utc_dataframe):
        """Test that transform preserves the data values"""
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")
        result = transformer.transform(sample_utc_dataframe)

        assert list(result["value"]) == [1, 2, 3, 4, 5]

    def test_transform_does_not_modify_original(self, sample_utc_dataframe):
        """Test that transform returns a copy, not modifying original"""
        original_tz = str(sample_utc_dataframe.index.tz)
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")
        transformer.transform(sample_utc_dataframe)

        assert str(sample_utc_dataframe.index.tz) == original_tz

    def test_transform_invalid_index_type(self):
        """Test that non-DatetimeIndex raises error"""
        df = pd.DataFrame({"value": [1, 2, 3]}, index=[0, 1, 2])
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")

        with pytest.raises(ValueError, match="DatetimeIndex"):
            transformer.transform(df)

    def test_transform_multiple_columns(self):
        """Test transform with multiple columns"""
        dates = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "price": [10.0, 20.0, 30.0],
                "load": [100, 200, 300],
            },
            index=dates,
        )

        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")
        result = transformer.transform(df)

        assert list(result.columns) == ["price", "load"]
        assert list(result["price"]) == [10.0, 20.0, 30.0]
        assert list(result["load"]) == [100, 200, 300]


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


class TestResampleTransformerInit:
    """Test ResampleTransformer initialization"""

    def test_init_default_values(self):
        """Test initialization with default values"""
        transformer = ResampleTransformer()
        assert transformer.freq == "1h"
        assert transformer.method == "linear"

    def test_init_custom_freq(self):
        """Test initialization with custom frequency"""
        transformer = ResampleTransformer(freq="15min")
        assert transformer.freq == "15min"

    def test_init_ffill_method(self):
        """Test initialization with ffill method"""
        transformer = ResampleTransformer(method="ffill")
        assert transformer.method == "ffill"

    def test_init_bfill_method(self):
        """Test initialization with bfill method"""
        transformer = ResampleTransformer(method="bfill")
        assert transformer.method == "bfill"

    def test_init_linear_method(self):
        """Test initialization with linear method"""
        transformer = ResampleTransformer(method="linear")
        assert transformer.method == "linear"

    def test_init_invalid_method(self):
        """Test initialization with invalid method raises error"""
        with pytest.raises(ValueError, match="Invalid method"):
            ResampleTransformer(method="invalid")

    def test_init_invalid_method_shows_valid_options(self):
        """Test that invalid method error shows valid options"""
        with pytest.raises(ValueError, match="linear|ffill|bfill"):
            ResampleTransformer(method="cubic")

    def test_init_with_columns(self):
        """Test initialization with columns"""
        transformer = ResampleTransformer(columns=["price"])
        assert transformer.columns == ["price"]

    def test_init_with_single_column_string(self):
        """Test initialization with single column string"""
        transformer = ResampleTransformer(columns="price")
        assert transformer.columns == ["price"]


class TestResampleTransformerTransform:
    """Test ResampleTransformer transform method"""

    @pytest.fixture
    def sample_hourly_dataframe(self):
        """Create a sample DataFrame with hourly index"""
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        return pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)

    @pytest.fixture
    def sample_daily_dataframe(self):
        """Create a sample DataFrame with daily index"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        return pd.DataFrame({"value": [10.0, 20.0, 30.0]}, index=dates)

    @pytest.fixture
    def sample_tz_aware_dataframe(self):
        """Create a sample DataFrame with timezone-aware index"""
        dates = pd.date_range("2024-01-01", periods=5, freq="h", tz="Europe/Warsaw")
        return pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)

    def test_transform_upsample_to_30min(self, sample_hourly_dataframe):
        """Test upsampling from hourly to 30-minute frequency"""
        transformer = ResampleTransformer(freq="30min", method="linear")
        result = transformer.transform(sample_hourly_dataframe)

        assert len(result) == 9  # 5 hours -> 9 half-hour periods
        assert result.index.freq == pd.Timedelta("30min")

    def test_transform_downsample_to_2h(self, sample_hourly_dataframe):
        """Test downsampling from hourly to 2-hour frequency"""
        transformer = ResampleTransformer(freq="2h", method="linear")
        result = transformer.transform(sample_hourly_dataframe)

        assert len(result) == 3  # 5 hours -> 3 two-hour periods
        assert result.index.freq == pd.Timedelta("2h")

    def test_transform_daily_to_hourly_linear(self, sample_daily_dataframe):
        """Test resampling daily to hourly with linear interpolation"""
        transformer = ResampleTransformer(freq="1h", method="linear")
        result = transformer.transform(sample_daily_dataframe)

        # Check that interpolation fills values between 10 and 20
        assert result.index.freq == pd.Timedelta("1h")
        assert result["value"].iloc[0] == 10.0
        # Linear interpolation should produce intermediate values
        assert result["value"].iloc[12] == pytest.approx(15.0, rel=0.01)

    def test_transform_ffill_method(self, sample_hourly_dataframe):
        """Test forward fill interpolation method"""
        transformer = ResampleTransformer(freq="30min", method="ffill")
        result = transformer.transform(sample_hourly_dataframe)

        # Forward fill should repeat previous values
        assert result["value"].iloc[0] == 1.0
        assert result["value"].iloc[1] == 1.0  # Filled forward from 1.0
        assert result["value"].iloc[2] == 2.0

    def test_transform_bfill_method(self, sample_hourly_dataframe):
        """Test backward fill interpolation method"""
        transformer = ResampleTransformer(freq="30min", method="bfill")
        result = transformer.transform(sample_hourly_dataframe)

        # Backward fill should use next values
        assert result["value"].iloc[0] == 1.0
        assert result["value"].iloc[1] == 2.0  # Filled backward from 2.0
        assert result["value"].iloc[2] == 2.0

    def test_transform_preserves_timezone(self, sample_tz_aware_dataframe):
        """Test that transform preserves timezone information"""
        transformer = ResampleTransformer(freq="30min", method="linear")
        result = transformer.transform(sample_tz_aware_dataframe)

        assert str(result.index.tz) == "Europe/Warsaw"

    def test_transform_preserves_column_names(self):
        """Test that transform preserves column names"""
        dates = pd.date_range("2024-01-01", periods=3, freq="h")
        df = pd.DataFrame(
            {"price": [10.0, 20.0, 30.0], "load": [100.0, 200.0, 300.0]},
            index=dates,
        )

        transformer = ResampleTransformer(freq="30min", method="linear")
        result = transformer.transform(df)

        assert list(result.columns) == ["price", "load"]

    def test_transform_multiple_columns_linear(self):
        """Test linear interpolation with multiple columns"""
        dates = pd.date_range("2024-01-01", periods=3, freq="h")
        df = pd.DataFrame(
            {"price": [10.0, 20.0, 30.0], "load": [100.0, 200.0, 300.0]},
            index=dates,
        )

        transformer = ResampleTransformer(freq="30min", method="linear")
        result = transformer.transform(df)

        # Check both columns are interpolated
        assert result["price"].iloc[1] == pytest.approx(15.0, rel=0.01)
        assert result["load"].iloc[1] == pytest.approx(150.0, rel=0.01)

    def test_transform_does_not_modify_original(self, sample_hourly_dataframe):
        """Test that transform returns a copy, not modifying original"""
        original_len = len(sample_hourly_dataframe)
        transformer = ResampleTransformer(freq="30min", method="linear")
        transformer.transform(sample_hourly_dataframe)

        assert len(sample_hourly_dataframe) == original_len

    def test_transform_invalid_index_type(self):
        """Test that non-DatetimeIndex raises error"""
        df = pd.DataFrame({"value": [1, 2, 3]}, index=[0, 1, 2])
        transformer = ResampleTransformer(freq="1h", method="linear")

        with pytest.raises(ValueError, match="DatetimeIndex"):
            transformer.transform(df)

    def test_transform_empty_dataframe(self):
        """Test transform with empty dataframe"""
        dates = pd.DatetimeIndex([], dtype="datetime64[ns]")
        df = pd.DataFrame({"value": []}, index=dates)
        transformer = ResampleTransformer(freq="1h", method="linear")

        result = transformer.transform(df)
        assert len(result) == 0

    def test_transform_single_row(self):
        """Test transform with single row dataframe"""
        dates = pd.date_range("2024-01-01", periods=1, freq="h")
        df = pd.DataFrame({"value": [1.0]}, index=dates)
        transformer = ResampleTransformer(freq="30min", method="linear")

        result = transformer.transform(df)
        assert len(result) == 1
        assert result["value"].iloc[0] == 1.0

    def test_transform_values_are_rounded(self, sample_hourly_dataframe):
        """Test that transformed values are rounded to 3 decimal places"""
        transformer = ResampleTransformer(freq="30min", method="linear")
        result = transformer.transform(sample_hourly_dataframe)

        # All values should be rounded to 3 decimal places
        for value in result["value"]:
            rounded = round(value, 3)
            assert value == rounded

    def test_transform_various_frequencies(self, sample_hourly_dataframe):
        """Test transform with various frequency strings"""
        frequencies = ["15min", "30min", "1h", "2h"]

        for freq in frequencies:
            transformer = ResampleTransformer(freq=freq, method="linear")
            result = transformer.transform(sample_hourly_dataframe)
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0

    def test_transform_specific_columns(self):
        """Test that interpolation is applied only to specified columns"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0], "load": [100.0, 200.0, 300.0]}, index=dates)

        # Resample to 12h, so we get one intermediate row
        transformer = ResampleTransformer(freq="12h", method="linear", columns=["price"])
        result = transformer.transform(df)

        assert len(result) == 5
        # Price should be interpolated (linear)
        # 10, nan, 20, nan, 30 -> 10, 15, 20, 25, 30
        assert result["price"].iloc[1] == 15.0

        # Load should NOT be interpolated (remain NaN)
        assert pd.isna(result["load"].iloc[1])

    def test_transform_columns_validation(self, sample_hourly_dataframe):
        """Test validation of columns existence"""
        transformer = ResampleTransformer(columns=["non_existent"])
        with pytest.raises(ValueError, match="Columns not found"):
            transformer.transform(sample_hourly_dataframe)


class TestResampleTransformerIsTransformer:
    """Test ResampleTransformer inheritance"""

    def test_resample_transformer_is_transformer(self):
        """Test that ResampleTransformer is a Transformer subclass"""
        transformer = ResampleTransformer()
        assert isinstance(transformer, Transformer)


class TestLagTransformerInit:
    """Test LagTransformer initialization"""

    def test_init_with_required_columns(self):
        """Test initialization with required columns"""
        transformer = LagTransformer(columns=["price"])
        assert transformer.columns == ["price"]
        assert transformer.lags == [1]
        assert transformer.freq == pd.Timedelta("1h")

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
        assert transformer.freq == pd.Timedelta("1d")

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
