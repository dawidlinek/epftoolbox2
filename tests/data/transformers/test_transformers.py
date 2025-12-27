import pytest
import pandas as pd
from zoneinfo import ZoneInfo

from epftoolbox2.data.transformers import Transformer, TimezoneTransformer


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
