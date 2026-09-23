import pytest
import pandas as pd

from epftoolbox2.data.transformers import Transformer, TimezoneTransformer, LagTransformer


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


class TestTimezoneTransformerDailySnap:
    """Test daily_columns midnight-snapping for winter and summer cases."""

    def _make_df(self, hourly_utc_start: str, periods: int, daily_utc_ts: str, daily_value: float):
        """Build a DataFrame with hourly rows and one sparse daily column."""
        idx = pd.date_range(hourly_utc_start, periods=periods, freq="h", tz="UTC")
        df = pd.DataFrame({"load": range(periods)}, index=idx)
        df["daily_min"] = float("nan")
        if daily_utc_ts in df.index:
            df.loc[daily_utc_ts, "daily_min"] = daily_value
        else:
            # Insert the daily row even if it's outside the hourly grid
            extra = pd.DataFrame(
                {"load": float("nan"), "daily_min": daily_value},
                index=pd.DatetimeIndex([pd.Timestamp(daily_utc_ts, tz="UTC")]),
            )
            df = pd.concat([df, extra]).sort_index()
        return df

    def test_winter_snap_23xx_to_next_midnight(self):
        """CET winter: ENTSOE UTC+2 anchor puts daily value at 23:00+01:00.
        Should snap to 00:00+01:00 of the *next* local day (Wednesday), not Tuesday."""
        # 2025-11-18 22:00 UTC = 2025-11-18 23:00+01:00 CET (1h before Wed midnight)
        df = self._make_df("2025-11-17 00:00", periods=72, daily_utc_ts="2025-11-18 22:00", daily_value=99.0)
        transformer = TimezoneTransformer(target_tz="Europe/Madrid", daily_columns=["daily_min"])
        result = transformer.transform(df)

        # Value must land at Wednesday midnight (2025-11-19 00:00+01:00)
        wednesday_midnight = pd.Timestamp("2025-11-19 00:00", tz="Europe/Madrid")
        assert result.loc[wednesday_midnight, "daily_min"] == 99.0

        # Tuesday midnight must NOT have the value
        tuesday_midnight = pd.Timestamp("2025-11-18 00:00", tz="Europe/Madrid")
        assert pd.isna(result.loc[tuesday_midnight, "daily_min"])

    def test_summer_snap_00xx_already_midnight(self):
        """CEST summer: ENTSOE UTC+2 anchor puts daily value at 00:00+02:00.
        Already at midnight — no shift should occur."""
        # 2025-07-15 22:00 UTC = 2025-07-16 00:00+02:00 CEST (midnight)
        df = self._make_df("2025-07-15 00:00", periods=72, daily_utc_ts="2025-07-15 22:00", daily_value=55.0)
        transformer = TimezoneTransformer(target_tz="Europe/Madrid", daily_columns=["daily_min"])
        result = transformer.transform(df)

        wednesday_midnight = pd.Timestamp("2025-07-16 00:00", tz="Europe/Madrid")
        assert result.loc[wednesday_midnight, "daily_min"] == 55.0

    def test_winter_snap_01xx_to_current_midnight(self):
        """UTC+2 period-start (00:00 UTC) converted to UTC+1: lands at 01:00+01:00.
        Should snap to 00:00+01:00 of the *same* local day."""
        # 2025-11-19 00:00 UTC = 2025-11-19 01:00+01:00 CET
        df = self._make_df("2025-11-18 00:00", periods=72, daily_utc_ts="2025-11-19 00:00", daily_value=77.0)
        transformer = TimezoneTransformer(target_tz="Europe/Madrid", daily_columns=["daily_min"])
        result = transformer.transform(df)

        target_midnight = pd.Timestamp("2025-11-19 00:00", tz="Europe/Madrid")
        assert result.loc[target_midnight, "daily_min"] == 77.0

    def test_already_at_midnight_no_shift(self):
        """Value already at local midnight: needs_shift must be False, no movement."""
        # 2025-11-18 23:00 UTC = 2025-11-19 00:00+01:00 CET (exactly midnight)
        df = self._make_df("2025-11-18 00:00", periods=72, daily_utc_ts="2025-11-18 23:00", daily_value=42.0)
        transformer = TimezoneTransformer(target_tz="Europe/Madrid", daily_columns=["daily_min"])
        result = transformer.transform(df)

        midnight = pd.Timestamp("2025-11-19 00:00", tz="Europe/Madrid")
        assert result.loc[midnight, "daily_min"] == 42.0

    def test_non_daily_columns_unaffected(self):
        """Columns not in daily_columns must not be moved."""
        idx = pd.date_range("2025-11-18 00:00", periods=48, freq="h", tz="UTC")
        df = pd.DataFrame({"load": range(48), "price": range(48)}, index=idx)
        transformer = TimezoneTransformer(target_tz="Europe/Madrid", daily_columns=["daily_min"])
        result = transformer.transform(df)
        # price column should be untouched (no snapping)
        assert list(result["price"]) == list(range(48))


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
