import pytest
import pandas as pd
from unittest.mock import Mock, patch
from epftoolbox2.data.sources.open_meteo import OpenMeteoSource


class TestOpenMeteoSourceInit:
    """Test OpenMeteoSource initialization"""

    def test_init_valid_params(self):
        """Test initialization with valid parameters"""
        source = OpenMeteoSource(latitude=52.23, longitude=21.01)
        assert source.latitude == 52.23
        assert source.longitude == 21.01
        assert source.horizon == 7
        assert source.model == "jma_seamless"
        assert source.prefix == ""
        assert len(source.columns) > 0

    def test_init_custom_horizon(self):
        """Test initialization with custom horizon"""
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, horizon=14)
        assert source.horizon == 14

    def test_init_custom_model(self):
        """Test initialization with custom model"""
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, model="icon_seamless")
        assert source.model == "icon_seamless"

    def test_init_custom_columns(self):
        """Test initialization with custom columns"""
        custom_cols = ["temperature_2m", "rain"]
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, columns=custom_cols)
        assert source.columns == custom_cols
        assert len(source.columns) == 2

    def test_init_with_prefix(self):
        """Test initialization with column prefix"""
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, prefix="weather")
        assert source.prefix == "weather"

    def test_init_invalid_latitude_too_high(self):
        """Test initialization with latitude > 90"""
        with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
            OpenMeteoSource(latitude=91.0, longitude=21.01)

    def test_init_invalid_latitude_too_low(self):
        """Test initialization with latitude < -90"""
        with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
            OpenMeteoSource(latitude=-91.0, longitude=21.01)

    def test_init_invalid_longitude_too_high(self):
        """Test initialization with longitude > 180"""
        with pytest.raises(ValueError, match="Longitude must be between -180 and 180"):
            OpenMeteoSource(latitude=52.23, longitude=181.0)

    def test_init_invalid_longitude_too_low(self):
        """Test initialization with longitude < -180"""
        with pytest.raises(ValueError, match="Longitude must be between -180 and 180"):
            OpenMeteoSource(latitude=52.23, longitude=-181.0)

    def test_init_invalid_horizon_zero(self):
        """Test initialization with horizon = 0"""
        with pytest.raises(ValueError, match="Horizon must be a positive integer"):
            OpenMeteoSource(latitude=52.23, longitude=21.01, horizon=0)

    def test_init_invalid_horizon_negative(self):
        """Test initialization with negative horizon"""
        with pytest.raises(ValueError, match="Horizon must be a positive integer"):
            OpenMeteoSource(latitude=52.23, longitude=21.01, horizon=-1)

    def test_validate_config_returns_true(self):
        """Test that _validate_config returns True on success"""
        source = OpenMeteoSource(latitude=52.23, longitude=21.01)
        result = source._validate_config()
        assert result is True

    def test_default_columns_exist(self):
        """Test that DEFAULT_COLUMNS is properly defined"""
        source = OpenMeteoSource(latitude=52.23, longitude=21.01)
        assert "temperature_2m" in source.columns
        assert "rain" in source.columns
        assert "wind_speed_10m" in source.columns


class TestOpenMeteoSourceValidation:
    """Test fetch validation"""

    def test_fetch_validates_timestamps(self):
        """Test that fetch validates end > start"""
        source = OpenMeteoSource(latitude=52.23, longitude=21.01)
        start = pd.Timestamp("2024-01-02", tz="UTC")
        end = pd.Timestamp("2024-01-01", tz="UTC")

        with pytest.raises(ValueError, match="End timestamp.*must be after start timestamp"):
            source.fetch(start, end)

    def test_fetch_converts_naive_timestamps_to_utc(self):
        """Test that naive timestamps are converted to UTC"""
        source = OpenMeteoSource(latitude=52.23, longitude=21.01)

        with patch.object(source, "_generate_chunks", return_value=[]) as mock_chunks:
            with patch.object(source.logger, "info"):
                start = pd.Timestamp("2024-01-01")
                end = pd.Timestamp("2024-01-02")

                source.fetch(start, end)

                call_args = mock_chunks.call_args[0]
                assert call_args[0].tz is not None
                assert call_args[1].tz is not None

    def test_fetch_handles_timezone_aware_timestamps(self):
        """Test that timezone-aware timestamps are handled correctly"""
        source = OpenMeteoSource(latitude=52.23, longitude=21.01)

        with patch.object(source, "_generate_chunks", return_value=[]) as mock_chunks:
            with patch.object(source.logger, "info"):
                start = pd.Timestamp("2024-01-01", tz="Europe/Warsaw")
                end = pd.Timestamp("2024-01-02", tz="Europe/Warsaw")

                source.fetch(start, end)

                call_args = mock_chunks.call_args[0]
                assert str(call_args[0].tz) == "UTC"
                assert str(call_args[1].tz) == "UTC"


class TestOpenMeteoSourceChunking:
    """Test date range chunking"""

    def test_generate_chunks_single_month(self):
        """Test chunking for a single month period"""
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, horizon=7)
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-15", tz="UTC")

        chunks = source._generate_chunks(start, end, months=3)
        assert len(chunks) == 1
        assert chunks[0][0] == start

    def test_generate_chunks_multiple_months(self):
        """Test chunking for multi-month period"""
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, horizon=7)
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-07-01", tz="UTC")

        chunks = source._generate_chunks(start, end, months=3)
        assert len(chunks) >= 2
        assert chunks[0][0] == start
        assert chunks[-1][1] == end

    def test_generate_chunks_validates_months_parameter(self):
        """Test that _generate_chunks validates months parameter"""
        source = OpenMeteoSource(latitude=52.23, longitude=21.01)
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-07-01", tz="UTC")

        with pytest.raises(ValueError, match="months parameter must be positive"):
            source._generate_chunks(start, end, months=0)

        with pytest.raises(ValueError, match="months parameter must be positive"):
            source._generate_chunks(start, end, months=-1)

    def test_generate_chunks_overlap_by_horizon(self):
        """Test that chunks overlap by horizon days"""
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, horizon=7)
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-07-01", tz="UTC")

        chunks = source._generate_chunks(start, end, months=3)

        if len(chunks) > 1:
            first_chunk_end = chunks[0][1]
            second_chunk_start = chunks[1][0]
            assert second_chunk_start < first_chunk_end


class TestOpenMeteoSourceAPIRequests:
    """Test API request methods"""

    @patch("epftoolbox2.data.sources.open_meteo.requests.Session")
    def test_fetch_chunk_success(self, mock_session, sample_weather_response):
        """Test successful API request"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_weather_response
        mock_session.return_value.get.return_value = mock_response

        source = OpenMeteoSource(latitude=52.23, longitude=21.01, horizon=2)
        source.session = mock_session.return_value

        result = source._fetch_chunk(
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-02", tz="UTC"),
        )

        assert isinstance(result, pd.DataFrame)

    @patch("epftoolbox2.data.sources.open_meteo.requests.Session")
    def test_fetch_chunk_handles_rate_limit(self, mock_session):
        """Test handling of rate limit errors"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "error": True,
            "reason": "Minutely API request limit exceeded. Please try again in one minute.",
        }
        mock_session.return_value.get.return_value = mock_response

        source = OpenMeteoSource(latitude=52.23, longitude=21.01)
        source.session = mock_session.return_value

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError, match="Failed to fetch data after"):
                source._fetch_chunk(
                    pd.Timestamp("2024-01-01", tz="UTC"),
                    pd.Timestamp("2024-01-02", tz="UTC"),
                )
            assert mock_sleep.called

    @patch("epftoolbox2.data.sources.open_meteo.requests.Session")
    def test_fetch_chunk_handles_concurrent_requests_error(self, mock_session):
        """Test handling of too many concurrent requests error"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "error": True,
            "reason": "Too many concurrent requests",
        }
        mock_session.return_value.get.return_value = mock_response

        source = OpenMeteoSource(latitude=52.23, longitude=21.01)
        source.session = mock_session.return_value

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError, match="Failed to fetch data after"):
                source._fetch_chunk(
                    pd.Timestamp("2024-01-01", tz="UTC"),
                    pd.Timestamp("2024-01-02", tz="UTC"),
                )
            assert mock_sleep.called

    @patch("epftoolbox2.data.sources.open_meteo.requests.Session")
    def test_fetch_chunk_retries_on_request_exception(self, mock_session):
        """Test retry logic on request exceptions"""
        import requests

        mock_session.return_value.get.side_effect = requests.RequestException("Network error")

        source = OpenMeteoSource(latitude=52.23, longitude=21.01)
        source.session = mock_session.return_value

        with patch("time.sleep"):
            with pytest.raises(requests.RequestException, match="Network error"):
                source._fetch_chunk(
                    pd.Timestamp("2024-01-01", tz="UTC"),
                    pd.Timestamp("2024-01-02", tz="UTC"),
                )

        assert mock_session.return_value.get.call_count == 5


class TestOpenMeteoSourceParsing:
    """Test weather data parsing methods"""

    def test_parse_weather_data_returns_dataframe(self, sample_weather_response):
        """Test that parse_weather_data returns a DataFrame"""
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, horizon=2)
        df = source._parse_weather_data(sample_weather_response)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_parse_weather_data_creates_correct_columns(self, sample_weather_response):
        """Test that parsing creates correctly named columns"""
        source = OpenMeteoSource(
            latitude=52.23,
            longitude=21.01,
            horizon=2,
            columns=["temperature_2m"],
        )
        df = source._parse_weather_data(sample_weather_response)

        assert "temperature_2m_d+1" in df.columns
        assert "temperature_2m_d+2" in df.columns

    def test_parse_weather_data_handles_missing_hourly(self):
        """Test that parsing handles missing hourly data"""
        source = OpenMeteoSource(latitude=52.23, longitude=21.01)
        df = source._parse_weather_data({})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_parse_weather_data_handles_missing_time(self):
        """Test that parsing handles missing time data"""
        source = OpenMeteoSource(latitude=52.23, longitude=21.01)
        df = source._parse_weather_data({"hourly": {}})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_parse_weather_data_creates_utc_index(self, sample_weather_response):
        """Test that the index is in UTC timezone"""
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, horizon=2)
        df = source._parse_weather_data(sample_weather_response)
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"

    def test_parse_weather_data_skips_incomplete_data(self, sample_weather_response_incomplete):
        """Test that parsing gracefully handles incomplete data"""
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, horizon=2)
        df = source._parse_weather_data(sample_weather_response_incomplete)
        assert isinstance(df, pd.DataFrame)

    def test_parse_weather_data_filters_future_forecasts(self):
        """Test that forecasts beyond now+13h are filtered out"""
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, horizon=2)

        # Create response with timestamps far in the future
        future_time = (pd.Timestamp.now(tz="UTC") + pd.DateOffset(days=365)).strftime("%Y-%m-%dT%H:00")
        response = {
            "hourly": {
                "time": [future_time],
                "temperature_2m_previous_day1": [15.0] * 50,
                "temperature_2m_previous_day2": [16.0] * 50,
            }
        }

        df = source._parse_weather_data(response)
        assert len(df) == 0


class TestOpenMeteoSourceFetch:
    """Test full fetch workflow"""

    @patch("epftoolbox2.data.sources.open_meteo.requests.Session")
    def test_fetch_returns_dataframe(self, mock_session, sample_weather_response):
        """Test that fetch returns a DataFrame"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_weather_response
        mock_session.return_value.get.return_value = mock_response

        source = OpenMeteoSource(latitude=52.23, longitude=21.01, horizon=2)
        source.session = mock_session.return_value

        result = source.fetch(
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-02", tz="UTC"),
        )

        assert isinstance(result, pd.DataFrame)

    @patch("epftoolbox2.data.sources.open_meteo.requests.Session")
    def test_fetch_applies_prefix(self, mock_session, sample_weather_response):
        """Test that fetch applies column prefix"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_weather_response
        mock_session.return_value.get.return_value = mock_response

        source = OpenMeteoSource(
            latitude=52.23,
            longitude=21.01,
            horizon=2,
            prefix="test",
            columns=["temperature_2m"],
        )
        source.session = mock_session.return_value

        df = source.fetch(
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-02", tz="UTC"),
        )

        assert all(col.startswith("test_") for col in df.columns)

    @patch("epftoolbox2.data.sources.open_meteo.requests.Session")
    def test_fetch_removes_duplicates(self, mock_session, sample_weather_response):
        """Test that fetch removes duplicate timestamps"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_weather_response
        mock_session.return_value.get.return_value = mock_response

        source = OpenMeteoSource(latitude=52.23, longitude=21.01, horizon=2)
        source.session = mock_session.return_value

        df = source.fetch(
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-02", tz="UTC"),
        )

        assert not df.index.duplicated().any()

    @patch("epftoolbox2.data.sources.open_meteo.requests.Session")
    def test_fetch_returns_empty_df_when_no_data(self, mock_session):
        """Test that fetch returns empty DataFrame when no data available"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"hourly": {"time": []}}
        mock_session.return_value.get.return_value = mock_response

        source = OpenMeteoSource(latitude=52.23, longitude=21.01)
        source.session = mock_session.return_value

        df = source.fetch(
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-02", tz="UTC"),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


@pytest.mark.integration
class TestOpenMeteoSourceIntegration:
    """Integration tests that hit the real API"""

    def test_fetch_weather_data_warsaw(self):
        """Test fetching weather data for Warsaw, Poland"""
        source = OpenMeteoSource(
            latitude=52.2297,
            longitude=21.0122,
            horizon=3,
        )
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")

        df = source.fetch(start, end)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert df.index.tz is not None

    def test_fetch_weather_data_limited_columns(self):
        """Test fetching with limited columns"""
        source = OpenMeteoSource(
            latitude=52.2297,
            longitude=21.0122,
            horizon=2,
            columns=["temperature_2m", "wind_speed_10m"],
        )
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")

        df = source.fetch(start, end)

        temp_cols = [col for col in df.columns if "temperature_2m" in col]
        wind_cols = [col for col in df.columns if "wind_speed_10m" in col]

        assert len(temp_cols) == 2
        assert len(wind_cols) == 2

    def test_fetch_weather_data_with_prefix(self):
        """Test fetching with column prefix"""
        source = OpenMeteoSource(
            latitude=52.2297,
            longitude=21.0122,
            horizon=2,
            prefix="weather",
            columns=["temperature_2m"],
        )
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")

        df = source.fetch(start, end)

        assert all(col.startswith("weather_") for col in df.columns)

    def test_fetch_long_period(self):
        """Test fetching data for a longer period (tests chunking)"""
        source = OpenMeteoSource(
            latitude=52.2297,
            longitude=21.0122,
            horizon=7,
            columns=["temperature_2m"],
        )
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-06-01", tz="UTC")

        df = source.fetch(start, end)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 1000
