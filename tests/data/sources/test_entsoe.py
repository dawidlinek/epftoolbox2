import pytest
import pandas as pd
from unittest.mock import Mock, patch
from epftoolbox2.data.sources.entsoe import (
    EntsoeSource,
    lookup_area,
    AREAS,
)


class TestAreaLookup:
    """Test area lookup functionality"""

    def test_lookup_valid_area(self):
        """Test looking up a valid area code"""
        name, code = lookup_area("PL")
        assert name == "PL"
        assert code == "10YPL-AREA-----S"

    def test_lookup_case_insensitive(self):
        """Test that lookup is case insensitive"""
        name1, code1 = lookup_area("pl")
        name2, code2 = lookup_area("PL")
        assert name1 == name2
        assert code1 == code2

    def test_lookup_invalid_area(self):
        """Test that invalid area raises ValueError"""
        with pytest.raises(ValueError, match="Invalid country code"):
            lookup_area("INVALID")

    def test_all_areas_have_codes(self):
        """Test that all areas in AREAS dict have valid codes"""
        for name, code in AREAS.items():
            assert isinstance(name, str)
            assert isinstance(code, str)
            assert len(code) > 0


class TestEntsoeSourceInit:
    """Test EntsoeSource initialization"""

    def test_init_valid_params(self):
        """Test initialization with valid parameters"""
        source = EntsoeSource("PL", "test-api-key", type=["load"])
        assert source.area_name == "PL"
        assert source.area_code == "10YPL-AREA-----S"
        assert source.api_key == "test-api-key"
        assert source.types == ["load"]

    def test_init_multiple_types(self):
        """Test initialization with multiple data types"""
        source = EntsoeSource("DE", "test-key", type=["load", "generation", "price"])
        assert len(source.types) == 3
        assert "load" in source.types
        assert "generation" in source.types
        assert "price" in source.types

    def test_init_invalid_country(self):
        """Test initialization with invalid country code"""
        with pytest.raises(ValueError, match="Invalid country code"):
            EntsoeSource("XX", "test-key", type=["load"])

    def test_init_empty_api_key(self):
        """Test initialization with empty API key"""
        with pytest.raises(ValueError, match="API key cannot be empty"):
            EntsoeSource("PL", "", type=["load"])

    def test_init_empty_types(self):
        """Test initialization with empty types list"""
        with pytest.raises(ValueError, match="At least one data type must be specified"):
            EntsoeSource("PL", "test-key", type=[])

    def test_init_invalid_type(self):
        """Test initialization with invalid data type"""
        with pytest.raises(ValueError, match="Invalid type"):
            EntsoeSource("PL", "test-key", type=["invalid_type"])


class TestEntsoeSourceChunking:
    """Test date range chunking"""

    def test_generate_chunks_single_month(self):
        """Test chunking for a single month period"""
        source = EntsoeSource("PL", "test-key", type=["load"])
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-15", tz="UTC")

        chunks = source._generate_chunks(start, end, months=3)
        assert len(chunks) == 1
        assert chunks[0][0] == start
        assert chunks[0][1] == end

    def test_generate_chunks_multiple_months(self):
        """Test chunking for multi-month period"""
        source = EntsoeSource("PL", "test-key", type=["load"])
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-07-01", tz="UTC")

        chunks = source._generate_chunks(start, end, months=3)
        assert len(chunks) == 2
        assert chunks[0][0] == start
        assert chunks[-1][1] == end

    def test_generate_chunks_exact_boundary(self):
        """Test chunking with exact 3-month boundary"""
        source = EntsoeSource("PL", "test-key", type=["load"])
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-04-01", tz="UTC")

        chunks = source._generate_chunks(start, end, months=3)
        assert len(chunks) == 1


class TestEntsoeSourceAPIRequests:
    """Test API request methods"""

    @patch("epftoolbox2.data.sources.entsoe.requests.Session")
    def test_api_request_success(self, mock_session):
        """Test successful API request"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<data>test</data>"
        mock_response.headers = {"content-type": "application/xml"}
        mock_session.return_value.get.return_value = mock_response

        source = EntsoeSource("PL", "test-key", type=["load"])
        source.session = mock_session.return_value

        result = source._api_request({"documentType": "A65"}, pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC"))

        assert result == "<data>test</data>"

    @patch("epftoolbox2.data.sources.entsoe.requests.Session")
    def test_api_request_no_matching_data(self, mock_session):
        """Test API request with no matching data"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<text>No matching data found</text>"
        mock_response.headers = {"content-type": "application/xml"}
        mock_session.return_value.get.return_value = mock_response

        source = EntsoeSource("PL", "test-key", type=["load"])
        source.session = mock_session.return_value

        result = source._api_request({"documentType": "A65"}, pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC"))

        assert result is None


class TestEntsoeSourceParsing:
    """Test XML parsing methods"""

    def test_resolution_to_timedelta_60min(self):
        """Test resolution conversion for 60 minutes"""
        source = EntsoeSource("PL", "test-key", type=["load"])
        result = source._resolution_to_timedelta("PT60M")
        assert result == "60min"

    def test_resolution_to_timedelta_15min(self):
        """Test resolution conversion for 15 minutes"""
        source = EntsoeSource("PL", "test-key", type=["load"])
        result = source._resolution_to_timedelta("PT15M")
        assert result == "15min"

    def test_resolution_to_timedelta_invalid(self):
        """Test resolution conversion for invalid format"""
        source = EntsoeSource("PL", "test-key", type=["load"])
        with pytest.raises(NotImplementedError):
            source._resolution_to_timedelta("INVALID")

    def test_parse_loads_returns_dataframe(self, sample_xml_load):
        """Test that parse_loads returns a DataFrame"""
        source = EntsoeSource("PL", "test-key", type=["load"])
        df = source._parse_loads(sample_xml_load, "A16")
        assert isinstance(df, pd.DataFrame)
        assert "load_actual" in df.columns

    def test_parse_generation_returns_dataframe(self, sample_xml_generation):
        """Test that parse_generation returns a DataFrame"""
        source = EntsoeSource("PL", "test-key", type=["generation"])
        df = source._parse_generation(sample_xml_generation)
        assert isinstance(df, pd.DataFrame)
        assert "solar" in df.columns

    def test_parse_prices_returns_dict(self, sample_xml_price):
        """Test that parse_prices returns a dict"""
        source = EntsoeSource("PL", "test-key", type=["price"])
        result = source._parse_prices(sample_xml_price)
        assert isinstance(result, dict)
        assert "60min" in result
        assert "15min" in result


@pytest.mark.integration
class TestEntsoeSourceIntegration:
    """Integration tests that hit the real API (require API key)"""

    def test_fetch_load_data_poland(self, entsoe_api_key):
        """Test fetching load data for Poland"""
        source = EntsoeSource("PL", entsoe_api_key, type=["load"])
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-03", tz="UTC")

        data = source.fetch(start, end)

        assert "load" in data
        assert isinstance(data["load"], pd.DataFrame)
        assert len(data["load"]) > 0
        assert data["load"].index.tz is not None

    def test_fetch_price_data_germany(self, entsoe_api_key):
        """Test fetching price data for Germany"""
        source = EntsoeSource("DE", entsoe_api_key, type=["price"])
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")

        data = source.fetch(start, end)

        assert "price" in data
        assert isinstance(data["price"], pd.DataFrame)
        assert "price" in data["price"].columns

    def test_fetch_generation_data_france(self, entsoe_api_key):
        """Test fetching generation data for France"""
        source = EntsoeSource("FR", entsoe_api_key, type=["generation"])
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")

        data = source.fetch(start, end)

        assert "generation" in data
        assert isinstance(data["generation"], pd.DataFrame)
        assert len(data["generation"].columns) > 0

    def test_fetch_all_types(self, entsoe_api_key):
        """Test fetching all data types together"""
        source = EntsoeSource("PL", entsoe_api_key, type=["load", "generation", "price"])
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")

        data = source.fetch(start, end)

        assert "load" in data
        assert "generation" in data
        assert "price" in data
        assert all(isinstance(df, pd.DataFrame) for df in data.values())
