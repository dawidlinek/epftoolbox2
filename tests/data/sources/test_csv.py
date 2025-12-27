import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from epftoolbox2.data.sources.csv import CsvSource


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a sample CSV file for testing"""
    csv_content = """datetime,price,load,temperature
2024-01-01 00:00:00,50.5,1000,5.2
2024-01-01 01:00:00,51.2,1020,5.0
2024-01-01 02:00:00,49.8,980,4.8
2024-01-01 03:00:00,48.5,950,4.5
2024-01-01 04:00:00,47.0,920,4.2
2024-01-01 05:00:00,52.0,1050,5.5
2024-01-02 00:00:00,55.0,1100,6.0
2024-01-02 01:00:00,53.5,1080,5.8
"""
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text(csv_content)
    return csv_file


@pytest.fixture
def sample_csv_semicolon(tmp_path):
    """Create a sample CSV file with semicolon separator"""
    csv_content = """datetime;price;load
2024-01-01 00:00:00;50.5;1000
2024-01-01 01:00:00;51.2;1020
"""
    csv_file = tmp_path / "test_semicolon.csv"
    csv_file.write_text(csv_content)
    return csv_file


@pytest.fixture
def sample_csv_custom_datetime(tmp_path):
    """Create a sample CSV file with custom datetime column name"""
    csv_content = """timestamp,value
2024-01-01 00:00:00,100
2024-01-01 01:00:00,200
"""
    csv_file = tmp_path / "test_custom_dt.csv"
    csv_file.write_text(csv_content)
    return csv_file


class TestCsvSourceInit:
    """Test CsvSource initialization"""

    def test_init_valid_params(self, sample_csv_file):
        """Test initialization with valid parameters"""
        source = CsvSource(file_path=str(sample_csv_file))
        assert source.file_path == sample_csv_file
        assert source.datetime_column == "datetime"
        assert source.columns is None
        assert source.prefix == ""
        assert source.separator == ","

    def test_init_with_all_params(self, sample_csv_file):
        """Test initialization with all parameters"""
        source = CsvSource(
            file_path=str(sample_csv_file),
            datetime_column="datetime",
            columns=["price", "load"],
            prefix="test",
            datetime_format="%Y-%m-%d %H:%M:%S",
            separator=",",
        )
        assert source.columns == ["price", "load"]
        assert source.prefix == "test"
        assert source.datetime_format == "%Y-%m-%d %H:%M:%S"

    def test_init_file_not_found(self):
        """Test initialization with non-existent file"""
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            CsvSource(file_path="nonexistent.csv")

    def test_init_invalid_extension(self, tmp_path):
        """Test initialization with invalid file extension"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("data")
        with pytest.raises(ValueError, match="File must have .csv extension"):
            CsvSource(file_path=str(txt_file))

    def test_init_empty_datetime_column(self, sample_csv_file):
        """Test initialization with empty datetime column"""
        with pytest.raises(ValueError, match="datetime_column cannot be empty"):
            CsvSource(file_path=str(sample_csv_file), datetime_column="")


class TestCsvSourceLoadData:
    """Test CSV data loading"""

    def test_load_data_basic(self, sample_csv_file):
        """Test basic data loading"""
        source = CsvSource(file_path=str(sample_csv_file))
        df = source._load_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 8
        assert "price" in df.columns
        assert "load" in df.columns
        assert "temperature" in df.columns

    def test_load_data_datetime_index(self, sample_csv_file):
        """Test that datetime is set as index"""
        source = CsvSource(file_path=str(sample_csv_file))
        df = source._load_data()

        assert df.index.name == "datetime"
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"

    def test_load_data_selected_columns(self, sample_csv_file):
        """Test loading only selected columns"""
        source = CsvSource(file_path=str(sample_csv_file), columns=["price", "load"])
        df = source._load_data()

        assert list(df.columns) == ["price", "load"]
        assert "temperature" not in df.columns

    def test_load_data_missing_column(self, sample_csv_file):
        """Test error when requested column doesn't exist"""
        source = CsvSource(file_path=str(sample_csv_file), columns=["price", "nonexistent"])
        with pytest.raises(ValueError, match="Columns not found in CSV"):
            source._load_data()

    def test_load_data_missing_datetime_column(self, tmp_path):
        """Test error when datetime column doesn't exist"""
        csv_file = tmp_path / "no_datetime.csv"
        csv_file.write_text("col1,col2\n1,2\n")

        source = CsvSource(file_path=str(csv_file), datetime_column="datetime")
        with pytest.raises(ValueError, match="Datetime column 'datetime' not found"):
            source._load_data()

    def test_load_data_custom_datetime_column(self, sample_csv_custom_datetime):
        """Test loading with custom datetime column name"""
        source = CsvSource(file_path=str(sample_csv_custom_datetime), datetime_column="timestamp")
        df = source._load_data()

        assert df.index.name == "timestamp"
        assert len(df) == 2

    def test_load_data_semicolon_separator(self, sample_csv_semicolon):
        """Test loading CSV with semicolon separator"""
        source = CsvSource(file_path=str(sample_csv_semicolon), separator=";")
        df = source._load_data()

        assert len(df) == 2
        assert "price" in df.columns


class TestCsvSourceFetch:
    """Test fetch method"""

    def test_fetch_all_data(self, sample_csv_file):
        """Test fetching all data in range"""
        source = CsvSource(file_path=str(sample_csv_file))
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-03", tz="UTC")

        df = source.fetch(start, end)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 8

    def test_fetch_partial_data(self, sample_csv_file):
        """Test fetching partial data range"""
        source = CsvSource(file_path=str(sample_csv_file))
        start = pd.Timestamp("2024-01-01 02:00:00", tz="UTC")
        end = pd.Timestamp("2024-01-01 04:00:00", tz="UTC")

        df = source.fetch(start, end)

        assert len(df) == 3  # 02:00, 03:00, 04:00

    def test_fetch_with_prefix(self, sample_csv_file):
        """Test fetching with column prefix"""
        source = CsvSource(file_path=str(sample_csv_file), prefix="csv")
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")

        df = source.fetch(start, end)

        assert "csv_price" in df.columns
        assert "csv_load" in df.columns
        assert "csv_temperature" in df.columns

    def test_fetch_validates_timestamps(self, sample_csv_file):
        """Test that fetch validates end > start"""
        source = CsvSource(file_path=str(sample_csv_file))
        start = pd.Timestamp("2024-01-02", tz="UTC")
        end = pd.Timestamp("2024-01-01", tz="UTC")

        with pytest.raises(ValueError, match="End timestamp.*must be after start timestamp"):
            source.fetch(start, end)

    def test_fetch_localizes_naive_timestamps(self, sample_csv_file):
        """Test that naive timestamps are localized to UTC"""
        source = CsvSource(file_path=str(sample_csv_file))
        start = pd.Timestamp("2024-01-01")  # naive
        end = pd.Timestamp("2024-01-02")  # naive

        df = source.fetch(start, end)

        # Should work and return data
        assert len(df) > 0

    def test_fetch_converts_timezone(self, sample_csv_file):
        """Test that timestamps are converted to UTC"""
        source = CsvSource(file_path=str(sample_csv_file))
        start = pd.Timestamp("2024-01-01", tz="Europe/Warsaw")
        end = pd.Timestamp("2024-01-02", tz="Europe/Warsaw")

        df = source.fetch(start, end)

        assert len(df) > 0

    def test_fetch_empty_range(self, sample_csv_file):
        """Test fetching data outside available range"""
        source = CsvSource(file_path=str(sample_csv_file))
        start = pd.Timestamp("2025-01-01", tz="UTC")
        end = pd.Timestamp("2025-01-02", tz="UTC")

        df = source.fetch(start, end)

        assert len(df) == 0

    def test_fetch_returns_copy(self, sample_csv_file):
        """Test that fetch returns a copy of the data"""
        source = CsvSource(file_path=str(sample_csv_file))
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")

        df1 = source.fetch(start, end)
        df1["price"] = 0  # Modify returned data

        df2 = source.fetch(start, end)

        # Original data should be unchanged
        assert df2["price"].iloc[0] != 0


class TestCsvSourceEdgeCases:
    """Test edge cases and error handling"""

    def test_data_sorted_by_index(self, tmp_path):
        """Test that data is sorted by datetime index"""
        csv_content = """datetime,value
2024-01-01 02:00:00,2
2024-01-01 00:00:00,0
2024-01-01 01:00:00,1
"""
        csv_file = tmp_path / "unsorted.csv"
        csv_file.write_text(csv_content)

        source = CsvSource(file_path=str(csv_file))
        df = source._load_data()

        # Check data is sorted
        assert df.iloc[0]["value"] == 0
        assert df.iloc[1]["value"] == 1
        assert df.iloc[2]["value"] == 2

    def test_custom_datetime_format(self, tmp_path):
        """Test parsing with custom datetime format"""
        csv_content = """datetime,value
01/01/2024 00:00,100
01/01/2024 01:00,200
"""
        csv_file = tmp_path / "custom_format.csv"
        csv_file.write_text(csv_content)

        source = CsvSource(file_path=str(csv_file), datetime_format="%d/%m/%Y %H:%M")
        df = source._load_data()

        assert len(df) == 2
        assert df.index[0].day == 1
        assert df.index[0].month == 1
        assert df.index[0].year == 2024

    def test_numeric_data_types(self, sample_csv_file):
        """Test that numeric columns are properly typed"""
        source = CsvSource(file_path=str(sample_csv_file))
        df = source._load_data()

        assert df["price"].dtype in [float, "float64"]
        assert df["load"].dtype in [int, "int64"]
