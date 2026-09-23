import pytest
import pandas as pd
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

        assert list(df.index.hour) == [2, 3]  # [start, end): 04:00 belongs to the next range

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


def _write(tmp_path, name: str, lines: list) -> str:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return str(path)


WHOLE_RANGE = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))


class TestCsvSourceTimezones:
    def test_mixed_offsets_from_a_local_aware_index(self, tmp_path):
        # What pandas writes for an Europe/Warsaw index across the fall-back switch.
        index = pd.date_range("2024-10-26 22:00", "2024-10-27 03:00", freq="h", tz="UTC").tz_convert("Europe/Warsaw")
        frame = pd.DataFrame({"price": range(len(index))}, index=index.rename("datetime"))
        path = tmp_path / "local_aware.csv"
        frame.to_csv(path)
        assert "+02:00" in path.read_text() and "+01:00" in path.read_text()

        df = CsvSource(str(path)).fetch(*WHOLE_RANGE)

        assert str(df.index.tz) == "UTC"
        assert df.index.is_unique
        assert list(df.index) == list(index.tz_convert("UTC"))
        assert list(df["price"]) == list(range(len(index)))

    def test_naive_local_times_with_timezone(self, tmp_path):
        # A chronological local-time export: 02:00 appears twice on the fall-back day.
        path = _write(
            tmp_path,
            "naive_local.csv",
            [
                "datetime,price",
                "2024-10-27 01:00,1",
                "2024-10-27 02:00,2",
                "2024-10-27 02:00,3",
                "2024-10-27 03:00,4",
            ],
        )
        df = CsvSource(path, timezone="Europe/Warsaw").fetch(*WHOLE_RANGE)

        expected = pd.to_datetime(
            ["2024-10-26 23:00", "2024-10-27 00:00", "2024-10-27 01:00", "2024-10-27 02:00"]
        ).tz_localize("UTC")
        assert list(df.index) == list(expected)
        assert list(df["price"]) == [1, 2, 3, 4]

    def test_naive_local_spring_forward(self, tmp_path):
        path = _write(
            tmp_path, "spring.csv", ["datetime,price", "2024-03-31 01:00,1", "2024-03-31 03:00,2", "2024-03-31 04:00,3"]
        )
        df = CsvSource(path, timezone="Europe/Warsaw").fetch(*WHOLE_RANGE)
        assert list(df.index.hour) == [0, 1, 2]  # 01:00 CET, 03:00 CEST, 04:00 CEST in UTC

    def test_nonexistent_local_time_raises(self, tmp_path):
        path = _write(tmp_path, "gap.csv", ["datetime,price", "2024-03-31 01:00,1", "2024-03-31 02:30,2"])
        with pytest.raises(ValueError, match="does not exist"):
            CsvSource(path, timezone="Europe/Warsaw").fetch(*WHOLE_RANGE)

    def test_ambiguous_local_time_that_cannot_be_inferred_raises(self, tmp_path):
        # A fall-back day with a single 02:00 row: which of the two 02:00s is it?
        path = _write(
            tmp_path,
            "one_two.csv",
            ["datetime,price", "2024-10-27 01:00,1", "2024-10-27 02:00,2", "2024-10-27 03:00,3"],
        )
        with pytest.raises(ValueError, match="ambiguous"):
            CsvSource(path, timezone="Europe/Warsaw").fetch(*WHOLE_RANGE)

    def test_timezone_is_ignored_for_aware_timestamps(self, tmp_path):
        path = _write(tmp_path, "aware.csv", ["datetime,price", "2024-01-01T00:00:00Z,1", "2024-01-01T01:00:00Z,2"])
        df = CsvSource(path, timezone="Europe/Warsaw").fetch(*WHOLE_RANGE)
        assert list(df.index.hour) == [0, 1]

    def test_mixed_naive_and_aware_raises(self, tmp_path):
        path = _write(tmp_path, "mixed.csv", ["datetime,price", "2024-01-01 00:00,1", "2024-01-01 01:00+01:00,2"])
        with pytest.raises(ValueError):
            CsvSource(path).fetch(*WHOLE_RANGE)

    def test_invalid_timezone_raises(self, sample_csv_file):
        with pytest.raises(ValueError, match="timezone"):
            CsvSource(str(sample_csv_file), timezone="Mars/Olympus_Mons")


class TestCsvSourceIntegrity:
    def test_duplicated_timestamps_raise(self, tmp_path):
        # A local-time export read as UTC (the default) has a repeated 02:00 on the fall-back day.
        path = _write(
            tmp_path, "dup.csv", ["datetime,price", "2024-10-27 01:00,1", "2024-10-27 02:00,2", "2024-10-27 02:00,3"]
        )
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            CsvSource(path).fetch(*WHOLE_RANGE)

    def test_duplicated_instants_with_different_offsets_raise(self, tmp_path):
        path = _write(tmp_path, "dup_aware.csv", ["datetime,price", "2024-01-01 01:00+01:00,1", "2024-01-01 00:00Z,2"])
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            CsvSource(path).fetch(*WHOLE_RANGE)

    def test_missing_timestamp_raises(self, tmp_path):
        path = _write(tmp_path, "nat.csv", ["datetime,price", "2024-01-01 00:00,1", ",2"])
        with pytest.raises(ValueError, match="no timestamp"):
            CsvSource(path).fetch(*WHOLE_RANGE)

    def test_end_is_exclusive(self, sample_csv_file):
        source = CsvSource(file_path=str(sample_csv_file))
        df = source.fetch(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC"))
        assert len(df) == 6
        assert df.index.max() == pd.Timestamp("2024-01-01 05:00", tz="UTC")


class TestCsvSourceCompression:
    def test_gzip_file(self, tmp_path):
        path = tmp_path / "data.csv.gz"
        pd.DataFrame({"datetime": ["2024-01-01 00:00", "2024-01-01 01:00"], "price": [1.0, 2.0]}).to_csv(
            path, index=False
        )
        df = CsvSource(str(path)).fetch(*WHOLE_RANGE)
        assert list(df["price"]) == [1.0, 2.0]

    def test_other_extension_still_rejected(self, tmp_path):
        path = tmp_path / "data.txt.gz"
        path.write_bytes(b"")
        with pytest.raises(ValueError, match="File must have .csv extension"):
            CsvSource(str(path))
