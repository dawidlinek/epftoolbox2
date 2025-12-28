import pytest
import pandas as pd
import tempfile
import shutil
from epftoolbox2.data.cache_manager import CacheManager


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def cache_manager(temp_cache_dir):
    """Create a CacheManager instance with temporary directory"""
    return CacheManager(cache_dir=temp_cache_dir)


@pytest.fixture
def sample_config():
    """Sample source configuration"""
    return {"source_type": "entsoe", "area_code": "10YPL-AREA-----S", "types": ["load", "price"]}


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing"""
    dates = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    return pd.DataFrame({"load": range(24), "price": range(100, 124)}, index=dates)


class TestCacheKeyGeneration:
    """Test cache key generation"""

    def test_same_config_same_key(self, cache_manager):
        """Same configuration should generate the same key"""
        config1 = {"source_type": "entsoe", "area_code": "PL", "types": ["load"]}
        config2 = {"source_type": "entsoe", "area_code": "PL", "types": ["load"]}

        key1 = cache_manager.get_cache_key(config1)
        key2 = cache_manager.get_cache_key(config2)

        assert key1 == key2

    def test_different_config_different_key(self, cache_manager):
        """Different configuration should generate different keys"""
        config1 = {"source_type": "entsoe", "area_code": "PL", "types": ["load"]}
        config2 = {"source_type": "entsoe", "area_code": "DE", "types": ["load"]}

        key1 = cache_manager.get_cache_key(config1)
        key2 = cache_manager.get_cache_key(config2)

        assert key1 != key2

    def test_key_deterministic(self, cache_manager):
        """Key generation should be deterministic"""
        config = {"source_type": "entsoe", "area_code": "PL", "types": ["load", "price"]}

        key1 = cache_manager.get_cache_key(config)
        key2 = cache_manager.get_cache_key(config)
        key3 = cache_manager.get_cache_key(config)

        assert key1 == key2 == key3


class TestCacheWriteRead:
    """Test cache write and read operations"""

    def test_write_and_read_cache(self, cache_manager, sample_config, sample_dataframe):
        """Write data to cache and read it back"""
        cache_key = cache_manager.get_cache_key(sample_config)
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")

        # Write to cache
        cache_manager.write_cache(cache_key, sample_dataframe, start, end, sample_config)

        # Read from cache
        result = cache_manager.read_cached_data(cache_key, start, end)

        assert result is not None
        assert len(result) == len(sample_dataframe)
        # Parquet doesn't preserve index frequency, so we skip that check
        pd.testing.assert_frame_equal(result, sample_dataframe, check_freq=False)

    def test_read_nonexistent_cache(self, cache_manager, sample_config):
        """Reading non-existent cache should return None"""
        cache_key = cache_manager.get_cache_key(sample_config)
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")

        result = cache_manager.read_cached_data(cache_key, start, end)

        assert result is None

    def test_write_empty_dataframe(self, cache_manager, sample_config):
        """Writing empty DataFrame should be skipped"""
        cache_key = cache_manager.get_cache_key(sample_config)
        empty_df = pd.DataFrame()
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")

        # Should not raise error
        cache_manager.write_cache(cache_key, empty_df, start, end, sample_config)

        # Should still be None
        result = cache_manager.read_cached_data(cache_key, start, end)
        assert result is None


class TestGapDetection:
    """Test missing range detection"""

    def test_find_missing_no_cache(self, cache_manager, sample_config):
        """With no cache, entire range should be missing"""
        cache_key = cache_manager.get_cache_key(sample_config)
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-31", tz="UTC")

        missing = cache_manager.find_missing_ranges(cache_key, start, end)

        assert len(missing) == 1
        assert missing[0] == (start, end)

    def test_find_missing_full_hit(self, cache_manager, sample_config, sample_dataframe):
        """With full cache coverage, no ranges should be missing"""
        cache_key = cache_manager.get_cache_key(sample_config)
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")

        # Cache the data
        cache_manager.write_cache(cache_key, sample_dataframe, start, end, sample_config)

        # Check missing ranges
        missing = cache_manager.find_missing_ranges(cache_key, start, end)

        assert len(missing) == 0

    def test_find_missing_partial_hit_before(self, cache_manager, sample_config, sample_dataframe):
        """Requesting range before cached data"""
        cache_key = cache_manager.get_cache_key(sample_config)

        # Cache Jan 15-31
        cache_start = pd.Timestamp("2024-01-15", tz="UTC")
        cache_end = pd.Timestamp("2024-02-01", tz="UTC")
        cache_manager.write_cache(cache_key, sample_dataframe, cache_start, cache_end, sample_config)

        # Request Jan 1-31
        req_start = pd.Timestamp("2024-01-01", tz="UTC")
        req_end = pd.Timestamp("2024-02-01", tz="UTC")

        missing = cache_manager.find_missing_ranges(cache_key, req_start, req_end)

        assert len(missing) == 1
        assert missing[0] == (req_start, cache_start)

    def test_find_missing_partial_hit_after(self, cache_manager, sample_config, sample_dataframe):
        """Requesting range after cached data"""
        cache_key = cache_manager.get_cache_key(sample_config)

        # Cache Jan 1-15
        cache_start = pd.Timestamp("2024-01-01", tz="UTC")
        cache_end = pd.Timestamp("2024-01-15", tz="UTC")
        cache_manager.write_cache(cache_key, sample_dataframe, cache_start, cache_end, sample_config)

        # Request Jan 1-31
        req_start = pd.Timestamp("2024-01-01", tz="UTC")
        req_end = pd.Timestamp("2024-02-01", tz="UTC")

        missing = cache_manager.find_missing_ranges(cache_key, req_start, req_end)

        assert len(missing) == 1
        assert missing[0] == (cache_end, req_end)

    def test_find_missing_gap_in_middle(self, cache_manager, sample_config):
        """Gap in the middle of cached ranges"""
        cache_key = cache_manager.get_cache_key(sample_config)

        # Create sample data for different periods
        dates1 = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        df1 = pd.DataFrame({"load": range(24)}, index=dates1)

        dates2 = pd.date_range("2024-01-15", periods=24, freq="h", tz="UTC")
        df2 = pd.DataFrame({"load": range(24)}, index=dates2)

        # Cache Jan 1-2 and Jan 15-16
        cache_manager.write_cache(cache_key, df1, pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC"), sample_config)
        cache_manager.write_cache(cache_key, df2, pd.Timestamp("2024-01-15", tz="UTC"), pd.Timestamp("2024-01-16", tz="UTC"), sample_config)

        # Request Jan 1-20
        req_start = pd.Timestamp("2024-01-01", tz="UTC")
        req_end = pd.Timestamp("2024-01-20", tz="UTC")

        missing = cache_manager.find_missing_ranges(cache_key, req_start, req_end)

        # Should have gap between Jan 2 and Jan 15, and after Jan 16
        assert len(missing) == 2
        assert missing[0] == (pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-15", tz="UTC"))
        assert missing[1] == (pd.Timestamp("2024-01-16", tz="UTC"), req_end)


class TestCacheInfo:
    """Test cache information retrieval"""

    def test_cache_info_no_cache(self, cache_manager, sample_config):
        """Cache info for non-existent cache"""
        cache_key = cache_manager.get_cache_key(sample_config)
        info = cache_manager.get_cache_info(cache_key)

        assert info["exists"] is False
        assert info["num_chunks"] == 0

    def test_cache_info_with_data(self, cache_manager, sample_config, sample_dataframe):
        """Cache info for existing cache"""
        cache_key = cache_manager.get_cache_key(sample_config)
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")

        cache_manager.write_cache(cache_key, sample_dataframe, start, end, sample_config)

        info = cache_manager.get_cache_info(cache_key)

        assert info["exists"] is True
        assert info["num_chunks"] == 1
        assert len(info["date_ranges"]) == 1
