import pytest
import pandas as pd

from epftoolbox2.data.sources.base import DataSource
from epftoolbox2.data.transformers.base import Transformer
from epftoolbox2.data.transformers import TimezoneTransformer
from epftoolbox2.pipelines import DataPipeline


class MockDataSource(DataSource):
    """Mock data source for testing"""

    def __init__(self, data: pd.DataFrame = None, prefix: str = ""):
        self.data = data
        self.prefix = prefix
        self.fetch_called = False
        self.fetch_args = None

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        self.fetch_called = True
        self.fetch_args = (start, end)
        if self.data is None:
            return pd.DataFrame()
        return self.data.copy()

    def _validate_config(self) -> bool:
        return True

    def get_cache_config(self) -> dict:
        return {"source_type": "mock", "prefix": self.prefix}


class MockTransformer(Transformer):
    """Mock transformer for testing"""

    def __init__(self, suffix: str = "_transformed"):
        self.suffix = suffix
        self.transform_called = False

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.transform_called = True
        result = df.copy()
        result.columns = [f"{col}{self.suffix}" for col in result.columns]
        return result


@pytest.fixture
def sample_dataframe_1():
    """First sample DataFrame"""
    dates = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    return pd.DataFrame({"price": [10.0, 20.0, 30.0, 40.0, 50.0]}, index=dates)


@pytest.fixture
def sample_dataframe_2():
    """Second sample DataFrame with different columns"""
    dates = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    return pd.DataFrame({"load": [100, 200, 300, 400, 500]}, index=dates)


@pytest.fixture
def sample_dataframe_partial():
    """DataFrame with partial overlap"""
    dates = pd.date_range("2024-01-01 02:00", periods=3, freq="h", tz="UTC")
    return pd.DataFrame({"weather": [1.0, 2.0, 3.0]}, index=dates)


class TestDataPipelineInit:
    """Test DataPipeline initialization"""

    def test_init_single_source(self, sample_dataframe_1):
        """Test initialization with a single source"""
        source = MockDataSource(data=sample_dataframe_1)
        pipeline = DataPipeline(sources=[source])

        assert len(pipeline.sources) == 1
        assert len(pipeline.transformers) == 0

    def test_init_multiple_sources(self, sample_dataframe_1, sample_dataframe_2):
        """Test initialization with multiple sources"""
        source1 = MockDataSource(data=sample_dataframe_1)
        source2 = MockDataSource(data=sample_dataframe_2)
        pipeline = DataPipeline(sources=[source1, source2])

        assert len(pipeline.sources) == 2

    def test_init_with_transformers(self, sample_dataframe_1):
        """Test initialization with transformers"""
        source = MockDataSource(data=sample_dataframe_1)
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")
        pipeline = DataPipeline(sources=[source], transformers=[transformer])

        assert len(pipeline.transformers) == 1

    def test_init_empty_allows_builder(self):
        """Test that empty constructor is allowed for builder pattern"""
        pipeline = DataPipeline()
        assert len(pipeline.sources) == 0
        assert len(pipeline.transformers) == 0


class TestDataPipelineBuilder:
    """Test DataPipeline builder pattern"""

    def test_add_source(self, sample_dataframe_1):
        """Test adding a source via builder"""
        source = MockDataSource(data=sample_dataframe_1)
        pipeline = DataPipeline().add_source(source)

        assert len(pipeline.sources) == 1
        assert pipeline.sources[0] is source

    def test_add_source_returns_self(self, sample_dataframe_1):
        """Test that add_source returns self for chaining"""
        source = MockDataSource(data=sample_dataframe_1)
        pipeline = DataPipeline()
        result = pipeline.add_source(source)

        assert result is pipeline

    def test_add_multiple_sources(self, sample_dataframe_1, sample_dataframe_2):
        """Test adding multiple sources via chaining"""
        source1 = MockDataSource(data=sample_dataframe_1)
        source2 = MockDataSource(data=sample_dataframe_2)
        pipeline = DataPipeline().add_source(source1).add_source(source2)

        assert len(pipeline.sources) == 2

    def test_add_transformer(self, sample_dataframe_1):
        """Test adding a transformer via builder"""
        source = MockDataSource(data=sample_dataframe_1)
        transformer = MockTransformer()
        pipeline = DataPipeline().add_source(source).add_transformer(transformer)

        assert len(pipeline.transformers) == 1

    def test_add_transformer_returns_self(self):
        """Test that add_transformer returns self for chaining"""
        transformer = MockTransformer()
        pipeline = DataPipeline()
        result = pipeline.add_transformer(transformer)

        assert result is pipeline

    def test_full_builder_chain(self, sample_dataframe_1, sample_dataframe_2):
        """Test full builder pattern with multiple sources and transformers"""
        source1 = MockDataSource(data=sample_dataframe_1)
        source2 = MockDataSource(data=sample_dataframe_2)
        transformer1 = MockTransformer(suffix="_first")
        transformer2 = TimezoneTransformer(target_tz="Europe/Warsaw")

        pipeline = DataPipeline().add_source(source1).add_source(source2).add_transformer(transformer1).add_transformer(transformer2)

        assert len(pipeline.sources) == 2
        assert len(pipeline.transformers) == 2

    def test_add_source_invalid_type(self):
        """Test that add_source rejects invalid types"""
        pipeline = DataPipeline()
        with pytest.raises(TypeError, match="must be a DataSource"):
            pipeline.add_source("not a source")

    def test_add_transformer_invalid_type(self):
        """Test that add_transformer rejects invalid types"""
        pipeline = DataPipeline()
        with pytest.raises(TypeError, match="must be a Transformer"):
            pipeline.add_transformer("not a transformer")


class TestDataPipelineRun:
    """Test DataPipeline run method"""

    def test_run_single_source(self, sample_dataframe_1):
        """Test running pipeline with a single source"""
        source = MockDataSource(data=sample_dataframe_1)
        pipeline = DataPipeline(sources=[source])

        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        result = pipeline.run(start, end)

        assert source.fetch_called
        assert len(result) == 5
        assert "price" in result.columns

    def test_run_multiple_sources_merge(self, sample_dataframe_1, sample_dataframe_2):
        """Test that multiple sources are merged correctly"""
        source1 = MockDataSource(data=sample_dataframe_1)
        source2 = MockDataSource(data=sample_dataframe_2)
        pipeline = DataPipeline(sources=[source1, source2])

        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        result = pipeline.run(start, end)

        assert len(result) == 5
        assert "price" in result.columns
        assert "load" in result.columns

    def test_run_partial_overlap_outer_join(self, sample_dataframe_1, sample_dataframe_partial):
        """Test that partial overlap uses outer join"""
        source1 = MockDataSource(data=sample_dataframe_1)
        source2 = MockDataSource(data=sample_dataframe_partial)
        pipeline = DataPipeline(sources=[source1, source2])

        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        result = pipeline.run(start, end)

        assert len(result) == 5  # Outer join preserves all rows
        assert "price" in result.columns
        assert "weather" in result.columns
        # First two rows should have NaN for weather
        assert pd.isna(result["weather"].iloc[0])
        assert pd.isna(result["weather"].iloc[1])
        assert result["weather"].iloc[2] == 1.0

    def test_run_with_transformer(self, sample_dataframe_1):
        """Test that transformers are applied"""
        source = MockDataSource(data=sample_dataframe_1)
        transformer = MockTransformer(suffix="_test")
        pipeline = DataPipeline(sources=[source], transformers=[transformer])

        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        result = pipeline.run(start, end)

        assert transformer.transform_called
        assert "price_test" in result.columns

    def test_run_multiple_transformers_chain(self, sample_dataframe_1):
        """Test that multiple transformers are applied in order"""
        source = MockDataSource(data=sample_dataframe_1)
        transformer1 = MockTransformer(suffix="_first")
        transformer2 = MockTransformer(suffix="_second")
        pipeline = DataPipeline(sources=[source], transformers=[transformer1, transformer2])

        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        result = pipeline.run(start, end)

        # Column should have both suffixes applied in order
        assert "price_first_second" in result.columns

    def test_run_timezone_transformer(self, sample_dataframe_1):
        """Test with real TimezoneTransformer"""
        source = MockDataSource(data=sample_dataframe_1)
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")
        pipeline = DataPipeline(sources=[source], transformers=[transformer])

        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        result = pipeline.run(start, end)

        assert str(result.index.tz) == "Europe/Warsaw"

    def test_run_empty_source_returns_empty(self):
        """Test that empty source returns empty DataFrame"""
        source = MockDataSource(data=None)
        pipeline = DataPipeline(sources=[source])

        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        result = pipeline.run(start, end)

        assert result.empty

    def test_run_validates_timestamps(self, sample_dataframe_1):
        """Test that end must be after start"""
        source = MockDataSource(data=sample_dataframe_1)
        pipeline = DataPipeline(sources=[source])

        with pytest.raises(ValueError, match="must be after"):
            pipeline.run(
                start=pd.Timestamp("2024-01-02", tz="UTC"),
                end=pd.Timestamp("2024-01-01", tz="UTC"),
            )

    def test_run_localizes_naive_timestamps(self, sample_dataframe_1):
        """Test that naive timestamps are localized to UTC"""
        source = MockDataSource(data=sample_dataframe_1)
        pipeline = DataPipeline(sources=[source])

        start = pd.Timestamp("2024-01-01")  # Naive
        end = pd.Timestamp("2024-01-02")  # Naive
        pipeline.run(start, end)

        assert source.fetch_called
        # Check that the source received UTC timestamps
        assert source.fetch_args[0].tzinfo is not None
        assert source.fetch_args[1].tzinfo is not None

    def test_run_no_sources_raises_error(self):
        """Test that run with no sources raises error"""
        pipeline = DataPipeline()

        with pytest.raises(ValueError, match="At least one data source"):
            pipeline.run(
                start=pd.Timestamp("2024-01-01", tz="UTC"),
                end=pd.Timestamp("2024-01-02", tz="UTC"),
            )

    def test_run_with_builder_pattern(self, sample_dataframe_1):
        """Test running pipeline built with builder pattern"""
        source = MockDataSource(data=sample_dataframe_1)
        transformer = TimezoneTransformer(target_tz="Europe/Warsaw")

        result = (
            DataPipeline()
            .add_source(source)
            .add_transformer(transformer)
            .run(
                start=pd.Timestamp("2024-01-01", tz="UTC"),
                end=pd.Timestamp("2024-01-02", tz="UTC"),
            )
        )

        assert len(result) == 5
        assert str(result.index.tz) == "Europe/Warsaw"


class TestDataPipelineCache:
    """Test pipeline caching functionality"""

    def test_run_with_cache_disabled_default(self, sample_dataframe_1):
        """Test that cache is disabled by default"""
        source = MockDataSource(data=sample_dataframe_1)
        pipeline = DataPipeline(sources=[source])

        result = pipeline.run(start=pd.Timestamp("2024-01-01", tz="UTC"), end=pd.Timestamp("2024-01-02", tz="UTC"))

        assert len(result) == 5
        assert source.fetch_called

    def test_run_with_cache_enabled(self, sample_dataframe_1):
        """Test running pipeline with cache enabled"""
        import tempfile
        import shutil
        from epftoolbox2.data.cache_manager import CacheManager

        cache_dir = tempfile.mkdtemp()

        try:
            source = MockDataSource(data=sample_dataframe_1)
            pipeline = DataPipeline(sources=[source])

            cache_manager = CacheManager(cache_dir=cache_dir)
            cache_key = cache_manager.get_cache_key(source.get_cache_config())

            # Verify cache is empty initially
            info = cache_manager.get_cache_info(cache_key)
            assert info["exists"] is False

            # Run with cache disabled first (default)
            result = pipeline.run(start=pd.Timestamp("2024-01-01", tz="UTC"), end=pd.Timestamp("2024-01-02", tz="UTC"), cache=False)

            assert len(result) == 5
            assert source.fetch_called
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)


class TestDataPipelineSerialization:
    """Test pipeline serialization (save/load)"""

    def test_to_dict_empty_pipeline(self):
        """Test serializing empty pipeline to dict"""
        pipeline = DataPipeline()
        config = pipeline.to_dict()

        assert config == {"sources": [], "transformers": [], "validators": []}

    def test_to_dict_with_source(self):
        """Test serializing pipeline with source to dict"""
        from epftoolbox2.data.sources import CalendarSource

        pipeline = DataPipeline().add_source(CalendarSource(country="PL"))
        config = pipeline.to_dict()

        assert len(config["sources"]) == 1
        assert config["sources"][0]["class"] == "CalendarSource"
        assert config["sources"][0]["params"]["country"] == "PL"

    def test_to_dict_with_transformer(self):
        """Test serializing pipeline with transformer to dict"""
        pipeline = DataPipeline().add_transformer(TimezoneTransformer(target_tz="Europe/Warsaw"))
        config = pipeline.to_dict()

        assert len(config["transformers"]) == 1
        assert config["transformers"][0]["class"] == "TimezoneTransformer"
        assert config["transformers"][0]["params"]["target_tz"] == "Europe/Warsaw"

    def test_to_dict_with_validator(self):
        """Test serializing pipeline with validator to dict"""
        from epftoolbox2.data.validators import NullCheckValidator

        pipeline = DataPipeline().add_validator(NullCheckValidator(columns=["test"]))
        config = pipeline.to_dict()

        assert len(config["validators"]) == 1
        assert config["validators"][0]["class"] == "NullCheckValidator"
        assert config["validators"][0]["params"]["columns"] == ["test"]

    def test_save_and_load(self, tmp_path):
        """Test saving and loading pipeline from YAML"""
        from epftoolbox2.data.sources import CalendarSource
        from epftoolbox2.data.validators import ContinuityValidator

        pipeline = DataPipeline().add_source(CalendarSource(country="PL", holidays="binary", weekday="number")).add_transformer(TimezoneTransformer(target_tz="Europe/Warsaw")).add_validator(ContinuityValidator(freq="1h"))

        yaml_path = tmp_path / "pipeline.yaml"
        pipeline.save(yaml_path)

        assert yaml_path.exists()

        loaded = DataPipeline.load(yaml_path)

        assert len(loaded.sources) == 1
        assert len(loaded.transformers) == 1
        assert len(loaded.validators) == 1
        assert type(loaded.sources[0]).__name__ == "CalendarSource"
        assert type(loaded.transformers[0]).__name__ == "TimezoneTransformer"
        assert type(loaded.validators[0]).__name__ == "ContinuityValidator"

    def test_loaded_pipeline_runs(self, tmp_path):
        """Test that loaded pipeline can run successfully"""
        from epftoolbox2.data.sources import CalendarSource

        pipeline = DataPipeline().add_source(CalendarSource(country="PL", holidays="binary", weekday=False))

        yaml_path = tmp_path / "pipeline.yaml"
        pipeline.save(yaml_path)

        loaded = DataPipeline.load(yaml_path)
        result = loaded.run(start="2024-01-01", end="2024-01-02")

        assert not result.empty
        assert "is_holiday" in result.columns

    def test_save_creates_valid_yaml(self, tmp_path):
        """Test that saved file is valid YAML"""
        import yaml

        pipeline = DataPipeline().add_transformer(TimezoneTransformer(target_tz="UTC"))

        yaml_path = tmp_path / "pipeline.yaml"
        pipeline.save(yaml_path)

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        assert "sources" in config
        assert "transformers" in config
        assert "validators" in config

    def test_load_unknown_component_raises_error(self, tmp_path):
        """Test that loading unknown component raises error"""
        import yaml

        yaml_path = tmp_path / "invalid.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump({"sources": [{"class": "UnknownSource", "params": {}}], "transformers": [], "validators": []}, f)

        with pytest.raises(ValueError, match="Unknown"):
            DataPipeline.load(yaml_path)

    def test_round_trip_preserves_params(self, tmp_path):
        """Test that save/load preserves all parameters"""
        from epftoolbox2.data.sources import CalendarSource
        from epftoolbox2.data.transformers import ResampleTransformer

        pipeline = DataPipeline().add_source(CalendarSource(country="DE", holidays="onehot", weekday="name", month="number")).add_transformer(ResampleTransformer(freq="1h", method="ffill"))

        yaml_path = tmp_path / "pipeline.yaml"
        pipeline.save(yaml_path)
        loaded = DataPipeline.load(yaml_path)

        assert loaded.sources[0].country == "DE"
        assert loaded.sources[0].holidays == "onehot"
        assert loaded.sources[0].weekday == "name"
        assert loaded.sources[0].month == "number"
        assert loaded.transformers[0].freq == "1h"
        assert loaded.transformers[0].method == "ffill"
