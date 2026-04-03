import pytest
import yaml
from pathlib import Path

from epftoolbox2.pipelines import DataPipeline, ModelPipeline, Workflow
from epftoolbox2.data.sources import CalendarSource, CsvSource
from epftoolbox2.data.transformers import ResampleTransformer, LagTransformer, TimezoneTransformer
from epftoolbox2.data.validators import NullCheckValidator, ContinuityValidator
from epftoolbox2.models import OLSModel, LassoCVModel
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.exporters import TerminalExporter, ExcelExporter


# ---------------------------------------------------------------------------
# DataPipeline serialization
# ---------------------------------------------------------------------------

class TestDataPipelineSerialization:

    def test_save_creates_yaml_file(self, tmp_path):
        pipeline = DataPipeline().add_source(CalendarSource(country="PL"))
        path = tmp_path / "dp.yaml"
        pipeline.save(path)
        assert path.exists()

    def test_yaml_structure(self, tmp_path):
        pipeline = (
            DataPipeline()
            .add_source(CalendarSource(country="PL", holidays="binary"))
            .add_transformer(ResampleTransformer(freq="1h"))
            .add_validator(NullCheckValidator(columns=["is_holiday"]))
        )
        path = tmp_path / "dp.yaml"
        pipeline.save(path)
        config = yaml.safe_load(path.read_text())
        assert "sources" in config
        assert "transformers" in config
        assert "validators" in config
        assert config["sources"][0]["class"] == "CalendarSource"
        assert config["transformers"][0]["class"] == "ResampleTransformer"
        assert config["validators"][0]["class"] == "NullCheckValidator"

    def test_round_trip_calendar_source(self, tmp_path):
        original = DataPipeline().add_source(
            CalendarSource(country="PL", holidays="binary", weekday="onehot", daylight=True)
        )
        path = tmp_path / "dp.yaml"
        original.save(path)
        loaded = DataPipeline.load(path)

        src = loaded.sources[0]
        assert isinstance(src, CalendarSource)
        assert src.country == "PL"
        assert src.holidays == "binary"
        assert src.weekday == "onehot"
        assert src.daylight is True

    def test_round_trip_resample_transformer(self, tmp_path):
        original = DataPipeline().add_transformer(ResampleTransformer(freq="15min", method="ffill"))
        path = tmp_path / "dp.yaml"
        original.save(path)
        loaded = DataPipeline.load(path)

        t = loaded.transformers[0]
        assert isinstance(t, ResampleTransformer)
        assert t.freq == "15min"
        assert t.method == "ffill"

    def test_round_trip_lag_transformer_list(self, tmp_path):
        original = DataPipeline().add_transformer(
            LagTransformer(columns=["price", "load"], lags=[1, 2, 7], freq="day")
        )
        path = tmp_path / "dp.yaml"
        original.save(path)
        loaded = DataPipeline.load(path)

        t = loaded.transformers[0]
        assert isinstance(t, LagTransformer)
        assert t.columns == ["price", "load"]
        assert t.lags == [1, 2, 7]
        assert t.freq == "day"

    def test_round_trip_lag_transformer_range(self, tmp_path):
        # range(-7, 1) must serialize correctly as a list
        original = DataPipeline().add_transformer(
            LagTransformer(columns=["price"], lags=range(-7, 1), freq="day")
        )
        path = tmp_path / "dp.yaml"
        original.save(path)
        loaded = DataPipeline.load(path)

        t = loaded.transformers[0]
        assert t.lags == list(range(-7, 1))

    def test_round_trip_null_check_validator(self, tmp_path):
        original = DataPipeline().add_validator(NullCheckValidator(columns=["price"], allow_nulls=True))
        path = tmp_path / "dp.yaml"
        original.save(path)
        loaded = DataPipeline.load(path)

        v = loaded.validators[0]
        assert isinstance(v, NullCheckValidator)
        assert v.columns == ["price"]
        assert v.allow_nulls is True

    def test_round_trip_multiple_components(self, tmp_path):
        original = (
            DataPipeline()
            .add_source(CalendarSource(country="PL"))
            .add_source(CalendarSource(country="DE", prefix="de"))
            .add_transformer(ResampleTransformer(freq="1h"))
            .add_transformer(LagTransformer(columns=["price"], lags=[1, 7], freq="day"))
            .add_validator(NullCheckValidator(columns=["price"]))
            .add_validator(ContinuityValidator(freq="1h"))
        )
        path = tmp_path / "dp.yaml"
        original.save(path)
        loaded = DataPipeline.load(path)

        assert len(loaded.sources) == 2
        assert len(loaded.transformers) == 2
        assert len(loaded.validators) == 2

    def test_csv_source_path_serializes_as_string(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("datetime,price\n2024-01-01,100\n")
        original = DataPipeline().add_source(CsvSource(file_path=str(csv_file)))
        path = tmp_path / "dp.yaml"
        original.save(path)
        config = yaml.safe_load(path.read_text())
        stored_path = config["sources"][0]["params"]["file_path"]
        assert isinstance(stored_path, str)

    def test_unknown_component_raises_on_load(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("sources:\n- class: NonExistentSource\n  params: {}\n")
        with pytest.raises(ValueError, match="Unknown source"):
            DataPipeline.load(path)


# ---------------------------------------------------------------------------
# ModelPipeline serialization
# ---------------------------------------------------------------------------

class TestModelPipelineSerialization:

    def test_save_creates_yaml_file(self, tmp_path):
        pipeline = ModelPipeline().add_model(OLSModel(predictors=["price"], name="OLS"))
        path = tmp_path / "mp.yaml"
        pipeline.save(path)
        assert path.exists()

    def test_yaml_structure(self, tmp_path):
        pipeline = (
            ModelPipeline()
            .add_model(OLSModel(predictors=["price"], name="OLS"))
            .add_evaluator(MAEEvaluator())
            .add_exporter(TerminalExporter())
        )
        path = tmp_path / "mp.yaml"
        pipeline.save(path)
        config = yaml.safe_load(path.read_text())
        assert "models" in config
        assert "evaluators" in config
        assert "exporters" in config
        assert config["models"][0]["class"] == "OLSModel"
        assert config["evaluators"][0]["class"] == "MAEEvaluator"
        assert config["exporters"][0]["class"] == "TerminalExporter"

    def test_round_trip_ols_model(self, tmp_path):
        predictors = ["load_actual", "price_d-1", "is_monday_d+{horizon}"]
        original = ModelPipeline().add_model(
            OLSModel(predictors=predictors, training_window=180, name="MyOLS")
        )
        path = tmp_path / "mp.yaml"
        original.save(path)
        loaded = ModelPipeline.load(path)

        m = loaded.models[0]
        assert isinstance(m, OLSModel)
        assert m.predictors == predictors
        assert m.training_window == 180
        assert m.name == "MyOLS"

    def test_round_trip_lasso_model(self, tmp_path):
        original = ModelPipeline().add_model(
            LassoCVModel(predictors=["price"], training_window=365, cv=10, max_iter=5000, name="Lasso")
        )
        path = tmp_path / "mp.yaml"
        original.save(path)
        loaded = ModelPipeline.load(path)

        m = loaded.models[0]
        assert isinstance(m, LassoCVModel)
        assert m.cv == 10
        assert m.max_iter == 5000

    def test_round_trip_mae_evaluator(self, tmp_path):
        original = ModelPipeline().add_evaluator(MAEEvaluator())
        path = tmp_path / "mp.yaml"
        original.save(path)
        loaded = ModelPipeline.load(path)
        assert isinstance(loaded.evaluators[0], MAEEvaluator)

    def test_round_trip_terminal_exporter(self, tmp_path):
        original = ModelPipeline().add_exporter(TerminalExporter(show=["summary", "hour"]))
        path = tmp_path / "mp.yaml"
        original.save(path)
        loaded = ModelPipeline.load(path)

        e = loaded.exporters[0]
        assert isinstance(e, TerminalExporter)
        assert e.show == ["summary", "hour"]

    def test_round_trip_excel_exporter_path_as_string(self, tmp_path):
        original = ModelPipeline().add_exporter(ExcelExporter(path="results/out.xlsx"))
        path = tmp_path / "mp.yaml"
        original.save(path)

        config = yaml.safe_load(path.read_text())
        assert isinstance(config["exporters"][0]["params"]["path"], str)

        loaded = ModelPipeline.load(path)
        assert loaded.exporters[0].path == Path("results/out.xlsx")

    def test_callable_predictor_raises_on_save(self, tmp_path):
        pipeline = ModelPipeline().add_model(
            OLSModel(predictors=[lambda h: f"col_d+{h}"], name="Bad")
        )
        with pytest.raises(ValueError, match="callable"):
            pipeline.save(tmp_path / "mp.yaml")

    def test_shared_predictor_list_yaml_alias_loads_correctly(self, tmp_path):
        # Same list object → PyYAML emits alias; both models must load independently
        predictors = ["load_actual", "price_d-1"]
        original = (
            ModelPipeline()
            .add_model(OLSModel(predictors=predictors, name="OLS"))
            .add_model(LassoCVModel(predictors=predictors, name="Lasso"))
        )
        path = tmp_path / "mp.yaml"
        original.save(path)
        loaded = ModelPipeline.load(path)

        assert loaded.models[0].predictors == predictors
        assert loaded.models[1].predictors == predictors

    def test_unknown_model_raises_on_load(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("models:\n- class: NonExistentModel\n  params: {predictors: [], name: x}\n")
        with pytest.raises(ValueError, match="Unknown model"):
            ModelPipeline.load(path)


# ---------------------------------------------------------------------------
# Workflow serialization
# ---------------------------------------------------------------------------

class TestWorkflowSerialization:

    @pytest.fixture
    def pipeline_yamls(self, tmp_path):
        dp_path = tmp_path / "dp.yaml"
        mp_path = tmp_path / "mp.yaml"
        DataPipeline().add_source(CalendarSource(country="PL")).save(dp_path)
        (
            ModelPipeline()
            .add_model(OLSModel(predictors=["is_holiday"], name="OLS"))
            .add_model(LassoCVModel(predictors=["is_holiday"], name="Lasso"))
            .add_evaluator(MAEEvaluator())
            .add_exporter(TerminalExporter())
        ).save(mp_path)
        return dp_path, mp_path

    def test_save_creates_yaml_file(self, tmp_path, pipeline_yamls):
        dp_path, mp_path = pipeline_yamls
        wf = Workflow(
            data_pipeline=dp_path,
            model_pipeline=mp_path,
            data_start="2023-01-01",
            data_end="2024-01-01",
            model_test_start="2023-10-01",
            model_test_end="2024-01-01",
        )
        out = tmp_path / "exp.yaml"
        wf.save(out)
        assert out.exists()

    def test_round_trip_basic_params(self, tmp_path, pipeline_yamls):
        dp_path, mp_path = pipeline_yamls
        original = Workflow(
            data_pipeline=str(dp_path),
            model_pipeline=str(mp_path),
            data_start="2023-01-01",
            data_end="2024-01-01",
            data_cache=True,
            model_test_start="2023-10-01",
            model_test_end="2024-01-01",
            model_target="price",
            model_horizon=5,
            max_processes=4,
            threads_per_process=8,
            cache_path="/tmp/cache",
        )
        out = tmp_path / "exp.yaml"
        original.save(out)
        loaded = Workflow.load(out)

        assert loaded.data_start == "2023-01-01"
        assert loaded.data_end == "2024-01-01"
        assert loaded.data_cache is True
        assert loaded.model_test_start == "2023-10-01"
        assert loaded.model_test_end == "2024-01-01"
        assert loaded.model_target == "price"
        assert loaded.model_horizon == 5
        assert loaded.max_processes == 4
        assert loaded.threads_per_process == 8
        assert loaded.cache_path == "/tmp/cache"

    def test_model_index_selects_single_model(self, tmp_path, pipeline_yamls):
        dp_path, mp_path = pipeline_yamls
        wf = Workflow(
            data_pipeline=str(dp_path),
            model_pipeline=str(mp_path),
            data_start="2023-01-01",
            data_end="2024-01-01",
            model_test_start="2023-10-01",
            model_test_end="2024-01-01",
            model_index=1,
        )
        out = tmp_path / "exp.yaml"
        wf.save(out)
        loaded = Workflow.load(out)
        assert loaded.model_index == 1

        mp = loaded._build_model_pipeline()
        assert len(mp.models) == 1
        assert mp.models[0].name == "Lasso"

    def test_model_index_none_runs_all_models(self, tmp_path, pipeline_yamls):
        dp_path, mp_path = pipeline_yamls
        wf = Workflow(
            data_pipeline=str(dp_path),
            model_pipeline=str(mp_path),
            data_start="2023-01-01",
            data_end="2024-01-01",
            model_test_start="2023-10-01",
            model_test_end="2024-01-01",
        )
        out = tmp_path / "exp.yaml"
        wf.save(out)
        loaded = Workflow.load(out)

        assert loaded.model_index is None
        mp = loaded._build_model_pipeline()
        assert len(mp.models) == 2

    def test_model_index_not_in_yaml_when_none(self, tmp_path, pipeline_yamls):
        dp_path, mp_path = pipeline_yamls
        wf = Workflow(
            data_pipeline=str(dp_path),
            model_pipeline=str(mp_path),
            data_start="2023-01-01",
            data_end="2024-01-01",
            model_test_start="2023-10-01",
            model_test_end="2024-01-01",
        )
        out = tmp_path / "exp.yaml"
        wf.save(out)
        config = yaml.safe_load(out.read_text())
        assert "environment" not in config or "model_index" not in config.get("environment", {})

    def test_relative_paths_resolved_from_yaml_dir(self, tmp_path, pipeline_yamls):
        dp_path, mp_path = pipeline_yamls
        wf = Workflow(
            data_pipeline="dp.yaml",
            model_pipeline="mp.yaml",
            data_start="2023-01-01",
            data_end="2024-01-01",
            model_test_start="2023-10-01",
            model_test_end="2024-01-01",
        )
        out = tmp_path / "exp.yaml"
        wf.save(out)
        loaded = Workflow.load(out)

        # _resolve should find dp.yaml relative to tmp_path
        assert loaded._resolve("dp.yaml") == tmp_path / "dp.yaml"

    def test_inline_override_replaces_transformers(self, tmp_path, pipeline_yamls):
        dp_path, mp_path = pipeline_yamls
        wf = Workflow(
            data_pipeline=str(dp_path),
            model_pipeline=str(mp_path),
            data_start="2023-01-01",
            data_end="2024-01-01",
            model_test_start="2023-10-01",
            model_test_end="2024-01-01",
        )
        out = tmp_path / "exp.yaml"
        wf.save(out)

        # Manually inject inline transformer override
        config = yaml.safe_load(out.read_text())
        config["data_pipeline"]["transformers"] = [
            {"class": "ResampleTransformer", "params": {"freq": "1h", "method": "ffill"}}
        ]
        out.write_text(yaml.dump(config, default_flow_style=False))

        loaded = Workflow.load(out)
        dp = loaded._build_data_pipeline()
        assert len(dp.transformers) == 1
        assert isinstance(dp.transformers[0], ResampleTransformer)
        assert dp.transformers[0].freq == "1h"

    def test_inline_model_override(self, tmp_path, pipeline_yamls):
        dp_path, mp_path = pipeline_yamls
        wf = Workflow(
            data_pipeline=str(dp_path),
            model_pipeline=str(mp_path),
            data_start="2023-01-01",
            data_end="2024-01-01",
            model_test_start="2023-10-01",
            model_test_end="2024-01-01",
        )
        out = tmp_path / "exp.yaml"
        wf.save(out)

        config = yaml.safe_load(out.read_text())
        config["model_pipeline"]["models"] = [
            {"class": "OLSModel", "params": {"predictors": ["is_holiday"], "training_window": 90, "name": "Inline"}}
        ]
        out.write_text(yaml.dump(config, default_flow_style=False))

        loaded = Workflow.load(out)
        mp = loaded._build_model_pipeline()
        assert len(mp.models) == 1
        assert mp.models[0].name == "Inline"
        assert mp.models[0].training_window == 90
