import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

from epftoolbox2.pipelines import DataPipeline, ModelPipeline, Workflow
from epftoolbox2.data.sources import CalendarSource, CsvSource
from epftoolbox2.data.transformers import ResampleTransformer, LagTransformer
from epftoolbox2.data.validators import CoverageValidator
from epftoolbox2.ensembles import AverageEnsemble
from epftoolbox2.models import (
    HistGBModel,
    LassoCVModel,
    LEARModel,
    NaiveModel,
    OLSModel,
    PooledModel,
    ResidualModel,
    RidgeModel,
)
from epftoolbox2.models.base import BaseModel
from epftoolbox2.evaluators import (
    BatteryArbitrage,
    BiasEvaluator,
    DMEvaluator,
    EconomicEvaluator,
    MAEEvaluator,
    MCSEvaluator,
    NMAEEvaluator,
    RMSEEvaluator,
    ScaledMAEEvaluator,
    ShapeMAEEvaluator,
    rMAEEvaluator,
)
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
            .add_validator(CoverageValidator(columns=["is_holiday"]))
        )
        path = tmp_path / "dp.yaml"
        pipeline.save(path)
        config = yaml.safe_load(path.read_text())
        assert "sources" in config
        assert "transformers" in config
        assert "validators" in config
        assert config["sources"][0]["class"] == "CalendarSource"
        assert config["transformers"][0]["class"] == "ResampleTransformer"
        assert config["validators"][0]["class"] == "CoverageValidator"

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
        original = DataPipeline().add_transformer(
            ResampleTransformer(freq="15min", agg={"default": "mean", "energy": "sum"}, fill={"price": None})
        )
        path = tmp_path / "dp.yaml"
        original.save(path)
        loaded = DataPipeline.load(path)

        t = loaded.transformers[0]
        assert isinstance(t, ResampleTransformer)
        assert t.freq == "15min"
        assert t.agg == {"default": "mean", "energy": "sum"}
        assert t.fill == {"price": None}

    def test_round_trip_lag_transformer_list(self, tmp_path):
        original = DataPipeline().add_transformer(LagTransformer(columns=["price", "load"], lags=[1, 2, 7], freq="day"))
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
        original = DataPipeline().add_transformer(LagTransformer(columns=["price"], lags=range(-7, 1), freq="day"))
        path = tmp_path / "dp.yaml"
        original.save(path)
        loaded = DataPipeline.load(path)

        t = loaded.transformers[0]
        assert t.lags == list(range(-7, 1))

    def test_round_trip_coverage_validator(self, tmp_path):
        original = DataPipeline().add_validator(CoverageValidator(columns=["price"], min_coverage=0.0))
        path = tmp_path / "dp.yaml"
        original.save(path)
        loaded = DataPipeline.load(path)

        v = loaded.validators[0]
        assert isinstance(v, CoverageValidator)
        assert v.columns == ["price"]
        assert v.min_coverage == 0.0

    def test_round_trip_multiple_components(self, tmp_path):
        original = (
            DataPipeline()
            .add_source(CalendarSource(country="PL"))
            .add_source(CalendarSource(country="DE", prefix="de"))
            .add_transformer(ResampleTransformer(freq="1h"))
            .add_transformer(LagTransformer(columns=["price"], lags=[1, 7], freq="day"))
            .add_validator(CoverageValidator(columns=["price"]))
            .add_validator(CoverageValidator(freq="1h"))
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


class TestDataPipelineSerializationFidelity:
    """pipelines M4 / empirical F7 and the "silently dropped parameter" findings."""

    def test_calendar_lat_lon_survive_round_trip(self, tmp_path):
        original = DataPipeline().add_source(CalendarSource("PL", daylight=True, lat=54.35, lon=18.65))  # Gdansk
        path = tmp_path / "dp.yaml"
        original.save(path)
        loaded = DataPipeline.load(path)

        assert (loaded.sources[0].lat, loaded.sources[0].lon) == (54.35, 18.65)
        a = original.run("2024-06-20", "2024-06-22")
        b = loaded.run("2024-06-20", "2024-06-22")
        pd.testing.assert_frame_equal(a, b)

    def test_every_registered_component_round_trips(self, tmp_path):
        from epftoolbox2.data.sources import EnergyChartsSource, EntsoeSource, OpenMeteoSource, PseSource
        from epftoolbox2.data.transformers import TimezoneTransformer
        from epftoolbox2.data.validators import EdaValidator
        from epftoolbox2.pipelines.data_pipeline import COMPONENT_REGISTRY

        csv = tmp_path / "p.csv"
        csv.write_text("ts;price\n2024-01-01 00:00;1.0\n")
        components = {
            "EntsoeSource": EntsoeSource("DE_LU", api_key="dummy", type=["price"], prefer_15min=True),
            "OpenMeteoSource": OpenMeteoSource(
                50.06,
                19.94,
                horizon=3,
                model="icon_seamless",
                columns=["temperature_2m"],
                prefix="krk",
                timezone="Europe/Warsaw",
            ),
            "EnergyChartsSource": EnergyChartsSource(
                "de",
                type=["forecast", "price"],
                bzn="CH",
                forecast_types=["day-ahead", "intraday"],
                production_types=["solar", "load"],
                prefix="ec",
                cache_dir=str(tmp_path / "ec"),
                use_cache=False,
            ),
            "CsvSource": CsvSource(str(csv), datetime_column="ts", columns=["price"], prefix="x", separator=";"),
            "CalendarSource": CalendarSource(
                "PL",
                holidays="onehot",
                weekday="name",
                hour="number",
                daylight=True,
                lat=50.06,
                lon=19.94,
                freq="15min",
            ),
            "PseSource": PseSource(datasets=["kse-load", "his-wlk-cal"], columns=["load_fcst", "pv"], prefix="pse"),
            "TimezoneTransformer": TimezoneTransformer("Europe/Berlin", daily_columns=["a"]),
            "ResampleTransformer": ResampleTransformer("15min", agg={"a": "sum"}, fill=None),
            "LagTransformer": LagTransformer(["a"], [1, -2], "15min"),
            "CoverageValidator": CoverageValidator(["a"], freq="15min", min_coverage=0.9),
            "EdaValidator": EdaValidator(["a"]),
        }
        kind = {name: section for section, names in COMPONENT_REGISTRY.items() for name in names}
        assert set(components) <= set(kind)  # components registered later: add them here too

        def public(obj):
            return {
                k: v
                for k, v in vars(obj).items()
                if not k.startswith("_") and k not in {"session", "console", "logger"}
            }

        for name, obj in components.items():
            pipeline = DataPipeline(**{kind[name]: [obj]})
            path = tmp_path / f"{name}.yaml"
            pipeline.save(path, include_secrets=True)
            new = getattr(DataPipeline.load(path), kind[name])[0]
            assert {k: str(v) for k, v in public(obj).items()} == {k: str(v) for k, v in public(new).items()}, name

    def test_unsupported_parameter_type_raises(self, tmp_path):
        transformer = LagTransformer(["a"], [1], "day")
        transformer.columns = {"a"}  # a set cannot be written to YAML
        with pytest.raises(TypeError, match="LagTransformer.columns"):
            DataPipeline().add_transformer(transformer).save(tmp_path / "dp.yaml")

    def test_callable_parameter_raises(self, tmp_path):
        transformer = LagTransformer(["a"], [1], "day")
        transformer.columns = lambda: ["a"]
        with pytest.raises(TypeError, match="LagTransformer.columns"):
            DataPipeline().add_transformer(transformer).to_dict()

    def test_numpy_values_are_saved_as_plain_python(self, tmp_path):
        transformer = LagTransformer(["price"], np.array([1, 2, 7]), "day")
        source = CalendarSource("PL", lat=np.float64(54.35), lon=np.float64(18.65))
        path = tmp_path / "dp.yaml"
        DataPipeline().add_source(source).add_transformer(transformer).save(path)

        assert "!!python" not in path.read_text()
        loaded = DataPipeline.load(path)
        assert loaded.transformers[0].lags == [1, 2, 7]
        assert all(type(v) is int for v in loaded.transformers[0].lags)
        assert type(loaded.sources[0].lat) is float and loaded.sources[0].lat == 54.35

    def test_tuple_is_saved_as_list(self, tmp_path):
        from epftoolbox2.data.sources import OpenMeteoSource

        path = tmp_path / "dp.yaml"
        DataPipeline().add_source(
            OpenMeteoSource(52.2, 21.0, columns=("temperature_2m", "rain"), timezone="Europe/Warsaw")
        ).save(path)
        assert DataPipeline.load(path).sources[0].columns == ["temperature_2m", "rain"]


class TestDataPipelineSettingsInYaml:
    def test_pipeline_settings_round_trip(self, tmp_path):
        original = DataPipeline(
            sources=[CalendarSource("PL")],
            source_validators=[CoverageValidator(freq="1h")],
            validators=[CoverageValidator(["is_holiday"])],
            timezone="Europe/Warsaw",
            on_invalid="raise",
        )
        path = tmp_path / "dp.yaml"
        original.save(path)
        config = yaml.safe_load(path.read_text())
        assert config["timezone"] == "Europe/Warsaw"
        assert config["on_invalid"] == "raise"
        assert config["source_validators"][0]["class"] == "CoverageValidator"

        loaded = DataPipeline.load(path)
        assert loaded.timezone == "Europe/Warsaw"
        assert loaded.on_invalid == "raise"
        assert isinstance(loaded.source_validators[0], CoverageValidator)
        assert loaded.source_validators[0].freq == "1h"
        assert isinstance(loaded.validators[0], CoverageValidator)

    def test_yaml_without_settings_keeps_defaults(self, tmp_path):
        path = tmp_path / "dp.yaml"
        path.write_text("sources:\n- class: CalendarSource\n  params: {country: PL}\n")
        loaded = DataPipeline.load(path)
        assert loaded.timezone is None
        assert loaded.on_invalid == "warn"
        assert loaded.source_validators == []

    def test_unknown_top_level_key_raises(self, tmp_path):
        path = tmp_path / "dp.yaml"
        path.write_text("timezon: Europe/Warsaw\nsources: []\n")
        with pytest.raises(ValueError, match="timezon"):
            DataPipeline.load(path)


class TestSecretsInYaml:
    """Known issue (rules/data.md) / pipelines M6: the API key is not written to YAML in plain text."""

    def _entsoe(self, key="real-secret-123"):
        from epftoolbox2.data.sources import EntsoeSource

        return EntsoeSource("PL", api_key=key, type=["price"])

    def test_save_writes_env_reference_instead_of_key(self, tmp_path):
        path = tmp_path / "dp.yaml"
        DataPipeline().add_source(self._entsoe()).save(path)
        text = path.read_text()
        assert "real-secret-123" not in text
        assert yaml.safe_load(text)["sources"][0]["params"]["api_key"] == "${ENTSOE_API_KEY}"

    def test_include_secrets_writes_the_key(self, tmp_path):
        path = tmp_path / "dp.yaml"
        DataPipeline().add_source(self._entsoe()).save(path, include_secrets=True)
        assert yaml.safe_load(path.read_text())["sources"][0]["params"]["api_key"] == "real-secret-123"

    def test_load_expands_env_reference(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ENTSOE_API_KEY", "key-from-env")
        path = tmp_path / "dp.yaml"
        DataPipeline().add_source(self._entsoe()).save(path)
        assert DataPipeline.load(path).sources[0].api_key == "key-from-env"

    def test_load_with_unset_variable_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ENTSOE_API_KEY", raising=False)
        path = tmp_path / "dp.yaml"
        DataPipeline().add_source(self._entsoe()).save(path)
        with pytest.raises(ValueError, match="ENTSOE_API_KEY"):
            DataPipeline.load(path)

    def test_custom_reference_is_kept_on_resave(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_ENTSOE_TOKEN", "abc")
        path = tmp_path / "dp.yaml"
        path.write_text(
            "sources:\n- class: EntsoeSource\n  params:\n    country_code: PL\n"
            "    api_key: ${MY_ENTSOE_TOKEN}\n    type: [price]\n"
        )
        loaded = DataPipeline.load(path)
        assert loaded.sources[0].api_key == "abc"
        loaded.save(tmp_path / "again.yaml")
        assert "${MY_ENTSOE_TOKEN}" in (tmp_path / "again.yaml").read_text()

    def test_env_reference_in_any_string_parameter(self, tmp_path, monkeypatch):
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "prices.csv").write_text("datetime,price\n2024-01-01 00:00,1.0\n")
        monkeypatch.setenv("EPF_DATA_DIR", str(tmp_path / "data"))
        path = tmp_path / "dp.yaml"
        path.write_text("sources:\n- class: CsvSource\n  params:\n    file_path: ${EPF_DATA_DIR}/prices.csv\n")
        loaded = DataPipeline.load(path)
        assert Path(loaded.sources[0].file_path) == tmp_path / "data" / "prices.csv"
        loaded.save(tmp_path / "again.yaml")
        assert "${EPF_DATA_DIR}/prices.csv" in (tmp_path / "again.yaml").read_text()


class TestRelativePathsInYaml:
    """Relative paths in a pipeline YAML are relative to the YAML file, like Workflow pipeline paths."""

    @pytest.fixture
    def layout(self, tmp_path, monkeypatch):
        (tmp_path / "data").mkdir()
        (tmp_path / "configs").mkdir()
        (tmp_path / "elsewhere" / "deeper").mkdir(parents=True)  # ../data does not exist from here
        (tmp_path / "data" / "prices.csv").write_text("datetime,price\n2024-01-01 00:00,1.0\n2024-01-01 01:00,2.0\n")
        monkeypatch.chdir(tmp_path / "elsewhere" / "deeper")
        return tmp_path

    def test_load_resolves_relative_to_yaml(self, layout):
        path = layout / "configs" / "dp.yaml"
        path.write_text("sources:\n- class: CsvSource\n  params:\n    file_path: ../data/prices.csv\n")
        loaded = DataPipeline.load(path)
        assert Path(loaded.sources[0].file_path).resolve() == (layout / "data" / "prices.csv").resolve()
        assert len(loaded.run("2024-01-01", "2024-01-02")) == 2

    def test_save_writes_relative_to_yaml(self, layout, monkeypatch):
        monkeypatch.chdir(layout)
        pipeline = DataPipeline().add_source(CsvSource("data/prices.csv"))  # relative to the working directory
        path = layout / "configs" / "dp.yaml"
        pipeline.save(path)
        assert yaml.safe_load(path.read_text())["sources"][0]["params"]["file_path"] == "../data/prices.csv"

        monkeypatch.chdir(layout / "elsewhere" / "deeper")
        assert len(DataPipeline.load(path).run("2024-01-01", "2024-01-02")) == 2

    def test_absolute_path_outside_yaml_dir_stays_absolute(self, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        csv = other / "p.csv"
        csv.write_text("datetime,price\n2024-01-01 00:00,1.0\n")
        path = tmp_path / "configs" / "dp.yaml"
        path.parent.mkdir()
        DataPipeline().add_source(CsvSource(str(csv))).save(path)
        assert yaml.safe_load(path.read_text())["sources"][0]["params"]["file_path"] == str(csv)

    def test_loaded_relative_path_stays_relative_on_resave(self, layout):
        path = layout / "configs" / "dp.yaml"
        path.write_text("sources:\n- class: CsvSource\n  params:\n    file_path: ../data/prices.csv\n")
        DataPipeline.load(path).save(layout / "configs" / "copy.yaml")
        DataPipeline.load(path).save(layout / "copy_at_root.yaml")
        assert "file_path: ../data/prices.csv" in (layout / "configs" / "copy.yaml").read_text()
        assert "file_path: data/prices.csv" in (layout / "copy_at_root.yaml").read_text()


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
        original = ModelPipeline().add_model(OLSModel(predictors=predictors, training_window=180, name="MyOLS"))
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

    def test_callable_predictor_in_list_raises_on_save(self, tmp_path):
        pipeline = ModelPipeline().add_model(OLSModel(predictors=[lambda h: f"col_d+{h}"], name="Bad"))
        with pytest.raises(TypeError, match=r"model 'Bad' .*callable predictors can't be saved to YAML yet"):
            pipeline.save(tmp_path / "mp.yaml")

    @pytest.mark.parametrize(
        "predictors",
        [
            pytest.param(lambda horizon: ["price_d-1", f"load_d+{horizon}"], id="horizon"),
            pytest.param(lambda horizon, hour: [f"load_h{hour:02d}_d+{horizon}"], id="horizon-and-hour"),
        ],
    )
    def test_predictor_function_raises_instead_of_saving_an_empty_list(self, tmp_path, predictors):
        # BaseModel keeps `predictors == []` and the function privately; a plain dump wrote `predictors: []`,
        # so the reloaded pipeline was a different study.
        pipeline = ModelPipeline().add_model(LassoCVModel(predictors=predictors, name="FromFunction"))
        path = tmp_path / "mp.yaml"
        with pytest.raises(TypeError) as error:
            pipeline.save(path)
        message = str(error.value)
        assert "'FromFunction'" in message and "LassoCVModel" in message
        assert "callable predictors can't be saved to YAML yet" in message
        assert "{horizon}" in message  # tells how to write them explicitly
        with pytest.raises(TypeError):
            pipeline.to_dict()
        assert not path.exists()

    def test_predictor_function_assigned_after_construction_raises(self, tmp_path):
        model = OLSModel(predictors=["price_d-1"], name="Late")
        model.predictors = lambda horizon: ["price_d-1"]
        with pytest.raises(TypeError, match="'Late'"):
            ModelPipeline().add_model(model).save(tmp_path / "mp.yaml")

    def test_only_the_model_with_a_function_is_named(self, tmp_path):
        pipeline = (
            ModelPipeline()
            .add_model(OLSModel(predictors=["price_d-1"], name="Fine"))
            .add_model(OLSModel(predictors=lambda horizon: ["price_d-1"], name="Dynamic"))
        )
        with pytest.raises(TypeError, match="'Dynamic'") as error:
            pipeline.save(tmp_path / "mp.yaml")
        assert "'Fine'" not in str(error.value)

    def test_failed_save_leaves_an_existing_file_alone(self, tmp_path):
        path = tmp_path / "mp.yaml"
        ModelPipeline().add_model(OLSModel(predictors=["price_d-1"], name="Good")).save(path)
        before = path.read_text()
        with pytest.raises(TypeError):
            ModelPipeline().add_model(OLSModel(predictors=lambda horizon: ["x"], name="Bad")).save(path)
        assert path.read_text() == before

    def test_shared_predictor_list_is_written_in_full_for_each_model(self, tmp_path):
        # The same list object used by two models used to be written once as `&id001` and then as `*id001`.
        predictors = ["load_actual", "price_d-1"]
        original = (
            ModelPipeline()
            .add_model(OLSModel(predictors=predictors, name="OLS"))
            .add_model(LassoCVModel(predictors=predictors, name="Lasso"))
        )
        path = tmp_path / "mp.yaml"
        original.save(path)

        text = path.read_text()
        assert "&id" not in text and "*id" not in text
        saved = yaml.safe_load(text)["models"]
        assert [m["params"]["predictors"] for m in saved] == [predictors, predictors]
        assert text.count("- load_actual") == 2  # inline in both models, not once and a reference

        loaded = ModelPipeline.load(path)
        assert loaded.models[0].predictors == predictors
        assert loaded.models[1].predictors == predictors
        assert loaded.models[0].predictors is not loaded.models[1].predictors  # editing one leaves the other

    def test_shared_parameter_lists_of_evaluators_and_exporters_are_written_in_full(self, tmp_path):
        show = ["summary", "horizon"]
        pipeline = ModelPipeline().add_exporter(TerminalExporter(show=show)).add_exporter(TerminalExporter(show=show))
        path = tmp_path / "mp.yaml"
        pipeline.save(path)
        assert "&id" not in path.read_text() and "*id" not in path.read_text()
        assert [e.show for e in ModelPipeline.load(path).exporters] == [show, show]

    def test_unknown_model_raises_on_load(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("models:\n- class: NonExistentModel\n  params: {predictors: [], name: x}\n")
        with pytest.raises(ValueError, match="Unknown model"):
            ModelPipeline.load(path)


class TupleParamModel(OLSModel):
    """A model with a tuple parameter, as a neural network has `hidden=(256, 128)`."""

    def __init__(self, predictors, hidden=(8,), **kwargs):
        super().__init__(predictors, **kwargs)
        self.hidden = hidden

    @property
    def _model_kwargs(self):
        return {"hidden": self.hidden}


@pytest.fixture
def tuple_param_model(monkeypatch):
    """Registers TupleParamModel (this module) so that a saved pipeline can name it."""
    from epftoolbox2.pipelines.model_pipeline import COMPONENT_REGISTRY as MODEL_REGISTRY

    monkeypatch.setitem(MODEL_REGISTRY["models"], "TupleParamModel", __name__)
    return TupleParamModel


class TestModelPipelineParameterFidelity:
    """Every public parameter is written, converted to plain YAML values, or refused: none is silently dropped."""

    def test_tuple_parameter_survives_as_a_list_and_the_model_reloads_with_it(self, tmp_path, tuple_param_model):
        path = tmp_path / "mp.yaml"
        ModelPipeline().add_model(tuple_param_model(["price_d-1"], hidden=(256, 128), name="Net")).save(path)

        assert yaml.safe_load(path.read_text())["models"][0]["params"]["hidden"] == [256, 128]
        reloaded = ModelPipeline.load(path).models[0]
        assert isinstance(reloaded, TupleParamModel)
        assert reloaded.hidden == [256, 128]  # not the default (8,)
        assert reloaded._model_kwargs == {"hidden": [256, 128]}  # what the worker processes rebuild the model from

    def test_predictors_given_as_a_tuple_are_saved(self, tmp_path):
        # A tuple used to be dropped, leaving a file that could not be loaded (predictors is required).
        path = tmp_path / "mp.yaml"
        ModelPipeline().add_model(OLSModel(predictors=("price_d-1", "load_d+{horizon}"), name="T")).save(path)
        assert ModelPipeline.load(path).models[0].predictors == ["price_d-1", "load_d+{horizon}"]

    def test_numpy_values_are_saved_as_plain_python(self, tmp_path, tuple_param_model):
        model = tuple_param_model(["price_d-1"], hidden=np.array([64, 32]), training_window=np.int64(180), name="Numpy")
        path = tmp_path / "mp.yaml"
        ModelPipeline().add_model(model).add_exporter(TerminalExporter(show=np.array(["summary"]))).save(path)

        assert "!!python" not in path.read_text()
        loaded = ModelPipeline.load(path)
        assert loaded.models[0].hidden == [64, 32]
        assert loaded.models[0].training_window == 180 and type(loaded.models[0].training_window) is int
        assert loaded.exporters[0].show == ["summary"]

    def test_path_parameter_is_saved_as_a_string(self, tmp_path):
        path = tmp_path / "mp.yaml"
        ModelPipeline().add_exporter(ExcelExporter(path=Path("results") / "out.xlsx")).save(path)
        assert yaml.safe_load(path.read_text())["exporters"][0]["params"]["path"] == str(Path("results") / "out.xlsx")

    def test_unsupported_model_parameter_raises_naming_model_and_parameter(self, tmp_path, tuple_param_model):
        model = tuple_param_model(["price_d-1"], hidden={64, 32}, name="WithSet")  # a set cannot be written to YAML
        path = tmp_path / "mp.yaml"
        with pytest.raises(TypeError, match=r"TupleParamModel\('WithSet'\)\.hidden"):
            ModelPipeline().add_model(model).save(path)
        assert not path.exists()

    def test_callable_model_parameter_raises(self, tmp_path):
        model = LassoCVModel(["price_d-1"], name="WithFunction")
        model.cv = lambda n: n
        with pytest.raises(TypeError, match=r"LassoCVModel\('WithFunction'\)\.cv.*callable"):
            ModelPipeline().add_model(model).to_dict()

    def test_unsupported_evaluator_and_exporter_parameters_raise(self, tmp_path):
        evaluator = rMAEEvaluator("Naive")
        evaluator.base_model = object()
        with pytest.raises(TypeError, match=r"rMAEEvaluator\.base_model"):
            ModelPipeline().add_evaluator(evaluator).save(tmp_path / "a.yaml")

        exporter = TerminalExporter()
        exporter.show = {"summary"}
        with pytest.raises(TypeError, match=r"TerminalExporter\.show"):
            ModelPipeline().add_exporter(exporter).save(tmp_path / "b.yaml")

    def test_the_strategy_is_saved_inside_the_evaluator_and_reloads(self, tmp_path):
        path = tmp_path / "mp.yaml"
        evaluator = EconomicEvaluator(BatteryArbitrage(power_mw=2, energy_mwh=8, efficiency=0.85))
        ModelPipeline().add_evaluator(evaluator).save(path)

        saved = yaml.safe_load(path.read_text())["evaluators"][0]
        assert saved == {
            "class": "EconomicEvaluator",
            "params": {
                "strategy": {
                    "class": "BatteryArbitrage",
                    "params": {"power_mw": 2, "energy_mwh": 8, "efficiency": 0.85},
                }
            },
        }
        loaded = ModelPipeline.load(path).evaluators[0]
        assert isinstance(loaded.strategy, BatteryArbitrage)
        assert (loaded.strategy.power_mw, loaded.strategy.energy_mwh, loaded.strategy.efficiency) == (2, 8, 0.85)

    def test_the_default_strategy_is_saved_in_full(self, tmp_path):
        path = tmp_path / "mp.yaml"
        ModelPipeline().add_evaluator(EconomicEvaluator()).save(path)
        strategy = yaml.safe_load(path.read_text())["evaluators"][0]["params"]["strategy"]
        assert strategy == {"class": "BatteryArbitrage", "params": {"power_mw": 1, "energy_mwh": 2, "efficiency": 0.9}}

    def test_an_unknown_strategy_raises_on_load(self, tmp_path):
        path = tmp_path / "mp.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "evaluators": [
                        {"class": "EconomicEvaluator", "params": {"strategy": {"class": "Nope", "params": {}}}}
                    ]
                }
            )
        )
        with pytest.raises(ValueError, match="Unknown strategy: Nope"):
            ModelPipeline.load(path)

    def test_an_unregistered_strategy_raises_on_save(self, tmp_path):
        class MyStrategy:
            def profit(self, forecast, actual, periods_per_hour=1):
                return actual.sum(axis=1)

        with pytest.raises(TypeError, match=r"EconomicEvaluator\.strategy"):
            ModelPipeline().add_evaluator(EconomicEvaluator(MyStrategy())).save(tmp_path / "mp.yaml")

    def test_a_workflow_loads_the_new_evaluators_inline(self):
        evaluator = ModelPipeline._load_component(
            "evaluators", {"class": "DMEvaluator", "params": {"benchmark": "O", "correction": None, "lag": 7}}
        )
        assert isinstance(evaluator, DMEvaluator)
        assert (evaluator.benchmark, evaluator.correction, evaluator.lag) == ("O", None, 7)

    def test_every_registered_component_round_trips_with_non_default_values(self, tmp_path):
        from epftoolbox2.exporters import CsvExporter
        from epftoolbox2.pipelines.model_pipeline import COMPONENT_REGISTRY as MODEL_REGISTRY

        originals = {
            "models": [
                OLSModel(
                    ["price_d-1", "load_d+{horizon}"], training_window=90, name="O", scaler="asinh", fit_intercept=True
                ),
                LassoCVModel(
                    ["price_d-1"],
                    training_window=60,
                    cv=3,
                    max_iter=77,
                    name="L",
                    fit_intercept=False,
                    scaler={"name": "standard", "params": {"scale_binary": True}},
                ),
                RidgeModel(
                    ["price_d-1", "load_d+{horizon}"],
                    training_window=80,
                    alphas=[0.25, 4.0, 64.0],
                    name="R",
                    fit_intercept=False,
                    scaler="asinh",
                ),
                NaiveModel(["load_d+{horizon}"], name="N", known_only=False),
                LEARModel(
                    ["price_d-1", "load_d+{horizon}"],
                    training_window=[14, "expanding"],
                    criterion="aic",
                    solver="cd",
                    name="E",
                    scaler={"name": "asinh", "params": {"standardise": False}},
                    newest_target="published",
                    refit="7D",
                ),
                PooledModel(
                    RidgeModel(["price_d-1"], training_window=70, alphas=[1.0, 8.0], name="inner", refit="7D"),
                    over=("zone", "horizon"),
                    zone_feature="code",
                    missing="zero",
                    name="P",
                ),
                HistGBModel(
                    ["price_d-1", "period"],
                    training_window=56,
                    refit="1D",
                    loss="l2",
                    max_iter=99,
                    learning_rate=0.1,
                    max_leaf_nodes=15,
                    min_samples_leaf=7,
                    l2_regularization=0.5,
                    max_features=0.8,
                    random_state=3,
                    name="H",
                ),
                # A base by name, and a base and learner nested in it.
                ResidualModel(
                    "O", HistGBModel(["price_d-1", "base_fc_d+{horizon}"], periods="each"), 91, "14D", 30, "Res"
                ),
                ResidualModel(
                    RidgeModel(["price_d-1"], training_window=40, alphas=[2.0], name="b"),
                    HistGBModel(["price_d-1", "base_fc_h05_d+{horizon}"], max_iter=12),  # its name isn't written
                    name="Res2",
                ),
            ],
            "ensembles": [AverageEnsemble(["O", "L"], top=1, by="validation", name="best")],
            "evaluators": [
                MAEEvaluator(),
                RMSEEvaluator(),
                rMAEEvaluator("O"),
                BiasEvaluator(),
                DMEvaluator(
                    "O",
                    variance="hac",
                    lag=7,
                    correction=None,
                    alpha=0.01,
                    interval=0.9,
                    block=14,
                    reps=500,
                    seed=3,
                ),
                MCSEvaluator(alpha=0.25, statistic="max", block=14, reps=2000, seed=5),
                EconomicEvaluator(BatteryArbitrage(power_mw=2, energy_mwh=8, efficiency=0.85)),
                ShapeMAEEvaluator(period="1h"),
                NMAEEvaluator(),
                ScaledMAEEvaluator("installed_capacity"),
            ],
            "exporters": [
                CsvExporter(
                    str(tmp_path / "out.csv"), extra_columns=["is_holiday"], summary_path=str(tmp_path / "s.csv")
                ),
                ExcelExporter(str(tmp_path / "out.xlsx"), sheets=["summary", "hour"]),
                TerminalExporter(show=["hour"]),
            ],
        }
        covered = {type(obj).__name__ for objs in originals.values() for obj in objs} | {"BatteryArbitrage"}
        assert covered == {name for names in MODEL_REGISTRY.values() for name in names}  # add a new component here

        pipeline = ModelPipeline()
        pipeline.models, pipeline.ensembles, pipeline.evaluators, pipeline.exporters = (
            originals[kind] for kind in ("models", "ensembles", "evaluators", "exporters")
        )
        path = tmp_path / "mp.yaml"
        pipeline.save(path)
        loaded = ModelPipeline.load(path)

        def public(obj):
            # A model inside another (`PooledModel.model`) by its own parameters.
            return {
                k: public(v) if isinstance(v, BaseModel) else str(v)
                for k, v in vars(obj).items()
                if not k.startswith("_")
            }

        for kind, objs in originals.items():
            reloaded = getattr(loaded, kind)
            assert [type(o) for o in reloaded] == [type(o) for o in objs], kind
            assert [public(o) for o in reloaded] == [public(o) for o in objs], kind


# ---------------------------------------------------------------------------
# Workflow serialization
# ---------------------------------------------------------------------------


class TestWorkflowYamlDates:
    """YAML reads an unquoted `2024-01-31` as a `datetime.date`, which must mean the same days as the quoted string."""

    @staticmethod
    def _write(tmp_path, quote):
        q = '"' if quote else ""
        DataPipeline(timezone="Europe/Warsaw").add_source(CalendarSource(country="PL")).save(tmp_path / "dp.yaml")
        ModelPipeline().add_model(OLSModel(predictors=["is_holiday"], name="OLS")).save(tmp_path / "mp.yaml")
        path = tmp_path / f"exp_{'quoted' if quote else 'bare'}.yaml"
        path.write_text(
            "data_pipeline:\n"
            "  path: dp.yaml\n"
            f"  start: {q}2024-01-01{q}\n"
            f"  end: {q}2024-01-31{q}\n"
            "model_pipeline:\n"
            "  path: mp.yaml\n"
            f"  test_start: {q}2024-01-10{q}\n"
            f"  test_end: {q}2024-01-12{q}\n"
        )
        return path

    def test_unquoted_dates_load_as_dates_and_run_the_same_range_as_quoted_ones(self, tmp_path):
        bare = Workflow.load(self._write(tmp_path, quote=False))
        quoted = Workflow.load(self._write(tmp_path, quote=True))
        assert not isinstance(bare.data_end, str)
        df_bare, df_quoted = bare._run_data(), quoted._run_data()
        assert df_bare.index.equals(df_quoted.index)
        assert df_bare.index.max() == pd.Timestamp("2024-01-31 23:00", tz="Europe/Warsaw")
        assert len(df_bare) == 31 * 24

    def test_unquoted_test_dates_run_the_same_days_as_quoted_ones(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MAX_PROCESSES", "1")
        monkeypatch.setenv("THREADS_PER_PROCESS", "1")
        bare = Workflow.load(self._write(tmp_path, quote=False))
        quoted = Workflow.load(self._write(tmp_path, quote=True))
        assert not isinstance(bare.model_test_start, str)
        data = bare._run_data().assign(price=lambda df: np.arange(len(df), dtype=float))
        data.index = data.index.tz_convert("Europe/Warsaw")
        kwargs = dict(target="price", horizon=1, freq="1h")
        refs = [
            OLSModel(["is_holiday"], training_window=5).run(
                data, test_start=wf.model_test_start, test_end=wf.model_test_end, **kwargs
            )
            for wf in (bare, quoted)
        ]
        assert refs[0].count == refs[1].count == 3 * 24
        key = lambda r: (r["hour"], r["horizon"], r["run_date"])  # noqa: E731
        assert sorted(refs[0]._results, key=key) == sorted(refs[1]._results, key=key)

    def test_a_saved_workflow_with_dates_reloads_unchanged(self, tmp_path):
        bare = Workflow.load(self._write(tmp_path, quote=False))
        out = tmp_path / "saved.yaml"
        bare.save(out)
        again = Workflow.load(out)
        assert again._run_data().index.equals(bare._run_data().index)


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
            {"class": "ResampleTransformer", "params": {"freq": "1h", "agg": "sum"}}
        ]
        out.write_text(yaml.dump(config, default_flow_style=False))

        loaded = Workflow.load(out)
        dp = loaded._build_data_pipeline()
        assert len(dp.transformers) == 1
        assert isinstance(dp.transformers[0], ResampleTransformer)
        assert dp.transformers[0].freq == "1h"
        assert dp.transformers[0].agg == "sum"

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


# ---------------------------------------------------------------------------
# No YAML anchors and aliases
# ---------------------------------------------------------------------------


class TestYamlWithoutAliases:
    """An object used twice is written in full each time (`&id001` / `*id001` are valid YAML, but hard to edit)."""

    @staticmethod
    def assert_no_aliases(text):
        assert "&id" not in text and "*id" not in text, text

    def test_dump_yaml_writes_shared_objects_in_full(self):
        from epftoolbox2.pipelines._yaml import dump_yaml

        shared_list, shared_dict = ["a", "b"], {"k": 1}
        data = {"first": shared_list, "second": shared_list, "d1": shared_dict, "d2": shared_dict}
        text = dump_yaml(data)

        self.assert_no_aliases(text)
        assert yaml.safe_load(text) == data
        assert list(yaml.safe_load(text)) == list(data)  # keys keep their order

    def test_data_pipeline_components_sharing_a_list(self, tmp_path):
        columns = ["price", "load"]
        pipeline = (
            DataPipeline()
            .add_transformer(LagTransformer(columns, [1], "day"))
            .add_transformer(LagTransformer(columns, [2], "day"))
        )
        path = tmp_path / "dp.yaml"
        pipeline.save(path)

        self.assert_no_aliases(path.read_text())
        assert [t.columns for t in DataPipeline.load(path).transformers] == [columns, columns]

    def test_data_pipeline_loaded_with_anchors_and_env_references(self, tmp_path, monkeypatch):
        # A `${VAR}` reference is written back as it was loaded: the same list object for both components here.
        monkeypatch.setenv("COLUMN", "price")
        source = tmp_path / "dp.yaml"
        source.write_text(
            "transformers:\n"
            "- class: LagTransformer\n"
            '  params: {columns: &c ["${COLUMN}", load], lags: [1], freq: day}\n'
            "- class: LagTransformer\n"
            "  params: {columns: *c, lags: [2], freq: day}\n"
        )
        saved = tmp_path / "saved.yaml"
        DataPipeline.load(source).save(saved)

        self.assert_no_aliases(saved.read_text())
        written = yaml.safe_load(saved.read_text())["transformers"]
        assert [t["params"]["columns"] for t in written] == [["${COLUMN}", "load"]] * 2
        assert [t.columns for t in DataPipeline.load(saved).transformers] == [["price", "load"]] * 2

    def test_workflow_loaded_with_anchors_in_its_inline_models(self, tmp_path):
        source = tmp_path / "experiment.yaml"
        source.write_text(
            "data_pipeline: {start: '2024-01-01', end: '2024-02-01'}\n"
            "model_pipeline:\n"
            "  test_start: '2024-01-20'\n"
            "  test_end: '2024-01-25'\n"
            "  models:\n"
            "  - class: OLSModel\n"
            "    params: {predictors: &p [price_d-1, 'load_d+{horizon}'], name: A}\n"
            "  - class: LassoCVModel\n"
            "    params: {predictors: *p, name: B}\n"
        )
        saved = tmp_path / "saved.yaml"
        Workflow.load(source).save(saved)

        self.assert_no_aliases(saved.read_text())
        models = Workflow.load(saved)._build_model_pipeline().models
        assert [m.name for m in models] == ["A", "B"]
        assert [m.predictors for m in models] == [["price_d-1", "load_d+{horizon}"]] * 2

    def test_workflow_saved_without_inline_sections_is_unchanged(self, tmp_path):
        out = tmp_path / "exp.yaml"
        Workflow(data_start="2024-01-01", data_end="2024-02-01", model_save_dir="results").save(out)
        config = yaml.safe_load(out.read_text())
        assert config["data_pipeline"]["start"] == "2024-01-01"
        assert config["model_pipeline"]["save_dir"] == "results"
        self.assert_no_aliases(out.read_text())

    def test_workflow_failed_save_leaves_an_existing_file_alone(self, tmp_path):
        out = tmp_path / "exp.yaml"
        out.write_text("keep: me\n")
        workflow = Workflow(data_start="2024-01-01", data_end="2024-02-01")
        workflow.model_save_dir = object()  # YAML cannot represent it
        with pytest.raises(yaml.YAMLError):
            workflow.save(out)
        assert out.read_text() == "keep: me\n"
