"""LambdaTransformer: custom columns/expressions, the index guard, and YAML serialization."""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.sources import CalendarSource
from epftoolbox2.data.transformers import LambdaTransformer
from epftoolbox2.pipelines import DataPipeline

TZ = "Europe/Warsaw"


def frame(n: int = 48) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz=TZ)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "load_forecast": rng.normal(20000, 1000, n),
            "wind_forecast": rng.uniform(0, 3000, n),
            "solar_forecast": rng.uniform(0, 2000, n),
        },
        index=index,
    )


# --------------------------------------------------------------------------- validation


def test_needs_exactly_one_of_func_or_columns():
    with pytest.raises(ValueError, match="exactly one"):
        LambdaTransformer()
    with pytest.raises(ValueError, match="exactly one"):
        LambdaTransformer(func=lambda df: df, columns={"a": "load_forecast"})


def test_columns_must_not_be_empty():
    with pytest.raises(ValueError, match="empty"):
        LambdaTransformer(columns={})


def test_columns_must_be_a_dict():
    with pytest.raises(TypeError, match="dict"):
        LambdaTransformer(columns=["load_forecast"])


def test_columns_values_must_be_str_or_callable():
    with pytest.raises(TypeError, match="string or a callable"):
        LambdaTransformer(columns={"a": 1.0})


def test_func_must_be_a_lambda_or_its_source():
    with pytest.raises(TypeError, match="lambda"):
        LambdaTransformer(func=42)
    with pytest.raises(ValueError, match="must be a lambda"):
        LambdaTransformer(func="not a lambda")
    with pytest.raises(ValueError, match="not valid Python"):
        LambdaTransformer(func="lambda df: df.assign(")


# --------------------------------------------------------------------------- columns: string expressions


def test_string_expression_is_same_row_arithmetic():
    df = frame()
    out = LambdaTransformer(columns={"residual_load": "load_forecast - wind_forecast - solar_forecast"}).transform(df)
    expected = df["load_forecast"] - df["wind_forecast"] - df["solar_forecast"]
    pd.testing.assert_series_equal(out["residual_load"], expected, check_names=False)
    # Original columns are untouched, new column appended.
    assert list(out.columns) == [*df.columns, "residual_load"]


def test_later_expression_can_reference_an_earlier_new_column():
    df = frame()
    df["installed_capacity"] = 30000.0
    out = LambdaTransformer(
        columns={
            "residual_load": "load_forecast - wind_forecast - solar_forecast",
            "residual_load_ratio": "residual_load / installed_capacity",
        }
    ).transform(df)
    expected = (df["load_forecast"] - df["wind_forecast"] - df["solar_forecast"]) / df["installed_capacity"]
    pd.testing.assert_series_equal(out["residual_load_ratio"], expected, check_names=False)


def test_string_expression_can_overwrite_an_existing_column():
    df = frame()
    out = LambdaTransformer(columns={"load_forecast": "load_forecast * 2"}).transform(df)
    pd.testing.assert_series_equal(out["load_forecast"], df["load_forecast"] * 2, check_names=False)


def test_backticked_column_name():
    df = frame().rename(columns={"load_forecast": "load forecast"})
    out = LambdaTransformer(columns={"x": "`load forecast` - wind_forecast"}).transform(df)
    expected = df["load forecast"] - df["wind_forecast"]
    pd.testing.assert_series_equal(out["x"], expected, check_names=False)


# --------------------------------------------------------------------------- columns: callables


def test_callable_expression():
    df = frame()
    out = LambdaTransformer(columns={"residual_load": lambda d: d["load_forecast"] - d["wind_forecast"]}).transform(df)
    pd.testing.assert_series_equal(out["residual_load"], df["load_forecast"] - df["wind_forecast"], check_names=False)


def test_callable_expression_sees_earlier_new_columns_too():
    df = frame()
    out = LambdaTransformer(
        columns={
            "a": "load_forecast - wind_forecast",
            "b": lambda d: d["a"] * 2,
        }
    ).transform(df)
    pd.testing.assert_series_equal(out["b"], out["a"] * 2, check_names=False)


# --------------------------------------------------------------------------- func


def test_func_full_control():
    df = frame()
    out = LambdaTransformer(func=lambda d: d.assign(spread=d["load_forecast"] - d["wind_forecast"])).transform(df)
    pd.testing.assert_series_equal(out["spread"], df["load_forecast"] - df["wind_forecast"], check_names=False)


def test_func_can_drop_columns():
    df = frame()
    out = LambdaTransformer(func=lambda d: d.drop(columns=["solar_forecast"])).transform(df)
    assert "solar_forecast" not in out.columns
    assert out.index.equals(df.index)


# --------------------------------------------------------------------------- index guard


def test_func_that_drops_rows_raises():
    df = frame()
    with pytest.raises(ValueError, match="index"):
        LambdaTransformer(func=lambda d: d.iloc[:-1]).transform(df)


def test_func_that_reorders_rows_raises():
    df = frame()
    with pytest.raises(ValueError, match="index"):
        LambdaTransformer(func=lambda d: d.iloc[::-1]).transform(df)


def test_func_that_resamples_raises():
    df = frame()
    with pytest.raises(ValueError, match="index"):
        LambdaTransformer(func=lambda d: d.resample("2h").mean()).transform(df)


# --------------------------------------------------------------------------- YAML serialization


def test_string_columns_round_trip(tmp_path):
    transformer = LambdaTransformer(columns={"residual_load": "load_forecast - wind_forecast - solar_forecast"})
    path = tmp_path / "dp.yaml"
    DataPipeline().add_transformer(transformer).save(path)

    loaded = DataPipeline.load(path).transformers[0]
    assert type(loaded) is LambdaTransformer
    assert loaded.func is None
    assert loaded.columns == transformer.columns


def _round_trip(transformer, tmp_path):
    path = tmp_path / "dp.yaml"
    DataPipeline().add_transformer(transformer).save(path)
    return path, DataPipeline.load(path).transformers[0]


def test_lambda_func_is_saved_as_source_text(tmp_path):
    transformer = LambdaTransformer(func=lambda df: df.assign(net=df["load_forecast"] - df["wind_forecast"]))
    path, loaded = _round_trip(transformer, tmp_path)

    assert "lambda df: df.assign(net=" in path.read_text()
    pd.testing.assert_frame_equal(loaded.transform(frame()), transformer.transform(frame()))


def test_lambda_in_columns_is_saved_as_source_text(tmp_path):
    transformer = LambdaTransformer(columns={"net": lambda df: df["load_forecast"] - df["wind_forecast"]})
    path, loaded = _round_trip(transformer, tmp_path)

    assert loaded.columns == {"net": 'lambda df: df["load_forecast"] - df["wind_forecast"]'}
    pd.testing.assert_frame_equal(loaded.transform(frame()), transformer.transform(frame()))


def test_lambda_text_can_use_numpy_and_pandas(tmp_path):
    transformer = LambdaTransformer(
        func="lambda df: df.assign(log_load=np.log(df['load_forecast']), hour=pd.Series(df.index.hour, df.index))"
    )
    _, loaded = _round_trip(transformer, tmp_path)
    out = loaded.transform(frame())
    np.testing.assert_allclose(out["log_load"], np.log(frame()["load_forecast"]))
    assert (out["hour"] == frame().index.hour).all()


def test_lambda_closing_over_a_variable_raises_on_save(tmp_path):
    base = 18.0
    transformer = LambdaTransformer(columns={"hdd": lambda df: (base - df["load_forecast"]).clip(lower=0)})
    with pytest.raises(TypeError, match="callable"):
        DataPipeline().add_transformer(transformer).save(tmp_path / "dp.yaml")


def test_named_function_raises_on_save(tmp_path):
    def net(df):
        return df["load_forecast"] - df["wind_forecast"]

    transformer = LambdaTransformer(columns={"net": net})
    with pytest.raises(TypeError, match="callable"):
        DataPipeline().add_transformer(transformer).save(tmp_path / "dp.yaml")


def test_two_different_lambdas_on_one_line_are_each_found(tmp_path):
    transformer = LambdaTransformer(
        columns={"a": lambda df: df["load_forecast"] * 2, "b": lambda df: df["wind_forecast"] + 1}
    )
    path, loaded = _round_trip(transformer, tmp_path)
    pd.testing.assert_frame_equal(loaded.transform(frame()), transformer.transform(frame()))


# --------------------------------------------------------------------------- DataPipeline integration


def test_data_pipeline_integration():
    pipeline = (
        DataPipeline()
        .add_source(CalendarSource(country="PL", weekday="number", holidays=False))
        .add_transformer(LambdaTransformer(columns={"weekend": "weekday >= 5"}))
    )
    df = pipeline.run("2024-01-01", "2024-01-08")
    assert "weekend" in df.columns
    assert (df["weekend"] == (df["weekday"] >= 5)).all()
