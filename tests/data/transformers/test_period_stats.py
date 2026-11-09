"""`PeriodStatsTransformer`: a column's mean over each hour of its local clock, on every row of the hour, and each row's
deviation from it.

The oracle groups the rows by their local wall-clock hour and UTC offset, a key the transformer doesn't use, so the hour the
clocks repeat in autumn is two hours of four quarter-hours (one per offset) and the hour they skip in spring has no rows. It is
checked on 15-minute (and 30-minute and hourly) data across both switches, in zones a whole hour and half an hour from UTC.
"""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.transformers import DSTNormalizeTransformer, PeriodStatsTransformer
from epftoolbox2.pipelines import DataPipeline
from epftoolbox2.pipelines.data_pipeline import COMPONENT_REGISTRY

SWITCHES = {"spring": "2024-03-31", "autumn": "2024-10-27"}


def frame(day: str, zone: str = "Europe/Warsaw", freq: str = "15min", seed: int = 0, days: int = 3) -> pd.DataFrame:
    """Random values on whole local days around `day`, with a few missing."""
    first = pd.Timestamp(day) - pd.Timedelta(days=1)
    index = pd.date_range(
        first.tz_localize(zone), (first + pd.Timedelta(days=days)).tz_localize(zone), freq=freq, inclusive="left"
    )
    rng = np.random.default_rng(seed)
    values = rng.normal(50, 10, len(index))
    values[rng.choice(len(index), size=3, replace=False)] = np.nan
    return pd.DataFrame({"x": values, "y": rng.normal(0, 1, len(index))}, index=index)


def oracle(df: pd.DataFrame, column: str, slots: int) -> pd.DataFrame:
    """The hour's mean and each row's deviation, by (local wall-clock hour, UTC offset): NaN unless the hour has `slots` rows,
    all with a value."""
    keys = [(ts.replace(minute=0, second=0, tzinfo=None), ts.utcoffset()) for ts in df.index]
    series = pd.Series(df[column].to_numpy(), index=pd.MultiIndex.from_tuples(keys))
    means = {}
    for key, values in series.groupby(level=[0, 1]):
        means[key] = values.mean() if len(values) == slots and values.notna().all() else np.nan
    mean = np.array([means[key] for key in keys])
    return pd.DataFrame({"mean": mean, "dev": df[column].to_numpy() - mean}, index=df.index)


@pytest.mark.parametrize("switch", sorted(SWITCHES))
@pytest.mark.parametrize(
    ("zone", "freq"), [("Europe/Warsaw", "15min"), ("Europe/Lisbon", "30min"), ("Asia/Kolkata", "15min")]
)
def test_hour_means_and_deviations_equal_the_oracle(switch, zone, freq):
    df = frame(SWITCHES[switch], zone, freq)
    out = PeriodStatsTransformer(["x", "y"], freq=freq).transform(df)

    assert list(out.columns) == ["x", "y", "x_hour_mean", "x_hour_dev", "y_hour_mean", "y_hour_dev"]
    pd.testing.assert_frame_equal(out[["x", "y"]], df)
    slots = 60 // int(pd.Timedelta(freq).total_seconds() // 60)
    for column in ("x", "y"):
        want = oracle(df, column, slots)
        np.testing.assert_allclose(out[f"{column}_hour_mean"], want["mean"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(out[f"{column}_hour_dev"], want["dev"], rtol=0, atol=1e-12)
        assert np.array_equal(out[f"{column}_hour_mean"].isna(), want["mean"].isna())
    assert out["x_hour_mean"].isna().any() and out["y_hour_mean"].notna().all()  # a missing value takes its hour alone


def test_the_repeated_autumn_hour_is_two_hours_of_four_and_the_skipped_one_has_no_rows():
    autumn = frame(SWITCHES["autumn"])
    autumn["x"] = np.arange(len(autumn), dtype=float)
    out = PeriodStatsTransformer("x").transform(autumn)
    wall = out.index.tz_localize(None)
    repeated = out[(wall >= pd.Timestamp("2024-10-27 02:00")) & (wall < pd.Timestamp("2024-10-27 03:00"))]
    assert len(repeated) == 8
    first, second = repeated.iloc[:4], repeated.iloc[4:]
    assert (first["x_hour_mean"] == first["x"].mean()).all()
    assert (second["x_hour_mean"] == second["x"].mean()).all()
    assert first["x_hour_mean"].iloc[0] != second["x_hour_mean"].iloc[0]
    assert list(first["x_hour_dev"]) == [-1.5, -0.5, 0.5, 1.5] == list(second["x_hour_dev"])

    spring = frame(SWITCHES["spring"])
    spring["x"] = np.arange(len(spring), dtype=float)
    out = PeriodStatsTransformer("x").transform(spring)
    day = out.loc["2024-03-31"]
    assert len(day) == 92 and not (day.index.hour == 2).any()
    assert day["x_hour_mean"].notna().all()  # 01:00 and 03:00 are whole hours, the switch between them
    assert (day.loc[day.index.hour == 3, "x_hour_dev"].to_numpy() == [-1.5, -0.5, 0.5, 1.5]).all()


def test_a_partial_hour_at_either_end_and_an_hour_with_a_missing_value_get_nan():
    index = pd.date_range("2024-01-10 10:30", "2024-01-10 14:15", freq="15min", tz="Europe/Warsaw", inclusive="left")
    df = pd.DataFrame({"x": np.arange(len(index), dtype=float)}, index=index)
    df.loc[pd.Timestamp("2024-01-10 12:15", tz="Europe/Warsaw"), "x"] = np.nan
    out = PeriodStatsTransformer("x").transform(df)
    hours = out.index.hour
    assert out.loc[hours.isin([10, 12, 14]), ["x_hour_mean", "x_hour_dev"]].isna().all().all()
    assert out.loc[hours.isin([11, 13]), ["x_hour_mean", "x_hour_dev"]].notna().all().all()


def test_hourly_data_has_its_value_as_the_mean_and_no_deviation():
    df = frame(SWITCHES["autumn"], freq="1h")
    out = PeriodStatsTransformer("x", freq="1h").transform(df)
    np.testing.assert_array_equal(out["x_hour_mean"], df["x"])
    assert (out["x_hour_dev"].dropna() == 0).all()


def test_a_naive_index_after_dst_normalisation_has_four_slots_every_hour():
    df = frame(SWITCHES["autumn"]).fillna(1.0)
    naive = DSTNormalizeTransformer(freq="15min").transform(df)
    out = PeriodStatsTransformer("x").transform(naive)
    assert out["x_hour_mean"].notna().all()
    blocks = out["x"].to_numpy().reshape(-1, 4)
    np.testing.assert_allclose(
        out["x_hour_mean"].to_numpy().reshape(-1, 4), blocks.mean(axis=1, keepdims=True).repeat(4, 1)
    )


def test_rows_up_to_an_hour_never_read_a_later_hour():
    df = frame(SWITCHES["autumn"])
    transformer = PeriodStatsTransformer(["x", "y"])
    base = transformer.transform(df)
    for end in ["2024-10-27 02:00+02:00", "2024-10-27 02:00+01:00", "2024-10-27 03:00+01:00", "2024-10-26 14:00+02:00"]:
        cut = pd.Timestamp(end)
        perturbed = df.copy()
        perturbed.loc[perturbed.index >= cut] = perturbed.loc[perturbed.index >= cut] * 3 + 100
        changed = transformer.transform(perturbed)
        new = ["x_hour_mean", "x_hour_dev", "y_hour_mean", "y_hour_dev"]
        pd.testing.assert_frame_equal(base.loc[base.index < cut, new], changed.loc[changed.index < cut, new])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"period": "1D"}, "period must be '1h'"),
        ({"period": "an hour"}, "period must be '1h'"),
        ({"freq": "2h"}, "freq must divide the period"),
        ({"freq": "25min"}, "freq must divide"),
        ({"stats": ["median"]}, "median"),
        ({"stats": []}, "non-empty"),
    ],
)
def test_bad_parameters_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PeriodStatsTransformer("x", **kwargs)


def test_it_rejects_data_off_its_grid_name_clashes_and_a_plain_index():
    df = frame(SWITCHES["autumn"])
    with pytest.raises(ValueError, match="15min grid"):
        PeriodStatsTransformer("x").transform(df.set_axis(df.index + pd.Timedelta(minutes=5)))
    with pytest.raises(ValueError, match="already exist"):
        PeriodStatsTransformer("x").transform(df.assign(x_hour_mean=1.0))
    with pytest.raises(ValueError, match="DatetimeIndex"):
        PeriodStatsTransformer("x").transform(df.reset_index(drop=True))


def test_a_utc_index_warns_and_a_duplicated_timestamp_is_refused_as_in_daily_stats():
    """A UTC index groups UTC hours, which in Kolkata (UTC+05:30) are 10:30 to 11:29 local: the information set, reading the
    frame's local hours, would take a realised series as known half an hour early, so it warns as DailyStatsTransformer does.
    A timestamp twice could make an hour with a missing slot look whole: both transformers refuse it."""
    from epftoolbox2.data.transformers import DailyStatsTransformer

    df = frame(SWITCHES["autumn"])
    with pytest.warns(UserWarning, match="index is UTC"):
        PeriodStatsTransformer("x").transform(df.tz_convert("UTC"))
    twice = pd.concat([df.drop(df.index[5]), df.iloc[[4]]]).sort_index()
    for transformer in (PeriodStatsTransformer("x"), DailyStatsTransformer("x", freq="15min")):
        with pytest.raises(ValueError, match="1 duplicate timestamp"):
            transformer.transform(twice)


def test_data_coarser_than_freq_is_refused_rather_than_all_nan():
    """Hourly data with the default freq="15min" has no hour of four slots: an error, not a column of NaN. With its own freq it
    runs; a column without any value still runs (its hours have their rows), NaN as it should be."""
    hourly = frame(SWITCHES["spring"], freq="1h")
    with pytest.raises(ValueError, match="no hour of the data has its 4 slots of 15min"):
        PeriodStatsTransformer("x").transform(hourly)
    assert PeriodStatsTransformer("x", freq="1h").transform(hourly)["x_hour_mean"].notna().any()
    empty = frame(SWITCHES["spring"]).assign(x=np.nan)
    assert PeriodStatsTransformer("x").transform(empty)["x_hour_mean"].isna().all()
    assert PeriodStatsTransformer("x").transform(hourly.iloc[:0].asfreq("15min")).empty


def test_one_statistic_alone_and_the_columns_in_order():
    df = frame(SWITCHES["spring"])
    assert list(PeriodStatsTransformer("x", stats="dev").transform(df).columns) == ["x", "y", "x_hour_dev"]
    assert list(PeriodStatsTransformer(["y", "x"], stats=["mean"]).transform(df).columns)[2:] == [
        "y_hour_mean",
        "x_hour_mean",
    ]


def test_it_is_registered_and_round_trips_through_yaml(tmp_path):
    assert COMPONENT_REGISTRY["transformers"]["PeriodStatsTransformer"] == "epftoolbox2.data.transformers.period_stats"
    transformer = PeriodStatsTransformer(["x", "y"], stats=["dev"], freq="30min")
    path = tmp_path / "dp.yaml"
    DataPipeline().add_transformer(transformer).save(path)
    loaded = DataPipeline.load(path).transformers[0]
    assert type(loaded) is PeriodStatsTransformer
    assert vars(loaded) == vars(transformer)


def test_the_tui_offers_it_with_its_parameters():
    import inspect

    from epftoolbox2.tui import PARAM_SCHEMAS, _TRANSFORMERS

    assert "PeriodStatsTransformer" in _TRANSFORMERS
    fields = [field["name"] for field in PARAM_SCHEMAS["PeriodStatsTransformer"]]
    assert fields == list(inspect.signature(PeriodStatsTransformer).parameters)
