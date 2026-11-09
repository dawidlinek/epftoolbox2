"""Daily-feature transformers: profiles and daily stats."""

import warnings

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.transformers import (
    DailyProfileTransformer,
    DailyStatsTransformer,
    DSTNormalizeTransformer,
    LagTransformer,
)
from epftoolbox2.pipelines import DataPipeline

TZ = "Europe/Warsaw"


def local_frame(start: str, end: str, freq: str = "1h", seed: int = 0) -> pd.DataFrame:
    index = pd.date_range(start, end, freq=freq, tz=TZ, inclusive="left")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"price": rng.normal(50, 10, len(index)), "load": rng.normal(1000, 50, len(index))}, index=index
    )


def local(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts, tz=TZ)


def assert_rows_up_to_day_unchanged(transformer, df: pd.DataFrame, issue_day: str, columns=None):
    """Changing every value dated after the issue day must not change any feature on rows up to that day."""
    day = pd.Timestamp(issue_day).date()
    perturbed = df.copy()
    later = df.index.date > day
    perturbed.loc[later, perturbed.select_dtypes("number").columns] *= 3.0
    perturbed.loc[later, perturbed.select_dtypes("number").columns] += 1000.0
    a, b = transformer.transform(df), transformer.transform(perturbed)
    new = [c for c in a.columns if c not in df.columns] if columns is None else columns
    upto = a.index.date <= day
    pd.testing.assert_frame_equal(a.loc[upto, new], b.loc[upto, new])


# --------------------------------------------------------------------------- DailyProfileTransformer


def test_profile_copies_the_days_values_onto_every_row():
    df = local_frame("2024-01-01", "2024-01-04")
    out = DailyProfileTransformer(columns="price").transform(df)

    names = [f"price_h{h:02d}" for h in range(24)]
    assert list(out.columns) == ["price", "load", *names]
    for day in ["2024-01-01", "2024-01-02", "2024-01-03"]:
        rows = out.loc[day]
        expected = df.loc[day, "price"].to_numpy()
        assert (rows[names].to_numpy() == expected).all()


def test_profile_of_quarter_hours_uses_slot_names():
    df = local_frame("2024-01-01", "2024-01-03", freq="15min")
    out = DailyProfileTransformer(columns="price", freq="15min").transform(df)
    assert "price_s00" in out.columns and "price_s95" in out.columns
    assert out.loc[local("2024-01-02 17:30"), "price_s05"] == df.loc[local("2024-01-02 01:15"), "price"]


def test_profile_on_dst_days_warns_and_leaves_skipped_slot_empty():
    df = local_frame("2024-03-30", "2024-04-02")
    with pytest.warns(UserWarning, match="DSTNormalizeTransformer"):
        out = DailyProfileTransformer(columns="price").transform(df)
    assert out.loc["2024-03-31", "price_h02"].isna().all()
    assert out.loc[local("2024-03-31 12:00"), "price_h03"] == df.loc[local("2024-03-31 03:00"), "price"]

    autumn = local_frame("2024-10-26", "2024-10-29")
    with pytest.warns(UserWarning):
        out = DailyProfileTransformer(columns="price").transform(autumn)
    first = autumn.loc[pd.Timestamp("2024-10-27 02:00+02:00"), "price"]
    assert (out.loc["2024-10-27", "price_h02"] == first).all()


def test_profile_after_dst_normalisation_is_complete_without_warning():
    df = DSTNormalizeTransformer().transform(local_frame("2024-03-30", "2024-04-02"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = DailyProfileTransformer(columns="price").transform(df)
    assert out.loc["2024-03-31", [f"price_h{h:02d}" for h in range(24)]].notna().all().all()


def test_profile_is_leak_free():
    df = local_frame("2024-10-20", "2024-11-03")
    for day in ["2024-10-26", "2024-10-27", "2024-10-28"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            assert_rows_up_to_day_unchanged(DailyProfileTransformer(columns=["price", "load"]), df, day)


def test_profile_rejects_off_grid_data_and_name_clashes():
    with pytest.raises(ValueError, match="grid"):
        DailyProfileTransformer(columns="price").transform(local_frame("2024-01-01", "2024-01-02", freq="15min"))
    df = DailyProfileTransformer(columns="price").transform(local_frame("2024-01-01", "2024-01-03"))
    with pytest.raises(ValueError, match="price_h00"):
        DailyProfileTransformer(columns="price").transform(df)


# --------------------------------------------------------------------------- DailyStatsTransformer


def test_daily_stats_per_local_day():
    df = local_frame("2024-01-01", "2024-01-04")
    out = DailyStatsTransformer(columns="price").transform(df)

    day = df.loc["2024-01-02", "price"]
    rows = out.loc["2024-01-02"]
    assert (rows["price_day_min"] == day.min()).all()
    assert (rows["price_day_max"] == day.max()).all()
    assert np.allclose(rows["price_day_mean"], day.mean())
    assert (rows["price_day_last"] == day.iloc[-1]).all()


def test_daily_stats_only_on_complete_days():
    # UTC bounds: the first local day starts at 01:00, so it is incomplete; so is a day with a missing value.
    index = pd.date_range("2024-01-01", "2024-01-05", freq="1h", tz="UTC", inclusive="left").tz_convert(TZ)
    df = pd.DataFrame({"price": np.arange(len(index), dtype=float)}, index=index)
    df.loc[local("2024-01-03 05:00"), "price"] = np.nan

    out = DailyStatsTransformer(columns="price", stats=["max"]).transform(df)

    assert out.loc["2024-01-01", "price_day_max"].isna().all()
    assert out.loc["2024-01-03", "price_day_max"].isna().all()
    assert out.loc["2024-01-02", "price_day_max"].notna().all()


def test_daily_stats_count_dst_days_as_complete():
    for start, end, day in [("2024-03-30", "2024-04-02", "2024-03-31"), ("2024-10-26", "2024-10-29", "2024-10-27")]:
        df = local_frame(start, end)
        out = DailyStatsTransformer(columns="price", stats=["mean", "last"]).transform(df)
        assert np.allclose(out.loc[day, "price_day_mean"], df.loc[day, "price"].mean())
        assert (out.loc[day, "price_day_last"] == df.loc[local(f"{day} 23:00"), "price"]).all()


def test_daily_stats_are_leak_free():
    df = local_frame("2024-10-20", "2024-11-03", freq="15min")
    stats = ["min", "max", "mean", "last", "peak"]
    for day in ["2024-10-26", "2024-10-27"]:
        assert_rows_up_to_day_unchanged(DailyStatsTransformer(["price", "load"], stats, freq="15min"), df, day)


def test_daily_stats_reject_unknown_stat():
    with pytest.raises(ValueError, match="median"):
        DailyStatsTransformer(columns="price", stats=["median"])


def _peak_oracle(df: pd.DataFrame, column: str, slots_per_hour: int, first: int = 8, end: int = 20) -> pd.Series:
    """The mean over the rows of each local date whose wall-clock hour is in [first, end), on dates with every slot the
    clock gives them (23, 24 or 25 hours), each with a value; NaN on the others."""
    wall = df.index.tz_localize(None)
    out = pd.Series(np.nan, index=df.index)
    for day in wall.normalize().unique():
        on = np.asarray(wall.normalize() == day)
        hours = 24 + {pd.Timestamp("2024-03-31"): -1, pd.Timestamp("2024-10-27"): 1}.get(day, 0)
        values = df.loc[on, column]
        if len(values) == hours * slots_per_hour and values.notna().all():
            out[on] = values[(wall[on].hour >= first) & (wall[on].hour < end)].mean()
    return out


@pytest.mark.parametrize("freq", ["1h", "15min"])
@pytest.mark.parametrize("peak_hours", [(8, 20), (0, 24), (17, 21), (2, 3)])
def test_daily_peak_is_the_mean_over_the_peak_hours_of_whole_days_across_both_switches(freq, peak_hours):
    per_hour = 4 if freq == "15min" else 1
    for start, end in [("2024-03-30", "2024-04-02"), ("2024-10-26", "2024-10-29")]:
        index = pd.date_range(start, end, freq=freq, tz="UTC", inclusive="left").tz_convert(TZ)  # partial first day
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"price": rng.normal(50, 10, len(index))}, index=index)
        # The whole day after the switch misses a slot outside the peak block: only the switch day is whole.
        df.loc[pd.Timestamp(end, tz=TZ) - pd.Timedelta(hours=21), "price"] = np.nan
        out = DailyStatsTransformer(
            columns="price", stats=["peak", "mean"], freq=freq, peak_hours=peak_hours
        ).transform(df)
        want = _peak_oracle(df, "price", per_hour, *peak_hours)
        np.testing.assert_allclose(out["price_day_peak"], want, rtol=0, atol=1e-12)
        assert np.array_equal(out["price_day_peak"].isna(), want.isna())
        spring = start.endswith("30")
        # 02:00 to 02:59 is no hour of the spring switch day and two of the autumn one.
        whole = 0 if spring and peak_hours == (2, 3) else (23 if spring else 25) * per_hour
        assert out["price_day_peak"].notna().sum() == whole
        if peak_hours == (0, 24):
            np.testing.assert_allclose(out["price_day_peak"], out["price_day_mean"], rtol=0, atol=1e-12)


def test_daily_peak_is_asked_for_and_the_default_statistics_are_as_before():
    df = local_frame("2024-01-01", "2024-01-04")
    assert list(DailyStatsTransformer(columns="price").transform(df).columns) == [
        "price",
        "load",
        "price_day_min",
        "price_day_max",
        "price_day_mean",
        "price_day_last",
    ]
    out = DailyStatsTransformer(columns="price", stats=["peak"]).transform(df)
    day = df.loc["2024-01-02", "price"]
    assert np.allclose(out.loc["2024-01-02", "price_day_peak"], day.between_time("08:00", "19:59").mean())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"stats": ["peak"], "peak_hours": (20, 8)}, "peak_hours"),
        ({"stats": ["peak"], "peak_hours": (8, 25)}, "peak_hours"),
        ({"stats": ["peak"], "peak_hours": (8.0, 20.0)}, "peak_hours"),
        ({"stats": ["peak"], "peak_hours": 8}, "peak_hours"),
        ({"stats": ["max"], "peak_hours": (7, 19)}, "means nothing"),
    ],
)
def test_daily_peak_hours_are_checked(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DailyStatsTransformer(columns="price", **kwargs)


def test_peak_hours_are_written_only_when_set_and_a_file_without_them_loads(tmp_path):
    """A pipeline saved now with the default block loads in a version from before `peak_hours`, and one written then loads now."""
    DataPipeline().add_transformer(DailyStatsTransformer(columns="price", stats=["min", "peak"])).save(
        tmp_path / "a.yaml"
    )
    assert "peak_hours" not in (tmp_path / "a.yaml").read_text()
    DataPipeline().add_transformer(DailyStatsTransformer("price", ["peak"], peak_hours=(7, 19))).save(
        tmp_path / "b.yaml"
    )
    assert "peak_hours" in (tmp_path / "b.yaml").read_text()
    old = tmp_path / "old.yaml"
    old.write_text(
        "transformers:\n- class: DailyStatsTransformer\n  params:\n    columns: [x]\n    stats: [min, max]\n"
    )
    loaded = DataPipeline.load(old).transformers[0]
    assert (loaded.stats, loaded.peak_hours) == (["min", "max"], (8, 20))


def test_daily_peak_block_is_the_report_s():
    from epftoolbox2.data.transformers._daily import PEAK_HOURS
    from epftoolbox2.results.breakdowns import PEAK_HOURS as REPORT_PEAK_HOURS

    assert PEAK_HOURS == REPORT_PEAK_HOURS == DailyStatsTransformer(columns="x").peak_hours


# --------------------------------------------------------------------------- UTC warnings and YAML


@pytest.mark.parametrize(
    "transformer",
    [
        DailyProfileTransformer(columns="price"),
        DailyStatsTransformer(columns="price"),
    ],
)
def test_local_day_features_warn_on_utc_index(transformer):
    index = pd.date_range("2024-01-01", periods=24 * 40, freq="1h", tz="UTC")
    df = pd.DataFrame({"price": np.arange(len(index), dtype=float)}, index=index)
    with pytest.warns(UserWarning, match="UTC"):
        transformer.transform(df)


@pytest.mark.parametrize(
    "transformer",
    [
        DailyProfileTransformer(columns=["price", "load"], freq="15min"),
        DailyStatsTransformer(columns="price", stats=["min", "last"], freq="15min"),
        DailyStatsTransformer(columns="price", stats=["peak", "max"], peak_hours=[7, 19]),
    ],
)
def test_yaml_round_trip(tmp_path, transformer):
    path = tmp_path / "dp.yaml"
    DataPipeline().add_transformer(transformer).save(path)
    loaded = DataPipeline.load(path).transformers[0]
    assert type(loaded) is type(transformer)
    assert vars(loaded) == vars(transformer)


def test_the_lag_of_a_day_statistic_is_not_the_statistic_of_a_lag_across_dst():
    """What the docs tell users: `x_d-1_day_max` summarises the lag over the rows of D, so the skipped spring 02:00 makes
    the day after that switch incomplete, and the day after the autumn switch leaves out the second 02:00, here the
    day's maximum. `x_day_max_d-1` lags the finished statistic and only misses its own 02:00 row in spring."""
    index = pd.date_range("2024-03-20", "2024-11-05", freq="h", tz="Europe/Warsaw", inclusive="left")
    values = np.full(len(index), 10.0)
    values[index.get_loc(pd.Timestamp("2024-10-27 02:00:00+01:00"))] = 99.0
    df = LagTransformer(columns="x", lags=[1], freq="day").transform(pd.DataFrame({"x": values}, index=index))
    df = DailyStatsTransformer(columns=["x", "x_d-1"], stats=["max"]).transform(df)
    df = LagTransformer(columns="x_day_max", lags=[1], freq="day").transform(df)

    spring, autumn = df.loc["2024-04-01"], df.loc["2024-10-28"]

    assert spring["x_d-1_day_max"].isna().all()
    assert spring["x_day_max_d-1"].isna().sum() == 1
    assert pd.isna(spring.loc[local("2024-04-01 02:00"), "x_day_max_d-1"])
    assert (autumn["x_d-1_day_max"] == 10.0).all()
    assert (autumn["x_day_max_d-1"] == 99.0).all()
