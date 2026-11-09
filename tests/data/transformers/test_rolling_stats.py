"""RollingStatsTransformer (step 7e): a statistic of a column's complete days over a window of local days ending `lag_days`
before the row's, against a day-by-day oracle and the renewables study's capacity proxy, over both clock changes, hourly and
every 15 minutes; it never reads after the window's newest day."""

import warnings

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.transformers import RollingStatsTransformer
from epftoolbox2.pipelines import DataPipeline

TZ = "Europe/Warsaw"


def local_frame(start, days, freq="1h", seed=0, zone=TZ):
    index = pd.date_range(start, pd.Timestamp(start) + pd.Timedelta(days=days), freq=freq, tz=zone, inclusive="left")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"gen": rng.uniform(0, 100, len(index))}, index=index)


def oracle(df, column, window, stat, lag, freq, min_days=None):
    """Day by day: a day's statistic where every slot its clock has holds a value, then the window's over the days that have
    one, NaN where fewer than `min_days` (all `window` days by default) do: a day before the frame's first has none."""
    wall = df.index.tz_localize(None)
    days = wall.normalize()
    step = pd.Timedelta(freq)
    per_day = {}
    for day in days.unique():
        values = df.loc[days == day, column]
        start = pd.Timestamp(day).tz_localize(df.index.tz, ambiguous=True, nonexistent="shift_forward")
        end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(
            df.index.tz, ambiguous=True, nonexistent="shift_forward"
        )
        if values.notna().sum() == (end - start) // step:
            per_day[day] = getattr(values, stat)()
    out = np.full(len(df), np.nan)
    for i, day in enumerate(days):
        newest = day - pd.Timedelta(days=lag)
        oldest = newest - pd.Timedelta(days=window - 1)
        held = [per_day[d] for d in pd.date_range(oldest, newest, freq="D") if d in per_day]
        if len(held) >= (window if min_days is None else min_days):
            out[i] = getattr(pd.Series(held), stat)()
    return out


@pytest.mark.parametrize(
    "freq, stat, window, lag, switch, min_days",
    [
        ("1h", "max", 7, 1, "2024-03-31", None),
        ("1h", "min", 3, 0, "2024-10-27", 1),
        ("1h", "mean", 5, 2, "2024-10-27", 3),
        ("15min", "max", 5, 0, "2024-03-31", 2),
        ("15min", "mean", 7, 1, "2024-10-27", None),
        ("15min", "min", 1, 1, "2024-03-31", None),
        ("1h", "max", 4, 1, "2024-10-27", 4),
    ],
)
def test_equals_a_day_by_day_oracle_over_a_switch_with_gaps(freq, stat, window, lag, switch, min_days):
    """Rows removed (a day without them is incomplete), NaN values and a whole day missing: none of those days counts, and a
    window with fewer than `min_days` days that do, every day of it by default, is NaN, as one that reaches before the frame's
    first day is. The switch days count with 23 or 25 hours (92 or 100 quarter-hours)."""
    start = pd.Timestamp(switch) - pd.Timedelta(days=10)
    df = local_frame(start, 20, freq=freq, seed=3)
    per_hour = 4 if freq == "15min" else 1
    df.iloc[30 * per_hour, 0] = np.nan  # day 2 incomplete by a NaN
    df = df.drop(df.index[70 * per_hour])  # day 3 incomplete by a missing row
    df = df[df.index.tz_localize(None).normalize() != start + pd.Timedelta(days=13)]  # a whole day gone
    transformer = RollingStatsTransformer(
        "gen", window_days=window, stat=stat, lag_days=lag, min_days=min_days, freq=freq
    )
    out = transformer.transform(df)
    name = f"gen_roll{window}{stat}" + (f"_d-{lag}" if lag else "")
    assert list(out.columns) == ["gen", name]
    expected = oracle(df, "gen", window, stat, lag, freq, min_days)
    np.testing.assert_allclose(out[name].to_numpy(), expected, rtol=1e-12, equal_nan=True)
    assert np.isfinite(expected).any()


def test_by_default_one_incomplete_day_takes_the_value_of_every_window_it_falls_in():
    """No "365-day" maximum of what is left: a missing hour on 5 January leaves the 7-day windows from 5 to 11 January without a
    value, which `min_days=6` keeps from the other six days (from 7 January, the first window with six complete days)."""
    df = local_frame("2024-01-01", 20, seed=1)
    df.loc[pd.Timestamp("2024-01-05 13:00", tz=TZ), "gen"] = np.nan
    strict = RollingStatsTransformer("gen", 7, lag_days=0).transform(df)["gen_roll7max"]
    by_day = strict.groupby(strict.index.tz_localize(None).normalize()).first()
    assert by_day.isna().tolist() == [True] * 11 + [False] * 9  # 1-6 Jan: before the frame; 5-11 Jan: the gap
    loose = RollingStatsTransformer("gen", 7, lag_days=0, min_days=6).transform(df)["gen_roll7max"]
    by_day = loose.groupby(loose.index.tz_localize(None).normalize()).first()
    assert by_day.isna().tolist() == [True] * 6 + [False] * 14


def test_the_switch_days_are_complete_with_their_own_number_of_hours():
    """Every hour of the 23- and 25-hour days holds a value: both days count (the repeated hour's two values both in)."""
    df = local_frame("2024-10-25", 5)
    df.loc[pd.Timestamp("2024-10-27 02:00+01:00"), "gen"] = 1000.0  # the second 02:00
    out = RollingStatsTransformer("gen", window_days=1, lag_days=0).transform(df)
    assert (out.loc["2024-10-27", "gen_roll1max"] == 1000.0).all()
    spring = RollingStatsTransformer("gen", window_days=1, lag_days=0).transform(local_frame("2024-03-30", 3))
    assert spring.loc["2024-03-31", "gen_roll1max"].notna().all() and len(spring.loc["2024-03-31"]) == 23


@pytest.mark.parametrize("lag", [0, 1, 3])
def test_it_never_reads_after_the_window_s_newest_day(lag):
    """Every value from day D - lag + 1 on multiplied and shifted: the rows of D and before keep their value."""
    df = local_frame("2024-03-20", 20, seed=5)
    day = pd.Timestamp("2024-04-02")
    later = df.index.tz_localize(None).normalize() > day - pd.Timedelta(days=lag)
    perturbed = df.copy()
    perturbed.loc[later, "gen"] = perturbed.loc[later, "gen"] * 7 + 500
    transformer = RollingStatsTransformer("gen", window_days=4, lag_days=lag)
    a, b = transformer.transform(df), transformer.transform(perturbed)
    upto = a.index.tz_localize(None).normalize() <= day
    pd.testing.assert_series_equal(a.loc[upto, transformer.output_column], b.loc[upto, transformer.output_column])
    # ... and the day after does move: the window's newest day is read.
    after = a.index.tz_localize(None).normalize() == day + pd.Timedelta(days=1)
    assert (a.loc[after, transformer.output_column] != b.loc[after, transformer.output_column]).any()


def test_nan_until_the_window_is_complete():
    df = local_frame("2024-01-01", 12)
    out = RollingStatsTransformer("gen", window_days=5, lag_days=1).transform(df)["gen_roll5max_d-1"]
    by_day = out.groupby(out.index.tz_localize(None).normalize()).first()
    assert by_day.iloc[:5].isna().all() and by_day.iloc[5:].notna().all()  # D-5..D-1 needs 1 January as D-5


def study_cap_proxy(act):
    """The renewables study's `gen_models.cap_proxy`, verbatim: cap[t] = max of daily-max actual over days t-366 .. t-2 (365
    days ending D-1 = T-2), indexed by the target day T."""
    dmax = pd.Series(np.nanmax(np.where(np.isfinite(act), act, -np.inf), axis=1))
    dmax[~np.isfinite(dmax)] = np.nan
    return dmax.shift(2).rolling(365, min_periods=60).max().to_numpy()


def test_the_capacity_proxy_is_the_renewables_study_s():
    """On a two-year solar-like series (growing, with whole hours missing), the column with `min_days=60` on the row of issue
    day D equals the study's `cap_proxy` of the target day T = D + 1 on every day, to the bit, NaN where it is NaN (before 60
    complete days). The study's 24-hour grid is
    built from the frame as its `to_grid` does it: the hour the spring switch skips as the mean of its neighbours, the
    repeated autumn hour as the mean of its two, which never move a day's maximum of a series that is 0 at night."""
    index = pd.date_range("2023-01-01", "2025-01-01", freq="1h", tz="Europe/Berlin", inclusive="left")
    rng = np.random.default_rng(11)
    hour = index.hour.to_numpy()
    days_in = (index.tz_localize(None).normalize() - pd.Timestamp("2023-01-01")).days.to_numpy()
    sun = np.clip(np.sin(np.pi * (hour - 5) / 15), 0, None)
    gen = (40_000 + 30 * days_in) * sun * rng.uniform(0.2, 1.0, len(index))
    df = pd.DataFrame({"generation_solar": gen}, index=index)
    df.iloc[rng.choice(len(df), 40, replace=False), 0] = np.nan  # incomplete days, left out by both
    proxy = RollingStatsTransformer("generation_solar", 365, "max", 1, min_days=60)  # the study's min_periods
    out = proxy.transform(df)["generation_solar_roll365max_d-1"]

    wall = index.tz_localize(None)
    days = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    grid = np.full((len(days), 24), np.nan)
    frame = pd.DataFrame({"v": df["generation_solar"].to_numpy(), "day": wall.normalize(), "hour": wall.hour})
    hourly = frame.groupby(["day", "hour"])["v"].mean()  # the repeated hour: the mean of its two
    grid[days.get_indexer(hourly.index.get_level_values(0)), hourly.index.get_level_values(1)] = hourly.to_numpy()
    # The skipped spring hour is NaN here (the study's mean of its neighbours is below their maximum), and a day with a
    # missing value of its own, incomplete for the transformer, is left out of the study's grid as a whole.
    grid[days.isin(wall.normalize()[df["generation_solar"].isna().to_numpy()])] = np.nan
    cap = study_cap_proxy(grid)  # by target day T

    per_day = out.groupby(wall.normalize()).first()  # by issue day D, the same value on every row of it
    issue = days[:-1]
    ours = per_day.reindex(issue).to_numpy()
    theirs = cap[1:]  # T = D + 1
    np.testing.assert_array_equal(ours, theirs)  # NaN where the other is
    assert np.isnan(ours[:61]).all() and np.isfinite(ours[365:]).all()


def test_names_and_lag_0():
    df = local_frame("2024-01-01", 3)
    assert RollingStatsTransformer("gen", 2, "mean", 0).transform(df).columns[-1] == "gen_roll2mean"
    assert RollingStatsTransformer("gen", 30, "min", 2).output_column == "gen_roll30min_d-2"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"column": ""}, "column"),
        ({"column": "gen", "window_days": 0}, "window_days"),
        ({"column": "gen", "window_days": 1.5}, "window_days"),
        ({"column": "gen", "window_days": True}, "window_days"),
        ({"column": "gen", "stat": "last"}, "stat"),
        ({"column": "gen", "lag_days": -1}, "lag_days"),
        ({"column": "gen", "window_days": 30, "min_days": 0}, "min_days"),
        ({"column": "gen", "window_days": 30, "min_days": 31}, "min_days"),
        ({"column": "gen", "window_days": 30, "min_days": 2.0}, "min_days"),
        ({"column": "gen", "freq": "7min"}, "whole slots"),
    ],
)
def test_values_that_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RollingStatsTransformer(**kwargs)


def test_frames_that_are_refused_or_warned():
    transformer = RollingStatsTransformer("gen", 2)
    with pytest.raises(ValueError, match="not in the frame"):
        transformer.transform(local_frame("2024-01-01", 2).rename(columns={"gen": "other"}))
    with pytest.raises(ValueError, match="grid"):
        transformer.transform(local_frame("2024-01-01", 1, freq="15min"))
    with pytest.raises(ValueError, match="already exist"):
        transformer.transform(transformer.transform(local_frame("2024-01-01", 3)))
    with pytest.warns(UserWarning, match="UTC days"):
        transformer.transform(local_frame("2024-01-01", 3, zone="UTC"))


def test_an_empty_frame_gets_an_empty_column():
    df = local_frame("2024-01-01", 2).iloc[:0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = RollingStatsTransformer("gen", 2).transform(df)
    assert list(out.columns) == ["gen", "gen_roll2max_d-1"] and out.empty


def test_yaml_round_trip(tmp_path):
    transformer = RollingStatsTransformer("generation_solar", 90, "mean", 2, min_days=60, freq="15min")
    pipeline = DataPipeline().add_transformer(transformer)
    path = tmp_path / "rolling.yaml"
    pipeline.save(path)
    loaded = DataPipeline.load(path).transformers[0]
    assert isinstance(loaded, RollingStatsTransformer)
    assert vars(loaded) == vars(pipeline.transformers[0])


def test_a_daily_frame_counts_the_switch_days_as_complete():
    """One row a day, `freq="1D"`: the day of the spring switch is a complete day of one row, so only the first row of the frame,
    whose window reaches before it, is NaN."""
    index = pd.date_range("2024-03-20", "2024-11-05", freq="D", tz="Europe/Berlin")
    df = pd.DataFrame({"x": np.arange(len(index), dtype=float)}, index=index)
    out = RollingStatsTransformer("x", window_days=1, stat="max", lag_days=1, freq="1D").transform(df)["x_roll1max_d-1"]
    assert out.isna().tolist() == [True] + [False] * (len(index) - 1)
    assert (out.iloc[1:].to_numpy() == df["x"].iloc[:-1].to_numpy()).all()


@pytest.mark.parametrize("freq", ["15min", "30min"])
def test_rows_further_apart_than_freq_are_refused(freq):
    """Hourly data read with a shorter `freq` would count every day as incomplete (a quarter of its slots): refused, as data
    off the grid is."""
    with pytest.raises(ValueError, match="no two rows are closer"):
        RollingStatsTransformer("gen", 2, freq=freq).transform(local_frame("2024-06-01", 3))


def test_two_daily_rows_across_the_autumn_switch_are_a_day_apart():
    index = pd.DatetimeIndex([pd.Timestamp("2024-10-27", tz=TZ), pd.Timestamp("2024-10-28", tz=TZ)])  # 25 hours apart
    out = RollingStatsTransformer("gen", 1, lag_days=0, freq="1D").transform(
        pd.DataFrame({"gen": [1.0, 2.0]}, index=index)
    )
    assert out["gen_roll1max"].tolist() == [1.0, 2.0]
