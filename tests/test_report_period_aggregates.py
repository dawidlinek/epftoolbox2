"""Hourly and block aggregates, the within-hour shape, the error split, normalised MAEs and `by_column()`.

Every answer is counted by hand or computed by a pandas oracle from the records, across both clock switches of
Europe/Berlin: the spring day has 23 hours (92 quarter-hours, the hour 02:00 skipped), the day after it lacks the hour no
forecast was issued from, and the autumn day repeats 02:00, which a run writes once with no actual.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.evaluators import (
    BiasEvaluator,
    EconomicEvaluator,
    MAEEvaluator,
    NMAEEvaluator,
    RMSEEvaluator,
    ScaledMAEEvaluator,
    ShapeMAEEvaluator,
)
from epftoolbox2.models._calendar import clock_periods
from epftoolbox2.pipelines.model_pipeline import ModelPipeline
from epftoolbox2.results.ref import ModelResultRef
from epftoolbox2.results.report import EvaluationReport, metric_columns

pytestmark = pytest.mark.filterwarnings("ignore:The comparison rests on")

ZONE = "Europe/Berlin"
SPRING, AUTUMN = "2024-03-31", "2024-10-27"
# Run dates: a plain day, the spring switch, two plain days, the autumn switch, a plain day. Horizon 1, so the targets are
# the day after each: the spring switch day, the day after it, ..., the autumn switch day, the day after it.
RUNS = ["2024-03-30", "2024-03-31", "2024-10-25", "2024-10-26", "2024-10-27"]
REPEATED_HOUR = 2  # 02:00 local, skipped in spring and repeated in autumn


def run_ids(run, target, freq):
    """The period ids a real run writes for (run date, target date): those the clock has on both days."""
    on_run = set(clock_periods(run, freq, ZONE))
    return [p for p in clock_periods(target, freq, ZONE) if p in on_run]


def actual_of(target, period):
    return 40.0 + 0.7 * period + 5 * np.sin(period / 3.0) + pd.Timestamp(target).dayofyear / 10


def records(freq="15min", error=lambda target, period: 1.0, runs=RUNS, drop=(), with_freq=True):
    """One model's records at horizon 1. `error(target, period)` is prediction minus actual; `drop` holds (target, period)
    keys the model lacks. The repeated autumn hour has no actual, as a run writes it."""
    per_hour = 4 if freq == "15min" else 1
    out = []
    for i, run in enumerate(runs):
        target = (pd.Timestamp(run) + pd.Timedelta(days=1)).date().isoformat()
        for period in run_ids(run, target, freq):
            if (target, period) in drop:
                continue
            actual = actual_of(target, period)
            prediction = actual + error(target, period)
            if target == AUTUMN and period // per_hour == REPEATED_HOUR:
                actual = None
            row = {
                "run_date": run,
                "target_date": target,
                "run_weekday": pd.Timestamp(run).dayofweek,
                "hour": period,
                "horizon": 1,
                "day_in_test": i,
                "prediction": prediction,
                "actual": actual,
            }
            if with_freq:
                row["freq"] = freq
            out.append(row)
    return out


def zoned(name, recs, freq="15min", zone=ZONE, horizon=1):
    return ModelResultRef(
        name=name,
        path=None,
        count=len(recs),
        test_start="",
        test_end="",
        horizon=horizon,
        freq=freq,
        _results=recs,
        zone=zone,
    )


def report_of(models, evaluators=(MAEEvaluator(),), freq="15min", zone=ZONE, source_data=None, horizon=1):
    refs = {name: zoned(name, recs, freq, zone, horizon) if zone else recs for name, recs in models.items()}
    return EvaluationReport(refs, list(evaluators), source_data=source_data)


def finite_frame(recs):
    df = pd.DataFrame(recs)
    df["actual"] = df["actual"].astype(float)
    return df


def hourly_oracle(recs):
    """Hours with all four quarter-hours finite, reduced by pandas: (kept, seen)."""
    df = finite_frame(recs)
    df["h"] = df["hour"] // 4
    keys = ["run_date", "target_date", "horizon", "h"]
    seen = df.groupby(keys).ngroups
    finite = df[np.isfinite(df["prediction"]) & np.isfinite(df["actual"])]
    g = finite.groupby(keys)
    out = g[["prediction", "actual"]].mean()[g.size() == 4].reset_index()
    return out.rename(columns={"h": "hour"}), seen


def as_frame(report, name):
    return pd.DataFrame(list(report.refs[name].iter_records())).sort_values(["run_date", "target_date", "hour"])


# ---------------------------------------------------------------------------------------------------------------------
# aggregate("1h")
# ---------------------------------------------------------------------------------------------------------------------

# Hours each target day has, by hand: the spring day 23 (02:00 skipped); the day after it 23 (no forecast was issued from
# 02:00 of the spring day); two plain days 24; the autumn day 24, of which 02:00 (no actual) is dropped; the day after 24.
HOURS_SEEN = 23 + 23 + 24 + 24 + 24
HOURS_KEPT = HOURS_SEEN - 1


@pytest.mark.parametrize("zone", [ZONE, None], ids=["zone", "plain lists"])
def test_hourly_aggregate_counts_hours_by_hand_across_both_switches(zone):
    hole = ("2024-10-26", 37)  # one quarter-hour of 09:00 on a plain day
    full, holey = records(), records(drop={hole})
    report = report_of({"full": full, "holey": holey}, zone=zone)

    with pytest.warns(UserWarning, match=r"left out hours .*'full': 1, 'holey': 2.*incomplete_hour"):
        hourly = report.aggregate("1h", agg="mean")

    counts = hourly.forecast_counts().set_index("model")
    assert counts.loc["full", ["planned", "produced", "missing", "incomplete_hour"]].tolist() == [
        HOURS_SEEN,
        HOURS_KEPT,
        1,
        1,
    ]
    # The hole costs the holey model its 09:00 alone; the full model keeps it.
    assert counts.loc["holey", ["planned", "produced", "missing", "incomplete_hour"]].tolist() == [
        HOURS_SEEN,
        HOURS_KEPT - 1,
        2,
        2,
    ]
    for name, recs in (("full", full), ("holey", holey)):
        got = as_frame(hourly, name)
        expected, seen = hourly_oracle(recs)
        assert seen == HOURS_SEEN
        expected = expected.sort_values(["run_date", "target_date", "hour"])
        assert (
            got[["run_date", "target_date", "hour"]].values.tolist()
            == expected[["run_date", "target_date", "hour"]].values.tolist()
        )
        np.testing.assert_allclose(got["prediction"], expected["prediction"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(got["actual"], expected["actual"], rtol=0, atol=1e-12)
        assert set(got["freq"]) == {"1h"}
    autumn = as_frame(hourly, "full").query("target_date == @AUTUMN")
    assert REPEATED_HOUR not in autumn["hour"].tolist() and len(autumn) == 23
    spring = as_frame(hourly, "full").query("target_date == @SPRING")
    assert REPEATED_HOUR not in spring["hour"].tolist() and len(spring) == 23
    assert hourly.summary().set_index("model").loc["holey", "forecasts"] == HOURS_KEPT - 1


def test_hourly_report_keeps_the_zone_so_days_are_judged_by_the_clock_again():
    report = report_of({"m": records()})
    with pytest.warns(UserWarning, match="incomplete_hour"):
        hourly = report.aggregate("1h", agg="mean")
    assert hourly._result_freqs() == {"m": "1h"}
    assert hourly.refs["m"].zone == ZONE
    with pytest.warns(UserWarning, match="incomplete_day"):
        daily = hourly.aggregate("1D", agg="mean")
    # Whole by the clock: the spring day (23 hours) and the plain days; not the day after spring (lacks 02:00) nor the
    # autumn day (its 02:00 was dropped).
    assert as_frame(daily, "m")["target_date"].tolist() == ["2024-03-31", "2024-10-26", "2024-10-28"]


def test_hourly_aggregate_of_max_min_and_spread_matches_pandas():
    recs = records(error=lambda target, period: np.cos(period))
    report = report_of({"m": recs})
    df = finite_frame(recs)
    df = df[np.isfinite(df["actual"])]
    df["h"] = df["hour"] // 4
    groups = df.groupby(["run_date", "target_date", "h"])
    for agg, reduce in (
        ("max", lambda g: g.max()),
        ("min", lambda g: g.min()),
        ("spread", lambda g: g.max() - g.min()),
    ):
        with pytest.warns(UserWarning):
            got = as_frame(report.aggregate("1h", agg=agg), "m")
        expected = reduce(groups["prediction"]).reset_index().sort_values(["run_date", "target_date", "h"])
        np.testing.assert_allclose(got["prediction"].to_numpy(), expected["prediction"].to_numpy(), atol=1e-12)


def test_hourly_aggregate_refuses_what_it_cannot_build():
    hourly = report_of({"m": records(freq="1h")}, freq="1h")
    with pytest.raises(ValueError, match="needs 15-minute results"):
        hourly.aggregate("1h")
    quarter = report_of({"m": records()})
    with pytest.raises(ValueError, match="block of a day"):
        quarter.aggregate("1h", agg="peak")
    with pytest.raises(ValueError, match="freq must be '1D' or '1h'"):
        quarter.aggregate("1W")
    with pytest.raises(ValueError, match="peak_hours sets the block"):
        quarter.aggregate("1D", agg="max", peak_hours=(7, 19))
    for bad in ((20, 8), (8,), (8.5, 20), (0, 25), "8-20"):
        with pytest.raises(ValueError, match="peak_hours must be"):
            quarter.aggregate("1D", agg="peak", peak_hours=bad)
    with pytest.raises(ValueError, match="no offpeak hours"):
        quarter.aggregate("1D", agg="offpeak", peak_hours=(0, 24))


def test_hourly_aggregate_keeps_the_battery_and_leaves_source_readers_out():
    report = report_of({"m": records()}, evaluators=[MAEEvaluator(), EconomicEvaluator(), ScaledMAEEvaluator("cap")])
    with pytest.warns(UserWarning) as caught:
        hourly = report.aggregate("1h")
    assert [type(e).__name__ for e in hourly.evaluators] == ["MAEEvaluator", "EconomicEvaluator"]
    assert any("ScaledMAEEvaluator" in str(w.message) for w in caught)
    with pytest.warns(UserWarning) as caught:
        daily = report.aggregate("1D", agg="peak")
    assert [type(e).__name__ for e in daily.evaluators] == ["MAEEvaluator"]


# ---------------------------------------------------------------------------------------------------------------------
# aggregate("1D", agg="peak" | "offpeak")
# ---------------------------------------------------------------------------------------------------------------------


def block_oracle(recs, agg, per_hour, peak_hours, day_ids):
    """Mean over the block of each forecast day, on the days with every slot of the block finite, by pandas."""
    df = finite_frame(recs)
    hour = df["hour"] // per_hour
    peak = (hour >= peak_hours[0]) & (hour < peak_hours[1])
    df = df[peak if agg == "peak" else ~peak]
    rows = []
    for (run, target, horizon), g in df.groupby(["run_date", "target_date", "horizon"]):
        wanted = {p for p in day_ids(target) if (peak_hours[0] <= p // per_hour < peak_hours[1]) == (agg == "peak")}
        g = g[np.isfinite(g["prediction"]) & np.isfinite(g["actual"])]
        if set(g["hour"]) == wanted:
            rows.append((run, target, g["prediction"].mean(), g["actual"].mean()))
    return pd.DataFrame(rows, columns=["run_date", "target_date", "prediction", "actual"])


@pytest.mark.parametrize("freq", ["1h", "15min"])
@pytest.mark.parametrize("zone", [ZONE, None], ids=["zone", "plain lists"])
@pytest.mark.parametrize("agg", ["peak", "offpeak"])
def test_peak_and_offpeak_match_a_pandas_oracle(freq, zone, agg):
    per_hour = 4 if freq == "15min" else 1
    hole_peak = ("2024-10-26", 9 * per_hour + (1 if per_hour == 4 else 0))  # 09:00 or 09:15
    recs = {"full": records(freq), "holey": records(freq, drop={hole_peak})}
    report = report_of(recs, freq=freq, zone=zone)

    if zone:
        day_ids = lambda target: clock_periods(target, freq, ZONE)  # noqa: E731
    else:
        every = pd.DataFrame(recs["full"] + recs["holey"]).groupby("target_date")["hour"]
        on_date = {date: set(ids) for date, ids in every}
        day_ids = lambda target: on_date[target]  # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aggregated = report.aggregate("1D", agg=agg)
    for name, model in recs.items():
        got = as_frame(aggregated, name)
        expected = block_oracle(model, agg, per_hour, (8, 20), day_ids)
        assert got["target_date"].tolist() == expected["target_date"].tolist()
        np.testing.assert_allclose(got["prediction"], expected["prediction"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(got["actual"], expected["actual"], rtol=0, atol=1e-12)

    kept = {name: as_frame(aggregated, name)["target_date"].tolist() for name in recs}
    if agg == "peak":
        # Every day's peak is whole, the switch days' too; the hole in 09:00 costs the holey model that day alone.
        assert kept["full"] == ["2024-03-31", "2024-04-01", "2024-10-26", "2024-10-27", "2024-10-28"]
        assert kept["holey"] == ["2024-03-31", "2024-04-01", "2024-10-27", "2024-10-28"]
    else:
        # The spring day is whole over its clock; the day after it lacks 02:00 with a zone (and without one, every model
        # lacks it, so it is taken at its word); the autumn day's 02:00 has no actual.
        day_after = ["2024-04-01"] if zone is None else []
        assert kept["full"] == ["2024-03-31", *day_after, "2024-10-26", "2024-10-28"]
        assert kept["holey"] == kept["full"]
    counts = aggregated.forecast_counts().set_index("model")
    assert (counts["planned"] == 5).all()
    assert counts.loc["full", "incomplete_day"] == 5 - len(kept["full"])


def test_peak_hours_set_the_block():
    recs = records(
        "1h", error=lambda target, period: float(period)
    )  # the error is the hour, so the mean says the block
    report = report_of({"m": recs}, freq="1h")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        peak = as_frame(report.aggregate("1D", agg="peak", peak_hours=(10, 12)), "m")
        off = as_frame(report.aggregate("1D", agg="offpeak", peak_hours=(0, 23)), "m")
    np.testing.assert_allclose(peak["prediction"] - peak["actual"], 10.5)
    np.testing.assert_allclose(off["prediction"] - off["actual"], 23.0)


def test_existing_daily_aggregates_are_unchanged_by_the_block_path():
    recs = records("1h", error=lambda target, period: np.sin(period))
    report = report_of({"m": recs}, freq="1h")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean = as_frame(report.aggregate("1D", agg="mean"), "m")
        whole_day_block = as_frame(report.aggregate("1D", agg="peak", peak_hours=(0, 24)), "m")
    # A peak block of the whole day is the daily mean on the days whole for both rules.
    both = mean.merge(whole_day_block, on=["run_date", "target_date"])
    np.testing.assert_allclose(both["prediction_x"], both["prediction_y"], rtol=0, atol=1e-12)
    assert set(mean["target_date"]) <= set(whole_day_block["target_date"])


# ---------------------------------------------------------------------------------------------------------------------
# ShapeMAEEvaluator
# ---------------------------------------------------------------------------------------------------------------------

SHAPE = (1.0, -1.0, 2.0, -2.0)  # within-hour error pattern, mean 0: its shape MAE is mean(|SHAPE|) = 1.5


def level_hour_shape(target, period):
    """An error made of a daily level, an hour profile and a within-hour shape: only the last is shape error."""
    day = pd.Timestamp(target).dayofyear % 7
    return 3.0 + day + 4.0 * np.sin(period // 4) + SHAPE[period % 4]


def test_shape_mae_measures_only_the_shape_within_the_hour():
    report = report_of({"m": records(error=level_hour_shape)}, evaluators=[ShapeMAEEvaluator(), MAEEvaluator()])
    value = report.summary().set_index("model").loc["m", "ShapeMAE"]
    assert value == pytest.approx(1.5, abs=1e-12)
    # A deviation from the daily mean (a mutant) would carry the hour profile, which is far from 1.5.
    df = finite_frame(records(error=level_hour_shape))
    df = df[np.isfinite(df["actual"])]
    e = df["prediction"] - df["actual"]
    daily = (e - e.groupby([df["target_date"]]).transform("mean")).abs().mean()
    assert abs(daily - 1.5) > 0.5


def test_shape_mae_skips_the_repeated_hour_and_hours_with_a_hole():
    def error(target, period):
        if target == AUTUMN and period // 4 == REPEATED_HOUR:
            return 100.0 * SHAPE[period % 4]  # no actual: must not count
        if target == "2024-10-26" and period // 4 == 9:
            return 50.0 * SHAPE[period % 4]  # loses a quarter-hour below: must not count
        return SHAPE[period % 4]

    recs = records(error=error, drop={("2024-10-26", 37)})
    report = report_of({"m": recs}, evaluators=[ShapeMAEEvaluator()])
    assert report.summary().loc[0, "ShapeMAE"] == pytest.approx(1.5, abs=1e-12)


def test_shape_mae_oracle_on_irregular_errors():
    rng = np.random.default_rng(3)
    noise = {(t, p): rng.normal(0, 3) for t in ["2024-04-01", "2024-03-31"] for p in range(96)}
    recs = records(error=lambda t, p: noise.get((t, p), 0.5 * p), runs=RUNS[:2])
    report = report_of({"m": recs}, evaluators=[ShapeMAEEvaluator()])
    df = finite_frame(recs)
    df["h"] = df["hour"] // 4
    g = df.groupby(["target_date", "h"])
    dev = (df["prediction"] - g["prediction"].transform("mean")) - (df["actual"] - g["actual"].transform("mean"))
    assert report.summary().loc[0, "ShapeMAE"] == pytest.approx(dev.abs().mean(), abs=1e-12)


def test_shape_mae_is_nan_without_quarter_hours():
    hourly = report_of({"m": records("1h")}, evaluators=[ShapeMAEEvaluator()], freq="1h")
    assert np.isnan(hourly.summary().loc[0, "ShapeMAE"])
    quarter = report_of({"m": records()}, evaluators=[ShapeMAEEvaluator()])
    assert quarter.by_hour()["ShapeMAE"].isna().all()  # one quarter-hour of each hour per group
    assert quarter.by_year()["ShapeMAE"].notna().all()
    with pytest.warns(UserWarning):
        assert np.isnan(quarter.aggregate("1h").summary().loc[0, "ShapeMAE"])
    with pytest.raises(ValueError, match="period must be '1h'"):
        ShapeMAEEvaluator(period="1D")


def test_shape_mae_called_directly_reads_the_period_ids():
    frame = finite_frame(records(error=level_hour_shape))
    frame["run_date"] = pd.to_datetime(frame["run_date"])
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    assert ShapeMAEEvaluator().compute(frame) == pytest.approx(1.5, abs=1e-12)


# ---------------------------------------------------------------------------------------------------------------------
# NMAEEvaluator and ScaledMAEEvaluator
# ---------------------------------------------------------------------------------------------------------------------


def test_nmae_is_divided_by_the_mean_actual_on_the_same_keys():
    recs = records("1h", error=lambda t, p: 2.0 if p % 2 else -1.0)
    for row in recs[:10]:
        row["prediction"] = None  # no forecast there: neither its error nor its actual counts
        row["actual"] = 10_000.0
    report = report_of({"m": recs}, evaluators=[NMAEEvaluator(), MAEEvaluator()], freq="1h")
    df = finite_frame(recs)
    df["prediction"] = df["prediction"].astype(float)
    keys = df[np.isfinite(df["prediction"]) & np.isfinite(df["actual"])]
    expected = (keys["prediction"] - keys["actual"]).abs().mean() / keys["actual"].mean()
    row = report.summary().iloc[0]
    assert row["nMAE"] == pytest.approx(expected, rel=1e-14)
    assert row["nMAE"] == pytest.approx(row["MAE"] / keys["actual"].mean(), rel=1e-14)
    every_actual = (keys["prediction"] - keys["actual"]).abs().mean() / df["actual"].mean()
    assert abs(row["nMAE"] - every_actual) > 1e-3  # what dividing by every actual (a mutant) would give


def hourly_source(columns, start="2024-03-29", end="2024-10-30"):
    index = pd.date_range(pd.Timestamp(start, tz=ZONE), pd.Timestamp(end, tz=ZONE), freq="1h", inclusive="left")
    return pd.DataFrame({name: fill(index) for name, fill in columns.items()}, index=index)


def first_repeated(index):
    """True on the first 02:00 of the autumn day (still summer time), the row a run doesn't forecast from."""
    offsets = np.array([ts.utcoffset() for ts in index])
    local = index.tz_localize(None)
    return (
        (local.normalize() == pd.Timestamp(AUTUMN)) & (local.hour == REPEATED_HOUR) & (offsets == pd.Timedelta(hours=2))
    )


def test_scaled_mae_divides_each_error_by_the_source_value_at_its_target_row():
    recs = records("1h", error=lambda t, p: 1.0 + p)
    autumn_two = next(r for r in recs if r["target_date"] == AUTUMN and r["hour"] == REPEATED_HOUR)
    autumn_two["actual"] = 50.0  # scored here, so the joined row shows
    source = hourly_source({"cap": lambda idx: 100.0 + idx.hour.to_numpy()})
    local = source.index.tz_localize(None)
    source.loc[first_repeated(source.index), "cap"] = 1.0  # the first 02:00 would give a huge error/cap
    source.loc[local == pd.Timestamp("2024-10-26 05:00"), "cap"] = 0.0
    source.loc[local == pd.Timestamp("2024-10-26 06:00"), "cap"] = np.nan
    report = report_of({"m": recs}, evaluators=[ScaledMAEEvaluator("cap")], freq="1h", source_data=source)

    row = report.summary().iloc[0]
    df = finite_frame(recs)
    df = df[np.isfinite(df["actual"])]
    cap = 100.0 + df["hour"].astype(float)  # the second 02:00 has the plain value
    scaled = ~((df["target_date"] == "2024-10-26") & df["hour"].isin([5, 6]))
    expected = ((df["prediction"] - df["actual"]).abs() / cap)[scaled].mean()
    assert row["MAE/cap"] == pytest.approx(expected, rel=1e-13)
    assert row["MAE/cap left out"] == 2
    assert {"MAE/cap", "MAE/cap left out"} <= set(metric_columns(report.summary()))
    assert report.compute_all()["hour"].set_index("hour").loc[5, "MAE/cap left out"] == 1


def test_scaled_mae_needs_source_data_with_its_column():
    recs = records("1h")
    with pytest.raises(ValueError, match="report's source data, and this report has none"):
        report_of({"m": recs}, evaluators=[ScaledMAEEvaluator("cap")], freq="1h").summary()
    source = hourly_source({"other": lambda idx: np.ones(len(idx))})
    with pytest.raises(ValueError, match="no column"):
        report_of({"m": recs}, evaluators=[ScaledMAEEvaluator("cap")], freq="1h", source_data=source).summary()
    quarter = hourly_source({"cap": lambda idx: np.ones(len(idx))})
    with pytest.raises(ValueError, match="at that resolution"):
        report_of({"m": records()}, evaluators=[ScaledMAEEvaluator("cap")], source_data=quarter).summary()
    with pytest.raises(ValueError, match="name of a source-data column"):
        ScaledMAEEvaluator("")


# ---------------------------------------------------------------------------------------------------------------------
# by_column()
# ---------------------------------------------------------------------------------------------------------------------


def test_by_column_groups_by_a_flag_joined_at_the_second_occurrence_of_the_repeated_hour():
    recs = records("1h", error=lambda t, p: 1.0 + (p % 3))

    def flags(index):
        local = index.tz_localize(None)
        flag = np.where((local.hour >= 6) & (local.hour < 12), 0.0, 1.0)
        flag[first_repeated(index)] = 0.0
        return flag

    source = hourly_source({"flag": flags})
    report = report_of({"m": recs}, evaluators=[MAEEvaluator(), BiasEvaluator()], freq="1h", source_data=source)

    table = report.by_column("flag")
    assert table["flag"].tolist() == [0.0, 1.0]
    df = finite_frame(recs)
    flag = np.where((df["hour"] >= 6) & (df["hour"] < 12), 0.0, 1.0)
    # The autumn 02:00 record takes the second 02:00 row (flag 1); the first occurrence (a mutant) would put it in 0.
    assert table.set_index("flag")["forecasts"].to_dict() == {0.0: int((flag == 0).sum()), 1.0: int((flag == 1).sum())}
    finite = np.isfinite(df["actual"])
    for value in (0.0, 1.0):
        part = df[(flag == value) & finite]
        expected = (part["prediction"] - part["actual"]).abs().mean()
        assert table.set_index("flag").loc[value, "MAE"] == pytest.approx(expected, rel=1e-14)
    assert table["forecasts"].sum() == report.summary().loc[0, "forecasts"]
    assert table.attrs["no_value"] == {}
    assert metric_columns(table) == ["MAE", "Bias"]


def test_by_column_cuts_a_numeric_column_into_bins_closed_on_the_left():
    recs = records("1h")
    source = hourly_source({"price": lambda idx: idx.hour.to_numpy() - 5.0})
    local = source.index.tz_localize(None)
    source.loc[local == pd.Timestamp("2024-10-26 13:00"), "price"] = np.nan
    report = report_of({"a": recs, "b": recs}, freq="1h", source_data=source)

    table = report.by_column("price", bins=[-np.inf, 0, 10, 20])  # 19:00-23:00 are 14-18: in [10, 20)
    labels = table.drop_duplicates("price")["price"].tolist()
    assert [(i.left, i.right, i.closed) for i in labels] == [(-np.inf, 0, "left"), (0, 10, "left"), (10, 20, "left")]
    counts = table[table["model"] == "a"].set_index("price")["forecasts"]
    df = finite_frame(recs)
    hours = df["hour"]
    assert counts.tolist() == [
        int((hours < 5).sum()),
        int(((hours >= 5) & (hours < 15)).sum()) - 1,  # the NaN price at 13:00 on one day is in no group
        int((hours >= 15).sum()),
    ]
    assert table.attrs["no_value"] == {"a": 1, "b": 1}
    assert counts.sum() + 1 == len(recs)
    with pytest.raises(ValueError, match="list of edges"):
        report.by_column("price", bins=4)
    with pytest.raises(ValueError, match="increasing edges"):
        report.by_column("price", bins=[1, 1])


def test_by_column_refuses_what_it_cannot_join():
    recs = records("1h")
    with pytest.raises(ValueError, match="source data, and this report has none"):
        report_of({"m": recs}, freq="1h").by_column("flag")
    source = hourly_source({"hour": lambda idx: idx.hour.to_numpy()})
    with pytest.raises(ValueError, match="already use that name"):
        report_of({"m": recs}, freq="1h", source_data=source).by_column("hour")


# ---------------------------------------------------------------------------------------------------------------------
# error_split()
# ---------------------------------------------------------------------------------------------------------------------


def split_oracle(recs):
    """The resolution study's decomposition (`evaluate.py`), on the days with 96 finite quarter-hours."""
    df = finite_frame(recs)
    df["e"] = df["prediction"] - df["actual"]
    days = [g.sort_values("hour")["e"].to_numpy() for _, g in df.groupby("target_date")]
    ef = np.array([d for d in days if len(d) == 96 and np.isfinite(d).all()])
    lvl = ef.mean(1, keepdims=True)
    ehf = ef.reshape(-1, 24, 4).mean(2)
    total = (ef**2).mean()
    return dict(
        days=len(ef),
        level=(lvl**2).mean() / total,
        hour=((ehf - lvl) ** 2).mean() / total,
        quarter=((ef.reshape(-1, 24, 4) - ehf[:, :, None]) ** 2).mean() / total,
    )


def test_error_split_is_the_studys_decomposition_on_whole_non_switch_days():
    recs = records(error=level_hour_shape)
    table = report_of({"m": recs}).error_split()
    row = table.iloc[0]
    expected = split_oracle(recs)
    # Whole non-switch days: the spring day has 92 quarter-hours, the day after it lacks 02:00, the autumn day's 02:00 has
    # no actual; the plain day before autumn and the day after it count.
    assert row["days"] == expected["days"] == 2
    assert row["forecasts"] == 2 * 96
    for part in ("level", "hour", "quarter"):
        assert row[part] == pytest.approx(expected[part], abs=1e-12)
    assert row["level"] + row["hour"] + row["quarter"] == pytest.approx(1.0, abs=1e-12)
    assert list(table.columns) == ["model", "days", "forecasts", "mse", "level", "hour", "quarter"]


def test_error_split_leaves_out_switch_days_by_the_clock():
    recs = records(error=level_hour_shape)
    for row in recs:
        if row["actual"] is None:
            row["actual"] = 30.0  # a plain list may hold a finite actual in the repeated hour
    with_zone = report_of({"m": recs}).error_split().iloc[0]
    without = report_of({"m": recs}, zone=None).error_split().iloc[0]
    assert with_zone["days"] == 2
    assert without["days"] == 3  # without a zone nothing tells the autumn day from a plain one


def test_error_split_warns_when_models_have_different_days_and_refuses_hourly_results():
    full = records(error=level_hour_shape)
    holey = records(error=level_hour_shape, drop={("2024-10-26", 37)})
    with pytest.warns(UserWarning, match="different numbers of whole days"):
        table = report_of({"full": full, "holey": holey}).error_split()
    assert table.attrs["unequal_days"] == {"full": 2, "holey": 1}
    with pytest.raises(ValueError, match="needs 15-minute results"):
        report_of({"m": records("1h")}, freq="1h").error_split()
    with pytest.raises(ValueError, match="period must be '1h'"):
        report_of({"m": full}).error_split(period="1D")


# ---------------------------------------------------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------------------------------------------------


def test_new_evaluators_round_trip_through_yaml(tmp_path):
    original = ModelPipeline().add_evaluator(
        ShapeMAEEvaluator(), NMAEEvaluator(), ScaledMAEEvaluator("installed_capacity"), RMSEEvaluator()
    )
    original.save(tmp_path / "mp.yaml")
    text = (tmp_path / "mp.yaml").read_text()
    assert "name:" not in text  # the column name of ScaledMAEEvaluator is a property, not a parameter
    loaded = ModelPipeline.load(tmp_path / "mp.yaml")
    shape, nmae, scaled, _ = loaded.evaluators
    assert isinstance(shape, ShapeMAEEvaluator) and shape.period == "1h"
    assert isinstance(nmae, NMAEEvaluator)
    assert isinstance(scaled, ScaledMAEEvaluator) and scaled.column == "installed_capacity"
    assert scaled.name == "MAE/installed_capacity"


def test_exporters_take_the_new_columns(tmp_path, capsys):
    from openpyxl import load_workbook

    from epftoolbox2.exporters import CsvExporter, ExcelExporter, TerminalExporter

    source = hourly_source({"cap": lambda idx: 100.0 + idx.hour.to_numpy()})
    evaluators = [MAEEvaluator(), NMAEEvaluator(), ScaledMAEEvaluator("cap"), ShapeMAEEvaluator()]
    report = report_of(
        {"a": records("1h"), "b": records("1h", error=lambda t, p: 2.0)}, evaluators, "1h", source_data=source
    )

    ExcelExporter(str(tmp_path / "out.xlsx"), sheets=["summary", "hour"]).export(report)
    names = load_workbook(tmp_path / "out.xlsx", read_only=True).sheetnames
    assert {"Summary", "hour_MAE_cap", "hour_MAE_cap left out", "hour_nMAE"} <= set(names)  # "/" is no sheet character
    CsvExporter(str(tmp_path / "out.csv"), summary_path=str(tmp_path / "summary.csv")).export(report)
    assert list(pd.read_csv(tmp_path / "summary.csv").columns)[2:] == [
        "MAE",
        "nMAE",
        "MAE/cap",
        "MAE/cap left out",
        "ShapeMAE",
    ]
    TerminalExporter(show=["summary"]).export(report)
    assert "MAE/cap" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------------------------------
# Review of step 7b
# ---------------------------------------------------------------------------------------------------------------------


def test_excel_sheet_titles_that_clash_after_the_cut_get_their_own_sheets(tmp_path):
    from openpyxl import load_workbook

    from epftoolbox2.exporters import ExcelExporter

    column = "installed_capacity_mw"  # MAE/... and MAE/... left out are one title once cut to 31 characters
    source = hourly_source({column: lambda idx: 100.0 + idx.hour.to_numpy()})
    local = source.index.tz_localize(None)
    source.loc[local.normalize() == pd.Timestamp("2024-10-26"), column] = np.nan  # 24 forecasts left out
    report = report_of(
        {"a": records("1h"), "b": records("1h", error=lambda t, p: 2.0)},
        [MAEEvaluator(), ScaledMAEEvaluator(column)],
        "1h",
        source_data=source,
    )
    path = tmp_path / "out.xlsx"
    ExcelExporter(str(path), sheets=["summary", "horizon", "hour_horizon", "run_weekday_horizon"]).export(report)
    book = load_workbook(path)
    assert all(len(name) <= 31 for name in book.sheetnames)
    assert len({name.casefold() for name in book.sheetnames}) == len(book.sheetnames)
    table = report.by_horizon().set_index("model")
    for prefix in ("horizon", "HourHorizon", "RunWeekdayHorizon"):
        titles = [name for name in book.sheetnames if name.startswith(f"{prefix}_MAE_inst")]
        assert len(titles) == 2, (prefix, book.sheetnames)
    first, second = (book[name] for name in book.sheetnames if name.startswith("horizon_MAE_inst"))
    assert first["A1"].value == f"MAE/{column}" and second["A1"].value == f"MAE/{column} left out"
    assert second["B2"].value == table.loc["a", f"MAE/{column} left out"] == 24
    assert first["B2"].value == pytest.approx(table.loc["a", f"MAE/{column}"])
    # a title that clashes with nothing keeps its name
    assert {"horizon_MAE", "horizon_forecasts", "Summary"} <= set(book.sheetnames)


def zoned_quarter_records(seed):
    """c_dst.py's records: clock-numbered 15-minute forecasts across both switches, horizons 1 and 2."""
    rng = np.random.default_rng(seed)
    spring, autumn = pd.Timestamp(SPRING), pd.Timestamp(AUTUMN)
    rows = []
    runs = list(pd.date_range("2024-03-28", "2024-04-02")) + list(pd.date_range("2024-10-24", "2024-10-29"))
    for d, run in enumerate(runs):
        for h in (1, 2):
            target = run + pd.Timedelta(days=h)
            for p in range(96):
                if (target == spring or run == spring) and 8 <= p < 12:
                    continue
                a = 50.0 + 20 * np.sin(p / 96 * 6.28) + p % 4
                rows.append(
                    {
                        "run_date": run.date().isoformat(),
                        "target_date": target.date().isoformat(),
                        "run_weekday": run.dayofweek,
                        "hour": p,
                        "horizon": h,
                        "day_in_test": d,
                        "prediction": float(a + rng.normal(0, 3)),
                        "actual": None if (target == autumn and 8 <= p < 12) else float(a),
                        "freq": "15min",
                    }
                )
    return rows


def test_hourly_then_daily_is_the_daily_aggregate_or_refused():
    models = {"base": zoned_quarter_records(1), "m": zoned_quarter_records(2)}
    zoned = report_of(models, [MAEEvaluator()], horizon=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chained = zoned.aggregate("1h", agg="mean").aggregate("1D", agg="mean")
        direct = zoned.aggregate("1D", agg="mean")
    for name in models:
        a, b = as_frame(chained, name), as_frame(direct, name)
        assert (
            a[["run_date", "target_date", "horizon"]].values.tolist()
            == b[["run_date", "target_date", "horizon"]].values.tolist()
        )
        assert AUTUMN not in a["target_date"].tolist()
        np.testing.assert_allclose(a["prediction"], b["prediction"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(a["actual"], b["actual"], rtol=0, atol=1e-12)
    # Without a zone the hourly report can't tell a dropped hour from one the clock skips: a daily aggregate of it is refused.
    plain = report_of(models, [MAEEvaluator()], zone=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hourly = plain.aggregate("1h", agg="mean")
        direct = plain.aggregate("1D", agg="mean")
    assert AUTUMN not in as_frame(direct, "m")["target_date"].tolist()
    for agg in ("mean", "max", "peak"):
        with pytest.raises(ValueError, match="incomplete_hour"):
            hourly.aggregate("1D", agg=agg)
    # With no hour dropped there is nothing to mistake, and the chain is the direct aggregate.
    whole = {name: [r for r in rows if r["target_date"] != AUTUMN] for name, rows in models.items()}
    plain = report_of(whole, [MAEEvaluator()], zone=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chained = plain.aggregate("1h", agg="mean").aggregate("1D", agg="mean")
        direct = plain.aggregate("1D", agg="mean")
    a, b = as_frame(chained, "m"), as_frame(direct, "m")
    assert a[["run_date", "target_date"]].values.tolist() == b[["run_date", "target_date"]].values.tolist()
    np.testing.assert_allclose(a["prediction"], b["prediction"], rtol=0, atol=1e-12)


def test_blocks_and_shape_refuse_period_ids_beyond_the_results_frequency():
    recs = records(with_freq=False)  # quarter-hour ids 0-95 that the report reads as hours ("1h", the default)
    report = EvaluationReport({"m": recs}, [MAEEvaluator(), ShapeMAEEvaluator()])
    for agg in ("peak", "offpeak"):
        with pytest.raises(ValueError, match="period ids up to 95"):
            report.aggregate("1D", agg=agg)
    with pytest.raises(ValueError, match="period ids up to 95"):
        report.summary()


def test_error_split_judges_a_day_by_all_its_finite_quarter_hours():
    rows = []
    for d, run in enumerate(pd.date_range("2024-10-25", "2024-10-28")):
        target = run + pd.Timedelta(days=1)
        n = 100 if target == pd.Timestamp(AUTUMN) else 96  # the autumn day numbered through, ids 0-99
        for p in range(n):
            a = 50.0 + p % 4
            rows.append(
                {
                    "run_date": run.date().isoformat(),
                    "target_date": target.date().isoformat(),
                    "hour": p,
                    "horizon": 1,
                    "prediction": a + (30.0 if p >= 96 else np.sin(p)),
                    "actual": a,
                    "freq": "15min",
                }
            )
    row = EvaluationReport({"m": rows}, []).error_split().iloc[0]
    assert row["days"] == 3  # the 100-quarter-hour day is not a whole day of 96
    assert row["forecasts"] == 3 * 96


def numbered_through(target, hours, run=None, value=lambda hour: float(hour)):
    """One day of hourly records numbered 0, 1, 2, ... over the local hours `hours`; the error is `value(local hour)`."""
    run = run or (pd.Timestamp(target) - pd.Timedelta(days=1)).date().isoformat()
    return [
        {"run_date": run, "target_date": target, "hour": i, "horizon": 1, "prediction": 50.0 + value(h), "actual": 50.0}
        for i, h in enumerate(hours)
    ]


def test_blocks_leave_out_switch_days_numbered_through():
    plain = numbered_through("2024-03-30", range(24))
    short = numbered_through(SPRING, [h for h in range(24) if h != 2])  # a Sunday, ids 0-22
    long = numbered_through(AUTUMN, [*range(3), 2, *range(3, 24)])  # 25 local hours, ids 0-24
    report = EvaluationReport({"m": plain + short + long}, [MAEEvaluator()])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert len(list(report.aggregate("1D", agg="mean").refs["m"].iter_records())) == 3  # whole_days takes all three
        for agg in ("peak", "offpeak"):
            daily = report.aggregate("1D", agg=agg)
            kept = list(daily.refs["m"].iter_records())
            assert [r["target_date"] for r in kept] == ["2024-03-30"], agg
            assert daily.forecast_counts().loc[0, "incomplete_day"] == 2
    peak = list(EvaluationReport({"m": plain}, []).aggregate("1D", agg="peak").refs["m"].iter_records())[0]
    assert peak["prediction"] - peak["actual"] == pytest.approx(13.5)  # local 08:00-19:59


def test_source_columns_need_the_results_time_zone():
    recs = records("1h")
    local = hourly_source({"flag": lambda idx: (idx.hour.to_numpy() < 7).astype(float)})
    for source in (local.tz_convert("UTC"), local.tz_localize(None)):
        report = report_of({"m": recs}, [MAEEvaluator()], "1h", source_data=source)
        with pytest.raises(ValueError, match="time zone"):
            report.by_column("flag")
        scaled = report_of({"m": recs}, [ScaledMAEEvaluator("flag")], "1h", source_data=source)
        with pytest.raises(ValueError, match="time zone"):
            scaled.summary()
    assert len(report_of({"m": recs}, [MAEEvaluator()], "1h", source_data=local).by_column("flag")) == 2
    # Results without a zone (plain lists) read the source's wall clock, as CsvExporter does.
    assert len(
        report_of({"m": recs}, [MAEEvaluator()], "1h", zone=None, source_data=local.tz_convert("UTC")).by_column("flag")
    )


def test_nmae_of_a_zero_mean_is_nan():
    zeros = pd.DataFrame({"prediction": [0.0, 0.0], "actual": [0.0, 0.0]})
    assert np.isnan(NMAEEvaluator().compute(zeros))
    assert np.isnan(NMAEEvaluator().compute(zeros.assign(prediction=[1.0, 2.0])))
