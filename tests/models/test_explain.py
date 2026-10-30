"""`ModelPipeline.explain`: every expected table here is written by hand from what a column name says and from the rule.

Suffixes are relative to the row's date D and the target day is T = D + horizon, so at horizon 2 `price_d-1` is T-3 and
`x_d+2` is T. Warsaw and the Brussels rules share a wall clock, so `known` reads directly: `price` is published 13:00 the
day before delivery, `load_forecast` 10:00 the day before, and `load_actual` 1 h after the end of its period. The issue time
is 11:45 on D.
"""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import OLSModel, Use
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.availability import _parse_ranges
from tests.synthetic import PREDICTORS, make_market_data

PUBLISHED_13 = "published 13:00 Europe/Brussels on D-1 of delivery"
PUBLISHED_10 = "published 10:00 Europe/Brussels on D-1 of delivery"
REALISED = "known 1h after the end of the period"
STATIC = "known in advance"
INFO = InformationSet("11:45", "Europe/Warsaw")
COLUMNS = ["predictor", "target-relative", "rule", "known", "status"]
TEST = ("2024-05-06", "2024-05-08")


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


def frame(names, freq="1h", start="2024-03-01", days=240, tz="Europe/Warsaw"):
    first = pd.Timestamp(start, tz=tz)
    index = pd.date_range(first, first + pd.Timedelta(days=days), freq=freq, inclusive="left")
    return pd.DataFrame(1.0, index=index, columns=names)


def explain(names, horizon, hour, freq="1h", test=TEST, info=INFO, echo=False, **frame_options):
    pipeline = ModelPipeline(information_set=info).add_model(OLSModel(list(names), name="m"))
    return pipeline.explain(
        frame([n.replace("{horizon}", str(horizon)) for n in names], freq, **frame_options),
        model="m",
        test=test,
        horizon=horizon,
        hour=hour,
        freq=freq,
        echo=echo,
    )


def assert_table(table, expected):
    assert table[COLUMNS].values.tolist() == [list(row) for row in expected]


# ---------------------------------------------------------------------------
# Hand-written tables
# ---------------------------------------------------------------------------


def test_hourly_horizon_1():
    expected = [
        ("price_d-1", "T-2 12:00", PUBLISHED_13, "D-2 13:00", "ok"),
        ("price_d-7", "T-8 12:00", PUBLISHED_13, "D-8 13:00", "ok"),
        ("price", "T-1 12:00", PUBLISHED_13, "D-1 13:00", "ok"),
        ("load_forecast_d+1", "T 12:00", PUBLISHED_10, "D 10:00", "ok"),
        ("load_actual_d-1", "T-2 12:00", REALISED, "D-1 14:00", "ok"),
        ("load_actual", "T-1 12:00", REALISED, "D 14:00", "late"),
        ("is_monday_d+1", "T 12:00", STATIC, "in advance", "ok"),
        ("price_h05_d-1", "T-2 05:00", PUBLISHED_13, "D-2 13:00", "ok"),
        ("price_day_max_d-1", "T-2 00:00 … 23:00", PUBLISHED_13, "D-2 13:00", "ok"),
        ("load_actual_h05", "T-1 05:00", REALISED, "D 07:00", "ok"),
        ("load_actual_h23", "T-1 23:00", REALISED, "D+1 01:00", "late"),
        ("load_actual_day_max", "T-1 00:00 … 23:00", REALISED, "D+1 01:00", "late"),
        ("load_actual_d-1_day_max", "T-2 00:00 … 23:00", REALISED, "D 01:00", "ok"),
        ("load_actual_d+1_h05", "T 05:00", REALISED, "D+1 07:00", "late"),
        ("mystery", "T-1 12:00", "-", "-", "no rule"),
    ]
    assert_table(explain([row[0] for row in expected], 1, 12)["predictors"], expected)


def test_hourly_horizon_2_reads_every_name_two_days_before_its_target():
    expected = [
        ("price_d-1", "T-3 12:00", PUBLISHED_13, "D-2 13:00", "ok"),
        ("price_d-7", "T-9 12:00", PUBLISHED_13, "D-8 13:00", "ok"),
        ("price", "T-2 12:00", PUBLISHED_13, "D-1 13:00", "ok"),
        ("load_forecast_d+2", "T 12:00", PUBLISHED_10, "D+1 10:00", "late"),
        ("load_forecast_d+1", "T-1 12:00", PUBLISHED_10, "D 10:00", "ok"),
        ("is_monday_d+2", "T 12:00", STATIC, "in advance", "ok"),
        ("price_h05_d-1", "T-3 05:00", PUBLISHED_13, "D-2 13:00", "ok"),
        ("price_day_max_d-1", "T-3 00:00 … 23:00", PUBLISHED_13, "D-2 13:00", "ok"),
        ("load_actual_h23", "T-2 23:00", REALISED, "D+1 01:00", "late"),
        ("load_actual_d+2_h05", "T 05:00", REALISED, "D+2 07:00", "late"),
    ]
    assert_table(explain([row[0] for row in expected], 2, 12)["predictors"], expected)


def test_a_horizon_template_expands_to_the_target_day():
    table = explain(["load_forecast_d+{horizon}", "price_d-1"], 2, 12)["predictors"]
    assert table["predictor"].tolist() == ["load_forecast_d+2", "price_d-1"]
    assert table["target-relative"].tolist() == ["T 12:00", "T-3 12:00"]


def test_15_minute_horizon_1():
    expected = [
        ("price_d-1", "T-2 12:00", PUBLISHED_13, "D-2 13:00", "ok"),
        ("load_forecast_d+1", "T 12:00", PUBLISHED_10, "D 10:00", "ok"),
        ("load_actual_d-1", "T-2 12:00", REALISED, "D-1 13:15", "ok"),
        ("load_actual", "T-1 12:00", REALISED, "D 13:15", "late"),
        ("price_s20_d-1", "T-2 05:00", PUBLISHED_13, "D-2 13:00", "ok"),
        ("load_actual_s95", "T-1 23:45", REALISED, "D+1 01:00", "late"),
        ("load_actual_s03", "T-1 00:45", REALISED, "D 02:00", "ok"),
        ("price_day_max_d-1", "T-2 00:00 … 23:45", PUBLISHED_13, "D-2 13:00", "ok"),
        ("load_actual_day_max", "T-1 00:00 … 23:45", REALISED, "D+1 01:00", "late"),
    ]
    assert_table(explain([row[0] for row in expected], 1, 48, freq="15min")["predictors"], expected)


def test_15_minute_horizon_2_at_a_quarter_past():
    expected = [
        ("price_d-1", "T-3 12:15", PUBLISHED_13, "D-2 13:00", "ok"),
        ("load_forecast_d+2", "T 12:15", PUBLISHED_10, "D+1 10:00", "late"),
        ("load_actual_d-1", "T-3 12:15", REALISED, "D-1 13:30", "ok"),
        ("load_actual_s95", "T-2 23:45", REALISED, "D+1 01:00", "late"),
        ("price_day_max_d-1", "T-3 00:00 … 23:45", PUBLISHED_13, "D-2 13:00", "ok"),
    ]
    assert_table(explain([row[0] for row in expected], 2, 49, freq="15min")["predictors"], expected)


def test_slots_are_grouped_in_runs_that_agree_on_everything_else():
    names = [f"price_h{h:02d}_d-1" for h in range(24)]
    names += [f"load_actual_h{h:02d}" for h in range(24)]
    names += ["is_monday_d+1", "load_actual_h05_d-1"]
    expected = [
        ("price_h00_d-1 … price_h23_d-1 (24)", "T-2 00:00 … 23:00", PUBLISHED_13, "D-2 13:00", "ok"),
        ("load_actual_h00 … load_actual_h09 (10)", "T-1 00:00 … 09:00", REALISED, "D 02:00 … D 11:00", "ok"),
        ("load_actual_h10 … load_actual_h23 (14)", "T-1 10:00 … 23:00", REALISED, "D 12:00 … D+1 01:00", "late"),
        ("is_monday_d+1", "T 12:00", STATIC, "in advance", "ok"),
        ("load_actual_h05_d-1", "T-2 05:00", REALISED, "D-1 07:00", "ok"),
    ]
    assert_table(explain(names, 1, 12)["grouped"], expected)


def test_quarter_hour_slots_group_and_slots_with_a_gap_do_not():
    names = [f"price_s{s:02d}_d-1" for s in range(96)]
    expected = [("price_s00_d-1 … price_s95_d-1 (96)", "T-2 00:00 … 23:45", PUBLISHED_13, "D-2 13:00", "ok")]
    assert_table(explain(names, 1, 48, freq="15min")["grouped"], expected)

    apart = explain(["price_h00_d-1", "price_h06_d-1", "price_h07_d-1"], 1, 12)["grouped"]
    assert apart["predictor"].tolist() == ["price_h00_d-1", "price_h06_d-1 … price_h07_d-1 (2)"]


def test_an_exact_rule_on_a_shifted_name_changes_when_it_is_known_not_what_it_holds():
    info = InformationSet("11:45", "Europe/Warsaw", rules={"x_d-3": Availability.published("09:00", 0)})
    table = explain(["x_d-3"], 2, 12, info=info)["predictors"]
    assert table["target-relative"].tolist() == ["T-5 12:00"]
    assert table["known"].tolist() == ["D 09:00"]


def test_an_hourly_model_cannot_tell_a_quarter_hour_slot():
    """`_s30` on hourly data can be a slot of any length (`_resolve_for`), so the view doesn't name a time for it."""
    table = explain(["load_actual_s30"], 1, 12)["predictors"]
    assert table["target-relative"].tolist() == ["?"]
    assert table["status"].tolist() == ["no rule"]


def test_a_rule_that_is_late_on_some_days_only_says_so():
    info = InformationSet("09:00", "Europe/Warsaw", rules={"weekly": Availability.weekly("friday", "10:00")})
    table = explain(["weekly_d+7"], 1, 12, test=("2024-05-08", "2024-05-12"), info=info)["predictors"]
    # D+7 on Wed 05-08 is Wed 05-15, published Fri 05-10 10:00: late, by two days. On Thu 05-09 and on Fri 05-10 (at 09:00,
    # an hour before it comes out) late as well. On Sat 05-11 and Sun 05-12 it is out already.
    assert table["status"].tolist() == ["late on 3 of 5 days"]
    assert table["known"].tolist() == ["D+2 10:00"]


# ---------------------------------------------------------------------------
# DST days: 2024-03-31 has 23 hours in Warsaw, 2024-10-27 has 25
# ---------------------------------------------------------------------------

SPRING = "2024-03-31"
AUTUMN = "2024-10-27"
DST_NAMES = [
    "price_d-1",
    "load_actual_d-1",
    "price_h02_d-1",
    "price_day_max_d-1",
    "load_actual_day_max",
    "load_actual_day_max_d-1",
]


@pytest.mark.parametrize("day", [SPRING, AUTUMN])
def test_a_dst_issue_day_keeps_the_labels_of_any_other_day(day):
    """Wall-clock labels, calendar days: a 23- or 25-hour day moves neither, and a day aggregate still ends at 23:00."""
    expected = [
        ("price_d-1", "T-2 12:00", PUBLISHED_13, "D-2 13:00", "ok"),
        ("load_actual_d-1", "T-2 12:00", REALISED, "D-1 14:00", "ok"),
        ("price_h02_d-1", "T-2 02:00", PUBLISHED_13, "D-2 13:00", "ok"),
        ("price_day_max_d-1", "T-2 00:00 … 23:00", PUBLISHED_13, "D-2 13:00", "ok"),
        ("load_actual_day_max", "T-1 00:00 … 23:00", REALISED, "D+1 01:00", "late"),
        ("load_actual_day_max_d-1", "T-2 00:00 … 23:00", REALISED, "D 01:00", "ok"),
    ]
    assert_table(explain(DST_NAMES, 1, 12, test=(day, day))["predictors"], expected)


@pytest.mark.parametrize("day", [SPRING, AUTUMN])
def test_a_dst_day_on_15_minute_data(day):
    names = ["price_d-1", "price_s08_d-1", "price_day_max_d-1", "load_actual_s95"]
    expected = [
        ("price_d-1", "T-2 12:00", PUBLISHED_13, "D-2 13:00", "ok"),
        ("price_s08_d-1", "T-2 02:00", PUBLISHED_13, "D-2 13:00", "ok"),
        ("price_day_max_d-1", "T-2 00:00 … 23:45", PUBLISHED_13, "D-2 13:00", "ok"),
        ("load_actual_s95", "T-1 23:45", REALISED, "D+1 01:00", "late"),
    ]
    assert_table(explain(names, 1, 48, freq="15min", test=(day, day))["predictors"], expected)


def test_the_hour_a_spring_switch_skips_has_no_row_and_the_view_says_so(capsys):
    table = explain(["price_d-1", "load_actual_d-1"], 1, 2, test=(SPRING, SPRING), echo=True)["predictors"]
    assert table["target-relative"].tolist() == ["T-2 02:00", "T-2 02:00"]
    assert table["known"].tolist() == ["D-2 13:00", "D-1 04:00"]
    assert f"No row at period 2 on 1 of the issue days ({SPRING})" in capsys.readouterr().out


def test_the_hour_an_autumn_switch_repeats_reads_the_same_wall_clock():
    table = explain(["price_d-1", "load_actual_d-1"], 1, 2, test=(AUTUMN, AUTUMN))["predictors"]
    assert table["target-relative"].tolist() == ["T-2 02:00", "T-2 02:00"]
    assert table["status"].tolist() == ["ok", "ok"]


def test_a_target_day_on_a_switch_reads_the_same_days():
    table = explain(["load_forecast_d+1", "price_d-1"], 1, 12, test=("2024-03-30", "2024-03-30"))["predictors"]
    assert table["target-relative"].tolist() == ["T 12:00", "T-2 12:00"]
    assert table["status"].tolist() == ["ok", "ok"]


def test_an_hour_lag_runs_in_elapsed_time_across_a_switch():
    """24 hours before 12:00 on the 23-hour day is 11:00 the day before, and on the 25-hour day 13:00 the day before."""
    assert explain(["x_h-24"], 1, 12, test=(SPRING, SPRING))["predictors"]["target-relative"].tolist() == ["T-2 11:00"]
    assert explain(["x_h-24"], 1, 12, test=(AUTUMN, AUTUMN))["predictors"]["target-relative"].tolist() == ["T-2 13:00"]
    ordinary = explain(["x_h-24"], 1, 12, test=TEST)["predictors"]
    assert ordinary["target-relative"].tolist() == ["T-2 12:00"]


def test_a_window_across_a_switch_gives_the_usual_label_and_says_how_many_days_differ():
    table = explain(["x_h-24"], 1, 12, test=("2024-03-29", "2024-04-01"))["predictors"]
    assert table["target-relative"].tolist() == ["T-2 12:00 (else on 1 of 4 days)"]


def test_a_utc_index_reads_its_own_clock():
    table = explain(["price_d-1"], 2, 12, tz="UTC")["predictors"]
    assert table["target-relative"].tolist() == ["T-3 12:00"]


def test_a_naive_index_has_no_switch_to_cross():
    data = frame(["x_h-24"]).tz_localize(None)
    pipeline = ModelPipeline(information_set=INFO).add_model(OLSModel(["x_h-24"], name="m"))
    table = pipeline.explain(data, model="m", test=(SPRING, SPRING), horizon=1, hour=12, echo=False)["predictors"]
    assert table["target-relative"].tolist() == ["T-2 12:00"]


# ---------------------------------------------------------------------------
# Printing, arguments, no issue time
# ---------------------------------------------------------------------------


def test_what_it_prints_names_t_and_never_a_lowercase_day_relative_to_the_target(capsys):
    explain(["load_forecast_d+1", "load_actual"], 1, 12, echo=True)
    out = capsys.readouterr().out
    assert "T = D + 1" in out
    assert "issued 11:45 Europe/Warsaw on D" in out
    assert "T 12:00" in out and "T-1 12:00" in out
    relative = out.replace("load_forecast_d+1", "")
    assert "d-" not in relative and "d+" not in relative


def test_echo_false_prints_nothing(capsys):
    explain(["price_d-1"], 1, 12, echo=False)
    assert capsys.readouterr().out == ""


def test_without_an_issue_time_it_still_says_what_each_name_holds():
    table = explain(["price_d-1", "price_day_max_d-1"], 2, 12, info=None)["predictors"]
    assert table["target-relative"].tolist() == ["T-3 12:00", "T-3 00:00 … 23:00"]
    assert table["status"].tolist() == ["no issue time", "no issue time"]


def test_a_column_the_data_lacks_is_a_status_not_an_error():
    pipeline = ModelPipeline(information_set=INFO).add_model(OLSModel(["price_d-1"], name="m"))
    table = pipeline.explain(frame(["other"]), model="m", test=TEST, echo=False)["predictors"]
    assert table["status"].tolist() == ["not in data"]
    assert table["target-relative"].tolist() == ["T-2 12:00"]


def test_callable_predictors_are_explained_for_that_horizon_and_hour():
    def by_horizon_and_hour(horizon, hour):
        return [f"price_d-{horizon}"] + (["load_actual_d-1"] if hour < 12 else [])

    pipeline = ModelPipeline(information_set=INFO).add_model(OLSModel(by_horizon_and_hour, name="m"))
    data = frame(["price_d-2", "load_actual_d-1"])
    table = pipeline.explain(data, model="m", test=TEST, horizon=2, hour=3, echo=False)["predictors"]
    assert table["target-relative"].tolist() == ["T-4 03:00", "T-3 03:00"]
    table = pipeline.explain(data, model="m", test=TEST, horizon=2, hour=12, echo=False)["predictors"]
    assert table["predictor"].tolist() == ["price_d-2"]


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"hour": 24}, "0-23"),
        ({"hour": -1}, "0-23"),
        ({"horizon": 0}, "horizon"),
        ({"test": ("2024-05-08", "2024-05-06")}, "starts after"),
        ({"freq": "30min"}, "Unsupported freq"),
    ],
)
def test_bad_arguments_raise(arguments, message):
    pipeline = ModelPipeline(information_set=INFO).add_model(OLSModel(["price_d-1"], name="m"))
    with pytest.raises(ValueError, match=message):
        pipeline.explain(frame(["price_d-1"]), model="m", **{"test": TEST, **arguments}, echo=False)


def test_an_unknown_model_raises():
    pipeline = ModelPipeline(information_set=INFO).add_model(OLSModel(["price_d-1"], name="m"))
    with pytest.raises(ValueError, match=r"Unknown model"):
        pipeline.explain(frame(["price_d-1"]), model="nope", test=TEST)


# ---------------------------------------------------------------------------
# Predictor functions of ctx
# ---------------------------------------------------------------------------

RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "price": Availability.published("13:00", 1),
    "early": Availability.published("10:15", 0),
}
RULED = InformationSet("11:45", "Europe/Warsaw", rules=RULES)
WINDOW = ("2023-04-05", "2023-04-14")
MISSING_DAYS = ("2023-04-07", "2023-04-10")  # `early` has no value on these issue days


@pytest.fixture(scope="module")
def df():
    data = make_market_data(noise=5.0, seed=7)
    data["early"] = data["load_d+1"]
    data.loc[data.index.tz_localize(None).normalize().isin(pd.to_datetime(MISSING_DAYS)), "early"] = np.nan
    return data


def by_hour_and_publication(ctx):
    names = PREDICTORS + (["price_d-1"] if ctx.period < 12 else [])
    if ctx.published("early"):
        names = names + ["early"]
    return names


def test_explain_prints_target_relative_days_and_status(df, capsys):
    frame_with_late = df.assign(**{"early_d+1": df["early"]})
    pipeline = ModelPipeline(information_set=RULED).add_model(
        OLSModel(PREDICTORS + ["price_d-1", "early_d+1"], training_window=45, name="m")
    )
    out = pipeline.explain(frame_with_late, model="m", horizon=1, hour=12, test=WINDOW)["predictors"]
    out = out.set_index("predictor")
    assert out.loc["price_d-1", "target-relative"] == "T-2 12:00"
    assert out.loc["load_d+1", "target-relative"] == "T 12:00"
    assert out.loc["price_d-1", "status"] == "ok"
    assert out.loc["early_d+1", "status"] == "late"
    printed = capsys.readouterr().out
    assert "T-2 12:00" in printed and "late" in printed


def test_explain_lists_each_distinct_list_of_a_function(df, capsys):
    pipeline = ModelPipeline(information_set=RULED).add_model(
        OLSModel(by_hour_and_publication, training_window=45, name="dyn")
    )
    lists = pipeline.explain(df, model="dyn", horizon=1, hour=12, test=WINDOW)["lists"]
    assert lists["forecasts"].sum() == 240 and len(lists) == 4
    assert "distinct list" in capsys.readouterr().out


def test_the_lists_table_is_written_by_hand(df):
    """10 issue days, 2 of them without `early`. Periods 0-11 add price_d-1. So the four lists are used on 8, 8, 2 and 2 days,
    and the two most used, tied at 96 forecasts, go to the one returned first: period 0 on the first day."""
    pipeline = ModelPipeline(information_set=RULED).add_model(
        OLSModel(by_hour_and_publication, training_window=45, name="dyn")
    )
    lists = pipeline.explain(df, model="dyn", horizon=1, hour=12, test=WINDOW, echo=False)["lists"]
    rows = lists[["horizon", "list", "forecasts", "issue days", "first issue day", "last issue day", "periods"]]
    assert rows.values.tolist() == [
        [1, 1, 96, 8, "2023-04-05", "2023-04-14", "0-11"],
        [1, 2, 96, 8, "2023-04-05", "2023-04-14", "12-23"],
        [1, 3, 24, 2, "2023-04-07", "2023-04-10", "0-11"],
        [1, 4, 24, 2, "2023-04-07", "2023-04-10", "12-23"],
    ]
    assert lists["adds"].tolist() == ["-", "", "", ""]
    assert lists["drops"].tolist() == ["-", "price_d-1", "early", "price_d-1, early"]
    assert lists["predictors"].tolist() == [6, 5, 5, 4]
    assert lists["train_on"].tolist() == ["all"] * 4


def test_lists_are_compared_within_their_horizon(df):
    def by_horizon(ctx):
        names = PREDICTORS + (["price_d-1"] if ctx.period < 18 else [])
        return Use(names + (["early"] if ctx.horizon == 2 and ctx.period == 0 else []))

    pipeline = ModelPipeline(information_set=RULED).add_model(OLSModel(by_horizon, training_window=45, name="dyn"))
    lists = pipeline.explain(df, model="dyn", horizon=2, hour=12, test=WINDOW, echo=False)["lists"]
    assert lists["horizon"].tolist() == [1, 1, 2, 2, 2]
    # Horizon 1: periods 0-17 are the most used, and it is the reference; `{horizon}` names differ from horizon 2's.
    first, second = lists.iloc[0], lists.iloc[1]
    assert (first["adds"], first["drops"]) == ("-", "-")
    assert (second["adds"], second["drops"]) == ("", "price_d-1")
    # Horizon 2: the reference is again periods 1-17, not any list of horizon 1.
    rows = lists[lists["horizon"] == 2].set_index("periods")
    assert (rows.loc["1-17", "adds"], rows.loc["1-17", "drops"]) == ("-", "-")
    assert (rows.loc["0", "adds"], rows.loc["0", "drops"]) == ("early", "")
    assert (rows.loc["18-23", "adds"], rows.loc["18-23", "drops"]) == ("", "price_d-1")


def test_a_list_is_judged_on_the_days_it_is_used(df):
    """Published Sunday 10:00 for the week from Monday, and issued 09:00: Monday's value is late on Sunday only."""
    rules = {**RULES, "weekly": Availability.weekly("sunday", "10:00")}
    info = InformationSet("09:00", "Europe/Warsaw", rules=rules)
    frame_ = df.assign(**{"weekly_d+1": 1.0})

    def not_on_sundays(ctx):
        return PREDICTORS + ([] if ctx.issue_day.weekday() == 6 else ["weekly_d+1"])

    model = OLSModel(not_on_sundays, training_window=45, name="m")
    pipeline = ModelPipeline(information_set=info).add_model(model)
    out = pipeline.explain(frame_, model="m", horizon=1, hour=12, test=WINDOW, echo=False)
    assert out["lists"]["issue days"].tolist() == [1, 9] or out["lists"]["issue days"].tolist() == [9, 1]
    assert out["predictors"].set_index("predictor").loc["weekly_d+1", "status"] == "ok"

    fixed = ModelPipeline(information_set=info).add_model(OLSModel(PREDICTORS + ["weekly_d+1"], name="f"))
    table = fixed.explain(frame_, model="f", horizon=1, hour=12, test=WINDOW, echo=False)["predictors"]
    assert table.set_index("predictor").loc["weekly_d+1", "status"] == "late on 1 of 10 days"


def test_a_late_value_doesnt_stop_the_view_at_any_level(df):
    def peeks(ctx):
        return PREDICTORS if ctx.value("price") > 0 else PREDICTORS + ["price_d-1"]

    for level in ("none", "warn", "raise"):
        pipeline = ModelPipeline(information_set=RULED, availability=level).add_model(
            OLSModel(peeks, training_window=45, name="m")
        )
        out = pipeline.explain(df, model="m", horizon=1, hour=12, test=WINDOW, echo=False)
        assert len(out["lists"]) >= 1


def test_the_view_and_the_listing_agree_on_every_period(df):
    """Two ways to the same rule: `explain` per forecast, the listing per run."""
    frame_ = df.assign(observed=1.0, **{"observed_d-1": 1.0, "early_d+1": 1.0})
    rules = {**RULES, "observed": Availability.realised("1h")}
    info = InformationSet("11:45", "Europe/Warsaw", rules=rules)
    names = PREDICTORS + ["observed", "observed_d-1", "early_d+1"]
    pipeline = ModelPipeline(information_set=info).add_model(OLSModel(names, training_window=45, name="m"))
    report = pipeline.run(frame_, "2023-04-05", "2023-04-05", horizon=1)
    listed = {
        (row["predictor"], period)
        for row in report.availability().to_dict("records")
        for period in _parse_ranges(row["periods"])
    }
    shown = set()
    for period in range(24):
        table = pipeline.explain(frame_, model="m", test=("2023-04-05", "2023-04-05"), hour=period, echo=False)
        shown |= {
            (row["predictor"], period) for row in table["predictors"].to_dict("records") if row["status"] == "late"
        }
    assert shown == listed
    assert ("observed", 10) in listed and ("observed", 9) not in listed and ("early_d+1", 0) in listed


def test_a_predictor_function_needs_the_test_period(df):
    pipeline = ModelPipeline(information_set=RULED).add_model(OLSModel(by_hour_and_publication, name="dyn"))
    with pytest.raises(ValueError, match="needs test="):
        pipeline.explain(df, model="dyn", horizon=1, hour=12)


def test_explaining_leaves_no_trace_on_a_later_run(df, tmp_path):
    model = OLSModel(by_hour_and_publication, training_window=45, name="dyn")
    plain = OLSModel(PREDICTORS, training_window=45, name="plain")
    pipeline = ModelPipeline(information_set=RULED).add_model(model).add_model(plain)
    before = {name: repr(getattr(model, name)) for name in vars(model)}
    pipeline.explain(df, model="dyn", horizon=1, hour=12, test=WINDOW, freq="1h", echo=False)
    pipeline.explain(df, model="plain", horizon=2, hour=3, test=WINDOW, echo=False)
    assert {name: repr(getattr(model, name)) for name in vars(model)} == before
    assert list(tmp_path.iterdir()) == []

    after = pipeline.run(df, "2023-04-05", "2023-04-05", horizon=1)
    fresh = (
        ModelPipeline(information_set=RULED)
        .add_model(OLSModel(by_hour_and_publication, training_window=45, name="dyn"))
        .add_model(OLSModel(PREDICTORS, training_window=45, name="plain"))
        .run(df, "2023-04-05", "2023-04-05", horizon=1)
    )
    assert after.predictions().equals(fresh.predictions())


def test_explaining_warns_about_nothing(df, recwarn):
    def asks_for_what_is_never_out(ctx):
        return PREDICTORS + (["early"] if ctx.published("early_d+1") else [])

    frame_ = df.assign(**{"early_d+1": 1.0, "mystery": 1.0})
    pipeline = ModelPipeline(information_set=RULED).add_model(
        OLSModel(asks_for_what_is_never_out, training_window=45, name="m")
    )
    pipeline.add_model(OLSModel(PREDICTORS + ["early_d+1", "mystery"], training_window=45, name="f"))
    pipeline.explain(frame_, model="m", horizon=1, hour=12, test=WINDOW, echo=False)
    pipeline.explain(frame_, model="f", horizon=1, hour=12, test=WINDOW, echo=False)
    assert [
        str(w.message) for w in recwarn if "epftoolbox2" in str(w.filename) or "test_explain" in str(w.filename)
    ] == []
