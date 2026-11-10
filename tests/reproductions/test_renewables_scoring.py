"""Tier S of the renewables study: experiment 2's price table (ridge, DE_LU and AT), its recovered shares, DM table and
sub-periods, and experiment 1's DE solar and wind table and DM list, recomputed by the library's report from the study's stored
forecasts.

The study's CSVs hold full precision, so each library value is checked within 1e-9 (relative) of the CSV cell, which is the
study's formula (`evaluate.py`) on the same keys; the DM statistics and p-values within 1e-10 (the study's Newey-West variance is
the library's: autocovariances over n, lag 6); and each cell `renewables/REPORT.md` prints is the library's value rounded. The
study scores every model on the keys where the actual and all of the zone's candidates are finite, which the fixture holds, so
each library table is on the study's sample. The fixture and its builder are in `tests/fixtures/renewables/`.
"""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.evaluators import BiasEvaluator, MAEEvaluator, NMAEEvaluator, RMSEEvaluator, ScaledMAEEvaluator
from epftoolbox2.results.report import EvaluationReport
from tests.reproductions.scoring import printed_equal, records, renewables, tables

# Every variant misses some hours of some days (a key needs all of the zone's candidates finite): the tests say so.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Days with only part of their periods in common"),
    pytest.mark.filterwarnings("ignore:The comparison rests on"),  # the validation half-year: 181 days
]

LAG = 6
ZONES = ("de_lu", "at")
SUB_PERIODS = {2025: "test_H2_2025", 2026: "test_2026"}


def close(value, cell, rel=1e-9):
    return abs(value - cell) <= rel * max(1.0, abs(cell))


def rounds_to(value, cell, digits):
    return abs(value - cell) <= 0.5 * 10**-digits + 1e-9


def price_frame(zone):
    frame = renewables("prices")
    return frame[frame["zone"] == zone].reset_index(drop=True)


def price_variants(frame):
    return [c for c in frame.columns if c not in ("zone", "target_date", "hour", "actual") and frame[c].notna().any()]


def in_period(recs, period):
    first, last = tables("renewables")["periods"][period]
    return [r for r in recs if first <= r["target_date"] <= last]


def gen_frame(kind, period):
    frame = renewables("generation")
    return frame[(frame["kind"] == kind) & (frame["period"] == period)].reset_index(drop=True)


def gen_source(frame):
    """The source data `ScaledMAEEvaluator("cap")` and `by_column("is_night")` read: a naive wall-clock hourly frame over the
    fixture's days (the study's hours, the DST-filled ones included), NaN where the study has no key."""
    stamps = pd.to_datetime(frame["target_date"]) + pd.to_timedelta(frame["hour"].astype(int), unit="h")
    index = pd.date_range(stamps.min().normalize(), stamps.max().normalize() + pd.Timedelta(hours=23), freq="1h")
    values = pd.DataFrame({"cap": frame["cap"].to_numpy(), "is_night": frame["is_night"].astype(float).to_numpy()})
    return values.set_index(pd.DatetimeIndex(stamps)).reindex(index)


@pytest.fixture(scope="module")
def priced():
    out = {}
    for zone in ZONES:
        frame = price_frame(zone)
        recs = {v: records(frame, v, "hour", "1h") for v in price_variants(frame)}
        report = EvaluationReport(recs, [MAEEvaluator(), RMSEEvaluator()])
        result = {
            "summary": report.summary().set_index("model"),
            "days": report.leaderboard().set_index("model")["n_days"],
            "by_year": report.by_year().set_index(["year", "model"]),
            "dm": {},
        }
        for row in tables("renewables")["price_dm"]:
            if row["zone"] != zone:
                continue
            pair = {name: in_period(recs[name], row["period"]) for name in (row["A"], row["B"])}
            tested = EvaluationReport(pair, []).tests(benchmark=row["B"], lag=LAG, interval=None)
            result["dm"][(row["period"], row["A"], row["B"])] = tested.iloc[0]
        out[zone] = result
    return out


@pytest.mark.parametrize("zone", ZONES)
def test_price_table_is_reproduced(zone, priced):
    summary, days, by_year = (priced[zone][k] for k in ("summary", "days", "by_year"))
    frame = price_frame(zone)
    rows = [r for r in tables("renewables")["price_table"] if r["zone"] == zone]
    assert {r["variant"][: -len("/ridge")] for r in rows} == set(price_variants(frame))
    for row in rows:
        variant = row["variant"][: -len("/ridge")]
        if row["period"] == "test":
            library = summary.loc[variant]
            assert library["forecasts"] == row["n"] and days[variant] == row["days"]
            e = frame[variant] - frame["actual"]
            assert abs(library["MAE"] - e.abs().mean()) <= 1e-9
        else:
            year = next(y for y, name in SUB_PERIODS.items() if name == row["period"])
            library = by_year.loc[(year, variant)]
            assert library["forecasts"] == row["n"]
        assert close(library["MAE"], row["MAE"]), (row, library["MAE"])
        assert close(library["RMSE"], row["RMSE"]), (row, library["RMSE"])
    # evaluate.py printed the price table at 3 decimals, which the report rounded again.
    for variant, cell in tables("renewables")["report_price"][zone].items():
        assert printed_equal(summary.loc[variant, "MAE"], cell, 2, 3), (variant, summary.loc[variant, "MAE"], cell)
    for variant, cell in tables("renewables")["report_price_rmse"].get(zone, {}).items():
        assert printed_equal(summary.loc[variant, "RMSE"], cell, 2, 3)
    errata = {
        (z, period, variant): (printed, value)
        for z, period, variant, printed, value in tables("renewables")["report_errata"]
    }
    for period, cells in tables("renewables")["report_price_sub"][zone].items():
        year = next(y for y, name in SUB_PERIODS.items() if name == period)
        for variant, cell in cells.items():
            value = by_year.loc[(year, variant), "MAE"]
            if (
                zone,
                period,
                variant,
            ) in errata:  # a cell the report misprints: the study's value, one digit off the cell
                assert rounds_to(value, errata[(zone, period, variant)][1], 4) and abs(value - cell) < 0.01
                assert not printed_equal(value, cell, 2, 3)
                continue
            assert printed_equal(value, cell, 2, 3), (period, variant, value, cell)


@pytest.mark.parametrize("zone", ZONES)
def test_recovered_shares_are_reproduced(zone, priced):
    mae = priced[zone]["summary"]["MAE"]
    for own, share in tables("renewables")["report_shares"][zone].items():
        a69 = "tso_de" if own.endswith("_de") else "tso"
        recovered = 100 * (mae["base"] - mae[own]) / (mae["base"] - mae[a69])
        assert round(recovered) == share, (own, recovered)


@pytest.mark.parametrize("zone", ZONES)
def test_price_dm_table_is_reproduced(zone, priced):
    rows = [r for r in tables("renewables")["price_dm"] if r["zone"] == zone]
    assert rows
    for row in rows:
        library = priced[zone]["dm"][(row["period"], row["A"], row["B"])]
        # The library's statistic is positive when A is better: the study's sign the other way round.
        assert abs(-library["statistic"] - row["DM"]) <= 1e-10, (row, library["statistic"])
        assert abs(library["p_value"] - row["p"]) <= 1e-10, (row, library["p_value"])
        assert close(library["mae"] - library["mae_benchmark"], row["dMAE"]), row
    for a, b, statistic, p_value in tables("renewables")["report_price_dm"][zone]:
        library = priced[zone]["dm"][("test", a, b)]
        # evaluate.py printed the DM table at 4 decimals.
        assert printed_equal(-library["statistic"], statistic, 2, 4), (a, b, library["statistic"], statistic)
        if p_value is not None:
            digits = len(repr(p_value).split(".")[1])  # as many as the cell prints
            assert printed_equal(library["p_value"], p_value, digits, 4), (a, b, library["p_value"])


# ---------------------------------------------------------------------------------------------------------------------
# Experiment 1: DE solar and wind
# ---------------------------------------------------------------------------------------------------------------------

GEN = [(kind, period) for kind in ("SOL", "WIND") for period in ("val", "test")]


@pytest.fixture(scope="module")
def generated():
    names = tables("renewables")["gen_names"]
    out = {}
    for kind, period in GEN:
        frame = gen_frame(kind, period)
        recs = {name: records(frame, name, "hour", "1h") for name in names.values()}
        evaluators = [MAEEvaluator(), RMSEEvaluator(), BiasEvaluator(), NMAEEvaluator(), ScaledMAEEvaluator("cap")]
        report = EvaluationReport(recs, evaluators, source_data=gen_source(frame))
        result = {
            "summary": report.summary().set_index("model"),
            "night": report.by_column("is_night").set_index(["is_night", "model"]),
            "dm": {},
        }
        for row in tables("renewables")["gen_dm"]:
            if (row["target"], row["period"]) == (kind, period):
                a, b = names[row["A"]], names[row["B"]]
                tested = EvaluationReport({a: recs[a], b: recs[b]}, []).tests(benchmark=b, lag=LAG, interval=None)
                result["dm"][(a, b)] = tested.iloc[0]
        out[(kind, period)] = result
    return out


@pytest.mark.parametrize("kind, period", GEN)
def test_generation_table_is_reproduced(kind, period, generated):
    summary, night = generated[(kind, period)]["summary"], generated[(kind, period)]["night"]
    names = tables("renewables")["gen_names"]
    rows = [r for r in tables("renewables")["gen_table"] if (r["target"], r["period"]) == (kind, period)]
    assert len(rows) == len(names)
    frame = gen_frame(kind, period)
    for row in rows:
        model = names[row["model"]]
        library = summary.loc[model]
        daylight = night.loc[(0.0, model), "MAE"] if kind == "SOL" else library["MAE"]
        assert library["forecasts"] == row["n"] == len(frame)
        assert library["MAE/cap left out"] == 0
        cells = {
            "MAE": library["MAE"],
            "RMSE": library["RMSE"],
            "bias": library["Bias"],
            "nMAE_pct": 100 * library["nMAE"],
            "MAE_cap_pct": 100 * library["MAE/cap"],
            "MAE_day": daylight,
            "mean_actual": library["MAE"] / library["nMAE"],
        }
        for cell, value in cells.items():
            assert close(value, row[cell]), (model, cell, value, row[cell])
        e = (frame[model] - frame["actual"]).abs()
        assert abs(library["MAE/cap"] - (e / frame["cap"]).mean()) <= 1e-9
        if kind == "SOL":
            assert abs(daylight - e[frame["is_night"] == 0].mean()) <= 1e-9
    if period == "test":
        for model, printed in tables("renewables")["report_gen"][kind].items():
            library = summary.loc[model]
            daylight = night.loc[(0.0, model), "MAE"] if kind == "SOL" else None
            values = (library["MAE"], 100 * library["nMAE"], 100 * library["MAE/cap"], daylight, library["Bias"])
            # evaluate.py printed the generation table at 2 decimals.
            for value, cell, digits in zip(values, printed, (0, 1, 2, 0, 0)):
                if cell is not None:
                    assert printed_equal(value, cell, digits, 2), (kind, model, value, cell)
        mean_actual = summary.loc["ridge", "MAE"] / summary.loc["ridge", "nMAE"]
        assert printed_equal(mean_actual, tables("renewables")["report_mean_actual"][kind], 0, 2)
    else:
        for model, cell in tables("renewables")["report_gen_val"][kind].items():
            assert printed_equal(summary.loc[model, "MAE"], cell, 0, 2), (kind, model)


@pytest.mark.parametrize("kind, period", GEN)
def test_generation_dm_list_is_reproduced(kind, period, generated):
    names = tables("renewables")["gen_names"]
    rows = [r for r in tables("renewables")["gen_dm"] if (r["target"], r["period"]) == (kind, period)]
    assert len(rows) == 4
    for row in rows:
        library = generated[(kind, period)]["dm"][(names[row["A"]], names[row["B"]])]
        assert abs(-library["statistic"] - row["DM"]) <= 1e-10, (row, library["statistic"])
        assert abs(library["p_value"] - row["p"]) <= 1e-10, (row, library["p_value"])
    if period == "test":
        for a, b, statistic in tables("renewables")["report_gen_dm"][kind]:
            library = generated[(kind, period)]["dm"][(a, b)]
            assert printed_equal(-library["statistic"], statistic, 1, 4), (a, b, library["statistic"], statistic)


def test_the_fixture_holds_the_studys_keys():
    prices = renewables("prices")
    assert prices.groupby("zone", observed=True).size().to_dict() == {"de_lu": 11140, "at": 10676}
    assert prices.groupby("zone", observed=True)["target_date"].nunique().to_dict() == {"de_lu": 467, "at": 467}
    generation = renewables("generation")
    counts = generation.groupby(["kind", "period"], observed=True).size().to_dict()
    assert counts == {("SOL", "test"): 11206, ("SOL", "val"): 4344, ("WIND", "test"): 11205, ("WIND", "val"): 4343}
    assert np.isfinite(generation.drop(columns=["kind", "period", "target_date", "hour"]).to_numpy(dtype=float)).all()
