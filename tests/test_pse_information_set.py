"""The built-in availability rules of PseSource's columns, and a model on a PSE frame through `verify`.

Table of the rules (Europe/Warsaw; a delivery day D):
`kse-load` load_fcst known 09:00 on D-1; `csdac-pln` and `rce-pln` 15:00 on D-1; `pdgsz` 16:00 on D-1; the actuals of `kse-load`,
all of `his-wlk-cal` and `przeplywy-mocy` at 13:00 on D+1. So a forecast issued at 11:45 on D has D+1's load forecast, the
prices of D (published on D-1) and the actuals up to D-2, and has not D-1's actual load (known at 13:00 on D).
"""

from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from epftoolbox2.data.sources import PseSource
from epftoolbox2.data.sources.pse import DATASET_COLUMNS
from epftoolbox2.data.transformers import LagTransformer, ResampleTransformer, TimezoneTransformer
from epftoolbox2.information_set import Availability, InformationSet, RuleChain, source_rules, vintage_of
from epftoolbox2.models import OLSModel
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import DataPipeline, ModelPipeline
from tests.pse_fake import WARSAW, local_day_starts, make_source, table

EVERY_DATASET = ["kse-load", "his-wlk-cal", "csdac-pln", "rce-pln", "pdgsz", "przeplywy-mocy"]
EXPECTED = {  # column: (time, days_before, class)
    "load_fcst": ("09:00", 1, "last_write_stamp"),
    "load_actual": ("13:00", -1, "last_write_stamp"),
    "csdac_pln": ("15:00", 1, "last_write_stamp"),
    "rce_pln": ("15:00", 1, "last_write_stamp"),
    "usage_fcst": ("16:00", 1, "row_stamp"),
}


def every_column_source(**kwargs):
    columns = [c for d in EVERY_DATASET for c in DATASET_COLUMNS[d]]
    return PseSource(datasets=EVERY_DATASET, columns=columns, **kwargs)


def test_every_column_has_the_rule_of_its_table():
    rules = source_rules([every_column_source()])

    assert set(rules) == {c for d in EVERY_DATASET for c in DATASET_COLUMNS[d]}
    for name, rule in rules.items():
        assert (rule.kind, rule.timezone) == ("published", WARSAW)
        assert (rule.time, rule.days_before, rule.vintage) == EXPECTED.get(name, ("13:00", -1, "last_write_stamp"))
        assert rule.note and not rule.bounded
    assert {n for n, r in rules.items() if r.days_before == -1} >= {
        "demand",
        "wi",
        "pv",
        "jg",
        "flow_de_pl",
        "flow_pl_de",
    }
    assert all(vintage_of(rules[name]) == "last_write_stamp" for name in ("load_actual", "demand", "flow_pl_cz"))
    assert vintage_of(rules["usage_fcst"]) == "row_stamp"


def test_a_prefix_and_a_column_choice_follow_the_source():
    source = PseSource(datasets=["kse-load", "csdac-pln"], columns=["load_actual", "csdac_pln"], prefix="pse")

    rules = source_rules([source])

    assert set(rules) == {"pse_load_actual", "pse_csdac_pln"}
    assert rules["pse_csdac_pln"].days_before == 1 and rules["pse_load_actual"].days_before == -1


def test_the_source_s_rules_replace_the_entsoe_defaults_of_the_same_names():
    """`load_actual` is the name of ENTSO-E's column too: with a PseSource in the pipeline it is PSE's, known a day later."""
    plain = InformationSet().resolve("load_actual_d-1")
    info = InformationSet("11:45", WARSAW).with_source_rules([PseSource(datasets=["kse-load"])])
    column = info.resolve("load_actual_d-1")

    assert (plain.layer, plain.rule.kind) == ("defaults", "realised")
    assert (column.layer, column.rule.kind, column.rule.days_before) == ("sources", "published", -1)
    assert info.resolve("load_fcst_d+1").layer == "sources"


def late(info, names, freq, index, start, end):
    """{predictor: periods at which it is not known when the forecast of D+1 is issued at 11:45 on D}."""
    periods = 24 if freq == "1h" else 96
    expanded = {(p, 1): names for p in range(periods)}
    found, unknown = info.violations(expanded, "csdac_pln", freq, index, start, end)
    assert unknown == set()
    out = {}
    for violation in found:
        out.setdefault(violation.predictor, set()).add(violation.period)
    return out


@pytest.mark.parametrize("freq, periods", [("1h", 24), ("15min", 96)])
def test_a_forecast_issued_at_1145_sees_the_next_day_s_load_forecast_and_not_the_day_before_s_actual(freq, periods):
    index = pd.date_range("2026-03-10", "2026-04-10", freq=freq, tz=WARSAW, inclusive="left")  # past the spring switch
    info = InformationSet("11:45", WARSAW).with_source_rules([every_column_source()])
    known = [
        "load_fcst_d+1",  # 09:00 on D
        "csdac_pln",  # D's prices, 15:00 on D-1
        "csdac_pln_d-1",
        "rce_pln",
        "usage_fcst",  # D's, 16:00 on D-1
        "load_actual_d-2",  # 13:00 on D-1
        "demand_d-2",
        "wi_d-2",
        "flow_de_pl_d-2",
    ]
    unknown_then = [
        "load_actual",  # D's own, 13:00 on D+1
        "load_actual_d-1",  # 13:00 on D, after the issue
        "demand_d-1",
        "pv_d-1",
        "flow_pl_de_d-1",
        "load_fcst_d+2",  # 09:00 on D+1
        "csdac_pln_d+1",  # 15:00 on D
        "rce_pln_d+1",
        "usage_fcst_d+1",
    ]

    found = late(info, known + unknown_then, freq, index, "2026-03-16", "2026-04-05")

    assert found == {name: set(range(periods)) for name in unknown_then}


def test_the_next_days_actual_is_known_at_13_00_on_the_day_after_not_before():
    """The rule in elapsed time: an issue at 13:15 sees D-1's actual, one at 12:45 does not."""
    index = pd.date_range("2026-03-10", "2026-04-10", freq="1h", tz=WARSAW, inclusive="left")
    after = InformationSet("13:15", WARSAW).with_source_rules([PseSource(datasets=["kse-load", "csdac-pln"])])
    before = InformationSet("12:45", WARSAW).with_source_rules([PseSource(datasets=["kse-load", "csdac-pln"])])

    assert late(after, ["load_actual_d-1"], "1h", index, "2026-03-16", "2026-04-05") == {}
    assert late(before, ["load_actual_d-1"], "1h", index, "2026-03-16", "2026-04-05") == {
        "load_actual_d-1": set(range(24))
    }


def test_a_user_rule_chain_for_a_pse_column_still_wins():
    chain = RuleChain(
        (
            Availability.published("14:30", -4, WARSAW).during(valid_until="2026-10-07"),
            Availability.published("14:30", -1, WARSAW).during(valid_from="2026-10-08"),
        )
    )
    info = InformationSet("11:45", WARSAW, rules={"load_actual": chain}).with_source_rules(
        [PseSource(datasets=["kse-load"])]
    )

    assert info.resolve("load_actual_d-1").layer == "rules"


# ---------------------------------------------------------------------------
# A model on a frame that went through the source
# ---------------------------------------------------------------------------


class InProcessPool:
    """Stands in for the worker pool (see tests/pipelines/test_verify.py)."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        _worker_init(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, task, *args):
        future = Future()
        try:
            future.set_result(task(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)
    with threadpool_limits(limits=1):
        yield


FIRST, LAST = "2026-03-25", "2026-05-22"


@pytest.fixture(scope="module")
def pse_frame():
    """Hourly local frame built from a fake PSE answer through PseSource: the price follows the load forecast of its own
    quarter-hour, the day before's price and the day before's actual load."""
    days = pd.date_range(FIRST, LAST).strftime("%Y-%m-%d").tolist()
    starts = pd.DatetimeIndex(np.concatenate([local_day_starts(d, "15min") for d in days]))
    rng = np.random.default_rng(3)
    n = len(starts)
    load_actual = 15000 + 2500 * np.sin(2 * np.pi * (starts.hour + starts.minute / 60 - 6) / 24) + rng.normal(0, 300, n)
    load_fcst = load_actual + rng.normal(0, 150, n)
    price = np.zeros(n)
    for i in range(n):
        previous = price[i - 96] if i >= 96 else 0.0
        earlier_load = load_actual[i - 96] if i >= 96 else 15000.0
        price[i] = 40 + 0.01 * load_fcst[i] + 0.4 * previous + 0.01 * earlier_load + rng.normal(0, 3)
    series = pd.DataFrame({"load_actual": load_actual, "load_fcst": load_fcst, "csdac_pln": price}, index=starts)

    source = make_source(
        {"kse-load": table("kse-load", days, series), "csdac-pln": table("csdac-pln", days, series)},
        ["kse-load", "csdac-pln"],
    )
    pipeline = (
        DataPipeline()
        .add_source(source)
        .add_transformer(TimezoneTransformer(WARSAW))
        .add_transformer(ResampleTransformer("1h"))
        .add_transformer(LagTransformer(["csdac_pln", "load_actual"], [1, 2], "day"))
        .add_transformer(LagTransformer(["load_fcst"], [-1, -2], "day"))
    )
    return source, pipeline.run(FIRST, LAST)


def verify(frame_and_source, predictors, **kwargs):
    source, frame = frame_and_source
    information_set = InformationSet("11:45", WARSAW).with_source_rules([source])
    model = OLSModel(predictors, training_window=30, name="OLS")
    return (
        ModelPipeline(information_set=information_set)
        .add_model(model)
        .verify(frame, "2026-05-04", "2026-05-20", target="csdac_pln", horizon=1, days=1, **kwargs)
    )


def test_the_frame_has_the_columns_the_rules_name(pse_frame):
    _, frame = pse_frame
    assert {"csdac_pln", "csdac_pln_d-1", "load_actual_d-2", "load_fcst_d+1"} <= set(frame.columns)
    assert str(frame.index.tz) == WARSAW and frame.index.is_unique
    spring = frame.loc["2026-03-29"]
    assert len(spring) == 23  # 92 quarters resampled to 23 hours


def test_a_model_on_values_known_at_the_issue_passes_verify(pse_frame):
    report = verify(pse_frame, ["load_fcst_d+{horizon}", "csdac_pln_d-1", "load_actual_d-2"])

    assert report.ok, report.problems()
    assert report.table()["status"].tolist() == ["ok"]
    assert report.table()["forecasts"].tolist() == [24]


def test_a_model_on_the_day_before_s_actual_load_is_a_leak(pse_frame):
    """`load_actual_d-1` is known at 13:00 on D, after the 11:45 issue: the probe moves the forecast with it."""
    report = verify(pse_frame, ["load_fcst_d+{horizon}", "csdac_pln_d-1", "load_actual_d-1"], positive_control=False)

    assert report.table()["status"].tolist() == ["leak"]
    assert not report.ok
