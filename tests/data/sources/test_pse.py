"""PseSource: parsing, paging, filters, DST days, retries and cache, against a fake of the reports API.

The fake answers in the shape the live API does (checked 2026-11-08): a JSON object with `value` (a list of rows) and
`nextLink` (the same URL with `$after`). A 15-minute row is stamped with the END of its period (`dtime` local, `dtime_utc`),
`business_date` is the local date of the period's start, and `pdgsz` is hourly with one row per version of an hour. Values are
a known function of the period's start, so what the parser did to the time axis shows in the numbers.
"""

import numpy as np
import pandas as pd
import pytest
import requests

from epftoolbox2.data.sources import PseSource
from epftoolbox2.data.sources.pse import DATASET_COLUMNS, FLOW_COLUMNS, PseApiError
from epftoolbox2.pipelines.data_pipeline import COMPONENT_REGISTRY, DataPipeline
from tests.pse_fake import (
    bare_source,
    API,
    AUTUMN,
    SPRING,
    WARSAW,
    FakeApi,
    answer,
    local_day_starts,
    make_source,
    quarter_rows,
    signal,
    table,
    utc,
)


def expected_index(first_day: str, last_day: str, step="15min") -> pd.DatetimeIndex:
    starts = [local_day_starts(day, step) for day in pd.date_range(first_day, last_day, freq="D").strftime("%Y-%m-%d")]
    return starts[0].append(starts[1:]) if len(starts) > 1 else starts[0]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("epftoolbox2.data.sources.pse.time.sleep", lambda seconds: None)


# ---------------------------------------------------------------------------
# Time axis and DST
# ---------------------------------------------------------------------------


def test_the_index_is_the_period_start_in_utc_with_known_values():
    days = ["2026-06-10", "2026-06-11"]
    source = make_source({"kse-load": table("kse-load", days)}, ["kse-load"])

    df = source.fetch(utc("2026-06-09 22:00"), utc("2026-06-11 22:00"))

    index = expected_index("2026-06-10", "2026-06-11")
    assert list(df.columns) == ["load_actual", "load_fcst"]
    assert str(df.index.tz) == "UTC" and df.index.equals(index)
    assert df.index[0] == utc("2026-06-09 22:00")  # 00:00 CEST: the period 00:00 - 00:15, stamped 00:15 by PSE
    np.testing.assert_array_equal(df["load_actual"].to_numpy(), signal(index, 9000))
    np.testing.assert_array_equal(df["load_fcst"].to_numpy(), signal(index, 5000))
    assert (df.dtypes.astype(str) == "float64").all()


@pytest.mark.parametrize(
    "day, quarters, first_utc, last_utc",
    [
        (SPRING, 92, "2026-03-28 23:00", "2026-03-29 21:45"),  # 23 hours: CET to CEST at 02:00
        (AUTUMN, 100, "2026-10-24 22:00", "2026-10-25 22:45"),  # 25 hours: CEST to CET at 03:00
    ],
)
def test_a_dst_day_has_92_or_100_quarters_without_a_gap_or_a_repeat(day, quarters, first_utc, last_utc):
    source = make_source({"csdac-pln": table("csdac-pln", [day])}, ["csdac-pln"])

    df = source.fetch(utc(first_utc), utc(last_utc) + pd.Timedelta(minutes=15))

    assert len(df) == quarters
    assert df.index[0] == utc(first_utc) and df.index[-1] == utc(last_utc)
    assert (np.diff(df.index.asi8) == pd.Timedelta(minutes=15).value).all()  # one step each, in elapsed time
    np.testing.assert_array_equal(df["csdac_pln"].to_numpy(), signal(df.index, 100))
    local = df.index.tz_convert(WARSAW)
    assert local[0].strftime("%H:%M") == "00:00" and local[-1].strftime("%H:%M") == "23:45"
    assert df["csdac_pln"].notna().all()


def test_rows_before_start_and_from_end_on_are_clipped():
    source = make_source({"csdac-pln": table("csdac-pln", ["2026-06-10", "2026-06-11"])}, ["csdac-pln"])

    df = source.fetch(utc("2026-06-10 10:07"), utc("2026-06-10 11:00"))  # starts inside a quarter

    assert df.index.tolist() == [utc("2026-06-10 10:15"), utc("2026-06-10 10:30"), utc("2026-06-10 10:45")]


def test_a_sub_range_gives_the_values_of_the_larger_range():
    days = ["2026-03-28", SPRING, "2026-03-30"]
    source = make_source({"kse-load": table("kse-load", days), "pdgsz": table("pdgsz", days)}, ["kse-load", "pdgsz"])

    whole = source.fetch(utc("2026-03-28 00:00"), utc("2026-03-31 00:00"))
    part = source.fetch(utc("2026-03-29 03:00"), utc("2026-03-29 17:00"))

    pd.testing.assert_frame_equal(part, whole.loc[part.index[0] : part.index[-1]])


def test_an_empty_or_reversed_range_is_an_empty_frame_and_asks_nothing():
    source = make_source({"kse-load": []}, ["kse-load"])

    df = source.fetch(utc("2026-06-10"), utc("2026-06-10"))

    assert df.empty and list(df.columns) == ["load_actual", "load_fcst"] and str(df.index.tz) == "UTC"
    assert source.session.get.calls == []


def test_a_range_without_rows_is_an_empty_frame_with_the_columns_and_a_warning():
    source = make_source({"kse-load": [], "csdac-pln": []}, ["kse-load", "csdac-pln"], prefix="pse")

    df = source.fetch(utc("2020-01-01"), utc("2020-01-03"))

    assert df.empty and list(df.columns) == ["pse_load_actual", "pse_load_fcst", "pse_csdac_pln"]
    assert str(df.index.tz) == "UTC"
    assert source.logger.warning.call_count == 2


def test_missing_values_stay_nan():
    rows = quarter_rows("kse-load", "2026-06-10")
    rows[3]["load_actual"] = None
    source = make_source({"kse-load": rows}, ["kse-load"])

    df = source.fetch(utc("2026-06-09 22:00"), utc("2026-06-10 22:00"))

    assert df["load_actual"].isna().sum() == 1 and np.isnan(df["load_actual"].iloc[3])
    assert df["load_fcst"].notna().all()


# ---------------------------------------------------------------------------
# Requests: filter, paging, chunks, retries
# ---------------------------------------------------------------------------


def test_the_filter_is_on_the_local_business_dates_of_the_range():
    source = make_source({"csdac-pln": table("csdac-pln", ["2026-03-28", SPRING, "2026-03-30"])}, ["csdac-pln"])

    # The 23-hour day exactly: 00:00 CET to 00:00 CEST.
    source.fetch(utc("2026-03-28 23:00"), utc("2026-03-29 22:00"))
    # 24 UTC hours from the same start reach 00:59 on the next local day.
    source.fetch(utc("2026-03-28 23:00"), utc("2026-03-29 23:00"))

    assert source.session.get.filters() == [
        "business_date ge '2026-03-29' and business_date le '2026-03-29'",
        "business_date ge '2026-03-29' and business_date le '2026-03-30'",
    ]
    first = source.session.get.calls[0]
    assert first["url"] == f"{API}/csdac-pln" and first["params"]["$first"] == PseSource.PAGE_SIZE


def test_a_long_range_is_asked_in_chunks_and_each_dataset_on_its_own_url():
    source = make_source(
        {"csdac-pln": table("csdac-pln", ["2026-06-01"]), "kse-load": table("kse-load", ["2026-06-01"])},
        ["csdac-pln", "kse-load"],
    )

    source.fetch(utc("2026-05-31 22:00"), utc("2026-08-09 22:00"))  # 70 local days, 2026-06-01 to 2026-08-09

    calls = source.session.get.calls
    assert [c["url"].rsplit("/", 1)[1] for c in calls] == ["csdac-pln"] * 3 + ["kse-load"] * 3
    assert source.session.get.filters()[:3] == [
        "business_date ge '2026-06-01' and business_date le '2026-07-01'",
        "business_date ge '2026-07-02' and business_date le '2026-08-01'",
        "business_date ge '2026-08-02' and business_date le '2026-08-09'",
    ]


def test_pages_are_followed_through_nextlink_and_give_the_rows_of_one_page(monkeypatch):
    days = ["2026-06-10", "2026-06-11"]
    tables = {"kse-load": table("kse-load", days)}
    whole = make_source(tables, ["kse-load"]).fetch(utc("2026-06-09 22:00"), utc("2026-06-11 22:00"))

    monkeypatch.setattr(PseSource, "PAGE_SIZE", 70)
    paged = make_source(tables, ["kse-load"])
    df = paged.fetch(utc("2026-06-09 22:00"), utc("2026-06-11 22:00"))

    pd.testing.assert_frame_equal(df, whole)
    calls = paged.session.get.calls
    assert len(calls) == 3  # 192 rows in pages of 70
    assert calls[0]["params"]["$first"] == 70
    assert all(c["params"] is None and "$after=" in c["url"] for c in calls[1:])  # nextLink is used as it is
    assert all(c["headers"]["User-Agent"] == "epftoolbox2" for c in calls)


def test_a_next_link_to_another_host_is_refused():
    body = {"value": quarter_rows("kse-load", "2026-06-10")[:2], "nextLink": "https://example.org/api/kse-load?x=1"}
    source = make_source({}, ["kse-load"])
    source.session.get = FakeApi({}, scripted=[answer(body)])

    with pytest.raises(PseApiError, match="unexpected nextLink"):
        source.fetch(utc("2026-06-09 22:00"), utc("2026-06-10 22:00"))
    assert len(source.session.get.calls) == 1


def test_transient_errors_are_retried_honouring_retry_after(monkeypatch):
    waits = []
    monkeypatch.setattr("epftoolbox2.data.sources.pse.time.sleep", waits.append)
    ok = {"value": quarter_rows("csdac-pln", "2026-06-10")}
    source = make_source({}, ["csdac-pln"])
    source.session.get = FakeApi(
        {},
        scripted=[
            answer({"error": {"message": "busy"}}, status=503),
            answer({"error": {"message": "slow down"}}, status=429, headers={"Retry-After": "7"}),
            requests.ConnectionError("reset"),
            answer(ok),
        ],
    )

    df = source.fetch(utc("2026-06-09 22:00"), utc("2026-06-10 22:00"))

    assert len(df) == 96
    assert waits == [2.0, 7.0, 8.0]  # backoff 2 s, then the server's 7 s, then 2 s * 2**2


def test_a_client_error_raises_with_the_reason_and_is_not_retried():
    source = make_source({}, ["csdac-pln"])
    source.session.get = FakeApi({}, scripted=[answer({"error": {"message": "Invalid filter"}}, status=400)])

    with pytest.raises(PseApiError, match="HTTP 400: Invalid filter"):
        source.fetch(utc("2026-06-09 22:00"), utc("2026-06-10 22:00"))
    assert len(source.session.get.calls) == 1


def test_retries_run_out_into_an_error_not_a_gap():
    source = make_source({}, ["csdac-pln"])
    source.session.get = FakeApi({}, scripted=[answer({}, status=500)] * (PseSource.MAX_RETRIES + 1))

    with pytest.raises(PseApiError, match=r"failed after 6 attempts: HTTP 500"):
        source.fetch(utc("2026-06-09 22:00"), utc("2026-06-10 22:00"))


@pytest.mark.parametrize("body", [{"nothing": 1}, {"value": "x"}])
def test_an_answer_that_is_not_a_table_raises(body):
    source = make_source({}, ["csdac-pln"])
    source.session.get = FakeApi({}, scripted=[answer(body)])

    with pytest.raises(PseApiError, match="no 'value' list"):
        source.fetch(utc("2026-06-09 22:00"), utc("2026-06-10 22:00"))


def test_a_field_the_api_dropped_raises():
    rows = quarter_rows("kse-load", "2026-06-10")
    for row in rows:
        del row["load_fcst"]
    source = make_source({"kse-load": rows}, ["kse-load"])

    with pytest.raises(PseApiError, match=r"fields \['load_fcst'\] missing"):
        source.fetch(utc("2026-06-09 22:00"), utc("2026-06-10 22:00"))


def test_two_rows_for_one_period_raise():
    rows = quarter_rows("csdac-pln", "2026-06-10")
    source = make_source({"csdac-pln": rows + rows[:1]}, ["csdac-pln"])

    with pytest.raises(PseApiError, match="two rows for one period"):
        source.fetch(utc("2026-06-09 22:00"), utc("2026-06-10 22:00"))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def test_his_wlk_cal_loads_four_columns_by_default_and_any_field_on_request():
    tables = {"his-wlk-cal": table("his-wlk-cal", ["2026-06-10"])}
    default = make_source(tables, ["his-wlk-cal"]).fetch(utc("2026-06-09 22:00"), utc("2026-06-10 22:00"))
    chosen = make_source(tables, ["his-wlk-cal"], columns=["pv", "jgm", "jga"]).fetch(
        utc("2026-06-09 22:00"), utc("2026-06-10 22:00")
    )

    assert list(default.columns) == ["demand", "wi", "pv", "jg"]
    np.testing.assert_array_equal(default["wi"].to_numpy(), signal(default.index, 2))
    assert list(chosen.columns) == ["pv", "jga", "jgm"]  # the dataset's order, not the order asked for
    assert list(chosen.columns) == [c for c in DATASET_COLUMNS["his-wlk-cal"] if c in {"pv", "jgm", "jga"}]
    assert chosen["jga"].isna().all() and (chosen["jgm"] == -1.5).all()


def test_flows_become_one_column_per_section_and_unknown_sections_are_left_out_with_a_warning():
    rows = table("przeplywy-mocy", ["2026-06-10"])
    rows.append({**rows[0], "section_code": "XX-PL", "value": 5.0})
    source = make_source({"przeplywy-mocy": rows}, ["przeplywy-mocy"])

    df = source.fetch(utc("2026-06-09 22:00"), utc("2026-06-10 22:00"))

    assert list(df.columns) == list(FLOW_COLUMNS) and len(df) == 96
    np.testing.assert_array_equal(df["flow_de_pl"].to_numpy(), signal(df.index, 1000))
    np.testing.assert_array_equal(df["flow_pl_de"].to_numpy(), signal(df.index, 2000))
    assert df["flow_cz_pl"].isna().all()  # a section PSE did not send in the window
    assert "XX-PL" in source.logger.warning.call_args.args[0]


def pdgsz_source(day, versions, **kwargs):
    return make_source({"pdgsz": table("pdgsz", [day], versions=versions)}, ["pdgsz"], **kwargs)


def day_range(day):
    hours = local_day_starts(day, "1h")
    return hours[0], hours[-1] + pd.Timedelta(hours=1)


def test_pdgsz_takes_the_latest_version_stamped_by_the_cutoff_on_the_day_before():
    """Versions 1 (D-1 15:40), 2 (D-1 17:00), 3 (D-1 18:30) and 4 (D 08:00, the active one): the cutoff picks neither the
    first nor the active version but the last one stamped at or before it."""
    versions = (
        ("2026-06-09 15:40:00.000", 1),
        ("2026-06-09 17:00:00.000", 2),
        ("2026-06-09 18:30:00.000", 3),
        ("2026-06-10 08:00:00.000", 4),
    )
    got = {}
    for cutoff in ("15:00", "15:40", "16:59", "17:00", "19:00", "23:59"):
        df = pdgsz_source("2026-06-10", versions, pdgsz_cutoff=cutoff).fetch(*day_range("2026-06-10"))
        assert len(df) == 96  # every hour over its four quarters, with or without a value
        got[cutoff] = set(df["usage_fcst"].dropna()), int(df["usage_fcst"].isna().sum())

    assert got == {
        "15:00": (set(), 96),  # nothing yet: NaN, not the first version
        "15:40": ({1.0}, 0),  # the stamp itself is at the cutoff
        "16:59": ({1.0}, 0),
        "17:00": ({2.0}, 0),
        "19:00": ({3.0}, 0),
        "23:59": ({3.0}, 0),  # D-1 only: the version of D 08:00 is never read
    }


def test_a_version_stamped_one_second_after_the_cutoff_is_excluded():
    versions = (
        ("2026-06-09 18:00:00.000", 1),
        ("2026-06-09 19:00:00.000", 2),
        ("2026-06-09 19:00:01.000", 3),
    )
    df = pdgsz_source("2026-06-10", versions).fetch(*day_range("2026-06-10"))
    assert set(df["usage_fcst"]) == {2.0}

    early = pdgsz_source("2026-06-10", versions, pdgsz_cutoff="18:59").fetch(*day_range("2026-06-10"))
    assert set(early["usage_fcst"]) == {1.0}


def test_a_single_version_written_on_the_delivery_day_is_nan_under_any_cutoff_on_the_day_before():
    """The 2024 to spring 2025 regime: one version per hour, stamped about 22:35 on the delivery day."""
    versions = (("2025-01-15 22:35:00.000", 1),)
    for cutoff in ("00:00", "19:00", "23:59"):
        df = pdgsz_source("2025-01-15", versions, pdgsz_cutoff=cutoff).fetch(*day_range("2025-01-15"))
        assert len(df) == 96 and df["usage_fcst"].isna().all()


@pytest.mark.parametrize("day, quarters", [(SPRING, 92), (AUTUMN, 100)])
def test_pdgsz_on_a_dst_day_has_23_or_25_hours_of_four_quarters(day, quarters):
    """The default versions: two on D-1 (15:40, 18:30) and the active one on D at 08:00, so the cutoff 19:00 gives 2."""
    df = pdgsz_source(day, None).fetch(*day_range(day))

    assert len(df) == quarters
    assert (np.diff(df.index.asi8) == pd.Timedelta(minutes=15).value).all()
    assert (df["usage_fcst"] == 2.0).all()


def test_the_cutoff_is_read_in_warsaw_time_on_the_day_before_a_dst_day():
    """D-1 of the spring switch day is 2026-03-28 (CET, UTC+1): 18:30 local is 17:30Z."""
    versions = (("2026-03-28 18:30:00.000", 1), ("2026-03-28 19:30:00.000", 2))
    df = pdgsz_source(SPRING, versions, pdgsz_cutoff="19:00").fetch(*day_range(SPRING))
    assert set(df["usage_fcst"]) == {1.0}


# ---------------------------------------------------------------------------
# Configuration, YAML, cache
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"datasets": []}, "at least one dataset"),
        ({"datasets": ["kse-load", "price-fcs"]}, r"Unknown PSE dataset\(s\) \['price-fcs'\]"),
        ({"datasets": ["kse-load", "kse-load"]}, "Duplicate datasets"),
        ({"datasets": ["kse-load"], "columns": ["pv"]}, r"Column\(s\) \['pv'\] are not in the datasets"),
        ({"datasets": ["kse-load", "csdac-pln"], "columns": ["load_fcst"]}, "No column of the dataset 'csdac-pln'"),
        ({"datasets": ["kse-load"], "columns": ["load_fcst", "load_fcst"]}, "Duplicate columns"),
    ],
)
def test_invalid_configuration_raises(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PseSource(**kwargs)


@pytest.mark.parametrize("cutoff", ["19", "7:00", "24:00", "19:60", "", None, 19])
def test_a_bad_pdgsz_cutoff_raises(cutoff):
    with pytest.raises(ValueError, match="pdgsz_cutoff"):
        PseSource(pdgsz_cutoff=cutoff)


def test_the_cutoff_is_part_of_the_cache_key():
    assert PseSource(pdgsz_cutoff="19:00").get_cache_config() != PseSource(pdgsz_cutoff="18:00").get_cache_config()
    assert PseSource().get_cache_config()["pdgsz_cutoff"] == "19:00"


def test_every_column_is_prefixed_by_default():
    columns = [c for d in DATASET_COLUMNS for c in PseSource(datasets=[d]).output_columns()]
    assert len(columns) == sum(len(v) for v in DATASET_COLUMNS.values()) - 13  # his-wlk-cal: 4 by default
    assert all(c.startswith("pse_") for c in columns)
    assert PseSource().output_columns() == ["pse_load_actual", "pse_load_fcst", "pse_csdac_pln"]
    assert PseSource().prefix == "pse"


def test_columns_prefix_and_the_default_datasets():
    assert PseSource().datasets == ["kse-load", "csdac-pln"]
    source = PseSource(datasets=["kse-load", "csdac-pln"], columns=["csdac_pln", "load_fcst"], prefix="pl")
    assert source.output_columns() == ["pl_load_fcst", "pl_csdac_pln"]


def test_the_cache_key_changes_with_every_parameter_that_changes_the_output():
    base = PseSource(datasets=["kse-load", "his-wlk-cal"], prefix="a").get_cache_config()
    assert base["source_type"] == "pse" and "schema_version" in base
    others = [
        PseSource(datasets=["kse-load", "his-wlk-cal"], prefix="b"),
        PseSource(datasets=["kse-load"], prefix="a"),
        PseSource(datasets=["kse-load", "his-wlk-cal"], columns=["load_fcst", "demand"], prefix="a"),
        PseSource(datasets=["his-wlk-cal", "kse-load"], prefix="a"),
    ]
    assert all(other.get_cache_config() != base for other in others)
    assert PseSource(datasets=["kse-load", "his-wlk-cal"], prefix="a").get_cache_config() == base


def test_yaml_round_trip_and_registry(tmp_path):
    assert COMPONENT_REGISTRY["sources"]["PseSource"] == "epftoolbox2.data.sources.pse"
    original = PseSource(datasets=["kse-load", "pdgsz"], columns=["load_fcst", "usage_fcst"], prefix="pse")
    path = tmp_path / "pipeline.yaml"
    DataPipeline().add_source(original).save(path)

    loaded = DataPipeline.load(path).sources[0]

    assert isinstance(loaded, PseSource)
    assert (loaded.datasets, loaded.columns, loaded.prefix) == (
        ["kse-load", "pdgsz"],
        ["load_fcst", "usage_fcst"],
        "pse",
    )
    assert "api_key" not in path.read_text()


def test_the_pipeline_cache_serves_a_second_run_without_a_request(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    days = pd.date_range("2026-06-01", "2026-06-06").strftime("%Y-%m-%d").tolist()
    source = make_source({"kse-load": table("kse-load", days)}, ["kse-load"])
    pipeline = DataPipeline().add_source(source)

    first = pipeline.run("2026-06-01", "2026-06-04", cache=True)
    asked = len(source.session.get.calls)
    second = pipeline.run("2026-06-01", "2026-06-04", cache=True)
    wider = pipeline.run("2026-06-01", "2026-06-05", cache=True)

    assert asked > 0 and len(first) == 4 * 96
    assert len(source.session.get.calls) > asked  # the extra day was fetched, and only it
    assert source.session.get.filters()[-1] == "business_date ge '2026-06-05' and business_date le '2026-06-06'"
    pd.testing.assert_frame_equal(second, first)
    pd.testing.assert_frame_equal(wider.iloc[: len(first)], first)


# ---------------------------------------------------------------------------
# The live API (keyless; a few days of settled history)
# ---------------------------------------------------------------------------

LIVE_DATASETS = ["kse-load", "his-wlk-cal", "csdac-pln", "rce-pln", "pdgsz", "przeplywy-mocy"]


@pytest.mark.integration
@pytest.mark.parametrize(
    "day, quarters, first_utc",
    [("2025-10-26", 100, "2025-10-25 22:00"), ("2026-03-29", 92, "2026-03-28 23:00")],
)
def test_live_dst_days_have_92_or_100_quarters_of_every_dataset(day, quarters, first_utc):
    source = bare_source(LIVE_DATASETS)
    start = utc(first_utc)

    df = source.fetch(start, start + pd.Timedelta(minutes=15 * quarters))

    assert len(df) == quarters and df.index[0] == start
    assert (np.diff(df.index.asi8) == pd.Timedelta(minutes=15).value).all()
    assert list(df.columns) == source.output_columns()
    assert df.notna().all().all() or set(df.columns[df.isna().all()]) <= {"jga", "jgo", "jgz1", "jgz3"}
    assert df["csdac_pln"].between(-500, 5000).all() and df["load_actual"].between(5000, 40000).all()
    assert set(df["usage_fcst"]) <= {0.0, 1.0, 2.0, 3.0}
    assert df["load_actual"].equals(df["demand"]) or (df["load_actual"] - df["demand"]).abs().max() < 500


@pytest.mark.integration
def test_live_the_index_is_the_start_of_the_period_pse_stamps_with_its_end():
    source = bare_source(["csdac-pln"])
    raw = source._request_table("csdac-pln", pd.Timestamp("2026-03-29"), pd.Timestamp("2026-03-29"))
    first, last = raw[0], raw[-1]

    df = source.fetch(utc("2026-03-28 23:00"), utc("2026-03-29 22:00"))

    assert first["period_utc"] == "23:00 - 23:15" and df.index[0] == utc("2026-03-28 23:00")
    assert last["period_utc"] == "21:45 - 22:00" and df.index[-1] == utc("2026-03-29 21:45")
    assert df["csdac_pln"].iloc[0] == pytest.approx(first["csdac_pln"])
    assert df["csdac_pln"].iloc[-1] == pytest.approx(last["csdac_pln"])
    assert len(raw) == len(df) == 92


@pytest.mark.integration
def test_live_a_range_before_the_history_is_empty_and_a_year_pages_through_nextlink():
    source = bare_source(["csdac-pln"])

    assert source.fetch(utc("2020-01-01"), utc("2020-01-03")).empty
    year = source.fetch(utc("2025-01-01"), utc("2026-01-01"))
    assert len(year) == 35040 and year.index.is_unique and year["csdac_pln"].notna().all()
