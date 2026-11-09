"""`ForecastSource`: another pipeline's stored forecasts as a column, at their target instants.

The placement grid writes an upstream's records by hand, one per slot the clock of its data zone has, with a value that names its
key, and reads them back: every value at its slot's instant (both instants of a repeated autumn slot, none of a skipped spring
one), no other row, and `x_d+1` on the row of D equal to the record issued on D at horizon 1. Every level of every axis (upstream
freq 1h, 15min and 1D; data zones Warsaw, Helsinki, Lisbon, London and UTC; both switches) is covered with every pair of levels,
in all 30 cells (they cost nothing). The rest pins what the source refuses and what it tells the caches.
"""

import datetime
import json
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.sources import DataSource, ForecastSource, UpstreamError
from epftoolbox2.data.transformers import LagTransformer
from epftoolbox2.models._calendar import clock_periods
from epftoolbox2.pipelines import DataPipeline
from epftoolbox2.pipelines.data_pipeline import COMPONENT_REGISTRY
from epftoolbox2.results.parquet_store import convert

ZONES = ["Europe/Warsaw", "Europe/Helsinki", "Europe/Lisbon", "Europe/London", "UTC"]
FREQS = ["1h", "15min", "1D"]
# Four target dates around each EU switch of 2024 (UTC has none, and is placed all the same).
SWITCHES = {"spring": "2024-03-29", "autumn": "2024-10-25"}
DAYS = 5
HORIZONS = (1, 2)
STEP = {"1h": 60, "15min": 15}
STAMP = "12:00"


def value_of(target_date: datetime.date, period: int, horizon: int) -> float:
    """A record's prediction, naming its key."""
    return target_date.toordinal() * 1000.0 + period + horizon / 10.0


def write_upstream(
    directory: Path,
    name: str = "up",
    zone: str = "Europe/Warsaw",
    freq: str = "1h",
    first: str = "2024-03-29",
    days: int = DAYS,
    issue_time: str = "09:30",
    issue_zone: str = "Europe/Brussels",
    horizons=HORIZONS,
    manifest: dict = None,
    records: list = None,
) -> list:
    """An upstream's results and manifest written by hand: a record per (target date, period on the clock, horizon)."""
    directory.mkdir(parents=True, exist_ok=True)
    start = datetime.date.fromisoformat(first)
    out = []
    for offset in range(days):
        target = start + datetime.timedelta(days=offset)
        periods = (0,) if freq == "1D" else clock_periods(target, freq, zone)
        for horizon in horizons:
            run = target - datetime.timedelta(days=horizon)
            for period in periods:
                out.append(
                    {
                        "run_date": run.isoformat(),
                        "target_date": target.isoformat(),
                        "run_weekday": run.weekday(),
                        "hour": period,
                        "horizon": horizon,
                        "day_in_test": offset,
                        "prediction": value_of(target, period, horizon),
                        "actual": None,
                        "coefficients": [],
                        "freq": freq,
                        "status": "ok",
                        "revision": 1,
                    }
                )
    records = out if records is None else records
    (directory / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    recorded = {
        "model_class": "OLSModel",
        "target": "x",
        "horizon": max(horizons),
        "freq": freq,
        "data_timezone": zone,
        "information_set": {"issue_time": issue_time, "issue_timezone": issue_zone, "unknown": "raise"},
        "availability": {"level": "warn", "times": "instants", "listing": []},
    }
    if freq == "1D":
        recorded["daily_stamp"] = STAMP
    recorded.update(manifest or {})
    (directory / f"{name}.manifest.json").write_text(json.dumps({k: v for k, v in recorded.items() if v is not None}))
    return records


def everything(source: ForecastSource) -> pd.DataFrame:
    return source.fetch(pd.Timestamp("2000-01-01", tz="UTC"), pd.Timestamp("2100-01-01", tz="UTC"))


GRID = [(freq, zone, switch) for freq in FREQS for zone in ZONES for switch in SWITCHES]


@pytest.mark.parametrize(("freq", "zone", "switch"), GRID)
def test_every_value_is_at_its_slot_s_instants_and_x_d_plus_1_is_the_forecast_issued_on_d(tmp_path, freq, zone, switch):
    records = write_upstream(tmp_path, zone=zone, freq=freq, first=SWITCHES[switch])
    for horizon in HORIZONS:
        frame = everything(ForecastSource(tmp_path, "up", "x", horizon=horizon))
        assert frame.index.tz is not None and str(frame.index.tz) == "UTC"
        mine = [r for r in records if r["horizon"] == horizon]
        expected = {}
        for record in mine:
            date = pd.Timestamp(record["target_date"])
            minutes = 12 * 60 if freq == "1D" else record["hour"] * STEP[freq]
            wall = date + pd.Timedelta(minutes=minutes)
            instants = {wall.tz_localize(zone, ambiguous=flag) for flag in (True, False)}  # both of a repeated slot
            for instant in instants:
                expected[instant.tz_convert("UTC")] = record["prediction"]
        assert frame.index.is_monotonic_increasing and not frame.index.has_duplicates
        assert dict(zip(frame.index, frame["x"])) == expected  # every value at its instants, and no other row

        if freq == "1D":
            continue
        # A slot the clock skips has no record and no row; a repeated one has its value on both of its rows.
        first = pd.Timestamp(SWITCHES[switch])
        local = pd.date_range(
            first.tz_localize(zone),
            (first + pd.Timedelta(days=DAYS + 1)).tz_localize(zone),
            freq=freq,
            inclusive="left",
        ).tz_convert("UTC")  # whole local days, one past the records
        column = frame["x"].reindex(local).tz_convert(zone).rename("x").to_frame()
        with warnings.catch_warnings():  # a UTC frame's day lags are UTC days, as they should be here
            warnings.filterwarnings("ignore", message="LagTransformer: day lags on a UTC index")
            lagged = LagTransformer(["x"], [-1], "day").transform(column)
        # `x_d+1` on the row of D holds the forecast for D+1 issued on D+1-horizon: on D at horizon 1.
        keyed = {(r["run_date"], r["hour"]): r["prediction"] for r in mine}
        checked = 0
        for row_wall, value in zip(lagged.index.tz_localize(None), lagged["x_d+1"]):
            run = (row_wall.date() + datetime.timedelta(days=1 - horizon)).isoformat()
            key = (run, int((row_wall.hour * 60 + row_wall.minute) // STEP[freq]))
            if np.isnan(value):  # D+1 is past the records, or its clock skips the slot
                assert key not in keyed, (row_wall, key)
            else:
                assert value == keyed[key], (row_wall, key)
                checked += 1
        assert checked >= (DAYS - 2) * 23 * 60 // STEP[freq]


def test_a_daily_upstream_is_placed_at_its_stamp_on_the_target_date(tmp_path):
    write_upstream(tmp_path, zone="Europe/Helsinki", freq="1D", first="2024-10-25")
    frame = everything(ForecastSource(tmp_path, "up", "x"))
    local = frame.index.tz_convert("Europe/Helsinki")
    assert set(local.strftime("%H:%M")) == {STAMP}
    assert [value_of(d.date(), 0, 1) for d in local.tz_localize(None).normalize()] == list(frame["x"])


def test_a_record_of_a_slot_the_clock_skips_has_no_row(tmp_path):
    records = write_upstream(tmp_path, first="2024-03-31", days=1, horizons=(1,))
    records.append({**records[0], "hour": 2, "prediction": -1.0})  # 02:00 on the spring switch day, which has none
    write_upstream(tmp_path, first="2024-03-31", days=1, horizons=(1,), records=records)
    frame = everything(ForecastSource(tmp_path, "up", "x"))
    assert -1.0 not in set(frame["x"]) and len(frame) == 23


def test_fetch_returns_start_to_end_and_a_sub_range_holds_the_same_values(tmp_path):
    write_upstream(tmp_path, first="2024-10-25")
    source = ForecastSource(tmp_path, "up", "x")
    whole = everything(source)
    start, end = pd.Timestamp("2024-10-26 22:00", tz="UTC"), pd.Timestamp("2024-10-27 05:00", tz="UTC")
    part = source.fetch(start, end)
    assert part.index.min() == start and part.index.max() < end
    pd.testing.assert_frame_equal(part, whole[(whole.index >= start) & (whole.index < end)])


def test_jsonl_and_parquet_upstreams_give_bitwise_equal_frames(tmp_path):
    write_upstream(tmp_path / "jsonl", first="2024-10-25", freq="15min", zone="Europe/Helsinki")
    shutil.copytree(tmp_path / "jsonl", tmp_path / "parquet")
    convert(tmp_path / "parquet", to="parquet")
    assert (tmp_path / "parquet" / "up.parquet").is_dir() and not (tmp_path / "parquet" / "up.jsonl").exists()
    for horizon in HORIZONS:
        a = everything(ForecastSource(tmp_path / "jsonl", "up", "x", horizon=horizon))
        b = everything(ForecastSource(tmp_path / "parquet", "up", "x", horizon=horizon))
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        assert np.array_equal(a["x"].to_numpy().view(np.int64), b["x"].to_numpy().view(np.int64))


@pytest.mark.parametrize(
    ("manifest", "match"),
    [
        ({"information_set": None}, "without an issue time"),
        ({"data_timezone": "None"}, "no named time zone"),
        ({"freq": "1D", "daily_stamp": None}, "daily_stamp"),
    ],
)
def test_an_upstream_that_can_t_say_when_or_where_raises(tmp_path, manifest, match):
    write_upstream(tmp_path, manifest=manifest)
    source = ForecastSource(tmp_path, "up", "x")
    with pytest.raises(UpstreamError, match=match):
        everything(source)
    with pytest.raises(UpstreamError, match=match):
        source.availability()


def test_a_daily_upstream_without_its_stamp_names_the_rerun(tmp_path):
    write_upstream(tmp_path, freq="1D", manifest={"daily_stamp": None})
    with pytest.raises(UpstreamError, match=r"Run the upstream again into .*resume"):
        everything(ForecastSource(tmp_path, "up", "x"))


def test_a_live_record_with_issued_at_raises(tmp_path):
    records = write_upstream(tmp_path)
    records[-1] = {**records[-1], "issued_at": "2024-03-30T09:31:00+01:00", "revision": 2}
    write_upstream(tmp_path, records=records)
    with pytest.raises(UpstreamError, match="issued_at"):
        everything(ForecastSource(tmp_path, "up", "x", horizon=1))


def test_no_manifest_raises_and_names_the_save_dir(tmp_path):
    write_upstream(tmp_path)
    (tmp_path / "up.manifest.json").unlink()
    with pytest.raises(UpstreamError, match="save_dir"):
        everything(ForecastSource(tmp_path, "up", "x"))


def test_an_empty_range_warns_and_a_horizon_the_upstream_lacks_raises(tmp_path):
    write_upstream(tmp_path, horizons=(1,))
    source = ForecastSource(tmp_path, "up", "x")
    with pytest.warns(UserWarning, match="no forecast of horizon 1"):
        out = source.fetch(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-02-01", tz="UTC"))
    assert out.empty and list(out.columns) == ["x"]
    with pytest.raises(UpstreamError, match="no forecast of horizon 3"):
        everything(ForecastSource(tmp_path, "up", "x", horizon=3))


def test_an_empty_upstream_raises(tmp_path):
    write_upstream(tmp_path, records=[])
    with pytest.raises(UpstreamError, match="no forecast of horizon 1"):
        everything(ForecastSource(tmp_path, "up", "x"))


@pytest.mark.parametrize(
    "kwargs",
    [{"horizon": 0}, {"horizon": 1.5}, {"horizon": True}, {"column": ""}, {"model": ""}, {"available": "25:00"}],
)
def test_bad_parameters_raise_when_built(tmp_path, kwargs):
    with pytest.raises(ValueError):
        ForecastSource(**{"path": str(tmp_path), "model": "up", "column": "x", **kwargs})


def test_available_states_a_later_time_never_an_earlier_one(tmp_path):
    write_upstream(tmp_path, issue_time="09:30", issue_zone="Europe/Brussels")
    rule = ForecastSource(tmp_path, "up", "x", available="10:15").availability()
    assert (rule.time, rule.timezone, rule.days_before, rule.day_zone) == (
        "10:15",
        "Europe/Brussels",
        1,
        "Europe/Warsaw",
    )
    with pytest.raises(UpstreamError, match="before the upstream's issue time"):
        ForecastSource(tmp_path, "up", "x", available="09:00").availability()


def test_the_rule_is_the_upstream_s_issue_time_on_its_data_zone_s_dates(tmp_path):
    write_upstream(tmp_path, zone="Europe/Helsinki", issue_time="08:45", issue_zone="Europe/Berlin")
    rule = ForecastSource(tmp_path, "up", "x", horizon=2).availability()
    assert rule.to_dict() == {
        "kind": "published",
        "time": "08:45",
        "days_before": 2,
        "timezone": "Europe/Berlin",
        "day_zone": "Europe/Helsinki",
        "vintage": "snapshot",
        "note": rule.note,
    }
    write_upstream(
        tmp_path / "data", issue_zone="data"
    )  # resolved by the run, but read as the data's zone all the same
    assert ForecastSource(tmp_path / "data", "up", "x").availability().timezone == "Europe/Warsaw"


def test_it_has_no_per_source_cache_and_a_content_digest_that_follows_the_upstream(tmp_path):
    records = write_upstream(tmp_path)
    source = ForecastSource(tmp_path, "up", "x")
    assert source.get_cache_config() is None
    first = source.content_digest()
    assert first == source.content_digest()
    results = tmp_path / "up.jsonl"
    before = results.stat()
    longer = [dict(record) for record in records]
    longer[0]["prediction"] = 1.0 / 3.0  # a longer number: the file's size changes
    write_upstream(tmp_path, records=longer)
    assert results.stat().st_size != before.st_size and source.content_digest() != first
    # Another value of the same length, and the first file's modification time put back (a coarse clock, or a copy that kept
    # it): the bytes still tell. The first bytes again give the first digest.
    same_size = [dict(record) for record in records]
    same_size[0]["prediction"] = records[0]["prediction"] + 1.0
    write_upstream(tmp_path, records=same_size)
    os.utime(results, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert results.stat().st_size == before.st_size and results.stat().st_mtime_ns == before.st_mtime_ns
    assert source.content_digest() != first
    write_upstream(tmp_path, records=records)
    assert source.content_digest() == first
    assert DataSource.content_digest(source) is None  # the hook's default


def test_the_pipeline_file_cache_refetches_after_the_upstream_changes_and_reuses_otherwise(tmp_path, monkeypatch):
    write_upstream(tmp_path / "up")
    calls = []
    fetch = ForecastSource.fetch

    def counted(self, start, end):
        calls.append((start, end))
        return fetch(self, start, end)

    monkeypatch.setattr(ForecastSource, "fetch", counted)
    pipeline = DataPipeline(timezone="Europe/Warsaw").add_source(ForecastSource(tmp_path / "up", "up", "x"))
    cache = tmp_path / "cache.parquet"
    first = pipeline.run("2024-03-29", "2024-04-01", cache=str(cache))
    again = pipeline.run("2024-03-29", "2024-04-01", cache=str(cache))
    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, again)
    records = [json.loads(line) for line in (tmp_path / "up" / "up.jsonl").read_text().splitlines()]
    for record in records:
        record["prediction"] += 1.0
    write_upstream(tmp_path / "up", records=records)
    changed = pipeline.run("2024-03-29", "2024-04-01", cache=str(cache))
    assert len(calls) == 2
    assert np.allclose(changed["x"].dropna().to_numpy(), first["x"].dropna().to_numpy() + 1.0)


def test_other_sources_fingerprints_have_no_content_key(tmp_path):
    from epftoolbox2.data.sources import CalendarSource
    from epftoolbox2.pipelines.data_pipeline import _component_fingerprint

    assert "content" not in _component_fingerprint(CalendarSource("PL", timezone="Europe/Warsaw"))
    write_upstream(tmp_path)
    assert "content" in _component_fingerprint(ForecastSource(tmp_path, "up", "x"))


def test_it_is_registered_and_round_trips_through_yaml_with_a_path_relative_to_the_file(tmp_path, monkeypatch):
    assert COMPONENT_REGISTRY["sources"]["ForecastSource"] == "epftoolbox2.data.sources.forecast"
    write_upstream(tmp_path / "runs" / "solar")
    monkeypatch.chdir(tmp_path)
    pipeline = DataPipeline(timezone="Europe/Warsaw").add_source(
        ForecastSource("runs/solar", "up", "solar_fc", horizon=2, available="10:00")
    )
    path = tmp_path / "configs" / "data.yaml"
    path.parent.mkdir()
    pipeline.save(path)
    text = path.read_text()
    assert "path: ../runs/solar" in text and "available: '10:00'" in text
    loaded = DataPipeline.load(path).sources[0]
    assert isinstance(loaded, ForecastSource)
    assert Path(loaded.path).resolve() == (tmp_path / "runs" / "solar").resolve()
    assert (loaded.model, loaded.column, loaded.horizon, loaded.available) == ("up", "solar_fc", 2, "10:00")


def test_the_tui_offers_it_with_its_parameters():
    import inspect

    from epftoolbox2.tui import _SOURCES, PARAM_SCHEMAS

    assert "ForecastSource" in _SOURCES
    names = [field["name"] for field in PARAM_SCHEMAS["ForecastSource"]]
    assert names == list(inspect.signature(ForecastSource).parameters)


def test_an_ensemble_upstream_reads_its_issue_time_from_its_members(tmp_path):
    write_upstream(tmp_path, name="a", issue_time="09:30")
    write_upstream(tmp_path, name="b", issue_time="09:30")
    shutil.copy(tmp_path / "a.jsonl", tmp_path / "avg.jsonl")
    ensemble = {
        "kind": "ensemble",
        "model_class": "AverageEnsemble",
        "freq": "1h",
        "horizon": 2,
        "data_timezone": "Europe/Warsaw",
        "member_results": {"a": {"manifest": "x", "results": "y"}, "b": {"manifest": "x", "results": "y"}},
    }
    (tmp_path / "avg.manifest.json").write_text(json.dumps(ensemble))
    source = ForecastSource(tmp_path, "avg", "x")
    assert source.availability().time == "09:30"
    assert not everything(source).empty
    write_upstream(tmp_path, name="b", issue_time="10:30")
    with pytest.raises(UpstreamError, match="different times"):
        source.availability()
    (tmp_path / "b.manifest.json").unlink()
    with pytest.raises(UpstreamError, match="no manifest"):
        source.availability()


def test_a_name_with_spaces_reads_the_store_s_file_name(tmp_path):
    write_upstream(tmp_path, name="solar_ridge")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not everything(ForecastSource(tmp_path, "Solar Ridge", "x")).empty


@pytest.mark.parametrize(
    ("manifest", "reason"),
    [
        ({"availability": None}, "ran at availability='none'"),
        (
            {
                "availability": {"level": "raise", "times": "instants", "listing": []},
                "information_set": {"issue_time": "09:30", "issue_timezone": "Europe/Brussels", "unknown": "warn"},
            },
            "unknown='warn'",
        ),
        (
            {
                "availability": {
                    "level": "warn",
                    "times": "instants",
                    "listing": [],
                    "stored_under_other_rules": {"listing": []},
                }
            },
            "listed under other rules",
        ),
        ({"data_unverified": True}, "accepted unverified"),
        ({"training_cut": {"rows": "dates", "stored": "unverified"}}, "accepted unverified"),
        ({"test_start": "2024-03-30", "test_end": "2024-04-02"}, "outside the test period 2024-03-30 to 2024-04-02"),
    ],
    ids=["none", "unknown", "other rules", "data unverified", "rows unverified", "outside its window"],
)
def test_an_upstream_whose_listing_doesn_t_vouch_for_every_forecast_is_late(tmp_path, manifest, reason):
    """Records issued 2024-03-27 to 2024-04-01 at horizon 2: a manifest whose listing doesn't cover them all is no clean upstream."""
    write_upstream(tmp_path, manifest={"test_start": "2024-03-27", "test_end": "2024-04-02"})
    clean = ForecastSource(tmp_path, "up", "x", horizon=2).chained_input()
    assert not clean.late_upstream and clean.unchecked == ()
    write_upstream(tmp_path, manifest={"test_start": "2024-03-27", "test_end": "2024-04-02", **manifest})
    found = ForecastSource(tmp_path, "up", "x", horizon=2).chained_input()
    assert found.late_upstream and any(reason in one for one in found.unchecked), found.unchecked


def test_a_run_with_windows_is_judged_by_them_not_by_its_test_period(tmp_path):
    write_upstream(
        tmp_path, manifest={"test_start": "2024-03-30", "test_end": "2024-04-02", "windows": {"test": ["x"]}}
    )
    assert not ForecastSource(tmp_path, "up", "x", horizon=2).chained_input().late_upstream


def test_the_path_expands_home_and_the_upstream_is_recorded_by_its_absolute_path(tmp_path, monkeypatch):
    write_upstream(tmp_path / "home" / "runs")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    source = ForecastSource("~/runs", "up", "x")
    assert source.path == "~/runs"  # as given, which YAML writes back
    assert not everything(source).empty
    assert source.chained_input().path == str((tmp_path / "home" / "runs").resolve())
    monkeypatch.chdir(tmp_path / "home")
    assert ForecastSource("runs", "up", "x").chained_input().path == str((tmp_path / "home" / "runs").resolve())
