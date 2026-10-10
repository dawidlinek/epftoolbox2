"""A source can tell the cache that its rows from some instant on are not settled yet (`df.attrs["settled_until"]`).

The mocked source below has a row for every hour, but those at or after its publication frontier are NaN
placeholders. Without the declaration the cache would record them as covered (the last real value is long past, so
the cache's own "still being published" rule does not hold them back) and never ask for them again.
"""

import json

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data import cache_manager as cache_manager_module
from epftoolbox2.data.cache_manager import SETTLED_UNTIL_ATTR, CacheManager, settled_until_of
from epftoolbox2.data.sources.base import DataSource
from epftoolbox2.pipelines import DataPipeline

GRID = pd.date_range("2023-12-01", "2024-03-01", freq="1h", tz="UTC", inclusive="left")


def utc(value) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


class PublishingSource(DataSource):
    """One row per hour, a ramp up to `published_until` and NaN from there on, which it declares as unsettled.
    It also returns a day of rows on both sides of the request, like ENTSO-E and Open-Meteo do. The frontier is
    private so that it is not part of the single-file cache's fingerprint: only the declaration decides there."""

    def __init__(self, published_until, name="publishing", declare=True):
        self._published_until = utc(published_until)
        self.name = name
        self.declare = declare
        self.calls = []

    def fetch(self, start, end):
        self.calls.append((start, end))
        index = GRID[(GRID >= start - pd.Timedelta("1D")) & (GRID < end + pd.Timedelta("1D"))]
        ramp = np.asarray((index - GRID[0]) / pd.Timedelta("1h"), dtype=float)
        df = pd.DataFrame({self.name: np.where(index < self._published_until, ramp, np.nan)}, index=index)
        if self.declare and self._published_until < end:
            df.attrs[SETTLED_UNTIL_ATTR] = self._published_until
        return df

    def _validate_config(self):
        return True

    def get_cache_config(self):
        return {"source_type": self.name, "schema_version": 1}


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def set_now(monkeypatch):
    def set_now(value):
        monkeypatch.setattr(cache_manager_module, "_now", lambda: utc(value))

    set_now("2100-01-01")  # far ahead: the cache's own unsettled window never matters unless a test moves it
    return set_now


def recorded_ranges(root) -> list:
    """The `(start, end)` ranges all per-source caches under `root` hold, sorted."""
    ranges = []
    for path in (root / ".cache" / "sources").glob("*/metadata.json"):
        for entry in json.loads(path.read_text())["cached_ranges"]:
            ranges.append((pd.Timestamp(entry["start"]), pd.Timestamp(entry["end"])))
    return sorted(ranges)


def run(source, start="2024-01-01", end="2024-01-15", cache=True):
    return DataPipeline().add_source(source).run(start, end, cache=cache)


class TestSettledUntilHelper:
    def test_reads_a_utc_timestamp(self):
        df = pd.DataFrame(index=GRID[:2])
        df.attrs[SETTLED_UNTIL_ATTR] = utc("2023-12-01 00:30")
        assert settled_until_of(df) == utc("2023-12-01 00:30")

    def test_an_iso_string_is_read_too(self):
        df = pd.DataFrame(index=GRID[:2])
        df.attrs[SETTLED_UNTIL_ATTR] = "2023-12-01T00:30:00+00:00"
        assert settled_until_of(df) == utc("2023-12-01 00:30")
        df.attrs[SETTLED_UNTIL_ATTR] = "2023-12-01T01:30:00+01:00"
        assert settled_until_of(df) == utc("2023-12-01 00:30")

    def test_naive_timestamp_is_utc_and_other_zones_are_converted(self):
        df = pd.DataFrame(index=GRID[:2])
        df.attrs[SETTLED_UNTIL_ATTR] = pd.Timestamp("2023-12-01 01:00")
        assert settled_until_of(df) == utc("2023-12-01 01:00")
        df.attrs[SETTLED_UNTIL_ATTR] = pd.Timestamp("2023-12-01 01:00", tz="Europe/Warsaw")
        assert settled_until_of(df) == utc("2023-12-01 00:00")

    def test_absent_or_missing_value_means_everything_is_settled(self):
        df = pd.DataFrame(index=GRID[:2])
        assert settled_until_of(df) is None
        df.attrs[SETTLED_UNTIL_ATTR] = pd.NaT
        assert settled_until_of(df) is None

    def test_a_value_that_is_no_timestamp_raises(self):
        df = pd.DataFrame(index=GRID[:2])
        df.attrs[SETTLED_UNTIL_ATTR] = "not a time"
        with pytest.raises(ValueError, match=SETTLED_UNTIL_ATTR):
            settled_until_of(df)


class TestCoveredRanges:
    def hourly(self, **attrs):
        df = pd.DataFrame({"x": np.arange(48.0)}, index=GRID[:48])
        df.attrs.update(attrs)
        return df

    def test_caps_the_range_at_settled_until(self, tmp_path, set_now):
        manager = CacheManager(cache_dir=str(tmp_path))
        df = self.hourly()
        assert manager.covered_ranges(df, GRID[0], GRID[48]) == [(GRID[0], GRID[48])]
        assert manager.covered_ranges(df, GRID[0], GRID[48], settled_until=GRID[30]) == [(GRID[0], GRID[30])]

    def test_reads_the_declaration_from_the_frame(self, tmp_path, set_now):
        manager = CacheManager(cache_dir=str(tmp_path))
        df = self.hourly(**{SETTLED_UNTIL_ATTR: GRID[30]})
        assert manager.covered_ranges(df, GRID[0], GRID[48]) == [(GRID[0], GRID[30])]

    def test_an_explicit_value_wins_over_the_frame(self, tmp_path, set_now):
        manager = CacheManager(cache_dir=str(tmp_path))
        df = self.hourly(**{SETTLED_UNTIL_ATTR: GRID[30]})
        assert manager.covered_ranges(df, GRID[0], GRID[48], settled_until=GRID[10]) == [(GRID[0], GRID[10])]

    def test_the_unsettled_window_still_applies_when_it_is_earlier(self, tmp_path, set_now):
        set_now("2023-12-04 12:00")  # the unsettled window starts 2023-12-02 00:00 UTC, which is GRID[24]
        manager = CacheManager(cache_dir=str(tmp_path))
        assert manager.covered_ranges(self.hourly(), GRID[0], GRID[48], settled_until=GRID[40]) == [(GRID[0], GRID[24])]

    def test_a_settled_until_before_the_range_records_nothing(self, tmp_path, set_now):
        manager = CacheManager(cache_dir=str(tmp_path))
        assert manager.covered_ranges(self.hourly(), GRID[10], GRID[48], settled_until=GRID[5]) == []

    def test_write_cache_stores_only_the_settled_rows(self, tmp_path, set_now):
        manager = CacheManager(cache_dir=str(tmp_path))
        config = {"source_type": "x"}
        key = manager.get_cache_key(config)
        df = self.hourly(**{SETTLED_UNTIL_ATTR: GRID[30]})
        assert manager.write_cache(key, df, GRID[0], GRID[48], config) == GRID[30]
        assert manager.find_missing_ranges(key, GRID[0], GRID[48]) == [(GRID[30], GRID[48])]
        assert len(manager.read_cached_data(key, GRID[0], GRID[48])) == 30


class TestSourceCache:
    def test_only_the_unsettled_part_is_fetched_again(self, in_tmp, set_now):
        first = run(PublishingSource("2024-01-10"))
        assert first["publishing"].loc["2024-01-09 23:00"] == (utc("2024-01-09 23:00") - GRID[0]) / pd.Timedelta("1h")
        assert first["publishing"].loc["2024-01-10":].isna().all()
        assert recorded_ranges(in_tmp) == [(utc("2024-01-01"), utc("2024-01-10"))]

        later = PublishingSource("2024-01-20")  # the rest was published since
        second = run(later)
        assert later.calls == [(utc("2024-01-10"), utc("2024-01-16"))]
        assert second["publishing"].notna().all()
        assert recorded_ranges(in_tmp) == [
            (utc("2024-01-01"), utc("2024-01-10")),
            (utc("2024-01-10"), utc("2024-01-16")),
        ]

        again = PublishingSource("2024-01-20")
        pd.testing.assert_frame_equal(run(again), second, check_freq=False)
        assert again.calls == []  # a full hit

    def test_a_split_run_equals_a_fresh_fetch(self, in_tmp, set_now):
        run(PublishingSource("2024-01-05"), end="2024-01-05")
        run(PublishingSource("2024-01-12"), end="2024-01-10")
        for published, end in [
            ("2024-01-12", "2024-01-15"),
            ("2024-01-14", "2024-01-15"),
            ("2024-02-01", "2024-01-20"),
        ]:
            cached = run(PublishingSource(published), end=end)
            fresh = run(PublishingSource(published), end=end, cache=False)
            pd.testing.assert_frame_equal(cached, fresh, check_freq=False)

    def test_a_declared_frontier_beyond_the_request_changes_nothing(self, in_tmp, set_now):
        source = PublishingSource("2024-02-01")
        run(source)
        assert source.calls == [(utc("2024-01-01"), utc("2024-01-16"))]
        assert recorded_ranges(in_tmp) == [(utc("2024-01-01"), utc("2024-01-16"))]

    def test_nothing_is_recorded_when_everything_is_unsettled(self, in_tmp, set_now):
        source = PublishingSource("2023-12-15")
        run(source)
        run(source)
        assert len(source.calls) == 2  # both runs fetched the whole range
        assert recorded_ranges(in_tmp) == []

    def test_without_the_declaration_the_placeholders_are_cached_for_good(self, in_tmp, set_now):
        """What the declaration prevents: the NaN rows count as covered, as no value is recent enough to hold them."""
        run(PublishingSource("2024-01-10", declare=False))
        assert recorded_ranges(in_tmp) == [(utc("2024-01-01"), utc("2024-01-16"))]

    @pytest.mark.parametrize(
        "now,end,expected_end",
        [
            ("2024-01-12 12:00", "2024-01-15", "2024-01-10"),  # the unsettled window starts 01-10, before the frontier
            ("2024-01-20 12:00", "2024-01-24", "2024-01-14"),  # the window starts 01-18, after the frontier
        ],
    )
    def test_the_earlier_of_the_frontier_and_the_unsettled_window_wins(self, in_tmp, set_now, now, end, expected_end):
        set_now(now)
        run(PublishingSource("2024-01-14"), end=end)
        assert recorded_ranges(in_tmp) == [(utc("2024-01-01"), utc(expected_end))]

    def test_the_declaration_does_not_reach_the_pipeline_output(self, in_tmp, set_now):
        for cache in (False, True, True):
            assert SETTLED_UNTIL_ATTR not in run(PublishingSource("2024-01-10"), cache=cache).attrs

    def test_one_source_settling_late_does_not_hold_back_another(self, in_tmp, set_now):
        pipeline = DataPipeline().add_source(PublishingSource("2024-01-10", name="a"))
        pipeline.add_source(PublishingSource("2024-02-01", name="b"))
        pipeline.run("2024-01-01", "2024-01-15", cache=True)
        assert recorded_ranges(in_tmp) == [
            (utc("2024-01-01"), utc("2024-01-10")),
            (utc("2024-01-01"), utc("2024-01-16")),
        ]


class TestFileCache:
    def meta(self, root):
        (path,) = (root / "cache").glob("*.meta.json")
        return json.loads(path.read_text())

    def test_an_entry_that_was_not_settled_up_to_the_end_is_fetched_again(self, in_tmp, set_now):
        first = PublishingSource("2024-01-10")
        run(first, cache="cache/")
        assert self.meta(in_tmp)["settled_until"] == utc("2024-01-10").isoformat()

        second = PublishingSource("2024-01-20")
        result = run(second, cache="cache/")
        assert len(second.calls) == 1
        assert result["publishing"].notna().all()
        assert self.meta(in_tmp)["settled_until"] is None

        third = PublishingSource("2024-01-20")
        pd.testing.assert_frame_equal(run(third, cache="cache/"), result, check_freq=False)
        assert third.calls == []

    def test_an_entry_settled_up_to_the_end_is_reused(self, in_tmp, set_now):
        # The range is [2024-01-01, 2024-01-16): a frontier at its end means every row in it is settled.
        run(PublishingSource("2024-01-16"), cache="cache/")
        assert self.meta(in_tmp)["settled_until"] is None
        again = PublishingSource("2024-01-16")
        run(again, cache="cache/")
        assert again.calls == []

    def test_the_earliest_frontier_of_several_sources_is_stored(self, in_tmp, set_now):
        pipeline = DataPipeline().add_source(PublishingSource("2024-01-12", name="a"))
        pipeline.add_source(PublishingSource("2024-01-08", name="b"))
        pipeline.run("2024-01-01", "2024-01-15", cache="cache/")
        assert self.meta(in_tmp)["settled_until"] == utc("2024-01-08").isoformat()

    def test_a_sidecar_written_before_the_field_existed_is_still_reused(self, in_tmp, set_now):
        run(PublishingSource("2024-01-20"), cache="cache/")
        (path,) = (in_tmp / "cache").glob("*.meta.json")
        meta = json.loads(path.read_text())
        del meta["settled_until"]
        path.write_text(json.dumps(meta))
        again = PublishingSource("2024-01-20")
        run(again, cache="cache/")
        assert again.calls == []

    def test_a_malformed_value_means_fetch_again(self, in_tmp, set_now):
        run(PublishingSource("2024-01-20"), cache="cache/")
        (path,) = (in_tmp / "cache").glob("*.meta.json")
        meta = json.loads(path.read_text())
        meta["settled_until"] = "garbage"
        path.write_text(json.dumps(meta))
        again = PublishingSource("2024-01-20")
        run(again, cache="cache/")
        assert len(again.calls) == 1
