import json
import multiprocessing
import threading
import time

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data import cache_manager as cache_manager_module
from epftoolbox2.data.cache_manager import CACHE_FORMAT_VERSION, CacheManager


def hourly(start, end, **cols):
    """Hourly UTC frame on [start, end) with the given column name -> offset added to a 0..n-1 ramp."""
    idx = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    return pd.DataFrame({name: np.arange(len(idx), dtype=float) + off for name, off in cols.items()}, index=idx)


def ts(s):
    return pd.Timestamp(s, tz="UTC")


@pytest.fixture
def temp_cache_dir(tmp_path):
    return str(tmp_path / "cache")


@pytest.fixture
def cache_manager(temp_cache_dir):
    return CacheManager(cache_dir=temp_cache_dir)


@pytest.fixture
def sample_config():
    return {"source_type": "entsoe", "area_code": "10YPL-AREA-----S", "types": ["load", "price"]}


@pytest.fixture
def sample_dataframe():
    dates = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    return pd.DataFrame({"load": range(24), "price": range(100, 124)}, index=dates)


@pytest.fixture
def frozen_now(monkeypatch):
    """Pin the cache's notion of "now" (default: far in the future, so nothing counts as unpublished)."""

    def set_now(value="2100-01-01"):
        monkeypatch.setattr(cache_manager_module, "_now", lambda: ts(value))

    set_now()
    return set_now


class TestCacheKeyGeneration:
    def test_same_config_same_key(self, cache_manager):
        config1 = {"source_type": "entsoe", "area_code": "PL", "types": ["load"]}
        config2 = {"source_type": "entsoe", "area_code": "PL", "types": ["load"]}
        assert cache_manager.get_cache_key(config1) == cache_manager.get_cache_key(config2)

    def test_different_config_different_key(self, cache_manager):
        config1 = {"source_type": "entsoe", "area_code": "PL", "types": ["load"]}
        config2 = {"source_type": "entsoe", "area_code": "DE", "types": ["load"]}
        assert cache_manager.get_cache_key(config1) != cache_manager.get_cache_key(config2)

    def test_key_deterministic(self, cache_manager):
        config = {"source_type": "entsoe", "area_code": "PL", "types": ["load", "price"]}
        assert len({cache_manager.get_cache_key(config) for _ in range(3)}) == 1

    def test_key_includes_source_class(self, cache_manager, sample_config):
        """Two source classes with the same get_cache_config() must not share a cache."""
        a = cache_manager.get_cache_key(sample_config, source_class="pkg.EntsoeSource")
        b = cache_manager.get_cache_key(sample_config, source_class="pkg.MyPatchedEntsoeSource")
        assert a != b

    def test_key_includes_cache_format_version(self, cache_manager, sample_config, monkeypatch):
        """Bumping the package-level cache format version invalidates every existing cache."""
        before = cache_manager.get_cache_key(sample_config)
        monkeypatch.setattr(cache_manager_module, "CACHE_FORMAT_VERSION", CACHE_FORMAT_VERSION + 1)
        assert cache_manager.get_cache_key(sample_config) != before

    def test_key_changes_with_schema_version(self, cache_manager, sample_config):
        """Sources bump "schema_version" in get_cache_config() when their parsing changes."""
        assert cache_manager.get_cache_key(sample_config) != cache_manager.get_cache_key(
            {**sample_config, "schema_version": 2}
        )


class TestCacheWriteRead:
    def test_write_and_read_cache(self, cache_manager, sample_config, sample_dataframe, frozen_now):
        cache_key = cache_manager.get_cache_key(sample_config)
        start, end = ts("2024-01-01"), ts("2024-01-02")

        cache_manager.write_cache(cache_key, sample_dataframe, start, end, sample_config)
        result = cache_manager.read_cached_data(cache_key, start, end)

        pd.testing.assert_frame_equal(result, sample_dataframe, check_freq=False)

    def test_files_are_parquet(self, cache_manager, sample_config, sample_dataframe, frozen_now, temp_cache_dir):
        from pathlib import Path

        cache_key = cache_manager.get_cache_key(sample_config)
        cache_manager.write_cache(cache_key, sample_dataframe, ts("2024-01-01"), ts("2024-01-02"), sample_config)
        files = sorted(p.name for p in (Path(temp_cache_dir) / cache_key).iterdir())
        assert [f for f in files if f.startswith("data_")] == ["data_20240101T000000_20240102T000000.parquet"]
        assert not [f for f in files if f.endswith(".tmp")]

    def test_round_trip_is_exact(self, cache_manager, sample_config, frozen_now):
        """tz, dtypes and every float bit survive the cache (CSV lost the last digit of some floats)."""
        idx = pd.date_range("2024-03-30", periods=96, freq="15min", tz="UTC")
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "f": rng.normal(size=96) * 1000 / 7,
                "thirds": np.arange(96) / 3,
                "i": np.arange(96, dtype="int64"),
                "b": rng.integers(0, 2, 96).astype(bool),
                "s": ["New Year"] * 95 + [None],
                "nan": [np.nan] * 48 + list(rng.normal(size=48)),
            },
            index=idx,
        )
        df.loc[idx[3], "f"] = 15.783333333333333
        df.loc[idx[4], "f"] = 9.662236272500001
        key = cache_manager.get_cache_key(sample_config)
        cache_manager.write_cache(key, df, idx[0], idx[-1] + pd.Timedelta("15min"), sample_config)
        back = cache_manager.read_cached_data(key, idx[0], idx[-1] + pd.Timedelta("15min"))

        pd.testing.assert_frame_equal(back, df, check_freq=False, check_exact=True)
        assert str(back.index.tz) == "UTC"

    def test_read_nonexistent_cache(self, cache_manager, sample_config):
        cache_key = cache_manager.get_cache_key(sample_config)
        assert cache_manager.read_cached_data(cache_key, ts("2024-01-01"), ts("2024-01-02")) is None

    def test_write_empty_dataframe(self, cache_manager, sample_config):
        cache_key = cache_manager.get_cache_key(sample_config)
        start, end = ts("2024-01-01"), ts("2024-01-02")
        assert cache_manager.write_cache(cache_key, pd.DataFrame(), start, end, sample_config) is None
        assert cache_manager.read_cached_data(cache_key, start, end) is None
        assert cache_manager.find_missing_ranges(cache_key, start, end) == [(start, end)]

    def test_rows_outside_the_written_range_are_not_stored(self, cache_manager, sample_config, frozen_now):
        """A source may return rows outside [start, end) (ENTSO-E load from start - 1 day, price at end)."""
        key = cache_manager.get_cache_key(sample_config)
        df = hourly("2023-12-31", "2024-01-03", load=0.0)
        cache_manager.write_cache(key, df, ts("2024-01-01"), ts("2024-01-02"), sample_config)

        back = cache_manager.read_cached_data(key, ts("2023-12-01"), ts("2024-02-01"))
        assert back.index.min() == ts("2024-01-01")
        assert back.index.max() == ts("2024-01-01 23:00")


class TestCoverage:
    """Only what the source actually returned is recorded as cached (entsoe H4, pipelines H3, empirical F5d)."""

    def test_coverage_ends_at_last_returned_timestamp(self, cache_manager, sample_config, frozen_now):
        key = cache_manager.get_cache_key(sample_config)
        df = hourly("2024-01-01", "2024-01-10", price=0.0)  # source had data only up to 01-09 23:00

        covered_end = cache_manager.write_cache(key, df, ts("2024-01-01"), ts("2024-02-01"), sample_config)

        assert covered_end == ts("2024-01-10")
        assert cache_manager.find_missing_ranges(key, ts("2024-01-01"), ts("2024-02-01")) == [
            (ts("2024-01-10"), ts("2024-02-01"))
        ]

    def test_coverage_never_reaches_into_unpublished_period(self, cache_manager, sample_config, frozen_now):
        """Data close to "now" may still be published or completed (actual load, next-day prices, weather runs)."""
        frozen_now("2024-06-10 09:30")
        key = cache_manager.get_cache_key(sample_config)
        # the source already returns tomorrow's prices, but actual load only up to "now"
        df = pd.concat(
            [hourly("2024-06-01", "2024-06-12", price=0.0), hourly("2024-06-01", "2024-06-10 09:00", load=0.0)], axis=1
        )

        covered_end = cache_manager.write_cache(key, df, ts("2024-06-01"), ts("2024-06-12"), sample_config)

        assert covered_end == ts("2024-06-10 09:30").floor("D") - cache_manager.unsettled_period
        missing = cache_manager.find_missing_ranges(key, ts("2024-06-01"), ts("2024-06-12"))
        assert missing == [(covered_end, ts("2024-06-12"))]

    def test_nothing_recorded_when_everything_is_recent(self, cache_manager, sample_config, frozen_now):
        frozen_now("2024-06-10 09:30")
        key = cache_manager.get_cache_key(sample_config)
        df = hourly("2024-06-10", "2024-06-12", price=0.0)
        assert cache_manager.write_cache(key, df, ts("2024-06-10"), ts("2024-06-12"), sample_config) is None
        assert cache_manager.read_cached_data(key, ts("2024-06-10"), ts("2024-06-12")) is None

    def test_retrieved_at_is_recorded(self, cache_manager, sample_config, sample_dataframe, frozen_now):
        frozen_now("2030-05-05 12:00")
        key = cache_manager.get_cache_key(sample_config)
        cache_manager.write_cache(key, sample_dataframe, ts("2024-01-01"), ts("2024-01-02"), sample_config)
        meta = json.loads((cache_manager.cache_dir / key / "metadata.json").read_text())
        assert meta["cache_format"] == CACHE_FORMAT_VERSION
        assert meta["cached_ranges"][0]["retrieved_at"].startswith("2030-05-05T12:00")

    def test_provisional_open_meteo_rows_are_never_recorded_as_covered(self, cache_manager, sample_config, frozen_now):
        """integration check f: an Open-Meteo `{col}_d+i` row is provisional exactly when the issue time of
        the day the forecast was made (at most `horizon` days before the row) hasn't passed yet — and that
        issue date is never later than the row's own date, so a provisional row is always dated today or
        later. That is always inside the 2-day unsettled window below, so it can never be recorded as
        covered here, regardless of the horizon or how the source itself decides "provisional"."""
        frozen_now("2024-06-10 08:00")  # before a source's issue time later today
        key = cache_manager.get_cache_key(sample_config)
        # A day-ahead weather forecast column, as an Open-Meteo fetch would return it: values through today,
        # today's own row possibly provisional (built from whatever run has been published so far).
        df = hourly("2024-06-01", "2024-06-11", temperature_2m_d_plus_1=0.0)

        covered_end = cache_manager.write_cache(key, df, ts("2024-06-01"), ts("2024-06-11"), sample_config)

        assert covered_end == ts("2024-06-08")  # today's floor (06-10) minus the 2-day unsettled period
        assert cache_manager.read_cached_data(key, ts("2024-06-10"), ts("2024-06-11")) is None
        assert cache_manager.find_missing_ranges(key, ts("2024-06-01"), ts("2024-06-11")) == [
            (ts("2024-06-08"), ts("2024-06-11"))
        ]

        # After the issue time passes, a later fetch for the same range still asks for (and can get a
        # different, settled value for) today's row instead of being served a stale provisional one.
        frozen_now("2024-06-10 11:00")
        assert cache_manager.find_missing_ranges(key, ts("2024-06-01"), ts("2024-06-11")) == [
            (ts("2024-06-08"), ts("2024-06-11"))
        ]


class TestCoverageGaps:
    """pipelines-integration H4-follow-up: a gap in the middle of a response used to be recorded as covered, so
    the missing hour was never fetched again — silently lost for good."""

    def test_internal_gap_is_not_recorded_as_covered(self, cache_manager, sample_config, frozen_now):
        key = cache_manager.get_cache_key(sample_config)
        # hourly 00:00, 01:00, 02:00, then a hole at 03:00, then 04:00
        idx = pd.DatetimeIndex(
            [ts("2024-01-01 00:00"), ts("2024-01-01 01:00"), ts("2024-01-01 02:00"), ts("2024-01-01 04:00")]
        )
        df = pd.DataFrame({"price": [1.0, 2.0, 3.0, 5.0]}, index=idx)

        cache_manager.write_cache(key, df, ts("2024-01-01 00:00"), ts("2024-01-01 05:00"), sample_config)

        missing = cache_manager.find_missing_ranges(key, ts("2024-01-01 00:00"), ts("2024-01-01 05:00"))
        assert missing == [(ts("2024-01-01 03:00"), ts("2024-01-01 04:00"))]

    def test_gap_is_fetched_and_cached_on_the_next_run(self, cache_manager, sample_config, frozen_now):
        """The hole found above is refetched next time, and once filled counts as fully covered."""
        key = cache_manager.get_cache_key(sample_config)
        idx = pd.DatetimeIndex(
            [ts("2024-01-01 00:00"), ts("2024-01-01 01:00"), ts("2024-01-01 02:00"), ts("2024-01-01 04:00")]
        )
        df = pd.DataFrame({"price": [1.0, 2.0, 3.0, 5.0]}, index=idx)
        cache_manager.write_cache(key, df, ts("2024-01-01 00:00"), ts("2024-01-01 05:00"), sample_config)

        gap_start, gap_end = ts("2024-01-01 03:00"), ts("2024-01-01 04:00")
        gap_df = pd.DataFrame({"price": [4.0]}, index=pd.DatetimeIndex([gap_start]))
        cache_manager.write_cache(key, gap_df, gap_start, gap_end, sample_config)

        assert cache_manager.find_missing_ranges(key, ts("2024-01-01 00:00"), ts("2024-01-01 05:00")) == []
        result = cache_manager.read_cached_data(key, ts("2024-01-01 00:00"), ts("2024-01-01 05:00"))
        assert list(result["price"]) == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_two_internal_gaps_split_into_three_runs(self, cache_manager, sample_config, frozen_now):
        key = cache_manager.get_cache_key(sample_config)
        present = ["00:00", "01:00", "03:00", "04:00", "05:00", "07:00"]
        idx = pd.DatetimeIndex([ts(f"2024-01-01 {h}") for h in present])
        df = pd.DataFrame({"price": range(len(idx))}, index=idx)

        cache_manager.write_cache(key, df, ts("2024-01-01 00:00"), ts("2024-01-01 08:00"), sample_config)

        missing = cache_manager.find_missing_ranges(key, ts("2024-01-01 00:00"), ts("2024-01-01 08:00"))
        assert missing == [
            (ts("2024-01-01 02:00"), ts("2024-01-01 03:00")),
            (ts("2024-01-01 06:00"), ts("2024-01-01 07:00")),
        ]

    def test_single_timestamp_response_is_cached(self, cache_manager, sample_config, frozen_now):
        """A source that returns exactly one row for the requested range used to never be cached at all, because
        `covered` computed from a zero-length step equalled `start` and failed the `covered > start` check."""
        key = cache_manager.get_cache_key(sample_config)
        df = pd.DataFrame({"price": [42.0]}, index=pd.DatetimeIndex([ts("2024-01-01")]))

        covered_end = cache_manager.write_cache(key, df, ts("2024-01-01"), ts("2024-01-08"), sample_config)

        assert covered_end == ts("2024-01-08")
        assert cache_manager.find_missing_ranges(key, ts("2024-01-01"), ts("2024-01-08")) == []
        result = cache_manager.read_cached_data(key, ts("2024-01-01"), ts("2024-01-08"))
        assert list(result["price"]) == [42.0]

    def test_single_timestamp_still_respects_the_unsettled_boundary(self, cache_manager, sample_config, frozen_now):
        frozen_now("2024-06-10 09:30")
        key = cache_manager.get_cache_key(sample_config)
        df = pd.DataFrame({"price": [1.0]}, index=pd.DatetimeIndex([ts("2024-06-01")]))

        covered_end = cache_manager.write_cache(key, df, ts("2024-06-01"), ts("2024-06-12"), sample_config)

        assert covered_end == ts("2024-06-10 09:30").floor("D") - cache_manager.unsettled_period


class TestResolutionChangeGaps:
    """entsoe-cache/pipelines review, finding 1: `covered_ranges` used to pick one "most common step" for the
    whole response. A fetch that changes resolution partway through (e.g. the ENTSO-E 15-minute go-live) made
    every diff on one side look like a gap relative to the other side's step, producing thousands of slivers and
    covering only a small fraction of the span -- refetched in full on every run."""

    def test_hourly_then_15min_does_not_cause_endless_refetch(self, cache_manager, sample_config, frozen_now):
        key = cache_manager.get_cache_key(sample_config)
        hourly_idx = pd.date_range("2024-01-01", periods=181 * 24, freq="h", tz="UTC")
        switch = hourly_idx[-1] + pd.Timedelta(hours=1)
        fine_idx = pd.date_range(switch, periods=46 * 96, freq="15min", tz="UTC")
        idx = hourly_idx.append(fine_idx)
        df = pd.DataFrame({"price": np.arange(len(idx), dtype=float)}, index=idx)
        start, end = idx[0], fine_idx[-1] + pd.Timedelta(minutes=15)

        cache_manager.write_cache(key, df, start, end, sample_config)
        missing = cache_manager.find_missing_ranges(key, start, end)

        total_span = end - start
        missing_span = sum((e - s for s, e in missing), pd.Timedelta(0))
        # A small boundary sliver refetched is acceptable and documented; the old bug refetched most of the span.
        assert len(missing) < 10, missing
        assert missing_span < pd.Timedelta(hours=6), missing_span
        assert missing_span < total_span * 0.01

    def test_internal_hour_gap_still_detected_across_a_long_hourly_run(self, cache_manager, sample_config, frozen_now):
        """(b): an internal missing hour in hourly data is still a gap, even once local (not global) steps decide."""
        key = cache_manager.get_cache_key(sample_config)
        idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC").delete(150)
        df = pd.DataFrame({"price": np.arange(len(idx), dtype=float)}, index=idx)
        start, end = ts("2024-01-01"), idx[-1] + pd.Timedelta(hours=1)

        cache_manager.write_cache(key, df, start, end, sample_config)
        missing = cache_manager.find_missing_ranges(key, start, end)

        gap = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")[150]
        assert missing == [(gap, gap + pd.Timedelta(hours=1))]

    def test_internal_quarter_hour_gap_still_detected_in_15min_data(self, cache_manager, sample_config, frozen_now):
        """(b): a missing quarter-hour in 15-minute data is still a gap."""
        key = cache_manager.get_cache_key(sample_config)
        full = pd.date_range("2024-01-01", periods=200, freq="15min", tz="UTC")
        idx = full.delete(150)
        df = pd.DataFrame({"price": np.arange(len(idx), dtype=float)}, index=idx)
        start, end = full[0], full[-1] + pd.Timedelta(minutes=15)

        cache_manager.write_cache(key, df, start, end, sample_config)
        missing = cache_manager.find_missing_ranges(key, start, end)

        gap = full[150]
        assert missing == [(gap, gap + pd.Timedelta(minutes=15))]

    def test_internal_week_gap_still_detected_in_weekly_data(self, cache_manager, sample_config, frozen_now):
        """(c): a missing week in weekly data is still a gap."""
        key = cache_manager.get_cache_key(sample_config)
        full = pd.date_range("2024-01-01", periods=20, freq="7D", tz="UTC")
        idx = full.delete(10)
        df = pd.DataFrame({"price": np.arange(len(idx), dtype=float)}, index=idx)
        start, end = full[0], full[-1] + pd.Timedelta(days=7)

        cache_manager.write_cache(key, df, start, end, sample_config)
        missing = cache_manager.find_missing_ranges(key, start, end)

        gap = full[10]
        assert missing == [(gap, gap + pd.Timedelta(days=7))]


class TestFewPointsDontInferAStep:
    """entsoe-cache/pipelines review, finding 2: with only one diff to go on, `covered_ranges` trusted it as the
    real step and extended coverage across it and beyond the last point, hiding a genuine gap."""

    def test_two_timestamps_cover_neither_the_gap_between_nor_beyond_them(
        self, cache_manager, sample_config, frozen_now
    ):
        key = cache_manager.get_cache_key(sample_config)
        idx = pd.DatetimeIndex([ts("2024-01-01"), ts("2024-01-15")])
        df = pd.DataFrame({"price": [1.0, 2.0]}, index=idx)

        covered_end = cache_manager.write_cache(key, df, ts("2024-01-01"), ts("2024-01-22"), sample_config)

        assert covered_end is None
        assert cache_manager.find_missing_ranges(key, ts("2024-01-01"), ts("2024-01-22")) == [
            (ts("2024-01-01"), ts("2024-01-22"))
        ]


class TestSeams:
    """entsoe H3, empirical F5b: files overlapping at their edges must not produce half-empty rows."""

    def test_each_file_is_sliced_to_its_own_range(self, cache_manager, sample_config, frozen_now):
        key = cache_manager.get_cache_key(sample_config)
        # file A was written by an older version: it holds a half-empty row at its end (price only, no load)
        a = hourly("2024-01-01", "2024-01-03", load=0.0, price=100.0)
        a.loc[ts("2024-01-03"), "price"] = 999.0
        cache_manager.write_cache(key, a, ts("2024-01-01"), ts("2024-01-03"), sample_config)
        path = cache_manager.cache_dir / key
        name_a = json.loads((path / "metadata.json").read_text())["cached_ranges"][0]["filename"]
        a.to_parquet(path / name_a)  # put the boundary row back into the file, as old caches have it

        b = hourly("2024-01-03", "2024-01-05", load=48.0, price=148.0)
        cache_manager.write_cache(key, b, ts("2024-01-03"), ts("2024-01-05"), sample_config)

        got = cache_manager.read_cached_data(key, ts("2024-01-01"), ts("2024-01-05"))
        expected = hourly("2024-01-01", "2024-01-05", load=0.0, price=100.0)
        pd.testing.assert_frame_equal(got, expected, check_freq=False)

    def test_newer_write_wins_over_overlapping_range(self, cache_manager, sample_config, frozen_now):
        key = cache_manager.get_cache_key(sample_config)
        cache_manager.write_cache(
            key, hourly("2024-01-01", "2024-01-05", x=0.0), ts("2024-01-01"), ts("2024-01-05"), {}
        )
        cache_manager.write_cache(
            key, hourly("2024-01-02", "2024-01-03", x=1000.0), ts("2024-01-02"), ts("2024-01-03"), {}
        )

        got = cache_manager.read_cached_data(key, ts("2024-01-01"), ts("2024-01-05"))
        assert got.index.is_unique and got.index.is_monotonic_increasing
        assert len(got) == 96
        assert got.loc["2024-01-02", "x"].min() == 1000.0
        assert got.loc["2024-01-03", "x"].iloc[0] == 48.0
        assert cache_manager.find_missing_ranges(key, ts("2024-01-01"), ts("2024-01-05")) == []


class TestBrokenFiles:
    """entsoe M4: a deleted or unreadable cache file means "missing", never a silently dropped source."""

    def _write(self, cm, cfg):
        key = cm.get_cache_key(cfg)
        cm.write_cache(key, hourly("2024-01-01", "2024-01-03", price=0.0), ts("2024-01-01"), ts("2024-01-03"), cfg)
        (fname,) = [r["filename"] for r in cm.get_cache_info(key)["date_ranges"]]
        return key, cm.cache_dir / key / fname

    def test_deleted_file_is_missing(self, cache_manager, sample_config, frozen_now):
        key, path = self._write(cache_manager, sample_config)
        path.unlink()
        assert cache_manager.find_missing_ranges(key, ts("2024-01-01"), ts("2024-01-03")) == [
            (ts("2024-01-01"), ts("2024-01-03"))
        ]
        assert cache_manager.read_cached_data(key, ts("2024-01-01"), ts("2024-01-03")) is None

    def test_unreadable_file_is_missing(self, cache_manager, sample_config, frozen_now):
        key, path = self._write(cache_manager, sample_config)
        path.write_bytes(b"not a parquet file")
        assert cache_manager.read_cached_data(key, ts("2024-01-01"), ts("2024-01-03")) is None
        assert cache_manager.find_missing_ranges(key, ts("2024-01-01"), ts("2024-01-03")) == [
            (ts("2024-01-01"), ts("2024-01-03"))
        ]

    def test_corrupt_metadata_means_empty_cache(self, cache_manager, sample_config, frozen_now):
        key, _ = self._write(cache_manager, sample_config)
        (cache_manager.cache_dir / key / "metadata.json").write_text('{"cached_ranges": [')
        assert cache_manager.find_missing_ranges(key, ts("2024-01-01"), ts("2024-01-03")) == [
            (ts("2024-01-01"), ts("2024-01-03"))
        ]


class TestAtomicityAndLocking:
    """pipelines H1, entsoe H5: concurrent SLURM array tasks share one cache directory."""

    def test_failed_write_leaves_previous_state_intact(self, cache_manager, sample_config, frozen_now, monkeypatch):
        key = cache_manager.get_cache_key(sample_config)
        first = hourly("2024-01-01", "2024-01-02", price=0.0)
        cache_manager.write_cache(key, first, ts("2024-01-01"), ts("2024-01-02"), sample_config)
        meta_before = (cache_manager.cache_dir / key / "metadata.json").read_text()

        def crash(self, path, *args, **kwargs):
            with open(path, "wb") as f:
                f.write(b"PAR1 half written")
            raise KeyboardInterrupt("killed mid-write")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", crash)
        with pytest.raises(KeyboardInterrupt):
            cache_manager.write_cache(
                key, hourly("2024-01-01", "2024-01-02", price=5.0), ts("2024-01-01"), ts("2024-01-02"), sample_config
            )
        monkeypatch.undo()
        frozen_now()

        assert (cache_manager.cache_dir / key / "metadata.json").read_text() == meta_before
        assert not list((cache_manager.cache_dir / key).glob("*.tmp"))
        got = cache_manager.read_cached_data(key, ts("2024-01-01"), ts("2024-01-02"))
        pd.testing.assert_frame_equal(got, first, check_freq=False)

    def test_writer_waits_for_lock_holder(self, cache_manager, sample_config, frozen_now, temp_cache_dir):
        key = cache_manager.get_cache_key(sample_config)
        other = CacheManager(cache_dir=temp_cache_dir)
        done = threading.Event()

        def write():
            other.write_cache(
                key, hourly("2024-01-01", "2024-01-02", price=0.0), ts("2024-01-01"), ts("2024-01-02"), sample_config
            )
            done.set()

        with cache_manager.lock(key):
            t = threading.Thread(target=write)
            t.start()
            time.sleep(0.5)
            assert not done.is_set(), "write_cache must wait while another holder has the key locked"
        t.join(timeout=10)
        assert done.is_set()

    def test_lock_is_reentrant(self, cache_manager, sample_config, sample_dataframe, frozen_now):
        key = cache_manager.get_cache_key(sample_config)
        with cache_manager.lock(key):
            with cache_manager.lock(key):
                cache_manager.write_cache(key, sample_dataframe, ts("2024-01-01"), ts("2024-01-02"), sample_config)
                assert cache_manager.read_cached_data(key, ts("2024-01-01"), ts("2024-01-02")) is not None

    def test_concurrent_processes_never_read_partial_data(self, temp_cache_dir):
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        procs = [ctx.Process(target=_write_then_read, args=(temp_cache_dir, queue)) for _ in range(4)]
        for p in procs:
            p.start()
        results = [queue.get(timeout=120) for _ in procs]
        for p in procs:
            p.join(timeout=60)
        expected = len(pd.date_range("2020-01-01", "2022-01-01", freq="h", inclusive="left"))
        assert results == [[("ok", expected)] * 3] * 4


def _write_then_read(cache_dir, queue):
    """One SLURM array task on a cold cache: write the same key, read it back."""
    cm = CacheManager(cache_dir)
    cfg = {"source_type": "race"}
    key = cm.get_cache_key(cfg)
    df = hourly("2020-01-01", "2022-01-01", price=0.0, load=1.0)
    out = []
    for _ in range(3):
        try:
            cm.write_cache(key, df, ts("2020-01-01"), ts("2022-01-01"), cfg)
            got = cm.read_cached_data(key, ts("2020-01-01"), ts("2022-01-01"))
            out.append(("ok", 0 if got is None else len(got)))
        except Exception as e:  # noqa: BLE001 - report any failure to the parent
            out.append((type(e).__name__, 0))
    queue.put(out)


class TestGapDetection:
    def test_find_missing_no_cache(self, cache_manager, sample_config):
        cache_key = cache_manager.get_cache_key(sample_config)
        start, end = ts("2024-01-01"), ts("2024-01-31")
        assert cache_manager.find_missing_ranges(cache_key, start, end) == [(start, end)]

    def test_find_missing_full_hit(self, cache_manager, sample_config, sample_dataframe, frozen_now):
        cache_key = cache_manager.get_cache_key(sample_config)
        start, end = ts("2024-01-01"), ts("2024-01-02")
        cache_manager.write_cache(cache_key, sample_dataframe, start, end, sample_config)
        assert cache_manager.find_missing_ranges(cache_key, start, end) == []

    def test_find_missing_partial_hit_before(self, cache_manager, sample_config, frozen_now):
        cache_key = cache_manager.get_cache_key(sample_config)
        cache_start, cache_end = ts("2024-01-15"), ts("2024-02-01")
        cache_manager.write_cache(
            cache_key, hourly(cache_start, cache_end, load=0.0), cache_start, cache_end, sample_config
        )

        missing = cache_manager.find_missing_ranges(cache_key, ts("2024-01-01"), ts("2024-02-01"))
        assert missing == [(ts("2024-01-01"), cache_start)]

    def test_find_missing_partial_hit_after(self, cache_manager, sample_config, frozen_now):
        cache_key = cache_manager.get_cache_key(sample_config)
        cache_start, cache_end = ts("2024-01-01"), ts("2024-01-15")
        cache_manager.write_cache(
            cache_key, hourly(cache_start, cache_end, load=0.0), cache_start, cache_end, sample_config
        )

        missing = cache_manager.find_missing_ranges(cache_key, ts("2024-01-01"), ts("2024-02-01"))
        assert missing == [(cache_end, ts("2024-02-01"))]

    def test_find_missing_gap_in_middle(self, cache_manager, sample_config, frozen_now):
        cache_key = cache_manager.get_cache_key(sample_config)
        for s, e in [("2024-01-01", "2024-01-02"), ("2024-01-15", "2024-01-16")]:
            cache_manager.write_cache(cache_key, hourly(s, e, load=0.0), ts(s), ts(e), sample_config)

        missing = cache_manager.find_missing_ranges(cache_key, ts("2024-01-01"), ts("2024-01-20"))
        assert missing == [(ts("2024-01-02"), ts("2024-01-15")), (ts("2024-01-16"), ts("2024-01-20"))]


class TestCacheInfo:
    def test_cache_info_no_cache(self, cache_manager, sample_config):
        info = cache_manager.get_cache_info(cache_manager.get_cache_key(sample_config))
        assert info["exists"] is False
        assert info["num_chunks"] == 0

    def test_cache_info_with_data(self, cache_manager, sample_config, sample_dataframe, frozen_now):
        cache_key = cache_manager.get_cache_key(sample_config)
        cache_manager.write_cache(cache_key, sample_dataframe, ts("2024-01-01"), ts("2024-01-02"), sample_config)
        info = cache_manager.get_cache_info(cache_key)
        assert info["exists"] is True
        assert info["num_chunks"] == 1
        assert len(info["date_ranges"]) == 1
