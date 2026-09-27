import hashlib
from unittest.mock import Mock

import pandas as pd
import pytest

from epftoolbox2.data.sources.benchmark import DATASETS, EpftoolboxBenchmarkSource
from epftoolbox2.data.transformers import LagTransformer, TimezoneTransformer
from epftoolbox2.pipelines.data_pipeline import DataPipeline

HEADERS = {
    "DE": "Date,Price,Ampirion Load Forecast,PV+Wind Forecast",
    "NP": "Date,Price,Grid load forecast,Wind power forecast",
    "PJM": "Date,Zonal COMED price,System load forecast,Zonal COMED load foecast",
    # the order of the exogenous columns doesn't matter: they are matched by keyword
    "BE": "Date,Prices,Generation forecast,System load forecast",
    "FR": "Date,Prices,System load forecast,Generation forecast",
}


def benchmark_csv(dataset="DE", first="2016-03-26", days=3):
    """Tiny file in the epftoolbox format: naive local time, 24 rows on every day (DST days included).

    Known answer: price = day of month * 100 + hour, exogenous 1 = price + 0.1 (load-like), exogenous 2 = + 0.2.
    """
    lines = [HEADERS[dataset]]
    for day in pd.date_range(first, periods=days, freq="D"):
        for hour in range(24):
            ts = day + pd.Timedelta(hours=hour)
            price = day.day * 100 + hour
            load, other = (price + 0.1, price + 0.2)
            if dataset in ("BE",):  # generation column first in this header
                load, other = other, load
            lines.append(f"{ts:%Y-%m-%d %H:%M:%S},{price},{load},{other}")
    return ("\n".join(lines) + "\n").encode()


def mock_zenodo(source, content, dataset="DE", md5=None):
    md5 = md5 or hashlib.md5(content).hexdigest()
    metadata = Mock(status_code=200)
    metadata.json.return_value = {
        "files": [
            {
                "key": f"{dataset}.csv",
                "checksum": f"md5:{md5}",
                "links": {"self": f"https://zenodo.org/api/records/4624805/files/{dataset}.csv/content"},
            },
            {"key": "OTHER.csv", "checksum": "md5:00", "links": {}},
        ]
    }
    download = Mock(status_code=200, content=content)
    source.session.get = Mock(side_effect=lambda url, **kwargs: metadata if url.endswith("4624805") else download)
    source.logger = Mock()
    return source.session.get


def fetch_all(source):
    return source.fetch(pd.Timestamp("2000-01-01", tz="UTC"), pd.Timestamp("2030-01-01", tz="UTC"))


class TestInit:
    def test_defaults_and_validation(self, tmp_path):
        source = EpftoolboxBenchmarkSource("de", cache_dir=str(tmp_path))
        assert source.dataset == "DE"
        assert source.prefix == ""
        assert source.index_timezone == "Etc/GMT-1"
        assert source.test_period == ("2016-01-04", "2017-12-31")
        assert source.get_cache_config() is None
        assert EpftoolboxBenchmarkSource("PJM").index_timezone == "Etc/GMT+5"
        with pytest.raises(ValueError, match="Unknown dataset"):
            EpftoolboxBenchmarkSource("ES")

    def test_default_cache_dir_is_user_cache(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        source = EpftoolboxBenchmarkSource("NP")
        assert source._path == tmp_path / "epftoolbox2" / "epftoolbox-benchmark" / "NP.csv"


class TestDownload:
    def test_downloads_verifies_and_caches(self, tmp_path):
        content = benchmark_csv()
        source = EpftoolboxBenchmarkSource("DE", cache_dir=str(tmp_path))
        get = mock_zenodo(source, content)
        df = fetch_all(source)
        assert len(df) == 72
        urls = [c.args[0] for c in get.call_args_list]
        assert urls == [
            "https://zenodo.org/api/records/4624805",
            "https://zenodo.org/api/records/4624805/files/DE.csv/content",
        ]
        assert (tmp_path / "DE.csv").read_bytes() == content
        assert (tmp_path / "DE.csv.md5").read_text() == hashlib.md5(content).hexdigest()

        # a second run uses the verified copy without any network access
        again = EpftoolboxBenchmarkSource("DE", cache_dir=str(tmp_path))
        again.session.get = Mock(side_effect=AssertionError("no network expected"))
        pd.testing.assert_frame_equal(fetch_all(again), df)

    def test_md5_mismatch_raises_and_saves_nothing(self, tmp_path):
        source = EpftoolboxBenchmarkSource("DE", cache_dir=str(tmp_path))
        mock_zenodo(source, benchmark_csv(), md5="0" * 32)
        with pytest.raises(ValueError, match="md5 mismatch"):
            fetch_all(source)
        assert list(tmp_path.iterdir()) == []

    def test_modified_cached_file_is_downloaded_again(self, tmp_path):
        content = benchmark_csv()
        source = EpftoolboxBenchmarkSource("DE", cache_dir=str(tmp_path))
        mock_zenodo(source, content)
        fetch_all(source)
        (tmp_path / "DE.csv").write_bytes(content.replace(b"2600,", b"9999,", 1))

        source = EpftoolboxBenchmarkSource("DE", cache_dir=str(tmp_path))
        get = mock_zenodo(source, content)
        df = fetch_all(source)
        assert get.call_count == 2
        assert df["price"].iloc[0] == 2600
        assert (tmp_path / "DE.csv").read_bytes() == content

    def test_file_placed_by_hand_is_verified_against_zenodo(self, tmp_path):
        content = benchmark_csv()
        (tmp_path / "DE.csv").write_bytes(content)
        source = EpftoolboxBenchmarkSource("DE", cache_dir=str(tmp_path))
        get = mock_zenodo(source, content)
        fetch_all(source)
        assert get.call_count == 1  # metadata only, no download
        assert (tmp_path / "DE.csv.md5").exists()

    def test_missing_file_in_record(self, tmp_path):
        source = EpftoolboxBenchmarkSource("FR", cache_dir=str(tmp_path))
        mock_zenodo(source, benchmark_csv("FR"), dataset="DE")
        with pytest.raises(ValueError, match="has no file FR.csv"):
            fetch_all(source)


class TestFormat:
    @pytest.mark.parametrize("dataset", sorted(DATASETS))
    def test_columns_named_by_meaning(self, tmp_path, dataset):
        source = EpftoolboxBenchmarkSource(dataset, cache_dir=str(tmp_path))
        mock_zenodo(source, benchmark_csv(dataset), dataset=dataset)
        df = fetch_all(source)
        expected = ["price", *DATASETS[dataset]["exogenous"].values()]
        assert list(df.columns) == expected
        first = df.iloc[0]
        assert first["price"] == 2600
        assert first[expected[1]] == pytest.approx(2600.1)  # load-type forecast
        assert first[expected[2]] == pytest.approx(2600.2)

    def test_prefix(self, tmp_path):
        source = EpftoolboxBenchmarkSource("NP", cache_dir=str(tmp_path), prefix="np")
        mock_zenodo(source, benchmark_csv("NP"), dataset="NP")
        assert list(fetch_all(source).columns) == ["np_price", "np_load_forecast", "np_wind_forecast"]

    def test_unrecognised_header_raises(self, tmp_path):
        content = benchmark_csv().replace(b"PV+Wind Forecast", b"Something else")
        source = EpftoolboxBenchmarkSource("DE", cache_dir=str(tmp_path))
        mock_zenodo(source, content)
        with pytest.raises(ValueError, match="can't identify the 'wind' column"):
            fetch_all(source)

    def test_fixed_offset_index_keeps_24_hours_on_dst_days(self, tmp_path):
        # 2016-03-27 is the spring DST switch in CET/CEST; the file still has a 02:00 row
        source = EpftoolboxBenchmarkSource("DE", cache_dir=str(tmp_path))
        mock_zenodo(source, benchmark_csv("DE", first="2016-03-26", days=3))
        df = fetch_all(source)
        assert str(df.index.tz) == "UTC"
        assert df.loc[pd.Timestamp("2016-03-27 01:00", tz="UTC"), "price"] == 2702  # 02:00 wall clock = UTC+1
        local = TimezoneTransformer(target_tz=source.index_timezone).transform(df)
        counts = local.groupby(local.index.date).size()
        assert (counts == 24).all()
        assert (local["price"] == local.index.day * 100 + local.index.hour).all()

    def test_autumn_dst_day_and_day_lags(self, tmp_path):
        # 2016-10-30 is the autumn switch; with the fixed offset a 1-day lag is always the same wall-clock hour
        source = EpftoolboxBenchmarkSource("DE", cache_dir=str(tmp_path))
        mock_zenodo(source, benchmark_csv("DE", first="2016-10-29", days=3))
        local = TimezoneTransformer(target_tz=source.index_timezone).transform(fetch_all(source))
        lagged = LagTransformer(columns=["load_forecast_amprion"], lags=[-1], freq="day").transform(local)
        row = lagged.loc[pd.Timestamp("2016-10-30 02:00", tz="Etc/GMT-1")]
        assert row["load_forecast_amprion_d+1"] == pytest.approx(3102.1)  # 2016-10-31 02:00

    def test_fetch_slices_by_utc(self, tmp_path):
        source = EpftoolboxBenchmarkSource("PJM", cache_dir=str(tmp_path))
        mock_zenodo(source, benchmark_csv("PJM", first="2016-01-05", days=2), dataset="PJM")
        df = source.fetch(pd.Timestamp("2016-01-05 05:00", tz="UTC"), pd.Timestamp("2016-01-05 10:00", tz="UTC"))
        # 00:00 EST = 05:00 UTC; half-open [start, end), so 10:00 UTC (05:00 EST) itself is excluded
        assert list(df["price"]) == [500, 501, 502, 503, 504]

    def test_fetch_excludes_a_row_exactly_at_end(self, tmp_path):
        """DST/timezone review finding 4: fetch used to include a row exactly at `end` (`df.loc[start:end]`),
        breaking the source contract's half-open [start, end) that DataPipeline and the per-source cache rely
        on."""
        source = EpftoolboxBenchmarkSource("PJM", cache_dir=str(tmp_path))
        mock_zenodo(source, benchmark_csv("PJM", first="2016-01-05", days=2), dataset="PJM")
        start, end = pd.Timestamp("2016-01-05 05:00", tz="UTC"), pd.Timestamp("2016-01-05 06:00", tz="UTC")
        df = source.fetch(start, end)
        assert list(df["price"]) == [500]
        assert df.index.max() < end


class TestPipelineTimezone:
    """DST/timezone review finding 4: `index_timezone` is a fixed offset the source always has (derived from
    `dataset`), never inherited from `DataPipeline.timezone`. A pipeline zone that disagrees with it was never
    compared, so the mismatch was silent: the same UTC index gets read back as a different (date, hour) once
    passed through a `TimezoneTransformer(target_tz="Europe/Berlin")` instead of the source's own offset."""

    def test_mismatched_pipeline_timezone_raises(self, tmp_path):
        source = EpftoolboxBenchmarkSource("DE", cache_dir=str(tmp_path))  # index_timezone == "Etc/GMT-1"
        with pytest.raises(ValueError, match="Etc/GMT-1"):
            DataPipeline(sources=[source], timezone="Europe/Berlin")

    def test_mismatch_raised_on_run_too_if_added_after_construction(self, tmp_path):
        """The check also fires from `run()` (via `_validate_settings`), not only at construction, since a
        source can be added afterwards with `add_source`."""
        source = EpftoolboxBenchmarkSource("DE", cache_dir=str(tmp_path))
        mock_zenodo(source, benchmark_csv("DE"))
        pipeline = DataPipeline(timezone="Europe/Berlin").add_source(source)
        with pytest.raises(ValueError, match="Etc/GMT-1"):
            pipeline.run("2016-03-26", "2016-03-29")

    def test_matching_pipeline_timezone_is_fine(self, tmp_path):
        source = EpftoolboxBenchmarkSource("DE", cache_dir=str(tmp_path))
        mock_zenodo(source, benchmark_csv("DE"))
        pipeline = DataPipeline(sources=[source], timezone=source.index_timezone)
        assert not pipeline.run("2016-03-26", "2016-03-29").empty

    def test_no_pipeline_timezone_is_fine(self, tmp_path):
        source = EpftoolboxBenchmarkSource("DE", cache_dir=str(tmp_path))
        mock_zenodo(source, benchmark_csv("DE"))
        assert not DataPipeline(sources=[source]).run("2016-03-26", "2016-03-29").empty


class TestSerialization:
    def test_yaml_round_trip(self, tmp_path):
        original = EpftoolboxBenchmarkSource("BE", cache_dir=str(tmp_path), prefix="be")
        path = tmp_path / "pipeline.yaml"
        DataPipeline().add_source(original).save(path)
        loaded = DataPipeline.load(path).sources[0]
        assert isinstance(loaded, EpftoolboxBenchmarkSource)
        assert (loaded.dataset, loaded.cache_dir, loaded.prefix) == ("BE", str(tmp_path), "be")


@pytest.mark.integration
def test_zenodo_record_publishes_md5_for_every_dataset():
    """Reads only the record metadata; it doesn't download the data files."""
    source = EpftoolboxBenchmarkSource("DE")
    for dataset in DATASETS:
        source.dataset = dataset
        entry = source._record_file()
        assert len(entry["md5"]) == 32
        assert entry["url"].startswith("https://zenodo.org/")
