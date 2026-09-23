from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.sources.settlement import SettlementPriceSource
from epftoolbox2.pipelines.data_pipeline import DataPipeline

TZ = "Europe/Warsaw"


def close_value(day):
    """Known answer: the close of trading day `day` is its ordinal date, e.g. 2024-03-15 -> 20240315."""
    return float(pd.Timestamp(day).strftime("%Y%m%d"))


def write_closes(path, days, columns=("TTF",), sep=",", date_fmt="%Y-%m-%d", scale=None):
    rows = {"date": [pd.Timestamp(d).strftime(date_fmt) for d in days]}
    for i, col in enumerate(columns):
        rows[col] = [close_value(d) + (scale or 0) * i for d in days]
    pd.DataFrame(rows).to_csv(path, index=False, sep=sep)
    return path


def weekdays(start, end, skip=()):
    return [d for d in pd.bdate_range(start, end) if d.strftime("%Y-%m-%d") not in skip]


def make_source(path, **kwargs):
    kwargs.setdefault("columns", {"TTF": "ttf"})
    kwargs.setdefault("timezone", TZ)
    source = SettlementPriceSource(file_path=str(path), **kwargs)
    source.logger = Mock()
    return source


def fetch(source, start, end):
    return source.fetch(pd.Timestamp(start, tz=TZ).tz_convert("UTC"), pd.Timestamp(end, tz=TZ).tz_convert("UTC"))


def value_on(df, local_time, column):
    return df.loc[pd.Timestamp(local_time, tz=TZ).tz_convert("UTC"), column]


def warnings_of(source):
    return [c.args[0] for c in source.logger.warning.call_args_list]


class TestInit:
    def test_defaults(self, tmp_path):
        path = write_closes(tmp_path / "f.csv", weekdays("2024-01-01", "2024-01-31"))
        source = SettlementPriceSource(file_path=str(path), columns=["TTF"], timezone=TZ)
        assert source.issue_time == "11:30"
        assert source.close_time == "23:59"
        assert source.close_timezone is None
        assert source.date_column == "date"
        assert source.trading_calendar == "weekdays"
        assert source.lags == [1]
        assert source.freq == "1h"
        assert source.get_cache_config() is None

    @pytest.mark.parametrize(
        "kwargs, error, match",
        [
            ({"lags": [-1]}, ValueError, "non-negative"),
            ({"lags": [1, 1]}, ValueError, "Duplicate lags"),
            ({"issue_time": "noon"}, ValueError, "issue_time must be"),
            ({"close_time": "25:00"}, ValueError, "close_time must be"),
            ({"timezone": "Nowhere/City"}, ValueError, "Invalid timezone"),
            ({"close_timezone": "Nowhere/City"}, ValueError, "Invalid timezone"),
            ({"trading_calendar": "EEX_MADE_UP"}, ValueError, "trading_calendar must be"),
            ({"freq": "2D"}, ValueError, "at most one day"),
            ({"columns": {"TTF": "x", "EUA": "x"}}, ValueError, "unique"),
            ({"columns": []}, ValueError, "at least one"),
        ],
    )
    def test_invalid_config(self, tmp_path, kwargs, error, match):
        path = write_closes(tmp_path / "f.csv", weekdays("2024-01-01", "2024-01-31"))
        kwargs = {"columns": ["TTF"], "timezone": TZ, **kwargs}
        with pytest.raises(error, match=match):
            SettlementPriceSource(file_path=str(path), **kwargs)

    def test_missing_file_and_bad_extension(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SettlementPriceSource(file_path=str(tmp_path / "none.csv"), columns=["TTF"], timezone=TZ)
        (tmp_path / "f.xlsx").write_text("x")
        with pytest.raises(ValueError, match="must be one of"):
            SettlementPriceSource(file_path=str(tmp_path / "f.xlsx"), columns=["TTF"], timezone=TZ)


class TestAsOfCloses:
    def test_previous_close_and_weekend_fill(self, tmp_path):
        path = write_closes(tmp_path / "f.csv", weekdays("2024-02-01", "2024-02-29"))
        df = fetch(make_source(path), "2024-02-05", "2024-02-13")  # Monday .. Tuesday a week later
        assert list(df.columns) == ["ttf_d-1"]
        assert str(df.index.tz) == "UTC"
        assert value_on(df, "2024-02-06 15:00", "ttf_d-1") == close_value("2024-02-05")  # Tue <- Mon
        assert value_on(df, "2024-02-10 15:00", "ttf_d-1") == close_value("2024-02-09")  # Sat <- Fri
        assert value_on(df, "2024-02-11 15:00", "ttf_d-1") == close_value("2024-02-09")  # Sun <- Fri
        assert value_on(df, "2024-02-12 15:00", "ttf_d-1") == close_value("2024-02-09")  # Mon <- Fri
        # every hour of a day carries the same daily value
        assert df.tz_convert(TZ).loc["2024-02-07", "ttf_d-1"].nunique() == 1

    def test_same_day_close_is_never_used_by_default(self, tmp_path):
        path = write_closes(tmp_path / "f.csv", weekdays("2024-02-01", "2024-02-29"))
        source = make_source(path, lags=[0, 1])
        df = fetch(source, "2024-02-05", "2024-02-12")
        weekday_rows = df.tz_convert(TZ).index.dayofweek < 5
        assert df.loc[weekday_rows, "ttf"].isna().all()
        # on a Saturday the prevailing close is Friday's, known since Friday evening
        assert value_on(df, "2024-02-10 15:00", "ttf") == close_value("2024-02-09")
        assert any("ttf: " in w and "use a larger lag" in w for w in warnings_of(source))

    def test_close_before_issue_time_is_usable_same_day(self, tmp_path):
        path = write_closes(tmp_path / "f.csv", weekdays("2024-02-01", "2024-02-29"))
        early = make_source(path, lags=[0], close_time="10:00", close_timezone="Europe/London")  # 11:00 CET
        late = make_source(path, lags=[0], close_time="12:00", close_timezone="Europe/London")  # 13:00 CET
        assert value_on(fetch(early, "2024-02-06", "2024-02-07"), "2024-02-06 15:00", "ttf") == close_value(
            "2024-02-06"
        )
        assert np.isnan(value_on(fetch(late, "2024-02-06", "2024-02-07"), "2024-02-06 15:00", "ttf"))

    def test_close_time_zone_follows_its_own_dst(self, tmp_path):
        # 06:00 New York is 12:00 CET in winter (after an 11:30 issue time) but 11:00 CET between the US and EU
        # spring switches (2024-03-10 and 2024-03-31).
        path = write_closes(tmp_path / "f.csv", weekdays("2024-02-01", "2024-04-30"))
        source = make_source(path, lags=[0], close_time="06:00", close_timezone="America/New_York")
        df = fetch(source, "2024-03-01", "2024-04-10")
        assert np.isnan(value_on(df, "2024-03-06 15:00", "ttf"))
        assert value_on(df, "2024-03-13 15:00", "ttf") == close_value("2024-03-13")
        assert np.isnan(value_on(df, "2024-04-03 15:00", "ttf"))

    def test_missing_close_on_trading_day_is_not_filled(self, tmp_path):
        path = write_closes(tmp_path / "f.csv", weekdays("2024-02-01", "2024-02-29", skip={"2024-02-14"}))
        source = make_source(path)
        df = fetch(source, "2024-02-12", "2024-02-18")
        assert np.isnan(value_on(df, "2024-02-15 15:00", "ttf_d-1"))  # needs Wed 02-14, which is missing
        assert value_on(df, "2024-02-16 15:00", "ttf_d-1") == close_value("2024-02-15")
        assert any("no close for 1 expected trading day(s) (2024-02-14)" in w for w in warnings_of(source))

    def test_rows_after_the_file_ends_stay_empty(self, tmp_path):
        path = write_closes(tmp_path / "f.csv", weekdays("2024-02-01", "2024-02-09"))
        df = fetch(make_source(path), "2024-02-08", "2024-02-16")
        assert value_on(df, "2024-02-12 15:00", "ttf_d-1") == close_value("2024-02-09")
        assert np.isnan(value_on(df, "2024-02-14 15:00", "ttf_d-1"))  # Tue 02-13 has no close: not stale-filled

    def test_holiday_calendar_fills_exchange_holidays(self, tmp_path):
        # Good Friday 2024-03-29 is an ICE Futures Europe holiday
        path = write_closes(tmp_path / "f.csv", weekdays("2024-03-01", "2024-04-30", skip={"2024-03-29"}))
        by_weekday = make_source(path)
        by_ice = make_source(path, trading_calendar="IFEU")
        on_saturday = "2024-03-30 15:00"
        assert np.isnan(value_on(fetch(by_weekday, "2024-03-25", "2024-04-03"), on_saturday, "ttf_d-1"))
        assert value_on(fetch(by_ice, "2024-03-25", "2024-04-03"), on_saturday, "ttf_d-1") == close_value("2024-03-28")
        assert not warnings_of(by_ice)

    def test_file_calendar_trusts_the_file(self, tmp_path):
        path = write_closes(tmp_path / "f.csv", weekdays("2024-02-01", "2024-02-29", skip={"2024-02-14"}))
        df = fetch(make_source(path, trading_calendar="file"), "2024-02-12", "2024-02-18")
        assert value_on(df, "2024-02-15 15:00", "ttf_d-1") == close_value("2024-02-13")

    def test_dst_days_get_one_value_per_local_day(self, tmp_path):
        path = write_closes(tmp_path / "f.csv", weekdays("2024-10-01", "2024-11-30"))
        df = fetch(make_source(path), "2024-10-26", "2024-10-29")
        local = df.tz_convert(TZ)
        switch_day = local.loc["2024-10-27", "ttf_d-1"]
        assert len(switch_day) == 25
        assert (switch_day == close_value("2024-10-25")).all()  # Sunday <- Friday
        assert (local.loc["2024-10-28", "ttf_d-1"] == close_value("2024-10-25")).all()  # Monday <- Friday

    def test_later_closes_never_leak(self, tmp_path):
        """Leakage probe: changing closes of D and later must not change any row of D."""
        days = weekdays("2024-02-01", "2024-02-29")
        base = write_closes(tmp_path / "a.csv", days)
        changed = pd.read_csv(base)
        changed.loc[changed["date"] >= "2024-02-14", "TTF"] = 1e9
        changed.to_csv(tmp_path / "b.csv", index=False)
        kwargs = dict(lags=[0, 1, 2])
        a = fetch(make_source(base, **kwargs), "2024-02-10", "2024-02-20")
        b = fetch(make_source(tmp_path / "b.csv", **kwargs), "2024-02-10", "2024-02-20")
        up_to_d = a.tz_convert(TZ).index.normalize() <= pd.Timestamp("2024-02-14", tz=TZ)
        pd.testing.assert_frame_equal(a[up_to_d], b[up_to_d])
        assert value_on(b, "2024-02-15 15:00", "ttf_d-1") == 1e9


class TestFiles:
    def test_column_mapping_and_list(self, tmp_path):
        path = write_closes(tmp_path / "f.csv", weekdays("2024-02-01", "2024-02-29"), columns=("TTF", "EUA"), scale=1)
        mapped = fetch(make_source(path, columns={"EUA": "eua_close"}), "2024-02-06", "2024-02-07")
        assert list(mapped.columns) == ["eua_close_d-1"]
        assert value_on(mapped, "2024-02-06 15:00", "eua_close_d-1") == close_value("2024-02-05") + 1
        listed = fetch(make_source(path, columns=["TTF", "EUA"]), "2024-02-06", "2024-02-07")
        assert list(listed.columns) == ["TTF_d-1", "EUA_d-1"]

    def test_missing_column_raises(self, tmp_path):
        path = write_closes(tmp_path / "f.csv", weekdays("2024-02-01", "2024-02-29"))
        with pytest.raises(ValueError, match="not found"):
            fetch(make_source(path, columns={"API2": "coal"}), "2024-02-06", "2024-02-07")

    def test_multiple_files_are_merged(self, tmp_path):
        ttf = write_closes(tmp_path / "ttf.csv", weekdays("2024-02-01", "2024-02-29"), columns=("TTF",))
        eua = write_closes(tmp_path / "eua.csv", weekdays("2024-02-01", "2024-02-29"), columns=("EUA",))
        source = SettlementPriceSource(
            file_path=[str(ttf), str(eua)], columns={"TTF": "ttf", "EUA": "eua"}, timezone=TZ, lags=[1, 2]
        )
        df = fetch(source, "2024-02-06", "2024-02-08")
        assert list(df.columns) == ["ttf_d-1", "ttf_d-2", "eua_d-1", "eua_d-2"]
        assert value_on(df, "2024-02-07 15:00", "eua_d-2") == close_value("2024-02-05")

    def test_conflicting_closes_raise(self, tmp_path):
        a = write_closes(tmp_path / "a.csv", weekdays("2024-02-01", "2024-02-15"))
        b = tmp_path / "b.csv"
        pd.DataFrame({"date": ["2024-02-15"], "TTF": [1.0]}).to_csv(b, index=False)
        source = SettlementPriceSource(file_path=[str(a), str(b)], columns=["TTF"], timezone=TZ)
        with pytest.raises(ValueError, match="Conflicting closes.*TTF on 2024-02-15"):
            fetch(source, "2024-02-06", "2024-02-07")

    def test_day_first_dates_and_separator(self, tmp_path):
        path = write_closes(tmp_path / "f.csv", weekdays("2024-02-01", "2024-02-29"), sep=";", date_fmt="%d.%m.%Y")
        source = make_source(path, separator=";", date_format="%d.%m.%Y")
        df = fetch(source, "2024-02-06", "2024-02-07")
        assert value_on(df, "2024-02-06 15:00", "ttf_d-1") == close_value("2024-02-05")

    def test_home_directory_is_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        write_closes(tmp_path / "f.csv", weekdays("2024-02-01", "2024-02-29"))
        df = fetch(make_source("~/f.csv"), "2024-02-06", "2024-02-07")
        assert value_on(df, "2024-02-06 15:00", "ttf_d-1") == close_value("2024-02-05")

    def test_parquet_without_engine_explains_what_to_install(self, tmp_path, monkeypatch):
        path = tmp_path / "f.parquet"
        path.write_bytes(b"PAR1")

        def no_engine(*args, **kwargs):
            raise ImportError("Unable to find a usable engine")

        monkeypatch.setattr(pd, "read_parquet", no_engine)
        with pytest.raises(ImportError, match="pip install pyarrow"):
            fetch(make_source(path), "2024-02-06", "2024-02-07")

    def test_parquet(self, tmp_path):
        pytest.importorskip("pyarrow")
        days = weekdays("2024-02-01", "2024-02-29")
        path = tmp_path / "f.parquet"
        pd.DataFrame({"date": days, "TTF": [close_value(d) for d in days]}).to_parquet(path)
        df = fetch(make_source(path), "2024-02-06", "2024-02-07")
        assert value_on(df, "2024-02-06 15:00", "ttf_d-1") == close_value("2024-02-05")


class TestSerialization:
    def test_yaml_round_trip(self, tmp_path):
        path = write_closes(tmp_path / "f.csv", weekdays("2024-02-01", "2024-02-29"), columns=("TTF", "EUA"))
        original = SettlementPriceSource(
            file_path=str(path),
            columns={"TTF": "ttf", "EUA": "eua"},
            timezone=TZ,
            issue_time="10:00",
            close_time="19:30",
            close_timezone="Europe/London",
            trading_calendar="IFEU",
            lags=[1, 2],
        )
        yaml_path = tmp_path / "pipeline.yaml"
        DataPipeline().add_source(original).save(yaml_path)
        loaded = DataPipeline.load(yaml_path).sources[0]
        assert isinstance(loaded, SettlementPriceSource)
        for name in (
            "file_path",
            "columns",
            "timezone",
            "issue_time",
            "close_time",
            "close_timezone",
            "date_column",
            "date_format",
            "trading_calendar",
            "lags",
            "freq",
            "separator",
        ):
            assert getattr(loaded, name) == getattr(original, name)

    def test_yaml_round_trip_file_list(self, tmp_path):
        a = write_closes(tmp_path / "a.csv", weekdays("2024-02-01", "2024-02-29"))
        b = write_closes(tmp_path / "b.csv", weekdays("2024-03-01", "2024-03-29"))
        original = SettlementPriceSource(file_path=[str(a), str(b)], columns=["TTF"], timezone=TZ)
        yaml_path = tmp_path / "pipeline.yaml"
        DataPipeline().add_source(original).save(yaml_path)
        assert DataPipeline.load(yaml_path).sources[0].file_path == [str(a), str(b)]
