from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.sources.pse import SERIES, PseSource
from epftoolbox2.pipelines.data_pipeline import DataPipeline

TZ = "Europe/Warsaw"

# Records as returned by api.raporty.pse.pl (rce-pln, business day 2024-10-27 = autumn DST switch; prices replaced):
# `dtime_utc` is the END of the quarter-hour, local `dtime` labels the repeated hour "02a", a period may end at "24:00".
REAL_RCE_RECORDS = [
    {
        "dtime": "2024-10-27 03:00:00",
        "period": "02:45 - 03:00",
        "rce_pln": 400.0,
        "dtime_utc": "2024-10-27 01:00:00",
        "period_utc": "00:45 - 01:00",
        "business_date": "2024-10-27",
        "publication_ts": "2024-10-26 13:42:15.651",
        "publication_ts_utc": "2024-10-26 11:42:15.651",
    },
    {
        "dtime": "2024-10-27 02a:15:00",
        "period": "03:00 - 02a:15",
        "rce_pln": 410.0,
        "dtime_utc": "2024-10-27 01:15:00",
        "period_utc": "01:00 - 01:15",
        "business_date": "2024-10-27",
        "publication_ts": "2024-10-26 13:42:15.651",
        "publication_ts_utc": "2024-10-26 11:42:15.651",
    },
    {
        "dtime": "2024-10-27 02:00:00",
        "period": "01:45 - 02:00",
        "rce_pln": 390.0,
        "dtime_utc": "2024-10-27 00:00:00",
        "period_utc": "23:45 - 24:00",
        "business_date": "2024-10-27",
        "publication_ts": "2024-10-26 13:42:15.651",
        "publication_ts_utc": "2024-10-26 11:42:15.651000",
    },
]


def make_records(entity, first_day, last_day, value_fn, published_fn):
    """Quarter-hour records in the API format for local business days first_day..last_day.

    value_fn(start_utc) -> dict of field values; published_fn(business_date) -> naive UTC publication timestamp.
    """
    records = []
    for day in pd.date_range(first_day, last_day, freq="D"):
        begin = day.tz_localize(TZ).tz_convert("UTC")
        stop = (day + pd.Timedelta(days=1)).tz_localize(TZ).tz_convert("UTC")
        for start in pd.date_range(begin, stop, freq="15min", inclusive="left"):
            end = start + pd.Timedelta(minutes=15)
            end_label = "24:00" if end.hour == 0 and end.minute == 0 else f"{end:%H:%M}"
            record = {
                "dtime_utc": f"{end:%Y-%m-%d %H:%M:%S}",
                "period_utc": f"{start:%H:%M} - {end_label}",
                "business_date": f"{day:%Y-%m-%d}",
                "publication_ts_utc": f"{published_fn(day):%Y-%m-%d %H:%M:%S.%f}",
            }
            record.update(value_fn(start))
            records.append(record)
    return records


def hour_code(start):
    """Distinct value per UTC quarter-hour: day-of-year * 100 + local hour, so hourly means are exact."""
    local = start.tz_convert(TZ)
    return local.dayofyear * 100 + local.hour


def day_ahead_published(day):
    """RCE-like: published on the day before delivery at 12:00 UTC."""
    return day - pd.Timedelta(days=1) + pd.Timedelta(hours=12)


def realised_published(day):
    """Realised-like: published on the day after delivery at 01:00 UTC (about 03:00 local)."""
    return day + pd.Timedelta(days=1, hours=1)


def mock_api(source, records_by_entity):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params))
        entity = url.rsplit("/", 1)[-1].split("?")[0]
        response = Mock(status_code=200)
        response.json.return_value = {"value": records_by_entity.get(entity, [])}
        response.raise_for_status.return_value = None
        return response

    source.session.get = Mock(side_effect=fake_get)
    source.logger = Mock()
    return calls


def rce_records(first_day, last_day, published_fn=day_ahead_published):
    return make_records("rce-pln", first_day, last_day, lambda s: {"rce_pln": float(hour_code(s))}, published_fn)


def demand_records(first_day, last_day, published_fn=realised_published):
    return make_records(
        "his-wlk-cal", first_day, last_day, lambda s: {"demand": float(hour_code(s)) + 0.5}, published_fn
    )


def local_row(df, local_time):
    return df.loc[pd.Timestamp(local_time, tz=TZ).tz_convert("UTC")]


class TestPseSourceInit:
    def test_defaults(self):
        source = PseSource(series=["rce"])
        assert source.series == ["rce"]
        assert source.issue_time == "11:30"
        assert source.lags == [0]
        assert source.as_of == "publication"
        assert source.timezone == TZ
        assert source.freq == "1h"
        assert source.prefix == "pse"

    def test_string_series_and_int_lag(self):
        source = PseSource(series="demand", lags=1)
        assert source.series == ["demand"]
        assert source.lags == [1]

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"series": []}, "At least one series"),
            ({"series": ["price"]}, "Unknown PSE series"),
            ({"series": ["rce", "rce"]}, "Duplicate series"),
            ({"series": ["rce"], "lags": [1, 1]}, "Duplicate lags"),
            ({"series": ["rce"], "lags": [1.5]}, "lags must be"),
            ({"series": ["rce"], "as_of": "latest"}, "as_of must be"),
            ({"series": ["rce"], "freq": "30min"}, "freq must be"),
            ({"series": ["rce"], "timezone": "Mars/Base"}, "Invalid timezone"),
            ({"series": ["rce"], "issue_time": "25:00"}, "issue_time must be"),
            ({"series": ["rce"], "issue_time": None, "lags": [1]}, "need issue_time"),
        ],
    )
    def test_invalid_config(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            PseSource(**kwargs)

    @pytest.mark.parametrize("issue_time", [1020, "1020", "11.30", "11:30 CET"])
    def test_issue_time_must_be_an_hh_mm_string(self, issue_time):
        # Unquoted `issue_time: 17:00` in hand-written YAML is read as the base-60 integer 1020, which Python 3.11+
        # would otherwise parse as 10:20.
        with pytest.raises(ValueError, match="issue_time must be"):
            PseSource(series=["rce"], issue_time=issue_time)

    def test_forecast_series_are_not_offered(self):
        # PSE keeps only the last version of its forecasts, so none may be offered as backtest predictors.
        assert all(kind in ("day_ahead", "realised") for _, _, kind in SERIES.values())
        assert not any("fcst" in field or "forecast" in name for name, (_, field, _) in SERIES.items())


class TestParsing:
    def test_real_records_use_period_start_in_utc(self):
        source = PseSource(series=["rce"], issue_time=None, freq="15min")
        frame = source._parse_records(REAL_RCE_RECORDS, ["rce_pln"])
        starts = list(frame["start"])
        # dtime_utc is the end of the quarter-hour; "23:45 - 24:00" ends at the next UTC midnight
        assert starts == [
            pd.Timestamp("2024-10-27 00:45", tz="UTC"),
            pd.Timestamp("2024-10-27 01:00", tz="UTC"),
            pd.Timestamp("2024-10-26 23:45", tz="UTC"),
        ]
        assert list(frame["minutes"]) == [15, 15, 15]
        assert frame["published"].iloc[0] == pd.Timestamp("2024-10-26 11:42:15.651")

    def test_raw_mode_fetch_of_real_records(self):
        source = PseSource(series=["rce"], issue_time=None, freq="15min")
        mock_api(source, {"rce-pln": REAL_RCE_RECORDS})
        df = source.fetch(pd.Timestamp("2024-10-26 23:45", tz="UTC"), pd.Timestamp("2024-10-27 01:00", tz="UTC"))
        assert str(df.index.tz) == "UTC"
        assert df.loc[pd.Timestamp("2024-10-26 23:45", tz="UTC"), "pse_rce"] == 390.0
        assert df.loc[pd.Timestamp("2024-10-27 00:45", tz="UTC"), "pse_rce"] == 400.0
        assert df.loc[pd.Timestamp("2024-10-27 01:00", tz="UTC"), "pse_rce"] == 410.0
        assert np.isnan(df.loc[pd.Timestamp("2024-10-27 00:00", tz="UTC"), "pse_rce"])

    def test_hourly_mean_and_incomplete_hours(self):
        records = make_records(
            "rce-pln",
            "2025-01-15",
            "2025-01-15",
            lambda s: {"rce_pln": float(s.minute)},  # 0, 15, 30, 45 -> hourly mean 22.5
            day_ahead_published,
        )
        records = [r for r in records if r["dtime_utc"] != "2025-01-15 05:15:00"]  # drop 05:00-05:15 UTC
        source = PseSource(series=["rce"], issue_time=None)
        mock_api(source, {"rce-pln": records})
        df = source.fetch(pd.Timestamp("2025-01-15 00:00", tz="UTC"), pd.Timestamp("2025-01-15 10:00", tz="UTC"))
        assert df.loc[pd.Timestamp("2025-01-15 04:00", tz="UTC"), "pse_rce"] == 22.5
        # an hour with a missing quarter is not passed off as an hourly mean
        assert np.isnan(df.loc[pd.Timestamp("2025-01-15 05:00", tz="UTC"), "pse_rce"])

    def test_pagination_follows_next_link(self):
        source = PseSource(series=["rce"], issue_time=None)
        records = rce_records("2025-01-15", "2025-01-15")
        pages = [
            {"value": records[:50], "nextLink": "https://api.raporty.pse.pl/api/rce-pln?$after=abc"},
            {"value": records[50:]},
        ]
        responses = []
        for page in pages:
            response = Mock(status_code=200)
            response.json.return_value = page
            responses.append(response)
        source.session.get = Mock(side_effect=responses)
        source.logger = Mock()
        df = source.fetch(pd.Timestamp("2025-01-15 00:00", tz="UTC"), pd.Timestamp("2025-01-15 20:00", tz="UTC"))
        assert source.session.get.call_count == 2
        second_call = source.session.get.call_args_list[1]
        assert second_call.args[0].endswith("$after=abc")
        assert second_call.kwargs["params"] is None
        assert df["pse_rce"].notna().all()

    def test_requested_business_days_cover_lags(self):
        source = PseSource(series=["rce"], lags=[7, -1])
        calls = mock_api(source, {"rce-pln": []})
        source.fetch(pd.Timestamp("2025-01-15", tz="UTC"), pd.Timestamp("2025-01-20", tz="UTC"))
        params = calls[0][1]
        # rows span local days 2025-01-15..2025-01-20; d-7 needs 01-08 and d+1 needs 01-21 (plus a day of margin)
        assert params["$filter"] == "business_date ge '2025-01-07' and business_date le '2025-01-22'"
        assert set(params["$select"].split(",")) == {
            "business_date",
            "dtime_utc",
            "period_utc",
            "publication_ts_utc",
            "rce_pln",
        }


class TestAsOfPublication:
    def fetch(self, source, start="2025-01-10", end="2025-01-20", **records):
        mock_api(source, records)
        return source.fetch(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))

    def test_day_ahead_price_is_known_on_its_delivery_day(self):
        source = PseSource(series=["rce"], issue_time="11:30", lags=[0, 1, -1])
        df = self.fetch(source, **{"rce-pln": rce_records("2025-01-05", "2025-01-25")})
        row = local_row(df, "2025-01-15 20:00")
        assert row["pse_rce"] == 15 * 100 + 20
        assert row["pse_rce_d-1"] == 14 * 100 + 20
        # D+1 is published on D at 12:00 UTC = 13:00 local, after the 11:30 issue time
        assert np.isnan(row["pse_rce_d+1"])
        assert df["pse_rce_d+1"].isna().all()

    def test_day_ahead_price_for_tomorrow_known_after_publication(self):
        source = PseSource(series=["rce"], issue_time="13:30", lags=[-1])
        df = self.fetch(source, **{"rce-pln": rce_records("2025-01-05", "2025-01-25")})
        assert local_row(df, "2025-01-15 20:00")["pse_rce_d+1"] == 16 * 100 + 20

    def test_realised_values_need_a_lag(self):
        source = PseSource(series=["demand"], issue_time="11:30", lags=[0, 1, 2])
        df = self.fetch(source, **{"his-wlk-cal": demand_records("2025-01-05", "2025-01-25")})
        assert df["pse_demand"].isna().all()  # D's demand is published on D+1
        row = local_row(df, "2025-01-15 20:00")
        assert row["pse_demand_d-1"] == 14 * 100 + 20.5
        assert row["pse_demand_d-2"] == 13 * 100 + 20.5
        source.logger.warning.assert_any_call(
            f"PSE: pse_demand is empty: none of its {len(df)} values was published by 11:30 on the row's day "
            "(as_of='publication'); use a larger lag"
        )

    def test_empty_column_hint_for_a_negative_lag_suggests_a_smaller_horizon(self):
        # lags=[-1] asks for tomorrow's realised demand, which is never published yet at issue time. "use a
        # larger lag" would be wrong here: a more negative lag looks even further ahead, making it worse.
        source = PseSource(series=["demand"], issue_time="11:30", lags=[-1])
        df = self.fetch(source, **{"his-wlk-cal": demand_records("2025-01-05", "2025-01-25")})
        assert df["pse_demand_d+1"].isna().all()
        source.logger.warning.assert_any_call(
            f"PSE: pse_demand_d+1 is empty: none of its {len(df)} values was published by 11:30 on the row's "
            "day (as_of='publication'); use a lag closer to zero"
        )

    def test_republished_value_is_dropped(self):
        def published(day):
            # 2025-01-14 was re-published a year later: its last version didn't exist on 2025-01-15
            if day == pd.Timestamp("2025-01-14"):
                return pd.Timestamp("2026-01-18 17:00")
            return realised_published(day)

        source = PseSource(series=["demand"], issue_time="11:30", lags=[1, 2])
        df = self.fetch(source, **{"his-wlk-cal": demand_records("2025-01-05", "2025-01-25", published)})
        assert np.isnan(local_row(df, "2025-01-15 20:00")["pse_demand_d-1"])
        assert np.isnan(local_row(df, "2025-01-16 20:00")["pse_demand_d-2"])
        assert local_row(df, "2025-01-16 20:00")["pse_demand_d-1"] == 15 * 100 + 20.5
        assert local_row(df, "2025-01-15 20:00")["pse_demand_d-2"] == 13 * 100 + 20.5
        warnings = [c.args[0] for c in source.logger.warning.call_args_list]
        assert any(f"pse_demand_d-1: 24 of {len(df)} values were published after 11:30" in w for w in warnings)

    def test_issue_time_follows_local_clock_across_dst(self):
        # Published at 09:45 UTC on the delivery day: 10:45 CET in winter, 11:45 CEST in summer; issue time 11:30 local
        def published(day):
            return day + pd.Timedelta(hours=9, minutes=45)

        source = PseSource(series=["rce"], issue_time="11:30", lags=[0])
        records = rce_records("2025-01-14", "2025-01-16", published) + rce_records(
            "2025-07-14", "2025-07-16", published
        )
        mock_api(source, {"rce-pln": records})
        winter = source.fetch(pd.Timestamp("2025-01-15", tz="UTC"), pd.Timestamp("2025-01-15 20:00", tz="UTC"))
        summer = source.fetch(pd.Timestamp("2025-07-15", tz="UTC"), pd.Timestamp("2025-07-15 20:00", tz="UTC"))
        assert winter["pse_rce"].notna().all()
        assert summer["pse_rce"].isna().all()

    def test_lags_use_local_calendar_days_across_autumn_dst(self):
        source = PseSource(series=["rce"], issue_time="11:30", lags=[1])
        df = self.fetch(source, "2024-10-26", "2024-10-30", **{"rce-pln": rce_records("2024-10-20", "2024-11-02")})
        # Day after the switch: 10:00 local on 10-28 takes 10:00 local on 10-27, not the value 24 h earlier
        assert local_row(df, "2024-10-28 10:00")["pse_rce_d-1"] == pd.Timestamp("2024-10-27").dayofyear * 100 + 10
        # The switch day itself has both 02:00 hours; each takes 02:00 of the day before
        switch_day = df.tz_convert(TZ)
        twos = switch_day[(switch_day.index.day == 27) & (switch_day.index.hour == 2)]["pse_rce_d-1"]
        assert len(twos) == 2
        assert (twos == pd.Timestamp("2024-10-26").dayofyear * 100 + 2).all()
        # 25 local hours on the switch day, all filled
        assert switch_day.loc["2024-10-27", "pse_rce_d-1"].notna().sum() == 25

    def test_lags_across_spring_dst_leave_missing_hour_empty(self):
        source = PseSource(series=["rce"], issue_time="11:30", lags=[1])
        df = self.fetch(source, "2025-03-29", "2025-04-02", **{"rce-pln": rce_records("2025-03-25", "2025-04-05")})
        # 2025-03-30 has no 02:00 local, so 02:00 on 03-31 has no "same hour yesterday"
        assert np.isnan(local_row(df, "2025-03-31 02:00")["pse_rce_d-1"])
        assert local_row(df, "2025-03-31 03:00")["pse_rce_d-1"] == pd.Timestamp("2025-03-30").dayofyear * 100 + 3

    def test_values_published_after_issue_time_never_leak(self):
        """Leakage probe: changing values published after the issue time must not change the output."""

        def published(day):
            return realised_published(day)

        base = demand_records("2025-01-05", "2025-01-25", published)
        perturbed = [dict(r) for r in base]
        for r in perturbed:
            if r["business_date"] == "2025-01-15":  # published 2025-01-16 02:00 local, after the 01:30 issue time
                r["demand"] = 1e9
        kwargs = dict(series=["demand"], issue_time="01:30", lags=[0, 1, 2])
        a = self.fetch(PseSource(**kwargs), **{"his-wlk-cal": base})
        b = self.fetch(PseSource(**kwargs), **{"his-wlk-cal": perturbed})
        days = a.tz_convert(TZ).index.normalize()
        known_before = days <= pd.Timestamp("2025-01-16", tz=TZ)
        pd.testing.assert_frame_equal(a[known_before], b[known_before])
        rows_17 = days == pd.Timestamp("2025-01-17", tz=TZ)
        assert (b.loc[rows_17, "pse_demand_d-2"] == 1e9).all()

    def test_spring_gap_lands_on_switch_day_plus_lag(self):
        source = PseSource(series=["rce"], issue_time="11:30", lags=[7, -1])
        df = self.fetch(source, "2025-03-25", "2025-04-10", **{"rce-pln": rce_records("2025-03-15", "2025-04-15")})
        assert np.isnan(local_row(df, "2025-04-06 02:00")["pse_rce_d-7"])  # 2025-03-30 + 7
        assert np.isnan(local_row(df, "2025-03-29 02:00")["pse_rce_d+1"])  # 2025-03-30 - 1
        assert df["pse_rce_d-7"].isna().sum() == 1
        source.logger.info.assert_any_call(
            f"PSE: pse_rce_d-7: 1 of {len(df)} rows have no source value "
            "(no data, or the hour missing on a spring DST day)"
        )

    @pytest.mark.parametrize("as_of", ["publication", "schedule"])
    @pytest.mark.parametrize("freq", ["1h", "15min"])
    @pytest.mark.parametrize("switch_day", ["2024-10-27", "2025-03-30"])
    def test_leakage_probe_across_dst(self, as_of, freq, switch_day):
        """Realised values of day D and later are unknown at 11:30 on D under both rules; changing them must not
        change any row of D or earlier, including DST switch days and quarter-hour output."""
        day = pd.Timestamp(switch_day)
        base = demand_records(day - pd.Timedelta(days=12), day + pd.Timedelta(days=12))
        perturbed = [dict(r, demand=1e9) if pd.Timestamp(r["business_date"]) >= day else r for r in base]
        kwargs = dict(series=["demand"], issue_time="11:30", lags=[0, 1, 2, 7], as_of=as_of, freq=freq)
        start, end = day - pd.Timedelta(days=3), day + pd.Timedelta(days=3)
        a = self.fetch(PseSource(**kwargs), start, end, **{"his-wlk-cal": base})
        b = self.fetch(PseSource(**kwargs), start, end, **{"his-wlk-cal": perturbed})
        days = a.tz_convert(TZ).index.tz_localize(None).normalize()
        pd.testing.assert_frame_equal(a[days <= day], b[days <= day])
        next_day = b.loc[days == day + pd.Timedelta(days=1), "pse_demand_d-1"].dropna()
        assert len(next_day) >= 23 * (4 if freq == "15min" else 1) and (next_day == 1e9).all()


class TestAsOfSchedule:
    def test_schedule_still_drops_a_value_actually_published_after_the_issue_time(self):
        # A known (not missing) publication_ts that is later than the issue time must still drop the value: the
        # nominal schedule is a fallback for when publication_ts is unknown, not permission to ignore a known,
        # later one. Every record here was really re-published long after delivery, so both rules must agree.
        def published(day):
            return pd.Timestamp("2026-01-18 17:00")  # everything re-published long after delivery

        records = demand_records("2025-01-05", "2025-01-25", published)
        strict = PseSource(series=["demand"], issue_time="11:30", lags=[0, 1])
        mock_api(strict, {"his-wlk-cal": records})
        nominal = PseSource(series=["demand"], issue_time="11:30", lags=[0, 1], as_of="schedule")
        mock_api(nominal, {"his-wlk-cal": records})
        start, end = pd.Timestamp("2025-01-10", tz="UTC"), pd.Timestamp("2025-01-20", tz="UTC")
        df_strict, df_nominal = strict.fetch(start, end), nominal.fetch(start, end)
        assert df_strict["pse_demand_d-1"].isna().all()
        assert df_nominal["pse_demand_d-1"].isna().all()
        # realised D is nominally published on D+1 06:00, so it is still unknown at 11:30 on D
        assert df_nominal["pse_demand"].isna().all()

    def test_schedule_day_ahead_published_previous_evening(self):
        source = PseSource(series=["rce"], issue_time="17:00", lags=[0, -1], as_of="schedule")
        mock_api(source, {"rce-pln": rce_records("2025-01-05", "2025-01-25")})
        df = source.fetch(pd.Timestamp("2025-01-10", tz="UTC"), pd.Timestamp("2025-01-20", tz="UTC"))
        assert df["pse_rce"].notna().all()
        assert df["pse_rce_d+1"].isna().all()  # nominally known from 18:00 on D

    def test_schedule_falls_back_to_nominal_only_when_publication_is_unknown(self):
        # _availability('schedule'): available once the nominal schedule time has passed AND, wherever
        # publication_ts is known, the actual publication has also happened by then; a missing publication_ts
        # (NaT) falls back to the nominal schedule alone.
        source = PseSource(series=["demand"], issue_time="11:30", lags=[1], as_of="schedule")
        frame = pd.DataFrame(
            {
                "business_date": [pd.Timestamp("2025-03-10")] * 3,
                "published": [
                    pd.Timestamp("2025-03-11 19:00"),  # later than the nominal schedule: this governs
                    pd.Timestamp("2025-03-11 03:00"),  # earlier than the nominal schedule: nominal still governs
                    pd.NaT,  # unknown: nominal alone applies
                ],
            }
        )
        available = source._availability(frame, "realised")
        nominal = pd.Timestamp("2025-03-11 05:00")  # business_date + 1 day, 06:00 Europe/Warsaw (winter) = 05:00 UTC
        assert available.iloc[0] == pd.Timestamp("2025-03-11 19:00")
        assert available.iloc[1] == nominal
        assert available.iloc[2] == nominal

    def test_values_published_after_issue_time_never_leak(self):
        """Leakage probe for as_of='schedule' (mirrors TestAsOfPublication's probe of the same name): a value
        that is nominally due by the schedule, but was actually (re-)published later according to
        publication_ts, must not leak into rows issued before that actual publication."""

        def published(day):
            if day == pd.Timestamp("2025-01-15"):
                # nominally known from 2025-01-16 06:00 local (well before that day's 11:30 issue time), but
                # actually republished at 19:00 UTC on 2025-01-16, after that day's issue time
                return pd.Timestamp("2025-01-16 19:00")
            return realised_published(day)

        base = demand_records("2025-01-05", "2025-01-25", published)
        perturbed = [dict(r) for r in base]
        for r in perturbed:
            if r["business_date"] == "2025-01-15":
                r["demand"] = 1e9

        kwargs = dict(series=["demand"], issue_time="11:30", lags=[0, 1, 2], as_of="schedule")
        start, end = pd.Timestamp("2025-01-10", tz="UTC"), pd.Timestamp("2025-01-20", tz="UTC")
        a_source, b_source = PseSource(**kwargs), PseSource(**kwargs)
        mock_api(a_source, {"his-wlk-cal": base})
        mock_api(b_source, {"his-wlk-cal": perturbed})
        a, b = a_source.fetch(start, end), b_source.fetch(start, end)

        days = a.tz_convert(TZ).index.normalize()
        known_before = days <= pd.Timestamp("2025-01-16", tz=TZ)
        pd.testing.assert_frame_equal(a[known_before], b[known_before])
        rows_17 = days == pd.Timestamp("2025-01-17", tz=TZ)
        assert (b.loc[rows_17, "pse_demand_d-2"] == 1e9).all()


class TestPseSourceConfig:
    def test_cache_config_covers_all_parameters(self):
        source = PseSource(series=["demand", "rce"], issue_time="10:00", lags=[7, 1], as_of="schedule", freq="15min")
        assert source.get_cache_config() == {
            "source_type": "pse",
            "schema_version": 1,
            "series": ["demand", "rce"],
            "issue_time": "10:00",
            "lags": [1, 7],
            "as_of": "schedule",
            "timezone": TZ,
            "freq": "15min",
            "prefix": "pse",
        }
        other = PseSource(series=["demand", "rce"], issue_time="11:00", lags=[7, 1], as_of="schedule", freq="15min")
        assert other.get_cache_config() != source.get_cache_config()

    def test_yaml_round_trip(self, tmp_path):
        path = tmp_path / "pipeline.yaml"
        original = PseSource(series=["rce", "wind"], issue_time="10:30", lags=[0, 1, 7], as_of="schedule", prefix="pl")
        DataPipeline().add_source(original).save(path)
        loaded = DataPipeline.load(path).sources[0]
        assert isinstance(loaded, PseSource)
        for name in ("series", "issue_time", "lags", "as_of", "timezone", "freq", "prefix"):
            assert getattr(loaded, name) == getattr(original, name)

    def test_yaml_round_trip_raw_mode(self, tmp_path):
        path = tmp_path / "pipeline.yaml"
        DataPipeline().add_source(PseSource(series=["rce"], issue_time=None)).save(path)
        loaded = DataPipeline.load(path).sources[0]
        assert loaded.issue_time is None
        assert loaded.lags == [0]


@pytest.mark.integration
def test_live_api_rce_and_demand():
    source = PseSource(series=["rce", "demand"], issue_time="11:30", lags=[0, 1])
    df = source.fetch(pd.Timestamp("2025-02-03", tz="UTC"), pd.Timestamp("2025-02-05", tz="UTC"))
    assert {"pse_rce", "pse_rce_d-1", "pse_demand", "pse_demand_d-1"} <= set(df.columns)
    assert df["pse_rce"].notna().all()
    assert df["pse_demand"].isna().all()
