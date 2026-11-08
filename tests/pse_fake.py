"""A fake of the PSE reports API for the tests of `PseSource`, in the shape the live API answers (checked 2026-11-08).

A 15-minute row is stamped with the END of its period (`dtime` local, `dtime_utc`), `business_date` is the local date of the
period's start, and `pdgsz` is hourly with one row per version of an hour. Values are a known function of the period's start
(`signal`), or the series a test passes, so what a parser did to the time axis shows in the numbers.
"""

import re
from typing import Optional
from unittest.mock import Mock
from urllib.parse import parse_qs, quote, urlparse

import numpy as np
import pandas as pd

from epftoolbox2.data.sources import PseSource

WARSAW = "Europe/Warsaw"
API = PseSource.API_URL
SPRING, AUTUMN = "2026-03-29", "2026-10-25"  # 23 and 25 hours in Poland


def utc(value):
    return pd.Timestamp(value, tz="UTC")


def signal(starts: pd.DatetimeIndex, offset: float = 0.0) -> np.ndarray:
    """A value that tells the period it belongs to: the number of quarter-hours since 2026-01-01, mod 1009."""
    return ((starts - utc("2026-01-01")) // pd.Timedelta(minutes=15) % 1009).to_numpy() + offset


def local_day_starts(day: str, step: str) -> pd.DatetimeIndex:
    first = pd.Timestamp(day, tz=WARSAW)
    nxt = (first + pd.Timedelta(hours=30)).normalize()
    return pd.date_range(first, nxt, freq=step, inclusive="left").tz_convert("UTC")


def stamp(instant: pd.Timestamp, local: bool, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    return (instant.tz_convert(WARSAW) if local else instant).strftime(fmt)


def quarter_rows(slug: str, day: str, series: Optional[pd.DataFrame] = None) -> list:
    """The rows of one business day of a 15-minute table, as the API gives them. `series` (columns indexed by the UTC start
    of a period) replaces the signal of the columns it has."""
    rows = []
    for start in local_day_starts(day, "15min"):
        end = start + pd.Timedelta(minutes=15)
        row = {
            "dtime": stamp(end, True),
            "period": f"{start.tz_convert(WARSAW):%H:%M} - {end.tz_convert(WARSAW):%H:%M}".replace(
                "- 00:00", "- 24:00"
            ),
            "dtime_utc": stamp(end, False),
            "period_utc": f"{start:%H:%M} - {end:%H:%M}",
            "business_date": day,
            "publication_ts": "2026-11-01 03:05:07.445",
            "publication_ts_utc": "2026-11-01 02:05:07.445",
        }
        one = pd.DatetimeIndex([start])

        def signal_of(column, offset):
            if series is not None and column in series:
                return float(series.at[start, column])
            return float(signal(one, offset)[0])

        if slug == "kse-load":
            row.update(load_fcst=signal_of("load_fcst", 5000), load_actual=signal_of("load_actual", 9000))
        elif slug == "csdac-pln":
            row["csdac_pln"] = signal_of("csdac_pln", 100)
        elif slug == "rce-pln":
            row["rce_pln"] = float(signal(one, 200)[0])
        elif slug == "his-wlk-cal":
            row.update(demand=float(signal(one, 1)[0]), wi=float(signal(one, 2)[0]), pv=float(signal(one, 3)[0]))
            row.update(jg=float(signal(one, 4)[0]), jga=None, jgm=-1.5, jgo=None, jgm1=0.0, jgm2=0.0, jgw1=7.0)
            row.update(jgw2=8.0, jgz1=None, jgz2=9.0, jgz3=None, jnwrb=10.0, swm_p=11.0, swm_np=12.0)
        elif slug == "przeplywy-mocy":
            for k, section in enumerate(["DE-PL", "PL-DE"]):
                rows.append({**row, "section_code": section, "value": float(signal(one, 1000 * (k + 1))[0])})
            continue
        rows.append(row)
    return rows


def pdgsz_rows(day: str, versions=(("2026-03-27 15:35:48.250", 1), ("2026-03-28 18:49:24.203", 2))) -> list:
    """Hourly rows with one row per version, listed newest version first. A version's `usage_fcst` is its number."""
    rows = []
    for start in local_day_starts(day, "1h"):
        for written, number in versions:
            written_utc = pd.Timestamp(written, tz=WARSAW).tz_convert("UTC")
            rows.append(
                {
                    "dtime": stamp(start, True, "%Y-%m-%d %H:%M"),
                    "dtime_utc": stamp(start, False, "%Y-%m-%d %H:%M"),
                    "is_active": number == versions[-1][1],
                    "usage_fcst": number,
                    "valid_to_ts": None,
                    "business_date": day,
                    "valid_from_ts": written,
                    "publication_ts": written,
                    "publication_ts_utc": stamp(written_utc, False, "%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "total_power_demand": None,
                }
            )
    rows.sort(key=lambda r: (r["dtime_utc"], -r["usage_fcst"]))  # newest version first within an hour
    return rows


def table(slug: str, days, series: Optional[pd.DataFrame] = None) -> list:
    if slug == "pdgsz":
        return [row for day in days for row in pdgsz_rows(day)]
    return [row for day in days for row in quarter_rows(slug, day, series)]


class FakeApi:
    """Stands in for `session.get`: filters on business_date, pages by `$first`/`$after` and follows nextLink."""

    def __init__(self, tables: dict, scripted=()):
        self.tables = tables
        self.scripted = list(scripted)  # responses to give first, before the table
        self.calls = []

    def __call__(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        if self.scripted:
            item = self.scripted.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        parsed = urlparse(url)
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        query.update(params or {})
        slug = parsed.path.rsplit("/", 1)[1]
        first_day, last_day = re.findall(r"'(\d{4}-\d{2}-\d{2})'", query["$filter"])
        rows = [r for r in self.tables[slug] if first_day <= r["business_date"] <= last_day]
        size, after = int(query["$first"]), int(query.get("$after", 0))
        body = {"value": rows[after : after + size]}
        if after + size < len(rows):
            body["nextLink"] = f"{API}/{slug}?$filter={quote(query['$filter'])}&$first={size}&$after={after + size}"
        return answer(body)

    def filters(self):
        return [c["params"]["$filter"] for c in self.calls if c["params"]]


def answer(body=None, status=200, headers=None, text=""):
    response = Mock(status_code=status, headers=headers or {}, reason="reason", text=text or str(body))
    response.json.return_value = body
    return response


def make_source(tables, dataset_names, monkeypatch=None, **kwargs):
    source = PseSource(datasets=dataset_names, **kwargs)
    source.session.get = FakeApi(tables)
    source.logger = Mock()
    return source
