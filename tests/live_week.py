"""The simulated week of step 8 (the operations spec's done criterion): a seeded truth, the publication schedule of each of its
cells, a fake ENTSO-E server that answers from what is published by the clock's now, and the pipelines that run on it.

The truth (`truth()`) is a 15-minute frame in UTC from 2026-07-01 to 2026-10-29 (local days, Europe/Berlin): the SDAC price, the
EXAA price, the A65 day-ahead load forecast and the actual load. Its schedule (`schedule()`) says when each cell appears, by the
local day d it belongs to: the price of d at 12:57 on d - 1, EXAA at 11:08 on d - 1, the load forecast at 10:12 on d - 1 and the
actual load one hour after its quarter-hour ends, with the week's exceptions:

- Thursday 2026-10-22: EXAA for Friday at 11:37 (a late source);
- Friday 2026-10-23: the load forecast for Saturday is never published (a missing input);
- Saturday 2026-10-24: every request answers 503 after 20 s from 08:00 to 11:25 (an outage, `OUTAGES`);
- Monday 2026-10-26: the load forecast for Tuesday, published at 10:12, is revised at 11:05 (`REVISED`);
- Tuesday 2026-10-27: one load forecast value for Wednesday is 1e6 (an implausible input, never corrected).

`FakeEntsoe` stands in for `EntsoeSource.session.get`: it answers A44 (classificationSequence positions 1 and 2) and A65 (A16
actual, A01 forecast; A31 has no data) with XML that the real parser reads, from the cells published by the clock's now. Each
request advances the clock by 1 s; during an outage it waits 20 s and answers 503; a request whose latency exceeds its timeout
raises `requests.Timeout` after advancing the clock by the timeout. No socket is opened and no real time passes.
"""

from __future__ import annotations

import datetime
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from epftoolbox2.clock import current_clock

ZONE = "Europe/Berlin"
FIRST_DAY, LAST_DAY = datetime.date(2026, 7, 1), datetime.date(2026, 10, 29)
WEEK = [
    datetime.date(2026, 10, 21) + datetime.timedelta(days=i) for i in range(7)
]  # Wednesday to Tuesday: the issue days
KEY = "simulated-week-key"  # a dummy ENTSO-E key: the fake server never reads it
QUARTER = pd.Timedelta(minutes=15)

LATE_EXAA = datetime.date(2026, 10, 23)  # its EXAA appears at 11:37 on Thursday 22
MISSING_A65 = datetime.date(2026, 10, 24)  # its load forecast never appears (Friday 23's issue)
REVISED_A65 = datetime.date(2026, 10, 27)  # its load forecast is revised at 11:05 on Monday 26
REVISED_AT = "11:05"
BAD_A65 = datetime.date(2026, 10, 28)  # one of its load forecast values is 1e6 (Tuesday 27's issue)
BAD_VALUE = 1e6
BAD_SLOT = "14:00"  # local time of the bad quarter-hour
OUTAGES = [(pd.Timestamp("2026-10-24 08:00", tz=ZONE), pd.Timestamp("2026-10-24 11:25", tz=ZONE))]
LATENCY_S, OUTAGE_LATENCY_S = 1.0, 20.0

COLUMNS = ("price", "price_exaa", "load_forecast", "load_actual")
WEEKDAY_COLUMNS = [
    f"is_{day}" for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
]
PRICE_LAGS = ["price_d-1", "price_d-2", "price_d-6"]
CALENDAR = [f"{column}_d+1" for column in WEEKDAY_COLUMNS]
BASE_PREDICTORS = [*PRICE_LAGS, *CALENDAR]


def main_predictors(ctx):
    """`main`'s list: the price lags, the calendar and the load forecast, and EXAA where `ctx.published` says it is out (live:
    known by the issue time under its rule and present now; in a backtest: known by the rule and present in the frame)."""
    names = [*BASE_PREDICTORS, "load_forecast_d+1"]
    if ctx.published("price_exaa_d+1"):
        names.append("price_exaa_d+1")
    return names


def local(day: datetime.date, hhmm: str) -> pd.Timestamp:
    """`hhmm` on the local day `day` in Berlin, as a UTC instant."""
    return pd.Timestamp(f"{day.isoformat()} {hhmm}").tz_localize(ZONE).tz_convert("UTC")


def _index() -> pd.DatetimeIndex:
    start = pd.Timestamp(FIRST_DAY.isoformat(), tz=ZONE)
    end = pd.Timestamp((LAST_DAY + datetime.timedelta(days=1)).isoformat(), tz=ZONE)
    return pd.date_range(start, end, freq="15min", inclusive="left").tz_convert("UTC")


def truth(seed: int = 2026) -> pd.DataFrame:
    """The final values of the four series (UTC index, 15 minutes), as the server holds them at the end of the week: Saturday's
    load forecast missing, Tuesday's revision and Wednesday's bad value in. `load_forecast_first` is the load forecast as first
    published (it differs from the final one on `REVISED_A65` only)."""
    index = _index()
    rng = np.random.default_rng(seed)
    wall = index.tz_convert(ZONE)
    days = wall.normalize().tz_localize(None)
    day_numbers = ((days - pd.Timestamp(FIRST_DAY)) // pd.Timedelta(days=1)).to_numpy()
    hour = wall.hour.to_numpy() + wall.minute.to_numpy() / 60
    weekday = wall.dayofweek.to_numpy()
    workday = (weekday < 5).astype(float)

    weather = np.zeros(day_numbers.max() + 1)  # a day's level of demand, as the weather moves it
    for d in range(1, len(weather)):
        weather[d] = 0.6 * weather[d - 1] + rng.normal(0, 3_000)
    load = (
        52_000
        + 9_000 * np.sin(2 * np.pi * (hour - 7) / 24)
        + 6_000 * workday
        + weather[day_numbers]
        + rng.normal(0, 400, len(index))
    )
    load_forecast = np.round(load + rng.normal(0, 900, len(index)))
    load_actual = np.round(load)
    level = np.zeros(day_numbers.max() + 1)
    for d in range(1, len(level)):
        level[d] = 0.7 * level[d - 1] + rng.normal(0, 8)
    price = np.round(
        -40
        + 0.0022 * load_forecast
        + level[day_numbers]
        + 12 * np.sin(2 * np.pi * (hour - 19) / 24)
        + rng.normal(0, 4, len(index)),
        2,
    )
    price_exaa = np.round(price + rng.normal(0, 15, len(index)), 2)  # a separate auction: close to SDAC, not equal

    frame = pd.DataFrame(
        {"price": price, "price_exaa": price_exaa, "load_forecast": load_forecast, "load_actual": load_actual},
        index=index,
    )
    local_day = pd.Series(wall.date, index=index)
    frame["load_forecast_first"] = frame["load_forecast"]
    revised = (local_day == REVISED_A65).to_numpy()
    frame.loc[revised, "load_forecast"] = np.round(frame.loc[revised, "load_forecast"] + 1_500)
    frame.loc[(local_day == MISSING_A65).to_numpy(), ["load_forecast", "load_forecast_first"]] = np.nan
    bad = local(BAD_A65, BAD_SLOT)
    frame.loc[bad, ["load_forecast", "load_forecast_first"]] = BAD_VALUE
    return frame


def schedule(frame: pd.DataFrame) -> pd.DataFrame:
    """When each cell of `truth()` appears on the server (UTC; NaT: never), and `load_forecast_revised`, when the revised value
    of the load forecast replaces the first one (NaT where there is no revision)."""
    index = frame.index
    wall = index.tz_convert(ZONE)
    day = pd.Series(wall.date, index=index)
    before = pd.Series(
        [d - datetime.timedelta(days=1) for d in day], index=index
    )  # the publishing day of a day-ahead value

    def at(hhmm: str, days: pd.Series) -> pd.Series:
        stamps = pd.to_datetime(days.astype(str) + " " + hhmm).dt.tz_localize(ZONE).dt.tz_convert("UTC")
        return pd.Series(stamps.to_numpy(), index=index)

    out = pd.DataFrame(index=index)
    out["price"] = at("12:57", before)
    out["price_exaa"] = at("11:08", before)
    out.loc[(day == LATE_EXAA).to_numpy(), "price_exaa"] = local(LATE_EXAA - datetime.timedelta(days=1), "11:37")
    out["load_forecast"] = at("10:12", before)
    out.loc[(day == MISSING_A65).to_numpy(), "load_forecast"] = pd.NaT
    out["load_actual"] = pd.Series(index + QUARTER + pd.Timedelta(hours=1), index=index)
    out["load_forecast_revised"] = pd.Series(pd.NaT, index=index, dtype=out["price"].dtype)
    out.loc[(day == REVISED_A65).to_numpy(), "load_forecast_revised"] = local(
        REVISED_A65 - datetime.timedelta(days=1), REVISED_AT
    )
    return out


# --- the fake ENTSO-E server ---------------------------------------------------------------------------------------------------

_PRICE_NS = "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"
_LOAD_NS = "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"
NO_DATA = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<Acknowledgement_MarketDocument '
    'xmlns="urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0"><mRID>1</mRID>'
    "<Reason><code>999</code><text>No matching data found for Data item X and interval Y.</text></Reason>"
    "</Acknowledgement_MarketDocument>"
)


def _stamp(ns: int) -> str:
    return pd.Timestamp(ns, tz="UTC").strftime("%Y-%m-%dT%H:%MZ")


class FakeEntsoe:
    """The ENTSO-E transparency API over `truth()` and `schedule()`, on the current clock (see the module docstring).

    `requests` lists each request as (clock now when it arrived, documentType, processType or classification position, status
    answered or "timeout"). Install it with `install(monkeypatch)`: every `requests.Session.get` of the process is its `get`."""

    def __init__(self, frame: Optional[pd.DataFrame] = None, outages=OUTAGES):
        frame = truth() if frame is None else frame
        when = schedule(frame)
        self.frame = frame
        self.outages = [(start.tz_convert("UTC"), end.tz_convert("UTC")) for start, end in outages]
        self._ns = frame.index.asi8
        self._values = {column: frame[column].to_numpy(dtype=float) for column in (*COLUMNS, "load_forecast_first")}
        self._published = {column: when[column].to_numpy(dtype="datetime64[ns]").astype(np.int64) for column in COLUMNS}
        never = np.iinfo(np.int64).max
        for column in COLUMNS:  # NaT reads as the smallest integer: never published is the largest
            self._published[column] = np.where(when[column].isna().to_numpy(), never, self._published[column])
        revised = when["load_forecast_revised"]
        self._revised = np.where(
            revised.isna().to_numpy(), never, revised.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        )
        self.requests: List[Tuple[pd.Timestamp, str, str, object]] = []

    def install(self, monkeypatch) -> "FakeEntsoe":
        fake = self

        def get(session, url, params=None, headers=None, timeout=None, **kwargs):
            return fake.get(url, params=params, headers=headers, timeout=timeout)

        monkeypatch.setattr(requests.Session, "get", get)
        return self

    # -- answering -------------------------------------------------------------------------------------------------------------

    def in_outage(self, now: pd.Timestamp) -> bool:
        return any(start <= now < end for start, end in self.outages)

    def get(self, url, params=None, headers=None, timeout=None):
        clock = current_clock()
        params = dict(params or {})
        now = clock.now()
        document = params.get("documentType")
        kind = params.get("processType") or params.get(
            "classificationSequence_AttributeInstanceComponent.position", "1"
        )
        latency = OUTAGE_LATENCY_S if self.in_outage(now) else LATENCY_S
        if timeout is not None and latency > float(timeout):
            clock.advance(float(timeout))
            self.requests.append((now, document, kind, "timeout"))
            raise requests.Timeout(f"the fake ENTSO-E server took longer than {timeout} s")
        clock.advance(latency)
        if latency == OUTAGE_LATENCY_S:
            self.requests.append((now, document, kind, 503))
            return self._response(params, "<html>Service Unavailable</html>", 503, "Service Unavailable")
        self.requests.append((now, document, kind, 200))
        start = pd.Timestamp(params["periodStart"], tz="UTC")
        end = pd.Timestamp(params["periodEnd"], tz="UTC")
        if document == "A44":
            if kind == "2":
                text = self._document(
                    "price.amount", [("price_exaa", "2")], start, end, now, "Publication_MarketDocument"
                )
            else:  # the plain request: SDAC (position 1) and, as the real API answers for DE_LU, EXAA (position 2) too
                text = self._document(
                    "price.amount",
                    [("price", "1"), ("price_exaa", "2")],
                    start,
                    end,
                    now,
                    "Publication_MarketDocument",
                )
        elif document == "A65" and kind == "A16":
            text = self._document("quantity", [("load_actual", None)], start, end, now, "GL_MarketDocument")
        elif document == "A65" and kind == "A01":
            text = self._document("quantity", [("load_forecast", None)], start, end, now, "GL_MarketDocument")
        else:
            text = None
        return self._response(params, NO_DATA if text is None else text, 200, "OK")

    def _values_at(self, column: str, rows: slice, now_ns: int) -> np.ndarray:
        values = self._values[column][rows].copy()
        published = self._published[column][rows]
        if column == "load_forecast":
            first = self._values["load_forecast_first"][rows]
            values = np.where(self._revised[rows] <= now_ns, values, first)
        values[published > now_ns] = np.nan
        return values

    def _document(self, tag: str, series, start, end, now, root) -> Optional[str]:
        lo, hi = np.searchsorted(self._ns, [start.value, end.value])
        rows = slice(lo, hi)
        parts = []
        for column, sequence in series:
            values = self._values_at(column, rows, now.value)
            have = np.flatnonzero(~np.isnan(values))
            if not have.size:
                continue
            ns = self._ns[rows]
            first, last = ns[have[0]], ns[have[-1]]
            positions = (ns[have] - first) // QUARTER.value + 1
            points = "".join(
                f"<Point><position>{p}</position><{tag}>{float(v)!r}</{tag}></Point>"
                for p, v in zip(positions.tolist(), values[have].tolist())
            )
            head = "<TimeSeries><mRID>1</mRID>"
            if sequence is not None:
                head += (
                    "<classificationSequence_AttributeInstanceComponent.position>"
                    f"{sequence}</classificationSequence_AttributeInstanceComponent.position>"
                )
            parts.append(
                f"{head}<curveType>A01</curveType><Period><timeInterval><start>{_stamp(first)}</start>"
                f"<end>{_stamp(last + QUARTER.value)}</end></timeInterval><resolution>PT15M</resolution>{points}"
                "</Period></TimeSeries>"
            )
        if not parts:
            return None
        ns = _PRICE_NS if root == "Publication_MarketDocument" else _LOAD_NS
        return f'<?xml version="1.0" encoding="utf-8"?>\n<{root} xmlns="{ns}"><mRID>x</mRID>{"".join(parts)}</{root}>'

    @staticmethod
    def _response(params, text: str, status: int, reason: str) -> requests.Response:
        response = requests.Response()
        response.status_code = status
        response._content = text.encode()
        response.headers.update({"Content-Type": "text/xml"})
        response.url = "https://web-api.tp.entsoe.eu/api"
        response.reason = reason
        response.encoding = "utf-8"  # as the real API declares it: no charset guessing over every answer
        return response


class QuietProgress:
    """Stands in for `rich.progress.Progress` in `EntsoeSource.fetch`: the progress bar's display thread, started and stopped
    on each of the week's ~150 refreshes, is time spent on nothing the week checks."""

    def __init__(self, *columns, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def add_task(self, *args, **kwargs):
        return 0

    def update(self, *args, **kwargs):
        pass

    def advance(self, *args, **kwargs):
        pass


class ParquetReads:
    """`pandas.read_parquet` remembering what it read of an unchanged file (same inode, size and modification time), a copy
    each time: every refresh of a poll reads every piece of both sources' caches again (about a fifth of the week's time), and
    a cache piece is written once, atomically, under a new name or a new inode. Other reads pass through."""

    def __init__(self, read):
        self.read = read
        self.kept = {}

    def __call__(self, path, *args, **kwargs):
        try:
            stat = os.stat(path)
        except (TypeError, OSError):
            return self.read(path, *args, **kwargs)
        if args or kwargs:
            return self.read(path, *args, **kwargs)
        key = (os.fspath(path), stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if key not in self.kept:
            self.kept[key] = self.read(path)
        return self.kept[key].copy()


def archive(fake: FakeEntsoe, at: pd.Timestamp) -> pd.DataFrame:
    """What the server holds at `at`, column by column (NaN where nothing is published): the truth as a reader at `at` sees it."""
    out = {}
    for column in COLUMNS:
        out[column] = fake._values_at(column, slice(None), at.value)
    return pd.DataFrame(out, index=fake.frame.index)


__all__ = [
    "BAD_A65",
    "BASE_PREDICTORS",
    "FakeEntsoe",
    "KEY",
    "WEEK",
    "ZONE",
    "archive",
    "local",
    "main_predictors",
    "schedule",
    "truth",
]
