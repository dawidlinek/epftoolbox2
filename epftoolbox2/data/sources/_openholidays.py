"""Fetch and parse school holiday periods from the OpenHolidays API.

Data: OpenHolidays API (https://www.openholidaysapi.org), made available under the Open Database License
(ODbL 1.0, https://opendatacommons.org/licenses/odbl/1-0/). Publications and derived databases must credit it.
"""

import time
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

ATTRIBUTION = "School holidays: OpenHolidays API (https://www.openholidaysapi.org), ODbL 1.0"

API_URL = "https://openholidaysapi.org"

# The API rejects requests spanning more than 1095 days.
MAX_DAYS_PER_REQUEST = 1000
# Every school year has holidays, so a stretch of more than a year without any is missing data, not a year of
# school. The query reaches further than that beyond the requested range, so each requested date can be judged.
MAX_GAP_DAYS = 366
QUERY_MARGIN_DAYS = 400


def get(session: requests.Session, path: str, params: dict) -> list:
    """GET one OpenHolidays endpoint, with retries on a server error or a network failure."""
    error: Exception = RuntimeError("no attempt made")
    for attempt in range(3):
        try:
            response = session.get(f"{API_URL}/{path}", params=params, headers={"accept": "text/json"}, timeout=30)
        except requests.RequestException as e:
            error = e
        else:
            if response.status_code < 500:
                response.raise_for_status()
                return response.json()
            error = requests.HTTPError(f"HTTP {response.status_code}")
        if attempt < 2:
            time.sleep(2**attempt)
    raise RuntimeError(f"OpenHolidays API request to {path} failed after 3 attempts: {error}") from error


def top_level_subdivisions(session: requests.Session, country: str) -> List[str]:
    """`country`'s own entries from the Subdivisions endpoint, not the nested `children` of each one (e.g.
    Poland's 16 voivodeships, not any finer level some countries report)."""
    return sorted({item["code"].upper() for item in get(session, "Subdivisions", {"countryIsoCode": country})})


def fetch_entries(session: requests.Session, country: str, first: pd.Timestamp, last: pd.Timestamp) -> list:
    """School holiday entries for `country` covering [first, last], in requests of at most 1000 days each."""
    entries = {}
    chunk_start = first
    while chunk_start <= last:
        chunk_end = min(chunk_start + pd.Timedelta(days=MAX_DAYS_PER_REQUEST - 1), last)
        params = {
            "countryIsoCode": country,
            "languageIsoCode": "EN",
            "validFrom": chunk_start.date().isoformat(),
            "validTo": chunk_end.date().isoformat(),
        }
        for entry in get(session, "SchoolHolidays", params):
            # Holidays that span a chunk boundary are returned twice.
            key = entry.get("id") or (entry["startDate"], entry["endDate"], str(entry.get("subdivisions")))
            entries[key] = entry
        chunk_start = chunk_end + pd.Timedelta(days=1)
    return list(entries.values())


def applies(entry: dict, target: Optional[str]) -> bool:
    codes = [s["code"].upper() for s in entry.get("subdivisions") or []]
    if entry.get("nationwide") or not codes:
        return True
    if target is None:
        return entry.get("regionalScope") == "National"
    # A holiday of a subdivision also applies to its children (e.g. DE-BY to DE-BY-AU), not the other way round.
    return any(target == code or target.startswith(f"{code}-") for code in codes)


def daily_flags(entries: list, target: Optional[str], days: pd.DatetimeIndex) -> np.ndarray:
    """1/0 per day in `days` (consecutive local dates), NaN where the data has a gap of more than a year."""
    intervals = sorted(
        (pd.Timestamp(e["startDate"]), pd.Timestamp(e["endDate"]))
        for e in entries
        if applies(e, target) and e.get("temporalScope", "FullDay") != "HalfDay"
    )
    flags = np.zeros(len(days))
    covered = np.ones(len(days), dtype=bool)
    previous_end = days[0] - pd.Timedelta(days=1)
    for start, end in intervals + [(days[-1] + pd.Timedelta(days=1), None)]:
        if (start - previous_end).days > MAX_GAP_DAYS:
            covered[(days > previous_end) & (days < start)] = False
        if end is not None:
            flags[days.searchsorted(start, "left") : days.searchsorted(end, "right")] = 1.0
            previous_end = max(previous_end, end)
    flags[~covered] = np.nan
    return flags
