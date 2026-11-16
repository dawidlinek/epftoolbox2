"""Monitoring a live directory (step 8h of the forecasting layer): how the forecasts it holds did against what happened, on the
same keys, with skill first.

`ModelPipeline.monitor(data, save_dir)` reads the records of a live directory (`results.live`) under a revision policy, takes each
key's actual from `data`, and scores every member and window through `LossPanel` (`evaluators._compare`): the models are compared
on forecast keys they share, with counts. Raw error is not the headline: a month in which prices swing wider raises every model's
RMSE and the benchmark's more (the live DE price RMSE rose 41 % from August to September while skill improved), so the tables lead
with skill, the ratio of a member's MAE to a benchmark's on the keys both hold.

**Definitions.** A *day* is one issue day at one horizon (what `LossPanel.day_ids` numbers): the delivered curve of one target
date, of 23, 24 or 25 hourly forecasts (92 to 100 quarter-hours) on a switch day, since the clock's periods are the records'. The
error of a key is `prediction - actual`. Per member and window, on the keys the member has a forecast and a known actual for:

- `forecasts`, `mae`, `rmse` (pooled), `rmse_daily` (the mean of the days' RMSEs, the arena's convention), `bias` (the mean
  error);
- `level`, the mean over days of |the day's mean error|, and `shape`, the MAE after removing each day's mean error. Per day
  `mean|e| <= |mean e| + mean|e - mean e|`, so `mae <= level + shape` when the days weigh the same; a day's level is what the
  forecast got wrong about the whole curve (a shared input or weather problem), its shape what it got wrong inside the day;
- per benchmark `b`: `skill_b` = member MAE / benchmark MAE on the keys both have (below 1: better than the benchmark),
  `common_b` (how many keys), `days_b` and `wins_b` (the days among them where the member's MAE is the lower);
- `participation`: the share of the window's issue days with a revision committed before the deadline (the cut-off), whatever
  the policy; `delivered`: the share with a delivery recorded; `member_days`: the issue days the member has a forecast on;
- for a `FallbackEnsemble`: `fallback_days`, the days a rung above 0 served, and `fallback_mae` on them;
- `quality_days`: the issue days whose `quality.json` (step 8f) records a missing cell, an imputed cell or a source that didn't
  refresh, of `quality_records` days that have one;
- with `backtest=`: per benchmark `backtest_skill_b`, and `live_vs_backtest_b`, the live skill over the backtest's skill, both on
  the target days of the window where the live and the backtest forecasts and the benchmark's all exist (`matched_days_b`).

**Windows** are the last `n` calendar issue days up to the newest *due* one: an issue day is due once its deadline has passed on the
clock (the day before today without a deadline), and none before the directory's first call counts. A key is scored once its
target's rule says the actual is known by now on the clock (`InformationSet`), whatever `data` holds, and its actual is the
data's value at the same wall-clock period `horizon` local days after the issue day (the one a run's `actual` is), so a
repeated autumn hour that no run could score isn't scored here either.

**Alerts** (`MonitorReport.alerts`, an empty list when nothing fires) come from a list of mappings, each with one rule key:

    - skill_above: 0.8            # member MAE / benchmark MAE above 0.8 on each of the last for_days scored days
      for_days: 3                 # default 1
      member: lear                # optional: all members but the benchmarks
      benchmark: naive            # optional: every benchmark
    - participation_below: 0.9    # share of the issue days of `window` with a committed revision
      window: 30                  # default: the first window
    - delivered_below: 0.9        # likewise, with a recorded delivery
    - rung_at_least: 1            # a fallback ensemble's day served by this rung or lower, on each of the last for_days days
      for_days: 1
      member: best                # optional: every fallback ensemble
    - stale_input: load_forecast  # the column's last valid timestamp, in each of the last for_days quality records, is
      age: 1                      # more than `age` days (default 1) before the issue day's midnight (or none)
      for_days: 1

A key the rule doesn't know raises, so a typo never silently disables an alert.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._clock import current_clock
from .._date_utils import shift_calendar_days
from ..evaluators._compare import LossPanel
from ..models._contract import period_index
from .live import POLICIES, LiveStore, _warn_dropped, local_instant
from .checks import CHECK_FAILED
from .store import OK_ONLY, LiveDirectoryError, read_records

# What a record is read for.
_COLUMNS = ["run_date", "target_date", "hour", "horizon", "prediction", "freq", "revision", "rung", "member"]
RUNG_CHANNEL = "\x00rung"  # a panel channel that carries each key's rung, as a "model" whose error is the rung
QUALITY_FILE = "quality.json"

ALERT_RULES = ("skill_above", "participation_below", "delivered_below", "rung_at_least", "stale_input")
_ALERT_OPTIONS = {
    "skill_above": ("for_days", "member", "benchmark"),
    "participation_below": ("window",),
    "delivered_below": ("window",),
    "rung_at_least": ("for_days", "member"),
    "stale_input": ("age", "for_days"),
}


@dataclass(frozen=True)
class Alert:
    """A rule of `monitor(alerts=...)` that fired: `rule`, a one-line `message`, the `member` and `benchmark` it concerns (None
    for none), the `value` observed, the `threshold` of the rule and `through`, the last issue day it looked at."""

    rule: str
    message: str
    member: Optional[str] = None
    benchmark: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    through: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def load_alerts(alerts: Any) -> List[Dict[str, Any]]:
    """`alerts` as a list of validated rule mappings: None (no alerts), a list of mappings, or the path of a YAML file (or its
    text) holding one. Each mapping names exactly one rule (`ALERT_RULES`) and the options that rule takes."""
    if alerts is None:
        return []
    if isinstance(alerts, (str, Path)):
        import yaml

        text = str(alerts)
        if isinstance(alerts, Path) or "\n" not in text:  # a name; a list of rules is YAML text of several lines
            try:
                if Path(alerts).is_file():
                    text = Path(alerts).read_text(encoding="utf-8")
            except OSError:  # a flow-style list on one line can be longer than a file name may be
                pass
        alerts = yaml.safe_load(text)
        if alerts is None:
            return []
    if isinstance(alerts, Mapping):
        alerts = [alerts]
    if not isinstance(alerts, (list, tuple)):
        raise TypeError(f"alerts is a list of mappings (or a YAML file holding one), got {type(alerts).__name__}")
    out = []
    for entry in alerts:
        if not isinstance(entry, Mapping):
            raise TypeError(f"an alert is a mapping such as {{'skill_above': 0.8, 'for_days': 3}}, got {entry!r}")
        rules = [key for key in entry if key in ALERT_RULES]
        if len(rules) != 1:
            raise ValueError(f"an alert names exactly one rule of {list(ALERT_RULES)}, got {dict(entry)!r}")
        rule = rules[0]
        unknown = [key for key in entry if key != rule and key not in _ALERT_OPTIONS[rule]]
        if unknown:
            raise ValueError(f"the alert {rule!r} takes {list(_ALERT_OPTIONS[rule])}, not {unknown}")
        spec = dict(entry)
        for count in ("for_days", "window"):
            if count in spec and (isinstance(spec[count], bool) or not isinstance(spec[count], int) or spec[count] < 1):
                raise ValueError(f"alert {rule!r}: {count} is a whole number of days from 1, got {spec[count]!r}")
        if rule == "stale_input":
            if not isinstance(spec[rule], str):
                raise ValueError(f"alert 'stale_input' names the column, got {spec[rule]!r}")
        elif isinstance(spec[rule], bool) or not isinstance(spec[rule], (int, float)) or not math.isfinite(spec[rule]):
            raise ValueError(f"alert {rule!r} takes a number, got {spec[rule]!r}")
        out.append(spec)
    return out


@dataclass
class MonitorMember:
    """One member of the pipeline with results in the live directory."""

    name: str
    path: Path
    ladder: Optional[Tuple[str, ...]] = None


class _Actuals:
    """The actual of a key and whether the target's rule says it is known by now, from the frame `data`.

    A key's actual is what a run's `actual` is: the value of `{target}_d+h` on the issue row, the last row of the issue day at the
    period, which is the data's value at the same wall-clock time `h` local calendar days later (`shift_calendar_days`, where a
    target landing on a repeated autumn hour from a day that has it once has none, as in the run)."""

    def __init__(self, data: pd.DataFrame, target: str, freq: str, information_set: Any, now: pd.Timestamp) -> None:
        if target not in data.columns:
            raise KeyError(f"monitor(target={target!r}): the data has no such column")
        series = data[target].sort_index()
        index = series.index
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError("monitor needs data with a DatetimeIndex")
        if not index.is_unique:
            raise ValueError("monitor: the data's index holds an instant twice")
        wall = index.tz_localize(None) if index.tz is not None else index
        rows = pd.DataFrame(
            {"date": wall.normalize().as_unit("ns"), "period": period_index(index, freq), "pos": np.arange(len(index))}
        )
        self.issue_rows = rows.groupby(["date", "period"])["pos"].last()
        self.series = series
        self.target = target
        self.freq = freq
        self.info = information_set
        self.now = now

    def lookup(self, keys: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """(actual, known) of every row of `keys` (`run_date`, `hour`, `horizon`): NaN and False where the data has no such row
        or value, or the target's rule puts its publication after now."""
        n = len(keys)
        actual = np.full(n, np.nan)
        known = np.zeros(n, dtype=bool)
        if not n:
            return actual, known
        index = self.series.index
        values = self.series.to_numpy(dtype=float)
        run_dates = pd.DatetimeIndex(keys["run_date"]).as_unit("ns")
        horizons = keys["horizon"].to_numpy()
        hours = keys["hour"].to_numpy()
        places, targets = [], []
        for horizon in np.unique(horizons):
            sel = np.flatnonzero(horizons == horizon)
            mi = pd.MultiIndex.from_arrays([run_dates[sel], hours[sel]], names=["date", "period"])
            pos = self.issue_rows.reindex(mi).to_numpy(dtype=float)
            have = np.flatnonzero(~np.isnan(pos))
            if len(have):
                places.append(sel[have])
                targets.append(shift_calendar_days(index[pos[have].astype(np.int64)], int(horizon), ambiguous="NaT"))
        if not places:
            return actual, known
        rows = np.concatenate(places)
        instants = targets[0].append(targets[1:]) if len(targets) > 1 else targets[0]
        found = index.get_indexer(instants)  # -1 for a NaT and for an instant the data lacks
        value = np.where(found >= 0, values[np.maximum(found, 0)], np.nan)
        usable = np.isfinite(value)
        if not usable.any():
            return actual, known
        rows, instants, value = rows[usable], instants[usable], value[usable]
        actual[rows] = value
        known[rows] = self._out(instants, run_dates[rows].min())
        return actual, known

    def _out(self, instants: pd.DatetimeIndex, first_day: pd.Timestamp) -> np.ndarray:
        """Whether the value at each of `instants` is out by now under the target's rule."""
        from ..information_set import StepsReader

        if self.info is None:
            raise ValueError(
                "monitor() needs the pipeline's information set to tell when an actual is known by now: pass issue_time= or "
                f"information_set= to ModelPipeline, with a rule for '{self.target}'"
            )
        # Each instant once and in time order, whatever batch it came in: a daily frame's rows are read as daily only where no
        # two share a date (`information_set._daily_rows`), so two horizons with one target must not repeat it.
        unique = instants.unique().sort_values()
        reader = StepsReader(self.info, unique, [str(first_day.date())], self.freq)
        minutes = reader.available(self.target, np.arange(len(unique)), np.zeros(len(unique), dtype=np.int64))
        out = np.asarray(minutes, dtype=float) <= (self.now - reader.origin) / pd.Timedelta(minutes=1)
        return out[unique.get_indexer(instants)]


def data_in_results_zone(members: Sequence[MonitorMember], data: pd.DataFrame) -> pd.DataFrame:
    """`data` in the time zone the members' results were made in (`data_timezone` of their manifests), which is what a record's
    date and period are read in: a frame in another zone (UTC, say) would put every hour on the wrong period and, near midnight,
    the wrong date, and score other keys without a word. The same instants are converted (`tz_convert`), so the frame is never
    refused for its zone alone; members whose manifests name different zones raise, and so does a naive frame for results made
    in a zone (its instants are unknown). Without a zone in any manifest (a directory written by hand) the frame is used as it is."""
    from .manifest import manifest_path, read_zone

    zones = {zone for zone in (read_zone(manifest_path(m.path)) for m in members) if zone is not None}
    if len(zones) > 1:
        raise ValueError(f"the members' results were made in different time zones: {sorted(zones)}")
    if not zones:
        return data
    (zone,) = zones
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.tz is None:
        raise ValueError(
            f"the results were made on data in {zone}, and the frame has a naive index, whose instants are unknown: localize it "
            f"(data.tz_localize('{zone}')) first"
        )
    return data if str(data.index.tz) == zone else data.tz_convert(zone)


def _read_keys(
    member: MonitorMember, committed: Mapping[str, Mapping[int, Any]], statuses: Optional[Iterable[str]] = OK_ONLY
) -> pd.DataFrame:
    """The forecasts of one member among the revisions `committed` lets through (`LiveStore.visible`'s shape; one per key, the
    highest of the day's revisions with a record): `run_date`, `target_date`, `hour`, `horizon`, `prediction`, `revision`, `rung`
    (NaN where the member has none). `statuses` other than ok read the keys whose chosen record has one (`check_failed`)."""
    columns = ["run_date", "target_date", "hour", "horizon", "prediction", "freq", "revision", "rung"]
    rows = [
        {c: r.get(c) for c in columns}
        for r in read_records(member.path, statuses, lambda record: record, columns=columns, committed=committed)
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if not len(frame):
        frame["freq"] = frame["freq"].astype(object)
    if statuses is OK_ONLY:
        frame = frame[frame["prediction"].notna()].reset_index(drop=True)
    frame["run_date_text"] = frame["run_date"].astype(str)
    frame["run_date"] = pd.to_datetime(frame["run_date"]).astype("datetime64[ns]")
    frame["target_date"] = pd.to_datetime(frame["target_date"]).astype("datetime64[ns]")
    for name in ("hour", "horizon"):
        frame[name] = frame[name].astype(np.int64)
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce").astype(float)
    frame["rung"] = pd.to_numeric(frame["rung"], errors="coerce").astype(float)
    return frame


def _delivered_view(latest: Mapping[str, Mapping[int, Any]], store: LiveStore) -> Dict[str, Dict[int, Any]]:
    """What `policy="delivered"` reads here: of each issue day's revisions committed by the cut-off (`latest`), those up to the newest
    one recorded as delivered, **whenever its delivery was recorded**. A delivery happens between the forecast deadline and the
    market's gate, or is marked later from a receipt, so the time of the line doesn't matter: the revision had to be committed by
    the deadline. (`LiveStore.visible("delivered", cutoff)`, the as-of view live readers need, also asks the delivery to be
    recorded by the cut-off.)"""
    heads: Dict[str, int] = {}
    for line in store.deliveries():
        day, revision = str(line.get("issue")), line.get("revision")
        if revision in latest.get(day, {}):
            heads[day] = revision
    return {
        day: {revision: commit for revision, commit in latest[day].items() if revision <= head}
        for day, head in heads.items()
    }


def _signed(frame: pd.DataFrame, actual: np.ndarray, known: np.ndarray) -> pd.DataFrame:
    """The frame `LossPanel` takes for the scored keys of one member: its prediction is the signed error and its actual 0, so
    `panel.errors` are the absolute errors and `panel.predictions` the signed ones."""
    scored = known & np.isfinite(frame["prediction"].to_numpy())
    kept = frame[scored]
    return pd.DataFrame(
        {
            "run_date": kept["run_date"].to_numpy(),
            "target_date": kept["target_date"].to_numpy(),
            "hour": kept["hour"].to_numpy(),
            "horizon": kept["horizon"].to_numpy(),
            "prediction": kept["prediction"].to_numpy() - actual[scored],
            "actual": np.zeros(int(scored.sum())),
        }
    )


def _rung_frame(frame: pd.DataFrame, known: np.ndarray) -> pd.DataFrame:
    """The rung of each scored key as a channel of the panel: an error that is the rung."""
    kept = frame[known & frame["rung"].notna().to_numpy()]
    return pd.DataFrame(
        {
            "run_date": kept["run_date"].to_numpy(),
            "target_date": kept["target_date"].to_numpy(),
            "hour": kept["hour"].to_numpy(),
            "horizon": kept["horizon"].to_numpy(),
            "prediction": kept["rung"].to_numpy(),
            "actual": np.zeros(len(kept)),
        }
    )


def _panel(
    frames: Mapping[str, pd.DataFrame], rungs: Mapping[str, pd.DataFrame], freq: str
) -> Tuple[LossPanel, List[str]]:
    """A `LossPanel` of the members' signed frames and, for each with a rung, its rung channel."""
    every = dict(frames)
    for name, frame in rungs.items():
        every[name + RUNG_CHANNEL] = frame
    whole = (1,) if freq == "1D" else (23, 24, 25, 92, 96, 100)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # actuals agree by construction, so this never hides one
        return LossPanel(every, whole_day_periods=whole), list(frames)


def _day_numbers(days: Iterable[str]) -> np.ndarray:
    return np.array([(pd.Timestamp(d) - pd.Timestamp("1970-01-01")).days for d in days], dtype=np.int64)


def _per_day(day_ids: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(the distinct day ids, each key's place among them, each day's count)."""
    ids, inverse = np.unique(day_ids, return_inverse=True)
    inverse = inverse.ravel()
    return ids, inverse, np.bincount(inverse, minlength=len(ids))


def _stats(panel: LossPanel, rows: np.ndarray, name: str) -> Dict[str, float]:
    """The absolute metrics of one member on the keys of `rows` it holds a scored forecast of."""
    j = panel.column(name)
    absolute = panel.errors[rows, j]
    keep = np.isfinite(absolute)
    rows, absolute = rows[keep], absolute[keep]
    n = len(rows)
    if not n:
        return dict(
            forecasts=0, mae=math.nan, rmse=math.nan, rmse_daily=math.nan, bias=math.nan, level=math.nan, shape=math.nan
        )
    signed = panel.predictions[rows, j]
    ids, inverse, count = _per_day(panel.day_ids(rows), signed)
    mean_signed = np.bincount(inverse, weights=signed, minlength=len(ids)) / count
    mean_square = np.bincount(inverse, weights=absolute**2, minlength=len(ids)) / count
    return dict(
        forecasts=n,
        mae=float(absolute.mean()),
        rmse=float(np.sqrt((absolute**2).mean())),
        rmse_daily=float(np.sqrt(mean_square).mean()),
        bias=float(signed.mean()),
        level=float(np.abs(mean_signed).mean()),
        shape=float(np.abs(signed - mean_signed[inverse]).mean()),
    )


def _skill(panel: LossPanel, rows: np.ndarray, member: str, benchmark: str) -> Dict[str, float]:
    """`member`'s MAE over `benchmark`'s on the keys of `rows` both hold, how many, and on how many days the member's day MAE is
    the lower."""
    common = panel.common(rows, [member, benchmark])
    n = len(common)
    if not n:
        return dict(skill=math.nan, common=0, days=0, wins=0)
    mine = panel.errors[common, panel.column(member)]
    theirs = panel.errors[common, panel.column(benchmark)]
    ids, inverse, count = _per_day(panel.day_ids(common), mine)
    day_mine = np.bincount(inverse, weights=mine, minlength=len(ids)) / count
    day_theirs = np.bincount(inverse, weights=theirs, minlength=len(ids)) / count
    bench = float(theirs.mean())
    return dict(
        skill=float(mine.mean()) / bench if bench > 0 else math.nan,
        common=n,
        days=len(ids),
        wins=int((day_mine < day_theirs).sum()),
    )


def _daily(panel: LossPanel, members: Sequence[str], benchmarks: Sequence[str]) -> pd.DataFrame:
    """One row per (member, day) of every scored key: its run date, horizon and target date, forecasts, MAE, RMSE, bias, and the
    day's skill against each benchmark on the keys both hold (`skill_<benchmark>`, NaN without any or a zero benchmark MAE)."""
    rows = panel.rows()
    frames = []
    for name in members:
        j = panel.column(name)
        mine = rows[np.isfinite(panel.errors[rows, j])]
        if not len(mine):
            continue
        ids, inverse, count = _per_day(panel.day_ids(mine), mine)
        absolute = panel.errors[mine, j]
        signed = panel.predictions[mine, j]
        first = np.zeros(len(ids), dtype=np.int64)
        first[inverse[::-1]] = np.arange(len(mine))[::-1]  # the first key of each day
        run_day = panel.run_days(mine)[first]
        horizon = panel.horizon[mine][first]
        frame = pd.DataFrame(
            {
                "member": name,
                "run_date": pd.to_datetime(run_day, unit="D"),
                "horizon": horizon,
                "target_date": pd.to_datetime(run_day + horizon, unit="D"),
                "forecasts": count,
                "mae": np.bincount(inverse, weights=absolute, minlength=len(ids)) / count,
                "rmse": np.sqrt(np.bincount(inverse, weights=absolute**2, minlength=len(ids)) / count),
                "bias": np.bincount(inverse, weights=signed, minlength=len(ids)) / count,
            }
        )
        for bench in benchmarks:
            col = f"skill_{bench}"
            frame[col] = math.nan
            if bench == name:
                continue
            both = panel.common(mine, [name, bench])
            if not len(both):
                continue
            b_ids, b_inverse, b_count = _per_day(panel.day_ids(both), both)
            ratio = np.full(len(ids), math.nan)
            m_sum = np.bincount(b_inverse, weights=panel.errors[both, j], minlength=len(b_ids))
            b_sum = np.bincount(b_inverse, weights=panel.errors[both, panel.column(bench)], minlength=len(b_ids))
            where = np.searchsorted(ids, b_ids)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio[where] = np.where(b_sum > 0, m_sum / b_sum, math.nan)
            frame[col] = ratio
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _calendar(
    store: LiveStore, now: pd.Timestamp, issue_days: Optional[Sequence[str]] = None
) -> Tuple[List[str], Optional[str]]:
    """(the issue days from the directory's first call to the newest due one, that one): an issue day is due once its deadline has
    passed on the clock, or once the day has ended without a deadline on file (today never counts then). Only the weekdays the
    pipeline issues on are issue days (`issue_days`, else `live.json`'s, else every day), so a pipeline that issues on Fridays
    alone isn't missing the other six."""
    from ..models._calendar import listed_days, resolve_issue_days

    first = min((str(line["issue"]) for line in store.calls() if line.get("issue")), default=None)
    if first is None:
        return [], None
    identity = store.identity()
    zone, deadline = identity.get("issue_timezone"), identity.get("deadline")
    if not zone:
        raise ValueError(
            f"{store.identity_path} records no issue_timezone: monitor() can't tell which issue days are due"
        )
    weekdays = resolve_issue_days(issue_days if issue_days is not None else identity.get("issue_days"))
    today = now.tz_convert(zone).date()
    candidate = pd.Timestamp(today)
    if deadline is None or local_instant(str(today), deadline, zone) > now:
        candidate -= pd.Timedelta(days=1)
    last = candidate.date().isoformat()
    if last < first:
        return [], None
    days = listed_days([d.date().isoformat() for d in pd.date_range(first, last, freq="D")], weekdays)
    return days, (days[-1] if days else None)


def _quality(
    store: LiveStore, revisions: Mapping[str, Mapping[int, Any]], days: Sequence[str]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Issue day -> its `quality.json` (of the highest of the day's `revisions` the policy reads), None for a day without a
    committed revision or a record."""
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for day in days:
        shown = revisions.get(day)
        if not shown:
            continue
        path = store.revision_dir(day, max(shown)) / QUALITY_FILE
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = None
        out[day] = record if isinstance(record, dict) else None
    return out


def _quality_row(day: str, revision: int, record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    row: Dict[str, Any] = dict(
        issue=day,
        revision=revision,
        record=record is not None,
        missing_cells=0,
        imputed_cells=0,
        gappy_columns="",
        imputed_columns="",
        failed_sources="",
        problem=False,
    )
    if record is None:
        return row
    columns = record.get("columns") or {}
    sources = record.get("sources") or {}
    gappy = [c for c, v in columns.items() if (v or {}).get("missing")]
    imputed = [c for c, v in columns.items() if (v or {}).get("imputed")]
    failed = [s for s, v in sources.items() if (v or {}).get("status") not in (None, "ok")]
    row.update(
        missing_cells=int(sum((v or {}).get("missing") or 0 for v in columns.values())),
        imputed_cells=int(sum((v or {}).get("imputed") or 0 for v in columns.values())),
        gappy_columns=", ".join(sorted(gappy)),
        imputed_columns=", ".join(sorted(imputed)),
        failed_sources=", ".join(sorted(failed)),
        problem=bool(gappy or imputed or failed),
    )
    return row


@dataclass
class MonitorReport:
    """How the forecasts of a live directory did (`ModelPipeline.monitor`; see the module docstring for every definition).

    `tables()` is one row per (window, member), skill first; `by_window()` the same cut by window; `daily()` one row per member
    and day; `rungs()` what each rung of a fallback ensemble served under each policy; `quality()` the input quality of each issue
    day; `alerts` the rules that fired (`Alert`). `through` is the newest due issue day, `now` the clock's instant the report was
    made at, and `missing` the members of the pipeline with no results in the directory."""

    windows: Tuple[int, ...]
    policy: str
    cutoff: Any
    benchmarks: Tuple[str, ...]
    members: Tuple[str, ...]
    missing: Tuple[str, ...]
    through: Optional[str]
    now: pd.Timestamp
    alerts: List[Alert] = field(default_factory=list)
    _tables: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    _daily: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    _rungs: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    _quality: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def tables(self) -> pd.DataFrame:
        """One row per (window, member): `window` (the number of issue days), `member`, then per benchmark `skill_<b>` (the
        member's MAE over the benchmark's on the keys both hold: below 1 beats it), `common_<b>`, `days_<b>`, `wins_<b>` (days
        the member's MAE is the lower), and with a backtest `backtest_skill_<b>`, `live_vs_backtest_<b>`, `matched_days_<b>`; then
        `forecasts`, `unscored` (forecasts whose actual isn't known yet), `mae`, `rmse`, `rmse_daily`, `bias`, `level`, `shape`,
        `issue_days` (due in the window), `committed_days`, `participation`, `delivered_days`, `delivered`, `member_days`,
        `fallback_days`, `fallback_mae`, `quality_records` and `quality_days`."""
        return self._tables.copy()

    def by_window(self) -> Dict[int, pd.DataFrame]:
        """`tables()` cut by window: window -> its rows without the `window` column."""
        return {
            int(window): frame.drop(columns="window").reset_index(drop=True)
            for window, frame in self._tables.groupby("window", sort=True)
        }

    def daily(self) -> pd.DataFrame:
        """One row per (member, issue day, horizon) with a scored forecast, in calendar order: `member`, `run_date`, `horizon`,
        `target_date`, `forecasts`, `mae`, `rmse`, `bias` and `skill_<b>` per benchmark."""
        return self._daily.copy()

    def rungs(self) -> pd.DataFrame:
        """What each rung of each `FallbackEnsemble` served, per policy and window: `policy`, `window`, `member`, `rung`,
        `rung_member` (the ladder's member at it), `days` and `forecasts` it served, `share` of the member's days, `scored` (of
        those forecasts, with a known actual) and `mae` on them. Empty without a fallback ensemble."""
        return self._rungs.copy()

    def quality(self) -> pd.DataFrame:
        """One row per issue day with a committed revision (of the policy's revision): `issue`, `revision`, `record` (whether it
        has a `quality.json`), `missing_cells`, `imputed_cells`, `gappy_columns`, `imputed_columns`, `failed_sources`, `problem`."""
        return self._quality.copy()


def _window_days(days: Sequence[str], through: Optional[str], window: int) -> List[str]:
    """The issue days of `days` in the window of `window` days ending at `through`."""
    if through is None:
        return []
    first = (pd.Timestamp(through) - pd.Timedelta(days=window - 1)).date().isoformat()
    return [d for d in days if d >= first]


def _frequency(frame: pd.DataFrame) -> str:
    known = frame["freq"].dropna()
    return str(known.iloc[0]) if len(known) else "1h"


def _served_days(frame: pd.DataFrame, through: Optional[str]) -> pd.DataFrame:
    """A fallback ensemble's days (issue day, horizon) up to the newest due issue day, with the highest rung that served a key of
    each, from every key read, scored or not: a fallback is known the day it is issued."""
    columns = ["run_date_text", "run_date", "horizon", "rung"]
    if not len(frame) or through is None:
        return pd.DataFrame(columns=columns)
    frame = frame[frame["run_date_text"] <= through]
    served = frame.groupby(["run_date_text", "horizon"], sort=True).agg(
        run_date=("run_date", "first"), rung=("rung", "max")
    )
    return served.reset_index()[columns]


def _fallback(
    panel: LossPanel,
    rows: np.ndarray,
    name: str,
    served: Optional[pd.DataFrame],
    in_window: Sequence[str],
    raw: pd.DataFrame,
) -> Tuple[int, float]:
    """(the days of the window a rung above 0 served, the MAE of the keys it served that have a known actual)."""
    if served is None:
        return 0, math.nan
    days = served[served["run_date_text"].isin(in_window) & (served["rung"] > 0)]
    channel = panel.column(name + RUNG_CHANNEL)
    own = panel.errors[rows, panel.column(name)]
    rung = panel.errors[rows, channel]
    chosen = np.isfinite(own) & np.isfinite(rung) & (rung > 0)
    return len(days), float(own[chosen].mean()) if chosen.any() else math.nan


def _backtest_panel(backtest: Any, names: Sequence[str], freq: str) -> LossPanel:
    """The backtest's forecasts of the members as a panel of signed errors (`_signed`): a report (`EvaluationReport`, anything with
    `_model_frames()`) or a mapping of member -> frame with the forecast keys, `prediction` and `actual`."""
    frames = backtest._model_frames() if hasattr(backtest, "_model_frames") else backtest
    if not isinstance(frames, Mapping):
        raise TypeError("backtest is an EvaluationReport or a mapping of member -> DataFrame of forecasts")
    out = {}
    for name in names:
        frame = frames.get(name)
        if frame is None or not len(frame):
            continue
        out[name] = pd.DataFrame(
            {
                "run_date": pd.to_datetime(frame["run_date"]).astype("datetime64[ns]").to_numpy(),
                "target_date": pd.to_datetime(frame["target_date"]).astype("datetime64[ns]").to_numpy(),
                "hour": np.asarray(frame["hour"], dtype=np.int64),
                "horizon": np.asarray(frame["horizon"], dtype=np.int64),
                "prediction": np.asarray(frame["prediction"], dtype=float) - np.asarray(frame["actual"], dtype=float),
                "actual": np.zeros(len(frame)),
            }
        )
    return _panel(out, {}, freq)[0]


def _key_codes(panel: LossPanel, rows: np.ndarray) -> np.ndarray:
    """One integer per forecast key (run date, horizon, hour) of `rows`, to match keys of two panels."""
    return (panel.run_days(rows) * 1000 + panel.horizon[rows]) * 1000 + panel.hour[rows]


def _backtest_ratio(
    panel: LossPanel, backtest: LossPanel, rows: np.ndarray, member: str, benchmark: str
) -> Dict[str, float]:
    """The live skill over the backtest's, both on the forecast keys (run date, target date, hour, horizon) where the live window and
    the backtest both have the member and the benchmark: one sample, however far the backtest reaches beyond the window."""
    out = {
        f"backtest_skill_{benchmark}": math.nan,
        f"live_vs_backtest_{benchmark}": math.nan,
        f"matched_days_{benchmark}": 0,
        f"matched_keys_{benchmark}": 0,
    }
    if member == benchmark or member not in backtest or benchmark not in backtest:
        return out
    live_rows = panel.common(rows, [member, benchmark])
    past_rows = backtest.common(backtest.rows(), [member, benchmark])
    live_codes, past_codes = _key_codes(panel, live_rows), _key_codes(backtest, past_rows)
    matched, in_live, in_past = np.intersect1d(live_codes, past_codes, return_indices=True)
    out[f"matched_keys_{benchmark}"] = int(len(matched))
    out[f"matched_days_{benchmark}"] = int(len(np.unique(matched // 1_000_000)))
    if not len(matched):
        return out

    def ratio(source: LossPanel, chosen: np.ndarray) -> float:
        mine = source.errors[chosen, source.column(member)].mean()
        theirs = source.errors[chosen, source.column(benchmark)].mean()
        return float(mine / theirs) if theirs > 0 else math.nan

    then = ratio(backtest, past_rows[in_past])
    now = ratio(panel, live_rows[in_live])
    out[f"backtest_skill_{benchmark}"] = then
    out[f"live_vs_backtest_{benchmark}"] = (
        now / then if then and math.isfinite(then) and math.isfinite(now) else math.nan
    )
    return out


def _rungs(
    ladders: Sequence[MonitorMember],
    visible: Mapping[str, Mapping[str, Mapping[int, Any]]],
    actuals: _Actuals,
    windows: Sequence[int],
    days: Sequence[str],
    through: Optional[str],
    freq: str,
) -> pd.DataFrame:
    """The rung table (`MonitorReport.rungs`): each fallback ensemble read under every policy."""
    columns = ["policy", "window", "member", "rung", "rung_member", "days", "forecasts", "share", "scored", "mae"]
    out = []
    for member in ladders:
        for policy in POLICIES:
            frame = _read_keys(member, visible[policy])
            actual, known = actuals.lookup(frame)
            panel, _ = _panel(
                {member.name: _signed(frame, actual, known)}, {member.name: _rung_frame(frame, known)}, freq
            )
            rows = panel.rows()
            run_days = panel.run_days(rows)
            own = panel.errors[:, panel.column(member.name)]
            rung = panel.errors[:, panel.column(member.name + RUNG_CHANNEL)]
            for window in windows:
                in_window = _window_days(days, through, window)
                raw = frame[frame["run_date_text"].isin(in_window)]
                every = raw[["run_date_text", "horizon"]].drop_duplicates()
                mine = rows[np.isin(run_days, _day_numbers(in_window))]
                for number, ladder_member in enumerate(member.ladder or ()):
                    served = raw[raw["rung"] == number]
                    count = len(served[["run_date_text", "horizon"]].drop_duplicates())
                    chosen = mine[np.isfinite(own[mine]) & (rung[mine] == number)]
                    out.append(
                        dict(
                            policy=policy,
                            window=window,
                            member=member.name,
                            rung=number,
                            rung_member=ladder_member,
                            days=count,
                            forecasts=len(served),
                            share=count / len(every) if len(every) else math.nan,
                            scored=len(chosen),
                            mae=float(own[chosen].mean()) if len(chosen) else math.nan,
                        )
                    )
    return pd.DataFrame(out, columns=columns)


def _evaluate(
    rules: Sequence[Dict[str, Any]],
    report: MonitorReport,
    served: Mapping[str, pd.DataFrame],
    quality: Mapping[str, Optional[Dict[str, Any]]],
    store: LiveStore,
    visible: Mapping[str, Any],
    delivered: Mapping[str, Any],
    days: Sequence[str],
) -> List[Alert]:
    """The alerts of `rules` that fire on the report (module docstring)."""
    fired: List[Alert] = []
    daily = report._daily
    for spec in rules:
        rule = next(key for key in spec if key in ALERT_RULES)
        threshold = spec[rule]
        if rule == "skill_above":
            wanted = spec.get("member")
            chosen = [spec["benchmark"]] if "benchmark" in spec else list(report.benchmarks)
            _require(wanted, report.members, "member")
            _require(spec.get("benchmark"), report.benchmarks, "benchmark")
            run = int(spec.get("for_days", 1))
            for name in [wanted] if wanted else [m for m in report.members if m not in report.benchmarks]:
                for bench in chosen:
                    column = f"skill_{bench}"
                    if bench == name or not len(daily) or column not in daily:
                        continue
                    mine = daily[(daily["member"] == name) & daily[column].notna()].sort_values(["run_date", "horizon"])
                    last = mine.tail(run)
                    if len(last) == run and (last[column] > threshold).all():
                        fired.append(
                            Alert(
                                rule,
                                f"{name}: skill against {bench} above {threshold} on each of the last {run} days "
                                f"(weakest {float(last[column].min()):.3f}, through {last['run_date'].iloc[-1]:%Y-%m-%d})",
                                member=name,
                                benchmark=bench,
                                value=float(last[column].min()),
                                threshold=float(threshold),
                                through=f"{last['run_date'].iloc[-1]:%Y-%m-%d}",
                            )
                        )
        elif rule in ("participation_below", "delivered_below"):
            window = int(spec.get("window", report.windows[0]))
            in_window = _window_days(days, report.through, window)
            if not in_window:
                continue
            seen = visible if rule == "participation_below" else delivered
            share = sum(1 for d in in_window if seen.get(d)) / len(in_window)
            if share < threshold:
                what = (
                    "committed a revision before the deadline" if rule == "participation_below" else "have a delivery"
                )
                fired.append(
                    Alert(
                        rule,
                        f"only {share:.0%} of the last {len(in_window)} issue days {what} (below {threshold:.0%})",
                        value=share,
                        threshold=float(threshold),
                        through=report.through,
                    )
                )
        elif rule == "rung_at_least":
            run = int(spec.get("for_days", 1))
            wanted = spec.get("member")
            _require(wanted, list(served), "member")
            for name in [wanted] if wanted else list(served):
                last = served[name].sort_values(["run_date", "horizon"]).tail(run)
                if len(last) == run and (last["rung"] >= threshold).all():
                    fired.append(
                        Alert(
                            rule,
                            f"{name}: served by rung {int(last['rung'].max())} (at least {int(threshold)}) on each of the last "
                            f"{run} days, through {last['run_date'].iloc[-1]:%Y-%m-%d}",
                            member=name,
                            value=float(last["rung"].min()),
                            threshold=float(threshold),
                            through=f"{last['run_date'].iloc[-1]:%Y-%m-%d}",
                        )
                    )
        else:  # stale_input
            column, age, run = threshold, float(spec.get("age", 1)), int(spec.get("for_days", 1))
            zone = store.zone()
            usable = []
            for day in sorted(d for d, record in quality.items() if record is not None):
                held = (quality[day].get("columns") or {}).get(column)
                if held is None:
                    continue  # the record doesn't say: no information, never a verdict
                last_valid = held.get("last_valid")
                gap = (
                    math.inf
                    if last_valid is None
                    else (pd.Timestamp(day, tz=zone) - pd.Timestamp(last_valid)).total_seconds() / 86400.0
                )
                usable.append((day, gap))
            tail = usable[-run:]
            if len(tail) == run and all(gap > age for _, gap in tail):
                weakest = min(gap for _, gap in tail)
                fired.append(
                    Alert(
                        rule,
                        f"input {column!r}: its last valid value is more than {age:g} days before the issue day on each of the "
                        f"last {run} days with a quality record (through {tail[-1][0]})",
                        value=None if math.isinf(weakest) else weakest,
                        threshold=age,
                        through=tail[-1][0],
                    )
                )
    return fired


def _require(value: Optional[str], known: Sequence[str], what: str) -> None:
    if value is not None and value not in known:
        raise ValueError(f"alert {what} {value!r} is not one of {list(known)}")


def monitor(
    members: Sequence[MonitorMember],
    benchmarks: Sequence[str],
    store: LiveStore,
    data: pd.DataFrame,
    information_set: Any,
    *,
    windows: Sequence[int],
    policy: str,
    cutoff: Any,
    backtest: Any,
    alerts: Any,
    target: str,
    missing: Sequence[str] = (),
    issue_days: Optional[Sequence[str]] = None,
) -> MonitorReport:
    """`ModelPipeline.monitor` over members already found in `store`: see the module docstring."""
    if not store.is_live():
        raise LiveDirectoryError(f"{store.save_dir} is no live directory (no live.json): nothing there to monitor")
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {list(POLICIES)}, got {policy!r}")
    windows = tuple(windows)
    if not windows or any(isinstance(w, bool) or not isinstance(w, int) or w < 1 for w in windows):
        raise ValueError(f"windows are whole numbers of days from 1, got {windows!r}")
    rules = load_alerts(alerts)
    names = [m.name for m in members]
    unknown = [b for b in benchmarks if b not in names]
    if unknown:
        raise ValueError(f"benchmarks {unknown} have no results in {store.save_dir}; its members are {names}")
    now = current_clock().now()
    days, through = _calendar(store, now, issue_days)

    visible = {"latest": store.visible("latest", cutoff), "first": store.visible("first", cutoff)}
    visible["delivered"] = _delivered_view(visible["latest"], store)
    if cutoff is not None and policy != "delivered":
        _warn_dropped(store.save_dir, store, policy, cutoff, visible[policy], None)
    keys = {m.name: _read_keys(m, visible[policy]) for m in members}
    failed = {m.name: _read_keys(m, visible[policy], (CHECK_FAILED,)) for m in members}
    freqs = {name: _frequency(frame) for name, frame in keys.items() if len(frame)}
    if len(set(freqs.values())) > 1:
        raise ValueError(f"monitor() needs one frequency, but these members' results have several: {freqs}")
    freq = next(iter(freqs.values()), "1h")
    actuals = _Actuals(data, target, freq, information_set, now)
    looked = {name: actuals.lookup(frame) for name, frame in keys.items()}
    signed = {name: _signed(keys[name], looked[name][0], looked[name][1]) for name in names}
    ladders = [m for m in members if m.ladder]
    rung_frames = {m.name: _rung_frame(keys[m.name], looked[m.name][1]) for m in ladders}
    panel, _ = _panel(signed, rung_frames, freq)

    committed_days, delivered, revisions = visible["latest"], visible["delivered"], visible[policy]
    quality = _quality(store, revisions, days)
    bt_panel = _backtest_panel(backtest, names, freq) if backtest is not None else None
    served = {m.name: _served_days(keys[m.name], through) for m in ladders}

    daily = _daily(panel, names, benchmarks)
    rows = panel.rows()
    run_days = panel.run_days(rows)
    table_rows = []
    for window in windows:
        in_window = _window_days(days, through, window)
        mine = rows[np.isin(run_days, _day_numbers(in_window))]
        committed = sum(1 for d in in_window if committed_days.get(d))
        handed = sum(1 for d in in_window if delivered.get(d))
        recorded = [d for d in in_window if quality.get(d) is not None]
        problems = sum(1 for d in recorded if _quality_row(d, 0, quality[d])["problem"])
        for name in names:
            raw = keys[name][keys[name]["run_date_text"].isin(in_window)]
            row: Dict[str, Any] = {"window": window, "member": name}
            stats = _stats(panel, mine, name)
            for bench in benchmarks:
                skill = (
                    dict(skill=math.nan, common=0, days=0, wins=0)
                    if bench == name
                    else _skill(panel, mine, name, bench)
                )
                row.update({f"{key}_{bench}": value for key, value in skill.items()})
                if bt_panel is not None:
                    row.update(_backtest_ratio(panel, bt_panel, mine, name, bench))
            row.update(stats)
            row["unscored"] = int(len(raw) - stats["forecasts"])
            lost = failed[name][failed[name]["run_date_text"].isin(in_window)]
            row["check_failed"] = len(lost)
            row["check_failed_days"] = int(lost["run_date_text"].nunique())
            row["issue_days"] = len(in_window)
            row["committed_days"] = committed
            row["participation"] = committed / len(in_window) if in_window else math.nan
            row["delivered_days"] = handed
            row["delivered"] = handed / len(in_window) if in_window else math.nan
            row["member_days"] = int(raw["run_date_text"].nunique())
            row["fallback_days"], row["fallback_mae"] = _fallback(panel, mine, name, served.get(name), in_window, raw)
            row["quality_records"] = len(recorded)
            row["quality_days"] = problems
            table_rows.append(row)
    tables = pd.DataFrame(table_rows)
    in_largest = set(_window_days(days, through, max(windows)))
    lost_keys = {
        name: int(frame["run_date_text"].isin(in_largest).sum()) for name, frame in failed.items() if len(frame)
    }
    lost_keys = {name: count for name, count in lost_keys.items() if count}
    if lost_keys:
        tables.attrs["check_failed"] = lost_keys
        warnings.warn(
            f"Members had forecasts turned into check_failed skips by checks before export: {lost_keys} (keys, in the largest "
            "window). Those days are out of their keys and of the common keys, so every score here is conditional on passing the "
            "checks, and a member's worst days can drop out of its own score; `check_failed` and `check_failed_days` in the table "
            "count them per window. To judge the delivered forecast, score the FallbackEnsemble (monitored.rungs()).",
            UserWarning,
            stacklevel=3,
        )
    if len(tables):
        lead = ["window", "member"]
        prefixes = (
            "skill_",
            "common_",
            "days_",
            "wins_",
            "backtest_skill_",
            "live_vs_backtest_",
            "matched_days_",
            "matched_keys_",
        )
        skills = [c for c in tables.columns if c.startswith(prefixes)]
        tables = tables[lead + skills + [c for c in tables.columns if c not in lead + skills]]
    quality_table = pd.DataFrame(
        [_quality_row(d, max(revisions[d]), quality.get(d)) for d in days if revisions.get(d)],
        columns=list(_quality_row("", 0, None)),
    )
    report = MonitorReport(
        windows=windows,
        policy=policy,
        cutoff=cutoff,
        benchmarks=tuple(benchmarks),
        members=tuple(names),
        missing=tuple(missing),
        through=through,
        now=now,
        _tables=tables,
        _daily=daily,
        _rungs=_rungs(ladders, visible, actuals, windows, days, through, freq),
        _quality=quality_table,
    )
    report.alerts = _evaluate(rules, report, served, quality, store, committed_days, delivered, days)
    return report


__all__ = ["Alert", "MonitorReport", "data_in_results_zone", "load_alerts", "monitor"]
