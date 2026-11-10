"""Another pipeline's stored forecasts as a data column (`ForecastSource`): chaining through the data.

A model's results in a `save_dir` (`ModelPipeline.results_path`: `<name>.jsonl` or the Parquet directory `<name>.parquet`, next to
`<name>.manifest.json`) come back as one column: the value at an instant is the forecast for that instant issued `horizon` local
days of the upstream's data zone before its date. The upstream's manifest says when that was (its issue time and zone) and which
zone its dates are of (`data_timezone`), so `epftoolbox2.information_set.source_rules` gives the column the rule
`Availability.published(issue time, days_before=horizon, timezone=issue zone, day_zone=data zone)`, and
`InformationSet.with_source_rules` registers what the downstream's checks read (`ChainedInput`): its coverage of the rows the fits
read, and whether the upstream itself used inputs after its issue time (`models._chained`).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..._date_utils import _occurrences
from .base import DataSource

# The record keys `fetch` reads; a Parquet directory reads these columns alone.
_READ = ("target_date", "hour", "horizon", "freq", "prediction", "issued_at")
_STEP_MINUTES = {"1h": 60, "15min": 15}
_DAILY = "1D"
_ENSEMBLE_KIND = "ensemble"
_NO_ZONE = (None, "None", "")
# The day statistics `agg` may name: `DailyStatsTransformer`'s, over each target date of the upstream's data zone.
AGGREGATES = ("min", "max", "mean", "peak")


class UpstreamError(ValueError):
    """The upstream results can't be read as a chained column: no manifest, no issue time, no zone, a live record, ..."""


@dataclass(frozen=True)
class Upstream:
    """What `ForecastSource` reads of an upstream run: its results, its manifest, and those of its leaf models (itself, or an
    ensemble's members under any nesting), which say when its forecasts were issued and what they read late."""

    results: Path
    manifest: Dict[str, Any]
    leaves: Tuple[Tuple[str, Dict[str, Any]], ...]  # (model name, its manifest)
    freq: str
    data_timezone: str
    issue_time: str
    issue_timezone: str
    daily_stamp: Optional[float]  # minutes after midnight of a daily run's rows; None at any other freq
    live: bool = False  # a live directory's revisions (`results.live`), read as committed by each issue day's cut-off

    def late_inputs(self) -> Tuple[str, ...]:
        """The inputs the upstream's leaf models listed as late or without a rule (`availability="warn"`'s listing), each once,
        as `model: predictor` for an ensemble's members."""
        found: List[str] = []
        for name, manifest in self.leaves:
            listing = (manifest.get("availability") or {}).get("listing") or []
            for row in listing:
                if row.get("status") != "ok":
                    label = row.get("predictor", "?")
                    found.append(label if len(self.leaves) == 1 else f"{name}: {label}")
        return tuple(dict.fromkeys(found))

    def unchecked(self, issued: Optional[Dict[str, Tuple[str, str]]] = None) -> Tuple[str, ...]:
        """Each leaf model whose listing doesn't vouch for every forecast it stored, with why: it ran at level "none" (no
        listing); at "raise" with `unknown` other than "raise" (a column without a rule passed unlisted); it was resumed under
        other rules than its manifest's (`stored_under_other_rules`), or over a mismatch (`data_unverified`, `training_cut`
        "stored": "unverified"); or (`issued`: model -> first and last run date of its records) it holds forecasts issued outside
        the test period its manifest describes, which no run it records checked."""
        from ...results.manifest import STORED_UNDER_OTHER_RULES, TRAINING_CUT_UNVERIFIED

        out: List[str] = []
        for name, manifest in self.leaves:
            availability = manifest.get("availability")
            unknown = (manifest.get("information_set") or {}).get("unknown", "raise")
            cut = manifest.get("training_cut") or {}
            span = (issued or {}).get(name)
            if not availability:
                out.append(f"{name} (ran at availability='none')")
            elif availability.get("level") == "raise" and unknown != "raise":
                out.append(
                    f"{name} (ran at availability='raise' with unknown={unknown!r}: a column without a rule passed)"
                )
            mark = (availability or {}).get(STORED_UNDER_OTHER_RULES)
            if mark is not None and mark.get("level") == "none":
                out.append(
                    f"{name} (holds forecasts an earlier run made at availability='none', on days no later run checked)"
                )
            elif mark is not None:
                out.append(f"{name} (holds forecasts listed under other rules than its manifest's)")
            if manifest.get("data_unverified") or cut.get("stored") == TRAINING_CUT_UNVERIFIED["stored"]:
                out.append(
                    f"{name} (its stored forecasts were accepted unverified, over a mismatch or without a manifest)"
                )
            if span is not None and not manifest.get("windows") and _outside(span, manifest):
                out.append(
                    f"{name} (holds forecasts issued {span[0]} to {span[1]}, outside the test period "
                    f"{manifest.get('test_start')} to {manifest.get('test_end')} its manifest checked)"
                )
        return tuple(out)


class ForecastSource(DataSource):
    """The forecasts a `ModelPipeline` stored in `path` (its `save_dir`) for `model`, at one horizon, as the column `column`.

    The value at instant t is the forecast for t issued `horizon` local days before t's date, dates of the upstream's data zone
    (its manifest's `data_timezone`), at the upstream's issue time (`issue_time` and `issue_timezone` of its information set):
    slot `hour` of `target_date` by that zone's clock. A slot the clock skips has no record and no row; a slot it repeats gives
    its value to both of its instants, as the forecast is for the wall-clock slot. A daily upstream (`freq="1D"`) puts its value
    at its frame's daily stamp (`daily_stamp` in its manifest). Lag the column like any other: `solar_fc_d+1` on the row of D is
    the forecast for D+1 issued on D. Other horizons are other sources with other column names.

    With `agg` the source emits `{column}_day_{agg}` instead: the day statistic of those forecasts over each target date, on
    every row of the date, as `DailyStatsTransformer(columns=column, stats=[agg], freq=...)` makes it on them in the upstream's
    data zone, bit for bit: NaN on a date missing a slot or holding a missing forecast. Every slot of a target date is issued at
    the same instant, so the statistic is known exactly when its forecasts are, and its column takes the same rule.

    The upstream must have run with an issue time, so that its forecasts' availability can be stated; `available="HH:MM"` (in
    the upstream issue zone) states a later time, for forecasts revised after it, never an earlier one. An ensemble's results
    read their issue time from its members' manifests.

    A live upstream (a live directory, `results.live`, step 8c) holds revisions: each issue day is read from the newest
    revision committed at or before `available` (else the issue time) on that day, in the issue zone, so a cell never holds a
    forecast that wasn't out when the rule says it is known; a day with no such revision has no value. A live upstream commits
    until its deadline, after its issue time, so give it `available` at or after the deadline: without, its on-schedule
    revisions are left out, which warns with the number of issue days. An upstream with no committed revision raises. Live
    records (with `issued_at`) outside a live directory are refused.

    Its availability rule and its upstream's identity come from `InformationSet.with_source_rules` (a `Workflow` applies it):
    a model run outside one must pass `information_set.with_source_rules(data_pipeline.sources)`, else the column has no rule.

    Args:
        path: The upstream run's `save_dir`.
        model: The upstream model's (or ensemble's) name.
        column: The name of the column this source emits.
        horizon: Which of the upstream's horizons to read, in days between issue day and target day.
        available: A later time than the upstream's issue time at which its forecasts are known ("HH:MM", upstream issue zone).
        agg: A day statistic of the forecasts instead of the forecasts, emitted as `{column}_day_{agg}`: "min", "max",
            "mean" or "peak" (the mean from 08:00 to 19:59). None (the default, or "none") emits the forecasts as `column`.
    """

    # Parameters a saved pipeline writes only when set, so files saved now load in versions from before them.
    _WRITTEN_WHEN_SET = {"agg": None}

    def __init__(
        self,
        path: Any,
        model: str,
        column: str,
        horizon: int = 1,
        available: Optional[str] = None,
        agg: Optional[str] = None,
    ):
        self.path = os.fspath(path) if isinstance(path, os.PathLike) else path
        self.model = model
        self.column = column
        self.horizon = horizon
        self.available = available
        self.agg = agg
        self._validate_config()

    def _validate_config(self) -> bool:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError(f"ForecastSource: path must be the upstream run's save_dir, got {self.path!r}")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError(f"ForecastSource: model must be the upstream model's name, got {self.model!r}")
        if not isinstance(self.column, str) or not self.column:
            raise ValueError(f"ForecastSource: column must be a non-empty name, got {self.column!r}")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, (int, np.integer)) or self.horizon < 1:
            raise ValueError(f"ForecastSource: horizon must be a positive integer, got {self.horizon!r}")
        self.horizon = int(self.horizon)
        if isinstance(self.agg, str) and self.agg.lower() == "none":
            self.agg = None  # the TUI's choice for none
        if self.agg is not None and self.agg not in AGGREGATES:
            raise ValueError(f"ForecastSource: agg must be None or one of {list(AGGREGATES)}, got {self.agg!r}")
        if self.available is not None:
            from ...information_set import _parse_time  # the information set imports the sources

            _parse_time(self.available)
        return True

    @property
    def output_column(self) -> str:
        """The name of the column it emits: `column`, or with `agg` `{column}_day_{agg}`."""
        return self.column if self.agg is None else f"{self.column}_day_{self.agg}"

    def get_cache_config(self) -> Optional[Dict]:
        # The upstream results grow as their pipeline runs: a per-source cache would serve stale forecasts. The single-file
        # cache sees a change through `content_digest`.
        return None

    # -- the upstream -----------------------------------------------------------------------------------------------------

    def _directory(self) -> Path:
        """`path`, with `~` expanded."""
        return Path(os.path.expanduser(self.path))

    def _stem(self, name: str) -> Path:
        return self._directory() / name.lower().replace(" ", "_")

    def results_path(self) -> Path:
        """The upstream's results: `<name>.jsonl` or the directory `<name>.parquet`, whichever exists (both raise)."""
        return self._results_of(self.model)

    def _results_of(self, name: str) -> Path:
        from ...results.store import SAVE_FORMATS

        stem = self._stem(name)
        jsonl = stem.with_name(stem.name + SAVE_FORMATS["jsonl"])
        parquet = stem.with_name(stem.name + SAVE_FORMATS["parquet"])
        if jsonl.exists() and parquet.is_dir():
            raise UpstreamError(
                f"ForecastSource({self.model!r}): results in both formats, {jsonl} and {parquet}; run `epf results convert "
                f"{self.path} --to <format>` to finish a conversion"
            )
        return parquet if parquet.is_dir() else jsonl

    def _manifest_of(self, name: str) -> Dict[str, Any]:
        path = self._stem(name).with_name(self._stem(name).name + ".manifest.json")
        if not path.exists():
            raise UpstreamError(
                f"ForecastSource({self.model!r}): no manifest {path}. Run the upstream pipeline with save_dir={self.path!r}, "
                "so that its results say when they were issued."
            )
        return json.loads(path.read_text())

    def _leaves(
        self, name: str, manifest: Dict[str, Any], seen: Tuple[str, ...] = ()
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """The leaf models under `name`: itself, or an ensemble's members' leaves, read from their manifests in `path`."""
        if manifest.get("kind") != _ENSEMBLE_KIND:
            return [(name, manifest)]
        if name in seen:
            raise UpstreamError(f"ForecastSource({self.model!r}): the ensemble {name!r} names itself among its members")
        out: List[Tuple[str, Dict[str, Any]]] = []
        for member in manifest.get("member_results") or {}:
            out.extend(self._leaves(member, self._manifest_of(member), (*seen, name)))
        if not out:
            raise UpstreamError(f"ForecastSource({self.model!r}): the ensemble {name!r} records no members")
        return out

    def upstream(self) -> Upstream:
        """The upstream as its manifests describe it, or `UpstreamError` where they can't say when its forecasts were known or
        where they are: no information set (no issue time), no zone (a naive index), a daily run without `daily_stamp`."""
        manifest = self._manifest_of(self.model)
        leaves = self._leaves(self.model, manifest)
        issues = set()
        for name, leaf in leaves:
            info = leaf.get("information_set")
            if not info or not info.get("issue_time"):
                raise UpstreamError(
                    f"ForecastSource({self.model!r}): {name!r} ran without an issue time, so nothing says when its forecasts "
                    "were known. Run it with ModelPipeline(issue_time=..., issue_timezone=...)."
                )
            issues.add((info["issue_time"], info.get("issue_timezone", "Europe/Brussels")))
        if len(issues) > 1:
            raise UpstreamError(
                f"ForecastSource({self.model!r}): its members were issued at different times {sorted(issues)}; an average of "
                "them is known when the latest is, which no one rule here states"
            )
        issue_time, issue_zone = issues.pop()
        zone = manifest.get("data_timezone")
        if zone in _NO_ZONE:
            zone = next(
                (leaf.get("data_timezone") for _, leaf in leaves if leaf.get("data_timezone") not in _NO_ZONE), None
            )
        if zone in _NO_ZONE or not _is_zone(zone):
            raise UpstreamError(
                f"ForecastSource({self.model!r}): the upstream data's index had no named time zone ({zone!r}), so its periods "
                "can't be placed in time. Run it on a frame with a zone, e.g. DataPipeline(timezone='Europe/Warsaw')."
            )
        if issue_zone == "data":
            issue_zone = zone
        freq = manifest.get("freq") or next((leaf.get("freq") for _, leaf in leaves if leaf.get("freq")), None)
        if freq not in (*_STEP_MINUTES, _DAILY):
            raise UpstreamError(f"ForecastSource({self.model!r}): the upstream's freq {freq!r} can't be placed in time")
        stamp = None
        if freq == _DAILY:
            stamps = {leaf.get("daily_stamp") for _, leaf in leaves}
            if None in stamps or len(stamps) != 1:
                raise UpstreamError(
                    f"ForecastSource({self.model!r}): a daily (freq='1D') upstream places each value at the time of day its "
                    "rows were stamped at, which its manifest doesn't record (`daily_stamp`, written since ForecastSource exists). "
                    f"Run the upstream again into {self.path!r} (a resume forecasts nothing new and records it)."
                )
            stamp = _minutes_of(stamps.pop())
            if self.agg is not None:
                raise UpstreamError(
                    f"ForecastSource({self.model!r}): agg={self.agg!r} takes a day statistic of a day's forecasts, and a daily "
                    "(freq='1D') upstream has one a day: read it without agg"
                )
        if self.available is not None and _minutes_of(self.available) < _minutes_of(issue_time):
            raise UpstreamError(
                f"ForecastSource({self.model!r}): available={self.available!r} is before the upstream's issue time "
                f"{issue_time} {issue_zone}: its forecasts can't be known before they were made"
            )
        return Upstream(
            self.results_path(),
            manifest,
            tuple(leaves),
            freq,
            str(zone),
            issue_time,
            str(issue_zone),
            stamp,
            live=self._is_live(leaves),
        )

    def _is_live(self, leaves: List[Tuple[str, Dict[str, Any]]]) -> bool:
        """Whether the upstream is a live directory's: `live.json` there and every leaf's manifest "live"; one without the
        other raises."""
        from ...results.store import MODE_LIVE, is_live_dir

        directory = is_live_dir(self._directory())
        modes = {name: leaf.get("mode") == MODE_LIVE for name, leaf in leaves}
        if any(live != directory for live in modes.values()):
            raise UpstreamError(
                f"ForecastSource({self.model!r}): {self.path} is "
                + ("a live directory" if directory else "no live directory")
                + f", and the manifests of {sorted(name for name, live in modes.items() if live != directory)} say "
                + ("otherwise" if directory else "mode 'live'")
                + ": live revisions and a backtest's results were mixed. Chain one run's results from its own directory."
            )
        return directory

    def _cutoff(self, upstream: Upstream):
        """Each issue day's cut-off of a live upstream: `available` (else the issue time) on that day in the issue zone."""
        from ...results.live import local_instant

        time = self.available or upstream.issue_time
        return lambda day: local_instant(day, time, upstream.issue_timezone)

    def _committed(self, upstream: Upstream):
        """What `read_records(committed=...)` reads of a live upstream (`results.live.LiveStore.visible`, policy "latest" by
        each issue day's cut-off), None for a backtest's; `UpstreamError` for a live one that committed nothing."""
        if not upstream.live:
            return None
        from ...results.live import LiveStore

        store = LiveStore(self._directory())
        if not store.committed():
            raise UpstreamError(
                f"ForecastSource({self.model!r}): the live directory {self.path} has no committed revision (no "
                f"`committed` line in {store.issues_path.name}): nothing it holds was ever out"
            )
        visible = store.visible("latest", cutoff=self._cutoff(upstream))
        everything = store.visible("latest", cutoff=None)
        later = sorted(day for day, revisions in everything.items() if max(revisions) > max(visible.get(day, {0: None})))
        if later:
            time = self.available or upstream.issue_time
            warnings.warn(
                f"ForecastSource({self.model!r}): on {len(later)} issue days (e.g. {later[0]}) the upstream committed a newer "
                f"revision after {time} {upstream.issue_timezone}, which this column doesn't read: each cell is the revision "
                "out by "
                + ("the upstream's issue time" if self.available is None else "`available`")
                + ". A live upstream commits until its deadline: pass available= at or after it (its rule is then the "
                "same time), or keep this cut-off knowingly.",
                UserWarning,
                stacklevel=4,
            )
        return visible

    def availability(self, upstream: Optional[Upstream] = None):
        """The column's rule: published at the upstream's issue time (or `available`) in its issue zone, `horizon` days before
        the delivery day, a local date of the upstream's data zone; its archive is a snapshot (one forecast per key). With `agg`
        the same: the statistic of a target date is known when the latest of its forecasts is, and all of them are issued at
        that one instant."""
        from ...information_set import Availability  # the information set imports the sources

        upstream = self.upstream() if upstream is None else upstream
        time = self.available or upstream.issue_time
        what = "forecasts" if self.agg is None else f"day {self.agg} of the forecasts"
        return Availability.published(
            time,
            self.horizon,
            timezone=upstream.issue_timezone,
            day_zone=upstream.data_timezone,
            vintage="snapshot",
            note=f"{what} of {self.model} (horizon {self.horizon}) stored in {self.path}, issued {time} "
            f"{upstream.issue_timezone} on {upstream.data_timezone} dates",
        )

    def chained_input(self, upstream: Optional[Upstream] = None):
        """What the downstream's checks and manifest read of this column (`information_set.ChainedInput`)."""
        from ...information_set import ChainedInput  # the information set imports the sources
        from ...results.manifest import manifest_digest

        upstream = self.upstream() if upstream is None else upstream
        # A live upstream's revisions are all made under its one live manifest, not over a test period: nothing to be outside.
        issued = (
            {}
            if upstream.live
            else {name: _issued_span(self._results_of(name), self.horizon) for name, _ in upstream.leaves}
        )
        return ChainedInput(
            column=self.output_column,
            path=str(self._directory().resolve()),
            model=self.model,
            horizon=self.horizon,
            issue_time=upstream.issue_time,
            issue_timezone=upstream.issue_timezone,
            data_timezone=upstream.data_timezone,
            available=self.available,
            digest=manifest_digest(upstream.manifest),
            late=upstream.late_inputs(),
            unchecked=upstream.unchecked(issued),
            freq=upstream.freq,
            agg=self.agg,
        )

    def content_digest(self) -> Optional[str]:
        """A hash of the upstream's manifest (without provenance, `manifest_digest`) and of its results' content (a JSONL file's
        bytes, a Parquet directory's catalog, which names its immutable parts), so that a single-file pipeline cache fetches
        again after the upstream changed, whatever the file system's clock, and reuses its file otherwise."""
        from ...results.manifest import manifest_digest
        from ...results.parquet_store import CATALOG

        manifest = self._manifest_of(self.model)
        results = self.results_path()
        path = results / CATALOG if results.is_dir() else results
        identity = _file_digest(path) if path.exists() else None
        content = {"manifest": manifest_digest(manifest), "results": identity}
        from ...results.live import ISSUES
        from ...results.store import is_live_dir

        if is_live_dir(self._directory()):  # a commit changes what is read without changing the results
            log = self._directory() / ISSUES
            content["commits"] = _file_digest(log) if log.exists() else None
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    # -- the column ---------------------------------------------------------------------------------------------------------

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """The upstream's forecasts of `horizon` at their target instants in `[start, end)`, UTC; with `agg` their statistic
        over each whole target date, taken before the range is cut, so a sub-range holds the same values."""
        from ...results.store import read_records

        upstream = self.upstream()
        results = upstream.results
        if not results.exists():
            raise UpstreamError(f"ForecastSource({self.model!r}): no results {results}")
        horizon = self.horizon
        found: Dict[str, set] = {"freq": set()}

        committed = self._committed(upstream)

        def pick(record: Dict[str, Any]) -> Optional[Tuple[str, int, float]]:
            if record.get("issued_at") is not None and not upstream.live:
                raise UpstreamError(
                    f"ForecastSource({self.model!r}): {results} holds live records (with `issued_at`, e.g. run_date "
                    f"{record.get('run_date')}) outside a live directory: which revision a chained value may use depends on "
                    "when it was committed, which only the live directory's issues.jsonl says. Chain them from it."
                )
            if record.get("horizon") != horizon:
                return None
            found["freq"].add(record.get("freq", upstream.freq))
            prediction = record.get("prediction")
            return record["target_date"], int(record["hour"]), math.nan if prediction is None else float(prediction)

        picked = list(read_records(results, columns=_READ, transform=pick, committed=committed))
        if not picked:
            raise UpstreamError(
                f"ForecastSource({self.model!r}): {results} has no forecast of horizon {horizon}"
                + (" in a revision committed by its issue day's cut-off" if upstream.live else "")
                + f". The upstream's horizons are 1 to {upstream.manifest.get('horizon')} (or its `horizons`)."
            )
        freqs = found["freq"] - {None}
        if freqs - {upstream.freq}:
            raise UpstreamError(
                f"ForecastSource({self.model!r}): records at {sorted(freqs)} in results whose manifest says {upstream.freq}"
            )
        frame = _placed(picked, upstream, self.column)
        if self.agg is not None:
            frame = _day_statistic(frame, upstream, self.column, self.agg)
        start, end = _utc(start), _utc(end)
        out = frame[(frame.index >= start) & (frame.index < end)]
        if out.empty:
            warnings.warn(
                f"ForecastSource({self.model!r}): no forecast of horizon {horizon} for [{start}, {end}); the upstream's "
                f"target instants run from {frame.index.min()} to {frame.index.max()}",
                UserWarning,
                stacklevel=2,
            )
        return out


def _placed(picked: List[Tuple[str, int, float]], upstream: Upstream, column: str) -> pd.DataFrame:
    """The records (target date, period, prediction) at their instants: the wall-clock slot of the period on the target date in
    the upstream's data zone (both instants of a repeated one, none of a skipped one), or a daily run's stamp."""
    dates = pd.DatetimeIndex(pd.to_datetime([one[0] for one in picked]))
    periods = np.array([one[1] for one in picked], dtype=np.int64)
    values = np.array([one[2] for one in picked], dtype=np.float64)
    if upstream.freq == _DAILY:
        minutes = np.full(len(periods), upstream.daily_stamp, dtype=np.float64)
    else:
        minutes = periods.astype(np.float64) * _STEP_MINUTES[upstream.freq]
    wall = dates + pd.to_timedelta(minutes, unit="min")
    first, second = _occurrences(pd.DatetimeIndex(wall), upstream.data_timezone)
    exists = ~first.isna()
    repeated = exists & (second != first)
    instants = np.concatenate([first[exists].asi8, second[repeated].asi8])
    placed = np.concatenate([values[exists], values[repeated]])
    order = np.argsort(instants, kind="stable")
    index = pd.DatetimeIndex(instants[order], tz="UTC")
    if index.has_duplicates:
        twice = index[index.duplicated()][:3]
        raise UpstreamError(f"ForecastSource: two forecasts for the instants {list(twice)}")
    return pd.DataFrame({column: placed[order]}, index=index)


def _day_statistic(frame: pd.DataFrame, upstream: Upstream, column: str, agg: str) -> pd.DataFrame:
    """`{column}_day_{agg}` on the rows of `frame` (UTC, at the target instants): `DailyStatsTransformer`'s statistic over each
    local date of the upstream's data zone, on whole dates only (every slot the date has on its clock, a repeated one counted
    on both of its rows, as any frame in that zone holds it), at the upstream's freq."""
    from ..transformers import DailyStatsTransformer

    local = frame.tz_convert(upstream.data_timezone)
    with warnings.catch_warnings():
        # An upstream on UTC dates is read on UTC days, which are its days.
        warnings.filterwarnings("ignore", message="DailyStatsTransformer: the index is UTC")
        stats = DailyStatsTransformer(columns=column, stats=[agg], freq=upstream.freq).transform(local)
    return stats[[f"{column}_day_{agg}"]].tz_convert("UTC")


def _outside(span: Tuple[str, str], manifest: Dict[str, Any]) -> bool:
    """Whether run dates from `span[0]` to `span[1]` reach outside the manifest's `[test_start, test_end]`."""
    start, end = manifest.get("test_start"), manifest.get("test_end")
    if start is None or end is None:
        return True
    return str(span[0]) < str(start)[:10] or str(span[1]) > str(end)[:10]


# (results path, horizon, the file's identity) -> (first, last) run date of its ok records at that horizon
_ISSUED: Dict[Tuple[Any, ...], Optional[Tuple[str, str]]] = {}


def _issued_span(results: Path, horizon: int) -> Optional[Tuple[str, str]]:
    """The first and last `run_date` of the ok records at `horizon` of a results file or directory, None without any."""
    from ...results.store import read_records

    if not results.exists():
        return None
    key = (str(results.resolve()), horizon, _identity(results))
    if key not in _ISSUED:
        dates = [
            str(day)
            for day in read_records(
                results,
                columns=("run_date", "horizon"),
                transform=lambda record: record.get("run_date") if record.get("horizon") == horizon else None,
            )
        ]
        _ISSUED[key] = (min(dates), max(dates)) if dates else None
        while len(_ISSUED) > 64:
            del _ISSUED[next(iter(_ISSUED))]
    return _ISSUED[key]


def _identity(results: Path) -> Any:
    """What says a results file or directory is as it was: a JSONL file's size, modification time and inode, a Parquet
    directory's catalog's."""
    from ...results.parquet_store import CATALOG

    path = results / CATALOG if results.is_dir() else results
    if not path.exists():
        return None
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ino)


def _file_digest(path: Path) -> str:
    """The sha256 of a file's bytes, read in blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _minutes_of(text: Any) -> float:
    hours, minutes = str(text).split(":")[:2]
    return int(hours) * 60.0 + int(minutes)


def _is_zone(zone: Any) -> bool:
    try:
        pd.Timestamp("2000-01-01").tz_localize(zone)
    except Exception:
        return False
    return True
