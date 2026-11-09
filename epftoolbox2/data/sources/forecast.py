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

# The record keys `fetch` reads; a Parquet directory reads these columns alone (`issued_at` is not in its schema: no Parquet
# record has one).
_READ = ("target_date", "hour", "horizon", "freq", "prediction", "issued_at")
_STEP_MINUTES = {"1h": 60, "15min": 15}
_DAILY = "1D"
_ENSEMBLE_KIND = "ensemble"
_NO_ZONE = (None, "None", "")


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

    def unchecked(self) -> Tuple[str, ...]:
        """The leaf models that ran at level "none", whose manifests hold no listing: nothing says their inputs were known."""
        return tuple(name for name, manifest in self.leaves if not manifest.get("availability"))


class ForecastSource(DataSource):
    """The forecasts a `ModelPipeline` stored in `path` (its `save_dir`) for `model`, at one horizon, as the column `column`.

    The value at instant t is the forecast for t issued `horizon` local days before t's date, dates of the upstream's data zone
    (its manifest's `data_timezone`), at the upstream's issue time (`issue_time` and `issue_timezone` of its information set):
    slot `hour` of `target_date` by that zone's clock. A slot the clock skips has no record and no row; a slot it repeats gives
    its value to both of its instants, as the forecast is for the wall-clock slot. A daily upstream (`freq="1D"`) puts its value
    at its frame's daily stamp (`daily_stamp` in its manifest). Lag the column like any other: `solar_fc_d+1` on the row of D is
    the forecast for D+1 issued on D. Other horizons are other sources with other column names.

    The upstream must have run with an issue time, so that its forecasts' availability can be stated; `available="HH:MM"` (in
    the upstream issue zone) states a later time, for forecasts revised after it, never an earlier one. Records of a live
    directory (with `issued_at`) are refused: which revision a cell may use depends on when it was issued, and that reading is
    not here yet. An ensemble's results read their issue time from its members' manifests.

    Its availability rule and its upstream's identity come from `InformationSet.with_source_rules` (a `Workflow` applies it):
    a model run outside one must pass `information_set.with_source_rules(data_pipeline.sources)`, else the column has no rule.

    Args:
        path: The upstream run's `save_dir`.
        model: The upstream model's (or ensemble's) name.
        column: The name of the column this source emits.
        horizon: Which of the upstream's horizons to read, in days between issue day and target day.
        available: A later time than the upstream's issue time at which its forecasts are known ("HH:MM", upstream issue zone).
    """

    def __init__(self, path: Any, model: str, column: str, horizon: int = 1, available: Optional[str] = None):
        self.path = os.fspath(path) if isinstance(path, os.PathLike) else path
        self.model = model
        self.column = column
        self.horizon = horizon
        self.available = available
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
        if self.available is not None:
            from ...information_set import _parse_time  # the information set imports the sources

            _parse_time(self.available)
        return True

    def get_cache_config(self) -> Optional[Dict]:
        # The upstream results grow as their pipeline runs: a per-source cache would serve stale forecasts. The single-file
        # cache sees a change through `content_digest`.
        return None

    # -- the upstream -----------------------------------------------------------------------------------------------------

    def _stem(self, name: str) -> Path:
        return Path(self.path) / name.lower().replace(" ", "_")

    def results_path(self) -> Path:
        """The upstream's results: `<name>.jsonl` or the directory `<name>.parquet`, whichever exists (both raise)."""
        from ...results.store import SAVE_FORMATS

        stem = self._stem(self.model)
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
        if self.available is not None and _minutes_of(self.available) < _minutes_of(issue_time):
            raise UpstreamError(
                f"ForecastSource({self.model!r}): available={self.available!r} is before the upstream's issue time "
                f"{issue_time} {issue_zone}: its forecasts can't be known before they were made"
            )
        return Upstream(
            self.results_path(), manifest, tuple(leaves), freq, str(zone), issue_time, str(issue_zone), stamp
        )

    def availability(self, upstream: Optional[Upstream] = None):
        """The column's rule: published at the upstream's issue time (or `available`) in its issue zone, `horizon` days before
        the delivery day, a local date of the upstream's data zone; its archive is a snapshot (one forecast per key)."""
        from ...information_set import Availability  # the information set imports the sources

        upstream = self.upstream() if upstream is None else upstream
        time = self.available or upstream.issue_time
        return Availability.published(
            time,
            self.horizon,
            timezone=upstream.issue_timezone,
            day_zone=upstream.data_timezone,
            vintage="snapshot",
            note=f"forecasts of {self.model} (horizon {self.horizon}) stored in {self.path}, issued {time} "
            f"{upstream.issue_timezone} on {upstream.data_timezone} dates",
        )

    def chained_input(self, upstream: Optional[Upstream] = None):
        """What the downstream's checks and manifest read of this column (`information_set.ChainedInput`)."""
        from ...information_set import ChainedInput  # the information set imports the sources
        from ...results.manifest import manifest_digest

        upstream = self.upstream() if upstream is None else upstream
        unchecked = upstream.unchecked()
        return ChainedInput(
            column=self.column,
            path=self.path,
            model=self.model,
            horizon=self.horizon,
            issue_time=upstream.issue_time,
            issue_timezone=upstream.issue_timezone,
            data_timezone=upstream.data_timezone,
            available=self.available,
            digest=manifest_digest(upstream.manifest),
            late=upstream.late_inputs(),
            unchecked=unchecked,
            freq=upstream.freq,
        )

    def content_digest(self) -> Optional[str]:
        """A hash of the upstream's manifest (without provenance, `manifest_digest`) and of its results' identity (a JSONL file's
        size and modification time, a Parquet directory's catalog), so that a single-file pipeline cache refetches after the
        upstream changed."""
        from ...results.manifest import manifest_digest
        from ...results.parquet_store import CATALOG

        manifest = self._manifest_of(self.model)
        results = self.results_path()
        if results.is_dir():
            catalog = results / CATALOG
            identity: Any = hashlib.sha256(catalog.read_bytes()).hexdigest() if catalog.exists() else None
        elif results.exists():
            stat = results.stat()
            identity = [stat.st_size, stat.st_mtime_ns]
        else:
            identity = None
        content = {"manifest": manifest_digest(manifest), "results": identity}
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    # -- the column ---------------------------------------------------------------------------------------------------------

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """The upstream's forecasts of `horizon` at their target instants in `[start, end)`, UTC."""
        from ...results.store import read_records

        upstream = self.upstream()
        results = upstream.results
        if not results.exists():
            raise UpstreamError(f"ForecastSource({self.model!r}): no results {results}")
        horizon = self.horizon
        found: Dict[str, set] = {"freq": set()}

        def pick(record: Dict[str, Any]) -> Optional[Tuple[str, int, float]]:
            if record.get("issued_at") is not None:
                raise UpstreamError(
                    f"ForecastSource({self.model!r}): {results} holds live records (with `issued_at`, e.g. run_date "
                    f"{record.get('run_date')}): which revision a chained value may use depends on when it was issued, which "
                    "this source doesn't read yet. Chain a backtest's results."
                )
            if record.get("horizon") != horizon:
                return None
            found["freq"].add(record.get("freq", upstream.freq))
            prediction = record.get("prediction")
            return record["target_date"], int(record["hour"]), math.nan if prediction is None else float(prediction)

        picked = list(read_records(results, columns=_READ, transform=pick))
        if not picked:
            raise UpstreamError(
                f"ForecastSource({self.model!r}): {results} has no forecast of horizon {horizon}. The upstream's horizons "
                f"are 1 to {upstream.manifest.get('horizon')} (or its `horizons`)."
            )
        freqs = found["freq"] - {None}
        if freqs - {upstream.freq}:
            raise UpstreamError(
                f"ForecastSource({self.model!r}): records at {sorted(freqs)} in results whose manifest says {upstream.freq}"
            )
        frame = _placed(picked, upstream, self.column)
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
