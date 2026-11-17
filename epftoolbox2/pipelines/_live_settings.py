"""The `live:` section of a `Workflow` YAML (step 8i): what `epf forecast`, `epf refit`, `epf status`, `epf mark-delivered` and
`epf monitor` read besides the pipelines.

    live:
      save_dir: live/de_price          # the live directory (relative to the YAML)
      deadline: "11:55"                # no revision is committed after it (quote it: YAML reads 11:55 as a number)
      wait_for: {load_forecast: "10:45"}   # or "rules"
      poll: 5min
      budget: 120s
      timeout: 30s
      retries: 3
      on_failure: cache
      grace: 30min
      fit_missing: previous
      deliver: live                    # the model or ensemble whose forecasts decide the status
      export: {path: "out/{issue}.csv", format: csv, grid: products}

Every key but `save_dir` is optional and, when absent, left to `ModelPipeline.forecast()`'s own default. The keys of
`forecast_options()` reach `forecast()` as written.
"""

import re
from typing import Any, Dict, Mapping, Optional

from ..results.forecast_result import FORMATS, GRIDS

FORECAST_KEYS = (
    "deadline",
    "wait_for",
    "poll",
    "budget",
    "timeout",
    "retries",
    "on_failure",
    "grace",
    "fit_missing",
    "deliver",
)
LIVE_KEYS = ("save_dir", *FORECAST_KEYS, "export")
EXPORT_KEYS = ("path", "format", "grid")
_HH_MM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def _clock_time(value: Any, what: str) -> str:
    if isinstance(value, str) and _HH_MM.match(value.strip()):
        return value.strip()
    hint = " (quote it: YAML reads an unquoted 11:55 as the number 715)" if isinstance(value, int) else ""
    raise ValueError(f"live.{what} is a time of the issue day as 'HH:MM', got {value!r}{hint}")


class LiveSettings:
    """The validated `live:` section: `save_dir`, the options `forecast()` takes (`forecast_options()`) and `export` (`path`, `format`, `grid`, or None).

    Checks before export are the pipeline's alone (`ModelPipeline(checks=...)`, `model_pipeline.checks`): they apply in backtests
    too, so live and backtest scores stay comparable. A `checks` key here is refused."""

    def __init__(
        self,
        save_dir: str,
        *,
        deadline: Optional[str] = None,
        wait_for: Any = None,
        poll: Any = None,
        budget: Any = None,
        timeout: Any = None,
        retries: Optional[int] = None,
        on_failure: Optional[str] = None,
        grace: Any = None,
        fit_missing: Optional[str] = None,
        deliver: Optional[str] = None,
        export: Optional[Mapping[str, Any]] = None,
    ):
        if not isinstance(save_dir, str) or not save_dir.strip():
            raise ValueError(f"live.save_dir is the live directory, a path: got {save_dir!r}")
        self.save_dir = save_dir
        self.deadline = None if deadline is None else _clock_time(deadline, "deadline")
        if wait_for is not None and wait_for != "rules":
            if not isinstance(wait_for, Mapping) or not all(isinstance(column, str) for column in wait_for):
                raise ValueError(f"live.wait_for is a mapping of column -> 'HH:MM', or 'rules', got {wait_for!r}")
            wait_for = {column: _clock_time(time, f"wait_for[{column}]") for column, time in wait_for.items()}
        self.wait_for = wait_for
        self.poll, self.budget, self.timeout, self.grace = poll, budget, timeout, grace
        if retries is not None and (isinstance(retries, bool) or not isinstance(retries, int) or retries < 0):
            raise ValueError(f"live.retries is a whole number of 0 or more, got {retries!r}")
        self.retries = retries
        self.on_failure, self.fit_missing, self.deliver = on_failure, fit_missing, deliver
        if export is not None:
            export = self._export(export)
        self.export = export

    @staticmethod
    def _export(raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise ValueError(f"live.export is a mapping of path, format and grid, got {raw!r}")
        unknown = sorted(set(raw) - set(EXPORT_KEYS))
        if unknown:
            raise ValueError(f"live.export: unknown key(s) {unknown}; the keys are {list(EXPORT_KEYS)}")
        path = raw.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"live.export.path is the file to write, a path that may hold {{issue}}: got {path!r}")
        out = {"path": path, "format": raw.get("format", "csv"), "grid": raw.get("grid", "products")}
        if out["format"] not in FORMATS:
            raise ValueError(f"live.export.format must be one of {list(FORMATS)}, got {out['format']!r}")
        if out["grid"] not in GRIDS:
            raise ValueError(f"live.export.grid must be one of {list(GRIDS)}, got {out['grid']!r}")
        return out

    @classmethod
    def from_dict(cls, raw: Any) -> "LiveSettings":
        if isinstance(raw, LiveSettings):
            return raw
        if not isinstance(raw, Mapping):
            raise ValueError(f"live: is a mapping, got {type(raw).__name__}")
        if "checks" in raw:
            raise ValueError(
                "live.checks is not a key: the checks before export belong to the pipeline, `model_pipeline.checks: "
                "{class: ForecastChecks, params: ...}` (or ModelPipeline(checks=...)), where they apply in backtests too, so "
                "live and backtest scores stay comparable"
            )
        unknown = sorted(set(raw) - set(LIVE_KEYS))
        if unknown:
            raise ValueError(f"live: unknown key(s) {unknown}; the keys are {list(LIVE_KEYS)}")
        if "save_dir" not in raw:
            raise ValueError("live: needs save_dir, the live directory")
        return cls(**dict(raw))

    def forecast_options(self) -> Dict[str, Any]:
        """The keys the section sets that `ModelPipeline.forecast()` takes, as written (an absent key keeps forecast's default)."""
        return {key: getattr(self, key) for key in FORECAST_KEYS if getattr(self, key) is not None}

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"save_dir": self.save_dir}
        out.update(self.forecast_options())
        if self.export is not None:
            out["export"] = dict(self.export)
        return out

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LiveSettings) and self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return f"LiveSettings({self.to_dict()!r})"
