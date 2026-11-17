from datetime import date, datetime
from typing import Any, Optional, Sequence, Union
from pathlib import Path
import os
import warnings
import numpy as np
import pandas as pd
import yaml

from .data_pipeline import DataPipeline
from .model_pipeline import ModelPipeline
from ..information_set import InformationSet
from ..models._calendar import every_horizon, issue_day_names, resolve_horizons, resolve_issue_days
from ..models._cards import preflight
from ..models._engine import engine_from_yaml
from ..results.report import EvaluationReport
from ..results.store import DEFAULT_SAVE_FORMAT, check_save_format
from ..logging import get_logger
from ._live_settings import FORECAST_KEYS, LiveSettings
from ._yaml import dump_yaml

logger = get_logger(__name__)

_DP_SECTIONS = ("sources", "transformers", "validators")
_MP_SECTIONS = ("models", "ensembles", "evaluators", "exporters")
_MP_INLINE_KEYS = (*_MP_SECTIONS, "information_set", "availability", "engine", "checks")


def _plain(value: Any) -> Any:
    """`value` as the plain Python values that YAML writes and reads back as themselves, in lists and dicts too.

    A numpy scalar or array becomes a Python number, bool, string or list, and a `pd.Timestamp` or `np.datetime64` its ISO
    text with the time of day and offset: `DataPipeline.run` reads the text `2024-01-31` as the whole day and a Timestamp as
    that instant, so only the full text keeps the instant. Anything else, such as a `datetime.date`, which YAML reads
    natively, stays as it is.
    """
    if isinstance(value, np.datetime64):
        value = pd.Timestamp(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        item = value.item()
        return item if isinstance(item, np.generic) else _plain(item)
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _as_day(day: Any) -> Any:
    """A date with a time of day (`datetime`, `pd.Timestamp`, `np.datetime64`) as its ISO day, as the models read days.
    A string and a `datetime.date` stay as they are."""
    if isinstance(day, np.datetime64):
        day = pd.Timestamp(day)
    return day.date().isoformat() if isinstance(day, datetime) else day


def _as_yaml_period(period: Any) -> Any:
    """A period as YAML takes it: a string stands for itself, and dates in any form become ISO days."""
    if isinstance(period, str):
        return period
    return [day.isoformat() if isinstance(day, date) else day for day in map(_as_day, period)]


def _read_model_index(raw: Any) -> Union[int, str, None]:
    """`model_index` as a number, or as written while a variable in it isn't set.

    `epf report`, `epf array` and `epf validate` read the array's YAML outside the array, where `$SLURM_ARRAY_TASK_ID` isn't
    set and nothing they do needs it. `run()` does, and refuses.
    """
    if raw is None:
        return None
    expanded = os.path.expandvars(str(raw))
    return str(raw) if "$" in expanded else int(expanded)


def refuse_forecast_only_into_save_dir(forecast_only: bool, save_dir: Optional[str]) -> None:
    """A Workflow with `model_pipeline.forecast_only: true` and a `save_dir` is refused when it is made (decision 12 of step 8):
    `run(forecast_only=True)` into a directory resumed the forecast it held and returned it as today's, so it raises, and so
    would every daily run of such a file. An array task's results go to `save_dir`, so a forecast-only one is refused too."""
    if forecast_only and save_dir:
        raise ValueError(
            "model_pipeline has forecast_only: true and save_dir: a forecast-only run into a directory would resume the "
            "forecast it holds and return it as today's, so it is refused. Remove save_dir to keep the forecasts in the "
            "report (forecast_only runs in memory), remove forecast_only to run a backtest of the window into save_dir, or "
            "issue live forecasts into a live directory with ModelPipeline.forecast() (revisions, stored fits, deadlines; "
            "see the docs page Live Forecasts)."
        )


class Workflow:
    """Orchestrates a full experiment: loads both pipelines from YAML, sets the
    execution environment, runs data fetching, then model training/evaluation.

    Pipeline components can be specified via a path to a pipeline YAML, inline
    in the workflow YAML, or both — inline sections override the corresponding
    section loaded from path.

    `model_index` selects a single model by index from the model list, enabling
    SLURM array jobs where each task runs one model. With `models_per_task: k`, task i runs the block of models
    `[i*k, (i+1)*k)` in the order of the list, the last block maybe short, and `array_size()` (CLI: `epf array
    experiment.yaml`) says how many tasks that makes:
        environment:
            model_index: $SLURM_ARRAY_TASK_ID
            models_per_task: 50
        model_pipeline:
            save_dir: results/

    Array mode requires `save_dir`: each task writes one JSONL per model there (resumable) and skips
    ensembles, evaluators and exporters, which need all models. After the array finishes, `report()` (CLI:
    `epf report experiment.yaml`) merges the files, computes the ensembles from them and runs the evaluators
    and exporters once. The block size only says which task runs a model: nothing stored records it, so results
    of an array at one size resume at another and merge to what one run gives, bit for bit.

    The windows of a run live in `model_pipeline` as `validation: [start, end]` (or `previous_year`) and
    `test: [start, end]`, next to the older `test_start` and `test_end`, and mean what they mean in
    `ModelPipeline.run`. A merge raises if the array's tasks ran on different windows.

    `model_pipeline.horizon: H` forecasts every horizon from 1 to H (7 if absent), and `horizons: [1, 3, 7]` (`model_horizons`)
    only those; giving both raises. `report()` passes the same to `ModelPipeline.report`. `issue_days: [friday]`
    (`model_issue_days`) issues forecasts on those weekdays only, as `ModelPipeline.run(issue_days=...)`, and is written only
    when it isn't every day.

    `model_pipeline.save_format: parquet` (`model_save_format`) writes each model's results as a Parquet directory, and
    `save_coefficients: false` (`model_save_coefficients`) writes `coefficients` as `[]`, as `ModelPipeline.run` does; each is
    written only when it isn't the default. `report()` reads either format.

    `model_pipeline.engine: on|off|check` sets the engine (`ModelPipeline(engine=...)`) over what the pipeline YAML says; without
    either, the default (on, a model whose folder was written on the model path resuming there). In array mode each task groups
    the models of its block, and nothing stored depends on the block.
    `model_pipeline.checks: {class: ForecastChecks, params: {...}}` sets the checks before export (`ModelPipeline(checks=...)`)
    over the pipeline YAML's, `checks: null` none.

    Example:
        >>> workflow = Workflow(
        ...     data_pipeline="data_pipeline.yaml",
        ...     model_pipeline="model_pipeline.yaml",
        ...     data_start="2023-05-01",
        ...     data_end="2024-08-01",
        ...     data_cache=True,
        ...     model_test_start="2024-06-01",
        ...     model_test_end="2024-07-01",
        ...     model_target="price",
        ...     model_horizon=7,
        ... )
        >>> workflow.save("experiment.yaml")
        >>> report = Workflow.load("experiment.yaml").run()
    """

    def __init__(
        self,
        data_start: Optional[str],
        data_end: Optional[str],
        model_test_start: Optional[str] = None,
        model_test_end: Optional[str] = None,
        data_pipeline: Optional[Union[str, Path]] = None,
        model_pipeline: Optional[Union[str, Path]] = None,
        data_cache: Union[bool, str] = False,
        model_target: str = "price",
        model_horizon: Optional[int] = None,
        model_freq: str = "1h",
        model_forecast_only: bool = False,
        max_processes: Optional[int] = None,
        threads_per_process: Optional[int] = None,
        cache_path: Optional[str] = None,
        model_index: Optional[Union[int, str]] = None,
        model_save_dir: Optional[str] = None,
        model_allow_manifest_mismatch: bool = False,
        model_require_manifest: bool = False,
        model_validation: Any = None,
        model_test: Optional[Sequence[Any]] = None,
        models_per_task: Optional[int] = None,
        model_horizons: Optional[Sequence[int]] = None,
        model_issue_days: Optional[Sequence[str]] = None,
        model_save_format: str = DEFAULT_SAVE_FORMAT,
        model_save_coefficients: bool = True,
        live: Union[None, dict, LiveSettings] = None,
    ):
        if live is None and (data_start is None or data_end is None):
            raise ValueError(
                "data_start and data_end are needed to run a backtest: a workflow without them is live-only"
            )
        resolve_horizons(model_horizon, model_horizons, 7)  # both, or a list that isn't one, raise before anything runs
        resolve_issue_days(model_issue_days)
        check_save_format(model_save_format)
        self.data_pipeline = str(data_pipeline) if data_pipeline is not None else None
        self.model_pipeline = str(model_pipeline) if model_pipeline is not None else None
        self.data_start = data_start
        self.data_end = data_end
        self.data_cache = data_cache
        self.model_test_start = model_test_start
        self.model_test_end = model_test_end
        self.model_target = model_target
        self.model_horizon = model_horizon
        self.model_horizons = model_horizons
        self.model_issue_days = model_issue_days
        self.model_freq = model_freq
        self.model_forecast_only = model_forecast_only
        self.max_processes = max_processes
        self.threads_per_process = threads_per_process
        self.cache_path = cache_path
        self.model_index = _read_model_index(model_index)
        self.models_per_task = models_per_task
        self.model_save_dir = str(model_save_dir) if model_save_dir is not None else None
        self.model_allow_manifest_mismatch = model_allow_manifest_mismatch
        self.model_require_manifest = model_require_manifest
        self.model_validation = model_validation
        self.model_test = model_test
        self.model_save_format = model_save_format
        self.model_save_coefficients = model_save_coefficients
        self.live = None if live is None else LiveSettings.from_dict(live)
        if self.live is not None and self.live.on_failure == "cache" and (cache_path or isinstance(data_cache, str)):
            raise ValueError(
                "live.on_failure: cache falls back to the per-source caches, which a single-file cache "
                "(data_pipeline.cache: <path> or environment.cache_path) doesn't have: use data_pipeline.cache: true, or "
                "another on_failure"
            )
        refuse_forecast_only_into_save_dir(self.model_forecast_only, self.model_save_dir)
        self._yaml_dir: Optional[Path] = None  # set by load(), used for relative path resolution
        self._dp_inline: dict = {}  # inline sources/transformers/validators overrides
        self._mp_inline: dict = {}  # inline models/evaluators/exporters overrides

    @property
    def models_per_task(self) -> Optional[int]:
        """How many models an array task runs: task `model_index` runs the k models from `model_index * k` on. None is one."""
        return self._models_per_task

    @models_per_task.setter
    def models_per_task(self, k: Optional[int]) -> None:
        if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 1):
            raise ValueError(f"models_per_task must be a positive integer, got {k!r}")
        self._models_per_task = k

    def array_size(self) -> int:
        """How many array tasks cover every model of the pipeline: indices 0 to `array_size() - 1`, whatever `model_index` is.

        `epf array experiment.yaml` prints that range as `sbatch --array=` takes it.
        """
        n = len(self._build_model_pipeline(all_models=True).models)
        if n == 0:
            raise ValueError("At least one model is required")
        return self._tasks_for(n)

    def _tasks_for(self, n_models: int) -> int:
        k = self.models_per_task or 1
        return (n_models + k - 1) // k

    def _block_of(self, models: list) -> list:
        """The models of task `model_index`, in the order of the list; an index past the last block raises."""
        k = self.models_per_task or 1
        first = self.model_index * k
        if not 0 <= first < len(models):
            per_task = f", {k} per task" if self.models_per_task is not None else ""
            valid = f"; the valid indices are 0-{self._tasks_for(len(models)) - 1}" if models else ""
            raise ValueError(
                f"model_index {self.model_index} is out of range: the pipeline has {len(models)} model(s){per_task}{valid}"
            )
        return models[first : first + k]

    def _resolve(self, relative_path: str) -> Path:
        """Resolve a path relative to the workflow YAML's directory."""
        p = Path(os.path.expandvars(relative_path))
        if not p.is_absolute() and self._yaml_dir is not None:
            return self._yaml_dir / p
        return p

    def _apply_environment(self) -> None:
        """Set env vars before any numpy/sklearn imports in worker processes."""
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["BLAS_NUM_THREADS"] = "1"
        os.environ["PYTHON_GIL"] = "0"
        if self.max_processes is not None:
            os.environ["MAX_PROCESSES"] = str(self.max_processes)
        if self.threads_per_process is not None:
            os.environ["THREADS_PER_PROCESS"] = str(self.threads_per_process)

    def _build_data_pipeline(self) -> DataPipeline:
        if self.data_pipeline is not None:
            dp = DataPipeline.load(self._resolve(self.data_pipeline))
        else:
            dp = DataPipeline()
        for section in _DP_SECTIONS:
            if section in self._dp_inline:
                # A relative file path of an inline component is the YAML's, as `save_dir` and a pipeline file's are.
                components = [
                    DataPipeline._load_component(section, c, base_dir=self._yaml_dir) for c in self._dp_inline[section]
                ]
                setattr(dp, section, components)
        return dp

    def _build_model_pipeline(self, all_models: bool = False) -> ModelPipeline:
        if self.model_pipeline is not None:
            mp = ModelPipeline.load(self._resolve(self.model_pipeline))
        else:
            mp = ModelPipeline()
        for section in _MP_SECTIONS:
            if section in self._mp_inline:
                components = [ModelPipeline._load_component(section, c) for c in self._mp_inline[section]]
                setattr(mp, section, components)
        if "information_set" in self._mp_inline:
            info = self._mp_inline["information_set"]
            mp.information_set = InformationSet.from_dict(info) if info else None
        if "availability" in self._mp_inline:
            mp.availability = self._mp_inline["availability"]
        if "engine" in self._mp_inline:
            mp.engine = engine_from_yaml(self._mp_inline["engine"])  # `engine: on` unquoted reads as True
        if "checks" in self._mp_inline:  # step 8f: `checks: {class: ForecastChecks, params: ...}`, or null for none
            checks = self._mp_inline["checks"]
            mp.checks = ModelPipeline._load_component("checks", checks) if checks else None
        if isinstance(self.model_index, int) and not all_models:
            mp._resolve_bases()  # a ResidualModel's base="name" is a model of the whole list, maybe of another task's slice
            mp.models = self._block_of(mp.models)
            mp._bases_from_the_whole_list = True
        return mp

    def _save_dir(self) -> Optional[Path]:
        return self._resolve(self.model_save_dir) if self.model_save_dir else None

    def _run_data(self, dp: Optional[DataPipeline] = None):  # noqa: ANN202
        if self.data_start is None or self.data_end is None:
            raise ValueError("data_pipeline.start and end are not set: this workflow is live-only (epf forecast)")
        cache = self.cache_path if self.cache_path else self.data_cache
        dp = dp if dp is not None else self._build_data_pipeline()
        return dp.run(start=self.data_start, end=self.data_end, cache=cache)

    def run(self) -> EvaluationReport:
        refuse_forecast_only_into_save_dir(self.model_forecast_only, self.model_save_dir)  # set after load, too
        self._apply_environment()

        if isinstance(self.model_index, str):
            raise ValueError(
                f"model_index is {self.model_index!r}, and that variable isn't set, so this is no array task. Run inside "
                "the array, or give the index: Workflow.model_index, or `epf run --model-index`."
            )
        if self.models_per_task is not None and self.model_index is None:
            warnings.warn(
                f"models_per_task={self.models_per_task} has no effect without model_index: this runs the whole pipeline.",
                stacklevel=2,
            )

        save_dir = self._save_dir()
        array_task = self.model_index is not None
        if array_task and save_dir is None:
            raise ValueError(
                "model_index runs part of the models (array mode) and needs model_pipeline.save_dir, where each task "
                "writes its results for `epf report` / Workflow.report() to merge."
            )
        mp = self._build_model_pipeline()
        # This task's models' extras, runtime packages and weights, before the data is fetched (`models._cards.preflight`).
        preflight(mp.models)
        # A ResidualModel's base="name" and results format, over every model of the pipeline, before the data is fetched: an
        # array task's slice may hold the base alone.
        (self._build_model_pipeline(all_models=True) if array_task else mp)._check_residual_models(
            self.model_save_format
        )
        if array_task:
            # Better now than when `epf report` runs, after every task has.
            self._build_model_pipeline(all_models=True)._check_before_any_fit(
                save_dir,
                self.model_validation,
                self.model_test,
                self.model_test_start,
                self.model_test_end,
                self.model_forecast_only,
            )
            # Ensembles, evaluators such as rMAE and the exporters need every model; report() runs them after all tasks.
            mp.ensembles, mp.evaluators, mp.exporters = [], [], []
        dp = self._build_data_pipeline()
        if mp.information_set is not None:
            # Knows the columns of the configured sources, e.g. a CalendarSource prefix or Open-Meteo variables, and a prefixed
            # SolarGeometryTransformer's.
            mp.information_set = mp.information_set.with_source_rules([*dp.sources, *dp.transformers])

        data = self._run_data(dp)
        if array_task and not self.model_forecast_only:
            # The evaluators run in report(), after every task: a column they read that the data lacks is refused now.
            self._build_model_pipeline(all_models=True)._check_source_columns(data)
        report = mp.run(
            data=data,
            test_start=self.model_test_start,
            test_end=self.model_test_end,
            target=self.model_target,
            horizon=self.model_horizon,
            horizons=self.model_horizons,
            issue_days=self.model_issue_days,
            freq=self.model_freq,
            save_dir=str(save_dir) if save_dir else None,
            forecast_only=self.model_forecast_only,
            allow_manifest_mismatch=self.model_allow_manifest_mismatch,
            require_manifest=self.model_require_manifest,
            validation=self.model_validation,
            test=self.model_test,
            save_format=self.model_save_format,
            save_coefficients=self.model_save_coefficients,
        )
        if array_task:
            logger.info(
                f"Workflow: task {self.model_index} ({', '.join(model.name for model in mp.models)}) written to {save_dir}; "
                "run `epf report` after all array tasks finish."
            )
        return report

    def report(self, allow_missing: bool = False) -> EvaluationReport:
        """Merge every model's results from `model_save_dir`, compute the ensembles from them, and run the
        evaluators and exporters.

        Use after an array job in which each task ran one model or a block of them. Raises FileNotFoundError naming each
        model whose results file is missing, unless `allow_missing` is set, which leaves out that model and every ensemble
        it is a member of. The data pipeline is only run when an exporter joins source columns (CsvExporter
        `extra_columns`) or an evaluator reads them (`ScaledMAEEvaluator`).
        """
        save_dir = self._save_dir()
        if save_dir is None:
            raise ValueError("report() reads the results from model_pipeline.save_dir, which is not set")
        mp = self._build_model_pipeline(all_models=True)
        needs_data = any(getattr(e, "extra_columns", None) for e in mp.exporters) or (
            not self.model_forecast_only and any(getattr(e, "source_columns", ()) for e in mp.evaluators)
        )
        return mp.report(
            save_dir,
            test_start=self.model_test_start,
            test_end=self.model_test_end,
            horizon=self.model_horizon,
            horizons=self.model_horizons,
            issue_days=self.model_issue_days,
            freq=self.model_freq,
            forecast_only=self.model_forecast_only,
            data=self._run_data() if needs_data else None,
            allow_missing=allow_missing,
            validation=self.model_validation,
            test=self.model_test,
        )

    # --- live calls (step 8i): `epf forecast`, `refit`, `status`, `mark-delivered` and `monitor` -------------------------------

    def _live_settings(self) -> LiveSettings:
        if self.live is None:
            raise ValueError(
                "this workflow has no live: section (live.save_dir at least): see the docs page Live Forecasts"
            )
        if self.model_index is not None:
            raise ValueError(
                "live calls run the whole pipeline: a workflow with environment.model_index is an array task"
            )
        return self.live

    def _live_pipelines(self) -> tuple:
        """The data pipeline and the model pipeline of a live call: as `run` builds them, with the information set knowing the
        configured sources' columns."""
        self._live_settings()
        self._apply_environment()
        mp = self._build_model_pipeline()
        dp = self._build_data_pipeline()
        if mp.information_set is not None:
            mp.information_set = mp.information_set.with_source_rules([*dp.sources, *dp.transformers])
        preflight(mp.models)
        return dp, mp

    def _live_dir(self) -> Path:
        return self._resolve(self._live_settings().save_dir)

    def _live_cache(self) -> Any:
        """The cache of a live refresh: the workflow's cache path or setting, and the per-source caches when it sets none (the
        fallback to the cache, `on_failure="cache"`, needs them)."""
        return (self.cache_path or self.data_cache) or True

    def _live_data_options(self) -> dict:
        """What a refresh of the data pipeline takes besides its range (`refit` and `monitor`): the live section's limits and
        the cache. `forecast()` takes the same from its own arguments."""
        from ..pipelines._live import seconds

        live = self._live_settings()
        options = {
            "cache": self._live_cache(),
            "budget": seconds(live.budget, "budget"),
            "timeout": seconds(live.timeout, "timeout"),
            "retries": live.retries,
            "on_failure": live.on_failure,
        }
        return {key: value for key, value in options.items() if value is not None}

    def _run_shape(self) -> dict:
        """The run arguments `forecast`, `refit` and `monitor` share with `run`."""
        return {
            "target": self.model_target,
            "horizon": self.model_horizon,
            "horizons": self.model_horizons,
            "freq": self.model_freq,
            "issue_days": self.model_issue_days,
        }

    def forecast(
        self,
        issue: Any = "today",
        *,
        ensure: bool = False,
        refresh: str = "auto",
        dry_run: bool = False,
        **options: Any,
    ):
        """`ModelPipeline.forecast` of the live section: the data pipeline refreshed, the models served into `live.save_dir`.

        The keys of the section that `forecast()` takes reach it as written; `options` (the same names) replace them for this
        call. Returns the `ForecastResult`; `export_result` writes `live.export`."""
        live = self._live_settings()
        unknown = sorted(set(options) - set(FORECAST_KEYS))
        if unknown:
            raise TypeError(f"forecast() takes the live keys {list(FORECAST_KEYS)}, not {unknown}")
        dp, mp = self._live_pipelines()
        merged = {**live.forecast_options(), **{key: value for key, value in options.items() if value is not None}}
        return mp.forecast(
            dp,
            issue,
            self._live_dir(),
            ensure=ensure,
            refresh=refresh,
            dry_run=dry_run,
            save_format=self.model_save_format,
            cache=self._live_cache(),
            **merged,
            **self._run_shape(),
        )

    def export_result(self, result: Any) -> Optional[Path]:
        """Write `live.export` for `result` (path with `{issue}` filled in, relative to the YAML): the delivered model's
        forecasts. Nothing for a call that has none (busy, late), a dry run, or without an export section. A noop (`--ensure`)
        or unchanged call exports from the committed revision's records only when the file for the issue is missing, so
        a retry after a failed export writes it and a repeat call leaves a delivered file alone. Returns the path written."""
        export = self._live_settings().export
        if export is None or result.dry_run or result.status in ("busy", "late"):
            return None
        path = self._resolve(export["path"].format(issue=result.issue))
        if (result.noop or result.unchanged) and path.exists():
            return None
        return result.export(path, format=export["format"], grid=export["grid"])

    def refit(self, issue: Any = "today", *, if_due: bool = True, refresh: str = "auto"):
        """`ModelPipeline.refit` into `live.save_dir` for the issue day: the data pipeline refreshed over the range the fits
        read (the live section's budget, timeout, retries and on_failure); `refresh="never"` reads the per-source caches alone."""
        if refresh not in ("auto", "never"):
            raise ValueError(f"refresh must be 'auto' or 'never', got {refresh!r}")
        from ..pipelines._live import data_span, issue_zone, resolve_issue

        dp, mp = self._live_pipelines()
        _, horizons = resolve_horizons(self.model_horizon, self.model_horizons, 7)
        day = resolve_issue(issue, issue_zone(mp.information_set, dp))
        options = self._live_data_options()
        if refresh == "never":  # as forecast(refresh="never"): no time for any request
            options.update(cache=True, budget=0, on_failure="cache")
        frame = dp.run(*data_span(mp.models, day, horizons, dp), **options)
        return mp.refit(frame, self._live_dir(), issue=day, if_due=if_due, **self._run_shape())

    def _live_store(self) -> Any:
        """The live directory's store, for the calls that read files alone: no pipeline is built, no extra imported, no key read."""
        from ..results.live import LiveStore
        from ..results.store import LiveDirectoryError

        store = LiveStore(self._live_dir())
        if not store.is_live():
            raise LiveDirectoryError(f"{store.save_dir} is no live directory yet: nothing has been forecast into it")
        return store

    def _issue_day(self, issue: Any, store: Any) -> Any:
        """`issue` as a date: "today" is read in the issue zone `live.json` records."""
        from ..pipelines._live import resolve_issue

        zone = store.zone()
        if zone is None:
            raise ValueError(
                f"{store.identity_path} records no issue_timezone, so 'today' can't be read: give the date"
            )
        return resolve_issue(issue, zone)

    def mark_delivered(self, issue: Any, revision: int, receipt: Union[str, Path], dry_run: bool = False) -> dict:
        """Record that `revision` of the issue day was delivered, as `ModelPipeline.mark_delivered` does, from the live directory
        alone; `issue` may be "today" (in the zone `live.json` records)."""
        store = self._live_store()
        day = self._issue_day(issue, store)
        with store.hold(create=not dry_run):
            return store.mark_delivered(day, revision, receipt, dry_run=dry_run)

    def status(self, issue: Any = "today") -> dict:
        """What the live directory holds for one issue day: its call lines in order, its committed revisions, its deliveries,
        the newest call's deadline and the newest committed revision's status. Reads the live directory's files only: no pipeline is built."""
        store = self._live_store()
        day = self._issue_day(issue, store).isoformat()
        calls = [line for line in store.calls() if line.get("issue") == day]
        deadlines = [line["deadline"] for line in calls if line.get("deadline")]
        committed = store.committed(day).get(day, {})
        latest = committed[max(committed)] if committed else None
        return {
            "issue": day,
            "calls": calls,
            "committed": sorted(committed),
            "latest": None if latest is None else {"revision": latest.revision, "status": latest.line.get("status")},
            "deliveries": store.deliveries(day),
            "deadline": deadlines[-1] if deadlines else None,  # the newest call's, an instant
        }

    def monitor(self, windows: Sequence[int] = (7, 30, 90), alerts: Any = None, policy: str = "latest"):
        """`ModelPipeline.monitor` of `live.save_dir`: the actuals are read from the data pipeline over the days the directory has
        revisions for, the cut-off is the live deadline (the one `live.json` records without one), and the weekdays are the
        pipeline's issue days."""
        from ..results.live import LiveStore

        dp, mp = self._live_pipelines()
        live = self._live_settings()
        store = LiveStore(self._live_dir())
        if not store.is_live():
            raise ValueError(f"{store.save_dir} is no live directory yet: nothing has been forecast into it")
        days = sorted(store.committed())
        if not days:
            raise ValueError(f"{store.save_dir} has no committed revision yet: nothing to monitor")
        _, horizons = resolve_horizons(self.model_horizon, self.model_horizons, 7)
        end = pd.Timestamp(days[-1]) + pd.Timedelta(days=max(horizons) + 2)
        frame = dp.run(days[0], end.date().isoformat(), **self._live_data_options())
        return mp.monitor(
            frame,
            store.save_dir,
            windows=windows,
            policy=policy,
            cutoff=live.deadline or "deadline",
            alerts=alerts,
            target=self.model_target,
            issue_days=self.model_issue_days,
        )

    def to_dict(self) -> dict:
        config: dict = {}

        env: dict = {}
        if self.max_processes is not None:
            env["max_processes"] = self.max_processes
        if self.threads_per_process is not None:
            env["threads_per_process"] = self.threads_per_process
        if self.cache_path is not None:
            env["cache_path"] = self.cache_path
        if self.model_index is not None:
            env["model_index"] = self.model_index
        if self.models_per_task is not None:
            env["models_per_task"] = self.models_per_task
        if env:
            config["environment"] = env

        dp: dict = {}
        if self.data_start is not None:  # a live-only workflow has none: forecast() computes the range from the models
            dp["start"] = self.data_start
        if self.data_end is not None:
            dp["end"] = self.data_end
        dp["cache"] = self.data_cache
        if self.data_pipeline is not None:
            dp["path"] = self.data_pipeline
        dp.update(self._dp_inline)
        config["data_pipeline"] = dp

        mp: dict = {"target": self.model_target}
        horizon, horizons = resolve_horizons(self.model_horizon, self.model_horizons, 7)
        if every_horizon(horizon, horizons):  # range(1, H + 1) is horizon: H
            mp["horizon"] = horizon
        else:
            mp["horizons"] = list(horizons)
        weekdays = resolve_issue_days(self.model_issue_days)
        if weekdays is not None:  # every day writes nothing
            mp["issue_days"] = issue_day_names(weekdays)
        mp.update({"freq": self.model_freq, "forecast_only": self.model_forecast_only})
        if self.model_test_start is not None:
            mp["test_start"] = _as_day(self.model_test_start)
        if self.model_test_end is not None:
            mp["test_end"] = _as_day(self.model_test_end)
        if self.model_validation is not None:
            mp["validation"] = _as_yaml_period(self.model_validation)
        if self.model_test is not None:
            mp["test"] = _as_yaml_period(self.model_test)
        if self.model_save_dir is not None:
            mp["save_dir"] = self.model_save_dir
        if self.model_save_format != DEFAULT_SAVE_FORMAT:
            mp["save_format"] = self.model_save_format
        if not self.model_save_coefficients:
            mp["save_coefficients"] = False
        if self.model_allow_manifest_mismatch:
            mp["allow_manifest_mismatch"] = True
        if self.model_require_manifest:
            mp["require_manifest"] = True
        if self.model_pipeline is not None:
            mp["path"] = self.model_pipeline
        mp.update(self._mp_inline)
        config["model_pipeline"] = mp
        if self.live is not None:
            config["live"] = self.live.to_dict()

        return _plain(config)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        config = self.to_dict()
        path.write_text(dump_yaml(config))
        logger.info(f"Workflow: Saved configuration to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Workflow":
        path = Path(path)
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        env = config.get("environment", {})
        dp = config["data_pipeline"]
        mp = config["model_pipeline"]

        if "live" in config and config["live"] is None:
            raise ValueError("live: needs save_dir, the live directory")
        workflow = cls(
            data_pipeline=dp.get("path"),
            model_pipeline=mp.get("path"),
            data_start=dp["start"] if "live" not in config else dp.get("start"),
            data_end=dp["end"] if "live" not in config else dp.get("end"),
            data_cache=dp.get("cache", False),
            model_test_start=mp.get("test_start"),
            model_test_end=mp.get("test_end"),
            model_target=mp.get("target", "price"),
            model_horizon=mp.get("horizon"),
            model_horizons=mp.get("horizons"),
            model_issue_days=mp.get("issue_days"),
            model_freq=mp.get("freq", "1h"),
            model_forecast_only=mp.get("forecast_only", False),
            max_processes=env.get("max_processes"),
            threads_per_process=env.get("threads_per_process"),
            cache_path=env.get("cache_path"),
            model_index=env.get("model_index"),
            model_save_dir=mp.get("save_dir"),
            model_allow_manifest_mismatch=mp.get("allow_manifest_mismatch", False),
            model_require_manifest=mp.get("require_manifest", False),
            model_validation=mp.get("validation"),
            model_test=mp.get("test"),
            models_per_task=env.get("models_per_task"),
            model_save_format=mp.get("save_format", DEFAULT_SAVE_FORMAT),
            model_save_coefficients=mp.get("save_coefficients", True),
            live=config.get("live"),
        )
        workflow._yaml_dir = path.parent
        workflow._dp_inline = {k: dp[k] for k in _DP_SECTIONS if k in dp}
        workflow._mp_inline = {k: mp[k] for k in _MP_INLINE_KEYS if k in mp}
        if "engine" in workflow._mp_inline:  # as written: "on" or "off", so that a save writes the word back
            workflow._mp_inline["engine"] = engine_from_yaml(workflow._mp_inline["engine"])
        logger.info(f"Workflow: Loaded configuration from {path}")
        return workflow
