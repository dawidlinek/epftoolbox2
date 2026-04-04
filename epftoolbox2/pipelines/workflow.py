from typing import Optional, Union
from pathlib import Path
import os
import yaml

from .data_pipeline import DataPipeline
from .model_pipeline import ModelPipeline
from ..results.report import EvaluationReport
from ..logging import get_logger

logger = get_logger(__name__)

_DP_SECTIONS = ("sources", "transformers", "validators")
_MP_SECTIONS = ("models", "evaluators", "exporters")


class Workflow:
    """Orchestrates a full experiment: loads both pipelines from YAML, sets the
    execution environment, runs data fetching, then model training/evaluation.

    Pipeline components can be specified via a path to a pipeline YAML, inline
    in the workflow YAML, or both — inline sections override the corresponding
    section loaded from path.

    `model_index` selects a single model by index from the model list, enabling
    SLURM array jobs where each task runs one model:
        environment:
            model_index: $SLURM_ARRAY_TASK_ID

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
        data_start: str,
        data_end: str,
        model_test_start: Optional[str] = None,
        model_test_end: Optional[str] = None,
        data_pipeline: Optional[Union[str, Path]] = None,
        model_pipeline: Optional[Union[str, Path]] = None,
        data_cache: Union[bool, str] = False,
        model_target: str = "price",
        model_horizon: int = 7,
        model_forecast_only: bool = False,
        max_processes: Optional[int] = None,
        threads_per_process: Optional[int] = None,
        cache_path: Optional[str] = None,
        model_index: Optional[int] = None,
    ):
        self.data_pipeline = str(data_pipeline) if data_pipeline is not None else None
        self.model_pipeline = str(model_pipeline) if model_pipeline is not None else None
        self.data_start = data_start
        self.data_end = data_end
        self.data_cache = data_cache
        self.model_test_start = model_test_start
        self.model_test_end = model_test_end
        self.model_target = model_target
        self.model_horizon = model_horizon
        self.model_forecast_only = model_forecast_only
        self.max_processes = max_processes
        self.threads_per_process = threads_per_process
        self.cache_path = cache_path
        self.model_index = model_index
        self._yaml_dir: Optional[Path] = None  # set by load(), used for relative path resolution
        self._dp_inline: dict = {}  # inline sources/transformers/validators overrides
        self._mp_inline: dict = {}  # inline models/evaluators/exporters overrides

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
                components = [DataPipeline._load_component(section, c) for c in self._dp_inline[section]]
                setattr(dp, section, components)
        return dp

    def _build_model_pipeline(self) -> ModelPipeline:
        if self.model_pipeline is not None:
            mp = ModelPipeline.load(self._resolve(self.model_pipeline))
        else:
            mp = ModelPipeline()
        for section in _MP_SECTIONS:
            if section in self._mp_inline:
                components = [ModelPipeline._load_component(section, c) for c in self._mp_inline[section]]
                setattr(mp, section, components)
        if self.model_index is not None:
            mp.models = [mp.models[self.model_index]]
        return mp

    def run(self) -> EvaluationReport:
        self._apply_environment()

        cache = self.cache_path if self.cache_path else self.data_cache

        df = self._build_data_pipeline().run(
            start=self.data_start,
            end=self.data_end,
            cache=cache,
        )

        report = self._build_model_pipeline().run(
            data=df,
            test_start=self.model_test_start,
            test_end=self.model_test_end,
            target=self.model_target,
            horizon=self.model_horizon,
            forecast_only=self.model_forecast_only,
        )

        return report

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
        if env:
            config["environment"] = env

        dp: dict = {"start": self.data_start, "end": self.data_end, "cache": self.data_cache}
        if self.data_pipeline is not None:
            dp["path"] = self.data_pipeline
        dp.update(self._dp_inline)
        config["data_pipeline"] = dp

        mp: dict = {"target": self.model_target, "horizon": self.model_horizon, "forecast_only": self.model_forecast_only}
        if self.model_test_start is not None:
            mp["test_start"] = self.model_test_start
        if self.model_test_end is not None:
            mp["test_end"] = self.model_test_end
        if self.model_pipeline is not None:
            mp["path"] = self.model_pipeline
        mp.update(self._mp_inline)
        config["model_pipeline"] = mp

        return config

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info(f"Workflow: Saved configuration to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Workflow":
        path = Path(path)
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        env = config.get("environment", {})
        dp = config["data_pipeline"]
        mp = config["model_pipeline"]

        model_index_raw = env.get("model_index")
        model_index = int(os.path.expandvars(str(model_index_raw))) if model_index_raw is not None else None

        workflow = cls(
            data_pipeline=dp.get("path"),
            model_pipeline=mp.get("path"),
            data_start=dp["start"],
            data_end=dp["end"],
            data_cache=dp.get("cache", False),
            model_test_start=mp.get("test_start"),
            model_test_end=mp.get("test_end"),
            model_target=mp.get("target", "price"),
            model_horizon=mp.get("horizon", 7),
            model_forecast_only=mp.get("forecast_only", False),
            max_processes=env.get("max_processes"),
            threads_per_process=env.get("threads_per_process"),
            cache_path=env.get("cache_path"),
            model_index=model_index,
        )
        workflow._yaml_dir = path.parent
        workflow._dp_inline = {k: dp[k] for k in _DP_SECTIONS if k in dp}
        workflow._mp_inline = {k: mp[k] for k in _MP_SECTIONS if k in mp}
        logger.info(f"Workflow: Loaded configuration from {path}")
        return workflow
