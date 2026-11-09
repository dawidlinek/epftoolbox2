from contextlib import contextmanager
from dataclasses import replace
from typing import Callable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union, Dict, Any
from pathlib import Path
import copy
import datetime
import importlib
import inspect
import json
import re
import warnings
import pandas as pd
import yaml

from .._date_utils import resolve_date
from ..information_set import DATA_ZONE, InformationSet, InformationSetError, check_level
from ..models._calendar import (
    DAILY,
    IssueCalendar,
    check_issue_days_in,
    horizon_arguments,
    issue_day_names,
    resolve_horizons,
    resolve_issue_days,
)
from ..models.base import BaseModel, _RunChecks
from ..models._worker import pool_native_threads
from ..models._engine import (
    ENGINE_CHECK,
    ENGINE_OFF,
    check_engine,
    engine_from_yaml,
    engine_groups,
    run_arguments,
    run_group,
)
from ..models._torch.groups import check_settings as check_stack_settings
from ..models._torch.groups import run_stack_group, stack_groups
from ..models.context import PredictorPlan, importable_path, load_function
from ..models.explain import explain_model, grouped_table
from ..models.pooled import PooledModel, check_pooled_model
from ..models.residual import ResidualModel
from ..models._pooled import (
    check_horizon_lists,
    check_resumes,
    member_of,
    needed_columns,
    run_pooled,
    shared_fit_checks,
    verify_pooled,
    with_missing_columns,
)
from ..models._cards import preflight, read_provenance
from ..ensembles._compute import compute
from ..ensembles.base import Ensemble
from ..evaluators.base import Evaluator
from ..evaluators.dm import DMEvaluator
from ..evaluators.economic import EconomicEvaluator
from ..exporters.base import Exporter
from ..results.availability import read_listing, warn_about
from ..results.manifest import (
    FrameHashes,
    ManifestMismatchError,
    is_ensemble_manifest,
    manifest_path,
    read_freq,
    read_horizons,
    read_issue_days,
    read_matching_lists,
    read_skipped,
    read_windows,
    read_zone,
    rows_unverified,
)
from ..results.matching import rows_of_refs, warn_about_thin
from ..results.panel import PanelReport
from ..results.report import EvaluationReport
from ..results.ref import ModelResultRef, issued_mae, split_by_window, warn_off_the_clock
from ..results.store import (
    DEFAULT_SAVE_FORMAT,
    SAVE_FORMATS,
    ResultsFormatError,
    check_no_other_format,
    check_save_format,
    read_records,
)
from ..results.windows import SKIP_TARGET_IN_TEST, TEST, RunWindows, check_validation_every, to_period
from ..results.vintages import read_vintages
from ..verification import VerifyReport, sample_issue_days, verify_model
from ..logging import get_logger
from ._yaml import class_path, dump_yaml, to_yaml_value

logger = get_logger(__name__)

# A zone of `run(data={zone: frame})` names a folder of `save_dir`.
_ZONE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
# The zone index a run over zones writes into its save_dir: the zones in the order first run. It alone makes a save_dir one of
# zones for `report`; no folder layout does.
ZONE_INDEX = "zone_run.json"


def read_zone_index(save_dir: Union[str, Path]) -> Optional[List[str]]:
    """The zones a run over zones recorded in `save_dir`, in order, or None if it holds no zone index."""
    path = Path(save_dir) / ZONE_INDEX
    if not path.is_file():
        return None
    try:
        zones = json.loads(path.read_text()).get("zones")
    except (OSError, ValueError, AttributeError) as exc:
        raise ValueError(f"{path} is no zone index of a run over zones: {exc}") from exc
    if not isinstance(zones, list) or not all(isinstance(zone, str) and _ZONE_NAME.fullmatch(zone) for zone in zones):
        raise ValueError(f"{path} is no zone index of a run over zones: its 'zones' must be a list of zone names")
    return zones


def check_zone_case(save_dir: Union[str, Path], zones: Sequence[str]) -> None:
    """A ValueError if a zone differs only in letter case from one `save_dir`'s zone index holds: on a file system that folds
    case (macOS, Windows) both are one folder, and the index would count it twice."""
    indexed = read_zone_index(save_dir) or []
    folded = {zone.lower(): zone for zone in indexed}
    clashes = [(zone, folded[zone.lower()]) for zone in zones if zone not in indexed and zone.lower() in folded]
    if clashes:
        shown = ", ".join(f"{zone!r} (the index has {other!r})" for zone, other in clashes)
        raise ValueError(
            f"{Path(save_dir)} holds a run over zones whose names differ from these only in case: {shown}. A folder on a file "
            "system that folds case is one for both: use the name the index has."
        )


def write_zone_index(save_dir: Union[str, Path], zones: Sequence[str]) -> None:
    """Record `zones` in `save_dir`'s zone index, after those it holds: a later run over fewer keeps the others. A zone that
    differs from an indexed one only in case raises (`check_zone_case`)."""
    root = Path(save_dir)
    check_zone_case(root, zones)
    root.mkdir(parents=True, exist_ok=True)
    merged = list(dict.fromkeys([*(read_zone_index(root) or []), *zones]))
    text = {
        "zones": merged,
        "about": "epftoolbox2: a run over zones, one folder per zone (ModelPipeline.run(data={zone: frame}))",
    }
    (root / ZONE_INDEX).write_text(json.dumps(text, indent=1) + "\n")


@contextmanager
def _zone_errors(zone: str) -> Iterator[None]:
    """A ValueError, TypeError or RuntimeError (a model that made no forecast) raised for one zone of a run over several, with
    the zone named first ("zone 'FI': ..."), of the same type where it can be built from a message alone."""
    try:
        yield
    except (ValueError, TypeError, RuntimeError) as exc:
        if f"zone {zone!r}" in str(exc):
            raise
        try:
            named = type(exc)(f"zone {zone!r}: {exc}")
        except Exception:
            raise exc from None
        raise named from exc


def _check_days_in_frame(
    frame: pd.DataFrame, windows: Optional[RunWindows], test_start: Optional[str], test_end: Optional[str]
) -> None:
    """A ValueError unless every first and last day the run names (each window's, or the test's) is a local date of `frame`, as
    `BaseModel._resolve_day` requires: checked for every zone before any zone's files are written."""
    named = [(f"{name} start", start) for name, (start, _) in windows.periods] if windows is not None else []
    named += [(f"{name} end", end) for name, (_, end) in windows.periods] if windows is not None else []
    if windows is None:
        named = [("test_start", test_start), ("test_end", test_end)]
    if not len(frame.index):
        raise ValueError("the frame has no rows")
    local = frame.index.tz_localize(None) if frame.index.tz is not None else frame.index
    dates = set(local.normalize().date)
    for label, day in named:
        if day is not None and datetime.date.fromisoformat(str(day)) not in dates:
            last = local.max().date()
            raise ValueError(
                f"'{label}' date '{day}' is not in the data (data ends at {last}). Fetch data through at least '{day}'."
            )


COMPONENT_REGISTRY = {
    "models": {
        "OLSModel": "epftoolbox2.models.ols",
        "LassoCVModel": "epftoolbox2.models.lasso",
        "RidgeModel": "epftoolbox2.models.ridge",
        "LEARModel": "epftoolbox2.models.lear",
        "NaiveModel": "epftoolbox2.models.naive",
        "PooledModel": "epftoolbox2.models.pooled",
        "TorchModel": "epftoolbox2.models.torch_model",
        "MLPHourlyModel": "epftoolbox2.models.mlp",
        "MLPDailyModel": "epftoolbox2.models.mlp",
        "HistGBModel": "epftoolbox2.models.hgb",
        "LightGBMModel": "epftoolbox2.models.gbm",
        "XGBoostModel": "epftoolbox2.models.gbm",
        "CatBoostModel": "epftoolbox2.models.gbm",
        "ResidualModel": "epftoolbox2.models.residual",
        "ShapeModel": "epftoolbox2.models.shape",
    },
    "ensembles": {
        "AverageEnsemble": "epftoolbox2.ensembles.average",
    },
    "evaluators": {
        "MAEEvaluator": "epftoolbox2.evaluators.mae",
        "RMSEEvaluator": "epftoolbox2.evaluators.rmse",
        "rMAEEvaluator": "epftoolbox2.evaluators.rmae",
        "BiasEvaluator": "epftoolbox2.evaluators.bias",
        "DMEvaluator": "epftoolbox2.evaluators.dm",
        "MCSEvaluator": "epftoolbox2.evaluators.mcs",
        "EconomicEvaluator": "epftoolbox2.evaluators.economic",
        "ShapeMAEEvaluator": "epftoolbox2.evaluators.shape",
        "NMAEEvaluator": "epftoolbox2.evaluators.normalised",
        "ScaledMAEEvaluator": "epftoolbox2.evaluators.normalised",
    },
    # What `EconomicEvaluator(strategy=...)` runs; saved inside the evaluator as `{class: ..., params: ...}`.
    "strategies": {
        "BatteryArbitrage": "epftoolbox2.evaluators.economic",
    },
    "exporters": {
        "CsvExporter": "epftoolbox2.exporters.csv",
        "ExcelExporter": "epftoolbox2.exporters.excel",
        "TerminalExporter": "epftoolbox2.exporters.terminal",
    },
}


def refuse_searched_on_test(models: Sequence[Any], test_start: Optional[str]) -> None:
    """A ValueError, before any fit, for a model a search chose (`tune.SearchResult.best`) on data through a day on or after
    the first test issue day: its test forecasts would be scored on days its settings were chosen on, as `RunWindows.fixed`
    refuses a validation period that doesn't end before the test starts."""
    if not test_start:
        return
    first = str(resolve_date(test_start))[:10]
    for model in models:
        search = getattr(model.model if isinstance(model, PooledModel) else model, "_search", None)
        through = search.get("data_through") if isinstance(search, dict) else None  # a model no search chose has none
        if through is not None and str(through) >= first:
            raise ValueError(
                f"{model.name}: the search {search.get('search')!r} chose it on data through {through} (its validation period "
                f"{tuple(search.get('validation') or ())}), and this run's test period starts on {first}: a test forecast must not "
                "be scored on days its settings were chosen on. Start the test after the validation end, or search on an "
                "earlier validation period."
            )


def _forecasts_of(ref: ModelResultRef, made: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """What a model's run forecast: target, horizons, freq and issue days, from the manifest of its results (what a report
    reads), else `made` (what `run` was told); None if neither says."""
    path = None if ref.path is None else manifest_path(ref.path)
    if path is not None and path.exists():
        stored = json.loads(path.read_text())
        horizon = stored.get("horizon")
        return {
            "target": stored.get("target"),
            "horizons": list(stored.get("horizons") or range(1, int(horizon or 0) + 1)),
            "freq": stored.get("freq"),
            "issue_days": stored.get("issue_days"),
        }
    return None if made is None else dict(made)


def search_choices(
    refs: Mapping[str, ModelResultRef], test: Optional[Tuple[str, str]], made: Optional[Mapping[str, Any]] = None
) -> Dict[str, List[Dict]]:
    """The rows of `report.choices()` of the models a search chose (`tune.SearchResult.best`, whose ref's provenance holds
    `search`), keyed `{search}:{model}`: the search's validation numbers, and the model's MAE on the `test` period of the run,
    from every forecast it has there (every horizon it made), so a report of any part of the run shows the same. A warning
    names a model whose run forecasts another target, horizons, freq or issue days than its search scored (`made`, or its
    manifest): its gap would compare two kinds of forecast."""
    out: Dict[str, List[Dict]] = {}
    for name, ref in refs.items():
        provenance = getattr(ref, "provenance", None)  # a stand-in ref may have none
        search = provenance.get("search") if isinstance(provenance, dict) else None
        if not isinstance(search, dict) or not search:
            continue
        forecasts = _forecasts_of(ref, made)
        if forecasts is not None and "target" in search:
            searched = {key: search.get(key) for key in ("target", "horizons", "freq", "issue_days")}
            differ = [key for key, value in searched.items() if json.dumps(value) != json.dumps(forecasts[key])]
            if differ:
                warnings.warn(
                    f"{name}: the search {search.get('search')!r} scored it on "
                    f"{', '.join(f'{key} {searched[key]}' for key in differ)}, and this run forecasts "
                    f"{', '.join(f'{key} {forecasts[key]}' for key in differ)}: its gap in report.choices() compares two kinds of "
                    "forecast.",
                    UserWarning,
                    stacklevel=3,
                )
        whole = replace(
            ref,
            test_start=test[0] if test else "",
            test_end=test[1] if test else "",
            window=TEST if ref.windows is not None else None,
            target_before=None,
            by_window=None,
            horizons=None,
            issue_days=None,
        )
        test_mae = issued_mae(whole.iter_records(cols=["run_date", "prediction", "actual"]), test) if test else None
        # A model from below the finalists was scored on a cheaper rung: its validation numbers are of another fidelity, and
        # its row shows none rather than a gap between two kinds of score.
        finalist = search.get("rung") == search.get("top_rung", search.get("rung"))
        out[f"{search.get('search')}:{name}"] = [
            {
                "chooser": search.get("search"),
                "segment": 0,
                "chosen": [name],
                "k": 1,
                "validation_mae": search.get("validation_mae") if finalist else None,
                "validation_forecasts": search.get("validation_forecasts") if finalist else None,
                "validation_share": search.get("validation_share") if finalist else None,
                "validation": tuple(search.get("validation") or (None, None)),
                "test": None if not test else tuple(test),
                "candidates": search.get("candidates"),
                "test_mae": test_mae,
                "search": dict(search),
            }
        ]
    return out


def _carded(models: Sequence[Any]) -> List[Any]:
    """The models `models._cards.preflight` checks: a `PooledModel`'s wrapped model in its place."""
    return [model.model if isinstance(model, PooledModel) else model for model in models]


class ModelPipeline:
    """Runs models over a test period, computes ensembles of them, then runs evaluators and exporters.

    Args:
        information_set: When each input becomes known and when forecasts are issued
            (see epftoolbox2.information_set). None means no issue time and no checks.
        issue_time: Shortcut for `information_set=InformationSet(issue_time, issue_timezone)` with the
            built-in availability rules, e.g. "11:45".
        issue_timezone: Zone of `issue_time`.
        availability: What a run does with a predictor (or the newest training target) that its rule puts after the
            issue time. "warn", the default with an issue time, goes on and lists it, and any column without a rule, in
            `report.availability()`, a warning, the terminal exporter and the manifest. "raise" refuses it before any
            fit, and `unknown` of the information set decides about columns without a rule. "none", the default
            without an issue time, checks and lists nothing; a predictor function still gets its answers from the
            information set. "warn" and "raise" need an issue time. The level doesn't change a forecast, unless a
            predictor function catches the error a late `ctx.value()` raises at "raise" and chooses another list, which it
            must not do. Changing the level alone resumes a run.
        engine: "off" (the default) runs each model on its own, one after another. "on" runs the models that can be
            fitted from shared statistics (OLS, Ridge, LassoCV and LEAR, `BaseModel._engine_capable`) in engine groups: the
            models of one refit rule share one read of the data and one pool of processes, which cuts the fixed cost of each
            model in a study of thousands. NaiveModel, a `train_on="matching"` list, a subclass that fits without its own
            `_fit_stats` and a `forecast_only` run go through their own run as with "off". The fits of OLS, Ridge, LassoCV
            and LEAR (`solver="lars"`) with `StandardScaler` or a VST come from exact sums shared in the group: the same bits
            in any group, within the engine's tolerances of "off" (see the docs page on forecasting). "check" is "on" with
            one fit in about 50 also run as "off" runs it, its gaps in the manifest (`engine_check`) and a warning about a
            model over its class's tolerance. `ENGINE_MAX_MODELS` (20,000) and `ENGINE_MAX_COLUMNS` (4,096) bound how many
            models and columns a group reads at once. `verify` runs each model through the engine as well.

    Whatever `engine`, the torch models of a run (`TorchModel`, the MLP presets) whose fits are vmap stacks on their device
    (`stack="vmap"`, or "auto" on a GPU) and that share a stack key (the module and its parameters but `dropout`, output,
    training window, refit, loss, batch size, epochs, patience, early stopping, weight decay, device, `tf32`, `pad_to`) are
    trained together, in stacks of `pad_to` members on one device: speed only, each forecast is its model's alone to the bit
    (`models/_torch/groups.py`, the docs page on neural networks). On a CPU only `nets.MLP` models (the MLP presets) with the
    "l1" or "mse" loss stack. The environment variable `TORCH_STACKS=off` runs each on its own, and `TORCH_STACKS=check` also
    trains a sample of the stacked calls alone and records the comparison (`stack_check` in the manifest).
    """

    def __init__(
        self,
        information_set: Optional[InformationSet] = None,
        issue_time: Optional[str] = None,
        issue_timezone: str = "Europe/Brussels",
        availability: Optional[str] = None,
        engine: str = ENGINE_OFF,
    ):
        if issue_time is not None:
            if information_set is not None:
                raise ValueError("Pass either issue_time/issue_timezone or information_set, not both")
            information_set = InformationSet(issue_time, issue_timezone)
        self.models: List[BaseModel] = []
        self.ensembles: List[Ensemble] = []
        self.evaluators: List[Evaluator] = []
        self.exporters: List[Exporter] = []
        self.information_set: Optional[InformationSet] = information_set
        self.availability = availability
        self.engine = engine

    @property
    def engine(self) -> str:
        """ "off", "on" or "check": whether engine-capable models run in engine groups (see the class docstring)."""
        return self._engine

    @engine.setter
    def engine(self, engine: str) -> None:
        self._engine = check_engine(engine)

    @property
    def availability(self) -> str:
        """The level in force: the one asked for, else "warn" with an information set and "none" without one."""
        if self._availability is not None:
            return self._availability
        return "warn" if self.information_set is not None else "none"

    @availability.setter
    def availability(self, level: Optional[str]) -> None:
        self._availability = None if level is None else check_level(level)
        self._check_availability()

    def _check_availability(self) -> None:
        """`information_set` can be set after the level, as `Workflow` and `epf model` do, so `run` asks again."""
        if self.availability != "none" and self.information_set is None:
            raise ValueError(
                f"availability={self.availability!r} checks predictors against the time a forecast is issued: pass "
                "issue_time or information_set too, or leave availability out."
            )

    def add_model(self, model: Union[BaseModel, PooledModel]) -> "ModelPipeline":
        """Add a model, or a `PooledModel` (one fit over the zones of `run(data={zone: frame})`)."""
        self.models.append(model)
        return self

    def add_ensemble(self, ensemble: Ensemble) -> "ModelPipeline":
        """Add an ensemble of models or other ensembles of this pipeline, named by `ensemble.members`.

        The members are resolved when the pipeline runs, so the order of `add_model` and `add_ensemble` doesn't matter. See
        `epftoolbox2.ensembles.AverageEnsemble` and the docs page on ensembles.
        """
        if not isinstance(ensemble, Ensemble):
            raise TypeError(f"add_ensemble takes an ensemble such as AverageEnsemble, got {type(ensemble).__name__}")
        ensemble._check()
        self.ensembles.append(ensemble)
        return self

    def add_evaluator(self, *evaluators: Evaluator) -> "ModelPipeline":
        self.evaluators.extend(evaluators)
        return self

    def add_exporter(self, exporter: Exporter) -> "ModelPipeline":
        self.exporters.append(exporter)
        return self

    def run(
        self,
        data: Union[pd.DataFrame, Mapping[str, pd.DataFrame]],
        test_start: str = None,
        test_end: str = None,
        target: str = "price",
        horizon: Optional[int] = None,
        freq: str = "1h",
        save_dir: Optional[str] = None,
        forecast_only: bool = False,
        allow_manifest_mismatch: bool = False,
        require_manifest: bool = False,
        validation: Union[Tuple[Any, Any], str, None] = None,
        test: Optional[Tuple[Any, Any]] = None,
        horizons: Optional[Sequence[int]] = None,
        issue_days: Optional[Sequence[str]] = None,
        save_format: str = DEFAULT_SAVE_FORMAT,
        save_coefficients: bool = True,
        validation_every: int = 1,
    ) -> EvaluationReport:
        """Run every model, compute the ensembles from what they stored, then (unless `forecast_only`) run the evaluators
        and exporters.

        `horizon=H` forecasts every horizon from 1 to H (7 if neither is given), `horizons` only the ones it lists, such as
        `[1, 3, 7]` or `range(3, 10)`; passing both raises. Each forecast is what a run with `horizon=max(horizons)` gives it, and
        the results record the list (see the docs page Time and Calendars).

        `issue_days` issues forecasts on the weekdays it names only, such as `["friday"]` for a week-ahead study (names in any
        case; all seven are every day). Training is untouched, so each forecast is what a run of every day gives it, its
        `day_in_test` still counts calendar days, and the other days are neither planned nor missing. A window without one of
        these weekdays raises. The results record them.

        `freq` is the resolution of the data and of the forecasts: one model per hour ("1h"), per quarter-hour ("15min"), or
        one per day ("1D") on a frame with one row per local date, every row at the time of day it stands for, such as
        `hourly.at_time("12:00")`: the stamp says what `x_h-1` is (11:00 here) and when a value is known, so stamp it at the
        time a forecast is made, not at midnight (see the docs page Time and Calendars). A daily run's records have `hour` 0 and `freq` "1D". `EconomicEvaluator` can't score it and raises before any fit.

        With `save_dir`, each model's results are resumable JSONL with a manifest recording the
        configuration that produced them (epftoolbox2.results.manifest). Resuming into results made with
        a different configuration raises `ManifestMismatchError`; `allow_manifest_mismatch` overwrites the
        manifest and resumes anyway, and `require_manifest` refuses results that predate the manifest
        feature instead of accepting them with a warning.

        `save_format="parquet"` writes each model's results as a directory of Parquet parts, `<name>.parquet/`, in place of
        `<name>.jsonl`: the same records, read by the same rules, for studies of thousands of models (see the docs page on
        caching). A model with results in the other format in `save_dir` raises before any fit; `epf results convert` turns
        one into the other exactly. `save_coefficients=False` writes each record's `coefficients` as `[]` and no other key
        differs; the manifest records it, and a resume with the other setting raises `ManifestMismatchError`.

        `test=(start, end)` is the test period, the same as `test_start` and `test_end` (passing both raises). With
        `validation`, each model forecasts the days of both windows in one run, into one results file, and nothing between
        them. Test and validation records are what a plain run over that window alone gives, apart from a `window` tag: each
        window reads its binary-column masks from its own first day. `validation=(start, end)` must end before the test
        starts. `validation="previous_year"` cuts the test period into 12-month segments, each served by the 12 months
        before it (`report.windows.segments`), and adds one year at the start, tagged "validation". `validation` alone
        forecasts only those days, and nothing of the test period is computed.

        `validation_every=k` issues on every k-th day of a fixed validation window alone, counted from its first day, which
        is always one: each forecast on such a day is the one a run of every day gives it, the other days are neither
        forecast nor planned, and the report scores the kept days. It makes a search's first looks cheap (`epftoolbox2.tune`,
        the rungs' `every_nth_day`). The test window is never thinned. It needs `validation=(start, end)`, not with
        `"previous_year"` or `issue_days`. The manifest records it, and a resume with another value raises
        `ManifestMismatchError`. The default, 1, is every day.

        The report is on the test window, and `report.validation` on the validation window, without forecasts for the first
        test issue day or later (see `EvaluationReport.validation`). Without a test window the validation report itself
        is returned. See the docs page on validation and test.

        Ensembles are computed after every model has run, from what the models stored. An ensemble that chooses (`top`) needs a
        validation window in this run, and a problem with one that shows before any fit raises before any fit. See the docs
        page on ensembles.

        `data` can be a dict of frames, `{"PL": df_pl, "CZ": df_cz, ...}`: a run over several zones. Every model then runs once
        per zone on that zone's frame, in its own data zone, into `save_dir/<zone>/` (one store and manifest per zone and
        model), and is exactly that model run on the zone's frame alone. A `PooledModel` is fitted once per unit over every zone
        and writes each zone's forecasts into that zone's store. Ensembles are computed per zone. The run returns a
        `PanelReport`, one `EvaluationReport` per zone. With `issue_timezone="data"` each zone issues at the issue time on its
        own clock. Exporters export one report, so a run over zones refuses them: export `report.zones[zone]`. See the docs page
        Several Zones.
        """
        check_stack_settings()  # TORCH_STACKS and TORCH_STACK_MEMBERS, before any model writes
        if isinstance(data, Mapping):
            return self._run_zones(
                data,
                test_start=test_start,
                test_end=test_end,
                target=target,
                horizon=horizon,
                freq=freq,
                save_dir=save_dir,
                forecast_only=forecast_only,
                allow_manifest_mismatch=allow_manifest_mismatch,
                require_manifest=require_manifest,
                validation=validation,
                test=test,
                horizons=horizons,
                issue_days=issue_days,
                save_format=save_format,
                save_coefficients=save_coefficients,
                validation_every=validation_every,
            )
        if not self.models:
            raise ValueError("At least one model is required")
        self._refuse_pooled("run")
        if save_dir and read_zone_index(save_dir) is not None:
            raise ValueError(
                f"{save_dir} holds a run over zones ({Path(save_dir) / ZONE_INDEX}); a run of one frame there would leave a "
                "report not knowing which to read. Use another save_dir, or run the frame as a zone, data={zone: frame}."
            )
        horizon, horizons = resolve_horizons(horizon, horizons, 7)
        weekdays = resolve_issue_days(issue_days)
        self._check_model_names()
        # Every model's extra, runtime packages and weights before anything else, so a missing one fails in seconds and names
        # every other problem with it (`models._cards.preflight`); a core model costs a lookup.
        preflight(self.models)
        self._check_availability()
        self._check_evaluators_for(freq)
        if not forecast_only:
            self._check_benchmarks()
            self._check_source_columns(data)
        windows, test_start, test_end = self._resolve_windows(
            validation, test, test_start, test_end, forecast_only, validation_every=validation_every
        )
        self._refuse_thinned_weekdays(windows, weekdays)
        if not forecast_only:
            refuse_searched_on_test(
                self.models, windows.test[0] if windows is not None and windows.test else test_start
            )
        self._check_ensembles(save_dir)
        self._refuse_top_without_validation(self.ensembles, windows)
        check_save_format(save_format)
        if save_dir:
            self._check_save_format(save_dir, save_format)
        if windows is not None:
            check_start, check_end = windows.first_day, windows.last_day
        elif forecast_only:
            check_start, check_end = resolve_date(test_start or "today"), resolve_date(test_end or "today")
        else:
            check_start, check_end = test_start, test_end
        for name, (start, end) in windows.periods if windows is not None else [("test", (check_start, check_end))]:
            check_issue_days_in(start, end, weekdays, name)
        days_argument = {} if weekdays is None else {"issue_days": issue_day_names(weekdays)}
        level = self.availability
        # Check every model's inputs before running any, so a bad predictor in the last model fails in seconds. A
        # predictor function is planned here, once, and the run uses that plan; each run uses its check, too. The frame
        # contract is checked once for the models that read the same days (`_RunChecks`, of this call alone).
        info = self._information_set_for(data)
        shared = _RunChecks(data)
        plans: Dict[str, PredictorPlan] = {}
        checks: Dict[str, Any] = {}
        listings: Dict[str, List[Dict]] = {}
        for model in self.models:
            plan = model._plan_predictors(
                data,
                target,
                horizon,
                freq,
                info,
                check_start,
                check_end,
                availability=level,
                windows=windows,
                horizons=horizons,
                weekdays=weekdays,
                shared=shared,
            )
            checks[model.name] = model._checked_inputs(
                data,
                target,
                horizon,
                freq,
                info,
                check_start,
                check_end,
                plan=plan,
                availability=level,
                windows=windows,
                horizons=horizons,
                weekdays=weekdays,
                shared=shared,
            )
            listings[model.name] = checks[model.name].listing
            if plan is not None:
                plans[model.name] = plan
        warn_about(listings, stacklevel=3)  # once, for every model, and before the hours a run can take
        shared = None

        # Of this call alone: a frame changed before the next one must be hashed again.
        frame_hashes = FrameHashes(data) if save_dir else None
        if frame_hashes is not None:
            frame_hashes.provenance()  # the package version and git SHA every manifest of the run records, read before any model

        def arguments(model: BaseModel) -> Dict[str, Any]:
            """What the model's run is called with."""
            save_to = str(self.results_path(save_dir, model, save_format)) if save_dir else None
            planned = {"plan": plans[model.name]} if model.name in plans else {}
            return dict(
                data=data,
                test_start=test_start,
                test_end=test_end,
                target=target,
                freq=freq,
                save_to=save_to,
                forecast_only=forecast_only,
                information_set=info,
                allow_manifest_mismatch=allow_manifest_mismatch,
                require_manifest=require_manifest,
                availability=level,
                warn_listing=False,
                windows=windows,
                frame_hashes=frame_hashes,
                **horizon_arguments(horizon, horizons),
                **days_argument,
                **planned,
                checked=checks.pop(model.name),  # held no longer than the model's run
                **({} if save_coefficients else {"save_coefficients": False}),
            )

        ran: Dict[str, ModelResultRef] = {}
        refs: dict[str, ModelResultRef] = {}
        # One hold of this process's OpenBLAS thread count for all the run's pools: putting it back restarts its threads.
        with pool_native_threads():
            if self.engine != ENGINE_OFF:
                # The engine groups first, each in one go (models/_engine.py), then the rest in list order, as with "off".
                capable = [
                    model for model in self.models if model._engine_capable(plans.get(model.name), forecast_only)
                ]
                for group in engine_groups(capable):
                    columns = {model.name: checks[model.name].used_predictors for model in group}
                    # `run_group` takes each model's arguments (its check with them) out of this dict when its chunk starts.
                    ran.update(
                        run_group(
                            group,
                            {model.name: run_arguments(**arguments(model)) for model in group},
                            columns,
                            check=self.engine == ENGINE_CHECK,
                        )
                    )
            # Torch stacks (models/_torch/groups.py): the torch models of one stack key train together, each forecast that
            # of its model alone, to the bit; then the rest in list order.
            for group in stack_groups([model for model in self.models if model.name not in ran], forecast_only):
                columns = {model.name: checks[model.name].used_predictors for model in group}
                ran.update(
                    run_stack_group(group, {model.name: run_arguments(**arguments(model)) for model in group}, columns)
                )
            for model in self.models:
                refs[model.name] = ran[model.name] if model.name in ran else model.run(**arguments(model))

        choices: Dict[str, List[Dict]] = {}
        if self.ensembles:
            paths = {model.name: self.results_path(save_dir, model) for model in self.models} if save_dir else {}
            on_file = self._stored_windows(paths)
            computed, choices = self._compute_ensembles(
                self._ordered_ensembles(),
                refs,
                on_file or windows,
                save_dir,
                provenance=None if frame_hashes is None else frame_hashes.provenance,
            )
            refs.update(computed)
        if not forecast_only:
            made = {"target": target, "horizons": list(horizons), "freq": freq, "issue_days": issue_day_names(weekdays)}
            choices.update(search_choices(refs, windows.test if windows is not None else (test_start, test_end), made))

        evaluators = [] if forecast_only else self.evaluators
        report = EvaluationReport(refs, evaluators, source_data=data, choices=choices)

        if not forecast_only:
            for exporter in self.exporters:
                exporter.export(report)

        return report

    def _information_set_for(self, data: pd.DataFrame) -> Optional[InformationSet]:
        """The pipeline's information set for a frame: with `issue_timezone="data"` issued in the frame's zone
        (`InformationSet.in_data_zone`), else itself."""
        return None if self.information_set is None else self.information_set.in_data_zone(data.index)

    def _refuse_pooled(self, action: str) -> None:
        """A `PooledModel` is fitted over the zones of `run(data={zone: frame})`: on one frame it raises before anything runs."""
        pooled = [model.name for model in self.models if isinstance(model, PooledModel)]
        if pooled:
            raise ValueError(
                f"{action}() of one frame: {pooled} pool zones, and fit over the frames of run(data={{zone: frame, ...}}). Run "
                "the zones as a dict (one zone is the model itself), or add the inner model in place of the PooledModel."
            )

    @staticmethod
    def _check_zones(frames: Mapping[str, Any]) -> List[str]:
        """The zones of a run over several, in their order: names that make a folder of `save_dir` (letters, digits, `_` and `-`,
        unique whatever their case), each a DataFrame."""
        zones = list(frames)
        if not zones:
            raise ValueError("run(data={...}) needs at least one zone's frame")
        bad = [zone for zone in zones if not isinstance(zone, str) or not _ZONE_NAME.fullmatch(zone)]
        if bad:
            raise ValueError(
                f"Zone names name the folders of save_dir, so they are letters, digits, '_' and '-', starting with a letter or "
                f"digit: {bad} aren't."
            )
        folded = [zone.lower() for zone in zones]
        if len(set(folded)) != len(folded):
            raise ValueError(f"Zone names must be unique whatever their case (they name folders): {zones}")
        frames_not = [zone for zone in zones if not isinstance(frames[zone], pd.DataFrame)]
        if frames_not:
            raise TypeError(f"run(data={{zone: frame}}) takes a DataFrame per zone; {frames_not} aren't")
        return zones

    def _run_zones(
        self,
        frames: Mapping[str, pd.DataFrame],
        test_start: Optional[str],
        test_end: Optional[str],
        target: str,
        horizon: Optional[int],
        freq: str,
        save_dir: Optional[str],
        forecast_only: bool,
        allow_manifest_mismatch: bool,
        require_manifest: bool,
        validation: Union[Tuple[Any, Any], str, None],
        test: Optional[Tuple[Any, Any]],
        horizons: Optional[Sequence[int]],
        issue_days: Optional[Sequence[str]],
        save_format: str,
        save_coefficients: bool,
        validation_every: int = 1,
    ) -> PanelReport:
        """`run` over several zones (see `run`): each zone's frame checked and run as `run` checks and runs one frame, every
        model's and zone's check before any fit, then each plain model per zone, each `PooledModel` over all of them
        (`models._pooled.run_pooled`), the ensembles per zone, and one report per zone."""
        if not self.models:
            raise ValueError("At least one model is required")
        zones = self._check_zones(frames)
        horizon, horizons = resolve_horizons(horizon, horizons, 7)
        weekdays = resolve_issue_days(issue_days)
        self._check_model_names()
        preflight(_carded(self.models))  # as in a run of one frame: every model's extra and weights first
        self._check_availability()
        self._check_evaluators_for(freq)
        if self.exporters and not forecast_only:
            raise ValueError(
                f"A run over zones returns one report per zone, and an exporter exports one report: remove the exporters "
                f"{[type(e).__name__ for e in self.exporters]} and export each `report.zones[zone]` yourself."
            )
        if not forecast_only:
            self._check_benchmarks()
            for zone in zones:
                self._check_source_columns(frames[zone])
        windows, test_start, test_end = self._resolve_windows(
            validation, test, test_start, test_end, forecast_only, validation_every=validation_every
        )
        self._refuse_thinned_weekdays(windows, weekdays)
        if not forecast_only:
            refuse_searched_on_test(
                self.models, windows.test[0] if windows is not None and windows.test else test_start
            )
        pooled = [model for model in self.models if isinstance(model, PooledModel)]
        plain = [model for model in self.models if not isinstance(model, PooledModel)]
        if pooled and windows is not None:
            raise ValueError(
                f"{[model.name for model in pooled]}: a PooledModel runs a test window alone in this step; run validation and "
                "test as two runs into two save_dirs."
            )
        dirs = {zone: None if not save_dir else Path(save_dir) / zone for zone in zones}
        for zone in zones:
            self._check_ensembles(dirs[zone])
        self._refuse_top_without_validation(self.ensembles, windows)
        check_save_format(save_format)
        if save_dir:
            check_zone_case(save_dir, zones)
            beside = [model.name for model in self.models if self.results_path(save_dir, model).exists()]
            if beside:
                raise ValueError(
                    f"{save_dir} holds the results of {beside} run on one frame; a run over zones writes into a folder per "
                    "zone, and a report couldn't tell which to read. Use another save_dir."
                )
            for zone in zones:
                self._check_save_format(dirs[zone], save_format)
        if windows is not None:
            check_start, check_end = windows.first_day, windows.last_day
        elif forecast_only:
            check_start, check_end = resolve_date(test_start or "today"), resolve_date(test_end or "today")
        else:
            check_start, check_end = test_start, test_end
        for name, (start, end) in windows.periods if windows is not None else [("test", (check_start, check_end))]:
            check_issue_days_in(start, end, weekdays, name)
        days_argument = {} if weekdays is None else {"issue_days": issue_day_names(weekdays)}
        level = self.availability
        infos = {zone: self._information_set_for(frames[zone]) for zone in zones}

        # Every model's and zone's inputs, before any fit (as `run` checks one frame's).
        common = dict(availability=level, windows=windows, horizons=horizons, weekdays=weekdays)
        plans: Dict[Tuple[str, str], PredictorPlan] = {}
        checks: Dict[Tuple[str, str], Any] = {}
        for zone in zones:
            frame, info = frames[zone], infos[zone]
            with _zone_errors(zone):
                # What `_prepare_run` would refuse after an earlier zone's files were written: every day the run names
                # must be a local date of each zone's frame.
                _check_days_in_frame(frame, windows, check_start, check_end)
                shared = _RunChecks(frame)
                for model in plain:
                    plan = model._plan_predictors(
                        frame, target, horizon, freq, info, check_start, check_end, shared=shared, **common
                    )
                    checks[(zone, model.name)] = model._checked_inputs(
                        frame, target, horizon, freq, info, check_start, check_end, plan=plan, shared=shared, **common
                    )
                    if plan is not None:
                        plans[(zone, model.name)] = plan
        members: Dict[str, Dict[str, BaseModel]] = {}
        member_frames: Dict[str, Dict[str, pd.DataFrame]] = {}
        for model in pooled:
            check_pooled_model(model.model)
            members[model.name], member_frames[model.name] = {}, {}
            by_zone = {}
            for zone in zones:
                member = member_of(model, model.identity(zones))
                check_horizon_lists(model, member, freq, horizons)
                with _zone_errors(zone):
                    frame = with_missing_columns(model, zone, frames[zone], needed_columns(member, freq, horizons))
                    shared = _RunChecks(frame)
                    member._plan_predictors(
                        frame, target, horizon, freq, infos[zone], check_start, check_end, shared=shared, **common
                    )
                    by_zone[zone] = member._checked_inputs(
                        frame, target, horizon, freq, infos[zone], check_start, check_end, shared=shared, **common
                    )
                members[model.name][zone], member_frames[model.name][zone] = member, frame
            by_zone = shared_fit_checks(
                model,
                members[model.name],
                member_frames[model.name],
                by_zone,
                infos,
                target,
                freq,
                check_start,
                check_end,
                weekdays,
                level,
            )
            checks.update({(zone, model.name): check for zone, check in by_zone.items()})
        # Once for every model and zone, before the hours a run can take.
        warn_about({f"{name} ({zone})": check.listing for (zone, name), check in checks.items()}, stacklevel=4)

        # Of this call alone, one per frame: a frame changed before the next call must be hashed again.
        hashes: Dict[int, FrameHashes] = {}

        def hashes_of(frame: pd.DataFrame) -> Optional[FrameHashes]:
            if not save_dir:
                return None
            if id(frame) not in hashes:
                hashes[id(frame)] = FrameHashes(frame)
                hashes[id(frame)].provenance()  # read before any model, as `run` reads it
            return hashes[id(frame)]

        if save_dir:
            for model in pooled:
                # Each zone's manifest records the other zones' data: another zone's frame moves this zone's forecasts.
                data_of = {
                    zone: hashes_of(member_frames[model.name][zone]).of_columns(
                        check_end, [target, *checks[(zone, model.name)].used_predictors]
                    )
                    for zone in zones
                }
                for zone, member in members[model.name].items():
                    member._pooled = {
                        **model.identity(zones),
                        "data": {other: data_of[other] for other in zones if other != zone},
                    }

        def arguments(zone: str, model: BaseModel, frame: pd.DataFrame, keep: bool = False) -> Dict[str, Any]:
            """What the model's (or a pooled model's member's) run in `zone` is called with; its check is let go of unless
            `keep`."""
            key = (zone, model.name)
            save_to = str(self.results_path(dirs[zone], model, save_format)) if save_dir else None
            return dict(
                data=frame,
                test_start=test_start,
                test_end=test_end,
                target=target,
                freq=freq,
                save_to=save_to,
                forecast_only=forecast_only,
                information_set=infos[zone],
                allow_manifest_mismatch=allow_manifest_mismatch,
                require_manifest=require_manifest,
                availability=level,
                warn_listing=False,
                windows=windows,
                frame_hashes=hashes_of(frame),
                **horizon_arguments(horizon, horizons),
                **days_argument,
                **({"plan": plans[key]} if key in plans else {}),
                checked=checks[key] if keep else checks.pop(key),  # held no longer than the model's run
                **({} if save_coefficients else {"save_coefficients": False}),
            )

        refs: Dict[str, Dict[str, ModelResultRef]] = {zone: {} for zone in zones}

        def run_plain(zone: str) -> None:
            frame = frames[zone]
            with _zone_errors(zone):
                if self.engine != ENGINE_OFF:
                    capable = [m for m in plain if m._engine_capable(plans.get((zone, m.name)), forecast_only)]
                    for group in engine_groups(capable):
                        columns = {model.name: checks[(zone, model.name)].used_predictors for model in group}
                        group_arguments = {
                            model.name: run_arguments(**arguments(zone, model, frame)) for model in group
                        }
                        refs[zone].update(run_group(group, group_arguments, columns, check=self.engine == ENGINE_CHECK))
                for group in stack_groups([m for m in plain if m.name not in refs[zone]], forecast_only):
                    columns = {model.name: checks[(zone, model.name)].used_predictors for model in group}
                    group_arguments = {model.name: run_arguments(**arguments(zone, model, frame)) for model in group}
                    refs[zone].update(run_stack_group(group, group_arguments, columns))
                for model in plain:
                    if model.name not in refs[zone]:
                        refs[zone][model.name] = model.run(**arguments(zone, model, frame))

        # What a resume can refuse comes from the zones whose folders hold results already: their pooled models' resumes
        # are checked, and their plain models run, before the zone index or any file of a zone new to the save_dir is written,
        # so a refused run leaves the save_dir as it was.
        # One hold of this process's OpenBLAS thread count for every zone's pools and the pooled models' (as in `run`).
        with pool_native_threads():
            on_file = [zone for zone in zones if save_dir and any(dirs[zone].glob("*.manifest.json"))]
            for model in pooled:
                check_resumes(
                    {
                        zone: (
                            members[model.name][zone],
                            arguments(zone, members[model.name][zone], member_frames[model.name][zone], keep=True),
                        )
                        for zone in on_file
                    }
                )
            for zone in on_file:
                run_plain(zone)
            if save_dir:
                write_zone_index(save_dir, zones)  # what tells a report this save_dir's folders are zones
            for zone in zones:
                if zone not in on_file:
                    run_plain(zone)
            for model in pooled:
                ran = run_pooled(
                    model,
                    {
                        zone: (member, arguments(zone, member, member_frames[model.name][zone]))
                        for zone, member in members[model.name].items()
                    },
                )
                for zone in zones:
                    refs[zone][model.name] = ran[zone]

        reports: Dict[str, EvaluationReport] = {}
        evaluators = [] if forecast_only else self.evaluators
        for zone in zones:
            zone_refs = {model.name: refs[zone][model.name] for model in self.models}
            choices: Dict[str, List[Dict]] = {}
            if self.ensembles:
                paths = {model.name: self.results_path(dirs[zone], model) for model in self.models} if save_dir else {}
                on_file = self._stored_windows(paths)
                first = hashes_of(frames[zone])
                computed, choices = self._compute_ensembles(
                    self._ordered_ensembles(),
                    zone_refs,
                    on_file or windows,
                    dirs[zone],
                    provenance=None if first is None else first.provenance,
                )
                zone_refs.update(computed)
            if not forecast_only:
                made = {
                    "target": target,
                    "horizons": list(horizons),
                    "freq": freq,
                    "issue_days": issue_day_names(weekdays),
                }
                choices.update(
                    search_choices(zone_refs, windows.test if windows is not None else (test_start, test_end), made)
                )
            reports[zone] = EvaluationReport(zone_refs, evaluators, source_data=frames[zone], choices=choices)
        return PanelReport(reports, target=target)

    @staticmethod
    def _resolve_windows(
        validation, test, test_start, test_end, forecast_only: bool, required: bool = True, validation_every: int = 1
    ):
        """(the windows of a run with validation or None, first test day, last test day), from `run`'s arguments.

        A `ValueError` if they are given twice, make no sense, or (with `required`) name no window at all, and for a
        `validation_every` other than 1 without a fixed validation window.
        """
        if check_validation_every(validation_every) != 1 and validation is None:
            raise ValueError(
                f"validation_every={validation_every} thins the validation window, and this run has none: pass "
                "validation=(start, end)"
            )
        if test is not None and (test_start is not None or test_end is not None):
            raise ValueError("Pass the test period once: test=(start, end) or test_start and test_end, not both")
        if forecast_only and validation is not None:
            raise ValueError("validation= is for backtests: forecast_only has no actuals to validate on")
        if test is None and forecast_only:
            return None, test_start, test_end  # today, unless they say otherwise
        if test is None and (test_start is not None or test_end is not None):
            if test_start is None or test_end is None:
                raise ValueError("test_start and test_end are required unless forecast_only=True")
            test = (test_start, test_end)
        if validation is not None:
            windows = RunWindows.resolve(validation, test, validation_every)
            return windows, *(windows.test or (None, None))
        if test is None:
            if not required:
                return None, None, None
            raise ValueError(
                "No window to forecast: pass test=(start, end), validation=(start, end) or both (test_start and test_end "
                "are required unless forecast_only=True)"
            )
        test_start, test_end = to_period(test, TEST)
        return None, test_start, test_end

    @staticmethod
    def _refuse_thinned_weekdays(windows: Optional[RunWindows], weekdays: Optional[Sequence[int]]) -> None:
        """`validation_every` and `issue_days` both choose issue days, and a run takes one of them."""
        if windows is not None and windows.validation_every != 1 and weekdays is not None:
            raise ValueError(
                f"validation_every={windows.validation_every} and issue_days={issue_day_names(weekdays)} both choose the "
                "days a run issues on: pass one of them"
            )

    @staticmethod
    def _stored_horizons(save_dir: Union[str, Path], paths: Dict[str, Path]) -> Optional[Tuple[int, ...]]:
        """The horizons every model's manifest records, or None if none does; models run with different ones raise."""
        found = {name: read_horizons(manifest_path(path)) for name, path in paths.items()}
        found = {name: horizons for name, horizons in found.items() if horizons is not None}
        distinct = set(found.values())
        if len(distinct) > 1:
            shared = sorted(set.intersection(*(set(horizons) for horizons in distinct)))
            shown = ", ".join(f"{name}: {list(horizons)}" for name, horizons in found.items())
            hint = f" such as horizons={shared}, the ones they share," if shared else ""
            raise ValueError(
                f"The models in {save_dir} were run with different horizons ({shown}), so a report that took them from the "
                f"manifests would plan different forecasts for each. Pass horizons={hint} or horizon= to say which to report."
            )
        return next(iter(distinct), None)

    def _stored_windows(self, paths: Dict[str, Path]) -> Optional[RunWindows]:
        """The windows every shard's manifest records, or None for plain runs; shards that disagree raise."""
        found = {name: read_windows(path) for name, path in paths.items()}
        distinct = {windows for windows in found.values()}
        if len(distinct) > 1:
            shown = "; ".join(f"{name}: {windows}" for name, windows in found.items())
            raise ValueError(
                f"These results were made with different windows, so their models can't be reported together ({shown}). "
                "Run every model with the same validation= and test=."
            )
        return next(iter(distinct), None)

    @staticmethod
    def _stored_issue_days(save_dir: Union[str, Path], paths: Dict[str, Path]) -> Optional[Tuple[int, ...]]:
        """The weekdays every model's manifest records it was run on, None for every day; models run on different ones raise.
        Results with no manifest say nothing and don't count."""
        found = {
            name: read_issue_days(manifest_path(path)) for name, path in paths.items() if manifest_path(path).exists()
        }
        if len(set(found.values())) > 1:
            shown = "; ".join(f"{name}: {issue_day_names(days) or 'every day'}" for name, days in found.items())
            raise ValueError(
                f"The models in {save_dir} were run on different weekdays ({shown}), so a report that took them from the "
                "manifests would plan different forecasts for each. Pass issue_days= to say which to report."
            )
        return next(iter(found.values()), None)

    @staticmethod
    def _record_freq(name: str, path: Path) -> Optional[str]:
        """The freq the records of results with no manifest carry, or None if they carry none; records of several raise."""
        if not path.exists():
            return None
        freqs = {r.get("freq") for r in read_records(path, statuses=None)} - {None}  # a half-written last line warns
        if len(freqs) > 1:
            raise ValueError(
                f"The results of {name!r} in {path} hold records of several frequencies ({sorted(freqs)}): their periods are "
                "different times of day, so they can't be one model's. Write each run into its own save_dir."
            )
        return next(iter(freqs), None)

    @staticmethod
    def _stored_freq(save_dir: Union[str, Path], paths: Dict[str, Path], told: Optional[str]) -> Optional[str]:
        """The freq every model's manifest records, or None if none does. Models run at different ones raise, and so does a
        `told` freq the manifests contradict: the report would plan another number of periods a day than the runs did, and count
        records as missing, or a gap as none. Results with no manifest say what freq their records carry."""
        found = {
            name: read_freq(manifest_path(path)) or ModelPipeline._record_freq(name, path)
            for name, path in paths.items()
        }
        found = {name: freq for name, freq in found.items() if freq is not None}
        shown = ", ".join(f"{name}: {freq!r}" for name, freq in found.items())
        if len(set(found.values())) > 1:
            raise ValueError(
                f"The models in {save_dir} were run at different frequencies ({shown}): their periods are different times of "
                "day, so one report can't plan them. Report each frequency from its own pipeline."
            )
        stored = next(iter(found.values()), None)
        if told is not None and stored is not None and told != stored:
            raise ValueError(
                f"report(freq={told!r}), but the results in {save_dir} were run at freq={stored!r} ({shown}). Leave freq out "
                "to report what the manifests record."
            )
        return stored

    def report(
        self,
        save_dir: Union[str, Path],
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
        horizon: Optional[int] = None,
        freq: Optional[str] = None,
        forecast_only: bool = False,
        data: Optional[pd.DataFrame] = None,
        allow_missing: bool = False,
        validation: Union[Tuple[Any, Any], str, None] = None,
        test: Optional[Tuple[Any, Any]] = None,
        horizons: Optional[Sequence[int]] = None,
        issue_days: Optional[Sequence[str]] = None,
    ) -> EvaluationReport:
        """Build one report from the per-model JSONL files in `save_dir`, then run evaluators and exporters.

        This is the merge step after runs that each wrote some models' results (e.g. a SLURM array with
        `Workflow(model_index=...)`). The test period and horizon (or `horizons`, as in `run`) must be those of the runs;
        with the time zone each manifest records they give each model's planned forecast count (`IssueCalendar`). Given neither
        `horizon` nor `horizons`, the report plans what the manifests record (7 without one), and models run with different
        horizons raise. Only records
        of the horizons planned are read, so `horizon=3` of a horizon-7 run is horizons 1 to 3 alone. Raises FileNotFoundError if a
        model has no results file, unless `allow_missing` is set. `data` is only needed by exporters that join source columns
        (`CsvExporter(extra_columns=)`) and evaluators that read them (`ScaledMAEEvaluator`), which raise without it.

        `issue_days` are the weekdays to report, as in `run`; if not given, the manifests say which the models were run on, and
        models run on different ones raise `ValueError`, as they would be compared on different forecasts. Only records issued on
        them are read and planned, so a report of the Fridays of a run of every day plans Fridays alone.

        `freq` is read from the manifests when not given (without one from the records, which raise if they carry several; "1h"
        if they carry none); models run at different frequencies raise
        `ValueError`, and so does a `freq` the manifests contradict, which would plan another number of periods a day than the runs
        did. `EconomicEvaluator` raises for daily results, as in `run`.

        Ensembles are computed here from the stored results of their members, every time, so they can't go stale, and written
        next to them. `allow_missing` leaves out an ensemble with a missing member, with a warning: it is never averaged over
        the members that happen to be there.

        Results that may hold records written before training rows were chosen by date (no manifest, one older than
        `training_cut`, or one that accepted such results, see `epftoolbox2.results.manifest`) are warned about by name, every
        time. With the zone their manifest records, a record of a slot the clock doesn't have is left out, with a warning.

        What the runs listed about inputs known after the issue time comes from the manifests, whatever this pipeline's
        `availability` is: `report.availability()`, one warning and the terminal exporter show it.

        `forecast_counts()` has the skip reasons the runs recorded in their manifests, of a window as a whole: a report on part of
        a plain run's days has `missing` and no reasons. A task that crashed recorded none, so what `missing` holds beyond them is
        that. `produced` counts the records inside the period, whatever else the file holds.

        Forecasts of a `train_on="matching"` list on no more days than the list has predictors are warned about again, from the
        stored records: an array task's warning went to that task's log.

        After runs with validation, `validation` and `test` say what `run` was told, and each must be what the manifests
        record. Without any window the report is on every window they record. Results whose manifests record different
        windows raise `ValueError`, as they would be compared on different forecasts. The report is what `run` returns: on
        the test window, with `report.validation`, or the validation report if there is no test window.

        A `save_dir` of a run over zones (`run(data={zone: frame})`) holds a folder per zone and the run's zone index,
        `zone_run.json`, which alone makes it one: the report is then a `PanelReport`, each zone's folder reported as above
        (`data` a dict of the zones' frames, if a zone's evaluators need it), in the run's order. A zone the index or a pooled
        model's manifest names whose folder holds none of the pipeline's results raises `FileNotFoundError` (`allow_missing`
        leaves it out with a warning). Exporters export one report, so they raise there. Any other layout reads as before:
        sub-folders of a `save_dir` are never zones by their names alone.
        """
        if not self.models:
            raise ValueError("At least one model is required")
        zones = read_zone_index(save_dir)
        if zones is not None:
            return self._report_zones(
                save_dir,
                zones,
                test_start=test_start,
                test_end=test_end,
                horizon=horizon,
                freq=freq,
                forecast_only=forecast_only,
                data=data,
                allow_missing=allow_missing,
                validation=validation,
                test=test,
                horizons=horizons,
                issue_days=issue_days,
            )
        if isinstance(data, Mapping):
            raise ValueError(f"report(data={{zone: frame}}) is for a save_dir of zone folders, and {save_dir} has none")
        told_weekdays = resolve_issue_days(issue_days)
        self._check_model_names()
        if not forecast_only:
            self._check_benchmarks()
        self._check_ensembles(save_dir)
        paths = {model.name: self.results_path(save_dir, model) for model in self.models}
        for name, path in paths.items():
            if is_ensemble_manifest(manifest_path(path)):
                raise ManifestMismatchError(
                    f"Model {name!r}: {path} is an ensemble's results, which go stale as soon as their members move unless "
                    "they are computed again. Add it to this pipeline with add_ensemble, or rename the model."
                )
        freq = self._stored_freq(save_dir, paths, freq) or freq or "1h"
        self._check_evaluators_for(freq)
        if not forecast_only:
            self._check_source_columns(data, required=True)
        if horizon is None and horizons is None:
            horizons = self._stored_horizons(save_dir, paths)
        horizon, horizons = resolve_horizons(horizon, horizons, 7)
        stored = self._stored_windows({name: path for name, path in paths.items() if path.exists()})
        weekdays = told_weekdays if issue_days is not None else self._stored_issue_days(save_dir, paths)
        windows, test_start, test_end = self._resolve_windows(
            validation, test, test_start, test_end, forecast_only, required=stored is None or forecast_only
        )
        if stored is not None and not forecast_only:
            windows = self._windows_to_report(windows, test_start, test_end, stored)
        elif windows is not None:
            raise ValueError(
                f"The results in {save_dir} were made without validation: their manifests record no windows, so there is "
                "no validation window to report. Run with validation=... first."
            )
        if windows is not None:
            test_start, test_end = windows.first_day, windows.last_day
        else:
            if forecast_only:
                test_start, test_end = resolve_date(test_start or "today"), resolve_date(test_end or "today")
            test_start, test_end = resolve_date(test_start), resolve_date(test_end)

        refs: Dict[str, ModelResultRef] = {}
        calendars: Dict[str, IssueCalendar] = {}
        missing = []
        for model in self.models:
            path = self.results_path(save_dir, model)
            if not path.exists():
                missing.append(f"{model.name} ({path})")
                continue
            zone = read_zone(manifest_path(path))
            for name, (start, end) in windows.periods if windows is not None else [("test", (test_start, test_end))]:
                check_issue_days_in(start, end, weekdays, name)
            calendars[model.name] = IssueCalendar(freq, zone, horizons, weekdays)
            ref = ModelResultRef(
                name=model.name,
                path=path,
                count=0 if windows is not None else calendars[model.name].count(test_start, test_end),
                test_start=test_start,
                test_end=test_end,
                horizon=horizon,
                freq=freq,
                skipped=(
                    {}
                    if windows is not None
                    else read_skipped(
                        manifest_path(path), period=(test_start, test_end), horizons=horizons, issue_days=weekdays
                    )
                ),
                availability_listing=read_listing(manifest_path(path)),
                vintages=read_vintages(manifest_path(path)),
                provenance=read_provenance(manifest_path(path)),
                windows=windows,
                matching_lists=read_matching_lists(manifest_path(path)),
                zone=zone,
                # Only the horizons and weekdays planned: a record of another is compared with nothing (`horizon=3` of a
                # horizon-7 run, the Fridays of a run of every day).
                horizons=horizons,
                issue_days=weekdays,
            )
            # Through the ref: the file can hold more than this period, and a line outside it is no forecast of it.
            ref.produced = ref.count_records()
            refs[model.name] = ref
        if missing and not allow_missing:
            raise FileNotFoundError(f"No results for model(s): {', '.join(missing)}. Run them first.")
        for name in missing:
            logger.warning(f"Report: no results for {name}; it is left out")
        old = [name for name, ref in refs.items() if rows_unverified(manifest_path(ref.path))]
        if old:
            warnings.warn(
                f"Report: the results of {old} may hold records written before training rows were chosen by date (their "
                "manifest is older, was accepted over that with allow_manifest_mismatch=True, or is missing), so the days after "
                "an autumn switch may hold a look-ahead. Run them again into a new save_dir.",
                stacklevel=2,
            )
        warn_off_the_clock(refs, stacklevel=2)

        parts = dict(refs)
        if windows is not None:
            parts = {}
            for name, ref in refs.items():
                ref.by_window = split_by_window(
                    ref,
                    windows,
                    calendars[name],
                    skipped={
                        w: read_skipped(manifest_path(ref.path), w, horizons=ref.horizons, issue_days=ref.issue_days)
                        for w, _ in windows.periods
                    },
                    listings={w: read_listing(manifest_path(ref.path), w) for w, _ in windows.periods},
                    target_before=stored.target_before,
                )
                parts.update({f"{name} in the {window} window": part for window, part in ref.by_window.items()})

        choices: Dict[str, List[Dict]] = {}
        reported = self._ensembles_with_their_members(set(refs))
        if reported:
            self._refuse_top_without_validation(reported, stored)
            # A forecast-only report covers some days of what is on file: the file on disk holds all of it.
            computed, choices = self._compute_ensembles(reported, refs, stored, None if forecast_only else save_dir)
            refs.update(computed)
            for name, ref in computed.items():
                if ref.by_window is None:
                    parts[name] = ref
                else:
                    parts.update({f"{name} in the {window} window": part for window, part in ref.by_window.items()})

        shortfall = {
            name: ref
            for name, ref in parts.items()
            if ref.produced is not None and ref.produced + ref.skipped.get(SKIP_TARGET_IN_TEST, 0) < ref.count
        }
        if shortfall:
            details = "; ".join(
                f"{name} ({ref.produced} of {ref.count}, {ref.count - ref.produced} missing"
                f"{''.join(f', {reason} {n}' for reason, n in sorted(ref.skipped.items()))})"
                for name, ref in shortfall.items()
            )
            warnings.warn(
                f"Report: model(s) with missing forecasts: {details}. Metrics are then computed on different "
                "forecast sets for these models; compare with care.",
                stacklevel=2,
            )
        warn_about({name: ref.availability_listing for name, ref in refs.items()}, stacklevel=3)

        if not forecast_only:
            # The test period on file, whichever part of it this report names, as the ensembles' rows have it.
            test_period = stored.test if stored is not None else (test_start, test_end)
            choices.update(
                search_choices(
                    {model.name: refs[model.name] for model in self.models if model.name in refs}, test_period
                )
            )
        report = EvaluationReport(refs, [] if forecast_only else self.evaluators, source_data=data, choices=choices)
        warn_about_thin(rows_of_refs(refs), stacklevel=2)  # on the whole file: a report holds one window
        if not forecast_only:
            for exporter in self.exporters:
                exporter.export(report)
        return report

    def _report_zones(self, save_dir: Union[str, Path], zones: List[str], data: Any = None, **told: Any) -> PanelReport:
        """`report` of the save_dir of a run over zones (its zone index, `ZONE_INDEX`): one report per zone (`data[zone]`, if
        given), in the run's order, in a `PanelReport`. A zone the index or a pooled model's manifest names whose folder holds
        none of the pipeline's results raises, unless `allow_missing`, which leaves it out with a warning: a panel without it
        would score every other zone as the whole run."""
        if self.exporters and not told.get("forecast_only"):
            raise ValueError(
                "A report of a run over zones is one report per zone, and an exporter exports one: remove the exporters and "
                "export each `report.zones[zone]` yourself."
            )
        if data is not None and not isinstance(data, Mapping):
            raise ValueError(
                f"{save_dir} holds a run over the zones {zones}: pass data as a dict of their frames, {{zone: frame}}"
            )
        root = Path(save_dir)
        named = [*self.models, *self.ensembles]
        expected = list(dict.fromkeys([*zones, *self._pooled_zones(root, zones)]))

        def has_results(zone: str) -> bool:
            folder = root / zone
            return folder.is_dir() and any(
                self.results_path(folder, model, fmt).exists() for model in named for fmt in SAVE_FORMATS
            )

        missing = [zone for zone in expected if not has_results(zone)]
        if missing and not told.get("allow_missing"):
            raise FileNotFoundError(
                f"{root} is a run over the zones {expected}, and {missing} hold none of this pipeline's results (a folder lost, "
                "or a run stopped before them): a panel without them would score the others as the whole run. Run them "
                "again, or pass allow_missing=True to leave them out."
            )
        if missing:
            warnings.warn(f"Report: zone(s) {missing} of {root} have no results; they are left out", stacklevel=3)
        reports = {}
        target = None
        for zone in expected:
            if zone in missing:
                continue
            folder = root / zone
            reports[zone] = self.report(folder, data=None if data is None else data.get(zone), **told)
            target = target or self._stored_target(folder)
        if not reports:
            raise FileNotFoundError(
                f"{root} is a run over the zones {expected}, and none holds this pipeline's results"
            )
        return PanelReport(reports, target=target)

    def _pooled_zones(self, root: Path, zones: Sequence[str]) -> List[str]:
        """Every zone a pooled model's manifest in these zones' folders records (`pooled["zones"]`), each once, in order."""
        found: List[str] = []
        for zone in zones:
            for model in self.models:
                if not isinstance(model, PooledModel):
                    continue
                path = manifest_path(self.results_path(root / zone, model))
                if path.exists():
                    try:
                        found += (json.loads(path.read_text()).get("pooled") or {}).get("zones") or []
                    except (OSError, ValueError, AttributeError):
                        continue
        return list(dict.fromkeys(found))

    def _stored_target(self, folder: Path) -> Optional[str]:
        """The target a model's manifest in `folder` records, or None if none does."""
        for model in self.models:
            path = manifest_path(self.results_path(folder, model))
            if path.exists():
                try:
                    return json.loads(path.read_text()).get("target")
                except (OSError, ValueError):
                    continue
        return None

    def _ordered_ensembles(self) -> List[Ensemble]:
        """The ensembles, each after the ensembles it is made of; `ValueError` for an unknown member or a cycle."""
        by_name = {ensemble.name: ensemble for ensemble in self.ensembles}
        models = {model.name for model in self.models}
        ordered: Dict[str, Ensemble] = {}

        def visit(ensemble: Ensemble, path: Tuple[str, ...]) -> None:
            if ensemble.name in ordered:
                return
            if ensemble.name in path:
                raise ValueError(f"These ensembles are members of each other: {' -> '.join([*path, ensemble.name])}")
            for member in ensemble.members:
                if member in by_name:
                    visit(by_name[member], (*path, ensemble.name))
                elif member not in models:
                    raise ValueError(
                        f"Ensemble {ensemble.name!r}: unknown member {member!r}. The pipeline has the models "
                        f"{sorted(models)} and the ensembles {sorted(by_name)}."
                    )
            ordered[ensemble.name] = ensemble

        for ensemble in self.ensembles:
            visit(ensemble, ())
        return list(ordered.values())

    def _check_before_any_fit(
        self,
        save_dir: Union[str, Path, None],
        validation: Union[Tuple[Any, Any], str, None],
        test: Optional[Tuple[Any, Any]],
        test_start: Optional[str],
        test_end: Optional[str],
        forecast_only: bool,
    ) -> None:
        """What `run` would refuse about the ensembles, for an array task, which runs some models and leaves them to the report."""
        self._check_model_names()
        windows, _, _ = self._resolve_windows(validation, test, test_start, test_end, forecast_only)
        self._check_ensembles(save_dir)
        self._refuse_top_without_validation(self.ensembles, windows)

    def _check_ensembles(self, save_dir: Union[str, Path, None]) -> None:
        """What can be known about the ensembles before any fit, whatever the windows."""
        names = {ensemble.name for ensemble in self.ensembles}
        for ensemble in self._ordered_ensembles():
            ensemble._check()
            nested = [member for member in ensemble.members if member in names]
            if ensemble.needs_validation and nested:
                raise ValueError(
                    f"Ensemble {ensemble.name!r} chooses on validation, so it can't have the ensembles {nested} as members: "
                    "their forecasts are already chosen or averaged on those days. A plain average of ensembles is fine."
                )
            if save_dir:
                path = self.results_path(save_dir, ensemble, "jsonl")
                if (path.exists() or manifest_path(path).exists()) and not is_ensemble_manifest(manifest_path(path)):
                    raise ManifestMismatchError(
                        f"Ensemble {ensemble.name!r} would overwrite {path}, which is not an ensemble's. Rename it, or use a "
                        "different save_dir."
                    )

    @staticmethod
    def _refuse_top_without_validation(ensembles: Sequence[Ensemble], windows: Optional[RunWindows]) -> None:
        if windows is not None and windows.validation is not None:
            return
        for ensemble in ensembles:
            if ensemble.needs_validation:
                raise ValueError(
                    f"Ensemble {ensemble.name!r} chooses its members on validation, and there is no validation window to "
                    "choose on: run with validation=(start, end) or validation='previous_year'."
                )

    def _ensembles_with_their_members(self, available: set) -> List[Ensemble]:
        """The ensembles whose members all have results, in their order; the others are left out with a warning."""
        reported = []
        known = set(available)
        for ensemble in self._ordered_ensembles():
            lacking = [member for member in ensemble.members if member not in known]
            if lacking:
                warnings.warn(
                    f"Report: ensemble {ensemble.name!r} is left out: no results for its member(s) {lacking}. It is never "
                    "averaged over the members that happen to be there.",
                    stacklevel=3,
                )
            else:
                reported.append(ensemble)
                known.add(ensemble.name)
        return reported

    def _compute_ensembles(
        self,
        ensembles: Sequence[Ensemble],
        refs: Dict[str, ModelResultRef],
        stored: Optional[RunWindows],
        save_dir: Union[str, Path, None],
        provenance: Optional[Callable[[], Tuple[str, Optional[str]]]] = None,
    ) -> Tuple[Dict[str, ModelResultRef], Dict[str, List[Dict]]]:
        """`({name: ref} of what a report shows, {name: rows of report.choices()})`, in the order they were added. `provenance`
        is a run's (`FrameHashes.provenance`), which its ensembles' manifests record as its models' do."""
        like = next(iter(refs.values()))
        held = dict(refs)
        leaves: Dict[str, List[str]] = {}
        test_only_names: set = set()
        computed: Dict[str, ModelResultRef] = {}
        choices: Dict[str, List[Dict]] = {}
        for ensemble in ensembles:
            test_only = ensemble.needs_validation or not test_only_names.isdisjoint(ensemble.members)
            if test_only:
                test_only_names.add(ensemble.name)
            path = self.results_path(save_dir, ensemble, "jsonl") if save_dir else None
            leaves[ensemble.name] = list(
                dict.fromkeys(leaf for member in ensemble.members for leaf in leaves.get(member, [member]))
            )
            result = compute(ensemble, held, stored, like, path, test_only, leaves[ensemble.name], provenance)
            held[ensemble.name] = result.ref
            # A ref with no window to show, as a `top` ensemble has in a validation report, is not a row of its tables.
            if result.ref.by_window is None or result.ref.by_window:
                computed[ensemble.name] = result.ref
            if result.choices:
                choices[ensemble.name] = result.choices
        order = [ensemble.name for ensemble in self.ensembles]
        if choices and stored is not None:
            # What the rows of `choices()` count in, whichever part of what is on file this run or report names.
            for ref in [*refs.values(), *computed.values()]:
                for part in [ref, *(ref.by_window or {}).values()]:
                    if part.windows is not None:
                        part.windows = replace(part.windows, mode=stored.mode, segments=stored.segments)
        return {name: computed[name] for name in order if name in computed}, {
            name: choices[name] for name in order if name in choices
        }

    @staticmethod
    def _windows_to_report(
        asked: Optional[RunWindows], test_start: Optional[str], test_end: Optional[str], stored: RunWindows
    ) -> RunWindows:
        """What a report covers after runs with windows: what `report` was told, which must be what the manifests record,
        or all of it if it was told nothing."""
        if asked is None:
            if test_start is None:
                return stored
            asked = RunWindows.fixed(None, (test_start, test_end))
        different = [
            f"{name} {period} (the manifests record {getattr(stored, name)})"
            for name, period in asked.periods
            if period != getattr(stored, name)
        ]
        if different:
            raise ValueError(
                f"These windows are not what the results were made with: {'; '.join(different)}. Pass the windows of the "
                "run, or none to report what the manifests record."
            )
        if asked.validation is not None and asked.test is not None:
            return stored
        # A window of the run alone: the validation window keeps the days it was issued on (`validation_every`).
        return asked if asked.validation is None else replace(asked, validation_every=stored.validation_every)

    def verify(
        self,
        data: Union[pd.DataFrame, Mapping[str, pd.DataFrame]],
        test_start: str,
        test_end: str,
        target: str = "price",
        horizon: Optional[int] = None,
        freq: str = "1h",
        days: int = 5,
        seed: int = 0,
        models: Optional[Sequence[str]] = None,
        positive_control: bool = True,
        rebuild: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        horizons: Optional[Sequence[int]] = None,
        issue_days: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
        co_batch: Optional[bool] = None,
        rerun: Optional[bool] = None,
    ) -> VerifyReport:
        """Check on a sample of issue days that no forecast depends on anything unknown when it is issued.

        For each model, `days` issue days are sampled with `seed` from [test_start, test_end] (the same days for every
        model, whichever ones and in whatever order). On each of them:

        - the leakage probe perturbs every value not known at the issue time, and what the model trains on from the
          issue day on (for a model with `newest_target="published"`, whose boundary is its target's rule, what that rule
          calls unknown). Forecasts must not move; if they do, the status is `leak`.
        - the positive control perturbs the freshest known values: those delivered or first known on the day before the
          issue day or later (see `epftoolbox2.information_set.perturb_recent`). Every forecast must move on some
          sampled day, and on every day some forecast of each horizon; if not, the probe isn't reaching the model and
          the status is `inconclusive`, which doesn't pass either. So is a model with an input that has no availability
          rule, which the probe can't perturb, and a predictor function with an input that no sampled day reads.

        A predictor function of `ctx` is planned on each frame the probe tests, so a list chosen by a value it can't know
        moves with it. A cell it read that holds another value in the perturbed frame is a `leak` too, whether or not the
        noise changed its list.

        `rebuild` treats `data` as source data, before feature engineering: the perturbation goes into the raw columns
        and `rebuild(frame) -> features` reproduces the transformers. Without it a transformer that writes a future
        value into a legally named column can't be seen, since the check only has the column's name to go by. An input
        with a non-static rule that `rebuild` doesn't build from its argument is `inconclusive` too.

        Only copies of the models run, in memory, so a later `run()` is unaffected. Each issue day costs four one-day
        runs per model (two for the probe, two for the control), six with `rebuild` or a target among the predictors; the
        rerun adds two, and the co-batch check as many as the probe and control take again, over the issue day's block and
        the other days of its call. A model that isn't deterministic shows up as a leak, or with `rerun` as `inconclusive`
        where a move is within `verification.NOISE_MARGIN` times the day's rerun difference. With `engine="on"` each run
        of a model the engine takes goes through it, as a group of one: its forecasts are those of any group, so the check
        covers the engine's way in too.

        `verify` probes the period you give it, not the windows of a run: to check a validation period as well as a test
        period, call it on each.

        The boundary is the same at every `availability` level: a late predictor is a `leak` also where `run` only lists
        it. With "warn" or "none" a predictor function that reads a late `ctx.value()` is a `leak` too; with "raise" that
        read raises `InformationSetError`. A predictor function is planned on the whole test period, so what it uses late
        on any issue day is a `leak`, sampled or not.

        Args:
            data: The frame the models run on, or the source data with `rebuild`.
            test_start: First issue day to sample from.
            test_end: Last issue day to sample from.
            target: Target column.
            horizon: Max horizon: every horizon from 1 to it, 1 if neither it nor `horizons` is given. Leave no horizon you
                run out: a leak in another one can't be seen (`run` defaults to 7).
            freq: "1h", "15min" or "1D"; must match the data.
            days: Issue days per model (all of the period if it has fewer).
            seed: Picks the issue days and the noise.
            models: Names of the models to verify. None verifies them all.
            positive_control: False skips the control, so `ok` only says that no forecast moved.
            rebuild: Builds the model's input from `data` (see above).
            horizons: The horizons to verify, as `run(horizons=...)` lists them, in place of `horizon`; passing both raises.
            issue_days: The weekdays a run issues on (`run(issue_days=["friday"])`): only days of these are sampled, and a
                predictor function is planned on them alone, so what it would read late on another weekday is no leak of
                this run's. None (or all seven) samples every day.
            device: Set on every copy of a model with an attribute named `device` ("cpu" to verify a GPU model on the
                CPU); `report.devices` names them, and a warning says so when no model has one.
            co_batch: The probe once more over runs widened to the other items of the issue day's call
                (`information_set.co_batch_probe`). None: for models that predict together, name their items or have a
                card (`verification.checked_by_default`).
            rerun: Run the issue day twice first, from other seeds of the global generators; any difference makes the model
                `inconclusive`. None: as `co_batch`.

        Raises:
            InformationSetError: The pipeline has no information set, or a checked column has no rule and the
                information set's `unknown` is "raise".
            MissingExtraError, MissingWeightsError: A model verified lacks its extra, a runtime package within its card's
                pins or its weights (the pre-flight of `run`, before any probe; `models._cards.preflight`).
        """
        horizon, horizons = resolve_horizons(horizon, horizons, 1)
        weekdays = resolve_issue_days(issue_days)
        if self.information_set is None:
            raise InformationSetError(
                "verify() needs to know when forecasts are issued: pass issue_time or information_set to ModelPipeline"
            )
        if not self.models:
            raise ValueError("At least one model is required")
        self._check_model_names()
        chosen = self._select_models(models)
        preflight(_carded(chosen))  # the extras, runtime packages and weights of the models verified, first
        if isinstance(data, Mapping):
            return self._verify_zones(
                data,
                chosen,
                test_start,
                test_end,
                target,
                horizon,
                horizons,
                freq,
                days,
                seed,
                positive_control,
                rebuild,
                weekdays,
            )
        self._refuse_pooled("verify")
        # With `rebuild` the frame the models read is the rebuilt one, whose zone is the market's (a source can be in UTC).
        info = self.information_set
        if info.issue_timezone == DATA_ZONE:
            info = info.in_data_zone((data if rebuild is None else rebuild(data.copy())).index)
        check_issue_days_in(resolve_date(test_start), resolve_date(test_end), weekdays)
        run_days = sample_issue_days(data.index, test_start, test_end, days, seed, issue_days=weekdays)
        report = VerifyReport(issue_days=run_days)
        with pool_native_threads():  # one hold for every probe's pool, as in `run`
            for model in chosen:
                report.checks.extend(
                    verify_model(
                        model,
                        data,
                        run_days,
                        info,
                        target=target,
                        freq=freq,
                        seed=seed,
                        control=positive_control,
                        rebuild=rebuild,
                        test_period=(test_start, test_end),
                        availability=self.availability,
                        **horizon_arguments(horizon, horizons),
                        **({} if weekdays is None else {"issue_days": issue_day_names(weekdays)}),
                        **({} if self.engine == ENGINE_OFF else {"engine": self.engine}),
                        device=device,
                        co_batch=co_batch,
                        rerun=rerun,
                    )
                )
        if device is not None and not report.devices:
            warnings.warn(
                f"verify(device={device!r}): no model verified has a `device`, so it changed nothing", stacklevel=2
            )
        return report

    def _verify_zones(
        self,
        frames: Mapping[str, pd.DataFrame],
        chosen: List[Any],
        test_start: str,
        test_end: str,
        target: str,
        horizon: Optional[int],
        horizons: Optional[Sequence[int]],
        freq: str,
        days: int,
        seed: int,
        positive_control: bool,
        rebuild: Optional[Callable[[pd.DataFrame], pd.DataFrame]],
        weekdays: Optional[Tuple[int, ...]],
    ) -> VerifyReport:
        """`verify` of a run over zones: each plain model on each zone's frame (`verify_model`), each `PooledModel` over all
        of them at once (`models._pooled.verify_pooled`); every check named "<model> (<zone>)". The issue days are sampled from
        the first zone's frame, the same for every model and zone."""
        if rebuild is not None:
            raise ValueError(
                "verify() over zones takes the frames the models read: rebuild= is for one frame's source data"
            )
        zones = self._check_zones(frames)
        infos = {zone: self.information_set.in_data_zone(frames[zone].index) for zone in zones}
        check_issue_days_in(resolve_date(test_start), resolve_date(test_end), weekdays)
        run_days = sample_issue_days(frames[zones[0]].index, test_start, test_end, days, seed, issue_days=weekdays)
        report = VerifyReport(issue_days=run_days)
        horizon_args = horizon_arguments(horizon, horizons)  # resolved by `verify`
        planned = tuple(range(1, horizon + 1)) if horizons is None else tuple(horizons)
        with pool_native_threads():  # one hold for every probe's pool, as in `run`
            for model in chosen:
                if isinstance(model, PooledModel):
                    if weekdays is not None:
                        raise ValueError(
                            f"verify() of PooledModel {model.name!r} probes every day of a refit block; issue_days= is for "
                            "plain models in this step"
                        )
                    check_pooled_model(model.model)
                    member = member_of(model, None)
                    needed = needed_columns(member, freq, planned)
                    zone_frames = {zone: with_missing_columns(model, zone, frames[zone], needed) for zone in zones}
                    # What the probe perturbs in each zone: every column with a rule, and those the model reads, but not a column
                    # of zeros `missing="zero"` added, which holds nothing a forecast could know late.
                    columns = {
                        zone: [
                            name
                            for name in frames[zone].columns
                            if name in {target, *needed}
                            or (isinstance(name, str) and infos[zone].resolve(name).rule is not None)
                        ]
                        for zone in zones
                    }
                    report.checks.extend(
                        verify_pooled(
                            model,
                            zone_frames,
                            infos,
                            run_days,
                            target,
                            freq,
                            horizon_args,
                            columns,
                            seed,
                            positive_control,
                        )
                    )
                    continue
                for zone in zones:
                    checks = verify_model(
                        model,
                        frames[zone],
                        run_days,
                        infos[zone],
                        target=target,
                        freq=freq,
                        seed=seed,
                        control=positive_control,
                        test_period=(test_start, test_end),
                        availability=self.availability,
                        **horizon_args,
                        **({} if weekdays is None else {"issue_days": issue_day_names(weekdays)}),
                        **({} if self.engine == ENGINE_OFF else {"engine": self.engine}),
                    )
                    report.checks.extend(replace(check, model=f"{check.model} ({zone})") for check in checks)
        return report

    def explain(
        self,
        data: pd.DataFrame,
        model: str,
        test: Optional[Tuple[Any, Any]] = None,
        horizon: Optional[int] = None,
        hour: Optional[int] = None,
        target: str = "price",
        freq: str = "1h",
        echo: bool = True,
        horizons: Optional[Sequence[int]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Show what a model's predictors mean for one forecast: an optional view, with no effect on a later run.

        The result has `predictors`, one row per predictor of the forecast of period `hour` at `horizon`: `predictor`,
        `target-relative` (when its value was delivered, relative to the target day T = D + horizon, so at horizon 2
        `price_d-1` at hour 12 is `T-3 12:00`), `rule`, `known` (relative to the issue day D, on the issue day it is
        latest), and `status` (`ok`, `late`, `late on 3 of 10 days`, `no rule`, `no issue time`, `not in data`). `grouped`
        is the same with runs of slots folded into one row (`price_h00 … price_h23_d-1 (24)`, `T-2 00:00 … 23:00`).

        For a predictor function of `ctx`, `test` is required and `lists` has one row per horizon up to `horizon` (or per
        horizon `horizons` lists) and list that horizon uses: how many forecasts and issue days use it, its first and last
        issue day, its periods, and what it `adds` to and `drops` from the most used list of that horizon. `predictors` then
        shows the list used most at (`horizon`, `hour`), judged on the days it is used. A late `ctx.value()` doesn't stop the view; see
        `report.availability()` for what a function read.

        With `train_on="matching"` lists `lists` also has `smallest days`, `median days` and `largest days`: how many days the
        forecasts of such a list would train on, as a run counts them, without fitting anything; a dash for any other list.

        Args:
            data: The frame the models run on.
            model: The model's name.
            test: (first, last) issue day. None takes every day of `data`, for fixed lists only.
            horizon: Days ahead of the issue day of the forecast shown, 1 if neither it nor `horizons` is given.
            hour: Its period of the target day: 0-23, 0-95 for 15-minute data, 0 for a daily frame. None is 12 (noon at 1h),
                or 0 at "1D".
            target: Target column.
            freq: "1h", "15min" or "1D"; must match the data.
            echo: Print the tables.
            horizons: The horizons a run lists (`run(horizons=...)`), in place of `horizon`: the forecast shown is at the
                largest, and `lists` has these alone. Passing both raises.
        """
        horizon, horizons = resolve_horizons(horizon, horizons, 1)
        if hour is None:
            hour = 0 if freq == DAILY else 12
        chosen = self._select_models([model])[0]
        if isinstance(chosen, PooledModel):
            chosen = chosen.model  # its predictors and rules in one zone's frame: what each zone's member reads
        if isinstance(chosen, ResidualModel):
            self._resolve_bases()  # a base="name" before the copy: the copy keeps what it names
        chosen = copy.deepcopy(chosen)  # planning leaves state on a model; this is only a look
        if test is not None:
            test = (resolve_date(test[0]), resolve_date(test[1]))
        info = self._information_set_for(data)
        if isinstance(chosen, ResidualModel):
            return self._explain_residual(chosen, data, info, test, horizon, hour, target, freq, echo, horizons)
        return explain_model(chosen, data, info, test, horizon, hour, target, freq, echo, horizons=horizons)

    def _explain_residual(self, model, data, info, test, horizon, hour, target, freq, echo, horizons):
        """`explain` of a `ResidualModel`: its learner's predictors, the base's forecasts among them shown as such (known at the
        issue time by construction, never the frame's), and the base's own predictors (`base_predictors`, `base_grouped`)."""
        from ..models.residual import is_base_name

        out = explain_model(model, data, info, test, horizon, hour, target, freq, False, horizons=horizons)
        table = out["predictors"].copy()
        on = table["predictor"].map(is_base_name)
        table.loc[on, ["rule", "known", "status"]] = ["the base's forecast, issued on D", "D", "base forecast"]
        out["predictors"], out["grouped"] = table, grouped_table(table)
        base = model._base_model()
        found = explain_model(base, data, info, test, horizon, hour, target, freq, False, horizons=horizons)
        out["base_predictors"], out["base_grouped"] = found["predictors"], found["grouped"]
        if echo:
            print(
                f"{model.name}: horizon {horizon}, period {hour}; the learner's inputs (base_fc_* are its base's forecasts)"
            )
            print(out["grouped"].to_string(index=False))
            print(f"{base.name}: the base's inputs")
            print(out["base_grouped"].to_string(index=False))
        return out

    def _select_models(self, names: Optional[Sequence[str]]) -> List[BaseModel]:
        if names is None:
            return self.models
        names = [names] if isinstance(names, str) else list(names)
        by_name = {model.name: model for model in self.models}
        unknown = [name for name in names if name not in by_name]
        if unknown:
            raise ValueError(f"Unknown model(s) {unknown}. The pipeline has: {list(by_name)}")
        return [by_name[name] for name in dict.fromkeys(names)]

    @staticmethod
    def results_path(save_dir: Union[str, Path], model: BaseModel, save_format: Optional[str] = None) -> Path:
        """Where a model's results live inside `save_dir`: `<name>.jsonl`, or the directory `<name>.parquet` (`save_format`).
        Without `save_format`, the one that exists, JSONL if neither does; both raise `ResultsFormatError`, as a model's
        results are in one format (a convert that stopped midway leaves both, and `epf results convert` again finishes it).
        An ensemble's are always JSONL."""
        stem = Path(save_dir) / model.name.lower().replace(" ", "_")
        if save_format is not None:
            return stem.with_name(stem.name + SAVE_FORMATS[check_save_format(save_format)])
        jsonl, parquet = (
            stem.with_name(stem.name + suffix) for suffix in (SAVE_FORMATS["jsonl"], SAVE_FORMATS["parquet"])
        )
        if jsonl.exists() and parquet.is_dir():
            raise ResultsFormatError(
                f"{model.name}: results in both formats, {jsonl} and {parquet}. A model's results are in one: run "
                f"`epf results convert {Path(save_dir)} --to <format>` again to finish a conversion, or remove the one no run "
                "wrote."
            )
        return parquet if parquet.is_dir() else jsonl

    def _check_save_format(self, save_dir: Union[str, Path], save_format: str) -> None:
        """Before any fit: no model may have results in the other format than this run writes (`check_no_other_format`), and a
        `ResidualModel`'s records, which hold `base_prediction`, are JSONL."""
        for model in self.models:
            if isinstance(model, ResidualModel) and save_format != "jsonl":
                raise ValueError(
                    f"{model.name}: a ResidualModel's records hold `base_prediction`, which the {save_format} results schema "
                    "has no column for: run with save_format='jsonl'"
                )
            check_no_other_format(self.results_path(save_dir, model, save_format))

    def _resolve_bases(self) -> None:
        """Give each `ResidualModel` with `base="name"` the model of that name added before it (`bind_base`; it runs a copy). A
        name added after it, or of no model of the list, raises; so does a ResidualModel bound in another pipeline. The one
        exception is a `Workflow` array task's slice (`_bases_from_the_whole_list`, set by the Workflow), whose models were
        resolved on the whole list before it was sliced: there a bound base whose model is in another task's slice stays."""
        for place, model in enumerate(self.models):
            if not isinstance(model, ResidualModel) or not isinstance(model.base, str):
                continue
            earlier = {other.name: other for other in self.models[:place]}
            if model.base in earlier:
                model.bind_base(earlier[model.base])
            elif not (getattr(self, "_bases_from_the_whole_list", False) and model._named_base is not None):
                later = [other.name for other in self.models[place + 1 :]]
                raise ValueError(
                    f"{model.name}: base={model.base!r} "
                    + ("is added after it" if model.base in later else "names no model of this pipeline")
                    + ": a ResidualModel's base is a model added before it, or a model itself. Models: "
                    + str([m.name for m in self.models])
                )

    def _check_residual_models(self, save_format: str) -> None:
        """What a run refuses of its `ResidualModel`s before anything else, the data pipeline of a `Workflow` included: each
        `base="name"` (`_check_model_names`, which resolves them) and results in another format than JSONL, whose schema has no
        column for `base_prediction`. For every model of the list, so a Workflow array task refuses even where its own slice
        holds the base alone."""
        self._check_model_names()
        residual = [m.name for m in self.models if isinstance(m, ResidualModel)]
        if residual and save_format != "jsonl":
            raise ValueError(
                f"{', '.join(residual)}: a ResidualModel's records hold `base_prediction`, which the {save_format} results "
                "schema has no column for: run with save_format='jsonl' (model_pipeline.save_format in a Workflow)"
            )

    def _check_model_names(self) -> None:
        """Every model's and ensemble's name, and the file of each `ResidualModel`'s base, once; and each `base="name"` resolved
        (`_resolve_bases`)."""
        named = [*self.models, *self.ensembles]
        names = [m.name for m in named]
        stems = [self.results_path("", m, "jsonl").name for m in named]
        stems += [
            ResidualModel.base_path(self.results_path("", m, "jsonl")).name
            for m in self.models
            if isinstance(m, ResidualModel)
        ]
        if len(set(names)) != len(names) or len(set(stems)) != len(stems):
            what = "Model and ensemble names" if self.ensembles else "Model names"
            raise ValueError(
                f"{what} must be unique (also after lower-casing and replacing spaces, since they name the "
                f"result files, and a ResidualModel's base writes <name>.base): {names}"
            )
        self._resolve_bases()

    def _check_evaluators_for(self, freq: str) -> None:
        """An evaluator that can't score results at `freq` raises before any fit: `EconomicEvaluator` schedules a strategy over
        the periods of a day, and a daily run has one."""
        if freq == DAILY and any(isinstance(evaluator, EconomicEvaluator) for evaluator in self.evaluators):
            raise ValueError(
                "EconomicEvaluator schedules a trading strategy over the periods of a day, and with freq='1D' a day has "
                "one forecast. Remove it from this pipeline, or score an hourly run of the same models."
            )

    def _check_source_columns(self, data: Optional[pd.DataFrame], required: bool = False) -> None:
        """An evaluator that reads columns of the data (`ScaledMAEEvaluator`) would only fail at the first table, after every
        fit: a column the data lacks raises here, and so does no data at all where it is `required` (a report from files)."""
        readers = [e for e in self.evaluators if getattr(e, "source_columns", ())]
        if not readers:
            return
        columns = list(dict.fromkeys(c for e in readers for c in e.source_columns))
        names = [f"{type(e).__name__}({', '.join(map(repr, e.source_columns))})" for e in readers]
        if data is None:
            if required:
                raise ValueError(
                    f"{', '.join(names)} read {columns} from the data the models ran on: pass data=... to report() (the "
                    "frame of the run; a Workflow passes its data pipeline's)."
                )
            return
        missing = [c for c in columns if c not in data.columns]
        if missing:
            raise ValueError(
                f"{', '.join(names)} read the column(s) {missing}, which the data doesn't have (it has {len(data.columns)} "
                "columns). Fix the column name or add it to the data; nothing was fitted."
            )

    def _check_benchmarks(self) -> None:
        """A mistyped benchmark would only show as empty columns, after every model has run."""
        names = [m.name for m in [*self.models, *self.ensembles]]
        for evaluator in self.evaluators:
            if isinstance(evaluator, DMEvaluator) and evaluator.benchmark not in names:
                raise ValueError(f"DMEvaluator: unknown benchmark {evaluator.benchmark!r}; models: {names}")

    @staticmethod
    def _check_predictors_can_be_saved(component: Any) -> None:
        """A model built with callable predictors keeps ``predictors == []`` and the function privately, so a plain
        dump would write an empty list: the reloaded pipeline would be a different study."""
        predictors = getattr(component, "predictors", None)
        if getattr(component, "_predictors_fn", None) is not None or callable(predictors):
            what = "its predictors are a function"
        elif isinstance(predictors, (list, tuple)) and any(callable(p) for p in predictors):
            what = "its predictors contain a function"
        else:
            return
        raise TypeError(
            f"Cannot save model {getattr(component, 'name', type(component).__name__)!r} "
            f"({type(component).__name__}): {what}, and callable predictors can't be saved to YAML yet. "
            "List the predictors explicitly instead: strings, with '{horizon}' where a column depends on the horizon. "
            "A function of `ctx` at the top level of a module can be saved, by its import path."
        )

    def _serialize_component(self, component: Any, omit: Sequence[str] = ()) -> Dict[str, Any]:
        """A component as `{class, params}`: its public attributes (but `omit`), each through `_serialize_param`."""
        class_name = type(component).__name__
        self._check_predictors_can_be_saved(component)
        label = f"{class_name}({component.name!r})" if isinstance(component, BaseModel) else class_name
        params = {
            key: self._serialize_param(component, key, value, f"{label}.{key}")
            for key, value in getattr(component, "__dict__", {}).items()
            if not key.startswith("_") and key not in omit
        }
        if getattr(component, "_ctx_fn", None) is not None:
            params["predictors"] = {"function": importable_path(component._ctx_fn, label)}
        if isinstance(component, BaseModel):
            for key, default in component._WRITTEN_WHEN_SET.items():
                if key in params and params[key] == default:
                    del params[key]  # written only when set, so files saved before it existed don't move
        out = {"class": class_name, "params": params}
        search = getattr(component, "_search", None)
        if isinstance(component, BaseModel) and isinstance(search, dict):
            # Where a search chose the model (`tune.SearchResult.best`): no setting of it, but what a run reads to refuse a test
            # on the days it was chosen on and to show its row in `report.choices()`, so a Workflow or an array task rebuilding
            # the model from YAML has it too. Written only for such a model.
            out["search"] = json.loads(json.dumps(search))
        return out

    def _serialize_param(self, owner: Any, key: str, value: Any, where: str) -> Any:
        """One parameter `key` of the component `owner` as YAML, the one place a parameter's kind is decided: a model inside it as
        `{class, params}` where its class lists `key` in `_NESTED_MODELS` (`ResidualModel`'s base and learner,
        `PooledModel`'s model; what `_load_component` rebuilds), less what its `_NESTED_OMIT` leaves out of that model; a
        strategy inside an evaluator the same; a class under a key its class lists in `_CLASS_PARAMS` (`TorchModel`'s
        `module`) by its import path (`class_path`, which raises for one defined in `__main__` or a function, which a load
        couldn't import; the component takes the path back); anything else a plain value (`to_yaml_value`, which raises for
        what YAML can't hold, a class among it). A model under a key the class doesn't list raises: a load couldn't rebuild
        it."""
        if inspect.isclass(value) and key in getattr(type(owner), "_CLASS_PARAMS", ()):
            return class_path(value, where)
        if isinstance(value, (BaseModel, PooledModel)):
            if key not in getattr(type(owner), "_NESTED_MODELS", ()):
                raise TypeError(
                    f"Cannot save {where}: it is a model ({type(value).__name__}), and {type(owner).__name__} doesn't list "
                    f"{key!r} in its _NESTED_MODELS, so a load couldn't rebuild it"
                )
            return self._serialize_component(value, omit=getattr(type(owner), "_NESTED_OMIT", {}).get(key, ()))
        if type(value).__name__ in COMPONENT_REGISTRY["strategies"]:
            return self._serialize_component(value)
        return to_yaml_value(value, where)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "models": [self._serialize_component(m) for m in self.models],
            **({"ensembles": [self._serialize_component(e) for e in self.ensembles]} if self.ensembles else {}),
            "evaluators": [self._serialize_component(e) for e in self.evaluators],
            "exporters": [self._serialize_component(e) for e in self.exporters],
        }
        if self.information_set is not None:
            out["information_set"] = self.information_set.to_dict()
        if self._availability is not None:  # what was asked for; left out, a load takes the default again
            out["availability"] = self._availability
        if self.engine != ENGINE_OFF:  # written only when set, so files saved before it existed don't move
            out["engine"] = self.engine
        return out

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        # The text is built before the file is opened: a pipeline that can't be saved must not truncate it.
        path.write_text(dump_yaml(self.to_dict()))
        logger.info(f"Pipeline: Saved configuration to {path}")

    @classmethod
    def _load_component(cls, component_type: str, config: Dict[str, Any]) -> Any:
        class_name = config["class"]
        params = dict(config.get("params") or {})
        if class_name not in COMPONENT_REGISTRY[component_type]:
            kind = "strategy" if component_type == "strategies" else component_type[:-1]
            raise ValueError(f"Unknown {kind}: {class_name}")
        if component_type == "evaluators" and isinstance(params.get("strategy"), dict):
            params["strategy"] = cls._load_component("strategies", params["strategy"])
        if component_type == "models" and isinstance(params.get("predictors"), dict):
            params["predictors"] = cls._load_predictor_function(class_name, params["predictors"])
        module = importlib.import_module(COMPONENT_REGISTRY[component_type][class_name])
        component_class = getattr(module, class_name)
        # A model inside a model (`PooledModel`'s model, `ResidualModel`'s base and learner), saved as {class, params}; a name
        # stays a name.
        for key in getattr(component_class, "_NESTED_MODELS", ()):
            if isinstance(params.get(key), dict):
                params[key] = cls._load_component("models", params[key])
        component = component_class(**params)
        if component_type == "models" and isinstance(config.get("search"), dict):
            component._search = dict(config["search"])  # where a search chose it (`_serialize_component`)
        return component

    @staticmethod
    def _load_predictor_function(class_name: str, saved: Dict[str, Any]) -> Callable:
        if set(saved) != {"function"}:
            raise ValueError(f"{class_name}: predictors must be a list or {{function: 'module:name'}}, got {saved}")
        return load_function(saved["function"])

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelPipeline":
        path = Path(path)
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        info = config.get("information_set")
        pipeline = cls(
            information_set=InformationSet.from_dict(info) if info else None,
            availability=config.get("availability"),
            engine=engine_from_yaml(config.get("engine", ENGINE_OFF)),  # `engine: on` unquoted reads as True
        )
        for c in config.get("models", []):
            pipeline.add_model(cls._load_component("models", c))
        for c in config.get("ensembles", []):
            pipeline.add_ensemble(cls._load_component("ensembles", c))
        for c in config.get("evaluators", []):
            pipeline.add_evaluator(cls._load_component("evaluators", c))
        for c in config.get("exporters", []):
            pipeline.add_exporter(cls._load_component("exporters", c))
        logger.info(f"Pipeline: Loaded configuration from {path}")
        return pipeline
