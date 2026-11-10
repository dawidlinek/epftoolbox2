"""The library's report reproduces the model-zoo leaderboard from stored research forecasts (Tier A of step 6).

This checks the path from result records to the leaderboard, not the models. Each member's stored validation and test
forecasts are written as result records (`window`, `day_in_test`, `actual` from the stored prices) through the library's store
into a save directory, and `ModelPipeline.report(
save_dir, validation=, test=)` with `AverageEnsemble`, `MAEEvaluator`, `rMAEEvaluator` and `DMEvaluator` must give the numbers
of `zoo_merge/out/leaderboard.csv` and of the follow-up tables behind plan section 4, within 1e-9: test MAE, rMAE, the ratios
to LEAR_CWA and TabPFN v2 and the DM p values, validation MAE, the members each top-k ensemble chooses and the variant picks.

Fixture ``tests/fixtures/zoo/``, built by ``epftoolbox2-research/library_check_step6/zoo_fixture/build.py`` (outside the repo).
The builder copies exactly the files it needs from S02 ``~/epf2_modeling_research/{zoo,recent}/flat`` and ``recent/data``,
indexes them with merge.py's rules (files written after merge.py ran are kept out of its index), recomputes merge.py's numbers
and stops if any kept row differs from leaderboard.csv by more than 1e-12 (the largest difference is 3.6e-15), if a top-k
ensemble chooses other members or if a variant pick differs.

- ``DE_LU_honest.npz``, ``AT_honest.npz``. ``dates`` are the target days of ``actual``, which is ``P[v0-7:i1]`` of
  ``recent/data/<market>.npz`` (``test = (i0, i1)``, ``v0 = i0 - 364``; float64, as the quarter-hour means of the test year
  have more than 4 decimals): rows 7..370 are the validation target days 2024-10-02..2025-09-30, rows 371..735 the test days
  2025-10-01..2026-09-30, and the weekly naive is ``actual[0:729]``. Then one array per stored series,
  ``<family>__<variant>__<seed>``, read from ``pred_<family>_<market>_<set>_<variant>[_s<k>]_{val,test}[_partXofY].csv`` as
  merge.py reads them (``uni_`` variants are of the ``uni`` input set, ``s0`` is a file without a seed suffix, and t0afts1/s2
  are seeds of t0aft): 729 x 24 values, validation then test, or 365 x 24 for Mitra-v2 in AT, which zoo_tabfm ran on the test
  year alone. Each series is int32 fixed point x 10^4, which the builder proved by a round trip gives back every value as
  ``float(text)`` bit for bit, stored as the 4 little-endian byte planes of the int32 values (uint8, shape (4, rows * 24)),
  which compress about 20 % better: ``decode`` below.
- The series: the 12 best by validation MAE of each of merge.py's candidate sets (all; clean, without t0-beta and Sundial;
  strict, also without Chronos-2 and Toto) with their validation-selected variants, which are every member a top-k can choose
  since merge.py searches k from 1 to 12; the members of plan section 4's table; the four LEAR windows; the three DNN windows,
  as DNN_CWA is among DE's 12 best; and every variant of Chronos-2 and TimesFM-3, so that the library picks the variant. Seeds
  are stored separately and averaged here. The follow-up members came after merge.py: Timestral-209M zero-shot, the monthly
  t0-alpha re-fine-tune (t0aftroll, 3 seeds) and the hybrids fixed a priori (learhgb and, in DE, ridgehgb at w0).
- ``expected.csv``: the kept rows of ``leaderboard.csv`` copied verbatim (val, test, vs_lear, vs_tabpfn, p_lear, p_tabpfn,
  variant), each top-k ensemble with its members in rank order from ``leaderboard.md``; the greedy (Caruana) rows, kept for
  reference only, as the library has no such construct; and the follow-up rows, recomputed by the builder and checked there
  against their printed tables at printed precision. ``source`` gives the file and line of each row.
- ``sources.csv``: every source file with its sha256 and size.

merge.py's conventions: MAE in EUR/MWh; ``rMAE`` = test MAE / the weekly naive's; ``vs_*`` = MAE / the reference's; ``p_*`` the
one-sided DM p value (epftoolbox's plain variance, on the daily mean absolute error) that the row beats the reference, which is
``norm.sf`` of the library's statistic. A forecast for target day D is issued on D - 1 (horizon 1). Every day of both years has
24 values (the research grid), so the manifests, the minimal ones a run over these windows writes, put the records on a UTC
clock, where every day has 24 hours.

What the library does differently, pinned here: it chooses on the validation forecasts for target days before the first test
issue day (`Segment.serves`), so the forecast for 2025-09-30, issued on 2025-09-29, does not count. It chooses on 363 days and
8,712 forecasts, merge.py on 364 days and 8,736. Every choice is the same but DE's all-members top-k, where the library keeps 7
members (not toto2b1) and merge.py 8; that row's numbers are checked on a plain average of merge.py's 8. The leaderboard's
validation MAEs are compared with a report of the validation year's 8,736 forecasts as a test window.

One test per setting builds everything it checks (a module fixture this costly would be built again in every xdist worker): 15-19
s each on a 12-core machine, most of it the report reading each member's records once per ensemble that averages it. The records
go through the Parquet store, which that report reads 2.6 times faster than JSONL files; a JSONL store gives the same numbers.
"""

import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from epftoolbox2.ensembles import AverageEnsemble
from epftoolbox2.evaluators import DMEvaluator, MAEEvaluator, rMAEEvaluator
from epftoolbox2.models import OLSModel
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results import EvaluationReport, Manifest, RunWindows
from epftoolbox2.results.manifest import TRAINING_CUT_BY_DATE, manifest_path
from epftoolbox2.results.parquet_store import ParquetResultStore
from epftoolbox2.results.store import with_status

FIXTURE = Path(__file__).parent / "fixtures" / "zoo"
EXPECTED = pd.read_csv(FIXTURE / "expected.csv", keep_default_na=False)
SETTINGS = ("DE_LU_honest", "AT_honest")
TOL = 1e-9
SCALE = 10_000

# Issue days: a forecast for target day D is issued on D - 1.
VALIDATION = ("2024-10-01", "2025-09-29")  # target days 2024-10-02..2025-09-30
TEST = ("2025-09-30", "2026-09-29")  # target days 2025-10-01..2026-09-30
N_VALIDATION, N_TEST = 364, 365

# merge.py's candidate sets of the top-k ensembles; k runs from 1 to 12.
POSSIBLY, UNKNOWN, LOWRISK = {"t0b", "t0bft"}, {"sundial"}, {"toto", "toto2b1"}
SETS = {
    "ENS_topk_all": lambda k: True,
    "ENS_topk_clean": lambda k: k not in POSSIBLY | UNKNOWN,
    "ENS_topk_strict": lambda k: k not in POSSIBLY | UNKNOWN | LOWRISK and k != "chronos2",
}
TOP = 12
VARIANT_PICKS = ("chronos2", "timesfm3")
LEAR_WINDOWS = ("w56", "w84", "w364", "w728")
BENCHMARKS = {"lear": "LEAR_CWA", "tabpfn": "tabpfn2"}
FOLLOW_UP_FAMILY = {"learhgb_w0": "learhgb", "ridgehgb_w0": "ridgehgb"}
# Chosen on 363 validation days, where merge.py chose on 364: the one choice that differs (see the module docstring).
CHOSEN_ON_363_DAYS = {
    ("DE_LU_honest", "ENS_topk_all"): {"t0b", "mitra2", "tabpfn2", "tabicl2", "t0bft", "t0aft", "tirex2"}
}


def decode(stored: np.ndarray) -> np.ndarray:
    """A series of the fixture as float64: the 4 byte planes of its int32 values x 10^4, or float64 as stored."""
    if stored.dtype == np.uint8:
        return np.ascontiguousarray(stored.T).view("<i4").reshape(-1, 24) / SCALE
    return stored


def seeds(data: dict, family: str, variant: str) -> list:
    """The stored seeds of a member, in seed order."""
    return [data[key] for key in sorted(data) if key.startswith(f"{family}__{variant}__")]


def plain_variant(text: str) -> str:
    return text.split(" (")[0]  # "ctx364_exog (3 seeds)"


class Setting:
    """One market: its members as arrays (729 x 24, validation then test), the expected rows and the ensembles."""

    def __init__(self, name: str):
        with np.load(FIXTURE / f"{name}.npz") as z:
            data = {key: (z[key] if key == "dates" else decode(z[key])) for key in z.files}
        self.name = name
        self.dates = data.pop("dates")
        actual = data.pop("actual")
        self.y = actual[7:]
        self.rows = EXPECTED[EXPECTED.setting == name].set_index("model")
        self.members = {}
        self.model_of = {}  # leaderboard row -> the library's model or ensemble that gives it
        for model, row in self.rows.iterrows():
            variant = plain_variant(row.variant)
            if row.kind == "member" and model == "DNN_CWA":  # merge.py's mean of the windows: a plain member here
                self.members[model] = np.mean([data[k] for k in sorted(data) if k.startswith("dnn__")], axis=0)
            elif row.kind == "member" and model not in ("LEAR_CWA", *VARIANT_PICKS):
                self.members[model] = np.mean(seeds(data, model, variant), axis=0)
            elif row.kind == "follow-up":
                self.members[model] = np.mean(seeds(data, FOLLOW_UP_FAMILY.get(model, model), variant), axis=0)
            self.model_of[model] = model
        for family in VARIANT_PICKS:
            for key in sorted(k for k in data if k.startswith(f"{family}__")):
                self.members[key.rsplit("__", 1)[0]] = data[key]
        for window in LEAR_WINDOWS:
            self.members[f"lear_{window}"] = data[f"lear__{window}__s0"]
        self.members["naive_weekly"] = actual[:-7]
        for setting, ensemble in CHOSEN_ON_363_DAYS:
            if setting == name:
                self.model_of[ensemble] = f"{ensemble}_merge_py"

    def member(self, model: str) -> str:
        """The plain member that stands for a leaderboard row in a top-k ensemble: a variant pick is an ensemble, so the
        variant merge.py picked (the pick must agree)."""
        return f"{model}__{plain_variant(self.rows.loc[model, 'variant'])}" if model in VARIANT_PICKS else model

    def best12(self, ensemble: str) -> list:
        """The 12 best members of a candidate set by merge.py's validation MAE, in that order."""
        rows = self.rows[(self.rows.kind == "member") & np.array([SETS[ensemble](k) for k in self.rows.index])]
        ranked = sorted(rows.index, key=lambda k: float(rows.loc[k, "val"]))[:TOP]
        assert "LEAR_CWA" not in ranked  # an ensemble can't be a member of a top ensemble
        return [self.member(k) for k in ranked]

    def merge_py_members(self, ensemble: str) -> list:
        chosen = {self.member(m) for m in self.rows.loc[ensemble, "members"].split(";")}
        return [m for m in self.best12(ensemble) if m in chosen]

    def ensembles(self) -> list:
        out = [AverageEnsemble([f"lear_{w}" for w in LEAR_WINDOWS], name="LEAR_CWA")]
        out += [AverageEnsemble(self.best12(name), top="validation", name=name) for name in SETS]
        for family in VARIANT_PICKS:
            out.append(AverageEnsemble([m for m in self.members if m.startswith(f"{family}__")], top=1, name=family))
        for setting, ensemble in CHOSEN_ON_363_DAYS:
            if setting == self.name:
                out.append(AverageEnsemble(self.merge_py_members(ensemble), name=f"{ensemble}_merge_py"))
        return out


def write_members(setting: Setting, save_dir: Path) -> None:
    """Each member's records through the library's store, beside the manifest a run over these windows writes."""
    first = datetime.date.fromisoformat(VALIDATION[0])
    keys = []
    for i in range(N_VALIDATION + N_TEST):
        day = first + datetime.timedelta(days=i)
        window, day_in_window = ("validation", i) if i < N_VALIDATION else ("test", i - N_VALIDATION)
        keys.append(
            (day.isoformat(), (day + datetime.timedelta(days=1)).isoformat(), day.weekday(), window, day_in_window)
        )
    assert keys[N_VALIDATION - 1][1] == setting.dates[7 + N_VALIDATION - 1] and keys[-1][1] == setting.dates[-1]
    digest = f"stored zoo forecasts, {setting.name}"
    for name, forecast in setting.members.items():
        offset = len(keys) - len(forecast)  # a series of the test year alone starts on its first day
        path = save_dir / f"{name.lower()}.parquet"
        store = ParquetResultStore(path)
        for i in range(len(forecast)):
            run, target, weekday, window, day_in_window = keys[offset + i]
            for hour in range(24):
                record = {
                    "run_date": run,
                    "target_date": target,
                    "run_weekday": weekday,
                    "hour": hour,
                    "horizon": 1,
                    "day_in_test": day_in_window,
                    "prediction": float(forecast[i, hour]),
                    "actual": float(setting.y[offset + i, hour]),
                    "freq": "1h",
                    "window": window,
                }
                store.save(with_status(record, refit_anchor=run))
        store.close()
        Manifest(
            model_class="StoredForecasts",
            predictors=[],
            predictor_layout_hash="stored",
            target="price",
            horizon=1,
            freq="1h",
            training_window="stored",
            model_params={},
            data_columns_hash=digest,
            data_timezone="UTC",
            information_set=None,
            test_start=VALIDATION[0],
            test_end=TEST[1],
            package_version="stored",
            git_sha=None,
            created_at="2026-11-08T00:00:00+00:00",
            updated_at="2026-11-08T00:00:00+00:00",
            windows=RunWindows.fixed(VALIDATION, TEST).to_dict(),
            actuals={"through": setting.dates[-1], "hash": digest},
            column_hashes={"price": digest},
            frame_hash=digest,
            training_cut=dict(TRAINING_CUT_BY_DATE),
            record_schema=2,
            save_format="parquet",
        ).write(manifest_path(path))


def pipeline(setting: Setting) -> ModelPipeline:
    out = ModelPipeline()
    for name in setting.members:
        out.add_model(OLSModel(predictors=["stored"], name=name))  # a stand-in: report() reads the results by name
    for ensemble in setting.ensembles():
        out.add_ensemble(ensemble)
    out.add_evaluator(MAEEvaluator())
    out.add_evaluator(rMAEEvaluator(base_model="naive_weekly"))
    out.add_evaluator(DMEvaluator(benchmark="LEAR_CWA", variance="plain", interval=None))
    return out


def test_fixture_holds_both_settings_on_full_years():
    assert sum(path.stat().st_size for path in FIXTURE.iterdir()) < 4_000_000
    for name in SETTINGS:
        setting = Setting(name)
        assert setting.y.shape == (N_VALIDATION + N_TEST, 24) and np.isfinite(setting.y).all()
        assert (setting.dates[7], setting.dates[7 + N_VALIDATION], setting.dates[-1]) == (
            "2024-10-02",
            "2025-10-01",
            "2026-09-30",
        )
        for member, forecast in setting.members.items():
            assert forecast.shape in ((N_VALIDATION + N_TEST, 24), (N_TEST, 24)), member
            assert np.isfinite(forecast).all(), member
        greedy = setting.rows[setting.rows.kind.str.startswith("greedy")]
        assert sorted(greedy.index) == ["ENS_greedy_all", "ENS_greedy_clean", "ENS_greedy_strict"]


@pytest.mark.parametrize("name", SETTINGS)
def test_the_report_reproduces_the_leaderboard(name, tmp_path):
    setting = Setting(name)
    write_members(setting, tmp_path)
    test_only = sorted(m for m, f in setting.members.items() if len(f) == N_TEST)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = pipeline(setting).report(tmp_path, validation=VALIDATION, test=TEST)
    # Nothing to warn about but a member with no validation year (Mitra-v2 in AT).
    assert [str(w.message) for w in caught] == [
        f"Report: model(s) with missing forecasts: {m} in the validation window (0 of {N_VALIDATION * 24}, "
        f"{N_VALIDATION * 24} missing). Metrics are then computed on different forecast sets for these models; "
        "compare with care."
        for m in test_only
    ]

    summary = report.summary().set_index("model")
    tests = {
        key: report.tests(benchmark=benchmark, variance="plain", interval=None).set_index("model")
        for key, benchmark in BENCHMARKS.items()
    }
    check_counts(report, summary, test_only)
    check_choices(setting, report)
    check_test_year(setting, summary, tests)
    check_lear_cwa_bit_for_bit(setting, report)
    check_validation_year(setting, report)


def check_counts(report, summary: pd.DataFrame, test_only: list) -> None:
    """8,760 test forecasts for every model and ensemble, on identical keys; validation as the library counts it."""
    counts = report.forecast_counts().set_index("model")
    assert (counts["planned"] == N_TEST * 24).all() and (counts["produced"] == N_TEST * 24).all()
    assert (summary["forecasts"] == N_TEST * 24).all() and (summary["DM n_forecasts"] == N_TEST * 24).all()
    # The forecast for 2025-09-30, the first test issue day, is on file and left out of the validation report.
    validation = report.validation.forecast_counts().set_index("model").drop(index=test_only)
    assert (validation["produced"] == (N_VALIDATION - 1) * 24).all()
    assert (validation["target_in_test"] == 24).all()


def check_choices(setting: Setting, report) -> None:
    """Each top-k ensemble ranks the 12 best, searches k from 1 to 12 and keeps merge.py's members; the picks agree."""
    choices = report.choices()
    table = choices.set_index("ensemble")
    assert (table["validation_forecasts"] == (N_VALIDATION - 1) * 24).all()
    for ensemble in SETS:
        member_mae = choices.attrs["member_mae"][ensemble][0]
        k_mae = choices.attrs["k_mae"][ensemble][0]
        assert sorted(member_mae) == sorted(setting.best12(ensemble)), ensemble
        assert sorted(int(k) for k in k_mae) == list(range(1, TOP + 1)), ensemble
        chosen = list(table.loc[ensemble, "chosen"])
        want = set(setting.merge_py_members(ensemble))
        if (setting.name, ensemble) in CHOSEN_ON_363_DAYS:
            want = CHOSEN_ON_363_DAYS[(setting.name, ensemble)]
            assert set(setting.merge_py_members(ensemble)) - want == {"toto2b1"}
        assert set(chosen) == want and table.loc[ensemble, "k"] == len(want), (ensemble, chosen)
        assert chosen == [m for m in setting.best12(ensemble) if m in want]  # in member order
    for family in VARIANT_PICKS:
        assert list(table.loc[family, "chosen"]) == [setting.member(family)], family


def check_test_year(setting: Setting, summary: pd.DataFrame, tests: dict) -> None:
    """Test MAE, rMAE, both ratios and both DM p values of every row, within 1e-9."""
    rows = setting.rows[setting.rows.kind.isin(["member", "ensemble", "reference", "follow-up"])]
    naive = float(rows.loc["naive_weekly", "test"])
    bad = {}
    for model, row in rows.iterrows():
        name = setting.model_of[model]
        found = {
            "test": summary.loc[name, "MAE"],
            "rmae": summary.loc[name, "rMAE"],
            "vs_lear": summary.loc[name, "DM ratio"],
            "vs_tabpfn": tests["tabpfn"]["ratio"].get(name, 1.0 if name == "tabpfn2" else np.nan),
            "p_lear": stats.norm.sf(tests["lear"]["statistic"].get(name, np.nan)),
            "p_tabpfn": stats.norm.sf(tests["tabpfn"]["statistic"].get(name, np.nan)),
        }
        want = {key: float(row[key]) if row[key] != "" else np.nan for key in found if key != "rmae"}
        want["rmae"] = float(row.test) / naive
        for key, value in want.items():
            if np.isnan(value) != np.isnan(found[key]) or abs(found[key] - value) > TOL:
                bad[(model, key)] = (found[key], value)
    assert not bad, bad
    assert summary.loc["LEAR_CWA", "DM p"] != summary.loc["LEAR_CWA", "DM p"]  # NaN: the benchmark itself
    assert len(rows) == 27  # 23 of the leaderboard and 4 follow-ups


def check_lear_cwa_bit_for_bit(setting: Setting, report) -> None:
    """The library's LEAR_CWA is merge.py's `np.mean` over the windows in its order, to the bit."""
    records = report.refs["LEAR_CWA"].iter_records(cols=["target_date", "hour", "prediction"])
    stored = sorted((r["target_date"], r["hour"], r["prediction"]) for r in records)
    mean = np.mean([setting.members[f"lear_{w}"] for w in LEAR_WINDOWS], axis=0)[N_VALIDATION:]
    assert np.array_equal(np.array([p for _, _, p in stored]).reshape(N_TEST, 24), mean)


def check_validation_year(setting: Setting, report) -> None:
    """merge.py's validation MAE, 364 days and 8,736 forecasts: a report of the validation year as a test window. An ensemble's
    is its average of what it chose, in member order, which is what it averages on test, bit for bit."""
    val = slice(0, N_VALIDATION)
    forecasts = {name: f[val] for name, f in setting.members.items() if len(f) == N_VALIDATION + N_TEST}
    forecasts["LEAR_CWA"] = np.mean([forecasts[f"lear_{w}"] for w in LEAR_WINDOWS], axis=0)
    chosen = report.choices().set_index("ensemble")["chosen"]
    for ensemble in SETS:
        forecasts[ensemble] = np.mean([forecasts[m] for m in setting.merge_py_members(ensemble)], axis=0)
        forecasts[f"{ensemble} as chosen"] = np.mean([forecasts[m] for m in chosen[ensemble]], axis=0)
    for family in VARIANT_PICKS:
        forecasts[family] = forecasts[setting.member(family)]
    first = datetime.date.fromisoformat(VALIDATION[0])
    days = [
        ((first + datetime.timedelta(days=i)).isoformat(), (first + datetime.timedelta(days=i + 1)).isoformat())
        for i in range(N_VALIDATION)
    ]
    records = {
        name: [
            {
                "run_date": run,
                "target_date": target,
                "hour": hour,
                "horizon": 1,
                "prediction": float(forecast[i, hour]),
                "actual": float(setting.y[i, hour]),
            }
            for i, (run, target) in enumerate(days)
            for hour in range(24)
        ]
        for name, forecast in forecasts.items()
    }
    summary = EvaluationReport(records, [MAEEvaluator()]).summary().set_index("model")
    assert (summary["forecasts"] == N_VALIDATION * 24).all()
    bad = {}
    for model, row in setting.rows.iterrows():
        if row.val == "" or row.kind.startswith("greedy"):
            continue
        found = summary.loc[model if model in forecasts else setting.member(model), "MAE"]
        if abs(found - float(row.val)) > TOL:
            bad[model] = (found, float(row.val))
    assert not bad, bad
    # On the 363 days it chooses on, the library's validation MAE of what it chose is that average's.
    library = report.choices().set_index("ensemble")["validation_mae"]
    for ensemble in SETS:
        mine = np.abs(setting.y[: N_VALIDATION - 1] - forecasts[f"{ensemble} as chosen"][: N_VALIDATION - 1]).mean()
        assert abs(library[ensemble] - mine) <= TOL, ensemble
