"""Build the week-ahead fixtures from the local copy of the published study (not run in CI: the source is outside the repo).

    uv run python tests/fixtures/week_ahead/build.py [SOURCE_DIR]

`SOURCE_DIR` defaults to `~/Desktop/min-max-hour-load-forecasting/data/parent`. It writes two files.

`PL_direct.csv`: the stored forecasts of the study for Poland, issued on the 104 Fridays from 2024-01-05 to 2025-12-26 at horizons
3 to 9, with the columns the tests read (the TSO's forecast, the regional NWP Lasso, the expert OLS and the actual daily minimum
and maximum load). The values are copied as text, so the file holds the source's numbers to the digit.

`PL_noon.parquet` (zstd): the study's daily frame for Poland as this library reads it, from 2021-10-20 to 2024-11-05, what the
forecasts issued on 2024-10-25 read (1,092 days of training rows before the newest at horizon 9, and the target of D+9). The study
stamps each row at local midnight though it holds the values as of noon, its issue time, so the rows are stamped at noon wall
clock instead (decision 11 of the step 3 spec): at midnight `load_actual_h-1` would read as 23:00 on D-1 and pass as known. The
targets `actual_daily_{min,max}` are built from the study's `_d+3` to `_d+9` target columns, each shifted back to its own date
(they agree wherever two give a value); the `_d+k` target columns themselves and `day` are dropped, since the library builds
targets itself and refuses those names. The other columns are those `study.columns()` names, as `pd.read_csv` parses them,
which is how the study read them.

`noon_frame` is also what the reproduction outside CI (`epftoolbox2-research/library_check_step3/week_ahead/`) reads every
market through.
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
try:
    from . import study  # imported by the tests, as tests.fixtures.week_ahead.build
except ImportError:  # run as a script
    sys.path.insert(0, str(HERE))
    import study

SOURCE = Path.home() / "Desktop" / "min-max-hour-load-forecasting" / "data" / "parent"
PARENT_CONFIG = Path.home() / "Desktop" / "min-max-hour-load-forecasting" / "src" / "mmh" / "parent_config.py"
DIRECT_COLUMNS = [
    "run_date",
    "horizon",
    "target",
    "actual",
    "tso_prediction",
    "regional_nwp_prediction",
    "expert_prediction",
]
FIRST, LAST = "2021-10-20", "2024-11-05"  # the CI frame: the rows the issue of 2024-10-25 reads, and a little margin
NOON = pd.Timedelta(hours=12)


def build_direct(source: Path, market: str = "PL") -> Path:
    """`{market}_direct.csv` with `DIRECT_COLUMNS`, rows in the source's order."""
    frame = pd.read_csv(source / "models_ecmwf" / f"{market}_direct.csv", dtype=str, keep_default_na=False)
    out = HERE / f"{market}_direct.csv"
    frame[DIRECT_COLUMNS].to_csv(out, index=False, lineterminator="\n")
    return out


def daily_targets(frame: pd.DataFrame, stat: str) -> pd.Series:
    """`actual_daily_{stat}` of each row's own date, from the study's `_d+k` targets shifted back k days. Raises where two
    disagree."""
    dates = frame.index.tz_localize(None).normalize()
    found = pd.Series(float("nan"), index=dates)
    for k in study.HORIZONS:
        values = pd.Series(frame[f"actual_daily_{stat}_d+{k}"].to_numpy(), index=dates + pd.Timedelta(days=k))
        values = values.reindex(dates)
        both = found.notna() & values.notna()
        if not (found[both] == values[both]).all():
            raise ValueError(f"actual_daily_{stat}: the _d+{k} column disagrees with a shorter one on some date")
        found = found.fillna(values)
    return pd.Series(found.to_numpy(), index=frame.index)


def at_noon(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Rows stamped at local midnight, restamped at noon wall clock of the same local date, in the same zone. Noon exists once on
    every date of a European zone, so no stamp is skipped or repeated, and a switch day's row is at noon too (11 or 13 hours
    after its midnight)."""
    local = index.tz_localize(None)
    if not (local == local.normalize()).all():
        raise ValueError("the rows aren't all stamped at local midnight")
    return (local + NOON).tz_localize(index.tz)


def noon_frame(source: Path, market: str, zone: str) -> pd.DataFrame:
    """The study's whole daily frame of `market`, stamped at noon wall clock in `zone`, with the columns of the lists and the
    two targets."""
    raw = pd.read_csv(source / "features" / f"data_{market}.csv", index_col=0)
    raw.index = pd.to_datetime(raw.index, utc=True).tz_convert(zone)
    frame = raw[study.columns()].copy()
    for stat in study.STATS:
        frame[f"actual_daily_{stat}"] = daily_targets(raw, stat)
    frame.index = at_noon(raw.index)
    return frame


def check_lists() -> None:
    """The lists of `study.py` are the study's, name for name and in order."""
    if not PARENT_CONFIG.exists():
        return
    sys.path.insert(0, str(PARENT_CONFIG.parent))
    import parent_config as parent  # noqa: E402

    def ours(names):
        return [name.replace("{horizon}", "{hz}") for name in names]

    assert ours(study.REGIONAL_NWP) == list(parent.SCENARIOS["regional_nwp"]), "regional_nwp differs from the study's"
    for stat in study.STATS:
        theirs = [name.replace("{key}", stat) for name in parent.BENCHMARKS["expert"]["predictors"]]
        assert ours(study.expert(stat)) == theirs, f"expert({stat}) differs from the study's"
    assert parent.TRAINING_WINDOW == study.TRAINING_WINDOW and list(parent.HORIZONS) == list(study.HORIZONS)


def build_noon(source: Path, market: str = "PL", zone: str = "Europe/Warsaw") -> Path:
    frame = noon_frame(source, market, zone)
    dates = frame.index.tz_localize(None).normalize()
    frame = frame[(dates >= pd.Timestamp(FIRST)) & (dates <= pd.Timestamp(LAST))]
    out = HERE / f"{market}_noon.parquet"
    frame.to_parquet(out, compression="zstd")
    return out


if __name__ == "__main__":
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCE
    check_lists()
    for written in (build_direct(source), build_noon(source)):
        print(f"{written} ({written.stat().st_size} bytes)")
