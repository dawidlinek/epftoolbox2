"""Forecasts of the original epftoolbox's LEAR (Lago, Marcjasz, De Schutter & Weron 2021) and the data they read, for
`tests/models/test_lear_lago.py`. Not run in CI: it needs the original code, which today's scikit-learn and numpy can't run.

Run once with a Python 3.8 venv holding scikit-learn 0.24.2 (what Lago et al. ran), numpy below 1.24, pandas 1.x, statsmodels,
scipy, matplotlib and pyarrow, and a clone of https://github.com/jeslago/epftoolbox at a commit on or after f234b08 (issue #8:
exogenous lags d-1 and d-7). The clone's `epftoolbox/models/_lear.py` is executed from its source with the clone on `sys.path`,
so the clone isn't installed and tensorflow isn't needed:

    VENV38/bin/python -I tests/fixtures/lago/build.py CLONE DATA_DIR tests/fixtures/lago --markets DE BE --days 0 --data --cv

DATA_DIR holds Zenodo record 4624805's `{market}.csv` (`EpftoolboxBenchmarkSource`'s cache). The original runs as its own
`evaluate_lear_in_test_dataset` does: the data up to the target day, that day's prices set to NaN, then
`LEAR(cw).recalibrate_and_forecast_next_day(df, cw, day)` for each calibration window cw in 56, 84, 1092 and 1456, on one BLAS
thread. A target day is given as its offset in Lago's test period (`--days`), or as `--sample N --seed S`: offsets 0 and 1 and
N - 2 drawn by `numpy.random.default_rng(S)`, the sample of the research's `orig_check.py`. Writes to OUT_DIR:

- `original_{M}.csv`: `issue_day` (the target day minus one, this library's convention), `target_day`, `window` (cw), `hour`,
  `forecast`, `alpha` (the penalty LassoLarsIC chose, on its normalised scale) and `sweeps` (Lasso's `n_iter_`), floats by repr;
- `xtest_{M}.csv`: `issue_day` and `x0` … `x246`, the original's test row before scaling, in its column order (by repr; the
  same for every window, which is checked);
- with `--data`, `data_{M}.parquet` (zstd): `price` and the two exogenous series under this library's names, in the file's
  column order, from the file's first hour through the last target day, on the file's naive wall-clock index, as pandas 1.5
  parsed the CSV for the original (level 19, about 1.1 MB for DE and BE);
- with `--cv`, the reference of `LEARModel(criterion="cv")`: the original with its `recalibrate` fitting old
  `LassoLarsCV(cv=KFold(5), normalize=True, max_iter=2500)` per hour on the same scaled rows (Invariant, the dummies as they
  are) instead of `LassoLarsIC` and `Lasso`, and forecasting with it. `cv_{M}.csv`: `issue_day`, `target_day`, `window`,
  `hour`, `forecast`, `alpha` (`alpha_`), `grid` (`len(cv_alphas_)`), `alpha_cap{CAP}` (`alpha_` with `max_n_alphas=CAP`, so
  that the grid is strided), by repr; `cv_coefs_{M}.parquet` (zstd): `issue_day`, `window`, `hour`, `column` and `coef`, the nonzero
  entries of `coef_` by their position in the original's column order (about 40 of 247 per fit);
- `SOURCES.json`: the clone's commit, the versions, and the sha256 of every file read and written.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

WINDOWS = (56, 84, 1092, 1456)
# Lago's Table 1: the first and last target day of each test period.
TEST = {
    "NP": ("2016-12-27", "2018-12-24"),
    "PJM": ("2016-12-27", "2018-12-24"),
    "FR": ("2015-01-04", "2016-12-31"),
    "BE": ("2015-01-04", "2016-12-31"),
    "DE": ("2016-01-04", "2017-12-31"),
}
# The two exogenous series in the order of the file's columns, under this library's names (`EpftoolboxBenchmarkSource`).
# FR's and BE's files have generation first.
EXOGENOUS = {
    "NP": ["load_forecast", "wind_forecast"],
    "PJM": ["load_forecast", "zonal_load_forecast"],
    "FR": ["generation_forecast", "load_forecast"],
    "BE": ["generation_forecast_fr", "load_forecast_fr"],
    "DE": ["load_forecast_amprion", "wind_solar_forecast"],
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_original(clone):
    """The clone's `_lear.py`, executed from its source as a module of its own."""
    sys.path.insert(0, str(clone))
    source = (Path(clone) / "epftoolbox" / "models" / "_lear.py").read_text()
    assert "for past_day in [1, 7]:" in source, "the clone predates f234b08 (exogenous lags d-1, d-2)"
    module = type(sys)("orig_lear")
    exec(compile(source, "orig_lear", "exec"), module.__dict__)
    return module


def read_market(data_dir, market):
    """The file as epftoolbox's `read_data` reads it: columns Price, Exogenous 1, Exogenous 2 in file order."""
    data = pd.read_csv(Path(data_dir) / f"{market}.csv", index_col=0)
    data.index = pd.to_datetime(data.index)
    data.columns = ["Price"] + [f"Exogenous {i}" for i in range(1, len(data.columns))]
    return data


def target_days(market, days, sample, seed):
    first, last = (pd.Timestamp(d) for d in TEST[market])
    n = (last - first).days + 1
    if sample:
        rng = np.random.default_rng(seed)
        offsets = sorted(set([0, 1] + [int(i) for i in rng.choice(n, size=max(0, sample - 2), replace=False)]))
    else:
        offsets = sorted(set(days))
    return [first + pd.Timedelta(days=offset) for offset in offsets], offsets


# `alpha_cap{CV_CAP}`: LassoLarsCV with max_n_alphas=CV_CAP, whose stride is 2 or more on every fixture fit.
CV_CAP = 50


def cv_recalibrate(self, Xtrain, Ytrain):
    """The original's `LEAR.recalibrate` with old `LassoLarsCV` in place of `LassoLarsIC` and `Lasso`: the same Invariant scaling
    of the target and of every input but the last seven (the dummies), then per hour `LassoLarsCV(cv=KFold(5), normalize=True,
    max_iter=2500)`, whose `predict` the original's `predict` calls. Also `alpha_` with `max_n_alphas=CV_CAP`."""
    from sklearn.linear_model import LassoLarsCV
    from sklearn.model_selection import KFold

    [Ytrain], self.scalerY = self.scaling([Ytrain], "Invariant")
    [Xtrain_no_dummies], self.scalerX = self.scaling([Xtrain[:, :-7]], "Invariant")
    Xtrain[:, :-7] = Xtrain_no_dummies
    self.models, self.capped = {}, {}
    for h in range(24):
        settings = dict(cv=KFold(5), normalize=True, max_iter=2500)
        self.models[h] = LassoLarsCV(**settings).fit(Xtrain, Ytrain[:, h])
        self.capped[h] = LassoLarsCV(max_n_alphas=CV_CAP, **settings).fit(Xtrain, Ytrain[:, h]).alpha_


def run_cv(orig, data, market, dates):
    """`cv_{M}.csv` rows and `cv_coefs_{M}.parquet` rows of the original with old LassoLarsCV, for each target day and window."""
    first = pd.Timestamp(TEST[market][0])
    train = data.loc[: first - pd.Timedelta(hours=1)]
    test = data.loc[first:]
    rows, coefs = [], []
    for cw in WINDOWS:
        model = orig.LEAR(calibration_window=cw)
        model.scaling = orig.scaling
        model.recalibrate = types.MethodType(cv_recalibrate, model)
        for date in dates:
            available = pd.concat([train, test.loc[: date + pd.Timedelta(hours=23)]], axis=0)
            available.loc[date : date + pd.Timedelta(hours=23), "Price"] = np.nan
            tic = time.perf_counter()
            forecast = model.recalibrate_and_forecast_next_day(df=available, next_day_date=date, calibration_window=cw)
            forecast = np.asarray(forecast, dtype=float).ravel()
            assert forecast.shape == (24,), forecast.shape
            issue = (date - pd.Timedelta(days=1)).date().isoformat()
            for hour in range(24):
                fitted = model.models[hour]
                assert fitted.coef_.shape == (247,), fitted.coef_.shape
                rows.append(
                    {
                        "issue_day": issue,
                        "target_day": date.date().isoformat(),
                        "window": cw,
                        "hour": hour,
                        "forecast": repr(float(forecast[hour])),
                        "alpha": repr(float(fitted.alpha_)),
                        "grid": int(len(fitted.cv_alphas_)),
                        f"alpha_cap{CV_CAP}": repr(float(model.capped[hour])),
                    }
                )
                for column in np.flatnonzero(fitted.coef_):
                    coefs.append((issue, cw, hour, int(column), float(fitted.coef_[column])))
            print(f"{market} cv cw {cw} {date.date()}: {time.perf_counter() - tic:.1f} s", flush=True)
    coefs = pd.DataFrame(coefs, columns=["issue_day", "window", "hour", "column", "coef"])
    return rows, coefs.astype({"window": "int16", "hour": "int8", "column": "int16"})


def run_market(orig, data, market, dates):
    """Forecast rows and test rows of the original for each target day and window."""
    first = pd.Timestamp(TEST[market][0])
    train = data.loc[: first - pd.Timedelta(hours=1)]
    test = data.loc[first:]
    rows, xtests, seconds = [], {}, {}
    for cw in WINDOWS:
        model = orig.LEAR(calibration_window=cw)
        build = model._build_and_split_XYs
        seen = {}

        def capturing(df_train, df_test=None, date_test=None, build=build, seen=seen):
            xtrain, ytrain, xtest = build(df_train=df_train, df_test=df_test, date_test=date_test)
            seen["xtest"] = xtest.copy()  # predict() scales its argument in place
            seen["n"] = xtrain.shape[0]
            return xtrain, ytrain, xtest

        model._build_and_split_XYs = capturing
        for date in dates:
            available = pd.concat([train, test.loc[: date + pd.Timedelta(hours=23)]], axis=0)
            available.loc[date : date + pd.Timedelta(hours=23), "Price"] = np.nan
            tic = time.perf_counter()
            forecast = model.recalibrate_and_forecast_next_day(df=available, next_day_date=date, calibration_window=cw)
            seconds[(cw, date)] = time.perf_counter() - tic
            forecast = np.asarray(forecast, dtype=float).ravel()
            assert forecast.shape == (24,) and seen["n"] == cw - 7, (forecast.shape, seen["n"])
            issue = (date - pd.Timedelta(days=1)).date().isoformat()
            x = seen["xtest"]
            assert x.shape[0] == 1, x.shape
            if issue in xtests:
                assert np.array_equal(xtests[issue], x[0]), f"{market} {issue}: the test row depends on the window"
            xtests[issue] = x[0].copy()
            for hour in range(24):
                fitted = model.models[hour]
                rows.append(
                    {
                        "issue_day": issue,
                        "target_day": date.date().isoformat(),
                        "window": cw,
                        "hour": hour,
                        "forecast": repr(float(forecast[hour])),
                        "alpha": repr(float(fitted.alpha)),
                        "sweeps": int(fitted.n_iter_),
                    }
                )
            print(f"{market} cw {cw} {date.date()}: {seconds[(cw, date)]:.1f} s", flush=True)
    return rows, xtests, seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("clone")
    parser.add_argument("data_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--markets", nargs="+", default=["DE", "BE"])
    parser.add_argument("--days", nargs="+", type=int, default=[0])
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", action="store_true", help="also write data_{M}.parquet")
    parser.add_argument(
        "--cv", action="store_true", help="also write cv_{M}.csv and cv_coefs_{M}.parquet (LassoLarsCV)"
    )
    args = parser.parse_args(argv)

    import sklearn
    import statsmodels

    orig = load_original(args.clone)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clone = Path(args.clone)
    commit = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    commit = commit.stdout.strip()
    sources = {
        "epftoolbox_commit": commit,
        "lear_py_sha256": sha256(clone / "epftoolbox" / "models" / "_lear.py"),
        "python": sys.version.split()[0],
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "inputs": {},
        "outputs": {},
        "seconds": {},
    }
    for market in args.markets:
        path = Path(args.data_dir) / f"{market}.csv"
        sources["inputs"][f"{market}.csv"] = sha256(path)
        data = read_market(args.data_dir, market)
        dates, offsets = target_days(market, args.days, args.sample, args.seed)
        sources.setdefault("test_offsets", {})[market] = offsets
        rows, xtests, seconds = run_market(orig, data, market, dates)
        written = [out / f"original_{market}.csv", out / f"xtest_{market}.csv"]
        pd.DataFrame(rows).to_csv(written[0], index=False, lineterminator="\n")
        xframe = pd.DataFrame(
            [[issue] + [repr(float(v)) for v in x] for issue, x in sorted(xtests.items())],
            columns=["issue_day"] + [f"x{i}" for i in range(len(next(iter(xtests.values()))))],
        )
        xframe.to_csv(written[1], index=False, lineterminator="\n")
        if args.data:
            last = max(dates) + pd.Timedelta(hours=23)
            frame = data.loc[:last].copy()
            frame.columns = ["price", *EXOGENOUS[market]]
            frame.index.name = "datetime"
            written.append(out / f"data_{market}.parquet")
            frame.to_parquet(written[-1], compression="zstd", compression_level=19)
        if args.cv:
            cv_rows, cv_coefs = run_cv(orig, data, market, dates)
            written += [out / f"cv_{market}.csv", out / f"cv_coefs_{market}.parquet"]
            pd.DataFrame(cv_rows).to_csv(written[-2], index=False, lineterminator="\n")
            cv_coefs.to_parquet(written[-1], index=False, compression="zstd", compression_level=19)
        for path in written:
            sources["outputs"][path.name] = sha256(path)
        sources["seconds"][market] = {f"{cw} {d.date()}": round(s, 2) for (cw, d), s in seconds.items()}
    (out / "SOURCES.json").write_text(json.dumps(sources, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
