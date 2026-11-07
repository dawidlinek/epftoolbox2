"""Behaviour the package keeps while it moves off calls that pandas 3.1 deprecates.

`float_precision="round_trip"` and the default sort of `pd.concat(..., axis=1)` over DatetimeIndex frames are the two with
numbers behind them: a CSV read must give back the bits that were written, and the joined sources must stay sorted.
"""

import warnings

import numpy as np
import pandas as pd

from epftoolbox2._frame_csv import load_frame_csv, save_frame_csv
from epftoolbox2.data.sources.base import DataSource
from epftoolbox2.pipelines import DataPipeline
from epftoolbox2.pipelines.data_pipeline import _read_frame, _write_frame


def _hard_floats(n=3000):
    """Doubles of random bit patterns (every exponent, subnormals too) next to the ones decimal text rounds worst."""
    bits = np.random.default_rng(0).integers(0, 2**63, size=n, dtype=np.int64).view(np.float64)
    bits = bits[np.isfinite(bits)]
    return np.concatenate([bits, [0.1, 0.30000000000000004, 1e-300, 5e-324, 1.7976931348623157e308, 2.0**53 + 2]])


def _frame(values):
    index = pd.date_range("2024-03-30", periods=len(values), freq="h", tz="UTC", name="datetime")
    return pd.DataFrame({"a": values, "b": values[::-1] * -1.0}, index=index)


def _bits(frame):
    return frame.to_numpy(dtype=np.float64).view(np.uint64)


def test_load_frame_csv_gives_back_the_bits_that_save_frame_csv_wrote(tmp_path):
    frame = _frame(_hard_floats())
    save_frame_csv(frame, tmp_path / "frame.csv")
    loaded = load_frame_csv(tmp_path / "frame.csv")
    assert np.array_equal(_bits(loaded), _bits(frame))


def test_the_single_file_cache_gives_back_the_bits_it_was_written_with(tmp_path):
    frame = _frame(_hard_floats())
    path = tmp_path / "data.csv"
    _write_frame(frame, path, path)
    meta = {"index_name": frame.index.name, "columns": list(frame.columns), "dtypes": [str(t) for t in frame.dtypes]}
    assert np.array_equal(_bits(_read_frame(path, meta)), _bits(frame))


class _Fixed(DataSource):
    def __init__(self, frame):
        self.frame = frame

    def fetch(self, start, end):
        return self.frame.copy()

    def _validate_config(self):
        return True

    def get_cache_config(self):
        return None


def test_sources_on_different_indexes_join_on_their_sorted_union_without_a_pandas_warning():
    hours = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    first = pd.DataFrame({"x": np.arange(6.0)}, index=hours[:6][::-1])  # newest first
    second = pd.DataFrame({"y": np.arange(5.0)}, index=hours[3:][np.random.default_rng(1).permutation(5)])
    pipeline = DataPipeline().add_source(_Fixed(first)).add_source(_Fixed(second))

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        joined = pipeline.run("2024-01-01", "2024-01-02")

    assert list(joined.columns) == ["x", "y"]  # the sources in order
    assert joined.index.equals(hours)
    assert joined["x"].dropna().to_dict() == first["x"].to_dict()
    assert joined["y"].dropna().to_dict() == second["y"].to_dict()
