"""The model's `{target}_d+{h}` columns shift by local calendar days (`BaseModel._shift_index_by_calendar_days`)."""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.models import OLSModel
from tests.synthetic import calendar_shift

TZ = "Europe/Warsaw"


@pytest.mark.parametrize("freq, periods_per_hour", [("1h", 1), ("15min", 4)])
@pytest.mark.parametrize("horizon", [1, 2, 7])
def test_targets_follow_calendar_days_and_skip_the_repeated_autumn_hour(freq, periods_per_hour, horizon):
    index = pd.date_range("2023-01-01", "2025-01-01", freq=freq, tz=TZ, inclusive="left")
    price = pd.Series(np.arange(len(index), dtype=float), index=index)

    target = price.reindex(OLSModel([])._shift_index_by_calendar_days(index, horizon)).to_numpy()
    reference = calendar_shift(price, horizon).to_numpy()  # first occurrence of a repeated hour

    differs = ~((target == reference) | (np.isnan(target) & np.isnan(reference)))
    # Only rows whose target is the repeated 02:00 hour of a fall-back day (2023-10-29, 2024-10-27) differ,
    # and there the model keeps NaN rather than picking one of the two prices.
    assert differs.sum() == 2 * periods_per_hour
    assert np.isnan(target[differs]).all()
    fall_back_days = {pd.Timestamp("2023-10-29").date(), pd.Timestamp("2024-10-27").date()}
    assert {(ts + pd.Timedelta(days=horizon)).date() for ts in index[differs]} == fall_back_days
    assert set(index[differs].hour) == {2}


def test_repeated_hour_rows_look_ahead_to_the_next_days_hour():
    index = pd.date_range("2024-10-26", "2024-10-30", freq="1h", tz=TZ, inclusive="left")
    shifted = OLSModel([])._shift_index_by_calendar_days(index, 1)
    first = pd.Timestamp("2024-10-27 02:00+02:00").tz_convert(TZ)
    second = pd.Timestamp("2024-10-27 02:00+01:00").tz_convert(TZ)
    assert shifted[index.get_loc(first)] == pd.Timestamp("2024-10-28 02:00", tz=TZ)
    assert shifted[index.get_loc(second)] == pd.Timestamp("2024-10-28 02:00", tz=TZ)
    assert shifted[index.get_loc(pd.Timestamp("2024-10-26 02:00", tz=TZ))] is pd.NaT


def test_utc_index_warns_that_days_are_utc_days():
    index = pd.date_range("2024-03-30", periods=72, freq="1h", tz="UTC")
    with pytest.warns(UserWarning, match="UTC"):
        OLSModel([])._shift_index_by_calendar_days(index, 1)
