"""Known methodology issues, pinned with xfail(strict=True).

Each test asserts the *correct* behaviour and currently fails. When an issue is fixed its test starts
passing, strict mode turns that into a failure, and the marker must be removed (move the test next to
the code it covers). Details: .claude/rules/{models,data,evaluation}.md.
"""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.transformers import DailyProfileTransformer
from epftoolbox2.information_set import Availability, InformationSet


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="`_sNN` profile names don't say their slot length; the information set reads each as a quarter-hour",
)
def test_a_half_hour_profile_slot_is_judged_at_its_own_time():
    """DailyProfileTransformer names the slots of every length but one hour `_sNN`. With freq="30min" on hourly data,
    `load_actual_s30` is the load at 15:00 on D, realised at 16:30 with a 1 h rule, but the information set reads it
    as quarter-hour 30 (07:30-07:45, known 08:45) and accepts it at every period of an 11:45 issue: a look-ahead."""
    index = pd.date_range("2024-01-01", periods=24 * 30, freq="h", tz="Europe/Warsaw")
    raw = pd.DataFrame({"load_actual": np.arange(len(index), dtype=float)}, index=index)
    data = DailyProfileTransformer(columns="load_actual", freq="30min").transform(raw)
    day = data.loc["2024-01-10"]
    assert (day["load_actual_s30"] == raw.loc["2024-01-10 15:00", "load_actual"]).all()  # 15:00 on every row
    info = InformationSet("11:45", "Europe/Brussels", rules={"load_actual_s*": Availability.realised("1h")})
    expanded = {(period, 1): ["load_actual_s30"] for period in range(24)}

    found, _ = info.violations(expanded, "price", "1h", index, "2024-01-10", "2024-01-16")

    assert {v.period for v in found} == set(range(24))
