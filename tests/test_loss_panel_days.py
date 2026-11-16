"""`LossPanel.day_ids` and `LossPanel.run_days`: the day (run date, horizon) and the run date of each key of a selection, which
the monitoring of a live directory groups by."""

import datetime

import numpy as np
import pandas as pd

from epftoolbox2.evaluators._compare import LossPanel

EPOCH = datetime.date(1970, 1, 1)


def frame(run_dates, horizons, hours, bias):
    rows = []
    for run in run_dates:
        for horizon in horizons:
            for hour in hours:
                target = pd.Timestamp(run) + pd.Timedelta(days=horizon)
                rows.append(
                    {
                        "run_date": pd.Timestamp(run),
                        "target_date": target,
                        "hour": hour,
                        "horizon": horizon,
                        "prediction": 10.0 + bias,
                        "actual": 10.0,
                    }
                )
    return pd.DataFrame(rows)


def test_a_day_is_one_run_date_and_horizon_and_a_run_date_is_its_epoch_day():
    runs = ["2026-10-21", "2026-10-22", "2026-10-24"]  # a hole on the 23rd
    panel = LossPanel({"a": frame(runs, (1, 2), range(3), 1.0), "b": frame(runs[:2], (1,), range(3), 2.0)})
    rows = panel.rows()
    days = panel.day_ids(rows)
    run_days = panel.run_days(rows)
    assert len(rows) == len(runs) * 2 * 3
    assert len(set(days.tolist())) == len(runs) * 2  # one id per (run date, horizon)
    # every id has one run date and one horizon, and three keys
    for day in set(days.tolist()):
        mine = rows[days == day]
        assert len(mine) == 3
        assert len(set(run_days[days == day].tolist())) == 1
        assert len(set(panel.horizon[mine].tolist())) == 1
    expected = [(datetime.date.fromisoformat(r) - EPOCH).days for r in runs]
    assert sorted(set(run_days.tolist())) == expected
    # ids are in calendar order: a later run date, or the same one and a larger horizon, has a larger id
    order = sorted({(int(a), int(h), int(d)) for a, h, d in zip(run_days, panel.horizon[rows], days)})
    assert [d for _, _, d in order] == sorted(d for _, _, d in order)
    # a selection: the second run date's keys only
    chosen = rows[run_days == expected[1]]
    assert np.all(panel.run_days(chosen) == expected[1]) and len(set(panel.day_ids(chosen).tolist())) == 2
