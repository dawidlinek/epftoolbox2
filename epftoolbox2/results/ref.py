from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict


@dataclass
class ModelResultRef:
    """Where one model's results live and how many forecasts the run planned and produced.

    `count` is the number of planned forecasts (periods x horizons x test days). `produced` is how many
    exist after the run (including ones resumed from an earlier run), and `skipped` counts the tasks that
    gave no forecast by reason (e.g. `missing_test_predictors`). `produced` is None for refs built from
    raw result lists.
    """

    name: str
    path: Optional[Path]
    count: int
    test_start: str
    test_end: str
    horizon: int
    freq: str = "1h"
    _results: Optional[List[Dict]] = field(default=None, repr=False)
    produced: Optional[int] = None
    skipped: Dict[str, int] = field(default_factory=dict)
