from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict


@dataclass
class ModelResultRef:
    name: str
    path: Optional[Path]
    count: int
    test_start: str
    test_end: str
    horizon: int
    _results: Optional[List[Dict]] = field(default=None, repr=False)
