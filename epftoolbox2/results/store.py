from pathlib import Path
import json
from typing import Set, Dict, List
from threading import Lock


class ResultStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = Lock()
        self._completed: Set[tuple] = set()
        self._load_existing()

    def _load_existing(self):
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self._completed.add((r["hour"], r["horizon"], r["day_in_test"]))

    def is_done(self, hour: int, horizon: int, day_in_test: int) -> bool:
        return (hour, horizon, day_in_test) in self._completed

    def save(self, result: Dict):
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(result) + "\n")
            self._completed.add((result["hour"], result["horizon"], result["day_in_test"]))

    def get_missing(self, all_tasks: List[tuple]) -> List[tuple]:
        return [t for t in all_tasks if not self.is_done(t[0], t[1], t[2])]

    def load_all(self) -> List[Dict]:
        results = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    results.append(json.loads(line))
        return results
