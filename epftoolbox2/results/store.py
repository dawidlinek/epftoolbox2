from pathlib import Path
import datetime
import json
import queue
import threading
from typing import Set, Dict, List, Iterator, Optional
from threading import Lock


class ResultStore:
    """Resumable JSONL storage for one model's results, keyed by `(hour, horizon, run_date)`.

    `run_date` (not `day_in_test`) identifies the forecast day: `day_in_test` is an offset from
    `test_start`, so the same offset means a different calendar day if `test_start` changes between
    runs into the same directory. Keying on the absolute date means a `test_start` change is handled
    correctly on its own; a change to anything else that defines a forecast (predictors, target,
    training window, model params, the data itself, ...) is caught separately by the manifest
    (`epftoolbox2.results.manifest`), which `BaseModel.run` checks before resuming.
    """

    _SENTINEL = object()

    def __init__(self, path: str, test_start: Optional[str] = None, batch_size: int = 500):
        self.path = Path(path)
        self.test_start = test_start
        self._lock = Lock()
        self._completed: Set[tuple] = set()
        self._batch_size = batch_size

        self._write_queue: queue.Queue = queue.Queue()
        self._writer = threading.Thread(target=self._writer_loop, daemon=True, name="store-writer")
        self._writer.start()

        self._load_existing()

    def save(self, result: Dict) -> None:
        key = (result["hour"], result["horizon"], result["run_date"])
        with self._lock:
            self._completed.add(key)
        self._write_queue.put(result)

    def flush(self) -> None:
        self._write_queue.put(self._SENTINEL)
        self._writer.join()

    def _run_date(self, day_in_test: int) -> str:
        if self.test_start is None:
            raise ValueError("ResultStore needs test_start to resolve day_in_test to a calendar date")
        base = datetime.date.fromisoformat(self.test_start)
        return (base + datetime.timedelta(days=day_in_test)).isoformat()

    def is_done(self, hour: int, horizon: int, day_in_test: int) -> bool:
        return (hour, horizon, self._run_date(day_in_test)) in self._completed

    def get_missing(self, all_tasks: List[tuple]) -> List[tuple]:
        return [t for t in all_tasks if not self.is_done(t[0], t[1], t[2])]

    def load_all(self) -> List[Dict]:
        return list(self.iter_lines())

    def iter_lines(self, cols: Optional[List[str]] = None) -> Iterator[Dict]:
        if not self.path.exists():
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    yield {k: r[k] for k in cols if k in r} if cols else r

    def _load_existing(self) -> None:
        for r in self.iter_lines(cols=["hour", "horizon", "run_date"]):
            if "run_date" not in r:
                continue  # pre-existing line from an older schema; recomputed rather than trusted
            self._completed.add((r["hour"], r["horizon"], r["run_date"]))

    def _writer_loop(self) -> None:
        buf: List[Dict] = []

        while True:
            try:
                item = self._write_queue.get(timeout=0.1)
            except queue.Empty:
                if buf:
                    self._write_batch(buf)
                    buf = []
                continue

            if item is self._SENTINEL:
                while True:
                    try:
                        extra = self._write_queue.get_nowait()
                        if extra is not self._SENTINEL:
                            buf.append(extra)
                    except queue.Empty:
                        break
                if buf:
                    self._write_batch(buf)
                return

            buf.append(item)

            while len(buf) < self._batch_size:
                try:
                    nxt = self._write_queue.get_nowait()
                    if nxt is self._SENTINEL:
                        self._write_batch(buf)
                        return
                    buf.append(nxt)
                except queue.Empty:
                    break

            if len(buf) >= self._batch_size:
                self._write_batch(buf)
                buf = []

    def _write_batch(self, items: List[Dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", buffering=1 << 20) as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
