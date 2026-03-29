from pathlib import Path
import json
import queue
import threading
from typing import Set, Dict, List, Iterator, Optional
from threading import Lock


class ResultStore:
    _SENTINEL = object()  # signals the writer thread to stop

    def __init__(self, path: str, batch_size: int = 500):
        self.path = Path(path)
        self._lock = Lock()                        # guards _completed only
        self._completed: Set[tuple] = set()
        self._batch_size = batch_size

        self._write_queue: queue.Queue = queue.Queue()
        self._writer = threading.Thread(
            target=self._writer_loop, daemon=True, name="store-writer"
        )
        self._writer.start()

        self._load_existing()

    def save(self, result: Dict) -> None:
        key = (result["hour"], result["horizon"], result["day_in_test"])
        with self._lock:
            self._completed.add(key)
        self._write_queue.put(result)   # non-blocking; no lock needed

    def flush(self) -> None:
        self._write_queue.put(self._SENTINEL)
        self._writer.join()

    def is_done(self, hour: int, horizon: int, day_in_test: int) -> bool:
        return (hour, horizon, day_in_test) in self._completed

    def get_missing(self, all_tasks: List[tuple]) -> List[tuple]:
        return [t for t in all_tasks if not self.is_done(t[0], t[1], t[2])]

    def load_all(self) -> List[Dict]:
        results = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    results.append(json.loads(line))
        return results

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
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self._completed.add((r["hour"], r["horizon"], r["day_in_test"]))

    def _writer_loop(self) -> None:
        """Background thread: drains the queue and writes batches to disk."""
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
