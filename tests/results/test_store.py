import pytest
import tempfile
from pathlib import Path
from epftoolbox2.results import ResultStore


class TestResultStore:
    @pytest.fixture
    def temp_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    def test_save_and_load(self, temp_file):
        store = ResultStore(temp_file)

        result = {
            "target_date": "2024-01-15",
            "hour": 12,
            "horizon": 1,
            "prediction": 45.5,
            "actual": 46.0,
        }
        store.save(result)

        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0]["prediction"] == 45.5

    def test_is_done(self, temp_file):
        store = ResultStore(temp_file)

        store.save({"target_date": "2024-01-15", "hour": 12, "horizon": 1})

        assert store.is_done("2024-01-15", 12, 1)
        assert not store.is_done("2024-01-15", 12, 2)
        assert not store.is_done("2024-01-16", 12, 1)

    def test_get_missing(self, temp_file):
        store = ResultStore(temp_file)

        store.save({"target_date": "2024-01-15", "hour": 0, "horizon": 1})
        store.save({"target_date": "2024-01-15", "hour": 1, "horizon": 1})

        all_tasks = [
            ("2024-01-15", 0, 1),
            ("2024-01-15", 1, 1),
            ("2024-01-15", 2, 1),
        ]

        missing = store.get_missing(all_tasks)
        assert len(missing) == 1
        assert missing[0] == ("2024-01-15", 2, 1)

    def test_resume_from_existing_file(self, temp_file):
        Path(temp_file).write_text('{"target_date": "2024-01-15", "hour": 5, "horizon": 2}\n')

        store = ResultStore(temp_file)

        assert store.is_done("2024-01-15", 5, 2)
        assert len(store.load_all()) == 1

    def test_empty_file(self, temp_file):
        store = ResultStore(temp_file)

        assert store.load_all() == []
        assert store.get_missing([("2024-01-15", 0, 1)]) == [("2024-01-15", 0, 1)]

    def test_append_mode(self, temp_file):
        store = ResultStore(temp_file)

        store.save({"target_date": "2024-01-15", "hour": 0, "horizon": 1})
        store.save({"target_date": "2024-01-15", "hour": 1, "horizon": 1})

        lines = Path(temp_file).read_text().strip().split("\n")
        assert len(lines) == 2
