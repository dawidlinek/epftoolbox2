import pytest
import tempfile
from pathlib import Path
from epftoolbox2.results import ResultStore

TEST_START = "2024-01-14"  # day_in_test=1 -> 2024-01-15, matching the fixtures' target_date below


class TestResultStore:
    @pytest.fixture
    def temp_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    def test_save_and_load(self, temp_file):
        store = ResultStore(temp_file, test_start=TEST_START)

        result = {
            "run_date": "2024-01-15",
            "target_date": "2024-01-15",
            "hour": 12,
            "horizon": 1,
            "day_in_test": 1,
            "prediction": 45.5,
            "actual": 46.0,
        }
        store.save(result)
        store.flush()

        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0]["prediction"] == 45.5

    def test_is_done(self, temp_file):
        store = ResultStore(temp_file, test_start=TEST_START)

        store.save({"run_date": "2024-01-15", "hour": 12, "horizon": 1, "day_in_test": 1})

        assert store.is_done(12, 1, 1)
        assert not store.is_done(12, 2, 1)
        assert not store.is_done(12, 1, 2)

    def test_get_missing(self, temp_file):
        store = ResultStore(temp_file, test_start=TEST_START)

        store.save({"run_date": "2024-01-15", "hour": 0, "horizon": 1, "day_in_test": 1})
        store.save({"run_date": "2024-01-15", "hour": 1, "horizon": 1, "day_in_test": 1})

        all_tasks = [
            (0, 1, 1),
            (1, 1, 1),
            (2, 1, 1),
        ]

        missing = store.get_missing(all_tasks)
        assert len(missing) == 1
        assert missing[0] == (2, 1, 1)

    def test_resume_from_existing_file(self, temp_file):
        Path(temp_file).write_text('{"run_date": "2024-01-17", "hour": 5, "horizon": 2, "day_in_test": 3}\n')

        store = ResultStore(temp_file, test_start=TEST_START)

        assert store.is_done(5, 2, 3)
        assert len(store.load_all()) == 1

    def test_resume_keys_on_the_calendar_date_not_the_offset(self, temp_file):
        """A day_in_test offset means a different date once test_start changes; resuming must not conflate them."""
        Path(temp_file).write_text('{"run_date": "2024-01-15", "hour": 5, "horizon": 2, "day_in_test": 1}\n')

        # Same day_in_test (1), but test_start moved a day later: day_in_test=1 is now 2024-01-16, a task the
        # file has no result for, so it must be treated as missing rather than already done.
        store = ResultStore(temp_file, test_start="2024-01-15")
        assert not store.is_done(5, 2, 1)

        # With the original test_start, the same file still resumes correctly.
        store = ResultStore(temp_file, test_start=TEST_START)
        assert store.is_done(5, 2, 1)

    def test_lines_without_run_date_are_not_trusted_for_resuming(self, temp_file):
        """A line from an older schema without run_date can still be loaded, but isn't treated as done."""
        Path(temp_file).write_text('{"hour": 5, "horizon": 2, "day_in_test": 3}\n')

        store = ResultStore(temp_file, test_start=TEST_START)

        assert not store.is_done(5, 2, 3)
        assert len(store.load_all()) == 1

    def test_empty_file(self, temp_file):
        store = ResultStore(temp_file, test_start=TEST_START)

        assert store.load_all() == []
        assert store.get_missing([(0, 1, 1)]) == [(0, 1, 1)]

    def test_append_mode(self, temp_file):
        store = ResultStore(temp_file, test_start=TEST_START)

        store.save({"run_date": "2024-01-15", "hour": 0, "horizon": 1, "day_in_test": 1})
        store.save({"run_date": "2024-01-15", "hour": 1, "horizon": 1, "day_in_test": 1})
        store.flush()

        lines = Path(temp_file).read_text().strip().split("\n")
        assert len(lines) == 2

    def test_is_done_without_test_start_raises(self, temp_file):
        store = ResultStore(temp_file)
        with pytest.raises(ValueError, match="test_start"):
            store.is_done(0, 1, 0)
