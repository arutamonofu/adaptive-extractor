"""Tests for LLM history logging."""

import json
from pathlib import Path
from unittest.mock import Mock

from aee.infrastructure.llm.history_logger import save_history, save_optimization_history


class TestSaveHistory:
    """Tests for save_history function."""

    def test_save_history_success(self, tmp_path: Path) -> None:
        """Test saving LLM history to file."""
        lm = Mock(history=[{"prompt": "test", "outputs": ["result"]}])
        output_path = tmp_path / "history.json"

        count = save_history(lm, output_path)

        assert count == 1
        assert output_path.exists()
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        # _clean_entry converts to {messages, outputs, model} format
        assert data[0]["messages"] == []
        assert data[0]["outputs"] == ["result"]

    def test_save_history_empty(self, tmp_path: Path) -> None:
        """Test with empty history - file should not be created."""
        lm = Mock(history=[])
        output_path = tmp_path / "history.json"

        count = save_history(lm, output_path)

        assert count == 0
        assert not output_path.exists()

    def test_save_history_no_attribute(self, tmp_path: Path) -> None:
        """Test with LM that has no history attribute."""
        lm = Mock(spec=[])  # no .history attribute
        output_path = tmp_path / "history.json"

        count = save_history(lm, output_path)

        assert count == 0
        assert not output_path.exists()

    def test_save_history_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that parent directories are created."""
        lm = Mock(history=[{"prompt": "test"}])
        output_path = tmp_path / "nested" / "dir" / "history.json"

        count = save_history(lm, output_path)

        assert count == 1
        assert output_path.exists()
        assert output_path.parent.exists()

    def test_save_history_unicode(self, tmp_path: Path) -> None:
        """Test saving history with unicode characters."""
        lm = Mock(history=[{"prompt": "Привет мир", "outputs": ["你好世界"]}])
        output_path = tmp_path / "history.json"

        count = save_history(lm, output_path)

        assert count == 1
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
        # _clean_entry converts to {messages, outputs, model} format
        assert data[0]["messages"] == []
        assert data[0]["outputs"][0] == "你好世界"


class TestSaveOptimizationHistory:
    """Tests for save_optimization_history function."""

    def test_save_both_histories(self, tmp_path: Path) -> None:
        """Test saving both student and teacher histories."""
        student = Mock(history=[{"prompt": "s1"}, {"prompt": "s2"}])
        teacher = Mock(history=[{"prompt": "t1"}])

        counts = save_optimization_history(student, teacher, tmp_path)

        assert counts["student"] == 2
        assert counts["teacher"] == 1
        assert list(tmp_path.glob("student_lm_*.json"))
        assert list(tmp_path.glob("teacher_lm_*.json"))

    def test_save_student_only(self, tmp_path: Path) -> None:
        """Test saving only student history (no teacher)."""
        student = Mock(history=[{"prompt": "s1"}])

        counts = save_optimization_history(student, None, tmp_path)

        assert counts["student"] == 1
        assert "teacher" not in counts
        student_files = list(tmp_path.glob("student_lm_*.json"))
        teacher_files = list(tmp_path.glob("teacher_lm_*.json"))
        assert len(student_files) == 1
        assert len(teacher_files) == 0

    def test_custom_timestamp(self, tmp_path: Path) -> None:
        """Test using custom timestamp in filename."""
        student = Mock(history=[{"prompt": "s1"}])
        timestamp = "20240101_120000"

        save_optimization_history(student, None, tmp_path, timestamp=timestamp)

        expected_file = tmp_path / f"student_lm_{timestamp}.json"
        assert expected_file.exists()

    def test_empty_histories_not_saved(self, tmp_path: Path) -> None:
        """Test that empty histories don't create files."""
        student = Mock(history=[])
        teacher = Mock(history=[])

        counts = save_optimization_history(student, teacher, tmp_path)

        assert counts["student"] == 0
        assert counts["teacher"] == 0
        assert not list(tmp_path.glob("*.json"))

    def test_returns_correct_counts(self, tmp_path: Path) -> None:
        """Test that returned counts match saved entries."""
        student = Mock(history=[{"p": i} for i in range(5)])
        teacher = Mock(history=[{"p": i} for i in range(3)])

        counts = save_optimization_history(student, teacher, tmp_path)

        assert counts == {"student": 5, "teacher": 3}
