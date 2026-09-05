"""Tests for preprocessing scripts — clean_comments, normalize, strip_docstrings, split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# pyrefly: ignore [missing-import]
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# pyrefly: ignore [missing-import]
from scripts.preprocessing.clean_comments import remove_comments
# pyrefly: ignore [missing-import]
from scripts.preprocessing.normalize import normalize
# pyrefly: ignore [missing-import]
from scripts.preprocessing.strip_docstrings import strip_docstrings
# pyrefly: ignore [missing-import]
from scripts.preprocessing.split import split_by_project


class TestRemoveComments:
    """Tests for the remove_comments function."""

    def test_removes_inline_comment(self):
        code = "x = 1  # this is a comment\ny = 2"
        result = remove_comments(code)
        assert "# this is a comment" not in result
        assert "x = 1" in result
        assert "y = 2" in result

    def test_removes_full_line_comment(self):
        code = "# full line comment\nx = 1"
        result = remove_comments(code)
        assert "# full line" not in result
        assert "x = 1" in result

    def test_preserves_string_with_hash(self):
        code = 'x = "hello # world"'
        result = remove_comments(code)
        assert "hello # world" in result

    def test_empty_code(self):
        assert remove_comments("") == ""
        assert remove_comments("   ") == ""

    def test_code_without_comments(self):
        code = "x = 1\ny = 2"
        result = remove_comments(code)
        assert "x = 1" in result
        assert "y = 2" in result


class TestNormalize:
    """Tests for the normalize function."""

    def test_replaces_tabs(self):
        assert normalize("\tx = 1") == "    x = 1"

    def test_normalizes_line_endings(self):
        result = normalize("x = 1\r\ny = 2\rz = 3")
        assert "\r" not in result
        assert "x = 1\ny = 2\nz = 3" == result

    def test_strips_trailing_newlines(self):
        result = normalize("\n\nx = 1\n\n")
        assert result == "x = 1"


class TestStripDocstrings:
    """Tests for the strip_docstrings function."""

    def test_strips_function_docstring(self):
        code = 'def foo():\n    """This is a docstring."""\n    return 1'
        result = strip_docstrings(code)
        assert "docstring" not in result
        assert "return" in result

    def test_preserves_regular_strings(self):
        code = 'def foo():\n    x = "not a docstring"\n    return x'
        result = strip_docstrings(code)
        assert "not a docstring" in result

    def test_no_docstring(self):
        code = "def foo():\n    return 1"
        result = strip_docstrings(code)
        assert "return" in result


class TestSplit:
    """Tests for the split functions."""

    def _make_samples(self, n: int = 100) -> list[dict]:
        return [
            {"sample_id": f"s{i}", "label": i % 2, "repository": f"repo_{i % 5}"}
            for i in range(n)
        ]

    def test_split_by_project_preserves_total(self):
        samples = self._make_samples(100)
        splits = split_by_project(samples, 0.8, 0.1, seed=42)
        total = sum(len(v) for v in splits.values())
        assert total == 100

    def test_split_by_project_has_all_keys(self):
        samples = self._make_samples()
        splits = split_by_project(samples, 0.8, 0.1, seed=42)
        assert set(splits.keys()) == {"train", "validation", "test"}

    def test_cross_project_split_no_leakage(self):
        samples = self._make_samples(100)
        splits = split_by_project(samples, 0.8, 0.1, seed=42)
        train_repos = {s["repository"] for s in splits["train"]}
        test_repos = {s["repository"] for s in splits["test"]}
        val_repos = {s["repository"] for s in splits["validation"]}
        # No repo overlap between train and test
        assert train_repos.isdisjoint(test_repos), "Data leakage: repo in both train and test!"
        assert train_repos.isdisjoint(val_repos), "Data leakage: repo in both train and validation!"

    def test_reproducibility(self):
        samples = self._make_samples()
        split1 = split_by_project(samples, 0.8, 0.1, seed=42)
        split2 = split_by_project(samples, 0.8, 0.1, seed=42)
        assert [s["sample_id"] for s in split1["train"]] == [s["sample_id"] for s in split2["train"]]
