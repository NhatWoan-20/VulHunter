"""Shared test fixtures for VulHunter test suite."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# pyrefly: ignore [missing-import]
import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


SAMPLE_VULNERABLE_CODE = """\
def login(username):
    query = "SELECT * FROM users WHERE name='" + username + "'"
    db.execute(query)
"""

SAMPLE_SAFE_CODE = """\
def login(username):
    query = "SELECT * FROM users WHERE name = %s"
    db.execute(query, (username,))
"""

SAMPLE_RECORD = {
    "sample_id": "test001",
    "pair_id": "test001",
    "data_source": "cvefixes",
    "quality_tier": "gold",
    "cve_id": "CVE-2024-0001",
    "repository": "test/repo",
    # Canonical keys (post-prepare_master / build_samples):
    "file_path": "app.py",
    "function_name": "login",
    # Raw aliases kept for backward compat with collection tests:
    "file": "app.py",
    "function": "login",
    "full_function_name": "login",
    "signature": "login(username)",
    "code": SAMPLE_VULNERABLE_CODE.strip(),
    "safe_code": SAMPLE_SAFE_CODE.strip(),
    "binary_label": 1,
    "cwe_ids": ["CWE-89"],
    "line_labels": [0, 1, 1],
    "vulnerable_lines": [2, 3],
    "severity": "HIGH",
}


@pytest.fixture
def sample_code() -> str:
    return SAMPLE_VULNERABLE_CODE


@pytest.fixture
def sample_safe_code() -> str:
    return SAMPLE_SAFE_CODE


@pytest.fixture
def sample_record() -> dict:
    return SAMPLE_RECORD.copy()


@pytest.fixture
def tmp_jsonl(tmp_path: Path) -> Path:
    """Create a temporary JSONL file with sample data."""
    path = tmp_path / "test_data.jsonl"
    records = [
        {
            "sample_id": f"test{i:03d}",
            "code": SAMPLE_VULNERABLE_CODE.strip() if i % 2 == 0 else SAMPLE_SAFE_CODE.strip(),
            "safe_code": SAMPLE_SAFE_CODE.strip(),
            "label": i % 2,
            "cwe_ids": ["CWE-89"] if i % 2 == 0 else [],
            "line_labels": [0, 1, 1] if i % 2 == 0 else [0, 0, 0],
            "vulnerable_lines": [2, 3] if i % 2 == 0 else [],
            "repository": f"repo_{i % 3}",
        }
        for i in range(20)
    ]
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path
