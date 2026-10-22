"""Unit and Integration tests for VulHunter FastAPI Deployment API."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multitask.model import VulHunterModel
from scripts.api_deployment import (
    extract_pdg_safe,
    graph_to_tensors,
    tokenize_code,
    init_service,
    app,
)

SAMPLE_VULN_CODE = """\
def run_command(user_input):
    import os
    os.system("ping " + user_input)
"""

SAMPLE_SAFE_CODE = """\
def add(a, b):
    return a + b
"""

SAMPLE_SYNTAX_ERROR = """\
def broken_syntax(
    print("missing paren"
"""


@pytest.fixture
def mock_checkpoint_file(tmp_path):
    """Fixture tạo checkpoint .pt hợp lệ để kiểm thử production startup."""
    ckpt_path = tmp_path / "test_best.pt"
    model = VulHunterModel(mode="fusion")
    ckpt_data = {
        "epoch": 5,
        "val_f1": 0.7519,
        "best_threshold": 0.25,
        "config": {
            "mode": "fusion",
            "model": {
                "semantic": {"backbone": "microsoft/codebert-base", "output_dim": 256},
                "graph": {"output_dim": 256},
                "fusion": {"hidden_dim": 256},
                "heads": {"binary": {"hidden_dim": 128}},
            },
        },
        "model_state_dict": model.state_dict(),
    }
    torch.save(ckpt_data, ckpt_path)
    return str(ckpt_path)


def test_extract_pdg_safe_valid_code():
    graph = extract_pdg_safe(SAMPLE_VULN_CODE)
    assert "nodes" in graph
    assert "edges" in graph
    assert len(graph["nodes"]) > 0


def test_extract_pdg_safe_handles_syntax_error_gracefully():
    # Không crash khi code bị lỗi cú pháp; trả về minimal fallback graph
    graph = extract_pdg_safe(SAMPLE_SYNTAX_ERROR)
    assert "nodes" in graph
    assert len(graph["nodes"]) >= 1


def test_graph_to_tensors_shapes():
    graph = extract_pdg_safe(SAMPLE_SAFE_CODE)
    device = torch.device("cpu")
    tensors = graph_to_tensors(graph, device)

    assert "node_types" in tensors
    assert "edge_index" in tensors
    assert "edge_type" in tensors
    assert "batch" in tensors
    assert tensors["edge_index"].ndim == 2
    assert tensors["edge_index"].size(0) == 2
    assert tensors["batch"].size(0) == len(tensors["node_types"])


def test_init_service_fail_fast_when_missing_checkpoint(monkeypatch):
    """Production test: Bắt buộc dừng server ngay nếu thiếu checkpoint."""
    monkeypatch.delenv("VULHUNTER_CHECKPOINT", raising=False)
    with pytest.raises(RuntimeError, match="Missing checkpoint path"):
        init_service(checkpoint_override=None)


def test_init_service_fail_fast_when_file_not_found(tmp_path):
    """Production test: Báo lỗi nếu file checkpoint không tồn tại."""
    fake_path = str(tmp_path / "does_not_exist.pt")
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        init_service(checkpoint_override=fake_path)


def test_api_client_endpoints(mock_checkpoint_file, monkeypatch):
    """Kiểm thử toàn bộ endpoints khi có checkpoint .pt hợp lệ."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi or httpx not installed in test environment")

    monkeypatch.setenv("VULHUNTER_CHECKPOINT", mock_checkpoint_file)

    with TestClient(app) as client:
        # 1. Test /health
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["model_mode"] == "fusion"
        assert data["checkpoint_loaded"] is True
        assert data["optimal_threshold"] == 0.25

        # 2. Test /predict with empty code (should 400)
        res_empty = client.post("/predict", json={"code": "   "})
        assert res_empty.status_code == 400

        # 3. Test /predict with valid code
        res_pred = client.post("/predict", json={"code": SAMPLE_VULN_CODE})
        assert res_pred.status_code == 200
        pred_data = res_pred.json()
        assert "is_vulnerable" in pred_data
        assert "vulnerability_probability" in pred_data
        assert pred_data["decision_threshold"] == 0.25
        assert 0.0 <= pred_data["vulnerability_probability"] <= 1.0

        # 4. Test /predict with custom threshold override
        res_custom = client.post("/predict", json={"code": SAMPLE_VULN_CODE, "threshold": 0.85})
        assert res_custom.status_code == 200
        assert res_custom.json()["decision_threshold"] == 0.85

        # 5. Test /predict/batch
        res_batch = client.post("/predict/batch", json={"codes": [SAMPLE_VULN_CODE, SAMPLE_SAFE_CODE]})
        assert res_batch.status_code == 200
        batch_data = res_batch.json()
        assert batch_data["total_samples"] == 2
        assert len(batch_data["predictions"]) == 2
