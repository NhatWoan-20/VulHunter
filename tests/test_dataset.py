"""Unit tests for VulHunterDataset — lazy zero-RAM indexing and collation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from src.utils.dataset import VulHunterDataset, collate_fn


@pytest.fixture
def dummy_dataset_files(tmp_path: Path):
    samples = [
        {
            "sample_id": f"s_{i}",
            "code": f"def func_{i}():\n    x = {i}\n    return x\n",
            "binary_label": i % 2,
            "severity": "HIGH" if i % 2 else "LOW",
            "cwe_ids": ["CWE-89"] if i % 2 else [],
            "input_ids_qwen": [100, 200, 300 + i],
            "attention_mask_qwen": [1, 1, 1],
            "token_line_ids_qwen": [0, 1, 2],
            "source_sink_labels": [-1, 1, 2],
            "line_labels": [0, 1, 0],
            "quality_tier": "gold",
        }
        for i in range(10)
    ]
    data_path = tmp_path / "test_data.jsonl"
    with open(data_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    graphs = [
        {
            "sample_id": f"s_{i}",
            "node_types": ["FunctionDef", "Assign", "Return"],
            "edge_index": [[0, 1], [1, 2]],
            "edge_type": [0, 1],
        }
        for i in range(10)
    ]
    graph_path = tmp_path / "test_graphs.jsonl"
    with open(graph_path, "w", encoding="utf-8") as f:
        for g in graphs:
            f.write(json.dumps(g) + "\n")

    return data_path, graph_path


def test_lazy_dataset_loading(dummy_dataset_files):
    data_path, graph_path = dummy_dataset_files
    ds = VulHunterDataset(data_path=data_path, graph_data_path=graph_path, max_length=512)

    # Check length
    assert len(ds) == 10
    assert len(ds.samples) == 10
    assert len(ds.graph_data) == 10

    # Check sample indexing
    item0 = ds[0]
    assert item0["sample_id"] == "s_0"
    assert item0["binary_label"] == 0
    assert isinstance(item0["input_ids"], torch.Tensor)
    assert item0["input_ids"].tolist() == [100, 200, 300]
    assert "node_types" in item0
    assert item0["node_types"] == ["FunctionDef", "Assign", "Return"]

    # Check slice indexing on samples
    slice_items = ds.samples[1:4]
    assert len(slice_items) == 3
    assert slice_items[0]["sample_id"] == "s_1"
    assert slice_items[2]["sample_id"] == "s_3"

    # Check iteration
    all_sids = [s["sample_id"] for s in ds.samples]
    assert all_sids == [f"s_{i}" for i in range(10)]


def test_dataloader_batching_with_lazy_dataset(dummy_dataset_files):
    data_path, graph_path = dummy_dataset_files
    ds = VulHunterDataset(data_path=data_path, graph_data_path=graph_path, max_length=512)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)

    batches = list(loader)
    assert len(batches) == 3  # 4 + 4 + 2
    b0 = batches[0]
    assert b0["input_ids"].shape[0] == 4
    assert b0["binary_labels"].shape[0] == 4
    assert "node_types" in b0
    assert "edge_index" in b0
