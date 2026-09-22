"""Tests for model components — Graph encoder, Fusion, Heads, Losses, Metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph.encoder import GraphEncoder, NodeTypeEmbedding, GATLayer, EDGE_TYPE_MAP
from src.fusion.cross_attention import CrossModalFusion, CrossAttentionBlock
from src.multitask.heads import BinaryHead
from src.utils.losses import BinaryClassificationLoss, FocalLoss
from src.utils.metrics import binary_metrics, find_best_threshold


class TestNodeTypeEmbedding:
    """Tests for the node type embedding layer."""

    def test_known_types(self):
        emb = NodeTypeEmbedding(num_types=64, embedding_dim=32)
        types = ["FunctionDef", "Assign", "Name"]
        result = emb(types)
        assert result.shape == (3, 32)

    def test_unknown_type_doesnt_crash(self):
        emb = NodeTypeEmbedding(num_types=64, embedding_dim=32)
        result = emb(["UnknownNodeType"])
        assert result.shape == (1, 32)


class TestGATLayer:
    """Tests for a single GAT layer."""

    def test_forward_shape(self):
        layer = GATLayer(in_dim=64, out_dim=8, num_heads=8, num_edge_types=5)
        x = torch.randn(10, 64)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_type = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        out = layer(x, edge_index, edge_type)
        assert out.shape == (10, 64)  # Residual keeps same dim

    def test_no_edges(self):
        layer = GATLayer(in_dim=64, out_dim=8, num_heads=8)
        x = torch.randn(5, 64)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
        out = layer(x, edge_index, edge_type)
        assert out.shape == (5, 64)


class TestGraphEncoder:
    """Tests for the full graph encoder."""

    def test_forward_shape(self):
        encoder = GraphEncoder(node_feature_dim=32, hidden_dim=64, output_dim=64, num_layers=2, num_heads=4)
        node_types = ["FunctionDef", "Assign", "Name", "Call", "Return"]
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_type = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)

        out = encoder(node_types, edge_index, edge_type, batch)
        assert out.shape == (1, 64)  # 1 graph, output_dim=64

    def test_with_node_embeddings(self):
        encoder = GraphEncoder(node_feature_dim=32, hidden_dim=64, output_dim=64, num_layers=2, num_heads=4)
        node_types = ["FunctionDef", "Assign", "Name"]
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        edge_type = torch.tensor([0, 3], dtype=torch.long)

        graph_out, node_out = encoder(node_types, edge_index, edge_type, return_node_embeddings=True)
        assert node_out.shape == (3, 64)




class TestCrossModalFusion:
    """Tests for the cross-modal fusion module."""

    @pytest.mark.parametrize("combine", ["mean", "concat", "gated"])
    def test_forward_2d(self, combine: str):
        fusion = CrossModalFusion(hidden_dim=64, num_heads=4, num_layers=1, combine=combine)
        sem = torch.randn(4, 64)
        graph = torch.randn(4, 64)
        out = fusion(sem, graph)
        assert out.shape == (4, 64)

    def test_forward_3d(self):
        fusion = CrossModalFusion(hidden_dim=64, num_heads=4, num_layers=2, combine="gated")
        sem = torch.randn(2, 10, 64)
        graph = torch.randn(2, 5, 64)
        out = fusion(sem, graph)
        assert out.shape == (2, 10, 64)

    def test_residual_alpha(self):
        """Verify residual_alpha retains semantic signal."""
        fusion = CrossModalFusion(hidden_dim=64, num_heads=4, num_layers=1, combine="gated")
        sem = torch.randn(2, 64)
        graph = torch.randn(2, 64)
        out_no_residual = fusion(sem, graph, residual_alpha=0.0)
        out_with_residual = fusion(sem, graph, residual_alpha=0.5)
        assert not torch.allclose(out_no_residual, out_with_residual, atol=1e-3)

    def test_padded_graph_input(self):
        """Verify fusion handles padded graph inputs (N, D) → (B, max_nodes, D)."""
        fusion = CrossModalFusion(hidden_dim=64, num_heads=4, num_layers=1, combine="gated")
        sem = torch.randn(2, 10, 64)
        graph_nodes = torch.randn(6, 64)
        graph_batch = torch.tensor([0, 0, 0, 0, 1, 1])
        out = fusion(sem, graph_nodes, graph_batch=graph_batch)
        assert out.shape == (2, 10, 64)


class TestHeads:
    """Tests cho prediction heads."""

    def test_binary_head(self):
        head = BinaryHead(input_dim=64)
        x = torch.randn(8, 64)
        out = head(x)
        assert out.shape == (8, 1)


class TestFocalLoss:
    """Tests for the Focal Loss function."""

    def test_output_is_scalar(self):
        loss = FocalLoss()
        logits = torch.randn(10, 1)
        targets = torch.randint(0, 2, (10,))
        result = loss(logits, targets)
        assert result.dim() == 0

    def test_perfect_prediction_low_loss(self):
        loss = FocalLoss()
        logits = torch.tensor([10.0, -10.0, 10.0])
        targets = torch.tensor([1, 0, 1])
        result = loss(logits, targets)
        assert result.item() < 0.1

    def test_bad_prediction_high_loss(self):
        loss = FocalLoss()
        logits = torch.tensor([-10.0, 10.0, -10.0])
        targets = torch.tensor([1, 0, 1])
        result = loss(logits, targets)
        assert result.item() > 1.0


class TestBinaryClassificationLoss:
    """Tests for binary classification loss."""

    def test_computes_total(self):
        criterion = BinaryClassificationLoss()
        binary_logits = torch.randn(8, 1)
        binary_targets = torch.randint(0, 2, (8,))
        losses = criterion(binary_logits=binary_logits, binary_targets=binary_targets)
        assert "total" in losses
        assert "binary" in losses
        assert losses["total"].item() > 0


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_binary_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        result = binary_metrics(y_true, y_pred)
        assert result.f1 == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.accuracy == 1.0

    def test_binary_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        result = binary_metrics(y_true, y_pred)
        assert result.f1 == 0.0

    def test_find_best_threshold(self):
        """Verify find_best_threshold chọn threshold tối ưu."""
        # Tạo data với threshold tối ưu khoảng 0.3
        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 2, size=n)
        y_prob = np.where(y_true == 1, np.random.uniform(0.5, 0.9, n), np.random.uniform(0.1, 0.4, n))
        best_thr, metrics = find_best_threshold(y_true, y_prob)
        assert 0.05 <= best_thr <= 0.95
        assert metrics.f1 > 0.5  # Ít nhất 0.5 F1 với synthetic data
