"""VulHunter Model — Binary vulnerability detection.

Main model class tích hợp:
    1. Semantic Encoder (LLM backbone, vd: Qwen2.5-Coder)
    2. Graph Encoder (GraphCodeBERT + GAT)
    3. Cross-Modal Fusion (Bidirectional Cross-Attention)
    4. Binary prediction head

Hỗ trợ 3 modes:
    - semantic_only: Chỉ dùng semantic encoder
    - graph_only: Chỉ dùng graph encoder
    - fusion: Kết hợp cả hai (đề xuất chính)

Ví dụ:
    >>> model = VulHunterModel(mode="fusion", config=model_config)
    >>> outputs = model(input_ids, attention_mask, node_types, edge_index, edge_type, batch)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.fusion.cross_attention import CrossModalFusion
from src.graph.encoder import GraphEncoder
from src.multitask.heads import BinaryHead
from src.semantic.encoder import SemanticEncoder

logger = logging.getLogger(__name__)


@dataclass
class ModelOutput:
    """Container cho VulHunterModel outputs.

    Attributes:
        binary_logits: Binary vulnerability logits shape ``(B, 1)``.
        fused_embedding: Pooled fused representation shape ``(B, output_dim)``.
    """
    binary_logits: torch.Tensor | None = None
    fused_embedding: torch.Tensor | None = None

    def __iter__(self):
        return iter((self.binary_logits, self.fused_embedding))

    def __getitem__(self, key: str):
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)


class VulHunterModel(nn.Module):
    """Hybrid multi-modal vulnerability detection model (binary classification).

    Combines semantic và graph understanding qua cross-modal fusion để
    dự đoán vulnerable (1) vs safe (0).

    Args:
        mode: Operating mode. Một trong "fusion", "semantic_only", "graph_only".
        semantic_config: Configuration dict cho SemanticEncoder.
        graph_config: Configuration dict cho GraphEncoder.
        fusion_config: Configuration dict cho CrossModalFusion.
        head_config: Configuration dict cho BinaryHead.
    """

    VALID_MODES = ("fusion", "semantic_only", "graph_only")

    def __init__(
        self,
        mode: str = "fusion",
        semantic_config: dict | None = None,
        graph_config: dict | None = None,
        fusion_config: dict | None = None,
        head_config: dict | None = None,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode phải là một trong {self.VALID_MODES}, got '{mode}'")
        self.mode = mode

        semantic_config = semantic_config or {}
        graph_config = graph_config or {}
        fusion_config = fusion_config or {}
        head_config = head_config or {}

        output_dim = semantic_config.get("output_dim", 256)

        # Khởi tạo encoders theo mode
        if mode in ("fusion", "semantic_only"):
            self.semantic_encoder = SemanticEncoder(**semantic_config)
            logger.info("Khởi tạo SemanticEncoder")

        if mode in ("fusion", "graph_only"):
            graph_config.setdefault("output_dim", output_dim)
            self.graph_encoder = GraphEncoder(**graph_config)
            logger.info("Khởi tạo GraphEncoder (unfreeze_top_n=%d)",
                        graph_config.get("unfreeze_top_n", 0))

        # Fusion module (chỉ cho fusion mode)
        if mode == "fusion":
            fusion_config.setdefault("hidden_dim", output_dim)
            # Extract residual_alpha before passing to constructor (not an init param)
            self._fusion_residual_alpha = float(fusion_config.pop("residual_alpha", 0.3))
            self.fusion = CrossModalFusion(**fusion_config)
            logger.info("Khởi tạo CrossModalFusion (residual_alpha=%.2f)", self._fusion_residual_alpha)

        # Binary prediction head (task chính)
        binary_cfg = head_config.get("binary", {})
        self.binary_head = BinaryHead(input_dim=output_dim, **binary_cfg)

        # Đảm bảo trainable params (LoRA adapters, projection, heads) đều ở float32
        # cho PyTorch AMP GradScaler compatibility (frozen backbone giữ ở FP16)
        for p in self.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()

        # Log parameter counts
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "VulHunterModel [%s] — Total params: %s, Trainable: %s (%.1f%%)",
            mode, f"{total:,}", f"{trainable:,}", 100.0 * trainable / total if total > 0 else 0,
        )

    def forward(
        self,
        # Semantic inputs
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        # Graph inputs
        node_types: list[str] | None = None,
        node_texts: list[str] | None = None,
        edge_index: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> ModelOutput:
        """Forward pass.

        Args:
            input_ids: Tokenized code, shape ``(B, L)``. Cần cho semantic/fusion.
            attention_mask: Attention mask, shape ``(B, L)``. Cần cho semantic/fusion.
            node_types: List node type strings. Cần cho graph/fusion.
            node_texts: List node text strings. Cần cho graph/fusion.
            edge_index: Edge indices, shape ``(2, E)``. Cần cho graph/fusion.
            edge_type: Edge type indices, shape ``(E,)``. Cần cho graph/fusion.
            batch: Graph batch vector, shape ``(N,)``. Cần cho graph/fusion.

        Returns:
            ModelOutput chứa binary_logits và fused_embedding.
        """
        output = ModelOutput()

        # ──── Encode ────
        if self.mode == "semantic_only":
            pooled = self.semantic_encoder(input_ids, attention_mask, return_sequence=False)
            fused_pooled = pooled

        elif self.mode == "graph_only":
            # Fix: DataParallel splits edge_index (2,E) along dim-0 → (1,E) per GPU.
            if edge_index is not None and edge_index.dim() == 1:
                edge_index = edge_index.new_zeros(2, 0)
            elif edge_index is not None and edge_index.dim() == 2 and edge_index.size(0) == 1:
                edge_index = edge_index.view(2, -1)
            graph_pooled = self.graph_encoder(
                node_types, edge_index, edge_type, batch, node_texts=node_texts,
            )
            fused_pooled = graph_pooled

        elif self.mode == "fusion":
            # Semantic branch
            sem_pooled, sem_seq = self.semantic_encoder(input_ids, attention_mask, return_sequence=True)
            # Graph branch
            # Fix: DataParallel splits edge_index (2,E) along dim-0 → (1,E) per GPU.
            # Reconstruct to (2, E//2) so the GAT layer sees the correct COO format.
            if edge_index is not None and edge_index.dim() == 1:
                # edge_index was squeezed to 1-D somehow — treat as no edges
                edge_index = edge_index.new_zeros(2, 0)
            elif edge_index is not None and edge_index.dim() == 2 and edge_index.size(0) == 1:
                # DataParallel gave us (1, E) — reshape to (2, E//2)
                edge_index = edge_index.view(2, -1)
            graph_pooled, node_emb = self.graph_encoder(
                node_types, edge_index, edge_type, batch,
                return_node_embeddings=True, node_texts=node_texts,
            )
            # Cross-attend code tokens to graph nodes
            # residual_alpha giữ semantic signal khi graph rỗng/noisy.
            fused_seq = self.fusion(
                sem_seq, node_emb, attention_mask, batch,
                residual_alpha=self._fusion_residual_alpha,
            )
            mask = attention_mask.unsqueeze(-1).to(fused_seq.dtype)
            fused_pooled = (fused_seq * mask).sum(1) / mask.sum(1).clamp(min=1)

        output.fused_embedding = fused_pooled

        # ──── Predict ────
        output.binary_logits = self.binary_head(fused_pooled)

        return output

    @classmethod
    def from_config(cls, config: dict) -> "VulHunterModel":
        """Tạo model từ nested configuration dict.

        Args:
            config: Dict với keys matching __init__ parameters.

        Returns:
            Initialized VulHunterModel.
        """
        model_cfg = config.get("model", config)
        return cls(
            mode=model_cfg.get("mode", "fusion"),
            semantic_config=model_cfg.get("semantic", {}),
            graph_config=model_cfg.get("graph", {}),
            fusion_config=model_cfg.get("fusion", {}),
            head_config=model_cfg.get("heads", {}),
        )
