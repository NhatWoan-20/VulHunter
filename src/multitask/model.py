"""VulHunter Model — Unified multi-modal, multi-task vulnerability detection model.

This is the main model class that integrates all components:
    1. Semantic Encoder (LLM backbone)
    2. Graph Encoder (GAT)
    3. Cross-Modal Fusion (Bidirectional Cross-Attention)
    4. Multi-Task Heads (Binary, CWE, Localization, Source/Sink)

It supports three operating modes:
    - semantic_only: Only uses the semantic encoder
    - graph_only: Only uses the graph encoder
    - fusion: Combines both (default, proposed approach)

Example:
    >>> model = VulHunterModel(mode="fusion", config=model_config)
    >>> outputs = model(input_ids, attention_mask, node_types, edge_index, edge_type, batch)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn

# pyrefly: ignore [missing-import]
from src.fusion.cross_attention import CrossModalFusion
# pyrefly: ignore [missing-import]
from src.graph.encoder import GraphEncoder
# pyrefly: ignore [missing-import]
from src.multitask.heads import BinaryHead, CWEHead, LocalizationHead, SeverityHead, SourceSinkHead
# pyrefly: ignore [missing-import]
from src.semantic.encoder import SemanticEncoder

logger = logging.getLogger(__name__)


@dataclass
class ModelOutput:
    """Container for multi-task model outputs.

    Attributes:
        binary_logits: Binary vulnerability logits of shape ``(B, 1)``.
        cwe_logits: CWE classification logits of shape ``(B, num_cwe_classes)``.
        localization_logits: Line-level vulnerability logits of shape ``(B, L, 1)``.
        source_sink_logits: Source/sink/normal logits of shape ``(B, L, 3)``.
        fused_embedding: Fused representation of shape ``(B, D)``.
    """
    binary_logits: torch.Tensor | None = None
    cwe_logits: torch.Tensor | None = None
    localization_logits: torch.Tensor | None = None
    source_sink_logits: torch.Tensor | None = None
    severity_logits: torch.Tensor | None = None
    fused_embedding: torch.Tensor | None = None

    def __iter__(self):
        return iter((
            self.binary_logits,
            self.cwe_logits,
            self.localization_logits,
            self.source_sink_logits,
            self.severity_logits,
            self.fused_embedding,
        ))

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)


class VulHunterModel(nn.Module):
    """Hybrid multi-modal vulnerability detection model.

    Combines semantic and graph understanding with multi-task prediction.

    Args:
        mode: Operating mode. One of "fusion", "semantic_only", "graph_only".
        semantic_config: Configuration dict for the SemanticEncoder.
        graph_config: Configuration dict for the GraphEncoder.
        fusion_config: Configuration dict for the CrossModalFusion.
        head_config: Configuration dict for prediction heads.
        num_cwe_classes: Number of CWE categories.
    """

    VALID_MODES = ("fusion", "semantic_only", "graph_only")

    def __init__(
        self,
        mode: str = "fusion",
        semantic_config: dict | None = None,
        graph_config: dict | None = None,
        fusion_config: dict | None = None,
        head_config: dict | None = None,
        num_cwe_classes: int = 10,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")
        self.mode = mode

        semantic_config = semantic_config or {}
        graph_config = graph_config or {}
        fusion_config = fusion_config or {}
        head_config = head_config or {}

        output_dim = semantic_config.get("output_dim", 256)

        # Initialize encoders based on mode
        if mode in ("fusion", "semantic_only"):
            self.semantic_encoder = SemanticEncoder(**semantic_config)
            logger.info("Initialized SemanticEncoder")

        if mode in ("fusion", "graph_only"):
            graph_config.setdefault("output_dim", output_dim)
            self.graph_encoder = GraphEncoder(**graph_config)
            logger.info("Initialized GraphEncoder")

        # Fusion module (only for fusion mode)
        if mode == "fusion":
            fusion_config.setdefault("hidden_dim", output_dim)
            self.fusion = CrossModalFusion(**fusion_config)
            logger.info("Initialized CrossModalFusion")

        # Multi-task prediction heads
        binary_cfg = head_config.get("binary", {})
        cwe_cfg = head_config.get("cwe", {})
        loc_cfg = head_config.get("localization", {})
        ss_cfg = head_config.get("source_sink", {})
        severity_cfg = head_config.get("severity", {})

        # Pop num_classes so they do not collide with the explicit kwargs below.
        cwe_num = cwe_cfg.pop("num_classes", None) or num_cwe_classes
        severity_num = severity_cfg.pop("num_classes", 4)
        ss_num = ss_cfg.pop("num_classes", 3)

        self.binary_head = BinaryHead(input_dim=output_dim, **binary_cfg)
        self.cwe_head = CWEHead(input_dim=output_dim, num_classes=cwe_num, **cwe_cfg)
        self.localization_head = LocalizationHead(input_dim=output_dim, **loc_cfg)
        self.source_sink_head = SourceSinkHead(input_dim=output_dim, num_classes=ss_num, **ss_cfg)
        self.severity_head = SeverityHead(input_dim=output_dim, num_classes=severity_num, **severity_cfg)

        # Ensure all trainable parameters (LoRA adapters, projection, heads) are in float32
        # for PyTorch AMP GradScaler compatibility while frozen backbone stays in FP16
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
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # Graph inputs
        node_types: Optional[list[str]] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        # Task control
        tasks: Optional[list[str]] = None,
    ) -> ModelOutput:
        """Forward pass through the full model.

        Args:
            input_ids: Tokenized code, shape ``(B, L)``. Required for semantic/fusion modes.
            attention_mask: Attention mask, shape ``(B, L)``. Required for semantic/fusion modes.
            node_types: List of node type strings for graph. Required for graph/fusion modes.
            edge_index: Edge indices, shape ``(2, E)``. Required for graph/fusion modes.
            edge_type: Edge type indices, shape ``(E,)``. Required for graph/fusion modes.
            batch: Graph batch vector, shape ``(N,)``. Required for graph/fusion modes.
            tasks: List of tasks to compute. Default: all tasks.
                Options: "binary", "cwe", "localization", "source_sink".

        Returns:
            ModelOutput containing logits for requested tasks.
        """
        if tasks is None:
            tasks = ["binary", "cwe", "localization", "source_sink", "severity"]

        output = ModelOutput()

        # ──── Encode ────
        if self.mode == "semantic_only":
            pooled, seq = self.semantic_encoder(input_ids, attention_mask, return_sequence=True)
            fused_pooled = pooled
            fused_seq = seq

        elif self.mode == "graph_only":
            graph_pooled, node_emb = self.graph_encoder(
                node_types, edge_index, edge_type, batch, return_node_embeddings=True,
            )
            fused_pooled = graph_pooled
            # For sequence-level tasks, expand graph pooled to fake sequence dim
            fused_seq = graph_pooled.unsqueeze(1)

        elif self.mode == "fusion":
            # Semantic branch
            sem_pooled, sem_seq = self.semantic_encoder(input_ids, attention_mask, return_sequence=True)
            # Graph branch
            graph_pooled, node_emb = self.graph_encoder(
                node_types, edge_index, edge_type, batch, return_node_embeddings=True,
            )
            # Fuse at pooled level
            fused_pooled = self.fusion(sem_pooled, graph_pooled)
            # For sequence-level tasks, use the semantic sequence
            fused_seq = sem_seq

        output.fused_embedding = fused_pooled

        # ──── Predict ────
        if "binary" in tasks:
            output.binary_logits = self.binary_head(fused_pooled)

        if "cwe" in tasks:
            output.cwe_logits = self.cwe_head(fused_pooled)

        if "localization" in tasks:
            output.localization_logits = self.localization_head(fused_seq)

        if "source_sink" in tasks:
            output.source_sink_logits = self.source_sink_head(fused_seq)

        if "severity" in tasks:
            output.severity_logits = self.severity_head(fused_pooled)

        return output

    @classmethod
    def from_config(cls, config: dict) -> "VulHunterModel":
        """Create a model from a nested configuration dictionary.

        Args:
            config: Configuration dict with keys matching __init__ parameters.

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
            num_cwe_classes=model_cfg.get("heads", {}).get("cwe", {}).get("num_classes", 10),
        )
