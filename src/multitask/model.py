"""VulHunter Model — Binary vulnerability detection.

Main model class tích hợp:
    1. Semantic Encoder (CodeBERT backbone: microsoft/codebert-base)
    2. Graph Encoder (Heterogeneous PDG + RGCN)
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
        node_logits: Auxiliary node-level logits shape ``(N, 3)`` if enabled.
    """
    binary_logits: torch.Tensor | None = None
    fused_embedding: torch.Tensor | None = None
    node_logits: torch.Tensor | None = None

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

        semantic_config = dict(semantic_config) if semantic_config else {}
        graph_config = dict(graph_config) if graph_config else {}
        fusion_config = dict(fusion_config) if fusion_config else {}
        head_config = dict(head_config) if head_config else {}

        output_dim = semantic_config.get("output_dim", 256)

        # Khởi tạo encoders theo mode
        if mode in ("fusion", "semantic_only"):
            encoder_type = semantic_config.pop("encoder_type", "codebert")
            self.semantic_encoder = SemanticEncoder(**semantic_config)
            logger.info("Khởi tạo SemanticEncoder")

        if mode in ("fusion", "graph_only"):
            graph_config.setdefault("output_dim", output_dim)
            # Shared Semantic Backbone & Hybrid Representation: Chia sẻ CodeBERT word_embeddings cho GraphEncoder
            if graph_config.get("node_encoder_type") in ("shared_codebert", "both", "hybrid"):
                if hasattr(self, "semantic_encoder"):
                    try:
                        graph_config["shared_word_embeddings"] = self.semantic_encoder.backbone.embeddings.word_embeddings
                        logger.info("Liên kết shared CodeBERT word_embeddings cho GraphEncoder (Shared Backbone / Hybrid)")
                    except Exception as e:
                        logger.warning("Không thể trích xuất word_embeddings từ semantic_encoder: %s", e)
                elif graph_config.get("load_pretrained_word_embeddings", True):
                    try:
                        from transformers import AutoModel
                        cb = AutoModel.from_pretrained("microsoft/codebert-base")
                        graph_config["shared_word_embeddings"] = cb.embeddings.word_embeddings
                        logger.info("Đã nạp pre-trained CodeBERT word_embeddings cho Graph-Only (Shared Backbone / Hybrid)")
                    except Exception as e:
                        logger.warning("Không thể nạp pre-trained CodeBERT word_embeddings: %s", e)
            self.graph_encoder = GraphEncoder(**graph_config)
            logger.info("Khởi tạo GraphEncoder")
            
        if mode == "graph_only":
            output_dim = graph_config.get("output_dim", output_dim)

        # Fusion module (chỉ cho fusion mode)
        if mode == "fusion":
            fusion_config.setdefault("hidden_dim", output_dim)
            # Extract residual_alpha before passing to constructor (not an init param)
            self._fusion_residual_alpha = float(fusion_config.pop("residual_alpha", 0.2))
            # FiLM Modulation / Cross-Modal Fusion: Cho phép cấu hình fusion_level: "token" (token-level) hoặc "pooled" (late fusion)
            self.fusion_level = fusion_config.pop("fusion_level", "token")
            self.fusion = CrossModalFusion(**fusion_config)
            logger.info(
                "Khởi tạo CrossModalFusion (fusion_level=%s, residual_alpha=%.2f)",
                self.fusion_level, self._fusion_residual_alpha,
            )

        # Binary prediction head (task chính)
        binary_cfg = head_config.get("binary", {})
        self.binary_head = BinaryHead(input_dim=output_dim, **binary_cfg)

        # Đảm bảo trainable params (Full Fine-tuning adapters, projection, heads) đều ở float32
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
        node_is_source: list[bool] | None = None,
        node_is_sink: list[bool] | None = None,
        node_is_sanitizer: list[bool] | None = None,
        return_node_logits: bool = False,
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
            node_is_source: Optional bool list indicating source nodes.
            node_is_sink: Optional bool list indicating sink nodes.
            node_is_sanitizer: Optional bool list indicating sanitizer nodes.
            return_node_logits: If True and node classifier exists, return node logits.

        Returns:
            ModelOutput chứa binary_logits, fused_embedding, và node_logits (optional).
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

            ret_nodes = return_node_logits and getattr(self.graph_encoder, "node_classifier", None) is not None
            res = self.graph_encoder(
                node_types=node_types,
                edge_index=edge_index,
                edge_type=edge_type,
                batch=batch,
                node_texts=node_texts,
                node_is_source=node_is_source,
                node_is_sink=node_is_sink,
                node_is_sanitizer=node_is_sanitizer,
                return_node_embeddings=ret_nodes,
            )
            if ret_nodes:
                graph_pooled, node_embs = res
                output.node_logits = self.graph_encoder.node_classifier(node_embs)
            else:
                graph_pooled = res
            fused_pooled = graph_pooled

        elif self.mode == "fusion":
            # Graph branch: truyền enriched node_texts và vulnerability flow flags
            if edge_index is not None and edge_index.dim() == 1:
                edge_index = edge_index.new_zeros(2, 0)
            elif edge_index is not None and edge_index.dim() == 2 and edge_index.size(0) == 1:
                edge_index = edge_index.view(2, -1)
                
            ret_nodes = return_node_logits and getattr(self.graph_encoder, "node_classifier", None) is not None
            res = self.graph_encoder(
                node_types=node_types,
                edge_index=edge_index,
                edge_type=edge_type,
                batch=batch,
                node_texts=node_texts,
                node_is_source=node_is_source,
                node_is_sink=node_is_sink,
                node_is_sanitizer=node_is_sanitizer,
                return_node_embeddings=ret_nodes,
            )
            if ret_nodes:
                graph_pooled, node_embs = res
                output.node_logits = self.graph_encoder.node_classifier(node_embs)
            else:
                graph_pooled = res

            # Cross-modal Modulation: Token-level Fusion vs Vector-level Late Fusion
            if getattr(self, "fusion_level", "token") == "token":
                # Lấy cả sequence (B, L, D) từ CodeBERT
                sem_pooled, sem_sequence = self.semantic_encoder(
                    input_ids, attention_mask, return_sequence=True
                )
                # Token-level modulation / fusion: graph context điều biến toàn bộ sequence 512 tokens
                fused_seq = self.fusion(
                    semantic=sem_sequence,
                    graph=graph_pooled,
                    semantic_mask=attention_mask,
                    residual_alpha=self._fusion_residual_alpha,
                )
                # Pooling lại sau khi đã fuse ở cấp độ token
                if getattr(self.semantic_encoder, "pooling", "last") == "mean":
                    mask_expanded = attention_mask.unsqueeze(-1).to(fused_seq.dtype)
                    fused_pooled = (fused_seq * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
                else:
                    # 'last' -> CLS token tại vị trí 0
                    fused_pooled = fused_seq[:, 0, :]
            else:
                # Vector-level / Late fusion (siêu ổn định, ít gradient steps)
                sem_pooled = self.semantic_encoder(input_ids, attention_mask, return_sequence=False)
                fused_pooled = self.fusion(
                    semantic=sem_pooled,
                    graph=graph_pooled,
                    semantic_mask=None,
                    graph_batch=None,
                    residual_alpha=self._fusion_residual_alpha,
                )

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




