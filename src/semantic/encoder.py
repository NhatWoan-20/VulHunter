"""Semantic Encoder — CodeBERT for Vulnerability Detection.

Architecture:
    Code String -> Tokenizer -> Transformer Backbone -> [CLS]/mean pooling -> Projection -> Embedding

Kỹ thuật tối ưu hóa (aligned với codebert_finetune_optimized.ipynb):
    - Full fine-tuning (~125M params, đủ nhỏ, không cần PEFT/LoRA)
    - Hỗ trợ freeze_layers: đóng băng N layer đầu của backbone
    - Gradient checkpointing: giảm VRAM, trade-off speed
    - Pooling modes: 'last' (CLS token, default) hoặc 'mean' (mean pooling)
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SemanticEncoder(nn.Module):
    """CodeBERT semantic encoder (Full Fine-Tuning).

    Phù hợp cho các baseline semantic-only với context ngắn (512 tokens).
    Không sử dụng PEFT/LoRA vì mô hình đủ nhỏ (~125M params) để full-finetune.

    Args:
        backbone: HuggingFace model ID. Default "microsoft/codebert-base".
        output_dim: Projected embedding dimension. Default 256.
        dropout: Dropout probability. Default 0.1.
        freeze_layers: Số transformer layer đầu của backbone bị đóng băng.
            0 = full fine-tuning (mặc định optimized).
            Ví dụ: 6 = freeze 6 layer đầu, fine-tune phần còn lại.
        gradient_checkpointing: Bật gradient checkpointing để giảm VRAM.
            Trade-off: tốc độ forward pass chậm hơn ~20% nhưng tiết kiệm ~40% VRAM.
        pooling: Chiến lược pooling. 'last' = CLS token (default), 'mean' = mean pooling.
        use_fp16: Không dùng, backbone luôn float32 (AMP quản lý FP16 externally).
    """

    def __init__(
        self,
        backbone: str = "microsoft/codebert-base",
        output_dim: int = 256,
        dropout: float = 0.1,
        freeze_layers: int = 0,
        gradient_checkpointing: bool = False,
        pooling: str = "last",
        use_fp16: bool = False,  # Được quản lý externally bởi torch.amp, không set tại đây
        **_extra_kwargs,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.pooling = pooling

        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("Install `transformers`: pip install transformers") from exc

        logger.info("Loading CodeBERT backbone: %s", backbone)
        self.backbone = AutoModel.from_pretrained(backbone)
        self.hidden_size = self.backbone.config.hidden_size

        # ── Gradient Checkpointing (kỹ thuật tối ưu VRAM) ──
        if gradient_checkpointing:
            if hasattr(self.backbone, "gradient_checkpointing_enable"):
                self.backbone.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            else:
                logger.warning("backbone không hỗ trợ gradient_checkpointing_enable()")

        # ── Freeze Layers (Tiered Fine-Tuning) ──
        # Đóng băng embedding + N layer transformer đầu tiên
        # Tương đương với việc chỉ fine-tune các layer cao hơn của backbone
        if freeze_layers > 0:
            frozen_count = 0
            # Luôn freeze embeddings
            for p in self.backbone.embeddings.parameters():
                p.requires_grad = False
                frozen_count += p.numel()

            # Freeze N encoder layers đầu
            encoder_layers = None
            if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
                encoder_layers = self.backbone.encoder.layer
            elif hasattr(self.backbone, "transformer") and hasattr(self.backbone.transformer, "layer"):
                encoder_layers = self.backbone.transformer.layer

            if encoder_layers is not None:
                actual_freeze = min(freeze_layers, len(encoder_layers))
                for i in range(actual_freeze):
                    for p in encoder_layers[i].parameters():
                        p.requires_grad = False
                        frozen_count += p.numel()
                logger.info(
                    "Froze embeddings + %d/%d encoder layers (%.1fM params frozen)",
                    actual_freeze, len(encoder_layers), frozen_count / 1e6
                )
            else:
                logger.warning("Không tìm thấy encoder layers để freeze (backbone structure khác)")

        # ── Projection Head ──
        self.projection = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "SemanticEncoder [%s pooling, gc=%s, freeze=%d]: %s/%s trainable (%.1f%%)",
            pooling, gradient_checkpointing, freeze_layers,
            f"{trainable:,}", f"{total:,}", 100.0 * trainable / total if total > 0 else 0,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_sequence: bool = False,
    ):
        """Forward pass.

        Args:
            input_ids: Token IDs, shape (B, L).
            attention_mask: Attention mask, shape (B, L).
            return_sequence: Nếu True, trả về (pooled, sequence) thay vì chỉ pooled.

        Returns:
            pooled: shape (B, output_dim).
            sequence (optional): shape (B, L, output_dim) nếu return_sequence=True.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # ── Pooling ──
        if self.pooling == "mean":
            # Mean pooling qua các token không padding
            last_hidden = outputs.last_hidden_state  # (B, L, H)
            mask_expanded = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
            pooled = (last_hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        else:
            # 'last' → CLS token (index 0), sử dụng pooler_output nếu có
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled = outputs.pooler_output
            else:
                pooled = outputs.last_hidden_state[:, 0, :]

        pooled_out = self.projection(pooled)

        if return_sequence:
            seq_out = self.projection(outputs.last_hidden_state)
            return pooled_out, seq_out

        return pooled_out

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
