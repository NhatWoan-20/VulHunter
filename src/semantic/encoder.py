"""Semantic Encoder — CodeBERT for Vulnerability Detection.

Architecture:
    Code String -> Tokenizer -> Transformer Backbone -> [CLS]/mean pooling -> Projection -> Embedding
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SemanticEncoder(nn.Module):
    """CodeBERT semantic encoder (Full Fine-Tuning).
    
    Phù hợp cho các baseline semantic-only với context ngắn (512 tokens).
    Không sử dụng Full Fine-tuning vì mô hình đủ nhỏ (~125M params) để full-finetune.
    """
    def __init__(
        self,
        backbone: str = "microsoft/codebert-base",
        output_dim: int = 256,
        dropout: float = 0.1,
        **_extra_kwargs,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("Install `transformers`: pip install transformers") from exc
            
        logger.info("Loading CodeBERT backbone: %s", backbone)
        # Using feature extraction
        self.backbone = AutoModel.from_pretrained(backbone)
        self.hidden_size = self.backbone.config.hidden_size
        
        self.projection = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_sequence: bool = False):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
            
        pooled_out = self.projection(pooled.to(next(self.projection.parameters()).dtype))
        
        if return_sequence:
            seq_out = self.projection(outputs.last_hidden_state.to(next(self.projection.parameters()).dtype))
            return pooled_out, seq_out
            
        return pooled_out

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

