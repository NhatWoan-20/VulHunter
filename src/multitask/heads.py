"""Prediction Heads — Task-specific prediction layers.

Currently supports binary vulnerability detection (vulnerable vs. safe).
Multi-task heads (CWE, Severity) sẽ được thêm sau khi hoàn thiện task chính.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BinaryHead(nn.Module):
    """Binary vulnerability detection head.

    Predicts whether a code snippet is vulnerable (1) or safe (0).

    Args:
        input_dim: Dimension of the fused representation.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout probability.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict binary vulnerability label.

        Args:
            x: Pooled fused representation of shape ``(B, input_dim)``.

        Returns:
            Logits of shape ``(B, 1)``. Apply sigmoid for probabilities.
        """
        return self.classifier(x)
