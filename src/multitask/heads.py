"""Multi-Task Prediction Heads — Task-specific prediction layers.

Each head takes the shared fused representation and produces predictions
for its corresponding task. All heads are lightweight MLP classifiers
that branch out from the shared backbone.

Heads:
    - BinaryHead: Vulnerable vs. safe (binary classification)
    - CWEHead: Vulnerability type classification (multi-class)
    - LocalizationHead: Line-level vulnerability prediction (sequence labeling)
    - SourceSinkHead: Source/Propagation/Sink detection (token classification)
"""
from __future__ import annotations

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
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


class SeverityHead(nn.Module):
    """Severity classification head for labeled vulnerable samples.

    Args:
        input_dim: Dimension of fused / pooled representation.
        num_classes: 4 severity tiers (LOW=0, MODERATE/MEDIUM=1, HIGH=2, CRITICAL=3).
            Samples with UNKNOWN severity are masked (label -1) and do not
            contribute to the loss. Must match ``SEVERITY_CLASSES`` in
            ``src/utils/dataset.py`` minus the masked UNKNOWN entry.
        hidden_dim: Hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(self, input_dim: int = 256, num_classes: int = 4, hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class CWEHead(nn.Module):
    """CWE classification head.

    Predicts the vulnerability type category (e.g., CWE-89, CWE-79).

    Args:
        input_dim: Dimension of the fused representation.
        num_classes: Number of CWE categories (including "none/other").
        hidden_dim: Hidden layer dimension.
        dropout: Dropout probability.
    """

    def __init__(self, input_dim: int = 256, num_classes: int = 10, hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict CWE category.

        Args:
            x: Pooled fused representation of shape ``(B, input_dim)``.

        Returns:
            Logits of shape ``(B, num_classes)``. Apply softmax for probabilities.
        """
        return self.classifier(x)


class LocalizationHead(nn.Module):
    """Line-level vulnerability localization head.

    Per-token projection -> max-pool over ``token_line_ids`` to per-line
    probabilities. Active from v3.2: ``tokenize_qwen.py`` now writes
    ``token_line_ids_qwen`` (via offset_mapping) and
    ``MultiTaskLoss._pool_tokens_to_lines`` supervises the head with
    ``loss_weights.localization=0.4``.

    Args:
        input_dim: Dimension of per-token features.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout probability.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict per-token vulnerability scores.

        Args:
            x: Per-token fused features of shape ``(B, L, input_dim)``.

        Returns:
            Per-token logits of shape ``(B, L, 1)``.
            Apply sigmoid to get vulnerability probabilities per token.
        """
        return self.classifier(x)


class SourceSinkHead(nn.Module):
    """Source/Sink/Normal detection head (token-level 3-class).

    Weakly supervised by ``scripts/preprocessing/generate_source_sink_labels.py``
    (lexicon + propagation heuristics). Active from v3.2 with
    ``loss_weights.source_sink=0.15`` — class-imbalanced but non-zero.

    Args:
        input_dim: Dimension of per-token features.
        num_classes: Number of classes (default: 3 for normal/source/sink).
        hidden_dim: Hidden layer dimension.
        dropout: Dropout probability.
    """

    def __init__(self, input_dim: int = 256, num_classes: int = 3, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict source/sink/normal labels per token.

        Args:
            x: Per-token fused features of shape ``(B, L, input_dim)``.

        Returns:
            Per-token logits of shape ``(B, L, num_classes)``.
        """
        return self.classifier(x)
