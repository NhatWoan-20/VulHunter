"""Loss Functions — Loss computation cho binary vulnerability detection.

Implements:
    - FocalLoss: For imbalanced binary classification
    - BinaryClassificationLoss: Wrapper cho FocalLoss
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Args:
        alpha: Weighting factor for positive class (0-1).
            Default 0.5 (balanced); nên auto-tune dựa trên pos/neg ratio.
        gamma: Focusing parameter (>=0). Higher gamma = giảm loss cho easy examples.
            Standard RetinaNet: gamma=2.0.
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.float().view(-1)
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class BinaryClassificationLoss(nn.Module):
    """Binary focal loss.

    Returns dict với keys 'binary' và 'total' để tương thích API cũ.
    """

    def __init__(self, focal_alpha: float = 0.5, focal_gamma: float = 2.0) -> None:
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(
        self,
        binary_logits: torch.Tensor,
        binary_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute binary focal loss.

        Args:
            binary_logits: Logits shape ``(B, 1)`` hoặc ``(B,)``.
            binary_targets: Targets shape ``(B,)`` values in {0, 1}.

        Returns:
            Dict với keys 'binary' và 'total'.
        """
        loss = self.focal_loss(binary_logits, binary_targets)
        return {"binary": loss, "total": loss}
