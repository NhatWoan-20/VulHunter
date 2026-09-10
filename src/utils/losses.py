"""Multi-Task Loss Functions — Loss computation for all vulnerability detection tasks.

Implements:
    - FocalLoss: For imbalanced binary classification
    - MultiTaskLoss: Weighted combination of per-task losses with curriculum support

The multi-task loss adaptively weights each task's contribution based on the loss
weights and, optionally, per-sample confidence (quality-aware sample weighting,
Pillar 4): gold CVEFixes samples keep full weight, silver GHSA samples are
down-weighted so noisy auto-derived diffs perturb gradients less.

"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

QUALITY_TIER_WEIGHTS = {
    "gold": 1.0,
    "silver": 0.85,
}


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""

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


class MultiTaskLoss(nn.Module):
    """Combined multi-task loss with per-task weighting.


    Args:
        loss_weights: Dict mapping task names to their loss weights.
        focal_alpha: Alpha for binary focal loss.
        focal_gamma: Gamma for binary focal loss.
        num_cwe_classes: unused but kept for API compat.
    """

    def __init__(
        self,
        loss_weights: Optional[dict[str, float]] = None,
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0,
        num_cwe_classes: int = 10,
    ) -> None:
        super().__init__()
        self.loss_weights = loss_weights or {
            "binary": 1.0,
            "cwe": 0.5,
            "severity": 0.2,
        }
        self.binary_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.cwe_loss = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)

    def update_weights(self, weights: dict[str, float]) -> None:
        self.loss_weights.update(weights)



    def forward(
        self,
        binary_logits: Optional[torch.Tensor] = None,
        binary_targets: Optional[torch.Tensor] = None,
        cwe_logits: Optional[torch.Tensor] = None,
        cwe_targets: Optional[torch.Tensor] = None,
        severity_logits: Optional[torch.Tensor] = None,
        severity_targets: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        device: Optional[torch.device] = None

        if sample_weights is not None:
            sample_weights = sample_weights.float()

        def _weighted_2d(loss_elem: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
            nonlocal device
            if device is None:
                device = loss_elem.device
            if sample_weights is None:
                return loss_elem[valid].mean() if valid is not None else loss_elem.mean()
            w = sample_weights if valid is None else torch.where(valid, sample_weights, torch.zeros_like(sample_weights))
            denom = w.sum().clamp(min=1e-6)
            return (loss_elem * w).sum() / denom

        if binary_logits is not None:
            device = binary_logits.device
            if binary_targets is not None and self.loss_weights.get("binary", 0) > 0:
                logits = binary_logits.view(-1)
                targets = binary_targets.float().view(-1)
                ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
                probs = torch.sigmoid(logits)
                p_t = probs * targets + (1 - probs) * (1 - targets)
                focal_w = (1 - p_t) ** self.binary_loss.gamma
                alpha_t = self.binary_loss.alpha * targets + (1 - self.binary_loss.alpha) * (1 - targets)
                losses["binary"] = _weighted_2d(alpha_t * focal_w * ce)
            else:
                losses["binary"] = (binary_logits * 0.0).sum()

        if cwe_logits is not None:
            device = cwe_logits.device
            computed_cwe = False
            if cwe_targets is not None and self.loss_weights.get("cwe", 0) > 0:
                valid = cwe_targets >= 0
                if valid.any():
                    ce = F.cross_entropy(cwe_logits[valid], cwe_targets[valid], reduction="none", label_smoothing=0.1)
                    losses["cwe"] = _weighted_2d(ce, None)
                    computed_cwe = True
            if not computed_cwe:
                losses["cwe"] = (cwe_logits * 0.0).sum()



        if severity_logits is not None:
            device = severity_logits.device
            computed_sev = False
            if severity_targets is not None and self.loss_weights.get("severity", 0) > 0:
                valid = severity_targets >= 0
                if valid.any():
                    _ce = F.cross_entropy(severity_logits[valid], severity_targets[valid], reduction="none", label_smoothing=0.05)
                    if sample_weights is not None:
                        w = sample_weights[valid]
                        losses["severity"] = (_ce * w).sum() / w.sum().clamp(min=1e-6)
                    else:
                        losses["severity"] = _ce.mean()
                    computed_sev = True
            if not computed_sev:
                losses["severity"] = (severity_logits * 0.0).sum()

        if device is None:
            device = torch.device("cpu")
        total = torch.tensor(0.0, device=device)
        for task, loss in losses.items():
            total = total + self.loss_weights.get(task, 1.0) * loss
        losses["total"] = total
        return losses
