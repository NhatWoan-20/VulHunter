"""Multi-Task Loss Functions — Loss computation for all vulnerability detection tasks.

Implements:
    - FocalLoss: For imbalanced binary classification
    - MultiTaskLoss: Weighted combination of per-task losses with curriculum support

The multi-task loss adaptively weights each task's contribution based on the loss
weights and, optionally, per-sample confidence (quality-aware sample weighting,
Pillar 4): gold CVEFixes samples keep full weight, silver GHSA samples are
down-weighted so noisy auto-derived diffs perturb gradients less.

Extensions in v3.2:
    - Localization now pools per-token logits -> per-line logits via max-pool
      using ``token_line_ids`` (B, L) where -1 marks special/pad tokens.
    - Source/Sink is a token-level 3-class CE with weak lexicon supervision.
"""
from __future__ import annotations

from typing import Optional

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F

QUALITY_TIER_WEIGHTS = {
    "gold": 1.0,
    "silver": 0.85,
}


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
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

    Extensions:
        - ``localization`` now expects ``token_line_ids`` to pool tokens->lines.
        - ``source_sink`` is a normal token-level CE (ignore_index=-1).

    Args:
        loss_weights: Dict mapping task names to their loss weights.
        focal_alpha: Alpha for binary focal loss.
        focal_gamma: Gamma for binary focal loss.
        num_cwe_classes: unused but kept for API compat.
    """

    def __init__(
        self,
        loss_weights: Optional[dict[str, float]] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        num_cwe_classes: int = 10,
    ) -> None:
        super().__init__()
        self.loss_weights = loss_weights or {
            "binary": 1.0,
            "cwe": 0.5,
            "localization": 0.4,
            "source_sink": 0.15,
            "severity": 0.2,
        }
        self.binary_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.cwe_loss = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)
        self.localization_loss = FocalLoss(alpha=0.5, gamma=2.0)
        self.source_sink_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def update_weights(self, weights: dict[str, float]) -> None:
        self.loss_weights.update(weights)

    @staticmethod
    def _pool_tokens_to_lines(
        token_logits: torch.Tensor,
        token_line_ids: torch.Tensor,
        line_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool per-token logits to per-line logits via max.

        Args:
            token_logits: (B, L) float
            token_line_ids: (B, L) long, -1 for special/pad
            line_targets: (B, max_lines) long, -1 for pad lines

        Returns:
            flat_line_logits (N_lines,), flat_line_targets (N_lines,), flat_sample_idx (N_lines,)
            where N_lines is total valid aligned lines across batch.
        """
        B = token_logits.size(0)
        line_logits: list[torch.Tensor] = []
        line_targets_flat: list[torch.Tensor] = []
        sample_idx_flat: list[torch.Tensor] = []
        for b in range(B):
            tl = token_line_ids[b]
            lt = line_targets[b]
            valid_mask = lt != -1
            if not valid_mask.any():
                continue
            for line_id in torch.where(valid_mask)[0].tolist():
                tok_mask = tl == line_id
                if not tok_mask.any():
                    continue
                logit = token_logits[b][tok_mask].max()
                line_logits.append(logit.unsqueeze(0))
                line_targets_flat.append(lt[line_id].float().unsqueeze(0))
                sample_idx_flat.append(torch.tensor([b], device=token_logits.device))
        if not line_logits:
            return (
                torch.empty(0, device=token_logits.device),
                torch.empty(0, device=token_logits.device),
                torch.empty(0, dtype=torch.long, device=token_logits.device),
            )
        return (
            torch.cat(line_logits, dim=0),
            torch.cat(line_targets_flat, dim=0),
            torch.cat(sample_idx_flat, dim=0),
        )

    def forward(
        self,
        binary_logits: Optional[torch.Tensor] = None,
        binary_targets: Optional[torch.Tensor] = None,
        cwe_logits: Optional[torch.Tensor] = None,
        cwe_targets: Optional[torch.Tensor] = None,
        localization_logits: Optional[torch.Tensor] = None,
        localization_targets: Optional[torch.Tensor] = None,
        source_sink_logits: Optional[torch.Tensor] = None,
        source_sink_targets: Optional[torch.Tensor] = None,
        severity_logits: Optional[torch.Tensor] = None,
        severity_targets: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
        token_line_ids: Optional[torch.Tensor] = None,
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

        if binary_logits is not None and binary_targets is not None and self.loss_weights.get("binary", 0) > 0:
            device = binary_logits.device
            logits = binary_logits.view(-1)
            targets = binary_targets.float().view(-1)
            ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            probs = torch.sigmoid(logits)
            p_t = probs * targets + (1 - probs) * (1 - targets)
            focal_w = (1 - p_t) ** self.binary_loss.gamma
            alpha_t = self.binary_loss.alpha * targets + (1 - self.binary_loss.alpha) * (1 - targets)
            losses["binary"] = _weighted_2d(alpha_t * focal_w * ce)

        if cwe_logits is not None and cwe_targets is not None and self.loss_weights.get("cwe", 0) > 0:
            device = cwe_logits.device
            valid = cwe_targets >= 0
            ce = F.cross_entropy(cwe_logits, cwe_targets, reduction="none", label_smoothing=0.1)
            losses["cwe"] = _weighted_2d(ce, valid)

        # Localization: pool tokens -> lines
        if (
            localization_logits is not None
            and localization_targets is not None
            and self.loss_weights.get("localization", 0) > 0
        ):
            device = localization_logits.device
            tok_logits = localization_logits.squeeze(-1)  # (B, L)
            if token_line_ids is not None:
                # ensure shape alignment: tok_logits seq len == token_line_ids seq len
                # If mismatch due to truncation/padding, slice to min
                min_L = min(tok_logits.size(1), token_line_ids.size(1))
                tok_logits_s = tok_logits[:, :min_L]
                tl = token_line_ids[:, :min_L]
                flat_logits, flat_targets, flat_sample = self._pool_tokens_to_lines(
                    tok_logits_s, tl, localization_targets
                )
                if flat_logits.numel() > 0:
                    if sample_weights is not None:
                        # per-line weight = sample weight of its sample
                        w = sample_weights[flat_sample]
                        ce = F.binary_cross_entropy_with_logits(flat_logits, flat_targets, reduction="none")
                        probs = torch.sigmoid(flat_logits)
                        p_t = probs * flat_targets + (1 - probs) * (1 - flat_targets)
                        focal_w = (1 - p_t) ** 2.0
                        alpha_t = 0.5 * flat_targets + 0.5 * (1 - flat_targets)
                        # alpha 0.5 => symmetric
                        loss_elem = alpha_t * focal_w * ce
                        losses["localization"] = (loss_elem * w).sum() / w.sum().clamp(min=1e-6)
                    else:
                        losses["localization"] = self.localization_loss(flat_logits, flat_targets)
            else:
                # Fallback: if no token_line_ids, treat as token-level (should not happen after fix)
                flat_logits = tok_logits.view(-1)
                flat_targets = localization_targets.float().view(-1) if localization_targets.dim() == 2 else localization_targets.float().view(-1)
                # Can't align shapes; skip
                pass

        if source_sink_logits is not None and source_sink_targets is not None and self.loss_weights.get("source_sink", 0) > 0:
            device = source_sink_logits.device
            B, L, C = source_sink_logits.shape
            # need to align L with targets
            min_L = min(L, source_sink_targets.size(1))
            logits_s = source_sink_logits[:, :min_L, :].reshape(B * min_L, C)
            targets_s = source_sink_targets[:, :min_L].reshape(B * min_L)
            valid = targets_s != -1
            if valid.any():
                ce = F.cross_entropy(logits_s[valid], targets_s[valid], reduction="none")
                if sample_weights is not None:
                    # expand sample_weights per token
                    w_tok = sample_weights.unsqueeze(1).expand(B, min_L).reshape(B * min_L)[valid]
                    losses["source_sink"] = (ce * w_tok).sum() / w_tok.sum().clamp(min=1e-6)
                else:
                    losses["source_sink"] = ce.mean()
            else:
                losses["source_sink"] = torch.tensor(0.0, device=device)

        if severity_logits is not None and severity_targets is not None and self.loss_weights.get("severity", 0) > 0:
            device = severity_logits.device
            valid = severity_targets >= 0
            if valid.any():
                _ce = F.cross_entropy(severity_logits[valid], severity_targets[valid], reduction="none", label_smoothing=0.05)
                if sample_weights is not None:
                    w = sample_weights[valid]
                    losses["severity"] = (_ce * w).sum() / w.sum().clamp(min=1e-6)
                else:
                    losses["severity"] = _ce.mean()

        if device is None:
            device = torch.device("cpu")
        total = torch.tensor(0.0, device=device)
        for task, loss in losses.items():
            total = total + self.loss_weights.get(task, 0.0) * loss
        losses["total"] = total
        return losses
