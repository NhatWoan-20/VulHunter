"""Metrics — Evaluation metrics cho binary vulnerability detection.

Cung cấp:
    - binary_metrics: Precision, Recall, F1, Accuracy, ROC-AUC, PR-AUC, MCC.
    - best_threshold: Tìm threshold tối ưu dựa trên F1.

All metrics operate on numpy arrays for compatibility with scikit-learn.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MetricResult:
    """Container cho binary classification metrics.

    Tất cả fields default 0.0 nếu không compute.
    """
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    auc: float = 0.0
    pr_auc: float = 0.0
    mcc: float = 0.0
    threshold: float = 0.5
    support: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "support": self.support,
            "threshold": round(self.threshold, 4),
        }
        if self.auc > 0:
            d["auc"] = round(self.auc, 4)
        d["pr_auc"] = round(self.pr_auc, 4)
        d["mcc"] = round(self.mcc, 4)
        return d


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    threshold: float = 0.5,
) -> MetricResult:
    """Compute binary classification metrics.

    Args:
        y_true: Ground truth labels, shape ``(N,)``, values in {0, 1}.
        y_pred: Predicted labels, shape ``(N,)``, values in {0, 1}.
        y_prob: Predicted probabilities for positive class, shape ``(N,)``.
            Nếu cung cấp, ROC-AUC và PR-AUC cũng được tính.
        threshold: Threshold đã dùng để tạo y_pred (cho logging).

    Returns:
        MetricResult với precision, recall, F1, accuracy, AUC.
    """
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    result = MetricResult(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        threshold=threshold,
        support=len(y_true),
    )

    # MCC — đặc biệt hữu ích với imbalanced data
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    result.mcc = float((tp * tn - fp * fn) / denominator) if denominator else 0.0

    # ROC-AUC và PR-AUC
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
            result.auc = float(roc_auc_score(y_true, y_prob))
            result.pr_auc = float(average_precision_score(y_true, y_prob))
        except (ImportError, ValueError):
            pass

    return result


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, MetricResult]:
    """Tìm threshold tối ưu dựa trên F1 score.

    Args:
        y_true: Ground truth labels, shape ``(N,)``.
        y_prob: Predicted probabilities, shape ``(N,)``.
        thresholds: Optional array of thresholds to test.
            Default: np.arange(0.05, 0.96, 0.05).

    Returns:
        Tuple of (best_threshold, MetricResult at best threshold).
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    best_thr, best_f1 = 0.5, 0.0
    best_result = binary_metrics(y_true, (y_prob >= 0.5).astype(int), y_prob, threshold=0.5)

    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        result = binary_metrics(y_true, preds, y_prob, threshold=float(thr))
        if result.f1 > best_f1:
            best_f1 = result.f1
            best_thr = float(thr)
            best_result = result

    return best_thr, best_result
