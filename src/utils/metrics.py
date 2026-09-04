"""Metrics — Evaluation metrics for all vulnerability detection tasks.

Provides metric computation for:
    - Binary detection: Precision, Recall, F1, ROC-AUC
    - CWE classification: Macro/Micro F1, Per-class F1
    - Line localization: Line-level Precision/Recall/F1, Top-k Accuracy
    - Source/Sink: Per-class F1

All metrics operate on numpy arrays for compatibility with scikit-learn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# pyrefly: ignore [missing-import]
import numpy as np


@dataclass
class MetricResult:
    """Container for computed metric values.

    All fields default to 0.0 if not computed.
    """
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    auc: float = 0.0
    support: int = 0
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "support": self.support,
        }
        if self.auc > 0:
            d["auc"] = round(self.auc, 4)
        if self.per_class:
            # pyrefly: ignore [bad-assignment]
            d["per_class"] = self.per_class
        return d


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> MetricResult:
    """Compute binary classification metrics.

    Args:
        y_true: Ground truth labels, shape ``(N,)``, values in {0, 1}.
        y_pred: Predicted labels, shape ``(N,)``, values in {0, 1}.
        y_prob: Predicted probabilities for positive class, shape ``(N,)``.
            If provided, ROC-AUC is also computed.

    Returns:
        MetricResult with precision, recall, F1, accuracy, and optionally AUC.
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    result = MetricResult(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        support=len(y_true),
    )

    # ROC-AUC
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            # pyrefly: ignore [missing-source-for-stubs]
            from sklearn.metrics import roc_auc_score
            # pyrefly: ignore [bad-assignment]
            result.auc = roc_auc_score(y_true, y_prob)
        except (ImportError, ValueError):
            pass

    return result


def multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> MetricResult:
    """Compute multi-class classification metrics.

    Args:
        y_true: Ground truth class indices, shape ``(N,)``.
        y_pred: Predicted class indices, shape ``(N,)``.
        class_names: Optional list of class name strings for per-class reporting.

    Returns:
        MetricResult with macro F1, accuracy, and per-class breakdown.
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))

    per_class = {}
    f1_scores = []
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        name = class_names[int(cls)] if class_names and int(cls) < len(class_names) else str(int(cls))
        per_class[name] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "support": int(np.sum(y_true == cls))}
        f1_scores.append(f)

    macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
    accuracy = np.sum(y_true == y_pred) / len(y_true) if len(y_true) > 0 else 0.0

    return MetricResult(
        f1=macro_f1,
        accuracy=accuracy,
        support=len(y_true),
        per_class=per_class,
    )


def localization_metrics(
    y_true: list[list[int]],
    y_pred: list[list[int]],
    top_k: int = 5,
) -> MetricResult:
    """Compute line-level vulnerability localization metrics.

    Evaluates how accurately the model identifies vulnerable lines.

    Args:
        y_true: List of ground truth line labels per sample.
            Each inner list has values in {0, 1} where 1 = vulnerable.
        y_pred: List of predicted line labels per sample.
        top_k: K value for top-k accuracy computation.

    Returns:
        MetricResult with line-level precision, recall, F1, and top-k accuracy.
    """
    all_tp = all_fp = all_fn = 0
    top_k_hits = 0
    total_samples = 0

    for true, pred in zip(y_true, y_pred):
        min_len = min(len(true), len(pred))
        true = true[:min_len]
        pred = pred[:min_len]

        true_set = {i for i, v in enumerate(true) if v == 1}
        pred_set = {i for i, v in enumerate(pred) if v == 1}

        all_tp += len(true_set & pred_set)
        all_fp += len(pred_set - true_set)
        all_fn += len(true_set - pred_set)

        # Top-k: did any of the top-k predictions hit a true vulnerable line?
        if true_set:
            total_samples += 1
            pred_top_k = sorted(range(len(pred)), key=lambda i: pred[i] if i < len(pred) else 0, reverse=True)[:top_k]
            if any(i in true_set for i in pred_top_k):
                top_k_hits += 1

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    top_k_acc = top_k_hits / total_samples if total_samples > 0 else 0.0

    return MetricResult(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=top_k_acc,
        support=total_samples,
    )


def compute_all_metrics(
    binary_true: Optional[np.ndarray] = None,
    binary_pred: Optional[np.ndarray] = None,
    binary_prob: Optional[np.ndarray] = None,
    cwe_true: Optional[np.ndarray] = None,
    cwe_pred: Optional[np.ndarray] = None,
    cwe_names: Optional[list[str]] = None,
    loc_true: Optional[list[list[int]]] = None,
    loc_pred: Optional[list[list[int]]] = None,
) -> dict[str, MetricResult]:
    """Compute all metrics for the multi-task model.

    Args:
        binary_true/pred/prob: Binary detection arrays.
        cwe_true/pred: CWE classification arrays.
        cwe_names: CWE class name strings.
        loc_true/pred: Line-level localization labels.

    Returns:
        Dictionary mapping task names to MetricResult objects.
    """
    results: dict[str, MetricResult] = {}

    if binary_true is not None and binary_pred is not None:
        results["binary"] = binary_metrics(binary_true, binary_pred, binary_prob)

    if cwe_true is not None and cwe_pred is not None:
        results["cwe"] = multiclass_metrics(cwe_true, cwe_pred, cwe_names)

    if loc_true is not None and loc_pred is not None:
        results["localization"] = localization_metrics(loc_true, loc_pred)

    return results
