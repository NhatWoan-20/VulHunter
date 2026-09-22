# 06 — Evaluation & Metrics

> **Version: 5.0** — Binary Classification Focus
> **Authoritative Specification**

---

## 1. Evaluation Protocol

Evaluation follows a **frozen-checkpoint** protocol:

| Tier | Benchmark | Data | Purpose |
|---|---|---|---|
| **In-Domain** | Unified Test Set | `data/splits/test.jsonl` (10% Master, repo-disjoint) | Overall performance |
| **Gold Sanity** | CVEFixes-only Test | test restricted to `data_source=="cvefixes"` | Gold-only baseline |
| **Out-of-Domain** | PyCode-Vul | External benchmark | Zero-shot generalization |

---

## 2. Evaluation Commands

```bash
# In-domain evaluation (all modes)
python scripts/evaluation/evaluate.py \
    --checkpoint models/checkpoints/best.pt \
    --test-data data/splits/test.jsonl \
    --graph-data data/processed/master_graphs.jsonl \
    --output outputs/metrics/evaluation_report.json

# With automatic threshold tuning (default)
python scripts/evaluation/evaluate.py \
    --checkpoint models/checkpoints/best.pt \
    --test-data data/splits/test.jsonl

# Use specific threshold
python scripts/evaluation/evaluate.py \
    --checkpoint models/checkpoints/best.pt \
    --threshold 0.5

# Skip threshold tuning
python scripts/evaluation/evaluate.py \
    --checkpoint models/checkpoints/best.pt \
    --no-tune
```

---

## 3. Binary Classification Metrics

### 3.1 Primary Metrics

| Metric | Formula | Target |
|---|---|---|
| **F1 Score** | $F1 = 2 \cdot \frac{P \cdot R}{P + R}$ | > 0.60 |
| **Precision** | $P = \frac{TP}{TP + FP}$ | > 0.70 |
| **Recall** | $R = \frac{TP}{TP + FN}$ | > 0.65 |
| **ROC-AUC** | Area under ROC curve | > 0.75 |

### 3.2 Secondary Metrics

| Metric | Description | Target |
|---|---|---|
| **Accuracy** | $(TP + TN) / (TP + TN + FP + FN)$ | > 0.70 |
| **PR-AUC** | Area under Precision-Recall curve | > 0.60 |
| **MCC** | Matthews Correlation Coefficient | > 0.40 |

### 3.3 Implementation

All metrics are computed in `src/utils/metrics.py`:

```python
from src.utils.metrics import binary_metrics, find_best_threshold

# Compute metrics at fixed threshold
metrics = binary_metrics(y_true, y_pred, y_prob, threshold=0.5)

# Find optimal threshold for F1
best_thr, optimal_metrics = find_best_threshold(y_true, y_prob)
```

---

## 4. Metric Targets by Performance Level

| Metric | Poor | Fair | Good | Excellent |
|---|---|---|---|---|
| ROC-AUC | < 0.55 | 0.55-0.65 | 0.65-0.75 | 0.75-0.85 | > 0.85 |
| F1 Score | < 0.20 | 0.20-0.40 | 0.40-0.60 | 0.60-0.75 | > 0.75 |
| Precision | — | — | — | > 0.70 | > 0.80 |
| Recall | — | — | — | > 0.65 | > 0.75 |

---

## 5. Comparison Strategy (RQ1)

Report all three branches on the same metrics:

| Branch | Val F1 | Test F1 | Test MCC | AUC-ROC | Precision | Recall |
|---|---|---|---|---|---|---|
| `semantic_only` | … | … | … | … | … | … |
| `graph_only` | … | … | … | … | … | … |
| `fusion` | … | … | … | … | … | … |

---

## 6. Diagnostic Visualizations

The training notebooks generate diagnostic plots:

1. **Confusion Matrix:** True vs Predicted labels
2. **ROC Curve:** TPR vs FPR with AUC
3. **Precision-Recall Curve:** P vs R with AP
4. **Probability Distribution:** P(vulnerable) for vulnerable vs safe samples
5. **Training Curves:** Loss and F1 over epochs

---

## 7. Output Schema

Evaluation results are saved to JSON:

```json
{
  "metrics": {
    "binary": {
      "precision": 0.75,
      "recall": 0.68,
      "f1": 0.71,
      "accuracy": 0.73,
      "auc": 0.82,
      "pr_auc": 0.68,
      "mcc": 0.46,
      "threshold": 0.45,
      "support": 3412
    }
  },
  "threshold": 0.45,
  "num_samples": 3412,
  "predictions_summary": {
    "total": 3412,
    "positive": 1234,
    "negative": 2178
  },
  "prob_stats": {
    "min": 0.01,
    "max": 0.99,
    "mean": 0.52,
    "std": 0.31
  },
  "evaluation_time_seconds": 45.2,
  "checkpoint": "models/checkpoints/best.pt",
  "test_data": "data/splits/test.jsonl"
}
```

---

## 8. Reproducibility

- Checkpoints contain full configuration for exact reproduction.
- `training_history.json` logs per-epoch metrics.
- Fixed seed (42) for deterministic evaluation.

