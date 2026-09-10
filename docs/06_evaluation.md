# 06 — Evaluation & Ablation: Multi-Tier Benchmark (Pillar 3)

> **Version: 4.0** — **3 trainable tasks**
> **Authoritative Specification**

---

## 1. Multi-Tier Benchmark Protocol

Evaluation follows a **frozen-checkpoint** protocol in fixed order:

| Tier | Benchmark | Data | Goal |
|---|---|---|---|
| **Benchmark 1** | Unified In-Domain Test | `data/splits/test.jsonl` (10% Master, repo-disjoint, 3,412 samples) | Overall performance across contemporary Python vulns |
| **Benchmark 2** *(aux)* | Gold CVEFixes re-check | test restricted to `data_source=="cvefixes"` | Gold-only sanity |
| **Benchmark 3** | Held-Out OOD (Zero-Shot) | `PyCode_Vul-test-set.csv` (final) + `PyCode_Vul-train-set.csv` (diagnostic) | Generalization to foreign corpus |

External PyCode-Vul is **evaluation-only**: never used for training, checkpoint selection, or splits. It is read via `scripts/evaluation/evaluate_external.py` (tokenizes on the fly).

---

## 2. Commands

```powershell
# Benchmark 1 — unified in-domain test (reports 3 tasks)
# graph/fusion need the master graph file; semantic_only does not
python scripts/evaluation/evaluate.py --checkpoint models/checkpoints/best.pt
python scripts/evaluation/evaluate.py --checkpoint models/checkpoints/best.pt --graph-data data/processed/master_graphs.jsonl
# Output: outputs/metrics/evaluation_report.json  (metrics.binary/cwe/severity)

# Benchmark 3 — OOD external (semantic_only checkpoints; PyCode-Vul has no graphs)
python scripts/evaluation/evaluate_external.py --checkpoint models/checkpoints/best.pt --split test
python scripts/evaluation/evaluate_external.py --checkpoint models/checkpoints/best.pt --split train
```

> A `fusion`/`graph_only` checkpoint cannot produce graph features for code without graphs; to claim external generalization for fusion, evaluate its semantic branch (or add PyCode-Vul graph extraction as an extension).

---

## 3. Required Metrics

| Task | Metrics (primary **bold**) | Threshold / Masking | Implementation |
|---|---|---|---|
| **Binary** | Accuracy, Precision, Recall, **F1**, ROC-AUC, PR-AUC, **MCC** | 0.5 fixed | `src/utils/metrics.py::binary_metrics` |
| **CWE** | **macro-F1**, weighted/micro F1, per-class F1+support (10 classes) | — | `multiclass_metrics` |
| **Severity** | accuracy & F1 over 4 tiers (labeled samples only) | mask UNKNOWN (-1) | `evaluate.py` severity block |

All 3 trainable tasks land in `outputs/metrics/evaluation_report.json` under `metrics.{binary,cwe,severity}` with identical schema across branches for auto table generation.

---

## 4. Comparison Strategy (RQs 1, 3)

Report all three branches on the same metric set over Benchmark 1 and Benchmark 3:

| Branch | Val F1 (bin) | Test F1 (bin) | Test MCC | CWE macro-F1 | PyCode-Vul Test F1 |
|---|---|---|---|---|---|
| `semantic_only` | … | … | … | … | … |
| `graph_only` | … | … | … | … | … |
| `fusion` | … | … | … | … | … |

---

## 5. Ablation Studies

| Exp | Configuration | Purpose |
|---|---|---|
| A1 / A2 / A3 | semantic_only / graph_only / fusion | modality value (RQ1) |
| A4 | fusion ∖ DFG / ∖ CFG | which graph relation matters |
| A5 | concat vs gated vs cross-attn | validate attention fusion (RQ2) |
| A6 *(P4)* | with vs without `quality_tier` weights | value of quality-aware loss |

All on the same Master splits/seed (42).

---

## 6. Statistical Significance & Error Analysis

- Report mean±std over ≥3 seeds (42/43/44) for fusion vs best baseline; paired bootstrap CI or Wilcoxon signed-rank, p<0.05 — now for **binary, CWE macro-F1**.
- Confusion matrix across CWE classes.

---

## 7. Reproducibility & Reporting

Each report (`outputs/metrics/*.json`, `training_history.json`) records checkpoint, mode, dataset path, threshold, seed, environment, and the full 3-task `metrics` dict. Schema is stable across branches.
