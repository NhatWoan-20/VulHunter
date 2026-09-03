# 06 — Evaluation & Ablation: Multi-Tier Benchmark (Pillar 3)

> **Version: 3.3 (2026-08-30)** — **5 trainable tasks + post-hoc explanation**
> **Authoritative Specification**

---

## 1. Multi-Tier Benchmark Protocol (v3.3)

Evaluation follows a **frozen-checkpoint** protocol in fixed order:

| Tier | Benchmark | Data | Goal | v3.3 extension |
|---|---|---|---|---|
| **Benchmark 1** | Unified In-Domain Test | `data/splits/test.jsonl` (10% Master, repo-disjoint, 3,412 samples) | Overall performance across contemporary Python vulns | + **localization** & **source/sink** metrics |
| **Benchmark 2** *(aux)* | Gold CVEFixes re-check | test restricted to `data_source=="cvefixes"` | Gold-only sanity | + localization on gold subset |
| **Benchmark 3** | Held-Out OOD (Zero-Shot) | `PyCode_Vul-test-set.csv` (final) + `PyCode_Vul-train-set.csv` (diagnostic) | Generalization to foreign corpus | binary/CWE only (no py graphs/weak labels) |
| **Benchmark 4** *(qualitative)* | Explainability audit | Sampled vulnerable predictions → `scripts/explain.py` Markdown reports | Faithfulness & actionability of remediation | NEW v3.3 — human/LLM judge on grounded lines & patch |

External PyCode-Vul is **evaluation-only**: never used for training, checkpoint selection, or splits. It is read via `scripts/evaluation/evaluate_external.py` (tokenizes on the fly, default Qwen2.5-Coder).

---

## 2. Commands (v3.3)

```powershell
# Benchmark 1 — unified in-domain test (now reports 5 tasks)
# graph/fusion need the master graph file; semantic_only does not
python scripts/evaluation/evaluate.py --checkpoint models/checkpoints/best.pt
python scripts/evaluation/evaluate.py --checkpoint models/checkpoints/best.pt --graph-data data/processed/master_graphs.jsonl
# Output: outputs/metrics/evaluation_report.json  (metrics.binary/cwe/severity/localization/source_sink)

# Benchmark 4 — explainability (post-hoc, no training)
# Ad-hoc
python scripts/explain.py --code-file app.py --cwe CWE-89 --severity HIGH --output report.md
# From a checkpoint prediction (auto-fills vulnerable lines + taint)
python scripts/explain.py --checkpoint models/checkpoints/best.pt --sample-id cvefixes:98919200308f75a4:vulnerable --test-data data/splits/test.jsonl --output report.md
# With LLM polish
python scripts/explain.py --code "def f(x): eval(x)" --cwe CWE-94 --severity CRITICAL --use-llm --model Qwen/Qwen2.5-Coder-3B-Instruct

# Benchmark 3 — OOD external (semantic_only checkpoints; PyCode-Vul has no graphs)
python scripts/evaluation/evaluate_external.py --checkpoint models/checkpoints/best.pt --split test
python scripts/evaluation/evaluate_external.py --checkpoint models/checkpoints/best.pt --split train
```

> A `fusion`/`graph_only` checkpoint cannot produce graph features for code without graphs; to claim external generalization for fusion, evaluate its semantic branch (or add PyCode-Vul graph extraction as an extension).

---

## 3. Required Metrics (v3.3)

| Task | Metrics (primary **bold**) | Threshold / Masking | Implementation |
|---|---|---|---|
| **Binary** | Accuracy, Precision, Recall, **F1**, ROC-AUC, PR-AUC, **MCC** | 0.5 fixed | `src/utils/metrics.py::binary_metrics` |
| **CWE** | **macro-F1**, weighted/micro F1, per-class F1+support (10 classes) | — | `multiclass_metrics` |
| **Severity** | accuracy & F1 over 4 tiers (labeled samples only) | mask UNKNOWN (-1) | `evaluate.py` severity block |
| **Localization** ★ v3.3 | **line-level P/R/F1**, Top-k accuracy (k=5), IoU | per-token logits **max-pooled by `token_line_ids`** vs `line_labels` | `localization_metrics` + `train.py`/`evaluate.py` max-pool decoding |
| **Source/Sink** ★ v3.3 | **per-class F1** (Normal/Source/Sink), macro-F1, support (token-level) | ignore_index=-1 (special/pad) | `multiclass_metrics` over `source_sink_labels` in `evaluate.py` |
| **Explanation** ★ v3.3 | Qualitative: groundedness (cite correct lines), patch correctness, faithfulness; optional LLM fluency | post-hoc | `src/explainability/` + `scripts/explain.py` |

All 5 trainable tasks land in `outputs/metrics/evaluation_report.json` under `metrics.{binary,cwe,severity,localization,source_sink}` with identical schema across branches for auto table generation. `training_history.json` logs per-epoch `val_metrics.localization` alongside binary.

**Localization scoring detail:** predictions are per-token `sigmoid(logit)`; for each valid line (`line_labels != -1` with ≥1 aligned token) the line score is the **max** token probability in that line (threshold 0.5). `localization_metrics` then computes line-level P/R/F1 and Top-k hit.

**Source/Sink scoring detail:** token-level `argmax` over 3 classes, scored only on tokens with `source_sink_labels != -1` and `attention_mask==1`; reported as multiclass per-class F1.

---

## 4. Comparison Strategy (RQs 1, 3, 5)

Report all three branches on the same metric set over Benchmark 1 and Benchmark 3:

| Branch | Val F1 (bin) | Test F1 (bin) | Test MCC | CWE macro-F1 | Loc F1 | S/S macro-F1 | PyCode-Vul Test F1 |
|---|---|---|---|---|---|---|---|
| `semantic_only` | … | … | … | … | … | … | … |
| `graph_only` | … | … | … | … | … | … | … |
| `fusion` | … | … | … | … | … | … | … |

RQ5 (v3.3) is answered by the Loc and S/S columns; RQ6 by the explanation audit.

---

## 5. Ablation Studies (v3.3)

| Exp | Configuration | Purpose |
|---|---|---|
| A1 / A2 / A3 | semantic_only / graph_only / fusion | modality value (RQ1) |
| A4 | fusion ∖ DFG / ∖ CFG | which graph relation matters |
| A5 | concat vs gated vs cross-attn | validate attention fusion (RQ2) |
| A6 *(P4)* | with vs without `quality_tier` weights | value of quality-aware loss |
| A7 *(v3.3)* | **λ_loc 0.0 vs 0.4** | value of localization supervision (RQ5) |
| A8 *(v3.3)* | **λ_ss 0.0 vs 0.15** | value of weak taint supervision (RQ5) |
| A9 *(v3.3)* | **explanation template vs +LLM polish** | explanation quality (RQ6) |

All on the same Master splits/seed (42).

---

## 6. Statistical Significance & Error Analysis

- Report mean±std over ≥3 seeds (42/43/44) for fusion vs best baseline; paired bootstrap CI or Wilcoxon signed-rank, p<0.05 — now for **binary, CWE macro-F1, and Loc F1**.
- Confusion matrix across CWE classes; FP (safe code resembling a sink e.g. `cursor.execute` with params) and FN (multi-hop taint / truncation) per branch.
- For localization: line-level confusion + qualitative examples of max-pool hits/misses; for source/sink: per-class precision on Source vs Sink lexicon hits.

---

## 7. Reproducibility & Reporting

Each report (`outputs/metrics/*.json`, `training_history.json`) records checkpoint, mode, dataset path, threshold, seed, environment, and the full 5-task `metrics` dict. Schema is stable across branches (v3.3 adds `localization` and `source_sink` keys; old reports without them remain valid).
