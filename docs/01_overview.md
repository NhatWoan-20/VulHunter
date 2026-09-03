# 01 — Project Overview & Research Objectives

> **Version: 3.3 (2026-08-30)** — **6/6 Tasks Active**
> **Authoritative Specification**

---

## 1. Problem Statement

Vulnerability detection in software source code is critical for cybersecurity. While modern static analysis tools (SAST) often suffer from high false positive rates and rule maintenance overhead, deep learning approaches offer promising automated detection capabilities.

In Python source code, vulnerabilities often stem from subtle semantic interactions (e.g., dynamic typing, dangerous built-ins like `eval`/`exec`, deserialization bugs) and complex data/control flow dependencies (e.g., taint propagation from request parameters to SQL queries).

**VulHunter** addresses this by combining:
1. **Semantic representations** from Code Large Language Models (capturing token-level context, API semantics, identifier naming).
2. **Structural representations** from Program Graphs (AST, CFG, DFG, Call Graph, capturing execution flow and data dependencies).
3. **Remediation intelligence** — precise line highlights, taint source/sink traces, and natural-language fix guidance — on top of detection.

---

## 2. Research Questions & Hypotheses

### Research Questions (RQs)

- **RQ1 (Modality Value):** Does combining semantic representations with structural program graphs outperform either modality alone (Semantic-only vs. Graph-only vs. Fusion)?
- **RQ2 (Cross-Modal Fusion):** Does bidirectional cross-attention effectively capture token-to-node structural alignments better than simple concatenation or early fusion?
- **RQ3 (Multi-Task Synergy):** Does auxiliary multi-task learning (CWE, severity, localization, taint) improve the generalization of binary vulnerability detection?
- **RQ4 (External Generalization):** Can a model trained on the repository-disjoint Master Dataset generalize zero-shot to the held-out external PyCode-Vul benchmark?
- **RQ5 (Fine-Grained Localization & Taint — v3.3):** Does token→line max-pool supervision (focal loss, `token_line_ids_qwen`) enable accurate line-level localization, and does weak lexicon supervision for source/sink provide useful taint signals without human taint labels?
- **RQ6 (Explainability — v3.3):** Can LLM post-processing grounded in predicted CWE/severity/lines/taint produce faithful, actionable remediation reports?

### Research Hypotheses (H)

- **H1:** The Cross-Modal Fusion model achieves a higher binary F1-score and MCC compared to both the Semantic-only and Graph-only baselines on repository-disjoint test splits.
- **H2:** Multi-task representation sharing improves minority CWE detection performance compared to isolated task training.
- **H3:** Merging silver GHSA data into the Master corpus with **quality-aware weighting** (0.85) improves CWE macro-F1 and OOD generalization without degrading gold CVEFixes performance.
- **H4 (v3.3):** Jointly supervising **line localization** (λ=0.4, token→line max-pool) and **source/sink** (λ=0.15, weak lexicon) does not harm binary F1 and yields usable fine-grained F1 for triage.
- **H5 (v3.3):** Offline CWE-aware templates already produce correct fix principles; optional LLM polish improves fluency without changing the remediation.

---

## 3. Supported Tasks & Output Schema — 6/6 Active (v3.3)

| # | Task | Input → Output | Supervision |
|---|---|---|---|
| 1 | **Binary Vulnerability Detection (Primary)** | Python function *f* → ŷ∈{0,1} + p∈[0,1] | `binary_label` |
| 2 | **CWE Classification** | → 10 classes (8 target CWEs + `none` + `CWE-Other`) | `cwe_ids` → `CWE_CLASSES` |
| 3 | **Severity Classification** | → Low / Moderate / High / Critical (masked if UNKNOWN) | `severity` → `SEVERITY_CLASSES` |
| 4 | **Line-Level Localization** | → binary vector **l**∈{0,1}^L (vulnerable lines) via per-token logits max-pooled by `token_line_ids_qwen` | `line_labels` (diff-derived) + `token_line_ids_qwen` (offset_mapping→bisect) |
| 5 | **Source/Sink Detection** | → per-token {Normal(0), Source(1), Sink(2)} taint tags | `source_sink_labels` (weak lexicon, `src/utils/taint.py`) |
| 6 | **Natural-Language Explanation** | → Markdown remediation report (root cause, taint flow, exploit, patch) | **post-hoc** — consumes tasks 1–5, no training loss; offline template + optional LLM (`src/explainability/`) |

> Tasks 4–6 were deferred/placeholder in v3.2; they are **fully wired in v3.3** — see `03_architecture.md §2.4`, `05_training.md §3.2`, `07_extensions.md`.

### Dataset Policy Summary (see `04_dataset.md`)

| Dataset | Use |
|---|---|
| CVEFixes (2,985 pairs) | **Gold** half of the Master training corpus (w=1.0) |
| GHSA (12,366 pairs after cleansing) | **Silver** half of the Master training corpus (w=0.85) |
| Master Dataset (`data/raw/master_methods.jsonl`, 15,351 pairs → 30,454 per-role samples) | Train/val/in-domain test (80/10/10 cross-dataset repository-disjoint, seed 42) |
| PyCode-Vul (14,248 / 3,563) | **Evaluation only**, final OOD benchmark, never in `data/splits/` |

---

## 4. Scope & Boundaries

- **Language:** Python source code (Python 3.8+ syntax).
- **Granularity:** Function-level samples (complete function definitions); each pair expands into a vulnerable and a safe role sample.
- **Learning Paradigm:** Supervised multi-task learning (5 trainable heads) + post-hoc explainability.
- **Supervision Rule:** Vulnerable parent version is used for inference; fixed child version provides supervision labels only (diff-derived line labels, weak taint lexicon).
- **Training corpus:** a single Master Dataset (gold CVEFixes + silver GHSA), quality-weighted; PyCode-Vul is excluded from training.

---

## 5. Success Criteria (v3.3)

- **Binary:** Fusion F1 > Semantic-only and Graph-only on both in-domain test and PyCode-Vul OOD.
- **CWE:** Macro-F1 improves with GHSA silver + quality weighting.
- **Localization:** Line-level F1 > 0.30 and Top-5 accuracy reported end-to-end (train + evaluate).
- **Source/Sink:** Per-class Source/Sink F1 reported; weak supervision does not degrade binary F1.
- **Explanation:** Every vulnerable prediction can be rendered as a Markdown report with grounded line numbers and a correct patch principle (template 100%, LLM optional).

---

## 6. Non-Functional Requirements

- **Reproducibility:** Deterministic repository-disjoint splitting (seed 42), fixed thresholds (0.5), version-pinned dependencies.
- **Modularity:** Swappable LLM backbones (Qwen2.5-Coder, DeepSeek-Coder, CodeBERT) and GNN layers (GAT, GCN, Graph Transformer) behind encoder interfaces.
- **Efficiency:** Gradient accumulation (eff. batch 32), FP16/BF16, layer freezing (28/36), and offline explanation — all on 8–24 GB VRAM.
