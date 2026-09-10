# 01 — Project Overview & Research Objectives

> **Version: 4.0** — **3/3 Tasks Active**
> **Authoritative Specification**

---

## 1. Problem Statement

Vulnerability detection in software source code is critical for cybersecurity. While modern static analysis tools (SAST) often suffer from high false positive rates and rule maintenance overhead, deep learning approaches offer promising automated detection capabilities.

In Python source code, vulnerabilities often stem from subtle semantic interactions (e.g., dynamic typing, dangerous built-ins like `eval`/`exec`, deserialization bugs) and complex data/control flow dependencies (e.g., taint propagation from request parameters to SQL queries).

**VulHunter** addresses this by combining:
1. **Semantic representations** from Code Large Language Models (capturing token-level context, API semantics, identifier naming).
2. **Structural representations** from Program Graphs (AST, CFG, DFG, Call Graph, capturing execution flow and data dependencies).

---

## 2. Research Questions & Hypotheses

### Research Questions (RQs)

- **RQ1 (Modality Value):** Does combining semantic representations with structural program graphs outperform either modality alone (Semantic-only vs. Graph-only vs. Fusion)?
- **RQ2 (Cross-Modal Fusion):** Does bidirectional cross-attention effectively capture token-to-node structural alignments better than simple concatenation or early fusion?
- **RQ3 (Multi-Task Synergy):** Does auxiliary multi-task learning (CWE, severity) improve the generalization of binary vulnerability detection?
- **RQ4 (External Generalization):** Can a model trained on the repository-disjoint Master Dataset generalize zero-shot to the held-out external PyCode-Vul benchmark?

### Research Hypotheses (H)

- **H1:** The Cross-Modal Fusion model achieves a higher binary F1-score and MCC compared to both the Semantic-only and Graph-only baselines on repository-disjoint test splits.
- **H2:** Multi-task representation sharing improves minority CWE detection performance compared to isolated task training.
- **H3:** Merging silver GHSA data into the Master corpus with **quality-aware weighting** (0.85) improves CWE macro-F1 and OOD generalization without degrading gold CVEFixes performance.

---

## 3. Supported Tasks & Output Schema — 3/3 Active

| # | Task | Input → Output | Supervision |
|---|---|---|---|
| 1 | **Binary Vulnerability Detection (Primary)** | Python function *f* → ŷ∈{0,1} + p∈[0,1] | `binary_label` |
| 2 | **CWE Classification** | → 10 classes (8 target CWEs + `none` + `CWE-Other`) | `cwe_ids` → `CWE_CLASSES` |
| 3 | **Severity Classification** | → Low / Moderate / High / Critical (masked if UNKNOWN) | `severity` → `SEVERITY_CLASSES` |

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
- **Learning Paradigm:** Supervised multi-task learning (3 trainable heads).
- **Supervision Rule:** Vulnerable parent version is used for inference; fixed child version provides supervision labels only.
- **Training corpus:** a single Master Dataset (gold CVEFixes + silver GHSA), quality-weighted; PyCode-Vul is excluded from training.

---

## 5. Success Criteria

- **Binary:** Fusion F1 > Semantic-only and Graph-only on both in-domain test and PyCode-Vul OOD.
- **CWE:** Macro-F1 improves with GHSA silver + quality weighting.

---

## 6. Non-Functional Requirements

- **Reproducibility:** Deterministic repository-disjoint splitting (seed 42), fixed thresholds (0.5), version-pinned dependencies.
- **Modularity:** Swappable LLM backbones (Qwen2.5-Coder) and GNN layers (GAT, GCN, Graph Transformer) behind encoder interfaces.
- **Efficiency:** Gradient accumulation (eff. batch 32), FP16/BF16, layer freezing (28/36).
