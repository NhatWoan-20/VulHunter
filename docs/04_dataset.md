# 04 — Data Engineering & Splitting: The Master Dataset Strategy

> **Version: 5.0** — Binary Classification Focus
> **Authoritative Specification**

This document defines the unified training corpus: a single **Master Dataset** built by
merging the gold **CVEFixes** corpus with the silver **GHSA** corpus, expanded per-role,
and split repository-disjoint.

---

## 1. Data Sources

### 1.1 CVEFixes (Gold Tier)

- **Source:** [secureIT-project/CVEfixes](https://github.com/secureIT-project/CVEfixes) (Zenodo DOI: `10.5281/zenodo.13118970`)
- **Format:** SQLite database extracted via `scripts/extraction/extract.py`
- **Output:** `data/raw/python_cvefixes_methods.jsonl`
- **Content:** Python function-level vulnerability-fixing pairs from CVEs
- **Quality:** Human-reviewed, high-quality (weight = 1.0)

### 1.2 GitHub Security Advisories (Silver Tier)

- **Source:** GitHub Advisory Database via GraphQL API
- **Extraction:** `scripts/collection/fetch_advisories.py` + `scripts/collection/extract_functions.py`
- **Output:** `data/raw/ghsa/ghsa_methods.jsonl`
- **Content:** Python function-level vulnerability-fixing pairs from GHSA
- **Quality:** Automatically extracted (weight = 0.85)

---

## 2. Preprocessing Pipeline

### 2.1 End-to-End Pipeline Orchestrator

```bash
# Full pipeline (recommended)
python scripts/preprocessing/run_pipeline.py

# Skip graph extraction (for semantic_only mode)
python scripts/preprocessing/run_pipeline.py --skip-graph

# Skip tokenization (for graph_only mode)
python scripts/preprocessing/run_pipeline.py --skip-tokenize
```

### 2.2 Pipeline Steps

```
data/raw/python_cvefixes_methods.jsonl ─┐
                                        ├─ prepare_master.py  (unify schema, noise filter)
data/raw/ghsa/ghsa_methods.jsonl ─────────┘
  ▼ data/raw/master_methods.jsonl          (15,351 pairs)
clean_comments.py                          (remove code comments)
normalize.py                              (normalize whitespace/indentation)
validate_ast.py                           (verify valid Python AST)
build_samples.py                           (pair → vulnerable + safe role)
  ▼ data/processed/master_semantic_samples.jsonl       (Docstrings retained for Semantic)
strip_docstrings.py                       (remove docstrings)
  ▼ data/processed/master_graph_samples.jsonl (For Graph Branch)
build_ast.py → build_cfg.py → build_dfg.py → build_call.py → merge_graphs.py
  ▼ data/processed/master_graphs.jsonl    (30,427 heterogeneous graphs)
tokenization (on-the-fly)                         (CodeBERT tokenization)
split.py                                 (80/10/10 repo-disjoint)
  ▼ data/splits/{train,validation,test}.jsonl
```

---

## 3. Noise Filtering & Quality Tiers

### 3.1 Strict Noise & Test-Code Cleansing

GHSA fix commits contain non-application code; such methods are dropped:

| Class | File Patterns Removed |
|---|---|
| Test / mock | `tests/`, `test_`, `testing/`, `mocks/`, `conftest.py`, `*_test.py`, `test.py`, `*_spec.py` |
| Build / config | `setup.py`, `fabfile.py`, `tasks.py` |

Applied during `prepare_master.py`: **4,683 GHSA methods removed** (27.5%).
CVEFixes gold intact (already reviewed).



---

## 4. Schema & Sample Format

### 4.1 Master Record Schema


```json
{
  "sample_id": "cvefixes:98919200308f75a4",
  "repository": "irmen/pyro3",
  "code": "def __init__(self, pidfile=None):\n    if not pidfile:\n        self.pidfile = \"/tmp/%s.pid\" % self.__class__.__name__.lower()\n    else:\n        self.pidfile = pidfile",
  "safe_code": "def __init__(self, pidfile=None):\n    if not pidfile:\n        self.pidfile = \"/var/run/pyro-%s.pid\" % self.__class__.__name__.lower()\n    else:\n        self.pidfile = pidfile",
  "binary_label": 1,
}
```

### 4.2 Per-Role Sample Schema

After `build_samples.py`, each pair expands into two samples:

```json
{
  "sample_id": "cvefixes:98919200308f75a4:vulnerable",
  "code": "def __init__(self, pidfile=None):\n    ...",
  "binary_label": 1,
  "input_ids": [151643, 29871, ...],
  "attention_mask": [1, 1, ...]
}
```

- `sample_id` namespaced by source and role → globally unique
- `binary_label`: 1 = vulnerable, 0 = safe

---

## 5. Repository-Disjoint Splitting (Pillar 2)

- **Group by canonical repository** across both GHSA and CVEFixes.
- **80 / 10 / 10** split, repos shuffled with `seed 42`.
- No repo straddles Train/Val/Test — prevents leakage of project-specific patterns.
- **Leakage check:** `split.py` computes pairwise repo overlap and warns on any violation.

### Split Statistics

| Split | Samples | CVEFixes | GHSA |
|---|---|---|---|
| train | 20,638 | 4,848 | 15,790 |
| validation | 6,404 | 322 | 6,082 |
| test | 3,412 | 626 | 2,786 |

Binary labels are balanced within every split (per-role vulnerable+safe twins).

---

## 6. External Benchmark (PyCode-Vul)

| Dataset | Purpose | Use |
|---|---|---|
| PyCode-Vul train (14,248) | Diagnostic OOD evaluation | Never in `data/splits/` |
| PyCode-Vul test (3,563) | Final OOD benchmark | Evaluation only |

PyCode-Vul is **evaluation-only**: never used for training or checkpoint selection.






