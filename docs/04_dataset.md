# 04 — Data Engineering & Splitting: The Master Dataset Strategy

> **Version: 6.0** — Unified Single-Sample & Multi-Source Focus
> **Authoritative Specification**

This document defines the unified training corpus: a single **Master Dataset** built by
merging **CVEFixes**, **GHSA**, and **Large Scale Datasets (Benign & Vulnerable)** into
a unified single-sample schema, cleansed, deduplicated, and split repository-disjoint.

---

## 1. Data Sources

### 1.1 CVEFixes (Paired)
- **Source:** [secureIT-project/CVEfixes](https://github.com/secureIT-project/CVEfixes) (Zenodo DOI: `10.5281/zenodo.13118970`)
- **Format:** Paired records (`code` + `safe_code`)
- **Location:** `data/raw/python_cvefixes_methods.jsonl` (~2,985 pairs)
- **Processing:** Exploded into vulnerable (`binary_label=1`) and safe (`binary_label=0`) records.

### 1.2 GitHub Security Advisories (GHSA, Paired)
- **Source:** GitHub Advisory Database via GraphQL API
- **Format:** Paired records (`code` + `safe_code`)
- **Location:** `data/raw/ghsa/ghsa_methods.jsonl` (~17,049 pairs)
- **Processing:** Exploded into vulnerable (`binary_label=1`) and safe (`binary_label=0`) records.

### 1.3 Large Scale Benign Dataset (Single-Sample)
- **Source:** Diverse non-vulnerable open-source Python repositories
- **Format:** Single-sample (`code`, `label=0`)
- **Location:** `data/raw/large/benign_samples.jsonl` (~90,000 samples)

### 1.4 Large Scale Vulnerable Dataset (Single-Sample)
- **Source:** Verified CVE / GHSA fix commits
- **Format:** Single-sample (`code`, `label=1`)
- **Location:** `data/raw/large/vulnerable_samples.jsonl` (~15,000 samples)

---

## 2. Preprocessing Pipeline

### 2.1 End-to-End Pipeline Orchestrator

```bash
# Full pipeline (recommended)
python scripts/preprocessing/run_pipeline.py

# Skip graph extraction (for semantic_only mode)
python scripts/preprocessing/run_pipeline.py --skip-graph

# Dry-run merge check only
python scripts/preprocessing/run_pipeline.py --dry-run-merge
```

### 2.2 Pipeline Steps

```
data/raw/python_cvefixes_methods.jsonl ─┐
data/raw/ghsa/ghsa_methods.jsonl ───────┼─ merge_all_sources.py (explode pairs, noise filter, SHA-1 dedup)
data/raw/large/benign_samples.jsonl ────┤
data/raw/large/vulnerable_samples.jsonl ─┘
  ▼ data/raw/master_samples.jsonl          (~101,414 records)
clean_comments.py                          (remove inline comments)
  ▼ data/processed/master_cleaned.jsonl
normalize.py                              (standardize whitespace, CRLF->LF, dedent)
  ▼ data/processed/master_normalized.jsonl
validate_ast.py                           (verify valid Python AST via 4 strategies)
  ▼ data/processed/master_validated.jsonl  (101,336 valid records)
build_samples.py                          (canonicalize schema, ensure unique sample_id)
  ▼ data/final/master_semantic_samples.jsonl
split.py                                  (80/10/10 repository-disjoint, seed 42)
  ▼ data/splits/{train,validation,test}.jsonl

[Branch: Graph / RGCN]
validate_ast.py
  ▼ data/processed/master_validated.jsonl
strip_docstrings.py                       (strip docstrings via AST NodeTransformer)
  ▼ data/processed/master_graph_ready.jsonl
build_pdg.py                              (construct PDG graph with AST, CFG, DFG, Call edges)
  ▼ data/final/master_pdg.jsonl
```

---

## 3. Cleansing & Deduplication

### 3.1 Strict Noise & Test-Code Cleansing
Files matching test/mock/build scaffolding patterns are removed across all sources:
- **Test / mock:** `tests/`, `test_`, `testing/`, `mocks/`, `conftest.py`, `*_test.py`, `test.py`, `*_spec.py`
- **Build / config:** `setup.py`, `fabfile.py`, `tasks.py`

### 3.2 Global SHA-1 Fingerprint Deduplication
To prevent data leakage between train and test sets, every code snippet is deduplicated globally:
$$\text{fp} = \text{SHA1}(\text{" ".join}(\text{code.split}()))$$
- Duplicate implementations across commits, trivial patches (`code == safe_code`), and overlapping CVEs between CVEFixes, GHSA, and Large datasets are strictly removed.

---

## 4. Schema & Sample Format

### 4.1 Master Record Schema (`data/raw/master_samples.jsonl`)
```json
{
  "sample_id": "cvefixes:98919200308f75a4:vuln",
  "repository": "irmen/pyro3",
  "code": "def __init__(self, pidfile=None):\n    ...",
  "binary_label": 1,
  "data_source": "cvefixes",
  "cve_id": "CVE-2011-2765",
  "ghsa_id": null,
  "severity": "HIGH",
  "cwe_ids": ["CWE-377"],
  "file": "Pyro/util.py",
  "function": "__init__"
}
```

### 4.2 Training-Ready Schema (`data/splits/{train,validation,test}.jsonl`)
```json
{
  "sample_id": "cvefixes:98919200308f75a4:vuln",
  "code": "def __init__(self, pidfile=None):\n    ...",
  "binary_label": 1
}
```

---

## 5. Repository-Disjoint Splitting (Pillar 2)

- **Group by canonical repository** (`owner/name`).
- Samples without repository (`unknown`) are safely assigned to `train` to prevent test-set leakage.
- **80 / 10 / 10** split with fixed `seed 42`.
- Pairwise repo overlap verification confirms **zero repository leakage** across all splits.

### Split Statistics

| Split | Total Samples | Vulnerable (1) | Safe (0) | Ratio (Safe:Vuln) |
|---|---|---|---|---|
| **Train** | 86,114 | 18,383 | 67,731 | ~3.7 : 1 |
| **Validation** | 8,446 | 2,445 | 6,001 | ~2.5 : 1 |
| **Test** | 6,776 | 1,872 | 4,904 | ~2.6 : 1 |
| **Total** | **101,336** | **22,700** | **78,636** | ~3.5 : 1 |

> [!TIP]
> Notice the class imbalance (~3.5:1 safe to vulnerable). When training, use `focal_loss` with appropriate `alpha` or weighted loss (`class_weight`) to optimize balanced precision and recall.

