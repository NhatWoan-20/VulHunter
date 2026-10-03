# Data Extraction & Master Dataset Preparation

> **Objective:** Unify paired sources (CVEFixes, GHSA) and large-scale single-sample datasets (benign, vulnerable) into a single canonical Master Dataset (`master_samples.jsonl`).

This directory handles merging all raw sources into a unified single-sample format with global SHA-1 fingerprint deduplication and noise cleansing.

## Workflow

```mermaid
flowchart TD
    A[python_cvefixes_methods.jsonl<br/>(2,985 pairs)] -->|Explode to singles| M{merge_all_sources.py}
    B[ghsa_methods.jsonl<br/>(17,049 pairs)] -->|Explode to singles| M
    C[large/benign_samples.jsonl<br/>(90,000 singles)] -->|Canonicalize| M
    D[large/vulnerable_samples.jsonl<br/>(15,000 singles)] -->|Canonicalize| M
    M -->|Noise Filter + Global SHA-1 Dedup| E[data/raw/master_samples.jsonl<br/>(~101k samples)]
```

## Files Description

- **`merge_all_sources.py`**: The authoritative dataset unification script. It:
  1. Explodes paired records (`code` + `safe_code`) into individual vulnerable (`binary_label=1`) and safe (`binary_label=0`) records.
  2. Ingests pre-single large benign and vulnerable records.
  3. Filters noise paths (test, mock, setup scaffolding).
  4. Applies global SHA-1 fingerprint deduplication on normalized code whitespace.
  5. Enforces valid length constraints (`20 <= len(code) <= 50,000`).

## Input / Output

- **Inputs**:
  - `data/raw/python_cvefixes_methods.jsonl`
  - `data/raw/ghsa/ghsa_methods.jsonl`
  - `data/raw/large/benign_samples.jsonl`
  - `data/raw/large/vulnerable_samples.jsonl`
- **Output**:
  - `data/raw/master_samples.jsonl`: The unified Master Dataset (~101,414 single-sample records).

## How to Run

```bash
# Run merge directly
python scripts/extraction/merge_all_sources.py

# Or dry-run to inspect statistics without writing
python scripts/extraction/merge_all_sources.py --dry-run
```
