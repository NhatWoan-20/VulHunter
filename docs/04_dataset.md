# 04 — Data Engineering & Splitting: The 5-Pillar Master Dataset Strategy

> **Version: 3.3 (2026-08-30)** — Master 1-Stage, **6/6 Tasks, 2 new token fields**
> **Authoritative Specification**

This document defines the unified training corpus: a single **Master Dataset** built by
merging the gold **CVEFixes** corpus with the silver **GHSA** corpus, expanded per-role,
split repository-disjoint, and enriched in **v3.3** with `token_line_ids_qwen` (for
line-level localization) and `source_sink_labels` (for weak taint supervision).

---

## PILLAR 1 — Data Engineering: Cleansing & Normalization

### 1.1 GHSA optimal line-label generation

`scripts/extraction/prepare_master.py` labels GHSA vulnerable lines robustly against
**comment / whitespace-only** diffs:

1. **Normalize before diffing:** dedent, strip trailing whitespace, and remove `COMMENT`
   tokens via tokenizer-based strip (`_strip_comments_preserve_lines`) that **preserves line
   count** — prevents a comment/indentation-only line from being mislabeled.
2. **Label rules** (`ghsa_line_labels`):
   - `delete` / `replace` opcodes → label `1`.
   - `equal` lines → label `0`.
   - Fix that only **inserted** lines → label the **adjacent previous line** (activation context).
3. Normalized lines map 1:1 to `code.splitlines()`, so `vulnerable_lines`/`line_labels` are
   exact for the vulnerable variant.

### 1.2 Strict noise & test-code cleansing

GHSA fix commits contain non-application code; such methods are dropped:

| Class | File patterns removed |
|---|---|
| Test / mock | `tests/`, `test_`, `testing/`, `mocks/`, `conftest.py`, `*_test.py`, `test.py`, `*_spec.py` |
| Build/config | `setup.py`, `fabfile.py`, `tasks.py` |

Applied to GHSA silver during `prepare_master.py`: **4,683 GHSA methods removed** (27.5%).
CVEFixes gold intact (already reviewed).

### 1.3 Schema unification & quality tiers

Every Master pair uses one canonical schema and carries `quality_tier`:

```json
{
  "sample_id": "ghsa:e083958cf1c6cf57",
  "pair_id": "ghsa:e083958cf1c6cf57",
  "data_source": "ghsa",
  "quality_tier": "silver",
  "cve_id": "CVE-2026-55558",
  "ghsa_id": "GHSA-vxj7-4xrp-5vr4",
  "repository": "cole/aiosmtplib",
  "sha": "...",
  "file_path": "src/aiosmtplib/protocol.py",
  "function_name": "start_tls",
  "code": "...",
  "safe_code": "...",
  "binary_label": 1,
  "severity": "MODERATE",
  "cwe_ids": ["CWE-74"],
  "line_labels": [0, 0, 1],
  "vulnerable_lines": [3]
}
```

- `sample_id` namespaced by source (`cvefixes:{id}` / `ghsa:{id}`) → globally unique;
  per-role ids become `{sample_id}:{role}` (`:vulnerable` / `:safe`) in `build_samples.py`.
- **CVEFixes → `gold` (w=1.0), GHSA → `silver` (w=0.85)**.
- `severity` `NAN` → `UNKNOWN` (masked, label `-1`).
- `repository` canonicalized to lowercase `owner/name` for cross-source grouping.

### 1.4 v3.3 token-level enrichment (post-split, in-place on `data/splits/*.jsonl`)

After `build_samples.py` + `split.py`, two preprocessing steps materialize new fields
**in place** on every split file (and as a mirror `data/tokenized/sem_qwen.jsonl`):

| Step | Script | New field(s) | How | Consumers |
|---|---|---|---|---|
| **Token→line alignment** | `scripts/preprocessing/tokenize_qwen.py` (v3.3) | `token_line_ids_qwen: int[]` (−1 = special/pad, else 0-indexed line id) + `offset_mapping_qwen: [int,int][]` + `tokens_qwen` | `return_offsets_mapping=True` on Qwen2.5-Coder; `token_line_ids = bisect(line_starts, offset_start)-1` per token (see `src/utils/dataset.py::_line_starts`) | `VulHunterDataset` → `collate_fn` pads to `(B, L)` → `MultiTaskLoss._pool_tokens_to_lines` (max-pool) → localization head |
| **Weak taint labels** | `scripts/preprocessing/generate_source_sink_labels.py` (NEW v3.3) | `source_sink_labels: int[]` (len == sequence, values 0 Normal / 1 Source / 2 Sink / −1 ignore-special) | Lexicon `src/utils/taint.py` (`SOURCE_SUBSTRINGS` / `SINK_SUBSTRINGS`; Sink > Source > Normal; safe samples → all 0) + line→token propagation via `token_line_ids` | `VulHunterDataset` → `collate_fn` → `MultiTaskLoss` CE (λ=0.15) + `evaluate.py` per-class taint metrics |

Example per-role record after v3.3:

```json
{
  "sample_id": "ghsa:e0839...:vulnerable",
  "code": "def f(x):\n    q = \"select * where id='\"+x+\"'\"\n    cursor.execute(q)\n",
  "line_labels": [0, 1, 1],
  "input_ids_qwen": [151643,  ...],
  "attention_mask_qwen": [1, 1, ...],
  "token_line_ids_qwen": [-1, 0, 0, 1, 1, 1, 1, 2, 2, -1],
  "offset_mapping_qwen": [[0,0],[0,3],[3,4], ...],
  "source_sink_labels": [-1, 0, 1, 0, 0, 0, 2, 2, -1],
  "quality_tier": "silver"
}
```

**Idempotence & backward compat:** both scripts re-derive deterministically; re-running after any
re-tokenization restores consistency. `VulHunterDataset` also **re-derives on the fly** if the
fields are absent (from `offset_mapping_qwen` + lexicon), but localization/source_sink are then
skipped for that sample (loss weight effectively 0) until re-tokenized — so old checkpoints and
old splits remain loadable.

---

## PILLAR 2 — Data Integrity: Cross-Dataset Repository-Disjoint Splitting

- **Group by canonical repository** across both GHSA and CVEFixes (verified: **232 shared repos**).
  No repo straddles Train/Val/Test — the attack *"repo X in GHSA→Train but X in CVEFixes→Test"*
  is impossible.
- **80 / 10 / 10**, repos shuffled with `seed 42`.
- **Leakage check:** `split.py` computes pairwise repo overlap and warns on any violation (currently 0).

Live result on the Master corpus (30,454 per-role samples):

| Split | Samples | Composition (cvefixes + ghsa, per-role) |
|---|---|---|
| train | 20,638 | 4,848 + 15,790 |
| validation | 6,404 | 322 + 6,082 |
| test | 3,412 | 626 + 2,786 |

Binary labels are 50/50 within every split (per-role vulnerable+safe twins).

---

## PILLAR 3 — Multi-Tier Benchmark (see `06_evaluation.md`)

| Tier | Benchmark | Source | Goal | v3.3 extension |
|---|---|---|---|---|
| **Benchmark 1** | Unified In-Domain Test | 10% Master test (repo-disjoint) | Overall performance across contemporary Python vulns | + line localization & source/sink metrics |
| **Benchmark 2** *(aux)* | Gold re-check | test restricted to `data_source=="cvefixes"` | Gold-only sanity | + localization on gold subset |
| **Benchmark 3** | Held-Out OOD (Zero-Shot) | PyCode-Vul test & train CSVs | Generalization | binary/CWE only (no py graphs/labels) |

External PyCode-Vul remains evaluation-only; it does not produce graphs or weak taint labels.

---

## 4. Master Pipeline (v3.3 path & artifact contract)

```
data/raw/python_cvefixes_methods.jsonl ─┐
                                        ├─ prepare_master.py  (Pillar 1: labels, noise-filter, schema, tiers)
data/raw/ghsa/ghsa_methods.jsonl ───────┘
  ▼ data/raw/master_methods.jsonl          (15,351 pairs)
clean_comments → normalize → validate_ast → strip_docstrings
  ▼ data/processed/master_graph_input.jsonl
build_samples.py                           (pair → vulnerable + safe role, per-role sample_id)
  ▼ data/final/master_samples.jsonl        (30,454)
split.py --cross-project --seed 42         (Pillar 2: repo-disjoint)
  ▼ data/splits/{train,validation,test}.jsonl   (20,638 / 6,404 / 3,412)
tokenize_qwen.py  ★ v3.3                   (in-place input_ids_qwen + token_line_ids_qwen + offset_mapping_qwen)
  │  mirror → data/tokenized/sem_qwen.jsonl + reports/preprocessing/tokenize_qwen.json
generate_source_sink_labels.py  ★ NEW v3.3 (in-place source_sink_labels + reports/preprocessing/source_sink.json)
  ▼
build_{ast,cfg,dfg,call}.py → merge_graphs.py
  ▼ data/processed/master_graphs.jsonl     (30,427 heterogeneous graphs, keyed by sample_id)
```

> Pipeline is fixed to `master` — just `python scripts/preprocessing/<script>.py`. Each script is hardcoded to `master_*` paths; to target a different dataset, edit the `INPUT`/`OUTPUT` constants at the top of the script directly.
> **After upgrading to v3.3:** re-run `tokenize_qwen.py` then `generate_source_sink_labels.py` once — old splits stay loadable.

---

## 5. Label Mappings (single source of truth)

### 5.1 CWE classes (`src/utils/dataset.py` — `CWE_CLASSES`)

| Idx | Class | Description |
|---|---|---|
| 0 | `none` | Safe / no CWE |
| 1–8 | `CWE-22,78,79,89,94,502,918,327` | 8 target CWEs |
| 9 | `CWE-Other` | Everything else |

`num_cwe_classes = 10` throughout (model, head, loss).

### 5.2 Severity (`SEVERITY_CLASSES`)

| String | Index |
|---|---|
| `UNKNOWN`/`NAN` | −1 (masked) |
| `LOW` / `MODERATE`+`MEDIUM` / `HIGH` / `CRITICAL` | 0..3 |

### 5.3 Quality tiers (`src/utils/losses.py::QUALITY_TIER_WEIGHTS`)

| Tier | Source | w_tier |
|---|---|---|
| `gold` | CVEFixes | 1.00 |
| `silver` | GHSA | 0.85 |

### 5.4 Taint classes (`src/utils/taint.py`)

| Idx | Tag | Meaning | Supervision |
|---|---|---|---|
| 0 | Normal | not on taint path | weak lexicon negative |
| 1 | Source | untrusted entry (`request.args`, `input(`, `os.environ`, …) | `SOURCE_SUBSTRINGS` |
| 2 | Sink | dangerous consumption (`cursor.execute`, `os.system`, `eval(`, `pickle.loads`, …) | `SINK_SUBSTRINGS` (priority over Source) |
| −1 | Ignore | special/pad token | never scored |

---

## 6. Quality Assurance

- Zero duplicated `sample_id` in `data/final/master_samples.jsonl` (30,454 unique, verified).
- Strictly disjoint repo sets across splits (verified via `split.py` report).
- 100% of samples carry `line_labels`; 100% carry `input_ids_qwen` after tokenize.
- **v3.3:** 100% carry `token_line_ids_qwen` after `tokenize_qwen.py`; 100% carry `source_sink_labels` after `generate_source_sink_labels.py` (safe → all-Normal, vulnerable → lexicon-derived).
- Graphs: 30,427 merged heterogeneous graphs; 0 missing types.
- Render order: `prepare_master → preprocessing → build_samples → split → tokenize_qwen → generate_source_sink_labels → graphs`.
