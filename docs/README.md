# VulHunter Research Specification & Documentation

> **Authoritative methodology — v3.3 (2026-08-30)** — Master 1-Stage, **6/6 Tasks Active**

VulHunter is a hybrid multi-modal vulnerability detection framework for Python source code combining semantic code representations (Transformers / Code LLMs) and structural program graphs (Heterogeneous AST / CFG / DFG / Call graphs).

> **Single source of truth:** `docs/` (this folder) is the only authoritative methodology. The legacy v1/v2 research draft and the original full-length specification are archived and **superseded** in `docs/archive/`.

---

## 1. Documentation Map

| Document | Topic | Key Content |
|---|---|---|
| [`01_overview.md`](01_overview.md) | Problem & Objectives | Research questions, scope, 6-task schema, success criteria |
| [`02_literature_review.md`](02_literature_review.md) | State of the Art | Semantic vs Graph vs Hybrid models, literature gaps |
| [`03_architecture.md`](03_architecture.md) | System Architecture | Encoders, Fusion, **5 trainable heads + Explainability** (v3.3) |
| [`04_dataset.md`](04_dataset.md) | Data & Preprocessing | Unified Master dataset + new `token_line_ids_qwen` / `source_sink_labels` fields |
| [`05_training.md`](05_training.md) | Training & Optimization | 5-loss multi-task (binary/CWE/severity/**localization/source_sink**), tiered LR |
| [`06_evaluation.md`](06_evaluation.md) | Evaluation & Ablation | 5-task metrics, explainability report |
| [`07_extensions.md`](07_extensions.md) | Research Extensions | Line localization, source/sink, LLM explanations — **all ACTIVE (v3.3)** |

---

## 2. Dataset Contract & Policy (v3.3 — verified, 6/6 tasks)

| Dataset | Location | Records | Role | Supervision (v3.3) |
|---|---|---|---|---|
| **CVEFixes (gold)** | `data/raw/python_cvefixes_methods.jsonl` | **2,985** pairs (659 CVEs, 361 repos) | Primary high-quality tier | binary, CWE, severity, diff-derived `line_labels` |
| **GHSA (silver)** | `data/raw/ghsa/ghsa_methods.jsonl` | **12,366** kept of 17,049 (4,683 noise removed) | Secondary tier | binary, CWE (96%), severity, auto-derived `line_labels` |
| **Master (consolidated)** | `data/raw/master_methods.jsonl` → per-role `data/final/master_samples.jsonl` | **15,351 pairs → 30,454 per-role samples** | **Single training corpus** (80/10/10) | + `token_line_ids_qwen` / `source_sink_labels` after v3.3 tokenization |
| **PyCode-Vul Train / Test** | `data/raw/external/PyCode_Vul-{train,test}-set.csv` | **14,248 / 3,563** functions | **Evaluation only**, final step | binary only |

### Hard rules (invariants)

1. **Master dataset** (gold + silver) is the single unified training corpus, namespaced `quality_tier` (`gold`/`silver`).
2. Splits are **fixed**: cross-dataset **repository-disjoint** (canonical lower-case `owner/name`), `--seed 42`, 80/10/10. Never re-split.
3. **PyCode-Vul** is **read-only and evaluation-only** — never enters `data/splits/`.
4. Every artifact (samples, tokens, graphs) is keyed by per-role `sample_id` (`{source}:{raw_id}:{role}`).
5. **Configs must be honored by code.** Hyperparameters in `configs/train/default.yaml`, architecture in `configs/model/default.yaml` — both injected by `train.py` and stored in each checkpoint.
6. **v3.3 fields:** after `tokenize_qwen.py` every split record carries `token_line_ids_qwen` (+ `offset_mapping_qwen`) for localization and `source_sink_labels` (after `generate_source_sink_labels.py`) for taint — see `04_dataset.md §4`.

---

## 3. Implementation Status Matrix — 6/6 Active (v3.3)

| Component | Status | Implementation Details |
|---|---|---|
| **Master Dataset build** | ✅ Implemented | `scripts/extraction/prepare_master.py` |
| **Semantic Branch** | ✅ Implemented | `src/semantic/encoder.py` (Qwen2.5-Coder, `return_offsets_mapping`) |
| **Graph Branch** | ✅ Implemented | `src/graph/encoder.py` (GAT, 5 edge types) |
| **Cross-Modal Fusion** | ✅ Implemented | `src/fusion/cross_attention.py` (gated) |
| **Multi-Task Heads** | ✅ Active (6/6) | Binary, CWE (10), Severity (4), **Localization** (token→line max-pool, λ=0.4), **Source/Sink** (per-token 3-class, λ=0.15), **Explanation** post-hoc — all wired (v3.3) |
| **Preprocessing** | ✅ Implemented (hardcoded `master_*`) | `scripts/preprocessing/*` — now includes `generate_source_sink_labels.py` |
| **Splitting** | ✅ Implemented + verified | `split.py` — cross-dataset repo-disjoint |
| **Tokenization** | ✅ Active (v3.3) | `tokenize_qwen.py` writes `token_line_ids_qwen` + `offset_mapping_qwen` (30,454 samples) |
| **Weak Taint Labels** | ✅ Active (v3.3) | `src/utils/taint.py` + `scripts/preprocessing/generate_source_sink_labels.py` → `source_sink_labels` |
| **Graph build + merge** | ✅ Implemented | `scripts/graph/*` + `merge_graphs.py` → `master_graphs.jsonl` (30,427) |
| **Training** | ✅ Active (v3.3) | `train.py` wires all 5 losses via `token_line_ids` + `source_sink_labels` |
| **In-Domain Evaluation** | ✅ Active (v3.3) | `scripts/evaluation/evaluate.py` reports binary/CWE/severity/**localization**/**source_sink** |
| **External Evaluation** | ✅ Implemented | `scripts/evaluation/evaluate_external.py` |
| **Explainability** | ✅ Active (v3.3) | `src/explainability/` (`prompts.py`, `generator.py`) + `scripts/explain.py` (offline template + optional LLM) |

*Full `semantic_only`/`fusion` training requires a GPU (`Qwen2.5-Coder-3B` not trainable on CPU). `graph_only` is CPU smoke-tested; `pytest 61 passed`.*

---

## 4. Canonical Data Pipeline (v3.3, fixed to `master`)

Pipeline is fixed to `master`. Each preprocessing/graph script is hardcoded to `master_*` paths; to target a different dataset, edit the `INPUT`/`OUTPUT` constants at the top of the script directly.

```
data/raw/databases/cvefixes.db ──extract.py──▶ data/raw/python_cvefixes_methods.jsonl   (2,985)
data/raw/ghsa/ghsa_methods.jsonl                                     (17,049; 4,683 dropped)
        │  scripts/extraction/prepare_master.py  (canonicalize repo, GHSA line labels)
        ▼
data/raw/master_methods.jsonl                     (15,351 pairs: 2,985 gold + 12,366 silver)
        │  clean_comments → normalize → validate_ast → strip_docstrings
        ▼
data/processed/master_graph_input.jsonl
        │  build_samples.py          (pair → vulnerable + safe role; per-role sample_id)
        ▼
data/final/master_samples.jsonl      (30,454 per-role samples; gold/silver tiers)
        │  split.py --seed 42        (cross-dataset repo-disjoint)
        ▼
data/splits/{train,validation,test}.jsonl       (20,638 / 6,404 / 3,412)
        │  tokenize_qwen.py          (in-place; mirror → data/tokenized/sem_qwen.jsonl)
        │    └─ writes token_line_ids_qwen + offset_mapping_qwen  ★ v3.3
        │  generate_source_sink_labels.py  ★ v3.3
        │    └─ writes source_sink_labels  (weak lexicon, src/utils/taint.py)
        ▼  graph builders + merge_graphs.py
data/processed/master_graphs.jsonl   (30,427 heterogeneous graphs keyed by sample_id)
```

The **diagram is the contract** — scripts must reproduce it exactly (hardcoded `master_*`). **After upgrading to v3.3:** re-run `tokenize_qwen.py` + `generate_source_sink_labels.py` once to materialize the two new fields (backward-compatible: missing fields simply skip their loss/metric).

---

## 5. Reproducible Workflow (v3.3)

No env var — just `python <script>.py` (each script defaults to `master_*` paths). Example (PowerShell):

```powershell
# 1. Build the unified Master corpus (gold CVEFixes + silver GHSA)
python scripts/extraction/prepare_master.py

# 2. Preprocess, expand to per-role samples, split
python scripts/preprocessing/clean_comments.py
python scripts/preprocessing/normalize.py
python scripts/preprocessing/validate_ast.py
python scripts/preprocessing/strip_docstrings.py
python scripts/preprocessing/build_samples.py
python scripts/preprocessing/split.py --seed 42

# 3. Tokenize splits in place + generate weak taint labels ★ v3.3
python scripts/preprocessing/tokenize_qwen.py              # needs network once (Qwen tokenizer) — writes token_line_ids_qwen
python scripts/preprocessing/generate_source_sink_labels.py # writes source_sink_labels (offline, idempotent)

# 4. Build + merge program graphs (optional, for graph_only/fusion)
python scripts/graph/build_ast.py
python scripts/graph/build_cfg.py
python scripts/graph/build_dfg.py
python scripts/graph/build_call.py
python scripts/graph/merge_graphs.py

# 5. Train the three branches (identical splits & schedule; GPU recommended)
python scripts/training/train.py --mode semantic_only --config configs/train/default.yaml
python scripts/training/train.py --mode graph_only --config configs/train/default.yaml --graph-data data/processed/master_graphs.jsonl
python scripts/training/train.py --mode fusion --config configs/train/default.yaml --graph-data data/processed/master_graphs.jsonl
# 5 losses active: binary 1.0 / cwe 0.5 / severity 0.2 / localization 0.4 / source_sink 0.15

# 6. In-domain evaluation (now reports 5 tasks)
python scripts/evaluation/evaluate.py --checkpoint models/checkpoints/best.pt
python scripts/evaluation/evaluate.py --checkpoint models/checkpoints/best.pt --graph-data data/processed/master_graphs.jsonl

# 7. Explain any prediction ★ v3.3
python scripts/explain.py --checkpoint models/checkpoints/best.pt --sample-id cvefixes:98919200308f75a4:vulnerable --output report.md
python scripts/explain.py --code-file app.py --cwe CWE-89 --severity HIGH --use-llm

# 8. External generalization (isolated, final step; semantic_only)
python scripts/evaluation/evaluate_external.py --checkpoint models/checkpoints/best.pt --split train
python scripts/evaluation/evaluate_external.py --checkpoint models/checkpoints/best.pt --split test
```

To target a different dataset, edit the `INPUT`/`OUTPUT` constants at the top of each script directly — no CLI switch or env var.

---

## 6. Vibe-Coding Guardrails (v3.3)

1. **Only `docs/` defines methodology.** Reconcile anything imported from `docs/archive/` here before adding it; never reintroduce the superseded policy as if it were current.
2. **Never touch PyCode-Vul / GHSA raw files** — read-only inputs. Derived files live outside `data/splits/`.
3. **Keep three branches comparable.** Data split, seed, LR, loss weights, and early stopping must be identical across `semantic_only`, `graph_only`, `fusion`.
4. **`sample_id` is sacred.** Samples, tokens, and graphs all use per-role `{source}:{raw_id}:{role}`. Graph data keyed otherwise silently fails to join.
5. **Pipeline is fixed to `master`.** All preprocessing/graph scripts are hardcoded to `master_*` paths; to target a different dataset, edit the `INPUT`/`OUTPUT` constants in the script header.
6. **Localization is active via `token_line_ids_qwen`.** Do **not** reintroduce the naive token↔line mismatch — always go through `token_line_ids_qwen` + `MultiTaskLoss._pool_tokens_to_lines` (max-pool). Missing alignment → skip the sample's localization loss.
7. **Checkpoint selection is on validation binary F1 only** (localization and source/sink F1 are logged but do not gate saving).
8. **Explanation is post-hoc.** Never let the explanation LLM influence training or checkpoint selection.

