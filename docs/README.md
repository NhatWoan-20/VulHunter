# VulHunter Research Specification & Documentation

> **Authoritative methodology — Master 1-Stage, 3 Tasks Active**

VulHunter is a hybrid multi-modal vulnerability detection framework for Python source code combining semantic code representations (Transformers / Code LLMs) and structural program graphs (Heterogeneous AST / CFG / DFG / Call graphs).

> **Single source of truth:** `docs/` (this folder) is the only authoritative methodology. The legacy v1/v2 research draft and the original full-length specification are archived and **superseded** in `docs/archive/`.

---

## 1. Documentation Map

| Document | Topic | Key Content |
|---|---|---|
| [`01_overview.md`](01_overview.md) | Problem & Objectives | Research questions, scope, 3-task schema, success criteria |
| [`02_literature_review.md`](02_literature_review.md) | State of the Art | Semantic vs Graph vs Hybrid models, literature gaps |
| [`03_architecture.md`](03_architecture.md) | System Architecture | Encoders, Fusion, **3 trainable heads** |
| [`04_dataset.md`](04_dataset.md) | Data & Preprocessing | Unified Master dataset |
| [`05_training.md`](05_training.md) | Training & Optimization | 3-loss multi-task (binary/CWE/severity), tiered LR |
| [`06_evaluation.md`](06_evaluation.md) | Evaluation & Ablation | 3-task metrics |
| [`07_extensions.md`](07_extensions.md) | Research Extensions | (Deprecated) Line localization, source/sink, LLM explanations |

---

## 2. Dataset Contract & Policy

| Dataset | Location | Records | Role | Supervision |
|---|---|---|---|---|
| **CVEFixes (gold)** | `data/raw/python_cvefixes_methods.jsonl` | **2,985** pairs (659 CVEs, 361 repos) | Primary high-quality tier | binary, CWE, severity |
| **GHSA (silver)** | `data/raw/ghsa/ghsa_methods.jsonl` | **12,366** kept of 17,049 (4,683 noise removed) | Secondary tier | binary, CWE, severity |
| **Master (consolidated)** | `data/raw/master_methods.jsonl` → per-role `data/final/master_samples.jsonl` | **15,351 pairs → 30,454 per-role samples** | **Single training corpus** (80/10/10) | |
| **PyCode-Vul Train / Test** | `data/raw/external/PyCode_Vul-{train,test}-set.csv` | **14,248 / 3,563** functions | **Evaluation only**, final step | binary only |

### Hard rules (invariants)

1. **Master dataset** (gold + silver) is the single unified training corpus, namespaced `quality_tier` (`gold`/`silver`).
2. Splits are **fixed**: cross-dataset **repository-disjoint** (canonical lower-case `owner/name`), `--seed 42`, 80/10/10. Never re-split.
3. **PyCode-Vul** is **read-only and evaluation-only** — never enters `data/splits/`.
4. Every artifact (samples, tokens, graphs) is keyed by per-role `sample_id` (`{source}:{raw_id}:{role}`).
5. **Configs must be honored by code.** Hyperparameters in `configs/train/*.yaml`, architecture in `configs/kaggle/model_kaggle.yaml` — both injected by `train.py` and stored in each checkpoint.

---

## 3. Implementation Status Matrix

| Component | Status | Implementation Details |
|---|---|---|
| **Master Dataset build** | ✅ Implemented | `scripts/extraction/prepare_master.py` |
| **Semantic Branch** | ✅ Implemented | `src/semantic/encoder.py` (Qwen2.5-Coder-1.5B-Instruct) |
| **Graph Branch** | ✅ Implemented | `src/graph/encoder.py` (GAT, 5 edge types) |
| **Cross-Modal Fusion** | ✅ Implemented | `src/fusion/cross_attention.py` (gated) |
| **Multi-Task Heads** | ✅ Active (3/3) | Binary, CWE (10), Severity (4) |
| **Preprocessing** | ✅ Implemented | `scripts/preprocessing/*` |
| **Splitting** | ✅ Implemented + verified | `split.py` — cross-dataset repo-disjoint |
| **Tokenization** | ✅ Active | `tokenize_qwen.py` |
| **Graph build + merge** | ✅ Implemented | `scripts/graph/*` + `merge_graphs.py` → `master_graphs.jsonl` |
| **Training** | ✅ Active | `train.py` wires 3 losses via Kaggle 2x T4 environments |
| **In-Domain Evaluation** | ✅ Active | `scripts/evaluation/evaluate.py` reports binary/CWE/severity |
| **External Evaluation** | ✅ Implemented | `scripts/evaluation/evaluate_external.py` |

*Full `semantic_only`/`fusion` training runs on Kaggle 2x T4 (16GB).*

---

## 4. Canonical Data Pipeline (fixed to `master`)

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
        │  tokenize_qwen.py          (in-place)
        ▼  graph builders + merge_graphs.py
data/processed/master_graphs.jsonl   (30,427 heterogeneous graphs keyed by sample_id)
```

The **diagram is the contract** — scripts must reproduce it exactly (hardcoded `master_*`).

---

## 5. Reproducible Workflow

No env var — just `python <script>.py` (each script defaults to `master_*` paths). Example:

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

# 3. Tokenize splits in place
python scripts/preprocessing/tokenize_qwen.py

# 4. Build + merge program graphs (optional, for graph_only/fusion)
python scripts/graph/build_ast.py
python scripts/graph/build_cfg.py
python scripts/graph/build_dfg.py
python scripts/graph/build_call.py
python scripts/graph/merge_graphs.py

# 5. Train the three branches (identical splits & schedule; GPU recommended)
python scripts/training/train.py --mode semantic_only --config configs/train/semantic.yaml
python scripts/training/train.py --mode graph_only --config configs/train/graph.yaml --graph-data data/processed/master_graphs.jsonl
python scripts/training/train.py --mode fusion --config configs/train/fusion.yaml --graph-data data/processed/master_graphs.jsonl
# 3 losses active: binary 1.0 / cwe 0.5 / severity 0.2

# 6. In-domain evaluation (reports 3 tasks)
python scripts/evaluation/evaluate.py --checkpoint models/checkpoints/best.pt
python scripts/evaluation/evaluate.py --checkpoint models/checkpoints/best.pt --graph-data data/processed/master_graphs.jsonl

# 7. External generalization (isolated, final step; semantic_only)
python scripts/evaluation/evaluate_external.py --checkpoint models/checkpoints/best.pt --split train
python scripts/evaluation/evaluate_external.py --checkpoint models/checkpoints/best.pt --split test
```

To target a different dataset, edit the `INPUT`/`OUTPUT` constants at the top of each script directly — no CLI switch or env var.

---

## 6. Vibe-Coding Guardrails

1. **Only `docs/` defines methodology.** Reconcile anything imported from `docs/archive/` here before adding it.
2. **Never touch PyCode-Vul / GHSA raw files** — read-only inputs. Derived files live outside `data/splits/`.
3. **Keep three branches comparable.** Data split, seed, LR, loss weights, and early stopping must be identical across `semantic_only`, `graph_only`, `fusion`.
4. **`sample_id` is sacred.** Samples, tokens, and graphs all use per-role `{source}:{raw_id}:{role}`. Graph data keyed otherwise silently fails to join.
5. **Pipeline is fixed to `master`.** All preprocessing/graph scripts are hardcoded to `master_*` paths; to target a different dataset, edit the `INPUT`/`OUTPUT` constants in the script header.
6. **Checkpoint selection is on validation binary F1 only.**
7. **Explanation is post-hoc (deprecated for primary training).** Never let the explanation LLM influence training or checkpoint selection.

