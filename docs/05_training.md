# 05 — Training & Optimization (1-Stage on the Master Dataset)

> **Version: 4.0** — **3 trainable heads**
> **Authoritative Specification**

---

## 1. Data Input Contract

Training consumes `data/splits/{train,validation}.jsonl` from the **Master Dataset** (Pillars 1–2) —
per-role records keyed by `{source}:{pair}:{role}`, each carrying:

- `code`, `binary_label`, `cwe_ids`, `severity`, `quality_tier` (gold/silver)
- `input_ids_qwen` + `attention_mask_qwen`
- For graph/fusion modes: per-sample heterogeneous graphs in `data/processed/master_graphs.jsonl`

`VulHunterDataset` loads all fields; `collate_fn` pads `input_ids_qwen` → `(B, max_seq)`.

---

## 2. Experimental Regimes

Three branches share identical data, seed, loss weights, scheduler, and early stopping:

| Branch | Modalities | Purpose |
|---|---|---|
| `semantic_only` | LLM backbone (seq + pool) | Pure semantic baseline |
| `graph_only` | GAT on AST+CFG+DFG+Call | Pure structural baseline |
| `fusion` (Proposed) | Semantic + Graph cross-attention | Proposed hybrid — supervises all 3 heads |

**Architecture source (Pillar 4):** `configs/kaggle/model_kaggle.yaml` (`--model-config`), injected by
`train.py` and saved into each checkpoint. Never change architecture for only one branch.

---

## 3. Pillar 4 — Quality-Aware Weighted Multi-Task Loss

Per-sample loss scales every task by the **quality tier** so noisier GHSA diffs perturb gradients less:

$$\mathcal{L}_{\text{sample}} = w_{\text{tier}}\cdot\Big[\lambda_{\text{bin}}\mathcal{L}_{\text{bin}} + \lambda_{\text{cwe}}\mathcal{L}_{\text{cwe}} + \lambda_{\text{sev}}\mathcal{L}_{\text{sev}}\Big]$$

### 3.1 Quality-tier weights ($w_{\text{tier}}$)

| Tier | Source | $w_{\text{tier}}$ |
|---|---|---|
| `gold` | CVEFixes (reviewed diffs) | **1.0** |
| `silver` | GHSA (auto-derived diffs) | **0.85** |

`VulHunterDataset.sample_weights` → `MultiTaskLoss(sample_weights=…)`.

### 3.2 Per-task weights & losses

| Task | λ | Loss | Input → Supervision | Note |
|---|---|---|---|---|
| **Binary** | **1.0** | `FocalLoss(alpha=0.25, gamma=2.0)` | `h_fused` → `binary_label` | primary |
| **CWE** | **0.5** | `CrossEntropy(label_smoothing=0.1)` | `h_fused` → 10 classes | benefits most from GHSA silver |
| **Severity** | **0.2** | masked `CrossEntropy(label_smoothing=0.05)` | `h_fused` → 4 tiers, skip UNKNOWN/-1 | — |

> **Config/code sync:** weights live identically in `configs/train/*.yaml` and
> `src/utils/losses.py::MultiTaskLoss` defaults; `train.py` honors the YAML via
> `criterion.update_weights()`. Keep them equal.

---

## 4. Pillar 4 — Optimization Hyperparameters

- **Full-finetune (Kaggle 1xP100):** backbone `2e-5`; graph / cross-attention / heads `2e-4` (**10×**).
- **Schedule:** linear warmup **10%** steps → cosine decay (`LambdaLR`).
- **Effective batch:** local `batch_size=8` × `grad_accum=4` → **32**.
- **Gradient clipping:** max norm 1.0.
- **Epochs:** 6 epochs, early stopping **patience 2** on **validation binary F1**.
- **Optimizer:** AdamW, weight decay 0.01, betas (0.9, 0.999).
- **Checkpoint:** `best.pt` on val binary F1; `training_history.json` logs per-epoch losses + `val_metrics`.

---

## 5. Curriculum & Two-Stage (removed)

Legacy curriculum and two-stage plans are **purged**. Default protocol is **single-stage end-to-end**
on the Master Dataset (GHSA included directly with quality down-weight).

---

## 6. Reproducibility

- Seed 42 everywhere; fixed repo-disjoint splits; deterministic checkpointing.
- All three branches via `scripts/training/train.py --config configs/train/<mode>.yaml [--graph-data …]`.
