# 05 — Training & Optimization (1-Stage on the Master Dataset)

> **Version: 3.3 (2026-08-30)** — **5 trainable heads, 6th post-hoc**
> **Authoritative Specification**

---

## 1. Data Input Contract (v3.3)

Training consumes `data/splits/{train,validation}.jsonl` from the **Master Dataset** (Pillars 1–2) —
per-role records keyed by `{source}:{pair}:{role}`, each carrying:

- `code`, `binary_label`, `cwe_ids`, `severity`, `line_labels`, `quality_tier` (gold/silver)
- **v3.3:** `input_ids_qwen` + `attention_mask_qwen` + **`token_line_ids_qwen`** (`-1`=special/pad) + **`offset_mapping_qwen`** + **`source_sink_labels`** (`-1`=ignore, else 0/1/2)
- For graph/fusion modes: per-sample heterogeneous graphs in `data/processed/master_graphs.jsonl`

`VulHunterDataset` loads all fields; `collate_fn` pads `token_line_ids` → `(B, max_seq)` and
`source_sink_labels` → `(B, max_seq)`. Missing enrichment (legacy splits) is re-derived on the fly
or skipped for that task — see `04_dataset.md §1.4`.

---

## 2. Experimental Regimes

Three branches share identical data, seed, loss weights, scheduler, and early stopping:

| Branch | Modalities | Purpose |
|---|---|---|
| `semantic_only` | LLM backbone (seq + pool) | Pure semantic baseline |
| `graph_only` | GAT on AST+CFG+DFG+Call | Pure structural baseline |
| `fusion` (Proposed) | Semantic + Graph cross-attention | Proposed hybrid — supervises all 5 heads |

**Architecture source (Pillar 4):** `configs/model/default.yaml` (`--model-config`), injected by
`train.py` and saved into each checkpoint. Never change architecture for only one branch.

---

## 3. Pillar 4 — Quality-Aware Weighted Multi-Task Loss (v3.3, 5 active losses)

Per-sample loss scales every task by the **quality tier** so noisier GHSA diffs perturb gradients less:

$$\mathcal{L}_{\text{sample}} = w_{\text{tier}}\cdot\Big[\lambda_{\text{bin}}\mathcal{L}_{\text{bin}} + \lambda_{\text{cwe}}\mathcal{L}_{\text{cwe}} + \lambda_{\text{sev}}\mathcal{L}_{\text{sev}} + \lambda_{\text{loc}}\mathcal{L}_{\text{loc}} + \lambda_{\text{ss}}\mathcal{L}_{\text{ss}}\Big]$$

### 3.1 Quality-tier weights ($w_{\text{tier}}$)

| Tier | Source | $w_{\text{tier}}$ |
|---|---|---|
| `gold` | CVEFixes (reviewed diffs) | **1.0** |
| `silver` | GHSA (auto-derived diffs) | **0.85** |

`VulHunterDataset.sample_weights` → `MultiTaskLoss(sample_weights=…)`; also used to weight per-line
and per-token losses in v3.3.

### 3.2 Per-task weights & losses — all wired (v3.3)

| Task | λ | Loss | Input → Supervision | Note |
|---|---|---|---|---|
| **Binary** | **1.0** | `FocalLoss(alpha=0.25, gamma=2.0)` | `h_fused` → `binary_label` | primary; checkpoint selection on val binary F1 |
| **CWE** | **0.5** | `CrossEntropy(label_smoothing=0.1)` | `h_fused` → 10 classes | benefits most from GHSA silver |
| **Severity** | **0.2** | masked `CrossEntropy(label_smoothing=0.05)` | `h_fused` → 4 tiers, skip UNKNOWN/-1 | — |
| **Localization** | **0.4** | `FocalLoss(alpha=0.5, gamma=2.0)` via **`_pool_tokens_to_lines`** (max-pool) | `H_seq (B×L×D)` → per-token logits → **max per `token_line_ids`** → per-line vs `line_labels` | **ACTIVE v3.3** — needs `token_line_ids_qwen`; re-run `tokenize_qwen.py` once after upgrade; fallback skips sample if all -1 |
| **Source/Sink** | **0.15** | masked `CrossEntropy(ignore_index=-1)` | `H_seq` → per-token 3-class {Normal,Source,Sink} vs `source_sink_labels` | **ACTIVE v3.3 (weak)** — lexicon `src/utils/taint.py` → `generate_source_sink_labels.py`; quality-weighted; truncation-safe |

> **Config/code sync:** weights live identically in `configs/train/default.yaml` and
> `src/utils/losses.py::MultiTaskLoss` defaults; `train.py` honors the YAML via
> `criterion.update_weights()`. Keep them equal.

**Localization detail:** `LocalizationHead` emits `(B, L, 1)` per-token logits. `MultiTaskLoss._pool_tokens_to_lines`
groups tokens by `token_line_ids` and takes the **max logit per line**; only lines with `line_labels != -1`
and ≥1 aligned token contribute. Per-line focal loss is then quality-weighted by the sample's `w_tier`.

**Source/Sink detail:** `SourceSinkHead` emits `(B, L, 3)`. `source_sink_labels` (`-1`=special/pad) come from
`src/utils/taint.py` lexicons (Source: `request.args`, `input(`, `os.environ`…; Sink: `cursor.execute`,
`os.system`, `eval(`, `pickle.loads`…; Sink > Source > Normal; safe→all Normal). Sliced to `min(L)` for
truncation safety; quality-weighted token CE.

**Explanation (6th task):** **no training loss** — post-hoc `src/explainability/` consumes the 5 heads' outputs.

---

## 4. Pillar 4 — Optimization Hyperparameters

- **Tiered LR (full-finetune, local 3B):** backbone `1.5e-5`; graph / cross-attention / heads `1.5e-4` (**10×**).
- **LoRA LR (Kaggle 2×T4, 3B LoRA — recommended):** LoRA adapters `2e-4` (**10×** full), heads `2e-4`; base backbone frozen. Effective batch `1 × 16 × 2 GPUs = 32` (`configs/kaggle/train_kaggle_3b_lora.yaml` + `model_kaggle_3b_lora.yaml`), `max_length 2048` (keeps full context — ~12GB FIT, avg code ~400 tokens), `gradient_checkpointing + fp16 + RsLoRA r32 alpha64`. See `src/semantic/encoder.py` `use_lora` + `configs/kaggle/model_kaggle_3b_lora.yaml`.
- **Schedule:** linear warmup **10%** steps → cosine decay (`LambdaLR`).
- **Effective batch:** local `batch_size=8` × `grad_accum=4` → **32**; Kaggle 3B LoRA `1 × 16 × 2 = 32`.
- **Gradient clipping:** max norm 1.0.
- **Epochs:** local 15–20 (Master ~7× larger than CVEFixes-only); **Kaggle 3B LoRA 6 epochs**, early stopping **patience 4 local / 2 Kaggle** on **validation binary F1** (localization/source_sink F1 logged but not gating).
- **Optimizer:** AdamW, weight decay 0.01, betas (0.9, 0.999).
- **Checkpoint:** `best.pt` on val binary F1; `training_history.json` logs per-epoch losses + `val_metrics` (binary + localization).

---

## 5. Curriculum & Two-Stage (removed)

Legacy curriculum and two-stage plans are **purged**. Default protocol is **single-stage end-to-end**
on the Master Dataset (GHSA included directly with quality down-weight).

---

## 6. Reproducibility

- Seed 42 everywhere; fixed repo-disjoint splits; deterministic checkpointing.
- All three branches via `scripts/training/train.py --config configs/train/default.yaml [--graph-data …]`.
- After upgrading to v3.3: `python scripts/preprocessing/tokenize_qwen.py && python scripts/preprocessing/generate_source_sink_labels.py` once before training.

