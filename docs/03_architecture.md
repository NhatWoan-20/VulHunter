# 03 — System Architecture

> **Version: 3.3 (2026-08-30)** — **6/6 Tasks Active**
> **Authoritative Specification**

---

## 1. High-Level Architecture

VulHunter comprises two complementary encoders, cross-modal fusion, **5 trainable multi-task heads** and one post-hoc explainability layer:

```
                           Python Function Code
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
 ┌──────────────────────┐                            ┌──────────────────────┐
 │   Semantic Branch    │                            │     Graph Branch     │
 │  (Tokenizer + LLM)   │                            │ (AST+CFG+DFG+Call GAT)│
 │  offset_mapping→     │                            │                      │
 │  token_line_ids      │                            │                      │
 └──────────┬───────────┘                            └──────────┬───────────┘
            │ H_sem ∈ ℝ^(B×L×D) seq + h_sem ∈ ℝ^(B×D) pool     │ H_graph ∈ ℝ^(B×N×D) / h_graph ∈ ℝ^(B×D)
            │                                                   │
            └────────────────────────┬──────────────────────────┘
                                     ▼
                      ┌──────────────────────────────┐
                      │     Cross-Modal Fusion       │
                      │ (Bidirectional Cross-Attn)   │
                      └──────────────┬───────────────┘
                                     │ h_fused ∈ ℝ^(B×D), H_seq ∈ ℝ^(B×L×D)
      ┌──────────┬──────────┬────────┼────────┬──────────────┐
      ▼          ▼          ▼        ▼        ▼              ▼
 ┌─────────┐ ┌─────────┐ ┌──────┐ ┌──────┐ ┌──────────┐ ┌──────────────┐
 │ Binary  │ │   CWE   │ │Severity│ │ Loc. │ │Source/ │ │ Explain.     │
 │  Head   │ │  Head   │ │ Head  │ │ Head │ │ Sink   │ │ (post-hoc)   │
 │ (Focal) │ │  (CE)   │ │ (CE)  │ │(Focal)│ │ Head   │ │ Markdown     │
 └─────────┘ └─────────┘ └──────┘ └──────┘ │  (CE)  │ │  report      │
                                            └────────┘ └──────────────┘
   5 trainable heads (joint loss)  ────────────────▶  LLM-optional
```

> **Pillar 4 config note:** concrete dims/layers/heads/dropout/`num_classes` live in **`configs/model/default.yaml`** and are injected by `train.py` via `--model-config`; the effective config is saved in each checkpoint so evaluation rebuilds the identical model. `configs/train/default.yaml` owns the 5 loss weights (binary 1.0 / cwe 0.5 / severity 0.2 / **localization 0.4 / source_sink 0.15** — v3.3, all active).

---

## 2. Component Specifications

### 2.1 Semantic Encoder (`src/semantic/encoder.py`)

- **Backbone:** `Qwen/Qwen2.5-Coder-3B-Instruct` (default) / `Qwen/Qwen2.5-Coder-1.5B-Instruct` for Kaggle.
- **Context:** 2,048 tokens (Kaggle 3B LoRA: **2,048** keeps full context — ~12GB FIT); `return_offsets_mapping=True` so `scripts/preprocessing/tokenize_qwen.py` can derive **`token_line_ids_qwen`** via `bisect(line_starts, offset)` and materialize `offset_mapping_qwen`.
- **Kaggle 2×T4 — LoRA for 3B (recommended):** `3B` full-finetune ~19 GB/GPU → OOM on T4 (DataParallel replicates per GPU). **LoRA** `r=32 alpha=64 dropout=0.05` target `[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]` + `RsLoRA` + `gradient_checkpointing` + `fp16` + `lr 2e-4` (10×) + `bs1 accum16 eff 32` + `len 2048` → **~12 GB/GPU FIT**, ~1.5–2 h, F1 +2% vs 1.5B full. Code: `src/semantic/encoder.py` `use_lora` via `peft` (`configs/kaggle/model_kaggle_3b_lora.yaml` + `train_kaggle_3b_lora.yaml`). Freeze path (`freeze_layers`) still used for 1.5B.
- **Layer Freezing (non-LoRA):** embedding + first `freeze_layers` (default 28/36) frozen; top layers fine-tuned.
- **Outputs:** masked **mean-pooled** `h_sem ∈ ℝ^D` for classification heads + full sequence `H_seq ∈ ℝ^(L×D)` for token-level heads (localization, source/sink).

### 2.2 Graph Encoder (`src/graph/encoder.py`)

- **Architecture:** 4-layer GAT, H=8 heads, `d_node=128`, `hidden=256`, `output=256`.
- **Edge Types (5):** `AST_CHILD`, `NEXT_STATEMENT`, `CONTROL_FLOW`, `DATA_FLOW`, `CALL` (`EDGE_TYPE_MAP`).
- **Readout:** mean-pool over nodes → `h_graph ∈ ℝ^D`.

### 2.3 Cross-Modal Fusion (`src/fusion/cross_attention.py`)

- **Mechanism:** bidirectional multi-head cross-attention `Softmax(QKᵀ/√d_k)V`; residual + LayerNorm: `h_fused = LayerNorm(h_sem + W_p[h_sem‖h_graph])`.
- **Combine:** `gated` (default; `concat`/`mean` switches) via `fusion.combine`.
- **Sequence path:** `H_seq` (semantic) is the sequence fed to localization/source-sink heads; fusion refines the pooled `h_fused` for classification heads.

### 2.4 Multi-Task Prediction Heads (`src/multitask/heads.py`) — 5 trainable

| Head | Input | Output | Loss (v3.3) |
|---|---|---|---|
| **Binary** | `h_fused` | logit ŷ_bin ∈ ℝ¹ | Focal α=0.25 γ=2.0, λ=1.0 |
| **CWE** | `h_fused` | logits ŷ_cwe ∈ ℝ¹⁰ | CE label_smooth 0.1, λ=0.5 |
| **Severity** | `h_fused` | logits ŷ_sev ∈ ℝ⁴ (masked if UNKNOWN=-1) | CE label_smooth 0.05, λ=0.2 |
| **Localization** | `H_seq` (B×L×D) | per-token logits → **max-pool by `token_line_ids`** → per-line p∈[0,1] (see §2.4.1) | Focal α=0.5 γ=2.0, **λ=0.4 ACTIVE** |
| **Source/Sink** | `H_seq` | per-token logits ŷ_ss ∈ ℝ³ (Normal/Source/Sink) | CE ignore=-1, **λ=0.15 ACTIVE** (weak lexicon) |

#### 2.4.1 Localization — token→line aggregation

`LocalizationHead` emits `(B, L, 1)` per-token logits. `src/utils/losses.py::_pool_tokens_to_lines` groups tokens by `token_line_ids` (B×L, -1 = special/pad) and takes the **max logit per line** as the line prediction; only lines with `line_labels != -1` and at least one aligned token contribute. Quality-weighted focal loss, truncation-safe.

#### 2.4.2 Source/Sink — weak supervision

`SourceSinkHead` (B×L×3) is supervised by `source_sink_labels` (B×L, -1 = ignore) derived offline by `src/utils/taint.py` + `scripts/preprocessing/generate_source_sink_labels.py` (Source substrings: `request.args`, `input(`, `os.environ`…; Sink: `cursor.execute`, `os.system`, `eval(`, `pickle.loads`…; Sink > Source > Normal). Safe samples → all Normal. CE, quality-weighted, truncation-safe (min-L slice).

### 2.5 Explainability Layer — post-hoc (`src/explainability/`) — ✅ Active (v3.3)

No training loss. Consumes the 5 heads' predictions + code to produce a **Markdown remediation report**:

- `prompts.py` — CWE knowledge base (`CWE_DESCRIPTIONS`, `REMEDIATION_HINTS`, `SEVERITY_GUIDANCE`), `SYSTEM_PROMPT`, `USER_PROMPT_TEMPLATE`, `format_code_block`, `build_user_prompt`.
- `generator.py` — `ExplanationGenerator` with two tiers: (1) **offline template** (deterministic, always succeeds) with CWE-aware root-cause + taint narrative + patch snippet per CWE; (2) optional **LLM polish** (`use_llm=True`) via local HF (`api_mode="hf"`, default Qwen2.5-Coder) or OpenAI-compatible endpoint (`api_mode="openai"`; env `OPENAI_API_KEY/BASE_URL/MODEL`, `VH_EXPLAIN_MODEL`).
- CLI: `scripts/explain.py` (ad-hoc `--code-file` or checkpoint-driven `--checkpoint --sample-id`).

> **Alignment:** per-role `sample_id = "{source}:{pair_id}:{role}"` links semantic tokens (`input_ids_qwen`, `token_line_ids_qwen`, `source_sink_labels`), graph nodes, and all 5 label vectors. `quality_tier` (gold/silver) → `sample_weights` scales every trainable loss (Pillar 4).
