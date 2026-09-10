# 03 — System Architecture

> **Version: 4.0** — **3/3 Tasks Active**
> **Authoritative Specification**

---

## 1. High-Level Architecture

VulHunter comprises two complementary encoders, cross-modal fusion, and **3 trainable multi-task heads**:

```
                           Python Function Code
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
 ┌──────────────────────┐                            ┌──────────────────────┐
 │   Semantic Branch    │                            │     Graph Branch     │
 │  (Tokenizer + LLM)   │                            │ (AST+CFG+DFG+Call GAT)│
 └──────────┬───────────┘                            └──────────┬───────────┘
            │ H_sem ∈ ℝ^(B×L×D) seq + h_sem ∈ ℝ^(B×D) pool     │ H_graph ∈ ℝ^(B×N×D) / h_graph ∈ ℝ^(B×D)
            │                                                   │
            └────────────────────────┬──────────────────────────┘
                                     ▼
                      ┌──────────────────────────────┐
                      │     Cross-Modal Fusion       │
                      │ (Bidirectional Cross-Attn)   │
                      └──────────────┬───────────────┘
                                     │ h_fused ∈ ℝ^(B×D)
      ┌──────────────────────────────┼──────────────────────────────┐
      ▼                              ▼                              ▼
 ┌─────────┐                    ┌─────────┐                    ┌──────┐
 │ Binary  │                    │   CWE   │                    │Severity│
 │  Head   │                    │  Head   │                    │ Head  │
 │ (Focal) │                    │  (CE)   │                    │ (CE)  │
 └─────────┘                    └─────────┘                    └──────┘
                 3 trainable heads (joint loss)
```

> **Pillar 4 config note:** concrete dims/layers/heads/dropout/`num_classes` live in **`configs/kaggle/model_kaggle.yaml`** and are injected by `train.py` via `--model-config`; the effective config is saved in each checkpoint so evaluation rebuilds the identical model. `configs/train/*.yaml` owns the 3 loss weights (binary 1.0 / cwe 0.5 / severity 0.2).

---

## 2. Component Specifications

### 2.1 Semantic Encoder (`src/semantic/encoder.py`)

- **Backbone:** `Qwen/Qwen2.5-Coder-1.5B-Instruct` for Kaggle 2x T4 environment.
- **Context:** 2,048 tokens.
- **Layer Freezing:** embedding + first `freeze_layers` (default 28/36) frozen; top layers fine-tuned.
- **Outputs:** masked **mean-pooled** `h_sem ∈ ℝ^D` for classification heads.

### 2.2 Graph Encoder (`src/graph/encoder.py`)

- **Architecture:** 4-layer GAT, H=8 heads, `d_node=128`, `hidden=256`, `output=256`.
- **Edge Types (5):** `AST_CHILD`, `NEXT_STATEMENT`, `CONTROL_FLOW`, `DATA_FLOW`, `CALL` (`EDGE_TYPE_MAP`).
- **Readout:** mean-pool over nodes → `h_graph ∈ ℝ^D`.

### 2.3 Cross-Modal Fusion (`src/fusion/cross_attention.py`)

- **Mechanism:** bidirectional multi-head cross-attention `Softmax(QKᵀ/√d_k)V`; residual + LayerNorm: `h_fused = LayerNorm(h_sem + W_p[h_sem‖h_graph])`.
- **Combine:** `gated` (default; `concat`/`mean` switches) via `fusion.combine`.
- **Sequence path:** `H_seq` (semantic) is the sequence fed to classification heads via pooling.

### 2.4 Multi-Task Prediction Heads (`src/multitask/heads.py`) — 3 trainable

| Head | Input | Output | Loss |
|---|---|---|---|
| **Binary** | `h_fused` | logit ŷ_bin ∈ ℝ¹ | Focal α=0.25 γ=2.0, λ=1.0 |
| **CWE** | `h_fused` | logits ŷ_cwe ∈ ℝ¹⁰ | CE label_smooth 0.1, λ=0.5 |
| **Severity** | `h_fused` | logits ŷ_sev ∈ ℝ⁴ (masked if UNKNOWN=-1) | CE label_smooth 0.05, λ=0.2 |

> **Alignment:** per-role `sample_id = "{source}:{pair_id}:{role}"` links semantic tokens (`input_ids_qwen`), graph nodes, and all 3 label vectors. `quality_tier` (gold/silver) → `sample_weights` scales every trainable loss.

