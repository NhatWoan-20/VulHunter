# 03 — System Architecture

> **Version: 5.0** — Binary Classification Focus
> **Authoritative Specification**

---

## 1. High-Level Architecture

VulHunter comprises two complementary encoders, cross-modal fusion, and **1 trainable binary prediction head**:

```
                           Python Function Code
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
 ┌──────────────────────┐                            ┌──────────────────────┐
 │   Semantic Branch     │                            │     Graph Branch     │
 │  (Qwen2.5-Coder +   │                            │ (AST+CFG+DFG+Call GAT)│
 │   LoRA, r=16/α=32)  │                            │  GraphCodeBERT        │
 │  Last-token pooling  │                            │  unfreeze top-6      │
 └──────────┬───────────┘                            └──────────┬───────────┘
             │ H_sem ∈ ℝ^(B×L×D) + h_sem ∈ ℝ^(B×D) pool      │ H_graph ∈ ℝ^(B×N×D) / h_graph ∈ ℝ^(B×D)
             │                                                   │
             └────────────────────────┬──────────────────────────┘
                                      ▼
                       ┌──────────────────────────────┐
                       │     Cross-Modal Fusion       │
                       │ (Gated Bidirectional Cross-Attn)│
                       │ + Residual Skip (α=0.3)     │
                       └──────────────┬───────────────┘
                                      │ h_fused ∈ ℝ^(B×D)
                                      ▼
                       ┌──────────────────────────────┐
                       │    Binary Prediction Head     │
                       │      (Focal Loss)           │
                       └──────────────┬───────────────┘
                                      │ p(vulnerable) ∈ [0, 1]
                                      ▼
                                 ŷ ∈ {0, 1}
```

> **Config source:** `configs/model/default.yaml` (`--model-config`); effective config is saved in each checkpoint for reproducibility.

---

## 2. Component Specifications

### 2.1 Semantic Encoder (`src/semantic/encoder.py`)

| Property | Value |
|---|---|
| **Backbone** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| **Context Length** | 2,048 tokens |
| **Pooling** | **Last-token pooling** (optimal for decoder-only LLMs) |
| **Fine-tuning** | **LoRA** (r=16, α=32, dropout=0.05, RSLoRA enabled) |
| **Target Modules** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| **Gradient Checkpointing** | Optional (enable for VRAM < 14GB) |
| **Output** | `h_sem ∈ ℝ^D` (pooled representation, D=256) |

### 2.2 Graph Encoder (`src/graph/encoder.py`)

| Property | Value |
|---|---|
| **Architecture** | 4-layer GAT, H=8 attention heads |
| **Node Features** | `d_node=128` |
| **Hidden/Output Dim** | 256 |
| **Edge Types (5)** | `AST_CHILD`, `NEXT_STATEMENT`, `CONTROL_FLOW`, `DATA_FLOW`, `CALL` (`EDGE_TYPE_MAP`) |
| **Readout** | Mean-pool over nodes → `h_graph ∈ ℝ^D` |
| **GraphCodeBERT** | Unfreeze top-6 layers (critical to avoid AUC=0.5 collapse) |

### 2.3 Cross-Modal Fusion (`src/fusion/cross_attention.py`)

| Property | Value |
|---|---|
| **Mechanism** | Bidirectional multi-head cross-attention |
| **Combine Mode** | `gated` (options: `concat`, `mean`) |
| **Residual Skip** | `h_fused = LayerNorm(h_sem + α * h_cross)` where α=0.3 |
| **Layers** | 2 cross-attention layers |
| **Heads** | 8 |
| **Output** | `h_fused ∈ ℝ^D` (same dimension as semantic/graph outputs) |

### 2.4 Binary Prediction Head (`src/multitask/heads.py`)

| Property | Value |
|---|---|
| **Input** | `h_fused ∈ ℝ^D` |
| **Architecture** | MLP: Linear(D→128) → GELU → Dropout → Linear(128→64) → GELU → Dropout → Linear(64→1) |
| **Output** | Logit ŷ ∈ ℝ¹ |
| **Loss Function** | Focal Loss (α=0.5, γ=2.0) with quality-tier sample weighting |

---

## 3. Three Operating Modes

VulHunter supports three modes, selected via `--mode` flag:

| Mode | Semantic Encoder | Graph Encoder | Fusion | Use Case |
|---|---|---|---|---|
| `semantic_only` | ✓ Qwen + LoRA | ✗ | ✗ | Semantic baseline |
| `graph_only` | ✗ | ✓ GraphCodeBERT + GAT | ✗ | Structural baseline |
| `fusion` (Proposed) | ✓ Qwen + LoRA | ✓ GraphCodeBERT + GAT | ✓ Gated Cross-Attn | **Main approach** |

---

## 4. Data Flow

1. **Input:** Python function code string
2. **Semantic Branch:** Tokenize with Qwen tokenizer → LoRA fine-tuned Qwen2.5-Coder → last-token pooling → `h_sem`
3. **Graph Branch:** Parse AST/CFG/DFG/Call → heterogeneous graph → GraphCodeBERT + GAT → mean pool → `h_graph`
4. **Fusion:** Cross-attend `h_sem` and `h_graph` → residual skip → `h_fused`
5. **Prediction:** Binary head → logit → sigmoid → `p(vulnerable)`

---

## 5. Model Configuration

All hyperparameters are centralized in `configs/model/default.yaml`:

```yaml
model:
  semantic:
    backbone: "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    output_dim: 256
    freeze_layers: 28
    pooling: "last"
    use_lora: true
    lora_r: 16
    lora_alpha: 32
  graph:
    use_graphcodebert: true
    unfreeze_top_n: 6  # Critical: avoid AUC=0.5 collapse
    num_layers: 4
    num_heads: 8
  fusion:
    combine: "gated"
    residual_alpha: 0.3
  heads:
    binary:
      hidden_dim: 128
      dropout: 0.3
```
