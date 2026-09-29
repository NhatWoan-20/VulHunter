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
 │   Semantic Branch    │                            │     Graph Branch     │
 │    (CodeBERT 125M    │                            │    (PDG with RGCN)   │
 │  + Full Fine-tuning  │                            │  NodeType Embedding  │
 │     CLS pooling      │                            │  + Stacked RGCN      │
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
                       │    Binary Prediction Head    │
                       │      (Focal Loss)            │
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
| **Backbone** | `microsoft/codebert-base` (125M) |
| **Context Length** | 512 tokens |
| **Pooling** | **[CLS] pooling** (RoBERTa encoder architecture) |
| **Fine-tuning** | **Full Fine-Tuning** with FP16 |
| **Projection** | Linear projection to D=256 with LayerNorm & GELU |
| **Output** | `h_sem ∈ ℝ^D` (pooled representation, D=256) |

### 2.2 Graph Encoder (`src/graph/encoder.py`)

| Property | Value |
|---|---|
| **Architecture** | Stacked RGCN layers |
| **Node Features** | `d_node=128` (NodeTypeEmbedding over AST node types) |
| **Hidden/Output Dim** | 256 |
| **Edge Types (3)** | `CONTROL_FLOW`, `DATA_FLOW`, `CALL` (`EDGE_TYPE_MAP`) |
| **Readout** | Mean + Max pool over nodes → Linear projection → `h_graph ∈ ℝ^D` |

### 2.3 Cross-Modal Fusion (`src/fusion/cross_attention.py`)

| Property | Value |
|---|---|
| **Mechanism** | Bidirectional multi-head cross-attention |
| **Combine Mode** | `gated` (options: `concat`, `mean`) |
| **Residual Skip** | `h_fused = (1-α) * h_cross + α * h_sem` where α=0.3 |
| **Layers** | 2 cross-attention layers |
| **Heads** | 8 |
| **Output** | `h_fused ∈ ℝ^D` (same dimension as semantic/graph outputs) |

### 2.4 Binary Prediction Head (`src/multitask/heads.py`)

| Property | Value |
|---|---|
| **Input** | `h_fused ∈ ℝ^D` |
| **Architecture** | MLP: Linear(D→128) → GELU → Dropout → Linear(128→64) → GELU → Dropout → Linear(64→1) |
| **Output** | Logit ŷ ∈ ℝ¹ |
| **Loss Function** | Focal Loss (α=0.5, γ=2.0) |

---

## 3. Three Operating Modes

VulHunter supports three modes, selected via `--mode` flag:

| Mode | Semantic Encoder | Graph Encoder | Fusion | Use Case |
|---|---|---|---|---|
| `semantic_only` | ✓ CodeBERT | ✗ | ✗ | Semantic baseline |
| `graph_only` | ✗ | ✓ RGCN on PDG | ✗ | Structural baseline |
| `fusion` (Proposed) | ✓ CodeBERT | ✓ RGCN on PDG | ✓ Gated Cross-Attn | **Main approach** |

---

## 4. Data Flow

1. **Input:** Python function code string
2. **Semantic Branch:** Tokenize with CodeBERT tokenizer → Full Fine-tuned CodeBERT → [CLS] pooling → `h_sem`
3. **Graph Branch:** Parse AST/CFG/DFG/Call into PDG → NodeType Embedding + RGCN → mean/max pool → `h_graph`
4. **Fusion:** Cross-attend `h_sem` and `h_graph` → residual skip → `h_fused`
5. **Prediction:** Binary head → logit → sigmoid → `p(vulnerable)`

---

## 5. Model Configuration

All hyperparameters are centralized in `configs/model/default.yaml`:

```yaml
model:
  semantic:
    backbone: "microsoft/codebert-base"
    output_dim: 256
    dropout: 0.1
    pooling: "cls"
  graph:
    node_feature_dim: 128
    hidden_dim: 256
    output_dim: 256
    num_layers: 4
    num_edge_types: 3
    dropout: 0.2
  fusion:
    hidden_dim: 256
    num_heads: 8
    num_layers: 2
    dropout: 0.1
    combine: "gated"
    residual_alpha: 0.3
  heads:
    binary:
      hidden_dim: 128
      dropout: 0.3
```



