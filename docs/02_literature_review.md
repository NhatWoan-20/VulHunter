# 02 — State of the Art & Literature Review

> **Version: 5.0** — Binary Classification Focus
> **Authoritative Specification**

---

## 1. Vulnerability Detection Approaches

Automated vulnerability detection approaches in software engineering can be broadly categorized into three paradigms:

```
┌─────────────────────────────────────────────────────────────┐
│                    Vulnerability Detection                   │
└──────────────────────────────┬──────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  Semantic-Based  │   │   Graph-Based   │   │   Hybrid-Based  │
│  (Transformers)  │   │   (GNNs / GAT)  │   │  (VulHunter)   │
│  - CodeBERT     │   │  - Devign (GGNN)│   │  - Cross-Modal  │
│  - CodeBERT   │   │  - LineVul (GAT)│   │    Attention    │
│  - DeepSeek     │   │  - Reveal (GGNN)│   │  - Residual     │
└─────────────────┘   └─────────────────┘   └─────────────────┘
```

---

## 2. Comparison of Paradigms

| Aspect | Semantic-Based (LLMs/Transformers) | Graph-Based (GNNs on AST/CFG/DFG) | Hybrid Multi-Modal (VulHunter) |
|---|---|---|---|
| **Primary Input** | Token sequence | Node & Edge adjacency matrix | Token sequence + Heterogeneous graph |
| **Strengths** | API semantics, naming conventions, comments, long-range token patterns | Control flow jumps, data dependencies, syntactic hierarchy | Combines contextual semantic understanding with exact data/control flow |
| **Weaknesses** | Blind to explicit non-local execution paths and pointer aliasing | Ignores natural language semantics, comments, and sub-token nuance | Higher architectural complexity and multi-modal alignment overhead |
| **Representative Works** | VulBERTa (2022), CodeGen (2023) | Devign (Zhou et al., 2019), Reveal (Chakraborty et al., 2021) | VulHunter (Ours), LineVul (Fu et al., 2022) |

---

## 3. Key Design Decisions in VulHunter

### 3.1 Why Last-Token Pooling for Decoder-Only LLMs?

Decoder-only models like CodeBERT have a special `<|endoftext|>` token at the end of every sequence. Taking the last meaningful token's hidden state captures the entire sequence context without needing additional pooling operations. This is:
- **Simple:** No masking or learned parameters needed.
- **Effective:** Proven in code generation tasks.
- **Efficient:** Single forward pass, no additional computation.

### 3.2 Why Full Fine-tuning for Fine-Tuning?

Full fine-tuning of large language models is computationally expensive and risks catastrophic forgetting. Full Fine-tuning (Low-Rank Adaptation):
- **Trainable Parameters:** Only ~0.1-1% of total parameters (vs. 100% for full fine-tuning).
- **VRAM Savings:** ~4-6GB reduction on 1.5B models.
- **Rank-Stabilized Full Fine-tuning (RSFull Fine-tuning):** Used for better convergence stability.

### 3.3 Why Unfreeze Top-6 Pure Structural Graph Layers?

When using Pure Structural Graph for graph encoding, leaving all layers frozen can lead to:
- **AUC Collapse:** Model outputs become random (AUC ≈ 0.5).
- **Reason:** Frozen layers cannot adapt to the vulnerability detection task's representation needs.

Solution: Unfreeze the top 6 out of 12 transformer layers to allow task-specific adaptation while keeping most of the pre-trained knowledge.

### 3.4 Why Residual Skip in Fusion?

When graph data is sparse, noisy, or unavailable, the fusion module should degrade gracefully to the semantic-only signal. Residual skip connections (`h_fused = α * h_sem + (1-α) * cross_attended`) ensure:
- **α = 0.3:** 30% semantic signal retained even with noisy graphs.
- **Graceful Degradation:** Model doesn't collapse when graph extraction fails.

### 3.5 Why Quality-Aware Sample Weighting?

The Master Dataset combines:
- **Gold samples (CVEFixes):** Human-reviewed, high-quality vulnerability fixes. Weight = 1.0.
- **Silver samples (GHSA):** Automatically extracted, may contain noise. Weight = 0.85.

This prevents noisy silver samples from overwhelming the gradient signal from gold samples.

---

## 4. Research Gaps Addressed by VulHunter

1. **Lack of Multi-Modal Interaction:** Most existing works either serialize graphs into tokens (losing structural topology) or embed token sequences into graph nodes with basic BoW (losing LLM attention dynamics). VulHunter uses **bidirectional cross-attention** between deep LLM contextual states and GAT node embeddings.

2. **Data Leakage in Benchmarks:** Many previous datasets randomly split function samples across train and test sets, allowing models to memorize project-specific identifiers. VulHunter enforces strict **repository-disjoint splitting**.

3. **Class Imbalance:** Vulnerability detection datasets are often imbalanced (more safe than vulnerable code). VulHunter uses **Focal Loss** to focus training on hard-to-classify examples.

4. **Threshold Selection:** Fixed 0.5 threshold is suboptimal for imbalanced datasets. VulHunter uses **threshold tuning** on validation set to maximize F1 score.




