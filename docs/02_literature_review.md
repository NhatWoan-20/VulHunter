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

### 3.1 Why CodeBERT as Semantic Backbone?

CodeBERT (`microsoft/codebert-base`) is a pre-trained Transformer encoder (~125M parameters) specifically tailored for programming languages. It leverages bidirectional contextual representations:
- **Rich Token Representation:** Effectively captures code semantics, control structures, and identifier names.
- **Feasible Full Fine-Tuning:** With ~125M parameters, CodeBERT can be fully fine-tuned directly on consumer-grade hardware (such as Kaggle 2x T4 GPUs) under FP16 mixed precision without requiring parameter-efficient approximations.
- **[CLS] Pooling:** Standard classification token representation aggregates whole-function context cleanly.

### 3.2 Why Heterogeneous Program Dependence Graphs (PDG) with RGCN?

Vulnerabilities often involve non-local data dependencies (e.g., untrusted user inputs flowing into dangerous sinks like `db.execute` or `os.system`):
- **Heterogeneous Relations:** Program graphs contain distinct edge relations: Control Flow, Data Flow, and Function Calls.
- **Relational Convolutions (RGCN):** Allows distinct relational weight matrices per relation type, capturing structural dependencies without losing edge semantics.
- **Structural Generalization:** Graph neural networks provide relational inductive bias complementary to sequential token patterns.

### 3.3 Why Residual Skip in Cross-Modal Fusion?

When graph extraction yields sparse, noisy, or empty structures (e.g. for functions with irregular syntax), the fusion module must degrade gracefully to semantic representations. Residual skip connections (`h_fused = α * h_sem + (1-α) * cross_attended`) ensure:
- **α = 0.3:** 30% semantic signal is preserved unconditionally, guarding against structural noise.
- **Robust Performance:** Prevents catastrophic drops in model accuracy when graph topology alone provides weak signals across disjoint repositories.

---

## 4. Research Gaps Addressed by VulHunter

1. **Lack of Multi-Modal Interaction:** Most existing works either serialize graphs into tokens (losing structural topology) or embed token sequences into graph nodes with basic BoW (losing LLM attention dynamics). VulHunter uses **bidirectional cross-attention** between deep CodeBERT contextual states and RGCN node embeddings.

2. **Data Leakage in Benchmarks:** Many previous datasets randomly split function samples across train and test sets, allowing models to memorize project-specific identifiers. VulHunter enforces strict **repository-disjoint splitting**.

3. **Class Imbalance:** Vulnerability detection datasets are often imbalanced (more safe than vulnerable code). VulHunter uses **Focal Loss** to focus training on hard-to-classify examples.

4. **Threshold Selection:** Fixed 0.5 threshold is suboptimal for imbalanced datasets. VulHunter uses **threshold tuning** on validation set to maximize F1 score.




