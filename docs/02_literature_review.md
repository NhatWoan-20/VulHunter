# 02 — State of the Art & Literature Review

> **Version: 3.3 (2026-08-30)**  
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
│  Semantic-Based │   │   Graph-Based   │   │   Hybrid-Based  │
│  (Transformers) │   │   (GNNs / GAT)  │   │  (VulHunter)    │
│  - CodeBERT     │   │  - Devign (GGNN)│   │  - Cross-Modal  │
│  - Qwen-Coder   │   │  - LineVul (GAT)│   │    Attention    │
│  - DeepSeek     │   │  - Reveal (GGNN)│   │  - Multi-Task   │
└─────────────────┘   └─────────────────┘   └─────────────────┘
```

---

## 2. Comparison of Paradigms

| Aspect | Semantic-Based (LLMs/Transformers) | Graph-Based (GNNs on AST/CFG/DFG) | Hybrid Multi-Modal (VulHunter) |
|---|---|---|---|
| **Primary Input** | Token sequence | Node & Edge adjacency matrix | Token sequence + Heterogeneous graph |
| **Strengths** | API semantics, naming conventions, docstrings, long-range token patterns | Control flow jumps, data dependencies, syntactic hierarchy | Combines contextual semantic understanding with exact data/control flow |
| **Weaknesses** | Blind to explicit non-local execution paths and pointer aliasing | Ignores natural language semantics, comments, and sub-token nuance | Higher architectural complexity and multi-modal alignment overhead |
| **Representative Works** | VulBERTa (2022), Rozière et al. (2023) | Devign (Zhou et al., 2019), Reveal (Chakraborty et al., 2021) | VulHunter (Ours), LineVul (Fu et al., 2022) |

---

## 3. Key Research Gaps Addressed by VulHunter

1. **Lack of Multi-Modal Interaction:** Most existing works either serialize graphs into tokens (losing structural topology) or embed token sequences into graph nodes with basic BoW (losing LLM attention dynamics). VulHunter uses **bidirectional cross-attention** between deep LLM contextual states and GAT node embeddings.
2. **Data Leakage in Benchmarks:** Many previous datasets randomly split function samples across train and test sets, allowing models to memorize project-specific identifiers. VulHunter enforces strict **repository-disjoint splitting**.
3. **Multi-Task Learning Absence:** Typical SAST DL models only predict binary labels. VulHunter jointly trains binary detection with **CWE categorization and severity classification**, leveraging shared representations.
