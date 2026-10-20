# VulHunter Source Library (`src/`)

> **Objective:** Core PyTorch neural network modules, data handling utilities for the VulHunter project.

This directory houses the foundational library code. It is designed to be highly modular, allowing the `semantic`, `graph`, and `fusion` components to be instantiated and tested independently or composed together in the main `multitask` model.

## Sub-Modules

### 1. `semantic/`
Contains the **Semantic Encoder** (`encoder.py`). This module wraps a pre-trained Code LLM (specifically `CodeBERT`) to extract semantic token representations from raw source code. It includes integrated support for Gradient Checkpointing and FP16 support to enable efficient fine-tuning of large models on constrained hardware (like Kaggle 2x T4s).

### 2. `graph/`
Contains the **Graph Encoder** (`encoder.py`). This implements a **Relational Graph Convolutional Network (RGCN)** designed to process heterogeneous program graphs (PDG Graph). It features relation-aware convolutions, allowing the network to distinguish between control-flow, data-flow, and call dependencies when aggregating neighborhood information.

### 3. `fusion/`
Contains the **Cross-Modal Fusion** module (`cross_attention.py`). This module implements a Bidirectional Cross-Attention mechanism. It bridges the gap between semantic tokens and structural graph representations, allowing the semantic context to attend to structural graphs and vice-versa, outputting a unified, gated representation.

### 4. `multitask/`
The apex of the model architecture.
- **`heads.py`**: Defines lightweight, task-specific prediction layers (Binary) that branch off from the fused representation.
- **`model.py`**: Defines the `VulHunterModel`, which composes the semantic encoder, graph encoder, fusion module, and the binary task head into a single, end-to-end trainable PyTorch `nn.Module`.

### 5. `utils/`
Core utilities shared across the project.
- **`dataset.py`**: Defines the `VulHunterDataset` and `collate_fn` for PyTorch DataLoaders with Zero-RAM Lazy Indexing (~15MB RAM total) and multi-worker prefetching.
- **`losses.py`**: Implements `FocalLoss` and `BinaryClassificationLoss` to handle class imbalance (78.4% benign : 21.6% vulnerable).
- **`metrics.py`**: Calculation functions for all benchmarks (Binary F1, Precision, Recall, Accuracy, ROC-AUC, PR-AUC, MCC) and dynamic threshold tuning (`find_best_threshold`).

---

> [!TIP]
> All modules in `src/` are intended to be imported by scripts in the `scripts/` directory. When developing new features, avoid adding executable CLI code here; instead, expose Python classes/functions and write a corresponding execution script in `scripts/`.




