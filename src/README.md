# VulHunter Source Library (`src/`)

> **Objective:** Core PyTorch neural network modules, data handling utilities for the VulHunter project.

This directory houses the foundational library code. It is designed to be highly modular, allowing the `semantic`, `graph`, and `fusion` components to be instantiated and tested independently or composed together in the main `multitask` model.

## Sub-Modules

### 1. `semantic/`
Contains the **Semantic Encoder** (`encoder.py`). This module wraps a pre-trained Code LLM (specifically `CodeBERT`) to extract semantic token representations from raw source code. It includes integrated support for Gradient Checkpointing and FP16 support to enable efficient fine-tuning of large models on constrained hardware (like Kaggle 2x T4s).

### 2. `graph/`
Contains the **Graph Encoder** (`encoder.py`). This implements a custom **Graph Attention Network (GAT)** designed to process heterogeneous program graphs (PDG Graph). It features edge-type-aware attention, allowing the network to distinguish between syntactic hierarchy and data-flow dependencies when aggregating neighborhood information.

### 3. `fusion/`
Contains the **Cross-Modal Fusion** module (`cross_attention.py`). This module implements a Bidirectional Cross-Attention mechanism. It bridges the gap between the semantic LLM tokens and the structural GAT nodes, allowing the semantic context to attend to structural graphs and vice-versa, outputting a unified, gated representation.

### 4. `multitask/`
The apex of the model architecture.
- **`heads.py`**: Defines lightweight, task-specific prediction layers (Binary) that branch off from the fused representation.
- **`model.py`**: Defines the `VulHunterModel`, which composes the semantic encoder, graph encoder, fusion module, and the 3 task heads into a single, end-to-end trainable PyTorch `nn.Module`.

### 5. `explainability/` (Archived)
Handles the post-hoc translation of model predictions into human-readable, actionable security reports.
- **`generator.py`**: The `ExplanationGenerator` can operate in two modes: a fast, offline heuristic mode (using pre-defined templates), or an LLM-assisted mode.
- **`prompts.py`**: Contains system prompts, and formatting logic.

### 6. `utils/`
Core utilities shared across the project.
- **`dataset.py`**: Defines the `VulHunterDataset` and `collate_fn` for PyTorch DataLoaders. It handles padding, heterogeneous graph batching (via PyTorch Geometric), and dynamic label extraction.
- **`metrics.py`**: Calculation functions for all benchmarks (Binary F1/MCC).

---

> [!TIP]
> All modules in `src/` are intended to be imported by scripts in the `scripts/` directory. When developing new features, avoid adding executable CLI code here; instead, expose Python classes/functions and write a corresponding execution script in `scripts/`.




