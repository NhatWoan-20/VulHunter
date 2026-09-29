# Model Training

> **Objective:** Train the multi-task, multi-modal VulHunter model.

This directory contains the central training script for the project. It orchestrates the loading of models, datasets, criteria (loss functions), and optimization schedules to train the network end-to-end.

## Files Description

- **`train.py`**: The master training script. It handles:
  - **3 Operating Modes**: `semantic_only` (CodeBERT only), `graph_only` (RGCN only), and `fusion` (Cross-Attention between CodeBERT and RGCN).
  - **Binary Optimization**: Optimizes binary classification head with Focal Loss.
  - **Hardware Acceleration**: Mixed Precision (AMP FP16), and Gradient Checkpointing, optimized for Kaggle 2x T4 (16GB) DataParallel setups.

## Configuration

Training behavior is heavily parameterized by YAML config files located in `configs/`:
- `configs/kaggle/model_kaggle.yaml`: Defines architecture parameters (hidden dims, layers, heads).
- `configs/train/*.yaml`: Defines optimization hyperparams (LR, epochs, loss weights) for each mode.

## How to Run

The model can be trained in 3 distinct operating modes. You can switch between them using the `--mode` flag.

### 1. Fusion Mode (Default)
Trains the full multi-modal architecture with cross-attention. Requires both tokenized data and graph data.

```bash
python scripts/training/train.py \
    --mode fusion \
    --model-config configs/kaggle/model_kaggle.yaml \
    --config configs/train/fusion.yaml \
    --train-data /kaggle/input/vulhunter-pre-tokenized/train.jsonl \
    --val-data /kaggle/input/vulhunter-pre-tokenized/validation.jsonl \
    --graph-data data/processed/master_pdg.jsonl
```

### 2. Semantic-Only Mode (LLM Only)
Trains only the CodeBERT semantic encoder. Does not require graph data.

```bash
python scripts/training/train.py \
    --mode semantic_only \
    --model-config configs/kaggle/model_kaggle.yaml \
    --config configs/train/semantic.yaml \
    --train-data /kaggle/input/vulhunter-pre-tokenized/train.jsonl \
    --val-data /kaggle/input/vulhunter-pre-tokenized/validation.jsonl
```

### 3. Graph-Only Mode (RGCN Only)
Trains only the Relational Graph Convolutional Network. Does not use the CodeBERT backbone.

```bash
python scripts/training/train.py \
    --mode graph_only \
    --config configs/train/graph.yaml \
    --train-data data/splits/train.jsonl \
    --val-data data/splits/validation.jsonl \
    --graph-data data/processed/master_pdg.jsonl
```

> [!IMPORTANT]
> The early stopping mechanism monitors the **validation binary F1 score**. The best checkpoint will be saved to `models/checkpoints/best.pt`.



