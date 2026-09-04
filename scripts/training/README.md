# Model Training

> **Objective:** Train the multi-task, multi-modal VulHunter model.

This directory contains the central training script for the project. It orchestrates the loading of models, datasets, criteria (loss functions), and optimization schedules to train the network end-to-end.

## Files Description

- **`train.py`**: The master training script. It handles:
  - **3 Operating Modes**: `semantic_only` (LLM only), `graph_only` (GAT only), and `fusion` (Cross-Attention between LLM and GAT).
  - **Multi-Task Optimization**: Jointly optimizes 5 loss heads (Binary, CWE, Severity, Localization, Source/Sink) weighted by sample quality (gold/silver).
  - **Hardware Acceleration**: Automatic DataParallel for Multi-GPU setups (like Kaggle 2xT4), Mixed Precision (AMP FP16), and Gradient Checkpointing.
  - **LoRA Support**: Seamless integration with PEFT/LoRA to fine-tune massive backbones (like Qwen2.5-Coder-3B) efficiently on limited VRAM.

## Configuration

Training behavior is heavily parameterized by YAML config files located in `configs/`:
- `configs/model/default.yaml`: Defines architecture parameters (hidden dims, layers, heads).
- `configs/train/default.yaml`: Defines optimization hyperparams (LR, epochs, loss weights).
- `configs/kaggle/*`: Specialized configs for Kaggle T4 setups (e.g., LoRA configurations).

## How to Run

The model can be trained in 3 distinct operating modes. You can switch between them using the `--mode` flag.

### 1. Fusion Mode (Default)
Trains the full multi-modal architecture with cross-attention. Requires both tokenized data and graph data.

```bash
python scripts/training/train.py \
    --mode fusion \
    --model-config configs/kaggle/model_kaggle_3b_lora.yaml \
    --config configs/kaggle/train_kaggle_3b_lora.yaml \
    --train-data /kaggle/input/vulhunter-pre-tokenized/train.jsonl \
    --val-data /kaggle/input/vulhunter-pre-tokenized/validation.jsonl \
    --graph-data data/processed/master_graphs.jsonl
```

### 2. Semantic-Only Mode (LLM Only)
Trains only the Qwen2.5-Coder semantic encoder. Does not require graph data.

```bash
python scripts/training/train.py \
    --mode semantic_only \
    --model-config configs/kaggle/model_kaggle_3b_lora.yaml \
    --config configs/kaggle/train_kaggle_3b_lora.yaml \
    --train-data /kaggle/input/vulhunter-pre-tokenized/train.jsonl \
    --val-data /kaggle/input/vulhunter-pre-tokenized/validation.jsonl
```

### 3. Graph-Only Mode (GAT Only)
Trains only the Graph Attention Network. Does not use the LLM backbone.

```bash
python scripts/training/train.py \
    --mode graph_only \
    --train-data data/splits/train.jsonl \
    --val-data data/splits/validation.jsonl \
    --graph-data data/processed/master_graphs.jsonl
```

> [!IMPORTANT]
> The early stopping mechanism monitors the **validation binary F1 score**. The best checkpoint will be saved to `models/checkpoints/best.pt`.
