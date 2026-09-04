# Evaluation & Benchmark

> **Objective:** Rigorously evaluate trained checkpoints on in-domain test sets and external zero-shot benchmarks.

This directory contains scripts to assess model performance across all tasks.

## Files Description

- **`evaluate.py`**: The primary evaluation script for the in-domain Master Dataset test split. It loads a trained checkpoint (`best.pt`) and computes comprehensive metrics for all 5 tasks:
  - Binary Classification (F1, MCC, AUC, Accuracy)
  - CWE Classification (Macro-F1, Precision, Recall)
  - Severity Prediction (F1, Accuracy)
  - Line-Level Localization (Token/Line F1, Precision, Recall)
  - Source/Sink Detection (F1, Precision, Recall)
  It outputs a detailed JSON report to `outputs/metrics/evaluation_report.json`.

- **`evaluate_external.py`**: Evaluates model generalization on a held-out, out-of-domain dataset (PyCode-Vul). Since PyCode-Vul lacks program graphs, this script only evaluates the semantic branch of the model (or `semantic_only` checkpoints). It tokenizes the raw source code on the fly using `Qwen2.5-Coder` and tests binary and CWE capabilities.

## How to Run

Evaluate on the in-domain test split (Fusion mode requires graph data):

```bash
python scripts/evaluation/evaluate.py \
    --checkpoint models/checkpoints/best.pt \
    --test-data data/splits/test.jsonl \
    --graph-data data/processed/master_graphs.jsonl
```

Evaluate zero-shot on the external PyCode-Vul test set:

```bash
python scripts/evaluation/evaluate_external.py \
    --checkpoint models/checkpoints/best.pt \
    --split test
```

> [!NOTE]
> Check the `outputs/metrics/` directory for the resulting JSON files. These metrics are used to compare `semantic_only`, `graph_only`, and `fusion` modalities for research evaluation.
