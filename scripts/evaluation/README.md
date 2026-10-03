# Evaluation & Benchmark

> **Objective:** Rigorously evaluate trained checkpoints on the in-domain test set.

This directory contains scripts to assess model performance across all tasks.

## Files Description

- **`evaluate.py`**: The primary evaluation script for the in-domain Master Dataset test split. It loads a trained checkpoint (`best.pt`) and computes comprehensive metrics:
  - Binary Classification (F1, MCC, AUC, Accuracy)
  It outputs a detailed JSON report to `outputs/metrics/evaluation_report.json`.

- **`evaluate_external.py`**: Reserved script for evaluating model generalization on a held-out, out-of-domain CSV dataset. Currently unused (no external benchmark is included in the project). If an external CSV dataset is provided in `data/raw/external/`, this script converts it to canonical JSONL format and evaluates the model's semantic branch.

## How to Run

Evaluate on the in-domain test split (Fusion mode requires graph data):

```bash
python scripts/evaluation/evaluate.py \
    --checkpoint models/checkpoints/best.pt \
    --test-data data/splits/test.jsonl \
    --graph-data data/processed/master_pdg.jsonl
```

> [!NOTE]
> Check the `outputs/metrics/` directory for the resulting JSON files. These metrics are used to compare `semantic_only`, `graph_only`, and `fusion` modalities for research evaluation.



