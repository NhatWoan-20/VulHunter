# 05 — Training & Optimization

> **Version: 5.0** — Binary Classification Focus
> **Authoritative Specification**

---

## 1. Data Input Contract

Training consumes `data/splits/{train,validation}.jsonl` from the **Master Dataset** —
per-role records keyed by `{source}:{pair}:{role}`, each carrying:

- `input_ids` + `attention_mask`
- For graph/fusion modes: per-sample heterogeneous graphs in `data/processed/master_pdg.jsonl`

`VulHunterDataset` loads all fields; `collate_fn` pads `input_ids` → `(B, max_seq)`.

---

## 2. Three Experimental Branches

Three branches share identical data, seed, loss function, scheduler, and early stopping:

| Branch | Modalities | Purpose |
|---|---|---|
| `semantic_only` | CodeBERT (seq + pool) | Pure semantic baseline |
| `graph_only` | GAT on PDG | Pure structural baseline |
| `fusion` (Proposed) | Semantic + Graph cross-attention | **Proposed hybrid** |

---

## 3. Quality-Aware Weighted Binary Focal Loss

Per-sample loss scales the focal loss by the **quality tier** so noisier GHSA diffs perturb gradients less:

$$\mathcal{L}_{\text{sample}} = w_{\text{tier}} \cdot \mathcal{L}_{\text{binary}}$$

### 3.1 Quality-Tier Weights

| Tier | Source | $w_{\text{tier}}$ |
|---|---|---|
| `gold` | CVEFixes (reviewed diffs) | **1.0** |
| `silver` | GHSA (auto-derived diffs) | **0.85** |

### 3.2 Focal Loss Hyperparameters

| Parameter | Value | Purpose |
|---|---|---|
| α (alpha) | 0.5 | Balanced weighting (auto-tuned based on pos/neg ratio) |
| γ (gamma) | 2.0 | Focus on hard examples |

---

## 4. Optimization Hyperparameters

### 4.1 Tiered Learning Rates

Different components have different learning rates:

| Component | Learning Rate | Reason |
|---|---|---|
| Backbone (Full Fine-tuning) | 2e-5 | Preserve pre-trained knowledge |
| Graph Encoder | 1e-4 | Randomly initialized, needs more adaptation |
| Binary Head | 2e-4 | Task-specific head, fastest adaptation |

### 4.2 Training Schedule

| Parameter | Fusion | Semantic | Graph |
|---|---|---|---|
| Epochs | 8-12 | 8-10 | 20-25 |
| Batch Size | 2 (per device) | 4 | 8 |
| Grad Accumulation | 4 | 4 | 2 |
| Effective Batch | 16-32 | 16 | 16 |
| Patience | 2-4 | 3 | 4-5 |

### 4.3 Common Settings

- **Scheduler:** Linear warmup (10%) → cosine decay
- **Optimizer:** AdamW, weight decay 0.01, betas (0.9, 0.999)
- **Gradient Clipping:** max norm 1.0
- **AMP:** FP16 mixed precision enabled via `--use-amp`
- **Early Stopping:** patience on validation binary F1

---

## 5. Threshold Tuning

Fixed 0.5 threshold is suboptimal for imbalanced datasets. VulHunter supports **automatic threshold tuning**:

```bash
# Enable threshold tuning in training
python scripts/training/train.py --tune-threshold ...

# Or tune threshold on evaluation
python scripts/evaluation/evaluate.py --checkpoint best.pt
```

The `find_best_threshold()` function scans thresholds from 0.05 to 0.95 and selects the threshold that maximizes F1 score on the validation set.

---

## 6. Training Commands

### 6.1 Local Training

```bash
# Semantic-only baseline
python scripts/training/train.py \
    --mode semantic_only \
    --config configs/train/semantic.yaml \
    --model-config configs/model/default.yaml \
    --use-amp

# Graph-only baseline
python scripts/training/train.py \
    --mode graph_only \
    --config configs/train/graph.yaml \
    --graph-data data/processed/master_pdg.jsonl \
    --use-amp

# Fusion (proposed)
python scripts/training/train.py \
    --mode fusion \
    --config configs/train/fusion.yaml \
    --graph-data data/processed/master_pdg.jsonl \
    --tune-threshold \
    --use-amp
```

### 6.2 Kaggle Training

Upload `notebooks/train_fusion.ipynb`, `notebooks/train_semantic_only.ipynb`, or `notebooks/train_graph_only.ipynb` to Kaggle with the pre-tokenized dataset.

---

## 7. Reproducibility

- **Seed 42** everywhere; fixed repo-disjoint splits; deterministic checkpointing.
- All configurations saved in checkpoints.
- Training history logged to `training_history.json`.

---

## 8. Roadmap: Future Multi-Task Extension


1. Train binary classification first (current phase)
2. Freeze binary head weights
3. Add auxiliary CWE/severity heads with lower learning rates
4. Joint training with weighted multi-task loss

The current architecture is designed to support this transition:
- `BinaryHead` is already modular
- `ModelOutput` can be extended with additional logits
- Quality-tier weighting already in place





