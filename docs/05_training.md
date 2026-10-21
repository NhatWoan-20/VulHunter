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
| `graph_only` | RGCN on PDG | Pure structural baseline |
| `fusion` (Proposed) | Semantic + Graph FiLM modulation (+ cross-attention) | **Proposed hybrid** |

---

## 3. Binary Focal Loss

To address class imbalance, VulHunter utilizes Binary Focal Loss:

$$\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

### 3.1 Focal Loss Hyperparameters

| Parameter | Value | Purpose |
|---|---|---|
| α (alpha) | `1 − pos_rate` (≈ 0.79) | Auto-tuned on the training split's pos/neg ratio (`α = 1 − pos_rate`); 0.5 is only the config fallback |
| γ (gamma) | 2.0 | Focuses gradient updates on hard examples |

---

## 4. Optimization Hyperparameters

### 4.1 Tiered Learning Rates

Different components have different learning rates. Values below are from the primary `fusion.yaml` profile; per-mode overrides live in each `configs/train/*.yaml` (e.g. `graph.yaml` uses `1e-3` for both the RGCN encoder and the head, `semantic_codebert.yaml` uses `1e-4` for the head since only CodeBERT is fine-tuned):

| Component | Learning Rate (fusion) | Reason |
|---|---|---|
| Backbone (Full Fine-tuning) | 2e-5 | Fine-tune CodeBERT Transformer layers |
| Graph Encoder | 1e-4 | Randomly initialized RGCN layers |
| Binary Head | 2e-4 | Task-specific prediction head |

### 4.2 Training Schedule

Training hyperparameters are defined per mode in `configs/train/*.yaml`. The table below reflects those shipped configs:

| Parameter | Fusion (`fusion.yaml`) | Semantic (`semantic_codebert.yaml`) | Graph (`graph.yaml`) |
|---|---|---|---|
| Epochs (early-stopping cap) | 12 | 15 | 30 |
| Batch Size (per device) | 2 | 8 | 16 |
| Grad Accumulation | 4 | 2 | 1 |
| Effective Batch | 8 (16 w/ DDP×2) | 16 | 16 |
| Patience | 4 | 3 | 5 |

> [!NOTE]
> Kaggle profile `configs/kaggle/train_kaggle.yaml` overrides the Fusion schedule for a 2x T4 session (8 epochs, per-device batch 2, grad accum 4, effective batch 16, patience 2).

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
    --config configs/train/semantic_codebert.yaml \
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





