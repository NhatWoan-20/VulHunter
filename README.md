<h1 align="center">🛡️ VulHunter</h1>

<p align="center">
  <strong>Hybrid Multi-Modal Binary Vulnerability Detection for Python</strong><br/>
  <em>CodeBERT (Semantic View) + Program Dependence Graph (RGCN) (Structural View) + Gated Bidirectional Cross-Attention (Fusion)</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-red?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-4.36%2B-yellow?logo=huggingface&logoColor=white" alt="Transformers">
  <img src="https://img.shields.io/badge/Kaggle-2xT4-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle">
  <img src="https://img.shields.io/badge/Tests-Passed-brightgreen" alt="Tests">
  <a href="https://doi.org/10.5281/zenodo.13118970"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.13118970-blue" alt="DOI"></a>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

<p align="center">
  <a href="#-quick-start--60s">Quick Start</a> •
  <a href="#-key-features">Key Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-dataset">Dataset</a> •
  <a href="#-run-on-kaggle-2xt4">Kaggle 2x T4</a> •
  <a href="#-training-modes">Training</a> •
  <a href="#-benchmarks--evaluation">Evaluation</a>
</p>

---

## 📌 What is VulHunter?

**VulHunter** is an end-to-end deep learning system for **binary vulnerability detection** in Python function-level code (vulnerable vs. safe).

Traditional vulnerability detectors rely either on purely syntactic sequence representations (LLMs/Transformers) which can miss non-local data-flow constraints, or purely on graph structures (AST/CFG/GNNs) which discard rich identifier semantics and comments. **VulHunter bridges this gap** by fusing two complementary representations:

1. **Semantic Perception**: Pretrained **CodeBERT** (Full Fine-tuning) for token semantics and control keywords.
2. **Structural Perception**: **Program Dependence Graph (PDG)** + Heterogeneous Graph Neural Network (**RGCN**) processing heterogeneous program graph edge types (control-flow, data-flow, call).
3. **Cross-Modal Fusion**: A **gated bidirectional cross-attention** mechanism with residual skip that dynamically balances semantic and structural signals.
4. **Binary Supervision**: Focal Loss giải quyết mất cân bằng mẫu.

---

## ⚡ Key Features

- **Single Task Focus**: Binary vulnerability detection (vulnerable=1 / safe=0) — đơn giản, hiệu quả.
- **3 Training Modes**: `semantic_only`, `graph_only`, `fusion` — train song song để so sánh.
- **Full Fine-Tuning**: Tối ưu trực tiếp toàn bộ backbone CodeBERT (~125M params) với FP16.
- **CLS Pooling**: Sử dụng vector [CLS] tiêu chuẩn của kiến trúc encoder-only RoBERTa.
- **Threshold Tuning**: Auto-find optimal F1 threshold trên val set.
- **MLOps Ready**: FastAPI deployment script.
- **Strict Leakage Prevention**: 80/10/10 split grouped strictly by GitHub repository.
- **Resource Efficient**: Fits Kaggle 2x T4 (16GB VRAM) với FP16 mixed precision.

---

## 🏗️ Architecture

```text
                                Python Function Code
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
       [Semantic Branch]                                [Structural Branch]
     CodeBERT (125M)                           Heterogeneous Program Graph
     + Full Fine-tuning                               (AST + CFG + DFG + Call)
     CLS pooling
                 │                                               │
        Per-token sequence                              NodeType Embedding + 
        representations                                 Stacked RGCN Layers
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                       Gated Bidirectional Cross-Attention
                       + Residual Skip (alpha=0.3)
                                         │
                                         ▼
                                Binary Prediction Head
                                     (Focal Loss)
                                         │
                                         ▼
                              P(vulnerable) ∈ [0, 1]
```

Configuration wiring được tách riêng ở `configs/model/default.yaml` và training schedules ở `configs/train/` (`semantic.yaml`, `graph.yaml`, `fusion.yaml`).

---

## 📦 Dataset

VulHunter trains trên consolidated **Master Dataset** kết hợp:

| Source | Raw Samples | Cleaned Pairs | Role Samples |
|:---|:---|:---:|:---:|
| **CVEFixes** | Zenodo SQL dump | **2,985** | 5,958 |
| **GHSA** | GitHub Security Advisories | **12,366** | 24,496 |
| **Master (Unified)** | CVEFixes + GHSA | **15,351** | **30,454** |

Split: **80/10/10 repository-disjoint** (strict repo-disjoint).

```json
{
  "sample_id": "cvefixes:98919200308f75a4:vulnerable",
  "code": "def run_query(q):\n    return db.execute('SELECT * WHERE id = ' + q)",
  "binary_label": 1,
  "input_ids": [13, 298, ...]
}
```

---

## ⚡ Quick Start — 60s

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/NhatWoan-20/VulHunter.git
cd VulHunter

# Setup virtual environment
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate

# Install PyTorch với CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Chuẩn bị dữ liệu (Local hoặc Kaggle)

```bash
# Full pipeline: master → samples → AST → graphs → tokenize → split
python scripts/preprocessing/run_pipeline.py

# Nếu chỉ cần semantic-only (không cần graph):
python scripts/preprocessing/run_pipeline.py --skip-graph

# Nếu chỉ cần graph-only (không cần tokenize):
python scripts/preprocessing/run_pipeline.py --skip-tokenize
```

### 3. Training

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

# Fusion (đề xuất chính)
python scripts/training/train.py \
    --mode fusion \
    --config configs/train/fusion.yaml \
    --graph-data data/processed/master_pdg.jsonl \
    --tune-threshold \
    --use-amp
```

### 4. Evaluation

```bash
python scripts/evaluation/evaluate.py \
    --checkpoint models/checkpoints/best.pt \
    --test-data data/splits/test.jsonl \
    --graph-data data/processed/master_pdg.jsonl \
    --output outputs/metrics/evaluation_report.json
```

### 5. FastAPI Deployment

```bash
uvicorn scripts.api_deployment:app --host 0.0.0.0 --port 8000

curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"code": "import os\ndef run():\n    os.system(user_input)"}'
```

---

## ☁️ Run on Kaggle (2x T4 GPUs)

> **Tự động clone:** Các notebooks sẽ tự động clone VulHunter từ GitHub khi chạy. Không cần upload dataset!

### Workflow

1. **Launch Kaggle Notebook**:
   - Accelerator: **GPU T4 x2**
   - Internet: **ON** | Persistence: **ON**
   - Add Input: `vulhunter-pre-tokenized` (dataset đã tokenized sẵn)
   - *Lưu ý: Nếu không có dataset, notebooks sẽ tự clone repo và bạn cần chạy preprocessing*

2. **Run Notebooks** (tự động clone):
   - `notebooks/train_fusion.ipynb` — Fusion mode (chính)
   - `notebooks/train_semantic_only.ipynb` — Semantic baseline
   - `notebooks/train_graph_only.ipynb` — Graph baseline

3. **Sau khi clone thành công**, notebook sẽ:
   - Cài đặt dependencies tự động
   - Import modules và hiển thị "All imports OK"
   - Sẵn sàng để train

---

## 🚀 Training Modes

#### Scenario A: Semantic-Only (CodeBERT)
```bash
python scripts/training/train.py \
  --mode semantic_only \
  --config configs/train/semantic.yaml \
  --model-config configs/kaggle/model_kaggle.yaml \
  --use-amp
```

#### Scenario B: Graph-Only Structural Baseline
```bash
python scripts/training/train.py \
  --mode graph_only \
  --config configs/train/graph.yaml \
  --graph-data data/processed/master_pdg.jsonl \
  --use-amp
```

#### Scenario C: Multi-Modal Fusion (CodeBERT + RGCN)
```bash
python scripts/training/train.py \
  --mode fusion \
  --config configs/train/fusion.yaml \
  --model-config configs/kaggle/model_kaggle.yaml \
  --graph-data data/processed/master_pdg.jsonl \
  --tune-threshold \
  --use-amp
```

### Tips:

- **`--tune-threshold`**: Auto-tune classification threshold trên val set sau mỗi epoch.
- **`--use-amp`**: Bật FP16 mixed precision.
- **`--data-parallel`**: Force single-node DataParallel (thay vì DDP).

---

## 📊 Benchmarks & Evaluation

Đánh giá theo multi-tier protocol:
1. **In-Domain**: Held-out 10% test split từ Master Dataset (repo-disjoint).
2. **Out-of-Domain** (future): Zero-shot trên PyCode-Vul.

### Metric Targets (binary classification)

| Metric | Rất kém | Kém | Trung bình | Tốt | Rất tốt |
|:---|:---:|:---:|:---:|:---:|:---:|
| AUC-ROC | <0.55 | 0.55-0.65 | 0.65-0.75 | 0.75-0.85 | >0.85 |
| F1 Score | <0.20 | 0.20-0.40 | 0.40-0.60 | 0.60-0.75 | >0.75 |
| Precision | — | — | — | >0.70 | >0.80 |
| Recall | — | — | — | >0.65 | >0.75 |

### Benchmark Results

> [!NOTE]
> **Experimental Phase:** Benchmark scores sẽ được populate sau khi training hoàn tất.

| Model Variant | Binary F1 | Binary MCC | AUC-ROC |
|:---|:---:|:---:|:---:|
| `graph_only` (PDG + RGCN) | *TBD* | *TBD* | *TBD* |
| `semantic_only` (CodeBERT) | *TBD* | *TBD* | *TBD* |
| **`fusion` (CodeBERT + RGCN)** | *TBD* | *TBD* | *TBD* |

---

## 📁 Project Structure

```text
VulHunter/
├── configs/                     # Hyperparameter & architecture specs
│   ├── model/default.yaml       # CodeBERT + RGCN + Gated Cross-Attention
│   ├── train/                   # Training profiles (semantic.yaml, graph.yaml, fusion.yaml)
│   └── kaggle/                  # Kaggle profiles
├── data/
│   ├── raw/databases/           # CVEFixes DB extraction scripts
│   └── splits/                  # Repo-disjoint train/validation/test splits
├── notebooks/                   # Kaggle notebooks (3 modes)
│   ├── train_fusion.ipynb       # ★ Main: fusion mode
│   ├── train_semantic_only.ipynb
│   └── train_graph_only.ipynb
├── src/                         # Core VulHunter Library
│   ├── semantic/encoder.py      # CodeBERT, CLS pooling
│   ├── graph/encoder.py         # Heterogeneous RGCN Graph Encoder
│   ├── fusion/cross_attention.py# Gated bidirectional cross-attention + residual
│   ├── multitask/model.py       # VulHunterModel (binary output)
│   ├── multitask/heads.py       # BinaryHead
│   └── utils/                   # Datasets, focal loss, metrics
├── scripts/
│   ├── extraction/prepare_master.py  # Master dataset builder
│   ├── preprocessing/           # Tokenize, comment strip, graph build, split
│   │   └── run_pipeline.py      # ★ End-to-end pipeline orchestrator
│   ├── graph/                   # PDG extraction
│   ├── training/train.py        # Distributed / AMP training runner
│   ├── evaluation/evaluate.py   # Binary classification evaluation
│   └── api_deployment.py        # FastAPI server
├── tests/                       # Test suite (61 unit tests)
└── docs/                        # Research methodology & specifications
```

---

## 🧪 Testing & Verification

```bash
# Run all unit tests
pytest tests -q

# Run with coverage
pytest tests --cov=src --cov-report=term-missing
```

---

## 🛠️ Tech Stack

- **Deep Learning**: [PyTorch 2.1+](https://pytorch.org/), [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- **Foundation Model**: [CodeBERT](https://github.com/microsoft/CodeBERT) (125M, `microsoft/codebert-base`)
- **Fine-tuning**: Full Fine-Tuning
- **Graph Neural Network**: Relational Graph Convolutional Network (RGCN) via PyG
- **Evaluation**: `scikit-learn`, `scipy`

---

## 🤝 Contributing & License

Contributions welcome! Đảm bảo pass existing unit tests (`pytest tests -q`).

Distributed under the **MIT License**.

### Acknowledgments
- **CVEFixes**: [secureIT-project/CVEfixes](https://github.com/secureIT-project/CVEfixes) (Zenodo DOI: `10.5281/zenodo.13118970`)
- **GitHub Security Advisories (GHSA)**: [GitHub Advisory Database](https://github.com/advisories)
- **CodeBERT**: Microsoft Research





